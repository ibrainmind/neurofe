"""Single-node study: one JR column, one TMS pulse. Which parameters can backprop recover?

With one node there is no coupling (g) and no delay (mu): only the local parameters
a, b, c1-c4 and the stimulation gain k remain. The node is read out through the
62-channel leadfield column of one real parcel, so amplitudes and sensor noise are
on the same scale as the 10-node and 200-node studies. The teacher is noise-free (R0-clean).

The approach is the one used for the network (core/diffsim.py, same equations as
neurofe's JansenRit): simulate with PyTorch, backpropagate the EEG loss through the whole
window, update with Adam. Many independent single-node models are stacked on the batch
axis, so one simulation trains hundreds of models at once.

Analyses
  A. Local identifiability: finite-difference Jacobian -> Fisher information -> Cramer-Rao
     bounds, eigen-directions, and bounds on combinations (c1*c2, c1/c2, c3*c4, c3/c4).
  B. Profile likelihood (global): fix one parameter on a grid, refit the others by
     backprop, record the best achievable misfit. Flat profile = not identifiable.
  C. Multi-start recovery: teachers x random starts, trained on 1 or 3 amplitudes,
     tested on all 5 (held-out amplitudes = generalisation).
  D. Ablation: neurofe-style raw parameters (Adam on raw values, relu floors, lr 0.05)
     versus log-scaled parameters.

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node
Outputs: results/single_node/{identifiability.json, study.npz, summary.json}; figures by
scripts/single_node_report.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites
from syn_brain.core.params import NOMINAL_RAW, OFFSET, draw_teacher
from syn_brain.core.ladder import REAL_BASELINE_STD_UV
from syn_brain.core import diffsim as ds

NAMES = ("a", "b", "c1", "c2", "c3", "c4", "k")
AMPS = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
TRAIN_SETS = {"1amp": [1000.0], "3amp": [500.0, 1000.0, 2000.0]}
WINDOW = (100, 400)
NOM = np.array([ds.NOMINAL_EFF[n] for n in NAMES])
SIGMA = REAL_BASELINE_STD_UV


def one_node_net(lm_col: np.ndarray) -> ds.Net:
    lm = torch.tensor(lm_col[:, None], dtype=torch.float64)
    return ds.Net(w_n=torch.zeros((1, 1), dtype=torch.float64), dist=torch.zeros((1, 1), dtype=torch.float64),
                  lm_dm=lm - lm.mean(0, keepdim=True), n=1, c=lm.shape[0])


def simulate(theta: torch.Tensor, net: ds.Net, grad_warmup_ms=100, warmup_ms=300):
    """theta: (M, 7) effective parameters of M independent nodes; each sees all 5 amplitudes.

    Returns baseline-corrected EEG in WINDOW, shape (M, 5, C, T).
    """
    M, A = theta.shape[0], len(AMPS)
    th = {n: torch.tensor(v, dtype=torch.float64) for n, v in ds.NOMINAL_EFF.items()}
    for i, n in enumerate(NAMES):
        th[n] = theta[:, i].repeat_interleave(A).view(M * A, 1, 1)
    proto = ds.Protocol(ki=np.ones((M * A, 1)), amp=np.tile(AMPS, M), warmup_ms=warmup_ms)
    e = ds.run(th, net, proto, noise_scale=0.0, integer_delays=True, grad_warmup_ms=grad_warmup_ms)
    return ds.baseline_correct(e[:, 0])[..., WINDOW[0]:WINDOW[1]].view(M, A, net.c, -1)


# ----------------------------------------------------------------------------- A
COMBOS = {"c1*c2": {"c1": 1, "c2": 1}, "c1/c2": {"c1": 1, "c2": -1},
          "c3*c4": {"c3": 1, "c4": 1}, "c3/c4": {"c3": 1, "c4": -1}}
PROTOCOLS = {"1 amplitude": [1000.0], "3 amplitudes": [500.0, 1000.0, 2000.0], "5 amplitudes": list(AMPS)}


def identifiability(truth: np.ndarray, net, h=1e-4):
    T, P = truth.shape
    thetas = [truth]
    for i in range(P):
        for s in (+1, -1):
            t = truth.copy()
            t[:, i] *= np.exp(s * h)
            thetas.append(t)
    with torch.no_grad():
        e = simulate(torch.tensor(np.concatenate(thetas)), net, grad_warmup_ms=None).numpy()
    e = e.reshape(1 + 2 * P, T, len(AMPS), net.c, -1)
    J = np.stack([(e[1 + 2 * i] - e[2 + 2 * i]) / (2 * h) for i in range(P)], -1)  # (T, A, C, t, P)
    out = []
    for t in range(T):
        res = {"peak_abs_uV": float(np.abs(e[0, t]).max()),
               "peak_by_amp_uV": np.abs(e[0, t]).max((1, 2)).tolist()}
        for lab, amps in PROTOCOLS.items():
            rows = [list(AMPS).index(a) for a in amps]
            Jt = J[t, rows].reshape(-1, P)
            F = Jt.T @ Jt / SIGMA ** 2
            cov = np.linalg.inv(F)
            se = np.sqrt(np.diag(cov))
            w, V = np.linalg.eigh(F)
            comb = {}
            for cname, coef in COMBOS.items():
                v = np.zeros(P)
                for n, c in coef.items():
                    v[NAMES.index(n)] = c
                comb[cname] = float(np.sqrt(v @ cov @ v))
            res[lab] = {"crb_rel_se": dict(zip(NAMES, se.tolist())), "combo_rel_se": comb,
                        "eigvals": w.tolist(), "eigvecs": V.T.tolist(),
                        "corr": (cov / np.outer(se, se)).tolist(),
                        "sensitivity_rms": dict(zip(NAMES, np.sqrt((Jt ** 2).mean(0)).tolist()))}
        out.append(res)
    return out, J


# ----------------------------------------------------------------------------- B, C, D
def build_jobs(truth, n_starts, prof_teachers, grid, rng):
    """One row per independent model. Returns dict of arrays describing every job."""
    jobs = {"kind": [], "teacher": [], "train": [], "z0": [], "fixed_idx": [], "fixed_z": [], "param": []}

    def add(kind, t, train, z0, fixed_idx=-1, fixed_z=0.0, param="log"):
        jobs["kind"].append(kind); jobs["teacher"].append(t); jobs["train"].append(train)
        jobs["z0"].append(z0); jobs["fixed_idx"].append(fixed_idx); jobs["fixed_z"].append(fixed_z)
        jobs["param"].append(param)

    T = truth.shape[0]
    tz = np.log(truth / NOM)
    starts = rng.uniform(np.log(0.5), np.log(1.5), (T, n_starts, len(NAMES)))
    starts[:, 0] = 0.0                                 # start 0 = nominal
    for train in TRAIN_SETS:
        for t in range(T):
            for s in range(n_starts):
                add("multistart", t, train, starts[t, s])
                add("multistart_raw", t, train, starts[t, s], param="raw")
    for train in TRAIN_SETS:
        for t in range(prof_teachers):
            for i in range(len(NAMES)):
                for gz in grid:
                    z0 = tz[t].copy()
                    z0[i] = tz[t, i] + gz
                    add("profile", t, train, z0, fixed_idx=i, fixed_z=tz[t, i] + gz)
    return {k: np.array(v) for k, v in jobs.items()}


def train_all(jobs, truth, net, iters, lr_log, lr_raw, log):
    M = len(jobs["kind"])
    T = truth.shape[0]
    with torch.no_grad():
        target_t = simulate(torch.tensor(truth), net)                          # (T, 5, C, t)
    target = target_t[torch.tensor(jobs["teacher"])]                          # (M, 5, C, t)
    amp_mask = torch.zeros((M, len(AMPS)), dtype=torch.float64)
    for j, tr in enumerate(jobs["train"]):
        for a in TRAIN_SETS[tr]:
            amp_mask[j, list(AMPS).index(a)] = 1.0
    scale = (target.pow(2).mean((2, 3)) * amp_mask).sum(1) / amp_mask.sum(1)

    is_raw = torch.tensor(jobs["param"] == "raw")
    z0 = torch.tensor(jobs["z0"], dtype=torch.float64)
    z = z0.clone().requires_grad_(True)
    # Raw (neurofe-style) parameterisation: theta = offset + relu(raw), raw initialised to match z0.
    off = torch.tensor([OFFSET[n] for n in NAMES], dtype=torch.float64)
    raw = (torch.tensor(NOM) * torch.exp(z0) - off).clone().requires_grad_(True)
    free = torch.ones((M, len(NAMES)), dtype=torch.float64)
    fixed_val = torch.zeros((M, len(NAMES)), dtype=torch.float64)
    for j in range(M):
        if jobs["fixed_idx"][j] >= 0:
            free[j, jobs["fixed_idx"][j]] = 0.0
            fixed_val[j, jobs["fixed_idx"][j]] = jobs["fixed_z"][j]
    opt = torch.optim.Adam([{"params": [z], "lr": lr_log}, {"params": [raw], "lr": lr_raw}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.02 * lr_log)
    hist_loss = np.zeros((iters, M))
    snap_every = max(1, iters // 40)
    hist_theta = []
    t0 = time.time()
    for it in range(iters):
        zz = z * free + fixed_val * (1 - free)
        th_log = torch.tensor(NOM) * torch.exp(zz)
        th_raw = off + torch.relu(raw)
        theta = torch.where(is_raw.unsqueeze(1), th_raw, th_log)
        e = simulate(theta, net)
        per_amp = (e - target).pow(2).mean((2, 3))                            # (M, 5)
        loss = (per_amp * amp_mask).sum(1) / amp_mask.sum(1) / scale
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        hist_loss[it] = loss.detach().numpy()
        if it % snap_every == 0 or it == iters - 1:
            hist_theta.append(theta.detach().numpy().copy())
        if it % 20 == 0 or it == iters - 1:
            sel = jobs["kind"] == "multistart"
            err = np.abs(np.log(theta.detach().numpy()[sel] / truth[jobs["teacher"][sel]]))
            log.info("it %4d  median loss: log %.2e raw %.2e profile %.2e | multistart median |log-err| %s (%.0fs)",
                     it, np.median(hist_loss[it][sel]), np.median(hist_loss[it][jobs["kind"] == "multistart_raw"]),
                     np.median(hist_loss[it][jobs["kind"] == "profile"]),
                     " ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, np.median(err, 0))), time.time() - t0)
    theta = theta.detach()
    with torch.no_grad():
        pred = simulate(theta, net)
    resid = (pred - target).pow(2).sum((2, 3)).numpy()                       # (M, 5) sum of squares
    rel = (resid / target.pow(2).sum((2, 3)).numpy()) ** 0.5
    return theta.numpy(), hist_loss, np.array(hist_theta), resid, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", type=int, default=10)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--prof-teachers", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lr-raw", type=float, default=0.05, help="neurofe's default learning rate")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--part", type=int, default=0, help="this process trains jobs[part::nparts]")
    ap.add_argument("--nparts", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = get_logger(f"single_node_part{args.part}of{args.nparts}")
    out_dir = RESULTS_DIR / "single_node"
    out_dir.mkdir(parents=True, exist_ok=True)

    anat = load_anatomy(10)
    site = int(pick_sites(anat, 1, seed=0)[0])
    net = one_node_net(anat.lm[:, site])
    truth = np.array([[draw_teacher(1, seed=s, spread=args.spread).effective[n] for n in NAMES]
                      for s in range(args.teachers)])

    # A. local identifiability (also at nominal, appended as the last row); part 0 only
    if args.part == 0:
        ident, J = identifiability(np.vstack([truth, NOM]), net)
    else:
        ident, J = None, None
    if ident is not None:
        _report_ident(ident, J, truth, out_dir, log)

    # B-D in one batch
    grid = np.log(np.array([0.5, 0.63, 0.8, 0.9, 1.0, 1.11, 1.25, 1.6, 2.0]))
    jobs = build_jobs(truth, args.starts, args.prof_teachers, grid, np.random.default_rng(123))
    jobs = {k: v[args.part::args.nparts] for k, v in jobs.items()}
    log.info("part %d/%d: training %d independent single-node models in one batch",
             args.part, args.nparts, len(jobs["kind"]))
    theta, hist_loss, hist_theta, resid, rel = train_all(jobs, truth, net, args.iters, args.lr, args.lr_raw, log)
    np.savez_compressed(out_dir / f"study_part{args.part}of{args.nparts}.npz", theta=theta, hist_loss=hist_loss,
                        hist_theta=hist_theta, resid=resid, rel=rel, truth=truth, grid=grid, amps=AMPS,
                        names=np.array(NAMES), **{f"job_{k}": v for k, v in jobs.items()})
    log.info("saved part %d/%d to %s", args.part, args.nparts, out_dir)


def _report_ident(ident, J, truth, out_dir, log):
    np.save(out_dir / "jacobian.npy", J)
    save_json({"names": NAMES, "amps": AMPS, "sigma_uV": SIGMA, "teachers": ident,
               "truth": truth, "nominal": NOM}, out_dir / "identifiability.json")
    for lab in PROTOCOLS:
        se = np.array([[r[lab]["crb_rel_se"][n] for n in NAMES] for r in ident[:-1]])
        cb = np.array([[r[lab]["combo_rel_se"][c] for c in COMBOS] for r in ident[:-1]])
        log.info("CRB median, %-12s %s | %s", lab, " ".join(f"{n}={100 * v:.2g}%" for n, v in zip(NAMES, np.median(se, 0))),
                 " ".join(f"{c}={100 * v:.2g}%" for c, v in zip(COMBOS, np.median(cb, 0))))



if __name__ == "__main__":
    main()
