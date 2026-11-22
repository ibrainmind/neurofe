"""Single-node virtual-TMS system identification.

A *teacher* JR node with neurofe's nominal parameters is stimulated with a set of
virtual-TMS protocols. A *student* JR node of the same form sees only the teacher's EEG
and is fitted by backpropagation (Adam), as neurofe does. The question is how the
choice of stimulus decides which parameters can be learned.

Sub-commands
  analyse   local identifiability (Fisher information / Cramer-Rao bounds) per protocol,
            for the 7 parameters neurofe fits and for 12 (adding A, B, v0, r, vmax);
            elasticities of input-output features; raw-parameter gradient scales.
            Fast (~1 min).
  train     batched backprop fits. Experiments:
              E1  every protocol x random starts, log-scaled parameters, noise-free
              E2  neurofe-style raw parameters (relu floors, Adam lr 0.05)
              E3  12 free parameters (neurofe's fixed ones released)
              E4  sensor noise at the real TMS-EEG level (0.42 uV)
              E5  profile likelihood: one parameter pinned off-truth, others refitted
              E6  4 perturbed teachers (robustness of E1)
              E7  driven operating point (tonic background drive p=110)
              E8  temporal hold-out: fit 0-60 ms after the pulse, predict the late response
            Split across processes with --part/--nparts.

Usage (repo root):
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms analyse
    for p in 0 1 2 3; do PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms train --part $p --nparts 4 & done; wait
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms_report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites
from syn_brain.core.ladder import REAL_BASELINE_STD_UV
from syn_brain.core.params import OFFSET
from syn_brain.core import node

OUT = RESULTS_DIR / "single_node_tms"
SIGMA = REAL_BASELINE_STD_UV
FIT = node.FIT_NAMES
ALL = node.ALL_NAMES
NOM_FIT = np.array([node.NOMINAL[n] for n in FIT])
NOM_ALL = np.array([node.NOMINAL[n] for n in ALL])

# Protocols: which stimulation conditions the experimenter records.
PROTOCOLS = {s: [s] for s in node.STIM_ORDER}
PROTOCOLS["DOSE"] = ["SP250", "SP1000", "SP4000"]
PROTOCOLS["RICH"] = ["SP250", "SP1000", "SP4000", "PP100", "TR10", "EI1000"]
PROTOCOLS["ALL"] = list(node.STIM_ORDER)
PROTOCOLS["SP1000@p110"] = ["SP1000@p110"]
PROTOCOLS["DOSE@p110"] = ["SP250@p110", "SP1000@p110", "SP4000@p110"]
E1_PROTOCOLS = list(node.STIM_ORDER) + ["DOSE", "RICH"]


def readout():
    anat = load_anatomy(10)
    site = int(pick_sites(anat, 1, seed=0)[0])
    return node.readout_gain(anat.lm[:, site]), int(anat.parcels[site])


def draw_teachers(n, spread=0.3, seed=1000):
    rng = np.random.default_rng(seed)
    return NOM_FIT * np.exp(rng.uniform(np.log(1 - spread), np.log(1 + spread), (n, len(FIT))))


# =============================================================================== analyse
def features(y: np.ndarray) -> dict:
    """Input-output features from the 9 condition responses y (9, T) of one model."""
    Y = {s: y[i] for i, s in enumerate(node.STIM_ORDER)}
    on = node.ONSET

    def peak(x, lo, hi, sign=1):
        seg = sign * x[lo:hi]
        i = int(np.argmax(seg))
        if 0 < i < len(seg) - 1:            # parabolic interpolation for sub-ms latency
            d = seg[i - 1] - 2 * seg[i] + seg[i + 1]
            off = 0.5 * (seg[i - 1] - seg[i + 1]) / d if d != 0 else 0.0
            val = seg[i] - 0.25 * (seg[i - 1] - seg[i + 1]) * off
        else:
            off, val = 0.0, seg[i]
        return sign * val, lo + i + off - on

    x = Y["SP1000"]
    p_amp, p_lat = peak(x, on, on + 80)
    n_amp, n_lat = peak(x, on + int(p_lat), node.T_WIN, -1)
    post = x[on + int(p_lat):]
    zc = np.where(post < 0)[0]
    zc = (zc[0] + p_lat) if len(zc) else np.nan
    if len(np.where(post < 0)[0]):
        j = np.where(post < 0)[0][0]
        zc = p_lat + j - 1 + post[j - 1] / (post[j - 1] - post[j])
    dose = peak(Y["SP4000"], on, on + 80)[0] / peak(Y["SP250"], on, on + 80)[0]
    single = Y["SP1000"]

    def ppr(name, isi):
        resid = Y[name] - single                       # response attributable to pulse 2
        return peak(resid, on + isi, on + isi + 80)[0] / p_amp
    return {
        "P1 amplitude": p_amp, "P1 latency": p_lat,
        "N1 amplitude": -n_amp, "N1 latency": n_lat,
        "zero-crossing": zc,
        "dose ratio 4000/250": dose / 16.0,
        "paired-pulse 30 ms": ppr("PP30", 30), "paired-pulse 100 ms": ppr("PP100", 100),
        "E+I / E-only": peak(Y["EI1000"], on, on + 80)[0] / p_amp,
        "2 ms / 30 ms pulse": peak(Y["SPshort"], on, on + 80)[0] / peak(Y["SPlong"], on, on + 80)[0],
    }


def fisher_at(points: np.ndarray, gain: float, h=1e-4):
    """Central-difference Jacobians in log-parameter space for every point and condition.

    points: (Q, 12) effective values. Returns y (Q, S, T) and J (Q, S, T, 12).
    """
    Q, P = points.shape
    thetas = [points]
    for i in range(P):
        for s in (+1, -1):
            t = points.copy()
            t[:, i] *= np.exp(s * h)
            thetas.append(t)
    th = np.concatenate(thetas)
    stims = [node.STIMS[s] for s in node.STIM_ORDER + node.DRIVEN_ORDER]
    with torch.no_grad():
        y = node.simulate({n: torch.tensor(th[:, i]) for i, n in enumerate(ALL)}, stims, gain=gain).numpy()
    y = y.reshape(1 + 2 * P, Q, len(stims), -1)
    J = np.stack([(y[1 + 2 * i] - y[2 + 2 * i]) / (2 * h) for i in range(P)], -1)
    return y[0], J


def crb(J_rows: np.ndarray, sigma: float, free: list[int]):
    """J_rows: (..., P) stacked over conditions and time. Returns stats for the free subset."""
    Jf = J_rows.reshape(-1, J_rows.shape[-1])[:, free]
    F = Jf.T @ Jf / sigma ** 2
    w, V = np.linalg.eigh(F)
    cov = np.linalg.pinv(F, rcond=1e-15)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"F": F, "se": se, "eig": w, "V": V, "corr": cov / np.outer(se, se),
            "cond": float(w[-1] / max(w[0], 1e-300))}


def analyse(log):
    OUT.mkdir(parents=True, exist_ok=True)
    gain, parcel = readout()
    teachers = draw_teachers(8)
    pts = np.vstack([NOM_ALL, np.hstack([teachers, np.tile(NOM_ALL[len(FIT):], (len(teachers), 1))])])
    t0 = time.time()
    y, J = fisher_at(pts, gain)
    log.info("Jacobians for %d points x %d conditions in %.0fs", len(pts), J.shape[1], time.time() - t0)
    np.savez_compressed(OUT / "jacobian.npz", y=y, J=J, points=pts, names=np.array(ALL),
                        stims=np.array(node.STIM_ORDER), gain=gain, parcel=parcel)

    res = {"gain": gain, "parcel": parcel, "sigma_uV": SIGMA, "names_fit": FIT, "names_all": ALL,
           "protocols": PROTOCOLS, "stim_labels": {s: node.STIMS[s].label for s in node.STIM_ORDER},
           "points": pts, "crb": {}}
    fit_idx = list(range(len(FIT)))
    all_idx = list(range(len(ALL)))
    order = node.STIM_ORDER + node.DRIVEN_ORDER
    for pname, conds in PROTOCOLS.items():
        ci = [order.index(c) for c in conds]
        per_point = []
        for q in range(len(pts)):
            r7 = crb(J[q, ci], SIGMA, fit_idx)
            r12 = crb(J[q, ci], SIGMA, all_idx)
            per_point.append({"se7": r7["se"], "eig7": r7["eig"], "V7": r7["V"], "corr7": r7["corr"],
                              "cond7": r7["cond"], "se12": r12["se"], "eig12": r12["eig"], "V12": r12["V"],
                              "corr12": r12["corr"], "cond12": r12["cond"]})
        res["crb"][pname] = per_point
        log.info("%-8s CRB rel SE at nominal (7 free): %s | cond %.1e", pname,
                 " ".join(f"{n}={100 * v:.3g}%" for n, v in zip(FIT, per_point[0]["se7"])), per_point[0]["cond7"])

    # feature elasticities d log|f| / d log theta, at nominal, 5% central differences
    h = 0.05
    th = [NOM_ALL]
    for i in range(len(ALL)):
        for s in (+1, -1):
            t = NOM_ALL.copy(); t[i] *= np.exp(s * h); th.append(t)
    th = np.array(th)
    with torch.no_grad():
        yy = node.simulate({n: torch.tensor(th[:, i]) for i, n in enumerate(ALL)},
                           [node.STIMS[s] for s in node.STIM_ORDER], gain=gain).numpy()
    F0 = features(yy[0])
    el = np.zeros((len(F0), len(ALL)))
    for i in range(len(ALL)):
        fp, fm = features(yy[1 + 2 * i]), features(yy[2 + 2 * i])
        for j, f in enumerate(F0):
            el[j, i] = (np.log(abs(fp[f])) - np.log(abs(fm[f]))) / (2 * h)
    res["features_nominal"] = F0
    res["elasticity"] = el
    res["feature_names"] = list(F0)
    res["y_nominal"] = yy[0]
    for f, v in F0.items():
        log.info("feature %-22s %8.3f | elasticities %s", f, v,
                 " ".join(f"{n}={e:+.2f}" for n, e in zip(ALL, el[list(F0).index(f)])))

    # one-at-a-time responses (+-30%) for the response gallery
    sweep = {}
    for i, n in enumerate(ALL):
        t = np.tile(NOM_ALL, (2, 1)); t[0, i] *= 0.7; t[1, i] *= 1.3
        with torch.no_grad():
            sweep[n] = node.simulate({m: torch.tensor(t[:, j]) for j, m in enumerate(ALL)},
                                     [node.STIMS[s] for s in node.STIM_ORDER], gain=gain).numpy()
    np.savez_compressed(OUT / "sweep.npz", **sweep)

    # neurofe raw parameterisation: relative Adam step and gradient scale
    res["raw_offset"] = [OFFSET[n] for n in FIT]
    res["adam_rel_step_lr005"] = (0.05 / NOM_FIT).tolist()
    g_log = np.sqrt((J[0, node.STIM_ORDER.index("SP1000"), :, :len(FIT)] ** 2).mean(0))   # dy/dlog(theta)
    res["grad_rms_log"] = g_log.tolist()
    res["grad_rms_raw"] = (g_log / NOM_FIT).tolist()                                      # dy/dtheta
    res["tonic_sweep"] = tonic_sweep(gain, log)
    save_json(res, OUT / "analysis.json")
    log.info("saved %s", OUT / "analysis.json")


def tonic_sweep(gain, log, ps=tuple(range(0, 241, 10))):
    """CRB at nominal for SP1000 and DOSE as the background drive p is raised."""
    import dataclasses
    out = {"p": list(ps), "SP1000": [], "DOSE": [], "peak": [], "late": [], "pre_std": [], "y": []}
    h, P = 1e-4, len(ALL)
    th = [NOM_ALL]
    for i in range(P):
        for s in (+1, -1):
            t = NOM_ALL.copy(); t[i] *= np.exp(s * h); th.append(t)
    th = np.array(th)
    for p in ps:
        st = [dataclasses.replace(node.STIMS[s], tonic=float(p)) for s in ("SP250", "SP1000", "SP4000")]
        with torch.no_grad():
            y = node.simulate({n: torch.tensor(th[:, i]) for i, n in enumerate(ALL)}, st, settle_ms=1500,
                              gain=gain).numpy()
        J = np.stack([(y[1 + 2 * i] - y[2 + 2 * i]) / (2 * h) for i in range(P)], -1)     # (S, T, P)
        out["SP1000"].append(crb(J[1], SIGMA, list(range(len(FIT))))["se"])
        out["DOSE"].append(crb(J, SIGMA, list(range(len(FIT))))["se"])
        y0 = y[0, 1]
        out["peak"].append(float(np.abs(y0).max())); out["late"].append(float(np.abs(y0[260:]).max()))
        out["pre_std"].append(float(y0[:100].std())); out["y"].append(y0)
        log.info("tonic p=%3d  pre-std %.3f uV | CRB SP1000 %s", p, out["pre_std"][-1],
                 " ".join(f"{n}={100 * v:.3g}%" for n, v in zip(FIT, out["SP1000"][-1])))
    return out


# =============================================================================== train
def build_jobs(n_starts: int, rng):
    """Every independent model to train. A job = (experiment, teacher, protocol, start, ...)."""
    jobs = []
    teachers = {"nominal": NOM_ALL.copy()}
    for i, t in enumerate(draw_teachers(4, seed=2000)):
        teachers[f"T{i}"] = np.concatenate([t, NOM_ALL[len(FIT):]])

    def starts(n, n_par, spread=0.5):
        return rng.uniform(np.log(1 - spread), np.log(1 + spread), (n, n_par))

    st7 = starts(n_starts, len(FIT))
    st12 = np.hstack([st7, starts(n_starts, len(ALL) - len(FIT), 0.3)])
    for p in E1_PROTOCOLS:
        for s in range(n_starts):
            jobs.append(dict(exp="E1", teacher="nominal", protocol=p, start=s, z0=st7[s], free="7", param="log", noise=0.0))
    for p in ("SP1000", "RICH"):
        for s in range(n_starts):
            jobs.append(dict(exp="E2", teacher="nominal", protocol=p, start=s, z0=st7[s], free="7", param="raw", noise=0.0))
            jobs.append(dict(exp="E3", teacher="nominal", protocol=p, start=s, z0=st12[s], free="12", param="log", noise=0.0))
    for p in ("SP1000", "DOSE", "RICH"):
        for s in range(n_starts):
            jobs.append(dict(exp="E4", teacher="nominal", protocol=p, start=s, z0=st7[s], free="7", param="log", noise=SIGMA))
    grid = np.log([0.5, 0.67, 0.8, 0.9, 1.11, 1.25, 1.5, 2.0])
    for p in ("SP1000", "RICH"):
        for i in range(len(FIT)):
            for gz in grid:
                z0 = np.zeros(len(FIT)); z0[i] = gz
                jobs.append(dict(exp="E5", teacher="nominal", protocol=p, start=0, z0=z0, free="7", param="log",
                                 noise=0.0, pin=i, pin_z=gz))
    for p in ("SP1000@p110", "DOSE@p110"):
        for s in range(n_starts):
            jobs.append(dict(exp="E7", teacher="nominal", protocol=p, start=s, z0=st7[s], free="7", param="log", noise=0.0))
    for tname in ("T0", "T1", "T2", "T3"):
        for p in ("SP1000", "DOSE", "RICH"):
            for s in range(n_starts // 2):
                z0 = st7[s] - np.log(teachers[tname][:len(FIT)] / NOM_FIT)   # same absolute starts, relative to truth
                jobs.append(dict(exp="E6", teacher=tname, protocol=p, start=s, z0=z0, free="7", param="log", noise=0.0))
    # E8: temporal hold-out -- fit only up to EARLY_MS after the (first) pulse, predict the rest
    for p in ("SP1000", "DOSE", "RICH"):
        for s in range(n_starts):
            jobs.append(dict(exp="E8", teacher="nominal", protocol=p, start=s, z0=st7[s], free="7", param="log",
                             noise=0.0, window="early"))
    for j in jobs:
        j.setdefault("pin", -1); j.setdefault("pin_z", 0.0); j.setdefault("window", "full")
    return jobs, teachers


EARLY_MS = 60


def train(args, log):
    OUT.mkdir(parents=True, exist_ok=True)
    gain, _ = readout()
    jobs, teachers = build_jobs(args.starts, np.random.default_rng(7))
    if args.only:
        jobs = [j for j in jobs if j["exp"] in args.only.split(",")]
    jobs = jobs[args.part::args.nparts]
    M = len(jobs)
    order = node.STIM_ORDER + node.DRIVEN_ORDER
    stims = [node.STIMS[s] for s in order]

    # teacher responses for every condition (noise-free), and per-job noisy copies
    tnames = list(teachers)
    tmat = np.array([teachers[t] for t in tnames])
    with torch.no_grad():
        ty = node.simulate({n: torch.tensor(tmat[:, i]) for i, n in enumerate(ALL)}, stims, gain=gain)   # (nT, S, T)

    rows_job, rows_stim, rows_cond = [], [], []
    for j, job in enumerate(jobs):
        for c in PROTOCOLS[job["protocol"]]:
            rows_job.append(j); rows_stim.append(node.STIMS[c]); rows_cond.append(order.index(c))
    rows_job = np.array(rows_job); rows_cond = np.array(rows_cond)
    tidx = np.array([tnames.index(jobs[j]["teacher"]) for j in rows_job])
    target = ty[tidx, rows_cond].clone()                                               # (R, T)
    gen = torch.Generator().manual_seed(1234 + args.part)
    noise = torch.tensor([jobs[j]["noise"] for j in rows_job], dtype=torch.float64)
    target = target + noise[:, None] * torch.randn(target.shape, generator=gen, dtype=torch.float64)
    target = target - target[:, :node.PRE].mean(-1, keepdim=True)
    R = len(rows_job)
    tmask = torch.ones((R, node.T_WIN), dtype=torch.float64)
    for r, j in enumerate(rows_job):
        if jobs[j]["window"] == "early":
            tmask[r, node.ONSET + EARLY_MS:] = 0.0
    rj = torch.tensor(rows_job)
    n_rows = torch.zeros(M, dtype=torch.float64).index_add_(0, rj, torch.ones(R, dtype=torch.float64))
    power = torch.zeros(M, dtype=torch.float64).index_add_(
        0, rj, (ty[tidx, rows_cond].pow(2) * tmask).sum(-1) / tmask.sum(-1)) / n_rows

    truth = np.array([teachers[job["teacher"]] for job in jobs])                     # (M, 12)
    z0 = np.zeros((M, len(ALL)))
    free = np.zeros((M, len(ALL)))
    for j, job in enumerate(jobs):
        k = 7 if job["free"] == "7" else 12
        z0[j, :k] = job["z0"][:k]
        free[j, :k] = 1
        if job["pin"] >= 0:
            free[j, job["pin"]] = 0
            z0[j, job["pin"]] = job["pin_z"]
    truth_t = torch.tensor(truth)
    free_t = torch.tensor(free)
    z = torch.tensor(z0).requires_grad_(True)
    is_raw = torch.tensor([job["param"] == "raw" for job in jobs])
    off = torch.tensor([OFFSET[n] for n in FIT], dtype=torch.float64)
    raw = (truth_t[:, :len(FIT)] * torch.exp(torch.tensor(z0[:, :len(FIT)])) - off).clone().requires_grad_(True)
    opt_log = torch.optim.Adam([z], lr=args.lr)
    opt_raw = torch.optim.Adam([raw], lr=args.lr_raw)                                 # neurofe: constant lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_log, args.iters, eta_min=0.03 * args.lr)

    def current_theta():
        th_log = truth_t * torch.exp(z * free_t + torch.tensor(z0) * (1 - free_t))
        th_raw = torch.cat([off + torch.relu(raw), truth_t[:, len(FIT):]], 1)
        return torch.where(is_raw.unsqueeze(1), th_raw, th_log)

    snap_every = max(1, args.iters // 60)
    hist_loss, hist_theta, hist_it = [], [], []
    log.info("part %d/%d: %d models, %d simulated rows", args.part, args.nparts, M, R)
    t0 = time.time()
    for it in range(args.iters):
        theta = current_theta()
        y = node.simulate_rows({n: theta[:, i] for i, n in enumerate(ALL)}, rows_job, rows_stim, gain=gain)
        sse = ((y - target).pow(2) * tmask).sum(-1) / tmask.sum(-1)
        loss = torch.zeros(M, dtype=torch.float64).index_add(0, rj, sse) / n_rows / power
        bad = ~torch.isfinite(loss)
        opt_log.zero_grad(); opt_raw.zero_grad()
        torch.where(bad, torch.zeros_like(loss), loss).sum().backward()
        for p in (z, raw):
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt_log.step(); opt_raw.step(); sched.step()
        with torch.no_grad():
            z.clamp_(-3, 3)
        if it % snap_every == 0 or it == args.iters - 1:
            hist_loss.append(loss.detach().numpy().copy())
            hist_theta.append(theta.detach().numpy().copy())
            hist_it.append(it)
        if it % 25 == 0 or it == args.iters - 1:
            l = loss.detach().numpy()
            log.info("it %4d  median rel loss %.2e  (non-finite %d)  %.0fs", it, np.nanmedian(l), int(bad.sum()), time.time() - t0)

    theta = current_theta().detach()
    with torch.no_grad():
        y = node.simulate({n: theta[:, i] for i, n in enumerate(ALL)}, stims, gain=gain)   # (M, S, T)
    tgt = ty[[tnames.index(job["teacher"]) for job in jobs]]
    rel = ((y - tgt).pow(2).sum(-1) / tgt.pow(2).sum(-1)).sqrt().numpy()                   # (M, S)
    tag = f"_{args.only.replace(',', '-')}" if args.only else ""
    np.savez_compressed(OUT / f"train{tag}_part{args.part}of{args.nparts}.npz",
                        theta=theta.numpy(), truth=truth, rel=rel, pred=y.numpy().astype(np.float32),
                        hist_loss=np.array(hist_loss), hist_theta=np.array(hist_theta), hist_it=np.array(hist_it),
                        jobs=np.array([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in job.items()}
                                       for job in jobs], dtype=object))
    log.info("saved part %d (%.0f min)", args.part, (time.time() - t0) / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyse", "train"])
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--lr-raw", type=float, default=0.05, help="neurofe's learning rate")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated experiments to train (default: all)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "analyse":
        analyse(get_logger("sn_tms_analyse"))
    else:
        train(args, get_logger(f"sn_tms_train_part{args.part}of{args.nparts}"))


if __name__ == "__main__":
    main()
