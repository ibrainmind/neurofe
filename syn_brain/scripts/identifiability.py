"""Step 1: which parameters can a TEP experiment identify at all? (local, Fisher information)

For a teacher, compute the Jacobian J = d EEG / d z of the noise-free, baseline-corrected
EEG with respect to log-scaled parameters z (theta = theta0 * exp(z)), by central finite
differences in float64. With white sensor noise of std sigma, the Fisher information is
F = J^T J / sigma^2 and the Cramer-Rao bound sqrt(diag(F^-1)) is the smallest achievable
*relative* standard error of each parameter for any unbiased estimator.

All conditions (sites x amplitudes) run in one batch, so each experiment design
("protocol") is just a subset of rows of J.

The noise-free model has no dependence on std_in, so std_in is excluded here.
Delays are fractional (mu differentiable); with neurofe's integer delays the mu column
would be exactly zero.

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.identifiability --teacher nominal
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.identifiability --teacher 0 --spread 0.5
Outputs: results/identifiability/<teacher>.json and .npz (Jacobian).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites, site_weights
from syn_brain.core.params import draw_teacher, nominal_effective
from syn_brain.core.ladder import REAL_BASELINE_STD_UV
from syn_brain.core import diffsim as ds

NAMES = tuple(n for n in ds.PARAM_NAMES if n != "std_in")
AMPS = (500.0, 1000.0, 2000.0)
N_SITES = 8


def protocols(n_sites=N_SITES, amps=AMPS):
    """name -> list of (site_index, amp_index) rows, and an optional time window."""
    ia = amps.index(1000.0)
    full = (100, 400)       # everything after baseline (pulse at 110)
    P = {
        "1 site": ([(0, ia)], full),
        "1 site, early 150 ms only": ([(0, ia)], (100, 260)),
        "1 site, 3 amplitudes": ([(0, j) for j in range(len(amps))], full),
        "2 sites": ([(s, ia) for s in range(2)], full),
        "4 sites": ([(s, ia) for s in range(4)], full),
        "8 sites": ([(s, ia) for s in range(n_sites)], full),
        "8 sites, 3 amplitudes": ([(s, j) for s in range(n_sites) for j in range(len(amps))], full),
    }
    return P


def analyse(J, names, sigma):
    """J: (n_obs, n_par). Returns CRB (relative SE), eigen-spectrum, correlations."""
    F = J.T @ J / sigma ** 2
    w, V = np.linalg.eigh(F)
    cov = np.linalg.pinv(F, rcond=1e-15)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(se, se)
    # Column-normalised condition: does the *direction* information exist at all?
    return {
        "crb_rel_se": dict(zip(names, se.tolist())),
        "eigvals": w.tolist(),
        "condition_number": float(w[-1] / max(w[0], 1e-300)),
        "weakest_direction": dict(zip(names, V[:, 0].tolist())),
        "second_weakest_direction": dict(zip(names, V[:, 1].tolist())),
        "corr": corr.tolist(),
        "sensitivity_rms": dict(zip(names, np.sqrt((J ** 2).mean(0)).tolist())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="nominal", help="'nominal' or an integer seed")
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--n-nodes", type=int, default=10)
    ap.add_argument("--h", type=float, default=1e-4, help="finite-difference step in log space")
    ap.add_argument("--sigma", type=float, default=REAL_BASELINE_STD_UV)
    ap.add_argument("--coupling-mode", default="laplacian", choices=["laplacian", "non_diffusive"],
                    help="laplacian = neurofe's zero-mean coupling (default); non_diffusive = "
                         "g*sum w_ij*S(E_j), gives nodes a non-zero mean drive (network_study_plan.txt N0.1)")
    ap.add_argument("--g-override", type=float, default=None,
                    help="use this g instead of the teacher's own value -- nominal g=1000 is "
                         "calibrated for the self-cancelling Laplacian and saturates U_BOUND "
                         "under non_diffusive coupling (network_study_plan.txt N0.2)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = f"t{args.teacher}" if args.teacher != "nominal" else "nominal"
    tag += "" if args.coupling_mode == "laplacian" else f"_{args.coupling_mode}"
    tag += "" if args.g_override is None else f"_g{args.g_override:g}"
    log = get_logger(f"identifiability_{tag}")
    out_dir = RESULTS_DIR / "identifiability"
    out_dir.mkdir(parents=True, exist_ok=True)

    anat = load_anatomy(args.n_nodes)
    if args.teacher == "nominal":
        truth = nominal_effective()
    else:
        truth = draw_teacher(anat.n_nodes, seed=int(args.teacher), spread=args.spread).effective
    if args.g_override is not None:
        truth = dict(truth); truth["g"] = args.g_override
    sites = pick_sites(anat, N_SITES, seed=0)
    ki = np.stack([site_weights(anat, int(s))[:, 0] for s in sites])
    # rows ordered (site, amp)
    proto = ds.Protocol(ki=np.repeat(ki, len(AMPS), axis=0), amp=np.tile(np.array(AMPS), N_SITES))
    net = ds.Net.from_anatomy(anat)
    base = {n: float(v) for n, v in truth.items()}

    def eeg_at(z_vec):
        z = torch.zeros(len(ds.PARAM_NAMES), dtype=torch.float64)
        for i, n in enumerate(NAMES):
            z[ds.PARAM_NAMES.index(n)] = z_vec[i]
        with torch.no_grad():
            e = ds.run(ds.theta_from_z(z, base=base), net, proto, noise_scale=0.0,
                      coupling_mode=args.coupling_mode)
        return ds.baseline_correct(e[:, 0]).numpy()          # (S*A, C, T)

    t0 = time.time()
    y0 = eeg_at(np.zeros(len(NAMES)))
    cols = []
    for i, n in enumerate(NAMES):
        dz = np.zeros(len(NAMES))
        dz[i] = args.h
        cols.append((eeg_at(dz) - eeg_at(-dz)) / (2 * args.h))
        log.info("column %s done (%.0fs)", n, time.time() - t0)
    Jfull = np.stack(cols, axis=-1)                           # (S*A, C, T, P)
    np.savez_compressed(out_dir / f"{tag}.npz", J=Jfull, y0=y0, names=np.array(NAMES),
                        sites=sites, amps=np.array(AMPS))

    res = {"teacher": tag, "truth": truth, "sigma_uV": args.sigma, "sites": sites,
           "coupling_mode": args.coupling_mode, "amps": AMPS, "names": NAMES,
           "peak_abs_uV": float(np.abs(y0).max()), "protocols": {}}
    for pname, (rows, (t_a, t_b)) in protocols().items():
        idx = [s * len(AMPS) + j for s, j in rows]
        J = Jfull[idx][:, :, t_a:t_b].reshape(-1, len(NAMES))
        res["protocols"][pname] = analyse(J, NAMES, args.sigma)
        se = res["protocols"][pname]["crb_rel_se"]
        log.info("%-28s cond %.1e  CRB rel.SE: %s", pname, res["protocols"][pname]["condition_number"],
                 " ".join(f"{n}={100 * v:.2g}%" for n, v in se.items()))
    save_json(res, out_dir / f"{tag}.json")
    log.info("saved %s", out_dir / tag)


if __name__ == "__main__":
    main()
