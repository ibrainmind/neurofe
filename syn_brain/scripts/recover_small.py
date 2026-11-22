"""Phase 1 gate: on a 10-node network, does student S1 recover a clean (R0) teacher?

R0 means the teacher has exactly the student's form: same anatomy, plain SC (w_bb = 0),
same stimulus mask, same integration step, trial noise only, no sensor noise.
The student starts from nominal values and must find the teacher's drawn values.

Usage (from the repo root):
    .venv/bin/python -m syn_brain.scripts.recover_small --teacher-seed 0 --epochs 120
Outputs: results/recovery_small/<run>.json and .png; log in logs/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites, pulse_stimulus, site_weights
from syn_brain.core.params import NOMINAL_RAW, draw_teacher, nominal_effective
from syn_brain.core.simulator import SimConfig, evoked
from syn_brain.core.student import build_student, fit_student, predict_evoked
from syn_brain.core.metrics import eeg_fit, noise_ceiling, param_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--n-nodes", type=int, default=10)
    ap.add_argument("--tier", default="S1")
    ap.add_argument("--prior", default="weak", choices=["weak", "pytepfit"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--spread", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = f"n{args.n_nodes}_{args.tier}_{args.prior}_lr{args.lr}_ep{args.epochs}_t{args.teacher_seed}"
    log = get_logger(f"recover_small_{run}")
    out_dir = RESULTS_DIR / "recovery_small"
    out_dir.mkdir(parents=True, exist_ok=True)

    anat = load_anatomy(args.n_nodes)
    site = int(pick_sites(anat, 1, seed=args.teacher_seed)[0])
    ki = site_weights(anat, site)
    cfg = SimConfig()
    stim = pulse_stimulus(anat.n_nodes, cfg.hidden_size)

    tp = draw_teacher(anat.n_nodes, seed=args.teacher_seed, spread=args.spread)
    t0 = time.time()
    teach_mean, teach_trials = evoked(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim,
                                      n_trials=args.trials, cfg=cfg, seed=100 + args.teacher_seed)
    teach_clean, _ = evoked(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim, n_trials=1,
                            cfg=cfg, noise_scale=0.0, seed=0)
    log.info("teacher %d (site %d) simulated in %.1fs; truth: %s", args.teacher_seed, site,
             time.time() - t0, {k: round(v, 3) for k, v in tp.effective.items()})

    model = build_student(anat, ki, tier=args.tier, prior=args.prior)
    eeg_train = np.broadcast_to(teach_mean, (args.epochs,) + teach_mean.shape).copy()
    fit = fit_student(model, eeg_train, epochs=args.epochs, lr=args.lr, stimulus=stim, log=log)

    rec = param_recovery(fit["fitted"], tp.effective, nominal_effective())
    for n, r in rec.items():
        log.info("  %-6s truth %9.3f  start %9.3f  fitted %9.3f  rel.err %+6.1f%%  progress %+5.2f",
                 n, r["truth"], r["start"], r["fitted"], 100 * r["rel_error"], r["progress"])

    stud_mean, _ = predict_evoked(model, anat, ki, stim, n_trials=args.trials, seed=7)
    stud_clean, _ = predict_evoked(model, anat, ki, stim, n_trials=1, noise_scale=0.0)
    fit_avg = eeg_fit(stud_mean, teach_mean)
    fit_clean = eeg_fit(stud_clean, teach_clean)
    nominal_mean, _ = evoked(dict(NOMINAL_RAW), anat.sc, anat.dist, anat.lm, ki, stim, n_trials=args.trials, cfg=cfg, seed=7)
    fit_nominal = eeg_fit(nominal_mean, teach_mean)
    ceiling = noise_ceiling(teach_trials)
    log.info("EEG fit (student avg vs teacher avg): %s", {k: round(v["pearson_mean"], 3) for k, v in fit_avg.items()})
    log.info("EEG fit (untrained nominal model):    %s", {k: round(v["pearson_mean"], 3) for k, v in fit_nominal.items()})
    log.info("rel. RMSE student %.3f vs nominal %.3f (full window)", fit_avg["full"]["rel_rmse"], fit_nominal["full"]["rel_rmse"])

    abs_err = np.array([abs(r["rel_error"]) for r in rec.values()])
    summary = {
        "run": run, "args": vars(args), "site": site,
        "truth_effective": tp.effective, "fitted_effective": fit["fitted"],
        "recovery": rec,
        "n_within_20pct": int((abs_err <= 0.20).sum()), "n_params": len(abs_err),
        "median_progress": float(np.nanmedian([r["progress"] for r in rec.values()])),
        "eeg_fit_avg": fit_avg, "eeg_fit_clean": fit_clean, "eeg_fit_nominal_baseline": fit_nominal,
        "noise_ceiling": ceiling,
        "fit_seconds": fit["seconds"], "sec_per_epoch": fit["sec_per_epoch"],
        "rmse_per_epoch": fit["rmse_per_epoch"],
    }
    save_json(summary, out_dir / f"{run}.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_ms = np.arange(400) - 110
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t_ms, teach_mean.T - teach_mean[:, :100].mean(1), color="k", lw=0.6)
    ax[0].set_title("Teacher (truth), trial average")
    ax[1].plot(t_ms, stud_mean.T - stud_mean[:, :100].mean(1), color="C0", lw=0.6)
    ax[1].set_title(f"Student {args.tier} after {args.epochs} epochs")
    names = list(rec)
    ax[2].bar(names, [rec[n]["progress"] for n in names])
    ax[2].axhline(1, color="g", ls="--", lw=1)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Progress from start to truth (1 = recovered)")
    ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run}.png", dpi=110)
    log.info("saved %s", out_dir / run)


if __name__ == "__main__":
    main()
