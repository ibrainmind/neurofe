"""Phase 1: check that synthetic TEPs look like real TMS-EEG before using them.

Simulates teachers on the full 200-parcel anatomy (nominal parameters and random
draws), averages trials, and compares coarse TEP features with the 20 real subjects
in PyTepFit's data: baseline noise, peak amplitude, GMFA peak and its latency,
and late-to-early GMFA ratio.

Usage (from the repo root):
    .venv/bin/python -m syn_brain.scripts.calibrate_teacher --teachers 3 --sites 2 --trials 100
Outputs: results/calibration/calibration.json, calibration.png; log in logs/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, load_real_eeg, pick_sites, pulse_stimulus, site_weights
from syn_brain.core.params import NOMINAL_RAW, draw_teacher
from syn_brain.core.simulator import SimConfig, evoked
from syn_brain.core.metrics import baseline_correct, gmfa, noise_ceiling, tep_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", type=int, default=3, help="random draws besides nominal")
    ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n-nodes", type=int, default=None, help="default: all 200 parcels")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = get_logger("calibrate_teacher")
    out_dir = RESULTS_DIR / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_real_eeg()
    real_feats = [tep_features(s) for s in real]
    real_summary = {k: {"median": float(np.median([f[k] for f in real_feats])),
                        "min": float(np.min([f[k] for f in real_feats])),
                        "max": float(np.max([f[k] for f in real_feats]))} for k in real_feats[0]}
    log.info("Real TMS-EEG (20 subjects): %s", {k: round(v["median"], 2) for k, v in real_summary.items()})

    anat = load_anatomy(args.n_nodes)
    sites = pick_sites(anat, args.sites, seed=0)
    cfg = SimConfig()
    stim = pulse_stimulus(anat.n_nodes, cfg.hidden_size)

    teachers = [("nominal", dict(NOMINAL_RAW))]
    for s in range(args.teachers):
        teachers.append((f"draw{s}", draw_teacher(anat.n_nodes, seed=s).raw))

    synth = []
    curves = {}
    for name, raw in teachers:
        for site in sites:
            t0 = time.time()
            mean, trials = evoked(raw, anat.sc, anat.dist, anat.lm, site_weights(anat, site), stim,
                                  n_trials=args.trials, cfg=cfg, seed=int(site))
            f = tep_features(mean)
            f["noise_ceiling"] = noise_ceiling(trials)
            f.update(teacher=name, site=int(site), finite=bool(np.isfinite(mean).all()))
            synth.append(f)
            curves[f"{name}_site{site}"] = gmfa(baseline_correct(mean))
            log.info("%s site %d: %.0fs  gmfa peak %.2f uV at %d ms, baseline std %.2f, late/early %.2f, ceiling(full) %.2f",
                     name, site, time.time() - t0, f["gmfa_peak_uV"], f["gmfa_peak_latency_ms"],
                     f["baseline_std_uV"], f["gmfa_late_over_early"], f["noise_ceiling"]["full"])

    save_json({"real_summary": real_summary, "real_per_subject": real_feats, "synthetic": synth,
               "args": vars(args), "sites": sites}, out_dir / "calibration.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_ms = np.arange(400) - 110
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for s in real:
        ax[0].plot(t_ms, gmfa(baseline_correct(s)), color="0.6", lw=0.8)
    ax[0].plot(t_ms, np.median([gmfa(baseline_correct(s)) for s in real], 0), color="k", lw=2, label="median")
    ax[0].set_title("Real TMS-EEG GMFA (20 subjects)")
    for k, c in curves.items():
        ax[1].plot(t_ms, c, lw=1, label=k)
    ax[1].set_title("Synthetic teacher GMFA")
    for a in ax:
        a.axvline(0, color="r", ls=":")
        a.set_xlabel("ms from pulse onset")
        a.set_ylabel("uV")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=120)
    log.info("Saved %s", out_dir)


if __name__ == "__main__":
    main()
