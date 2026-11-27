"""Step 2 gate: does a clean backprop fit recover a noise-free teacher on the 10-node network?

Teacher and student share the exact model (core/diffsim.py). The teacher's parameters are
drawn within +/- spread of nominal; the student starts at nominal (or a random point,
--init-seed) and fits the 9 parameters the noise-free EEG depends on.

After fitting, it predicts every condition of the full 8-site x 3-amplitude design,
so conditions not used for training measure generalisation (Step 4).

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.recover_clean --teacher-seed 0 --sites 1
Ablation flags (Step 3): --integer-delays, --grad-warmup -1 (no gradient through warm-up).
Outputs: results/recovery_clean/<run>.json and .png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites, site_weights
from syn_brain.core.params import draw_teacher
from syn_brain.core import diffsim as ds
from syn_brain.core.fitting import fit

NAMES = tuple(n for n in ds.PARAM_NAMES if n != "std_in")
ALL_AMPS = (500.0, 1000.0, 2000.0)


def rel_rmse(sim, ref, window=(100, 400)):
    s = ds.baseline_correct(sim)[..., window[0]:window[1]]
    r = ds.baseline_correct(ref)[..., window[0]:window[1]]
    return ((s - r).pow(2).mean((-1, -2)) / r.pow(2).mean((-1, -2))).sqrt().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--n-nodes", type=int, default=10)
    ap.add_argument("--sites", type=int, default=1, help="number of training sites (of 8)")
    ap.add_argument("--amps", type=float, nargs="+", default=[1000.0], help="training amplitudes")
    ap.add_argument("--init-seed", type=int, default=-1, help="-1: start at nominal; else random start")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--integer-delays", action="store_true")
    ap.add_argument("--coupling-mode", default="laplacian", choices=["laplacian", "non_diffusive"],
                     help="TEACHER's coupling mode -- the actual data-generating process")
    ap.add_argument("--student-coupling-mode", default=None, choices=["laplacian", "non_diffusive"],
                     help="STUDENT's assumed coupling mode when fitting; defaults to match "
                          "--coupling-mode (the usual case). Set differently to test model "
                          "misspecification robustness -- e.g. a non_diffusive (bounded, more "
                          "physiologically plausible) teacher fit by a laplacian student, matching "
                          "what neurofe's real framework actually assumes. NOTE: g is not a shared "
                          "quantity across coupling modes (different coupling functions entirely) "
                          "-- when mismatched, g's 'truth'/'rel_error' in the output is NOT a "
                          "recovery metric, only the student's own fitted value for inspection; "
                          "a, b, c1-c4, mu, k ARE node-intrinsic and still have a shared ground truth.")
    ap.add_argument("--g-override", type=float, default=None,
                    help="use this g instead of the teacher's own drawn value (see identifiability.py "
                         "N0.2 note: nominal g saturates U_BOUND under non_diffusive coupling)")
    ap.add_argument("--grad-warmup", type=int, default=200, help="ms of warm-up differentiated; -1 = none")
    ap.add_argument("--amp-scale-by-k", action="store_true",
                     help="scale every stimulus amplitude by NOMINAL_EFF['k']/teacher['k'], so a "
                          "teacher with below-nominal stimulus gain gets a stronger pulse instead of "
                          "a weaker effective drive -- see DECISIONS.md 2026-11-24 (per-teacher k "
                          "correlates with recovery failure, same mechanism as single-node's "
                          "per-teacher-p calibration)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    student_coupling_mode = args.student_coupling_mode or args.coupling_mode
    mismatched = student_coupling_mode != args.coupling_mode
    amps_tag = "-".join(str(int(a)) for a in args.amps)
    run = (f"n{args.n_nodes}_s{args.sites}_a{amps_tag}_init{args.init_seed}"
           f"{'_intdel' if args.integer_delays else ''}{'_nowarmgrad' if args.grad_warmup < 0 else ''}"
           f"{'_' + args.coupling_mode if args.coupling_mode != 'laplacian' else ''}"
           f"{'_stu' + student_coupling_mode if mismatched else ''}"
           f"{'_g' + f'{args.g_override:g}' if args.g_override is not None else ''}"
           f"{'_kscale' if args.amp_scale_by_k else ''}"
           f"{args.tag}_t{args.teacher_seed}")
    log = get_logger(f"recover_clean_{run}")
    out_dir = RESULTS_DIR / "recovery_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    anat = load_anatomy(args.n_nodes)
    net = ds.Net.from_anatomy(anat)
    sites = pick_sites(anat, 8, seed=0)
    ki_all = np.stack([site_weights(anat, int(s))[:, 0] for s in sites])
    truth = draw_teacher(anat.n_nodes, seed=args.teacher_seed, spread=args.spread).effective
    if args.g_override is not None:
        truth = dict(truth); truth["g"] = args.g_override
    truth_z = np.array([np.log(truth[n] / ds.NOMINAL_EFF[n]) for n in NAMES])
    th_truth = {n: torch.tensor(truth[n], dtype=torch.float64) for n in ds.PARAM_NAMES}

    # Only ever boost amplitude for a below-nominal-gain teacher, never reduce it below the fixed
    # baseline for an above-nominal one -- symmetric scaling under-drove (and hurt) those teachers,
    # see DECISIONS.md 2026-11-24 "Per-teacher amplitude calibration".
    amp_scale = max(1.0, ds.NOMINAL_EFF["k"] / truth["k"]) if args.amp_scale_by_k else 1.0
    amps_eff = tuple(a * amp_scale for a in ALL_AMPS)
    log.info("amp_scale (nominal_k/teacher_k) = %.3f  -> effective amps %s",
             amp_scale, " ".join(f"{a:.0f}" for a in amps_eff))

    # Full design: every (site, amp); training = first `sites` sites x `amps`.
    # ALL_AMPS/args.amps stay nominal (500/1000/2000) for indexing; amps_eff carries the actual
    # (possibly k-rescaled) stimulus delivered to the model.
    full = ds.Protocol(ki=np.repeat(ki_all, len(ALL_AMPS), axis=0), amp=np.tile(np.array(amps_eff), 8))
    train_rows = [s * len(ALL_AMPS) + ALL_AMPS.index(a) for s in range(args.sites) for a in args.amps]
    with torch.no_grad():
        teach_full = ds.run(th_truth, net, full, noise_scale=0.0, integer_delays=args.integer_delays,
                        coupling_mode=args.coupling_mode)[:, 0]
    train = ds.Protocol(ki=full.ki[train_rows], amp=full.amp[train_rows])
    target = teach_full[train_rows].numpy()

    if args.init_seed < 0:
        z0 = np.zeros(len(NAMES))
    else:
        z0 = np.random.default_rng(args.init_seed).uniform(np.log(0.5), np.log(1.5), len(NAMES))
    log.info("truth log-offsets: %s", " ".join(f"{n}={v:+.2f}" for n, v in zip(NAMES, truth_z)))
    log.info("start log-offsets: %s", " ".join(f"{n}={v:+.2f}" for n, v in zip(NAMES, z0)))
    if mismatched:
        log.info("MISMATCHED coupling: teacher=%s, student=%s -- g has no shared ground truth here",
                  args.coupling_mode, student_coupling_mode)

    res = fit(target, net, train, NAMES, z0=z0, iters=args.iters, lr=args.lr,
              integer_delays=args.integer_delays, coupling_mode=student_coupling_mode,
              grad_warmup_ms=None if args.grad_warmup < 0 else args.grad_warmup,
              log=log, truth_z=truth_z)
    zf = res["z"][-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = (zf - z0) / (truth_z - z0)
    rel_err = np.exp(zf - truth_z) - 1
    rec = {n: {"truth": truth[n], "fitted": res["theta"][n], "start": ds.NOMINAL_EFF[n] * np.exp(z0[i]),
               "rel_error": float(rel_err[i]), "progress": float(progress[i])}
           for i, n in enumerate(NAMES)}
    for n, r in rec.items():
        log.info("  %-4s truth %9.3f start %9.3f fitted %9.3f rel.err %+7.1f%% progress %+5.2f",
                 n, r["truth"], r["start"], r["fitted"], 100 * r["rel_error"], r["progress"])

    # Generalisation: student vs teacher on every condition (training ones flagged).
    zs = torch.zeros(len(ds.PARAM_NAMES), dtype=torch.float64)
    z_start = torch.zeros(len(ds.PARAM_NAMES), dtype=torch.float64)
    for i, n in enumerate(NAMES):
        zs[ds.PARAM_NAMES.index(n)] = float(zf[i])
        z_start[ds.PARAM_NAMES.index(n)] = float(z0[i])
    with torch.no_grad():
        stud_full = ds.run(ds.theta_from_z(zs), net, full, noise_scale=0.0, integer_delays=args.integer_delays,
                           coupling_mode=student_coupling_mode)[:, 0]
        start_full = ds.run(ds.theta_from_z(z_start), net, full, noise_scale=0.0, integer_delays=args.integer_delays,
                            coupling_mode=student_coupling_mode)[:, 0]
    err_stud = rel_rmse(stud_full, teach_full)
    err_start = rel_rmse(start_full, teach_full)
    is_train = np.zeros(len(err_stud), bool)
    is_train[train_rows] = True
    gen = {"train_rel_rmse": float(err_stud[is_train].mean()),
           "heldout_rel_rmse": float(err_stud[~is_train].mean()) if (~is_train).any() else None,
           "start_train_rel_rmse": float(err_start[is_train].mean()),
           "start_heldout_rel_rmse": float(err_start[~is_train].mean()) if (~is_train).any() else None,
           "per_condition": [{"site": int(r // 3), "amp_nominal": ALL_AMPS[r % 3],
                              "amp_delivered": amps_eff[r % 3], "train": bool(is_train[r]),
                              "student": float(err_stud[r]), "start": float(err_start[r])}
                             for r in range(len(err_stud))]}
    log.info("rel. RMSE  train %.4f (start %.3f)   held-out %.4f (start %.3f)",
             gen["train_rel_rmse"], gen["start_train_rel_rmse"],
             gen["heldout_rel_rmse"] or np.nan, gen["start_heldout_rel_rmse"] or np.nan)

    save_json({"run": run, "args": vars(args), "names": NAMES, "truth": truth, "recovery": rec,
               "teacher_coupling_mode": args.coupling_mode, "student_coupling_mode": student_coupling_mode,
               "mismatched_coupling": mismatched,
               "median_abs_rel_error": float(np.median(np.abs(rel_err))),
               "n_within_10pct": int((np.abs(rel_err) <= 0.10).sum()),
               "loss": res["loss"], "z_history": res["z"], "truth_z": truth_z, "z0": z0,
               "generalisation": gen, "seconds": res["seconds"]}, out_dir / f"{run}.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(res["loss"])
    ax[0].set_title("Training loss (relative MSE)")
    ax[0].set_xlabel("iteration")
    for i, n in enumerate(NAMES):
        ax[1].plot(res["z"][:, i] - truth_z[i], label=n)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("log(fitted / truth) per parameter")
    ax[1].legend(fontsize=7, ncol=3)
    ax[2].bar(NAMES, 100 * rel_err)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Final relative error, %")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run}.png", dpi=110)
    log.info("saved %s", out_dir / run)


if __name__ == "__main__":
    main()
