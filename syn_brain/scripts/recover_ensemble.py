"""Ensemble/restart wrapper around recover_clean.py: pick the lowest-training-loss restart.

This is the fix reports/single_node_conclusions.pdf sections 5-6 validated for a
CRB-good-but-backprop-bad single-node problem (a single start can get stuck in a bad basin
even when the Fisher information says the true answer is findable) -- applied here at
network scale for the first time (docs/network_study_plan.txt N0.3, flagged but not built
until this session's teacher-6/teacher-3 diagnosis made it the clear next step).

This script does NOT run any simulation itself -- it assumes a set of recover_clean.py runs
for the SAME teacher/design, differing only in --init-seed, have already been produced (one
nominal start, --init-seed -1, plus several random starts, --init-seed 1..N), and picks the
one with the lowest final TRAINING loss (never held-out loss -- that would be cheating: real
data never gives you the held-out error to select on).

Usage (from the repo root), after running recover_clean.py once per restart with a shared
--tag and varying --init-seed:
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.recover_ensemble \\
        --teacher-seed 6 --tag _N2gateD_ens --init-seeds -1 1 2 3 4 5 6 7
Outputs: results/recovery_ensemble/<run>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from syn_brain.core.paths import RESULTS_DIR, get_logger, save_json


def restart_run_name(teacher_seed: int, tag: str, init_seed: int, sites: int, amps: list[float],
                      n_nodes: int, coupling_mode: str, g_override: float | None,
                      amp_scale_by_k: bool, integer_delays: bool, grad_warmup: int,
                      student_coupling_mode: str | None = None) -> str:
    """Reconstruct the filename recover_clean.py would have written for one restart.

    Mirrors recover_clean.main()'s `run` string exactly -- kept in sync by hand since the two
    scripts don't share this bit of string-building code. `student_coupling_mode`: pass only when
    it differs from `coupling_mode` (recover_clean.py's --coupling-mode is the TEACHER's mode;
    a mismatched run adds a `_stu<mode>` segment right after it).
    """
    amps_tag = "-".join(str(int(a)) for a in amps)
    mismatched = student_coupling_mode is not None and student_coupling_mode != coupling_mode
    return (f"n{n_nodes}_s{sites}_a{amps_tag}_init{init_seed}"
            f"{'_intdel' if integer_delays else ''}{'_nowarmgrad' if grad_warmup < 0 else ''}"
            f"{'_' + coupling_mode if coupling_mode != 'laplacian' else ''}"
            f"{'_stu' + student_coupling_mode if mismatched else ''}"
            f"{'_g' + f'{g_override:g}' if g_override is not None else ''}"
            f"{'_kscale' if amp_scale_by_k else ''}"
            f"{tag}_t{teacher_seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-seed", type=int, required=True)
    ap.add_argument("--tag", required=True, help="the --tag shared by every restart's recover_clean.py call")
    ap.add_argument("--init-seeds", type=int, nargs="+", required=True,
                     help="every --init-seed value used, e.g. -1 1 2 3 4 5 6 7")
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--amps", type=float, nargs="+", default=[500.0, 1000.0, 2000.0])
    ap.add_argument("--n-nodes", type=int, default=10)
    ap.add_argument("--coupling-mode", default="laplacian", help="TEACHER's coupling mode")
    ap.add_argument("--student-coupling-mode", default=None,
                     help="STUDENT's coupling mode if different from --coupling-mode (mismatch run)")
    ap.add_argument("--g-override", type=float, default=None)
    ap.add_argument("--amp-scale-by-k", action="store_true")
    ap.add_argument("--integer-delays", action="store_true")
    ap.add_argument("--grad-warmup", type=int, default=200)
    args = ap.parse_args()
    log = get_logger(f"recover_ensemble_{args.tag}_t{args.teacher_seed}")

    in_dir = RESULTS_DIR / "recovery_clean"
    restarts = []
    for seed in args.init_seeds:
        name = restart_run_name(args.teacher_seed, args.tag, seed, args.sites, args.amps,
                                 args.n_nodes, args.coupling_mode, args.g_override,
                                 args.amp_scale_by_k, args.integer_delays, args.grad_warmup,
                                 student_coupling_mode=args.student_coupling_mode)
        path = in_dir / f"{name}.json"
        if not path.exists():
            log.warning("missing restart init_seed=%d: %s (skipped)", seed, path)
            continue
        j = json.load(open(path))
        restarts.append({"init_seed": seed, "path": str(path),
                          "train_loss": j["loss"][-1],
                          "names": j["names"], "recovery": j["recovery"],
                          "generalisation": j["generalisation"], "truth": j["truth"]})

    if not restarts:
        raise SystemExit("no restart results found -- run recover_clean.py for each --init-seed first")

    losses = [r["train_loss"] for r in restarts]
    best = restarts[int(np.argmin(losses))]
    names = best["names"]

    log.info("restarts: %s", " ".join(f"seed{r['init_seed']}={r['train_loss']:.3e}" for r in restarts))
    log.info("best: init_seed=%d, train_loss=%.3e (vs median %.3e, worst %.3e)",
              best["init_seed"], best["train_loss"], float(np.median(losses)), max(losses))
    for n in names:
        rec = best["recovery"][n]
        log.info("  %-4s truth %9.3f fitted %9.3f rel.err %+7.1f%%", n, rec["truth"], rec["fitted"],
                  100 * rec["rel_error"])
    c1c2_t = best["recovery"]["c1"]["truth"] * best["recovery"]["c2"]["truth"]
    c1c2_f = best["recovery"]["c1"]["fitted"] * best["recovery"]["c2"]["fitted"]
    c3c4_t = best["recovery"]["c3"]["truth"] * best["recovery"]["c4"]["truth"]
    c3c4_f = best["recovery"]["c3"]["fitted"] * best["recovery"]["c4"]["fitted"]
    log.info("  c1*c2 rel.err %+.1f%%   c3*c4 rel.err %+.1f%%", 100 * (c1c2_f / c1c2_t - 1),
              100 * (c3c4_f / c3c4_t - 1))
    log.info("train rel.RMSE (best restart) %.4f   held-out %.4f", best["generalisation"]["train_rel_rmse"],
              best["generalisation"]["heldout_rel_rmse"] or float("nan"))

    out_dir = RESULTS_DIR / "recovery_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = restart_run_name(args.teacher_seed, args.tag, best["init_seed"], args.sites, args.amps,
                                 args.n_nodes, args.coupling_mode, args.g_override,
                                 args.amp_scale_by_k, args.integer_delays, args.grad_warmup,
                                 student_coupling_mode=args.student_coupling_mode)
    save_json({"teacher_seed": args.teacher_seed, "tag": args.tag, "n_restarts": len(restarts),
               "all_train_losses": {r["init_seed"]: r["train_loss"] for r in restarts},
               "best_init_seed": best["init_seed"], "best_source": best["path"],
               "names": names, "recovery": best["recovery"], "generalisation": best["generalisation"],
               "truth": best["truth"],
               "teacher_coupling_mode": args.coupling_mode,
               "student_coupling_mode": args.student_coupling_mode or args.coupling_mode},
              out_dir / f"ensemble_{out_name}.json")
    log.info("saved %s", out_dir / f"ensemble_{out_name}")


if __name__ == "__main__":
    main()
