"""Build the firing-state / bifurcation follow-up report (PDF).

Reads syn_brain/results/single_node_tms/{analysis.json, train_E9-E10_part0of1.npz} and
syn_brain/results/bifurcation_followup/{settle_diagnostic.json, fig_settle_diagnostic.png},
plus the pre-fix comparison run saved aside during this investigation.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.bifurcation_report
Output: syn_brain/reports/bifurcation_followup.pdf
"""

from __future__ import annotations

import json

import numpy as np
from reportlab.platypus import PageBreak, Spacer

from syn_brain.core.paths import SYN_ROOT
from syn_brain.core.report_style import (BODY, BOX, BUL, CAP, H1, H2, P, SMALL, TITLE, VERDICT_BAD, VERDICT_OK,
                                         bullets, build, fig, pct, table)
from syn_brain.scripts.single_node_tms import OUT, FIT, readout

REPORT = SYN_ROOT / "reports" / "bifurcation_followup.pdf"
BIF_OUT = SYN_ROOT / "results" / "bifurcation_followup"
OLD_NPZ = "/tmp/claude-1084/-home-william-Documents-playground-jr-synthetic-neurofe/bba1a7a5-785a-497a-abca-68b116670858/scratchpad/train_E9-E10_settle300_OLD.npz"

PROTOS = ["SP1000@p120", "DOSE@p120", "SP1000@p150", "DOSE@p150"]
FIT_IDX = {n: i for i, n in enumerate(FIT)}


def load(npz_path):
    d = dict(np.load(npz_path, allow_pickle=True))
    return d


def sel(d, **kw):
    return np.array([all(j.get(k) == v for k, v in kw.items()) for j in d["jobs"]])


def combo_err(theta, truth, a, b):
    return np.abs(np.log((theta[:, FIT_IDX[a]] * theta[:, FIT_IDX[b]]) /
                         (truth[:, FIT_IDX[a]] * truth[:, FIT_IDX[b]]))) * 100


def recovery_table(d, an):
    rows = [["protocol", "exp", "c1·c2 % (excitatory loop)", "c3·c4 % (inhibitory loop)", "CRB c1·c2 / c3·c4 (%)"]]
    out = {}
    for prot in PROTOS:
        crb = np.array(an["crb"][prot][0]["se7"]) * 100
        crb_c1c2 = float(np.hypot(crb[FIT_IDX["c1"]], crb[FIT_IDX["c2"]]))
        crb_c3c4 = float(np.hypot(crb[FIT_IDX["c3"]], crb[FIT_IDX["c4"]]))
        for exp in ("E9", "E10"):
            m = sel(d, exp=exp, protocol=prot)
            if not m.any():
                continue
            theta, truth = d["theta"][m, :len(FIT)], d["truth"][m, :len(FIT)]
            c1c2 = float(np.median(combo_err(theta, truth, "c1", "c2")))
            c3c4 = float(np.median(combo_err(theta, truth, "c3", "c4")))
            rows.append([prot, exp, f"{c1c2:.1f}", f"{c3c4:.1f}", f"{crb_c1c2:.1f} / {crb_c3c4:.1f}"])
            out[(prot, exp)] = dict(c1c2=c1c2, c3c4=c3c4, crb_c1c2=crb_c1c2, crb_c3c4=crb_c3c4)
    return rows, out


def content():
    an = json.load(open(OUT / "analysis.json"))
    d_new = load(OUT / "train_E9-E10_part0of1.npz")
    d_old = load(OLD_NPZ)
    rows_new, new = recovery_table(d_new, an)
    rows_old, old = recovery_table(d_old, an)
    sd = json.load(open(BIF_OUT / "settle_diagnostic.json"))

    headline_old = old[("SP1000@p120", "E9")]["c3c4"]
    headline_new = new[("SP1000@p120", "E9")]["c3c4"]
    p150_old = old[("SP1000@p150", "E9")]["c3c4"]
    p150_new = new[("SP1000@p150", "E9")]["c3c4"]
    got_worse = headline_new > headline_old and p150_new > p150_old

    s = []
    s.append(P("Firing-state follow-up: the p=120 bifurcation and the settle-time check", TITLE))
    s.append(P("syn_brain single-node virtual-TMS study — supplementary investigation", CAP))
    s.append(Spacer(1, 6))

    s.append(P("Verdicts", H1))
    s.append(P(f"Is the simulator's settle time too short near the firing-state bifurcation? &rarr; "
              f"<b>YES, confirmed and fixed.</b> At tonic drive p=115-220 the pre-stimulus baseline was "
              f"still drifting by up to 0.51 &micro;V at the old default (settle_ms=300) — bigger than the "
              f"0.42 &micro;V sensor noise floor this study calibrates against — and converges to "
              f"&lt;0.005 &micro;V by settle_ms=1500 (classic critical slowing down at a bifurcation). "
              f"Default changed to 1500 in core/node.py.", VERDICT_OK))
    s.append(P(f"Does fixing it close the p=120 backprop recovery gap? &rarr; <b>NO — the opposite.</b> "
              f"c3&middot;c4 (inhibitory loop) median error, SP1000@p120, noise-free: {headline_old:.0f}% "
              f"(old, contaminated) &rarr; {headline_new:.0f}% (new, corrected) against a CRB bound of "
              f"{new[('SP1000@p120','E9')]['crb_c3c4']:.1f}%. p=150 got worse too: {p150_old:.0f}% &rarr; "
              f"{p150_new:.0f}%. <b>This rules settle time out as the cause</b> of the p=120 vs p=150 gap: "
              f"the EEG fit itself is equally good in both runs (loss ~3-6e-3 either way) but the recovered "
              f"PARAMETERS are no closer to truth with cleaner dynamics — a sign of a genuinely flat/"
              f"multi-valued loss landscape, not a numerical artifact.", VERDICT_BAD if got_worse else VERDICT_OK))
    s.append(P("Gate (docs/single_node_study_plan.txt Part A): c1·c2 and c3·c4 bounds &lt; 20% at realistic "
              "noise, backprop median error &lt; 2x the CRB bound &rarr; <b>FAILS at p=120 even after the "
              "fix; p=150 is closer but still fails on c1·c2.</b> See Table 1.", BOX))

    s.append(P("What changed / what's next", H2))
    s.append(P("core/node.py's default settle_ms changed from 300 to 1500 for every caller — a genuine "
              "correctness fix, kept regardless of the result above (E7 was not re-run this session; its "
              "p=110 residual was already small). Since settle time is ruled out, a continuation experiment "
              "(warm-start from the converged p=150 solution, step the drive down to p=120) was run to test "
              "whether the student simply crosses its own quiet/active threshold during a cold &plusmn;50% "
              "random start. <b>Result: it does not reliably fix p=120 either</b> — see \"Continuation "
              "experiment\" below for the full, mixed result and the final recommendation (p&ge;150, no "
              "further Adam tuning at p=120).", BODY))
    s.append(PageBreak())

    s.append(P("Background", H1))
    s.append(P("This follows directly from the firing-state follow-up (background_drive_note.txt section 5): "
              "E9/E10 pushed the single JR node's tonic drive past the quiet-state jump (p=120, right at the "
              "threshold; p=150, solidly on the active branch) to test whether the inhibitory/excitatory loop "
              "gains (c3&middot;c4, c1&middot;c2) become recoverable by backprop once the node's sigmoid is "
              "engaged, as the Cramer-Rao bound predicts. The first run found p=150 recovered well but p=120 "
              "did not, despite an equally tight CRB bound at both points — a sign of an optimizer or "
              "simulation problem, not a non-identifiability. This report tests the most obvious candidate "
              "cause (an under-settled simulator), fixes it on its own merits, and reports that it was NOT "
              "the explanation for the p=120 vs p=150 gap.", BODY))

    s.append(P("Method: the settle-time diagnostic", H1))
    s.append(P("core/node.py always starts its Euler integration from a fixed, near-zero state and runs "
              "<tt>settle_ms</tt> of simulated time <i>under torch.no_grad()</i> before recording — so the "
              "settle phase is never differentiated; it only decides which (possibly still-relaxing) state "
              "seeds the recorded, differentiated 400 ms window. If that state is not yet the true fixed "
              "point, the recorded response is contaminated by a residual transient. The check: simulate the "
              "nominal node under a pure tonic drive (no pulse) at p = 0-220 and settle_ms = 300/900/1500, and "
              "measure how much the pre-stimulus baseline (which should be perfectly flat once settled) still "
              "drifts over its first 100 ms.", BODY))
    s.append(fig(BIF_OUT / "fig_settle_diagnostic.png",
                "Figure 1. Left: size of the residual pre-stimulus drift at three settle times, log scale, "
                "against the 0.42 &micro;V sensor noise floor. Right: the baseline offset itself. Both panels "
                "show a sharp, narrow spike in residual error starting right at the p&asymp;115 jump and "
                "persisting to at least p=220 — classic critical slowing down at a bifurcation, not a smooth "
                "trend. settle=300 ms (the old default) is contaminated across this whole range; settle=1500 "
                "ms (the new default) is clean (&lt;0.005 &micro;V) everywhere tested, 0-240."))

    s.append(P("Results: does the fix restore the predicted recovery?", H1))
    s.append(P("Re-ran E9 (noise-free) and E10 (0.42 &micro;V sensor noise) with the corrected settle time, "
              "same recipe as before (10 random starts, &plusmn;50%, 300 Adam iterations, log-scaled "
              "parameters). Comparing the identifiable combinations (not the individual sloppy c's) against "
              "their Cramer-Rao bound at the same operating point: <b>every number got worse, not better</b>, "
              "at both p=120 and p=150.", BODY))
    s.append(table(rows_new, [0.22, 0.1, 0.23, 0.23, 0.22]))
    s.append(P("Table 1. Recovery AFTER the settle-time fix (settle_ms=1500, the new default).", CAP))
    s.append(table(rows_old, [0.22, 0.1, 0.23, 0.23, 0.22]))
    s.append(P("Table 2. The same recovery BEFORE the fix (settle_ms=300, the original E9/E10 run).", CAP))
    s.append(P("Both runs converge to a comparably good EEG fit (final relative loss ~3-6&times;10&#8315;&sup3; "
              "either way, over all 10 starts) — the optimizer is not failing to fit the data, it is finding "
              "DIFFERENT parameters that fit the data about equally well. That is the signature of a flat or "
              "multi-valued region of the loss surface, not of a numerical artifact: cleaning up the dynamics "
              "gave the optimizer a more faithful landscape, not an easier one.", BODY))

    s.append(P("How this conclusion could be wrong", H1))
    s.extend(bullets([
        "<b>Risk 1 — the settle-time contamination itself could be a measurement artifact.</b> Checked: "
        "Figure 1 is a direct, model-free measurement (no fitting involved) of the gap between two simulate() "
        "calls at different settle_ms, confirmed against two independently-generated result files "
        "(analyse()'s per-protocol Jacobian and its separate tonic_sweep, which already used settle_ms=1500) "
        "before trusting the number — both agreed once the gain scaling was applied consistently.",
        "<b>Risk 2 — 'both runs got worse' could mean the new run just had unluckier random starts, not that "
        "settle time truly doesn't matter.</b> Partially checked: both runs share the same seeded "
        "&plusmn;50% starting points (same RNG call in build_jobs), so this is not a resampling artifact — "
        "but 10 starts is still few enough that a full explanation should also check the per-start spread, "
        "not just the median (not done in this report; the raw per-start arrays are in the saved .npz files).",
        "<b>Risk 3 — the real cause could be something this report didn't test</b> (e.g. the Adam learning "
        "rate / schedule being tuned for the quiet-state loss scale, not the firing-state one). The "
        "continuation experiment (next step, below) tests the leading hypothesis (basin-of-attraction / "
        "branch-crossing) but would not catch a learning-rate-only explanation; flagged, not ruled out.",
    ]))

    s.append(PageBreak())
    s.append(P("Continuation experiment: does a smarter start fix it?", H1))
    cont = json.load(open(BIF_OUT / "continuation_results.json"))
    rows_cont = [["stage", "c1·c2 % (median of 10)", "c3·c4 % (median of 10)"],
                ["start: p=150, cold-start Adam (already converged)",
                 f"{new[('SP1000@p150','E9')]['c1c2']:.1f}", f"{new[('SP1000@p150','E9')]['c3c4']:.1f}"]]
    for p in ("140", "130", "120"):
        rows_cont.append([f"warm-started -> p={p} ({'stepping stone' if p != '120' else 'real target'})",
                          f"{cont['history'][p]['c1c2']:.1f}", f"{cont['history'][p]['c3c4']:.1f}"])
    s.append(P("Tested the leading hypothesis directly: take the already-converged p=150 solutions (10 "
              "starts) and warm-start a continuation down through p=140 and p=130 (artificial stepping-stone "
              "targets, 100 Adam iterations each) to the real p=120 target, instead of a cold &plusmn;50% "
              "random start.", BODY))
    s.append(table(rows_cont, [0.5, 0.25, 0.25]))
    s.append(P("Table 3. Continuation path. Compare the final row to Table 1's cold-start p=120 numbers "
              "(c1&middot;c2 75-87%, c3&middot;c4 52-59%).", CAP))
    s.append(fig(BIF_OUT / "fig_continuation_spread.png",
                "Figure 2. Per-start error (10 starts, log scale), not just the median. p=150 cold-start is "
                "a fairly tight, mostly-decent cluster. p=120 cold-start is a tight cluster of consistently "
                "mediocre fits. p=120 via continuation is NOT a tighter version of either — it is sharply "
                "bimodal: most starts land in a much WORSE basin for c3·c4 (nearly all near 200%, versus "
                "~50% cold-start), while c1·c2 splits into a cluster of quite good fits and 3 catastrophic "
                "outliers (300-600%)."))
    s.append(P("Verdict: continuation does not reliably fix p=120 — it changes the FAILURE MODE rather than "
              "removing it. The median c3&middot;c4 error gets worse (52% &rarr; 222%); the median c1&middot;c2 "
              "error gets better (81% &rarr; 17%, passing the 20% gate) but only because a minority of starts "
              "land in an excellent basin while most do not, and a few land in a catastrophic one. This is "
              "strong direct evidence for genuine multi-modality in the loss landscape at and below the "
              "p&asymp;120-140 region — not an optimizer-initialization problem continuation can patch, and "
              "not something settle time was ever going to fix either. p=150's spread (Figure 2, leftmost "
              "column) has no such catastrophic outliers in this sample.", BODY))
    s.append(P("<b>Final recommendation:</b> use p&ge;150 (plain cold-start Adam, no continuation needed) as "
              "the default operating point for Part A of the study plan — it is both simpler and empirically "
              "more reliable than anything tested at or below p=140. If a future need specifically requires "
              "fitting AT the p&asymp;115-140 region, don't keep tuning Adam: this is a multi-modal-posterior "
              "problem, which points at methods designed for that (study plan Part C: C6 simulation-based "
              "inference or C7 HMC/NUTS, which can characterize multiple modes directly, rather than C2-C4 "
              "which assume a single basin).", BOX))

    s.append(P("Reproduce", H1))
    s.append(P("<font face='DVM'>PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms analyse<br/>"
              "PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms train --only E9,E10<br/>"
              "PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.bifurcation_report</font>", SMALL))
    return s


def main():
    build(content(), REPORT, "Firing-state follow-up: the p=120 bifurcation and the settle-time fix",
         "syn_brain · bifurcation follow-up")


if __name__ == "__main__":
    main()
