"""Final single-node conclusion report: combines the original rest-state study
(reports/single_node_tms_study.pdf, E1-E8) with the firing-state follow-up (E9-E12, this
investigation) into one answer to "are parameters identified and do fits generalize on the
single-node setup".

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_conclusions_report
Output: syn_brain/reports/single_node_conclusions.pdf
"""

from __future__ import annotations

from reportlab.platypus import PageBreak, Spacer

from syn_brain.core.paths import SYN_ROOT
from syn_brain.core.report_style import BODY, BOX, BUL, CAP, H1, H2, P, SMALL, TITLE, VERDICT_BAD, VERDICT_OK, bullets, build, fig, table

REPORT = SYN_ROOT / "reports" / "single_node_conclusions.pdf"
BIF_OUT = SYN_ROOT / "results" / "bifurcation_followup"


def content():
    s = []
    s.append(P("Single-node JR identifiability study: the conclusion", TITLE))
    s.append(P("Combining the rest-state study (E1-E8, reports/single_node_tms_study.pdf) with the "
              "firing-state follow-up (E9-E12, reports/bifurcation_followup.pdf)", CAP))
    s.append(Spacer(1, 6))

    s.append(P("Verdicts", H1))
    s.append(P("<b>Are the parameters identifiable?</b> &rarr; Only two of seven, everywhere: "
              "<b>k</b> (stimulus gain) and <b>a</b> (excitatory rate) are recovered to &lt;1% and &lt;5% "
              "error under every protocol and operating point tested. Everything about the E-I loop "
              "(<b>b, c1, c2, c3, c4</b>) is, at best, conditionally identifiable as the PRODUCTS "
              "c1&middot;c2 and c3&middot;c4 (never the individual c's), and only when the node is pushed "
              "onto the active branch (p&ge;150) — at rest they carry 56-252% error. "
              "<b>A, B, vmax, v0, r are gauge/non-identifiable by construction</b> (exact scaling "
              "symmetries) and must stay fixed by convention.", VERDICT_OK))
    s.append(P("<b>Does the firing state actually fix the loop-gain parameters?</b> &rarr; "
              "<b>Only partially, and not robustly.</b> At the nominal teacher, p=150 brings "
              "c3&middot;c4 to 14-24% (near/at the 20% gate) and c1&middot;c2 to 20-38% (misses it). "
              "But across 4 other teachers (&plusmn;30% draws, E12), the SAME recipe gives c3&middot;c4 "
              "7-28% and c1&middot;c2 13-95% — one teacher's excitatory loop fit fails almost completely "
              "(91-95% error). p=120 fails outright (genuine landscape multi-modality, see "
              "bifurcation_followup.pdf).", VERDICT_BAD))
    s.append(P("<b>Do fits generalize?</b> &rarr; <b>No, not reliably, at either operating point.</b> "
              "Fits reproduce their own training window and stimulus well, but: late-window prediction "
              "carries 5-30% error even for a full-window fit (worse at the firing state than at rest); "
              "a temporal hold-out (fit 0-60ms, predict the rest, E8/E11) pushes that to 46-91% with a "
              "perfect model and no noise; and a held-out stimulus amplitude, nearly free at rest (&lt;4%), "
              "costs 24% at the firing state. <b>Generalization fails almost exactly where identifiability "
              "is weak</b> — the loop parameters that are hard to pin down are also the ones that control "
              "what a fit does NOT predict.", VERDICT_BAD))
    s.append(P("Part A gate (docs/single_node_study_plan.txt): c1&middot;c2 and c3&middot;c4 bounds &lt;20% "
              "at realistic noise, generalizes, no sustained oscillation &rarr; <b>FAILS overall.</b> The "
              "point estimate at one teacher can look like a pass (c3&middot;c4 at p=150); robustness "
              "across teachers and any generalization test both fail it.", BOX))
    s.append(PageBreak())

    s.append(P("1. What is identifiable (Table 7, original study, confirmed and extended here)", H1))
    rows = [["Param.", "Physical meaning", "Rest (CRB, best protocol)", "p=150 (this study)", "Class", "Verdict"],
           ["k", "stimulus gain", "0.12-0.2%", "0.2-1.0%", "Stiff", "IDENTIFIABLE, everywhere"],
           ["a", "1/tau_e, excitatory rate", "0.19-2.5%", "0.8-1.0%", "Stiff", "IDENTIFIABLE, everywhere"],
           ["b", "1/tau_i, inhibitory rate", "0.15-20.6%", "3.4-8.3%", "Conditional", "needs E+I data or firing state"],
           ["c1·c2", "excitatory loop gain", "56-252% (CRB); progress -0.04/-0.68 (moves AWAY)",
            "20-95% (E9/E12, 5 teachers)", "Sloppy/Conditional", "WEAKLY identifiable at best; not robust"],
           ["c3·c4", "inhibitory loop gain", "129-252% (CRB)", "7-28% (E9/E12, 5 teachers)",
            "Conditional", "IDENTIFIABLE only at p>=150, mostly robust"],
           ["c1, c2, c3, c4 individually", "which synapse in the loop", "sloppy everywhere (only products enter the EEG)",
            "same", "Sloppy", "NOT IDENTIFIABLE, by construction (section 2 theory)"],
           ["A, B, vmax, v0, r", "EPSP/IPSP size, max rate, threshold, slope", "exact scaling symmetries with k, c's",
            "same", "Gauge", "NOT IDENTIFIABLE from EEG; fix by convention"]]
    s.append(table(rows, [0.09, 0.17, 0.18, 0.16, 0.11, 0.29], font=7.0))
    s.append(P("Table 1. Identifiability classification. CRB = Cramér-Rao bound (theoretical best); "
              "'this study' = actual backprop recovery, median over teachers/starts.", CAP))

    s.append(P("2. Generalization: fails at both operating points", H1))
    s.append(fig(BIF_OUT / "fig_generalization_summary.png",
                "Figure 1. SP1000, nominal teacher. 'Same-stimulus, full window' is the headline number "
                "usually reported — it looks fine at both operating points. Every other test (late window, "
                "temporal hold-out, held-out amplitude) is far worse, and mostly worse at the firing state "
                "than at rest, not better."))
    rows2 = [["Test", "Rest (p=0)", "Firing state (p=150)", "Source"],
            ["Same-stimulus, full window", "0.49%", "4.8%", "E1 / E9"],
            ["Same-stimulus, late window (full-window fit)", "5.2%", "26%", "E8 / this study (E9 fits, late-only score)"],
            ["Same-stimulus, late window (early-only fit)", "91%", "46%", "E8 / E11 (temporal hold-out)"],
            ["Held-out stimulus amplitude (whole window)", "3.6%", "4-25%", "E1 / this study (E9 fits predicting SP250/SP4000)"]]
    s.append(table(rows2, [0.34, 0.18, 0.24, 0.24]))
    s.append(P("Table 2. Median relative RMSE. The temporal hold-out result reproduces, in the cleanest "
              "possible setting (perfect model, no noise), the same failure mode seen in the real-data "
              "tests this whole study was designed to explain.", CAP))

    s.append(P("3. Robustness across teachers: the firing-state result is NOT one-size-fits-all", H1))
    s.append(fig(BIF_OUT / "fig_robustness_p150.png",
                "Figure 2. 5 starts per teacher, SP1000@p150. The nominal teacher's c3·c4 recovery "
                "(median ~20%) is roughly the middle of the pack across 4 perturbed teachers (7-28%); its "
                "c1·c2 recovery (median ~38%) is also mid-pack, but T2 fails almost completely (median "
                "~90%, worse than at rest). A single teacher's result does not characterize the method."))
    s.append(P("This matters for Part B of the study plan (disease markers): a detectability or attribution "
              "claim for c1&middot;c2 or c3&middot;c4 made from one teacher's fit cannot be trusted without "
              "checking it holds across a range of ground truths first — exactly what T2 shows can fail.", BODY))
    s.append(PageBreak())

    s.append(P("How this conclusion could be wrong", H1))
    s.extend(bullets([
        "<b>Small samples.</b> E9/E12 use 10 and 5 starts respectively, and 4 perturbed teachers. The "
        "qualitative pattern (loop gains sloppy/non-robust, generalization poor) is consistent with the "
        "much larger original E1-E8 study (402 models) at rest, which is reassuring, but the firing-state "
        "numbers specifically could shift with more starts/teachers.",
        "<b>Single optimizer.</b> Everything here uses Adam with log-scaled parameters, 300 iterations — "
        "the best of the methods tried in the original study (C1), but still a local, point-estimate "
        "method. The study plan's Part C methods (C2 second-order, C5-C7 posterior-based) might do better; "
        "not tested here.",
        "<b>Single node.</b> Network effects (shared input across nodes, the real stimulus spreading to "
        "neighbours) could change what's identifiable, for better or worse. This conclusion is about the "
        "method and the model class at the single-node level only.",
    ]))

    s.append(P("Bottom line", H1))
    s.append(P("On this single-node setup, with neurofe's own backprop fitting method: <b>k and a are "
              "solid and ready to use as biomarkers today. The E-I loop gains (c1·c2, c3·c4) are "
              "not</b> — at rest they are unidentifiable in practice, and pushing to the firing state "
              "recovers them only sometimes, for one teacher at a time, and with no guarantee the fit says "
              "anything about a held-out time window or stimulus. This is not a close call that more tuning "
              "would fix; it was tested directly (continuation experiment, bifurcation_followup.pdf) and "
              "the result was genuine multi-modality / landscape ruggedness, not an initialization problem.",
              BOX))

    s.append(P("Recommended next step", H1))
    s.append(P("<b>Not</b> the network (Phase 2) for the loop-gain parameters' sake — it would repeat this "
              "exact failure at far greater cost and be much harder to diagnose. The single-node setup is "
              "the cheap place to fix the method; the network is not. <b>Before Part B (disease markers) or "
              "Phase 2</b>, test at least one method from Part C that reports honest uncertainty instead of "
              "a point estimate (cheapest first: multi-start with the spread reported, not just the median; "
              "then variational Laplace or SBI) on this same single-node p=150 problem. A method that "
              "correctly reports 'c1&middot;c2 is not determined by this data' is more useful than one that "
              "silently returns a confident but wrong number. Network-level work on background drive / "
              "non-diffusive coupling (to fix the missing late-TEP-activity finding) can proceed in "
              "parallel — but that fixes the teacher's <i>realism</i>, not the <i>identifiability</i> "
              "problem documented here, and the two should not be conflated.", BOX))

    s.append(P("Reproduce", H1))
    s.append(P("<font face='DVM'>PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms train "
              "--only E9,E10,E11,E12<br/>"
              "PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_conclusions_report</font>", SMALL))
    return s


def main():
    build(content(), REPORT, "Single-node JR identifiability study: the conclusion",
         "syn_brain · single-node study conclusion")


if __name__ == "__main__":
    main()
