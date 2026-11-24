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

    s.append(P("4. Does parameter accuracy actually CAUSE generalization — checked, not assumed", H1))
    s.append(P("\"A student that learned the teacher's parameters should generalize\" sounds like a direct "
              "consequence of the model being deterministic — but that only holds for EXACT recovery. With "
              "partial recovery, and in the presence of the sloppy/gauge directions this study keeps finding "
              "(individual c1, c2, c3, c4 don't matter on their own; only their products do), it is not "
              "obvious that \"more accurate parameters\" and \"better generalization\" track each other in "
              "practice. Checked directly: for every already-fitted model in E9/E11/E12, correlated its "
              "OWN parameter-recovery error against its OWN generalization error (Spearman rank "
              "correlation, since the relationship need not be linear).", BODY))
    s.append(fig(BIF_OUT / "fig_generalization_correlation.png",
                "Figure 3. Top row (temporal hold-out, n=20 models): parameter error vs late-window "
                "prediction error for the same model. Bottom row (held-out stimulus amplitude, n=30 models "
                "across 5 teachers): parameter error vs prediction error on an amplitude never trained on."))
    s.append(P("The answer is yes, specifically for the quantities the theory says should matter: b "
              "(&rho;=0.65), c1&middot;c2 (&rho;=0.51) and c3&middot;c4 (&rho;=0.71, held-out-amplitude test) "
              "all correlate significantly with generalization failure. a (&rho;=0.78) and k (&rho;=0.47), "
              "the two stiff parameters, correlate strongly too. <b>Individual c1 does not</b> "
              "(&rho;=-0.14, not significant) — exactly as predicted by the linear theory (section 2 of the "
              "study plan), which says c1 only enters the EEG as the product c1&middot;c2, so its individual "
              "recovery error should be noise with respect to prediction, not signal. This is a clean, "
              "independent confirmation that the structural identifiability analysis is not just "
              "theoretically motivated but actually explains which recovery errors matter for real "
              "generalization behaviour.", BODY))
    s.append(P("Practical implication: don't screen fitted models by overall parameter error (it mixes "
              "signal and noise); screen by error in the identifiable combinations (b, c1&middot;c2, "
              "c3&middot;c4) and the stiff parameters (a, k) — those are the ones that predict whether a fit "
              "will actually generalize.", BOX))
    s.append(PageBreak())

    s.append(P("5. A practical fix: ensemble + richer stimulus, NOT hard parameter bounds", H1))
    s.append(P("Section 3 showed a single fit is unreliable at the firing state. Three candidate practical "
              "fixes were tested directly at the hardest case (E-dominant, m=1.87, p=105) rather than assumed: "
              "a bigger ensemble of random restarts (20, up from 10), a hard box constraint on individual "
              "c1,c2,c3,c4 (clamped to 0.5-2.0x the population-nominal range, NOT each teacher's own truth), "
              "and checking whether the lowest-training-loss restart in an ensemble is actually the most "
              "accurate one.", BODY))
    s.append(table([
        ["Condition", "c1·c2 median (20 starts)", "c3·c4 median (20 starts)"],
        ["E-dominant, free (no bounds)", "26.2%", "12.4%"],
        ["E-dominant, bounded (hard clamp)", "63.6%", "27.3%"],
        ["balanced, free", "38.2%", "17.3%"],
        ["balanced, bounded", "37.2%", "18.9%"],
    ], [0.4, 0.3, 0.3]))
    s.append(P("Table 3. Hard box constraints do not help, and actively hurt in the E-dominant case (median "
              "error roughly DOUBLES despite the true values sitting comfortably inside the bounds). Likely "
              "cause: a hard clamp has zero gradient the instant a parameter hits the wall, and in a landscape "
              "already known to be rugged near these states, that can trap the optimizer against the wall "
              "instead of letting it find its way to the true combination. A soft/differentiable penalty "
              "(not tested here) would not have this failure mode and is the better way to add prior "
              "knowledge if that's wanted.", CAP))
    s.append(P("The ensemble itself, though, is informative and useful -- but only combined with the right "
              "stimulus. Checked whether \"pick the lowest-loss restart\" is a trustworthy selection rule "
              "(the only thing you could do with real data, where the true parameters are unknown):", BODY))
    s.append(fig(BIF_OUT / "fig_loss_vs_accuracy.png",
                "Figure 4. Same E-dominant point, 20 restarts, lowest-loss start circled. With SP1000 alone "
                "(left), the single best-FITTING model has 64% c1·c2 error -- among the WORST in the whole "
                "ensemble, despite a training loss orders of magnitude lower than any other start. With RICH "
                "(right), the lowest-loss start is also the most accurate one (3.1%/0.8%), and loss orders "
                "the rest of the ensemble sensibly too."))
    s.append(P("Spearman correlation between final loss and parameter error, across the 20 restarts: with "
              "SP1000 alone, c1&middot;c2 rho=+0.09 (p=0.70, no relationship) and c3&middot;c4 rho=+0.56 "
              "(p=0.01, moderate); with RICH, c1&middot;c2 rho=+0.83 and c3&middot;c4 rho=+0.81 (both "
              "p&lt;0.001). This is the mechanistic reason richer stimulation matters: it doesn't just tighten "
              "the Cramer-Rao bound, it makes the LOSS SURFACE ITSELF discriminating, so that \"which fit "
              "looks best\" becomes a trustworthy question to ask. With a single pulse, that question has no "
              "good answer -- the single best-fitting model can be confidently, invisibly wrong.", BODY))
    s.append(P("<b>Validated practical recipe:</b> use the RICH protocol (or similar multi-condition "
              "stimulation), run &ge;20 random restarts, and select (or report) the lowest-training-loss one. "
              "Do NOT add naive hard bounds on individual c1-c4. This is a cheap, already-tested alternative "
              "to building a full posterior method, though it only gives a point estimate (with an implicit, "
              "unquantified confidence from how tightly the ensemble agrees) rather than honest calibrated "
              "uncertainty -- a real posterior method remains the more rigorous long-term answer.", BOX))
    s.extend(bullets([
        "Caveat: tested at ONE point (E-dominant, nominal-shape teacher) with ONE seed. The sign of the "
        "SP1000-vs-RICH difference (no correlation vs strong correlation) is a clean, large, mechanistically "
        "sensible effect, but the exact correlation values would move with a different point or more restarts.",
    ]))

    s.append(P("How this conclusion could be wrong", H1))
    s.extend(bullets([
        "<b>The correlation check (section 4) uses n=20-30 models.</b> Spearman p-values around 0.01-0.02 "
        "(c2, c1&middot;c2, k) are real but not overwhelming at this sample size; the clearly strong results "
        "(a, c3&middot;c4, c4: p&lt;0.002) and the clearly null result (c1: p=0.55) are the ones to lean on.",
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
