"""Executive summary of the whole syn_brain effort: what we have, what we found, what next, product view.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.executive_summary_report
Output: syn_brain/reports/executive_summary.pdf  (text only; no results files needed)
"""

from __future__ import annotations

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether, PageBreak

from syn_brain.core.paths import SYN_ROOT
from syn_brain.core.report_style import (BOX, CAP, TITLE, VERDICT_BAD, VERDICT_OK, P, build, bullets, table)
from syn_brain.core.report_style import H1 as _H1, H2 as _H2

H1 = ParagraphStyle("h1k", parent=_H1, keepWithNext=1)
H2 = ParagraphStyle("h2k", parent=_H2, keepWithNext=1)
OUT = SYN_ROOT / "reports" / "executive_summary.pdf"
DATE = "27 November 2026 (v5)"


def T(rows, widths, cap, font=7.4, head=None):
    return KeepTogether(([head] if head is not None else []) + [table(rows, widths, font=font), P(cap, CAP)])


def content():
    s = [P("syn_brain: where we are", TITLE),
         P(f"Executive summary of the synthetic-brain identifiability work · {DATE} · "
           "<b>Status:</b> single-node study closed (recipe + calibrated posterior). 10-node gate run in full with "
           "Laplacian coupling; the follow-up coupling study found a simulator artifact that changes its conclusions, "
           "so the coupling comparison must be rerun before the coupling is chosen. CDC margins paper: S1–S3 done, "
           "paused at S4 for decisions.", CAP)]

    # ------------------------------------------------------------------ verdicts
    s.append(P("Verdicts", H1))
    s.append(P(
        "<b>Can neurofe's backprop fitting, as published, recover Jansen-Rit (JR) parameters from TMS-EEG?</b> → "
        "<b>Only two, reliably: the stimulation gain k and the excitatory time constant a.</b> With a single pulse at rest, "
        "the loop parameters stay where the fit started; A, B and vmax can never be identified from EEG (exact "
        "symmetries).", VERDICT_BAD))
    s.append(P(
        "<b>Do fitted models generalize?</b> → <b>No, not as published.</b> With a perfect model and no noise, fitting "
        "the first 60 ms and predicting the rest gives 46–91% late-window error, reproducing the real-data failures "
        "<b>without any model mismatch</b>: the failure is in the method and the stimulation design.", VERDICT_BAD))
    s.append(P(
        "<b>With the right recipe, single-node loop strengths are measurable with honest uncertainty</b> → <b>Yes, in "
        "well-behaved active states.</b> Rich stimulation + p = 150 + 15 restarts + lowest-loss selection + Laplace "
        "posterior, 24 random subjects: median error 7.5% (c1·c2) and 4.0% (c3·c4); 90% intervals cover the truth 96% "
        "of the time; small winner's-curse bias (−6% / −3%).", VERDICT_OK))
    s.append(P(
        "<b>10-node network, full gate (8 subjects, 5 restarts, Laplacian coupling)</b> → <b>Passes its three named "
        "criteria, but not cleanly.</b> Median error a 0.3%, k 0.7%, c3·c4 4.2%, mu 0.4% (all 8 subjects move toward "
        "the truth), c1·c2 12.0%; <b>coupling g fails (27%, biased low for high-g subjects)</b>. Not yet covered: "
        "interval coverage (no network posterior), stability of the fitted networks, enough restarts (loss spread "
        "1–6 orders of magnitude across restarts), and an amplitude calibration that uses the true gain.", BOX))
    s.append(P(
        "<b>New, 27 Nov: is the Laplacian 'high-g instability' real?</b> → <b>Mostly not; it is an artifact of our "
        "simulator.</b> core/diffsim.py holds the coupling's instantaneous self-term for 1 ms; neurofe updates it "
        "every 0.1 ms. Replaying the 8 gate subjects with no clip: held → 6 of 8 diverge (56–72 Hz oscillation); "
        "neurofe's update → 1 of 8 (subject 3, a genuine instability that persists at a 10× finer step). The g bias, the "
        "'no bounded equilibrium' finding and the matched-Laplacian numbers rest on this artifact for 6 of 8 subjects "
        "(teacher and student share it), so they must be rerun.", VERDICT_BAD))
    s.append(P(
        "<b>Which coupling: Laplacian (neurofe) or non-diffusive (rate-based)?</b> → <b>Not decided by the evidence "
        "yet.</b> The three-way comparison favoured Laplacian, but its Laplacian arm is artifact-affected and its "
        "non-diffusive arm used one fixed g, the plain (not centred) form and too few restarts. The mismatch test is "
        "the robust part: <b>a Laplacian model fitted to rate-coupled data gets E/I badly wrong</b> (c1·c2 98%, c3·c4 "
        "54%, twice the fit error). That argues for choosing the coupling on physiology and then making it "
        "identifiable, not for keeping the easier-to-fit one.", BOX))
    s.append(P("Decisions needed from William", H2))
    s += bullets([
        "<b>Coupling ruling.</b> Your 26 Nov decision (non-diffusive primary, Laplacian as the published-pipeline "
        "reference) and the remote session's 27 Nov decision (keep Laplacian) conflict; the remote session did not have "
        "the local entry. Recommendation: keep your decision, pending the rerun below.",
        "<b>Approve the fix and rerun:</b> update the self-term every sub-step in core/diffsim.py with a regression test "
        "against core/simulator.py; stability-screen teachers; rerun four arms (Laplacian fixed, non-diffusive plain "
        "with drawn g, non-diffusive centred, mismatched) on the same 8 subjects with 8–10 restarts. About one day of "
        "compute on the remote server.",
        "<b>CDC margins paper:</b> answer the ten S4 questions (Q1–Q10) so the workflow can resume.",
        "Still open: observable amplitude calibration (not the true gain); real TMS-EEG test; paper authorship.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ inventory
    s.append(P("1  What we have", H1))
    s.append(T([["Area", "What it is", "Where"],
                ["Simulators", "Network JR teacher identical to neurofe's forward pass; differentiable network "
                               "simulator (self-term hold artifact found 27 Nov, fix pending); single-node simulator with "
                               "programmable virtual TMS", "core/simulator.py, core/diffsim.py, core/node.py"],
                ["Analysis", "Fisher / Cramér–Rao and profile likelihood; exact fixed points + eigenvalues; Laplace "
                             "posterior and calibration; network stability checks; coupling diagnostics",
                 "scripts/single_node_tms.py, laplace_calibration.py, core/stability.py, diagnose_*.py"],
                ["Reports (PDF)", "Reports and plans; see Table 2", "reports/"],
                ["Plans, notes, papers", "Study plans; decision log; paper draft; control formulation; CDC margins "
                                         "paper workflow (plan, theory, code, results)",
                 "docs/, reports/DECISIONS.md, paper/"],
                ["Data", "PyTepFit leadfield, connectome, distances; real TMS-EEG on demand; method papers",
                 "external/pytepfit/, literature/"],
                ["Tooling", "One-command setup and re-run scripts; tests", "setup_venv.sh, scripts/*.sh, tests/"]],
               [0.14, 0.58, 0.28], "Table 1. Inventory. Results and logs are not in git; they live on the machine that "
                                  "ran them."))
    s.append(T([["Report", "Question it answers", "Headline"],
                ["single_node_tms_study.pdf", "What can backprop recover from one node at rest?",
                 "Only k and a; raw parameterisation is a separate failure"],
                ["bifurcation_followup.pdf", "Does the active state fix it?", "p = 150 partly; p = 120 multi-modal"],
                ["single_node_conclusions.pdf", "Final single-node verdict, fixes, posterior",
                 "Recipe + Laplace posterior: accurate and calibrated at p = 150"],
                ["ei_balance_bifurcation_map.pdf", "How do E/I balance and drive shape identifiability?",
                 "Single pulse useless when E dominates; rich protocol restores bounds"],
                ["jr_parameters_explainer.pdf", "What does each parameter mean, which are identifiable?",
                 "Equations, classes, bifurcations, linearised loop gains"],
                ["pd_connectivity_study_plan.pdf", "Can node connections carry PD-type mechanisms? (plan)",
                 "Staged plan with MRI priors; coupling form now under review"],
                ["closed_loop_framework.pdf", "What problem do control tools let us solve?",
                 "Measure the loop, not its parts; margins, input design, certificates"],
                ["executive_summary.pdf", "This document", "—"]],
               [0.3, 0.36, 0.34], "Table 2. Reports in syn_brain/reports/.", font=7.1))
    s.append(PageBreak())

    # ------------------------------------------------------------------ timeline
    s.append(P("2  Everything tested so far, in order", H1))
    s.append(T([["Step", "What was tested", "Result"],
                ["Phase 1 (10 nodes)", "neurofe's procedure on a clean 10-node teacher",
                 "Fail: only k; mu has zero gradient"],
                ["E1–E12 (single node)", "Rest and active states; 12+ stimulation conditions; hold-outs",
                 "Only k, a at rest; p = 150 partly works; hold-out 46–91% error"],
                ["E/I-balance map", "Drive × E/I balance; CRB; exact bifurcations",
                 "E-dominant: single pulse ~no information; rich 18–24%; bistable fold"],
                ["Fixes + Laplace posterior", "Restarts, lowest-loss, 24 random subjects",
                 "7.5% / 4.0%; 96% coverage; −6% / −3% bias"],
                ["Network N0–N2 pilot", "Coupling option; 10-node Fisher bounds; 1-subject pilot",
                 "Bounds &lt; 3%; Laplacian pilot recovers almost everything"],
                ["N2 gate, first passes", "8 subjects, single start; amplitude calibration; restart pilot",
                 "2/8 → 5/8 converge; failures are optimiser, not information"],
                ["<b>N2 gate, full (new)</b>", "8 subjects × 5 restarts, Laplacian",
                 "<b>a, k, c3·c4, mu pass; c1·c2 12%; g 27% (biased low); adversarial review: 5 restarts too few, "
                 "oracle calibration, no coverage, fitted-network stability unchecked</b>"],
                ["<b>N2.5 coupling study (new)</b>", "g's clip saturation; raising the clip; non-diffusive matched; "
                                                     "mismatched (rate-coupled data, Laplacian model)",
                 "<b>Clip kills g's gradient for high-g subjects; raising it does not help; non-diffusive (g = 100) "
                 "worse; mismatched worst (c1·c2 98%, c3·c4 54%)</b>"],
                ["<b>Audit (27 Nov, new)</b>", "Self-term held 1 ms (diffsim) vs every 0.1 ms (neurofe), 8 subjects, "
                                               "no clip",
                 "<b>Held: 6/8 diverge at 56–72 Hz; neurofe's update: 1/8 (genuine). The high-g instability is "
                 "mostly an artifact</b>"]],
               [0.2, 0.38, 0.42], "Table 3. Studies and tests, chronologically (details in reports/DECISIONS.md).",
               font=7.0))

    s.append(T(head=P("The coupling comparison, and how solid each conclusion is", H2),
               rows=[["Median error", "Laplacian (matched)", "Non-diffusive (matched, g = 100)",
                      "Mismatched (rate data, Laplacian model)"],
                     ["a / k", "0.3% / 0.3%", "1.8% / 0.2%", "4.5% / 2.4%"],
                     ["mu", "0.4%", "59.5%", "37.5%"],
                     ["c1·c2 / c3·c4", "12.0% / 4.2%", "75.2% / 9.0%", "98.1% / 54.4%"],
                     ["g", "27.1%", "58.4%", "— (no shared truth)"],
                     ["train / held-out fit error", "0.078 / 0.068", "0.081 / 0.083", "0.170 / 0.180"],
                     ["<b>How solid</b>", "<b>Not solid</b>: 6/8 subjects' dynamics contaminated by the hold "
                                          "artifact (shared by teacher and student); rerun needed",
                      "<b>Valid as measured, weak as a comparison</b>: one fixed g, plain form (degree-dependent mean "
                      "drive trades with the operating point), 5 restarts",
                      "<b>Probably solid</b>: fitted g 9–40, below where the artifact acts; fit error doubles"]],
               widths=[0.17, 0.25, 0.29, 0.29],
               cap="Table 4. Three-way comparison from network_study_plan.txt Part N2.5, with this audit's assessment.",
               font=6.9))

    # ------------------------------------------------------------------ scorecard
    s.append(P("3  The scorecard", H1))
    s.append(T([["Parameter", "Meaning", "Single pulse, rest", "Active (p = 150), rich + recipe",
                 "E-dominant active", "Network, 10 nodes (full gate, Laplacian; pending rerun)"],
                ["k", "stimulation gain", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "recovered (0.7%)"],
                ["a", "excitatory time constant", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "recovered (0.3%)"],
                ["b", "inhibitory time constant", "WEAKLY", "IDENTIFIABLE", "WEAKLY", "recovered (1.5%)"],
                ["c3·c4", "inhibitory loop", "NOT", "IDENTIFIABLE (4.0%)", "WEAKLY (~20%)", "recovered (4.2%)"],
                ["c1·c2", "excitatory loop", "NOT", "IDENTIFIABLE (7.5%)", "NOT (~40%)", "partly (12%)"],
                ["c1…c4 alone", "which synapse", "NOT", "NOT", "NOT", "c3, c4 fair; c1/c2 split no"],
                ["A, B, vmax", "PSP sizes, max rate", "NOT (symmetry)", "NOT", "NOT", "not (fixed)"],
                ["mu", "conduction velocity", "—", "—", "—", "recovered (0.4%)"],
                ["g", "network coupling", "—", "—", "—", "Fisher &lt; 3%; backprop 27% (artifact-linked)"],
                ["coupling form", "Laplacian vs rate", "—", "—", "—", "<b>wrong form → wrong E/I</b>"]],
               [0.13, 0.15, 0.14, 0.22, 0.14, 0.22], "Table 5. Identifiability by state. Network column: full gate "
                                                     "with Laplacian coupling, no noise, matched teacher; 6 of 8 "
                                                     "subjects affected by the self-term artifact, so these numbers "
                                                     "are provisional.",
               font=6.9))
    s.append(PageBreak())

    # ------------------------------------------------------------------ what it means
    s.append(P("4  What it means", H1))
    s += bullets([
        "<b>The published method's failures are mostly fixable method problems</b> (raw parameterisation, integer "
        "delays, single-condition stimulation, single fits). With those fixed, the clean 10-node case recovers the "
        "single-node-identifiable quantities and conduction velocity.",
        "<b>The bottleneck is usually the optimiser, not the information</b>, at every scale, with one new exception: "
        "g's failure was traced to the simulator, not to the optimiser or the data.",
        "<b>The coupling form is a modelling decision with large consequences.</b> If the brain couples regions "
        "through firing rates and we fit a Laplacian model, the E/I estimates are wrong even when the fit looks fine. "
        "The coupling must be chosen on physiological grounds and then made identifiable (centred form, protocol, "
        "restarts), not chosen because it is easier to fit in a matched test.",
        "<b>Implementation details are part of the model.</b> A 1 ms hold on one term of the coupling turned a stable "
        "loop into a 60 Hz oscillator for most subjects; teacher and student shared it, so the matched test could not "
        "see it. Every network result needs a check against the reference simulator (core/simulator.py = neurofe).",
        "<b>The protocol is part of the measurement</b>; stimulation must be calibrated per subject with an observable "
        "quantity.",
    ])

    # ------------------------------------------------------------------ review
    s.append(P("5  Review of the latest remote work", H1))
    s += bullets([
        "<b>What was done well.</b> The full gate was run and reported honestly; an adversarial review retracted a false "
        "'bonus' claim and listed what the gate does not cover; the g diagnosis was quantitative (clip gradient "
        "survival vs error, ρ −0.79) and the 'raise the clip' fix was tested and rejected on evidence.",
        "<b>What was missed.</b> diffsim's coupling differs from neurofe's (self-term held per ms). The observed "
        "signature — coupling settling at a constant 3–4× the clip at every clip level — is exactly that of a limit "
        "cycle sized by the saturation, which a high-gain loop destabilised by a sample-and-hold produces. Checking "
        "against core/simulator.py would have shown it.",
        "<b>A reasoning gap.</b> 'Matched non-diffusive and mismatched are both worse than matched Laplacian, so keep "
        "Laplacian' compares how easy each model is to fit, not which one is true. The mismatch result says the "
        "opposite: getting the coupling right matters more than ease of fitting.",
        "<b>Process.</b> The remote decision to keep Laplacian conflicts with William's earlier decision, which was only "
        "in the local log. Pushing DECISIONS.md before remote runs would avoid this.",
    ])

    # ------------------------------------------------------------------ next steps
    s.append(T(head=P("6  Next steps", H1), rows=[["#", "Step", "Why", "Effort", "Gate / output"],
                ["1", "Fix core/diffsim.py: update the self-term every sub-step; regression test against "
                      "core/simulator.py", "Removes the artifact", "hours", "Max difference vs simulator.py at "
                                                                              "float precision"],
                ["2", "Stability screen for teachers (exact equilibria + delayed linearisation; redraw unstable, e.g. "
                      "subject 3)", "Unstable teachers are not valid subjects", "hours", "All teachers certified "
                                                                                          "stable"],
                ["3", "Four-arm rerun, 8 subjects × 8–10 restarts: Laplacian (fixed), non-diffusive plain (drawn g), "
                      "non-diffusive centred, mismatched", "Decide the coupling on clean evidence", "~1 day compute",
                 "Same thresholds as N2"],
                ["4", "Coupling decision (William), then network posterior and observable amplitude calibration",
                 "Gate coverage criterion; real-data feasibility", "3–5 days", "90% coverage 80–98%"],
                ["5", "Realistic teacher (finer step, noise); late-window scores", "Removes the inverse crime", "2–3 days",
                 "Gate holds under realistic data"],
                ["6", "CDC margins paper: resume at S4 after Q1–Q10", "Single node; unaffected by the coupling issue",
                 "~1 day agents", "PDF draft"],
                ["7", "Real TMS-EEG with the cleaned-up fitting", "The decisive real-data test", "1–2 weeks",
                 "Real-data hold-out"]],
               widths=[0.04, 0.38, 0.25, 0.12, 0.21], cap="Table 6. Steps 1–2 before anything else at network scale; "
                                                          "20 nodes only after step 4.", font=6.9))

    # ------------------------------------------------------------------ CDC paper
    s.append(P("7  CDC margins paper (control line), status", H1))
    s += bullets([
        "<b>Done (S1–S3):</b> fold and modulus margins are identifiable invariants of the small-signal response "
        "(sympy-verified proof), invariant to the unknown EEG readout gain; numerics validated (&lt; 0.5%); Popov "
        "certificates track the margin (ρ 0.91), with no false certificates, but only for small signals.",
        "<b>S4 (precision):</b> H1 and H2 pass (margin bounds 2.8% / 5.2%; 28/32 intervals cover), but margins are not "
        "uniquely precise (c3·c4 is as good with the rich protocol); they win where a loop saturates.",
        "<b>Important finding:</b> on the active branch, shifting E/I toward excitation <i>raises</i> the local margin "
        "(the excitatory loop saturates). Hyperexcitability shows up as nearness to a state switch (distance to the "
        "fold, large-pulse threshold), not as a small local margin. The paper is being reframed around this.",
    ])

    # ------------------------------------------------------------------ product
    s.append(P("8  Product-driven view", H1))
    s += bullets([
        "<b>Not productizable:</b> per-patient digital twins; claims about specific synapses; per-edge maps.",
        "<b>Candidate A — a TMS-EEG excitability / inhibition profile:</b> still the best candidate, with per-subject "
        "intensity calibration and ensembles of fits. New requirement: the network model's coupling form must be "
        "justified, because a wrong form biases the E/I readout.",
        "<b>Readout design:</b> report loop-level quantities (margins, loop gains, distance to a state switch) with "
        "intervals and the protocol that makes them identifiable.",
    ])

    # ------------------------------------------------------------------ housekeeping
    s.append(P("9  Open issues and housekeeping", H1))
    s += bullets([
        "A stash-pop merge conflict in DECISIONS.md and network_study_plan.txt was resolved by keeping both sides; the "
        "stash entry is still in the stash list.",
        "Network results are matched, noise-free teachers; CLAUDE.md asks for a finer-step, different-integrator "
        "teacher with noise.",
        "Uncommitted local work: closed_loop_framework.pdf, control_formulation.txt, paper/cdc_margins/, the audit "
        "script. Zheng Wang's manuscripts stay out of git.",
    ])
    s.append(P("How this summary could be wrong", H2))
    s += bullets([
        "The artifact audit replays true teacher parameters, not the fits; the effect on fitted numbers is inferred, not "
        "measured, until the rerun. Teacher 3's instability is genuine at 0.01 ms but its mechanism (delays vs local "
        "loop) is not yet identified.",
        "The single-node calibration covers 24 well-behaved subjects at one operating point. Product views are "
        "opinions, not measurements.",
    ])
    return s


def main():
    build(content(), OUT, "syn_brain executive summary", f"syn_brain · executive summary · {DATE}")


if __name__ == "__main__":
    main()
