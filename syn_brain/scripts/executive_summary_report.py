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
DATE = "24 November 2026"


def T(rows, widths, cap, font=7.4):
    return KeepTogether([table(rows, widths, font=font), P(cap, CAP)])


def content():
    s = [P("syn_brain: where we are", TITLE),
         P(f"Executive summary of the synthetic-brain identifiability work · {DATE} (v2) · "
           "<b>Status:</b> single-node study closed with a validated recipe and posterior; network phase ready to "
           "start in stages (section 6)", CAP)]

    # ------------------------------------------------------------------ verdicts
    s.append(P("Verdicts", H1))
    s.append(P(
        "<b>Can neurofe's backprop fitting, as published, recover Jansen-Rit (JR) parameters from TMS-EEG?</b> → "
        "<b>Only two, reliably: the stimulation gain k and the excitatory time constant a.</b> With a single pulse at rest, "
        "the loop parameters stay where the fit started; A, B and vmax can never be identified from EEG (exact "
        "symmetries).", VERDICT_BAD))
    s.append(P(
        "<b>Do fitted models generalize?</b> → <b>No, not as published.</b> With a perfect model and no noise, fitting "
        "the first 60 ms and predicting the rest gives 46–91% late-window error. This reproduces the real-data failures "
        "(temporal hold-out, held-out site) <b>without any model mismatch</b>: the failure is in the method and the "
        "stimulation design.", VERDICT_BAD))
    s.append(P(
        "<b>New: with the right recipe, the loop strengths become measurable, with honest uncertainty.</b> → <b>Yes, "
        "in well-behaved active states.</b> Rich stimulation + active operating point (p = 150) + 15 restarts + "
        "lowest-loss selection + a Laplace posterior, tested on 24 random synthetic subjects with honest starting "
        "values: median error 7.5% (excitatory loop c1·c2) and 4.0% (inhibitory loop c3·c4); 90% intervals contain "
        "the truth 96% of the time. One catch: a small systematic underestimate (−5.9% / −3.3%; 75% of estimates "
        "too low), most likely a \"winner's curse\" from picking the best of many restarts.", VERDICT_OK))
    s.append(P(
        "<b>Where it still fails</b> → <b>Excitation-dominant states.</b> The corrected E/I map shows a single pulse "
        "carries essentially no information there (bounds in the hundreds of thousands of percent), because the "
        "excitatory interneurons saturate. Rich stimulation restores the bounds (18–24%), but backprop still misses "
        "c1·c2 (~40% error). Near the quiet/active transition the model is bistable (hysteresis), and fits are "
        "multi-modal.", BOX))
    s.append(P(
        "<b>When to move to 10–20 nodes</b> → <b>Start now in two stages.</b> Stage 1 (now, ~days): the 10-node "
        "Fisher analysis, which needs no fitting. Stage 2 (in ~1–2 weeks): the 10-node fitting study, once three "
        "short single-node items are closed: the winner's-curse check, calibration in harder states, and a network "
        "coupling that gives nodes background drive. Without that last item, every network node stays quiet and the "
        "network study would only repeat the at-rest failure. 20 nodes follow only if 10 passes (section 6).", BOX))
    s.append(P("Decisions needed from William", H2))
    s += bullets([
        "Approve the staged network plan (section 6), in particular adding background drive / rate-based coupling to "
        "the network simulator before any 10-node fitting.",
        "Approve the three remaining single-node checks (winner's curse, harder-state calibration, holdout-based "
        "selection) as the gate for Stage 2.",
        "Paper: a tentative draft exists (paper/draft_v0.txt); decide authorship and whether to involve the original "
        "method authors.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ inventory
    s.append(P("1  What we have", H1))
    s.append(T([["Area", "What it is", "Where"],
                ["Simulators", "Network JR teacher identical to neurofe's forward pass; differentiable network "
                               "simulator; single-node simulator with programmable virtual TMS (pulse shapes, E/I "
                               "targeting, tonic drive), batched", "core/simulator.py, core/diffsim.py, core/node.py"],
                ["Analysis", "Fisher / Cramér–Rao and profile likelihood; exact fixed points + eigenvalues "
                             "(bifurcations); Laplace posterior and calibration across random teachers",
                 "scripts/single_node_tms.py, jr_parameters_explainer.py, laplace_calibration.py"],
                ["Reports (PDF)", "Eight reports; see Table 2", "reports/"],
                ["Plans, notes, paper", "Single-node study plan; background-drive note; re-run guide; decision log; "
                                        "tentative paper draft", "docs/, reports/DECISIONS.md, paper/"],
                ["Data", "PyTepFit leadfield, connectome, distances in git; real TMS-EEG fetched on demand; the three "
                         "method papers", "external/pytepfit/, literature/"],
                ["Tooling", "One-command setup and single-node re-run; tests", "setup_venv.sh, "
                                                                              "scripts/run_single_node_tms.sh, tests/"]],
               [0.14, 0.58, 0.28], "Table 1. Inventory. Results and logs are not in git; they live on the machine that "
                                  "ran them."))
    s.append(T([["Report", "Question it answers", "Headline"],
                ["single_node_tms_study.pdf", "What can backprop recover from one node at rest, 12 stimulation "
                                              "conditions?", "Only k and a; raw parameterisation is a separate failure"],
                ["bifurcation_followup.pdf", "Does the active state (p = 120, 150) fix it?",
                 "p = 150 partly; p = 120 is a multi-modal trap"],
                ["single_node_conclusions.pdf", "Final single-node verdict, fixes, posterior",
                 "Recipe + Laplace posterior: accurate and calibrated at p = 150; small downward bias"],
                ["ei_balance_bifurcation_map.pdf (corrected)", "How do E/I balance and drive shape states and "
                                                              "identifiability?",
                 "Single pulse useless when E dominates; rich protocol restores bounds; hysteresis at the fold"],
                ["jr_parameters_explainer.pdf", "What does each parameter mean, which are identifiable?",
                 "Equations, classes, bifurcations, hysteresis, integrator caveat"],
                ["pd_connectivity_study_plan.pdf", "Can node connections carry PD-type mechanisms? (plan)",
                 "Linear, cortex-only coupling cannot; staged plan with MRI priors"],
                ["executive_summary.pdf", "This document", "—"]],
               [0.3, 0.36, 0.34], "Table 2. Reports in syn_brain/reports/.", font=7.1))
    s.append(PageBreak())

    # ------------------------------------------------------------------ timeline
    s.append(P("2  Everything tested so far, in order", H1))
    s.append(T([["Step", "What was tested", "Result"],
                ["Phase 1 calibration", "Do synthetic TEPs look like real TMS-EEG?",
                 "No: one large early peak, little late activity (late/early 0.15–0.19 vs 0.84 real)"],
                ["Phase 1 gate (10 nodes)", "Does tier S1 recover a clean 10-node teacher?",
                 "Fail: only k; mu has zero gradient; b, c4, std_in absorb mismatch"],
                ["E1–E8 (single node, rest)", "402 models; 12 stimulation conditions; noise; raw vs log; profiles; "
                                              "teachers; temporal hold-out",
                 "Near-linear filter at rest; only k, a; raw parameters ~250× worse; hold-out 91% error"],
                ["E9–E12, p = 120 / 150", "Active state; settle fix; continuation; 4 teachers; hold-out",
                 "p = 150 partly works; p = 120 multi-modal; c1·c2 13–95% across teachers; hold-out 46%"],
                ["Mechanism study", "Why do c1·c2 / c3·c4 fail?", "Wrong regime for some teachers; loops "
                                                                  "cross-couple when nonlinear"],
                ["E/I-balance map (corrected 24 Nov)", "Drive × E/I balance; CRB; backprop at 3 points; exact "
                                                       "bifurcation analysis",
                 "E-dominant: single pulse ~no information; rich 18–24%; bistable fold; one 'oscillation' was an "
                 "integrator artifact"],
                ["Accuracy → generalization", "Does better recovery predict better prediction?",
                 "Yes, for identifiable combinations; not for individual c1"],
                ["Practical fixes", "Ensembles, hard bounds, lowest-loss selection",
                 "Rich + many restarts + lowest loss works; hard bounds double the error"],
                ["<b>Laplace posterior (new)</b>", "24 random teachers, honest (nominal) starts, rich protocol, "
                                                   "p = 150", "<b>Median error 7.5% / 4.0%; 96% coverage of 90% "
                                                              "intervals; −6% / −3% bias</b>"]],
               [0.2, 0.38, 0.42], "Table 3. Studies and tests, chronologically (details in reports/DECISIONS.md).",
               font=7.0))

    # ------------------------------------------------------------------ scorecard
    s.append(P("3  The scorecard", H1))
    s.append(T([["Parameter", "Meaning", "Single pulse, rest", "Active (p = 150), rich protocol + recipe",
                 "E-dominant active", "Network (expected)"],
                ["k", "stimulation gain", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "per site, if leadfield "
                                                                                          "fixed"],
                ["a", "excitatory time constant", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "global"],
                ["b", "inhibitory time constant", "WEAKLY", "IDENTIFIABLE", "WEAKLY", "weakly"],
                ["c3·c4", "inhibitory loop", "NOT", "IDENTIFIABLE (4.0%, calibrated)", "WEAKLY (~20%)",
                 "only with mean drive"],
                ["c1·c2", "excitatory loop", "NOT", "IDENTIFIABLE (7.5%, calibrated)", "NOT (~40%)",
                 "only with mean drive; hardest"],
                ["c1…c4 alone", "which synapse", "NOT", "NOT", "NOT", "not"],
                ["A, B, vmax", "PSP sizes, max rate", "NOT (symmetry)", "NOT", "NOT", "not"],
                ["v0, r", "threshold, slope", "fix", "fix", "fix", "fix"],
                ["g, w_bb, mu, leadfield", "network", "—", "—", "—", "g partly; edges / leadfield not; mu not"]],
               [0.13, 0.15, 0.14, 0.24, 0.14, 0.2], "Table 4. Identifiability by state. 'Recipe' = rich protocol, "
                                                    "≥ 15 restarts, lowest-loss selection, Laplace interval; tested on "
                                                    "24 well-behaved random subjects. Network column is a prediction.",
               font=6.9))
    s.append(PageBreak())

    # ------------------------------------------------------------------ what it means
    s.append(P("4  What it means", H1))
    s += bullets([
        "<b>The method as published cannot support claims about the local circuit</b>; the recipe can, in the right "
        "brain state, and now with intervals that are honest about their uncertainty.",
        "<b>The protocol is part of the measurement.</b> Rich stimulation doesn't just tighten bounds; it makes the "
        "best fit also the most accurate fit. With a single pulse, the best fit can be the most wrong one.",
        "<b>Brain state decides what is measurable.</b> Excitation-dominant columns hide their excitatory loop "
        "(saturation); near the quiet/active transition the column is bistable and fits are multi-modal.",
        "<b>Generalization failures in real data are expected from the published method itself</b>, so the method "
        "and protocol come before modelling more of the brain.",
    ])

    # ------------------------------------------------------------------ next steps
    s.append(P("5  Next steps", H1))
    s.append(T([["#", "Step", "Why", "Effort", "Output"],
                ["1", "Winner's-curse check: does the bias grow with the number of restarts? Fix by selecting on a "
                      "held-out stimulation condition, or by averaging the best few restarts",
                 "Removes a known, systematic error before anything is built on point estimates", "2–3 days",
                 "Unbiased (or bias-corrected) recipe"],
                ["2", "Calibration in harder states: E-dominant, near the fold, near oscillation", "Disease "
                                                                                                    "mechanisms will "
                                                                                                    "push subjects there",
                 "3–5 days", "Map of where intervals can be trusted"],
                ["3", "10-node Fisher spectrum (tiers S1, S2), no fitting", "Measures how many combinations a "
                                                                           "network can identify", "1–2 days",
                 "Identifiable count per tier; go/no-go for S2"],
                ["4", "Network coupling with background drive (tonic p per node, or rate-based coupling), validated "
                      "in the simulator", "Without it every network node is quiet and loops are invisible",
                 "3–5 days", "Nodes in a stable active state; late TEP activity"],
                ["5", "10-node fitting study (section 6)", "First real test at network scale", "1–2 weeks",
                 "Network scorecard; calibration; hold-out tests"],
                ["6", "Real-data gates: drug-response validity, test–retest, added value over plain TEP features",
                 "Required before any product claim", "2–4 weeks each", "Go/no-go for product"],
                ["7", "Paper: figures from existing results; network and real-data sections", "Publishable now in "
                                                                                              "outline",
                 "ongoing", "Submission-ready draft in ~6–10 weeks"]],
               [0.04, 0.34, 0.27, 0.12, 0.23], "Table 5. Steps 1, 3 and 4 can run in parallel this week.", font=7.0))
    s.append(PageBreak())

    # ------------------------------------------------------------------ network timing
    s.append(P("6  When to move to 10 or 20 nodes", H1))
    s.append(P(
        "The single node has now produced what a network study needs from it: a recipe that works in well-behaved "
        "states and a calibrated posterior. What it has <i>not</i> settled can mostly be done in parallel. The one "
        "real blocker is the network model itself: neurofe's coupling gives nodes no background drive, so in a "
        "network every node rests in the quiet state where the loops cannot be seen. Fitting a network before fixing "
        "that would repeat the single-node at-rest failure at much higher cost."))
    s.append(T([["Stage", "When", "What", "Gate to continue"],
                ["1  Structural, no fitting", "Now", "10-node Fisher spectrum for tiers S1 (global parameters) and "
                                                     "S2 (+ per-edge gains), at rest and with background drive; "
                                                     "single site vs several sites",
                 "Know how many combinations are identifiable vs fitted"],
                ["2  10-node fitting", "In ~1–2 weeks, after steps 1, 2, 4 of section 5",
                 "Tier S1 + one coupling gain g + per-node or global drive; rich protocol at 2–4 stimulation sites; "
                 "recipe + Laplace posterior; 10–20 random teachers; temporal and held-out-site tests",
                 "Calibrated intervals (80–98% coverage); loop products within 20%; held-out site predicted"],
                ["3  20 nodes", "Only if 10 nodes pass", "Same design; mainly more edges and compute",
                 "Same gates; check that the identifiable count grows with data, not just with parameters"],
                ["4  50 / 100 / 200 nodes", "Later", "Scaling per study/00_overview.md",
                 "Above ~60 nodes there are more sources than the 62 EEG channels, so spatial identifiability "
                 "changes; treat this as a separate question"]],
               [0.17, 0.17, 0.38, 0.28], "Table 6. Staged network plan.", font=7.0))
    s += bullets([
        "<b>Why 10 before 20:</b> at 10 nodes a fitting epoch takes ~5 s and the per-edge tier has ~100 extra parameters "
        "instead of ~400, so failures are cheap to diagnose. 20 nodes adds edges and compute, not a new question.",
        "<b>Why stay below ~60 nodes first:</b> with fewer brain regions than EEG channels the read-out can in "
        "principle separate the sources; above that it cannot, and identifiability changes for reasons unrelated to "
        "the neural model.",
        "<b>Start with tier S1 (global parameters + one coupling gain).</b> Per-edge gains (S2) and the fitted leadfield "
        "(S3) are expected to be unidentifiable; Stage 1 will say how much S2 is worth.",
        "<b>Several stimulation sites</b>: each site informs the connections leaving it; held-out-site prediction is "
        "the network version of the temporal hold-out.",
    ])

    # ------------------------------------------------------------------ product
    s.append(P("7  Product-driven view: clinical and wellness", H1))
    s += bullets([
        "<b>Not productizable:</b> a per-patient \"digital twin\" or full JR parameter map; claims about specific "
        "synapses; network connection maps.",
        "<b>Stronger than a week ago — candidate A, a TMS-EEG excitability / inhibition profile.</b> Response gain, "
        "excitatory and inhibitory time constants, and now inhibitory and excitatory loop strengths <i>with calibrated "
        "intervals</i>, measured with a defined rich protocol in an active state. Still to prove: drug-response "
        "validity, test–retest, added value over plain TEP features. Caveat: excitation-dominant states, often the "
        "clinically interesting ones, remain the weak spot for the excitatory loop.",
        "<b>B, protocol-design software</b> (which stimulation makes a marker identifiable): near-term, low "
        "regulatory burden; the rich-protocol and E/I results are its core.",
        "<b>C, MRI + TMS-EEG for early PD:</b> far; depends on the network stage and subcortical modelling.",
        "<b>D, wellness from resting EEG (no TMS):</b> untested; resting-state identifiability is still open.",
        "<b>Clinical vs wellness:</b> clinical needs validity, reliability and added value on patient data, then "
        "medical-software clearance; start as a research and pharma tool. Wellness avoids diagnostic claims but "
        "needs the resting-state question answered first.",
        "<b>Design principle:</b> report only identifiable quantities, with intervals, with the protocol that makes "
        "them identifiable, and say \"not determined\" when the data don't decide.",
    ])

    # ------------------------------------------------------------------ housekeeping
    s.append(P("8  Open issues and housekeeping", H1))
    s += bullets([
        "Winner's-curse bias in the recipe (section 5, step 1).",
        "Teacher realism: synthetic TEPs lack late activity until background drive is added (same fix as step 4).",
        "Phase spec files (study/phaseN_*.md) were never written; worth writing for the network phase.",
        "Misspecification rungs R2–R6 are declared but not built; R0–R1 only so far.",
        "Integrator caveat near oscillation boundaries (neurofe's 0.1 ms Euler step).",
        "Results and logs live on the machines that produced them, not in git.",
    ])
    s.append(P("How this summary could be wrong", H2))
    s += bullets([
        "The calibration result covers 24 well-behaved subjects at one safe operating point; harder states are "
        "untested (step 2).",
        "The network column and the product views are predictions and opinions, not measurements; Stages 1–2 and "
        "the real-data gates are designed to test them.",
    ])
    return s


def main():
    build(content(), OUT, "syn_brain executive summary", f"syn_brain · executive summary · {DATE}")


if __name__ == "__main__":
    main()
