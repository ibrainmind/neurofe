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


def T(rows, widths, cap, font=7.4):
    return KeepTogether([table(rows, widths, font=font), P(cap, CAP)])


def content():
    s = [P("syn_brain: where we are", TITLE),
         P("Executive summary of the synthetic-brain identifiability work · 23 November 2026 · "
           "<b>Status:</b> single-node study closed (gate FAIL); network phase on hold", CAP)]

    # ------------------------------------------------------------------ verdicts
    s.append(P("Verdicts", H1))
    s.append(P(
        "<b>Can neurofe's backprop fitting recover Jansen-Rit (JR) parameters from TMS-EEG?</b> → <b>Only two, reliably: "
        "the stimulation gain k and the excitatory time constant a.</b> Inhibitory timing b and the inhibitory loop "
        "strength c3·c4 are recoverable only with a richer stimulation protocol and an active brain state, and not "
        "consistently across subjects. The excitatory loop c1·c2 is not reliable anywhere. A, B, vmax, v0 and r can never "
        "be identified from EEG (exact mathematical symmetries).", VERDICT_BAD))
    s.append(P(
        "<b>Do fitted models generalize?</b> → <b>No.</b> Fits reproduce what they were trained on, but with a perfect "
        "model and no noise, fitting the first 60 ms and predicting the rest gives 46–91% error in the late response. "
        "This reproduces the real-data generalization failure (temporal hold-out, held-out site) <b>without any model "
        "mismatch</b>: the failure is in the method and the stimulation design, not proof that the brain isn't JR.",
        VERDICT_BAD))
    s.append(P(
        "<b>Is there a practical recipe that works better?</b> → <b>Yes, partly.</b> Log-scaled parameters (about 250× "
        "lower loss than neurofe's raw parameters), a rich protocol (several amplitudes, paired pulses, a train, E+I "
        "injection), an active operating point away from the transitions (background drive p ≈ 150), and ≥ 20 restarts "
        "keeping the lowest-loss one. With a single pulse, the best-fitting model can be the most wrong; with the rich "
        "protocol, the best fit is the most accurate. Hard parameter bounds make things worse.", VERDICT_OK))
    s.append(P(
        "<b>What about networks (10–200 nodes)?</b> → <b>Expected to be harder, not easier.</b> The 10-node Phase 1 gate "
        "already failed (only k recovered; conduction velocity gets no gradient). neurofe's coupling gives nodes no "
        "background drive, so every node stays in the quiet state where loop parameters are invisible. Network work for "
        "loop parameters is on hold.", BOX))
    s.append(P(
        "<b>Product view</b> → A per-patient \"digital twin\" or full JR parameter map is <b>not</b> productizable on this "
        "evidence. A narrow product that reports a few identifiable quantities <b>with honest uncertainty</b>, paired with "
        "the stimulation protocol that makes them identifiable, is plausible, but must first beat plain TEP features on "
        "real data (section 6).", BOX))
    s.append(P("Decisions needed from William", H2))
    s += bullets([
        "Approve the next method test: an uncertainty-aware estimator (multi-start spread → variational Laplace or "
        "simulation-based inference) on the same single-node problem, before any network or disease-marker work.",
        "Approve the neurofe engineering fixes (log-scaled parameters, mean drive / rate coupling, fractional delays, "
        "fixed leadfield, smaller parameter set) as a parallel track.",
        "Choose the product direction to validate first (section 6): a TMS-EEG excitability/inhibition profile, "
        "protocol-design software, or MRI + TMS-EEG for early PD.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ inventory
    s.append(P("1  What we have", H1))
    s.append(T([["Area", "What it is", "Where"],
                ["Simulators", "Network JR teacher identical to neurofe's forward pass (tested to float precision); "
                               "differentiable network simulator; single-node simulator with programmable virtual TMS "
                               "(pulse shapes, E/I targeting, tonic drive), batched to train hundreds of models at once",
                 "core/simulator.py, core/diffsim.py, core/node.py"],
                ["Study code", "Teacher draws, student tiers S1–S3, metrics, misspecification ladder (R0–R1 built); "
                               "single-node experiments E1–E12; Fisher / Cramér–Rao and profile-likelihood analysis",
                 "core/*, scripts/single_node_tms.py"],
                ["Reports (PDF)", "Seven reports; see Table 2", "reports/"],
                ["Plans and notes", "Single-node study plan (network drive, disease markers, methods); background-drive "
                                    "note; re-run guide; decision log", "docs/*.txt, reports/DECISIONS.md"],
                ["Data", "PyTepFit leadfield, connectome, distances in git; real TMS-EEG fetched on demand",
                 "external/pytepfit/"],
                ["Tooling", "One-command environment setup, one-command single-node re-run, tests (4, ~1 min)",
                 "setup_venv.sh, scripts/run_single_node_tms.sh, tests/"]],
               [0.14, 0.6, 0.26], "Table 1. Inventory. Results and logs are not in git; they live on the machine that "
                                  "ran them."))
    s.append(T([["Report", "Question it answers", "Headline"],
                ["single_node_tms_study.pdf", "Which parameters can backprop recover from one node at rest, under 12 "
                                              "virtual-TMS conditions?",
                 "Only k and a; c's stay where they started; neurofe's raw parameterisation is a separate failure"],
                ["bifurcation_followup.pdf", "Does pushing the node to the active state (p = 120, 150) fix it?",
                 "p = 150 partly; p = 120 is a multi-modal trap near the transition"],
                ["single_node_conclusions.pdf", "Final single-node verdict incl. generalization, robustness, fixes",
                 "Gate FAIL; generalization fails; RICH + ≥ 20 restarts + lowest loss is the working recipe"],
                ["ei_balance_bifurcation_map.pdf", "How do E/I balance and drive shape states and identifiability?",
                 "Single pulse useless when excitation dominates; rich protocol restores most of it"],
                ["jr_parameters_explainer.pdf", "What does each parameter mean, and which are identifiable?",
                 "Equations, classes, bifurcations, hysteresis; integrator caveat"],
                ["pd_connectivity_study_plan.pdf", "Can node connections carry PD-type mechanisms? (plan)",
                 "Linear, cortex-only coupling cannot; staged plan with MRI priors"],
                ["executive_summary.pdf", "This document", "—"]],
               [0.28, 0.38, 0.34], "Table 2. Reports in syn_brain/reports/.", font=7.1))
    s.append(PageBreak())

    # ------------------------------------------------------------------ timeline
    s.append(P("2  Everything tested so far, in order", H1))
    s.append(T([["Step", "What was tested", "Result"],
                ["Phase 1 calibration", "Do synthetic TEPs look like real TMS-EEG (20 subjects)?",
                 "No: one large early peak, almost no late activity (late/early ratio 0.15–0.19 vs 0.84 real)"],
                ["Phase 1 gate (10 nodes)", "Does tier S1 recover a clean 10-node teacher?",
                 "Fail: only k; mu has zero gradient (integer delays); b, c4, std_in absorb mismatch"],
                ["E1–E8 (single node, rest)", "402 models; 12 stimulation conditions; noise; raw vs log; 12 free "
                                              "parameters; profiles; teachers; temporal hold-out",
                 "Node is a near-linear filter at rest; only k, a; raw parameters ~250× worse; hold-out 91% error"],
                ["Background-drive sweep", "Does input from the rest of the brain help?",
                 "Yes in theory: above p ≈ 115 the c3 bound drops 129% → ~5%"],
                ["E9–E10 + settle fix", "Fit at p = 120 / 150, with and without noise",
                 "p = 150 better; p = 120 fails; settle-time bug fixed but not the cause"],
                ["Continuation (p 150 → 120)", "Warm-start through the transition", "Bimodal fits: multi-modal "
                                                                                     "landscape near the fold"],
                ["Mechanism study", "Why do c1·c2 / c3·c4 fail?", "Wrong regime for some teachers; some draws have "
                                                                  "no stable active state; loops cross-couple"],
                ["E11–E12", "Temporal hold-out and 4 teachers at p = 150", "Hold-out 46% error; c1·c2 13–95% across "
                                                                          "teachers"],
                ["E/I-balance map", "15 × 19 grid of drive × E/I balance; CRB + backprop at 3 points",
                 "E-dominant: single pulse useless (CRB > 2500%); rich protocol ~10–19%"],
                ["Accuracy → generalization", "Does better parameter recovery predict better prediction?",
                 "Yes, for the identifiable combinations; not for individual c1 (as theory predicts)"],
                ["Practical fixes", "Bigger ensembles, hard bounds, lowest-loss selection",
                 "RICH + ≥ 20 restarts + lowest loss works; hard bounds double the error"],
                ["Equation-level bifurcations", "Fixed points and stability across the E/I map",
                 "Quiet/active hysteresis (p ≈ 37–115); one 'limit cycle' is an Euler-step artifact; E-dominant "
                 "loop saturates"]],
               [0.2, 0.4, 0.4], "Table 3. Studies and tests, chronologically (details in reports/DECISIONS.md).",
               font=7.0))

    # ------------------------------------------------------------------ scorecard
    s.append(P("3  The scorecard (the deliverable defined in study/00_overview.md)", H1))
    s.append(T([["Parameter", "Meaning", "Single node, rest", "Single node, active (p ≈ 150)", "Network (expected)"],
                ["k", "stimulation gain", "IDENTIFIABLE", "IDENTIFIABLE", "identifiable per site if leadfield fixed"],
                ["a", "excitatory time constant", "IDENTIFIABLE", "IDENTIFIABLE", "identifiable (global)"],
                ["b", "inhibitory time constant", "WEAKLY (needs E+I protocol)", "WEAKLY", "weakly"],
                ["c3·c4", "inhibitory loop strength", "NOT IDENTIFIABLE", "WEAKLY (rich protocol; varies by "
                                                                         "subject)", "not, with current coupling"],
                ["c1·c2", "excitatory loop strength", "NOT IDENTIFIABLE", "WEAKLY at balance; NOT when E dominates",
                 "not"],
                ["c1…c4 individually", "which synapse", "NOT IDENTIFIABLE", "NOT IDENTIFIABLE", "not"],
                ["A, B, vmax", "PSP sizes, max rate", "NOT IDENTIFIABLE (symmetry)", "same", "same"],
                ["v0, r", "threshold, slope", "NOT NECESSARY / fix", "same", "same"],
                ["g, w_bb, mu, leadfield", "network coupling, edges, delays", "—", "—",
                 "g partly; edges and leadfield not; mu not (no gradient)"]],
               [0.15, 0.18, 0.19, 0.26, 0.22], "Table 4. Network column is a prediction from the single-node results "
                                                "and Phase 1, not yet measured.", font=7.0))
    s.append(PageBreak())

    # ------------------------------------------------------------------ what it means
    s.append(P("4  What it means, in three sentences", H1))
    s += bullets([
        "<b>TMS-EEG reliably tells you how strongly the cortex responds and how fast its excitatory synapses work</b>; "
        "it tells you about the local circuit (excitation/inhibition loops) only with the right stimulation, in the "
        "right brain state, and even then only as loop strengths, never individual connections.",
        "<b>A good fit is not evidence of correct parameters.</b> With a single pulse, the best-fitting model can be the "
        "most wrong one; only richer stimulation makes \"fits best\" mean \"is right\".",
        "<b>Generalization failures seen in real data are expected from the method itself</b>, so fixing the method and "
        "the protocol comes before modelling more of the brain.",
    ])

    # ------------------------------------------------------------------ next steps
    s.append(P("5  Next steps, in priority order", H1))
    s.append(T([["#", "Step", "Why", "Effort", "Gate / output"],
                ["1", "Uncertainty-aware estimator on the single node (multi-start spread; then variational Laplace or "
                      "simulation-based inference)", "A method must be able to say \"not determined\" instead of a "
                                                         "confident wrong number", "1–2 weeks",
                 "Calibrated intervals (90% interval holds truth 80–98% of the time)"],
                ["2", "neurofe fixes: log-scaled parameters, mean drive or rate-based coupling, fractional delays, "
                      "fixed leadfield, reduced parameter set, integrator check", "Removes known, avoidable failures; "
                                                                                  "makes late TEP activity possible",
                 "1–2 weeks (parallel)", "Synthetic TEPs match real late/early ratio"],
                ["3", "Fisher spectrum of the 10-node network (S1, S2)", "Measure, don't guess, which network "
                                                                        "combinations survive", "hours",
                 "Number of identifiable combinations per tier"],
                ["4", "Real-data validity: drug effects (e.g. GABAergic) and test-retest on published TMS-EEG data",
                 "The first test that the quantities mean something biologically", "2–4 weeks",
                 "Estimates move as predicted; stable across sessions"],
                ["5", "Added value: compare against plain TEP features (amplitude, latency, N100) for a clinical "
                      "contrast", "A model product must beat the waveform it is fitted to", "2–4 weeks",
                 "Better discrimination than plain features"],
                ["6", "Resting-state identifiability (study Q1a, not yet done)", "Needed for any EEG-only (no TMS) or "
                                                                                 "wellness use", "1 week",
                 "Which quantities resting spectra can identify"]],
               [0.04, 0.33, 0.27, 0.12, 0.24], "Table 5. Recommended sequence. Steps 1–3 are synthetic and cheap; "
                                               "4–5 are the first real-data gates.", font=7.0))
    s.append(PageBreak())

    # ------------------------------------------------------------------ product
    s.append(P("6  Product-driven view: clinical and wellness", H1))
    s.append(P("What is not productizable on current evidence", H2))
    s += bullets([
        "A per-patient \"digital twin\" or full map of JR parameters: most of its numbers would reflect the fitting "
        "procedure, not the person.",
        "Claims about specific synapses (e.g. \"reduced c4\") or synapse loss vs receptor dysfunction: mathematically "
        "invisible to EEG.",
        "Network-level parameter maps (per-edge connectivity, leadfield-fitted sources).",
    ])
    s.append(P("Candidate products, from nearest to furthest", H2))
    s.append(T([["Candidate", "What it delivers", "Who uses it", "What must be proven first", "Distance"],
                ["A. TMS-EEG excitability / inhibition profile",
                 "A few model quantities with confidence: response gain, excitatory and inhibitory time constants, "
                 "inhibitory loop strength, measured with a defined rich protocol",
                 "Research labs, pharma (target engagement), epilepsy and psychiatry research",
                 "Drug-response validity, test-retest, added value over plain TEP features", "nearest (research-grade "
                                                                                           "first)"],
                ["B. Protocol-design software", "For a chosen marker, the stimulation protocol that makes it "
                                                "identifiable (from the Fisher tools we already have)",
                 "TMS-EEG labs, device makers", "That recommended protocols improve real test-retest reliability",
                 "near; low regulatory burden"],
                ["C. MRI + TMS-EEG for early PD", "Model quantities constrained by MRI (E-field fixes k; connectome; "
                                                  "nigral imaging as priors)",
                 "Movement-disorder clinics, trials", "Subcortical loop model works synthetically; MRI priors improve "
                                                      "identification; then cohort data", "far (multi-year)"],
                ["D. Wellness (EEG only, no TMS)", "E.g. an E/I-balance or 'cortical state' index from resting EEG",
                 "Consumers, wellness apps", "Resting-state identifiability (not yet studied); reliability on consumer "
                                             "hardware", "uncertain; untested"]],
               [0.17, 0.27, 0.17, 0.25, 0.14], "Table 6. Candidate products. None is validated; distance reflects "
                                               "the evidence needed, not the commercial opportunity.", font=6.9))
    s.append(P("Clinical vs wellness", H2))
    s += bullets([
        "<b>Clinical</b> (diagnosis, monitoring, treatment selection) needs validity, reliability and added value on "
        "patient data, then regulatory clearance as medical software. Product A is the realistic entry point, sold "
        "first as a research tool (e.g. measuring whether a drug engages inhibitory circuits) before any diagnostic "
        "claim. The E/I results matter here: the states of most clinical interest (too much excitation, e.g. epilepsy "
        "or hyperexcitability) are exactly where a standard single-pulse protocol tells you least, so the protocol is "
        "part of the product.",
        "<b>Wellness</b> avoids diagnostic claims and the regulatory path, but consumer devices record resting EEG "
        "without TMS, and we have not tested what resting EEG can identify. The model does predict a resting signature "
        "(the ~6–11 Hz ringing whose frequency and damping depend on drive and E/I balance), so step 6 is a cheap way to "
        "find out whether an 'E/I index' from resting EEG is honest or would be curve fitting.",
        "<b>Common design principle:</b> report only identifiable quantities, always with uncertainty, always with the "
        "protocol that makes them identifiable, and say \"not determined\" when the data don't decide. That is both the "
        "scientifically defensible product and the one that survives scrutiny.",
    ])
    s.append(P("Gates before any product investment", H2))
    s.append(P("1 synthetic (reduced model + uncertainty passes R0–R1) → 2 drug-response validity → 3 test-retest "
               "reliability → 4 added value over plain features. Failing gate 2 or 4 means a research tool, not a "
               "product, and is better learned before scaling to networks or cohorts."))

    # ------------------------------------------------------------------ housekeeping
    s.append(P("7  Open issues and housekeeping", H1))
    s += bullets([
        "Teacher realism is unresolved: synthetic TEPs still lack late activity (Phase 1 Finding 1); fix via mean drive "
        "or rate-based coupling (docs/background_drive_note.txt).",
        "Phase spec files (study/phaseN_*.md) were never written; the single-node study plan has stood in for them "
        "(DECISIONS.md, 22 Nov). Worth writing before network work restarts.",
        "Only rung R0 (and R1 noise) of the misspecification ladder has been run; R2–R6 are declared, not built.",
        "Integrator caveat: near the oscillation boundary, neurofe's 0.1 ms Euler step can create oscillations the "
        "model does not have; teacher and student time steps must be chosen with that in mind.",
        "Results and logs live on the machines that produced them, not in git; the explainer and some reports need "
        "those files to rebuild.",
    ])
    s.append(P("How this summary could be wrong", H2))
    s += bullets([
        "Everything rests on single-node, synthetic, best-case experiments with one optimizer family (Adam); a better "
        "estimator (step 1) could recover more than reported, though it cannot exceed the Cramér–Rao bounds.",
        "The network column and all product judgments are predictions and opinions, not measurements; steps 3–5 are "
        "designed to test them.",
    ])
    return s


def main():
    build(content(), OUT, "syn_brain executive summary", "syn_brain · executive summary · 23 November 2026")


if __name__ == "__main__":
    main()
