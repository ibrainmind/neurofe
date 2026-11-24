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
DATE = "24 November 2026 (v3)"


def T(rows, widths, cap, font=7.4):
    return KeepTogether([table(rows, widths, font=font), P(cap, CAP)])


def content():
    s = [P("syn_brain: where we are", TITLE),
         P(f"Executive summary of the synthetic-brain identifiability work · {DATE} · "
           "<b>Status:</b> single-node study closed with a validated recipe and posterior; 10-node network pilot "
           "passed; network gate (≥ 8 subjects) next", CAP)]

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
        "<b>New: at 10 nodes, the global JR parameters are recoverable — including the coupling g and the "
        "conduction velocity mu, which Phase 1 could not move.</b> → <b>Yes, in a pilot.</b> With the cleaned-up "
        "fitting (log-scaled parameters, fractional delays, 8 stimulation sites × 3 amplitudes) and neurofe's own "
        "coupling, one start recovers a, b, g, c3, c4, mu, k within 4% and c1·c2 within 17%, from starting errors up "
        "to 96%. An independent earlier run with the same design agrees. Even with one site and one amplitude, most "
        "parameters were recovered and held-out sites predicted within 2–8%. <b>So Phase 1 failed mainly because of "
        "the fitting method, not the network.</b> Caveats: 1–2 subjects, no noise, teacher and student share code, "
        "only whole-window prediction tested.", VERDICT_OK))
    s.append(P(
        "<b>Where it still fails</b> → <b>Excitation-dominant states</b> (single node: no information from a single "
        "pulse; c1·c2 ~40% even with rich stimulation), the c1-vs-c2 split (never identifiable), and late-window "
        "prediction (46–91% error for the single node; not yet tested for networks).", BOX))
    s.append(P(
        "<b>When to move to 20 nodes</b> → <b>After the 10-node gate, in ~1–2 weeks.</b> First: ≥ 8 random subjects "
        "at 10 nodes with held-out-site and late-window tests, a more realistic teacher (finer time step + sensor "
        "noise), and a Fisher check of per-edge gains (tier S2) at 10 nodes. Then 20 nodes at tier S1 — one change "
        "at a time (section 6).", BOX))
    s.append(P("Decisions needed from William", H2))
    s += bullets([
        "Approve the 10-node gate design (section 6): ≥ 8 subjects, held-out-site and late-window tests, realistic "
        "teacher (finer time step, sensor noise). Background-drive coupling is no longer a prerequisite.",
        "Approve applying the cleaned-up fitting to real TMS-EEG (Ye's data) at tier S1: now the most informative "
        "real-data test (section 6, step 5).",
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
                ["Laplace posterior", "24 random teachers, honest (nominal) starts, rich protocol, p = 150",
                 "Median error 7.5% / 4.0%; 96% coverage of 90% intervals; −6% / −3% bias"],
                ["<b>Network plan N0–N1 (new)</b>", "Coupling option with background drive; 10-node Fisher bounds, "
                                                    "8 sites × 3 amplitudes", "<b>Bounds < 3% for all parameters with "
                                                                             "neurofe's own coupling; the new coupling "
                                                                             "saturates at nominal g (bug found)</b>"],
                ["<b>Network pilot N2 (new)</b>", "10 nodes, one subject, one start, both couplings",
                 "<b>neurofe's coupling recovers everything but the c1/c2 split; the new coupling gets stuck</b>"]],
               [0.2, 0.38, 0.42], "Table 3. Studies and tests, chronologically (details in reports/DECISIONS.md).",
               font=7.0))

    # ------------------------------------------------------------------ scorecard
    s.append(P("3  The scorecard", H1))
    s.append(T([["Parameter", "Meaning", "Single pulse, rest", "Active (p = 150), rich protocol + recipe",
                 "E-dominant active", "Network, 10-node pilot"],
                ["k", "stimulation gain", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "recovered (pilot)"],
                ["a", "excitatory time constant", "IDENTIFIABLE", "IDENTIFIABLE", "IDENTIFIABLE", "recovered (pilot)"],
                ["b", "inhibitory time constant", "WEAKLY", "IDENTIFIABLE", "WEAKLY", "recovered (pilot)"],
                ["c3·c4", "inhibitory loop", "NOT", "IDENTIFIABLE (4.0%, calibrated)", "WEAKLY (~20%)",
                 "recovered (pilot, ~2%)"],
                ["c1·c2", "excitatory loop", "NOT", "IDENTIFIABLE (7.5%, calibrated)", "NOT (~40%)",
                 "partly (pilot, −17% to −29%)"],
                ["c1…c4 alone", "which synapse", "NOT", "NOT", "NOT", "c3, c4 yes; c1/c2 split no"],
                ["A, B, vmax", "PSP sizes, max rate", "NOT (symmetry)", "NOT", "NOT", "not (fixed)"],
                ["v0, r", "threshold, slope", "fix", "fix", "fix", "fix"],
                ["g, mu", "coupling, conduction velocity", "—", "—", "—", "<b>recovered (pilot)</b>"],
                ["w_bb, leadfield", "per-edge gains, read-out", "—", "—", "—", "untested (S2/S3)"]],
               [0.13, 0.15, 0.14, 0.24, 0.14, 0.2], "Table 4. Identifiability by state. 'Recipe' = rich protocol, "
                                                    "≥ 15 restarts, lowest-loss selection, Laplace interval; tested on "
                                                    "24 well-behaved random subjects. Network column: 10-node pilot "
                                                    "(1–2 subjects, tier S1, no noise) — not yet the gate.",
               font=6.9))
    s.append(PageBreak())

    # ------------------------------------------------------------------ what it means
    s.append(P("4  What it means", H1))
    s += bullets([
        "<b>The published method's failures are mostly fixable method problems</b>: raw parameterisation, integer "
        "delays, single-condition stimulation and single fits. With those fixed, a 10-node network recovers its global "
        "parameters, including coupling and conduction velocity, in the clean synthetic case.",
        "<b>Networks help, not hurt, global parameters.</b> Many stimulation sites probe the same shared parameters "
        "from different angles, and strongly coupled neighbours receive large transient inputs that likely push them "
        "into the nonlinear range a single quiet node never reaches (hypothesis, to check).",
        "<b>The protocol is part of the measurement</b>, at both scales: rich, multi-condition stimulation makes the "
        "best fit also the most accurate fit.",
        "<b>What stays hard</b>: excitation-dominant states, the split between c1 and c2, and (single node) predicting "
        "the late response. The late-window question is the open test for networks.",
        "<b>Strategic consequence:</b> if a clean network fit works at R0, the real-data generalization failures "
        "point to the published pipeline's choices (raw parameters, per-edge gains and fitted leadfield, single site) "
        "or to model mismatch. Applying the cleaned-up fitting to real data is now the most informative next test.",
    ])

    # ------------------------------------------------------------------ review
    s.append(P("5  Review of the new network results", H1))
    s.append(P(
        "The 10-node pilot changes the plan, so it was checked before being adopted here:"))
    s += bullets([
        "<b>Not a starting-point artifact.</b> The default teacher is drawn within ±50% of nominal and the student starts "
        "at nominal: starting errors were up to 96% (g +91%, c1 +96%, b +49%, k −28%; c1·c2 +61%), and the initial "
        "prediction error was 38%. Final errors of 0.1–4% are genuine recovery.",
        "<b>Reproduced independently.</b> A run from 21 November with the same design (before the coupling option "
        "existed) gives the same picture at 150 iterations: b, g, c3, mu, k recovered (progress ≈ 1); c1·c2 −29% and "
        "still improving (pilot: −17% at 300).",
        "<b>Single site also works, mostly.</b> Two runs with one site and one amplitude recovered most parameters and "
        "predicted the 7 held-out sites within 2.0% and 8.1% (whole window); one subject's g stayed stuck (+62%).",
        "<b>Not yet tested:</b> (1) late-window and temporal hold-out at network scale — the single node's main "
        "failure; (2) a realistic teacher: teacher and student share code, time step and integrator, with no noise "
        "(the 'inverse crime' CLAUDE.md requires avoiding or declaring); (3) more than 1–2 subjects; (4) tiers S2/S3, "
        "which the published real-data pipeline uses.",
    ])

    # ------------------------------------------------------------------ next steps
    s.append(P("6  Next steps and the network timeline", H1))
    s.append(T([["#", "Step", "Why", "Effort", "Gate / output"],
                ["1", "10-node gate N2: ≥ 8 random subjects, tier S1, neurofe's coupling; held-out sites (train 6, "
                      "test 2), late-window and temporal hold-out", "Turns the pilot into a result; tests the single "
                                                                    "node's main failure at network scale",
                 "3–5 days", "Median error < 20%; held-out sites predicted; late window reported"],
                ["2", "Realistic teacher: neurofe-exact simulator at a 10× finer step + 0.42 µV sensor noise (R1)",
                 "Removes the inverse crime; the single node showed integrator effects near transitions", "2–3 days",
                 "Same gate under realistic data"],
                ["3", "Network ensemble + Laplace posterior (port the single-node calibration)",
                 "Honest uncertainty at network scale", "3–5 days", "90% intervals cover 80–98%"],
                ["4", "Fisher check of tier S2 (per-edge gains) at 10 nodes", "Decide whether per-edge gains are "
                                                                              "worth fitting at all",
                 "1–2 days", "Identifiable count for S2"],
                ["5", "Cleaned-up fitting on real TMS-EEG (Ye's data), tier S1, with Ye's hold-out tests",
                 "If held-out sites now transfer, the published pipeline's choices were the problem; if not, model "
                 "mismatch is", "1–2 weeks", "Real-data cross-site and temporal hold-out"],
                ["6", "Single-node loose ends: winner's-curse bias, calibration in harder states", "Lower priority "
                                                                                                  "now; still needed "
                                                                                                  "before disease work",
                 "1 week", "Bias-corrected recipe"],
                ["7", "Paper: draft exists (paper/draft_v0.txt); add the network section", "The network result "
                                                                                           "strengthens it", "ongoing",
                 "Submission draft"]],
               [0.04, 0.36, 0.27, 0.11, 0.22], "Table 5. Steps 1, 2 and 4 can start now in parallel.", font=7.0))
    s.append(T([["Stage", "When", "What"],
                ["10 nodes, tier S1", "<b>Now</b> (pilot passed)", "Steps 1–3 above"],
                ["10 nodes, tier S2", "After step 4's Fisher check", "Per-edge gains; separately from node count"],
                ["20 nodes, tier S1", "<b>In ~1–2 weeks</b>, once the 10-node gate passes",
                 "Same design; change one thing at a time"],
                ["50 / 100 / 200 nodes", "Later", "Above ~60 nodes there are more sources than the 62 EEG channels; "
                                                 "treat as a separate question"],
                ["Disease markers (Part B)", "After the network gate and the single-node loose ends",
                 "Resume with a tested recovery method"]],
               [0.2, 0.3, 0.5], "Table 6. When to scale.", font=7.2))
    s.append(P("Background-drive coupling is no longer a prerequisite: with neurofe's own coupling the pilot already "
               "recovers the loop parameters, while the new coupling got stuck in a poor solution from a single start "
               "(it may still help with restarts or for teacher realism; secondary track)."))

    # ------------------------------------------------------------------ product
    s.append(P("7  Product-driven view: clinical and wellness", H1))
    s += bullets([
        "<b>Not productizable:</b> a per-patient \"digital twin\"; claims about specific synapses (c1 vs c2); per-edge "
        "connection maps (untested, expected unidentifiable).",
        "<b>Candidate A — a TMS-EEG excitability / inhibition profile — got stronger again.</b> Global quantities "
        "(response gain, excitatory and inhibitory time constants, loop strengths, and now network coupling and "
        "conduction velocity) appear recoverable from a <b>multi-site, multi-intensity</b> protocol, with calibrated "
        "intervals at the single-node level. Multi-site TMS-EEG within one session is feasible in practice, so the "
        "protocol can be part of the product. Still to prove on real data: hold-out transfer, drug-response validity, "
        "test–retest, added value over plain TEP features.",
        "<b>B, protocol design</b> (which sites and intensities make a marker identifiable): near-term, low regulatory "
        "burden; the network Fisher tools now support it directly.",
        "<b>C, MRI + TMS-EEG for early PD:</b> far; needs subcortical modelling. <b>D, wellness from resting EEG:</b> "
        "untested.",
        "<b>Clinical vs wellness:</b> clinical needs validity, reliability and added value on patient data, then "
        "medical-software clearance; start as a research and pharma tool. Wellness avoids diagnostic claims but needs "
        "resting-state identifiability first.",
        "<b>Design principle:</b> report only identifiable quantities, with intervals, with the protocol that makes "
        "them identifiable, and say \"not determined\" when the data don't decide.",
    ])

    # ------------------------------------------------------------------ housekeeping
    s.append(P("8  Open issues and housekeeping", H1))
    s += bullets([
        "Network results so far are R0 (matched teacher, no noise); CLAUDE.md asks for a finer-step, different-"
        "integrator teacher with noise unless a spec says 'matched' (step 2).",
        "The new coupling option saturates neurofe's input bound (500) at nominal g; any use needs g ≈ 100–120.",
        "Winner's-curse bias in the single-node recipe; harder-state calibration (step 6).",
        "Phase spec files (study/phaseN_*.md) were never written; docs/network_study_plan.txt now serves for the "
        "network phase.",
        "Misspecification rungs R2–R6 not built; integrator caveat near oscillation boundaries.",
        "Results and logs live on the machines that produced them, not in git.",
    ])
    s.append(P("How this summary could be wrong", H2))
    s += bullets([
        "The network result rests on 1–2 subjects with a matched, noise-free teacher and whole-window metrics; it "
        "could weaken with realistic data or on the late window (steps 1–2 test exactly this).",
        "The single-node calibration covers 24 well-behaved subjects at one safe operating point; harder states are "
        "untested. Product views are opinions, not measurements.",
    ])
    return s


def main():
    build(content(), OUT, "syn_brain executive summary", f"syn_brain · executive summary · {DATE}")


if __name__ == "__main__":
    main()
