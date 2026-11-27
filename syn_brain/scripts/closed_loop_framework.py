"""Build "Measuring brain loops: a closed-loop identification framework for stimulation-evoked EEG" (PDF).

The framework document: the problem, control-theoretic foundations, four core problems, and applications
(cortical TMS-EEG, network scaling, subcortical observer-controller modelling, Parkinson's disease, the EEG
product, neuromodulation). Supersedes the earlier subcortex_modeling_plan.pdf, whose content is section 5.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.closed_loop_framework
Output: syn_brain/reports/closed_loop_framework.pdf  (needs no results files)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, KeepTogether, PageBreak

from syn_brain.core.paths import RESULTS_DIR, SYN_ROOT
from syn_brain.core.report_style import BOX, CAP, TITLE, W, P, build, bullets, table
from syn_brain.core.report_style import H1 as _H1, H2 as _H2

H1 = ParagraphStyle("h1k", parent=_H1, keepWithNext=1)
H2 = ParagraphStyle("h2k", parent=_H2, keepWithNext=1)
OUT = SYN_ROOT / "reports" / "closed_loop_framework.pdf"
FIG_DIR = RESULTS_DIR / "plans"
DATE = "27 November 2026"


# ============================================================================ figures
def _box(ax, x, y, w, h, text, fc, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arr(ax, x0, y0, x1, y1, col="#333", style="-|>", ls="-", rad=0.0, lab=None, lx=None, ly=None, fs=6.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=col, lw=1.2,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if lab:
        ax.text(lx, ly, lab, fontsize=fs, color=col, ha="center")


def lure_figure(path):
    """A Jansen-Rit column as a Lur'e system: linear synaptic dynamics in feedback with bounded sigmoids."""
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.30, 0.55, 0.30, 0.28, "Linear dynamics G(s)\nsynaptic kernels\nH_a = a/(s+a)², H_b = b/(s+b)²\n"
                                     "gains A, B, k; delays", "#cfe3f5")
    _box(ax, 0.30, 0.08, 0.30, 0.26, "Static nonlinearities\nsigmoids S(·), slope 0 ≤ S' ≤ 0.70\n"
                                     "(+ input bounds), connection c's", "#f8d7da")
    _box(ax, 0.02, 0.58, 0.17, 0.22, "Stimulation u(t)\n(TMS pulses,\nsites, intensities)", "#fdebd0")
    _box(ax, 0.78, 0.58, 0.19, 0.22, "EEG y(t)\n(leadfield read-out)", "#d9f0d3")
    _arr(ax, 0.19, 0.69, 0.30, 0.69, "#d68910")
    _arr(ax, 0.60, 0.69, 0.78, 0.69, "#2a9d8f")
    _arr(ax, 0.60, 0.60, 0.60, 0.30, rad=-0.6, lab="internal states\n(P, E, I)", lx=0.73, ly=0.40)
    _arr(ax, 0.30, 0.22, 0.30, 0.60, rad=-0.6, lab="loop feedback\n(λ_E positive, λ_I negative)", lx=0.15, ly=0.33)
    ax.text(0.5, -0.04, "Measurable: the input-output map u → y (gains, poles, loop gains, margins). "
                        "Not measurable from y alone: how it is split among the parts.", ha="center", fontsize=6.9)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def loop_figure(path):
    """Two-state observer-controller loop with routed, signed subcortical pathways and perturbation entry points."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.36, 0.70, 0.28, 0.16, "Cortex (plant)\nstate x(t)\nJR network or linear plant", "#cfe3f5")
    _box(ax, 0.05, 0.27, 0.27, 0.17, "Striatum (D1 / D2)\nbasal ganglia route\nestimate x̂_BG", "#eeeeee")
    _box(ax, 0.68, 0.27, 0.27, 0.17, "Thalamus\nrelay + own route\nestimate x̂_TH", "#d9f0d3")
    _box(ax, 0.37, 0.03, 0.26, 0.13, "GPi / SNr output\n(inhibitory)", "#f8d7da")
    _arr(ax, 0.40, 0.70, 0.22, 0.44, "#1f6fb4", lab="observation C_BG\n(cortex→striatum SC)", lx=0.19, ly=0.58)
    _arr(ax, 0.60, 0.70, 0.78, 0.44, "#1f6fb4", lab="observation C_TH\n(cortex→thalamus SC)", lx=0.60, ly=0.53)
    _arr(ax, 0.18, 0.27, 0.40, 0.12, "#555")
    _arr(ax, 0.63, 0.09, 0.78, 0.27, "#c0392b", style="-[", lab="inhibitory", lx=0.77, ly=0.15)
    _arr(ax, 0.88, 0.44, 0.64, 0.75, "#2a9d8f", rad=0.25, lab="controller K_TH\n(thalamocortical feedback)",
         lx=0.91, ly=0.70)
    ax.text(0.22, 0.86, "TMS / stimulation\n(cortical sites)", ha="center", fontsize=7.2, color="#d68910")
    _arr(ax, 0.30, 0.86, 0.36, 0.82, "#d68910")
    ax.text(0.80, 0.86, "EEG read-out\n(leadfield)", ha="center", fontsize=7.2, color="#d68910")
    _arr(ax, 0.64, 0.82, 0.72, 0.86, "#d68910")
    ax.text(0.04, 0.10, "dopamine δ\n(medication ON/OFF)", fontsize=7, color="#d68910", ha="left")
    _arr(ax, 0.10, 0.17, 0.12, 0.27, "#d68910", ls="--")
    ax.text(0.30, 0.02, "DBS / STN LFP", fontsize=7, color="#d68910", ha="center")
    _arr(ax, 0.34, 0.045, 0.37, 0.07, "#d68910", ls="--")
    ax.text(0.5, -0.06, "Two states: cortex x and subcortical estimates x̂ (with their own error dynamics and delays). "
                        "Orange = where the loop can be perturbed or observed.", ha="center", fontsize=6.8)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def fig(path, caption, width):
    from PIL import Image as PI
    w_, h_ = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h_ / w_), P(caption, CAP)])


def T(rows, widths, cap, font=7.2):
    return KeepTogether([table(rows, widths, font=font), P(cap, CAP)])


def TH(head, rows, widths, cap, font=7.2):
    """A table kept on the same page as its section heading."""
    return KeepTogether([head, table(rows, widths, font=font), P(cap, CAP)])


# ============================================================================ document
def content():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = [P("Measuring brain loops", TITLE),
         P("A closed-loop identification framework for stimulation-evoked EEG — control-theoretic foundations, four "
           "core problems, and applications including subcortical observer–controller modelling", _H2),
         P(f"syn_brain framework and plan · {DATE} · draft for discussion with the team and with Zheng Wang and "
           "colleagues · builds on reports/executive_summary.pdf, jr_parameters_explainer.pdf (section 5.6) and "
           "pd_connectivity_study_plan.pdf", CAP)]

    # -------------------------------------------------------------- summary
    s.append(P("Summary", H1))
    s.append(P(
        "<b>The problem.</b> Fitting neural-mass models to stimulation-evoked EEG and reading off their parameters does "
        "not work reliably: our synthetic studies show that most parameters are not identifiable, good fits hide wrong "
        "circuits, and fits do not generalize. The underlying reason is control-theoretic: a cortical column (and a "
        "cortico-subcortical loop) is a <b>feedback system</b>, and feedback makes outputs insensitive to internal "
        "parameters. <b>The proposal:</b> treat TMS-EEG as <b>closed-loop system identification</b> — measure each "
        "person's loop (loop gains, resonances, distance to instability) with a stimulation protocol designed to reveal "
        "it, report it with honest uncertainty, and use it to choose and dose interventions. Four core problems follow: "
        "<b>measurement</b> (which loop quantities are identifiable), <b>design</b> (which stimulation reveals them), "
        "<b>state</b> (how close each loop is to instability), and <b>estimation</b> (finding the answer when the data "
        "contain it). Subcortical observer–controller modelling (section 5) is "
        "one application of the framework, alongside cortical TMS-EEG, network scaling, Parkinson's disease, an "
        "EEG product, and neuromodulation.", BOX))

    # -------------------------------------------------------------- 1 problem
    s.append(P("1  The problem and why parameter fitting fails", H1))
    s += bullets([
        "<b>What the study found</b> (executive_summary.pdf): with neurofe's procedure only the stimulation gain and the "
        "excitatory time constant are reliably recovered from one node; fits with near-perfect training error predict "
        "the late response with 46–91% error even with a perfect model; loop parameters become measurable only with "
        "rich stimulation in an active state, and the excitatory loop vanishes when excitation dominates (its "
        "interneurons saturate). With a fixed procedure, a 10-node network recovers the single-node-identifiable "
        "quantities and conduction velocity (full 8-subject gate: a, k, c3·c4, mu all under 5%), but single fits often "
        "missed answers the Fisher bounds said were in the data. The coupling strength g failed for a different reason: "
        "a sample-and-hold in our simulator destabilised a high-gain feedback term (section 2.9).",
        "<b>The control reading:</b> negative feedback reduces the sensitivity of the output to internal parameters "
        "(sensitivity function S = 1/(1 + L)); exact symmetries hide how gains are split among parts; and the "
        "input-output map constrains only a few loop-level quantities (section 5.6 of the explainer). Fitting the parts "
        "of a feedback system from its output is the wrong question; measuring the loop is the right one.",
    ])
    lure_figure(FIG_DIR / "lure_loop.png")
    s.append(fig(FIG_DIR / "lure_loop.png", "Figure 1. A Jansen-Rit column as a control loop (a Lur'e system): linear "
                                            "synaptic dynamics in feedback with bounded sigmoid nonlinearities, "
                                            "stimulation as input and EEG as output. The data constrain the input-output "
                                            "map, not how it is divided among the parts.", 0.82 * W))

    # -------------------------------------------------------------- 2 foundations
    s.append(P("2  Control-theoretic foundations, and how each connects to JR network identification", H1))
    s.append(P(
        "The JR network identification problem, in control terms: N nodes, each a linear synaptic filter (excitatory "
        "kernel A·a/(s+a)², inhibitory kernel B·b/(s+b)²) in feedback with bounded sigmoids (slopes 0 ≤ S' ≤ 0.70) and "
        "neurofe's input bounds (500·tanh(·/500)); nodes coupled through g·W with conduction delays d_ij/mu; inputs are "
        "TMS pulses at chosen sites (gain k); outputs are 62 EEG channels through the leadfield. We want the parameters "
        "(or loop-level functions of them) from stimulation-evoked responses. Table 1 summarises the concepts; the cards "
        "below give, for each, the control idea, what it corresponds to in this problem, the evidence we already have, "
        "and what we can do with it."))
    s.append(T([["Concept", "Connection to JR network identification", "Status"],
                ["2.1 Loop gain; small-gain, circle, Popov", "Certify stability of teachers and of the fitted region",
                 "Evidence: unstable-ish teachers found only by long simulation"],
                ["2.2 Fold and Hopf as loop conditions", "Predict quiet/active switches and oscillation per spatial mode",
                 "Confirmed for one node; network untested"],
                ["2.3 Gain and phase margins", "Predict which subjects and regimes will be hard to fit",
                 "Consistent with p = 120, Euler artifact, teachers 3/6"],
                ["2.4 Sensitivity and the waterbed effect", "Where information lives; how to weight the loss",
                 "Explains sloppiness and late-window failure"],
                ["2.5 Structural identifiability", "Which parameters the site→channel transfer matrix determines",
                 "Fisher: all S1 &lt; 3% with 8 sites × 3 amplitudes"],
                ["2.6 Persistent excitation; input design", "Which sites, amplitudes, intervals; amplitude relative to "
                                                           "subject gain", "Amplitude calibration fixed 2 of 6 failures"],
                ["2.7 Closed-loop identification", "Separating local loops from network loops (the 'late = network' "
                                                   "question)", "Not yet tested"],
                ["2.8 Observers and Kalman filtering", "Single-trial likelihood; unknown background drive",
                 "Not yet used"],
                ["2.9 Passivity / dissipativity", "Stability for any network size; choice of coupling",
                 "A 1 ms hold destabilised the Laplacian self-term (6/8 subjects); coupling choice open"],
                ["2.10 Model reduction (Hankel)", "How many combinations a network can reveal",
                 "Not yet computed"],
                ["2.11 System-ID estimators (subspace, ERA, multiple shooting)", "Fix 'information present, optimiser "
                                                                                 "fails'", "Restarts work but cost "
                                                                                           "compute; to test"],
                ["2.12 Loop shaping", "Dosing interventions", "Later"]],
               [0.3, 0.42, 0.28], "Table 1. Control concepts at a glance; details in 2.1–2.12.", font=7.0))

    def card(title, idea, jr, evidence, todo):
        rows = [["Control idea", idea], ["In the JR network", jr], ["Evidence so far", evidence],
                ["What we can do", todo]]
        return KeepTogether([P(title, H2), table(rows, [0.18, 0.82], header=False, zebra=False, font=7.4)])

    s.append(card(
        "2.1  Loop gain; small-gain, circle and Popov criteria",
        "A loop made of a linear system G(s) and static nonlinearities whose slopes lie in a sector [0, κ] is stable "
        "if a frequency-domain condition on G holds: κ·‖G‖ &lt; 1 (small gain), or the less conservative circle and "
        "Popov criteria. These are global guarantees: no oscillation and no second resting state anywhere in the "
        "certified region.",
        "The network is exactly this structure: block-diagonal sigmoids (κ = 0.70) and input bounds, around a linear "
        "system made of every node's synaptic kernels plus the coupling g·W with delays. Parameter sets (teachers, or "
        "the student during fitting) can be certified without simulating them.",
        "Teachers drawn ±50% include unstable ones: subject 3 of the 8-subject gate oscillates at ~60 Hz even with "
        "neurofe's exact update and a 10× finer step, found only by long simulation; six others looked unstable only "
        "because of a discretisation artifact (section 2.9). The single-node study needed exact fixed-point analysis to separate real from apparent "
        "oscillations.",
        "Screen teacher draws with a circle/Popov test before using them; add a soft penalty that keeps the student in "
        "a certified region during fitting, so the optimiser cannot wander into oscillatory parameter sets. Both are "
        "cheap compared with simulation."))
    s.append(card(
        "2.2  Fold and Hopf as loop conditions",
        "A static feedback loop loses its resting state (fold / saddle-node) when its DC loop gain reaches 1, and starts "
        "oscillating (Hopf) when its frequency response reaches gain 1 at −180° (a Nyquist encirclement).",
        "For one node the conditions are scalar: fold at λ_E − λ_I = 1, oscillation at the Nyquist crossing of the "
        "node's loop. In a network the loop is a matrix: node gains times g·W with delays. Each spatial mode (eigenvector "
        "of the loop matrix) has its own loop gain, so the network destabilises along the mode that reaches 1 first, "
        "e.g. a hub-dominated mode.",
        "Single node: fold at p ≈ 115 with quiet and active states coexisting for p ≈ 37–115 (hysteresis, confirmed "
        "by simulation); one 'oscillation' at m = 1, p = 270 is created by the Euler step. Network: not yet analysed "
        "this way.",
        "Compute per-mode loop gains for each teacher: predicts which subjects sit near a switch or an oscillation "
        "before any fitting, and which spatial pattern a stimulation response will ring in. Validate against the "
        "single-node bifurcation map first (milestone M1)."))
    s.append(card(
        "2.3  Gain and phase margins",
        "How far a loop is from instability: how much the gain (gain margin) or phase (phase margin) can change before "
        "it starts oscillating; similarly the distance of the DC loop gain from 1 for a fold.",
        "Margins of each node's loop and of the network's modes, as functions of drive, E/I balance, coupling g and "
        "delays. Small margins mean strong ringing, slow decay, sensitivity to small parameter or integrator "
        "differences, and multiple coexisting states.",
        "Every fragile case so far had a small margin: p = 120 (just past the fold; fits multi-modal), m = 1 at p = 270 "
        "(damping −0.1 /s; the Euler step flips it to oscillation), teachers 3 and 6 (slow decay; hardest to fit).",
        "Use margins (i) to predict fit difficulty and set restarts or step size accordingly, (ii) to choose "
        "integration step (finer near small margins), (iii) as candidate clinical markers (distance to oscillation in "
        "epilepsy, beta-loop margin in PD), once shown identifiable."))
    s.append(card(
        "2.4  Sensitivity functions and the waterbed effect",
        "Negative feedback divides the effect of internal changes on the output by 1 + L (sensitivity S = 1/(1 + L)); "
        "Bode's integral says suppression at some frequencies must be paid for by amplification at others.",
        "Strong inhibitory loops (λ_I ≈ 4 in the balanced column) make the slow response insensitive to many parameters; "
        "the information reappears near the resonance (~6–11 Hz ringing) and in the late window. Network coupling adds "
        "loops, so parameters inside strongly coupled sub-networks are further hidden.",
        "Single node: sloppy loop parameters; their information sits in a late dip holding only 0.1–0.5% of the signal "
        "power, which a whole-window loss nearly ignores; temporal hold-out fails (46–91%). Network: whole-window "
        "metrics so far.",
        "Compute parameter sensitivities per frequency band for the network; weight the loss accordingly (pre-whitening "
        "or a late-window term) so the optimiser sees the informative band; report late-window prediction as a standard "
        "metric."))
    s.append(card(
        "2.5  Structural identifiability from the transfer function",
        "With perfect data, parameters are identifiable exactly when they are determined by the input-output map "
        "(transfer-function coefficients, or the Markov parameters of the impulse response).",
        "Linearised around rest, the network maps TMS at the stimulated sites to EEG channels through a transfer matrix "
        "H(s) = C·(sI − A(θ))⁻¹·B_sites, with delays entering as e^(−s·d_ij/mu). Conduction velocity is identifiable only "
        "if delays enter smoothly (fractional delays), which is why neurofe's integer delays give mu no gradient. "
        "Per-edge gains (tier S2) are identifiable only as far as H(s) depends on them.",
        "Single node: only A·k, a, b and the loop gains are visible. Network: Fisher bounds &lt; 3% for every tier-S1 "
        "parameter with 8 sites × 3 amplitudes; mu recovered (2.6%) once delays are fractional; teacher 6 is weaker "
        "with one site (c1 24%) but fine with 8.",
        "Rank test of the Jacobian of H(s) with respect to the parameters for tiers S1 and S2 and for 10/20/50 nodes: "
        "a quick count of identifiable combinations before training (complements Fisher, which adds noise)."))
    s.append(card(
        "2.6  Persistent excitation and optimal input design",
        "Parameters can only be separated if the input excites the system where their effects differ (in frequency, "
        "amplitude and entry point). For nonlinear systems the excitation level relative to the plant gain matters: too "
        "weak and the system looks linear and uninformative, too strong and it saturates.",
        "Design variables: stimulation sites (where the input enters the network), amplitudes (which nonlinear regime is "
        "probed), pulse intervals and trains (which frequencies), E/I targeting (which loop the input enters). The "
        "subject's gain k scales the effective excitation, so a fixed protocol excites subjects unequally.",
        "Single node: the rich protocol makes loss predict accuracy (ρ 0.83 vs 0.09 for one pulse). Network gate: one "
        "amplitude → 2/8 subjects converge; 3× more conditions barely helped; failures tracked low k (ρ −0.95); boosting "
        "amplitude by the true k fixed 2 subjects — but that calibration uses information real data do not have.",
        "D-optimal design (maximise det of the Fisher matrix) over sites, amplitudes and intervals; per-subject "
        "excitation normalisation from an observable response, e.g. titrating intensity to a target evoked amplitude, "
        "the analogue of clinical motor-threshold calibration."))
    s.append(card(
        "2.7  Closed-loop identification",
        "Inside a feedback loop, plant and controller cannot be separated from internal signals alone: an external "
        "excitation must enter the loop, and where it enters decides what can be separated.",
        "Each node is a closed loop (E and I interneurons), and nodes close further loops through g·W. The question "
        "'is late TEP activity local or network-driven' (the basis of virtual-lesion conclusions in the literature) is "
        "a closed-loop identification question: local and network loops can only be told apart with excitation entering "
        "at different points (several sites; E vs I targeting; for subcortex, DBS or medication).",
        "Single node: injecting into the inhibitory population pins b (0.4%). Network: multi-site stimulation recovers "
        "node and coupling parameters jointly in the pilot, but coupling g was the hardest in practice (27% median "
        "in the full gate, largely a simulator artifact, section 2.9) — local and network feedback trade off.",
        "Formulate the local-vs-network attribution test (synthetic teachers with known late-activity origin) as a "
        "closed-loop identification problem; choose excitation points that make the split identifiable; the same logic "
        "for observer vs controller in subcortical models (section 5)."))
    s.append(card(
        "2.8  Observers and Kalman filtering",
        "Latent states of a dynamical system can be estimated from noisy outputs by an observer (Kalman filter for "
        "linear-Gaussian systems; extended or unscented versions for nonlinear ones); the filter also gives the exact "
        "likelihood of the data (prediction-error form), and observability tells which states are recoverable.",
        "Hidden node states (P, E, I) behind 62 channels; trial-to-trial noise std_in; an unknown background drive "
        "(unknown-input estimation). Current fits compare one deterministic run with a trial average, so noise and "
        "trial variability are not modelled.",
        "Phase 1: std_in, b and c4 absorbed mismatch, converging to the same values whatever the truth. Network: all "
        "fits so far noise-free and trial-averaged.",
        "Kalman-based likelihood for noisy single trials (prediction-error method); unknown-input observers for "
        "background drive; observability of each source given the leadfield (which nodes the EEG can 'see')."))
    s.append(card(
        "2.9  Passivity and dissipativity",
        "Interconnections of passive systems through diffusive (difference-of-states) coupling remain stable for any "
        "number of nodes and any coupling graph; non-passive couplings can destabilise large networks.",
        "neurofe's Laplacian coupling g·Σ w_ij(E_j − E_i) is diffusive, but its self-term −g·d_i·E_i is a high-gain "
        "negative loop (DC gain ~20–40). Rate coupling g·Σ w_ij·S(E_j) is positive and bounded, has no self-term, and "
        "adds a degree-dependent mean drive. A node's passivity index depends on its operating point.",
        "Our differentiable simulator held the Laplacian self-term for 1 ms (neurofe updates it every 0.1 ms). That "
        "sample-and-hold destabilised the loop at 56–72 Hz for 6 of 8 gate subjects; the input clip then sized the "
        "resulting limit cycle, which looked like 'no bounded equilibrium at high g' and biased g low. With neurofe's "
        "update only 1 of 8 is unstable. A Laplacian model fitted to rate-coupled data gets E/I badly wrong "
        "(c1·c2 98%, c3·c4 54% error).",
        "Compute node passivity indices across operating points; derive size-independent stability conditions for 20 "
        "to 200 nodes and an admissible range of g for each coupling family, before scaling up. Treat the simulator's "
        "discretisation as part of the plant: check sampled-data stability (hold + Euler step) of every high-gain "
        "term, and test each simulator against neurofe's reference forward pass. For rate coupling, the positive "
        "interconnection admits delay-independent small-gain tests (paper/control_formulation.txt)."))
    s.append(card(
        "2.10  Model reduction: Hankel singular values",
        "The Hankel singular values of the map from inputs to outputs rank how strongly each internal mode is both "
        "excited and observed; only modes with large values can be identified from data.",
        "Inputs: the stimulation sites; outputs: 62 EEG channels. The number of significant Hankel values bounds the "
        "number of parameter combinations a given protocol can reveal; above ~60 nodes there are more sources than "
        "channels.",
        "Sloppy Fisher spectra at the single node; the network count is not yet computed.",
        "Compute Hankel values for 10/20/50/200 nodes and candidate site sets; choose sites that spread them; predict "
        "how identifiability scales with network size before training."))
    s.append(card(
        "2.11  System-identification estimators: subspace methods, impulse-response realisation, multiple shooting",
        "System identification avoids hard non-convex fits by (i) subspace methods and the Eigensystem Realization "
        "Algorithm (ERA), which estimate a linear state-space model directly from input-output or impulse-response data "
        "without iterative optimisation, (ii) multiple shooting, which fits short trajectory segments with continuity "
        "constraints and greatly reduces bad local minima, (iii) prediction-error refinement from a good start.",
        "A TMS pulse is close to an impulse, and responses around rest are close to linear, so ERA / subspace "
        "identification on the evoked responses gives the network's poles and residues directly. These map to loop "
        "gains, time constants, coupling and delays, giving a starting point near the right basin for the full JR fit.",
        "The recurring bottleneck at every scale is the optimiser, not the information: single node (p = 120, "
        "E-dominant), network gate (teachers 3 and 6 had good Fisher bounds but stuck fits). Restarts fix it (24× lower "
        "loss for teacher 6) but cost compute; a continuation attempt made fits bimodal.",
        "Compare ERA/subspace initialisation and multiple shooting against random restarts: convergence rate across "
        "subjects and compute cost. If they work, they replace most restarts (core problem P4)."))
    s.append(card(
        "2.12  Loop shaping",
        "Designing inputs or controllers to give a loop a desired frequency response and margins.",
        "Stimulation changes effective loop gains (via operating point and direct input); repetitive TMS, DBS and "
        "medication reshape cortical and subcortical loops.",
        "Not yet studied.",
        "Once loop quantities and margins are measurable (P1, P3), choose stimulation to move a loop away from "
        "instability: the step from measurement to closed-loop neuromodulation."))

    # -------------------------------------------------------------- 3 core problems
    s.append(P("3  Four core problems", H1))
    s.append(T([["Problem", "Question", "Control tools", "Success criterion (fixed now)"],
                ["P1 Measurement", "Which loop quantities are identifiable? Define a 'loop fingerprint': loop gains "
                                   "λ_E, λ_I, dominant poles (ringing frequency, damping), stability margins",
                 "Transfer-function identifiability; Fisher bounds and calibrated posterior on loop-level quantities",
                 "Bounds < 20% and 90% intervals covering 80–98% in all three E/I regimes, including the "
                 "excitation-dominant one where the c's fail"],
                ["P2 Design", "Which stimulation reveals the fingerprint (sites, intensities, pulse intervals, train "
                              "rates, E/I targeting, medication or DBS contrasts)?",
                 "Optimal input design; closed-loop identification conditions",
                 "A designed protocol matching or beating the hand-picked rich protocol with fewer conditions"],
                ["P3 State", "How close is each loop to instability (fold and oscillation margins)?",
                 "Small-gain / circle criteria; Nyquist; margins",
                 "Margins identifiable with calibrated intervals; predictive of fit fragility and response changes"],
                ["P4 Estimation", "When the data contain the answer, how do we find it reliably and cheaply?",
                 "Subspace / ERA initialisation, multiple shooting, per-subject excitation normalisation, posterior",
                 "≥ 7 of 8 network subjects converge without oracle calibration, at lower compute than 8–10 random "
                 "restarts"]],
               [0.12, 0.34, 0.26, 0.28], "Table 2. The four core problems.", font=7.0))
    s.append(P(
        "Why this matters: quantities defined on the input-output map are identifiable by construction, and they are "
        "the language of intervention: \"inhibitory loop gain\" for GABAergic function, \"oscillation margin\" for "
        "epilepsy, \"gain of the beta-generating loop\" for Parkinson's disease, all with uncertainty and with the "
        "protocol that makes them measurable. The study's message becomes: <b>measure the loop, not its parts</b>."))

    # -------------------------------------------------------------- 4 applications
    s.append(PageBreak())
    s.append(P("4  Applications within the framework", H1))
    s.append(T([["Application", "What the framework provides", "Status / next step"],
                ["A  Cortical TMS-EEG (single node)", "Loop fingerprint of a cortical column; analytic stability map; "
                                                      "designed protocol", "Linearisation done (explainer 5.6); "
                                                                           "milestone M1 below"],
                ["B  Network scaling (10–200 nodes)", "Passivity / small-gain guarantees; Hankel bound on identifiable "
                                                      "modes; multi-site input design",
                 "Full 8-subject gate: a, k, c3·c4, mu recovered; g failed because of a simulator hold artifact; "
                 "coupling comparison to be rerun (network_study_plan.txt N2.6)"],
                ["C  Subcortical loops (observer–controller)", "Closed-loop identification of thalamus / basal ganglia "
                                                              "routes; which perturbations separate observer from "
                                                              "controller", "Section 5 (with Zheng Wang et al.)"],
                ["D  Parkinson's disease", "Dopaminergic loop gain and beta-loop margins as identifiable markers; "
                                           "MRI priors on observation channels",
                 "pd_connectivity_study_plan.pdf (Parts 2–5)"],
                ["E  EEG product (sleep / meditation)", "State-space estimation from EEG time series; identifiable, "
                                                       "scale-free indices; test–retest", "Gate O4 in the PD plan; "
                                                                                          "section 5.4"],
                ["F  Neuromodulation", "Loop shaping: choose stimulation to move a loop's margins", "Later, once P3 "
                                                                                                   "is solved"]],
               [0.26, 0.44, 0.3], "Table 3. Applications.", font=7.1))

    # -------------------------------------------------------------- 5 subcortex
    s.append(P("5  Application C: subcortical observer–controller modelling", H1))
    s.append(P(
        "Zheng Wang and colleagues model thalamus and basal ganglia as an <b>observer–controller</b> of the cortex: "
        "subcortex receives a view of cortical activity through cortex→subcortex connections, estimates the cortical "
        "state (observer gain L) and feeds back corrections (controller gains K). It addresses exactly the loops "
        "Parkinson's disease disrupts and uses MRI connectivity naturally. In its current form, fitted to static resting "
        "fMRI connectivity, the indices are not yet measurements: static connectivity constrains only the combined "
        "closed loop (a closed-loop identification problem in exactly the sense of section 2), correlations hide "
        "absolute gains, and some fits are partly circular. Detailed assessment: pd_connectivity_study_plan.pdf, "
        "section 9."))
    s.append(P("5.1  The right view", H2))
    s += bullets([
        "<b>A hypothesis about function, not yet a measurement method.</b> \"Subcortex estimates and corrects cortex\" "
        "predicts specific signatures (delayed correction, error dynamics, changes under perturbation); it becomes a "
        "measurement once its indices are recoverable and track known manipulations.",
        "<b>A good fit is not evidence for the gains:</b> any split between cortical and subcortical feedback that yields "
        "the same closed loop fits equally well; in the manuscripts the cortex-only model fits slightly better.",
        "<b>Keep the structure and interpretation:</b> anatomical observation channels, route-specific quantities, and a "
        "language that connects to interventions.",
    ])
    s.append(P("5.2  Five principles", H2))
    s += bullets([
        "<b>Fit dynamics and perturbations, not static connectivity.</b> Observer and controller differ in time (the "
        "observer acts on a delayed estimate with its own error dynamics); this shows in lagged covariances, spectra and "
        "perturbation responses, at the tens-of-milliseconds scale of EEG, MEG, TMS-EEG and intracranial recordings.",
        "<b>Model the full two-state loop</b> (cortex and subcortex's estimate of it), with basal ganglia output "
        "inhibitory and routed through the thalamus, thalamus and basal ganglia as separate routes, and directed "
        "anatomical priors.",
        "<b>Estimate a few meaningful parameters:</b> either a few cost and noise weights per route (inverse optimal "
        "control) or, recommended for PD, the physiology of a linearised Jansen-Rit + basal ganglia loop, so that L and K "
        "are functions of dopaminergic gain δ.",
        "<b>Validate without circularity:</b> held-out sessions or subjects, subject-swap tests, comparison against an "
        "equal-size generic loop model, indices defined from identifiable (scale-free) combinations, and the validation "
        "ladder (synthetic recovery → Fisher / profile → calibrated posterior → test–retest → known manipulation → added "
        "value).",
        "<b>Use data in which the loop is perturbed</b> (Table 4).",
    ])
    loop_figure(FIG_DIR / "observer_loop.png")
    s.append(fig(FIG_DIR / "observer_loop.png", "Figure 2. Proposed formulation: a cortical plant observed by two "
                                                "routed subcortical estimators, with signed basal ganglia output through "
                                                "the thalamus. Orange marks where the loop can be perturbed or observed.",
                 0.78 * W))
    s.append(T([["Data", "Perturbs / observes", "What it can identify (expected)"],
                ["Resting fMRI connectivity (static)", "nothing; zero-lag correlations", "the combined closed loop only"],
                ["Resting EEG/MEG (time-resolved)", "spontaneous dynamics at loop timescales",
                 "loop time constants; partly the observer/controller split"],
                ["Multi-site TMS-EEG", "cortex, at chosen sites and intensities",
                 "cortical parameters and coupling (10-node pilot); routes as far as they shape the response"],
                ["Medication ON/OFF (PD)", "dopamine: basal ganglia route gain, in a known direction",
                 "the basal ganglia route weight, by contrast"],
                ["DBS ON/OFF, STN local field potentials", "the subcortical loop directly", "subcortical estimates and "
                                                                                           "gains most directly"],
                ["Combinations", "the loop from both sides", "the most complete identification"]],
               [0.28, 0.32, 0.4], "Table 4. Data types and what they can identify (to be confirmed by stage S1).",
               font=7.0))
    s.append(P("5.3  Staged joint path (about 4–6 weeks)", H2))
    s.append(T([["Stage", "Weeks", "Work", "Gate (fixed before running)"],
                ["S1 Identifiability first", "1–2", "Synthetic observer–controller subjects with known gains; static "
                                                   "connectivity vs lagged covariance vs stimulation responses, ± a "
                                                   "dopamine-like change; Fisher bounds and Laplace posterior",
                 "Report which indices are recoverable from which data; static connectivity expected to give only the "
                 "closed loop (negative control)"],
                ["S2 Re-formulation", "2–4", "Two-state loop with routed, signed pathways; normative or mechanistic "
                                            "parameterisation", "Split and basal ganglia route weight within 25% in ≥ 8 "
                                                                "of 10 synthetic subjects; 90% intervals cover 80–98%"],
                ["S3 Link to dopamine", "3–4", "Observer indices as functions of δ, STN–GPe coupling, atrophy and "
                                               "cortical change", "Indices track δ monotonically, not mimicked by "
                                                                  "atrophy or cortical change"],
                ["S4 First real test", "4–6", "Time-resolved or perturbational data, held-out validation, equal-size "
                                              "alternative, test–retest", "Better held-out prediction than the "
                                                                          "alternative; ICC ≥ 0.7"]],
               [0.16, 0.07, 0.45, 0.32], "Table 5. Staged path for application C.", font=6.9))
    s.append(P("5.4  Implications for the observer-based EEG product", H2))
    s += bullets([
        "Estimate from the EEG time series as a state-space model, not from summary covariances.",
        "Show that K_norm and L_norm are recoverable in synthetic data and stable across nights before calling them "
        "biomarkers; define them from identifiable, scale-free loop quantities.",
        "Benchmark against plain spectral features: an AUC of 0.965 shows that the features differ between states, not "
        "that they measure \"regulation effort\" or \"observer efficiency\".",
    ])

    # -------------------------------------------------------------- 6 roadmap
    s.append(TH(P("6  Roadmap", H1), [["Milestone", "When", "Work", "Gate"],
                ["M1 Loop fingerprint (single node)", "weeks 1–2", "Analytic loop tool (transfer function, Bode / "
                                                                   "Nyquist, DC loop gain, margins); check fold at "
                                                                   "λ_E − λ_I = 1 and oscillation at the Nyquist "
                                                                   "crossing against the bifurcation map; Fisher bounds "
                                                                   "and posterior calibration for the fingerprint in "
                                                                   "three regimes", "P1 criterion (Table 2)"],
                ["M2 Designed protocol", "weeks 2–3", "Frequency-domain sensitivities; information-optimal pulse "
                                                      "intervals, trains and E/I targeting", "P2 criterion"],
                ["M3 Subcortical stage S1", "weeks 1–2 (parallel, with Zheng)", "Section 5.3 S1",
                 "Table 5"],
                ["M3b Estimation (network)", "weeks 1–2 (parallel)", "ERA / subspace initialisation and multiple "
                                                                    "shooting vs random restarts on the 8-subject "
                                                                    "10-node gate; observable amplitude normalisation; "
                                                                    "circle/Popov screening of teachers (2.1, 2.6, "
                                                                    "2.11)", "P4 criterion"],
                ["M4 Network", "after the 10-node gate", "Fingerprint and per-mode margins at 10 nodes; passivity / "
                                                         "small-gain guarantees; Hankel bound; then 20 nodes",
                 "P1 at network scale"],
                ["M5 Real data", "after M1–M2", "Fingerprint on real TMS-EEG with held-out tests; drug-response and "
                                                "test–retest where available", "Calibrated, reproducible, beats plain "
                                                                               "TEP features"]],
               [0.2, 0.15, 0.43, 0.22], "Table 6. Roadmap. Gates are set now and will not be changed after results.",
               font=6.9))
    s += bullets([
        "<b>If M1 passes</b> — the fingerprint is identifiable where the c's are not, especially in excitation-dominant "
        "states — it becomes the main result of the paper (paper/draft_v0.txt) and the core of a product.",
        "<b>If it fails</b>, we learn that some loop properties are fundamentally hidden from EEG: also publishable, "
        "and it tells the PD and product work what not to claim.",
    ])

    # -------------------------------------------------------------- 7 contribution + questions
    s.append(P("7  What we bring, and questions for Zheng and colleagues", H1))
    s += bullets([
        "<b>Available now:</b> teacher–student harness with ground truth; Fisher / Cramér–Rao bounds, profile "
        "likelihood, calibrated Laplace posterior (24 subjects); differentiable JR network with fractional delays "
        "(10-node pilot); equation-level bifurcation analysis and the regime linearisation; the PD loop model and disease "
        "teachers (pd_connectivity_study_plan.pdf).",
        "<b>Questions:</b> construction of Ac and A* and the number of free parameters; whether the cortical gain and the "
        "observer are re-solved in the null without subcortical feedback, and how it relates to the best-fitting "
        "cortex-only model; the derivation of the closed-loop operator; the normalisation of K_norm / L_norm and the "
        "meditation analysis behind the AUC; availability of repeated or perturbational recordings; interest in running "
        "stage S1 and milestone M1 jointly.",
    ])
    s.append(P("Basis: syn_brain study results (reports/), and review of the LCF_HCP (three versions), CRRP and CRRP-RL "
               "manuscripts and the sleep digital-twin deck (unpublished drafts, not for distribution).", CAP))
    return s


def main():
    build(content(), OUT, "Measuring brain loops: a closed-loop identification framework",
          f"syn_brain · closed-loop identification framework · {DATE}")


if __name__ == "__main__":
    main()
