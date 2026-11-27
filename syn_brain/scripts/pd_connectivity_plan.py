"""Build the study plan "Connections between JR nodes, toward early Parkinson's disease detection" (PDF).

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.pd_connectivity_plan
Output: syn_brain/reports/pd_connectivity_study_plan.pdf
"""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate, Spacer

from syn_brain.core.paths import RESULTS_DIR, SYN_ROOT
from reportlab.lib.styles import ParagraphStyle

from syn_brain.core.report_style import BOX, CAP, SMALL, TITLE, W, P, bullets, table
from syn_brain.core.report_style import H1 as _H1, H2 as _H2

H1 = ParagraphStyle("h1k", parent=_H1, keepWithNext=1)     # never leave a heading at the bottom of a page
H2 = ParagraphStyle("h2k", parent=_H2, keepWithNext=1)

PLAN = SYN_ROOT / "reports" / "pd_connectivity_study_plan.pdf"
FIG_DIR = RESULTS_DIR / "plans"


# ----------------------------------------------------------------------------- schematic
def loop_schematic(path):
    """Textbook cortico-basal ganglia-thalamic loop with signed connections."""
    pos = {"Cortex (JR)": (0.5, 0.92), "Thalamus": (0.88, 0.55), "Striatum D1": (0.18, 0.62),
           "Striatum D2": (0.42, 0.62), "GPe": (0.42, 0.36), "STN": (0.66, 0.30), "GPi / SNr": (0.30, 0.10),
           "SNc\n(dopamine)": (0.04, 0.35)}
    col = {"Cortex (JR)": "#cfe3f5", "Thalamus": "#d9f0d3", "SNc\n(dopamine)": "#f6d6a8"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(-0.08, 1.02); ax.set_ylim(0.0, 1.02); ax.axis("off")
    for n, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 0.075, y - 0.04), 0.15, 0.08, boxstyle="round,pad=0.01",
                                    fc=col.get(n, "#eeeeee"), ec="#333333", lw=0.8))
        ax.text(x, y, n, ha="center", va="center", fontsize=7.5)

    def arrow(a, b, sign, rad=0.0, style="-"):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        c = {"+": "#1f6fb4", "-": "#c0392b", "m": "#d68910"}[sign]
        head = "-|>" if sign != "-" else "-["
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle=head,
                                     mutation_scale=9, color=c, lw=1.2, ls=style, shrinkA=16, shrinkB=16))

    arrow("Cortex (JR)", "Striatum D1", "+", 0.1)
    arrow("Cortex (JR)", "Striatum D2", "+", 0.05)
    arrow("Cortex (JR)", "STN", "+", -0.15)               # hyperdirect
    arrow("Cortex (JR)", "Thalamus", "+", -0.15)
    arrow("Thalamus", "Cortex (JR)", "+", -0.15)
    arrow("Striatum D1", "GPi / SNr", "-", 0.1)           # direct pathway
    arrow("Striatum D2", "GPe", "-", 0.0)                 # indirect pathway
    arrow("GPe", "STN", "-", 0.3)
    arrow("STN", "GPe", "+", 0.3)
    arrow("STN", "GPi / SNr", "+", 0.1)
    arrow("GPi / SNr", "Thalamus", "-", 0.25)
    arrow("SNc\n(dopamine)", "Striatum D1", "m", -0.1, "--")
    arrow("SNc\n(dopamine)", "Striatum D2", "m", 0.25, "--")
    ax.text(-0.07, 0.02, "blue →  excitatory (glutamate)     red ⊣  inhibitory (GABA)     orange dashed  dopaminergic "
            "modulation (D1 ↑ gain, D2 ↓ gain)", fontsize=6.8)
    ax.text(0.615, 0.52, "hyperdirect", fontsize=6.5, color="#1f6fb4", rotation=-62)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def img(path, caption, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP)])


# ----------------------------------------------------------------------------- content
def content():
    s = []
    s += [P("Connections between Jansen-Rit nodes", TITLE),
          P("A study plan toward early Parkinson's disease detection with model-based TMS-EEG", H2),
          P("syn_brain study plan · proposed 22 November 2026 · revised 24 November 2026 after the single-node "
            "posterior and the 10-node network pilot; coupling note updated 27 November · builds on reports/executive_summary.pdf, "
            "docs/network_study_plan.txt and docs/background_drive_note.txt", CAP)]

    s.append(P("Summary", H1))
    s.append(P(
        "In a discussion with a collaborator working on MRI and early detection of Parkinson's disease (PD), the question "
        "came up of how Jansen-Rit (JR) nodes are connected in our network model. The answer, that connections are "
        "approximately linear, is correct, and the single-node study shows it is a substantive limitation. neurofe's "
        "coupling is linear, diffusive, symmetric and excitatory only, provides no mean drive, and the anatomy contains "
        "no subcortical nodes. PD is primarily a disorder of loops (basal ganglia–thalamus–cortex) and of "
        "dopamine-dependent gain on them, so the present model cannot represent it. <b>Revision (24 Nov):</b> one "
        "premise has changed. A single quiet node hides its loops, but a 10-node network with neurofe's own coupling "
        "turned out to identify its global parameters, loop strengths included, from multi-site, multi-intensity "
        "stimulation once the fitting procedure is fixed (pilot). So mean drive is not needed for identifiability; "
        "the case for new connections now rests on representing PD (sign, direction, subcortical loops). This plan "
        "(1) completes the coupling ladder with the PD-relevant rungs, (2) adds a minimal "
        "cortico-basal ganglia-thalamic loop with dopamine-dependent gain, (3) measures how small a dopamine-like change "
        "TMS-EEG can detect, (4) quantifies how much MRI-derived priors improve identification, and (5) tests, and "
        "builds on, the observer–controller formulation of cortico-subcortical loops developed by Zheng Wang and "
        "colleagues (sections 9–10, added 24 Nov).", BOX))

    # ------------------------------------------------------------- 1
    s.append(P("1  Why the connection question matters", H1))
    s += bullets([
        "<b>Connections are where MRI enters the model.</b> Diffusion MRI supplies the structural connectome (weights) "
        "and tract lengths (delays). What the model does with those numbers decides whether MRI can inform it.",
        "<b>PD lives in loops and connections.</b> Loss of nigral dopamine changes the effective gain of striatal "
        "pathways; the classic electrophysiological signature is exaggerated beta-band (13–30 Hz) synchrony emerging "
        "from the loop. Several key projections are inhibitory (striatum → GPe/GPi, GPi → thalamus). A linear, "
        "excitatory-only coupling cannot generate or represent this.",
        "<b>The present anatomy is cortex only.</b> The Schaefer 200-parcel atlas has no substantia nigra, striatum, "
        "subthalamic nucleus, pallidum or thalamus, the structures where PD begins and where MRI early-detection "
        "markers are measured.",
        "<b>Early detection means small effects.</b> The relevant question is not whether a large lesion changes the "
        "EEG, but whether a 10–20% change is detectable and correctly attributed. That is an identifiability question, "
        "which the synthetic teacher/student framework answers directly.",
    ])

    # ------------------------------------------------------------- 2
    s.append(P("2  The current coupling and what PD would need", H1))
    s.append(P(
        "neurofe couples nodes through the excitatory population with a delayed Laplacian: input<sub>i</sub> = "
        "g·Σ<sub>j</sub> w<sub>ij</sub>·(E<sub>j</sub>(t − d<sub>ij</sub>) − E<sub>i</sub>(t)), with w the "
        "log-normalised, symmetric diffusion-MRI connectome, d = tract length / conduction velocity, and zero-mean "
        "noise as the only other input. <b>Update 27 November:</b> the network study now supports the change argued "
        "for here: a Laplacian model fitted to data generated with rate coupling recovers E/I badly (c1·c2 98%, "
        "c3·c4 54% median error), so the coupling form must be chosen on physiological grounds. The apparent "
        "high-coupling instability of the Laplacian was mostly a discretisation artifact in our simulator "
        "(executive_summary.pdf); the coupling comparison is being rerun."))
    rows = [["Aspect", "neurofe now", "Needed for PD-type mechanisms"],
            ["What is transmitted", "a membrane-potential-like state E (can be negative)",
             "firing rate S(E) (positive), so connected nodes drive each other"],
            ["Form", "linear, diffusive (difference of states)", "nonlinear through the sigmoid; non-diffusive"],
            ["Sign", "excitatory only", "signed: inhibitory projections (striatum, pallidum, GPi → thalamus)"],
            ["Direction", "symmetric (DTI is undirected)", "directed for known pathways (anatomical priors)"],
            ["Mean drive", "none: zero when all nodes are at rest", "tonic input p, or positive rate coupling"],
            ["Nodes", "cortical parcels only", "plus striatum (D1/D2), GPe, STN, GPi/SNr, thalamus"],
            ["Neuromodulation", "none", "dopamine as a gain on striatal populations"],
            ["What sets weights", "anatomy (streamline counts) × fitted per-edge gain",
             "anatomy for existence and delay; effective gain fitted, with MRI-informed priors"]]
    s += [table(rows, [0.2, 0.37, 0.43]), P("Table 1. Connection properties, current model against what is needed.", CAP)]

    # ------------------------------------------------------------- 3
    s.append(P("3  What the study has shown so far", H1))
    rows = [["Finding", "Consequence for connections and PD"],
            ["Single node: with no drive a node rests at E − I ≈ −1.9 mV, far below threshold (6 mV); it behaves like "
             "a linear filter and its loop gains are invisible to a single pulse (bounds 56–252%)",
             "An isolated, quiet column cannot reveal its circuit"],
            ["Single node: an active state (p ≈ 150) plus rich stimulation, ≥ 15 restarts and a Laplace posterior "
             "recover loop strengths with calibrated uncertainty (4–8% error, 96% coverage, 24 subjects)",
             "A validated estimation recipe now exists for the detectability and attribution tests (Parts 3–4)"],
            ["<b>10-node network (pilot):</b> with neurofe's own coupling and 8 sites × 3 intensities, all global "
             "parameters have bounds &lt; 3%, and backprop recovers a, b, g, c3, c4, mu, k within 4%",
             "<b>Mean drive is not needed for identifiability</b>; multi-site stimulation of a network constrains the "
             "shared parameters. Connection changes are visible."],
            ["10-node network: coupling with mean drive (firing rates) saturates neurofe's input bound at nominal g; "
             "best bounds at g ≈ 100–120, but the fit got stuck from a single start",
             "Rate-based coupling needs rescaling of g and restarts; a secondary option, not a prerequisite"],
            ["Excitation-dominant columns saturate their excitatory interneurons; quiet and active states coexist "
             "near the transition (hysteresis)",
             "Disease states that push circuits toward excitation or toward the transition are the hardest to "
             "measure; protocols must be designed for them"],
            ["A, B, vmax are exactly degenerate with the c's and k",
             "EEG alone cannot separate synaptic strength from synapse number; external measurements are needed to "
             "break such ties"],
            ["Stimulation gain k is stiff but confounded with A; atrophy changes k",
             "Structural MRI with E-field modelling can fix k, a direct role for MRI"],
            ["Paired-pulse ratios are exactly 1 at the nominal state",
             "Clinical paired-pulse paradigms (intracortical inhibition) cannot be reproduced until the state is "
             "active"],
            ["Fitting the early window predicts the late window with 91% error, even with a perfect model",
             "Late TEP components, which carry loop information, need dedicated protocols and loss weighting"]]
    s += [table(rows, [0.5, 0.5]), P("Table 2. Results relevant to the connection question (details in "
                                     "reports/executive_summary.pdf and docs/network_study_plan.txt). Network results "
                                     "are a pilot: 1–2 synthetic subjects, matched noise-free teacher.", CAP)]
    s.append(PageBreak())

    # ------------------------------------------------------------- 4
    s.append(P("4  Questions", H1))
    rows = [["#", "Question", "If yes", "If no"],
            ["Q1", "Does the connection form decide whether the network reaches an active state, and whether changes in "
                   "connection strength become visible in TMS-EEG?",
             "Switch neurofe's coupling before any disease modelling", "Connection form is not the bottleneck; look "
                                                                       "at local parameters and stimulation"],
            ["Q2", "Can the gains of a cortico-basal ganglia-thalamic loop be identified from cortical TMS-EEG?",
             "A model-based PD marker from TMS-EEG is feasible", "TMS-EEG must be combined with other data (MRI, "
                                                                 "resting-state spectra)"],
            ["Q3", "What is the smallest dopamine-like gain change TMS-EEG can detect and attribute correctly?",
             "Defines the stage at which detection is possible", "Early detection needs a different modality or "
                                                                  "protocol"],
            ["Q4", "Which MRI-derived measures, used as priors, most improve identification of the disease parameters?",
             "Concrete joint MRI + TMS-EEG protocol", "MRI and TMS-EEG inform different parameters; keep them "
                                                     "separate"],
            ["Q5", "Can the observer–controller indices (observer vs controller gains, basal ganglia route weight, "
                   "K_norm / L_norm) be recovered, and do they track dopaminergic loop gain?",
             "A compact, interpretable PD read-out that also takes MRI priors", "They are summaries of the fit, not "
                                                                                "measurements; use the JR loop "
                                                                                "parameters instead"]]
    s += [table(rows, [0.05, 0.43, 0.26, 0.26]), P("Table 3. Questions and what each answer implies.", CAP)]

    # ------------------------------------------------------------- 5
    s.append(P("5  Part 1: coupling-form ladder (Q1)", H1))
    s.append(P(
        "Each rung changes one property of the connections in the synthetic teacher; the student has the same form "
        "(rung R0 of the main study). Networks: 10 cortical nodes (fast), then the 200-node anatomy."))
    rows = [["Rung", "Coupling into node i", "Tests"],
            ["L0", "g·Σ w<sub>ij</sub>(E<sub>j</sub>(t−d<sub>ij</sub>) − E<sub>i</sub>)  (neurofe now)", "baseline"],
            ["L1", "L0 + tonic drive p (global, then per node)", "mean drive alone"],
            ["L2", "g·Σ w<sub>ij</sub>·S(E<sub>j</sub>(t−d<sub>ij</sub>))  (rate coupling)", "positive, nonlinear transmission"],
            ["L3", "L2 with signed, directed weights for known inhibitory or one-way pathways", "sign and direction"],
            ["L4", "L3 + subcortical nodes (Part 2)", "loops through basal ganglia and thalamus"]]
    s += [table(rows, [0.08, 0.62, 0.3]), P("Table 4. Coupling ladder.", CAP)]
    s.append(P("Measured at every rung:", SMALL))
    s += bullets([
        "Operating point of every node (resting E − I, sigmoid slope), and whether the network oscillates spontaneously.",
        "TEP realism against the real-data features already measured (README, Finding 1): GMFA peak and latency, "
        "late/early GMFA ratio, baseline variability.",
        "Cramér–Rao bounds and profile likelihoods for g, per-edge gains and the loop gains; backprop recovery with "
        "log-scaled parameters.",
        "Generalisation: temporal hold-out (fit early, predict late) and held-out stimulation site.",
    ])
    s.append(P("<b>Decision.</b> The lowest rung that gives an active operating point, realistic late TEP features, and "
               "bounds below 20% on the coupling parameters becomes the default teacher, and the proposed change to "
               "neurofe's coupling."))
    s.append(P("Status, 24 November (criteria unchanged; reported against them)", H2))
    rows = [["Rung", "Status", "Against the Q1 criteria"],
            ["L0", "Measured at 10 nodes (Fisher + backprop pilot)", "Coupling bounds &lt; 20%: <b>pass</b>. Active "
                                                                     "operating point: <b>no</b> (nodes quiet). Late/early "
                                                                     "ratio: not re-measured (0.15–0.19 earlier, real "
                                                                     "0.84): <b>likely fail</b>"],
            ["L1", "Single node only (background-drive sweep)", "Active state reachable (p ≳ 115); network not tested"],
            ["L2", "Implemented (coupling option); Fisher sweep over g; pilot", "Bounds &lt; 2% at g ≈ 100–120; fit "
                                                                                "stuck from one start; late/early "
                                                                                "ratio not measured"],
            ["L3", "Not started", "The first PD-relevant rung"],
            ["L4", "Not started (Part 2)", "—"]]
    s += [table(rows, [0.08, 0.4, 0.52]), P("Table 4b. No rung passes all of Q1 yet. Identifiability no longer "
                                            "requires a new coupling; teacher realism (late activity) and PD "
                                            "representation still do.", CAP)]

    # ------------------------------------------------------------- 6
    s.append(PageBreak())
    s.append(P("6  Part 2: a minimal cortico-basal ganglia-thalamic loop (Q2)", H1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sp = FIG_DIR / "pd_loop_schematic.png"
    loop_schematic(sp)
    s.append(img(sp, "Figure 1. Textbook cortico-basal ganglia-thalamic loop used as the teacher. Each box is one "
                     "neural-mass population; the cortex is a JR node (or the JR network). Dopamine from SNc "
                     "modulates the gain of the D1 (direct) and D2 (indirect) striatal populations.", width=0.8 * W))
    s += bullets([
        "<b>Model.</b> Cortex: the JR node from core/node.py (or the 10-node network). Subcortical nuclei: single "
        "populations with second-order synaptic kernels and sigmoids, as in published mean-field models of this loop "
        "(e.g. van Albada and Robinson 2009; Moran et al. 2011; to be confirmed in a short literature review). "
        "Thalamus feeds back to the cortical excitatory input.",
        "<b>Dopamine.</b> A single parameter δ scales striatal gain: D1 gain × (1 + δ), D2 gain × (1 − δ). Healthy "
        "δ = 0; dopamine loss is δ &lt; 0.",
        "<b>Sanity checks before any fitting.</b> The healthy teacher has no excessive beta; reducing δ increases "
        "beta power and synchrony in the loop and in the cortical EEG; cortical TEPs stay within the real-data ranges.",
        "<b>Observation.</b> Only cortical EEG is observed (as in TMS-EEG), with TMS at M1 and at least one non-motor "
        "site; optionally a pre-stimulus resting spectrum.",
        "<b>Identification.</b> Fisher bounds and profile likelihoods for δ, the loop gains, and the cortical "
        "parameters; backprop and Levenberg–Marquardt fits from random starts.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------- 7
    s.append(P("7  Part 3: detectability of early, PD-like changes (Q3)", H1))
    rows = [["ID", "Disease teacher (change from healthy)", "Represents"],
            ["D1", "δ = −0.05, −0.10, −0.20, −0.40", "progressive dopamine loss"],
            ["D2", "STN ↔ GPe coupling × 1.2", "loop change downstream of dopamine loss"],
            ["D3", "cortical stimulation gain k × 0.85", "cortical atrophy (confound control)"],
            ["D4", "D1 (δ = −0.10) + D3", "early PD with atrophy"],
            ["D5", "cortical inhibitory loop (c3·c4) × 0.8", "a cortical, non-dopaminergic change (specificity control)"]]
    s += [table(rows, [0.08, 0.47, 0.45]), P("Table 5. Disease teachers.", CAP)]
    s.append(P("For each disease teacher, stimulation protocol (single pulse, paired pulses at clinical intervals, a short "
               "train) and brain state (rest, active):"))
    s += bullets([
        "<b>Detectability:</b> Fisher distance between healthy and diseased EEG, d' = (Δθ<super>T</super> F Δθ)"
        "<super>1/2</super>, at the real noise level, and the number of trials needed for d' = 2.",
        "<b>Attribution:</b> fitted separately to healthy and diseased teachers, does the student change δ, or does it "
        "explain the difference with k or cortical parameters? (D3 and D5 are the controls.)",
        "<b>Minimum detectable effect:</b> the smallest |δ| detected and correctly attributed in ≥ 8 of 10 fits.",
    ])

    # ------------------------------------------------------------- 8
    s.append(P("8  Part 4: MRI-derived priors (Q4)", H1))
    s.append(P(
        "MRI cannot measure synaptic gains, but it can constrain the parameters that make them unidentifiable. A prior "
        "enters the identifiability analysis simply as added precision, F<sub>post</sub> = F<sub>EEG</sub> + "
        "Σ<sub>prior</sub><super>−1</super>, so the value of each MRI measure can be computed before acquiring it."))
    rows = [["MRI measure", "Model parameter it could constrain", "Status"],
            ["Diffusion MRI connectome, tract lengths", "existence and anatomical weight of connections; delays",
             "used now"],
            ["Structural MRI + E-field modelling (scalp-to-cortex distance, coil geometry)", "stimulation gain k",
             "standard tools; to add"],
            ["Cortical thickness, atrophy", "priors on k and on local cortical gain", "hypothesis"],
            ["Nigral imaging (neuromelanin-sensitive, iron-sensitive, diffusion free-water)",
             "prior on dopaminergic gain δ", "hypothesis: mapping to δ needs calibration"],
            ["Resting-state fMRI functional connectivity", "coupling g, effective connectivity of the loop",
             "hypothesis"]]
    s += [table(rows, [0.38, 0.38, 0.24]), P("Table 6. Candidate MRI constraints. 'Hypothesis' rows are to be "
                                             "reviewed with the MRI collaborator.", CAP)]
    s += bullets([
        "<b>Experiment.</b> Repeat Part 3 with each prior added at a range of precisions; report how much each shrinks "
        "the bound on δ and how it changes the minimum detectable effect and attribution.",
        "<b>Output.</b> A ranked list of MRI measures by information gain for the PD parameters, i.e. which scans are "
        "worth adding to a TMS-EEG protocol.",
    ])

    # ------------------------------------------------------------- 9
    # ------------------------------------------------------------- 9 (new)
    s.append(PageBreak())
    s.append(P("9  The observer–controller work (Zheng Wang et al.): what it offers, what it needs", H1))
    s.append(P(
        "Zheng Wang and colleagues model the cortex as a 'plant' and thalamus plus basal ganglia (and in one version the "
        "cerebellum) as an <b>observer–controller</b>: subcortex receives a view of cortical activity through "
        "cortex→subcortex connections, estimates the cortical state (a Kalman-type observer gain L), and feeds back "
        "corrections (controller gains K). This is an attractive formalism for this plan: it is about exactly the "
        "loops PD disrupts, it uses MRI connectivity to define the observation channels, and it proposes "
        "interpretable summary indices. Five manuscripts and a product deck were reviewed (unpublished drafts in "
        "literature/, not for distribution)."))
    rows = [["Document", "What it does", "Data and fit", "Key results"],
            ["LCF (3 versions; latest LCF_HCP_1p_final)",
             "Linear cortical plant; thalamus + BG as one observer (Kalman gain L, Ks = L·Bs·R⁻¹) and a cortical "
             "controller Kc (LQR); model FC from the closed-loop Lyapunov equation",
             "HCP, 100 subjects, Schaefer-200 + 31 subcortical; static resting fMRI FC; gains from Riccati/LQR with "
             "fixed cost weights (grid-searched)",
             "FC fit r ≈ 0.82; time-split test r 0.76; Bs = 0 null 0.80 → 0.21; Kc ≈ inverse SC degree "
             "(r ≈ −0.96)"],
            ["CRRP", "Three nested observer routes (thalamus ⊂ BG ⊂ cerebellum); a per-parcel thalamus/BG mixing weight "
                     "g learned online against the cerebellar estimate",
             "HCP, ~200 subjects; target drift A* from each subject's FC (inverse Lyapunov)",
             "FC fit r 0.76; route energies THA < BG < CB; Allocation 0.379 vs 0.370 shuffled"],
            ["CRRP-RL", "Route weights (simplex) in a reinforcement-learning agent; BG as value/prediction-error route",
             "Simulation only (100 seeds)", "Faster early learning; slower reversal; BG weight grows with training"],
            ["Sleep digital-twin deck", "Observer-based EEG engine; biomarkers K_norm (regulation effort), L_norm "
                                        "(observer efficiency)", "Meditation EEG (details not given)",
             "'AUC up to 0.965' distinguishing regulated states"]]
    s += [table(rows, [0.17, 0.31, 0.27, 0.25], font=7.0),
          P("Table 6b. The observer–controller documents (as reported by the authors).", CAP)]
    s.append(P("Assessment through this study's identifiability lens", H2))
    s += bullets([
        "<b>What it offers.</b> A principled way to describe cortico-subcortical loops as estimation and control; MRI "
        "connectivity defines the observation channels (a natural Part 4 prior); route-specific (thalamic, basal "
        "ganglia, cerebellar) quantities that map onto PD hypotheses; reproducible SC-derived gain maps (split-half "
        "r 0.69 for Kc).",
        "<b>The observer-vs-controller split is not identifiable from static FC.</b> Model FC depends only on the "
        "closed-loop operator Acl = Ac + Kc − Ks·Bsᵀ, so any split of the same Acl between cortical and subcortical "
        "feedback gives the same FC; the division of labour is set by the chosen cost weights. Consistent with this, "
        "the cortex-only model fits slightly better than the full model.",
        "<b>Correlation-based FC cannot fix absolute gain magnitudes</b> (it is invariant to scaling Σ and Q), so "
        "'energies', stability margins and un-normalised K or L are not measurable from FC; normalised indices "
        "(K_norm, L_norm) need their own identifiability check before use as biomarkers.",
        "<b>Fits are partly circular.</b> In CRRP the target drift is built from the same FC it is scored against; "
        "in LCF the gains follow from SC and fixed costs (Kc ≈ inverse degree), so high FC correlation mostly reflects "
        "the construction. Validation is within-run (time-split); there is no synthetic ground-truth recovery, "
        "cross-session test–retest, or subject-swap control.",
        "<b>Points to clarify with the authors</b>: the Bs = 0 null (r 0.21) and the cortex-only model (best fit) "
        "appear to describe the same closed loop; the closed-loop operator is stated rather than derived (a full "
        "observer-based loop has twice the state dimension); route orderings persist after shuffling (built in by "
        "nesting); CRRP's mixing weight g is weakly constrained (route estimates differ by ~10⁻⁶).",
        "<b>Not yet about PD.</b> No dopamine parameter, no direct/indirect pathway signs, thalamus and BG pooled in "
        "the main model, no EEG/TMS-EEG or patient data.",
        "<b>The same lesson as our study:</b> a good fit does not identify the gains. The remedy is also the same: "
        "perturbational, multi-condition data and explicit identifiability tests, which this plan can supply.",
    ])

    # ------------------------------------------------------------- 10 (new)
    s.append(P("10  Part 5: observer–controller identifiability and its link to the PD loop (Q5)", H1))
    s.append(P(
        "Aim: turn the observer–controller indices into quantities that are provably recoverable, and give them a "
        "mechanistic meaning in the PD loop of Part 2. Uses the teacher–student and Fisher/Laplace tools already "
        "validated in this study."))
    rows = [["ID", "Experiment", "Question"],
            ["O1", "Synthetic observer–controller teachers with known L, Kc, Ks and route weights; generate (a) static "
                   "FC, (b) multi-site stimulation-evoked responses, (c) both with a dopamine-like change in the BG "
                   "route; refit with the authors' procedure and with a Laplace posterior",
             "Which indices are identifiable from FC (expected: only the closed loop) vs from perturbational data?"],
            ["O2", "Linearise the Part 2 JR + basal-ganglia loop around its operating point and express it as an "
                   "observer–controller; read off L and K as functions of δ (dopamine gain), STN–GPe coupling and "
                   "cortical parameters",
             "Do Ks / K_norm / L_norm track δ monotonically and specifically (vs atrophy D3 and cortical change D5)?"],
            ["O3", "Fisher-based protocol design: choose stimulation sites (cortical, and DBS-like subcortical in "
                   "synthetic data) and ON/OFF-medication contrasts that make the BG route identifiable",
             "Minimum protocol that identifies the BG route and δ"],
            ["O4", "Product gate for K_norm / L_norm (sleep / meditation engine): synthetic recovery + test–retest on "
                   "repeated recordings, before any biomarker claim",
             "Are the deck's indices reliable quantities or fit-dependent summaries?"]]
    s += [table(rows, [0.06, 0.58, 0.36], font=7.1),
          P("Table 6c. Part 5 experiments. Pass criteria Q5 are in section 12, fixed before running.", CAP)]
    s += bullets([
        "<b>Expected outcome (stated now so it can be falsified):</b> from static FC only the closed-loop operator is "
        "identifiable; the observer/controller split and route weights need perturbations (stimulation, medication "
        "state). With multi-site stimulation, the BG route weight becomes identifiable to the extent that BG output "
        "reaches the stimulated cortical sites through the thalamus.",
        "<b>Why this matters for PD:</b> if O2 shows a monotonic, specific link between δ and an observer index, the "
        "observer formulation becomes a compact, interpretable read-out of dopaminergic loop gain: a candidate early "
        "PD marker that can also absorb MRI priors through the observation channels.",
        "<b>Questions for Zheng and colleagues:</b> the exact construction of Ac and A*; whether Kc and L are re-solved "
        "in the Bs = 0 null; the closed-loop derivation; how K_norm and L_norm are normalised; and the meditation data "
        "behind the deck. Ideally run O1 jointly.",
    ])

    s.append(P("11  Methods and code", H1))
    s += bullets([
        "Reuse: core/node.py (JR node, programmable stimulation, tonic drive); core/diffsim.py (network, fractional "
        "delays, log-scaled parameters, coupling_mode switch for L0/L2); scripts/identifiability.py and "
        "scripts/recover_clean.py (network Fisher and recovery); scripts/laplace_calibration.py (validated posterior).",
        "New: core/loop.py (subcortical populations, signed and directed coupling, dopamine gain); L3 in the network "
        "simulator.",
        "Fitting: the validated recipe (rich multi-condition stimulation, ≥ 15–20 restarts from population-nominal "
        "starts, lowest-loss selection, Laplace interval); simulation-based inference where posteriors are "
        "non-Gaussian (near transitions).",
        "Every change to neurofe's coupling is first validated here, then proposed as a neurofe change with tests.",
    ])

    # ------------------------------------------------------------- 10
    s.append(P("12  Pass criteria (to agree before running)", H1))
    rows = [["Item", "Passes if"],
            ["Q1 coupling form", "active operating point; late/early GMFA ratio within the real-data range; coupling "
                                 "bounds &lt; 20%"],
            ["Q2 loop identifiability", "δ bound &lt; 25% from cortical TMS-EEG alone in the active state"],
            ["Q3 detectability", "|δ| = 0.10 detected (d' &gt; 2) with ≤ 200 trials, correctly attributed in ≥ 8 of 10 "
                                 "fits, with D3 and D5 not misattributed to δ"],
            ["Q4 MRI priors", "at least one MRI prior halves the δ bound or the minimum detectable effect"],
            ["Q5 observer split", "from perturbational data (multi-site stimulation, ± dopamine change), the "
                                  "observer-vs-controller split and the BG route weight are recovered within 25% in "
                                  "≥ 8 of 10 synthetic subjects, with 90% intervals covering 80–98%; from static FC "
                                  "alone this is expected to fail (negative control)"],
            ["Q5 biomarkers", "normalised gains (K_norm, L_norm) recovered within 20% in synthetic data and, on real "
                              "data, test–retest ICC ≥ 0.7 across sessions"]]
    s += [table(rows, [0.25, 0.75]), P("Table 7. Proposed thresholds.", CAP)]

    # ------------------------------------------------------------- 11
    s.append(P("13  Schedule", H1))
    rows = [["Weeks", "Work", "Gate"],
            ["1–2", "Prerequisite shared with the main study: the 10-node gate (≥ 8 subjects, held-out sites, late "
                    "window, realistic teacher). In parallel: L3 (signed, directed) at 10 nodes; late/early TEP ratio "
                    "for L0–L3", "Network gate passed; default coupling chosen (Q1)"],
            ["3–4", "Literature review of basal ganglia mean-field models; core/loop.py; sanity checks (beta with δ)",
             "Loop teacher validated"],
            ["5–6", "Loop identifiability (Q2); disease teachers D1–D5, detectability and attribution (Q3)",
             "Minimum detectable effect"],
            ["7", "MRI priors (Q4) with the MRI collaborator's input on measures and their reliability",
             "Ranked MRI measures"],
            ["8", "Part 5 (O1–O3): observer–controller identifiability from FC vs perturbational data; link to "
                  "the JR/BG loop (can start in parallel with weeks 3–6)", "Q5 verdict"],
            ["9", "Write-up; proposed neurofe coupling change; plan for real data", "—"]]
    s += [table(rows, [0.08, 0.64, 0.28]), P("Table 8. Schedule (estimate, one person, CPU).", CAP)]

    # ------------------------------------------------------------- 12
    s.append(P("14  Risks and limitations", H1))
    s += bullets([
        "Neural-mass models of subcortical nuclei are coarse; results show what is possible in principle, not that a "
        "real PD marker exists.",
        "Dopamine as a single gain δ is a simplification of a multi-receptor, multi-timescale system.",
        "Synthetic detectability is an upper bound: real data add noise, misspecification and non-cortical "
        "components (auditory and somatosensory responses in TMS-EEG).",
        "Mappings from MRI measures to model parameters are hypotheses until calibrated on data.",
        "Cortical EEG may see subcortical changes only weakly; if so, Q2 fails and resting-state spectra or other "
        "modalities become necessary.",
    ])

    # ------------------------------------------------------------- 13
    s.append(P("15  Questions for the MRI collaborator", H1))
    s += bullets([
        "Which MRI markers do you consider most reliable for early PD, and at what stage do they change?",
        "Is there a quantitative relation between any of them and nigrostriatal dopamine function (e.g. against "
        "dopamine-transporter imaging) that could calibrate a prior on δ?",
        "Are there datasets with both MRI and TMS-EEG (or EEG) in early PD or prodromal cohorts?",
        "How reliable are subcortical tractography (nigrostriatal, pallido-thalamic, cortico-subthalamic) weights and "
        "lengths at the resolution available?",
        "Would resting-state fMRI connectivity of the motor loop be available in the same subjects?",
    ])
    return s


def footer(c, doc):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.8 * cm, 1.1 * cm, "syn_brain · study plan · connections between JR nodes and early PD detection")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
    c.restoreState()


def main():
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PLAN), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.7 * cm, title="Connections between JR nodes: study plan toward early PD detection",
                            author="syn_brain study")
    doc.build(content(), onFirstPage=footer, onLaterPages=footer)
    print("wrote", PLAN)


if __name__ == "__main__":
    main()
