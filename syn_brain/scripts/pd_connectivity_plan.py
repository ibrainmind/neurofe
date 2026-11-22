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

from syn_brain.scripts.single_node_tms_report import BOX, CAP, SMALL, TITLE, W, P, bullets, table
from syn_brain.scripts.single_node_tms_report import H1 as _H1, H2 as _H2

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
          P(f"syn_brain study plan · proposed {datetime.date.today():%d %B %Y} · builds on "
            "reports/single_node_tms_study.pdf and docs/background_drive_note.txt", CAP)]

    s.append(P("Summary", H1))
    s.append(P(
        "In a discussion with a collaborator working on MRI and early detection of Parkinson's disease (PD), the question "
        "came up of how Jansen-Rit (JR) nodes are connected in our network model. The answer, that connections are "
        "approximately linear, is correct, and the single-node study shows it is a substantive limitation. neurofe's "
        "coupling is linear, diffusive, symmetric and excitatory only, provides no mean drive, and the anatomy contains "
        "no subcortical nodes. Every node therefore rests in a quiet, near-linear state in which loop mechanisms are "
        "invisible in TMS-EEG. PD is primarily a disorder of loops (basal ganglia–thalamus–cortex) and of "
        "dopamine-dependent gain on them, so the present model cannot represent it. This plan (1) tests which "
        "connection form lets the network enter an active state and makes connection changes visible, (2) adds a minimal "
        "cortico-basal ganglia-thalamic loop with dopamine-dependent gain, (3) measures how small a dopamine-like change "
        "TMS-EEG can detect, and (4) quantifies how much MRI-derived priors improve identification.", BOX))

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
        "noise as the only other input."))
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
    s.append(P("3  What the single-node study already shows", H1))
    rows = [["Finding (single-node study)", "Consequence for connections and PD"],
            ["Laplacian coupling gives no mean drive; a node with no drive rests at E − I ≈ −1.9 mV, far below "
             "threshold (6 mV), sigmoid slope 0.03 of a possible 0.7",
             "Changing connection strength barely changes the resting state; the network stays quiet and "
             "near-linear"],
            ["A quiet node behaves like a linear filter; loop gains appear only in an active state (tonic drive "
             "p ≳ 115 moves the node to threshold and the c3 bound from 129% to ~5%)",
             "Loop-level disease mechanisms are invisible unless the network is driven into an active state"],
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
    s += [table(rows, [0.5, 0.5]), P("Table 2. Single-node results relevant to the connection question "
                                     "(details in reports/single_node_tms_study.pdf).", CAP)]
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
                                                     "separate"]]
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
    s.append(P("9  Methods and code", H1))
    s += bullets([
        "Reuse: core/node.py (JR node, programmable stimulation, tonic drive), core/diffsim.py (network), the Fisher, "
        "profile-likelihood and batched-training code of scripts/single_node_tms.py.",
        "New: core/loop.py (subcortical populations, signed and directed coupling, dopamine gain); a coupling-form "
        "switch in the network simulator (L0–L3).",
        "Fitting: log-scaled parameters throughout; Levenberg–Marquardt for the global parameters; simulation-based "
        "inference for full posteriors once the model is fixed.",
        "Every change to neurofe's coupling is first validated here, then proposed as a neurofe change with tests.",
    ])

    # ------------------------------------------------------------- 10
    s.append(P("10  Pass criteria (to agree before running)", H1))
    rows = [["Item", "Passes if"],
            ["Q1 coupling form", "active operating point; late/early GMFA ratio within the real-data range; coupling "
                                 "bounds &lt; 20%"],
            ["Q2 loop identifiability", "δ bound &lt; 25% from cortical TMS-EEG alone in the active state"],
            ["Q3 detectability", "|δ| = 0.10 detected (d' &gt; 2) with ≤ 200 trials, correctly attributed in ≥ 8 of 10 "
                                 "fits, with D3 and D5 not misattributed to δ"],
            ["Q4 MRI priors", "at least one MRI prior halves the δ bound or the minimum detectable effect"]]
    s += [table(rows, [0.25, 0.75]), P("Table 7. Proposed thresholds.", CAP)]

    # ------------------------------------------------------------- 11
    s.append(P("11  Schedule", H1))
    rows = [["Weeks", "Work", "Gate"],
            ["1–2", "Coupling ladder L0–L3 on 10 nodes; operating points; TEP realism; identifiability",
             "Default coupling chosen (Q1)"],
            ["3–4", "Literature review of basal ganglia mean-field models; core/loop.py; sanity checks (beta with δ)",
             "Loop teacher validated"],
            ["5–6", "Loop identifiability (Q2); disease teachers D1–D5, detectability and attribution (Q3)",
             "Minimum detectable effect"],
            ["7", "MRI priors (Q4) with the MRI collaborator's input on measures and their reliability",
             "Ranked MRI measures"],
            ["8", "Write-up; proposed neurofe coupling change; plan for real data", "—"]]
    s += [table(rows, [0.08, 0.64, 0.28]), P("Table 8. Schedule (estimate, one person, CPU).", CAP)]

    # ------------------------------------------------------------- 12
    s.append(P("12  Risks and limitations", H1))
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
    s.append(P("13  Questions for the MRI collaborator", H1))
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
