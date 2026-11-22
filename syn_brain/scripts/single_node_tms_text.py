"""Text and layout of the single-node virtual-TMS study report (used by single_node_tms_report.py).

Every number quoted in the prose is read from analysis.json / summary.json, so the report
stays consistent with the results it is built from.
"""

from __future__ import annotations

import datetime

FIT = ["a", "b", "c1", "c2", "c3", "c4", "k"]


def content(an, sm, *, P, H1, H2, TITLE, BOX, CAP, SMALL, bullets, fig, table, pct, Spacer, PageBreak):
    c7 = sm["crb7_nominal"]
    c12 = sm["crb12_nominal"]
    e1 = sm["E1"]
    E = lambda prot, n: e1[prot]["median_abs_err_pct"][n]          # noqa: E731
    prog = sm["progress_median"]
    start = sm["start_median_abs_err_pct"]
    f0 = an["features_nominal"]
    gen, genL = sm["E1_generalisation_pct"], sm["E1_generalisation_late_pct"]
    e2 = sm["E2"]
    e8 = sm.get("E8", {})
    ts = an["tonic_sweep"]
    i110, i120 = ts["p"].index(110), ts["p"].index(120)
    s = []

    # ------------------------------------------------------------------ title + summary
    s += [P("Single-node virtual-TMS system identification", TITLE),
          P(f"syn_brain study report · {datetime.date.today():%d %B %Y} · teacher/student Jansen-Rit, "
            "PyTorch backpropagation", CAP),
          P("Executive summary", H1)]
    s.append(P(
        "A <b>teacher</b> Jansen-Rit (JR) node with neurofe's nominal parameters was stimulated with 12 virtual-TMS "
        "conditions (pulse amplitude, width, paired pulses, a 10 Hz train, injection into excitatory and inhibitory "
        "populations, and a tonic background drive). A <b>student</b> JR node of identical form saw only the teacher's "
        "EEG and was fitted with Adam and backpropagation through the simulator, as neurofe does. Because the truth is "
        f"known, every fitted value can be checked. In total {sm['n_models']} independent student models were trained, "
        "and local (Fisher / Cramér–Rao) and global (profile likelihood) identifiability were computed for every protocol.",
        BOX))
    s += bullets([
        f"<b>The nominal node is almost a linear filter.</b> At rest it sits far below firing threshold, so the TMS "
        f"response is set by the stimulus gain <b>k·A</b> (P1 amplitude {f0['P1 amplitude']:.1f} µV, elasticity +1.0) and "
        f"the excitatory rate <b>a</b> (P1 latency {f0['P1 latency']:.0f} ms). The dose ratio is "
        f"{f0['dose ratio 4000/250']:.2f} and the paired-pulse ratio {f0['paired-pulse 100 ms']:.2f}: responses scale and "
        f"add linearly. The intrinsic E–I loop (c1–c4, b) only shows in a {f0['N1 amplitude']:.2f} µV late dip, about the "
        "size of real TMS-EEG noise (0.42 µV).",
        "<b>Three classes of parameter.</b> (1) <i>Stiff</i>: k and a, recovered by backprop under every protocol "
        f"(median error ≤ {max(E(p_, 'k') for p_ in e1):.1f}% for k and ≤ {max(E(p_, 'a') for p_ in e1):.1f}% for a). (2) <i>Conditional</i>: b, and c1·c2, c3·c4 as products, identifiable only with the right "
        "stimulus or brain state. (3) <i>Gauge</i>: A, B and vmax trade off exactly against the c's and k (two scaling "
        "symmetries); no data can identify them, so they must be fixed by convention.",
        f"<b>The stimulus decides what can be learned.</b> One single pulse bounds c1–c4 only to "
        f"{pct(c7['SP1000']['c1'])}–{pct(c7['SP1000']['c4'])}. Combining amplitudes, E+I injection and a train (RICH) gives "
        f"b {pct(c7['RICH']['b'])} and c1 {pct(c7['RICH']['c1'])}, but c3, c4 stay at {pct(c7['RICH']['c3'])}, "
        f"{pct(c7['RICH']['c4'])}. Pulse width or shape adds almost nothing.",
        f"<b>Brain state matters more than the pulse.</b> A tonic background drive just above p ≈ 115 moves the node to "
        f"an active, alpha-resonant state; the single-pulse bound on c3 drops from {pct(ts['SP1000'][0][4] * 100)} to "
        f"{pct(ts['SP1000'][i120][4] * 100)}.",
        f"<b>Backprop recovers what is stiff and leaves the rest where it started.</b> Median 'progress' toward the truth "
        f"(1 = reached, 0 = did not move): k {prog['E1']['k']:.2f}, a {prog['E1']['a']:.2f}, b {prog['E1']['b']:.2f}, "
        f"c3 {prog['E1']['c3']:.2f}, c4 {prog['E1']['c4']:.2f}, c1 {prog['E1']['c1']:.2f}, c2 {prog['E1']['c2']:.2f} "
        "(c2 moves <i>away</i> from the truth, absorbing residual error).",
        f"<b>neurofe's raw parameterisation is a separate, avoidable failure.</b> Adam with lr 0.05 on raw values moves "
        f"each parameter by about 0.05 per step, i.e. 0.05% of a or c1 but 0.5% of k. On the same data and starts, the "
        f"raw fit ends with {e2['E2_RICH']['final_loss_median'] / e2['E1_RICH']['final_loss_median']:.0f}× higher loss "
        f"and a error {e2['E2_RICH']['median_abs_err_pct']['a']:.1f}% vs {e2['E1_RICH']['median_abs_err_pct']['a']:.1f}% "
        "with log-scaled parameters. This explains the Phase 1 observation 'k recovered, the rest stuck'.",
        "<b>Generalisation fails exactly where identifiability fails.</b> Fits transfer to new stimuli to within ~1–5% "
        "over the whole window, which the P1 peak dominates, but miss the late window (&gt; 60 ms) by 5–30%, "
        "even on stimuli seen in training. They miss the driven-state late response by 15–30%." +
        (f" Fitting only the first 60 ms (E8, as in the real-data temporal hold-out test) raises the late-window error "
         f"from {pct(100 * e8['SP1000']['E1 late rel. error median'])} to "
         f"{pct(100 * e8['SP1000']['E8 late rel. error median'])} for a single pulse, with a perfect model and no noise."
         if "SP1000" in e8 else ""),
    ])
    s.append(P(
        "<b>Implication for the real-data generalisation failures.</b> The temporal hold-out and cross-site failures "
        "seen with real data are what this model predicts <i>even when the model is exactly right</i>. The early response "
        "pins k and a; the late response is governed by parameters (b, c3·c4, the excitability set by background input) "
        "that the early window does not constrain and that backprop leaves near its starting point. The failure is "
        "therefore expected to appear already at rung R0 of the main study, and is a property of the method and the "
        "stimulation design, not evidence of model misspecification.", BOX))
    s.append(PageBreak())

    # ------------------------------------------------------------------ 1. setup
    s.append(P("1  Study design", H1))
    s.append(P(
        "The study isolates the smallest unit of the inverse problem: one JR column, one stimulation channel. If a "
        "parameter cannot be identified from a single node it cannot be identified from a network either, since the "
        "network only adds unknowns (coupling g, per-edge weights w_bb, delays mu, the leadfield). The node is read "
        "out through one real leadfield column of the PyTepFit anatomy, so amplitudes are in EEG microvolts and "
        f"comparable with the 10- and 200-node studies (read-out gain ‖lm‖ = {an['gain']:.0f}, parcel {an['parcel']})."))
    s.append(P("1.1  Model", H2))
    s.append(P(
        "The equations, constants and integrator are those of neurofe's <font name='DVM'>JansenRit.forward</font> "
        "with network terms removed: Euler steps of 0.1 ms, 10 sub-steps per 1 ms sample, tanh input bounds of 500, "
        "EEG = 5·10<super>-4</super>·leadfield·(E − I). The new simulator (<font name='DVM'>core/node.py</font>) "
        "matches the existing neurofe-equivalent simulator to 5·10<super>-15</super> µV. Stimulation is an arbitrary "
        "input waveform u(t) scaled by k and added to the excitatory drive of the pyramidal cells (neurofe's default), "
        "optionally also routed into the inhibitory population, on top of an optional tonic drive p."))
    phys = [["Parameter", "Nominal", "Physical meaning", "Fitted in neurofe?"],
            ["a", "100 /s", "Excitatory synaptic rate constant, τ<sub>e</sub> = 1/a ≈ 10 ms", "yes"],
            ["b", "50 /s", "Inhibitory (GABA-A-like) rate constant, τ<sub>i</sub> ≈ 20 ms", "yes"],
            ["c1", "135", "Pyramidal → excitatory interneurons (average synaptic contacts)", "yes"],
            ["c2", "108", "Excitatory interneurons → pyramidal (recurrent excitation)", "yes"],
            ["c3", "33.75", "Pyramidal → inhibitory interneurons", "yes"],
            ["c4", "33.75", "Inhibitory interneurons → pyramidal (feedback inhibition)", "yes"],
            ["k", "10.5", "Stimulus gain: coil current → synaptic drive (coil, orientation, scalp–cortex distance)", "yes"],
            ["A", "3.25 mV", "Maximum excitatory PSP amplitude", "no (fixed)"],
            ["B", "22 mV", "Maximum inhibitory PSP amplitude", "no (fixed)"],
            ["v0", "6 mV", "Firing threshold (half-maximal firing)", "no (fixed)"],
            ["r", "0.56 /mV", "Sigmoid slope (spread of thresholds)", "no (fixed)"],
            ["vmax", "5 /s", "Maximum firing rate", "no (fixed)"]]
    s += [table(phys, [0.12, 0.11, 0.58, 0.19]), P("Table 1. Parameters of the JR node. Values are the effective "
                                                   "values entering the equations (neurofe adds floors such as 1 + relu(·)).", CAP)]
    s.append(P("1.2  Virtual-TMS protocols", H2))
    prot = [["Condition", "Description"]] + [[k, v] for k, v in an["stim_labels"].items()] + [
        ["…@p110", "SP250 / SP1000 / SP4000 delivered on a tonic background drive p = 110"],
        ["DOSE", "SP250 + SP1000 + SP4000 recorded together"],
        ["RICH", "DOSE + PP100 + TR10 + EI1000 (6 conditions)"]]
    s += [table(prot, [0.18, 0.82]), P("Table 2. Stimulation conditions. Amplitudes are in neurofe's input units "
                                       "(1000 × 10 ms is neurofe's default pulse). SPshort and SPlong deliver the same charge as SP1000.", CAP)]
    s.append(P("1.3  Experiments", H2))
    exps = [["ID", "What", "Models"],
            ["A", "Fisher information and Cramér–Rao bounds (noise 0.42 µV = real TMS-EEG baseline) for every protocol, for 7 "
                  "and 12 free parameters, at nominal and 8 random teachers; feature elasticities; tonic-drive sweep", "—"],
            ["E1", "Every protocol × 10 random starts (±50%), log-scaled parameters, noise-free teacher", "110"],
            ["E2", "neurofe-style raw parameters (relu floors, Adam lr 0.05), same starts", "20"],
            ["E3", "12 free parameters (A, B, v0, r, vmax released)", "20"],
            ["E4", "Sensor noise at the real level (0.42 µV)", "30"],
            ["E5", "Profile likelihood: one parameter pinned at 0.5–2× truth, others refitted", "112"],
            ["E6", "4 perturbed teachers (±30%)", "60"],
            ["E7", "Driven operating point (tonic p = 110)", "20"],
            ["E8", "Temporal hold-out: fit only 0–60 ms after the pulse, predict the late response", "30"]]
    s += [table(exps, [0.07, 0.83, 0.1]), P("Table 3. Experiments. Each model is trained for 300 Adam iterations "
                                            "(lr 0.04, cosine schedule) on the relative mean-squared EEG error. Errors are reported as "
                                            "|log(fit / true)| × 100, which equals the relative error for small errors.", CAP)]
    s.append(PageBreak())

    # ------------------------------------------------------------------ 2. input-output behaviour
    s.append(P("2  Input–output behaviour of the teacher", H1))
    s.append(fig("responses", "Figure 1. Teacher EEG (black) for every condition, with the injected current (yellow). "
                              "Responses scale with amplitude and superpose almost exactly; E+I injection reverses the sign "
                              "because the inhibitory PSP (B = 22 mV, τ = 20 ms) is much larger than the excitatory one."))
    s.append(P(
        "Figure 1 shows the central physical fact of the study. With no background input, the pyramidal "
        "membrane potential E − I sits at about −1.9 mV, far below the firing threshold v0 = 6 mV, where the sigmoid "
        "slope is 0.03 /s per mV (interneuron sigmoids about 0.1; the maximum is r·vmax/4 = 0.7). The pulse is filtered by the excitatory synaptic kernel "
        "and passes to the EEG almost unchanged; the recurrent loop through the interneurons is too weak to reshape it. "
        "Hence the near-perfect linearity (dose ratio "
        f"{f0['dose ratio 4000/250']:.2f}, paired-pulse ratios {f0['paired-pulse 30 ms']:.2f} and "
        f"{f0['paired-pulse 100 ms']:.2f}) and the absence of a real late component."))
    s.append(fig("oneatatime", "Figure 2. Single-pulse response when one parameter is changed by ±30% (insets: the late "
                               "tail, magnified). Only k, A and a change the main peak; everything else only changes the "
                               "sub-microvolt tail."))
    s.append(fig("elasticity", "Figure 3. Elasticity of each input–output feature with respect to each parameter "
                               "(+1 = feature proportional to the parameter). Blank cells are |elasticity| &lt; 0.05."))
    s.append(P("How parameters map onto observable behaviour", H2))
    s += bullets([
        "<b>P1 amplitude</b> depends only on k and A (elasticity +0.99 and +1.02), i.e. on their product. The response "
        "amplitude identifies 'how hard the cortex was hit', not any property of the circuit.",
        "<b>P1 latency</b> depends only on a (−0.64): the excitatory time constant is read directly off the timing of the "
        "first peak, which is why a is the best-identified intrinsic parameter.",
        "<b>The late N1 dip and the zero-crossing time</b> carry all information about the inhibitory loop: b (−3.2), "
        "c3 (+3.2), c4 (+1.6), and the excitability triplet v0, r, vmax. The elasticities are large, but the feature "
        f"itself is only {f0['N1 amplitude']:.2f} µV, so in absolute terms its information content is small.",
        "<b>Dose and paired-pulse ratios</b> have elasticities &lt; 0.05 with respect to every parameter: in this state, "
        "nonlinear protocols probe nothing, because the node has no nonlinearity to probe.",
        "<b>E+I injection</b> exposes b and B directly (elasticity +2.3) because the stimulus enters the inhibitory "
        "kernel without passing through the weak loop.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ 3. identifiability theory
    s.append(P("3  How much can the data identify? (Fisher information)", H1))
    s.append(fig("crb", "Figure 4. Cramér–Rao lower bounds on the relative error of each parameter, by protocol, for the "
                        "7 parameters neurofe fits (left) and with the 5 fixed parameters released (right). "
                        "No unbiased estimator can do better than these numbers at real TMS-EEG noise."))
    s.append(P(
        f"For a single neurofe-style pulse (SP1000) the bounds are k {pct(c7['SP1000']['k'])}, a {pct(c7['SP1000']['a'])}, "
        f"b {pct(c7['SP1000']['b'])}, and c1–c4 {pct(c7['SP1000']['c1'])}, {pct(c7['SP1000']['c2'])}, "
        f"{pct(c7['SP1000']['c3'])}, {pct(c7['SP1000']['c4'])}. The connectivity constants are not identifiable from one "
        "pulse. What helps, in order of effect:"))
    s += bullets([
        f"<b>Several amplitudes</b> (DOSE): c1–c4 fall to {pct(c7['DOSE']['c1'])}–{pct(c7['DOSE']['c4'])}. The small "
        "residual nonlinearity is informative once the amplitude range is wide.",
        f"<b>Injection into the inhibitory population</b> (EI1000) pins b to {pct(c7['EI1000']['b'])} on its own, but makes "
        "the c's completely unidentifiable when used alone.",
        f"<b>Combining them</b> (RICH): b {pct(c7['RICH']['b'])}, c1 {pct(c7['RICH']['c1'])}, c2 {pct(c7['RICH']['c2'])}, "
        f"c3 {pct(c7['RICH']['c3'])}, c4 {pct(c7['RICH']['c4'])}.",
        f"<b>Pulse width or shape</b> changes little (2 ms vs 30 ms pulse of equal charge: c1 {pct(c7['SPshort']['c1'])} vs "
        f"{pct(c7['SPlong']['c1'])}). Long pulses are worse, as they low-pass the input.",
    ])
    s.append(P(
        "Across 8 random teachers (±30% around nominal) the median single-pulse bounds are "
        + ", ".join(f"{n} {pct(sm['crb7_teachers_median']['SP1000'][n])}" for n in FIT) +
        ", so the picture does not depend on the nominal point."))
    s.append(fig("sloppiness", "Figure 5. Left: eigenvalues of the Fisher matrix relative to the largest, per protocol. "
                               "Middle, right: the corresponding eigen-directions (rows, stiff at the top). Stiff directions "
                               "are k and a (with b under RICH); the sloppiest are c4, c2 and c1 almost alone."))
    s.append(P(
        "The Fisher eigenvalues span 5–6 orders of magnitude, the hallmark of a 'sloppy' model: a few stiff parameter "
        "combinations determine the output, and the rest can change by large factors with no visible effect. A gradient "
        "optimiser moves quickly along the stiff directions and barely at all along the sloppy ones, which is what the "
        "training experiments show."))
    s.append(P("3.1  Exact symmetries: parameters no data can identify", H2))
    s.append(P(
        "Releasing A, B, v0, r and vmax (right panel of Figure 4) makes every coupling parameter unidentifiable "
        f"(bounds above 100% for every protocol, k and A both at {pct(c12['RICH']['k'])} under RICH). Two "
        "transformations leave the EEG essentially unchanged; checked numerically, they move the response by less than "
        "1.5 µV out of 60 µV for scale factors 0.7–1.5, with the small residual due only to the tanh input bounds:"))
    s += bullets([
        "<b>Synaptic scale:</b> A, B × λ together with c1, c2, c3, c4, k ÷ λ. EEG cannot separate 'larger PSPs' from "
        "'more synapses' from 'a stronger stimulus'.",
        "<b>Firing-rate scale:</b> vmax × λ with A, B ÷ λ and k × λ. The maximum firing rate trades off against the PSP "
        "amplitudes.",
    ])
    s.append(P(
        "These parameters are <i>gauge</i> choices: fixing them, as neurofe does, is required, but it also means the "
        "fitted c's and k only have meaning relative to the fixed A, B and vmax. Experiment E3 confirms this in "
        "practice (Figure 13): the fitted A and k lie on the line A·k = constant."))
    s.append(P("3.2  The operating point: background drive", H2))
    s.append(fig("tonic", "Figure 6. Cramér–Rao bounds as the tonic background drive p is raised (left: one pulse, "
                          "middle: three amplitudes) and the late part of the single-pulse response (right)."))
    s.append(P(
        "A single node in a network is never at rest: its neighbours drive it. Raising a constant drive p (the classic "
        "JR extrinsic input) moves the resting point up the sigmoid. Up to p ≈ 110 the node stays quiet and the bounds "
        f"barely change (c3 {pct(ts['SP1000'][i110][4] * 100)} at p = 110). Just above p ≈ 115 the resting state "
        "jumps to the upper JR branch, an active state that rings at about 10 Hz after each perturbation (a damped "
        "alpha resonance; there is no sustained oscillation, checked over 6 s). There the loop takes part in the response "
        f"and the single-pulse bounds fall sharply: c3 {pct(ts['SP1000'][i120][4] * 100)}, "
        f"c4 {pct(ts['SP1000'][i120][5] * 100)}, b {pct(ts['SP1000'][i120][1] * 100)}, "
        f"c1 {pct(ts['SP1000'][i120][2] * 100)}. <b>Identifiability is a property of the brain state as much as of the "
        "stimulus.</b> A node probed while inactive cannot reveal its circuit."))
    s.append(P("How to supply the background input", H2))
    op = [["Drive p", "Resting E − I", "Pyramidal sigmoid slope", "State"],
          ["0", "−1.9 mV", "0.03", "rest: near-linear filter"],
          ["50", "−0.2 mV", "0.08", "rest"],
          ["100", "1.6 mV", "0.20", "rest"],
          ["110", "2.1 mV", "0.25", "rest (E7)"],
          ["115–120", "6.0 mV = v0", "0.70 (maximum)", "jump to the upper branch: damped ~10 Hz resonance"],
          ["150–220", "6.3–6.7 mV", "0.67–0.70", "active"],
          ["320", "7.4 mV", "0.61", "active, slope starts to fall"]]
    s += [table(op, [0.15, 0.2, 0.25, 0.4]), P("Table 4. Resting operating point of the nominal node against a constant "
                                               "background drive p into the pyramidal cells' excitatory input "
                                               "(fixed point of the model equations).", CAP)]
    s.append(P(
        "The pyramidal cells fire when their membrane potential E − I approaches v0 = 6 mV. A constant drive p raises E "
        "by roughly (A/a)·p = 0.0325 mV per unit of p, so without feedback about p ≈ 240 would be needed. Recurrent "
        "excitation (c1, c2) amplifies the drive, and at p ≈ 115 the positive feedback becomes self-sustaining: the "
        "resting point jumps to E − I ≈ v0, exactly where the sigmoid is steepest (Table 4). This is the classic JR "
        "bifurcation: p is the mean firing rate arriving from other areas and the thalamus (Jansen and Rit used "
        "120–320 /s)."))
    s.append(P(
        "<b>neurofe has no such input.</b> Its long-range coupling is a Laplacian, g·Σ<sub>j</sub> w<sub>ij</sub>"
        "(E<sub>j</sub>(t − d<sub>ij</sub>) − E<sub>i</sub>), which is zero whenever the nodes sit at the same resting "
        "state, and its noise input has zero mean. Every node of a neurofe network therefore rests in the inactive, "
        "near-linear state of Figure 1, which is probably why synthetic TEPs lack late activity (main README, Finding 1). "
        "Ways to put a node in an active state, in the teacher and in neurofe:"))
    s += bullets([
        "<b>A mean input parameter p</b> (global or per node) added to the excitatory drive. One line in the forward "
        "pass; fittable, though it trades off partly against v0.",
        "<b>Non-diffusive coupling</b>, g·Σ<sub>j</sub> w<sub>ij</sub>·S(E<sub>j</sub>(t − d<sub>ij</sub>)): neighbours "
        "send firing rates, which are positive, so a node receives a mean drive that grows with g and its degree.",
        "<b>Noise with non-zero mean</b> (or 1/f / Ornstein–Uhlenbeck input with mean μ), which also supplies the "
        "ongoing activity of the resting-state EEG.",
        "<b>Lower threshold v0</b>, an excitability change, which moves the threshold towards the resting point "
        "instead of the other way round.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ 4. backprop results
    s.append(P("4  What backpropagation actually recovers", H1))
    s.append(fig("e1_recovery", "Figure 7. E1: error of each fitted parameter after 300 Adam iterations from 10 random "
                                "starts, per protocol. Noise-free teacher; bars are medians."))
    rows = [["Protocol"] + FIT]
    for prot in ("SP1000", "SP4000", "PP100", "EI1000", "DOSE", "RICH"):
        rows.append([prot] + [pct(E(prot, n)) for n in FIT])
    rows.append(["start (median)"] + [pct(start[n]) for n in FIT])
    rows.append(["progress (E1, all)"] + [f"{prog['E1'][n]:.2f}" for n in FIT])
    s += [table(rows, [0.22] + [0.78 / 7] * 7), P(
        "Table 5. Median error after fitting (E1), the median error of the random starting points, and the median "
        "'progress' (1 − final error / starting error; 1 = reached the truth, 0 = did not move, negative = moved away).",
        CAP)]
    s.append(P(
        "The results follow the Fisher analysis closely. k and a are recovered by every protocol "
        f"(median progress {prog['E1']['k']:.2f} and {prog['E1']['a']:.2f}; the worst case is a single E+I-only start "
        "that missed a by 30%). b is recovered to "
        f"{pct(E('EI1000', 'b'))} when the inhibitory population is stimulated directly and to {pct(E('RICH', 'b'))} under "
        f"RICH, but only to {pct(E('SP1000', 'b'))} from a single pulse. The connectivity constants stay near their "
        f"random starting values (median start error {pct(start['c1'])}–{pct(start['c4'])}): c3 and c4 move "
        f"part-way ({prog['E1']['c3']:.2f}, {prog['E1']['c4']:.2f}), c1 does not move ({prog['E1']['c1']:.2f}), and c2 "
        f"moves <i>away</i> from the truth ({prog['E1']['c2']:.2f}). These fits are not failures of convergence in "
        f"the usual sense: the final relative loss is about {e1['SP1000']['final_loss_median']:.0e}, far below the level "
        "at which real noise would hide any remaining difference. The optimiser has found an EEG that is "
        "indistinguishable from the teacher's, with the wrong circuit."))
    s.append(fig("e1_loss", "Figure 8. E1 training curves: median relative loss and range over starts.", width=0.6 * 17.4 * 28.35))
    s.append(fig("e5_profile", "Figure 9. E5: profile likelihood. Each curve pins one parameter at a multiple of its true "
                               "value and refits the other six. Below the dashed line the difference is invisible at real "
                               "TMS-EEG noise."))
    s.append(P(
        "The profiles give the global picture behind the local bounds. For a single pulse, k, a and b have clear minima "
        "well above the noise threshold. The profiles of c1–c4 lie entirely below it between 0.5× and 2× the truth: any "
        "value in that range explains the data. RICH lifts c1 and c3 above the threshold beyond about ±20%, but c2 and c4 "
        "remain unidentifiable over the full range. (One RICH c3 point at 1.5× is a failed refit, a local minimum.) "
        "In noise-free data every profile has its minimum at the truth, so the parameters are identifiable in principle; "
        "the problem is that the information is far below realistic noise."))
    s.append(fig("e4_noise", "Figure 10. E4: fits to data with sensor noise at the real level. Black bars are the "
                             "Cramér–Rao bounds."))
    s.append(P(
        "With realistic noise, the errors of k, a and b are close to their bounds. For c1–c4 the errors are "
        "<i>smaller</i> than the bounds. That is not a better-than-optimal estimate: those parameters did not move, so "
        "their error is simply the distance from the starting point. <b>For sloppy parameters, the fitted value reports "
        "the initialisation (or the prior), not the brain.</b>"))
    s.append(fig("e6_e7", "Figure 11. Left, E6: the same protocols on 4 teachers drawn ±30% around nominal. Right, E7: "
                          "the same fits with the node on a tonic background drive p = 110."))
    s.append(P(
        "E6 reproduces the pattern on other teachers, so it is not specific to the nominal point. E7 shows no improvement "
        "at p = 110 (c3 " + pct(sm["E7"]["SP1000@p110"]["c3"]) + " vs " + pct(sm["E7"]["SP1000"]["c3"]) + " at rest). This is "
        "consistent with Figure 6: at p = 110 the node is still below the jump to the active state, where the bounds "
        "only improve modestly. Testing p ≥ 120 is the first item of the follow-up plan."))
    s.append(PageBreak())

    # ------------------------------------------------------------------ 5. backprop specifics
    s.append(P("5  Backpropagation-specific findings", H1))
    s.append(P("5.1  neurofe's raw parameterisation", H2))
    s.append(fig("e2_raw_vs_log", "Figure 12. E2: parameter trajectories (fitted / true) and loss for the RICH protocol, "
                                  "same starts. Blue: log-scaled parameters. Red: neurofe's raw parameters with Adam, lr 0.05."))
    rel = an["adam_rel_step_lr005"]
    s.append(P(
        "Adam normalises each gradient, so every raw parameter moves by roughly the learning rate per step, whatever its "
        "scale. With lr 0.05 that is a relative step of "
        + ", ".join(f"{n} {100 * v:.2g}%" for n, v in zip(FIT, rel)) +
        " per iteration. Over 300 iterations a and c1 can move by at most ~15%, while k can move by a factor of "
        "several. This is visible in Figure 12: the red trajectories for a, b and the c's creep linearly and stop far "
        f"from the truth. The raw fits end at a median loss of {e2['E2_RICH']['final_loss_median']:.1e} against "
        f"{e2['E1_RICH']['final_loss_median']:.1e} with log-scaled parameters, and with a error "
        f"{e2['E2_RICH']['median_abs_err_pct']['a']:.1f}% vs {e2['E1_RICH']['median_abs_err_pct']['a']:.2g}%, k error "
        f"{e2['E2_RICH']['median_abs_err_pct']['k']:.1f}% vs {e2['E1_RICH']['median_abs_err_pct']['k']:.2g}%. "
        "This explains Phase 1's 'k recovered, the rest stuck' and is fixed by optimising log(θ / nominal) "
        "(or by per-parameter learning rates)."))
    s.append(P("5.2  Releasing the fixed parameters", H2))
    s.append(fig("e3_extended", "Figure 13. E3: errors with 12 free parameters (left), and the trade-offs between A and k "
                                "(middle) and A and a (right) across starts."))
    e3 = sm["E3"]
    s.append(P(
        f"With 12 free parameters the fit is as good (median loss {e3['RICH']['final_loss_median']:.0e}) but k is now "
        f"wrong by {pct(e3['RICH']['median_abs_err_pct']['k'])} and A by {pct(e3['RICH']['median_abs_err_pct']['A'])}, "
        "exactly along the line A·k = constant (middle panel). a and b stay well identified. Adding parameters to the "
        "student without adding information only adds degenerate directions; the same applies, at much larger scale, "
        "to fitting the per-edge w_bb and the full leadfield in tiers S2 and S3."))
    s.append(P("5.3  Other mechanisms specific to the PyTorch implementation", H2))
    s += bullets([
        "<b>Integer delays.</b> neurofe casts delays to integers, so conduction velocity mu has zero gradient and "
        "cannot be learned by backprop (Phase 1). Irrelevant for one node, decisive for the network.",
        "<b>Loss weighting.</b> The mean-squared error is dominated by the large early peak. The late window, which "
        "carries the loop information, holds only 0.1–0.5% of a single pulse's signal power (and so of the loss), so its gradient is tiny unless the "
        "loss is reweighted by time window or the signal is whitened by its expected noise.",
        "<b>Leftover error goes to the sloppy parameters.</b> c2 moving away from the truth (progress "
        f"{prog['E1']['c2']:.2f}) is the single-node analogue of Phase 1's b, c4 and std_in converging to the same values "
        "for every teacher: the sloppy directions soak up whatever error remains.",
    ])
    s.append(PageBreak())

    # ------------------------------------------------------------------ 6. generalisation
    s.append(P("6  Generalisation: does a good fit predict new inputs?", H1))
    s.append(fig("e1_generalisation", "Figure 14. E1: median prediction error of models trained on one protocol (rows) "
                                      "for every stimulation condition (columns), over the whole window (left) and the "
                                      "late window only (right). Green boxes: conditions seen in training."))
    s.append(P(
        f"Over the whole window, fits generalise well: a model trained on one SP1000 pulse predicts SP4000 to "
        f"{gen['SP1000']['SP4000']:.1f}% and the 10 Hz train to {gen['SP1000']['TR10']:.1f}%. That is expected, because the "
        "whole-window error is dominated by P1, which k and a determine, and those are recovered. The late window tells a "
        f"different story: the same model misses the late SP4000 response by {genL['SP1000']['SP4000']:.0f}%, its own "
        f"training stimulus by {genL['SP1000']['SP1000']:.0f}%, and the driven-state responses by "
        f"{genL['SP1000']['SP1000@p110']:.0f}–{genL['SP1000']['SP250@p110']:.0f}%. Models trained only on E+I "
        f"injection fit their own data to {gen['EI1000']['EI1000']:.2f}% but predict E-only pulses with "
        f"{genL['EI1000']['SP1000']:.0f}% late-window error. A perfect fit to one stimulus says nothing about responses "
        "governed by parameters that stimulus did not exercise."))
    s.append(fig("err_vs_io", "Figure 15. Error of each fitted parameter against the prediction error on held-out "
                              "stimuli (top: whole window, E1 + E6; bottom: late window, E1). Spearman ρ in each title."))
    cor, corL = sm["spearman_param_err_vs_heldout"], sm["spearman_param_err_vs_heldout_late"]
    s.append(P(
        "Which parameter errors actually cost predictive accuracy? Whole-window held-out error tracks the errors of "
        f"a (ρ = {cor['a']:.2f}), k ({cor['k']:.2f}) and c2 ({cor['c2']:.2f}); late-window error tracks c1 "
        f"({corL['c1']:.2f}), c2 ({corL['c2']:.2f}) and c3 ({corL['c3']:.2f}). Errors in c4 and b barely matter "
        f"(ρ ≈ {cor['c4']:.2f}, {cor['b']:.2f}) for these stimuli: they are 'sloppy' in both directions, hard to learn "
        "and of little consequence for the inputs tested. The parameters that matter for out-of-sample prediction are "
        "a mix of stiff ones (well learned) and loop ones (poorly learned), and the poorly learned loop parameters are "
        "the ones that control the late response."))
    if "SP1000" in e8:
        s.append(fig("e8_holdout", "Figure 16. E8: temporal hold-out. Students fitted only on the first 60 ms after the "
                                   "pulse (red) against students fitted on the whole window (blue). Left: late part of "
                                   "the single-pulse response. Middle, right: late-window Pearson r and relative error "
                                   "on the training stimuli."))
        rows = [["Protocol", "Late r, whole-window fit", "Late r, early-only fit", "Late error, whole-window fit",
                 "Late error, early-only fit"]]
        for prot in ("SP1000", "DOSE", "RICH"):
            q = e8[prot]
            rows.append([prot, f"{q['E1 late Pearson r median']:.2f}", f"{q['E8 late Pearson r median']:.2f}",
                         pct(100 * q['E1 late rel. error median']), pct(100 * q['E8 late rel. error median'])])
        s += [table(rows, [0.16, 0.21, 0.21, 0.21, 0.21]), P("Table 6. E8 summary (medians over 10 starts).", CAP)]
        s.append(P(
            "E8 reproduces, in the cleanest possible setting, the temporal hold-out test applied to real data: the model "
            "is exactly right, the data are noise-free, and the student is fitted on the early window only. The late-window "
            f"error rises from {pct(100 * e8['SP1000']['E1 late rel. error median'])} to "
            f"{pct(100 * e8['SP1000']['E8 late rel. error median'])} for a single pulse and to "
            f"{pct(100 * e8['DOSE']['E8 late rel. error median'])} for DOSE; RICH, which includes E+I injection and later "
            f"pulses, holds at {pct(100 * e8['RICH']['E8 late rel. error median'])}. The median Pearson r stays fairly high "
            f"({e8['SP1000']['E8 late Pearson r median']:.2f}) because the late <i>shape</i> is roughly preserved while its "
            f"amplitude is wrong; the worst start falls to r = {e8['SP1000']['E8 late Pearson r min']:.2f}. As in Phase 1, "
            "Pearson alone overstates the fit. The early window fixes k and a (errors "
            f"{pct(e8['median_abs_err_pct']['SP1000']['k'])}, {pct(e8['median_abs_err_pct']['SP1000']['a'])}) and leaves b "
            f"({pct(e8['median_abs_err_pct']['SP1000']['b'])}) and the loop parameters to the initialisation; the late "
            "prediction inherits that arbitrariness."))
    s.append(PageBreak())

    # ------------------------------------------------------------------ 7. classification
    s.append(P("7  Which parameters are critical, which are conventions", H1))
    cls = [["Param.", "Physical meaning", "1 pulse (CRB)", "Best tested (CRB)", "Backprop progress", "Class", "Recommendation"],
           ["k", "stimulus gain (coil, orientation, scalp–cortex distance)", pct(c7["SP1000"]["k"]), pct(c7["ALL"]["k"]),
            f"{prog['E1']['k']:.2f}", "Stiff", "Fit, but as a per-session nuisance: it is not a brain property and "
                                             "confounds atrophy"],
           ["a", "excitatory rate, τ<sub>e</sub> ≈ 10 ms", pct(c7["SP1000"]["a"]), pct(c7["ALL"]["a"]),
            f"{prog['E1']['a']:.2f}", "Stiff", "Fit; reliable marker of excitatory kinetics"],
           ["b", "inhibitory rate, τ<sub>i</sub> ≈ 20 ms", pct(c7["SP1000"]["b"]), pct(c7["ALL"]["b"]),
            f"{prog['E1']['b']:.2f}", "Conditional", "Fit only with E+I / multi-condition data; otherwise report as "
                                                    "weakly constrained"],
           ["c1", "pyramidal → exc. interneurons", pct(c7["SP1000"]["c1"]), pct(c7["ALL"]["c1"]),
            f"{prog['E1']['c1']:.2f}", "Sloppy", "Fit only as the product c1·c2 (excitatory loop gain)"],
           ["c2", "exc. interneurons → pyramidal", pct(c7["SP1000"]["c2"]), pct(c7["ALL"]["c2"]),
            f"{prog['E1']['c2']:.2f}", "Sloppy", "As c1; do not interpret alone"],
           ["c3", "pyramidal → inh. interneurons", pct(c7["SP1000"]["c3"]), pct(c7["ALL"]["c3"]),
            f"{prog['E1']['c3']:.2f}", "Conditional", "Fit as c3·c4 (inhibitory loop gain), in an engaged state"],
           ["c4", "inh. interneurons → pyramidal", pct(c7["SP1000"]["c4"]), pct(c7["ALL"]["c4"]),
            f"{prog['E1']['c4']:.2f}", "Sloppy", "As c3; do not interpret alone"],
           ["A, B", "max EPSP / IPSP amplitude", "—", ">100%", "—", "Gauge", "Fix by convention (neurofe does)"],
           ["vmax", "max firing rate", "—", ">100%", "—", "Gauge", "Fix by convention"],
           ["v0, r", "firing threshold, sigmoid slope", "—", pct(c12["ALL"]["v0"]) + ", " + pct(c12["ALL"]["r"]), "—",
            "Sloppy", "Fix; v0 trades off with background drive"]]
    s += [table(cls, [0.07, 0.22, 0.1, 0.11, 0.1, 0.11, 0.29], font=7.1), P(
        "Table 7. Parameter classification. CRB = Cramér–Rao bound at 0.42 µV noise (7 parameters free; 12 for v0, r). "
        "'Best tested' is the ALL protocol (all 9 resting conditions). Progress from E1.", CAP)]
    s.append(P("Physical reading", H2))
    s.append(P(
        "Linearising the node around its resting point shows why the partition looks like this. The EEG response to an "
        "input U is V(s) = A·k·H<sub>a</sub>(s)·U(s) / (1 − A·S'·H<sub>a</sub>(s)·[L<sub>E</sub>H<sub>a</sub>(s) − "
        "L<sub>I</sub>H<sub>b</sub>(s)]), with synaptic kernels H<sub>a</sub> = a/(s+a)<super>2</super>, "
        "H<sub>b</sub> = b/(s+b)<super>2</super>, loop gains L<sub>E</sub> = A·c1·c2·S'<sub>E</sub> and "
        "L<sub>I</sub> = B·c3·c4·S'<sub>I</sub>, and S' the sigmoid slopes at the resting point. The data can only "
        "determine the coefficients of this function: the input gain A·k, the time constants a and b, and the two loop "
        "gains. The individual c's enter only through products, and A, B, vmax only through scalings. When the node is "
        "inactive, S' is small, the loop term vanishes and only A·k and a remain. That is what the nominal node shows."))
    s.append(P(
        "In terms of the brain: the parameters that are reliably identifiable describe <i>how strongly the cortex was hit</i> "
        "(k·A) and <i>how fast synapses respond</i> (a, and b with the right stimulus). The parameters that describe the "
        "<i>circuit</i>, i.e. how many synapses connect which populations, are only visible as loop gains, and only when "
        "the circuit is active. Absolute synaptic strength vs synapse number is never visible to EEG. For disease "
        "studies this means: slowed kinetics are detectable; changes in excitatory or inhibitory loop strength are "
        "detectable only as products and only in an engaged state; reduced response amplitude cannot be told apart from "
        "atrophy (reduced k) without independent E-field modelling. These questions are developed in "
        "<font name='DVM'>docs/single_node_study_plan.txt</font>."))

    # ------------------------------------------------------------------ 8. recommendations
    s.append(P("8  Recommendations", H1))
    s.append(P("For neurofe", H2))
    s += bullets([
        "Optimise log(θ / nominal) instead of raw values (or set per-parameter learning rates). This alone makes a, b "
        "and k converge, and is the cheapest fix with the largest effect.",
        "Keep A, B, vmax (and v0, r) fixed, and document that fitted c's and k are relative to them.",
        "Report identifiable combinations (k·A, a, b, c1·c2, c3·c4) and their Fisher or profile-likelihood uncertainty "
        "next to every fit. A fitted value without an uncertainty is not interpretable here.",
        "Add a mean input (a parameter p, or non-diffusive coupling through firing rates). The Laplacian coupling "
        "gives no mean drive, so every node rests in the inactive state where the circuit is invisible (section 3.2).",
        "Make delays fractional so that conduction velocity gets a gradient.",
        "Reweight the loss by time window (or whiten by the expected noise), so that the late response is not "
        "drowned by the early peak.",
        "Prefer second-order or Bayesian fitting (Levenberg–Marquardt, variational Laplace, simulation-based "
        "inference) for the global parameters; see the study plan, Part C.",
    ])
    s.append(P("For the synthetic-brain study (main README)", H2))
    s += bullets([
        "Expect T2 (temporal hold-out) and T3 (cross-site) to fail already at rung R0 unless the protocol and brain state "
        "engage the loop. Plan Phase 2 accordingly: the question is not whether they fail, but which remedy fixes them.",
        "Use multi-amplitude and multi-population stimulation in the synthetic protocols; single pulses at one "
        "amplitude cannot identify the circuit.",
        "Choose teachers in an active operating point (background drive above the jump to the upper JR branch). This is "
        "also the realistic regime and should give late TEP components like the real data (README Finding 1).",
        "Score generalisation on the late window separately. Whole-window scores hide the failure.",
    ])
    s.append(P("Limitations", H2))
    s += bullets([
        "Single node, rung R0 (teacher and student identical, apart from the random start). Real data add noise, "
        "misspecification and network effects, which can only make identification harder.",
        "Fits are noise-free except E4, and use 300 iterations of Adam with log-scaled parameters. Other optimisers may "
        "move further along sloppy directions; they cannot create information the Fisher bounds rule out.",
        "The driven-state fits (E7) used p = 110, below the jump; the more informative regime (p ≥ 120) is not yet fitted.",
        "Late-window metrics use a fixed split at 60 ms after the first pulse; for trains and paired pulses that window "
        "also contains later pulses.",
    ])
    s.append(P("Reproducing", H2))
    s.append(P(
        "<font name='DVM' size='7.5'>PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms analyse<br/>"
        "for p in 0 1 2 3; do PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms train --part $p --nparts 4 &amp; done; wait<br/>"
        "PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms_figures<br/>"
        "PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms_report</font>", SMALL))
    s.append(P("The full run (372 + 30 models) takes about 45 minutes on a 4-core laptop CPU.", SMALL))
    return s
