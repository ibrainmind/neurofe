"""E/I-balance bifurcation map report: identifiability of the excitatory/inhibitory loop
around the limit cycle, across quiet / E-dominant / balanced / I-dominant active states.

Corrected 2026-11-24: the regime classification now uses the analytic fixed-point +
eigenvalue method from jr_parameters_explainer.py (equilibria()) instead of Euler-simulation,
after that independent analysis found (a) a bistable/hysteresis fold region the simulation-only
approach couldn't see (cold starts always land on the quiet branch), and (b) that the specific
point used for the original hysteresis test (balanced, p=270) was a numerical artifact of
neurofe's 0.1 ms Euler step, not a real limit cycle.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.ei_balance_report
Output: syn_brain/reports/ei_balance_bifurcation_map.pdf
"""

from __future__ import annotations

import json

import numpy as np
from reportlab.platypus import PageBreak, Spacer

from syn_brain.core.paths import SYN_ROOT
from syn_brain.core.report_style import BODY, BOX, BUL, CAP, H1, H2, P, SMALL, TITLE, VERDICT_BAD, VERDICT_OK, bullets, build, fig, table

REPORT = SYN_ROOT / "reports" / "ei_balance_bifurcation_map.pdf"
OUT = SYN_ROOT / "results" / "bifurcation_followup"


def region_table(raw, crb):
    regime = raw["regime"]; m = raw["m"]
    active = regime == 2   # CORRECTED scheme: 0=quiet,1=bistable,2=active-stable,3=Euler-artifact,4=genuine oscillation
    mgrid = np.broadcast_to(m[None, :], regime.shape)
    regions = {
        "balanced (0.77&lt;m&lt;1.3)": active & (np.abs(np.log(mgrid)) < np.log(1.3)),
        "E-dominant (m&gt;=1.3, too much E)": active & (mgrid >= 1.3),
        "I-dominant (m&lt;=0.77, too much I)": active & (mgrid <= 1 / 1.3),
    }
    rows = [["Region", "n", "c1·c2 CRB, SP1000", "c1·c2 CRB, RICH", "c3·c4 CRB, SP1000", "c3·c4 CRB, RICH"]]
    for name, mask in regions.items():
        vals = []
        for combo in ("c1c2", "c3c4"):
            for prot in ("SP1000", "RICH"):
                se = crb[f"se_{combo}_{prot}"][mask] * 100
                vals.append(np.nanmedian(se))
        rows.append([name, str(mask.sum()), f"{vals[0]:.0f}%", f"{vals[1]:.0f}%", f"{vals[2]:.0f}%", f"{vals[3]:.0f}%"])
    return rows


def content():
    raw = np.load(OUT / "phase_diagram_raw.npz")
    crb = np.load(OUT / "phase_diagram_crb.npz")
    bp = json.load(open(OUT / "ei_backprop_results.json"))

    s = []
    s.append(P("The E/I-balance bifurcation map", TITLE))
    s.append(P("syn_brain single-node study -- identifiability around quiet / active / limit-cycle states, "
              "and across E-dominant, balanced, and I-dominant operating points. <b>Corrected 2026-11-24</b> "
              "to match the analytic equation-level analysis in "
              "<font face='DVM'>reports/jr_parameters_explainer.pdf</font> -- see the correction note below.",
              CAP))
    s.append(Spacer(1, 6))

    s.append(P("Correction note (2026-11-24)", H2))
    s.append(P("The original version of this report classified each grid point's dynamical regime by "
              "simulating it (neurofe's Euler integrator, 0.1 ms step) and checking whether the response "
              "kept oscillating. An independent equation-level analysis "
              "(<font face='DVM'>jr_parameters_explainer.pdf</font>: exact fixed points, solved analytically, "
              "classified by the eigenvalues of the linearised equations -- immune to integration-step error "
              "by construction) found two things the simulation-based map got wrong, both now fixed here: "
              "(1) an entire <b>bistable (fold) region</b> was invisible to a cold-start-only simulation -- "
              "every cold start lands on the quiet branch even where a stable active branch also exists, so "
              "that whole region was silently mislabelled \"quiet\"; (2) the specific point used for the "
              "hysteresis test in section 3 (balanced, p=270) is a <b>numerical artifact</b> -- the true "
              "equations are stable there (just very weakly damped), but neurofe's Euler step adds enough "
              "spurious energy to make the simulation ring forever. Re-derived the whole phase diagram "
              "(Figure 1) and the \"is it a trap\" section (3) from the analytic method; the CRB/backprop "
              "findings (section 2) did not need correction -- all three validation points were independently "
              "re-checked and are genuinely stable, undamped active states.", BOX))

    s.append(P("Verdicts", H1))
    s.append(P("<b>Is there a clean, normal transition between quiet and active, and does it bend with "
              "E/I balance?</b> &rarr; Yes, with one addition: it is not a single line but a <b>band</b> -- "
              "over a range of p the quiet and active states coexist (a fold), and which one the node is in "
              "depends on its history. E-dominant states (m&gt;1) need less drive to engage and the fold "
              "sits lower; I-dominant states need more.", VERDICT_OK))
    s.append(P("<b>Where do pathological (sustained-oscillation) states live?</b> &rarr; <b>Revised.</b> A "
              "narrower, lower genuine band than first reported: confirmed analytically (no stable state "
              "exists at all) mainly for balanced-to-I-leaning m at moderate-high p. The part of the original "
              "band at m&asymp;1 and high p (p&gt;250) is <b>not</b> a real limit cycle -- it is neurofe's "
              "Euler step turning a very weakly damped (but genuinely stable) ringing into an apparently "
              "sustained one. This matters: it was exactly the point used below for the hysteresis test.",
              VERDICT_BAD))
    s.append(P("<b>Can the system get 'stuck' (hysteresis)?</b> &rarr; <b>Yes -- revised from \"not found\".</b> "
              "The original test ramped down from what turned out to be the Euler artifact above, so it "
              "never actually tested a real sustained oscillation, and it never crossed the fold region "
              "either. The equation-level analysis found genuine hysteresis exactly at the quiet/active fold: "
              "for the balanced case the two states coexist for p in [36.5, 114.6), confirmed by direct "
              "simulation (settle active at p=150, step down: stays active at p=110/80/50 while a cold start "
              "at those same p is quiet). E-dominant columns are bistable already from p=0.", VERDICT_BAD))
    s.append(P("<b>Does richer stimulation fix identifiability around this whole map, not just one point?</b> "
              "&rarr; <b>Mostly yes, with one persistent exception -- and the corrected map makes the case "
              "stronger, not weaker.</b> With the fold-hidden active region now correctly included, the "
              "E-dominant active-stable sample is much larger (45 to 99 of 285 grid points) and the single-"
              "pulse failure is even more extreme (CRB median in the hundreds of thousands of percent -- "
              "effectively zero information). A richer protocol ('RICH': paired-pulse + train + E/I-targeted "
              "injection) still restores good bounds almost everywhere (median 10-24%). Backprop CONFIRMS "
              "this is real for c3&middot;c4 and for balanced/I-dominant c1&middot;c2 -- but <b>c1&middot;c2 "
              "in E-dominant states remains hard even with RICH</b> (39.8% backprop vs a 19-24% CRB bound).",
              VERDICT_BAD))

    s.append(P("What this means for an E/I-balance biomarker", H2))
    s.append(P("The states that would matter most clinically -- 'too much E' -- are exactly the states "
              "where a conventional single-pulse TMS-EEG protocol carries essentially no information at all "
              "about the loop gains. A richer protocol closes most of this gap. The one piece that remains "
              "stubbornly hard is the excitatory loop gain (c1&middot;c2) specifically when excitation "
              "already dominates -- worth flagging explicitly to William/Zheng as a known, tested limit "
              "rather than an untested assumption. Separately: a real patient could be sitting in the newly-"
              "found bistable band, where the SAME background drive gives a quiet or an active response "
              "depending on recent history -- a confound for any protocol that assumes a single, repeatable "
              "operating point per subject.", BODY))
    s.append(PageBreak())

    s.append(P("1. The phase diagram", H1))
    s.append(P("Grid: tonic drive p = 0-280 (step 15), E/I balance m = 1/3 to 3 (15 log-spaced values; "
              "c1,c2 scaled by &radic;m, c3,c4 by 1/&radic;m, m=1 nominal/balanced). Each point is now "
              "classified by solving for every fixed point of the continuous equations exactly (root-finding "
              "on a fine grid) and checking the sign of the leading eigenvalue of the linearised dynamics at "
              "each one (<font face='DVM'>jr_parameters_explainer.py: equilibria()</font>) -- this cannot be "
              "fooled by the integration step, unlike the simulate-and-watch approach used in the first "
              "version of this report (see correction note above).", BODY))
    s.append(fig(OUT / "fig_phase_diagram_full.png",
                "Figure 1. Top-left: corrected regime map -- quiet, bistable (fold, new), active-stable, "
                "active but only oscillating because of neurofe's Euler step (artifact, now isolated to a "
                "thin sliver near m=1, p>250), and genuine sustained oscillation. Top-middle: E/I dominance, "
                "active-stable points only. Top-right: the Euler integrator's damping bias (always positive -- "
                "it only ever pushes toward instability, never away from it). Bottom row: CRB on c1·c2, "
                "single pulse vs RICH, and the median trend for both loop-gain combinations across the "
                "(now larger and correct) active-stable region. Stars mark the three backprop validation "
                "points from section 2, independently re-confirmed as genuinely stable."))

    s.append(P("2. Backprop validation: does the richer protocol actually work?", H1))
    s.append(P("CRB is a local, best-case bound -- this whole study has repeatedly found the optimizer can "
              "fail to reach it. Ran real 10-start Adam recovery (300 iterations, log-scaled, noise-free) at "
              "three representative active-and-stable points.", BODY))
    rows = [["Point (m, p)", "Protocol", "c1·c2 median (backprop)", "c3·c4 median (backprop)", "Passes 20% gate?"]]
    labels = {"balanced": "balanced (m=1.0, p=150)", "E_dominant": "E-dominant (m=1.87, p=105)",
             "I_dominant": "I-dominant (m=0.53, p=240)"}
    for name, label in labels.items():
        for prot in ("SP1000", "RICH"):
            r = bp[name][prot]
            ok = "c3·c4 yes" if r["c3c4_med"] < 20 else "c3·c4 no"
            ok += ", c1·c2 yes" if r["c1c2_med"] < 20 else ", c1·c2 no"
            rows.append([label if prot == "SP1000" else "", prot, f"{r['c1c2_med']:.1f}%", f"{r['c3c4_med']:.1f}%", ok])
    s.append(table(rows, [0.27, 0.12, 0.22, 0.22, 0.17]))
    s.append(P("Table 1. RICH consistently beats SP1000 (5 of 6 comparisons strictly better, one tied). "
              "c3·c4 passes the 20% gate with RICH at 2 of 3 points (misses narrowly at E-dominant, 20.5%). "
              "c1·c2 passes only at balanced (19.1%, right at the edge) -- it stays the hardest parameter "
              "combination everywhere, worst in the E-dominant state that matters most clinically.", CAP))

    s.append(table(region_table(raw, crb), [0.34, 0.06, 0.15, 0.15, 0.15, 0.15]))
    s.append(P("Table 2. CRB (theoretical best case) by region, median over all active grid points in that "
              "region -- the broader pattern Table 1's three spot-checks sit inside. E-dominant is the "
              "outlier: SP1000 is off the scale (1000s of percent); RICH brings it back in line with the "
              "other regions.", CAP))

    s.append(P("3. Is the limit cycle a trap? -- corrected", H1))
    s.append(P("<b>The original test in this section used the wrong point.</b> It built up what looked like a "
              "confirmed, plateaued limit cycle at (balanced, p=270) and showed that ramping drive down exits "
              "it cleanly. We now know that point is the Euler-step artifact described above: the true "
              "equations are stable there (damping Re(&lambda;) = -0.096 /s, barely negative), so there was no "
              "real limit cycle to escape from in the first place. That specific finding (\"no hysteresis\") "
              "is therefore withdrawn -- it was a true statement about the simulator's behaviour at that "
              "point, not a true statement about the model's dynamics.", VERDICT_BAD))
    s.append(P("<b>The real answer, from the equation-level analysis:</b> hysteresis is real, at the "
              "quiet/active fold, not at a limit-cycle boundary. For the balanced column (m=1) the quiet and "
              "active states coexist for p in [36.5, 114.6). Direct simulation confirms it: settle into the "
              "active state at p=150, then step the drive down -- the column STAYS active at p=110, 80 and 50 "
              "(E-I = 5.98, 5.70, 5.33 mV) where a cold start at those same drive levels is quiet (2.09, 0.79, "
              "-0.23 mV), and only drops to quiet between p=50 and p=40. E-dominant columns are bistable "
              "already at p=0 -- there is no drive level at which an E-dominant column is forced to be quiet.",
              BODY))
    s.append(P("The second case tested in the original version (an independently-perturbed, intrinsically "
              "fragile teacher, 'T2', that oscillates even at modest drive and was confirmed by a 30-second "
              "simulation to have a genuinely non-decaying ~6 mV oscillation) was NOT part of the clean "
              "(p, m) sweep and has not been independently re-verified by the analytic method -- its earlier "
              "\"exits cleanly, no hysteresis\" finding is left as-is but should be read as unconfirmed by "
              "this stronger method, not as contradicting the fold-hysteresis result above (they concern "
              "different mechanisms: T2's is plausibly a genuine Hopf limit cycle; the fold is a separate, "
              "coexisting-fixed-points phenomenon).", BODY))
    s.extend(bullets([
        "Practical implication: a real subject could sit in the bistable band and show either a quiet or an "
        "active response to the identical background state, depending on what happened just before the "
        "measurement -- a genuine confound for any protocol that assumes one repeatable operating point per "
        "subject, separate from (and in addition to) the identifiability problems documented elsewhere in "
        "this report.",
    ]))

    s.append(P("How this conclusion could be wrong", H1))
    s.extend(bullets([
        "<b>The m-balance axis is one specific, clean parameterization</b> (c1,c2 vs c3,c4 scaled "
        "oppositely, holding their geometric mean fixed) -- real disease-driven parameter shifts need not "
        "move along this exact direction. The T2/T3 teachers from the earlier robustness check (full "
        "independent ±30% draws) land in genuinely oscillatory regimes this clean sweep's analytic method "
        "has not been extended to check (equilibria() as written only supports the m-axis, not arbitrary "
        "a/b/c1-c4 perturbations) -- other parameter directions can reach dynamics this 2D slice doesn't show. "
        "The map is a useful cross-section, not the whole picture.",
        "<b>Grid resolution.</b> 15x19 points with linear/log spacing could miss thin bistable slivers or "
        "narrow true-limit-cycle regions between grid points -- the analytic method removes integration-step "
        "error but not grid coarseness.",
        "<b>Backprop validation used only 10 starts at 3 points.</b> Consistent with the much larger E9-E12 "
        "study's conclusion (loop gains are genuinely harder to recover than CRB implies, especially "
        "c1·c2), which is reassuring, but the exact medians here could shift with more starts. All three "
        "points were independently re-confirmed as genuinely stable (not artifacts) by the corrected method.",
    ]))

    s.append(P("Recommended next step", H1))
    s.append(P("Adopt RICH-style protocols (paired-pulse, train, E/I-targeted injection) as the default for "
              "any future firing-state fitting -- the single-pulse-only design used in E7/E9/E10 was "
              "leaving most of the achievable identifiability on the table, especially for E-dominant states, "
              "now known to be an even larger effect than first measured. The remaining open problem is "
              "specific and well-characterized: c1&middot;c2 in E-dominant states. If that combination "
              "matters clinically, it needs either more stimulus diversity still (not yet tried: "
              "higher-amplitude dosing, longer trains) or an uncertainty-aware method (Part C) rather than "
              "more point-estimate tuning -- section 6 of reports/single_node_conclusions.pdf has since built "
              "and validated a first such method (Laplace approximation). Newly added: extend "
              "<font face='DVM'>equilibria()</font> to arbitrary (not just m-axis) parameter perturbations, "
              "so the T2/T3-style independently-drawn teachers can be placed on this same analytic map rather "
              "than relying on simulation alone.", BOX))

    s.append(P("Reproduce", H1))
    s.append(P("<font face='DVM'>Phase diagram (corrected): .venv/bin/python -c \"import sys; "
              "sys.path.insert(0,'.'); from syn_brain.scripts.jr_parameters_explainer import equilibria\" "
              "-- see scripts/jr_parameters_explainer.py for the full stability-map code.<br/>"
              "CRB: see scripts referenced in reports/DECISIONS.md (2026-11-22 entries).<br/>"
              "Backprop validation and hysteresis checks: standalone scripts, not yet folded into "
              "single_node_tms.py (logged as a follow-up if this becomes a permanent part of the "
              "study).</font>", SMALL))
    return s


def main():
    build(content(), REPORT, "The E/I-balance bifurcation map", "syn_brain · E/I-balance bifurcation map")


if __name__ == "__main__":
    main()
