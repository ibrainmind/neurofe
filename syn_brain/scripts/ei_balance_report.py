"""E/I-balance bifurcation map report: identifiability of the excitatory/inhibitory loop
around the limit cycle, across quiet / E-dominant / balanced / I-dominant active states.

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
    active = regime == 1
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
              "and across E-dominant, balanced, and I-dominant operating points", CAP))
    s.append(Spacer(1, 6))

    s.append(P("Verdicts", H1))
    s.append(P("<b>Is there a clean, normal transition between quiet and active, and does it bend with "
              "E/I balance?</b> &rarr; Yes. Sweeping tonic drive p against an E/I-balance multiplier m "
              "(c1,c2 scaled by &radic;m; c3,c4 by 1/&radic;m; m=1 is nominal/balanced) gives a smooth, "
              "monotonic quiet-to-active boundary: E-dominant states (m&gt;1) need less drive to engage, "
              "I-dominant states need more. This matches the obvious biological intuition and is a useful "
              "sanity check that the model is behaving sensibly.", VERDICT_OK))
    s.append(P("<b>Where do pathological (sustained-oscillation) states live?</b> &rarr; In one genuine band: "
              "roughly balanced-to-slightly-I-leaning (m 0.5-1.5) at high drive (p above ~210-270, depending "
              "on m). An apparent second oscillating island at strongly E-dominant, low drive was CHECKED "
              "AND REJECTED: it was a slowly-damping spiral (critical slowing down), not a true limit cycle "
              "-- caught only by comparing two time-separated measurement windows, not one.", VERDICT_OK))
    s.append(P("<b>Can the system get 'stuck' in the limit cycle (hysteresis)?</b> &rarr; Not found, in the "
              "two cases directly tested. Ramping tonic drive back down from inside a confirmed, plateaued "
              "limit cycle (both for the balanced/nominal case and for an independently-drawn, intrinsically "
              "fragile teacher) exits the oscillation cleanly, landing on exactly the same quiet fixed point "
              "a cold start reaches at that same drive level. No bistability via drive alone, in this model, "
              "for these two cases.", VERDICT_OK))
    s.append(P("<b>Does richer stimulation fix identifiability around this whole map, not just one point?</b> "
              "&rarr; <b>Mostly yes, with one persistent exception.</b> A single pulse (SP1000) is "
              "catastrophically uninformative for E-dominant states (CRB median &gt;2500%). A richer protocol "
              "(paired-pulse + train + E/I-targeted injection, 'RICH') restores good bounds almost everywhere "
              "(median 10-19%). Backprop CONFIRMS this is real, not just theoretical, for c3&middot;c4 and for "
              "balanced/I-dominant c1&middot;c2 -- but <b>c1&middot;c2 in E-dominant states remains hard even "
              "with RICH</b> (39.8% backprop vs a 19% CRB bound: the optimizer still underperforms the theory "
              "there).", VERDICT_BAD))

    s.append(P("What this means for an E/I-balance biomarker", H2))
    s.append(P("The states that would matter most clinically -- 'too much E' -- are exactly the states "
              "where a conventional single-pulse TMS-EEG protocol is least informative, by a huge margin "
              "(1000s of percent CRB vs tens of percent for balanced/I-dominant states). A richer protocol "
              "closes most of this gap. The one piece that remains stubbornly hard is the excitatory loop "
              "gain (c1&middot;c2) specifically when excitation already dominates -- worth flagging explicitly "
              "to William/Zheng as a known, tested limit rather than an untested assumption.", BODY))
    s.append(PageBreak())

    s.append(P("1. The phase diagram", H1))
    s.append(P("Grid: tonic drive p = 0-280 (step 15), E/I balance m = 1/3 to 3 (15 log-spaced values). For "
              "each point, the raw (undemeaned) integrator state is read directly -- not through "
              "simulate()'s baseline-corrected output, which hides the absolute operating point -- after an "
              "8s settle, then classified using TWO successive 3s windows: a genuine limit cycle plateaus "
              "(window 2 amplitude close to window 1); a damped spiral decays a lot between them. This "
              "correction mattered: the first-pass single-window check wrongly flagged part of the quiet-to-"
              "active transition as oscillatory.", BODY))
    s.append(fig(OUT / "fig_phase_diagram_full.png",
                "Figure 1. Top row: dynamical regime, E/I dominance (raw E/(E+I), active points only), and "
                "limit-cycle amplitude. Bottom row: CRB on c1·c2 under a single pulse vs the RICH protocol "
                "(log color scale, same range both panels -- SP1000 is almost uniformly at the dark/bad end), "
                "and the median CRB trend across all active m for both loop-gain combinations. Stars mark the "
                "three points used for the backprop validation in section 2."))

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

    s.append(P("3. Is the limit cycle a trap?", H1))
    s.append(P("Tested directly rather than assumed. Case 1 (balanced/nominal, m=1): built up a confirmed, "
              "plateaued limit cycle at p=270 (amplitude 1.2 mV, stable over 15s), then stepped drive down "
              "through 240/210/180/150/100. The oscillation decayed smoothly and was fully gone by p=180, "
              "landing at E-I=6.486 mV, 6.280 mV at p=150 -- matching a cold start at p=150 (6.280 mV) to 4 "
              "decimal places. Case 2 (an independently-perturbed, intrinsically fragile teacher that "
              "oscillates even at modest drive): built up its limit cycle at p=150 (amplitude 6.1 mV), "
              "stepped down through 100 (still oscillating, 5.1 mV) to 60 -- oscillation vanished completely "
              "(E-I=0.257 mV), again matching a cold start at p=60 exactly. Neither case shows hysteresis: "
              "the backward (ramp-down) transition lands at essentially the same drive level as the forward "
              "one found by scanning from a cold start. <b>No evidence that reducing drive alone fails to "
              "resolve the abnormal oscillation</b>, for either case tested.", BODY))
    s.extend(bullets([
        "Caveat: only two points in parameter space were tested for hysteresis, both varying along the "
        "balance (m) axis or an independent full-parameter draw -- not an exhaustive search. A different "
        "route through parameter space (e.g. varying a, b, or v0 instead of the loop gains) could behave "
        "differently; this is evidence against a simple story, not a proof that no path to hysteresis "
        "exists.",
    ]))

    s.append(P("How this conclusion could be wrong", H1))
    s.extend(bullets([
        "<b>The m-balance axis is one specific, clean parameterization</b> (c1,c2 vs c3,c4 scaled "
        "oppositely, holding their geometric mean fixed) -- real disease-driven parameter shifts need not "
        "move along this exact direction. The T2/T3 teachers from the earlier robustness check (full "
        "independent ±30% draws) landed in a genuinely oscillatory regime this clean sweep did NOT "
        "originally find (a 'pure E-dominant, low-drive' island that turned out to be a damped-spiral "
        "artifact on THIS axis) -- meaning other parameter directions can reach genuine instability that "
        "this 2D slice doesn't show. The map is a useful cross-section, not the whole picture.",
        "<b>Grid resolution.</b> 15x19 points with linear/log spacing could miss thin spikes in the "
        "identifiability landscape or narrow true-limit-cycle slivers between grid points.",
        "<b>Backprop validation used only 10 starts at 3 points.</b> Consistent with the much larger E9-E12 "
        "study's conclusion (loop gains are genuinely harder to recover than CRB implies, especially "
        "c1·c2), which is reassuring, but the exact medians here could shift with more starts.",
    ]))

    s.append(P("Recommended next step", H1))
    s.append(P("Adopt RICH-style protocols (paired-pulse, train, E/I-targeted injection) as the default for "
              "any future firing-state fitting -- the single-pulse-only design used in E7/E9/E10 was "
              "leaving most of the achievable identifiability on the table, especially for E-dominant states. "
              "The remaining open problem is specific and now well-characterized: c1&middot;c2 in E-dominant "
              "states. If that combination matters clinically, it needs either more stimulus diversity still "
              "(not yet tried: higher-amplitude dosing, longer trains, at the firing state) or an "
              "uncertainty-aware method (Part C) rather than more point-estimate tuning.", BOX))

    s.append(P("Reproduce", H1))
    s.append(P("<font face='DVM'>Phase diagram + CRB: see scripts referenced in reports/DECISIONS.md "
              "(2026-11-22 entries). Backprop validation and hysteresis checks: standalone scripts, not yet "
              "folded into single_node_tms.py (logged as a follow-up if this becomes a permanent part of "
              "the study).</font>", SMALL))
    return s


def main():
    build(content(), REPORT, "The E/I-balance bifurcation map", "syn_brain · E/I-balance bifurcation map")


if __name__ == "__main__":
    main()
