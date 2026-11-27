"""Part N2.5 diagnostic: does the U_BOUND clip saturate during stimulation for high-g teachers?

No new training -- replays each N2-gate teacher's own TRUE parameters (the actual physical
operating point, not a fitted guess) through an instrumented copy of core/diffsim.py's step_ms,
logging rE = noise + g*coupling + c2*E_s (pre-clip) alongside the U_BOUND=500 clip, for every
node and timestep of the real training protocol (4 sites x 3 amplitudes). If |rE| routinely
exceeds ~U_BOUND for the high-true-g teachers but not the low-true-g ones, that confirms the
hypothesis in docs/network_study_plan.txt Part N2.5: the clip is silently flattening d(rE_bd)/dg
near the truth whenever truth-g is high, which would explain the "low-g attractor" pattern found
in the N2-gate fits without needing any new training to check.

Usage:
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.diagnose_g_clip
"""

from __future__ import annotations

import json

import numpy as np
import torch

from syn_brain.core.paths import RESULTS_DIR, get_logger
from syn_brain.core.anatomy import load_anatomy, pick_sites, site_weights
from syn_brain.core import diffsim as ds

U_BOUND = ds.U_BOUND


def replay(theta: dict, net: ds.Net, proto: ds.Protocol, warmup_ms=1000, T=400):
    """Minimal re-implementation of diffsim.run()'s inner loop (Laplacian only), instrumented
    to return max|rE| (pre-clip) and max|g*coupling| per timestep, across all nodes/conditions."""
    A = torch.tensor(ds.FIXED_RAW["A"], dtype=torch.float64)
    Bg = torch.tensor(ds.FIXED_RAW["B"], dtype=torch.float64)
    vmax, v0, r = ds.FIXED_RAW["vmax"], ds.FIXED_RAW["v0"], ds.FIXED_RAW["r"]
    a, b, g = theta["a"], theta["b"], theta["g"]
    c1, c2, c3, c4 = theta["c1"], theta["c2"], theta["c3"], theta["c4"]
    mu, k = theta["mu"], theta["k"]
    dt = 1e-4
    hidden = 10

    w_n = net.w_n
    dg = -torch.diag(w_n.sum(1))
    d = net.dist / mu
    i0 = d.detach().floor().long().clamp(max=ds.DELAYS_MAX - 2)
    frac = d - i0.to(torch.float64)
    S, N = proto.n_cond, net.n
    i0b = i0.unsqueeze(0).expand(S, N, N)
    i1b = (i0 + 1).unsqueeze(0).expand(S, N, N)

    ki = torch.tensor(proto.ki, dtype=torch.float64).unsqueeze(-1)
    amp = torch.tensor(proto.amp, dtype=torch.float64).view(S, 1, 1)

    x = torch.zeros((6, S, N, 1), dtype=torch.float64)
    hE = torch.zeros((S, N, ds.DELAYS_MAX), dtype=torch.float64)
    P, E, I, Pv, Ev, Iv = x

    def delayed(hE):
        Ed0 = torch.gather(hE, 2, i0b).transpose(1, 2)
        Ed1 = torch.gather(hE, 2, i1b).transpose(1, 2)
        return Ed0 + frac.T * (Ed1 - Ed0)

    max_abs_rE = 0.0
    max_abs_gcoupling = 0.0

    def step(P, E, I, Pv, Ev, Iv, hE, u_amp, record):
        nonlocal max_abs_rE, max_abs_gcoupling
        Ed = delayed(hE)
        coupling = (w_n * Ed).sum(2, keepdim=True) + dg @ E
        for _ in range(hidden):
            P_s = vmax / (1 + torch.exp(r * (v0 - (E - I))))
            E_s = vmax / (1 + torch.exp(r * (v0 - c1 * P)))
            I_s = vmax / (1 + torch.exp(r * (v0 - c3 * P)))
            gc = g * coupling
            rE = gc + c2 * E_s
            if record:
                max_abs_rE = max(max_abs_rE, float(rE.abs().max()))
                max_abs_gcoupling = max(max_abs_gcoupling, float(gc.abs().max()))
            rP_bd = U_BOUND * torch.tanh(P_s / U_BOUND)
            rE_bd = U_BOUND * torch.tanh(rE / U_BOUND)
            if u_amp is not None:
                rE_bd = rE_bd + k * ki * u_amp
            rI_bd = U_BOUND * torch.tanh(c4 * I_s / U_BOUND)
            P, E, I, Pv, Ev, Iv = (P + dt * Pv, E + dt * Ev, I + dt * Iv,
                                   Pv + dt * (A * a * rP_bd - 2 * a * Pv - a ** 2 * P),
                                   Ev + dt * (A * a * rE_bd - 2 * a * Ev - a ** 2 * E),
                                   Iv + dt * (Bg * b * rI_bd - 2 * b * Iv - b ** 2 * I))
        hE = torch.cat([E, hE[:, :, :-1]], dim=2)
        return P, E, I, Pv, Ev, Iv, hE

    with torch.no_grad():
        for _ in range(warmup_ms):
            P, E, I, Pv, Ev, Iv, hE = step(P, E, I, Pv, Ev, Iv, hE, None, record=False)
        p0, p1 = proto.pulse
        for t in range(T):
            u_amp = amp if p0 <= t < p1 else None
            # record from the pulse onward through the recorded window (where fitting gradients
            # actually act), not during the no-input warm-up (zero coupling there, uninformative).
            P, E, I, Pv, Ev, Iv, hE = step(P, E, I, Pv, Ev, Iv, hE, u_amp, record=(t >= p0))
    return max_abs_rE, max_abs_gcoupling


def main():
    log = get_logger("diagnose_g_clip")
    anat = load_anatomy(10)
    net = ds.Net.from_anatomy(anat)
    sites = pick_sites(anat, 8, seed=0)
    ki_all = np.stack([site_weights(anat, int(s))[:, 0] for s in sites])
    amps = (500.0, 1000.0, 2000.0)
    # training design used by the N2 gate: 4 of 8 sites x all 3 amplitudes
    proto = ds.Protocol(ki=np.repeat(ki_all[:4], len(amps), axis=0), amp=np.tile(np.array(amps), 4))

    in_dir = RESULTS_DIR / "recovery_ensemble"
    files = {0: "init13", 1: "init12", 2: "init11", 3: "init12", 4: "init13", 5: "init11",
             6: "init10", 7: "init13"}
    rows = []
    for t, initseed in files.items():
        j = json.load(open(in_dir / f"ensemble_n10_s4_a500-1000-2000_{initseed}_kscale_N2full_ens_t{t}.json"))
        truth = j["truth"]
        th = {n: torch.tensor(float(v), dtype=torch.float64) for n, v in truth.items()}
        max_rE, max_gc = replay(th, net, proto)
        rows.append((t, truth["g"], max_rE, max_gc))
        log.info("teacher %d  true_g=%8.1f  max|rE|=%8.1f  max|g*coupling|=%8.1f  %s",
                  t, truth["g"], max_rE, max_gc,
                  "SATURATING" if max_rE > 0.9 * U_BOUND else "clear")

    log.info("--- sorted by true g ---")
    for t, g, max_rE, max_gc in sorted(rows, key=lambda r: r[1]):
        log.info("teacher %d  true_g=%8.1f  max|rE|=%8.1f (U_BOUND=%.0f)  max|g*coupling|=%8.1f",
                  t, g, max_rE, U_BOUND, max_gc)


if __name__ == "__main__":
    main()
