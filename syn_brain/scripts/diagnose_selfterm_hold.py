"""Part N2.5 audit: is the Laplacian 'no bounded equilibrium at high g' real, or a discretisation artifact?

core/diffsim.py computes the whole Laplacian coupling, INCLUDING the instantaneous self term
dg @ E = -d_i * E_i, once per 1 ms sample and holds it for the 10 Euler sub-steps. neurofe
(models/jansen_rit.py, line ~417) and core/simulator.py recompute dg @ E at EVERY 0.1 ms sub-step;
only the delayed neighbour sum is held per ms. The self term is a high-gain negative loop
(DC gain g*d_i*A/a ~ 20-40), so a 0.5-1 ms hold can destabilise it at ~50-100 Hz.

This script replays the 8 N2-gate teachers (draw_teacher(10, seed=t, spread=0.5), true
parameters, training protocol: 4 sites x 500/1000/2000 with the clamped amp-by-k scale) under:
  held     = diffsim semantics (self term held per ms)
  substep  = neurofe / simulator.py semantics (self term updated every sub-step)
each with U_BOUND = 500 (as trained) and U_BOUND = 1e6 (effectively no clip), and reports the
tail (150-400 ms) mean |g*coupling| / U_BOUND, late-window rms of E-I and the peak.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.diagnose_selfterm_hold
Writes syn_brain/results/n2_5/selfterm_hold.json
"""
import json, sys
from syn_brain.core.paths import RESULTS_DIR
import numpy as np, torch
from syn_brain.core.anatomy import load_anatomy, pick_sites, site_weights
from syn_brain.core import diffsim as ds
from syn_brain.core.params import draw_teacher
NOMINAL_EFF = ds.NOMINAL_EFF

anat = load_anatomy(10); net = ds.Net.from_anatomy(anat)
sites = pick_sites(anat, 8, seed=0)
ki_all = np.stack([site_weights(anat, int(s))[:, 0] for s in sites])
amps = np.array((500.0, 1000.0, 2000.0))
A = ds.FIXED_RAW["A"]; Bg = ds.FIXED_RAW["B"]; vmax, v0, r = ds.FIXED_RAW["vmax"], ds.FIXED_RAW["v0"], ds.FIXED_RAW["r"]
dt, hidden = 1e-4, 10
w_n = net.w_n; deg = w_n.sum(1); dg = -torch.diag(deg)

def run(th, scale, mode, U, warm=1000, T=400):
    ki = torch.tensor(np.repeat(ki_all[:4], 3, axis=0), dtype=torch.float64).unsqueeze(-1)
    amp = torch.tensor(np.tile(amps, 4) * scale, dtype=torch.float64).view(-1, 1, 1)
    S, N = ki.shape[0], net.n
    a, b, g, c1, c2, c3, c4, mu, k = (float(th[n]) for n in ("a","b","g","c1","c2","c3","c4","mu","k"))
    d = net.dist / mu; i0 = d.floor().long().clamp(max=ds.DELAYS_MAX - 2); frac = d - i0.double()
    i0b = i0.unsqueeze(0).expand(S, N, N); i1b = (i0 + 1).unsqueeze(0).expand(S, N, N)
    P = E = I = Pv = Ev = Iv = torch.zeros((S, N, 1), dtype=torch.float64)
    hE = torch.zeros((S, N, ds.DELAYS_MAX), dtype=torch.float64)
    tail_gc, maxv, out = [], 0.0, []
    for t in range(warm + T):
        Ed0 = torch.gather(hE, 2, i0b).transpose(1, 2); Ed1 = torch.gather(hE, 2, i1b).transpose(1, 2)
        Ed = Ed0 + frac.T * (Ed1 - Ed0)
        LEd = (w_n * Ed).sum(2, keepdim=True)
        held_self = dg @ E
        u = amp if (warm + 0 <= t < warm + 10) else None
        for _ in range(hidden):
            self_t = held_self if mode == "held" else dg @ E
            gc = g * (LEd + self_t)
            P_s = vmax / (1 + torch.exp(r * (v0 - (E - I)))); E_s = vmax / (1 + torch.exp(r * (v0 - c1 * P)))
            I_s = vmax / (1 + torch.exp(r * (v0 - c3 * P)))
            rE = gc + c2 * E_s
            rP = U * torch.tanh(P_s / U); rEb = U * torch.tanh(rE / U); rI = U * torch.tanh(c4 * I_s / U)
            if u is not None: rEb = rEb + k * ki * u
            P, E, I, Pv, Ev, Iv = (P + dt * Pv, E + dt * Ev, I + dt * Iv,
                                   Pv + dt * (A * a * rP - 2 * a * Pv - a * a * P),
                                   Ev + dt * (A * a * rEb - 2 * a * Ev - a * a * E),
                                   Iv + dt * (Bg * b * rI - 2 * b * Iv - b * b * I))
        hE = torch.cat([E, hE[:, :, :-1]], dim=2)
        if t >= warm:
            v = (E - I)[..., 0]; out.append(v.numpy().copy())
            if t >= warm + 150: tail_gc.append(float(gc.abs().mean()))
    if not torch.isfinite(E).all(): return dict(finite=False)
    y = np.array(out)                                   # (T, S, N)
    late = y[300:] - y[300:].mean(0)
    return dict(finite=True, tail_gc_over_U=float(np.mean(tail_gc)) / U, late_rms=float(np.sqrt((late ** 2).mean())),
                peak=float(np.abs(y).max()))

rows = []
for t in range(8):
    th = draw_teacher(10, seed=t, spread=0.5).effective
    scale = max(1.0, NOMINAL_EFF["k"] / float(th["k"]))
    res = {}
    for mode in ("held", "substep"):
        for U in (500.0, 1e6):
            res[f"{mode}_U{int(U)}"] = run(th, scale, mode, U)
    gdmax = float(th["g"]) * float(deg.max())
    rows.append(dict(t=t, g=float(th["g"]), g_dmax=gdmax, **res))
    print(t, round(float(th["g"])), {k: (v if not v.get("finite") else {kk: round(vv, 4) for kk, vv in v.items() if kk != "finite"}) for k, v in res.items()}, flush=True)
out = RESULTS_DIR / "n2_5"; out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "selfterm_hold.json", "w"), indent=1)
