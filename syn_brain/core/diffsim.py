"""Differentiable Jansen-Rit network for identifiability analysis and clean fitting.

Same equations as core/simulator.py (and therefore neurofe's JansenRit.forward), with
four deliberate differences that the Phase 2 ablation can switch on and off:

1. Parameters are given as *effective* values (the ones entering the equations),
   learned as log-scaled offsets z: theta = nominal * exp(z). Every parameter then
   moves on the same O(1) scale, whatever its magnitude, and there is no relu floor.
2. Delays can be fractional (linear interpolation between the two neighbouring
   1 ms samples of the delay buffer), so conduction velocity mu gets a gradient.
   `integer_delays=True` reproduces neurofe's int cast exactly.
3. Noise is optional and reproducible: `noise_seed` fixes the random stream, so
   two runs with different parameters see the same noise (common random numbers).
   That makes finite differences and trial-averaged gradients smooth.
4. A leading batch axis holds (condition x trial): each condition can have its own
   stimulation mask `ki` and pulse amplitude.

Everything runs in float64 by default: finite-difference Jacobians need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .params import FIXED_RAW, Y0, nominal_effective

U_BOUND = 500.0
S2O_COEF = 1e-4
DELAYS_MAX = 500

PARAM_NAMES = ("a", "b", "g", "c1", "c2", "c3", "c4", "std_in", "mu", "k")
NOMINAL_EFF = nominal_effective()


def theta_from_z(z: torch.Tensor, names=PARAM_NAMES, base: dict | None = None) -> dict:
    """theta[name] = base[name] * exp(z[i]); base defaults to the nominal effective values."""
    base = base or NOMINAL_EFF
    return {n: base[n] * torch.exp(z[i]) for i, n in enumerate(names)}


def z_from_theta(theta: dict, names=PARAM_NAMES, base: dict | None = None) -> torch.Tensor:
    base = base or NOMINAL_EFF
    return torch.tensor([np.log(float(theta[n]) / base[n]) for n in names], dtype=torch.float64)


@dataclass
class Net:
    """Anatomy as tensors, ready for the simulator."""
    w_n: torch.Tensor       # (N, N) normalised coupling
    dist: torch.Tensor      # (N, N) mm
    lm_dm: torch.Tensor     # (C, N) de-meaned leadfield
    n: int
    c: int

    @classmethod
    def from_anatomy(cls, anat, w_bb: np.ndarray | None = None, dtype=torch.float64):
        sc = torch.tensor(anat.sc, dtype=dtype)
        w = sc if w_bb is None else torch.exp(torch.tensor(w_bb, dtype=dtype)) * sc
        w_sym = 0.5 * (w + w.T)
        w_n = torch.log1p(w_sym) / torch.linalg.norm(torch.log1p(w_sym))
        lm = torch.tensor(anat.lm, dtype=dtype)
        c = lm.shape[0]
        lm_dm = lm - lm.mean(0, keepdim=True)
        return cls(w_n=w_n, dist=torch.tensor(anat.dist, dtype=dtype), lm_dm=lm_dm,
                   n=anat.n_nodes, c=c)


@dataclass
class Protocol:
    """What the experimenter does: where and how hard to stimulate, and what to record.

    ki    : (S, N) stimulation masks, one row per condition
    amp   : (S,) pulse amplitude per condition
    pulse : (start, stop) sample indices of the pulse in the recorded window
    T     : recorded samples (1 kHz)
    warmup_ms : no-input settling time before the window
    """
    ki: np.ndarray
    amp: np.ndarray
    pulse: tuple = (110, 120)
    T: int = 400
    warmup_ms: int = 1000

    @property
    def n_cond(self) -> int:
        return self.ki.shape[0]


def run(theta: dict, net: Net, proto: Protocol, n_trials: int = 1, noise_seed: int | None = None,
        noise_scale: float = 1.0, integer_delays: bool = False, step_size: float = 1e-4,
        hidden: int = 10, grad_warmup_ms: int | None = None, state0=None, return_state=False,
        dtype=torch.float64):
    """Simulate every condition (x n_trials) and return EEG (S, n_trials, C, T).

    noise_scale = 0 gives a deterministic run (std_in then has no effect).
    grad_warmup_ms: if set, the first (warmup_ms - grad_warmup_ms) of the warm-up run
    without autograd and only the last grad_warmup_ms are differentiated. Starting from
    a settled state, that is enough to carry the parameter dependence of the baseline.
    state0: optional (hx, hE) to start from instead of the fixed small initial state.
    """
    S, R, N, C = proto.n_cond, n_trials, net.n, net.c
    B = S * R
    gen = None
    if noise_scale > 0:
        gen = torch.Generator().manual_seed(0 if noise_seed is None else noise_seed)

    A = torch.tensor(FIXED_RAW["A"], dtype=dtype)
    Bg = torch.tensor(FIXED_RAW["B"], dtype=dtype)
    vmax, v0, r = FIXED_RAW["vmax"], FIXED_RAW["v0"], FIXED_RAW["r"]
    cy0 = FIXED_RAW["cy0"]
    a, b, g = theta["a"], theta["b"], theta["g"]
    c1, c2, c3, c4 = theta["c1"], theta["c2"], theta["c3"], theta["c4"]
    std_in = theta["std_in"] * noise_scale
    mu, k = theta["mu"], theta["k"]
    dt = step_size

    w_n = net.w_n
    dg = -torch.diag(w_n.sum(1))
    d = net.dist / mu                                    # (N, N) delay in ms
    if integer_delays:
        i0 = d.detach().floor().long()
        frac = None
    else:
        i0 = d.detach().floor().long().clamp(max=DELAYS_MAX - 2)
        frac = d - i0.to(dtype)                          # carries the gradient wrt mu
    i0b = i0.unsqueeze(0).expand(B, N, N)
    i1b = (i0 + 1).unsqueeze(0).expand(B, N, N)

    ki = torch.tensor(np.repeat(proto.ki, R, axis=0), dtype=dtype).unsqueeze(-1)       # (B, N, 1)
    amp = torch.tensor(np.repeat(proto.amp, R), dtype=dtype).view(B, 1, 1)

    if state0 is None:
        x = torch.zeros((6, B, N, 1), dtype=dtype)       # P, E, I, Pv, Ev, Iv
        hE = torch.zeros((B, N, DELAYS_MAX), dtype=dtype)
    else:
        x, hE = state0
        x = x.repeat_interleave(R, dim=1) if x.shape[1] == S else x
        hE = hE.repeat_interleave(R, dim=0) if hE.shape[0] == S else hE
    P, E, I, Pv, Ev, Iv = x

    def delayed(hE):
        Ed0 = torch.gather(hE, 2, i0b).transpose(1, 2)
        if frac is None:
            return Ed0
        Ed1 = torch.gather(hE, 2, i1b).transpose(1, 2)
        return Ed0 + frac.T * (Ed1 - Ed0)               # frac[i, j] -> Ed[b, j, i]

    def step_ms(P, E, I, Pv, Ev, Iv, hE, u_amp):
        LEd = (w_n * delayed(hE)).sum(2, keepdim=True)
        for _ in range(hidden):
            if gen is not None:
                noise = std_in * torch.randn((B, N, 1), generator=gen, dtype=dtype)
            else:
                noise = 0.0
            P_s = vmax / (1 + torch.exp(r * (v0 - (E - I))))
            E_s = vmax / (1 + torch.exp(r * (v0 - c1 * P)))
            I_s = vmax / (1 + torch.exp(r * (v0 - c3 * P)))
            rE = noise + g * (LEd + dg @ E) + c2 * E_s
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

    nograd = proto.warmup_ms if grad_warmup_ms is None else proto.warmup_ms - grad_warmup_ms
    if grad_warmup_ms is None:
        nograd = 0
    with torch.no_grad():
        for _ in range(nograd):
            P, E, I, Pv, Ev, Iv, hE = step_ms(P, E, I, Pv, Ev, Iv, hE, None)
    P, E, I, Pv, Ev, Iv, hE = (t.detach() for t in (P, E, I, Pv, Ev, Iv, hE)) if nograd else (P, E, I, Pv, Ev, Iv, hE)
    for _ in range(proto.warmup_ms - nograd):
        P, E, I, Pv, Ev, Iv, hE = step_ms(P, E, I, Pv, Ev, Iv, hE, None)

    out = []
    p0, p1 = proto.pulse
    for t in range(proto.T):
        u_amp = amp if p0 <= t < p1 else None
        P, E, I, Pv, Ev, Iv, hE = step_ms(P, E, I, Pv, Ev, Iv, hE, u_amp)
        out.append(S2O_COEF * cy0 * (net.lm_dm @ (E - I))[..., 0] - Y0)     # (B, C)
    eeg = torch.stack(out, dim=-1).view(S, R, C, proto.T)
    if return_state:
        return eeg, (torch.stack([P, E, I, Pv, Ev, Iv]), hE)
    return eeg


def baseline_correct(eeg: torch.Tensor, pre: int = 100) -> torch.Tensor:
    return eeg - eeg[..., :pre].mean(-1, keepdim=True)
