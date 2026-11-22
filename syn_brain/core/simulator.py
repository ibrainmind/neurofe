"""Batched Jansen-Rit simulator for the teacher (the synthetic brain).

Mirrors neurofe's JansenRit.forward equation by equation (bounds, coupling, noise,
leadfield readout), but runs many trials at once along a leading batch axis. One
batch of 100 trials costs little more than a single trial, which makes generating
trial-averaged evoked responses practical on a laptop CPU.

tests/test_simulator.py checks that, with batch=1 and the same random seed, this
simulator reproduces JansenRit.forward to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .params import FIXED_RAW, OFFSET, Y0

# Constants hard-coded in JansenRit.forward.
U_BOUND = 500.0
S2O_COEF = 1e-4
DELAYS_MAX = 500


@dataclass
class SimConfig:
    step_size: float = 1e-4     # integration step, s
    tr: float = 1e-3            # output sampling period, s
    warmup_ms: int = 1000       # matches JansenRit.predict(warmup_batches=20, batch_size=50)

    @property
    def hidden_size(self) -> int:
        return int(round(self.tr / self.step_size))


def _eff(raw: dict, name: str) -> torch.Tensor:
    v = torch.tensor(float(raw[name]), dtype=torch.float32)
    if name in OFFSET:
        return OFFSET[name] + torch.relu(v)
    return v


def coupling_matrix(sc: np.ndarray, w_bb: np.ndarray, mask: np.ndarray | None = None) -> torch.Tensor:
    """w_n exactly as in JansenRit.forward: exp(w_bb)*SC, symmetrised, log1p, unit norm."""
    sc_t = torch.tensor(sc, dtype=torch.float32)
    w = torch.exp(torch.tensor(w_bb, dtype=torch.float32)) * sc_t
    w_sym = 0.5 * (w + w.T)
    w_n = torch.log1p(w_sym) / torch.linalg.norm(torch.log1p(w_sym))
    if mask is not None:
        w_n = w_n * torch.tensor(mask, dtype=torch.float32)
    return w_n


def random_initial_state(n_trials: int, n_nodes: int, rng: np.random.Generator):
    """Initial states drawn like JansenRit.get_initial_state / get_initial_delay_state."""
    hx = rng.uniform(-0.01, 0.01, (n_trials, n_nodes, 3, 2))
    hE = rng.uniform(0, 0.1, (n_trials, n_nodes, DELAYS_MAX))
    return torch.tensor(hx, dtype=torch.float32), torch.tensor(hE, dtype=torch.float32)


@torch.no_grad()
def simulate(raw: dict, sc: np.ndarray, dist: np.ndarray, lm: np.ndarray, ki: np.ndarray,
             stimulus: np.ndarray, w_bb: np.ndarray | None = None,
             hx: torch.Tensor | None = None, hE: torch.Tensor | None = None,
             n_trials: int = 1, step_size: float = 1e-4, tr: float = 1e-3,
             noise_scale: float = 1.0, return_states: bool = False, seed: int | None = None):
    """Run the JR network for len(stimulus time axis) samples.

    Parameters
    ----------
    raw : dict of raw parameter values (keys of NOMINAL_RAW; FIXED_RAW fills the rest).
    stimulus : (N, hidden_size, T) legacy-format input, the same array given to neurofe.
    hx, hE : optional (B, N, 3, 2) and (B, N, 500) initial states; random if None.
    noise_scale : multiplies the excitatory noise. 1.0 = the model as fitted.
        (0 gives a noise-free run, which JansenRit.forward cannot produce because
        its noise std has a hard lower bound of 150.)

    Returns
    -------
    eeg : (B, C, T) numpy array; states (B, N, 3, 2, T) if return_states.
    hx, hE : final states, so a run can be continued.
    """
    if seed is not None:
        torch.manual_seed(seed)
    p = dict(FIXED_RAW)
    p.update(raw)
    n = sc.shape[0]
    n_chans = lm.shape[0]
    hidden = int(round(tr / step_size))
    assert stimulus.shape[0] == n and stimulus.shape[1] == hidden, \
        f"stimulus must be (N={n}, hidden={hidden}, T), got {stimulus.shape}"
    T = stimulus.shape[2]

    if hx is None or hE is None:
        rng = np.random.default_rng(seed)
        hx, hE = random_initial_state(n_trials, n, rng)
    B = hx.shape[0]

    A, a = torch.relu(torch.tensor(p["A"], dtype=torch.float32)), _eff(p, "a")
    Bg, b = torch.relu(torch.tensor(p["B"], dtype=torch.float32)), _eff(p, "b")
    g = _eff(p, "g")
    c1, c2, c3, c4 = (_eff(p, c) for c in ("c1", "c2", "c3", "c4"))
    std_in = _eff(p, "std_in") * noise_scale
    vmax = torch.tensor(p["vmax"], dtype=torch.float32)
    v0 = torch.tensor(p["v0"], dtype=torch.float32)
    r = torch.tensor(p["r"], dtype=torch.float32)
    mu = _eff(p, "mu")
    k = _eff(p, "k")
    cy0 = torch.tensor(p["cy0"], dtype=torch.float32)
    y0 = torch.tensor(Y0, dtype=torch.float32)
    dt = torch.tensor(step_size, dtype=torch.float32)

    w_n = coupling_matrix(sc, np.zeros((n, n)) if w_bb is None else w_bb)
    dg = -torch.diag(torch.sum(w_n, dim=1))
    delays = (torch.tensor(dist, dtype=torch.float32) / mu).type(torch.int64)
    delays_b = delays.unsqueeze(0).expand(B, n, n)

    ki_t = torch.tensor(ki, dtype=torch.float32).reshape(n, 1)
    u_all = torch.tensor(stimulus, dtype=torch.float32)
    lm_t = torch.tensor(lm, dtype=torch.float32)
    lm_dm = lm_t - (1 / n_chans) * torch.matmul(torch.ones((1, n_chans)), lm_t)

    P, E, I = hx[:, :, 0:1, 0], hx[:, :, 1:2, 0], hx[:, :, 2:3, 0]   # (B, N, 1)
    Pv, Ev, Iv = hx[:, :, 0:1, 1], hx[:, :, 1:2, 1], hx[:, :, 2:3, 1]

    eeg = torch.zeros((B, n_chans, T))
    states = torch.zeros((B, n, 3, 2, T)) if return_states else None

    for t in range(T):
        # Ed[b, j, i] = hE[b, i, delays[i, j]], as gather + transpose in JansenRit.forward.
        Ed = torch.gather(hE, 2, delays_b).transpose(1, 2)
        LEd = torch.sum(w_n * Ed, dim=2, keepdim=True)          # (B, N, 1)
        for s in range(hidden):
            u = u_all[:, s:s + 1, t]                           # (N, 1)
            E_noise = std_in * torch.randn(B, n, 1)
            P_sigm = vmax / (1 + torch.exp(r * (v0 - (E - I))))
            E_sigm = vmax / (1 + torch.exp(r * (v0 - (c1 * P))))
            I_sigm = vmax / (1 + torch.exp(r * (v0 - (c3 * P))))
            rP = P_sigm
            rE = E_noise + g * (LEd + torch.matmul(dg, E)) + c2 * E_sigm
            rI = c4 * I_sigm
            rP_bd = U_BOUND * torch.tanh(rP / U_BOUND)
            rE_bd = U_BOUND * torch.tanh(rE / U_BOUND) + k * ki_t * u
            rI_bd = U_BOUND * torch.tanh(rI / U_BOUND)
            P_n = P + dt * Pv
            E_n = E + dt * Ev
            I_n = I + dt * Iv
            Pv_n = Pv + dt * (A * a * rP_bd - 2 * a * Pv - a ** 2 * P)
            Ev_n = Ev + dt * (A * a * rE_bd - 2 * a * Ev - a ** 2 * E)
            Iv_n = Iv + dt * (Bg * b * rI_bd - 2 * b * Iv - b ** 2 * I)
            P, E, I, Pv, Ev, Iv = P_n, E_n, I_n, Pv_n, Ev_n, Iv_n
        hE = torch.cat([E, hE[:, :, :-1]], dim=2)
        eeg[:, :, t] = (S2O_COEF * cy0 * torch.matmul(lm_dm, E - I) - y0)[:, :, 0]
        if return_states:
            states[..., t] = torch.stack([torch.cat([P, E, I], 2), torch.cat([Pv, Ev, Iv], 2)], dim=3)

    hx_out = torch.stack([torch.cat([P, E, I], 2), torch.cat([Pv, Ev, Iv], 2)], dim=3)
    out = (eeg.numpy(),)
    if return_states:
        out += (states.numpy(),)
    return out + (hx_out, hE)


def evoked(raw: dict, sc, dist, lm, ki, stimulus, w_bb=None, n_trials: int = 100,
           cfg: SimConfig = SimConfig(), noise_scale: float = 1.0, seed: int = 0,
           chunk: int = 100):
    """Trial-averaged evoked response with a warm-up before the stimulus window.

    Each trial starts from a random state, runs `cfg.warmup_ms` with no input (as
    JansenRit.predict does), then the stimulus window. Returns (mean, trials):
    mean (C, T), trials (n_trials, C, T).
    """
    n = sc.shape[0]
    hidden = cfg.hidden_size
    warm = np.zeros((n, hidden, cfg.warmup_ms))
    out = []
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    done = 0
    while done < n_trials:
        bsz = min(chunk, n_trials - done)
        hx, hE = random_initial_state(bsz, n, rng)
        _, hx, hE = simulate(raw, sc, dist, lm, ki, warm, w_bb, hx, hE,
                             step_size=cfg.step_size, tr=cfg.tr, noise_scale=noise_scale)
        eeg, _, _ = simulate(raw, sc, dist, lm, ki, stimulus, w_bb, hx, hE,
                             step_size=cfg.step_size, tr=cfg.tr, noise_scale=noise_scale)
        out.append(eeg)
        done += bsz
    trials = np.concatenate(out, axis=0)
    return trials.mean(0), trials
