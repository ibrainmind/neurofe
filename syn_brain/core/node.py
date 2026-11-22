"""Single Jansen-Rit node with programmable virtual-TMS input, for system-identification studies.

Same equations, constants and integration scheme as neurofe's JansenRit.forward (and
core/simulator.py, core/diffsim.py) with the network terms removed: Euler steps of
0.1 ms, 10 sub-steps per 1 ms output sample, the same tanh input bounds (500), the same
effective-value floors, and EEG = 5e-4 * leadfield * (E - I).

What is new here is the stimulus. neurofe injects one 10 ms rectangular pulse, scaled by
k, into the excitatory-interneuron drive of the pyramidal cells (the `rE` channel). Here
the input is an arbitrary waveform u(t) per condition, and it can also be routed into
the inhibitory channel (`rI`), so a "virtual TMS" protocol is just a set of waveforms.

With one node, the 62-channel EEG is rank one: leadfield column x waveform. Fitting the
full EEG with a sum-of-squares loss is therefore the same as fitting the scalar waveform
y(t) = 5e-4 * ||lm_dm|| * (E - I)(t), and channel noise of std sigma becomes noise of std
sigma on y. Everything below works on y, in microvolts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .params import FIXED_RAW, NOMINAL_RAW, OFFSET, to_effective

U_BOUND = 500.0
S2O = 1e-4 * FIXED_RAW["cy0"]          # 5e-4, as in neurofe
DT = 1e-4                               # 0.1 ms
HIDDEN = 10                             # sub-steps per 1 ms sample

# Parameters neurofe fits for a single node (g, mu and std_in have no effect here).
FIT_NAMES = ("a", "b", "c1", "c2", "c3", "c4", "k")
# Parameters neurofe holds fixed; included in the extended analyses.
EXTRA_NAMES = ("A", "B", "v0", "r", "vmax")
ALL_NAMES = FIT_NAMES + EXTRA_NAMES

NOMINAL = {n: to_effective(n, NOMINAL_RAW[n]) for n in FIT_NAMES}
NOMINAL.update({n: FIXED_RAW[n] for n in EXTRA_NAMES})

PHYSICAL = {
    "a": ("1/tau_e, excitatory synaptic rate", "1/s", "Rate constant of the excitatory PSP kernel (tau_e = 1/a ~ 10 ms). Sets how fast pyramidal and excitatory responses rise and decay."),
    "b": ("1/tau_i, inhibitory synaptic rate", "1/s", "Rate constant of the inhibitory (GABA-A-like) PSP kernel (tau_i = 1/b ~ 20 ms). Sets the duration of the inhibitory rebound."),
    "c1": ("pyramidal -> excitatory interneurons", "-", "Average number of synaptic contacts (Braitenberg & Schuz ratio 1.0 x C)."),
    "c2": ("excitatory interneurons -> pyramidal", "-", "Recurrent excitatory feedback gain (0.8 x C)."),
    "c3": ("pyramidal -> inhibitory interneurons", "-", "Drive to the inhibitory pool (0.25 x C)."),
    "c4": ("inhibitory interneurons -> pyramidal", "-", "Inhibitory feedback gain (0.25 x C)."),
    "k": ("stimulus gain", "-", "How strongly the coil current is converted into synaptic drive. Depends on coil, orientation, scalp-to-cortex distance; not a brain property."),
    "A": ("max excitatory PSP", "mV", "Peak amplitude of the excitatory PSP (3.25 mV)."),
    "B": ("max inhibitory PSP", "mV", "Peak amplitude of the inhibitory PSP (22 mV)."),
    "v0": ("firing threshold", "mV", "Mean membrane potential at half-maximal firing (6 mV)."),
    "r": ("sigmoid slope", "1/mV", "Spread of firing thresholds across the population (0.56 /mV)."),
    "vmax": ("max firing rate", "1/s", "Maximal population firing rate (5 /s, as coded in neurofe)."),
}


# ------------------------------------------------------------------------------- stimuli
@dataclass(frozen=True)
class Stim:
    """One stimulation condition: a list of rectangular pulses (onset_ms, width_ms, amp).

    `to_i` is the fraction of the pulse also routed into the inhibitory channel
    (0 = neurofe's default, excitatory-only injection).
    """
    name: str
    pulses: tuple
    to_i: float = 0.0
    label: str = ""
    tonic: float = 0.0      # constant background drive into rE (classic JR "p"), not scaled by k

    def waveform(self, T: int) -> np.ndarray:
        u = np.zeros(T)
        for on, w, amp in self.pulses:
            u[on:on + w] += amp
        return u


T_WIN = 400          # samples (ms) per window, as in PyTepFit
ONSET = 110          # pulse onset sample; 0-110 is pre-stimulus baseline
PRE = 100            # baseline-correction samples

STIMS = {
    "SP250":   Stim("SP250", ((ONSET, 10, 250.0),), label="single pulse, weak (250)"),
    "SP1000":  Stim("SP1000", ((ONSET, 10, 1000.0),), label="single pulse, neurofe default (1000 x 10 ms)"),
    "SP4000":  Stim("SP4000", ((ONSET, 10, 4000.0),), label="single pulse, strong (4000)"),
    "SPshort": Stim("SPshort", ((ONSET, 2, 5000.0),), label="brief pulse, 2 ms x 5000 (same charge)"),
    "SPlong":  Stim("SPlong", ((ONSET, 30, 333.3),), label="long pulse, 30 ms x 333 (same charge)"),
    "PP30":    Stim("PP30", ((ONSET, 10, 1000.0), (ONSET + 30, 10, 1000.0)), label="paired pulse, ISI 30 ms"),
    "PP100":   Stim("PP100", ((ONSET, 10, 1000.0), (ONSET + 100, 10, 1000.0)), label="paired pulse, ISI 100 ms"),
    "TR10":    Stim("TR10", tuple((ONSET + 100 * i, 10, 1000.0) for i in range(3)), label="10 Hz train, 3 pulses"),
    "EI1000":  Stim("EI1000", ((ONSET, 10, 1000.0),), to_i=0.5, label="pulse into E and I (50%)"),
}
STIM_ORDER = list(STIMS)
# The same pulses delivered on top of a tonic background drive, at three operating
# points (fixed point of the nominal node; see docs/background_drive_note.txt):
#   p=110            just below the jump; still the quiet, near-linear state (E7)
#   p=120            just above the jump onto the upper branch (threshold crossing)
#   p=150            solidly on the upper branch, away from the jump (stable "active" state)
P_DRIVEN = 110.0
FIRING_PS = (110.0, 120.0, 150.0)
for _p in FIRING_PS:
    for _n in ("SP250", "SP1000", "SP4000"):
        _tag = f"@p{int(_p)}"
        STIMS[_n + _tag] = Stim(_n + _tag, STIMS[_n].pulses, tonic=_p,
                                label=STIMS[_n].label + f", tonic drive p={int(_p)}")
DRIVEN_ORDER = [n for n in STIMS if n not in STIM_ORDER]

# settle_ms=300 (the old default, still fine at p=0) is NOT enough once tonic drive is on:
# relaxation to the fixed point slows sharply near and above the p~115-120 jump (critical
# slowing down at the bifurcation). At gain~250 (a typical readout), the resulting
# incompletely-settled pre-stimulus baseline drifts by up to ~0.5 uV between p=115 and
# p=200 at settle_ms=300 -- bigger than the 0.42 uV sensor noise floor this study is
# calibrated against -- and is still ~0.02 uV at settle_ms=900. It is <0.005 uV by
# settle_ms=1500 at every p tested (0-240). See docs/background_drive_note.txt section 6.
SETTLE_MS_DEFAULT = 1500


# ------------------------------------------------------------------------------- simulate
def simulate(theta: dict, stims: list[Stim], settle_ms: int = SETTLE_MS_DEFAULT, T: int = T_WIN,
             gain: float = 1.0, dtype=torch.float64) -> torch.Tensor:
    """Noise-free single-node response for every (model, condition) pair.

    theta: name -> tensor of shape (M,) (or scalar), effective values. Names missing from
           theta take their nominal value.
    stims: S conditions; each of the M models sees all S.
    gain:  ||lm_dm|| of the read-out parcel, so that y is in EEG microvolts.

    Returns baseline-corrected y, shape (M, S, T). The first `settle_ms` run without
    autograd from a fixed small state; the recorded window (which starts with 110 ms of
    pre-stimulus baseline) is differentiated, so the parameter dependence of the resting
    state is carried by the pre-stimulus segment.
    """
    M = 1
    for n in ALL_NAMES:
        v = torch.as_tensor(theta.get(n, NOMINAL[n]))
        M = max(M, v.numel())
    S = len(stims)
    idx = np.repeat(np.arange(M), S)
    rows = [stims[j] for j in np.tile(np.arange(S), M)]
    return simulate_rows(theta, idx, rows, settle_ms, T, gain, dtype).view(M, S, T)


def simulate_rows(theta: dict, model_idx, stims_rows: list[Stim], settle_ms: int = SETTLE_MS_DEFAULT,
                  T: int = T_WIN, gain: float = 1.0, dtype=torch.float64) -> torch.Tensor:
    """Like `simulate`, but row i runs model theta[model_idx[i]] under stims_rows[i].

    Lets models with different training protocols share one batch. Returns (B, T).
    """
    idx = torch.as_tensor(np.asarray(model_idx), dtype=torch.long)
    th = {}
    for n in ALL_NAMES:
        v = torch.as_tensor(theta.get(n, NOMINAL[n]), dtype=dtype)
        th[n] = v.expand(int(idx.max()) + 1)[idx] if v.dim() == 0 else v[idx]
    u = torch.tensor(np.stack([s.waveform(T) for s in stims_rows]), dtype=dtype)   # (B, T)
    to_i = torch.tensor([s.to_i for s in stims_rows], dtype=dtype)
    tonic = torch.tensor([s.tonic for s in stims_rows], dtype=dtype)
    B = len(stims_rows)
    a, b, c1, c2, c3, c4, k = (th[n] for n in FIT_NAMES)
    A, Bg, v0, r, vmax = (th[n] for n in EXTRA_NAMES)
    Aa, Bb, a2, b2, twoa, twob = A * a, Bg * b, a * a, b * b, 2 * a, 2 * b

    x = torch.zeros((6, B), dtype=dtype)
    P, E, I, Pv, Ev, Iv = x

    def sub(P, E, I, Pv, Ev, Iv, uE, uI):
        P_s = vmax / (1 + torch.exp(r * (v0 - (E - I))))
        E_s = vmax / (1 + torch.exp(r * (v0 - c1 * P)))
        I_s = vmax / (1 + torch.exp(r * (v0 - c3 * P)))
        rP = U_BOUND * torch.tanh(P_s / U_BOUND)
        rE = U_BOUND * torch.tanh(c2 * E_s / U_BOUND) + uE
        rI = U_BOUND * torch.tanh(c4 * I_s / U_BOUND) + uI
        return (P + DT * Pv, E + DT * Ev, I + DT * Iv,
                Pv + DT * (Aa * rP - twoa * Pv - a2 * P),
                Ev + DT * (Aa * rE - twoa * Ev - a2 * E),
                Iv + DT * (Bb * rI - twob * Iv - b2 * I))

    zero = torch.zeros(B, dtype=dtype)
    with torch.no_grad():
        for _ in range(settle_ms * HIDDEN):
            P, E, I, Pv, Ev, Iv = sub(P, E, I, Pv, Ev, Iv, tonic, zero)
    P, E, I, Pv, Ev, Iv = (t.detach() for t in (P, E, I, Pv, Ev, Iv))
    out = []
    for t in range(T):
        if bool((u[:, t] != 0).any()):
            uE = k * u[:, t]
            uI = to_i * uE
            uE = uE + tonic
        else:
            uE, uI = tonic, zero
        for _ in range(HIDDEN):
            P, E, I, Pv, Ev, Iv = sub(P, E, I, Pv, Ev, Iv, uE, uI)
        out.append(E - I)
    y = gain * S2O * torch.stack(out, -1)
    return y - y[..., :PRE].mean(-1, keepdim=True)


def readout_gain(lm_col: np.ndarray) -> float:
    lm = lm_col - lm_col.mean()
    return float(np.linalg.norm(lm))
