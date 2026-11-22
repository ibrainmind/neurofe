"""Misspecification ladder: one configuration per rung, so a rung is a flag, not a code branch.

Phase 1 implements R0 and R1. Later rungs are declared so the plan is visible in code,
and raise NotImplementedError until Phase 3 builds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rung:
    name: str
    description: str
    sensor_noise_uV: float = 0.0     # std of added sensor noise on the trial average
    w_bb_sigma: float = 0.0          # teacher per-edge SC perturbation
    implemented: bool = True
    extra: dict = field(default_factory=dict)


# Baseline (pre-pulse) std of the real trial-averaged TMS-EEG, median of 20 subjects,
# measured by scripts/calibrate_teacher.py.
REAL_BASELINE_STD_UV = 0.42

RUNGS = {
    "R0": Rung("R0", "Clean: teacher has the student's exact form; trial noise only"),
    "R1": Rung("R1", "Sensor noise at the real recordings' baseline level",
               sensor_noise_uV=REAL_BASELINE_STD_UV),
    "R2": Rung("R2", "Regional heterogeneity in excitability", implemented=False),
    "R3": Rung("R3", "Student gets a corrupted SC", implemented=False),
    "R4": Rung("R4", "Stimulus spreads beyond the target parcel", implemented=False),
    "R5": Rung("R5", "Non-cortical late component added", implemented=False),
    "R6": Rung("R6", "Data from run_net_sim (different model form)", implemented=False),
}


def get_rung(name: str) -> Rung:
    r = RUNGS[name]
    if not r.implemented:
        raise NotImplementedError(f"Rung {name} is planned for Phase 3")
    return r


def add_sensor_noise(eeg: np.ndarray, std_uV: float, rng: np.random.Generator,
                     smooth_ms: int = 5) -> np.ndarray:
    """Add temporally smoothed Gaussian noise with the requested per-sample std.

    Smoothing with a short boxcar makes the noise less white, closer to residual
    EEG background; the result is rescaled so its std is exactly `std_uV`.
    """
    if std_uV <= 0:
        return eeg.copy()
    w = rng.normal(size=eeg.shape)
    if smooth_ms > 1:
        k = np.ones(smooth_ms) / smooth_ms
        w = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), -1, w)
    w *= std_uV / w.std()
    return eeg + w
