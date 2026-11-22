"""Ground-truth parameter draws for teachers, and JRParams builders for teacher and student.

neurofe's JansenRit.forward does not use most raw parameter values directly: it applies
lower bounds such as `mu_eff = 1.5 + relu(mu)`. Draws and recovery errors are expressed
in these *effective* values, the ones that actually enter the equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .paths import REPO_ROOT  # noqa: F401  (puts neurofe on sys.path)
from neurofe.models.jansen_rit import JRParams
from neurofe.models.utils import Par

# Raw nominal values, as in neurofe's examples/example_train_jr.py.
NOMINAL_RAW = {
    "a": 100.0, "b": 50.0, "g": 1000.0,
    "c1": 135.0, "c2": 135 * 0.8, "c3": 135 * 0.25, "c4": 135 * 0.25,
    "std_in": 100.0, "mu": 1.0, "k": 10.0,
}
FIXED_RAW = {"A": 3.25, "B": 22.0, "vmax": 5.0, "v0": 6.0, "r": 0.56, "cy0": 5.0}
Y0 = 2.0

# effective = offset + relu(raw), mirroring JansenRit.forward.
OFFSET = {"a": 1.0, "b": 1.0, "g": 0.01, "c1": 0.01, "c2": 0.01, "c3": 0.01,
          "c4": 0.01, "std_in": 150.0, "mu": 1.5, "k": 0.5}

GLOBAL_FITTED = list(NOMINAL_RAW)

# Prior standard deviations passed to Par (neurofe turns them into precision 1/std).
PRIOR_STD = {
    # Values used in neurofe's example_train_jr.py.
    "pytepfit": {"a": 0.5, "b": 1.0, "g": 0.1, "c1": 0.2, "c2": 0.4, "c3": 0.8,
                 "c4": 0.8, "std_in": 10.0, "mu": 2.5, "k": 0.3},
    # Weak priors: std equal to the nominal value, so the data dominate.
    "weak": {n: abs(v) if v else 1.0 for n, v in NOMINAL_RAW.items()},
}


def to_effective(name: str, raw: float) -> float:
    return OFFSET.get(name, 0.0) + max(raw, 0.0) if name in OFFSET else raw


def to_raw(name: str, eff: float) -> float:
    return eff - OFFSET[name] if name in OFFSET else eff


def nominal_effective() -> dict[str, float]:
    return {n: to_effective(n, v) for n, v in NOMINAL_RAW.items()}


@dataclass
class TeacherParams:
    """Ground truth for one synthetic subject."""
    raw: dict[str, float]
    w_bb: np.ndarray              # (N, N) per-edge log-gain on SC; zeros = plain SC
    seed: int
    spread: float
    meta: dict = field(default_factory=dict)

    @property
    def effective(self) -> dict[str, float]:
        return {n: to_effective(n, v) for n, v in self.raw.items()}


def draw_teacher(n_nodes: int, seed: int, spread: float = 0.15,
                 w_bb_sigma: float = 0.0, vary=GLOBAL_FITTED) -> TeacherParams:
    """Draw global parameters log-uniformly within +/- spread of nominal (effective values).

    `w_bb_sigma` > 0 perturbs every SC edge by a log-normal gain (needed for S2/S3 tests);
    0 keeps the plain SC, which matches student tier S1 exactly.
    """
    rng = np.random.default_rng(seed)
    eff = nominal_effective()
    raw = dict(NOMINAL_RAW)
    for n in vary:
        f = np.exp(rng.uniform(np.log(1 - spread), np.log(1 + spread)))
        raw[n] = to_raw(n, eff[n] * f)
    w = rng.normal(0.0, w_bb_sigma, (n_nodes, n_nodes)) if w_bb_sigma > 0 else np.zeros((n_nodes, n_nodes))
    w = 0.5 * (w + w.T)
    return TeacherParams(raw=raw, w_bb=w, seed=seed, spread=spread,
                         meta={"w_bb_sigma": w_bb_sigma, "varied": list(vary)})


def teacher_jrparams(tp: TeacherParams, lm: np.ndarray, ki: np.ndarray) -> JRParams:
    """All-fixed JRParams carrying the teacher's ground truth (for neurofe's own forward)."""
    kw = {n: Par(v) for n, v in tp.raw.items()}
    kw.update({n: Par(v) for n, v in FIXED_RAW.items()})
    kw["y0"] = Par(Y0 * np.ones((lm.shape[0], 1)))
    kw["lm"] = Par(lm)
    kw["ki"] = Par(ki)
    return JRParams(**kw)


TIERS = ("S1", "S2", "S3")


def student_jrparams(tier: str, lm: np.ndarray, ki: np.ndarray,
                     prior: str = "weak", init: dict[str, float] | None = None) -> tuple[JRParams, bool]:
    """Student parameters for a capacity tier. Returns (JRParams, fit_connectivity).

    S1: global JR parameters + stimulation gain k.
    S2: S1 + per-edge w_bb.
    S3: S2 + the full leadfield lm (PyTepFit-style).
    Initial values and prior means are nominal, never the teacher's truth.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    stds = PRIOR_STD[prior]
    init = init or NOMINAL_RAW
    kw = {n: Par(init[n], NOMINAL_RAW[n], stds[n], True) for n in GLOBAL_FITTED}
    kw.update({n: Par(v) for n, v in FIXED_RAW.items()})
    kw["y0"] = Par(Y0 * np.ones((lm.shape[0], 1)))
    kw["ki"] = Par(ki)
    if tier == "S3":
        kw["lm"] = Par(lm, lm, np.ones_like(lm), True)
    else:
        kw["lm"] = Par(lm)
    return JRParams(**kw), tier in ("S2", "S3")


def fitted_effective(model) -> dict[str, float]:
    """Effective values of the global parameters currently held by a neurofe model."""
    out = {}
    for n in GLOBAL_FITTED:
        raw = float(getattr(model.param, n).value().detach())
        out[n] = to_effective(n, raw)
    return out
