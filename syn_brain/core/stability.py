"""Linear stability of the quiet (p=0) network state.

Why this exists: `scripts/diagnose_g_clip.py` found that some randomly-drawn network teachers
have NO bounded equilibrium under Laplacian coupling (the settled-window coupling magnitude
scales with whatever clip is set, rather than converging) -- a genuine linear instability of
diffusively-coupled damped oscillators above a critical coupling gain, not a numerical artifact
(docs/network_study_plan.txt, "RAISING U_BOUND: TESTED AND DISCONFIRMED"). `draw_teacher()] never
screened for this -- every global parameter including `g` is drawn independently and uniformly,
so some draws land past the threshold purely by chance, exactly the same mistake the single-node
study already found and fixed once for tonic-drive oscillation (T2/T3, 2026-11-22) but never
carried over to network scale.

The check: linearise diffsim.py's equations (same FIXED_RAW constants, same sigmoid, same
coupling term) about the quiet fixed point, which is independent of network structure (Laplacian
coupling is exactly zero-mean when every node is at the same state -- confirmed empirically at
N0.2). Network coupling only enters through the GRAPH LAPLACIAN eigenvalues of the connectivity
(coupling = W @ E_delayed - D @ E = -(D-W) @ E = -L @ E, ignoring conduction delay -- delay only
adds phase lag, which can only make an already-marginal mode worse, so this is a NECESSARY, not
sufficient, condition for full stability; documented, not hidden). Each Laplacian eigenmode lambda
gives an independent 6x6 linear system; lambda=0 (the uniform/consensus mode) reproduces the
isolated single-node Jacobian exactly, which both validates this code against
scripts/jr_parameters_explainer.py's equilibria() and explains why g's instability is a genuinely
NETWORK-scale phenomenon (it cannot appear in the single-node study at all).

Clip-free by design: the tanh(.../U_BOUND) clip is a numerical safety net, not part of the
physical model (see the U_BOUND-disconfirmation entry above) -- stability is assessed on the
unclipped equations, which is the right question ("does a bounded equilibrium exist at all"),
not "does the clip happen to look stable in a 400ms simulation window."
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .diffsim import U_BOUND
from .params import FIXED_RAW


def _S(v, vmax, v0, r):
    return vmax / (1 + np.exp(r * (v0 - v)))


def quiet_fixed_point(theta: dict) -> list[tuple[float, float, float]]:
    """All roots (P, E, I) of the isolated node's p=0 fixed-point equation. Usually one
    (the quiet rest state); more if the node is already bistable with no external drive.

    Includes the U_BOUND clip here (unlike the network-coupling term in _jacobian_at below):
    this is about finding the correct REST point the real simulator actually settles to, which
    is a clipped equation like every other quantity diffsim.py simulates; omitting the clip
    created spurious extra roots at large P where c2*S(c1*P) can exceed U_BOUND (c2 up to ~540
    at nominal, already past 500) -- caught by comparing against
    scripts/jr_parameters_explainer.py's equilibria(), which has one root here, not three."""
    A, B = FIXED_RAW["A"], FIXED_RAW["B"]
    vmax, v0, r = FIXED_RAW["vmax"], FIXED_RAW["v0"], FIXED_RAW["r"]
    a, b = theta["a"], theta["b"]
    c1, c2, c3, c4 = theta["c1"], theta["c2"], theta["c3"], theta["c4"]
    U = U_BOUND

    def EI(P):
        E = A / a * (U * np.tanh(c2 * _S(c1 * P, vmax, v0, r) / U))
        I = B / b * (U * np.tanh(c4 * _S(c3 * P, vmax, v0, r) / U))
        return E, I

    def g_of(P):
        E, I = EI(P)
        return A / a * (U * np.tanh(_S(E - I, vmax, v0, r) / U)) - P

    Ps = np.linspace(0, A / a * vmax * 1.02, 4000)
    gs = np.array([g_of(P) for P in Ps])
    roots = []
    for i in np.where(np.sign(gs[:-1]) != np.sign(gs[1:]))[0]:
        P = brentq(g_of, Ps[i], Ps[i + 1])
        E, I = EI(P)
        roots.append((P, E, I))
    return roots


def _jacobian_at(theta: dict, P: float, E: float, I: float, lam: float) -> np.ndarray:
    """6x6 Jacobian of (P, E, I, Pv, Ev, Iv)' at a fixed point, for network-Laplacian
    eigenvalue `lam` (lam=0 is the isolated-node / uniform-mode case)."""
    A, B = FIXED_RAW["A"], FIXED_RAW["B"]
    vmax, v0, r = FIXED_RAW["vmax"], FIXED_RAW["v0"], FIXED_RAW["r"]
    a, b, g = theta["a"], theta["b"], theta["g"]
    c1, c2, c3, c4 = theta["c1"], theta["c2"], theta["c3"], theta["c4"]

    def Sp(v):
        s = _S(v, vmax, v0, r)
        return r * s * (1 - s / vmax)

    sP = Sp(E - I)
    sE = c1 * Sp(c1 * P)
    sI = c3 * Sp(c3 * P)

    J = np.zeros((6, 6))
    J[0, 3] = J[1, 4] = J[2, 5] = 1.0
    # Pv' = A*a*S(E-I) - 2a*Pv - a^2*P
    J[3, 1] = A * a * sP
    J[3, 2] = -A * a * sP
    J[3, 3] = -2 * a
    J[3, 0] = -a ** 2
    # Ev' = A*a*(c2*S(c1*P) - g*lam*E) - 2a*Ev - a^2*E   (coupling_k = -lam*E_k)
    J[4, 0] = A * a * c2 * sE
    J[4, 1] = -A * a * g * lam - a ** 2
    J[4, 4] = -2 * a
    # Iv' = B*b*c4*S(c3*P) - 2b*Iv - b^2*I
    J[5, 0] = B * b * c4 * sI
    J[5, 2] = -b ** 2
    J[5, 5] = -2 * b
    return J


def laplacian_eigs(w_n: np.ndarray) -> np.ndarray:
    """Eigenvalues of the graph Laplacian L = D - W implied by diffsim.py's coupling term
    (coupling = W @ E_delayed - D @ E, D = diag(row sums of W)). Symmetric W -> real eigs >= 0."""
    D = np.diag(w_n.sum(1))
    L = D - w_n
    return np.linalg.eigvalsh(0.5 * (L + L.T))


def worst_growth_rate(theta: dict, lams: np.ndarray) -> float:
    """Max real part of any Jacobian eigenvalue, over every quiet fixed point and every
    network-Laplacian eigenmode. Negative = stable (damped); >=0 = no bounded equilibrium."""
    roots = quiet_fixed_point(theta)
    worst = -np.inf
    for (P, E, I) in roots:
        for lam in lams:
            ev = np.linalg.eigvals(_jacobian_at(theta, P, E, I, lam))
            worst = max(worst, float(ev.real.max()))
    return worst


def is_network_stable(theta: dict, lams: np.ndarray, margin: float = 0.0) -> tuple[bool, float]:
    """(stable, worst_growth_rate). `margin` (1/s) requires that much extra damping headroom,
    e.g. margin=1.0 rejects anything within 1/s of the instability boundary, not just anything
    literally past it -- recommended for teacher screening so draws aren't marginal."""
    worst = worst_growth_rate(theta, lams)
    return worst < -margin, worst


def critical_g(theta: dict, lam_max: float, lo: float = 1.0, hi: float = 1e6) -> float | None:
    """Bisect for the largest g (holding every other parameter fixed) at which the network
    (with worst-case eigenmode lam_max) is still stable. None if stable throughout [lo, hi]
    or unstable throughout (both reported by the caller via worst_growth_rate at the bounds)."""
    def stable_at(g):
        th = dict(theta, g=g)
        return worst_growth_rate(th, np.array([lam_max])) < 0

    if stable_at(hi):
        return None  # stable even at hi -- no threshold found in range
    if not stable_at(lo):
        return None  # already unstable at lo -- threshold is below range, not just at g
    for _ in range(60):
        mid = np.sqrt(lo * hi)  # bisect in log-space, g spans orders of magnitude
        if stable_at(mid):
            lo = mid
        else:
            hi = mid
    return lo
