"""Explainer: the Jansen-Rit node parameters, what they mean, and which are identifiable (PDF).

Builds syn_brain/reports/jr_parameters_explainer.pdf from the single-node study results
(results/single_node_tms/: analysis.json, summary.json, jacobian.npz, fig_*.png) plus a few
new illustrations computed here with core/node.py.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.jr_parameters_explainer
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate

from syn_brain.core import node
from syn_brain.core.paths import SYN_ROOT
from syn_brain.scripts.single_node_tms import OUT, FIT, ALL, NOM_ALL, PROTOCOLS
from syn_brain.core.report_style import BOX, CAP, SMALL, TITLE, W, P, bullets, table, pct
from syn_brain.core.report_style import H1 as _H1, H2 as _H2

H1 = ParagraphStyle("h1k", parent=_H1, keepWithNext=1)
H2 = ParagraphStyle("h2k", parent=_H2, keepWithNext=1)
EQ = ParagraphStyle("eq", parent=SMALL, fontName="DV", fontSize=8.6, leading=13, leftIndent=18, spaceBefore=3,
                    spaceAfter=6)
PDF = SYN_ROOT / "reports" / "jr_parameters_explainer.pdf"
FIG = OUT / "explainer"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
CLS_COL = {"stiff": "#2a9d8f", "conditional": "#e9c46a", "sloppy": "#f4a261", "gauge": "#9d0208"}
U = 500.0
NOM = {n: float(v) for n, v in zip(ALL, NOM_ALL)}


# ============================================================================ computations
def fixed_point(p, th=None):
    """Resting state of the node under constant drive p (damped fixed-point iteration from rest)."""
    t = dict(NOM, **(th or {}))
    S = lambda v: t["vmax"] / (1 + np.exp(t["r"] * (t["v0"] - v)))           # noqa: E731
    Pp = E = I = 0.0
    for _ in range(60000):
        Pn = t["A"] / t["a"] * U * np.tanh(S(E - I) / U)
        En = t["A"] / t["a"] * (U * np.tanh(t["c2"] * S(t["c1"] * Pp) / U) + p)
        In = t["B"] / t["b"] * U * np.tanh(t["c4"] * S(t["c3"] * Pp) / U)
        Pp, E, I = 0.98 * Pp + 0.02 * Pn, 0.98 * E + 0.02 * En, 0.98 * I + 0.02 * In
    v = E - I
    s = S(v)
    return v, s, t["r"] * s * (1 - s / t["vmax"])


def combo_crb(J, conds, sigma):
    """CRB on log-products and log-ratios of c's (7 fitted parameters free), nominal teacher."""
    ci = [(node.STIM_ORDER + node.DRIVEN_ORDER).index(c) for c in conds]
    Jf = J[0, ci][..., :len(FIT)].reshape(-1, len(FIT))
    cov = np.linalg.pinv(Jf.T @ Jf / sigma ** 2, rcond=1e-15)
    out = {}
    for name, coef in {"c1·c2": {"c1": 1, "c2": 1}, "c1/c2": {"c1": 1, "c2": -1},
                       "c3·c4": {"c3": 1, "c4": 1}, "c3/c4": {"c3": 1, "c4": -1}}.items():
        v = np.zeros(len(FIT))
        for n, c in coef.items():
            v[FIT.index(n)] = c
        out[name] = float(np.sqrt(v @ cov @ v)) * 100
    return out


# ============================================================================ E/I balance and bifurcations
# E/I balance multiplier m, as in reports/ei_balance_bifurcation_map.pdf: c1, c2 x sqrt(m); c3, c4 / sqrt(m).
# Loop gains then scale as c1*c2 ~ m and c3*c4 ~ 1/m, so m^2 is the ratio of excitatory to inhibitory loop strength.
STUDY_POINTS = {"balanced": (1.0, 150), "E-dominant": (1.87, 105), "I-dominant": (0.53, 240)}
DT_NEUROFE = 1e-4


def ei_params(m):
    t = dict(NOM)
    s_ = np.sqrt(m)
    t["c1"] *= s_; t["c2"] *= s_; t["c3"] /= s_; t["c4"] /= s_
    return t


def _S(v, t):
    return t["vmax"] / (1 + np.exp(t["r"] * (t["v0"] - v)))


def _rhs(x, t, p):
    P_, E, I, Pv, Ev, Iv = x
    rP = U * np.tanh(_S(E - I, t) / U)
    rE = U * np.tanh(t["c2"] * _S(t["c1"] * P_, t) / U) + p
    rI = U * np.tanh(t["c4"] * _S(t["c3"] * P_, t) / U)
    a, b, A, B = t["a"], t["b"], t["A"], t["B"]
    return np.array([Pv, Ev, Iv, A * a * rP - 2 * a * Pv - a * a * P_, A * a * rE - 2 * a * Ev - a * a * E,
                     B * b * rI - 2 * b * Iv - b * b * I])


def equilibria(p, m=1.0):
    """All fixed points of the continuous JR equations at drive p and balance m, with linear stability.

    A fixed point has P = (A/a)·S(E - I), E = (A/a)·(c2·S(c1·P) + p), I = (B/b)·c4·S(c3·P) (with neurofe's tanh
    bounds), so it is a root of one scalar function of P; all roots are found on a fine grid.
    """
    from scipy.optimize import brentq
    t = ei_params(m)

    def EI(P_):
        return (t["A"] / t["a"] * (U * np.tanh(t["c2"] * _S(t["c1"] * P_, t) / U) + p),
                t["B"] / t["b"] * U * np.tanh(t["c4"] * _S(t["c3"] * P_, t) / U))

    def g(P_):
        E, I = EI(P_)
        return t["A"] / t["a"] * U * np.tanh(_S(E - I, t) / U) - P_
    Ps = np.linspace(0, t["A"] / t["a"] * t["vmax"] * 1.02, 6000)
    gs = g(Ps)
    out = []
    for i in np.where(np.sign(gs[:-1]) != np.sign(gs[1:]))[0]:
        P_ = brentq(g, Ps[i], Ps[i + 1])
        E, I = EI(P_)
        x = np.array([P_, E, I, 0.0, 0.0, 0.0])
        J = np.zeros((6, 6))
        for k_ in range(6):
            dx = np.zeros(6); dx[k_] = 1e-6 * max(1.0, abs(x[k_]))
            J[:, k_] = (_rhs(x + dx, t, p) - _rhs(x - dx, t, p)) / (2 * dx[k_])
        ev = np.linalg.eigvals(J)
        lead = ev[np.argmax(ev.real)]
        w = abs(lead.imag)
        sE = t["r"] * _S(t["c1"] * P_, t) * (1 - _S(t["c1"] * P_, t) / t["vmax"])
        sI = t["r"] * _S(t["c3"] * P_, t) * (1 - _S(t["c3"] * P_, t) / t["vmax"])
        out.append(dict(v=E - I, P=P_, E=E, I=I, stable=bool(ev.real.max() < 0), re=float(lead.real),
                        f=w / (2 * np.pi), re_euler=float(lead.real + w ** 2 * DT_NEUROFE / 2), sE=sE, sI=sI))
    return out


def simulate_np(cases, T_s, dt=DT_NEUROFE, x0=None):
    """Batch Euler simulation (neurofe's integrator) of the noise-free JR node; returns E - I at 1 kHz and state."""
    ts = [ei_params(m) for m, _ in cases]
    g = lambda k_: np.array([t[k_] for t in ts])                                        # noqa: E731
    a, b, A, B, c1, c2, c3, c4, vm, v0, r = (g(k_) for k_ in "a b A B c1 c2 c3 c4 vmax v0 r".split())
    p = np.array([c[1] for c in cases], float)
    S = lambda v: vm / (1 + np.exp(r * (v0 - v)))                                       # noqa: E731
    P_, E, I, Pv, Ev, Iv = np.zeros((6, len(cases))) if x0 is None else x0
    rec, out = int(round(1e-3 / dt)), []
    for i in range(int(round(T_s / dt))):
        rP = U * np.tanh(S(E - I) / U); rE = U * np.tanh(c2 * S(c1 * P_) / U) + p; rI = U * np.tanh(c4 * S(c3 * P_) / U)
        P_, E, I, Pv, Ev, Iv = (P_ + dt * Pv, E + dt * Ev, I + dt * Iv, Pv + dt * (A * a * rP - 2 * a * Pv - a * a * P_),
                                Ev + dt * (A * a * rE - 2 * a * Ev - a * a * E), Iv + dt * (B * b * rI - 2 * b * Iv - b * b * I))
        if i % rec == 0:
            out.append(E - I)
    return np.array(out), np.array([P_, E, I, Pv, Ev, Iv])


def fig_bifurcation(path):
    ps = np.arange(0, 282, 2.0)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True)
    for col, (lab, m) in enumerate((("I-dominant  m = 0.53", 0.53), ("balanced  m = 1", 1.0),
                                    ("E-dominant  m = 1.87", 1.87))):
        ax, axd = axes[0, col], axes[1, col]
        act = []
        for p_ in ps:
            eq = equilibria(p_, m)
            for e in eq:
                ax.plot(p_, e["v"], "o", ms=2.2, color="k" if e["stable"] else "#d62828",
                        mfc="k" if e["stable"] else "none", mew=0.6)
            top = max(eq, key=lambda q: q["v"])
            act.append((p_, top["re"], top["re_euler"], top["f"], len(eq)))
        act = np.array(act)
        bist = act[act[:, 4] >= 3, 0]
        if len(bist):
            for a_ in (ax, axd):
                a_.axvspan(bist.min(), bist.max(), color="#e9c46a", alpha=0.25, lw=0)
        ax.axhline(NOM["v0"], color="0.6", ls=":", lw=0.8)
        ax.set_title(lab); ax.set_ylim(-3, 10)
        axd.plot(act[:, 0], act[:, 1], color="#264653", label="damping Re(λ), true equations")
        axd.plot(act[:, 0], act[:, 2], color="#264653", ls="--", lw=0.9, label="same with neurofe's Euler step")
        axd.axhline(0, color="k", lw=0.6)
        axd.set_ylim(-30, 8); axd.set_xlabel("background drive p")
        axf = axd.twinx(); axf.plot(act[:, 0], act[:, 3], color="#e76f51", lw=1); axf.set_ylim(0, 14)
        axf.spines["right"].set_visible(True)
        if col == 2:
            axf.set_ylabel("ringing frequency (Hz)", color="#e76f51")
        else:
            axf.set_yticklabels([])
        for name, (mm, pp) in STUDY_POINTS.items():
            if abs(mm - m) < 1e-6:
                ax.plot(pp, max(equilibria(pp, mm), key=lambda q: q["v"])["v"], "*", color="#1f6fb4", ms=10)
    axes[0, 0].set_ylabel("resting E − I (mV)")
    axes[1, 0].set_ylabel("damping of active state (1/s)")
    axes[1, 0].legend(fontsize=6.3, frameon=False, loc="lower left")
    axes[0, 0].plot([], [], "o", color="k", ms=3, label="stable state")
    axes[0, 0].plot([], [], "o", color="#d62828", mfc="none", ms=3, label="unstable state")
    axes[0, 0].fill_between([], [], color="#e9c46a", alpha=0.25, label="bistable: quiet and active coexist")
    axes[0, 0].legend(fontsize=6.3, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_stability_map(path):
    ms = np.exp(np.linspace(np.log(1 / 3), np.log(3), 25))
    ps = np.arange(0, 285, 7.5)
    cls = np.zeros((len(ms), len(ps)))
    for i, m in enumerate(ms):
        for j, p_ in enumerate(ps):
            eq = equilibria(p_, m)
            st = [e for e in eq if e["stable"]]
            if not st:
                c = 4                                             # no stable state: sustained oscillation
            elif len(st) >= 2:
                c = 1                                             # quiet and active both stable
            elif st[0]["v"] >= 2.5:                               # active: above the highest quiet state (fold)
                c = 3 if st[0]["re_euler"] > 0 else 2             # 3: oscillates only with neurofe's Euler step
            else:
                c = 0                                             # quiet
            cls[i, j] = c
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d8f3dc", "#e9c46a", "#f4a261", "#b5838d", "#d62828"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.pcolormesh(ps, ms, cls, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")
    ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_yticks([1 / 3, 0.5, 1, 2, 3]); ax.set_yticklabels(["1/3", "0.5", "1", "2", "3"])
    for name, (mm, pp) in STUDY_POINTS.items():
        ax.plot(pp, mm, "*", color="#1f6fb4", ms=11, mec="w"); ax.text(pp + 5, mm * 1.05, name, fontsize=7, color="#1f6fb4")
    ax.set_xlabel("background drive p"); ax.set_ylabel("E/I balance m  (>1: more excitation)")
    labels = ["quiet", "bistable (quiet + active)", "active, stable", "active; oscillates only\nwith neurofe's Euler step",
              "sustained oscillation\n(no stable state)"]
    for c_, l_ in zip(cmap.colors, labels):
        ax.plot([], [], "s", color=c_, ms=8, label=l_)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Stability map from the equations (fixed points + eigenvalues)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_hysteresis(path):
    down = [150, 130, 110, 90, 70, 50, 40, 35, 30, 20, 10]
    _, x = simulate_np([(1.0, 150)], 6.0)
    ramp = []
    for p_ in down:
        y, x = simulate_np([(1.0, p_)], 5.0, x0=x)
        ramp.append(y[-1000:, 0].mean())
    y, _ = simulate_np([(1.0, p_) for p_ in down], 6.0)
    cold = y[-1000:].mean(0)
    ps = np.arange(0, 160, 1.0)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for p_ in ps:
        for e in equilibria(p_, 1.0):
            ax.plot(p_, e["v"], ".", ms=1.6, color="0.6" if e["stable"] else "#f4a261")
    ax.plot(down, ramp, "-o", color="#d62828", ms=4, label="ramp down from the active state")
    ax.plot(down, cold, "s", color="#1f6fb4", ms=5, mfc="none", label="cold start at each p")
    lo = min(p_ for p_ in ps if len(equilibria(p_, 1.0)) >= 3)
    hi = max(p_ for p_ in ps if len(equilibria(p_, 1.0)) >= 3)
    ax.axvspan(lo, hi + 1, color="#e9c46a", alpha=0.25, lw=0)
    ax.text((lo + hi) / 2, 8.3, f"bistable window p ≈ {lo:.0f}–{hi + 1:.0f}", ha="center", fontsize=7.5)
    ax.set_xlabel("background drive p"); ax.set_ylabel("resting E − I (mV)"); ax.set_ylim(-3, 9)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("Hysteresis at the quiet/active transition (balanced, m = 1)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return dict(lo=lo, hi=hi + 1, down=down, ramp=[float(v) for v in ramp], cold=[float(v) for v in cold])


def dt_check(cache):
    """Is a limit cycle real or a product of neurofe's Euler step? Compare dt = 0.1 ms and 0.01 ms (cached)."""
    if cache.exists():
        return json.load(open(cache))
    cases = [(1.0, 270), (1.0, 255), (0.8, 200)]
    out = {}
    for dt in (1e-4, 1e-5):
        y, _ = simulate_np(cases, 30.0, dt=dt)
        seg = y[:30000].reshape(30, 1000, len(cases))
        amp = seg.max(1) - seg.min(1)
        out[f"{dt:g}"] = {f"m={m} p={p_}": [float(amp[s_ - 1, k_]) for s_ in (5, 15, 30)]
                          for k_, (m, p_) in enumerate(cases)}
    json.dump(out, open(cache, "w"), indent=1)
    return out


# ============================================================================ figures
def fig_column(path):
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(0, 1); ax.axis("off")
    box = {"pyr": (0.5, 0.78, "Pyramidal cells\nmembrane potential  v = E − I\nfiring  S(E − I)", "#cfe3f5"),
           "ein": (0.16, 0.30, "Excitatory\ninterneurons\nfiring  S(c1·P)", "#d9f0d3"),
           "iin": (0.84, 0.30, "Inhibitory\ninterneurons\nfiring  S(c3·P)", "#f8d7da")}
    for k_, (x, y, t, c) in box.items():
        ax.add_patch(FancyBboxPatch((x - 0.16, y - 0.1), 0.32, 0.2, boxstyle="round,pad=0.01", fc=c, ec="#333",
                                    lw=0.8))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.6)

    def arr(x0, y0, x1, y1, col, txt, tx, ty, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3,
                                     connectionstyle=f"arc3,rad={rad}"))
        ax.text(tx, ty, txt, fontsize=7, color=col, ha="center")
    arr(0.36, 0.70, 0.16, 0.40, "#333", "c1 · (A, a kernel → P)", 0.12, 0.60, 0.15)
    arr(0.64, 0.70, 0.84, 0.40, "#333", "c3 · (A, a kernel → P)", 0.88, 0.60, -0.15)
    arr(0.27, 0.40, 0.42, 0.68, "#1f6fb4", "c2 · excitatory\nPSP (A, a) → E", 0.39, 0.47, 0.15)
    arr(0.73, 0.40, 0.58, 0.68, "#c0392b", "c4 · inhibitory\nPSP (B, b) → I", 0.61, 0.47, -0.15)
    arr(0.5, 1.0, 0.5, 0.89, "#d68910", "", 0.5, 0.95)
    ax.text(0.5, 0.965, "TMS input  k·u(t)   +   background drive  p   (into E, via the A, a kernel)", fontsize=7.4,
            color="#d68910", ha="center")
    ax.text(0.5, 0.06, "EEG ∝ E − I   (sigmoid  S(v) = vmax / (1 + exp(r·(v0 − v))) converts potential to firing rate)",
            fontsize=7.4, ha="center")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_kernels_sigmoid(path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    t = np.linspace(0, 0.15, 600)
    ax = axes[0]
    for A_, a_, ls, lab in ((3.25, 101, "-", "nominal A = 3.25 mV, a = 101/s"), (3.25 * 1.3, 101, "--", "A × 1.3"),
                            (3.25, 101 * 0.7, ":", "a × 0.7 (slower)")):
        ax.plot(t * 1e3, A_ * a_ * t * np.exp(-a_ * t), color="#1f6fb4", ls=ls, label=lab)
    ax.set_title("Excitatory PSP kernel  h(t) = A·a·t·e^(−a·t)"); ax.set_xlabel("ms"); ax.set_ylabel("mV per unit input·s")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    for B_, b_, ls, lab in ((22, 51, "-", "nominal B = 22 mV, b = 51/s"), (22, 51 * 0.7, ":", "b × 0.7 (slower)")):
        ax.plot(t * 1e3, B_ * b_ * t * np.exp(-b_ * t), color="#c0392b", ls=ls, label=lab)
    ax.set_title("Inhibitory PSP kernel  (B, b)"); ax.set_xlabel("ms"); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[2]
    v = np.linspace(-6, 16, 400)
    S = NOM["vmax"] / (1 + np.exp(NOM["r"] * (NOM["v0"] - v)))
    ax.plot(v, S, color="k")
    ax.axhline(NOM["vmax"], color="0.7", ls=":", lw=0.8); ax.text(-5.5, 4.75, "vmax = 5 /s", fontsize=6.8)
    ax.axvline(NOM["v0"], color="0.7", ls=":", lw=0.8); ax.text(6.3, 0.3, "v0 = 6 mV", fontsize=6.8)
    for p_, c_ in ((0, "#555"), (110, "#2a9d8f"), (150, "#d62828")):
        vv, ss, _ = fixed_point(p_)
        ax.plot(vv, ss, "o", color=c_, ms=6)
        off = {0: (-4.5, 1.2), 110: (1.6, 0.9), 150: (2.0, 0.9)}[p_]
        ax.annotate(f"rest at p = {p_}", (vv, ss), (vv + off[0], ss + off[1]),
                    fontsize=6.8, color=c_, arrowprops=dict(arrowstyle="-", color=c_, lw=0.6))
    ax.set_title("Sigmoid: potential → firing rate (slope set by r)"); ax.set_xlabel("membrane potential v (mV)")
    ax.set_ylabel("firing rate (/s)")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_operating_point(path, ts):
    ps = np.arange(0, 241, 5)
    vs, fs, sl = zip(*[fixed_point(p_) for p_ in ps])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax = axes[0]
    ax.plot(ps, vs, color="k"); ax.axhline(NOM["v0"], color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("background drive p"); ax.set_ylabel("resting E − I (mV)"); ax.set_title("Operating point")
    ax2 = ax.twinx(); ax2.plot(ps, sl, color="#d62828", lw=1); ax2.set_ylabel("sigmoid slope", color="#d62828")
    ax2.spines["right"].set_visible(True)
    for a_ in (ax,):
        for p_, c_ in ((110, "#2a9d8f"), (120, "#e9c46a"), (150, "#d62828")):
            a_.axvline(p_, color=c_, lw=0.8, ls="--")
    ax = axes[1]
    p_ = np.array(ts["p"]); se = np.array(ts["SP1000"]) * 100
    for j, n in enumerate(FIT):
        if n in ("b", "c1", "c3", "c4"):
            ax.semilogy(p_, se[:, j], label=n)
    for pv, c_ in ((110, "#2a9d8f"), (120, "#e9c46a"), (150, "#d62828")):
        ax.axvline(pv, color=c_, lw=0.8, ls="--")
    ax.set_xlabel("background drive p"); ax.set_ylabel("Cramér–Rao bound (%)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("What one pulse can identify")
    ax = axes[2]
    yy = np.array(ts["y"]); t = np.arange(400) - node.ONSET
    for pv, c_ in ((0, "#555"), (110, "#2a9d8f"), (150, "#d62828")):
        ax.plot(t, yy[list(p_).index(pv)], color=c_, lw=1, label=f"p = {pv}")
    ax.set_xlim(-10, 290); ax.set_ylim(-1.6, 2.5); ax.set_xlabel("ms after pulse"); ax.set_ylabel("µV (zoom)")
    ax.legend(fontsize=7, frameon=False); ax.set_title("Late response (zoom)")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_identifiability(path, an, sm, ts):
    names = list(FIT) + ["A", "B", "vmax", "v0", "r"]
    cls = {"k": "stiff", "a": "stiff", "b": "conditional", "c1": "sloppy", "c2": "sloppy", "c3": "conditional",
           "c4": "sloppy", "A": "gauge", "B": "gauge", "vmax": "gauge", "v0": "sloppy", "r": "sloppy"}
    i120 = ts["p"].index(120)
    cap = 2000
    rows = {"single pulse, at rest": [an["crb"]["SP1000"][0]["se7"][FIT.index(n)] * 100 if n in FIT else
                                      an["crb"]["SP1000"][0]["se12"][ALL.index(n)] * 100 for n in names],
            "rich protocol, at rest": [an["crb"]["RICH"][0]["se7"][FIT.index(n)] * 100 if n in FIT else
                                       an["crb"]["RICH"][0]["se12"][ALL.index(n)] * 100 for n in names],
            "single pulse, p = 120": [ts["SP1000"][i120][FIT.index(n)] * 100 if n in FIT else np.nan for n in names]}
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    x = np.arange(len(names))
    for k_, (lab, v) in enumerate(rows.items()):
        v = np.minimum(np.array(v, float), cap)
        ax.bar(x + (k_ - 1) * 0.26, v, 0.25, label=lab, color=["#264653", "#2a9d8f", "#d62828"][k_],
               alpha=0.9)
    ax.set_yscale("log"); ax.set_ylim(0.08, 3000)
    ax.axhline(20, color="k", ls=":", lw=0.8); ax.text(len(names) - 0.6, 23, "20%", fontsize=7)
    pr = sm["progress_median"]["E1"]
    ax.set_xticks(x, [f"{n}\n{pr[n]:+.2f}" if n in pr else f"{n}\n—" for n in names])
    for i, n in enumerate(names):
        ax.get_xticklabels()[i].set_color(CLS_COL[cls[n]])
        ax.get_xticklabels()[i].set_fontweight("bold")
    ax.text(-0.75, 0.0, "backprop\nprogress →", transform=ax.get_xaxis_transform(), fontsize=6.5, ha="right",
            va="top", color="#444")
    ax.set_ylabel("best achievable error (%, log)")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title("How precisely the TMS-EEG data can pin down each parameter (Cramér–Rao bound, 0.42 µV noise). "
                 "Bars capped at 2000%.", fontsize=8.5)
    for c_, lab in zip(CLS_COL.values(), ("stiff", "conditional", "sloppy", "gauge")):
        ax.plot([], [], "s", color=c_, label=lab)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=6.8, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_gauge(path):
    st = [node.STIMS["SP1000"], node.STIMS["EI1000"]]
    base = {n: torch.tensor([NOM[n]], dtype=torch.float64) for n in ALL}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    with torch.no_grad():
        y0 = node.simulate(base, st, gain=248.36).numpy()[0]
        t = np.arange(400) - node.ONSET
        for lam, c_ in ((0.7, "#e76f51"), (1.5, "#2a9d8f")):
            g1 = dict(base); g2 = dict(base); g3 = dict(base)
            for n in ("A", "B"):
                g1[n] = base[n] * lam
            for n in ("c1", "c2", "c3", "c4", "k"):
                g1[n] = base[n] / lam
            g2["vmax"] = base["vmax"] * lam; g2["A"] = base["A"] / lam; g2["B"] = base["B"] / lam
            g2["k"] = base["k"] * lam
            g3["A"] = base["A"] * lam                    # A alone: NOT a symmetry, for contrast
            y1 = node.simulate(g1, st, gain=248.36).numpy()[0]
            y2 = node.simulate(g2, st, gain=248.36).numpy()[0]
            y3 = node.simulate(g3, st, gain=248.36).numpy()[0]
            axes[0].plot(t, y1[0], color=c_, lw=1, ls="--", label=f"λ = {lam}")
            axes[1].plot(t, y2[0], color=c_, lw=1, ls="--", label=f"λ = {lam}")
            axes[2].plot(t, y3[0], color=c_, lw=1, label=f"A × {lam} only")
    for ax, ttl in zip(axes, ("A, B × λ  and  c1–c4, k ÷ λ", "vmax × λ, A, B ÷ λ, k × λ", "contrast: A × λ alone")):
        ax.plot(t, y0[0], color="k", lw=1.6, alpha=0.5, label="nominal", zorder=0)
        ax.set_xlim(-10, 200); ax.set_title(ttl); ax.set_xlabel("ms"); ax.legend(fontsize=6.8, frameon=False)
    axes[0].set_ylabel("EEG (µV)")
    fig.suptitle("Gauge symmetries: very different parameter sets, the same EEG (single pulse)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_combos(path, J, sigma):
    rows = {p_: combo_crb(J, PROTOCOLS[p_], sigma) for p_ in ("SP1000", "DOSE", "RICH")}
    fig, ax = plt.subplots(figsize=(7, 2.9))
    labs = ["c1·c2", "c1/c2", "c3·c4", "c3/c4"]
    x = np.arange(len(labs))
    for k_, (p_, r_) in enumerate(rows.items()):
        ax.bar(x + (k_ - 1) * 0.26, [min(r_[l], 2000) for l in labs], 0.25, label=p_,
               color=["#264653", "#e9c46a", "#2a9d8f"][k_])
    ax.set_yscale("log"); ax.set_xticks(x, labs); ax.axhline(20, color="k", ls=":", lw=0.8)
    ax.set_ylabel("Cramér–Rao bound (%)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Products of connection constants are better identified than ratios (at rest)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return rows


def img(path, caption, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP)])


# ============================================================================ document
REMOTE_FIG = FIG / "ei_phase_from_remote_p2_0.png"     # extracted from reports/ei_balance_bifurcation_map.pdf


def extract_remote_figure():
    """The E/I phase-diagram data live on the machine that ran it; reuse the figure embedded in its PDF."""
    if REMOTE_FIG.exists():
        return True
    try:
        import fitz                                         # PyMuPDF, optional
    except ImportError:
        return False
    d = fitz.open(SYN_ROOT / "reports" / "ei_balance_bifurcation_map.pdf")
    for page in d:
        for im in page.get_images(full=True):
            pix = fitz.Pixmap(d, im[0])
            if pix.width > 1500:
                (fitz.Pixmap(fitz.csRGB, pix) if pix.n > 4 else pix).save(REMOTE_FIG)
                return True
    return False


def ei_section():
    s = [PageBreak(), P("5  E/I balance and the bifurcations around it", H1)]
    s.append(P("5.1  What E/I balance means in the model", H2))
    s.append(P(
        "Excitation and inhibition reach the pyramidal cells through two loops: the excitatory loop (c1, c2) and the "
        "inhibitory loop (c3, c4). Following reports/ei_balance_bifurcation_map.pdf, a single balance multiplier m "
        "shifts weight between them while keeping their geometric mean fixed: c1, c2 × √m and c3, c4 ÷ √m. The loop "
        "strengths then scale as c1·c2 ∝ m and c3·c4 ∝ 1/m, so <b>m² is the ratio of excitatory to inhibitory loop "
        "strength</b>. m = 1 is the nominal, balanced column; m &gt; 1 is 'too much excitation' (as with lost or weakened "
        "inhibition); m &lt; 1 is 'too much inhibition'. Together with the background drive p, m decides which state the "
        "column is in."))
    s.append(P("5.2  Three states and two kinds of transition", H2))
    s.append(P(
        "For every (p, m) the equations were solved for all resting states, and each was tested for stability from the "
        "eigenvalues of the linearised equations. The eigenvalue with the largest real part also gives the column's "
        "ringing frequency and how fast that ringing dies out (the damping)."))
    s += bullets([
        "<b>Quiet</b>: the pyramidal cells rest well below threshold; the column behaves like a linear filter and the "
        "loops are invisible.",
        "<b>Active</b>: the column rests near threshold (E − I ≈ 6 mV), the sigmoid is steep, the loops take part in the "
        "response, and a pulse is followed by damped ringing at 6–11 Hz.",
        "<b>Sustained oscillation</b> (limit cycle): no stable resting state exists; the column oscillates on its own, "
        "a model of abnormal rhythmic activity.",
        "<b>Fold</b> (quiet ↔ active): over a range of p the quiet and the active state <i>both</i> exist, separated by "
        "an unstable state. The quiet state disappears at the upper end, the active state at the lower end. Inside "
        "the range the column's state depends on its history: hysteresis.",
        "<b>Hopf</b> (active → oscillation): the damping of the active state's ringing falls to zero and the ringing "
        "becomes self-sustaining. Approaching it, ringing lasts longer and longer (critical slowing down), which is "
        "also why slowly decaying ringing is easily mistaken for a sustained oscillation.",
    ])
    fig_bifurcation(FIG / "bifurcation.png")
    s.append(img(FIG / "bifurcation.png", "Figure 6. Bifurcation diagrams for an inhibition-dominant, the balanced and an "
                                          "excitation-dominant column. Top: every resting state against background drive "
                                          "(black = stable, red circles = unstable; yellow band = quiet and active states "
                                          "coexist; stars = the three points where backprop was tested). Bottom: damping "
                                          "(below zero = ringing dies out) and ringing frequency of the active state (of the only state where there is "
                                          "one); the "
                                          "dashed line is the damping as neurofe's Euler integrator effectively sees it. "
                                          "Excitation shifts the quiet-to-active transition to lower drive; inhibition "
                                          "shifts it higher."))
    s.append(PageBreak())
    s.append(P("5.3  The E/I phase map", H2))
    have = extract_remote_figure()
    if have:
        s.append(img(REMOTE_FIG, "Figure 7. The E/I phase map from reports/ei_balance_bifurcation_map.pdf (simulated "
                                 "from a cold start on a 15 × 19 grid). Top: state, E/I dominance of the active state, "
                                 "oscillation amplitude. Bottom: Cramér–Rao bound on c1·c2 for a single pulse and for the "
                                 "rich protocol, and the median bounds against drive. It shows where TMS can and cannot "
                                 "identify the loops."))
    fig_stability_map(FIG / "stability_map.png")
    s.append(img(FIG / "stability_map.png", "Figure 8. The same plane computed from the equations instead of cold-start "
                                            "simulation. It agrees with Figure 7 on the main structure and adds two "
                                            "things a cold-start scan cannot see: the bistable band along the "
                                            "quiet/active boundary, and which oscillations exist only because of "
                                            "neurofe's integration step.", width=0.85 * W))
    s += bullets([
        "<b>Agreement.</b> The quiet/active boundary bends with balance: an excitation-dominant column becomes active at "
        "much lower drive, an inhibition-dominant one needs more. Genuine sustained oscillations sit in a band of "
        "slightly inhibition-leaning columns (m ≈ 0.55–0.95) at drive above p ≈ 160, at 6–11 Hz.",
        "<b>Bistability along the boundary.</b> For the balanced column the quiet and the active state coexist from "
        "p ≈ 37 to 115 (Figure 9). A cold-start scan only ever finds the quiet one there, which is why Figure 7 shows "
        "these cells as quiet. Excitation-dominant columns are bistable already at zero drive: once activated, they "
        "can stay active with no input at all.",
        "<b>The oscillation of the balanced column at very high drive is a numerical effect.</b> At m = 1 the active "
        "state stays stable up to p ≈ 280 in the equations, but its damping becomes so weak (−0.1 /s at p = 270) that "
        "neurofe's Euler step of 0.1 ms, which slightly amplifies weakly damped oscillations (by about ω²·dt/2 ≈ "
        "+0.24 /s at 11 Hz), turns it into a sustained 1.2 mV oscillation. With a 10× smaller step it decays (Table 4). "
        "The oscillations of inhibition-leaning columns are real at both step sizes.",
    ])
    s.append(P("5.4  Hysteresis at the quiet/active transition", H2))
    h = fig_hysteresis(FIG / "hysteresis.png")
    s.append(img(FIG / "hysteresis.png", "Figure 9. Balanced column: settle in the active state at p = 150, then lower "
                                         "the drive step by step (red), and compare with a fresh start at each drive "
                                         "(blue). Inside the yellow band the column stays active although a fresh start "
                                         "is quiet: its state depends on its history.", width=0.72 * W))
    s.append(P(
        f"Lowering the drive from the active state, the column stays active far below the drive that switched it on "
        f"(E − I = {h['ramp'][h['down'].index(50)]:.2f} mV at p = 50, against {h['cold'][h['down'].index(50)]:.2f} mV from a "
        f"fresh start) and drops to the quiet state only between p = 50 and 40. In the equations the active state exists "
        f"down to p ≈ {h['lo']:.0f}, but near that edge it is barely separated from the unstable state, so a step in drive "
        f"knocks it over a little earlier. Raising the drive from the quiet state, it stays quiet up to p ≈ 115. The E/I balance report found no hysteresis when leaving a <i>sustained oscillation</i> by lowering "
        "the drive, and this analysis agrees: the oscillation band has no coexisting stable states in the equations "
        "(Figure 8). The hysteresis is at the quiet/active boundary, which that test did not cross. Physically, an "
        "active column can be held active by much weaker input than it took to switch it on."))
    dtc = dt_check(FIG / "dt_check.json")
    rows = [["Column", "Step 0.1 ms (neurofe): oscillation at 5 / 15 / 30 s", "Step 0.01 ms", "Verdict"]]
    for k_, verdict in (("m=1.0 p=270", "numerical: decays with the finer step"),
                        ("m=1.0 p=255", "numerical: decays faster with the finer step"),
                        ("m=0.8 p=200", "real limit cycle at both step sizes")):
        rows.append([k_.replace("m=", "m = ").replace(" p=", ", p = "),
                     " / ".join(f"{v:.2g}" for v in dtc["0.0001"][k_]) + " mV",
                     " / ".join(f"{v:.2g}" for v in dtc["1e-05"][k_]) + " mV", verdict])
    s += [table(rows, [0.17, 0.33, 0.22, 0.28]),
          P("Table 4. Peak-to-peak oscillation from a cold start with neurofe's 0.1 ms Euler step and a 10× finer step. "
            "Near a Hopf transition the integrator, not the model, can decide whether the column oscillates. This "
            "matters for synthetic tests where teacher and student use different time steps.", CAP)]
    s.append(P("5.5  Identifiability across the map", H2))
    s.append(table([["Region (median over active points)", "c1·c2, single pulse", "c1·c2, rich", "c3·c4, single pulse",
                     "c3·c4, rich"],
                    ["balanced (0.77 &lt; m &lt; 1.3)", "24%", "9%", "22%", "5%"],
                    ["E-dominant (m ≥ 1.3, too much E)", "2807%", "19%", "3329%", "10%"],
                    ["I-dominant (m ≤ 0.77, too much I)", "110%", "15%", "36%", "4%"]],
                   [0.34, 0.165, 0.165, 0.165, 0.165]))
    s.append(P("Table 5. Cramér–Rao bounds by region (from reports/ei_balance_bifurcation_map.pdf, Table 2). "
               "Rich = SP1000 + paired pulse + 10 Hz train + E+I injection.", CAP))
    s.append(KeepTogether([table([["Point (m, p)", "Protocol", "c1·c2 backprop", "c3·c4 backprop"],
                    ["balanced (1.0, 150)", "single pulse / rich", "19.3% / 19.1%", "15.0% / 9.1%"],
                    ["E-dominant (1.87, 105)", "single pulse / rich", "48.8% / 39.8%", "23.7% / 20.5%"],
                    ["I-dominant (0.53, 240)", "single pulse / rich", "47.0% / 22.0%", "15.9% / 6.9%"]],
                   [0.3, 0.24, 0.23, 0.23]),
                           P("Table 6. Median backprop error, 10 random starts (from the same report, Table 1). The 20% "
                             "gate is met for c3·c4 at two of three points with the rich protocol, for c1·c2 only at "
                             "balance.", CAP)]))
    rows = [["Point (m, p)", "Active E − I", "Ringing", "Damping", "Exc. interneuron slope", "Inh. interneuron slope"]]
    for name, (mm, pp) in STUDY_POINTS.items():
        e = max(equilibria(pp, mm), key=lambda q: q["v"])
        rows.append([f"{name} ({mm}, {pp})", f"{e['v']:.2f} mV", f"{e['f']:.1f} Hz", f"{e['re']:.1f} /s",
                     f"{e['sE']:.3f}", f"{e['sI']:.3f}"])
    s += [table(rows, [0.24, 0.13, 0.11, 0.12, 0.2, 0.2]),
          P("Table 7. Why identifiability differs: the state of the column at the three tested points (sigmoid slopes, "
            "maximum 0.70). A loop is only visible in the EEG if its interneurons sit on the steep part of their sigmoid.",
            CAP)]
    s.append(P("Mechanisms", H2))
    s += bullets([
        "<b>Too much excitation saturates the excitatory loop.</b> At the E-dominant point the excitatory interneurons "
        "fire at their maximum: their sigmoid slope is 0.000. Changing c1 or c2 then changes nothing the pyramidal cells "
        "receive, so the excitatory loop strength L<sub>E</sub> = A·c1·c2·S'<sub>E</sub> is invisible to any stimulus "
        "that does not first pull the column out of saturation. That is why a single pulse gives bounds in the "
        "thousands of percent there, and why c1·c2 stays the hardest quantity even with the rich protocol (whose E+I "
        "injection partly does pull the column out of saturation).",
        "<b>Strong damping shortens the information-carrying ringing.</b> The E-dominant column damps its ringing at "
        "−17 /s (gone in ~60 ms); the balanced column at −2.2 /s (rings for ~0.5 s). Less ringing means fewer cycles "
        "carrying information about the loops.",
        "<b>Near the transitions, fits are fragile.</b> Close to the fold a student with slightly wrong c's can sit on "
        "the other branch (the p = 120 failures, section 4). Close to the Hopf transition the response depends on tiny "
        "parameter or integrator differences (Table 4). The well-behaved region for fitting is the active state away "
        "from both transitions.",
    ])
    s.append(P("5.6  What this means for an E/I-balance biomarker", H2))
    s += bullets([
        "The clinically interesting state, too much excitation, is exactly where a conventional single-pulse protocol "
        "is least informative; the reason is saturation of the excitatory interneurons, not noise. Richer protocols "
        "recover most of the inhibitory loop but not the excitatory one.",
        "The quantity that is robustly measurable across the map is the inhibitory loop strength c3·c4 with a rich "
        "protocol; the excitatory loop c1·c2 is measurable only near balance.",
        "The state of a column (quiet or active) is not determined by its parameters alone: in the bistable band it "
        "depends on history. Two recordings with the same parameters and the same input can differ, and a fit can "
        "converge to the wrong branch.",
        "Sustained oscillations near the Hopf boundary are a natural model of abnormal rhythms, but whether a weakly "
        "damped column oscillates in simulation can depend on the integration step; check with a finer step before "
        "interpreting.",
    ])
    return s


def content(an, sm, ts, combos):
    c7 = an["crb"]
    i120, i150 = ts["p"].index(120), ts["p"].index(150)
    se = lambda prot, n: c7[prot][0]["se7"][FIT.index(n)] * 100                     # noqa: E731
    s120 = lambda n: ts["SP1000"][i120][FIT.index(n)] * 100                          # noqa: E731
    s150 = lambda n: ts["SP1000"][i150][FIT.index(n)] * 100                          # noqa: E731
    se12 = lambda prot, n: c7[prot][0]["se12"][ALL.index(n)] * 100                   # noqa: E731
    pr = sm["progress_median"]["E1"]
    e1 = sm["E1"]
    f0 = an["features_nominal"]
    s = []
    s += [P("The Jansen-Rit node: what each parameter means", TITLE),
          P("and which ones TMS-EEG can identify, with a focus on E/I balance and the bifurcations around it", _H2),
          P(f"syn_brain explainer · {datetime.date.today():%d %B %Y} · based on the single-node virtual-TMS study "
            "(reports/single_node_tms_study.pdf)", CAP)]
    s.append(P(
        "A Jansen-Rit (JR) node models one cortical column as three interacting neural populations. Each parameter has "
        "a physical meaning, but not every parameter can be measured from EEG. This note explains each parameter in "
        "turn, shows what it does to the TMS response, and states, with numbers from the synthetic teacher/student "
        "study, whether it can be identified. Section 5 is the main focus: how the balance between excitation and "
        "inhibition decides whether the column is quiet, active or oscillating, where it switches between these "
        "states, and what that means for identification. Four classes recur throughout:", BOX))
    s.append(table([["Class", "Meaning", "Parameters"],
                    ["<font color='#2a9d8f'><b>Stiff</b></font>", "pinned by almost any TMS-EEG recording", "k, a"],
                    ["<font color='#b8860b'><b>Conditional</b></font>",
                     "identifiable only with the right stimulus or brain state", "b, c3 (c3·c4), p"],
                    ["<font color='#e76f51'><b>Sloppy</b></font>",
                     "changes by large factors hardly change the EEG", "c1, c2, c4, v0, r"],
                    ["<font color='#9d0208'><b>Gauge</b></font>",
                     "no data can identify them; fixed by convention", "A, B, vmax"]], [0.15, 0.6, 0.25]))

    # -------------------------------------------------------------- equations
    s.append(P("1  The nominal JR equations (as implemented in neurofe)", H1))
    s.append(P(
        "Three state variables are postsynaptic potentials: <b>P</b>, the potential the pyramidal cells' output "
        "induces in the interneurons; <b>E</b>, the excitatory potential on the pyramidal cells; <b>I</b>, the "
        "inhibitory potential on the pyramidal cells. The pyramidal membrane potential is v = E − I, and the EEG is "
        "proportional to it. Each potential obeys a second-order synaptic filter driven by a firing rate:"))
    s.append(P("d²P/dt² = A·a · S(E − I) − 2a · dP/dt − a² · P", EQ))
    s.append(P("d²E/dt² = A·a · [ c2 · S(c1 · P) + k · u(t) + p ] − 2a · dE/dt − a² · E", EQ))
    s.append(P("d²I/dt² = B·b · c4 · S(c3 · P) − 2b · dI/dt − b² · I", EQ))
    s.append(P("S(v) = vmax / (1 + exp( r · (v0 − v) ))          EEG(t) = 5·10<super>−4</super> · leadfield · (E − I)", EQ))
    s.append(P(
        "u(t) is the injected TMS current (a 10 ms pulse of amplitude 1000 in neurofe), k converts it into synaptic "
        "drive, and p is a constant background drive (zero in neurofe; see section 4). neurofe additionally bounds each "
        "firing-rate input with 500·tanh(·/500), integrates with Euler steps of 0.1 ms, and fits effective values such "
        "as a = 1 + relu(100) = 101 /s. In a network, the long-range coupling g·Σ<sub>j</sub> w<sub>ij</sub>"
        "(E<sub>j</sub>(t − d<sub>ij</sub>) − E<sub>i</sub>) is added to the E equation."))
    FIG.mkdir(parents=True, exist_ok=True)
    fig_column(FIG / "column.png")
    s.append(img(FIG / "column.png", "Figure 1. The JR column. Arrows carry firing rates; each arrow ends in a synaptic "
                                     "filter (A, a for excitation, B, b for inhibition) scaled by a connection constant c.",
                 width=0.85 * W))
    s.append(KeepTogether([table([["Parameter", "Nominal", "Unit", "Physical meaning", "Fitted in neurofe?"],
                    ["k", "10.5", "—", "stimulus gain: TMS current → synaptic drive", "yes"],
                    ["a", "101", "1/s", "excitatory synaptic rate; τ<sub>e</sub> = 1/a ≈ 10 ms", "yes"],
                    ["b", "51", "1/s", "inhibitory synaptic rate; τ<sub>i</sub> = 1/b ≈ 20 ms", "yes"],
                    ["c1, c2", "135, 108", "—", "pyramidal ↔ excitatory interneuron contacts (loop)", "yes"],
                    ["c3, c4", "33.75, 33.75", "—", "pyramidal ↔ inhibitory interneuron contacts (loop)", "yes"],
                    ["A", "3.25", "mV", "maximum excitatory PSP amplitude", "no (fixed)"],
                    ["B", "22", "mV", "maximum inhibitory PSP amplitude", "no (fixed)"],
                    ["vmax", "5", "1/s", "maximum population firing rate", "no (fixed)"],
                    ["v0", "6", "mV", "firing threshold (half-maximal firing)", "no (fixed)"],
                    ["r", "0.56", "1/mV", "sigmoid steepness (spread of thresholds)", "no (fixed)"],
                    ["p", "0 (neurofe)", "1/s", "background input from the rest of the brain", "not present"]],
                   [0.12, 0.13, 0.08, 0.48, 0.19]),
                           P("Table 1. Parameters. The c's follow Jansen and Rit's anatomical ratios 1 : 0.8 : 0.25 : "
                             "0.25 of C = 135.", CAP)]))
    fig_kernels_sigmoid(FIG / "kernels.png")
    s.append(img(FIG / "kernels.png", "Figure 2. The building blocks. Left, middle: the synaptic kernels; A and B set "
                                      "the height, a and b how fast the response rises and decays. Right: the sigmoid "
                                      "and where the node rests for three background drives. At p = 0 it rests far "
                                      "below threshold, where the curve is almost flat."))

    # -------------------------------------------------------------- overview
    s.append(PageBreak())
    s.append(P("2  The overview: what changes the response, and what can be identified", H1))
    s.append(img(OUT / "fig_oneatatime.png", "Figure 3. Single-pulse EEG response when one parameter is changed by "
                                             "±30% (insets: the late tail, magnified). Only k, A and a visibly change the "
                                             "main peak; all other parameters only change a sub-microvolt late tail."))
    fig_identifiability(FIG / "identifiability.png", an, sm, ts)
    s.append(img(FIG / "identifiability.png", "Figure 4. Best achievable relative error per parameter (lower is better) "
                                              "for one neurofe pulse at rest, for the rich multi-condition protocol at "
                                              "rest, and for one pulse at background drive p = 120. Gauge and fixed "
                                              "parameters (A, B, vmax, v0, r) are shown with all 12 parameters free. "
                                              "Numbers under the axis: median backprop progress toward the truth "
                                              "(1 = reached, 0 = did not move, negative = moved away). Label colour = class."))
    s.append(P(
        "Two facts organise everything below. First, at rest the node is almost a linear filter: the pyramidal "
        "potential sits at −1.9 mV, far below the 6 mV threshold, where the sigmoid slope is 0.03 of a possible 0.7, so "
        "the interneuron loop barely participates. Second, the EEG can only see certain <i>combinations</i> of "
        "parameters. Linearising the equations gives the response to an input U:"))
    s.append(P("V(s) = A·k·H<sub>a</sub>(s)·U(s) / ( 1 − A·S'·H<sub>a</sub>(s)·[ L<sub>E</sub>·H<sub>a</sub>(s) − "
               "L<sub>I</sub>·H<sub>b</sub>(s) ] ),   L<sub>E</sub> = A·c1·c2·S'<sub>E</sub>,   "
               "L<sub>I</sub> = B·c3·c4·S'<sub>I</sub>", EQ))
    s.append(P(
        "with H<sub>a</sub> = a/(s+a)<super>2</super>, H<sub>b</sub> = b/(s+b)<super>2</super> and S' the sigmoid "
        "slopes at rest. Only A·k, a, b and the two loop gains L<sub>E</sub>, L<sub>I</sub> are visible; the loop "
        "gains are multiplied by S', which is tiny at rest."))

    # -------------------------------------------------------------- per parameter
    s.append(PageBreak())
    s.append(P("3  Parameter by parameter", H1))

    def section(title, meaning, equation, eeg, ident, verdict, cls):
        col = {"stiff": "#2a9d8f", "conditional": "#b8860b", "sloppy": "#e76f51", "gauge": "#9d0208"}[cls]
        return [P(f"{title}   <font color='{col}' size='8.5'>[{cls}]</font>", H2),
                table([["Physical meaning", meaning], ["Where it acts", equation], ["What it does to the EEG", eeg],
                       ["Identifiability (study results)", ident], ["Verdict", verdict]], [0.24, 0.76], header=False,
                      zebra=False, font=7.6)]

    s += section("k — stimulus gain",
                 "How strongly the TMS pulse drives the column. Set by the coil, its orientation, and the distance "
                 "from scalp to cortex. A property of the experiment and the head, not of the brain circuit.",
                 "Multiplies the input u(t) in the E equation.",
                 f"Scales the whole response: elasticity of the P1 peak +0.99 (P1 = {f0['P1 amplitude']:.1f} µV at "
                 "nominal). Does not change its shape.",
                 f"Bound {pct(se('SP1000', 'k'))} from one pulse; backprop error {pct(e1['SP1000']['median_abs_err_pct']['k'])}, "
                 f"progress {pr['k']:+.2f}. Exactly degenerate with A: only A·k is measured (Figure 10).",
                 "Identified, but as a nuisance. Cortical atrophy increases scalp-to-cortex distance and lowers k, so a "
                 "smaller response cannot be blamed on the circuit unless k is fixed independently (e.g. MRI-based "
                 "E-field modelling).", "stiff")
    s += section("a — excitatory synaptic rate (τ<sub>e</sub> = 1/a ≈ 10 ms)",
                 "Speed of excitatory (glutamatergic) synaptic transmission and dendritic integration.",
                 "Rate constant of the excitatory kernel in the P and E equations.",
                 f"Sets the timing of the first peak: elasticity of P1 latency −0.64 (P1 at {f0['P1 latency']:.0f} ms). "
                 "Smaller a = slower, broader response.",
                 f"Bound {pct(se('SP1000', 'a'))} from one pulse, {pct(se('RICH', 'a'))} with the rich protocol; backprop "
                 f"error {pct(e1['SP1000']['median_abs_err_pct']['a'])}, progress {pr['a']:+.2f}.",
                 "The best-identified circuit parameter. A reliable marker of excitatory kinetics.", "stiff")
    s += section("b — inhibitory synaptic rate (τ<sub>i</sub> = 1/b ≈ 20 ms)",
                 "Speed of GABA-A-type inhibition; changed by e.g. benzodiazepines or GABAergic dysfunction.",
                 "Rate constant of the inhibitory kernel in the I equation.",
                 "At rest it only shapes the small late dip (elasticity of N1 amplitude −3.2, but N1 is only "
                 f"{f0['N1 amplitude']:.2f} µV). With TMS also injected into inhibitory cells it shapes the main "
                 "response.",
                 f"Bound {pct(se('SP1000', 'b'))} from one pulse, {pct(se('EI1000', 'b'))} with E+I injection, "
                 f"{pct(s120('b'))} at p = 120; backprop progress {pr['b']:+.2f}.",
                 "Identifiable with the right stimulus (E+I targeting, several conditions) or an active state; poorly "
                 "constrained by a single pulse at rest.", "conditional")
    s += section("c1, c2 — the excitatory loop",
                 "Average number of synaptic contacts: pyramidal → excitatory interneurons (c1) and back (c2). Together "
                 "they set how much the column re-excites itself. Synapse loss (e.g. in neurodegeneration) would "
                 "reduce them.",
                 "c1 inside the interneuron sigmoid S(c1·P); c2 scales its output into E.",
                 "Only through the loop gain L<sub>E</sub> ∝ c1·c2, multiplied by the tiny resting sigmoid slope: "
                 "a sub-microvolt change in the late tail.",
                 f"Bounds c1 {pct(se('SP1000', 'c1'))}, c2 {pct(se('SP1000', 'c2'))} from one pulse; product c1·c2 "
                 f"{pct(combos['SP1000']['c1·c2'])}, ratio c1/c2 {pct(combos['SP1000']['c1/c2'])}. At p = 120: c1 "
                 f"{pct(s120('c1'))}, c2 {pct(s120('c2'))}. Backprop progress c1 {pr['c1']:+.2f}, c2 {pr['c2']:+.2f} "
                 "(c2 moves away from the truth).",
                 "Not identifiable individually at rest. Interpret only the product c1·c2, and only in an active state. "
                 "The hardest loop to recover even at p = 150 (one of 4 perturbed teachers fails at 91–95%).", "sloppy")
    s += section("c3, c4 — the inhibitory loop",
                 "Contacts pyramidal → inhibitory interneurons (c3) and back (c4): the strength of feedback inhibition. "
                 "Interneuron loss or dysfunction would reduce them.",
                 "c3 inside S(c3·P); c4 scales the inhibitory interneuron output into I.",
                 "Through the loop gain L<sub>I</sub> ∝ c3·c4: deepens and lengthens the late dip (elasticities "
                 "+3.2 and +1.6 on N1 amplitude).",
                 f"Bounds c3 {pct(se('SP1000', 'c3'))}, c4 {pct(se('SP1000', 'c4'))} at rest; c3·c4 "
                 f"{pct(combos['SP1000']['c3·c4'])}. At p = 120: c3 {pct(s120('c3'))}, c4 {pct(s120('c4'))}. Backprop "
                 f"progress c3 {pr['c3']:+.2f}, c4 {pr['c4']:+.2f}.",
                 "c3 (and c3·c4) become identifiable in principle once the node is in an active state; c4 alone stays "
                 "weak. In practice backprop recovers c3·c4 at p = 150 only inconsistently (7–28% across 4 teachers). "
                 "Interpret c3·c4 as 'inhibitory loop strength', with uncertainty.", "conditional")
    s += section("A, B — maximum PSP amplitudes",
                 "Peak size of a single excitatory (A = 3.25 mV) or inhibitory (B = 22 mV) postsynaptic potential: "
                 "receptor density and synaptic efficacy.",
                 "Gain of the excitatory (A) and inhibitory (B) kernels.",
                 "A scales the whole response like k (elasticity +1.02); B scales the inhibitory part.",
                 f"With all 12 parameters free: A {pct(se12('RICH', 'A'))}, B {pct(se12('RICH', 'B'))} even with the rich "
                 "protocol. Scaling A and B up while scaling the c's and k down leaves the EEG unchanged (Figure 10). "
                 "Backprop with A free recovers k·A but not A or k (Figure 12).",
                 "Never identifiable from EEG: 'bigger synapses' and 'more synapses' look the same. Must stay fixed; "
                 "fitted c's and k are only meaningful relative to them.", "gauge")
    s += section("vmax — maximum firing rate",
                 "Highest average firing rate of a population (5 /s as coded in neurofe).",
                 "Height of the sigmoid S.",
                 "Changes the late dip strongly (elasticity +3.6 on N1), not the main peak.",
                 f"Bound {pct(se12('RICH', 'vmax'))} with 12 free. Trades off exactly against A, B and k "
                 "(vmax × λ with A, B ÷ λ and k × λ gives the same EEG).",
                 "Gauge: fix by convention.", "gauge")
    s += section("v0, r — firing threshold and sigmoid slope",
                 "v0: the mean membrane potential at which half the population fires (excitability). r: how spread "
                 "the individual thresholds are (how sharp the population switches on).",
                 "Position (v0) and steepness (r) of the sigmoid S.",
                 "Both decide how far the resting state is from threshold, and therefore whether the loop takes part "
                 "(elasticities on N1: v0 −3.1, r −0.4).",
                 f"With 12 free, rich protocol: v0 {pct(se12('RICH', 'v0'))}, r {pct(se12('RICH', 'r'))}. v0 trades off "
                 "against the background drive p: raising p or lowering v0 both move the node toward threshold.",
                 "Sloppy; keep fixed. A change in excitability (v0) cannot be told apart from a change in background "
                 "input (p) without extra data.", "sloppy")

    # -------------------------------------------------------------- p
    s.append(PageBreak())
    s.append(P("4  p — the background drive (operating point)", H1))
    s.append(P(
        "p is the steady input a column receives from the rest of the brain: firing arriving from other cortical "
        "areas and the thalamus, independent of the TMS pulse. In Jansen and Rit's original model it is a firing rate "
        "of 120–320 /s. It enters the E equation exactly like the recurrent excitation c2·S(c1·P), whose maximum is "
        "c2·vmax = 540, so p = 150 is about 28% of the strongest possible recurrent drive. In a real brain p reflects "
        "arousal, thalamocortical input and ongoing activity. <b>neurofe has no p:</b> its network coupling is a "
        "difference of potentials, which is zero when all nodes rest in the same state, so every node rests at p = 0."))
    rows = [["p", "Stable state(s): E − I", "Pyramidal firing", "Sigmoid slope", "State"]]
    for p_, state in ((0, "isolated cortex; nearly silent; linear filter"),
                      (50, "quiet from a cold start, but active if it was active before (bistable)"),
                      (110, "bistable; quiet from a cold start (E7)"), (120, "only the active state exists; just "
                                                                         "past the jump"),
                      (150, "firmly active; ~10 Hz damped alpha ringing"),
                      (220, "active; ringing decays slowly (critical slowing down)")):
        st = [e for e in equilibria(p_) if e["stable"]]
        t_ = ei_params(1.0)
        fr = [_S(e["v"], t_) for e in st]
        sl = [t_["r"] * f * (1 - f / t_["vmax"]) for f in fr]
        rows.append([str(p_), " / ".join(f"{e['v']:.2f} mV" for e in st), " / ".join(f"{f:.2f} /s" for f in fr),
                     " / ".join(f"{x:.2f}" for x in sl), state])
    s += [table(rows, [0.06, 0.2, 0.17, 0.13, 0.44]),
          P("Table 2. Stable resting states of the balanced node against background drive (fixed points and "
            "eigenvalues of the equations above). Two values = two stable states (quiet / active). The quiet state "
            "disappears at p ≈ 115; once active, the node stays active down to p ≈ 37 (section 5).", CAP)]
    fig_operating_point(FIG / "operating_point.png", ts)
    s.append(img(FIG / "operating_point.png", "Figure 5. Left: resting potential (black) and sigmoid slope (red) "
                                              "against p; dashed lines mark p = 110, 120, 150. Middle: Cramér–Rao bounds "
                                              "from one pulse fall sharply once the node crosses into the active state. "
                                              "Right: the late response gains a ~10 Hz ringing in the active state."))
    s += bullets([
        f"<b>p = 110 vs 120 vs 150.</b> Below the jump (110) the node is quiet and the loop is invisible (c3 bound "
        f"{pct(ts['SP1000'][ts['p'].index(110)][FIT.index('c3')] * 100)}). At 120 and 150 the node is in almost the same "
        f"state, firing at about half its maximum with the steepest sigmoid, so the bounds are similar (c3 "
        f"{pct(s120('c3'))} and {pct(s150('c3'))}). What differs is the safety margin: 120 is only ~4% above the "
        "point where the quiet state vanishes, so slightly different c's can move that point above 120 and leave a "
        "student in the quiet state; 150 is ~23% above it. Note that at p = 110 the active state also exists: whether "
        "the node is quiet or active there depends on its history (hysteresis, section 5).",
        "<b>Backprop at p = 120 vs 150</b> (E9–E12; reports/bifurcation_followup.pdf and "
        "reports/single_node_conclusions.pdf). At p = 120 the median error on c3·c4 stayed at 52–59%, far above the "
        "17% bound, also after longer settling; warm-starting from p = 150 made the fits bimodal rather than better. "
        "Near the jump a student with slightly wrong parameters can rest on the other (quiet) branch, a cliff in the "
        "loss surface that gradients cannot cross. At p = 150, c3·c4 is recovered, but inconsistently: 7–28% across "
        "4 perturbed teachers, and one teacher's c1·c2 fails at 91–95%. p ≥ 150 is the better operating point, but "
        "even there the loop parameters are not reliably identified, and fits do not generalise to the late window or "
        "to held-out amplitudes.",
        "<b>Identifiability of p itself.</b> If p is unknown it must be fitted, and it trades off with v0 (both move "
        "the node toward threshold). In the study p was treated as known.",
        "<b>Physically:</b> p = 0 is cortex cut off from all input; p ≈ 150 resembles awake resting cortex with an "
        "alpha resonance; p ≈ 115–120 is a tipping point where small changes in input switch the state.",
    ])

    s += ei_section()

    # -------------------------------------------------------------- evidence figures
    s.append(PageBreak())
    s.append(P("6  Evidence: degeneracies and flat directions", H1))
    fig_gauge(FIG / "gauge.png")
    s.append(img(FIG / "gauge.png", "Figure 10. Gauge symmetries. Left and middle: two families of very different "
                                    "parameter sets (scaled by λ = 0.7 or 1.5) give EEG responses indistinguishable "
                                    "from the nominal one. Right, for contrast: changing A alone clearly changes the "
                                    "response. Only the combinations are visible."))
    combo_rows = fig_combos(FIG / "combos.png", np.load(OUT / "jacobian.npz")["J"], an["sigma_uV"])
    s.append(img(FIG / "combos.png", "Figure 11. Precision of products vs ratios of the connection constants. The EEG "
                                     "constrains the loop strengths (products) far better than how they are split "
                                     "between the two connections (ratios).", width=0.72 * W))
    s.append(img(OUT / "fig_e3_extended.png", "Figure 12. When A, B, v0, r and vmax are also fitted (E3), backprop "
                                              "still fits the EEG, but A and k drift along the line A·k = constant "
                                              "(middle): the data fix only their product."))
    s.append(img(OUT / "fig_e5_profile.png", "Figure 13. Profile likelihood: pin one parameter at 0.5–2× its true value, "
                                             "refit the rest. k, a, b rise above the noise threshold (dashed): "
                                             "identifiable. c1–c4 stay below it over most of the range: any value fits "
                                             "the data."))

    # -------------------------------------------------------------- summary
    s.append(P("7  Summary", H1))
    rows = [["Param.", "Meaning in one line", "Class", "Use it as"],
            ["k", "how hard the pulse hits the cortex", "stiff", "per-session nuisance; fix with E-field modelling if "
                                                              "possible"],
            ["a", "excitatory synaptic speed", "stiff", "reliable marker"],
            ["b", "inhibitory synaptic speed", "conditional", "marker with E+I / multi-condition protocols or active "
                                                             "state"],
            ["c1, c2", "excitatory loop contacts", "sloppy", "only as c1·c2, only in an active state; invisible when "
                                                             "excitation dominates (interneurons saturate)"],
            ["c3, c4", "inhibitory loop contacts", "conditional", "as c3·c4 ('inhibitory loop strength') in an active "
                                                                  "state, with a rich protocol"],
            ["m", "E/I balance: excitatory vs inhibitory loop strength", "conditional",
             "decides quiet / active / oscillating and how informative TMS is; identified best near balance"],
            ["A, B", "PSP amplitudes", "gauge", "fixed convention; never interpret"],
            ["vmax", "maximum firing rate", "gauge", "fixed convention"],
            ["v0, r", "threshold, threshold spread", "sloppy", "fixed; v0 trades off with p"],
            ["p", "background input from the rest of the brain", "conditional",
             "sets the operating point; choose p ≈ 150 (active, away from the jump); history-dependent between "
             "p ≈ 37 and 115; missing in neurofe"]]
    s += [table(rows, [0.09, 0.33, 0.13, 0.45]), P("Table 8. One-line summary of every parameter.", CAP)]
    s.append(P(
        "In short: TMS-EEG measures <i>how hard the cortex was hit</i> (k·A) and <i>how fast its synapses respond</i> "
        "(a, and b with the right stimulus). It measures the <i>circuit</i> (the c's) only as loop strengths, and only "
        "when the column is active, which in a model requires a background drive p that neurofe does not have. How "
        "informative TMS is depends strongly on the E/I balance: near balance the loops are measurable with a rich "
        "protocol; when excitation dominates, the excitatory loop saturates and becomes invisible. Absolute synaptic "
        "strength, synapse number and maximum firing rate are never visible to EEG."))
    return s


def footer(c, doc):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.8 * cm, 1.1 * cm, "syn_brain · explainer · Jansen-Rit node parameters and identifiability")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
    c.restoreState()


def main():
    an = json.load(open(OUT / "analysis.json"))
    sm = json.load(open(OUT / "summary.json"))
    ts = an["tonic_sweep"]
    combos = {p_: combo_crb(np.load(OUT / "jacobian.npz")["J"], PROTOCOLS[p_], an["sigma_uV"])
              for p_ in ("SP1000", "DOSE", "RICH")}
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.7 * cm, title="Jansen-Rit node parameters and identifiability",
                            author="syn_brain study")
    doc.build(content(an, sm, ts, combos), onFirstPage=footer, onLaterPages=footer)
    print("wrote", PDF)


if __name__ == "__main__":
    main()
