"""Figures and summary tables for the single-node virtual-TMS study.

Reads results/single_node_tms/{analysis.json, sweep.npz, train_part*.npz};
writes results/single_node_tms/fig_*.png and summary.json.

    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_tms_figures
"""

from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from syn_brain.core import node
from syn_brain.core.paths import save_json
from syn_brain.scripts.single_node_tms import OUT, FIT, ALL, NOM_FIT, PROTOCOLS, E1_PROTOCOLS

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 200, "axes.titlesize": 9, "axes.titleweight": "bold"})
PCOL = {"a": "#1f77b4", "b": "#ff7f0e", "c1": "#2ca02c", "c2": "#d62728", "c3": "#9467bd", "c4": "#8c564b",
        "k": "#e377c2", "A": "#7f7f7f", "B": "#bcbd22", "v0": "#17becf", "r": "#393b79", "vmax": "#ad494a"}
ORDER = node.STIM_ORDER + node.DRIVEN_ORDER
T_MS = np.arange(node.T_WIN) - node.ONSET
SHORT = {"SP250": "SP 250", "SP1000": "SP 1000", "SP4000": "SP 4000", "SPshort": "2 ms", "SPlong": "30 ms",
         "PP30": "PP 30", "PP100": "PP 100", "TR10": "10 Hz", "EI1000": "E+I",
         "SP250@p110": "SP250 @p", "SP1000@p110": "SP1000 @p", "SP4000@p110": "SP4000 @p"}


SHORTPHYS = {"a": "1/τ excitatory", "b": "1/τ inhibitory", "c1": "P → E interneurons", "c2": "E interneurons → P",
             "c3": "P → I interneurons", "c4": "I interneurons → P", "k": "stimulus gain", "A": "max EPSP",
             "B": "max IPSP", "v0": "firing threshold", "r": "sigmoid slope", "vmax": "max firing rate"}


def fmt(v):
    return ">1e3" if v >= 1000 else (f"{v:.0f}" if v >= 10 else f"{v:.2g}")


def save(fig, name):
    fig.savefig(OUT / f"fig_{name}.png", bbox_inches="tight")
    plt.close(fig)


def load_train():
    files = sorted(glob.glob(str(OUT / "train*_part*of*.npz")))
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("theta", "truth", "rel", "pred", "jobs")}
    d["hist_loss"] = np.concatenate([p["hist_loss"] for p in parts], 1)
    d["hist_theta"] = np.concatenate([p["hist_theta"] for p in parts], 1)
    d["hist_it"] = parts[0]["hist_it"]
    d["jobs"] = list(d["jobs"])
    return d


def sel(d, **kw):
    return np.array([all(j.get(k) == v for k, v in kw.items()) for j in d["jobs"]])


def main():
    an = json.load(open(OUT / "analysis.json"))
    sweep = dict(np.load(OUT / "sweep.npz"))
    jac = np.load(OUT / "jacobian.npz")
    summary = {}

    # ------------------------------------------------------------ 1. stimuli and teacher responses
    y = jac["y"][0]                                   # nominal teacher, all conditions (S, T)
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), sharex=True)
    for ax, s in zip(axes.flat, ORDER):
        i = ORDER.index(s)
        u = node.STIMS[s].waveform(node.T_WIN)
        ax2 = ax.twinx()
        ax2.fill_between(T_MS, 0, u, color="#f2c14e", alpha=0.45, step="post", lw=0)
        ax2.set_ylim(0, 5500); ax2.set_yticks([]); ax2.spines["right"].set_visible(False)
        ax.plot(T_MS, y[i], color="k", lw=1.1)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_title(SHORT[s] + ("  (tonic p=110)" if "@" in s else ""), fontsize=8.5)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_xlim(-30, 290)
    for ax in axes[-1]:
        ax.set_xlabel("ms from first pulse")
    for ax in axes[:, 0]:
        ax.set_ylabel("EEG (µV)")
    fig.suptitle("Teacher (nominal JR node) response to each virtual-TMS condition — black: EEG, yellow: injected current", fontsize=9.5)
    fig.tight_layout()
    save(fig, "responses")

    # ------------------------------------------------------------ 2. one-at-a-time gallery
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), sharex=True)
    i1000 = node.STIM_ORDER.index("SP1000")
    for ax, n in zip(axes.flat, ALL):
        r = sweep[n]
        ax.plot(T_MS, y[i1000], color="k", lw=1.0, label="nominal")
        ax.plot(T_MS, r[0, i1000], color=PCOL[n], lw=1.0, ls="--", label="×0.7")
        ax.plot(T_MS, r[1, i1000], color=PCOL[n], lw=1.0, label="×1.3")
        ax.set_title(f"{n}: {SHORTPHYS[n]}", fontsize=7.8)
        ax.set_xlim(-10, 200)
        ins = ax.inset_axes([0.5, 0.42, 0.48, 0.5])
        for k, (c, ls) in enumerate(((PCOL[n], "--"), (PCOL[n], "-"))):
            ins.plot(T_MS, r[k, i1000], color=c, lw=0.8, ls=ls)
        ins.plot(T_MS, y[i1000], color="k", lw=0.8)
        ins.set_xlim(40, 280); lo = min(r[:, i1000, 160:].min(), y[i1000, 160:].min())
        ins.set_ylim(lo * 1.3, max(0.3, -lo * 0.6)); ins.tick_params(labelsize=6)
        ins.set_title("late tail (zoom)", fontsize=6)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="center right")
    for ax in axes[-1]:
        ax.set_xlabel("ms")
    fig.suptitle("Single-pulse response (SP 1000) when one parameter is changed by ±30%", fontsize=9.5)
    fig.tight_layout()
    save(fig, "oneatatime")

    # ------------------------------------------------------------ 3. elasticities
    el = np.array(an["elasticity"])
    fn = an["feature_names"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    lim = 3.5
    im = ax.imshow(np.clip(el, -lim, lim), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(el.shape[0]):
        for j in range(el.shape[1]):
            if abs(el[i, j]) >= 0.05:
                ax.text(j, i, f"{el[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="w" if abs(el[i, j]) > 2 else "k")
    ax.set_xticks(range(len(ALL)), ALL)
    ax.set_yticks(range(len(fn)), [f"{f}  ({an['features_nominal'][f]:.3g})" for f in fn])
    ax.axvline(len(FIT) - 0.5, color="k", lw=1)
    ax.text(3, len(fn) + 0.3, "fitted by neurofe", ha="center", va="top", fontsize=8, style="italic")
    ax.text(len(FIT) + 2, len(fn) + 0.3, "fixed in neurofe", ha="center", va="top", fontsize=8, style="italic")
    plt.colorbar(im, ax=ax, label="elasticity  d log|feature| / d log θ", shrink=0.8)
    ax.set_title("How each parameter moves each input-output feature (nominal node)")
    fig.tight_layout()
    save(fig, "elasticity")

    # ------------------------------------------------------------ 4. CRB heatmap
    prot_show = E1_PROTOCOLS + ["ALL", "SP1000@p110", "DOSE@p110"]
    se7 = np.array([an["crb"][p][0]["se7"] for p in prot_show])
    se12 = np.array([an["crb"][p][0]["se12"] for p in prot_show])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [7, 12]})
    for ax, se, names, ttl in ((axes[0], se7, FIT, "7 parameters free (neurofe)"),
                               (axes[1], se12, ALL, "12 free (A, B, v0, r, vmax released)")):
        im = ax.imshow(se * 100, norm=LogNorm(0.1, 1000), cmap="viridis_r", aspect="auto")
        for i in range(se.shape[0]):
            for j in range(se.shape[1]):
                v = se[i, j] * 100
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6,
                        color="w" if v > 30 else "k")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(prot_show)), [p.replace("@p110", " @p110") for p in prot_show] if ax is axes[0] else [])
        ax.set_title(ttl)
    plt.colorbar(im, ax=axes, label="Cramér–Rao bound on relative error (%)", shrink=0.85)
    fig.suptitle("Best achievable precision per stimulation protocol (sensor noise 0.42 µV, nominal teacher). Lower = identifiable", fontsize=9.5)
    save(fig, "crb")
    summary["crb7_nominal"] = {p: dict(zip(FIT, (100 * s).round(3).tolist())) for p, s in zip(prot_show, se7)}
    summary["crb12_nominal"] = {p: dict(zip(ALL, (100 * s).round(3).tolist())) for p, s in zip(prot_show, se12)}
    se7_t = np.array([[an["crb"][p][q]["se7"] for q in range(1, len(an["points"]))] for p in prot_show])
    summary["crb7_teachers_median"] = {p: dict(zip(FIT, (100 * np.median(s, 0)).round(3).tolist()))
                                       for p, s in zip(prot_show, se7_t)}

    # ------------------------------------------------------------ 5. sloppiness: eigen-spectra and eigenvectors
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.35})
    ax = axes[0]
    for k, p in enumerate(["SP1000", "DOSE", "RICH", "EI1000", "DOSE@p110"]):
        w = np.array(an["crb"][p][0]["eig7"])
        ax.scatter(np.full(len(w), k), w / w.max(), marker="_", s=300, color="C%d" % k)
    ax.set_yscale("log"); ax.set_xticks(range(5), ["SP1000", "DOSE", "RICH", "E+I", "DOSE@p"], rotation=20)
    ax.set_ylabel("Fisher eigenvalue / largest"); ax.set_title("Sloppiness: eigen-spectrum (7 params)")
    for ax, p in zip(axes[1:], ["SP1000", "RICH"]):
        V = np.array(an["crb"][p][0]["V7"])             # columns = eigenvectors (ascending)
        w = np.array(an["crb"][p][0]["eig7"])
        im = ax.imshow(V[:, ::-1].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(FIT)), FIT)
        ax.set_yticks(range(len(FIT)), [f"{x:.0e}" for x in (w / w.max())[::-1]])
        if p == "SP1000":
            ax.set_ylabel("eigenvalue (rel.)  stiff → sloppy")
        ax.set_title(f"Eigen-directions, {p}")
    plt.colorbar(im, ax=axes[1:], shrink=0.8, label="loading")
    save(fig, "sloppiness")

    # ------------------------------------------------------------ 6. tonic sweep
    ts = an["tonic_sweep"]
    p = np.array(ts["p"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, key in zip(axes[:2], ("SP1000", "DOSE")):
        se = np.array(ts[key]) * 100
        for j, n in enumerate(FIT):
            ax.semilogy(p, se[:, j], color=PCOL[n], label=n, marker=".", ms=3)
        ax.axvline(115, color="0.6", ls=":", lw=1)
        ax.set_xlabel("tonic background drive p"); ax.set_title(f"CRB vs operating point, {key}")
        ax.set_ylabel("rel. error bound (%)")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    ax = axes[2]
    yy = np.array(ts["y"])
    for pv, c in ((0, "k"), (110, "C0"), (150, "C3"), (220, "C2")):
        ax.plot(T_MS, yy[list(p).index(pv)], color=c, lw=0.9, label=f"p={pv}")
    ax.set_xlim(-20, 290); ax.set_ylim(-1.5, 3); ax.set_xlabel("ms"); ax.set_ylabel("µV (zoom)")
    ax.legend(fontsize=7, frameon=False); ax.set_title("SP 1000 response, zoom on late part")
    fig.text(0.5, -0.02, "dotted line (p ≈ 115): the resting state jumps to the upper JR branch, a damped ~10 Hz (alpha) resonance; no sustained oscillation", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "tonic")

    # ------------------------------------------------------------ training results
    files = glob.glob(str(OUT / "train*_part*of*.npz"))
    if not files:
        save_json(summary, OUT / "summary.json")
        print("no training results yet; analysis figures written")
        return
    d = load_train()
    err = np.log(d["theta"] / d["truth"])                         # (M, 12) log error
    err0 = np.log(d["hist_theta"][0] / d["truth"])                # at the start
    summary["n_models"] = len(d["jobs"])
    # late-window metrics (nominal-teacher models only): t >= 60 ms after the first pulse
    LATE = node.ONSET + 60
    ynom = jac["y"][0]                                            # (S, T) nominal teacher, ORDER
    nomj = np.array([j["teacher"] == "nominal" for j in d["jobs"]])
    dif = d["pred"][..., LATE:] - ynom[None, :, LATE:]
    late_rel = np.sqrt((dif ** 2).sum(-1) / (ynom[None, :, LATE:] ** 2).sum(-1))
    pc = d["pred"][..., LATE:] - d["pred"][..., LATE:].mean(-1, keepdims=True)
    tc = ynom[None, :, LATE:] - ynom[None, :, LATE:].mean(-1, keepdims=True)
    late_r = (pc * tc).sum(-1) / np.sqrt((pc ** 2).sum(-1) * (tc ** 2).sum(-1) + 1e-30)
    late_rel[~nomj] = np.nan; late_r[~nomj] = np.nan
    # progress: 1 = reached truth, 0 = did not move, < 0 = moved away
    prog = 1 - np.abs(err) / np.maximum(np.abs(err0), 1e-9)
    summary["start_median_abs_err_pct"] = dict(zip(FIT, (100 * np.median(np.abs(err0[sel(d, exp="E1")][:, :len(FIT)]), 0)).round(1).tolist()))
    summary["progress_median"] = {}
    for exp in ("E1", "E2", "E3", "E4", "E6", "E7", "E8", "E9", "E10", "E11", "E12"):
        mm = sel(d, exp=exp)
        if mm.any():
            npar = len(ALL) if exp == "E3" else len(FIT)
            summary["progress_median"][exp] = dict(zip(ALL[:npar], np.median(prog[mm][:, :npar], 0).round(3).tolist()))

    # 7. E1 recovery per protocol vs CRB
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    W = 0.11
    rec = {}
    for i, prot in enumerate(E1_PROTOCOLS):
        m = sel(d, exp="E1", protocol=prot)
        e = np.abs(err[m][:, :len(FIT)]) * 100
        rec[prot] = {"median_abs_err_pct": dict(zip(FIT, np.median(e, 0).round(3).tolist())),
                     "max_abs_err_pct": dict(zip(FIT, e.max(0).round(3).tolist())),
                     "final_loss_median": float(np.median(d["hist_loss"][-1][m]))}
        for j, n in enumerate(FIT):
            x = i + (j - 3) * W
            ax.scatter(np.full(e.shape[0], x) + np.random.default_rng(j).uniform(-0.02, 0.02, e.shape[0]),
                       np.maximum(e[:, j], 1e-3), s=5, color=PCOL[n], alpha=0.6, lw=0, label=n if i == 0 else None)
            ax.plot([x - W / 2, x + W / 2], [np.median(e[:, j])] * 2, color=PCOL[n], lw=2)
    ax.axhline(10, color="0.5", ls=":", lw=1); ax.text(len(E1_PROTOCOLS) - 0.5, 11, "10%", fontsize=7, color="0.4")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 300)
    ax.set_xticks(range(len(E1_PROTOCOLS)), [SHORT.get(p, p) for p in E1_PROTOCOLS])
    ax.set_ylabel("|relative error| of fitted value (%)")
    ax.set_title("E1  Backprop recovery from 10 random starts (±50%) per protocol, noise-free teacher. Bars = median")
    ax.legend(ncol=7, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "e1_recovery")
    summary["E1"] = rec

    # 8. E1 generalisation matrix: train protocol x test condition
    G = np.array([[np.median(d["rel"][sel(d, exp="E1", protocol=p)][:, ORDER.index(s)]) for s in ORDER]
                  for p in E1_PROTOCOLS]) * 100
    GL = np.array([[np.median(late_rel[sel(d, exp="E1", protocol=p)][:, ORDER.index(s)]) for s in ORDER]
                   for p in E1_PROTOCOLS]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, M_, ttl in ((axes[0], G, "whole window"), (axes[1], GL, "late window only (> 60 ms after the pulse)")):
        im = ax.imshow(M_, norm=LogNorm(0.01, 100), cmap="magma_r", aspect="auto")
        for i in range(M_.shape[0]):
            for j in range(M_.shape[1]):
                trained = ORDER[j] in PROTOCOLS[E1_PROTOCOLS[i]]
                ax.text(j, i, f"{M_[i, j]:.2g}" if M_[i, j] < 100 else f"{M_[i, j]:.0f}", ha="center", va="center", fontsize=5.6,
                        color="w" if M_[i, j] > 3 else "k", fontweight="bold" if trained else "normal")
                if trained:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#2a9d8f", lw=1.5))
        ax.set_xticks(range(len(ORDER)), [SHORT[s] for s in ORDER], rotation=40, ha="right")
        ax.set_title(ttl)
        ax.set_xlabel("test condition (green box = seen in training)")
    axes[0].set_yticks(range(len(E1_PROTOCOLS)), [SHORT.get(p, p) for p in E1_PROTOCOLS])
    axes[0].set_ylabel("trained on")
    plt.colorbar(im, ax=axes, label="median relative RMS prediction error (%)", shrink=0.85)
    fig.suptitle("E1  Does a fit generalise to stimuli it never saw?", fontsize=9.5)
    save(fig, "e1_generalisation")
    summary["E1_generalisation_pct"] = {p: dict(zip(ORDER, G[i].round(4).tolist())) for i, p in enumerate(E1_PROTOCOLS)}
    summary["E1_generalisation_late_pct"] = {p: dict(zip(ORDER, GL[i].round(4).tolist())) for i, p in enumerate(E1_PROTOCOLS)}
    held_late = np.array([np.median([late_rel[i, ORDER.index(s)] for s in ORDER if s not in PROTOCOLS[d["jobs"][i]["protocol"]]] or [np.nan])
                          for i in range(len(d["jobs"]))])

    # 9. parameter error vs I/O error (all E1/E6 models): which parameter errors matter?
    m = sel(d, exp="E1") | sel(d, exp="E6")
    held = np.array([np.median([d["rel"][i, ORDER.index(s)] for s in ORDER if s not in PROTOCOLS[d["jobs"][i]["protocol"]]] or [np.nan])
                     for i in range(len(d["jobs"]))])
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(2, len(FIT), figsize=(11, 4.8), sharey="row")
    corr, corr_late = {}, {}
    m_late = sel(d, exp="E1")
    for j, n in enumerate(FIT):
        for row, (mm, yv_all, store, lab) in enumerate(((m, held, corr, "held-out stimuli,\nwhole window (%)"),
                                                        (m_late, held_late, corr_late, "held-out stimuli,\nlate window (%)"))):
            ax = axes[row, j]
            x = np.abs(err[mm, j]) * 100; yv = yv_all[mm] * 100
            ok = np.isfinite(yv)
            ax.loglog(np.maximum(x[ok], 1e-3), np.maximum(yv[ok], 1e-3), ".", ms=3, color=PCOL[n], alpha=0.6)
            rho = spearmanr(x[ok], yv[ok]).correlation
            store[n] = float(rho)
            ax.set_title(f"{n}   ρ={rho:.2f}", fontsize=8)
            if row == 1:
                ax.set_xlabel("|param err| (%)")
            if j == 0:
                ax.set_ylabel(lab)
    fig.suptitle("Parameter error vs prediction error on held-out stimuli (top: E1+E6, whole window; bottom: E1, late window). Spearman ρ", fontsize=9)
    fig.tight_layout()
    save(fig, "err_vs_io")
    summary["spearman_param_err_vs_heldout"] = corr
    summary["spearman_param_err_vs_heldout_late"] = corr_late

    # 10. E2 raw vs log
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.8), sharex=True)
    it = d["hist_it"]
    for ax, n in zip(axes.flat, FIT):
        j = FIT.index(n)
        for exp, c in (("E1", "C0"), ("E2", "C3")):
            mm = sel(d, exp=exp, protocol="RICH")
            tr = d["hist_theta"][:, mm, j] / d["truth"][mm, j]
            ax.plot(it, tr, color=c, lw=0.7, alpha=0.6)
        ax.axhline(1, color="k", lw=0.6); ax.set_title(n); ax.set_ylim(0.4, 1.7)
    ax = axes.flat[-1]
    for exp, c, lab in (("E1", "C0", "log-scaled θ (this study)"), ("E2", "C3", "raw θ, Adam lr 0.05 (neurofe)")):
        mm = sel(d, exp=exp, protocol="RICH")
        ax.semilogy(it, d["hist_loss"][:, mm], color=c, lw=0.7, alpha=0.6)
        ax.plot([], [], color=c, label=lab)
    ax.set_title("relative loss"); ax.legend(fontsize=6.5, frameon=False)
    for a in axes[-1]:
        a.set_xlabel("Adam iteration")
    fig.suptitle("E2  Same data (RICH protocol), same starts: neurofe's raw parameterisation vs log-scaled. y = fitted / true", fontsize=9.5)
    fig.tight_layout()
    save(fig, "e2_raw_vs_log")
    summary["E2"] = {}
    for exp in ("E1", "E2"):
        for prot in ("SP1000", "RICH"):
            mm = sel(d, exp=exp, protocol=prot)
            summary["E2"][f"{exp}_{prot}"] = {"median_abs_err_pct": dict(zip(FIT, (100 * np.median(np.abs(err[mm, :len(FIT)]), 0)).round(2).tolist())),
                                             "final_loss_median": float(np.median(d["hist_loss"][-1][mm]))}

    # 11. E3 extended parameters
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    for k, (prot, c) in enumerate((("SP1000", "C1"), ("RICH", "C0"))):
        mm = sel(d, exp="E3", protocol=prot)
        e = np.abs(err[mm]) * 100
        for j in range(len(ALL)):
            ax.scatter(np.full(e.shape[0], j + (k - 0.5) * 0.3), np.maximum(e[:, j], 1e-3), s=6, color=c, alpha=0.7, lw=0,
                       label=prot if j == 0 else None)
        summary.setdefault("E3", {})[prot] = {"median_abs_err_pct": dict(zip(ALL, np.median(e, 0).round(2).tolist())),
                                              "final_loss_median": float(np.median(d["hist_loss"][-1][mm])),
                                              "heldout_median_pct": float(np.nanmedian(held[mm]) * 100)}
    ax.set_yscale("log"); ax.set_xticks(range(len(ALL)), ALL); ax.axvline(len(FIT) - 0.5, color="k", lw=0.8)
    ax.set_ylabel("|rel. error| (%)"); ax.legend(fontsize=7, frameon=False); ax.set_title("E3  12 free parameters")
    mm = sel(d, exp="E3", protocol="RICH")
    for ax, (p1, p2) in zip(axes[1:], (("A", "k"), ("A", "a"))):
        i1, i2 = ALL.index(p1), ALL.index(p2)
        ax.scatter(err[mm, i1], err[mm, i2], s=10, color="C0")
        mm1 = sel(d, exp="E3", protocol="SP1000")
        ax.scatter(err[mm1, i1], err[mm1, i2], s=10, color="C1")
        ax.set_xlabel(f"log({p1} fit/true)"); ax.set_ylabel(f"log({p2} fit/true)")
        ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"trade-off {p1} vs {p2}")
    fig.tight_layout()
    save(fig, "e3_extended")

    # 12. E4 noise vs CRB
    fig, ax = plt.subplots(figsize=(8, 3.4))
    summary["E4"] = {}
    for k, prot in enumerate(("SP1000", "DOSE", "RICH")):
        mm = sel(d, exp="E4", protocol=prot)
        e = np.abs(err[mm, :len(FIT)]) * 100
        cr = np.array(an["crb"][prot][0]["se7"]) * 100
        for j, n in enumerate(FIT):
            x = j + (k - 1) * 0.25
            ax.scatter(np.full(e.shape[0], x), np.maximum(e[:, j], 1e-3), s=5, color="C%d" % k, alpha=0.6, lw=0,
                       label=prot if j == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [cr[j]] * 2, color="k", lw=1.4)
        summary["E4"][prot] = {"median_abs_err_pct": dict(zip(FIT, np.median(e, 0).round(2).tolist())),
                               "crb_pct": dict(zip(FIT, cr.round(2).tolist()))}
    ax.set_yscale("log"); ax.set_xticks(range(len(FIT)), FIT); ax.set_ylabel("|rel. error| (%)")
    ax.legend(fontsize=7, frameon=False, ncol=3); ax.set_ylim(1e-2, 500)
    ax.set_title("E4  Fits to noisy data (0.42 µV, real TMS-EEG level). Black bars = Cramér–Rao bound")
    save(fig, "e4_noise")

    # 13. E5 profile likelihood
    from syn_brain.scripts.single_node_tms import SIGMA
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    summary["E5"] = {}
    for ax, prot in zip(axes, ("SP1000", "RICH")):
        ci = [ORDER.index(c) for c in PROTOCOLS[prot]]
        power = float((ynom[ci] ** 2).mean())
        thr = 3.84 * SIGMA ** 2 / (node.T_WIN * len(ci) * power)       # 95% chi2(1) at 0.42 uV noise
        for j, n in enumerate(FIT):
            mm = sel(d, exp="E5", protocol=prot, pin=j)
            gz = np.array([d["jobs"][i]["pin_z"] for i in np.where(mm)[0]])
            l = d["hist_loss"][-1][mm]
            o = np.argsort(gz)
            ax.loglog(np.exp(gz[o]), np.maximum(l[o], 1e-10), "-o", ms=3, color=PCOL[n], label=n)
            summary["E5"].setdefault(prot, {})[n] = dict(zip([f"{v:.2f}" for v in np.exp(gz[o])], l[o].tolist()))
        summary["E5"][prot]["threshold_95"] = thr
        ax.axhline(thr, color="k", ls="--", lw=0.9)
        ax.text(0.52, thr * 1.4, "95% threshold at real noise (0.42 µV)", fontsize=7)
        ax.axvline(1, color="0.7", lw=0.6)
        ax.set_xticks([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2]); ax.set_xticklabels(["0.5", "0.67", "0.8", "1", "1.25", "1.5", "2"])
        ax.minorticks_off()
        ax.set_xlabel("pinned value / true value"); ax.set_title(f"E5  profile likelihood, {prot}")
    axes[0].set_ylabel("best relative loss after refitting the others")
    axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper center")
    fig.text(0.5, -0.03, "Below the dashed line the change is invisible at real TMS-EEG noise: the parameter is not identifiable over that range.", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "e5_profile")

    # 14. E6 + E7
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    ax = axes[0]
    summary["E6"] = {}
    for k, prot in enumerate(("SP1000", "DOSE", "RICH")):
        mm = sel(d, exp="E6", protocol=prot)
        e = np.abs(err[mm, :len(FIT)]) * 100
        for j, n in enumerate(FIT):
            ax.scatter(np.full(e.shape[0], j + (k - 1) * 0.25), np.maximum(e[:, j], 1e-3), s=5, color="C%d" % k, alpha=0.6, lw=0,
                       label=prot if j == 0 else None)
        summary["E6"][prot] = dict(zip(FIT, np.median(e, 0).round(2).tolist()))
    ax.set_yscale("log"); ax.set_xticks(range(len(FIT)), FIT); ax.legend(fontsize=7, frameon=False)
    ax.set_ylabel("|rel. error| (%)"); ax.set_title("E6  4 perturbed teachers (±30%)")
    ax = axes[1]
    summary["E7"] = {}
    for k, (exp, prot) in enumerate((("E1", "SP1000"), ("E7", "SP1000@p110"), ("E1", "DOSE"), ("E7", "DOSE@p110"))):
        mm = sel(d, exp=exp, protocol=prot)
        e = np.abs(err[mm, :len(FIT)]) * 100
        for j, n in enumerate(FIT):
            ax.scatter(np.full(e.shape[0], j + (k - 1.5) * 0.2), np.maximum(e[:, j], 1e-3), s=5, color="C%d" % (k + 3),
                       alpha=0.7, lw=0, label=prot if j == 0 else None)
        summary["E7"][prot] = dict(zip(FIT, np.median(e, 0).round(2).tolist()))
    ax.set_yscale("log"); ax.set_xticks(range(len(FIT)), FIT); ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("E7  resting (p=0) vs driven (p=110) operating point")
    fig.tight_layout()
    save(fig, "e6_e7")

    # 14b. E9 + E10: firing-state recovery (tonic p=120, 150 -- past the jump, vs E7's p=110)
    if sel(d, exp="E9").any():
        from syn_brain.scripts.single_node_tms import FIRING_PROTOCOLS
        ps_firing = sorted({int(p.split("@p")[1]) for p in FIRING_PROTOCOLS})
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        ax = axes[0]
        combos = [("E1", "SP1000"), ("E7", "SP1000@p110")] + [("E9", f"SP1000@p{p}") for p in ps_firing]
        summary["E9"] = {}
        for k, (exp, prot) in enumerate(combos):
            mm = sel(d, exp=exp, protocol=prot)
            e = np.abs(err[mm, :len(FIT)]) * 100
            for j, n in enumerate(FIT):
                ax.scatter(np.full(e.shape[0], j + (k - 1.5) * 0.15), np.maximum(e[:, j], 1e-3), s=5,
                          color="C%d" % k, alpha=0.7, lw=0, label=prot if j == 0 else None)
            summary["E9"][prot] = dict(zip(FIT, np.median(e, 0).round(2).tolist()))
        ax.set_yscale("log"); ax.set_xticks(range(len(FIT)), FIT); ax.legend(fontsize=6.5, frameon=False, ncol=2)
        ax.set_ylabel("|rel. error| (%)")
        ax.set_title("E9  noise-free recovery vs operating point (p=0, 110, 120, 150)")
        ax = axes[1]
        summary["E10"] = {}
        for k, p in enumerate(ps_firing):
            for q, base in enumerate(("SP1000", "DOSE")):
                prot = f"{base}@p{p}"
                mm = sel(d, exp="E10", protocol=prot)
                e = np.abs(err[mm, :len(FIT)]) * 100
                cr = np.array(an["crb"][prot][0]["se7"]) * 100
                c = "C%d" % (2 * k + q)
                for j, n in enumerate(FIT):
                    x = j + (2 * k + q - 1.5) * 0.18
                    ax.scatter(np.full(e.shape[0], x), np.maximum(e[:, j], 1e-3), s=5, color=c, alpha=0.7, lw=0,
                              label=prot if j == 0 else None)
                    ax.plot([x - 0.08, x + 0.08], [cr[j]] * 2, color="k", lw=1.2)
                summary["E10"][prot] = {"median_abs_err_pct": dict(zip(FIT, np.median(e, 0).round(2).tolist())),
                                        "crb_pct": dict(zip(FIT, cr.round(2).tolist()))}
        ax.set_yscale("log"); ax.set_xticks(range(len(FIT)), FIT); ax.legend(fontsize=6, frameon=False, ncol=2)
        ax.set_ylim(1e-2, 500)
        ax.set_title("E10  recovery at 0.42 µV noise, firing state. Black bars = CRB")
        fig.suptitle("Pushing the node onto the firing branch: does the E-I loop become recoverable?", fontsize=9.5)
        fig.tight_layout()
        save(fig, "e9_e10_firing")

    # E8 temporal hold-out vs E1 full-window fits
    if sel(d, exp="E8").any():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
        ax = axes[0]
        i1 = ORDER.index("SP1000")
        ax.plot(T_MS, ynom[i1], color="k", lw=1.6, label="teacher", zorder=5)
        for exp, c, lab in (("E1", "C0", "fit on whole window (E1)"), ("E8", "C3", "fit on 0–60 ms only (E8)")):
            mm = np.where(sel(d, exp=exp, protocol="SP1000"))[0]
            for q, i in enumerate(mm):
                ax.plot(T_MS, d["pred"][i, i1], color=c, lw=0.7, alpha=0.7, label=lab if q == 0 else None)
        ax.axvspan(60, 290, color="0.92", zorder=0); ax.text(170, 1.6, "held-out late window", ha="center", fontsize=7)
        ax.set_xlim(-10, 290); ax.set_ylim(-2.0, 2.0); ax.set_xlabel("ms"); ax.set_ylabel("µV (zoom)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.set_title("SP 1000: late window, zoomed")
        summary["E8"] = {}
        for k, (metric, arr, ttl) in enumerate((("late Pearson r", late_r, "late-window Pearson r"),
                                                ("late rel. error", late_rel, "late-window rel. error"))):
            ax = axes[1 + k]
            for q, prot in enumerate(("SP1000", "DOSE", "RICH")):
                ci = [ORDER.index(c) for c in PROTOCOLS[prot]]
                for e_, (exp, c) in enumerate((("E1", "C0"), ("E8", "C3"))):
                    mm = sel(d, exp=exp, protocol=prot)
                    v = np.nanmean(arr[mm][:, ci], 1)
                    x = q + (e_ - 0.5) * 0.3
                    ax.scatter(np.full(len(v), x), v if k == 0 else v * 100, s=8, color=c, alpha=0.7, lw=0,
                               label=exp if q == 0 else None)
                    summary["E8"].setdefault(prot, {})[f"{exp} {metric} median"] = float(np.median(v))
                    summary["E8"][prot][f"{exp} {metric} min"] = float(np.min(v))
            ax.set_xticks(range(3), ["SP1000", "DOSE", "RICH"]); ax.set_title(ttl)
            if k == 1:
                ax.set_yscale("log"); ax.set_ylabel("%")
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle("E8  Temporal hold-out on the training stimuli: fit up to 60 ms after the pulse, predict the rest", fontsize=9.5)
        fig.tight_layout()
        save(fig, "e8_holdout")
        mm = sel(d, exp="E8")
        summary["E8"]["median_abs_err_pct"] = {prot: dict(zip(FIT, (100 * np.median(np.abs(err[sel(d, exp="E8", protocol=prot), :len(FIT)]), 0)).round(2).tolist()))
                                               for prot in ("SP1000", "DOSE", "RICH")}

    # loss curves E1
    fig, ax = plt.subplots(figsize=(6, 3))
    for k, prot in enumerate(("SP1000", "DOSE", "RICH", "EI1000")):
        mm = sel(d, exp="E1", protocol=prot)
        ax.semilogy(it, np.median(d["hist_loss"][:, mm], 1), label=prot, color="C%d" % k)
        ax.fill_between(it, d["hist_loss"][:, mm].min(1), d["hist_loss"][:, mm].max(1), color="C%d" % k, alpha=0.15)
    ax.set_xlabel("Adam iteration"); ax.set_ylabel("relative loss"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("E1  training curves (median and range over starts)")
    save(fig, "e1_loss")
    save_json(summary, OUT / "summary.json")
    print("figures and summary written")


if __name__ == "__main__":
    main()
