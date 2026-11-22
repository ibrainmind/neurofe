"""Figures and tables for the single-node study (reads results/single_node/*).

Usage (from the repo root, after all parts of scripts/single_node.py have finished):
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.single_node_figures
Outputs: results/single_node/fig_*.png and tables.json
"""

from __future__ import annotations

import glob
import json

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from syn_brain.core.paths import RESULTS_DIR, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites
from syn_brain.scripts.single_node import (AMPS, NAMES, NOM, SIGMA, TRAIN_SETS, one_node_net, simulate)

OUT = RESULTS_DIR / "single_node"
COL = {"a": "#1f77b4", "b": "#ff7f0e", "c1": "#2ca02c", "c2": "#d62728", "c3": "#9467bd", "c4": "#8c564b", "k": "#e377c2"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_parts():
    files = sorted(glob.glob(str(OUT / "study_part*of*.npz")))
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    keys = ["theta", "resid", "rel"] + [k for k in parts[0] if k.startswith("job_")]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["hist_loss"] = np.concatenate([p["hist_loss"] for p in parts], axis=1)
    for k in ("truth", "grid", "amps", "names"):
        d[k] = parts[0][k]
    return d, len(files)


def project(eeg, lm_col):
    """Single-node EEG is rank 1 (leadfield x waveform): project onto the leadfield pattern."""
    u = lm_col - lm_col.mean()
    u = u / np.linalg.norm(u)
    return np.tensordot(eeg, u, axes=([-2], [0]))


def main():
    anat = load_anatomy(10)
    site = int(pick_sites(anat, 1, seed=0)[0])
    lm_col = anat.lm[:, site]
    net = one_node_net(lm_col)
    ident = json.load(open(OUT / "identifiability.json"))
    J = np.load(OUT / "jacobian.npy")                       # (T+1, A, C, t, P); last teacher = nominal
    d, nparts = load_parts()
    truth = d["truth"]
    tz = np.log(truth / NOM)
    t_ms = np.arange(100, 400) - 110
    tables = {"n_parts": nparts}

    # ---------------------------------------------------------------- fig 1: responses
    with torch.no_grad():
        e_nom = simulate(torch.tensor(NOM[None]), net).numpy()[0]           # (5, C, t)
        e_t = simulate(torch.tensor(truth[:4]), net).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, a in enumerate(AMPS):
        ax[0].plot(t_ms, project(e_nom[j], lm_col), label=f"pulse {int(a)}", color=plt.cm.viridis(j / 4))
    ax[0].set_title("Nominal node: response to 5 pulse amplitudes")
    ax[0].set_xlabel("ms after pulse")
    ax[0].set_ylabel("EEG projected on leadfield (uV)")
    ax[0].legend(fontsize=7)
    for i in range(4):
        ax[1].plot(t_ms, project(e_t[i, 2], lm_col), label=f"teacher {i}")
    ax[1].plot(t_ms, project(e_nom[2], lm_col), "k--", label="nominal")
    ax[1].set_title("Different teachers, pulse 1000")
    ax[1].set_xlabel("ms after pulse")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_responses.png", dpi=150)
    plt.close(fig)
    tables["peak_uV_nominal_by_amp"] = dict(zip([str(int(a)) for a in AMPS],
                                                np.abs(e_nom).max((1, 2)).round(2).tolist()))

    # ---------------------------------------------------------------- fig 2: sensitivity waveforms
    Jn = J[-1]                                                                # nominal (A, C, t, P)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), sharey=False)
    for k, j in enumerate([1, 2, 4]):
        for p, n in enumerate(NAMES):
            ax[k].plot(t_ms, project(Jn[j, :, :, p], lm_col), color=COL[n], label=n)
        ax[k].axhline(0, color="k", lw=0.5)
        ax[k].set_title(f"dEEG / dlog(param), pulse {int(AMPS[j])}")
        ax[k].set_xlabel("ms after pulse")
    ax[0].set_ylabel("uV per 100% change")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_sensitivity.png", dpi=150)
    plt.close(fig)
    # shape similarity of sensitivity waveforms at pulse 1000 (cosine)
    S = np.stack([Jn[2, :, :, p].ravel() for p in range(len(NAMES))])
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    tables["sensitivity_cosine_pulse1000"] = (S @ S.T).round(3).tolist()

    # ---------------------------------------------------------------- fig 3: CRB
    labs = ["1 amplitude", "3 amplitudes", "5 amplitudes"]
    T = len(truth)
    crb = {lab: np.array([[ident["teachers"][t][lab]["crb_rel_se"][n] for n in NAMES] for t in range(T)]) for lab in labs}
    combos = list(ident["teachers"][0]["1 amplitude"]["combo_rel_se"])
    crbc = {lab: np.array([[ident["teachers"][t][lab]["combo_rel_se"][c] for c in combos] for t in range(T)]) for lab in labs}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5), gridspec_kw={"width_ratios": [7, 4]})
    w = 0.27
    for k, lab in enumerate(labs):
        med = np.median(crb[lab], 0) * 100
        lo, hi = np.percentile(crb[lab], 25, 0) * 100, np.percentile(crb[lab], 75, 0) * 100
        x = np.arange(len(NAMES)) + (k - 1) * w
        ax[0].bar(x, med, w, label=lab, color=plt.cm.Blues(0.4 + 0.25 * k))
        ax[0].errorbar(x, med, [med - lo, hi - med], fmt="none", color="k", lw=0.7)
        med = np.median(crbc[lab], 0) * 100
        ax[1].bar(np.arange(len(combos)) + (k - 1) * w, med, w, color=plt.cm.Blues(0.4 + 0.25 * k))
    for a_ in ax:
        a_.set_yscale("log")
        a_.axhline(10, color="g", ls="--", lw=0.8)
        a_.axhline(100, color="r", ls="--", lw=0.8)
    ax[0].set_xticks(range(len(NAMES)), NAMES)
    ax[0].set_ylabel("best possible relative SE, %")
    ax[0].set_title("Cramer-Rao bound per parameter (median of 10 teachers, IQR)")
    ax[0].legend(fontsize=7)
    ax[1].set_xticks(range(len(combos)), combos)
    ax[1].set_title("Combinations")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_crb.png", dpi=150)
    plt.close(fig)
    tables["crb_median_pct"] = {lab: dict(zip(NAMES, (np.median(crb[lab], 0) * 100).round(2).tolist())) for lab in labs}
    tables["crb_combo_median_pct"] = {lab: dict(zip(combos, (np.median(crbc[lab], 0) * 100).round(2).tolist())) for lab in labs}

    # ---------------------------------------------------------------- fig 4: correlations + weakest direction
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, lab in enumerate(["1 amplitude", "5 amplitudes"]):
        C = np.median(np.array([ident["teachers"][t][lab]["corr"] for t in range(T)]), 0)
        im = ax[k].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax[k].set_xticks(range(len(NAMES)), NAMES)
        ax[k].set_yticks(range(len(NAMES)), NAMES)
        for i in range(len(NAMES)):
            for j in range(len(NAMES)):
                ax[k].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax[k].set_title(f"Estimate correlations, {lab}")
        tables[f"corr_median_{lab}"] = C.round(3).tolist()
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    for k, lab in enumerate(labs):
        V = np.array(ident["teachers"][-1][lab]["eigvecs"])               # nominal, rows = eigvecs ascending
        v = V[0] * np.sign(V[0][np.argmax(np.abs(V[0]))])
        ax[2].bar(np.arange(len(NAMES)) + (k - 1) * 0.27, v, 0.27, label=lab, color=plt.cm.Blues(0.4 + 0.25 * k))
    ax[2].set_xticks(range(len(NAMES)), NAMES)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("Least identifiable direction (nominal)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_correlations.png", dpi=150)
    plt.close(fig)
    tables["eig_nominal"] = {lab: {"eigvals": ident["teachers"][-1][lab]["eigvals"],
                                   "eigvecs": np.array(ident["teachers"][-1][lab]["eigvecs"]).round(3).tolist()}
                             for lab in labs}

    # ---------------------------------------------------------------- fig 5: profile likelihood
    kind = d["job_kind"]
    prof = kind == "profile"
    n_prof_t = int(d["job_teacher"][prof].max()) + 1
    fig, ax = plt.subplots(2, len(NAMES), figsize=(16, 5.5), sharey="row")
    tables["profile"] = {}
    for r, tr in enumerate(TRAIN_SETS):
        n_amp = len(TRAIN_SETS[tr])
        amp_idx = [list(AMPS).index(a) for a in TRAIN_SETS[tr]]
        for p, n in enumerate(NAMES):
            rows_all = []
            for t in range(n_prof_t):
                sel = prof & (d["job_train"] == tr) & (d["job_fixed_idx"] == p) & (d["job_teacher"] == t)
                off = d["job_fixed_z"][sel] - tz[t, p]
                order = np.argsort(off)
                chi2 = d["resid"][sel][:, amp_idx].sum(1) / SIGMA ** 2
                ax[r, p].plot(np.exp(off[order]), chi2[order], "o-", ms=3, color=plt.cm.tab10(t), lw=1)
                rows_all.append({"factor": np.exp(off[order]).round(3).tolist(), "dchi2": chi2[order].round(2).tolist()})
            ax[r, p].axhline(3.84, color="r", ls="--", lw=0.8)
            ax[r, p].set_xscale("log")
            ax[r, p].set_yscale("symlog", linthresh=1)
            ax[r, p].set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
            if r == 0:
                ax[r, p].set_title(n)
            if r == 1:
                ax[r, p].set_xlabel("fixed value / truth")
            tables["profile"][f"{tr}/{n}"] = rows_all
        ax[r, 0].set_ylabel(f"{'1 amplitude' if tr == '1amp' else '3 amplitudes'}\nbest delta-chi2")
    fig.suptitle("Profile likelihood: fix one parameter, refit the rest by backprop. Below the red line (3.84) = "
                 "cannot be excluded at 95%", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_profiles.png", dpi=150)
    plt.close(fig)

    # fig 6: compensation along the profile (teacher 0, 1 amplitude)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for k, pname in enumerate(["c1", "c3", "b"]):
        p = NAMES.index(pname)
        sel = prof & (d["job_train"] == "1amp") & (d["job_fixed_idx"] == p) & (d["job_teacher"] == 0)
        off = d["job_fixed_z"][sel] - tz[0, p]
        order = np.argsort(off)
        th = d["theta"][sel][order] / truth[0]
        for q, n in enumerate(NAMES):
            if q != p:
                ax[k].plot(np.exp(off[order]), th[:, q], "o-", ms=3, color=COL[n], label=n)
        ax[k].plot(np.exp(off[order]), np.exp(off[order]), "k:", lw=0.8)
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")
        ax[k].set_xlabel(f"{pname} fixed at (x truth)")
        ax[k].set_title(f"How the others compensate when {pname} is wrong")
    ax[0].set_ylabel("refitted value / truth")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_compensation.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- fig 7: multi-start recovery
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    tables["multistart"] = {}
    for k, (kd, tr) in enumerate([(a, b) for a in ("multistart", "multistart_raw") for b in TRAIN_SETS]):
        sel = (kind == kd) & (d["job_train"] == tr)
        err = 100 * (d["theta"][sel] / truth[d["job_teacher"][sel]] - 1)
        for p, n in enumerate(NAMES):
            ax[k].scatter(p + rng.uniform(-0.25, 0.25, err.shape[0]), np.clip(err[:, p], -80, 80), s=7,
                          color=COL[n], alpha=0.6)
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].axhspan(-5, 5, color="g", alpha=0.08)
        ax[k].set_xticks(range(len(NAMES)), NAMES)
        nice = ("log-scaled" if kd == "multistart" else "neurofe-style raw") + (", 1 amplitude" if tr == "1amp" else ", 3 amplitudes")
        ax[k].set_title(nice)
        amp_idx = [list(AMPS).index(a) for a in TRAIN_SETS[tr]]
        tables["multistart"][f"{kd}/{tr}"] = {
            "within_5pct": dict(zip(NAMES, (np.abs(err) <= 5).mean(0).round(2).tolist())),
            "within_20pct": dict(zip(NAMES, (np.abs(err) <= 20).mean(0).round(2).tolist())),
            "median_abs_err_pct": dict(zip(NAMES, np.median(np.abs(err), 0).round(1).tolist())),
            "median_final_loss": float(np.median(d["hist_loss"][-1, sel])),
            "median_rel_rmse_by_amp": dict(zip([str(int(a)) for a in AMPS], np.median(d["rel"][sel], 0).round(4).tolist())),
            "median_train_dchi2": float(np.median(d["resid"][sel][:, amp_idx].sum(1) / SIGMA ** 2)),
            "n_fits": int(sel.sum()),
        }
    ax[0].set_ylabel("fitted vs truth, % (clipped at +/-80)")
    fig.suptitle("Recovery from 6 starts x 10 teachers (green band = within 5%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_recovery.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- fig 8: loss curves + generalisation
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for kd, ls in (("multistart", "-"), ("multistart_raw", "--")):
        for tr, c in zip(TRAIN_SETS, ("C0", "C1")):
            sel = (kind == kd) & (d["job_train"] == tr)
            ax[0].semilogy(np.median(d["hist_loss"][:, sel], 1), ls=ls, color=c,
                           label=f"{'log' if kd == 'multistart' else 'raw'}, {tr}")
    ax[0].set_xlabel("Adam iteration")
    ax[0].set_ylabel("median relative MSE (training)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Training loss")
    for tr, c in zip(TRAIN_SETS, ("C0", "C1")):
        sel = (kind == "multistart") & (d["job_train"] == tr)
        med = np.median(d["rel"][sel], 0)
        ax[1].plot(AMPS, med, "o-", color=c, label=f"trained on {TRAIN_SETS[tr]}")
        for a in TRAIN_SETS[tr]:
            j = list(AMPS).index(a)
            ax[1].plot(a, med[j], "s", ms=9, mfc="none", color=c)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(AMPS, [str(int(a)) for a in AMPS])
    ax[1].set_xlabel("pulse amplitude (squares = used for training)")
    ax[1].set_ylabel("median relative RMSE vs teacher")
    ax[1].set_title("Generalisation to unseen amplitudes (log-scaled fits)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_loss_generalisation.png", dpi=150)
    plt.close(fig)

    save_json(tables, OUT / "tables.json")
    print(json.dumps({k: tables[k] for k in ("crb_median_pct", "crb_combo_median_pct", "multistart",
                                              "peak_uV_nominal_by_amp")}, indent=1))


if __name__ == "__main__":
    main()
