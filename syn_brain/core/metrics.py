"""Metrics for the recovery study: EEG fit by window, noise ceiling, parameter recovery."""

from __future__ import annotations

import numpy as np

from .anatomy import PULSE


def baseline_correct(eeg: np.ndarray, pre: int = 100) -> np.ndarray:
    """Subtract each channel's mean over the first `pre` samples (before the pulse)."""
    return eeg - eeg[..., :pre].mean(axis=-1, keepdims=True)


def gmfa(eeg: np.ndarray) -> np.ndarray:
    """Global mean field amplitude: std across channels at each time point."""
    return eeg.std(axis=-2)


def _pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(-1) / den


# Windows in samples of the 400-sample analysis window (1 kHz). The pulse is at
# samples 110:120, so "early" = first 150 ms after pulse onset, "late" = after that.
WINDOWS = {
    "full": (0, 400),
    "early": (PULSE[0], PULSE[0] + 150),
    "late": (PULSE[0] + 150, 400),
}


def eeg_fit(sim: np.ndarray, ref: np.ndarray) -> dict:
    """Per-channel Pearson (mean and median), GMFA correlation and relative RMSE, by window."""
    sim = baseline_correct(sim)
    ref = baseline_correct(ref)
    out = {}
    for name, (a, b) in WINDOWS.items():
        s, r = sim[:, a:b], ref[:, a:b]
        rho = _pearson_rows(s, r)
        g = float(np.corrcoef(gmfa(s), gmfa(r))[0, 1])
        rmse = float(np.sqrt(np.mean((s - r) ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12))
        out[name] = {"pearson_mean": float(np.nanmean(rho)), "pearson_median": float(np.nanmedian(rho)),
                     "gmfa_r": g, "rel_rmse": rmse}
    return out


def noise_ceiling(trials: np.ndarray, n_splits: int = 20, seed: int = 0) -> dict:
    """Split-half reliability of a trial-averaged response, Spearman-Brown corrected.

    The best Pearson any model can expect against the full average, by window.
    """
    rng = np.random.default_rng(seed)
    n = trials.shape[0]
    res = {k: [] for k in WINDOWS}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        h1 = baseline_correct(trials[perm[: n // 2]].mean(0))
        h2 = baseline_correct(trials[perm[n // 2:]].mean(0))
        for k, (a, b) in WINDOWS.items():
            r = np.nanmean(_pearson_rows(h1[:, a:b], h2[:, a:b]))
            res[k].append(2 * r / (1 + r))
    return {k: float(np.mean(v)) for k, v in res.items()}


def tep_features(eeg: np.ndarray) -> dict:
    """Coarse shape features used to check that synthetic TEPs look like real ones."""
    x = baseline_correct(eeg)
    g = gmfa(x)
    post = g[PULSE[1]:]
    return {
        "baseline_std_uV": float(x[:, :100].std()),
        "post_peak_abs_uV": float(np.abs(x[:, PULSE[1]:]).max()),
        "gmfa_peak_uV": float(post.max()),
        "gmfa_peak_latency_ms": int(np.argmax(post) + PULSE[1] - PULSE[0]),
        "gmfa_late_over_early": float(g[WINDOWS["late"][0]:].mean() / (g[WINDOWS["early"][0]:WINDOWS["early"][1]].mean() + 1e-12)),
    }


def param_recovery(fitted: dict, truth: dict, start: dict) -> dict:
    """Relative error per parameter, and how far the fit moved from its start towards the truth.

    progress = 1 means the fit landed on the truth, 0 means it stayed at the start
    (nominal), negative means it moved away.
    """
    out = {}
    for n in truth:
        t, f, s = truth[n], fitted[n], start[n]
        rel = (f - t) / t
        gap = t - s
        progress = (f - s) / gap if abs(gap) > 1e-9 * max(abs(t), 1) else float("nan")
        out[n] = {"truth": t, "fitted": f, "start": s, "rel_error": rel, "progress": progress}
    return out
