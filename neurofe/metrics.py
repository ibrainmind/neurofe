"""Goodness-of-fit metrics for comparing simulated and empirical signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats, spatial


def pearson_correlation(
    simulated: npt.NDArray[np.floating],
    empirical: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Per-channel Pearson correlation between simulated and empirical signals.

    Parameters
    ----------
    simulated : ndarray (n_channels, n_timepoints)
        Simulated signal (e.g. result.eeg_test).
    empirical : ndarray (n_channels, n_timepoints)
        Empirical signal.

    Returns
    -------
    r : ndarray (n_channels,)
        Pearson correlation coefficient per channel.
    p : ndarray (n_channels,)
        Two-tailed p-value per channel.
    """
    n_channels = simulated.shape[0]
    r = np.zeros(n_channels)
    p = np.zeros(n_channels)
    for ch in range(n_channels):
        r[ch], p[ch] = stats.pearsonr(simulated[ch], empirical[ch])
    return r, p


def cosine_similarity(
    simulated: npt.NDArray[np.floating],
    empirical: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Per-channel cosine similarity between simulated and empirical signals.

    Parameters
    ----------
    simulated : ndarray (n_channels, n_timepoints)
        Simulated signal.
    empirical : ndarray (n_channels, n_timepoints)
        Empirical signal.

    Returns
    -------
    similarity : ndarray (n_channels,)
        Cosine similarity per channel (1.0 = identical direction).
    """
    n_channels = simulated.shape[0]
    similarity = np.zeros(n_channels)
    for ch in range(n_channels):
        similarity[ch] = 1 - spatial.distance.cosine(simulated[ch], empirical[ch])
    return similarity


def fc_correlation(
    simulated: npt.NDArray[np.floating],
    empirical: npt.NDArray[np.floating],
) -> float:
    """Pearson correlation between functional connectivity matrices.

    Computes the FC matrix (Pearson correlation across time) for both
    simulated and empirical signals, then correlates the lower-triangular
    elements of the two FC matrices.

    Parameters
    ----------
    simulated : ndarray (n_channels, n_timepoints)
        Simulated signal.
    empirical : ndarray (n_channels, n_timepoints)
        Empirical signal.

    Returns
    -------
    r : float
        Pearson correlation between the lower-triangular FC vectors.
    """
    fc_sim = np.corrcoef(simulated)
    fc_emp = np.corrcoef(empirical)
    mask = np.tril_indices(fc_sim.shape[0], -1)
    r, _ = stats.pearsonr(fc_sim[mask], fc_emp[mask])
    return float(r)


def rmse(
    simulated: npt.NDArray[np.floating],
    empirical: npt.NDArray[np.floating],
) -> float:
    """Root mean squared error between simulated and empirical signals.

    Parameters
    ----------
    simulated : ndarray (n_channels, n_timepoints)
        Simulated signal.
    empirical : ndarray (n_channels, n_timepoints)
        Empirical signal.

    Returns
    -------
    error : float
        RMSE across all channels and timepoints.
    """
    return float(np.sqrt(np.mean((simulated - empirical) ** 2)))


def evaluate(
    simulated: npt.NDArray[np.floating],
    empirical: npt.NDArray[np.floating],
) -> dict[str, object]:
    """Compute all goodness-of-fit metrics at once.

    Parameters
    ----------
    simulated : ndarray (n_channels, n_timepoints)
        Simulated signal.
    empirical : ndarray (n_channels, n_timepoints)
        Empirical signal.

    Returns
    -------
    metrics : dict
        Keys:
        - 'pearson_r': ndarray (n_channels,) per-channel Pearson r
        - 'pearson_p': ndarray (n_channels,) per-channel p-values
        - 'cosine_similarity': ndarray (n_channels,) per-channel cosine similarity
        - 'fc_correlation': float, FC matrix correlation
        - 'rmse': float, root mean squared error
    """
    r, p = pearson_correlation(simulated, empirical)
    return {
        'pearson_r': r,
        'pearson_p': p,
        'cosine_similarity': cosine_similarity(simulated, empirical),
        'fc_correlation': fc_correlation(simulated, empirical),
        'rmse': rmse(simulated, empirical),
    }
