"""Anatomy for the synthetic brain: SC, distances and leadfield from PyTepFit's bundled data.

Also builds smaller sub-networks (for fast unit tests) and stimulation-site masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import PYTEPFIT_DATA

# PyTepFit convention: the analysis window is samples 900:1300 of the 2000-sample
# epoch (1 kHz), and the stimulus pulse sits at window samples 110:120.
WINDOW = (900, 1300)
PULSE = (110, 120)
PULSE_AMP = 1000.0


@dataclass
class Anatomy:
    sc: np.ndarray        # (N, N) normalised structural connectivity
    dist: np.ndarray      # (N, N) inter-parcel distance, mm
    lm: np.ndarray        # (C, N) leadfield
    parcels: np.ndarray   # indices into the full 200-parcel atlas

    @property
    def n_nodes(self) -> int:
        return self.sc.shape[0]

    @property
    def n_channels(self) -> int:
        return self.lm.shape[0]


def normalise_sc(counts: np.ndarray) -> np.ndarray:
    """Same normalisation as neurofe's example_train_jr.py: symmetrise, log1p, unit norm."""
    sc = 0.5 * (counts + counts.T)
    sc = np.log1p(sc)
    return sc / np.linalg.norm(sc)


def load_raw():
    import pandas as pd
    d = PYTEPFIT_DATA
    counts = pd.read_csv(d / "Schaefer2018_200Parcels_7Networks_count.csv",
                         header=None, sep=" ").values.astype(float)
    dist = pd.read_csv(d / "Schaefer2018_200Parcels_7Networks_distance.csv",
                       header=None, sep=" ").values.astype(float)
    lm = np.load(d / "leadfield", allow_pickle=True).astype(float)
    return counts, dist, lm


def load_anatomy(n_nodes: int | None = None) -> Anatomy:
    """Full 200-parcel anatomy, or an evenly spread sub-network of n_nodes parcels.

    The sub-network re-normalises SC over the kept parcels and keeps their true
    distances and leadfield columns, so delays and sensor geometry stay realistic.
    """
    counts, dist, lm = load_raw()
    n_full = counts.shape[0]
    if n_nodes is None or n_nodes >= n_full:
        idx = np.arange(n_full)
    else:
        idx = np.linspace(0, n_full - 1, n_nodes).round().astype(int)
    sub = counts[np.ix_(idx, idx)]
    return Anatomy(sc=normalise_sc(sub), dist=dist[np.ix_(idx, idx)],
                   lm=lm[:, idx], parcels=idx)


def load_real_eeg() -> np.ndarray:
    """Real TMS-EEG evoked responses, (subjects, channels, 400) in the PyTepFit window."""
    import scipy.io
    x = scipy.io.loadmat(PYTEPFIT_DATA / "only_high_trial.mat")["only_high_trial"]
    return x[:, :, WINDOW[0]:WINDOW[1]].astype(float)


def tract_distance(anat: Anatomy, fill: float = np.inf) -> np.ndarray:
    """Distances with 'no tract' made explicit.

    PyTepFit's distance matrix holds tract lengths and is 0 where two parcels have no
    tract (about 30% of off-diagonal pairs). Read naively, 0 means "adjacent", which
    made every unconnected parcel look like a neighbour of the stimulation site.
    Off-diagonal zeros are replaced by `fill`.
    """
    d = anat.dist.astype(float).copy()
    off = ~np.eye(d.shape[0], dtype=bool)
    d[off & (d == 0)] = fill
    return d


def pick_sites(anat: Anatomy, n_sites: int, seed: int = 0) -> np.ndarray:
    """Spread stimulation centres across the cortex by farthest-point sampling on distance."""
    rng = np.random.default_rng(seed)
    n = anat.n_nodes
    n_sites = min(n_sites, n)
    dist = tract_distance(anat, fill=np.nanmax(anat.dist))   # unconnected = as far as the farthest tract
    chosen = [int(rng.integers(n))]
    while len(chosen) < n_sites:
        dmin = dist[:, chosen].min(axis=1)
        dmin[chosen] = -1
        chosen.append(int(np.argmax(dmin)))
    return np.array(chosen)


def site_weights(anat: Anatomy, centre: int, sigma_mm: float = 10.0) -> np.ndarray:
    """Per-node stimulation weights `ki` (N, 1): a Gaussian fall-off around the centre parcel.

    Unconnected parcels (tract length 0 in the data) get no stimulation.
    """
    d = tract_distance(anat)[centre]
    w = np.exp(-d ** 2 / (2 * sigma_mm ** 2))
    w[centre] = 1.0
    return w[:, None]


def pulse_stimulus(n_nodes: int, hidden_size: int, n_tp: int = WINDOW[1] - WINDOW[0],
                   pulse=PULSE, amp: float = PULSE_AMP) -> np.ndarray:
    """Stimulus array in neurofe's legacy 3D format (N, hidden_size, n_tp).

    Every node receives the same pulse; `ki` decides who actually feels it.
    """
    u = np.zeros((n_nodes, hidden_size, n_tp))
    u[:, :, pulse[0]:pulse[1]] = amp
    return u
