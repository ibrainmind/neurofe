"""
Example: Evaluate goodness of fit between simulated and empirical TMS-EEG
==========================================================================

This script demonstrates how to:
1. Load saved fitting results
2. Compare simulated vs empirical EEG using correlation and cosine similarity

Usage:
    python example_goodness_of_fit.py --data_dir /path/to/data --results_dir /path/to/results
"""

import argparse
import os
import pickle
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


def load_fitting_results(results_dir):
    """Load all subject fitting result files."""
    pck_files = sorted(glob.glob(os.path.join(results_dir, '*_fittingresults_stim_exp.pkl')))
    if pck_files and pck_files[-1].endswith('_fittingresults_stim_exp.pkl'):
        pck_files.pop()
    pck_files.sort(key=lambda var: [
        int(x) if x.isdigit() else x
        for x in re.findall(r'[^0-9]|[0-9]+', var)
    ])
    return pck_files


def compute_channel_correlation(empirical, simulated):
    """Compute per-channel Pearson correlation between empirical and simulated EEG.

    Parameters
    ----------
    empirical : ndarray (n_channels, n_timepoints)
    simulated : ndarray (n_channels, n_timepoints)

    Returns
    -------
    corr_r : ndarray (n_channels,)
    corr_p : ndarray (n_channels,)
    similarity : ndarray (n_channels,)
    """
    n_channels = empirical.shape[0]
    corr_r = np.zeros(n_channels)
    corr_p = np.zeros(n_channels)
    similarity = np.zeros(n_channels)

    for ch in range(n_channels):
        emp_ch = empirical[ch, :]
        sim_ch = simulated[ch, :]

        emp_norm = (emp_ch - emp_ch.min()) / (emp_ch.max() - emp_ch.min())
        sim_norm = (sim_ch - sim_ch.min()) / (sim_ch.max() - sim_ch.min())

        similarity[ch] = 1 - spatial.distance.cosine(emp_norm, sim_norm)

        r, p = scipy.stats.pearsonr(emp_norm, sim_norm)
        corr_r[ch] = r
        corr_p[ch] = p

    return corr_r, corr_p, similarity


def main(data_dir, results_dir):
    # =========================================================================
    # 1. Load empirical data and fitting results
    # =========================================================================
    print("Loading data...")

    high_file = os.path.join(data_dir, 'only_high_trial.mat')
    only_high_trial = scipy.io.loadmat(high_file)['only_high_trial']

    pck_files = load_fitting_results(results_dir)
    n_subjects = min(only_high_trial.shape[0], len(pck_files))
    print(f"  Found {n_subjects} subjects")

    all_sim_EEG = []
    for sbj in range(n_subjects):
        with open(pck_files[sbj], 'rb') as f:
            data = pickle.load(f)
        all_sim_EEG.append(data.output_sim.eeg_test)
    all_sim_EEG = np.array(all_sim_EEG)

    # =========================================================================
    # 2. Per-subject channel-wise correlation
    # =========================================================================
    print("Computing correlations...")

    all_corr_r = np.zeros((n_subjects, only_high_trial.shape[1]))
    all_similarity = np.zeros((n_subjects, only_high_trial.shape[1]))

    for sbj in range(n_subjects):
        emp = only_high_trial[sbj, :, 900:1300]
        sim = all_sim_EEG[sbj]
        corr_r, corr_p, sim_cos = compute_channel_correlation(emp, sim)
        all_corr_r[sbj] = corr_r
        all_similarity[sbj] = sim_cos

    # =========================================================================
    # 3. Plot subject-wise mean correlation
    # =========================================================================
    print("Plotting...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    mean_r = np.mean(all_corr_r, axis=1)
    color_map = plt.get_cmap('coolwarm')
    colors = [color_map(r) for r in mean_r]

    axes[0].bar(range(n_subjects), mean_r, color=colors)
    axes[0].set_xlabel('Subject')
    axes[0].set_ylabel('Mean Pearson r')
    axes[0].set_title('Simulated vs Empirical EEG Correlation')

    mean_sim = np.mean(all_similarity, axis=1)
    colors_sim = [color_map(s) for s in mean_sim]

    axes[1].bar(range(n_subjects), mean_sim, color=colors_sim)
    axes[1].set_xlabel('Subject')
    axes[1].set_ylabel('Mean Cosine Similarity')
    axes[1].set_title('Simulated vs Empirical EEG Similarity')

    plt.tight_layout()
    plt.savefig('goodness_of_fit.png', dpi=150)
    print("Saved plot to goodness_of_fit.png")
    plt.show()

    print(f"\nSummary across {n_subjects} subjects:")
    print(f"  Mean Pearson r:         {mean_r.mean():.3f} +/- {mean_r.std():.3f}")
    print(f"  Mean Cosine Similarity: {mean_sim.mean():.3f} +/- {mean_sim.std():.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate goodness of fit')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory with fitting result .pkl files')
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
