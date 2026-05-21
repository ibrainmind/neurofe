"""
Example: Train JR model on synthetic data (no real data needed)
================================================================

This script demonstrates the full pipeline using randomly generated
synthetic data, so you can test the code without having access to
the actual TMS-EEG dataset.

Usage:
    python example_synthetic_data.py
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from neurofe import JansenRit
from neurofe.models.jansen_rit import JRParams


def generate_synthetic_data(node_size, output_size, n_timepoints, n_trials):
    """Generate synthetic SC, leadfield, distance, and EEG data for testing."""

    sc = np.random.rand(node_size, node_size) * 0.01
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0)
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    dist = np.random.rand(node_size, node_size) * 10
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0)

    lm = np.random.randn(output_size, node_size) * 0.01

    eeg = np.random.randn(n_trials, output_size, n_timepoints) * 1e-5

    stim_weights = np.random.rand(node_size)

    return sc, dist, lm, eeg, stim_weights


def main():
    start_time = time.time()

    # =========================================================================
    # Configuration
    # =========================================================================
    node_size = 20
    output_size = 10
    n_timepoints = 200
    n_trials = 5
    batch_size = 20
    step_size = 0.0001
    num_epochs = 5
    tr = 0.001

    # =========================================================================
    # Generate synthetic data
    # =========================================================================
    print("Generating synthetic data...")
    sc, dist, lm, eeg, stim_weights = generate_synthetic_data(
        node_size, output_size, n_timepoints, n_trials)

    ki0 = stim_weights[:, np.newaxis]

    # =========================================================================
    # Configure JR parameters (override defaults for fitted params)
    # =========================================================================
    print("Configuring JR model...")

    par = JRParams(
        a=[100, 0.5],
        b=[50, 1],
        g=[1000, 0.1],
        c1=[135, 0.2],
        c2=[135 * 0.8, 0.4],
        c3=[135 * 0.25, 0.8],
        c4=[135 * 0.25, 0.8],
        std_in=[100, 10],
        mu=[1., 2.5],
        k=[10, 0.3],
        ki=[ki0, 0],
    )

    # =========================================================================
    # Build model — sizes inferred from data
    # =========================================================================
    model = JansenRit(
        sc=sc, leadfield=lm, dist=dist,
        step_size=step_size, tr=tr, batch_size=batch_size,
        fit_connectivity=True,
        param=par,
    )

    # =========================================================================
    # Train
    # =========================================================================
    print(f"Training for {num_epochs} epochs...")

    u = np.zeros((node_size, int(tr / step_size), n_timepoints))
    u[:, :, 50:60] = 1000

    result = model.fit(eeg, epochs=num_epochs, stimulus=u)

    print(f"Final loss: {result.loss[-1]:.4f}")

    # =========================================================================
    # Inference
    # =========================================================================
    print("Running inference...")

    u_test = np.zeros((node_size, int(tr / step_size), n_timepoints))
    u_test[:, :, 50:60] = 1000

    result = model.predict(eeg, stimulus=u_test)

    # =========================================================================
    # Visualize
    # =========================================================================
    print("Plotting...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(result.loss)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')

    axes[0, 1].plot(result.eeg_test.T)
    axes[0, 1].set_title('Simulated EEG (test)')
    axes[0, 1].set_xlabel('Time (samples)')

    axes[1, 0].plot((result.E_test - result.I_test).T)
    axes[1, 0].set_title('Source E-I balance (test)')
    axes[1, 0].set_xlabel('Time (samples)')

    axes[1, 1].plot(result.eeg_train.T)
    axes[1, 1].set_title('Simulated EEG (train, last epoch)')
    axes[1, 1].set_xlabel('Time (samples)')

    plt.tight_layout()
    plt.savefig('synthetic_results.png', dpi=150)
    print("Saved plot to synthetic_results.png")
    plt.show()

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
