"""
Example: Parameter exploration with the standalone NumPy JR simulation
=======================================================================

This script demonstrates how to use the NumPy-based Jansen-Rit simulator
to explore how different parameter settings affect neural dynamics,
without running the full PyTorch fitting pipeline.

Usage:
    python example_param_exploration.py
"""

import numpy as np
import matplotlib.pyplot as plt

from neurofe.sim import run_net_sim


def main():
    # =========================================================================
    # 1. Single-node simulation with default parameters
    # =========================================================================
    print("Running single-node simulation...")

    df = run_net_sim(
        K=1,
        T=50000,
        T_transient=10000,
        weights=np.zeros((1, 1)),
        delays=np.zeros((1, 1)).astype(int),
        noise_std=0.005,
        A=3.25,
        B=22.0,
        a=100,
        b=50,
        J=135,
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df['pyc_c'][0])
    axes[0].set_title('Pyramidal current (pyc_c)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(df.index, df['ein_c'][0])
    axes[1].set_title('Excitatory interneuron current (ein_c)')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(df.index, df['iin_c'][0])
    axes[2].set_title('Inhibitory interneuron current (iin_c)')
    axes[2].set_xlabel('Time (a.u.)')
    axes[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('single_node_sim.png', dpi=150)
    print("Saved single_node_sim.png")

    # =========================================================================
    # 2. Sweep over global coupling strength (g)
    # =========================================================================
    print("\nSweeping global coupling strength (g)...")

    n_nodes = 5
    weights = np.random.rand(n_nodes, n_nodes) * 0.01
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0)
    delays = (np.random.rand(n_nodes, n_nodes) * 5).astype(int)
    delays = np.maximum(delays, delays.T)

    g_values = [0.1, 0.5, 1.0, 5.0]
    fig, axes = plt.subplots(len(g_values), 1, figsize=(12, 3 * len(g_values)), sharex=True)

    for i, g_val in enumerate(g_values):
        df = run_net_sim(
            K=n_nodes,
            T=30000,
            T_transient=10000,
            weights=weights,
            delays=delays,
            g=g_val,
            noise_std=0.005,
        )

        eeg_like = df['ein_c'].values - df['iin_c'].values
        axes[i].plot(df.index, eeg_like, alpha=0.7)
        axes[i].set_title(f'g = {g_val}')
        axes[i].set_ylabel('E - I')

    axes[-1].set_xlabel('Time (a.u.)')
    plt.tight_layout()
    plt.savefig('coupling_sweep.png', dpi=150)
    print("Saved coupling_sweep.png")

    # =========================================================================
    # 3. Effect of stimulation
    # =========================================================================
    print("\nSimulating TMS pulse...")

    df_stim = run_net_sim(
        K=n_nodes,
        T=30000,
        T_transient=10000,
        weights=weights,
        delays=delays,
        g=1.0,
        noise_std=0.005,
        stim_nodes=[0],
        stim_type='square',
        stim_on=5000,
        stim_off=5100,
        stim_amp=5.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df_stim.index, df_stim['stim'][0], color='red')
    axes[0].set_title('Stimulation input (node 0)')
    axes[0].set_ylabel('Amplitude')

    eeg_stim = df_stim['ein_c'].values - df_stim['iin_c'].values
    axes[1].plot(df_stim.index, eeg_stim, alpha=0.7)
    axes[1].set_title('E - I response (all nodes)')
    axes[1].set_xlabel('Time (a.u.)')
    axes[1].set_ylabel('E - I')

    plt.tight_layout()
    plt.savefig('stimulation_response.png', dpi=150)
    print("Saved stimulation_response.png")

    plt.show()
    print("Done.")


if __name__ == '__main__':
    main()
