"""
Example: Train a Jansen-Rit model on empirical TMS-EEG data
=============================================================

This script demonstrates the full pipeline:
1. Load empirical data (SC, leadfield, EEG, stimulation weights)
2. Configure JR model parameters
3. Train the model
4. Run inference (test)
5. Visualize results

Usage:
    python example_train_jr.py --data_dir /path/to/data
"""

import argparse
import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from neurofe import JansenRit
from neurofe.models.jansen_rit import JRParams
from neurofe.models.utils import Par


def main(data_dir):
    start_time = time.time()

    # =========================================================================
    # 1. Load data (user handles file formats)
    # =========================================================================
    print("Loading data...")

    sc_file = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_count.csv')
    sc = np.array(pd.read_csv(sc_file, header=None, sep=' ').values)
    sc = 0.5 * (sc + sc.T)
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    dist_file = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_distance.csv')
    dist = np.array(pd.read_csv(dist_file, header=None, sep=' ').values)

    high_file = os.path.join(data_dir, 'only_high_trial.mat')
    data_high = scipy.io.loadmat(high_file)['only_high_trial']
    print(f"  Empirical data shape: {data_high.shape}")

    leadfield_file = os.path.join(data_dir, 'leadfield')
    lm = np.load(leadfield_file, allow_pickle=True)
    print(f"  Leadfield shape: {lm.shape}")

    stim_weights_file = os.path.join(data_dir, 'stim_weights.npy')
    stim_weights = np.load(stim_weights_file)
    ki0 = stim_weights[:, np.newaxis]

    # =========================================================================
    # 2. Configure model
    # =========================================================================
    print("Configuring model...")

    node_size = stim_weights.shape[0]
    output_size = lm.shape[0]
    batch_size = 50
    step_size = 0.0001
    num_epochs = 120
    tr = 0.001
    subject_idx = 0

    # JRParams convention:
    #   Par(val)                              -> fixed (not optimized)
    #   Par(val, prior_mean, prior_std, True) -> learned with Gaussian prior
    par = JRParams(
        # PSP kernels (excitatory: A, a ; inhibitory: B, b)
        A=Par(3.25),                                  # fixed
        a=Par(100, 100, 0.5, True),
        B=Par(22),                                    # fixed
        b=Par(50, 50, 1.0, True),
        # Long-range coupling gain (lateral P -> P)
        g=Par(1000, 1000, 0.1, True),
        # Intracolumnar connectivity
        c1=Par(135, 135, 0.2, True),
        c2=Par(135 * 0.8, 135 * 0.8, 0.4, True),
        c3=Par(135 * 0.25, 135 * 0.25, 0.8, True),
        c4=Par(135 * 0.25, 135 * 0.25, 0.8, True),
        # Stochastic input
        std_in=Par(100, 100, 10.0, True),
        # Sigmoid (kept fixed at canonical Jansen-Rit values)
        vmax=Par(5),
        v0=Par(6),
        r=Par(0.56),
        # EEG observation model — y0 as per-channel offset
        y0=Par(2 * np.ones((output_size, 1))),
        cy0=Par(5),
        # Conduction velocity
        mu=Par(1.0, 1.0, 2.5, True),
        # Stimulation
        k=Par(10, 10, 0.3, True),
        ki=Par(ki0),                                  # per-node weights, fixed
        # Leadfield — fixed at empirical value.
        # To learn it instead, pass:
        #   lm=Par(lm, lm, np.ones((output_size, node_size)), True)
        lm=Par(lm),
    )

    # =========================================================================
    # 3. Build model — sizes inferred from sc and leadfield
    # =========================================================================
    # fit_connectivity=True learns the per-region gains for the three
    # long-range pathways (w_ff, w_bb, w_ll). The leadfield itself is
    # controlled separately by how `lm` is wrapped in JRParams above.
    print("Building JansenRit model...")

    model = JansenRit(
        sc=sc, leadfield=lm, dist=dist,
        step_size=step_size, tr=tr, batch_size=batch_size,
        fit_connectivity=True,
        param=par,
    )

    # =========================================================================
    # 4. Prepare training data
    # =========================================================================
    data_mean = np.array([data_high[subject_idx]] * num_epochs)
    ts = data_mean[:, :, 900:1300]
    print(f"  Training data shape: {ts.shape}")

    u = np.zeros((node_size, 10, 400))
    u[:, :, 110:120] = 1000

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("Training...")

    result = model.fit(ts, epochs=num_epochs, stimulus=u)

    # =========================================================================
    # 6. Inference (test)
    # =========================================================================
    print("Running inference...")

    u_test = np.zeros((node_size, 10, 400))
    u_test[:, :, 110:120] = 1000

    result = model.predict(ts, stimulus=u_test)

    # =========================================================================
    # 7. Visualize results
    # =========================================================================
    print("Plotting results...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # states_test has shape (node_size, pop_size, state_size, n_timepoints).
    # pop_size axis: [P=0, E=1, I=2]; state_size axis: [current=0, voltage=1].
    E_test = result.states_test[:, 1, 0, :]
    I_test = result.states_test[:, 2, 0, :]
    axes[0].plot((E_test - I_test).T)
    axes[0].set_title('Simulated source signal (E - I)')
    axes[0].set_xlabel('Time (samples)')

    axes[1].plot(result.eeg_test.T)
    axes[1].set_title('Simulated EEG')
    axes[1].set_xlabel('Time (samples)')

    axes[2].plot(data_high[subject_idx].T[900:1300, :])
    axes[2].set_title('Empirical EEG')
    axes[2].set_xlabel('Time (samples)')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

    # =========================================================================
    # 8. Save results
    # =========================================================================
    output_file = f'sub_{subject_idx}_fitting_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'model_state': model.state_dict(), 'result': result}, f)
    print(f"Results saved to {output_file}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train JR model on TMS-EEG data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing SC, leadfield, EEG files')
    args = parser.parse_args()
    main(args.data_dir)
