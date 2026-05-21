# Examples

All example scripts are in the `examples/` directory. Install the example dependencies first:

```bash
pip install -e ".[examples]"
```

## usecase.py — minimal API overview

A concise script showing the core workflow: load data, fit, predict, and lesion.

```bash
python examples/usecase.py
```

Covers:
- Data loading
- Model creation
- Fitting to empirical EEG
- Running prediction
- Virtual dissection

## example_synthetic_data.py — full pipeline without real data

Demonstrates the entire pipeline using randomly generated synthetic data. No real TMS-EEG dataset required.

```bash
python examples/example_synthetic_data.py
```

What it does:
1. Generates random SC, leadfield, distance, and EEG matrices (20 nodes, 10 channels, 200 timepoints)
2. Creates a `JansenRit` model with custom `JRParams`
3. Trains for 5 epochs
4. Runs prediction
5. Plots training loss and simulated vs empirical EEG

This is the best starting point if you want to verify your installation works.

## example_train_jr.py — training on real TMS-EEG data

Full training pipeline with empirical data loading, model fitting, and visualization.

```bash
python examples/example_train_jr.py --data_dir /path/to/data
```

Expects the following files in `data_dir`:
- `Schaefer2018_200Parcels_7Networks_count.csv` — structural connectivity
- `Schaefer2018_200Parcels_7Networks_distance.csv` — distance matrix
- `only_high_trial.mat` — empirical TMS-EEG data
- `leadfield` — leadfield matrix
- `stim_weights.npy` — stimulation weights

What it does:
1. Loads and preprocesses all input data
2. Configures `JRParams` with fitted parameter ranges
3. Trains for 120 epochs with RMSE loss
4. Runs inference
5. Plots E-I activity, simulated EEG, and empirical comparison

## example_param_exploration.py — parameter sweeps

Uses the standalone NumPy simulator (`run_net_sim`) for fast parameter exploration without the PyTorch fitting pipeline.

```bash
python examples/example_param_exploration.py
```

What it does:
1. **Single-node simulation** — runs one JR column with default parameters and plots pyramidal, excitatory, and inhibitory currents
2. **Coupling sweep** — varies global coupling strength `g` across a 5-node network and shows how synchronization changes
3. **Stimulation response** — applies a square TMS pulse to one node and visualizes the network's response

Generates three plots: `single_node_sim.png`, `coupling_sweep.png`, `stimulation_response.png`.

## example_goodness_of_fit.py — evaluating model quality

Post-training evaluation of how well the fitted model matches empirical data.

```bash
python examples/example_goodness_of_fit.py --data_dir /path/to/data --results_dir /path/to/results
```

What it does:
1. Loads saved fitting results (pickle files) from `results_dir`
2. Computes per-channel Pearson correlation between simulated and empirical EEG
3. Calculates cosine similarity
4. Produces summary statistics and visualizations
