# neurofe

Fit neural mass models to empirical TMS-EEG data using gradient-based optimization.

Based on: Momi et al. (2022) *Stimulation mapping and whole-brain modeling reveal gradients of excitability and recurrence in cortical networks*.

## Features

- **Jansen-Rit neural mass model** — full PyTorch implementation with conduction delays, structural connectivity, and leadfield projection
- **Gradient-based fitting** — fit model parameters to empirical TMS-EEG data using Adam optimizer with Bayesian priors
- **Virtual dissection** — lesion brain regions and observe the effect on simulated EEG
- **Wong-Wang-Deco model** — parameter framework defined (forward model contributions welcome)
- **Standalone NumPy simulator** — lightweight Jansen-Rit simulator for parameter exploration

## Installation

```bash
pip install neurofe
```

From source:

```bash
git clone https://github.com/neurofe/neurofe.git
cd neurofe
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[examples]"   # matplotlib, pandas for running examples
pip install -e ".[viz]"        # mne, nibabel, etc. for brain surface visualization
pip install -e ".[dev]"        # pytest for development
```

## Quick start

```python
import numpy as np
import neurofe

# Load your data as numpy arrays
sc = np.load("sc.npy")           # (n_regions, n_regions) structural connectivity
lm = np.load("leadfield.npy")   # (n_channels, n_regions) leadfield matrix
dist = np.load("dist.npy")      # (n_regions, n_regions) distance matrix
eeg = np.load("eeg.npy")        # (n_channels, n_timepoints) empirical EEG

# Create and fit the model
model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)
result = model.fit(eeg, epochs=120)
result = model.predict(eeg)

# Access results
result.eeg_test    # simulated EEG
result.E_test      # excitatory activity
result.I_test      # inhibitory activity
result.loss        # training loss history
```

## Virtual dissection

Lesion specific brain regions and observe the effect on simulated EEG:

```python
# Lesion nodes 5 and 10 — returns a new model, original is unchanged
lesioned_model = model.lesion(nodes=[5, 10])
result_lesioned = lesioned_model.predict(eeg)
```

## Documentation

- [Getting Started](docs/getting_started.md) — installation, input data, and quick start
- [Models](docs/models.md) — model API, parameters, and loss functions
- [Virtual Dissection](docs/virtual_dissection.md) — lesion API, usage patterns, and interpretation
- [Examples](docs/examples.md) — walkthrough of all example scripts

## Examples

See the [examples/](examples/) directory:

- `example_synthetic_data.py` — full pipeline with randomly generated data (no real data needed)
- `example_train_jr.py` — training on empirical TMS-EEG data
- `example_param_exploration.py` — parameter sweeps with the standalone NumPy simulator
- `example_goodness_of_fit.py` — evaluating model fit quality

## Citation

If you use neurofe in your research, please cite:

```bibtex
@article{momi2022stimulation,
  title={Stimulation mapping and whole-brain modeling reveal gradients of excitability and recurrence in cortical networks},
  author={Momi, Davide and others},
  journal={Nature Communications},
  year={2022}
}
```

## License

MIT. See [LICENSE](LICENSE).
