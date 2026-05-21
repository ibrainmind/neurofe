# Getting Started

## Installation

### From PyPI

```bash
pip install neurofe
```

### From source

```bash
git clone https://github.com/neurofe/neurofe.git
cd neurofe
pip install -e .
```

### Optional dependencies

```bash
pip install -e ".[examples]"   # matplotlib, pandas for running examples
pip install -e ".[sim]"        # pandas for the standalone NumPy simulator
pip install -e ".[viz]"        # mne, nibabel, etc. for brain surface visualization
pip install -e ".[dev]"        # pytest for development
```

## Requirements

- Python >= 3.10
- numpy >= 1.21
- torch >= 2.0
- scipy >= 1.7
- scikit-learn >= 1.0

## Input data

neurofe requires three matrices that describe the brain's anatomy, plus empirical EEG data to fit against:

| Input | Shape | Description |
|---|---|---|
| `sc` | `(n_regions, n_regions)` | Structural connectivity matrix (e.g. from diffusion MRI tractography) |
| `leadfield` | `(n_channels, n_regions)` | Leadfield matrix mapping source activity to EEG sensors |
| `dist` | `(n_regions, n_regions)` | Physical distance matrix between regions (used for conduction delays) |
| `eeg` | `(n_channels, n_timepoints)` | Empirical EEG data to fit the model against |

These can come from any source — load them however you prefer (`.npy`, `.mat`, `.csv`, etc.).

## Quick start

```python
import numpy as np
import neurofe

# Load your data
sc = np.load("sc.npy")
lm = np.load("leadfield.npy")
dist = np.load("dist.npy")
eeg = np.load("eeg.npy")

# Create a Jansen-Rit model (sizes are inferred from the data)
model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)

# Define a TMS stimulus
u = np.zeros((sc.shape[0], 10, eeg.shape[-1]))
u[:, :, 110:120] = 1000  # pulse at timepoints 110-120

# Fit model parameters to empirical EEG
result = model.fit(eeg, epochs=120, stimulus=u)

# Run inference with the fitted model
result = model.predict(eeg, stimulus=u)

# Access outputs
result.eeg_test    # simulated EEG
result.E_test      # excitatory activity per node
result.I_test      # inhibitory activity per node
result.loss        # training loss history
```

## Trying without real data

If you don't have real TMS-EEG data, you can test the pipeline with synthetic data:

```python
import numpy as np
import neurofe

n_regions, n_channels, n_timepoints = 20, 10, 200

sc = np.random.rand(n_regions, n_regions) * 0.01
sc = 0.5 * (sc + sc.T)
np.fill_diagonal(sc, 0)

lm = np.random.randn(n_channels, n_regions) * 0.01
dist = np.random.rand(n_regions, n_regions) * 10
dist = 0.5 * (dist + dist.T)
eeg = np.random.randn(n_channels, n_timepoints) * 0.1

model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)
result = model.fit(eeg, epochs=5)
result = model.predict(eeg)
```

See `examples/example_synthetic_data.py` for a complete version with plotting.

## Next steps

- [Models](models.md) — model API and parameter configuration
- [Virtual Dissection](virtual_dissection.md) — lesion brain regions
- [Examples](examples.md) — walkthrough of example scripts
