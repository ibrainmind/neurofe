# Models

neurofe provides two neural mass model classes. Both inherit from `BaseModel` and share the same `fit()` / `predict()` / `lesion()` API.

## JansenRit

The Jansen-Rit model simulates EEG signals using coupled neural mass columns with excitatory and inhibitory populations.

### Constructor

```python
from neurofe import JansenRit

model = JansenRit(
    sc=sc,                    # (n_regions, n_regions) structural connectivity
    leadfield=lm,             # (n_channels, n_regions) leadfield matrix
    dist=dist,                # (n_regions, n_regions) distance matrix
    step_size=0.0001,         # integration step size in seconds (0.1 ms)
    tr=0.001,                 # batch temporal resolution in seconds (1 ms)
    batch_size=50,            # time points per batch
    fit_connectivity=True,    # learn long-range connection gains (w_ff/w_bb/w_ll)
    param=None,               # JRParams object (None = paper defaults)
)
```

Whether the leadfield itself is learned is controlled by how `lm` is wrapped
inside `JRParams` (see below), not by a constructor flag.

### Configuring parameters with JRParams

Each parameter is wrapped in a `Par` object:

- `Par(value)` — fixed (not optimized).
- `Par(value, prior_mean, prior_std, fit_hyper=True)` — learned with a
  Gaussian prior `N(prior_mean, prior_std)` for regularization.

```python
from neurofe import JRParams
from neurofe.models.utils import Par

# Use all defaults (from Momi et al. 2022) — every parameter fixed
params = JRParams()

# Override specific parameters
params = JRParams(
    g=Par(500, 500, 100, True),     # global coupling: fitted with prior N(500, 100)
    A=Par(3.25),                    # excitatory gain: fixed at 3.25
    std_in=Par(100, 100, 50, True), # input noise std: fitted with prior N(100, 50)
)

model = JansenRit(sc=sc, leadfield=lm, dist=dist, param=params)
```

#### Default parameters

| Parameter | Default | Description |
|---|---|---|
| `A` | 3.25 | Excitatory synaptic gain |
| `a` | 100 | Excitatory time constant (1/s) |
| `B` | 22 | Inhibitory synaptic gain |
| `b` | 50 | Inhibitory time constant (1/s) |
| `g` | 1000 | Global coupling strength |
| `c1` | 135 | Connectivity constant C1 |
| `c2` | 108 | Connectivity constant C2 (0.8 * C1) |
| `c3` | 33.75 | Connectivity constant C3 (0.25 * C1) |
| `c4` | 33.75 | Connectivity constant C4 (0.25 * C1) |
| `std_in` | 100 | Input noise standard deviation |
| `vmax` | 5 | Maximum firing rate |
| `v0` | 6 | Firing threshold |
| `r` | 0.56 | Sigmoid steepness |
| `y0` | 2 | EEG output baseline |
| `mu` | 0.5 | Conduction velocity parameter |
| `k` | 5 | Stimulation gain |
| `cy0` | 5 | EEG scaling coefficient |
| `ki` | 1 | Stimulation input gain |

### State variables

The JR model tracks 6 state variables per node:

| State | Name in results | Description |
|---|---|---|
| E | `result.E_test` | Excitatory post-synaptic potential |
| Ev | `result.Ev_test` | Excitatory derivative |
| I | `result.I_test` | Inhibitory post-synaptic potential |
| Iv | `result.Iv_test` | Inhibitory derivative |
| P | `result.P_test` | Pyramidal membrane potential |
| Pv | `result.Pv_test` | Pyramidal derivative |

EEG is computed as: `eeg = leadfield @ (E - I)` (scaled).

## WongWangDeco

The Wong-Wang-Deco model simulates BOLD signals using mean-field reduction of spiking neurons. The parameter framework is defined but **the forward model is not yet implemented**.

```python
from neurofe import WongWangDeco, WWDParams

model = WongWangDeco(sc=sc, leadfield=lm, dist=dist)
# model.fit(...)  # raises NotImplementedError
```

Contributions to implement the forward model are welcome.

## fit()

Train the model by fitting parameters to empirical EEG data.

```python
result = model.fit(
    eeg,                      # (n_channels, n_timepoints) empirical EEG
    epochs=120,               # number of training epochs
    lr=0.05,                  # Adam optimizer learning rate
    stimulus=0,               # external stimulation (0 = none)
    loss="rmse",              # loss function: "rmse", "beamform", or "fc"
    early_stop_corr=0.85,     # FC correlation threshold for early stopping
    early_stop_after=110,     # minimum epoch before early stopping
)
```

### Loss functions

| Loss | Description |
|---|---|
| `"rmse"` | Root mean squared error between simulated and empirical EEG |
| `"beamform"` | Beamforming-based loss using the leadfield |
| `"fc"` | Functional connectivity correlation (Pearson r) |

### Stimulus format

The stimulus array has shape `(n_regions, hidden_size, n_timepoints)` where `hidden_size = tr / step_size` (default 10). Example TMS pulse:

```python
u = np.zeros((n_regions, 10, n_timepoints))
u[:, :, 110:120] = 1000  # pulse at timepoints 110-120
```

## predict()

Run inference with the trained model.

```python
result = model.predict(
    eeg,                      # empirical EEG (for determining time length)
    stimulus=0,               # external stimulation
    warmup_batches=20,        # warm-up batches before recording output
)
```

## SimulationResult

The `result` object returned by `fit()` and `predict()` contains:

| Attribute | Description |
|---|---|
| `result.eeg_train` | Simulated EEG from training |
| `result.eeg_test` | Simulated EEG from prediction |
| `result.E_test`, `result.I_test`, ... | State variable time series |
| `result.loss` | Training loss history |
| `result.weights` | Structural connectivity weight history |

## Metrics

The `neurofe.metrics` module provides goodness-of-fit metrics for evaluating how well simulated signals match empirical data (see Momi et al. 2022).

```python
import neurofe

# After fitting and predicting
result = model.predict(eeg)

# Individual metrics
r, p = neurofe.metrics.pearson_correlation(result.eeg_test, eeg)
sim = neurofe.metrics.cosine_similarity(result.eeg_test, eeg)
fc_r = neurofe.metrics.fc_correlation(result.eeg_test, eeg)
error = neurofe.metrics.rmse(result.eeg_test, eeg)

# Or compute all at once
m = neurofe.metrics.evaluate(result.eeg_test, eeg)
print(f"Mean Pearson r: {m['pearson_r'].mean():.3f}")
print(f"FC correlation: {m['fc_correlation']:.3f}")
print(f"RMSE: {m['rmse']:.4f}")
```

| Function | Returns | Description |
|---|---|---|
| `pearson_correlation(sim, emp)` | `(r, p)` arrays `(n_channels,)` | Per-channel Pearson correlation |
| `cosine_similarity(sim, emp)` | array `(n_channels,)` | Per-channel cosine similarity |
| `fc_correlation(sim, emp)` | float | Correlation between FC matrices |
| `rmse(sim, emp)` | float | Root mean squared error |
| `evaluate(sim, emp)` | dict | All of the above in one call |

## Standalone NumPy simulator

For quick parameter exploration without PyTorch, use the standalone simulator:

```python
from neurofe.sim import run_net_sim

df = run_net_sim(
    K=5,                         # number of nodes
    T=30000,                     # simulation length (time steps)
    weights=weights,             # (K, K) connectivity matrix
    delays=delays,               # (K, K) delay matrix
    g=1.0,                       # global coupling
    stim_nodes=[0],              # stimulate node 0
    stim_type='square',          # square pulse
    stim_on=5000, stim_off=5100, # pulse timing
    stim_amp=5.0,                # pulse amplitude
)

# Returns a pandas DataFrame with columns:
# 'pyc_c', 'pyc_v', 'ein_c', 'ein_v', 'iin_c', 'iin_v', 'stim'
```
