# Virtual Dissection

Virtual dissection (lesioning) allows you to remove specific brain regions from the network and observe how the simulated EEG changes. This is a key analysis from [Momi et al. (2022)](https://doi.org/10.1038/s41467-022-28462-x) for understanding the causal role of individual regions in cortical dynamics.

## How it works

Calling `model.lesion(nodes)` returns a **new model** (the original is unchanged) with:

1. **Structural connectivity** zeroed out — all rows and columns in the SC matrix for the lesioned nodes are set to zero
2. **Distance matrix** zeroed out — no conduction delays to/from lesioned nodes
3. **Connection gains** zeroed out — fitted `w_bb` weights for lesioned nodes are cleared
4. **Leadfield** zeroed out (optional) — lesioned nodes no longer contribute to simulated EEG

The lesioned nodes still exist in the model (array dimensions don't change), but they are fully disconnected from the network.

## Usage

### PyTorch model

```python
import neurofe

model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)
result = model.fit(eeg, epochs=120, stimulus=u)

# Lesion nodes 5 and 10
lesioned_model = model.lesion(nodes=[5, 10])

# The original model is unchanged
result_intact = model.predict(eeg, stimulus=u)

# Run prediction with the lesioned model
result_lesioned = lesioned_model.predict(eeg, stimulus=u)

# Compare EEG
eeg_intact = result_intact.eeg_test
eeg_lesioned = result_lesioned.eeg_test
```

### Standalone NumPy simulator

```python
from neurofe.sim import run_net_sim

# Intact simulation
df_intact = run_net_sim(K=20, weights=w, delays=d, g=1.0)

# Lesioned simulation
df_lesioned = run_net_sim(K=20, weights=w, delays=d, g=1.0, lesion_nodes=[5, 10])
```

## Parameters

```python
lesioned_model = model.lesion(
    nodes=[5, 10],          # indices of nodes to remove
    remove_leadfield=True,  # also zero out leadfield columns (default: True)
)
```

Setting `remove_leadfield=False` keeps the leadfield intact, so the lesioned nodes still contribute their (now isolated) activity to the EEG. This can be useful for distinguishing between network effects (how does disconnection change dynamics?) and observation effects (what if we just stop measuring this node?).

## Example: systematic lesioning

Lesion each node one at a time and measure the impact on simulated EEG:

```python
import numpy as np
import neurofe

model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)
model.fit(eeg, epochs=120)

# Intact baseline
result_intact = model.predict(eeg)
eeg_intact = result_intact.eeg_test

# Lesion each node and measure EEG change
n_regions = sc.shape[0]
impact = np.zeros(n_regions)

for node in range(n_regions):
    lesioned = model.lesion(nodes=[node])
    result_lesioned = lesioned.predict(eeg)
    eeg_lesioned = result_lesioned.eeg_test

    # Impact = RMSE between intact and lesioned EEG
    impact[node] = np.sqrt(np.mean((eeg_intact - eeg_lesioned) ** 2))

# Nodes with highest impact are most critical to the network
critical_nodes = np.argsort(impact)[::-1]
print("Most critical nodes:", critical_nodes[:5])
```

## Relationship to the paper

In Momi et al. (2022), virtual dissection is used to characterize each brain region's contribution to overall cortical dynamics. Regions with high excitability gradients show larger changes in simulated EEG when lesioned, revealing the network's dependence on specific hubs.
