import numpy as np
import neurofe

# User handles their own data loading — any format they want
sc = np.load("sc.npy")               # (n_regions, n_regions)
lm = np.load("leadfield.npy")        # (n_channels, n_regions)
dist = np.load("dist.npy")           # (n_regions, n_regions)
eeg = np.load("eeg.npy")             # (n_channels, n_timepoints)

# Create model — sizes inferred from data, defaults from paper
model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist)

# Define stimulus
u = np.zeros((sc.shape[0], 10, eeg.shape[-1]))
u[:, :, 110:120] = 1000  # TMS pulse

# Fit to data
result = model.fit(eeg, epochs=120, stimulus=u)

# Predict
result = model.predict(eeg, stimulus=u)

# Access results
result.eeg_test       # simulated EEG
result.E_test         # excitatory activity
result.I_test         # inhibitory activity
result.loss           # training loss history

# Evaluate goodness of fit
m = neurofe.metrics.evaluate(result.eeg_test, eeg)
m['pearson_r']          # per-channel Pearson correlation
m['cosine_similarity']  # per-channel cosine similarity
m['fc_correlation']     # functional connectivity matrix correlation
m['rmse']               # root mean squared error

# Virtual dissection — lesion nodes and observe effect on EEG
lesioned_model = model.lesion(nodes=[5, 10])   # returns a new model
result_lesioned = lesioned_model.predict(eeg, stimulus=u)
result_lesioned.eeg_test  # simulated EEG without nodes 5 and 10
