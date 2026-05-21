"""
Visualization functions for neurofe simulation data.

These functions require optional dependencies: mne, nibabel, surfer, mayavi.
Install them separately if you need source-level visualization.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

try:
    import mne
    import nibabel as nib
    import os.path as op
    from surfer import Brain
    from mayavi import mlab
    from PIL import Image
    from IPython import display
    _HAS_VIZ_DEPS = True
except ImportError:
    _HAS_VIZ_DEPS = False


def _check_viz_deps():
    if not _HAS_VIZ_DEPS:
        raise ImportError(
            "Visualization functions require optional dependencies: "
            "mne, nibabel, pysurfer, mayavi, Pillow. "
            "Install them with: pip install mne nibabel pysurfer mayavi Pillow"
        )


def plot_source_activity(data, fwd, noise_cov, inv, real_src_path, annot2use, subjects_dir, subject,
                        eeg_dir, method='dSPM', hemi='both', clim='auto',
                         time_viewer=True, views=['dorsal'],
                         surface='inflated',
                         alpha=0.5, size=(800, 400)):
    """Compute and plot simulated time series on the cortical surface using MNE.

    Parameters
    ----------
    data : ndarray (n_regions, n_timepoints)
        Simulated source time series.
    fwd : Forward
        MNE forward solution.
    noise_cov : Covariance
        MNE noise covariance.
    inv : InverseOperator
        MNE inverse operator.
    real_src_path : str
        Path to a source-estimate file.
    annot2use : str
        Parcellation annotation name.
    subjects_dir : str
        Path to FreeSurfer subjects directory.
    subject : str
        Subject folder name.
    eeg_dir : str
        Path to EEG epochs file.
    method : str
        Source localization method ('dSPM', 'sLORETA', 'eLORETA'). Default 'dSPM'.
    hemi : str
        Hemisphere to display ('lh', 'rh', 'both', 'split'). Default 'both'.
    clim : str or dict
        Colorbar limits. Default 'auto'.
    time_viewer : bool
        Show time viewer GUI. Default True.
    views : list
        Brain views. Default ['dorsal'].
    surface : str
        FreeSurfer surface ('white', 'inflated', etc.). Default 'inflated'.
    alpha : float
        Surface transparency. Default 0.5.
    size : tuple
        Window size in pixels. Default (800, 400).

    Returns
    -------
    stc_sim : SourceEstimate
        Simulated source estimate.
    sim_EEG : Evoked
        Simulated EEG object.
    """
    _check_viz_deps()

    stcs = mne.read_source_estimate(real_src_path)

    lh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir +
                                    subject + '/label/lh.' + annot2use))[0]
    rh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir +
                                        subject + '/label/rh.' + annot2use))[0]
    rh_regmap_fixed = rh_regmap + np.max(lh_regmap)
    rh_regmap_fixed[rh_regmap_fixed == np.max(lh_regmap)] = 0
    lh_vtx2use = lh_regmap[stcs.vertices[0]]
    rh_vtx2use = rh_regmap_fixed[stcs.vertices[1]]
    regions_idx = np.concatenate([lh_vtx2use, rh_vtx2use])

    leadfield = fwd['sol']['data']
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    leadfield = fwd_fixed['sol']['data']

    new_leadfield = []
    for xx in range(1, np.unique(regions_idx).shape[0]):
        new_leadfield.append(np.mean(np.squeeze(leadfield[:, np.where(regions_idx == xx)]), axis=1))

    new_leadfield = np.array(new_leadfield).T

    EEG = new_leadfield.dot(data).T
    EEG = (EEG.T - EEG.mean(axis=1).T).T
    EEG = EEG - EEG.mean(axis=0)
    simulated_EEG = EEG.T

    epochs = mne.read_epochs(eeg_dir)
    evoked = epochs.average()
    sim_EEG = evoked.copy()
    sim_EEG.data = simulated_EEG * 1e-5
    sim_EEG.times = sim_EEG.times[:simulated_EEG.data.shape[1]]

    inv_sim = mne.minimum_norm.make_inverse_operator(sim_EEG.info, forward=fwd, noise_cov=noise_cov)
    stc_sim = mne.minimum_norm.apply_inverse(sim_EEG, inv, method=method)

    mne.viz.set_3d_backend("notebook")
    os.environ['SUBJECTS_DIR'] = subjects_dir
    brain = stc_sim.plot(subjects_dir=subjects_dir, subject=subject, clim=clim,
                         hemi=hemi, time_viewer=time_viewer, views=views,
                         surface=surface, alpha=alpha, size=size)

    return stc_sim, sim_EEG
