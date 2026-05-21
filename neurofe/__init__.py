"""
neurofe - Cortical Network Model Fitting
===========================================

A modular framework for fitting neural mass models (Jansen-Rit, Wong-Wang-Deco)
to empirical EEG data using gradient-based optimization.

Basic usage::

    import numpy as np
    import neurofe

    # Your data as numpy arrays
    sc = np.load("sc.npy")           # (n_regions, n_regions)
    lm = np.load("leadfield.npy")    # (n_channels, n_regions)
    dist = np.load("dist.npy")       # (n_regions, n_regions)
    eeg = np.load("eeg.npy")         # (n_channels, n_timepoints)

    params = neurofe.JRParams(
        A=neurofe.Par(3.25, 3.25, 0.1, True),
        g=neurofe.Par(200, 200, 1, True),
        lm=neurofe.Par(lm, lm, 0.01 * (np.abs(lm) + 0.01), True),
    )
    model = neurofe.JansenRit(sc=sc, leadfield=lm, dist=dist, param=params)
    result = model.fit(eeg, epochs=120, stimulus=u)
    result = model.predict(eeg, stimulus=u_test)
"""

__version__ = "0.1.0"

from .models.jansen_rit import JansenRit, JRParams
from .models.wong_wang_deco import WongWangDeco, WWDParams
from .models.base import BaseModel, SimulationResult
from .models.utils import Par
from . import metrics
