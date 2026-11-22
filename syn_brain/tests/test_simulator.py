"""The batched teacher simulator must reproduce neurofe's JansenRit.forward exactly.

If this fails, the teacher and the student are not the same model, and rung R0
("clean") would silently contain a misspecification.
"""

import numpy as np
import torch

from syn_brain.core import paths  # noqa: F401
from syn_brain.core.anatomy import load_anatomy, pulse_stimulus, site_weights
from syn_brain.core.params import draw_teacher, teacher_jrparams
from syn_brain.core.simulator import random_initial_state, simulate
from neurofe import JansenRit


def _run_both(w_bb_sigma: float, n_nodes: int = 10, T: int = 60):
    anat = load_anatomy(n_nodes)
    tp = draw_teacher(anat.n_nodes, seed=3, w_bb_sigma=w_bb_sigma)
    ki = site_weights(anat, centre=2)
    stim = pulse_stimulus(anat.n_nodes, 10, n_tp=T, pulse=(20, 30))

    hx, hE = random_initial_state(1, anat.n_nodes, np.random.default_rng(0))

    torch.manual_seed(123)
    eeg_ours, _, _ = simulate(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim,
                              w_bb=tp.w_bb, hx=hx.clone(), hE=hE.clone())

    model = JansenRit(sc=anat.sc, leadfield=anat.lm, dist=anat.dist,
                      batch_size=T, fit_connectivity=False,
                      param=teacher_jrparams(tp, anat.lm, ki))
    model.w_bb = torch.tensor(tp.w_bb, dtype=torch.float32)
    ext = torch.zeros((anat.n_nodes, 10, T, 3))
    ext[:, :, :, 0] = torch.tensor(stim, dtype=torch.float32)
    torch.manual_seed(123)
    with torch.no_grad():
        out, _ = model(ext, hx[0].clone(), hE[0].clone())
    return eeg_ours[0], out["eeg"].numpy()


def test_matches_neurofe_plain_sc():
    ours, ref = _run_both(w_bb_sigma=0.0)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-5)


def test_matches_neurofe_perturbed_sc():
    ours, ref = _run_both(w_bb_sigma=0.5)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-5)


def test_batch_trials_are_independent():
    anat = load_anatomy(10)
    tp = draw_teacher(anat.n_nodes, seed=1)
    stim = pulse_stimulus(anat.n_nodes, 10, n_tp=30, pulse=(5, 15))
    eeg, _, _ = simulate(tp.raw, anat.sc, anat.dist, anat.lm, site_weights(anat, 0),
                         stim, n_trials=4, seed=0)
    assert eeg.shape == (4, anat.n_channels, 30)
    assert not np.allclose(eeg[0], eeg[1])
