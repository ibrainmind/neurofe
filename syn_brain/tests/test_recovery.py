"""Fast guard for the recovery pipeline (about 1 minute).

The full Phase 1 gate (120 epochs, several teachers) is scripts/recover_small.py;
this test only checks that the pipeline runs end to end against a clean R0 teacher
on a 10-node network and that the optimiser actually changes the parameters.

It deliberately does not assert that training error falls epoch by epoch:
JansenRit.fit runs one continuous simulation across epochs with no warm-up, so
epoch 0 starts from a quiet state and has the lowest error; epochs 1-3 jump about
3x before slowly declining (Phase 1 finding, see README).
"""

import numpy as np
import torch

from syn_brain.core import paths  # noqa: F401
from syn_brain.core.anatomy import load_anatomy, pulse_stimulus, site_weights
from syn_brain.core.params import draw_teacher, nominal_effective
from syn_brain.core.simulator import SimConfig, evoked
from syn_brain.core.student import build_student, fit_student


def test_fit_runs_and_updates_parameters():
    torch.manual_seed(0)
    anat = load_anatomy(10)
    ki = site_weights(anat, 3)
    cfg = SimConfig()
    stim = pulse_stimulus(anat.n_nodes, cfg.hidden_size)
    tp = draw_teacher(anat.n_nodes, seed=5)
    teach, _ = evoked(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim, n_trials=20, cfg=cfg, seed=1)

    epochs = 6
    model = build_student(anat, ki, tier="S1")
    fit = fit_student(model, np.broadcast_to(teach, (epochs,) + teach.shape).copy(),
                      epochs=epochs, stimulus=stim)
    rmse = fit["rmse_per_epoch"]
    assert rmse.shape == (epochs,)
    assert np.all(np.isfinite(rmse))
    start = nominal_effective()
    moved = [n for n, v in fit["fitted"].items() if abs(v - start[n]) > 1e-3 * abs(start[n])]
    assert len(moved) >= 3, f"optimiser barely moved any parameter: {fit['fitted']}"
    # Known limitation: delays are cast to integers, so mu receives no gradient.
    assert abs(fit["fitted"]["mu"] - start["mu"]) < 1e-6
