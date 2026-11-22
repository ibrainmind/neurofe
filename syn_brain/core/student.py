"""Student: neurofe's JansenRit fitted by backprop to the teacher's evoked EEG."""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import torch

from .paths import REPO_ROOT  # noqa: F401
from neurofe import JansenRit

from .anatomy import Anatomy, pulse_stimulus
from .params import NOMINAL_RAW, fitted_effective, student_jrparams
from .simulator import SimConfig, evoked


def build_student(anat: Anatomy, ki: np.ndarray, tier: str = "S1", prior: str = "weak",
                  step_size: float = 1e-4, batch_size: int = 50) -> JansenRit:
    par, fit_conn = student_jrparams(tier, anat.lm, ki, prior=prior)
    return JansenRit(sc=anat.sc, leadfield=anat.lm, dist=anat.dist,
                     step_size=step_size, tr=1e-3, batch_size=batch_size,
                     fit_connectivity=fit_conn, param=par)


def fit_student(model: JansenRit, eeg: np.ndarray, epochs: int, lr: float = 0.05,
                stimulus: np.ndarray | None = None, log=None, quiet: bool = True,
                track_every: int = 1) -> dict:
    """Run JansenRit.fit one epoch at a time so parameter trajectories and timing can be logged.

    Calling fit(epochs=1) repeatedly would reset the optimiser and the model state,
    so instead we wrap the model's own fit with epochs=N and record the per-epoch
    printout; parameters are read at the end. For trajectories, use track_every>0
    with fit_student_tracked().
    """
    if stimulus is None:
        stimulus = pulse_stimulus(model.node_size, model.hidden_size, n_tp=eeg.shape[-1])
    t0 = time.time()
    buf = io.StringIO()
    with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
        res = model.fit(eeg, epochs=epochs, lr=lr, stimulus=stimulus)
    elapsed = time.time() - t0
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("epoch")]
    if log is not None and lines:
        log.info("fit done in %.1fs (%.2fs/epoch); last: %s", elapsed, elapsed / epochs, lines[-1])
    n_b = int(eeg.shape[-1] / model.batch_size)
    rmse_epoch = np.asarray(res.loss_main).reshape(epochs, n_b).mean(1) if len(res.loss_main) else None
    return {"seconds": elapsed, "sec_per_epoch": elapsed / epochs, "rmse_per_epoch": rmse_epoch,
            "fitted": fitted_effective(model), "result": res}


def student_raw(model: JansenRit) -> dict:
    return {n: float(getattr(model.param, n).value().detach()) for n in NOMINAL_RAW}


def student_w_bb(model: JansenRit) -> np.ndarray:
    return model.w_bb.detach().numpy().copy()


def student_lm(model: JansenRit) -> np.ndarray:
    return model.param.lm.value().detach().numpy().copy()


def predict_evoked(model: JansenRit, anat: Anatomy, ki: np.ndarray, stimulus: np.ndarray,
                   n_trials: int = 100, noise_scale: float = 1.0, seed: int = 1,
                   cfg: SimConfig | None = None):
    """Trial-averaged student prediction, using the batched simulator with the student's values.

    Averaging many runs (or noise_scale=0) is the fair comparison with a trial-averaged
    teacher; a single noisy neurofe.predict run is not.
    """
    cfg = cfg or SimConfig(step_size=float(model.step_size))
    mean, trials = evoked(student_raw(model), anat.sc, anat.dist, student_lm(model), ki, stimulus,
                          w_bb=student_w_bb(model), n_trials=n_trials, cfg=cfg,
                          noise_scale=noise_scale, seed=seed)
    return mean, trials
