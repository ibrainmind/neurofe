"""Clean fitting loop (Step 2): full-window backprop through core/diffsim.py.

Differences from neurofe's JansenRit.fit, each switchable for the Step 3 ablation:
  - log-scaled parameters z (theta = nominal * exp(z)), no relu floors
  - fractional delays (mu gets a gradient)
  - warm-up before every forward pass, so each fit starts from a settled state
  - the whole 300 ms after the baseline in one backward pass (no 50 ms truncation)
  - noise-free student fitted to a noise-free teacher (R0-clean), or K-trial averages
    with common random numbers
  - several stimulation conditions in one loss
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import diffsim as ds


def fit(target: np.ndarray, net: ds.Net, proto: ds.Protocol, names, z0=None, base=None,
        iters: int = 200, lr: float = 0.03, window=(100, 400), integer_delays=False,
        grad_warmup_ms: int | None = 200, noise_scale=0.0, n_trials=1, noise_seed=0,
        log=None, log_every=10, truth_z=None):
    """Fit log-parameters `names` so the student's EEG matches `target` (S, C, T).

    Parameters not in `names` stay at `base` (default nominal). Loss is the MSE of
    baseline-corrected EEG in `window`, divided by the target's mean square.
    Returns dict with z history, loss history, final theta.
    """
    base = dict(base or ds.NOMINAL_EFF)
    idx = [ds.PARAM_NAMES.index(n) for n in names]
    z_fit = torch.zeros(len(names), dtype=torch.float64) if z0 is None else torch.tensor(z0, dtype=torch.float64)
    z_fit.requires_grad_(True)
    opt = torch.optim.Adam([z_fit], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    y = ds.baseline_correct(torch.tensor(target, dtype=torch.float64))[..., window[0]:window[1]]
    scale = y.pow(2).mean()
    hist_z, hist_loss = [], []
    t0 = time.time()
    for it in range(iters):
        z = torch.zeros(len(ds.PARAM_NAMES), dtype=torch.float64)
        z = z.index_put((torch.tensor(idx),), z_fit)
        e = ds.run(ds.theta_from_z(z, base=base), net, proto, n_trials=n_trials, noise_seed=noise_seed,
                   noise_scale=noise_scale, integer_delays=integer_delays, grad_warmup_ms=grad_warmup_ms)
        e = ds.baseline_correct(e.mean(1))[..., window[0]:window[1]]
        loss = (e - y).pow(2).mean() / scale
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist_loss.append(float(loss))
        hist_z.append(z_fit.detach().numpy().copy())
        if log and (it % log_every == 0 or it == iters - 1):
            msg = f"it {it:4d} loss {float(loss):.3e} ({time.time() - t0:.0f}s)"
            if truth_z is not None:
                err = z_fit.detach().numpy() - truth_z
                msg += "  log-err " + " ".join(f"{n}={v:+.2f}" for n, v in zip(names, err))
            log.info(msg)
    z_final = hist_z[-1]
    theta = {n: base[n] * float(np.exp(z_final[i])) for i, n in enumerate(names)}
    return {"z": np.array(hist_z), "loss": np.array(hist_loss), "theta": theta,
            "seconds": time.time() - t0}
