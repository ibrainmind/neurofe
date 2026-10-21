from __future__ import annotations

import numpy as np
import torch
from torch.nn.parameter import Parameter


class Par:
    """Parameter wrapper for neural mass model fitting.

    Wraps a parameter value with optional prior distribution for
    Bayesian-style regularization during gradient-based fitting.

    Parameters
    ----------
    val : float or ndarray
        Initial parameter value.
    prior_mean : float or ndarray, optional
        Prior mean for regularization. Required if fit_hyper=True.
    prior_std : float or ndarray, optional
        Prior standard deviation (controls regularization strength).
        Required if fit_hyper=True.

        NOTE: PyTepFit's ``ParamsJR`` takes ``[value, v]`` where ``v`` is the
        prior PRECISION, not the standard deviation -- its loss uses ``v``
        directly as ``(lb + relu(v)) * (theta - mean)**2``.  Copying those
        numbers into this slot inverts every prior: the reference's ``g``
        precision of 0.1 becomes 10, pinning the global coupling 100x too
        hard.  Pass ``1 / v`` when porting values from that model.
    fit_hyper : bool
        Whether this parameter should be fitted. Default False.

    Examples
    --------
    >>> Par(3.25)                          # fixed parameter
    >>> Par(3.25, 3.25, 0.1, True)         # fitted with prior N(3.25, 0.1)
    >>> Par(lm, lm, 0.01 * np.abs(lm), True)  # fitted array parameter
    """

    def __init__(self, val, prior_mean=None, prior_std=None, fit_hyper=False):
        if isinstance(val, np.ndarray):
            val_tensor = torch.tensor(val.copy(), dtype=torch.float32)
        else:
            val_tensor = torch.tensor(float(val), dtype=torch.float32)

        self.fit_hyper = fit_hyper

        if fit_hyper and prior_mean is not None and prior_std is not None:
            if isinstance(prior_mean, np.ndarray):
                self.prior_mean = Parameter(
                    torch.tensor(prior_mean.copy(), dtype=torch.float32))
            else:
                self.prior_mean = Parameter(
                    torch.tensor(float(prior_mean), dtype=torch.float32))

            if isinstance(prior_std, np.ndarray):
                pvi = 1.0 / (prior_std + 1e-8)
                self.prior_var_inv = Parameter(
                    torch.tensor(pvi.astype(np.float32)))
            else:
                self.prior_var_inv = Parameter(
                    torch.tensor(1.0 / (float(prior_std) + 1e-8),
                                 dtype=torch.float32))

            self.val = Parameter(val_tensor.clone())
        else:
            self.val = val_tensor
            self.prior_mean = val_tensor.clone()
            self.prior_var_inv = torch.tensor(0.0, dtype=torch.float32)

    def value(self):
        """Return the current parameter value tensor."""
        return self.val


def sys2nd(
    A: torch.Tensor,
    a: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Second-order system dynamics for the Jansen-Rit model.

    Computes: A*a*u - 2*a*v - a^2*x
    This represents the second-order differential equation governing
    post-synaptic potential dynamics.
    """
    return A * a * u - 2 * a * v - a ** 2 * x


def sigmoid(
    x: torch.Tensor,
    vmax: torch.Tensor,
    v0: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid transfer function (wave-to-pulse operator).

    Converts average membrane potential to average firing rate.
    """
    return vmax / (1 + torch.exp(r * (v0 - x)))
