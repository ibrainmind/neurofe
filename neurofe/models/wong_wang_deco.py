from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from .base import BaseModel
from .utils import Par


class WWDParams:
    """Parameter specification for the Wong-Wang-Deco neural mass model.

    Each parameter is a ``Par`` object. Use ``Par(value)`` for fixed
    parameters and ``Par(value, prior_mean, prior_std, True)`` for
    fitted parameters.
    """

    def __init__(self, **kwargs) -> None:
        defaults: dict[str, Par] = {
            "std_in": Par(0.02),
            "std_out": Par(0.02),
            "W_E": Par(1.),
            "tau_E": Par(100.),
            "gamma_E": Par(0.641 / 1000.),
            "W_I": Par(0.7),
            "tau_I": Par(10.),
            "gamma_I": Par(1. / 1000.),
            "I_0": Par(0.32),
            "I_external": Par(0.),
            "g": Par(20.),
            "g_EE": Par(.1),
            "g_IE": Par(.1),
            "g_EI": Par(0.1),
            "aE": Par(310),
            "bE": Par(125),
            "dE": Par(0.16),
            "aI": Par(615),
            "bI": Par(177),
            "dI": Par(0.087),
            "alpha": Par(0.32),
            "rho": Par(0.34),
            "k1": Par(2.38),
            "k2": Par(2.0),
            "k3": Par(0.48),
            "V": Par(.02),
            "E0": Par(0.34),
            "tau_s": Par(0.65),
            "tau_f": Par(0.41),
            "tau_0": Par(0.98),
            "mu": Par(0.5),
        }
        for var, val in defaults.items():
            setattr(self, var, val)
        for var, val in kwargs.items():
            setattr(self, var, val)


class WongWangDeco(BaseModel):
    """Wong-Wang-Deco neural mass model for simulating BOLD signals.

    This model uses a mean-field reduction of spiking neurons with
    NMDA/GABA synaptic dynamics to simulate resting-state BOLD signals.

    Note: forward() is not yet implemented.
    """

    state_names: list[str] = ['E', 'I', 'x', 'f', 'v', 'q']
    model_name: str = "WWD"
    output_name: str = "bold"
    pop_size: int = 1
    state_size: int = 6

    def __init__(
        self,
        sc: npt.NDArray[np.floating],
        leadfield: npt.NDArray[np.floating],
        dist: npt.NDArray[np.floating],
        step_size: float = 0.0001,
        tr: float = 0.001,
        batch_size: int = 50,
        fit_connectivity: bool = True,
        param: WWDParams | None = None,
    ) -> None:
        super().__init__(
            sc=sc, leadfield=leadfield, dist=dist,
            step_size=step_size, tr=tr, batch_size=batch_size,
            fit_connectivity=fit_connectivity, param=param)

    def _default_params(self) -> WWDParams:
        return WWDParams()

    def get_initial_state(self) -> torch.Tensor:
        return torch.tensor(
            0.45 * np.random.uniform(0, 1, (self.node_size, self.state_size))
            + np.array([0, 0, 0, 1.0, 1.0, 1.0]),
            dtype=torch.float32)

    def forward(
        self,
        external: torch.Tensor,
        hx: torch.Tensor,
        hE: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError(
            "WWD forward model not yet implemented. "
            "Contributions welcome."
        )
