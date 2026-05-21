from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def generate_noise(
    node_size: int,
    hidden_size: int,
    batch_size: int,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate input and output noise tensors for a forward pass.

    Parameters
    ----------
    node_size : int
        Number of brain regions.
    hidden_size : int
        Number of integration steps per batch element.
    batch_size : int
        Batch size.
    input_size : int
        Number of noise channels per state.

    Returns
    -------
    noise_in : tensor (node_size, hidden_size, batch_size, input_size)
    noise_out : tensor (node_size, batch_size)
    """
    noise_in: torch.Tensor = torch.tensor(
        np.random.randn(node_size, hidden_size, batch_size, input_size),
        dtype=torch.float32)
    noise_out: torch.Tensor = torch.tensor(
        np.random.randn(node_size, batch_size),
        dtype=torch.float32)
    return noise_in, noise_out


def create_batched_input(
    ts: npt.NDArray[np.floating],
    batch_size: int,
    batch_idx: int,
) -> torch.Tensor:
    """Extract a batch of empirical data.

    Parameters
    ----------
    ts : ndarray (n_channels, n_timepoints)
        Single trial/epoch of EEG data.
    batch_size : int
        Number of timepoints per batch.
    batch_idx : int
        Batch index.

    Returns
    -------
    ts_batch : tensor (n_channels, batch_size)
    """
    start: int = batch_idx * batch_size
    end: int = (batch_idx + 1) * batch_size
    return torch.tensor(ts.T[start:end, :].T, dtype=torch.float32)


def normalize_sc(sc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Normalize structural connectivity matrix.

    Parameters
    ----------
    sc : ndarray (node_size, node_size)
        Raw structural connectivity.

    Returns
    -------
    sc_norm : ndarray (node_size, node_size)
        Normalized SC with values in [0, 1].
    """
    sc_norm: npt.NDArray[np.floating] = sc / sc.max()
    return sc_norm
