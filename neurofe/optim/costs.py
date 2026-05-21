from __future__ import annotations

import torch

from ..models.base import BaseModel


class Costs:
    """Loss functions for model fitting.

    Parameters
    ----------
    method : int
        0: Euclidean distance (default for EEG)
        1: Beamforming + distance
        2: Functional connectivity correlation (for BOLD)
    """

    def __init__(self, method: int) -> None:
        self.method: int = method

    def cost_dist(self, sim: torch.Tensor, emp: torch.Tensor) -> torch.Tensor:
        """Root mean squared error between simulated and empirical signals."""
        return torch.sqrt(torch.mean((sim - emp) ** 2))

    def cost_beamform(self, model: BaseModel, emp: torch.Tensor) -> torch.Tensor:
        """Beamforming-based cost using the leadfield matrix."""
        corr: torch.Tensor = torch.matmul(emp, emp.T)
        corr_inv: torch.Tensor = torch.inverse(corr)
        corr_inv_s: torch.Tensor = torch.inverse(torch.matmul(model.lm.T, torch.matmul(corr_inv, model.lm)))
        W: torch.Tensor = torch.matmul(corr_inv_s, torch.matmul(model.lm.T, corr_inv))
        return torch.trace(torch.matmul(W, torch.matmul(corr, W.T)))

    def cost_r(self, logits_series_tf: torch.Tensor, labels_series_tf: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of Pearson correlation between
        simulated and empirical functional connectivity."""
        node_size: int = logits_series_tf.shape[0]

        labels_series_tf_n: torch.Tensor = labels_series_tf - torch.reshape(
            torch.mean(labels_series_tf, 1), [node_size, 1])
        logits_series_tf_n: torch.Tensor = logits_series_tf - torch.reshape(
            torch.mean(logits_series_tf, 1), [node_size, 1])

        cov_sim: torch.Tensor = torch.matmul(logits_series_tf_n, torch.transpose(logits_series_tf_n, 0, 1))
        cov_def: torch.Tensor = torch.matmul(labels_series_tf_n, torch.transpose(labels_series_tf_n, 0, 1))

        FC_sim_T: torch.Tensor = torch.matmul(torch.matmul(
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))), cov_sim),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))))
        FC_T: torch.Tensor = torch.matmul(torch.matmul(
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))), cov_def),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))))

        ones_tri: torch.Tensor = torch.tril(torch.ones_like(FC_T), -1)
        zeros: torch.Tensor = torch.zeros_like(FC_T)
        mask: torch.Tensor = torch.greater(ones_tri, zeros)

        FC_tri_v: torch.Tensor = torch.masked_select(FC_T, mask)
        FC_sim_tri_v: torch.Tensor = torch.masked_select(FC_sim_T, mask)

        FC_v: torch.Tensor = FC_tri_v - torch.mean(FC_tri_v)
        FC_sim_v: torch.Tensor = FC_sim_tri_v - torch.mean(FC_sim_tri_v)

        corr_FC: torch.Tensor = (
            torch.sum(torch.multiply(FC_v, FC_sim_v))
            * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_v, FC_v))))
            * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_sim_v, FC_sim_v))))
        )

        losses_corr: torch.Tensor = -torch.log(0.5000 + 0.5 * corr_FC)
        return losses_corr

    def cost_eff(self, model: BaseModel, sim: torch.Tensor, emp: torch.Tensor) -> torch.Tensor:
        """Select and compute the appropriate loss function."""
        if self.method == 0:
            return self.cost_dist(sim, emp)
        elif self.method == 1:
            return self.cost_beamform(model, emp) + self.cost_dist(sim, emp)
        else:
            return self.cost_r(sim, emp)
