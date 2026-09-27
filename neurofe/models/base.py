from __future__ import annotations

import copy
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity

from .utils import Par


class SimulationResult:
    """Container for storing model outputs during training and testing.

    Attributes
    ----------
    loss : ndarray
        Training loss history.
    eeg_train : ndarray or None
        Simulated EEG from the last training epoch.
    eeg_test : ndarray or None
        Simulated EEG from prediction/evaluation.
    states_train : ndarray or None
        Model states from the last training epoch.
    states_test : ndarray or None
        Model states from prediction/evaluation.
    weights : list
        Structural connectivity weight history.
    loss_main : ndarray
        Per-batch data-fit term (RMSE), unweighted.  Bounded below by 0,
        so unlike `loss` this is readable as fit quality.
    loss_prior : ndarray
        Per-batch prior term.  Unbounded below -- the -log(prior_var_inv)
        summand is negative whenever prior_std < 1 and can be driven
        arbitrarily negative by sharpening priors, independently of the fit.
    """

    def __init__(self) -> None:
        self.loss: npt.NDArray[np.floating] = np.array([])
        self.loss_main: npt.NDArray[np.floating] = np.array([])
        self.loss_prior: npt.NDArray[np.floating] = np.array([])
        self.eeg_train = None
        self.eeg_test = None
        self.states_train = None
        self.states_test = None
        self.weights: list = []


class BaseModel(torch.nn.Module):
    """Abstract base class for neural mass models.

    Subclasses must define:
        - model_name: str identifying the model (e.g. "JR", "WWD")
        - state_names: list of state variable names, one per population
        - output_name: str for the output signal type (e.g. "eeg", "bold")
        - pop_size: int number of neural populations per node/parcel
          (e.g. 3 for Jansen-Rit: pyramidal, excitatory, inhibitory)
        - state_size: int number of state variables per population. For
          second-order PSP kernels like Jansen-Rit this is 2 (the
          post-synaptic current and its time derivative); models with
          first-order dynamics use 1.
        - forward(): the forward simulation step
        - get_initial_state(): returns initial state tensor of shape
          (node_size, pop_size, state_size)
    """

    model_name: str
    state_names: list[str]
    output_name: str
    # Number of neural populations per node (override in subclass).
    pop_size: int = 3
    # Number of state variables per population (override in subclass).
    # 2 for second-order PSP kernels (current + derivative), 1 for
    # first-order dynamics.
    state_size: int = 2

    def __init__(
        self,
        sc: npt.NDArray[np.floating],
        leadfield: npt.NDArray[np.floating],
        dist: npt.NDArray[np.floating],
        step_size: float = 0.0001,
        tr: float = 0.001,
        batch_size: int = 50,
        fit_connectivity: bool = True,
        param: object | None = None,
        mask: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sc : ndarray (node_size, node_size)
            Structural connectivity matrix.
        leadfield : ndarray (output_size, node_size)
            Leadfield matrix mapping source activity to EEG channels.
        dist : ndarray (node_size, node_size)
            Distance matrix for computing conduction delays.
        step_size : float
            Integration step size in seconds. Default 0.0001 (0.1 ms).
        tr : float
            Repetition time / sampling period in seconds. Default 0.001 (1 ms).
        batch_size : int
            Number of **time points** processed in one forward call —
            i.e. the TBPTT (truncated backprop through time) window
            length, NOT the usual ML "number of independent samples per
            batch". The model is a stateful SDE: each forward advances
            `batch_size` time steps starting from the previous state,
            computes loss against the matching `batch_size`-long
            empirical slice, and backpropagates within that window.
            State is then detached and carried into the next window.
            Larger value -> longer gradient chain (more memory, risk of
            vanishing/exploding gradients); smaller value -> shorter
            gradient horizon (cheaper, may miss longer dependencies).
            Default 50 (= 50 ms at the default `tr = 0.001`).
        fit_connectivity : bool
            Whether to fit connection gains. Default True.
        param : object, optional
            Parameter specification object. If None, uses model defaults.
        mask : ndarray (node_size, node_size), optional
            Binary mask for connectivity. Default ones (all connections).
        """
        super().__init__()

        # Infer sizes from data
        node_size: int = sc.shape[0]
        output_size: int = leadfield.shape[0]

        # Validate inputs
        assert sc.shape == (node_size, node_size), \
            f"SC must be square, got {sc.shape}"
        assert leadfield.shape[1] == node_size, \
            f"Leadfield columns ({leadfield.shape[1]}) must match SC size ({node_size})"
        assert dist.shape == (node_size, node_size), \
            f"Distance matrix shape {dist.shape} must match SC shape {sc.shape}"

        self.node_size: int = node_size
        self.output_size: int = output_size
        # NOTE: `batch_size` here is the TBPTT window length (number of
        # time steps per forward call), not the conventional ML
        # "samples per batch". See __init__ docstring.
        self.batch_size: int = batch_size
        self.step_size: torch.Tensor = torch.tensor(step_size, dtype=torch.float32)
        self.hidden_size: int = int(tr / step_size)
        self.tr: float = tr
        self.sc: npt.NDArray[np.floating] = sc
        self.dist: torch.Tensor = torch.tensor(dist, dtype=torch.float32)
        self.fit_connectivity: bool = fit_connectivity
        self.mask: npt.NDArray[np.floating] = (
            mask if mask is not None else np.ones((node_size, node_size)))

        # Initialize param from subclass defaults if not provided
        if param is None:
            param = self._default_params()
        self.param = param

        # Register parameters from spec
        self._init_params_from_spec(param)

    @abstractmethod
    def _default_params(self) -> object:
        """Return default parameter object for this model."""
        raise NotImplementedError

    def _init_params_from_spec(self, param: object) -> None:
        """Register Par parameters on the module for gradient tracking.

        Handles both new Par objects and legacy [value, prior_std] lists.
        """
        self._par_names: list[str] = []
        for attr_name in sorted(dir(param)):
            if attr_name.startswith('_') or callable(getattr(param, attr_name)):
                continue
            attr = getattr(param, attr_name)
            if isinstance(attr, Par):
                self._par_names.append(attr_name)
                if attr.fit_hyper:
                    self.register_parameter(
                        f'_p_{attr_name}_val', attr.val)
                    self.register_parameter(
                        f'_p_{attr_name}_pm', attr.prior_mean)
                    self.register_parameter(
                        f'_p_{attr_name}_pvi', attr.prior_var_inv)
            elif isinstance(attr, list) and len(attr) >= 2:
                # Legacy [value, prior_std] format — convert to Par
                val, std = attr[0], attr[1]
                if np.any(np.array(std) > 0):
                    p = Par(val, val, std, fit_hyper=True)
                else:
                    p = Par(val)
                setattr(param, attr_name, p)
                self._par_names.append(attr_name)
                if p.fit_hyper:
                    self.register_parameter(
                        f'_p_{attr_name}_val', p.val)
                    self.register_parameter(
                        f'_p_{attr_name}_pm', p.prior_mean)
                    self.register_parameter(
                        f'_p_{attr_name}_pvi', p.prior_var_inv)

    def _compute_prior_loss(self) -> torch.Tensor:
        """Compute prior regularization loss across all fitted parameters."""
        loss_prior: list[torch.Tensor] = []
        lb: float = 0.001
        m: torch.nn.ReLU = torch.nn.ReLU()
        for name in self._par_names:
            p: Par = getattr(self.param, name)
            # std_in is fitted but deliberately unregularized, as in the
            # reference implementation: the noise level is identifiable
            # straight from the pre-stimulus baseline and needs no prior.
            if p.fit_hyper and name != 'std_in':
                loss_prior.append(
                    torch.sum(
                        (lb + m(p.prior_var_inv))
                        * (m(p.val) - m(p.prior_mean)) ** 2)
                    + torch.sum(-torch.log(lb + m(p.prior_var_inv))))
        if loss_prior:
            return sum(loss_prior)
        return torch.tensor(0.0)

    def lesion(
        self,
        nodes: list[int] | npt.NDArray[np.intp],
        remove_leadfield: bool = True,
    ) -> BaseModel:
        """Perform virtual dissection by lesioning (disconnecting) specified nodes.

        Returns a new model with all structural connectivity to/from the
        specified nodes set to zero. The original model is not modified.

        Parameters
        ----------
        nodes : list of int or ndarray
            Indices of nodes to lesion.
        remove_leadfield : bool
            If True, also zero out the leadfield for lesioned nodes,
            so they no longer contribute to simulated EEG. Default True.

        Returns
        -------
        lesioned_model : BaseModel
            A deep copy of this model with the specified nodes disconnected.
        """
        nodes = np.asarray(nodes, dtype=int)

        # Validate node indices
        if np.any(nodes < 0) or np.any(nodes >= self.node_size):
            raise ValueError(
                f"Node indices must be in [0, {self.node_size}), got {nodes}")

        lesioned = copy.deepcopy(self)

        # Zero out SC and mask for lesioned nodes
        lesioned.sc[nodes, :] = 0
        lesioned.sc[:, nodes] = 0
        lesioned.mask[nodes, :] = 0
        lesioned.mask[:, nodes] = 0

        # Zero out distance for lesioned nodes
        with torch.no_grad():
            lesioned.dist[nodes, :] = 0
            lesioned.dist[:, nodes] = 0

        # Zero out leadfield contribution
        if remove_leadfield:
            lm_par = getattr(lesioned.param, 'lm', None)
            if lm_par is not None and isinstance(lm_par, Par):
                with torch.no_grad():
                    if isinstance(lm_par.val, Parameter):
                        lm_par.val[:, nodes] = 0
                    else:
                        lm_par.val[:, nodes] = 0

        return lesioned

    @abstractmethod
    def forward(
        self,
        external: torch.Tensor,
        hx: torch.Tensor,
        hE: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Forward simulation step.

        Parameters
        ----------
        external : tensor (node_size, hidden_size, batch_size, pop_size)
            External stimulation input.
        hx : tensor
            Current model state.
        hE : tensor (node_size, delays_max)
            Activity history buffer for delayed coupling.

        Returns
        -------
        next_state : dict[str, torch.Tensor]
            Must contain 'current_state', output signal (e.g. 'eeg'),
            and optionally 'states'.
        hE : torch.Tensor
            Updated history buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_initial_state(self) -> torch.Tensor:
        raise NotImplementedError

    def get_initial_delay_state(self) -> torch.Tensor:
        """Return initial delay buffer."""
        delays_max: int = 500
        return torch.tensor(
            np.random.uniform(0, 0.1, (self.node_size, delays_max)),
            dtype=torch.float32)

    def fit(
        self,
        eeg: npt.NDArray[np.floating],
        epochs: int = 120,
        lr: float = 0.05,
        stimulus: npt.NDArray[np.floating] | None = None,
        loss: str = "rmse",
    ) -> SimulationResult:
        """Train the model by fitting to empirical EEG data.

        Parameters
        ----------
        eeg : ndarray (n_channels, n_timepoints) or (n_epochs, n_channels, n_timepoints)
            Empirical EEG data. If 2D, it is broadcast across epochs.
        epochs : int
            Number of training epochs. Default 120.
        lr : float
            Learning rate for Adam optimizer. Default 0.05.
        stimulus : ndarray or None
            External stimulation input. Shape
            (node_size, hidden_size, n_timepoints, pop_size) or legacy 3D
            (node_size, hidden_size, n_timepoints). None means no input.
        loss : str
            Loss function: "rmse", "beamform", or "fc". Default "rmse".

        Returns
        -------
        result : SimulationResult
            Training results including loss history and parameter evolution.
        """
        from ..optim.costs import Costs

        loss_methods: dict[str, int] = {"rmse": 0, "beamform": 1, "fc": 2}
        cost: Costs = Costs(loss_methods.get(loss, 0))

        # Handle 2D input
        if eeg.ndim == 2:
            eeg = np.array([eeg] * epochs)
        elif eeg.ndim == 3 and eeg.shape[0] < epochs:
            repeats: int = int(np.ceil(epochs / eeg.shape[0]))
            eeg = np.tile(eeg, (repeats, 1, 1))[:epochs]

        w_cost: int = 10
        result: SimulationResult = SimulationResult()
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=lr, eps=1e-7)

        hx: torch.Tensor = self.get_initial_state()
        hE: torch.Tensor = self.get_initial_delay_state()

        mask_e = np.tril_indices(self.output_size, -1)
        mask_sc = np.tril_indices(self.node_size, -1)

        loss_his: list = []
        main_his: list = []
        prior_his: list = []
        fit_sc: list = []

        # Number of TBPTT windows per epoch. Here `batch_size` is the
        # number of time steps processed in one forward call (a window
        # along the time axis), so `num_batches` counts how many such
        # windows fit into the epoch's time series. Any leftover time
        # points beyond a multiple of `batch_size` are dropped.
        num_batches: int = int(eeg.shape[2] / self.batch_size)

        for i_epoch in range(epochs):
            eeg_epoch: npt.NDArray = eeg[i_epoch % eeg.shape[0]]
            epoch_eeg: list = []
            epoch_states: list = []
            epoch_main: list = []
            epoch_prior: list = []

            for i_batch in range(num_batches):
                optimizer.zero_grad()

                # External input
                external: torch.Tensor
                if stimulus is not None:
                    start = i_batch * self.batch_size
                    end = (i_batch + 1) * self.batch_size
                    if stimulus.ndim == 3:
                        # Legacy 3D format: (node_size, hidden_size, n_tp)
                        ext_3d = stimulus[:, :, start:end]
                        external = torch.zeros(
                            (self.node_size, self.hidden_size,
                             self.batch_size, self.pop_size),
                            dtype=torch.float32)
                        external[:, :, :, 0] = torch.tensor(
                            ext_3d, dtype=torch.float32)
                    else:
                        external = torch.tensor(
                            stimulus[:, :, start:end, :],
                            dtype=torch.float32)
                else:
                    external = torch.zeros(
                        (self.node_size, self.hidden_size,
                         self.batch_size, self.pop_size),
                        dtype=torch.float32)

                # Forward
                next_state: dict[str, torch.Tensor]
                hE_new: torch.Tensor
                next_state, hE_new = self(external, hx, hE)

                # Empirical batch
                ts_batch: torch.Tensor = torch.tensor(
                    eeg_epoch[:, i_batch * self.batch_size:
                              (i_batch + 1) * self.batch_size],
                    dtype=torch.float32)

                # Loss
                loss_main: torch.Tensor = cost.cost_eff(
                    self, next_state[self.output_name], ts_batch)
                loss_prior: torch.Tensor = self._compute_prior_loss()
                loss_val: torch.Tensor = w_cost * loss_main + loss_prior

                loss_his.append(loss_val.detach().numpy())
                main_his.append(loss_main.detach().numpy())
                prior_his.append(loss_prior.detach().numpy())
                epoch_main.append(float(loss_main.detach()))
                epoch_prior.append(float(loss_prior.detach()))
                epoch_eeg.append(
                    next_state[self.output_name].detach().numpy())
                if 'states' in next_state:
                    epoch_states.append(
                        next_state['states'].detach().numpy())

                loss_val.backward(retain_graph=True)
                optimizer.step()

                hx = next_state['current_state'].detach()
                hE = hE_new.detach()

                if hasattr(self, 'sc_fitted'):
                    fit_sc.append(
                        self.sc_fitted.detach().numpy()[mask_sc].copy())

            # Epoch summary
            result.eeg_train = np.concatenate(epoch_eeg, axis=1)
            if epoch_states:
                result.states_train = np.concatenate(epoch_states, axis=-1)

            fc: npt.NDArray = np.corrcoef(eeg.mean(0))
            fc_sim: npt.NDArray = np.corrcoef(result.eeg_train[:, 10:])

            fc_corr: float = np.corrcoef(
                fc_sim[mask_e], fc[mask_e])[0, 1]
            cos_sim: float = np.diag(
                cosine_similarity(result.eeg_train, eeg.mean(0))).mean()
            # The data term and the prior term are reported separately: the
            # total mixes a bounded RMSE with a prior term that is unbounded
            # below, so a falling total is consistent with the priors
            # collapsing while the fit stands still.  Both are means over the
            # epoch's batches, not the last batch alone.
            mean_main: float = float(np.mean(epoch_main))
            mean_prior: float = float(np.mean(epoch_prior))
            print(f'epoch: {i_epoch}, loss: {mean_main * w_cost + mean_prior:.4f} '
                  f'(rmse {mean_main:.4f} x{w_cost} + prior {mean_prior:.4f}), '
                  f'FC corr: {fc_corr:.4f}, cos_sim: {cos_sim:.4f}')

        result.loss = np.array(loss_his)
        result.loss_main = np.array(main_his)
        result.loss_prior = np.array(prior_his)
        result.weights = np.array(fit_sc) if fit_sc else np.array([])
        self._last_result: SimulationResult = result
        return result

    def predict(
        self,
        eeg: npt.NDArray[np.floating] | None = None,
        stimulus: npt.NDArray[np.floating] | None = None,
        warmup_batches: int = 20,
        n_timepoints: int | None = None,
    ) -> SimulationResult:
        """Run inference with the trained model.

        Parameters
        ----------
        eeg : ndarray (n_channels, n_timepoints), optional
            Empirical EEG data (used for determining time length).
            If None, must provide n_timepoints.
        stimulus : ndarray or None
            External stimulation input. Shape
            (node_size, hidden_size, n_timepoints, pop_size) or legacy 3D.
        warmup_batches : int
            Number of warm-up batches before recording output. Default 20.
        n_timepoints : int, optional
            Number of output time points. Inferred from eeg if not given.

        Returns
        -------
        result : SimulationResult
            Result with test outputs (result.eeg_test, result.states_test).
        """
        result: SimulationResult
        if hasattr(self, '_last_result'):
            result = self._last_result
        else:
            result = SimulationResult()

        if eeg is not None:
            n_tp: int = eeg.shape[-1]
        elif n_timepoints is not None:
            n_tp = n_timepoints
        else:
            n_tp = warmup_batches * self.batch_size

        hx: torch.Tensor = self.get_initial_state()
        hE: torch.Tensor = self.get_initial_delay_state()

        total_tp: int = warmup_batches * self.batch_size + n_tp
        num_batches: int = total_tp // self.batch_size

        # Build full stimulus array
        u_hat: npt.NDArray = np.zeros(
            (self.node_size, self.hidden_size, total_tp, self.pop_size))
        if stimulus is not None:
            offset: int = warmup_batches * self.batch_size
            if stimulus.ndim == 3:
                u_hat[:, :, offset:offset + stimulus.shape[2], 0] = stimulus
            else:
                end = offset + stimulus.shape[2]
                u_hat[:, :, offset:end, :] = stimulus

        test_eeg: list = []
        test_states: list = []

        for i_batch in range(num_batches):
            start: int = i_batch * self.batch_size
            end_b: int = (i_batch + 1) * self.batch_size
            external: torch.Tensor = torch.tensor(
                u_hat[:, :, start:end_b, :], dtype=torch.float32)

            with torch.no_grad():
                next_state: dict[str, torch.Tensor]
                hE_new: torch.Tensor
                next_state, hE_new = self(external, hx, hE)

            if i_batch >= warmup_batches:
                test_eeg.append(
                    next_state[self.output_name].detach().numpy())
                if 'states' in next_state:
                    test_states.append(
                        next_state['states'].detach().numpy())

            hx = next_state['current_state'].detach()
            hE = hE_new.detach()

        if test_eeg:
            result.eeg_test = np.concatenate(test_eeg, axis=1)
        if test_states:
            result.states_test = np.concatenate(test_states, axis=-1)

        return result
