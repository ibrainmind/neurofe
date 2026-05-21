from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch.nn.parameter import Parameter

from .base import BaseModel
from .utils import Par


class JRParams:
    """Parameter specification for the Jansen-Rit neural mass model.

    Each parameter is a ``Par`` object. Use ``Par(value)`` for fixed
    parameters and ``Par(value, prior_mean, prior_std, True)`` for
    fitted parameters.

    Each cortical column has three populations — pyramidal (P), excitatory
    interneurons (E), and inhibitory interneurons (I) — coupled by a sigmoid
    firing-rate function and second-order alpha-type post-synaptic potential
    (PSP) kernels. Long-range inputs are routed through three pathways
    (lateral P→P, feedforward P→E, feedback P→I) with conduction delays.

    Parameters
    ----------
    Synaptic PSP kernel (second-order alpha function)
        A : float, default 3.25
            Peak amplitude of the excitatory PSP (mV). Sets the gain of
            P- and E-population synapses.
        a : float, default 100
            Inverse time constant of the excitatory PSP (1/s). Larger
            ``a`` means faster rise/decay of the EPSP.
        B : float, default 22
            Peak amplitude of the inhibitory PSP (mV). Sets the gain of
            I-population synapses.
        b : float, default 50
            Inverse time constant of the inhibitory PSP (1/s). Larger
            ``b`` means faster rise/decay of the IPSP.

    Long-range coupling gains (scale the network-level inputs)
        g : float, default 1000
            Global coupling gain for the lateral (P→P) pathway.
        g_f : float, default 40
            Coupling gain for the feedforward (P→E) pathway.
        g_b : float, default 40
            Coupling gain for the feedback (P→I) pathway.

    Intracolumnar connectivity constants (Jansen & Rit, 1995)
        c1 : float, default 135
            P → E presynaptic gain (number of synapses).
        c2 : float, default 108 (=135*0.8)
            P → E postsynaptic gain.
        c3 : float, default 33.75 (=135*0.25)
            P → I presynaptic gain.
        c4 : float, default 33.75 (=135*0.25)
            P → I postsynaptic gain.

    Sigmoid firing-rate function S(v) = vmax / (1 + exp(r*(v0 - v)))
        vmax : float, default 5
            Maximum firing rate of the population (1/s).
        v0 : float, default 6
            Half-activation potential — membrane potential at which the
            firing rate reaches ``vmax/2`` (mV).
        r : float, default 0.56
            Steepness (slope) of the sigmoid at ``v0`` (1/mV).

    Stochastic input
        std_in : float, default 100
            Standard deviation of the Gaussian noise injected into each
            population at every integration step.

    Conduction delays
        mu : float, default 0.5
            Mean conduction velocity used to convert the inter-node
            distance matrix into integer step delays
            (``delay = distance / mu``).

    External stimulation
        k : float, default 5
            Gain applied to the external stimulation input that drives
            the pyramidal population.
        ki : float, default 1
            Per-node stimulation indicator/scaling factor (e.g. binary
            mask of stimulated sites).

    EEG observation model
        y0 : float, default 2
            Constant offset subtracted from the simulated EEG.
        cy0 : float, default 5
            Amplitude scaling of the source-to-sensor projection.
        lm : ndarray or float, default 1
            Leadfield matrix (n_channels, n_nodes) mapping pyramidal
            source activity to scalp EEG channels. If left as the scalar
            default, :class:`JansenRit` will replace it with the
            leadfield passed to the constructor.

    Examples
    --------
    >>> params = JRParams(
    ...     A=Par(3.25, 3.25, 0.1, True),
    ...     g=Par(200, 200, 1, True),
    ... )
    """

    def __init__(self, **kwargs) -> None:
        defaults: dict[str, Par] = {
            # Excitatory PSP kernel: A = peak amplitude, a = inverse time constant
            "A": Par(3.25),
            "a": Par(100),
            # Inhibitory PSP kernel: B = peak amplitude, b = inverse time constant
            "B": Par(22),
            "b": Par(50),
            # Long-range coupling gains
            "g": Par(1000),    # lateral (P -> P)
            "g_f": Par(40),    # feedforward (P -> E)
            "g_b": Par(40),    # feedback (P -> I)
            # Intracolumnar connectivity constants
            "c1": Par(135),         # P -> E presynaptic
            "c2": Par(135 * 0.8),   # P -> E postsynaptic
            "c3": Par(135 * 0.25),  # P -> I presynaptic
            "c4": Par(135 * 0.25),  # P -> I postsynaptic
            # Stochastic input
            "std_in": Par(100),
            # Sigmoid firing-rate function
            "vmax": Par(5),    # maximum firing rate
            "v0": Par(6),      # half-activation potential
            "r": Par(0.56),    # sigmoid slope at v0
            # EEG observation model
            "y0": Par(2),      # EEG offset
            "cy0": Par(5),     # source-to-sensor amplitude scaling
            # Conduction delays
            "mu": Par(0.5),    # mean conduction velocity
            # External stimulation
            "k": Par(5),       # stimulation gain
            "ki": Par(1),      # per-node stimulation scaling
            # Leadfield (replaced by constructor when leadfield is provided)
            "lm": Par(1),
        }
        for var, val in defaults.items():
            setattr(self, var, val)
        for var, val in kwargs.items():
            setattr(self, var, val)


class JansenRit(BaseModel):
    """Jansen-Rit neural mass model for simulating EEG signals.

    Implements a network of coupled Jansen-Rit columns with three types
    of long-range connectivity (feedforward, feedback, lateral), conduction
    delays, and Bayesian-style parameter fitting via gradient optimization.

    Three populations per node:
        - P (pyramidal)
        - E (excitatory interneurons)
        - I (inhibitory interneurons)

    Three connectivity pathways:
        - w_ll: lateral (pyramidal-to-pyramidal), symmetrized
        - w_ff: feedforward (pyramidal-to-excitatory)
        - w_bb: feedback (pyramidal-to-inhibitory)

    Parameters
    ----------
    sc : ndarray (node_size, node_size)
        Structural connectivity matrix.
    leadfield : ndarray (output_size, node_size)
        Leadfield matrix mapping source activity to EEG channels.
    dist : ndarray (node_size, node_size)
        Distance matrix for computing conduction delays.
    step_size : float
        Integration step size in seconds. Default 0.0001.
    tr : float
        Sampling period in seconds. Default 0.001.
    batch_size : int
        TBPTT window length — number of **time steps** processed in one
        forward call, NOT the usual ML "samples per batch". The model
        is a stateful SDE: each forward advances `batch_size` time
        steps, loss is computed against the matching empirical slice,
        and gradients flow only within that window before state is
        detached and the next window starts. Default 50 time steps
        (= 50 ms at the default `tr = 0.001`).
    fit_connectivity : bool
        Whether to fit connection gains. Default True.
    param : JRParams, optional
        Parameter specification. If None, uses defaults.
    mask : ndarray (node_size, node_size), optional
        Binary mask for connectivity. Default ones (all connections).

    Examples
    --------
    >>> import numpy as np
    >>> from neurofe import JansenRit, Par, JRParams
    >>> sc = np.random.rand(20, 20) * 0.01
    >>> sc = 0.5 * (sc + sc.T)
    >>> lm = np.random.randn(10, 20) * 0.01
    >>> dist = np.random.rand(20, 20) * 10
    >>> dist = 0.5 * (dist + dist.T)
    >>> params = JRParams(
    ...     A=Par(3.25, 3.25, 0.1, True),
    ...     g=Par(200, 200, 1, True),
    ...     lm=Par(lm, lm, 0.01 * (np.abs(lm) + 0.01), True),
    ... )
    >>> model = JansenRit(sc=sc, leadfield=lm, dist=dist, param=params)
    """

    # Three neural populations per cortical column / parcel:
    #   P = pyramidal cells, E = excitatory interneurons,
    #   I = inhibitory interneurons.
    state_names: list[str] = ['P', 'E', 'I']
    model_name: str = "JR"
    output_name: str = "eeg"
    # One column has pop_size populations, each with state_size state
    # variables, giving a per-node state tensor of shape
    # (pop_size, state_size) = (3, 2).
    pop_size: int = 3
    # state_size = 2 because the Jansen-Rit alpha-function PSP is a
    # second-order ODE rewritten as two coupled first-order equations:
    #   index 0 -> post-synaptic current  (P, E, I)
    #   index 1 -> its time derivative    (Pv, Ev, Iv)
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
        param: JRParams | None = None,
        mask: npt.NDArray[np.floating] | None = None,
    ) -> None:
        # Ensure lm param is set from leadfield if not provided
        if param is None:
            param = JRParams(lm=Par(leadfield))
        elif (not hasattr(param, 'lm')
              or not isinstance(getattr(param, 'lm'), Par)
              or getattr(param, 'lm').val.dim() == 0):
            param.lm = Par(leadfield)

        super().__init__(
            sc=sc, leadfield=leadfield, dist=dist,
            step_size=step_size, tr=tr, batch_size=batch_size,
            fit_connectivity=fit_connectivity, param=param, mask=mask)

        self._init_sc_parameters()

    def _default_params(self) -> JRParams:
        """Return default JR parameters."""
        return JRParams()

    def _init_sc_parameters(self) -> None:
        """Initialize feedforward, feedback, and lateral connectivity gains.

        Each connectivity matrix uses a (node_size, 2) parameterization:
        w = diag(exp(w[:,0])) @ SC @ diag(exp(w[:,1]))
        """
        small_constant: float = 0.05
        zsmat: npt.NDArray = (
            np.zeros((self.node_size, 2)) + small_constant)

        if self.fit_connectivity:
            self.w_bb = Parameter(
                torch.tensor(zsmat.copy(), dtype=torch.float32))
            self.w_ff = Parameter(
                torch.tensor(zsmat.copy(), dtype=torch.float32))
            self.w_ll = Parameter(
                torch.tensor(zsmat.copy(), dtype=torch.float32))
        else:
            self.w_bb = torch.tensor(
                np.zeros((self.node_size, self.node_size)),
                dtype=torch.float32)
            self.w_ff = torch.tensor(
                np.zeros((self.node_size, self.node_size)),
                dtype=torch.float32)
            self.w_ll = torch.tensor(
                np.zeros((self.node_size, self.node_size)),
                dtype=torch.float32)

    def get_initial_state(self) -> torch.Tensor:
        """Return random initial state (node_size, pop_size, state_size)."""
        return torch.tensor(
            np.random.uniform(-0.01, 0.01,
                              (self.node_size, self.pop_size, self.state_size)),
            dtype=torch.float32)

    def forward(
        self,
        external: torch.Tensor,
        hx: torch.Tensor,
        hE: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for the Jansen-Rit model.

        Parameters
        ----------
        external : tensor (node_size, hidden_size, batch_size, pop_size)
            External stimulation input. The third axis is the time axis
            of length `batch_size` (the TBPTT window length — number of
            time steps this forward call advances, NOT a sample-batch
            dimension).
        hx : tensor (node_size, pop_size, state_size)
            Current state [P/E/I populations, current/voltage].
        hE : tensor (node_size, delays_max)
            Pyramidal activity history buffer for delayed coupling.

        Returns
        -------
        next_state : dict
            'current_state': tensor (node_size, pop_size, state_size)
            'eeg':    tensor (n_channels, batch_size)
                — last axis is time (one EEG sample per TR within the
                  window).
            'states': tensor (node_size, pop_size, state_size, batch_size)
                — last axis is time (full P/E/I trajectories within
                  the window).
        hE : tensor
            Updated history buffer.
        """
        m = torch.nn.ReLU()
        con_1 = torch.tensor(1.0, dtype=torch.float32)
        u_2ndsys_ub: int = 500
        lb: float = 0.0

        # Extract model parameters
        A = m(self.param.A.value())
        a = m(self.param.a.value())
        B = m(self.param.B.value())
        b = m(self.param.b.value())
        g = lb * con_1 + m(self.param.g.value())
        c1 = lb * con_1 + m(self.param.c1.value())
        c2 = lb * con_1 + m(self.param.c2.value())
        c3 = lb * con_1 + m(self.param.c3.value())
        c4 = lb * con_1 + m(self.param.c4.value())
        std_in = m(self.param.std_in.value())
        vmax = self.param.vmax.value()
        v0 = self.param.v0.value()
        r = self.param.r.value()
        y0 = self.param.y0.value()
        mu = 0.1 * con_1 + m(self.param.mu.value())
        k = 5.0 * con_1 + m(self.param.k.value())
        cy0 = self.param.cy0.value()
        ki = self.param.ki.value()
        g_f = lb * con_1 + m(self.param.g_f.value())
        g_b = lb * con_1 + m(self.param.g_b.value())
        lm = self.param.lm.value()

        dt = self.step_size
        n_nodes = self.node_size
        n_chans = self.output_size

        # Unpack states: (node_size, pop_size, state_size)
        P = hx[:, 0:1, 0]   # pyramidal current
        E = hx[:, 1:2, 0]   # excitatory current
        I = hx[:, 2:3, 0]   # inhibitory current
        Pv = hx[:, 0:1, 1]  # pyramidal voltage
        Ev = hx[:, 1:2, 1]  # excitatory voltage
        Iv = hx[:, 2:3, 1]  # inhibitory voltage

        sc = torch.tensor(self.sc, dtype=torch.float32)
        mask_t = torch.tensor(self.mask, dtype=torch.float32)

        # Compute three connectivity matrices
        if self.sc.shape[0] > 1:
            # Feedback: pyramidal to inhibitory
            w_b = (torch.diag(torch.exp(self.w_bb[:, 0]))
                   @ sc
                   @ torch.diag(torch.exp(self.w_bb[:, 1])))
            w_n_b = w_b / torch.linalg.norm(w_b) * mask_t
            self.sc_m_b = w_n_b
            dg_b = -torch.diag(torch.sum(w_n_b, dim=1))

            # Feedforward: pyramidal to excitatory
            w_f = (torch.diag(torch.exp(self.w_ff[:, 0]))
                   @ sc
                   @ torch.diag(torch.exp(self.w_ff[:, 1])))
            w_n_f = w_f / torch.linalg.norm(w_f) * mask_t
            self.sc_m_f = w_n_f
            dg_f = -torch.diag(torch.sum(w_n_f, dim=1))

            # Lateral: pyramidal to pyramidal (symmetrized)
            w_l = (torch.diag(torch.exp(self.w_ll[:, 0]))
                   @ sc
                   @ torch.diag(torch.exp(self.w_ll[:, 1])))
            w_sym = 0.5 * (w_l + w_l.T)
            w_n_l = w_sym / torch.linalg.norm(w_sym) * mask_t
            self.sc_fitted = w_n_l
            dg_l = -torch.diag(torch.sum(w_n_l, dim=1))
        else:
            dg_l = dg_b = dg_f = 0
            w_n_l = w_n_b = w_n_f = 0

        # Compute conduction delays
        self.delays = (self.dist / mu).type(torch.int64)

        eeg_window: list[torch.Tensor] = []
        states_window: list[torch.Tensor] = []

        # Iterate over time within the TBPTT window. `batch_size` here
        # is the number of TR-sized time points produced per forward
        # call (window length), not a sample batch.
        for i_window in range(self.batch_size):
            # Delayed coupling (computed once per TR)
            Ed = torch.transpose(
                hE.clone().gather(1, self.delays), 0, 1)

            LEd_p2i = (torch.reshape(torch.sum(w_n_b * Ed, 1),
                                     (n_nodes, 1))
                       + torch.matmul(dg_b, I - E))
            LEd_p2e = (torch.reshape(torch.sum(w_n_f * Ed, 1),
                                     (n_nodes, 1))
                       + torch.matmul(dg_f, E - I))
            LEd_p2p = (torch.reshape(torch.sum(w_n_l * Ed, 1),
                                     (n_nodes, 1))
                       + torch.matmul(dg_l, P))

            for step_i in range(self.hidden_size):
                # External input (pyramidal population)
                u_stim = external[:, step_i:step_i + 1, i_window, 0]

                # Stochastic noise
                P_noise = std_in * torch.randn(n_nodes, 1)
                E_noise = std_in * torch.randn(n_nodes, 1)
                I_noise = std_in * torch.randn(n_nodes, 1)

                # Firing rates (sigmoid)
                P_sigm = vmax / (1 + torch.exp(r * (v0 - (E - I))))
                E_sigm = vmax / (1 + torch.exp(r * (v0 - (c1 * P))))
                I_sigm = vmax / (1 + torch.exp(r * (v0 - (c3 * P))))

                # Input to each population:
                #   Local + Long-range + Noise + External
                rP = P_sigm + g * LEd_p2p + P_noise + k * ki * u_stim
                rE = c2 * E_sigm + g_f * LEd_p2e + E_noise
                rI = c4 * I_sigm + g_b * LEd_p2i + I_noise

                # Bound inputs for stability
                rP_bd = u_2ndsys_ub * torch.tanh(rP / u_2ndsys_ub)
                rE_bd = u_2ndsys_ub * torch.tanh(rE / u_2ndsys_ub)
                rI_bd = u_2ndsys_ub * torch.tanh(rI / u_2ndsys_ub)

                # Euler integration
                P_tp1 = P + dt * Pv
                E_tp1 = E + dt * Ev
                I_tp1 = I + dt * Iv
                Pv_tp1 = Pv + dt * (A * a * rP_bd
                                     - 2 * a * Pv - a ** 2 * P)
                Ev_tp1 = Ev + dt * (A * a * rE_bd
                                     - 2 * a * Ev - a ** 2 * E)
                Iv_tp1 = Iv + dt * (B * b * rI_bd
                                     - 2 * b * Iv - b ** 2 * I)

                # Saturation for stability
                P = 1000 * torch.tanh(P_tp1 / 1000)
                E = 1000 * torch.tanh(E_tp1 / 1000)
                I = 1000 * torch.tanh(I_tp1 / 1000)
                Pv = 1000 * torch.tanh(Pv_tp1 / 1000)
                Ev = 1000 * torch.tanh(Ev_tp1 / 1000)
                Iv = 1000 * torch.tanh(Iv_tp1 / 1000)

            # Update delay buffer
            hE = torch.cat([P, hE[:, :-1]], dim=1)

            # EEG output: normalize leadfield and project source to channels
            lm_t = (lm.T / torch.sqrt((lm ** 2).sum(1))).T
            lm_t_dm = (lm_t
                        - (1 / n_chans)
                        * torch.matmul(torch.ones((1, n_chans)), lm_t))
            temp = cy0 * torch.matmul(lm_t_dm, E - I) - y0
            eeg_window.append(temp)

            states_window.append(
                torch.cat([
                    torch.cat([P, E, I], dim=1)[:, :, None],
                    torch.cat([Pv, Ev, Iv], dim=1)[:, :, None],
                ], dim=2)[:, :, :, None])

        self.lm_t = lm_t_dm

        current_state = torch.cat([
            torch.cat([P, E, I], dim=1)[:, :, None],
            torch.cat([Pv, Ev, Iv], dim=1)[:, :, None],
        ], dim=2)

        next_state: dict[str, torch.Tensor] = {
            'current_state': current_state,
            'eeg': torch.cat(eeg_window, dim=1),
            'states': torch.cat(states_window, dim=3),
        }

        return next_state, hE
