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

    Long-range coupling gain
        g : float, default 1000
            Global coupling gain.  Scales the single long-range pathway,
            which carries delayed E from every parcel into the excitatory
            population.

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
            # Long-range coupling gain (single pathway, delayed E -> E)
            "g": Par(1000),
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

    One long-range pathway: w_bb, a per-edge gain on the empirical SC
    carrying delayed E between parcels.  The external stimulus enters the
    same population, so it propagates through this pathway directly.

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

        # Source-to-sensor gain.  The reference implementation hard-codes 1e-4
        # for a leadfield whose rows have norm ~5e2; deriving it from the
        # actual row norms keeps the readout on the same scale for leadfields
        # built by other pipelines, instead of silently mis-scaling them.
        self.s2o_coef: float = float(
            1.0 / np.median(np.linalg.norm(np.asarray(leadfield), axis=1)))

    def _default_params(self) -> JRParams:
        """Return default JR parameters."""
        return JRParams()

    def _init_sc_parameters(self) -> None:
        """Initialize the long-range connectivity gains.

        One gain per edge, exponentiated so it stays positive:
        ``w = exp(w_bb) * SC``.  A per-node (node_size, 2) parameterization
        was tried and is cheaper, but it can only rescale rows and columns --
        it cannot reweight individual edges, which is what the reference
        model fits.
        """
        small_constant: float = 0.05
        if self.fit_connectivity:
            self.w_bb = Parameter(torch.tensor(
                np.zeros((self.node_size, self.node_size)) + small_constant,
                dtype=torch.float32))
        else:
            self.w_bb = torch.tensor(
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
        # Bounds and coefficients are those of the reference implementation
        # (PyTepFit RNNJANSEN.forward).  They are not cosmetic: conduct_lb
        # keeps the delays inside the 500-step buffer, noise_std_lb sets the
        # ongoing drive, and s2o_coef brings the raw leadfield into microvolts.
        u_2ndsys_ub: int = 500      # bound on the input to the 2nd-order system
        conduct_lb: float = 1.5     # lower bound on conduction velocity
        noise_std_lb: float = 150.  # lower bound on input-noise std
        lb: float = 0.01            # lower bound on local gains
        s2o_coef: float = 1e-4      # source-to-sensor coefficient
        k_lb: float = 0.5           # lower bound on the stimulation gain

        # Extract model parameters
        A = m(self.param.A.value())
        a = con_1 + m(self.param.a.value())
        B = m(self.param.B.value())
        b = con_1 + m(self.param.b.value())
        g = lb * con_1 + m(self.param.g.value())
        c1 = lb * con_1 + m(self.param.c1.value())
        c2 = lb * con_1 + m(self.param.c2.value())
        c3 = lb * con_1 + m(self.param.c3.value())
        c4 = lb * con_1 + m(self.param.c4.value())
        std_in = noise_std_lb * con_1 + m(self.param.std_in.value())
        vmax = self.param.vmax.value()
        v0 = self.param.v0.value()
        r = self.param.r.value()
        y0 = self.param.y0.value()
        mu = conduct_lb * con_1 + m(self.param.mu.value())
        k = k_lb * con_1 + m(self.param.k.value())
        cy0 = self.param.cy0.value()
        ki = self.param.ki.value()
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
            # One long-range pathway, as in the reference: delayed E from every
            # parcel, weighted by a per-edge gain on the empirical SC.  The
            # log1p is applied to the symmetrised product before normalising --
            # it compresses the heavy tail of the streamline counts so a few
            # dense edges cannot dominate the unit-norm budget.
            w = torch.exp(self.w_bb) * sc
            w_sym = 0.5 * (w + w.T)
            w_n = (torch.log1p(w_sym)
                   / torch.linalg.norm(torch.log1p(w_sym))) * mask_t
            self.sc_fitted = w_n
            dg = -torch.diag(torch.sum(w_n, dim=1))
        else:
            dg = 0
            w_n = 0

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

            LEd = torch.reshape(torch.sum(w_n * Ed, 1), (n_nodes, 1))

            for step_i in range(self.hidden_size):
                # External input (pyramidal population)
                u_stim = external[:, step_i:step_i + 1, i_window, 0]

                # Stochastic noise.  Only the excitatory population is
                # driven: the reference multiplies the P and I noise terms by
                # a literal 0, so injecting into all three would deliver
                # several times the intended ongoing activity.
                E_noise = std_in * torch.randn(n_nodes, 1)

                # Firing rates (sigmoid)
                P_sigm = vmax / (1 + torch.exp(r * (v0 - (E - I))))
                E_sigm = vmax / (1 + torch.exp(r * (v0 - (c1 * P))))
                I_sigm = vmax / (1 + torch.exp(r * (v0 - (c3 * P))))

                # Input to each population.  The long-range term is a
                # Laplacian on delayed E and reaches the excitatory population
                # only; P and I receive purely local input.
                rP = P_sigm
                rE = (E_noise
                      + g * (LEd + torch.matmul(dg, E))
                      + c2 * E_sigm)
                rI = c4 * I_sigm

                # Bound inputs for stability
                rP_bd = u_2ndsys_ub * torch.tanh(rP / u_2ndsys_ub)
                rE_bd = u_2ndsys_ub * torch.tanh(rE / u_2ndsys_ub)
                rI_bd = u_2ndsys_ub * torch.tanh(rI / u_2ndsys_ub)

                # External stimulation drives the excitatory interneurons --
                # the extrinsic input port of the canonical Jansen-Rit column
                # -- and is added AFTER the tanh, so a strong pulse is not
                # squashed by a bound meant for the recurrent terms.  Folding
                # it into rP before the tanh (as an earlier version did) caps
                # the delivered drive at u_2ndsys_ub=500 no matter the
                # amplitude, ~16x less than the reference delivers, and leaves
                # no visible early response.
                rE_bd = rE_bd + k * ki * u_stim

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

                # No state saturation: the reference leaves the states
                # unbounded (its 1000*tanh lines are commented out) and the
                # inputs are already bounded above by u_2ndsys_ub.
                P, E, I = P_tp1, E_tp1, I_tp1
                Pv, Ev, Iv = Pv_tp1, Ev_tp1, Iv_tp1

            # Update delay buffer with E -- the variable the stimulus enters,
            # so the drive propagates directly.  Buffering P instead routes it
            # through P_sigm = sigmoid(E - I), which is bounded by vmax and
            # squashes the evoked response before it can leave the parcel.
            hE = torch.cat([E, hE[:, :-1]], dim=1)

            # EEG output: average-reference the leadfield (do NOT row-normalize
            # it -- that gives every channel the same total sensitivity and
            # erases the amplitude topography) and scale by s2o_coef.
            lm_t_dm = (lm
                        - (1 / n_chans)
                        * torch.matmul(torch.ones((1, n_chans)), lm))
            temp = s2o_coef * cy0 * torch.matmul(lm_t_dm, E - I) - y0
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
