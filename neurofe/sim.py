"""
Standalone NumPy-based Jansen-Rit simulation for parameter exploration.
Not used in the main fitting pipeline.
"""

from numpy import exp, sin, floor
import numpy as np
import pandas as pd


def run_net_sim(T=40000, K=1, Dt=0.0001,
                dt=0.1, Pi=3.14159, g=0.9, noise_std=0.005,
                T_transient=10000, stim_freq=35., stim_amp=0.5,
                weights=None, delays=None,
                stim_nodes=None, stim_pops=['pyc_v'],
                stim_type='sinewave', stim_on=None, stim_off=None,
                nu_max=2.5,
                v0=6.,
                r=0.56,
                a_1=1,
                a_2=0.8,
                a_3=0.25,
                a_4=0.25,
                a=100,
                A=3.25,
                b=50,
                B=22.0,
                J=135,
                lesion_nodes=None):
    """Run a network simulation of the Jansen-Rit model using NumPy.

    Parameters
    ----------
    K : int
        Number of nodes.
    T : int
        Simulation length in time steps.
    weights : ndarray (K, K)
        Connectivity weight matrix.
    delays : ndarray (K, K)
        Conduction delay matrix.
    stim_nodes : list of int, optional
        Indices of nodes receiving stimulation.
    stim_type : str
        Type of stimulation ('sinewave', 'spiketrain', 'square', 'rand').
    lesion_nodes : list of int, optional
        Indices of nodes to lesion (virtual dissection). All connectivity
        to/from these nodes is set to zero.

    Returns
    -------
    df_all : DataFrame
        Concatenated time series for all populations and stimulation.
    """

    n_nodes = K

    # Apply virtual dissection: disconnect lesioned nodes
    if lesion_nodes is not None:
        lesion_nodes = np.asarray(lesion_nodes, dtype=int)
        weights = weights.copy()
        delays = delays.copy()
        weights[lesion_nodes, :] = 0
        weights[:, lesion_nodes] = 0
        delays[lesion_nodes, :] = 0
        delays[:, lesion_nodes] = 0
    # Initialize variables
    Xi_pyc_c = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])
    Xi_pyc_v = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])
    Xi_ein_c = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])
    Xi_ein_v = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])
    Xi_iin_c = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])
    Xi_iin_v = np.random.uniform(low=-1., high=1., size=[K, T + T_transient])

    pyc_c_all = np.zeros_like(Xi_pyc_c)
    pyc_v_all = np.zeros_like(Xi_pyc_v)
    ein_c_all = np.zeros_like(Xi_ein_c)
    ein_v_all = np.zeros_like(Xi_ein_v)
    iin_c_all = np.zeros_like(Xi_iin_c)
    iin_v_all = np.zeros_like(Xi_iin_v)

    Ts = np.arange(0, T + T_transient)

    if stim_type == 'sinewave':
        state_input = 1.
        stim = state_input * stim_amp * sin(2 * Pi * stim_freq * Ts * Dt)
    elif stim_type == 'spiketrain':
        stim_period = 1. / (stim_freq * Dt)
        spiketrain = np.zeros_like(Ts)
        t_cnt = 0
        t = 0
        while t < Ts[-4]:
            t += 1
            if t >= t_cnt:
                spiketrain[t] = stim_amp
                spiketrain[t + 1] = stim_amp
                spiketrain[t + 2] = 0
                spiketrain[t + 3] = 0
                t_cnt = t_cnt + stim_period
        stim = spiketrain
    elif stim_type == 'square':
        stim = np.zeros_like(Ts)
        stim[(Ts >= stim_on) & (Ts <= stim_off)] = stim_amp
    elif stim_type == 'rand':
        stim = np.random.randn(Ts.shape[0]) * stim_amp

    stim_temp = np.zeros((n_nodes, stim.shape[0]))

    if stim_nodes is None:
        stim = stim_temp
    else:
        stim_temp[stim_nodes] = stim
        stim = stim_temp
        stim_nodes = np.array(stim_nodes)
        nostim_nodes = np.array([k for k in range(n_nodes) if k not in stim_nodes])
        if nostim_nodes.shape[0] != 0:
            stim[nostim_nodes, :] = 0.

    stim_pyc_c = stim_pyc_v = stim_ein_c = stim_ein_v = stim_iin_c = stim_iin_v = np.zeros_like(stim)
    if 'pyc_v' in stim_pops: stim_pyc_v = stim
    if 'pyc_c' in stim_pops: stim_pyc_c = stim
    if 'ein_c' in stim_pops: stim_ein_c = stim
    if 'ein_v' in stim_pops: stim_ein_v = stim
    if 'iin_c' in stim_pops: stim_iin_c = stim
    if 'iin_v' in stim_pops: stim_iin_v = stim

    phi_n_scaling = (0.1 * 325 * (0.32 - 0.12) * 0.5) ** 2 / 2.

    pyc_c = np.zeros(n_nodes)
    pyc_v = np.zeros(n_nodes)
    ein_c = np.zeros(n_nodes)
    ein_v = np.zeros(n_nodes) + phi_n_scaling * 0.00000
    iin_c = np.zeros(n_nodes)
    iin_v = np.zeros(n_nodes)

    fs = (1000. * Dt) / dt

    delays_lookup = floor(delays).astype(int)
    maxdelay = int(np.max(delays_lookup)) + 1

    for t in Ts[maxdelay:-1]:
        noise_e = np.random.rand(n_nodes)
        noise_i = np.random.rand(n_nodes)
        noise_p = np.random.rand(n_nodes)

        diff = ein_c_all[:, :t]
        Ed = diff[:, ::-1][np.arange(n_nodes), delays.T].T
        L = weights - 1 * np.diag(weights.sum(1))
        ccen = np.sum(L * Ed.T, axis=1)

        gccen = g * (ccen)

        sigm_ein_iin = 2.0 * nu_max / (1. + exp(r * (v0 - (ein_c - iin_c))))
        sigm_pyc_1 = 2.0 * nu_max / (1. + exp(r * (v0 - (a_1 * J * pyc_c))))
        sigm_pyc_3 = 2.0 * nu_max / (1. + exp(r * (v0 - (a_3 * J * pyc_c))))

        pyc_u = (sigm_ein_iin)
        ein_u = a_2 * J * sigm_pyc_1 + gccen + stim_ein_c[:, t] + noise_std * noise_e
        iin_u = a_4 * J * sigm_pyc_3

        pyc_in = A * a * (pyc_u) - (2.0 * a * pyc_v) - ((a ** 2) * pyc_c)
        ein_in = A * a * (ein_u) - (2.0 * a * ein_v) - ((a ** 2) * ein_c)
        iin_in = B * b * (iin_u) - (2.0 * b * iin_v) - ((b ** 2) * iin_c)

        pyc_c = pyc_c + dt * (pyc_v) + 0 * np.sqrt(dt) * noise_std * noise_p
        ein_c = ein_c + dt * (ein_v) + 0 * np.sqrt(dt) * noise_std * noise_e
        iin_c = iin_c + dt * (iin_v) + 0 * np.sqrt(dt) * noise_std * noise_i

        pyc_v = pyc_v + dt * (pyc_in)
        ein_v = ein_v + dt * (ein_in)
        iin_v = iin_v + dt * (iin_in)

        pyc_v_all[:, t + 1] = pyc_v
        pyc_c_all[:, t + 1] = pyc_c
        ein_v_all[:, t + 1] = ein_v
        ein_c_all[:, t + 1] = ein_c
        iin_v_all[:, t + 1] = iin_v
        iin_c_all[:, t + 1] = iin_c

    df_all = pd.concat({
        'stim': pd.DataFrame(stim.T[T_transient:]),
        'pyc_c': pd.DataFrame(pyc_c_all.T[T_transient:]),
        'pyc_v': pd.DataFrame(pyc_v_all.T[T_transient:]),
        'ein_c': pd.DataFrame(ein_c_all.T[T_transient:]),
        'ein_v': pd.DataFrame(ein_v_all.T[T_transient:]),
        'iin_c': pd.DataFrame(iin_c_all.T[T_transient:]),
        'iin_v': pd.DataFrame(iin_v_all.T[T_transient:])
    }, axis=1)
    df_all.index = (Ts[T_transient:] - T_transient) * (Dt * 10000.)

    return df_all
