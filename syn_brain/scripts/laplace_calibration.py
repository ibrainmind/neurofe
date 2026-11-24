"""Laplace-approximation calibration check for c1*c2 / c3*c4 recovery at the firing state.

For each of a batch of independently-drawn random teachers (not the clean E/I-balance axis --
full independent +-30% perturbation of all 7 fitted parameters, screened for a stable,
non-oscillating active state at p=150):

  1. Fit an ensemble of N_STARTS restarts (RICH protocol: SP1000+PP100+TR10+EI1000, nominal-
     centered starts -- NOT centered on each teacher's own truth, since real data never gives
     you that), take the lowest-training-loss restart as the point (MAP) estimate.
  2. Compute the Fisher information / Laplace covariance at that MAP point (central differences
     in log-parameter space), giving a Gaussian approximate posterior for log(c1*c2), log(c3*c4).
  3. Check calibration: does the MAP's 90% credible interval actually contain the true value?

Run as:
    PYTHONPATH=. .venv/bin/python -m syn_brain.scripts.laplace_calibration --seed-start 100 --n-teachers 6 --tag partA

Writes syn_brain/results/bifurcation_followup/laplace_calib_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from syn_brain.core import node
from syn_brain.core.ladder import REAL_BASELINE_STD_UV
from syn_brain.core.paths import SYN_ROOT
from syn_brain.scripts.single_node_tms import ALL, FIT, NOM_ALL, NOM_FIT, readout

FIT_IDX = {n: i for i, n in enumerate(FIT)}
SIGMA = REAL_BASELINE_STD_UV
OUT = SYN_ROOT / "results" / "bifurcation_followup"

N_STARTS = 15
ITERS = 300
P_POINT = 150.0
PROTO_NAMES = ("SP1000", "PP100", "TR10", "EI1000")
START_SPREAD = 0.5   # log-uniform +-50% around NOMINAL (not truth)
DRAW_SPREAD = 0.3     # teacher draw spread (matches draw_teachers default elsewhere)


def draw_candidate_teachers(n, seed):
    rng = np.random.default_rng(seed)
    fac = np.exp(rng.uniform(np.log(1 - DRAW_SPREAD), np.log(1 + DRAW_SPREAD), (n, len(FIT))))
    fit_vals = NOM_FIT * fac
    full = np.tile(NOM_ALL, (n, 1))
    full[:, :len(FIT)] = fit_vals
    return full


def screen_stability(teachers, p=P_POINT, settle_ms=8000, window_ms=3000):
    """Vectorized: returns boolean mask of teachers that are active (sigmoid slope>0.3) and
    non-oscillating (two-window plateau check, as in the phase-diagram script)."""
    G = teachers.shape[0]
    th = {n: torch.tensor(teachers[:, i]) for i, n in enumerate(ALL)}
    a, b, c1, c2, c3, c4, k = (th[n] for n in FIT)
    A, Bg, v0, r, vmax = (th[n] for n in node.EXTRA_NAMES)
    Aa, Bb, a2, b2, twoa, twob = A * a, Bg * b, a * a, b * b, 2 * a, 2 * b
    P = torch.zeros(G, dtype=torch.float64); E = P.clone(); I = P.clone()
    Pv = P.clone(); Ev = P.clone(); Iv = P.clone()
    uE = torch.full((G,), float(p), dtype=torch.float64); uI = torch.zeros(G, dtype=torch.float64)

    def sub(P, E, I, Pv, Ev, Iv):
        P_s = vmax / (1 + torch.exp(r * (v0 - (E - I))))
        E_s = vmax / (1 + torch.exp(r * (v0 - c1 * P)))
        I_s = vmax / (1 + torch.exp(r * (v0 - c3 * P)))
        rP = node.U_BOUND * torch.tanh(P_s / node.U_BOUND)
        rE = node.U_BOUND * torch.tanh(c2 * E_s / node.U_BOUND) + uE
        rI = node.U_BOUND * torch.tanh(c4 * I_s / node.U_BOUND) + uI
        return (P + node.DT * Pv, E + node.DT * Ev, I + node.DT * Iv,
                Pv + node.DT * (Aa * rP - twoa * Pv - a2 * P),
                Ev + node.DT * (Aa * rE - twoa * Ev - a2 * E),
                Iv + node.DT * (Bb * rI - twob * Iv - b2 * I))

    def window(n_ms):
        nonlocal P, E, I, Pv, Ev, Iv
        traj = []
        for t in range(n_ms * node.HIDDEN):
            P, E, I, Pv, Ev, Iv = sub(P, E, I, Pv, Ev, Iv)
            if t % (20 * node.HIDDEN) == 0:
                traj.append((E - I).clone())
        return torch.stack(traj)

    with torch.no_grad():
        for _ in range(settle_ms * node.HIDDEN):
            P, E, I, Pv, Ev, Iv = sub(P, E, I, Pv, Ev, Iv)
        w1 = window(window_ms)
        w2 = window(window_ms)
    amp1 = (w1.max(0).values - w1.min(0).values).numpy()
    amp2 = (w2.max(0).values - w2.min(0).values).numpy()
    ei = w2.mean(0).numpy()
    v0v, rv, vmaxv = teachers[:, 9], teachers[:, 10], teachers[:, 11]
    slope = rv * vmaxv * np.exp(rv * (v0v - ei)) / (1 + np.exp(rv * (v0v - ei)))**2
    stable = amp2 < 0.05
    active = slope > 0.3
    return stable & active


def combo_logerr(theta, truth, a, b):
    return np.log((theta[:, FIT_IDX[a]] * theta[:, FIT_IDX[b]]) / (truth[:, FIT_IDX[a]] * truth[:, FIT_IDX[b]]))


def fit_ensemble(teachers, gain, rng):
    """teachers: (T,12). Returns MAP theta (T,12) -- best of N_STARTS restarts per teacher,
    nominal-centered starts, RICH protocol at P_POINT."""
    T = teachers.shape[0]
    base = {"SP1000": node.STIMS["SP1000"], "PP100": node.STIMS["PP100"],
           "TR10": node.STIMS["TR10"], "EI1000": node.STIMS["EI1000"]}
    import dataclasses
    stims = [dataclasses.replace(base[n], tonic=P_POINT) for n in PROTO_NAMES]
    M = T * N_STARTS
    truth_rep = np.repeat(teachers, N_STARTS, axis=0)   # (M,12)
    truth_t = torch.tensor(truth_rep)
    rows_job = np.repeat(np.arange(M), len(stims))
    rows_stim = stims * M
    with torch.no_grad():
        target = node.simulate_rows({n: truth_t[:, i] for i, n in enumerate(ALL)}, rows_job, rows_stim, gain=gain)
    target = target - target[:, :node.PRE].mean(-1, keepdim=True)

    # z parameterized relative to NOMINAL (population reference), not each teacher's own truth
    nom_fit_t = torch.tensor(NOM_FIT)
    z0 = rng.uniform(np.log(1 - START_SPREAD), np.log(1 + START_SPREAD), (M, len(FIT)))
    z = torch.tensor(z0).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS, eta_min=0.03 * 0.04)
    rj = torch.tensor(rows_job)
    n_rows = torch.zeros(M, dtype=torch.float64).index_add_(0, rj, torch.ones(len(rows_job), dtype=torch.float64))
    power = torch.zeros(M, dtype=torch.float64).index_add_(0, rj, target.pow(2).mean(-1)) / n_rows

    t0 = time.time()
    for it in range(ITERS):
        theta_fit = nom_fit_t.unsqueeze(0) * torch.exp(z)   # NOMINAL-centered, not truth-centered
        theta_full = torch.cat([theta_fit, truth_t[:, len(FIT):]], 1)
        y = node.simulate_rows({n: theta_full[:, i] for i, n in enumerate(ALL)}, rows_job, rows_stim, gain=gain)
        sse = (y - target).pow(2).mean(-1)
        loss = torch.zeros(M, dtype=torch.float64).index_add(0, rj, sse) / n_rows / power
        opt.zero_grad(); loss.sum().backward(); opt.step(); sched.step()
        with torch.no_grad():
            z.clamp_(-3, 3)
        if it % 50 == 0 or it == ITERS - 1:
            print(f"    it={it:4d} median_loss={loss.detach().median():.3e} {time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        theta_fit = (nom_fit_t.unsqueeze(0) * torch.exp(z)).numpy()
        final_loss = loss.detach().numpy()
    theta_fit = theta_fit.reshape(T, N_STARTS, len(FIT))
    final_loss = final_loss.reshape(T, N_STARTS)
    best = final_loss.argmin(axis=1)
    map_theta = theta_fit[np.arange(T), best]   # (T,7)
    return map_theta


def laplace_se(map_fit, teachers_extra, gain, h=1e-3):
    """Central-difference Fisher at the MAP point (in log-param space), RICH protocol.
    Returns SE for log(c1*c2), log(c3*c4) via full covariance (not independence approx)."""
    T = map_fit.shape[0]
    import dataclasses
    base = {"SP1000": node.STIMS["SP1000"], "PP100": node.STIMS["PP100"],
           "TR10": node.STIMS["TR10"], "EI1000": node.STIMS["EI1000"]}
    stims = [dataclasses.replace(base[n], tonic=P_POINT) for n in PROTO_NAMES]
    P = len(FIT)
    full = np.concatenate([map_fit, teachers_extra], axis=1)   # (T, 12)
    thetas = [full]
    for i in range(P):
        for s in (+1, -1):
            t = full.copy(); t[:, i] *= np.exp(s * h)
            thetas.append(t)
    th_stack = np.concatenate(thetas)   # (1+2P)*T, 12
    rows_th_idx = np.repeat(np.arange(th_stack.shape[0]), len(stims))
    rows_stim = stims * th_stack.shape[0]
    with torch.no_grad():
        y = node.simulate_rows({n: torch.tensor(th_stack[:, i]) for i, n in enumerate(ALL)},
                               rows_th_idx, rows_stim, gain=gain).numpy()
    Tn = y.shape[-1]
    y = y.reshape(1 + 2 * P, T, len(stims), Tn)
    J = np.stack([(y[1 + 2 * i] - y[2 + 2 * i]) / (2 * h) for i in range(P)], -1)  # (T,ncond,Tn,7)
    Jr = J.reshape(T, -1, P)
    se_c1c2 = np.zeros(T); se_c3c4 = np.zeros(T)
    i1, i2, i3, i4 = FIT_IDX["c1"], FIT_IDX["c2"], FIT_IDX["c3"], FIT_IDX["c4"]
    for t in range(T):
        Jt = Jr[t]
        F = Jt.T @ Jt / SIGMA**2
        cov = np.linalg.pinv(F, rcond=1e-15)
        se_c1c2[t] = np.sqrt(max(cov[i1, i1] + cov[i2, i2] + 2 * cov[i1, i2], 0))
        se_c3c4[t] = np.sqrt(max(cov[i3, i3] + cov[i4, i4] + 2 * cov[i3, i4], 0))
    return se_c1c2, se_c3c4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--n-teachers", type=int, default=6)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    gain, _ = readout()
    # draw more candidates than needed, screen for stability, keep the first n_teachers survivors
    candidates = draw_candidate_teachers(args.n_teachers * 6, seed=args.seed_start)
    ok = screen_stability(candidates)
    print(f"stability screen: {ok.sum()}/{len(ok)} candidates survive", flush=True)
    teachers = candidates[ok][:args.n_teachers]
    if len(teachers) < args.n_teachers:
        print(f"WARNING: only {len(teachers)} stable teachers found, wanted {args.n_teachers}")

    rng = np.random.default_rng(args.seed_start + 1)
    map_fit = fit_ensemble(teachers, gain, rng)
    se_c1c2, se_c3c4 = laplace_se(map_fit, teachers[:, len(FIT):], gain)

    truth_logc1c2 = np.log(teachers[:, FIT_IDX["c1"]] * teachers[:, FIT_IDX["c2"]])
    truth_logc3c4 = np.log(teachers[:, FIT_IDX["c3"]] * teachers[:, FIT_IDX["c4"]])
    map_logc1c2 = np.log(map_fit[:, FIT_IDX["c1"]] * map_fit[:, FIT_IDX["c2"]])
    map_logc3c4 = np.log(map_fit[:, FIT_IDX["c3"]] * map_fit[:, FIT_IDX["c4"]])

    z90 = 1.645
    cov_c1c2 = (np.abs(map_logc1c2 - truth_logc1c2) < z90 * se_c1c2).tolist()
    cov_c3c4 = (np.abs(map_logc3c4 - truth_logc3c4) < z90 * se_c3c4).tolist()

    results = dict(
        teachers=teachers.tolist(), map_fit=map_fit.tolist(),
        se_c1c2=se_c1c2.tolist(), se_c3c4=se_c3c4.tolist(),
        map_logc1c2=map_logc1c2.tolist(), truth_logc1c2=truth_logc1c2.tolist(),
        map_logc3c4=map_logc3c4.tolist(), truth_logc3c4=truth_logc3c4.tolist(),
        covered_c1c2=cov_c1c2, covered_c3c4=cov_c3c4,
    )
    out_path = OUT / f"laplace_calib_{args.tag}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"c1c2 90% coverage: {np.mean(cov_c1c2)*100:.0f}%  ({sum(cov_c1c2)}/{len(cov_c1c2)})")
    print(f"c3c4 90% coverage: {np.mean(cov_c3c4)*100:.0f}%  ({sum(cov_c3c4)}/{len(cov_c3c4)})")
    print(f"wrote {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
