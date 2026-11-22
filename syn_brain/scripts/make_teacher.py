"""Generate synthetic subjects (teachers): ground truth + trial-averaged EEG for several sites.

Each teacher is saved to data/teachers/<rung>_n<N>_t<seed>/ as
    truth.json   ground-truth parameters (raw and effective), sites, settings
    eeg.npz      mean (S, C, T): trial average per site, with the rung's sensor noise
                 clean (S, C, T): noise-free teacher run per site
                 half1, half2 (S, C, T): split-half averages, for the noise ceiling
                 ki (S, N, 1): stimulation masks; w_bb (N, N)

Usage (from the repo root):
    .venv/bin/python -m syn_brain.scripts.make_teacher --seeds 0 1 2 3 --rung R0 --sites 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from syn_brain.core.paths import DATA_DIR, get_logger, save_json
from syn_brain.core.anatomy import load_anatomy, pick_sites, pulse_stimulus, site_weights
from syn_brain.core.ladder import add_sensor_noise, get_rung
from syn_brain.core.params import draw_teacher
from syn_brain.core.simulator import SimConfig, evoked


def make_one(seed: int, rung_name: str, n_nodes, n_sites: int, n_trials: int,
             spread: float, w_bb_sigma: float | None, log) -> None:
    rung = get_rung(rung_name)
    anat = load_anatomy(n_nodes)
    cfg = SimConfig()
    stim = pulse_stimulus(anat.n_nodes, cfg.hidden_size)
    sigma = rung.w_bb_sigma if w_bb_sigma is None else w_bb_sigma
    tp = draw_teacher(anat.n_nodes, seed=seed, spread=spread, w_bb_sigma=sigma)
    sites = pick_sites(anat, n_sites, seed=1000 + seed)
    rng = np.random.default_rng(2000 + seed)

    means, cleans, h1s, h2s, kis = [], [], [], [], []
    for s in sites:
        t0 = time.time()
        ki = site_weights(anat, int(s))
        mean, trials = evoked(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim, w_bb=tp.w_bb,
                              n_trials=n_trials, cfg=cfg, seed=int(rng.integers(1 << 30)))
        clean, _ = evoked(tp.raw, anat.sc, anat.dist, anat.lm, ki, stim, w_bb=tp.w_bb,
                          n_trials=1, cfg=cfg, noise_scale=0.0, seed=0)
        half = n_trials // 2
        noise = rung.sensor_noise_uV
        means.append(add_sensor_noise(mean, noise, rng))
        # Each half average carries sqrt(2) more sensor noise than the full average.
        h1s.append(add_sensor_noise(trials[:half].mean(0), noise * np.sqrt(2), rng))
        h2s.append(add_sensor_noise(trials[half:].mean(0), noise * np.sqrt(2), rng))
        cleans.append(clean)
        kis.append(ki)
        log.info("teacher %d site %d done in %.0fs", seed, s, time.time() - t0)

    tag = f"{rung_name}_n{anat.n_nodes}_t{seed}"
    out = DATA_DIR / "teachers" / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "eeg.npz", mean=np.array(means), clean=np.array(cleans),
                        half1=np.array(h1s), half2=np.array(h2s), ki=np.array(kis),
                        w_bb=tp.w_bb, parcels=anat.parcels)
    save_json({"tag": tag, "seed": seed, "rung": rung_name, "rung_description": rung.description,
               "n_nodes": anat.n_nodes, "sites": sites, "n_trials": n_trials, "spread": spread,
               "w_bb_sigma": sigma, "sensor_noise_uV": rung.sensor_noise_uV,
               "raw": tp.raw, "effective": tp.effective,
               "sim": {"step_size": cfg.step_size, "tr": cfg.tr, "warmup_ms": cfg.warmup_ms}},
              out / "truth.json")
    log.info("saved %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rung", default="R0")
    ap.add_argument("--n-nodes", type=int, default=None, help="default: all 200 parcels")
    ap.add_argument("--sites", type=int, default=8)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--spread", type=float, default=0.15)
    ap.add_argument("--w-bb-sigma", type=float, default=None, help="override the rung's value")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = get_logger("make_teacher")
    for s in args.seeds:
        make_one(s, args.rung, args.n_nodes, args.sites, args.trials, args.spread, args.w_bb_sigma, log)


if __name__ == "__main__":
    main()
