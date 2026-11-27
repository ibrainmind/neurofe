# syn_brain: Synthetic-Brain Recovery Study

**Status:** Phase 1 (build and calibrate) code is in place; the Phase 1 gate **did not pass**. See [Phase 1 results](#phase-1-results-2026-11-21) before starting Phase 2. The follow-up **network-scale study has since passed its N2 gate** (below) with two flagged caveats under active investigation — see `docs/network_study_plan.txt` for the current plan.

## Network-scale study: N0-N2 gate passed, g's bias flagged for diagnosis (2026-11-24 to 2026-11-26)

**A 10-node network (tier S1, Laplacian coupling, the same coupling neurofe always used) now recovers k, a, c3·c4 and mu with median error under 5% across 8 independently-drawn teachers, and mu — which had zero gradient in the original Phase 1 run because neurofe casts conduction delays to integers — is reliably recovered for the first time in this study (median progress ≈1.0, i.e. full recovery, in every teacher).** Getting here took several rounds of diagnosis, each one changing the plan's working hypothesis: a non-diffusive coupling fix (meant to give nodes a non-zero mean drive, since the existing zero-mean Laplacian leaves the network at rest) was built and Fisher-tested first, but a single-teacher pilot showed it get stuck in a bad optimization basin while the plain Laplacian — once combined with the harness fixes the single-node study had already found necessary (log-scaled parameters, fractional delays, a rich multi-site/multi-amplitude stimulus) — recovered everything from one start. The real 8-teacher gate then failed in substance on its first two attempts (most teachers plateaued far from the truth) until the actual cause was isolated: teachers whose randomly-drawn stimulus gain `k` was far below nominal were effectively under-driven by the fixed stimulus protocol. A per-teacher amplitude calibration helped some teachers but not all, and a direct stability/Fisher check on the two worst teachers showed both were "CRB-good-but-backprop-bad" (the same recurring failure mode the single-node study found for c1·c2) rather than genuinely under-informative — which motivated building a 5-restart ensemble wrapper (the network-scale version of the single-node study's own validated RICH+restarts+lowest-loss recipe). With that wrapper, the full 8-teacher gate passes comfortably on all pre-registered criteria. Full numbers and the diagnostic trail: `docs/network_study_plan.txt` Part N2; decision-by-decision narrative: `reports/DECISIONS.md` (2026-11-24 through 2026-11-26).

**Two things the N2 pass does not cover, carried forward rather than buried in the headline "PASS":** (1) `g`, the network's global coupling-strength parameter, fails badly (median error 27%, systematically biased low in 6 of 8 teachers) even with the ensemble wrapper — it was never one of the three named gate parameters, so the pass is real but narrower than it sounds; (2) no network-scale Laplace/credible-interval calibration was run, unlike the single-node study, so these are point estimates with no check on whether reported confidence would be honest. **Decision (2026-11-26, after a holistic review): diagnose g's bias before either scaling up (Part N3) or building the missing calibration check**, rather than doing either next. Two free re-analyses of already-saved results (no new training) narrowed this down further: g's own Fisher/CRB is excellent (0.003% — the best-identified parameter in the model) and not part of any sloppy direction, ruling out an information problem; but the 8 already-fit teachers show a clear pattern — the 3 with the lowest true `g` converge almost exactly, the 4 with the highest true `g` all converge to a much lower, narrower band regardless of which direction their restart started from, which looks like an attractor pulling high-g fits down rather than noise. The leading hypothesis is a gradient-killing saturation artifact in the coupling term's safety clip (`core/diffsim.py`'s `U_BOUND=500` tanh clip — the same clip already confirmed to silently kill gradients for the non-diffusive coupling fix at high g) triggering during the stimulation pulse for high-true-g teachers specifically; not yet confirmed. The next concrete step (not yet run) is a cheap instrumented replay of the 8 already-fit parameter sets to check this directly, before committing compute to Part N3 or the network Laplace wrapper. Full reasoning: `reports/DECISIONS.md` 2026-11-26 ("Holistic review after the N2 pass"); the new diagnostic step is written up as Part N2.5 in `docs/network_study_plan.txt`.

## Single-node study: final conclusion (2026-11-22)

**Are the parameters identifiable and do fits generalize? No, not reliably — only k (stimulus gain) and a (excitatory rate) are.** The E-I loop gains (c1·c2, c3·c4) are unidentifiable at rest, only sometimes recoverable at the firing state (p≥150, and not robustly across different ground-truth teachers), and fits don't generalize to held-out time windows or stimulus amplitudes at either operating point. **Validated directly (not assumed) that parameter accuracy drives generalization:** correlating per-model recovery error against per-model generalization error shows the theoretically-identifiable quantities (a, c3·c4, c4, b, k, c1·c2 — all p<0.03) predict generalization failure, while individual c1 does not (p=0.55) — exactly matching the structural theory. Full writeup: `reports/single_node_conclusions.pdf`. **A practical fix was found and tested: RICH protocol + ≥20 restarts + pick the lowest-training-loss one — NOT hard parameter bounds.** Hard box constraints on individual c1-c4 roughly *double* the error (tested directly). With a single pulse, the best-fitting model in an ensemble tells you nothing about accuracy (one restart had near-zero loss and 64% c1·c2 error); with RICH, the best-fitting model reliably is the most accurate one (loss↔accuracy correlation 0.09→0.83). This is a cheap point-estimate fix, not a substitute for honest uncertainty. **Built and validated a real posterior (Laplace approximation) across 24 independently-drawn random teachers (parallelized via Workflow):** median point-estimate error 4-8%, 90% credible intervals achieve ~96% coverage — both well-calibrated at a safe operating point (p=150). Found one real flaw: a small, statistically significant systematic downward bias (18/24 teachers, p=0.01-0.02), likely "winner's curse" from selecting the best of 15 restarts by lowest loss — flagged, not yet corrected. **Recommendation: characterize/fix the bias and extend calibration to harder operating points (E-dominant) before moving to Part B (disease markers) or network-scale work.** Decision log: `reports/DECISIONS.md`.

## E/I-balance bifurcation map (2026-11-22, corrected 2026-11-24)

Mapped the full phase diagram (tonic drive x E/I balance): a clean, monotonic quiet-to-active boundary that bends with balance. **Corrected** after an independent equation-level analysis (`reports/jr_parameters_explainer.pdf`, exact fixed points + eigenvalues) found two errors in the original Euler-simulation-based map: (1) a **bistable/hysteresis fold region** was invisible to cold-start-only simulation and had been silently mislabelled "quiet"; (2) the specific point used for the original "no hysteresis" test was a **numerical artifact** of neurofe's 0.1 ms Euler step (the true equations are stable there), not a real limit cycle. Real hysteresis exists instead at the quiet/active fold itself. The corrected active region is larger, making the single-pulse-vs-RICH identifiability gap for "too much E" states even more extreme than first reported (CRB median ~265,000%, not ~2,800%). c1·c2 in E-dominant states still stays hard even with the richer protocol. Full report: `reports/ei_balance_bifurcation_map.pdf`.

## Single-node virtual-TMS study (2026-11-22)

One teacher JR node (neurofe nominal values) identified by a student JR node through backpropagation, under 12 virtual-TMS conditions. Report: `reports/single_node_tms_study.pdf`; follow-up plan: `docs/single_node_study_plan.txt`; open item on background drive: `docs/background_drive_note.txt`. To re-run: `bash syn_brain/scripts/run_single_node_tms.sh` (details in `docs/single_node_rerun.txt`). Follow-up plan on node connections and early Parkinson's disease detection: `reports/pd_connectivity_study_plan.pdf`.

- At nominal values a node with no background input is almost a linear filter. Only k·A (P1 amplitude) and a (P1 latency) are well identified; the E–I loop (c1–c4, b) leaves a 0.6 µV trace.
- A, B and vmax are exactly degenerate with the c's and k (two scaling symmetries), so they must stay fixed.
- neurofe's raw parameterisation (Adam, lr 0.05) is the main reason Phase 1 only recovered k: same data, log-scaled parameters reach ~250× lower loss.
- Fitting 0–60 ms and predicting the rest (temporal hold-out) gives a 91% late-window error with a perfect model and no noise. Expect T2 to fail at R0.
- neurofe's Laplacian coupling supplies no mean drive, so nodes never reach the active state where the loop becomes identifiable. This probably explains Finding 1 below.

### Firing-state follow-up: E9/E10, settle-time bug, and the p=120 bifurcation (2026-11-22)

E7 (tonic drive p=110) stayed just below the jump to the active branch and showed no gain over rest, as predicted. Added p=120 (the threshold crossing) and p=150 (settled on the active branch) and ran backprop recovery (E9 noise-free, E10 at 0.42 µV). Full writeup and figures: `reports/bifurcation_followup.pdf`; narrative: `docs/background_drive_note.txt` section 5; decisions: `reports/DECISIONS.md`.

- **Found and fixed a real bug:** the simulator's 300 ms default settle time isn't enough once tonic drive is on near/above the p≈115 jump (critical slowing down at the bifurcation) — pre-stimulus baseline drift up to 0.51 µV, bigger than the 0.42 µV noise floor. `core/node.py`'s default `settle_ms` raised 300→1500.
- **The fix did not explain the recovery gap — it ruled it out.** After correcting settle time, recovery at both p=120 and p=150 got *worse*, not better, while the EEG fit itself stayed equally good — the signature of a flat/multi-valued loss landscape, not a numerical artifact.
- **Continuation (warm-start from p=150, step down to p=120) doesn't fix it either** — it trades one failure mode for another. Per-start spread (not just the median) shows p=120 is genuinely multi-modal: most starts land in a much worse c3·c4 basin, while c1·c2 splits into a good cluster plus catastrophic outliers.
- **Closed recommendation:** use p≥150 (plain cold-start Adam) as the default operating point for Part A. Don't keep tuning the optimizer at p≈115–140 — treat it as a multi-modal-posterior region and use methods built for that (study plan Part C: C6 simulation-based inference, C7 HMC/NUTS) if that region specifically needs fitting.

## Phase 1 results (2026-11-21)

Everything below was run on a 4-core Intel Mac (CPU only), PyTorch 2.2.2, in the repo's `.venv`.

### What works

- **The teacher simulator is exact.** `core/simulator.py` reproduces neurofe's `JansenRit.forward` to floating-point precision, with both plain and perturbed SC (`tests/test_simulator.py`). Teacher and student are therefore the same model in rung R0. The simulator runs many trials as one batch: 100 trials of one site on the 200-parcel brain take about 45 s.
- **Fitting cost is known.** One `JansenRit.fit` epoch takes about 5 s at both 10 and 200 parcels, because Python overhead dominates. A standard 120-epoch fit takes about 10 minutes, and about 3 fits can run in parallel on this machine.

### Finding 1: synthetic TEPs with nominal parameters don't look like real TMS-EEG

`scripts/calibrate_teacher.py` compared teachers (nominal parameters plus random draws, 2 sites each, 100 trials) with the 20 real subjects in PyTepFit's data:

| Feature | Real TMS-EEG (median of 20) | Synthetic teacher |
| --- | --- | --- |
| GMFA peak | 6.1 µV at ~105 ms | 24–38 µV at 14 ms |
| Late/early GMFA ratio | 0.84 | 0.15–0.19 |
| Baseline std | 0.42 µV | ~0 µV (the model's own noise barely reaches the EEG) |

The synthetic response is one large early peak that decays by ~50 ms. With so little late activity, the temporal hold-out test (T2) would be trivially easy or meaningless. **Before Phase 2, the teacher needs realistic late dynamics.** Options:

1. Use teachers *fitted to real subjects*: fit S3 to a real TEP and use the fitted model as ground truth.
2. Search the parameter space, including `w_bb`, for draws whose TEPs match the real features above.

Figure: `results/calibration/calibration.png`.

### Finding 2: the Phase 1 gate fails; S1 doesn't recover a clean teacher

`scripts/recover_small.py` used 3 teachers on a 10-node network at R0, student S1, weak priors, 120 epochs, lr 0.05. The student starts at nominal values; a progress of 1 means it reached the truth.

| Parameter | Behaviour across the 3 teachers | Why |
| --- | --- | --- |
| `k` (stimulation gain) | **Recovered** (errors −2.6%, +6.3%, +0.3%) | Directly scales the early peak; by far the largest gradient |
| `mu` (conduction velocity) | **Never moves** | Delays are cast to integers (`(dist / mu).type(torch.int64)`), so `mu` gets **no gradient**. It can't be learned by backprop in neurofe or in the PyTepFit code this mirrors |
| `g`, `c1`, `c2` | Stay at their starting values | Gradients about 10⁻⁶, about 10⁴× smaller than `a` or `k` |
| `b`, `c4`, `std_in` | Move to the **same values whatever the truth** (`b` ≈ 40, `c4` ≈ 42, `std_in` ≈ 282) | They absorb mismatch between the fitting procedure and the data, not information about the teacher |
| `a`, `c3` | Inconsistent | — |

EEG fit after training: Pearson 0.985–0.995, the same as the **untrained** nominal model (0.979–0.997). Relative RMSE improves about 25% (for example 0.113 → 0.082). Pearson on these responses is dominated by the leadfield pattern of the stimulated site and barely depends on the parameters, so high Pearson says little about whether a fit is right.

What this means for the questions:

- **Q1 (identifiability) is answered "no" for most parameters, even in the best case.** That supports the concern behind Ye's results: a good fit does not mean the parameters were found.
- **Conduction velocity is a structural blind spot.** Correcting `mu` by hand (one of Ye's attempts) can't be matched by the optimizer, because it never updates `mu`.
- **The proposed Q1 threshold is flawed.** Teachers are drawn within ±15% of nominal, so a student that never moves already lands "within 20% of truth". Use the progress measure, or draw over a wider range.

### Other quirks found

- **Training error rises before it falls.** `JansenRit.fit` runs one continuous simulation across epochs with no warm-up. Epoch 0 starts from a quiet state and has the lowest error (0.088); epochs 1–3 jump to about 0.30, then error slowly falls to about 0.19. Error at the end of training is higher than after epoch 0.
- **Training and prediction see different conditions.** Training compares one noisy run against a trial average with no warm-up; `predict` uses a 1 s warm-up. Evaluation here uses the batched simulator with 100-trial averages, matching how the teacher is generated.
- **The noise floor can't be removed.** neurofe hard-codes a noise std lower bound of 150, so a noise-free student run is impossible in neurofe. The batched simulator has `noise_scale` (0 = noise-free) for that comparison.

### Suggested next steps (to decide with Ye before Phase 2)

1. Fix teacher realism (Finding 1), most likely with option 1, teachers fitted to real subjects.
2. Re-run the gate with a wider draw range (±40%), the progress criterion, the PyTepFit priors, and a higher learning rate for large-scale parameters such as `g`. This tells apart "not identifiable" from "optimizer can't reach it".
3. Decide whether `mu` should become differentiable, for example with fractional delays by linear interpolation of the history buffer. This changes neurofe itself.

### How to reproduce

From the repo root:

```bash
.venv/bin/python -m pytest syn_brain/tests -q                     # 4 tests, ~40 s
.venv/bin/python -m syn_brain.scripts.calibrate_teacher --teachers 2 --sites 2
for s in 0 1 2; do .venv/bin/python -m syn_brain.scripts.recover_small --teacher-seed $s --epochs 120 & done; wait
.venv/bin/python -m syn_brain.scripts.make_teacher --seeds 0 1 2 3 --rung R0 --sites 8   # Phase 2 data
```

PyTepFit's anatomy (leadfield, connectome, distances; MIT license) is vendored in `external/pytepfit/`. Its real TMS-EEG file, needed only by `calibrate_teacher.py`, is fetched with `bash syn_brain/scripts/fetch_pytepfit_data.sh`. Set `SYN_BRAIN_PYTEPFIT_DATA` to use another copy.

## Why this study

Ye's validation tests found that models fitted with the Momi et al. (2022) method fail to generalize:

- **Temporal hold-out:** fitted on −50 to +150 ms, the model does not predict the late window. Pearson is 0.054 on TMS-EEG and 0.258 on CCEP, against in-window fits of 0.87 and 0.81.
- **Cross-site transfer:** fitted jointly across stimulation sites, the model scores 0.08–0.18 on a held-out site, against 0.73–0.78 in-sample.

With real data, three causes are possible, and we can't tell them apart:

1. **Identifiability:** the data don't pin the parameters down, so many parameter sets fit equally well.
2. **Optimization or capacity:** the fit overfits or stops in a poor local minimum. neurofe fits a per-edge connectivity gain `w_bb` and the full leadfield `lm`, which adds up to tens of thousands of free values.
3. **Misspecification:** the brain is not a Jansen-Rit (JR) network, or the data contain non-cortical signal, such as auditory and somatosensory responses in late TMS-EEG.

This study builds a **synthetic brain**: a JR network whose parameters we choose. The synthetic brain is the *teacher*. A second JR model, the *student*, sees only the teacher's EEG and is fitted with neurofe's PyTorch backpropagation (`JansenRit.fit`). Because we know the true parameters, every failure can be traced to a cause.

- The student still fails Ye's tests when the teacher has exactly the same form → the problem is in the method (cause 1 or 2).
- The student passes → Ye's real-data failures point to misspecification or confounds in the data. We then add mismatches one at a time to find which one breaks generalization.

## Questions

| # | Question | If yes | If no |
| --- | --- | --- | --- |
| Q1 | **Identifiability.** Fitted on the full window at one site, does the student recover the teacher's parameters, not just its EEG? | A good fit says something about the parameters | Fitted values should not be read as properties of the brain |
| Q2 | **Temporal extrapolation.** Fitted on 0–150 ms, does the student predict the teacher's 150–400 ms? | The real-data failure is caused by the data, not the method | The early window doesn't constrain the late-controlling parameters, even in the best case |
| Q3 | **Cross-site transfer.** Fitted jointly on k sites, does the student predict a held-out site? | Shared parameters can be learned; the real-data gap comes from misspecification | The method can't learn parameters that transfer |
| Q4 | **Breaking point.** If Q2 and Q3 pass in the clean case, which added mismatch makes them fail? | Points to the real-data property to fix or model | — |
| Q5 | **Remedies.** Which changes to the student close the gap? | A concrete fix to ship in neurofe | — |

## Design

### Teacher (the synthetic brain)

- **Anatomy:** structural connectivity, distance matrix and leadfield bundled with PyTepFit, so dimensions, delays and sensor geometry match the real-data tests.
- **Ground-truth parameters:** global JR parameters (`A`, `a`, `B`, `b`, `g`, `c1`–`c4`, `v0`, `r`, `mu`) drawn from physiological ranges around the `JRParams` defaults. `w_bb` is a log-normal perturbation of the SC.
- **Subjects:** 8 teachers from independent parameter draws. This replaces n = 1 with a distribution.
- **Stimulation sites:** 8 per teacher, spread across lobes, including both neighbouring and distant pairs. Each site gets a pulse into the pyramidal population through the `ki` mask.
- **Observation:** 100 noisy trials per site (`std_in`), averaged into an evoked response. Sensor noise is added at an SNR matched to the real TMS-EEG and CCEP recordings, then the real pipeline's preprocessing and −50 to +400 ms window are applied.

### Student (the model under test)

neurofe `JansenRit`, using the same SC, distances and leadfield as the teacher. It comes in three capacity tiers:

| Tier | Fitted parameters | Approx. free values |
| --- | --- | --- |
| S1 Global | Global JR parameters and stimulation gain `k` | ~15 |
| S2 + connectivity | S1 plus per-edge `w_bb` | ~15 + N² |
| S3 PyTepFit default | S2 plus the full leadfield `lm` | ~15 + N² + channels × N |

N is the number of parcels. S3 is the setup used in the real-data tests.

### Avoiding the "inverse crime"

When the same code generates and fits the data, the test can be too easy. To guard against this:

- The teacher integrates at 0.1 ms and the student at 1 ms.
- Teacher and student use independent noise realizations.
- One rung of the ladder generates data with the separate NumPy simulator (`neurofe/sim.py`, `run_net_sim`), which uses a different coupling form.

## Experiments

### Misspecification ladder

Each rung adds one realistic mismatch. The first rung where the tests fail identifies the likely cause.

| Rung | What changes in the teacher | Real-data property it imitates |
| --- | --- | --- |
| R0 Clean | Nothing: same form as the student, trial noise only | Best case; tests identifiability and optimization alone |
| R1 Noise | Sensor noise at the real SNR, finite trials | Measurement noise |
| R2 Regional heterogeneity | Excitability (`A` or `v0`) varies by node; the student has only global values | Regional excitability gradients |
| R3 Wrong connectivity | The student gets a corrupted SC (noise, missing long-range edges, false short edges) | Tractography errors |
| R4 Stimulus mismatch | The teacher's input spreads to neighbouring parcels with distance fall-off; the student assumes a single-parcel pulse | SEEG current spread and white-matter stimulation |
| R5 Non-cortical confound | A site-independent late component (fixed central topography, 100–250 ms) is added | Auditory and somatosensory responses in TMS-EEG |
| R6 Model-form mismatch | Data generated by `run_net_sim` | The real brain is not exactly a JR network |

Rungs are cumulative by default (R3 includes R1 and R2). Each mismatch is also run on its own on top of R1, to isolate its effect.

### Tests run at every rung

1. **T1 Single-site full-window fit:** fit −50 to +400 ms at one site; report the EEG fit and parameter recovery (Q1).
2. **T2 Temporal hold-out:** fit −50 to +150 ms, then free-run to +400 ms. Compare the late window using a noise-free student run and a 100-run noise average, never a single noisy run (Q2).
3. **T3 Leave-one-site-out:** fit on 7 sites, predict the 8th, rotating through all 8 (Q3). Three parameter-sharing designs:
    1. **All shared:** one parameter set for all sites (the design used in the real-data test).
    2. **Global shared, per-site gain:** global JR parameters and `w_bb` shared; stimulation gain `k` per site.
    3. **Global shared, regional excitability:** as in 2, plus one excitability value per parcel, shared across sites.
4. **T4 Remedies:** repeat T2 and T3 across tiers S1–S3, number of training sites (1, 2, 4, 7), and prior strength on `w_bb` (Q5).

T2 and T3 reuse the existing hold-out scripts from the real-data tests, pointed at synthetic data, so code differences stay out of the comparison.

## Metrics, baselines and decision criteria

### Metrics

- **EEG fit:** per-channel Pearson, GMFA correlation and relative RMSE, reported separately for the early (0–150 ms) and late (150–400 ms) windows.
- **Parameter recovery:** relative error per global parameter; rank correlation between fitted and true `w_bb`.
- **Identifiability:** spread of fitted parameters across 5 random restarts. Same EEG fit with different parameters means those parameters are not identifiable.
- **Functional recovery:** lesion the same parcel in teacher and student and compare the predicted change in EEG. This checks whether the virtual-lesion analysis holds up even when individual parameters are not recovered.

### Baselines

- **Noise ceiling:** split-half correlation of the teacher's trials, the best any model can score.
- **Nearest-site baseline:** predict a held-out site with the response of the closest training site.
- **Linear propagation baseline:** spread the stimulus along the SC with distance delays and project it through the leadfield, with no JR dynamics.

### Pass criteria (proposed, to agree before running)

| Question | Passes if |
| --- | --- |
| Q1 Identifiability | Global parameters within 20% of truth for at least 6 of 8 teachers, and restarts agree within 10% |
| Q2 Temporal extrapolation | Late-window Pearson at least 70% of the noise ceiling |
| Q3 Cross-site transfer | Held-out Pearson within 0.10 of in-sample, and at least 0.10 above the better baseline |

### Interpreting results

| Outcome | Conclusion | Action |
| --- | --- | --- |
| Fails at R0 | The method can't recover parameters or extrapolate even when the model is exactly right | Change the method: fewer free parameters, multi-site training, stronger priors (T4 shows which) |
| Passes R0–R1, fails at rung Rk | Mismatch k is the leading candidate for the real-data failure | Model or remove that mismatch in the real data, then re-run the real-data tests |
| Passes every rung | The ladder does not explain the real-data results | Audit the real-data pipeline: preprocessing, trial alignment, leadfield and SC registration |

## Layout

```
syn_brain/
  README.md               this plan and the results so far
  core/                   library code
    paths.py              folder layout, logging, JSON saving; puts neurofe on the path
    anatomy.py            PyTepFit SC, distances, leadfield, real TEPs; sub-networks; site masks
    params.py             ground-truth draws, effective-value transforms, teacher/student JRParams, tiers S1–S3
    simulator.py          batched JR simulator for the teacher (exact copy of JansenRit.forward)
    student.py            build and fit the student; trial-averaged predictions
    metrics.py            EEG fit by window, GMFA, noise ceiling, TEP features, parameter recovery
    ladder.py             rung configurations (R0, R1 implemented; R2–R6 declared)
  scripts/                runnable entry points (python -m syn_brain.scripts.<name>)
    calibrate_teacher.py  compare synthetic TEPs with real TMS-EEG
    recover_small.py      Phase 1 gate: 10-node R0 + S1 parameter recovery
    make_teacher.py       generate synthetic subjects (8 sites each) for Phase 2
  tests/                  pytest: simulator equivalence, recovery pipeline
  data/                   generated teachers (not committed)
  results/                JSON and figures per experiment (not committed)
  logs/                   run logs (not committed)
```

Still to build for Phase 2: test protocols T1–T4 (wrapping the existing hold-out scripts) and the nearest-site and linear propagation baselines. Once the study is done, reusable parts move into `neurofe/validation/` so other users can run the same tests on their own data.

## Compute

The full grid (7 rungs × 3 tiers × 8 teachers × restarts and folds) runs to thousands of fits, and each T3 joint fit costs about 7 single-site fits. A single fit hasn't been timed yet, so the budget is set in week 1. To keep it tractable:

- Run the full grid only at R0 and R1.
- At R2–R6, run tier S3 with sharing designs 1 and 2, then expand around the first rung that fails.
- Start with 4 teachers; add the other 4 only where results are borderline.

## Timeline (estimate: one person with GPU access)

| Phase | Weeks | Work | Gate at the end |
| --- | --- | --- | --- |
| 1 Build and calibrate | 1–2 | Teacher generator, R0 config, student tiers, recovery unit test; time one fit; check that teacher EEG resembles real TEP and CCEP | S1 recovers R0 parameters on 10 nodes; pass thresholds agreed |
| 2 Best case (R0–R1) | 3–4 | T1–T3 on all tiers and sharing designs, 4 teachers | **Decision:** method problem (fails at R0) or data and model mismatch (passes) |
| 3 Ladder and remedies (R2–R6) | 5–7 | One mismatch per rung; T4 remedies | Breaking point and best remedy identified |
| 4 Write up and ship | 8 | Report; merge the validation module and a Limitations section into neurofe; re-run the real-data tests with the best remedy | — |

If the student fails at R0, Phase 3 shifts from the ladder to method remedies.

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| The ladder misses the real cause | Every rung passes, yet real data still fail | Audit the real-data pipeline (see Interpreting results) |
| Teacher parameters drawn from unrealistic ranges | Results don't carry over to real recordings | Before fitting, check that teacher EEG matches real amplitude, latency and GMFA shape |
| Inverse crime makes R0 too easy | An optimistic pass | Finer teacher time step, independent noise, rung R6 |
| Compute exceeds budget | Grid cut short | Week-1 timing run and a staged grid |
| Thresholds set after seeing results | Biased conclusions | Fix thresholds before the first run |

A synthetic pass can't prove the method works on real brains; it can only rule causes in or out. A failure at R0 is conclusive. A pass at every rung narrows the search.

## Open questions

**For Ye**

- Do the proposed pass thresholds match what you'd accept as "generalizes"?
- Can the hold-out scripts from your two tests be shared as they are, so T2 and T3 can reuse them?
- What SNR and trial counts do the real TMS-EEG and CCEP recordings have? R1 needs them for calibration.

**For Dave**

- Which parameters are meant to be subject-level and which condition-specific? This decides the right sharing design in T3.
- Is there a reason late components shouldn't be predictable from the early window in this model class?
- Is there a multi-site, multi-subject empirical dataset we could use once the synthetic results are in?
