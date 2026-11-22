# syn_brain: Synthetic-Brain Recovery Study

**Status:** planned, not implemented yet. This folder will hold the study code and results.

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

## Planned layout

```
syn_brain/
  README.md          this plan
  configs/           one config per rung (R0–R6) and student tier (S1–S3)
  teacher.py         draw teacher parameters, simulate trials per site, add sensor noise
  protocols.py       tests T1–T4, wrapping the existing hold-out scripts
  baselines.py       noise ceiling, nearest-site and linear propagation baselines
  run_study.py       runs a rung × tier × test grid and saves results
  results/           outputs (not committed)
tests/
  test_recovery.py   R0 + S1 on a 10-node network must recover its parameters
```

Metrics that are useful beyond this study (parameter recovery, restart spread, noise ceiling) go into `neurofe/metrics.py`. Once the study is done, reusable parts move into `neurofe/validation/` so other users can run the same tests on their own data.

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
