# Decision log (append-only)

Format: date — decision — one-line reasoning. Only decisions that could plausibly have gone
another way and changed a conclusion; not routine implementation choices.

## 2026-11-22 — Adopting CLAUDE.md process without restarting at Phase 0

`syn_brain/CLAUDE.md`, `docs/prompt.md` and `docs/overview.md` (provided by William) define a
phase-gated process (Phase 0-5, `study/phaseN_*.md` specs, `reports/DECISIONS.md`). Those
phase specs don't exist yet, and writing them now would mean re-deriving pass criteria for
work already done. **Decision:** treat the existing single-node virtual-TMS study
(`docs/single_node_study_plan.txt`, which already fixes its own pass criteria in section 6
before running) as the live spec in force, and apply the new process's reporting style
(one-page verdicts, PDF output, decision log, adversarial self-review) to it going forward,
rather than re-deriving a Phase 0-5 structure retroactively. `study/00_overview.md` is kept
as the stated long-term goal; phase spec files are not authored unless/until a true restart
is wanted.

## 2026-11-22 — Firing-state follow-up (E9/E10) scope

Tested only tonic drive p=120 (threshold crossing) and p=150 (settled active branch), SP1000
and DOSE protocols, 10 starts, noise-free (E9) and 0.42 uV noise (E10) — not the full Part A
input-family sweep (A1-A6) in the study plan. Reasoning: this directly answers the question
asked ("push the node into the firing state and check recovery") at low compute; the other
input families are a separate, larger follow-up.

## 2026-11-22 — settle_ms default raised from 300 to 1500 in core/node.py

Diagnostic (this session): the pre-stimulus baseline at tonic drive p=115-200 is still
drifting by up to 0.5 uV at the old default `settle_ms=300` (bigger than the 0.42 uV sensor
noise floor), settling to <0.005 uV by `settle_ms=1500` at every p from 0-240 checked.
Affects every experiment that uses tonic drive (E7, E9, E10); does not affect p=0 experiments
(already converged by 300 ms there). **Decision:** change the default in `simulate`/
`simulate_rows` rather than add a per-call override, so every current and future caller gets
the fix without having to remember to ask for it. Cost: ~2.7x more compute per iteration for
any tonic-driven fit. Re-ran E9/E10 with the fix (old settle=300 results kept in
`/tmp/.../scratchpad/train_E9-E10_settle300_OLD.npz` for the before/after comparison in the
report, not committed). E7 was not re-run this session (p=110 residual was already small,
~0.003 uV, at the old default) — flagged in the report as a residual gap, not re-tested, to
keep this session's compute bounded.

**Outcome (2026-11-22, same day):** the fix did NOT improve recovery — c1·c2/c3·c4 median
error got worse at both p=120 and p=150 after correcting settle_ms, while the EEG fit itself
stayed equally good (loss ~3-6e-3 either way). This rules settle time out as the cause of the
p=120-vs-p=150 recovery gap; it is kept anyway as a genuine correctness fix. Full report:
`reports/bifurcation_followup.pdf`. Leading remaining hypothesis: the student crosses its own
quiet/active branch during cold ±50% random-start search, which settle time cannot fix.
Decision: proceed to the continuation experiment (warm-start from p=150, step down to p=120)
rather than re-ask William before testing this — matches the standing rule to make a
reasonable call on an unspecified design choice and log it, not stop and ask.

**Continuation experiment outcome (2026-11-22, same day):** warm-starting from the converged
p=150 solution and stepping down through p=140/130 to p=120 does NOT reliably fix recovery —
it changes the failure mode rather than removing it. Per-start spread (not just the median)
shows p=120-via-continuation is sharply bimodal: c3·c4 median got worse (52%->222%, nearly
all starts land near 200%+); c1·c2 median improved (81%->17%, passing the 20% gate) but only
because a minority of starts land in an excellent basin while most do not, plus 3 catastrophic
outliers (300-600%). p=150 cold-start Adam has no such catastrophic outliers in the same
10-start sample. **Closing this investigation with a final recommendation**: use p>=150 (plain
cold-start Adam) as the default operating point for Part A; p~115-140 is treated as a
multi-modal-posterior region, pointing at study-plan Part C methods built for that (C6
simulation-based inference, C7 HMC/NUTS) rather than further Adam/optimizer engineering. Not
pursued further this session, to respect the compute-budget standing rule and because the
practical question (what operating point to use) is already answered. Full writeup:
`reports/bifurcation_followup.pdf`.

## 2026-11-22 — Closing the single-node study with an explicit identifiability+generalization verdict before any network-phase work

William flagged (correctly) that the firing-state investigation never answered the parent
study's actual questions: are parameters identified, and do fits generalize? Added two more
experiments (E11: temporal hold-out at p=150, mirrors E8; E12: 4 perturbed teachers at p=150,
mirrors E6) plus a free re-analysis of existing E9 fits for cross-amplitude generalization.
Combined with the original rest-state study (E1-E8), wrote a single final verdict:
`reports/single_node_conclusions.pdf`. Headline: k and a are identifiable everywhere;
c3·c4 is identifiable only at p>=150 and only inconsistently (7-28% across 4 perturbed
teachers, one teacher's c1·c2 fails at 91-95%); generalization (temporal hold-out, held-out
amplitude) fails at BOTH operating points, worse in absolute terms at p=150 than at rest for
the late window. **Decision: do not proceed to network-scale work (Phase 2) for the loop-gain
parameters.** Recommend testing an uncertainty-aware method (Part C) on this same single-node
problem first -- it is the cheap place to find out whether any method can honestly report
"not identified" instead of a confident wrong point estimate. Network-level background-drive
work (fixing the missing late-TEP-activity finding) is a separate, parallel, lower-priority
track that improves teacher realism, not identifiability -- flagged explicitly so the two
aren't conflated in future planning.

## 2026-11-22 — Mechanism behind the c1·c2/c3·c4 recovery failure (William asked "why", not just "that it fails")

Investigated directly rather than leaving it at "sloppy/multi-modal". Found three distinct,
compounding causes:

1. **Protocol design flaw (fixable):** a single fixed p=150 does not put every teacher in the
   same dynamical regime. Direct state-space check (6s settle + 1s stability window, bypassing
   `simulate()`'s baseline correction which masked this earlier): nominal/T0/T1 land in a clean,
   non-oscillating active state at p=150 (spread <0.001 mV over 1s); T2 and T3 sustain large
   oscillations there (spread 1.5-7.6 mV) -- a direct violation of the study plan's own risk
   criterion ("no sustained oscillation"), undetected until now because the per-teacher E12 run
   didn't check it. T3 has a stable active window at p=240 instead (spread 0.003 mV) -- fixable
   by per-teacher calibration.
2. **A structural dynamical limit for some parameter draws (not fixable by choice of p):** T2
   has NO tonic drive level that is both "engaged" (sigmoid slope > 0.6) and non-oscillating --
   scanning p=70-95 in small steps shows the quiet state (slope 0.14-0.21) jumps straight into a
   ~4-5 mV sustained oscillation (slope 0.68) with no intervening stable active window. For this
   draw, the model cannot be brought to a state where the loop is both engaged and measurable
   without inducing abnormal rhythmic activity -- a genuine property of that parameter set, not
   a fitting failure. Plausibly relevant to real E/I-balance biology: parameter shifts that
   reduce inhibition are exactly the kind that could abolish a quiet engaged state this way.
3. **Genuine nonlinear cross-coupling between the loops (not fixable by operating point alone):**
   Fisher eigenvectors at p=150 (nominal) show the two softest directions (10,000x and 500x
   below the stiffest) are dominated by a mixed (c1, c2, c4) combination, not a clean c1·c2
   pair -- c3 loads only weakly on them. The excitatory and inhibitory loops share the pyramidal
   population and become cross-coupled once genuinely nonlinear (sigmoid-engaged), which the
   linear small-signal theory (section 2 of the study plan) does not capture. This likely
   explains why c3·c4 cleans up with a good operating point but c1·c2 does not, even for
   dynamically well-behaved teachers (T0 at p=150, dynamically stable, still gets c1·c2=53%).

Decision: not rerunning full E9-style training with per-teacher calibrated p this session (T3
only; T2 should be flagged as excluded/pathological, not "fixed") -- the stability-screening
result is already conclusive and cheap to obtain; a full backprop confirmation is the natural
next step if this becomes a priority, but the mechanism is established either way. Recommended
next experiment instead: richer stimuli (paired-pulse, train, E+I injection) AT the firing
state, to test whether they resolve the (c1,c2,c4) entanglement the way they tightened bounds
at rest in the original study.

## 2026-11-22 — E/I-balance bifurcation map (major effort, per William/Zheng's request)

Built a full phase diagram: tonic drive p (0-280) x an E/I-balance multiplier m (c1,c2 *=
sqrt(m), c3,c4 /= sqrt(m); m=1 nominal). Vectorized batch simulation made the whole 15x19 grid
cheap (~20s per classification pass), so this was done properly rather than at a few spot
points. Full report: `reports/ei_balance_bifurcation_map.pdf`.

Key findings:
- Clean, monotonic quiet-to-active boundary that bends with E/I balance (E-dominant needs less
  drive to engage) -- model behaves sensibly.
- First-pass regime classification (settle 6s + one 1s window) WRONGLY flagged part of the
  quiet-to-active transition as a limit cycle. Caught by comparing two time-separated windows:
  a genuine limit cycle plateaus; what was flagged was a slowly-damping spiral (critical
  slowing down, again) that fully settles by ~10-15s. Corrected before trusting the diagram.
  Also found: a sustained-oscillation island only exists in ONE place (balanced-to-slightly-
  I-leaning, m 0.5-1.5, high drive p>210ish) -- the earlier T2/T3 oscillation (full independent
  parameter draws, not this clean m-axis) is NOT reproduced by this 2D slice, meaning other
  parameter directions reach instability that this axis doesn't show.
- Hysteresis test (does reducing drive escape a confirmed, plateaued limit cycle): tested on
  the nominal/balanced case and on the independently-fragile T2 teacher. Neither showed
  hysteresis -- ramping drive down exits the oscillation and lands on exactly the same quiet
  state a cold start reaches at that drive level (matched to ~3 decimal places). No bistability
  via drive alone found, for the two cases tested (explicitly flagged as not exhaustive).
- CRB by region: SP1000 alone is catastrophic for E-dominant ("too much E") states (median
  >2500% on both loop gains) -- a single pulse is nearly uninformative there. RICH (paired-
  pulse + train + E/I injection) restores good bounds everywhere (median 10-19%).
- Backprop confirmation (10 starts, 3 representative points, not just CRB): RICH genuinely
  helps (beats SP1000 in 5/6 comparisons). c3·c4 passes the 20% gate with RICH at balanced and
  I-dominant points; misses narrowly at E-dominant (20.5%). **c1·c2 remains hard everywhere even
  with RICH, worst in the E-dominant state (39.8%)** -- the one persistent, now well-
  characterized limit, consistent with the (c1,c2,c4) Fisher entanglement found earlier.

Decision: this is now a solid enough answer to report back rather than keep iterating further
this session -- the remaining open item (c1·c2 in E-dominant states) is specific and actionable
(more stimulus diversity, or an uncertainty-aware method) rather than vague. Standalone
analysis scripts, not yet folded into single_node_tms.py/build_jobs() as permanent experiments
-- worth doing if this becomes a recurring part of the study rather than a one-off investigation.

## 2026-11-22 — Validated (not assumed) that parameter accuracy predicts generalization

William asked for direct empirical validation of "if the student learned the teacher's
parameters, it should generalize" rather than taking it as a logical given (it is only exactly
true for perfect recovery; with partial recovery and known sloppy/gauge directions it's an
empirical question). Correlated per-model parameter-recovery error against per-model
generalization error, reusing already-saved E9/E11/E12 fits -- no new training needed.
Added as section 4 of `reports/single_node_conclusions.pdf`.

Result: clean confirmation, with the right structure. Error in the theoretically-identifiable
quantities (a rho=0.78, c3·c4 rho=0.71, c4 rho=0.57, b rho=0.65, k rho=0.47, c2/c1·c2 rho=0.51,
all p<0.03) correlates significantly with both temporal-hold-out and held-out-amplitude
generalization error. Error in individual c1 does NOT (rho=-0.14, p=0.55) -- exactly as the
linear theory predicts (c1 only enters the EEG as the product c1·c2, so its individual error
should be noise, not signal, with respect to prediction). This independently corroborates the
structural identifiability analysis using real generalization behavior, not just Fisher theory.
Practical takeaway folded into the report: screen fitted models by error in identifiable
combinations (b, c1·c2, c3·c4) and stiff parameters (a, k), not by raw overall parameter error.

## 2026-11-23 — Tested three practical fixes for unreliable c1·c2/c3·c4 recovery (ensemble, hard bounds, loss-as-selector)

William asked for a concrete fix rather than just a diagnosis, and specifically to try bounding
individual c1-c4. Tested directly at the hardest case (E-dominant, m=1.87, p=105), 20 restarts
each condition. Added as section 5 of `reports/single_node_conclusions.pdf`.

**Hard box constraints on c1-c4 (clamped to 0.5-2.0x population-nominal) do NOT help -- they
roughly DOUBLE the median error** (E-dominant c1·c2: 26.2%->63.6%; c3·c4: 12.4%->27.3%), even
though the true values sit comfortably inside the bounds. Likely mechanism: a hard clamp has
zero gradient the instant a parameter hits the wall, which can trap the optimizer against the
wall in a landscape already known to be rugged, rather than helping it find the true answer.
**Decision: don't pursue hard bounds further; a soft/differentiable penalty would avoid this
failure mode but was not tested this session** (flagged as the correct follow-up if priors are
wanted).

**A bigger ensemble (20 vs 10 starts) + picking the lowest-training-loss restart IS a valid,
cheap practical fix -- but only together with the RICH stimulus protocol.** Checked directly
whether "best loss" predicts "best parameters" across 20 restarts, comparing SP1000-only vs
RICH: with SP1000 alone, final loss does NOT predict c1·c2 accuracy at all (Spearman rho=+0.09,
p=0.70) -- the single best-FITTING model in the whole ensemble had 64% c1·c2 error, among the
WORST of the 20, despite a training loss orders of magnitude lower than every other start. With
RICH, loss strongly predicts accuracy (rho=+0.83, p<0.001) and the best-loss start is genuinely
the most accurate (3.1%/0.8% error). **Mechanism: richer stimulation doesn't just tighten the
Cramer-Rao bound, it makes the loss surface itself discriminating, so "which fit looks best" is
only a trustworthy question with the right stimulus.** This directly explains the earlier
settle-time finding ("both runs converge to comparably good loss despite very different
parameters") -- that was always an SP1000-only comparison.

**Decision: validated practical recipe going forward is RICH protocol + >=20 restarts + select
by lowest loss, NOT hard parameter bounds.** This is a cheap point-estimate fix, not a
substitute for a real posterior method (no calibrated uncertainty), but it is now a tested,
concrete answer rather than a hopeful assumption. Caveat: tested at one point with one seed;
the qualitative SP1000-vs-RICH contrast is large and mechanistically well-explained, but exact
correlation values would shift elsewhere.

## 2026-11-22 — Equation-level bifurcation analysis of the E/I map (for the JR parameter explainer)

While updating `reports/jr_parameters_explainer.pdf` with the E/I-balance results, computed the
fixed points and their linear stability (eigenvalues) directly from the node equations over the
same (p, m) plane (`scripts/jr_parameters_explainer.py`: `equilibria`, `fig_stability_map`), and
checked the key claims by direct simulation. Agrees with `ei_balance_bifurcation_map.pdf` on the
main structure (boundary bends with m; real oscillation band at m ~0.55-0.95, p >~ 160). Two
findings that qualify that report:

1. **Hysteresis exists at the quiet/active transition.** For m = 1 the quiet and active states
   coexist for p in [36.5, 114.6) (fold). Direct test: settle active at p = 150, step drive down
   -> stays active at p = 110/80/50 (E-I 5.98/5.70/5.33 mV) while cold starts are quiet
   (2.09/0.79/-0.23 mV); drops to quiet between p = 50 and 40. E-dominant columns are bistable
   already at p = 0. The earlier "no hysteresis" result concerned leaving the limit cycle, and
   its ramp did not compare against a cold start inside this window -- both findings stand.
   Cold-start-only maps label this band "quiet".
2. **The m = 1, high-drive "limit cycle" is produced by neurofe's Euler step, not the model.** The
   active state at m = 1 stays stable in the equations up to p ~ 280 but is very weakly damped
   (Re ~ -0.1 /s at p = 270); explicit Euler at dt = 0.1 ms adds ~ +w^2 dt/2 ~ +0.24 /s at 11 Hz.
   Simulated 30 s from a cold start: dt = 0.1 ms -> sustained 1.19 mV at p = 270; dt = 0.01 ms ->
   decays to 0.11 mV. The m = 0.8 limit cycle is identical at both steps (6.9 vs 6.8 mV) -- real.
   Implication for the inverse-crime rule (teacher at finer dt than student): near a Hopf
   boundary, teacher and student can disagree about whether the column oscillates at all.

Also: excitatory-interneuron sigmoid slope is 0.000 at the E-dominant test point (m = 1.87,
p = 105) vs 0.106 balanced and 0.300 I-dominant -- the excitatory loop is saturated, which
explains why c1*c2 is unidentifiable there regardless of stimulus. No thresholds changed; no new
fits run.

## 2026-11-24 — Built and validated a Laplace posterior across 24 random teachers (closes the posterior-method gap)

Used Workflow (William opted in) to run 4 parallel batches of 6 teachers each (24 total,
independently-drawn, +-30% perturbation of all 7 fitted parameters, screened for a stable
non-oscillating active state at p=150), each fit with the validated recipe (RICH, 15 restarts,
lowest-loss selected), then a Laplace approximation (Fisher/CRB covariance at the MAP point)
for a 90% credible interval on log(c1*c2) and log(c3*c4). Added as section 6 of
`reports/single_node_conclusions.pdf`.

**Methodological correction applied here and recommended going forward:** every earlier
experiment in this study drew random restarts within +-50% of each teacher's OWN TRUE value --
convenient, but real data never gives you that. This run starts every restart from the
POPULATION-NOMINAL value instead (same reference regardless of the teacher's actual deviation),
which is the honest version of not knowing a patient's true state in advance. Mathematically,
`log(theta/nominal)` vs `log(theta/truth)` are the same optimization problem (differ only by a
constant shift in log-space) -- the choice only matters for where random starts are drawn, not
for what the optimizer finds once underway.

**Result: point estimates are accurate (median |error| 7.5% c1*c2, 4.0% c3*c4 -- both comfortably
under the 20% target) and credible intervals are well-calibrated (96% empirical coverage vs a
90% target, n=24).** This is BETTER than earlier single-point checks because this sample is
dynamically well-behaved teachers at a safe operating point, not a deliberately hard stress
test (E-dominant, oscillation-adjacent) -- not a contradiction of earlier findings, a different
(easier, more representative-of-typical-subjects) part of parameter space.

**But found a real, small, systematic bias**: 18/24 teachers (75%) have their MAP point estimate
BELOW the true value, for both combinations (Wilcoxon p=0.01-0.02, not noise). Likely cause:
"winner's curse" -- picking the lowest-loss result out of 15 restarts selects for whichever
random perturbation happens to reduce loss, not just for accuracy, a well-known bias wherever
an estimate is chosen as the best of many. Coverage is still good (intervals are wide enough to
absorb the bias), but the point estimate itself is not unbiased. Not fixed this session --
flagged as the next concrete check (does bias grow with ensemble size, as the hypothesis
predicts; does a holdout-based selection rule remove it).

Decision: this closes the "build a real posterior, check calibration" recommendation from the
previous entry. Recommended next steps (not yet done): characterize/correct the winner's-curse
bias; extend calibration to harder operating points (E-dominant, oscillation-adjacent) before
trusting this for Part B disease-marker work, since disease mechanisms will plausibly push some
patients toward exactly those harder states.
