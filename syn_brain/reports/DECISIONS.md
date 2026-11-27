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

## 2026-11-24 — Corrected ei_balance_bifurcation_map.pdf to match the equation-level analysis above

Re-derived the whole phase diagram using `jr_parameters_explainer.equilibria()` (exact fixed
points + eigenvalues, immune to integration-step error) in place of the original Euler-simulation
classification, on the same (p, m) grid. Rebuilt Figure 1 and rewrote the Verdicts and section 3
("Is the limit cycle a trap?") to match.

Confirmed on the full 285-point grid: of the 20 points the old method called "limit cycle", 17
are genuinely confirmed (no stable fixed point exists at all) and only 2 (both at m~1, p=255-270
-- exactly the point used for the original hysteresis test) are the Euler-step artifact. The
bigger correction is a 27-point **bistable (fold) region** the simulation-only method could never
see (a cold start always lands on the quiet branch even where a stable active branch also
coexists) -- this was previously silently folded into "quiet". The corrected "active, stable"
region is larger (134 vs ~80 points), especially for E-dominant m (45->99 points), which makes
the single-pulse-vs-RICH identifiability gap even more extreme than first reported (E-dominant
SP1000 CRB median now ~265,000% / ~29,700%, not ~2,800% / ~3,300% -- the earlier number
under-sampled the true active region).

The three backprop-validation points (balanced p=150, E-dominant p=105, I-dominant p=240) were
individually re-checked with `equilibria()` and are all genuinely stable, undamped active
states -- no correction needed to Table 1 or to the recommendation in
`single_node_conclusions.pdf` section 6 (Laplace calibration, also run at p=150).

**Withdrawn finding:** section 3's original "no hysteresis" result (ramping down from m=1,
p=270) tested a point now known to be numerically stable in the true equations -- there was no
real limit cycle there to escape from. **Replacement finding (real, confirmed):** genuine
hysteresis exists at the quiet/active fold itself (p in [36.5, 114.6) for the balanced case,
bistable from p=0 for E-dominant columns) -- a different mechanism than the one originally
tested, and not previously known to this study.

Not pursued further this session: extending `equilibria()` to arbitrary (not just m-axis)
parameter draws, so the T2/T3-style independently-perturbed teachers (whose oscillation was only
checked by long simulation, not analytically) can be placed on the same exact map.

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

## 2026-11-24 — Network study plan written; Part N0/N1 (coupling fix + Fisher sweep) executed

Wrote `docs/network_study_plan.txt` (format matches `single_node_study_plan.txt`), after reading
`core/fitting.py`, `scripts/identifiability.py`, `scripts/recover_clean.py` directly and finding a
parallel effort had already built most of the harness this plan originally proposed building from
scratch — revised the plan's inventory (section 2) to reflect that before writing the rest of it.

**N0.1/N0.2 (coupling fix + tiny-scale sanity check): done.** Added `coupling_mode` to
`core/diffsim.py` (`"laplacian"` default, unchanged; new `"non_diffusive"`: neighbours send a
firing rate `S(E_j)`, always >=0, so mean drive grows with g and in-degree, instead of the
zero-mean Laplacian). 3-node scan confirms it works as intended (g=0 -> -1.861 mV rest baseline;
g=200 -> +5.978 to +5.996 mV, at/above v0). 10-node scan at nominal g=1000.01: fully engaged
(6.023 mV), zero oscillation amplitude at every tested g (180-400) over 9 s — no bistability/
oscillation risk introduced by the fix itself, for this teacher.

**N0.3 (thread coupling_mode through identifiability.py/recover_clean.py): done, and caught a real
bug before it became a false network-level finding.** Nominal g=1000.01 is calibrated for the
self-cancelling Laplacian; under `non_diffusive` coupling at that same g, `g*coupling` reaches
2000-5000 per node (confirmed by direct computation, not inferred), which saturates
`core/diffsim.py`'s `U_BOUND=500` safety clip (`tanh(rE/U_BOUND)*U_BOUND`) and kills gradients.
**Decision: added `--g-override` to both scripts rather than silently using each teacher's own
drawn g**, and swept it (300/200/150/120/100/70/50) to characterize the actual operating range —
exactly the kind of harness bug this plan's own risk list (section 8) said N0.3 existed to catch.
The ensemble wrapper (>=15-20 restarts, nominal-centered) from N0.3's task list is **not yet
built** — flagged as the next thing to add if the N2 single-start pilot below looks promising.

**N1 (Fisher/CRB sweep, 10 nodes, tier S1): done, gate passed comfortably.** Full CRB table (8
sites x 3 amplitudes protocol) is in `docs/network_study_plan.txt` section 4. Three findings:

1. non_diffusive at the teacher's own nominal g (1000.01) is catastrophic (g/c1/c2/mu blow out to
   35-1400%) — this is the U_BOUND bug above, not a property of the coupling idea.
2. Once g is in a non-saturating range, non_diffusive has a clean interior sweet spot at
   g~=100-120: every parameter under ~2% (mu=0.93%, the long-standing Phase-1 blind spot, and g
   itself=0.68%, both excellent). Degrades in opposite directions either side (g>=150:
   re-saturating; g<=70: under-engaged, c3/c4 hit hardest).
3. **Unexpected, and the finding that most changes this plan's framing:** the existing Laplacian
   (rest, zero mean drive — what Phase 1 actually used) is ALREADY excellent at network scale for
   every parameter (worst case c4=2.7%, including mu=0.003% and g=0.0033%) — unlike the
   single-node study, where rest-state c1-c4 bounds were 56-250% with one pulse. Likely cause:
   pooling 24 largely-independent site x amplitude conditions constrains shared GLOBAL parameters
   even without any single node individually engaged. **This does not mean Phase 1 should have
   succeeded at rest — it demonstrably failed, with this same Laplacian coupling, using neurofe's
   real JansenRit.fit.** The CRB-says-rest-is-fine vs Phase-1-backprop-got-stuck gap is now the
   single most important open question, and is exactly why N2 (below) runs both coupling modes
   side by side rather than assuming the coupling fix is the fix that matters. Candidate
   non-coupling explanations the single-node study independently showed matter and that N0.3's
   harness fixes regardless of coupling mode: raw (non-log) parameterisation, integer delays,
   single-pulse/single-site stimulus, single point estimate with no restarts.

**N2 pilot (backprop recovery, single start, both coupling modes) launched this session, same
teacher/sites/amplitudes as N1** — direct background processes (`nohup`), not `Workflow`, after
this session's earlier experience of `Workflow` agents' turns ending before a detached background
job actually finished.

**N2 pilot result (one teacher, single nominal-centered start, 300 iterations, full 8-site x
3-amplitude design used for training — this is a single-teacher pilot, NOT yet the >=8-teacher
N2 gate in section 5):**

  Laplacian (rest coupling): loss drops cleanly from 0.11 to 5.5e-6. Recovery at iter 299: a
  +0.1%, b -0.2%, g +0.6%, c3 +2.2%, c4 -3.7%, mu -0.2%, k -0.0% — all excellent, including g
  and mu. c1*c2 = -16.5% error, c3*c4 = -1.6% error (both under the 20% single-node target,
  individual c1/c2 are off by design — only their product is identifiable). Train rel. RMSE
  0.0036 (from a start of 0.383).

  Non_diffusive, g=100 (the CRB "sweet spot" from N1): loss plateaus at ~7.5e-3 by iteration 60
  and never moves again — training rel. RMSE only reaches 0.128 (vs Laplacian's 0.0036, ~35x
  worse). Recovery at iter 299: g +114%, c1 +141%, c2 +168%, c3 -44.6%, c4 +84.4%, mu -33%.
  c1*c2 = +547% error (catastrophic); c3*c4 happens to land at +2.1% (close, likely because the
  fit drifted along the known c3/c4 degenerate direction rather than actually constraining it).

**This directly confirms the CRB-vs-backprop gap N1 flagged, and resolves the open question in
the opposite direction from what motivated this coupling fix in the first place:**

- The Laplacian (rest) coupling — Phase 1's own, unchanged coupling — recovers essentially
  everything from a SINGLE nominal-centered start, no ensemble needed, once the other N0.3
  harness fixes are in place (log-scaled parameters, fractional delays, the rich 8-site x
  3-amplitude design). This strongly suggests Phase 1's actual failure was never about missing
  mean drive — it was the four OTHER things the single-node study already found matter (raw
  parameterisation, integer delays, single-pulse/single-site stimulus, no restarts/ensemble),
  all of which this pilot already fixes and none of which touch the coupling term.
- Non_diffusive coupling, even at its CRB-favoured operating point, gets stuck in a bad basin
  from a single start and does not recover — a textbook Fisher-good-but-landscape-bad failure,
  the same pattern single-node E9-E12 diagnosed for c1*c2 at a bad operating point, now appearing
  for the coupling term itself. The missing N0.3 ensemble/restart wrapper is the natural next
  check (does one of >=15 nominal-centered restarts escape this basin) before concluding
  non_diffusive coupling is unusable, not just unlucky here.

**Decision: do not adopt the non-diffusive coupling fix as a prerequisite for the network study.**
The existing Laplacian coupling, combined with the harness fixes already built this session
(log-scale params, fractional delays via `core/diffsim.py`, rich multi-site/multi-amplitude
design), already clears a single-teacher pilot version of the N2 bar. Proceed to the real N2 gate
(>=8 independent teachers, Laplacian coupling, still a single start per teacher as a first pass)
before adding an ensemble wrapper or revisiting non_diffusive coupling — both are now
lower-priority than this result initially suggested.

## 2026-11-24 — Ran the real 8-teacher N2 gate with a held-out split; FAILS in substance despite a technical pass

Trained on 4 of 8 sites x ONE amplitude (1000) — chosen to leave a genuine held-out set (those 4
sites at 500/2000, plus the other 4 sites at all 3 amplitudes) — 8 independently-drawn teachers
(seed 0-7), Laplacian coupling, single nominal-centered start each, 300 iterations, same as the
single-teacher pilot otherwise. Full numbers and root-cause analysis in
`docs/network_study_plan.txt` Part N2's "8-TEACHER N2 GATE RESULT" block.

**By the letter of the pre-registered pass criteria (median |error| <20% for k, a, c3*c4; mu shows
positive progress): technically PASSES** (a=0.8%, k=1.2%, c3*c4=7.6%, mu progress positive for all
8). **Flagging this as misleading rather than reporting a clean pass** (standing adversarial-
self-review rule): only 2 of 8 teachers (seed 0, seed 5) actually converged to a good EEG fit at
all — the other 6 plateaued at a loss 10,000-100,000x worse, with `g` off by 23-111% and `c1*c2`
off by 55-470%, and held-out RMSE tracked training RMSE closely (median 0.184 vs 0.180) — so this
is NOT a generalisation gap, it is most teachers simply failing to fit the training data.
**Verdict: FAIL in substance.**

**Root cause, diagnosed not just observed:** the held-out split I chose cut training richness from
24 conditions (the successful pilot's 8 sites x 3 amplitudes) down to 4 (4 sites x 1 amplitude) —
a much bigger cut than intended, made in service of creating a held-out set. This reproduces the
single-node study's own earlier finding almost exactly (`single_node_conclusions.pdf` section 5):
with a low-diversity stimulus, loss does not predict accuracy and the optimizer gets stuck in bad
basins; a RICH multi-amplitude design was what fixed it there. The one teacher that converged
perfectly here (seed 0) is the SAME teacher the single-teacher pilot used (seed 0), where it also
converged perfectly with the FULL 24-condition design — consistent with "this teacher has an
easy-to-find basin even with little data; most do not."

**Decision: redo the held-out split rather than reach for the ensemble/restart wrapper next.** The
wrapper fixes "stuck in a bad basin"; this result looks more like "not enough information to find
any basin" for most teachers, which an ensemble restart cannot generally fix (restarting 15-20
times from the same impoverished data just finds many different bad basins). New design: train on
4 of 8 sites x ALL 3 amplitudes (12 conditions, same per-site richness that worked in the pilot,
now on half the sites), hold out the other 4 sites entirely — isolates "does training on half the
anatomy generalise to the rest" from "is 1 amplitude enough signal to fit at all" (demonstrated
here that it is not, for 6/8 teachers), which this first attempt accidentally conflated.

## 2026-11-24 — Reran the N2 gate with 3x training richness: aggregate pass/fail unchanged, but found the real governing factor (per-teacher stimulus gain, not protocol richness)

Same 8 teachers, same coupling/start/iteration count, training richness tripled (4 sites x 1
amplitude -> 4 sites x all 3 amplitudes, other 4 sites held out entirely). Full numbers in
`docs/network_study_plan.txt` Part N2's "RERUN RESULT" block.

**Result: essentially no change in the aggregate picture.** Median |rel_error| barely moved
(g 33.3%->30.3%, c1*c2 60.0%->56.5%, c3*c4 7.6%->6.0%, a/k/mu still <5%) despite 3x the training
data. Per-teacher final loss improved for 4/8 teachers, got WORSE for 1/8 (t6), and the two
clearly-good teachers (t0, t5) and two clearly-bad ones (t1, t3) stayed exactly as good/bad as
before. **Conclusion: training richness (how many sites/amplitudes) is not the dominant factor —
something about specific teachers is.**

**Found the real factor by checking what the three consistently-bad teachers (t1, t3, t6) have in
common across BOTH designs: their drawn stimulus gain `k` is 45-50% below nominal** (log-offset
-0.66, -0.57, -0.56 — the three most extreme negative k draws of all 8, by a wide margin; next
closest is -0.10). Spearman correlation of k's log-offset against log10(final training loss):
rho=-0.79 (p=0.02) in the first design, rho=-0.95 (p=0.0003) in the second — strong, and gets
STRONGER with more training data, ruling out "not enough data" as the explanation for the
correlation itself. The SIGN matters (weaker-than-nominal k correlates with failure), not just the
magnitude of the k deviation.

**Mechanism: this is the same structural problem the single-node study already diagnosed, showing
up through a different nuisance parameter.** The 2026-11-22 "Mechanism behind the c1.c2/c3.c4
recovery failure" entry found that a single fixed tonic drive `p` does not put every single-node
teacher in the same dynamical regime. Here, the network pilot's stimulus is pulsed (not tonic), so
the equivalent nuisance parameter is the teacher's own stimulus GAIN `k`: the same fixed pulse
amplitudes (500/1000/2000) produce a weaker effective drive for a low-k teacher, plausibly leaving
that teacher's network less engaged for this fixed protocol — structurally identical diagnosis,
different knob.

**Decision: do not reach for the ensemble/restart wrapper yet — that is the right fix for "stuck in
one of many basins," and this looks more like "wrong regime for this teacher's own gain," a
different failure mode requiring a different fix** (same distinction the single-node study made
between a protocol-design fix and a landscape/optimizer fix). Next experiment (not yet run):
calibrate stimulus amplitude per teacher (e.g. scale amps by nominal_k / teacher_k, or screen each
teacher for an engaged operating amplitude before fitting) rather than reusing one fixed amplitude
set for every random draw, mirroring the single-node per-teacher-p calibration recommendation from
2026-11-22.

## 2026-11-24 — Per-teacher amplitude calibration: partially confirms the k-mismatch diagnosis, finds a new asymmetry, and surfaces a probably-pathological teacher

Added `--amp-scale-by-k` to `scripts/recover_clean.py`: every stimulus amplitude is multiplied by
`NOMINAL_EFF["k"] / teacher["k"]`, so a below-nominal-gain teacher gets a stronger pulse and an
above-nominal-gain teacher gets a weaker one (symmetric compensation). Reran the same 8 teachers,
same design otherwise (4 of 8 sites x all 3 amplitudes, other 4 held out). Full numbers in
`docs/network_study_plan.txt` Part N2's "PER-TEACHER AMPLITUDE CALIBRATION RESULT" block.

**Aggregate improved — best of the three designs tried**: median train rel. RMSE 0.135 (vs 0.180,
0.195), held-out 0.103 (vs 0.184, 0.145), `c1*c2` median |error| 41.6% (vs 60.0%, 56.5%). But three
distinct, more specific findings sit underneath that aggregate improvement, not one clean story:

1. **The k-hypothesis is directly confirmed for the two most extreme cases.** t1 (k 66% below
   nominal, amplitude boosted 1.94x) and t3 (k 57% below nominal, boosted 1.77x) both went from
   clearly stuck (loss 0.071, 0.056) to clearly converged (0.0031, 0.0067) — a clean, large effect
   exactly where the diagnosis predicted it.
2. **Symmetric compensation has a real cost on the other side.** t5 (k 40% ABOVE nominal, amplitude
   REDUCED to 0.67x of baseline) regressed 4x (loss 0.0010->0.0041), and t7 (mildly below-nominal
   k, amplitude only slightly boosted) also regressed (0.0069->0.024) despite getting more, not
   less, amplitude — both losing information relative to the fixed-amplitude baseline. **The
   compensation should only ever increase amplitude, never decrease it below baseline** — reducing
   drive for an already-adequately-driven teacher throws away signal it was using, even though
   that teacher's own k "says" it needs less.
3. **t6 breaks the simple story and needs its own diagnosis.** It has almost the same below-nominal
   k as t3 (-0.56 vs -0.57) and got almost the same amplitude boost (1.75x vs 1.77x) — yet t3 was
   fixed and t6 was not. More strikingly, t6's loss got MONOTONICALLY WORSE across all three
   designs tried this session regardless of what changed (0.018 -> 0.031 -> 0.040), independent of
   both training richness and amplitude scale. This looks like the network-scale analogue of the
   single-node study's T2/T3 pathological teachers (2026-11-22 entry) — a draw with no good
   operating point for THIS stimulus family, not a calibration problem amplitude can fix.

**Decision: two follow-ups, neither run yet.** (a) Clamp `amp_scale` at a maximum of 1.0
(`max(1, nominal_k/teacher_k)`) so weak-k teachers still get boosted but strong-k teachers are
never driven below the fixed baseline — should recover t5/t7 without losing t1/t3's fix, if the
"reduction causes harm" diagnosis is right. (b) Run a direct stability/engagement check on teacher
6 specifically (the `equilibria()`-style two-window method from `ei_balance_bifurcation_map.pdf`,
not another protocol tweak) to test whether it has any good operating point at all, the same
diagnostic step that identified T2 as structurally unfixable in the single-node study rather than
a target for more stimulus engineering.

## 2026-11-25 — Teacher-6 diagnosis: not uniquely pathological, and the whole per-teacher-amplitude investigation turns out to be a CRB-good-but-backprop-bad problem

Ran both follow-ups from the previous entry. Full numbers in `docs/network_study_plan.txt` Part
N2's "TEACHER-6 DIAGNOSIS" block.

**Stability check (two-window, teachers 0/3/6, no stimulation and 12 site/amplitude pulse
conditions each) overturns the working hypothesis.** Teacher 0 decays cleanly everywhere. But
teachers 3 AND 6 both show slow/non-decaying residual dynamics — several conditions for BOTH
teachers have late-window amplitude >= 90% of early-window (some actually growing), and teacher
3's no-stimulus baseline residual (0.451 mV after 5s) is larger than teacher 6's (0.237 mV).
**This directly contradicts "teacher 6 is uniquely pathological, teacher 3 is not"** — by this
measure teacher 3, which the amplitude fix cleanly repaired, is the more dynamically sloppy of the
two.

**Fisher/CRB comparison (`identifiability.py --teacher 3` and `--teacher 6`, Laplacian coupling)
changes the diagnosis again, in a cleaner direction.** Teacher 3's CRB is excellent at every
protocol, including the sparsest single-site design (c1=0.63%, c2=0.003%, c4=9.9% — all comfortably
under 20% with ONE stimulation site, no amplitude boost needed for information reasons). Teacher
6's CRB is worse at sparse protocols (1 site: c1=24%, c2=15%, c4=18%) but still clears 20%
comfortably at the actual training design's richness (8 sites x 3 amplitudes: everything <3%).

**Reinterpreting the whole per-teacher-amplitude investigation in light of this:** teacher 3's CRB
was already excellent BEFORE any amplitude boost — so the amp-scale fix did not supply missing
information, it fixed an optimisation problem (a weak pulse gives Adam too small a gradient signal
to escape a bad start, even though the true answer was always locatable in principle). Teacher 6's
CRB, while weaker at sparse protocols, is still good enough at the training design actually used —
yet real backprop (even with its own amplitude boost applied) remains stuck far worse than CRB
predicts. **Both teacher 3 (originally) and teacher 6 (still) are CRB-good-but-backprop-bad cases —
the single most recurring failure mode in this entire study, at every scale tried so far — not
protocol/information problems solvable by per-teacher amplitude tuning.**

**Decision: stop treating per-teacher amplitude calibration as the primary fix; build and test the
N0.3 ensemble/restart wrapper next.** It is now motivated by both teacher 3's original failure
(before the amp fix) and teacher 6's persistent one, not a secondary nice-to-have — this directly
mirrors the single-node study's own validated recipe (RICH protocol + >=15-20 restarts +
lowest-loss selection, `single_node_conclusions.pdf` sections 5-6), which this network study has
built everything around except the restart wrapper itself. Keep the amp-scale clamp as a
complementary, low-cost fix (it did not hurt already-good teachers once clamped) — the two fixes
address different mechanisms and should be used together, not treated as alternatives.

## 2026-11-25 — Built and pilot-tested the ensemble/restart wrapper: confirmed as a real fix, not just a hypothesis

Added `scripts/recover_ensemble.py` — a lightweight aggregator, not a new simulation path: it
reads N independent `recover_clean.py` runs for the same teacher that differ only in
`--init-seed`, and selects the one with the lowest TRAINING loss (never held-out — that would be
selecting on data you would not have for real). This is exactly the recipe
`single_node_conclusions.pdf` sections 5-6 validated at the single-node level, ported to network
scale for the first time. Full numbers in `docs/network_study_plan.txt` Part N2's "ENSEMBLE/RESTART
WRAPPER" block.

**Pilot, compute-bounded per the standing rule** (full 8-teacher x >=15-restart x 300-iteration
would be many hours — flagged rather than just run): targeted only the 3 teachers still failing
after the amplitude clamp (t4, t6, t7), 2-3 random restarts each, 150 iterations.

**Teacher 6 (the single worst teacher of all 8, loss 0.0402): a different random start reaches loss
0.0017 — a 24x improvement**, and dramatically better than its own other two restarts (0.064,
0.085) — directly confirming this was a stuck-basin problem, not an information limit. `c3*c4` goes
from badly wrong to +6.9% (now passes the 20% bar); `c1*c2` improves to -23.4% (close). **Teacher 7:
best of 3 restarts reaches loss 0.0054 (was 0.0237, 4.4x better)** — clearly improved, not yet a
clean fit (`c3*c4`/`c1*c2` still miss the bar). **Teacher 4: only 2 restarts (smaller pilot), modest
~25% loss improvement but individual parameters moved in a DIFFERENT (not clearly better)
direction** — looks like a different point along the same (b, c1, c2, c4) sloppy-direction
trade-off this study already documented for the single node (2026-11-22 "Mechanism behind the
c1.c2/c3.c4 recovery failure" entry) rather than a clean fix; inconclusive with so few restarts.

**Verdict: the ensemble/restart wrapper is a validated, working fix, not a hypothesis anymore.** It
took the worst-performing teacher in the whole N2 gate from the bottom of the pack to a
near-passing fit using just 3 cheap, short restarts. This closes the loop opened by the
teacher-6/teacher-3 diagnosis entry above: CRB-good-but-backprop-bad was the right call, and
restarts — not more protocol engineering — is the right fix.

**Not yet done, flagged rather than auto-run (compute-budget rule):** a full 8-teacher N2 gate with
the ensemble wrapper as standard (proper ensemble size, ~8-10 restarts, full 300 iterations for a
polished number) is the natural next step to get the real N2 verdict, but is several hours of
wall-clock even at full 8-way parallelism (this 150-iteration, 2-3-restart pilot alone took ~40
minutes) — an explicit decision point, not an automatic continuation.

## 2026-11-26 — Full 8-teacher N2 gate, final verdict: PASS on the pre-registered criteria, with two caveats flagged rather than hidden

William asked to run it. 5-restart ensemble per teacher (reused single-start + 4 fresh random
restarts), full 300 iterations, same design as the prior attempts (Laplacian, amp-scale clamp, 4 of
8 sites x 3 amplitudes training, other 4 held out). ~5 hours wall-clock across 4 sequential
8-way-parallel batches. Full numbers in `docs/network_study_plan.txt` Part N2's "FULL N2 GATE,
FINAL VERDICT" block.

**Against the plan's own pre-registered criteria: PASS, by a comfortable margin, not a borderline
call.** Median |rel_error|: a=0.3%, k=0.7%, c3*c4=4.2% (all well under the 20% bar); mu progress
positive in all 8 teachers (median +1.01, essentially full recovery — the parameter that had zero
gradient in neurofe's real Phase 1 fit is now reliably recovered). c1*c2, not a named gate
criterion, also comes in at 12.0% — better than any single-node operating point tested in this
whole project.

**Two things the gate does not cover, flagged rather than left implicit:**
1. `g` fails badly (median |error| 27.1%, systematically under-estimated in 6/8 teachers by
   21-65%, "progress" scores ranging -4.85 to +23.88 — not noise around a right answer). It was
   never one of the three named gate parameters (the criteria were scoped to single-node-
   identifiable quantities before this session's g behavior was known), but "PASS" here certifies
   loop-gain and conduction-delay recovery, not network-coupling-strength recovery — that
   distinction must travel with the verdict, not get lost in a one-word summary.
2. No Laplace/credible-interval coverage check was run at network scale (the fuller N2 design also
   asked for this, mirroring `laplace_calibration.py`'s single-node validation). This result is
   point estimates only — accurate on average, but with no check on whether the model is over- or
   under-confident about that accuracy.

Decision: report this as PASS with both caveats attached, not as an unqualified pass — consistent
with the standing adversarial-self-review rule and with how every other milestone in this study has
been reported.

## 2026-11-26 — Holistic review after the N2 pass: diagnose g's bias before N3 scale-up or network-scale Laplace calibration, not either of those next

Reviewed the whole study (CLAUDE.md, both study plans, the full decision log, README, and what's
actually on disk in `results/`) to decide the next concrete step now that N2 has passed with the two
flagged caveats (g fails badly; no calibration check run). Did not run any new training — did two
free analyses of data already on disk, which changed the shape of the decision.

**Analysis 1 (zero new compute): the already-computed nominal-teacher Fisher/CRB data
(`results/identifiability/nominal.json`, "8 sites, 3 amplitudes" protocol) rules out "g is a sloppy/
degenerate direction" as the explanation for its N2 failure.** g's CRB there is 0.0033% — the single
best-identified parameter in the whole table. Its correlations with every other parameter are modest
(|corr| 0.02–0.54, strongest with c2 at -0.54 and mu at +0.48 — not a near-1.0 degeneracy), and it
loads at ~0 (5.98e-5, 5.66e-4) on both of the two softest eigendirections (which are instead
dominated by (c1, c2, c3, c4), matching the single-node study's own (b, c1, c2, c4) entanglement
finding). **Locally, around the true value, g is one of the best-constrained parameters in the model
— the N2 failure is not an information problem.**

**Analysis 2 (zero new compute): directly inspecting the 8 already-saved N2-gate ensemble results
(`results/recovery_ensemble/*_N2full_ens_t*.json`) shows a clear, non-random directional pattern, not
scatter around the right answer.** Sorting teachers by true g: t0=523, t1=586, t6=750 (the three
LOWEST true g) all converge close to truth (fitted 523/556/594, errors 0%/-5%/-21%). t3=1206, t7=1172,
t2=1223, t4=1461 (the four HIGHEST true g) all converge to fitted values clustered in a much narrower,
much lower band (622/482/813/505) **regardless of which direction their own restart started from** —
e.g. t4 started at 1219 (already close to its truth of 1461) and was pulled DOWN to 505, while t3
started at 616 (needing to move UP to reach 1206) and only crept up to 622. t5=881 (moderate) is the
one exception, overshooting from 968 to 1062 instead of moving down to 881. This looks like fits with
a high true g are being pulled toward a g~500-800 attractor band regardless of where they start, not
like noisy estimation around the correct value.

**Mechanism hypothesis, grounded in a bug this exact study already found and fixed once (N0.3):**
`core/diffsim.py`'s `step_ms` computes `rE = noise + g * coupling + c2 * E_s`, then clips with
`rE_bd = U_BOUND * tanh(rE / U_BOUND)`, `U_BOUND = 500`. N0.3 already proved this clip silently kills
gradients once its argument routinely exceeds ~500 (that is exactly why non_diffusive coupling at the
teacher's nominal g=1000 was "catastrophic" rather than just suboptimal). Laplacian `coupling` is
zero-mean at rest, so this was not suspected for Laplacian — but during a stimulation pulse (the only
time g's gradient is actually exercised), `g * coupling` for a HIGH true g is exactly the kind of term
that could intermittently push `rE` past the clip for some nodes/timesteps, attenuating d(rE_bd)/dg
locally in a way that would look precisely like "gradient descent prefers a lower g than the truth
whenever the truth is high" — matching the observed pattern. This has not been checked yet (would
need to replay the already-fit teacher parameter sets through the simulator and log per-node/
per-step `rE` during the pulse for the 4 high-g teachers vs. the 3 that converged cleanly); it is a
specific, falsifiable hypothesis, not a certainty, but it is the most parsimonious explanation given
everything else in this study's history of this same clip causing this same class of symptom.

**Decision: the next concrete step is a short, cheap diagnostic of g's systematic bias — not Part N3
(scale up) and not the missing network-scale Laplace/calibration check, both of which were the other
candidates on the table.** Reasoning:
- Scaling up (N3.a/b/c) while a central, already-flagged, structurally-unexplained bias sits in the
  one parameter that controls the whole network's coupling strength risks compounding an unresolved
  problem with a new one (more nodes or more tiers), exactly the confound the plan's own N3.a/N3.b
  "don't combine" rule exists to avoid, just one level up.
- Building the network Laplace/calibration wrapper now would calibrate a credible interval around a
  point estimate already known to be systematically biased for a third of the teachers tested — the
  single-node study's own experience (the winner's-curse bias entry, 2026-11-24) says a calibration
  result obtained on top of a known, unexplained bias is not trustworthy and would need re-doing
  anyway once the bias is understood.
- The diagnostic itself is nearly free: two of the analyses above used only data already on disk; the
  remaining check (log `rE` against `U_BOUND` during replay of the 8 existing fits) requires no new
  training, only instrumented forward passes of parameter sets already fit — well inside the "many
  small runs" compute-budget rule, and a strict subset of an afternoon, not the multi-hour cost of a
  fresh N3 design or an 8-teacher x restarts Laplace calibration.
This is a judgment call the plan's own structure anticipated requiring ("something else you judge
more important"), not a stop-and-ask case: the N2 gate did not fail with no fallback (it passed, with
caveats), nothing here contradicts what a paper describes, and the choice of what to check next does
not itself change what was already concluded about a, k, c3*c4, or mu.

**If the U_BOUND-clip hypothesis is confirmed:** the fix is likely a higher clip bound or a softer
saturation function for the coupling term specifically (or capping teacher draws' g to a range that
provably stays under the clip during the stimulation protocols in use) — cheap to apply and re-test
on the same 8 teachers before deciding whether to re-open N2's verdict on g specifically.
**If it is NOT confirmed** (rE stays well under 500 even for the high-g teachers): the systematic
"low-g attractor" pattern is a genuine landscape/optimizer artifact independent of the clip, and the
next step would be a larger ensemble (10-15+ restarts, vs. N2's 5) targeted at exactly the 4 high-g
teachers only, to see whether more restarts find the true high-g basin at all or whether it is
unreachable from nominal-centered starts regardless of restart count (the latter would be a genuine,
reportable network-scale finding about g, not an optimizer tuning problem).
Either branch is cheap, decided by the same instrumented-replay run, and should be resolved before
committing compute to N3 or the network Laplace check.

## 2026-11-26 — Adversarial review of the N2 verdict: corrected, not an unqualified pass

Spawned a fresh review agent (no inherited framing) specifically to try to break the "FULL N2 GATE,
FINAL VERDICT" entry above, per the standing adversarial-self-review rule. It independently
re-derived every headline number from the raw JSON files and confirmed the arithmetic is correct and
the pre-registration is genuine (checked via git diff of the "Pass criteria" subsection across
commits, plus result-file mtimes corroborating the narrated order of attempts). But it found real
problems with how the result was characterized, not with the numbers themselves:

1. **A factually false "bonus finding" is retracted.** The verdict claimed network-level c1*c2
   (12.0%) is "better recovered than at any single-node operating point tested." This is
   contradicted by this project's own 2026-11-23 entry (3.1%, one teacher, RICH + 20 restarts) and
   2026-11-24 entry (7.5% median, 24 teachers) -- both already on record, both lower (better) than
   12.0%. Struck from the verdict; `docs/network_study_plan.txt` updated with the correction and the
   real numbers for comparison.
2. **A third, previously-unlisted gate requirement was silently uncovered.** "No uncontrolled
   oscillation" is a named N2 pass criterion (section "PASS CRITERIA SUMMARY"), but every stability
   check this session was run on TEACHER parameters before fitting, never on the FITTED student
   parameters afterward -- this criterion has not actually been evaluated, which is different from
   "evaluated and passed." Added as an explicit open item, not assumed benign.
3. **The 5-restart ensemble is demonstrably too small**, evidenced by the data itself, not just a
   compute-budget argument: per-teacher loss across the 5 restarts spans 1-6 orders of magnitude
   (e.g. teacher 0: 1.5e-8 to 1.8e-2), far from a converged cluster, for 7 of 8 teachers. The
   reported "best" restart per teacher, especially for the weakest teachers (t4/t6/t7) and for g and
   c1*c2 generally, should be read as provisional, not a ceiling on what the method can do.
4. **A second, unflagged ground-truth leak**: `--amp-scale-by-k` sets delivered stimulus amplitude
   from the teacher's OWN TRUE k -- the same category of leak the 2026-11-24 entry explicitly caught
   and fixed for restart-centering (nominal-centered, not truth-centered), but missed here. Doesn't
   invalidate k's own recovery, but means this exact protocol isn't yet something a real experiment
   (which doesn't know a patient's true k in advance) could reproduce as-is.
5. Minor: a "6/8 teachers, 21-65%" quote for g's bias conflated sign (true for 6/8) with magnitude
   range (true for only 5/8) -- corrected.

**Decision: corrected `docs/network_study_plan.txt`'s final-verdict section in place** (not
rewritten from scratch -- the three pre-registered criteria genuinely pass and that stands) to add
all of the above, retract the false claim, and restate the verdict as "partial pass: single-node-
identifiable parameters survive at network scale; network-coupling-strength (g) does not, and two
gate-adjacent checks (oscillation stability on fitted networks, adequate restart count) remain
undone" rather than a bare "PASS." This does not change the next step already decided in the
holistic-review entry above (diagnose g's bias next, not N3 or Laplace calibration) -- if anything
it reinforces it, since an undersized ensemble and an unflagged leak are both reasons to treat
today's g numbers as preliminary, not as a settled negative result either.

## 2026-11-26 — g's U_BOUND-clip hypothesis: confirmed quantitatively, fix is a genuine design choice (stopping to ask)

Built `scripts/diagnose_g_clip.py` (instrumented reimplementation of `diffsim.py`'s inner loop,
no new training -- replays each N2-gate teacher's own TRUE parameters through the real training
protocol) to test the Part N2.5 hypothesis. First run gave implausible numbers (rE in the hundreds
of thousands); **verified against real `ds.run()`'s actual output EEG before trusting it** (sane,
max 30µV, no NaN) rather than either assuming a bug or assuming the hypothesis — the huge numbers
turned out to be a real, very brief (~10-20ms) transient spike right after the pulse, present for
EVERY teacher regardless of true g, which the clip successfully keeps the forward simulation
bounded through even though the instrumented pre-clip value is enormous.

**The real signal is in the settled post-transient window (150-400ms), not the transient**, and it
is unambiguous: computed the clip's local gradient-survival factor (sech^2(rE/U_BOUND): 1.0 =
undamped, 0.0 = dead) there for all 8 teachers.

```
teacher  true_g   grad-survival   g rel.error
t0        523        0.835          +0.0%
t1        586        0.686          -5.1%
t6        750        0.604         -20.8%
t5        881        0.551         +20.6%
t2       1223        0.385         -33.5%
t3       1206        0.377         -48.4%
t7       1172        0.427         -58.9%
t4       1461        0.424         -65.4%
```

Spearman true_g vs. grad-survival: rho=-0.90 (p=0.002). Grad-survival vs. |g error|: rho=-0.79
(p=0.02). True_g vs. |g error| directly: rho=0.88 (p=0.004). **This confirms the hypothesis with
real quantitative force, not just a plausible story**: g mechanically scales the term the clip
attenuates, the clip measurably kills more of the gradient as true g rises, and that attenuation
measurably predicts the recovery error -- a complete, falsifiable, now-verified causal chain.

**Decision: do not pick a fix and re-run the multi-hour N2 gate on a guess -- stopping to ask
William.** This is a genuine design choice, not an obvious bugfix: raising `U_BOUND` would reduce
gradient death for high-g teachers' settled dynamics, but it would also shrink the safety margin
that specifically prevented a MORE severe failure this same investigation already found (N0.3:
non_diffusive coupling's catastrophic blow-up at the teacher's nominal g=1000, caused by this exact
clip saturating during early, far-from-truth fitting iterations, when parameters can be far more
extreme than any individual teacher's own true value ever is). Trading "g biased low at convergence"
for "training occasionally diverges early" is not obviously a better trade without testing it, and
testing it costs real compute (at minimum a few teachers re-fit, likely the full gate again to be
sure). Per CLAUDE.md's stop-and-ask condition ("a required design choice is not specified and
different choices could change the conclusion"), reporting this finding and asking for direction
on how to proceed (e.g. a global U_BOUND increase with a stability check first, a targeted fix only
on the coupling term rather than globally, or accepting this as a genuine reported limitation of
the current model/clip combination at high network-coupling strength) rather than choosing alone.

**William chose "raise U_BOUND, test stability first." Result: the stability check PASSED (no
NaN/Inf for candidates 2000/5000/10000 across all 8 teachers + nominal start) but the gradient-
survival check that followed DISCONFIRMS this as a fix.** Raising U_BOUND meaningfully helps the
teachers that were already decent (t0 grad-survival 0.84->0.99, t1 0.69->0.95, t5 0.55->0.71,
t6 0.60->0.60 at U_BOUND=10000) but barely moves the four worst teachers at all (t2/t3/t4/t7 stay
at 0.37-0.44 even at 20x the original bound). Pushed further to find out why: measured the
settled-window tail coupling magnitude relative to whatever U_BOUND is set to, across U_BOUND =
500 / 5,000 / 50,000 / 500,000, for the four worst teachers:

```
                    ratio of tail mean|g*coupling| to U_BOUND
teacher   U=500    U=5,000   U=50,000   U=500,000
t2        38.6x      4.6x       3.3x        3.2x
t3        19.9x      4.8x       3.7x        3.8x
t4        12.5x      4.0x       3.4x        3.0x
t7        12.5x      3.6x       3.2x        2.7x
```

**The ratio converges to a constant ~3-4x and stops shrinking, across three orders of magnitude of
U_BOUND.** This means these four teachers' network dynamics do not have a bounded, non-saturating
equilibrium that a higher clip would eventually reach -- raising U_BOUND just lets the system
settle into a proportionally larger steady state, always a few times over whatever the clip is set
to. **U_BOUND is not an optional safety margin for these parameter draws; it is the only thing
preventing the Laplacian-coupled network from diverging outright at high g.** Raising it is
therefore DISCONFIRMED as a fix, not merely insufficient -- it cannot work by construction, proven
by direct test rather than argued from the mechanism alone.

Decision: report this back to William rather than proceeding with a fix just shown not to work.
The two live options are now: (a) investigate the underlying instability itself (why does this
specific coupling formulation have no bounded equilibrium above some g threshold -- is it anatomy-
/connectivity-specific, or a property of the Laplacian coupling term in general at this parameter
scale), which is a genuinely open research question, not a quick diagnostic; or (b) accept and
document this as a real, now rigorously characterized finding: this coupling formulation cannot
reliably recover network-coupling-strength g once it exceeds roughly the range where t0/t1/t6 sit
(~500-750), and proceed to N3/Laplace calibration with that limitation explicitly scoped (e.g.
restrict future teacher draws' g to the range this method can actually recover, rather than
treating the whole g axis as in-scope).

## 2026-11-26 — Tested non-diffusive coupling as a fix for g: it doesn't help, and breaks mu

William asked whether non-diffusive coupling (bounded firing-rate transmission between nodes,
structurally immune to Laplacian's "no bounded equilibrium at high g" instability -- see the
previous entry) was worth pursuing, after I explained the physiological argument for it (real
inter-areal connections are rate-limited spikes, not raw potentials). Confirmed the structural
claim first (clean decay, no NaN/Inf, stable bounded EEG across g=50-300 under non_diffusive) before
spending retraining compute on it. Full numbers in `docs/network_study_plan.txt` Part N2.5's
"NON-DIFFUSIVE COUPLING AS AN ALTERNATIVE" block.

**Ran the full 5-restart ensemble gate with non_diffusive coupling (teacher=student, g=100, same 8
teachers, same held-out design as the Laplacian N2 gate). Result: g recovery is WORSE, not better
(median |error| 58.4% vs Laplacian's 27.1%, with a +1351% outlier), and mu -- Laplacian's cleanest
win -- is newly broken** (median |error| 59.5%; mu's "progress" is negative, i.e. moved AWAY from
truth, for 4 of 8 teachers, versus unanimous positive progress under Laplacian). `c1*c2` degrades
from a passing 12.0% to a failing 75.2%. Train/held-out RMSE (0.081/0.083) is comparable to
Laplacian's (0.078/0.068) -- this is not a case of the student failing to fit the data; it fits
about as well while landing on substantially wrong parameters, meaning the loss landscape itself is
more degenerate under non_diffusive coupling, not just harder to optimize.

**Likely mechanism, flagged as an open question rather than asserted as confirmed:** Laplacian
coupling has a self-cancelling term (`dg @ E`, always pulling a node back toward its neighbours'
average) that non_diffusive coupling has no analogue of (purely additive/positive, no restoring
term). Losing that structural feature plausibly entangles g's network-wide drive with mu's
conduction-delay shaping of that same drive -- which would explain both symptoms (g worse, mu newly
broken) as one new degeneracy rather than two separate problems. Not yet directly verified.

**Decision: do not adopt non-diffusive coupling as a replacement for Laplacian.** The physiologically
more plausible choice is not the more identifiable one here -- it trades a clean, well-localized,
now precisely characterized failure (g only, only above ~750-1000 true g) for a diffuse one across
multiple parameters. Keep both coupling modes in the codebase (already supported via
`coupling_mode=`), with Laplacian as the default; non_diffusive is a secondary option for specific
questions (e.g. the mismatched-coupling robustness test below), not a fix.

**Still running, not yet concluded:** a mismatched-coupling test (non_diffusive teacher -- the more
physiologically plausible process -- fit by a Laplacian student, i.e. what neurofe's real framework
actually assumes) to test whether neurofe's existing tooling stays trustworthy for the
coupling-independent parameters (a, b, c1-c4, mu, k) even if real brains don't actually follow
neurofe's own coupling assumption. Added `--student-coupling-mode` to `recover_clean.py` (teacher
and student can now use different coupling modes; `g` has no shared ground truth across modes and
is excluded from that comparison's pass/fail judgment) and a matching fix to
`recover_ensemble.py`'s filename reconstruction. Result to follow in a separate entry.

## 2026-11-27 — Mismatched-coupling robustness result: neurofe's Laplacian framework is NOT robust to this kind of model misspecification

Full numbers in `docs/network_study_plan.txt` Part N2.5's "MISMATCHED-COUPLING ROBUSTNESS TEST,
RESULT" block. Same 8 teachers, non_diffusive-generated data (g=100), fit by a Laplacian student
(`--student-coupling-mode laplacian`), 5-restart ensemble, same held-out design as both matched runs.

**The mismatched condition is the worst of all three on every measure that matters, by a wide
margin.** `c1*c2` is essentially destroyed (98.1% median error, one teacher at +3146%); `c3*c4`
degrades to 54.4% — worse than EITHER matched condition (4.2% Laplacian, 9.0% non_diffusive). Train
and held-out RMSE are roughly double either matched run's (0.170/0.180 vs ~0.08/~0.08 for both) —
meaning this isn't only a parameter-recovery problem, a Laplacian student genuinely cannot reproduce
non_diffusive-generated dynamics as well as a matched model can, at any parameter setting. The
student's fitted "effective Laplacian g" clusters very low (9-40, one outlier at 204) — far below
any matched-Laplacian teacher's true g (500-1500), consistent with the student compensating for the
mismatch by using near-zero coupling and absorbing the error into badly wrong local loop gains.

**This is the single most practically important finding of the whole N2.5 investigation.** If real
cortical coupling is closer to the bounded, rate-based mechanism (the more physiologically grounded
assumption — see this session's earlier discussion of TMS afterdischarges and rate-limited synaptic
transmission), applying neurofe's actual Laplacian-only framework to real TMS-EEG data would not
just fail to recover network-coupling-strength (already known from the matched-Laplacian result) —
it would produce substantially WRONG E/I-balance (loop-gain) estimates, the exact quantity this
entire study (single-node and network phases both) treats as the primary scientific target. This is
a limitation of neurofe's modeling assumption itself, not of this identifiability study's method.

**Decision: do not switch coupling modes** (both matched-non_diffusive and mismatched are worse than
matched-Laplacian on every measure) — **treat this as a documented, now-quantified caveat that must
travel with every future result this framework produces**: loop-gain recovery is reliable only to
the extent Laplacian coupling is itself a reasonable approximation of the true coupling mechanism,
which has not been validated against real brain connectivity data. This is a finding to carry into
`executive_summary.pdf`/`README.md`, not a blocker — proceed with Laplacian-based N3 work, which
this full three-way comparison continues to support as the best available choice, but never present
a recovered c1*c2/c3*c4 without this caveat attached.

**This closes the N2.5 coupling-mode investigation** (g-clip diagnosis -> U_BOUND disconfirmation ->
non-diffusive alternative tested and rejected -> mismatch robustness tested and found wanting).
Recommended next step: proceed to Part N3 (scale-up) with Laplacian coupling as the established
default, g's recovery explicitly scoped to the range already characterized as reliable (~500-750),
and this session's coupling-misspecification caveat carried forward into any future reporting of
loop-gain results.
## 2026-11-26 - CDC margins paper, S1 (theory) advanced

- **Correction to control_formulation.txt section 1.1 (not edited; the file is read-only
  for the paper workflow):** in node.py, neurofe/models/jansen_rit.py and core/diffsim.py,
  the single-node TMS input k*u and the tonic drive p are added *outside* the tanh rate
  bound: rE = U tanh(c2 S(c1 P)/U) + k u + p. The note says the input enters inside, which
  is wrong. In the network code, coupling and process noise do enter inside the tanh.
  kappa = 5e-4 ||lm|| k A / a.
- Stimuli with to_i > 0 (EI1000, part of RICH) also drive the I channel, so the
  small-signal response is a 1x2 transfer. The margins and the identifiability lemma
  (eta = (kappa, a, b, lamE, lamI) is generically identifiable from T(s); needs a != b,
  which is sharp) are unaffected. The S2 and S4 code must model the second input.
- At the three active study points m_F is between 1.9 and 4.5 (far from the fold) and
  m_M is between 0.09 and 0.59. **m_M is the primary margin for the paper.** H1 (CRB of
  m_F <= 10%) is kept unchanged but is noted as lenient; S4 will report the SE of m_M
  with equal prominence.
- At the E-dominant point, lamE = 2.5e-4 (a pole almost cancels the zero at -a): lamE is
  structurally identifiable but not in practice. This is consistent with the earlier
  "E-dominant collapse" result.
- Artifacts: syn_brain/paper/cdc_margins/theory/{theory.txt, check_symbolic.py};
  steps/S1/round1_{worker,review,decision}.txt.

## 2026-11-26 - CDC margins paper, S2 (numerical core) worker round 1

- **Stability by margins = m_F > 0 AND Nyquist winding Z = 0**, not "m_F > 0 and m_M > 0".
  m_M > 0 also holds at unstable equilibria (theory.txt F3), so the literal test disagrees
  with the eigenvalues at every grid point where the active state is unstable (92.4%
  agreement). code/margins.py reports a signed m_M (negative if Z != 0). With that, S2(b)
  agrees with the eigenvalues at 100% of the 3724 grid points (4436 equilibria). This is how
  plan.txt S2(b) is read; the threshold is not changed.
- **S2(c), "m_M -> 0", was made operational before running** as m_M < 1e-3 at the
  bisected boundary (segment tolerance 1e-10 of a cell). At a fold, the critical equilibria
  are the colliding pair, whatever their stability: 19 of the 56 folds create a saddle plus
  an unstable pair, and no stable state changes there. The first run used "min m_M over
  stable equilibria", which wrongly marked those 19 folds as failures. Explainer class edges
  that are not bifurcations (quiet|active by the v >= 2.5 threshold: 26 edges; active|Euler-only
  oscillation: 7 edges) are excluded and listed.
- The lower-p edge of the explainer's "bistable" region is a Hopf of the upper branch, not a
  fold. There m_F is 0.46-3.1 while m_M -> 0. This is more support for m_M as the primary
  margin.
- **node.py note for S4:** with SETTLE_MS_DEFAULT = 1500, the balanced point (m=1, p=150;
  damping 2.05 /s under Euler) is not fully settled. v is still 0.17 mV off the equilibrium
  at the pulse and keeps ringing, which adds about 2% to node.py's small-pulse error there.
  In EEG units this is about 0.02 uV at gain 250, below the 0.42 uV noise. S4 teachers near
  the Hopf set should start at the equilibrium or settle for longer.
- Artifacts: syn_brain/paper/cdc_margins/{code/margins.py, experiments/s2_validate.py,
  results/s2_*.json, figures/s2_*.png, steps/S2/round1_worker.txt}.

## 2026-11-26 - CDC margins paper, S2 decision (round 1): ADVANCE

- S2 (a), (b), (c) pass; checked independently by the reviewer. Thresholds unchanged.
- **Ruling on S2(b):** "m_M > 0" means the signed margin (m_F > 0 and Nyquist count Z = 0),
  as the S1 decision required. The literal unsigned reading (92.4%) stays on record. The
  paper must not claim that m_M > 0 alone certifies stability, and the 100% agreement is a
  code-consistency check, not a result.
- S2(c) accepted, with its pre-fixed operational threshold m_M < 1e-3 and the corrected
  critical-equilibrium metric (colliding pair at folds), which was logged as post hoc.
- **Top-2 risk for S7:** the real SP250/SP1000 pulses are strongly nonlinear (30-96% off
  the T(s) response), so the linear identifiability lemma does not by itself cover the
  actual protocols. S4 must state that it works in the nonlinear regime (and, if cheap,
  add a small-amplitude arm), and the paper must state the small-signal assumption.
- Housekeeping carried forward: caption fixes in s2 figures; file hashes in results _meta;
  quote the full Hopf f* range of 1.3-11.2 Hz.
- Artifact: syn_brain/paper/cdc_margins/steps/S2/round1_decision.txt.

## 2026-11-26 - CDC margins paper, S3 worker (round 1): nonlinear certificate

- Installed cvxpy 1.7.5 (with clarabel 0.11.1, scs 3.3.2, osqp 1.1.3) into .venv; solver CLARABEL.
- Design choices (one line each): (1) the certified-region box uses exact chord sectors of the
  nonlinearities on |dv| <= Rv, |dP| <= RP (two radii, since q2 = q3 = dP), not the symmetric relative
  sectors of theory 5.4(b), which give tiny regions where a slope is saturated (E-dominant sigma_E = 0.57);
  rho* itself is computed exactly as in 5.4(a). (2) Every LMI solution is re-verified in numpy after
  projecting multipliers to their cones; only verified solutions count. (3) Region objective = largest
  linearised 10 ms pulse certified, with an exponential-rate term (10% of the linear decay rate).
- The operational falsification test (fixed before the run) flagged 2 of 23,414 initial states (one
  equilibrium near the Hopf set, m_M = 0.011) only by "final scaled norm not below the initial one at 3 s";
  V was monotone and the 30 s follow-up converged (transient non-normal growth). Both readings are kept in
  results/s3_summary.json; no threshold was changed.
- Artifacts: syn_brain/paper/cdc_margins/{code/certificate.py, experiments/s3_certificate.py,
  results/s3_*.json, figures/s3_*.png, steps/S3/round1_worker.txt}.

## 2026-11-26 - Network coupling: non-diffusive becomes the primary model (William's decision)

- **Decision (William):** for physiological plausibility, the network work moves to non-diffusive
  (rate-based) coupling, g * sum_j w_ij * S(E_j(t - tau_ij)): long-range input is the excitatory
  firing of other regions, as in the original coupled JR columns, David & Friston and TVB.
  The Laplacian coupling (neurofe / PyTepFit default) is kept only as the "published pipeline"
  reference arm in the neuroscience paper (one comparison), not as the model for new claims.
- **Why:** (1) the Laplacian self term -g*d_i*E_i is a node-self-inhibition proportional to its
  own connectivity, with no anatomical counterpart; (2) it is a high-gain negative self-loop
  (DC gain g*d_i*A/a ~ 23-39) that is fragile to holds and delays; the team observed network
  instabilities with it; (3) non-diffusive coupling is a positive, bounded, monotone
  interconnection, which also suits the control-theory line (positive-systems / diagonal-
  stability certificates), see paper/control_formulation.txt.
- **Noted, not yet verified:** core/diffsim.py computes the whole coupling, including the
  instantaneous self term dg @ E, once per ms and holds it for 10 sub-steps; neurofe
  (models/jansen_rit.py:417) updates dg @ E every sub-step. A single-loop estimate puts the
  held self-loop's critical degree at d_i ~ 0.64 (1 ms hold) to 1.25 (0.5 ms), against actual
  degrees of median ~0.7, max 1.0-1.2 (10-200 nodes). This may explain part of the observed
  Laplacian instability in syn_brain runs; relevant only to the reference arm now.
- **Known risks of the switch (from N0/N1, entries above):** at the Laplacian-calibrated
  g = 1000 the non-diffusive input saturates (g*coupling 2000-5000); at g = 100 single-start
  fits stuck in a bad basin. Both predate the restart wrapper and amplitude calibration.
  Operating point now depends on node degree (mean drive g*d_i*S_bar).
- **Variant to pilot:** centred non-diffusive coupling g * sum_j w_ij * (S(E_j(t - tau_ij)) - S_bar)
  (S_bar = resting rate), optionally with row-normalised W (d_i = 1): zero mean input at rest
  (no saturation at large g, degree-independent operating point) without the Laplacian
  self-loop. Plain vs centred to be decided by the pilot below.
- **Next:** (1) pilot on 3 teachers: Fisher bounds + restart wrapper for plain non-diffusive
  (g ~ 100) and centred non-diffusive; stability screen of teachers; (2) choose the variant;
  (3) rerun the 8-subject network gate with restarts under the chosen coupling. The existing
  network gate results (Laplacian) remain valid as the reference arm.
- Single-node work (including the CDC margins paper, paper/cdc_margins/) is unaffected.

## 2026-11-26 - CDC margins S3 decision: ADVANCE; nonlinear certificate is local/small-signal

- **Decision:** S3 round 1 advances; all three pass criteria met. The 2 operational 3 s flags (transient
  non-normal norm growth, monotone V, converged at 30 s / 60 s) are ruled NOT false certificates under
  plan.txt's definition ("diverges or oscillates"); no threshold changed.
- **Scope for the paper:** the certified region covers pulse amplitudes ~2-37 vs 1000 for SP1000, so the
  nonlinear certificate is local and small-signal; rho* is reported as robustness to slope uncertainty,
  not as a certificate of the response to real TMS pulses. Region construction uses exact chord sectors
  on an explicit box (deviation from theory 5.4(b), to be documented in S6).
- Decision file: syn_brain/paper/cdc_margins/steps/S3/round1_decision.txt.

## 2026-11-26 - CDC margins S4 (worker, round 1): precision of margins; H1 and H2 met, Laplace tails too light

- **Design choices (fixed before results):** CRB from node.py's Euler simulator, 7 fitted log-parameters,
  sigma 0.42 uV, RICH = SP250+SP1000+SP4000+PP100+TR10+EI1000 (explainer definition, not the 4-condition set
  in laplace_calibration.py). Posterior check: 32 teachers (>= 12 required; 12 would give a calibrated method
  only ~61% chance of landing in the [80, 98]% band), teacher = RK4 at 0.01 ms (finer step AND different
  integrator, per CLAUDE.md) + 0.42 uV noise, 15 nominal-centred restarts x 300 Adam iterations, then an
  added Levenberg-Marquardt polish to the MAP before Laplace (Adam-only MAP results kept alongside).
- **Correction made before any coverage result:** the teacher filter first also required the 1.5 s settle
  to be within 1e-3 mV of the equilibrium; that removed exactly the near-Hopf (small m_M) teachers, so it
  was relaxed to 1 mV (stable equilibrium Z = 0, m_F > 0 is still required).
- **Results:** H1 PASS (CRB m_F 2.3% balanced, 5.2% I-dominant, RICH). H2 PASS (m_F 90% coverage 28/32 =
  87.5%). But the z-scores have heavy tails (sd 1.48; robust sd 0.87; one at 5.7). The misses are neither
  optimiser failures (LM started at the truth returns the same MAP) nor dt bias (< 0.3 SE). So the Laplace
  intervals are right for the bulk and too narrow in the tails. The paper should say so.
- Files: syn_brain/paper/cdc_margins/steps/S4/round1_worker.txt, results/s4_*.json.

## 2026-11-26 - CDC margins S4 decision (round 1): ITERATE; stationarity, Laplace step, likelihood

- **Decision:** S4 does not advance yet, which is also consistent with William's request to hold and
  review. H1 (CRB m_F 2.3% / 5.2%) and H2 (28/32) pass robustly under every variant the reviewer tried,
  so the verdict is not in question. But the numbers that S5 and S6 will quote need three fixes first.
- **Fix 1, settle transient:** the 1.5 s settle from zero leaves near-Hopf nodes ringing. This adds
  artificial information. At the balanced point the RICH CRB of m_M goes 14% -> 26% and c1 goes
  2.1% -> 6.9% once the node is stationary. Round 2 makes a stationary start (at the equilibrium, or a
  settle >= 8 s) primary. The 1.5 s numbers become a sensitivity column.
- **Fix 2, Laplace step:** the Laplace SEs are recomputed with finite-difference step h = 1e-4.
  Teacher #26's miss was an artefact of h = 1e-3.
- **Fix 3, likelihood:** SEs that account for baseline correction (correct likelihood or sandwich) are
  reported next to the naive iid SEs, which are about 6-15% too small.
- **Framing:** "margins are precise under a rich protocol and stay identifiable where saturation makes
  the raw gains non-identifiable". Not "margins precise, parameters not".
- Thresholds are unchanged. Decision file: syn_brain/paper/cdc_margins/steps/S4/round1_decision.txt.

## 2026-11-27 - Audit of the N2.5 coupling conclusions: most of the Laplacian "high-g instability" is a diffsim discretisation artifact

Reviewed the N2.5 entries above (g clip, U_BOUND, non-diffusive matched, mismatched) against a
discrepancy noted on 2026-11-26: core/diffsim.py computes the Laplacian coupling, INCLUDING the
instantaneous self term dg @ E (= -d_i E_i), once per 1 ms sample and holds it for the 10 Euler
sub-steps; neurofe (models/jansen_rit.py ~l.417) and core/simulator.py update dg @ E every 0.1 ms
sub-step. The self term is a high-gain negative loop (DC gain g*d_i*A/a ~ 20-40), and a 0.5-1 ms
hold on it puts a pole pair across the imaginary axis at ~50-100 Hz once g*d_i exceeds ~600-1300.

Test (scripts/diagnose_selfterm_hold.py, results/n2_5/selfterm_hold.json): the 8 N2-gate teachers'
TRUE parameters, training protocol, held (diffsim) vs per-sub-step (neurofe) self term, with
U_BOUND = 500 and 1e6 (no clip).
```
teacher  true g   held, no clip         per-sub-step, no clip   held U=500: tail g*coupl/U   sub-step U=500
t0        523     bounded (rms 5.9)     bounded (rms 0.5)        0.24                         0.16
t1        586     bounded (rms 15)      bounded (rms 0.7)        0.55                         0.13
t6        750     DIVERGES (rms 1127)   bounded (rms 1.0)        0.72                         0.18
t5        881     DIVERGES (rms 697)    bounded (rms 0.7)        0.83                         0.22
t7       1172     DIVERGES (rms 1362)   bounded (rms 17)         1.24                         0.46
t3       1206     DIVERGES (rms 1856)   DIVERGES (rms 826)       1.42                         0.67
t2       1223     DIVERGES (rms 1473)   bounded (rms 6.6)        1.46                         0.55
t4       1461     DIVERGES (rms 1521)   bounded (rms 0.4)        1.39                         0.22
```
(late-window rms of E-I in mV). Held-mode divergence oscillates at 56-72 Hz, as predicted for the
held self-loop. Teacher 3 also diverges with the per-sub-step self term and at a 10x finer step
(0.01 ms: rms 783), at ~60 Hz: a GENUINE instability (low a = 55/s, high g; delayed neighbour
terms), not a discretisation effect. Teacher 4 is bounded at 0.01 ms too (rms 0.11).

What this does to the N2.5 conclusions:
- "Laplacian coupling has no bounded equilibrium above g ~750-1000; U_BOUND is the only thing
  preventing divergence": NOT SOLID. True for diffsim's held implementation; false for neurofe's
  own semantics in 6 of the 7 teachers that diverged. Genuine instability: 1/8 (teacher 3).
- "g is biased low because the clip kills its gradient": the clip mechanism is right, but its
  root cause is the hold artifact (tail g*coupling/U 1.2-1.5 held vs 0.2-0.7 per-sub-step).
  The matched-Laplacian gate (teacher and student share the artifact, an inverse crime) was fit
  on artifact-contaminated dynamics for 6/8 teachers; its numbers (including the good a, mu,
  c3*c4) must be re-run with the corrected simulator before they are quoted as Laplacian results.
- "Non-diffusive is less identifiable than Laplacian": NOT SOLID as a coupling comparison. The
  non-diffusive arm is not affected by the artifact (no self term), but it is compared with an
  artifact-affected baseline, used one fixed g = 100 for all teachers, the plain (non-centred)
  form whose degree-dependent mean drive g*d_i*S_bar trades off against the operating point (an
  identifiability problem the centred variant removes, never tested), and 5 restarts that the
  adversarial review already showed are too few (loss spread 1-6 orders of magnitude).
- "A Laplacian student mis-estimates E/I on non-diffusive data": PROBABLY SOLID qualitatively.
  The fitted Laplacian g stayed at 9-40, far below where the artifact bites, and train/held-out
  RMSE doubled. But its implication is the opposite of the recorded decision: if the coupling
  form is wrong, the E/I estimates are wrong, so the coupling should be chosen on physiological
  grounds and then made identifiable, not chosen because it is easier to fit in a matched test.
- The 2026-11-27 decision "do not switch coupling modes" conflicts with William's 2026-11-26
  decision (non-diffusive primary, Laplacian as reference arm; entry above, recorded locally
  and not yet pushed when the remote session ran). It needs William's ruling.

Proposed next steps (not run): (1) fix core/diffsim.py so dg @ E is updated every sub-step, add
a regression test against core/simulator.py (which is checked against neurofe); (2) stability-
screen teachers (exact equilibria + delayed linearisation) and redraw unstable ones; (3) rerun
the comparison as FOUR arms on the same 8 teachers with 8-10 restarts: Laplacian (fixed),
non-diffusive plain with drawn g, non-diffusive centred, mismatched; (4) only then decide the
coupling. Nothing in core/ was changed by this audit.
