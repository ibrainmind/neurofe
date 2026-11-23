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
