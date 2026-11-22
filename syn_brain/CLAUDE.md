CLAUDE.md — JR Identifiability Study
Who I am and how I work with you

I am William, the study lead. My attention is the bottleneck. Your job is to run the study in study/ autonomously and give me decisions, not raw results.

Project goal (one sentence)

Determine which Jansen-Rit (JR) parameters can actually be identified from EEG-like data with the team's PyTorch backprop fitting framework, starting from a single node and scaling to networks of 10-200 nodes, and whether a better estimation method exists.

Where things are

Model + fitting framework: `neurofe/models/jansen_rit.py` (the shipped PyTorch JR model and `JansenRit.fit`); `syn_brain/core/node.py` (single-node teacher/student simulator, exact copy of the same equations, used for all identifiability work); `syn_brain/core/simulator.py` (network-scale version, checked equal to `JansenRit.forward` to floating-point precision, `tests/test_simulator.py`).
Study specs: `study/00_overview.md` (read first). Phase-by-phase specs (`study/phaseN_*.md`) are NOT yet written — see DECISIONS.md entry 2026-11-22 for why, and what stands in for them today.
Report template: `templates/report_template.md`.
Write reports to: `reports/<short-name>.md` (+ PDF via the matching `scripts/*_report.py`, see DECISIONS.md).
Decision log (append-only): `reports/DECISIONS.md`.
Configs: not yet split out; current work is driven by `scripts/single_node_tms.py`'s PROTOCOLS/build_jobs(), results: `results/<study>/`.

Standing rules

Work phase by phase. Do not start a phase until the previous phase's gate is PASS or I have written a decision in reports/DECISIONS.md.
Thresholds are fixed before results. Use the pass criteria written in the phase spec (for the single-node study: `docs/single_node_study_plan.txt` section 6). Never change a threshold after seeing results. If you think a threshold is wrong, say so in the report; do not change it.
Sanity checks block reporting. Every phase lists required checks. If one fails, fix the cause (usually a bug) and rerun. Never report results built on a failed check.
Avoid the inverse crime. The teacher (data generator) always uses a finer time step (dt_teacher <= dt_student / 10) and a different integrator from the student, plus realistic measurement noise, unless a spec explicitly says "matched".
Reproducibility. Every run is config-driven, seeded, and saves its config, git commit hash, and runtime next to its results where practical. Same config + seed must reproduce the result.
Adversarial self-review. Before finishing any phase report, write a section "How this conclusion could be wrong", then actually check the top 2 risks with a quick experiment, and report what you found.
Compute budget. Prefer many small runs over a few huge ones. If a phase would take more than ~8 hours of compute, propose a reduced design in the report instead of running it.
Use existing tools. Use the existing framework for the "their method" arm. For alternative methods, use well-known libraries (e.g., scipy, torch, emcee or sbi) rather than writing estimators from scratch.

When to stop and ask me (only these)

A phase gate FAILS and the spec gives no fallback.
The existing framework's code contradicts a paper description in a way that changes what we are testing.
A required design choice is not specified and different choices could change the conclusion. Otherwise, make a reasonable choice, record it in reports/DECISIONS.md with one line of reasoning, and keep going.

How to report

Use templates/report_template.md exactly. One page of verdicts on top; details below. Plain language. Every plot must have a one-sentence caption saying what it shows and why it matters. Deliver as PDF (reportlab, see `core/report_style.py` for shared styles) — William reads PDFs, not chat scrollback or raw JSON.
