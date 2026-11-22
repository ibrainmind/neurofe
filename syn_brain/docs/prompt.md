Prompts to type into Claude Code
Start of the study (once)

Read CLAUDE.md and study/00_overview.md. Then read the existing JR fitting code and run Phase 0 (study/phase0_harness.md). Work autonomously until the Phase 0 gate passes or you hit a stop-and-ask condition in CLAUDE.md. Finish with the report in reports/phase0_harness.md using the template.

Continue to the next phase (after you read a report and agree)

Phase <N> gate accepted. Read study/phase<N+1>_*.md and run it autonomously following CLAUDE.md. Finish with the report.

When you disagree with a report

Phase <N>: I don't accept <specific point>. Investigate <what you want checked>, update the report, and tell me only what changed.

Independent review (run in a NEW session, not the one that did the work)

Read reports/phase<N>_*.md and the code and results it cites. Assume its main conclusion is wrong. Find the most likely reason, check it with a quick experiment, and write reports/phase<N>_review.md: AGREE or DISAGREE with the verdict, with evidence. One page maximum.

Status check (when you have 2 minutes)

Summarize in 5 lines: current phase, gate status, anything blocked on me, and the most important result so far.