Study overview: what can a JR model really identify?
Core questions
Q1. Structural identifiability. With perfect, noise-free data, which JR parameters (or combinations) are identifiable from (a) resting-state spectra and (b) evoked responses (TMS-like pulse)?
Q2. Practical identifiability. With realistic noise and data length, which of those remain identifiable, and how precisely?
Q3. Their method. Does the PyTorch backprop fitting framework recover the parameters that Q1–Q2 say are recoverable, and does it ever appear to recover parameters that Q1–Q2 say are not (a sign of curve fitting)?
Q4. Better methods. Is there an estimator that recovers the identifiable set more accurately, more reliably, or with honest uncertainty?
Q5. Scaling. How do the answers change from 1 node to 10, 50, 100, 200 nodes with structural connectivity, a leadfield, and 64 / 32 / 16 / 8 channels?
Phases and gates
Phase	Spec	Answers	Gate to continue
0	phase0_harness.md	Harness is correct	All sanity checks pass
1	phase1_structural.md	Q1	Identifiable set written down for both data types
2	phase2_practical.md	Q2	Recovery precision table for each parameter
3	phase3_bp_recovery.md	Q3	Verdict on their method for 1 node
4	phase4_method_comparison.md	Q4	Recommended estimator + parameterization
5	phase5_network_scaling.md	Q5	Scaling verdicts per network size
The one deliverable that matters

At the end, a table: for each parameter (or parameter combination), at each scale, one of:

IDENTIFIABLE — recovered reliably; worth fitting and interpreting.
WEAKLY IDENTIFIABLE — recoverable only as part of a combination or with priors; fit as a combination.
NOT IDENTIFIABLE — data cannot determine it; fix to literature value.
NOT NECESSARY — changing it does not meaningfully change the output in the realistic range; fix it. Each entry must be justified by Phase 1 analysis plus Phase 2–3 evidence.