# Propagation v0: attachment-bound prototype

This isolated prototype follows the user's v0 specification. It changes the student to frozen STVG plus an episodically adapted output-side propagation head. It neither changes the registered geometric method nor resumes the paused full VidSTG/CG tasks. Reuse the same first64 exposed HC2 development parents, one query each, seed41. The initial trajectory B0 is the existing single-ReferDINO ERef with unchanged Strong temporal inference and physical MBR.

## Differences from the previous cached-policy pilot

There are at most three candidates, obtained from separate forward/backward support-map components, rather than four projection-interval/secondary-peak hypotheses. The six student features are memory cosine, previous-region feature cosine, sparse connection, original-region feature cosine, negative bounded geometric displacement, and the original indicator. **Support u is not a student feature.** The student initializes all six weights to zero; its valid original candidate has prior0.7 and alternatives share0.3. Initial greedy output is exactly B0.

Teacher feedback is softmax((u +0.25L +0.1H -0.1G)/0.2), without the original-candidate prior. Student logits are log(pi0)+phi*x, with no student temperature. Memory writes use omega=0.2*c*u(selected); invalid evidence and event-external frames keep B0 and do not write memory. No unconditional wrong-candidate write or forced observation memory reset is added.

Distillation uses sum(c*KL(qE||pi))/(sum(c)+1e-12), plus0.05 times the all-path/all-node mean KL(pi||pi0). Adam learning rate is0.1, gradient norm limit1, at most3 rounds and2 paths per round. OPD fits on newly visited hard student histories; FixedKD reuses two fixed teacher histories; Direct has full available gradients through a soft recurrent history and feedback. Its final output also uses an actual hard greedy trajectory.

After each update, decode one real greedy path. A lower fixed evidence score restores all parameters and Adam state and stops that query; there is no backtracking search or GT stopping. The score is memory-free and additive:0.5 mean observed u +0.5 mean unobserved c*u +0.25 mean event-adjacent c_prev*c*L -0.1 mean all-node G. Empty groups contribute zero. The same score/check is shared by all adapted controls. Exact finite-candidate dynamic programming is retained as a strong direct-evidence baseline, not presented as a weak control.

## Prelocked unspecified implementation details

Use the original Source-feature FP32 cosine/top20/temperature0.1 matching, then FP64 mask transport and policy arithmetic. Reuse the same selected ReferDINO instance's **predicted masks**, not GT segmentation. Keep separate nearest past/future maps and per-source cycle Dice r. Combined support is sum(r*P)/sum(r), and node confidence is the available-source cycle mean; no forward/backward agreement multiplier erases disagreement.

Extract4-connected components strictly above0.5 of each map's own peak. Rank by cycle reliability times component support mass, preserve the native teacher box as an observation alternative, deduplicate at IoU0.7 and retain at most two alternatives. The original candidate is always retained. Confidence below0.1, invalid visual evidence or the frozen event-external range permits only the original. Geometry is G=1-exp(-d^2), where d=norm(center deviation divided by original size, log-size deviation)/2. All values are fixed before reading this prototype's GT, without a parameter search.

## Ordered execution and decision rule

Stage1 builds all64 banks, samples two initial student histories per query and produces evidence-DP, memory-greedy and no-memory-greedy paths. These artifacts are fully sealed before independent labels are opened. No fitting is run at this stage.

The user's instruction to diagnose first is made executable by a one-time **whole-prototype** development gate:

1. Mean per-query unobserved candidate-oracle IoU gain over B0 is strictly positive. Original-only fallback nodes remain in the denominator.
2. Within the same visited state, valid-candidate teacher feedback has query-macro non-tied pair accuracy above0.5 and query-macro Spearman above0. GT-tied pairs are excluded; score ties count0.5. Non-identifiable fallback/constant states are recorded as coverage, not assigned a fabricated correlation.
3. Memory-greedy minus no-memory-greedy has nonnegative mean full vIoU and unobserved sIoU differences, to tolerance1e-12, with at least one strictly positive. This checks that the added history has a net benefit before attempting to repair it with distillation.

Report10,000 parent bootstrap intervals for these diagnostics; the gate uses the predeclared point estimates, not a post-hoc significance threshold. It is a feasibility screen, not proof of reliability. If any check fails, retain the negative diagnostic and stop before adaptation. Do not modify thresholds, candidate construction or feedback using the revealed GT. If all pass, run the three prelocked adaptation controls on all64 parents, then seal all outputs before independent final scoring. The panel and the GT-assisted family go/no-go decision are development-exposed, not independent confirmation. GT never chooses an individual output, feedback, stopping step or rollback.

Distance analysis includes every fixed unobserved node with GT, whether changed, gated or rolled back. Normalize distance to the nearest teacher observation by the median physical spacing of the eight observations; use prelocked bins(0,0.25],(0.25,0.5],>0.5. Observed nodes are separate.

## Reproduction and cost

Use `python -B scripts/propagation_v0/run.py --stage evidence --run`, then the independent `diagnose.py --run-after-stage1-seal`. Only a passing fingerprint-bound gate permits `run.py --stage adapt --run`; final `score.py --run-after-seal` requires complete stage2. Configuration is in scripts/propagation_v0/config.json. Stages bind code, configuration, original caches, masks, reference outputs and source identities; failed attempts are retained rather than mixing fingerprints.

This cache-based experiment has zero Source/decoder/encoder/expert/RGB calls. Its incremental construction/adaptation costs are measurable. Reusing a cache does not establish end-to-end deployment latency for the changed pipeline; do not replace the missing cold full-pipeline timing with a sum of unrelated old timings. Raw masks, region descriptors, tracks, private IDs and transient states remain local.

## Source boundary

[HTR](https://arxiv.org/html/2403.19407v2) supports studying reference-based propagation but uses supervised trained modules. [GKD](https://arxiv.org/html/2306.13649v3) motivates distillation on student-generated histories and a supervised gradient that does not differentiate through discrete sampling. Neither establishes this v0's STVG performance or necessity relative to direct evidence decoding. OPD uses backpropagation too; its distinction here is the source of visited states, not the absence of gradients.
