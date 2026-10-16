# One matched instance-evidence experiment

The whole historical candidate audit motivated one prospective Source-seeded RGB foreground hypothesis. E0 retains old proposals and current response-guided Lite OPD. E1 replaces the last proposal slot with the new hypothesis and keeps the same updater. E2 uses exactly E1 references with original validated OPD. Frozen and corrected Temporal are included. All three use the same corrected time interface, four physical centers, Source checkpoints, seed keys and fixed HC lr .0045/Vid .027; no parameter or prompt search is performed.

The new mask uses one locked1.25 support and central half-size foreground seed, three original GrabCut iterations, complete foreground and five-frame local DIS propagation. Source remains ID1 and other first-eight IDs stay unchanged. This is Source-conditioned evidence, not independent object discovery. It may merge background or remove a useful old last candidate, and cannot reliably recover a referent outside a wrong Source prior. Invalid candidates retain the original pool.

The32 sources are previously exposed development, including12 result-stratified diagnostic sources. Both complete Student groups pass independent saved rollout/Adam/solver/acceptance and actual full-model final-box checks before joint prediction sealing and GT scoring. No-reference/no-update sources remain in the denominator. Geometry measurements use exact observed GT positions only; missing annotations remain unknown. Candidate-pool quality is not full-Tube vIoU, and low box IoU is not an identity annotation.

## Complete vIoU paired results

| Domain | Contrast | Mean delta (pp) | Positive/negative/unchanged |
|---|---|---:|---:|
|hc2|E1_minus_E0|+0.140518|3/5/8|
|hc2|E1_minus_E2|+0.294844|6/3/7|
|hc2|E0_minus_Temporal|+0.160716|6/3/7|
|hc2|E1_minus_Temporal|+0.301234|6/3/7|
|hc2|E1_minus_Frozen|+2.147881|9/5/2|
|hc2|E2_minus_Temporal|+0.006390|5/4/7|
|vidstg|E1_minus_E0|+0.289384|4/1/11|
|vidstg|E1_minus_E2|+0.232767|4/3/9|
|vidstg|E0_minus_Temporal|+0.636849|5/2/9|
|vidstg|E1_minus_Temporal|+0.926233|5/2/9|
|vidstg|E1_minus_Frozen|+1.096185|5/4/7|
|vidstg|E2_minus_Temporal|+0.693467|4/3/9|

E1-E0 isolates this evidence modification; E1-E2 isolates the updater under identical new references. All160 five-arm rows, supervised/unsupervised metrics, all candidate scalar diagnostics, update admission/nonzero counts, costs, negative and unchanged sources are supplied. The independent saved mathematics audit does not claim a second complete backward pass. Historical source/proposal costs are inherited and are not zero raw-deployment cost.

E1-E2 is a complete updater comparison, not an isolated projection-direction ablation: current Lite also differs from original validated OPD in weighting/control-variate/acceptance details. Independent projection benefit cannot be attributed from this contrast alone.

The first saved-output reader converted float32 boxes directly while the original complete-model arm conversion uses float64. This caused an exact-array check failure after the E0 GPU group completed successfully. The original output, source and failure are preserved. A separate reader applies exactly the original float64 conversion; all other reader AST expressions and tolerances remain unchanged. E0 GPU is not rerun. The continuation executes E1 once and independently checks both complete groups.

Prospective input metadata also retains an unused optional old tensor hash in E1 windows. Independent readback records128 such stale optional hashes and45 fresh E0 windows where the optional hash is absent. All actual mask archives, rendered NPY file/tensor hashes and center geometry are checked. This metadata defect does not change generated inputs or reuse any response; original fields and the explicit correction receipt remain available.

No automatic confirmation, cross-backbone result or fullset claim follows from a completed measurement. The final decision receipt must state whether a shared recipe has evidence to justify confirmation. A generic RGB/box API is not an empirical second-backbone experiment. CURRENT remains unchanged. Private media, GT coordinates, weights, tensors, real Query-caption and raw Teacher responses are excluded. This is a finite evidence test, not a new tracking model or another optimization sweep.

## Development decision and coverage

The pre-score rule selects E1 for one fixed64-source confirmation, with no retuning. E1-E0 is+0.141/+0.289pp(HC/Vid); E1-E2+0.295/+0.233pp. E1-Temporal+0.301/+0.926pp. E1's changed-evidence signs are3 positive/5 negative/8 unchanged in HC and4/1/11 in Vid, so the positive means do not imply uniform improvement. This is the exposed32 development screen, not confirmed domain robustness. The old64 parents are disjoint from this32 but have historical project exposure.

On common exact-GT windows, nonSource pool coverage at IoU .5 increases18→30 of43 in HC and4→11 of26 in Vid.21/38 respective windows have no exact GT and remain unknown. Pool-max mean IoU rises .423→.635/.321→.465, while actual R1-selected mean rises only .529→.542/.259→.282. Thus the candidate modification improves geometric opportunity more than the Teacher extracts from it. These are observation-level geometry, not Tube vIoU or semantic identity annotations.

E0/E1/E2 execute22/23/23 actual fits,83/78/75 accepted steps, and21 nonzero final parameter updates each. All68 fits have locked config/seed and complete-model readback checks. No-reference, rejected and zero-net-update cases are retained.
