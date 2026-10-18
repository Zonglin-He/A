# Measured interpretation: E-retaining QR conditional-feedback closed loop32

The two-round experiment executed real student updates and real second-round teacher refresh. It did not establish useful trajectory propagation or on-policy superiority. All three QR branches have lower complete-vIoU means than cached center-local replacement in both domains. Stop expansion of this particular Gaussian-endpoint/QR/full-step-Adam combination; retain OPD and student-conditioned feedback as research routes. This result does not test every OPD objective or resolve the known finite decoder-response problem.

## Five final outputs

| Output, complete vIoU% | HC16 | Vid16 |
|---|---:|---:|
| E+MBR | 35.305477 | 15.755931 |
| Center-query local replacement | 35.484322 | 15.819496 |
| Fixed-center teacher, QR Gaussian KL | 35.280435 | 15.798980 |
| Fixed initial sampling, QR Gaussian KL | 35.318092 | 12.252162 |
| On-policy sampling/feedback refresh, QR Gaussian KL | 35.281913 | 12.259547 |

Final step2 only. All32 five outputs sealed before GT. Time interval/tIoU exactly identical across arms, all variables episodic per query/branch. No best-step choice or learning-rate search.

| On-policy contrast, complete vIoU pp | HC mean [95% paired CI] | Vid mean [95% paired CI] |
|---|---:|---:|
| Versus E | -0.023563 [-0.155253,+0.096960] | -3.496385 [-10.535778,+0.059125] |
| Versus center-local | -0.202409 [-0.468321,+0.014179] | -3.559949 [-10.646643,+0.005060] |
| Versus fixed-center QR | +0.001479 [-0.116927,+0.128965] | -3.539434 [-10.636219,+0.018505] |
| Versus fixed-sample QR | -0.036178 [-0.109805,+0.001393] | +0.007385 [-0.000488,+0.022647] |

Intervals cross zero; the large Vid mean loss is not a proven population-wide average effect. It is a real severe failure retained in the main16 denominator. The resource stop follows lack of improvement and the loss tail, not a universal impossibility claim.

## Did sparse feedback improve unobserved actual decoder frames?

| Unobserved actual sIoU%, no interpolation positions | HC | Vid |
|---|---:|---:|
| E and center-local | 56.660359 | 45.448473 |
| Fixed-center QR | 56.624296 | 45.463944 |
| Fixed-sample QR | 56.671723 | 42.631042 |
| On-policy QR | 56.615278 | 42.642713 |

On-policy-minus-E is HC-0.045082pp[-0.243116,+0.123285], Vid-2.805761[-8.467765,+0.084681]. Observed actual-node changes are HC+0.045037pp[-0.101543,+0.227807], Vid-3.370926[-10.206244,+0.126689]. Thus this run does not establish reliable sparse-to-unobserved-frame benefit. Vid on-policy-minus-fixed-sample unobserved sIoU is a small positive+0.011671pp[+0.000019,+0.033945], which is retained; both remain substantially below E in mean. That small relative recovery is not overall deployment benefit.

All16/domain remain in complete metrics, including HC1/Vid5 no-E-support queries. Missing actual-node GT stays unknown; observed support is15HC/12Vid queries, while unobserved actual support is16/16 each. Complete on-policy positive/negative/unchanged counts are HC6/6/4 and Vid1/5/10. One Vid complete loss exceeds5pp; none in HC. Benchmark interpolation and actual decoder-frame contributions are separately reported, not relabeled dense visual supervision.

## The severe Vid failure and a useful counterexample

NV005 is retained and dominates the Vid negative mean:

| NV005 complete vIoU% | E | Center local | Fixed-center QR | Fixed-sample QR | On-policy QR |
|---|---:|---:|---:|---:|---:|
| Value | 64.071553 | 64.499419 | 64.672815 | 7.829448 | 7.950116 |

On-policy loses56.121437pp on that query. In saved round2 it has4/16 admitted candidate-anchor feedback entries. Its actual QR step norm is0.364919; total output step L2 jumps to2.279980. Re-evaluating the same sparse Gaussian objective on the already saved post-step boxes, with the exact same detached targets/mask, changes loss from0.290149 to87.749891. Fixed-sample shows the same problem:0.290372 to87.715286. Fixed-center round2 has QR norm0.372883 but output step L2 only0.054496, and same-target loss0.278076 to0.266075.

These are cached non-GT objective readbacks, not new model forwards, new predictions or intermediate GT scoring. They show that similar parameter-step norms can produce very different decoded-output responses, and the severe sample-based update did not realize its own teacher objective. They do not isolate intermediate-reference detach, Adam momentum, finite step size and decoder nonlinearity. No new optimizer variant or safety gate was run.

After removing NV005 only for post-hoc sensitivity, the other15 on-policy-minus-E complete-vIoU mean is about+0.01195pp. The main result remains all16, and this small residual does not establish on-policy superiority. Positive cases are retained too: NV016 gains+0.389348pp on-policy versus E; NH013 gains+0.400817pp. Fixed-center improves NV005 by+0.601262pp; the result is not evidence that all teacher feedback harms QR adaptation.

## Did the closed loop actually run?

Yes. QR256 is the only updated variable, pretrained STVG/DINO/LN weights remain exact, and every step0 output equals E bit-exact. All32 fixed-sample/on-policy first teacher targets, masks, gradients, QR steps, post-step outputs and Adam states match. Actual first-step output changes occur on11HC/8Vid queries; those queries provide19 genuine changed-center round2 cases. The remaining13 have no first-step output change, and their final fixed/on-policy outputs are bit-exact after matched teacher batching. No-feedback scheduled steps skip Adam; no random candidate-mean state update or momentum-only movement is used.

There are96 final fits,192 scheduled rounds and118 actual Adam steps;74 scheduled rounds are no-ops. Four-frame sparse feedback, not dense independent teacher evidence, drives every update. Actual means/output displacements and admission masks are saved per round. No temporal updates were made.

## Implementation and cost boundaries

All three branches use identical diagonal-Gaussian endpoint forward KL, covariance set by existing source geometry scales, fixed16 normalization and existing per-domain Adam LR HC.01/Vid.03. This loss is algebraically source-scaled endpoint squared error. It is not the original IoU-reward OPD likelihood objective or a replication of autoregressive GKD. The experiment tests state-conditioned feedback refresh through a frozen spatial decoder with E retained, not independent novelty or universal OPD superiority.

One pre-GT engineering correction was necessary: old fixed round2 feedback came from a24-candidate teacher batch while live feedback used4. Original13 completed partial queries, scripts/runtime and prefixes are preserved. Both second-round conditions now use the same4-candidate decoder layout; only the center differs. Loss/LR/scale/threshold/noises/roster/final step and original deadline did not change. No scores or GT were read before this correction. No scientific outputs were stitched from the retired path. A subsequent audit-counter label was corrected to distinguish192 scheduled rounds from118 actual optimizer steps, without changing any predictions or task scores.

Final worker GPU lease wall time251.102898s. Prior engineering allocation is conservatively charged134.675070s, including controller/startup/stop delay; combined upper bound385.777968s, about6.43min, below the original900s cap. The combined number is a conservative bound, not a precise GPU-kernel-time measurement. Final worker completed32 Source restoration forwards/104 DINO image-text encodings/656 conditional reference queries. The preserved engineering prefix adds13 completed Source captures/48 completed DINO encodings/132 completed conditional queries. Unreceipted interrupted work is not invented as completed and is covered by the conservative cost bound. First-use historical E/original DINO/calibration costs were reused and are not included. Full breakdown is public.

## Decision

The specific E-retaining QR256 sparse Gaussian-endpoint/full-step-Adam combination did not meet the continuation condition: on-policy did not exceed fixed feedback and center-local in both domains, and it did not improve unobserved actual frames. Stop this combination without adding rounds, broadening the cohort or changing the step size based on target scores. Preserve E+MBR and center-local as references; preserve OPD/conditional-feedback research as unresolved. The severe finite-update response prevents interpreting this as proof that student-state-conditioned teacher feedback itself is useless.
