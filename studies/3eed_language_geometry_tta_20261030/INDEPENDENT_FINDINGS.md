# Independent round4 findings

Final-code audit and independent CPU scoring are complete. Neither locked three-step adaptation arm is adopted. This audit did not run a source model, GPU computation, training, new adaptation, a parameter sweep, or a new prediction. It writes only this audit directory.

The implementation binds CODE_LOCK f409fa13165631707703112cb4a44da8adfd80f6bfb1b7a406fdf8f779413fff and mechanism a470d5daa48f8016a357d646a33c10a96a9a12ff945459e53c9c506c710be3aa. Development prediction seal is dbad6a7d85f5835bc0d114c234862059d8c094fd68acfdb4320e6239edf857d8. Confirmation master 080f4a00d1d4a77323a1012e2a8ee0f0cc4d858f67aa8dfcc06ef8dd422c786a and all 981 referenced files were verified before this auditor opened confirmation target metadata. The code path and receipt establish this auditor's gate; they cannot prove the absence of any unlogged external process accessing labels.

The independent scorer uses general three-dimensional convex intersection; the parent's implementation uses horizontal polygon clipping plus height. They agree on all 1,824 selected-box IoUs: DEV608 maximum difference 3.56e-15, CONFIRM1,216 maximum difference 8.44e-15. Candidate diagnostic selected/best IoUs and fixed-query GT substitutions add 672 compared numeric cells, maximum difference 1.09e-14. All 2,688 top-k/coverage/rank-recoverability Boolean cells and all arm threshold counts, invalid counts and platform/scene denominators agree. These are numerical checks, not independent experimental replications. The same source checkpoint and observations are shared.

## Primary Waymo results

All entries are mean oriented IoU in percent; PCA is the fixed standard axis-aware conversion for each of all 256 reference candidates. Geometry updates keep selected query, source sizes and PCA yaw fixed. Query arms select from exactly the same reference pool and PCA geometry.

| Arm | DEV24 / 12 scenes | CONFIRM48 / 24 scenes |
|---|---:|---:|
| Frozen raw yaw0 diagnostic | 49.68696 | 54.28061 |
| Frozen single-view BBF + PCA | 56.85255 | 60.90302 |
| IoU-weighted three-full-view TTAug | 53.93627 | 61.20654 |
| Spatial teacher direct, no update | 53.93627 | 61.20654 |
| Query0 / query1 / query3 / no-spatial | 53.93627 | 61.20654 |
| Geometry analytic direct, no update | 55.22413 | 59.76930 |
| Geometry1 | 56.72648 | 60.69175 |
| Geometry3 primary | 55.17187 | 59.44379 |
| Original frozen BBS head + PCA | 29.78259 | 41.23440 |
| Simple arithmetic three-view BBF | 53.93627 | 61.15338 |

Query3's confirmation change from the strong frozen single-view baseline is +0.30352 percentage points, with scene-bootstrap descriptive 95% interval [-0.41510,+1.46541] pp. It is exactly zero relative to matched TTAug and direct teacher: all final selected boxes are identical across those controls in both cohorts, including auxiliary platforms. The update changes scores and parameters but supplies no measured selected-box gain. The +0.30352 pp cannot be attributed to parameter adaptation. Simple arithmetic averaging is distinct from the mechanism's geometry-IoU-weighted aggregation; both remain reported.

Geometry3's confirmation change is -1.45922 pp versus frozen PCA, scene-bootstrap interval [-2.43929,-0.57723] pp; versus its same-information analytic direct solution it is -0.32551 pp, interval [-0.65176,-0.05886] pp. The individual signed differences relative to frozen are 5 positive, 19 negative and 24 unchanged. DEV is also negative: -1.68068 pp versus frozen. Geometry1 is better than Geometry3 in these samples, but three steps remain the predeclared primary; this observation is not a new one-step method selection.

## Measurement and implementation

All new source predictions and evidence caches were sealed before labels; adaptations consumed only frozen GT-free cache tensors, text and corrected calibration/XYZ inputs. The source checkpoint is the released union checkpoint, not a source-only checkpoint. The cancelled long training remains zero epochs. No source backbone is updated here.

Actual nonzero parameter updates occurred in 29/32 DEV and 47/64 confirmation query episodes. Query dimensionality is 64 with 512 trainable rank-four residual parameters, down initialized orthogonally and up zero. The real sealed outputs satisfy exact zero/no-spatial bridges to TTAug, and code assertions plus sealed traces establish per-episode reset. Largest embedding step/total changes were .034292/.055661 (DEV) and .017610/.024738 (confirmation), below the locked .05/.1 caps. The original token denominator and native logit/span anchor remain intact. Query updates are cached feature-adapter adaptation, not full-model adaptation.

Geometry's three center latents changed in 19/32 DEV and 33/64 confirmation episodes. No-observation controls return the frozen source exactly. Observed local axes alone move; all sizes/yaws remain byte-identical. The analytic direct output matches the independently reconstructed interval optimum with the same .15m cap. Latent bounds and nonincreasing unlabeled objectives pass. Float32 output rounding changes the reported source-coordinate displacement by at most 1.76e-6m, within the locked tolerance. Geometry is output-latent optimization and must not be described as a learned feature update.

The sealed traces and final code provide actual gradient/update/reset evidence. Serialized trainable parameter states were not saved, so this auditor did not independently replay adaptation or inspect final parameter tensors. This limitation does not affect independent box scoring.

## Teacher quality after prediction seal

The horizontal-region parser was informative for 31/32 DEV and 52/64 confirmation inputs. For Waymo it parsed 23/24 and 44/48, respectively. On confirmation, all 48 GT centroids project into the calibrated image; 40/44 satisfy the parsed broad horizontal region. This is feasibility, not instance discrimination. Of the four Waymo confirmation cases recoverable at IoU>.5 by selecting another candidate, all four give the best-IoU candidate the same spatial penalty as the original selected candidate. The broad left/right constraints do not distinguish those competing candidates. No direct teacher changes its TTAug top1 selection.

Posterior expected candidate IoU changes by -.000643 on Waymo DEV and +.003622 on confirmation, while top1 is unchanged. These soft-distribution changes are not task-level adaptation gains. Unsupported projections receive zero penalty; global normalization can increase their posterior mass. This is the locked teacher definition and an explicit limitation, not a silently replaced conditional-normalization teacher.

The geometry teacher accepts 18 Waymo DEV axes (14 horizontal, 4 vertical) and 42 confirmation axes (32 horizontal, 10 vertical). The target GT center lies inside 0/18 and 2/42 corresponding intervals. Confirmation center-distance change is favorable in 7 cases and unfavorable in 21, with the remaining 20 unchanged. These are conditional post-seal diagnostics. They mix visible-face estimation, crop contamination, frozen size/yaw and wrong-instance selection; they do not isolate any one cause or establish that every accepted normal belongs to the target. Near-face sign is a camera heuristic, not a true occlusion/instance observation. The present assumptions do not supply a sufficiently reliable center target.

The final parser has a finite object vocabulary and does not generally reject negation. Synthetic relative-to-van/bicycle and negated-left statements are accepted as global left. The defined lexical edge patterns trigger zero times in the actual 32 DEV and 64 confirmation parsed fragments, and re-parsing all 96 inputs matches the sealed traces exactly. Thus the synthetic edge cases limit general parser claims; they are not evidence that the measured cohort errors arose from those cases. An old unlocked DEV text-only summary has one stale unsupported entry; final trace and final parser are authoritative.

## Error decomposition and annotation association

On Waymo confirmation, fixed-PCA recall at IoU>.5 is 43/48 at top1 and 47/48 at top5,10,32,256. At IoU>.25 it is 45/48 at top1 and 48/48 at all larger reported k. One case has no IoU>.5 candidate; four are rank-recoverable. The geometric best256 mean is 69.89363%, a label-assisted recall diagnostic, not a deployable oracle or a proof of exact physical identity. Raw yaw0 recall at IoU>.5 is only 28/48 at top1 and 29/48 at all256.

Two of the four fixed-PCA rank-recoverable confirmation cases overlap a non-target annotation above .5 while the target overlap is at most .5; both annotated distractors share the target class. The available `others` schema has class and boxes but no stable instance identifiers or certified exhaustive catalog. This supports annotated distractor affinity and a ranking problem, without making an exact-instance identity claim.

At the same selected query, GT-center injection reaches 68.25107%, GT-size 64.44498% and GT-yaw 68.08972%. These are nonadditive, label-assisted conditional diagnostics. GT-center replacement can teleport a wrong instance to the target; GT-yaw is not a maximum under fixed wrong center/size and harms 7 confirmation cases. They must not be interpreted as separable causal regression-error contributions or test-time signals.

## Scope, cost and decision

Primary Waymo VAL results use new research scenes; source checkpoint selection history is not erased. Drone/Quad data use newly researched TRAIN scenes exposed to union pretraining, and remain separate auxiliary results. DEV has one Drone scene and two Quad scenes; confirmation has two scenes per auxiliary platform. No pooled headline and no one-scene population confidence interval is justified. The bootstrap draws scenes, not resampled views, and intervals are descriptive.

The complete cached CPU adaptation loops sum to 3.058s for DEV32 and 5.338s for confirmation64. These timings include the cached runner's two update arms and associated controls/teacher construction, omit source GPU forwards and cache creation, and are not full deployment latency.

Do not adopt query3 or geometry3 under this protocol. Keep all finite arms, signed tails, failure logs and source controls. The experiment rejects the measured mechanism contribution; it does not reject all conditional-query TTA or all geometric adaptation. Further hypotheses require a new protocol and authorization rather than a post-confirmation parameter adjustment.

Machine-readable receipts: `INDEPENDENT_NUMERIC_AUDIT.json`, `DEV_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json`, `CONFIRM_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json`, `CONFIRM_ANNOTATION_ASSOCIATION_PRIVATE.json`, `DEV_TEACHER_QUALITY_PRIVATE.json`, `CONFIRM_TEACHER_QUALITY_PRIVATE.json`, and the two teacher supplements. All files marked PRIVATE and all candidate-IoU arrays are GT-derived and must not enter adaptation or a public export.
