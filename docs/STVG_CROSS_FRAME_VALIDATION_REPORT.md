# Final Cross-Frame Validation and recipe freeze

2026-10-15. Eight historically exposed development sources, four per direction. This finite mechanism comparison is distinct from the authorized full Table 1 evaluation.

## Selection fixed before new annotation scoring

A is the current train-reference validated Gaussian IoU OPD. B additionally rejects a proposal if agreement at fixed non-training SAM2 frames worsens. The original 22 training anchors, source checkpoints, five-step budget, reward, sampling and optimizer are unchanged. No new detector, tracker, R1 or GT-Oracle fitting was run.

Validation positions were locked before visibility inspection: at most eight integer-quantile source-grid positions inside the QA interval, excluding training frames. Of 55 candidates, 35 had visible selected-ID masks and 20 remained unknown. Five cases had at least two valid positions; three fell back to A. The active validation set contains 33 positions; the two single-mask cases are excluded by the predeclared fallback rule.

B was eligible only if both domains retained informative accepted updates and their paired mean non-training Teacher IoU was no worse than A (tolerance 1e-7). HC had two evaluable sources and B-A = -0.000159379795; Vid had three and +0.001591818458. The global rule therefore selected A before new GT was accessed. This tests propagation agreement with the same imperfect teacher, not correct semantic identity.

## Complete full-tube comparison

Equal source means; values in percent. Labels were read only after every new prediction and the A/B choice were sealed. Historical development exposure remains.

| Arm | HC vIoU | Vid vIoU | HC dense sIoU | Vid dense sIoU |
|---|---:|---:|---:|---:|
| Frozen | 22.210 | 14.490 | 47.895 | 48.458 |
| Temporal QA | 31.286 | 22.442 | 47.895 | 48.458 |
| Original Spatial | 21.840 | 21.291 | 44.358 | 69.942 |
| Temporal + Original Spatial | 32.436 | 31.899 | 44.358 | 69.942 |
| Spatial A (train validation) | 23.840 | 21.291 | 47.363 | 69.942 |
| Temporal + A [frozen recipe] | 33.601 | 31.899 | 47.363 | 69.942 |
| Spatial B (cross-frame) | 24.321 | 21.732 | 48.281 | 70.689 |
| Temporal + B | 34.276 | 32.508 | 48.281 | 70.689 |
| Two-candidate Temporal | 27.255 | 18.265 | 47.895 | 48.458 |
| Two-candidate Temporal + A | 27.244 | 26.287 | 47.363 | 69.942 |
| Temporal + Teacher Geometry | 36.410 | 34.788 | 63.631 | 76.713 |
| Two-candidate + Geometry | 35.576 | 27.694 | 63.631 | 76.713 |

B has higher post-seal development vIoU than A (+0.674 pp HC, +0.609 pp Vid). That positive observation is retained, but does not retrospectively change the label-free selection rule. Teacher Geometry-only remains stronger than either Student version; dense teacher geometry and four-anchor adaptation have different supervision densities.

| Case | A - Temporal (pp) | B - Temporal (pp) | B - A (pp) | Fast - A (pp) |
|---|---:|---:|---:|---:|
| HC01 | +0.730 | +0.730 | +0.000 | +5.492 |
| HC02 | +10.061 | +10.061 | +0.000 | -30.172 |
| HC03 | +0.000 | +0.000 | +0.000 | +0.000 |
| HC04 | -1.531 | +1.167 | +2.698 | -0.750 |
| V01 | +18.618 | +21.052 | +2.434 | -3.303 |
| V02 | +6.265 | +6.265 | +0.000 | -20.306 |
| V03 | +0.000 | +0.000 | +0.000 | +2.232 |
| V04 | +12.945 | +12.945 | +0.000 | -1.070 |

## One bounded speed variant

The only method-changing speed trial removed expanded/contracted temporal candidates, retaining Native and Projection. Eight full-head replays first reproduced the original four-candidate parameters and intervals exactly, then eight actual two-candidate CPU Adam fits were sealed. No new teacher generation was needed because the exact original responses were cached.

The predeclared allowed loss was 1 percentage point per domain. The complete two-branch result lost 6.357 pp HC and 5.612 pp Vid, so this deletion was rejected. Historical request-count/component-time savings are not a fresh end-to-end throughput measurement. No subsequent prompt, learning-rate, radius or temperature search is authorized by this result.

## Execution and verification

The actual B worker loaded the two source models once each, ran seven source captures, seven fits and seven final full-model re-decodes in 54.499 seconds. V03 reused its sealed no-information/no-update output and is not counted as a new capture or fit. All eight output receipts were independently checked for source, frames, pixel binding, bytes and hashes. The worker exited normally under its own RAM/GPU/disk guard. This incremental Student time excludes all earlier expert work and is not Table 1 cost.

Independent endpoint geometry, parameter/Adam rollback, cached-to-full readout, official/independent metric agreement and summary aggregation were checked. Saved mathematical readback does not claim an additional independent GPU backward. All 96 case-arm rows, 120 metric summaries, paired parent-bootstrap contrasts, negative outcomes, neutral outcomes and unknown support are exported.

## Frozen next-stage method and scope

Use one episodic recipe for both domains: original four-candidate Caption-Query Alignment with 512-parameter Adam, official frozen YOLOv12-X/SAM2.1/R1 identity evidence, WHO/WHEN separation, and A: 1792-parameter Gaussian IoU OPD with training-reference rejection and full parameter/Adam rollback. CURRENT is unchanged. No new algorithm search follows this finite comparison.

Next execution is engineering qualification of model residency, exact per-video cache reuse, recovery and disk growth, followed by 32 fixed non-current-development parents per direction and the authorized 13,785-query official evaluation. HC2 is validation, VidSTG is test. Those phases are not completed by this report. Every official query prediction must be sealed before new official-label evaluation; no official labels select the frozen recipe. Old stopped queues are not resumed.

STVG-R1 external training overlap with target evaluation remains incompletely established. Describe the method as frozen external-expert-guided TTA; do not claim no target-domain teacher prior.
