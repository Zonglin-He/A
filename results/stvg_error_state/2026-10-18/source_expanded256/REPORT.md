# Frozen Source-start E expanded evaluation: HC128 + Vid128

The exact fixed parent/query roster is unchanged. Targets are historically analyzed; this is expanded evaluation, not independent confirmation. Original E32 beta1/16, geometric calibration, four-frame DINO support, candidate pool and full-trajectory correction are frozen. All three arms share final-distribution MBR.

## Explicit starting-state change

The human stopped original online-state recovery and selected a frozen Source start. The cancelled recovery retained all71 completed checkpoints and two Before captures. Its partial inherited states are NOT inputs to this experiment. Source_no_correction is the frozen trajectory; A/E are output corrections of that exact trajectory. This measures frozen-start correction and does not establish incremental improvement over inherited online OPD.

| Target | Source vIoU % | A vIoU % | E vIoU % | E−Source pp | E−A pp | E−Source unobserved sIoU pp | E−A unobserved sIoU pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | 22.614738 | 22.843150 | 25.160755 | 2.546018 | 2.317605 | 4.012292 | 4.890564 |
| vidstg | 14.037325 | 16.913820 | 16.956040 | 2.918715 | 0.042221 | 4.491397 | 1.271243 |

All256 outputs sealed before new GT scoring. GT/old labels never calibrate or route E. No-reference/no-op/unknown and negative cases remain. Missing annotations are unknown, not failures. Per-parent paired10000 bootstrap CIs, positive/negative/neutral counts, <-5/<-20pp harm tails, leave-one-out sensitivity, equal-domain macro CIs and existing shared-bin motion/area/event-length strata are in RESULTS.json. One query per parent makes query and parent means equal.

## Cost and limits

Exact full capture reuse: 21; new frozen Source forwards: 235. ZERO new DINO frames, optimizer steps or model training. Actual capture/output/score wall and cache work are in ACTUAL_COSTS.json; cancelled recovery is disclosed separately and excluded from method latency.

The original E32 failed two-domain no-regression gate remains. No hyperparameter scan, method promotion, domain-specific routing or dense OPD follows automatically. Original E32 parent overlap is HC11/Vid5; original tuning parents were excluded before sampling.

Independent exhaustive objective/output and metric/aggregate audits passed. Root actually inspected the two aggregate plots and both domains at the four original planned image slots, with original pixel SHA checks. This is an engineering output view, not new identity/semantic labels. Remote publication and archive completion are recorded by separate final receipts.


## Historical exposure sensitivity

Source_no_correction means frozen Source boxes with the SAME final-MBR interval used by A/E; it is not the Native-decoder benchmark Source Only row. Unobserved actual-frame sIoU uses all known GT positions on the actual sampled decoder grid except the four planned expert slots, independent of the predicted MBR support.

Metadata-only checking before new target scoring found HC18 source-train and7 source-validation calibration parent overlaps (25 total); Vid0. They are different video files, query text and sampled inputs, not exact-query reuse. Main256 is unchanged. A pre-score lock additionally reports HC103/Vid128 excluding those parents; this is exposure sensitivity, not a replacement main cohort, parameter-selection set or independent confirmation.

| Exposure-excluded target | Parents | E−Source full-v pp | E−A full-v pp | E−Source unobserved s pp | E−A unobserved s pp |
|---|---:|---:|---:|---:|---:|
| hc2 | 103 | +2.366082 | +1.958538 | +3.414883 | +4.596557 |
| vidstg | 128 | +2.918715 | +0.042221 | +4.491397 | +1.271243 |

[All signed results](RESULTS.json) · [exposure sensitivity](EXPOSURE_SENSITIVITY.json) · [actual protocol and costs](ACTUAL_PROTOCOL.json) · [independent audit](ROOT_INDEPENDENT_AUDIT.json) · [paired intervals](EXPANDED_SOURCE.svg) · [all parents](ALL_PARENTS.svg).


## Interpretation and negative tails

E improves full vIoU and unobserved actual-frame sIoU relative to the same-clock frozen Source in both domains. Relative to A, HC improves by 2.317605 pp full vIoU (paired95% interval [0.543592,4.176199]); Vid improves by only 0.042221 pp ([-0.786659,0.878388]), so this does not establish a Vid advantage over A. The equal-domain E−A mean is +1.179913 pp ([0.175674,2.213189]). Severe E−Source full-v losses below −5 pp remain10/128 HC and5/128 Vid, including4/128 and3/128 below −20 pp; negative samples are preserved. All intervals are descriptive for this historically exposed expanded cohort. No automatic confirmation, online-method promotion, or dense OPD is authorized by a positive mean.
