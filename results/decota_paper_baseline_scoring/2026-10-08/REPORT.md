# Sealed baseline evaluation, 2026-10-08

The human authorized this limited post-seal evaluation after pausing EATA. No inference, Fisher preparation, source-media download, OPD full-roster evaluation or parameter search was performed in this stage.

VidSTG test uses 10,303 official queries from 732 videos. HC2 validation uses 3,482 official clips/queries from 237 parent movies. Each query is averaged across three original orders; the primary source macro then averages queries within parent sources and weights parents equally. The official query macro weights every query equally. All values below are percentages; differences are percentage points.

EATA is a single completed direction supplement. Target-trained reference uses supervised same-dataset source training and is reported separately from TTA. The registered OPD main method has no complete prediction roster here; development tuning scores are not inserted. Historical target sources have been exposed, so this is not untouched confirmation.

## HC-source → VidSTG test

| Method | Source vIoU | Source Δv (95% CI) | Source tIoU | Source sIoU | Source R@.3 | Source R@.5 | Query vIoU | Query R@.3 | Query R@.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source Only | 12.065 | 0.000 [0.000, 0.000] | 32.569 | 34.499 | 14.272 | 3.127 | 11.129 | 12.831 | 3.300 |
| DINO-Refine | 15.372 | 3.308 [2.909, 3.720] | 32.569 | 38.272 | 20.078 | 7.828 | 13.985 | 18.024 | 6.920 |
| Target-trained reference | 21.449 | 9.385 [8.588, 10.215] | 45.316 | 44.099 | 31.730 | 15.884 | 19.296 | 28.137 | 13.103 |
| TENT-STVG | 9.512 | -2.553 [-2.847, -2.267] | 25.819 | 34.520 | 10.727 | 2.382 | 8.721 | 9.774 | 2.271 |
| SAR-STVG | 12.065 | 0.000 [0.000, 0.000] | 32.569 | 34.499 | 14.272 | 3.127 | 11.129 | 12.831 | 3.300 |

### State transfer and negative tails

| Method | Inherited Δv | Current Δv | Gross source gains | Gross source losses | Source harm >5pp | Source harm >20pp | Updates / arrivals | Recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO-Refine | 0.000 | 3.308 | 3.765 | 0.457 | 16 | 2 | 0 / 30909 | 0 |
| TENT-STVG | -2.541 | -0.011 | 0.289 | 2.841 | 138 | 2 | 30909 / 30909 | 0 |
| SAR-STVG | 0.000 | 0.000 | 0.000 | -0.000 | 0 | 0 | 0 / 30909 | 0 |

The inherited effect compares saved Before with frozen Source; the current effect compares saved After with Before. Spatial and temporal terms telescope exactly to the total difference. Reference checkpoint differences are not interpreted as adaptation or inherited state.

## Vid-source → HC2 validation

| Method | Source vIoU | Source Δv (95% CI) | Source tIoU | Source sIoU | Source R@.3 | Source R@.5 | Query vIoU | Query R@.3 | Query R@.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source Only | 21.072 | 0.000 [0.000, 0.000] | 41.516 | 49.400 | 30.536 | 8.899 | 20.017 | 27.570 | 7.639 |
| DINO-Refine | 21.888 | 0.817 [0.172, 1.443] | 41.516 | 44.761 | 31.623 | 10.422 | 20.673 | 28.920 | 9.075 |
| Target-trained reference | 29.895 | 8.824 [7.804, 9.850] | 57.835 | 49.903 | 48.430 | 14.970 | 29.056 | 45.951 | 14.302 |
| TENT-STVG | 17.437 | -3.635 [-4.258, -3.048] | 34.580 | 49.293 | 21.376 | 5.906 | 16.598 | 20.132 | 5.035 |
| SAR-STVG | 20.994 | -0.077 [-0.120, -0.040] | 41.370 | 49.399 | 30.471 | 8.653 | 19.939 | 27.465 | 7.486 |
| EATA-STVG (one-direction supplement) | 21.072 | 0.000 [-0.000, 0.001] | 41.517 | 49.400 | 30.536 | 8.899 | 20.017 | 27.570 | 7.639 |

### State transfer and negative tails

| Method | Inherited Δv | Current Δv | Gross source gains | Gross source losses | Source harm >5pp | Source harm >20pp | Updates / arrivals | Recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO-Refine | 0.000 | 0.817 | 2.076 | 1.259 | 18 | 0 | 0 / 10446 | 0 |
| TENT-STVG | -3.627 | -0.008 | 0.420 | 4.055 | 75 | 2 | 10446 / 10446 | 0 |
| SAR-STVG | -0.077 | 0.000 | 0.024 | 0.101 | 0 | 0 | 15 / 10446 | 0 |
| EATA-STVG | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 3 / 10446 | 0 |

The inherited effect compares saved Before with frozen Source; the current effect compares saved After with Before. Spatial and temporal terms telescope exactly to the total difference. Reference checkpoint differences are not interpreted as adaptation or inherited state.

## Verification and interpretation

All 120,726 selected prediction payloads and 13,785 shared source inputs were checked against independent GT-free byte receipts before this stage read labels. Original barriers and science/runtime locks remain unchanged. The new evaluation barrier records the limited authorized scope, including the unfinished EATA direction.

Every metric readout was checked with an independent dense geometry implementation. Extracted official evaluator functions were also compared on ordinal multiples of 101 in both datasets, and synthetic contracts cover no extrapolation, clipping, half-open temporal endpoints and strict thresholds. Recorded parameter and payload chains were checked for every online arrival; recorded optimizer snapshots and qualification gradient arithmetic were verified. Full production gradients and model Jacobians were not retained and were not replayed.

The independent root auditor reproduced query-then-source means, source-clustered official query means, 10,000 paired source bootstrap intervals, negative tails, and scalar telescoping identities. Per-order, quality, duration, track motion, query-type, source-exposure, positive/failure chains and cost details are in the accompanying anonymous scalar files. Quality strata use only admitted frames with known GT support; unknown frames stay explicit.

TENT/SAR/EATA are fixed STVG ports with decoder LayerNorm scope (19,968 affine coordinates), not unchanged classification implementations. Qualification arithmetic and complete saved-state integrity do not prove scientific usefulness; interpret measured effects and actual update coverage together.

EATA inference remains user-paused. Evaluation completion does not imply the original complete baseline table, full OPD results or all paper stages have completed. No scores were used to select a new configuration.
