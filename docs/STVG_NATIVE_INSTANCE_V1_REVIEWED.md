# Native detector–tracker and semantic-reference spatial update: fixed eight-source readout

## Scope and evidence

This report covers eight historically exposed development sources, four from HC and four from VidSTG. It is not an independent confirmation set. Predictions were sealed before the final metric read. The final evaluation uses the official dataset annotations; no metric below is a GT-free estimate. The model output and training/teacher traces are not treated as proof that the chosen detector track is the semantically correct instance.

The five primary arms are Temporal-only (T), the native teacher's direct track geometry, and T combined with direct, IoU, or reliability-weighted-IoU spatial adaptation. F and the three spatial-only arms are attribution controls. F/T are the preserved source baselines; the T+S arms use the same old temporal interval while changing the spatial readout. Every source-arm row, including unknown metrics and exact no-update rows, is retained below. Domain summaries remain separate; no pooled cross-domain estimate is reported.

The R1 checkpoint is the STVG-R1 “7B” model family, with an inspected parameter count of 8,292,166,656 (about 8.292 billion parameters). The native spatial reference path also runs YOLOv12 and SAM2 tracking over all target categories, so its cost includes a separate detection/segmentation stage. The YOLO checkpoint compatibility path uses half-precision fused QKV through PyTorch SDPA and does not claim bitwise equivalence to the upstream flash-attention path. Tracking used a bounded two-frame decode cache. These are implementation constraints that affect reproducibility and resource cost.

The reliability-weighted-IoU rule uses observed mask presence and adjacent-mask IoU as a proxy weight. Rapid motion can lower adjacent-mask IoU even for a valid moving target, so the value can penalize motion. It is not calibrated confidence, identity probability, or a semantic correctness score.

The sealed metric set does not estimate full candidate-pool detection recall. A valid native R1 selection can still be scored even when no fixed QA-center mask supplies a spatial update anchor; that case remains an exact no-update student branch, not evidence that the teacher abstained.

## Per-domain arm summaries

Metrics are percentages of their 0-to-1 values. The parenthesized counts are `valid / unknown` among four historical sources. Unknown cells remain unknown; they are not imputed as zero.

| Domain | Arm | vIoU % (valid/unknown) | tIoU % (valid/unknown) | dense sIoU % (valid/unknown) | supervised sIoU % (valid/unknown) | unsupervised sIoU % (valid/unknown) |
|---|---|---:|---:|---:|---:|---:|
| HC | Temporal only (T) | 31.286 (4/0) | 60.486 (4/0) | 47.895 (4/0) | 60.801 (3/1) | 47.851 (4/0) |
| HC | Native teacher direct | 27.683 (4/0) | 45.509 (4/0) | 47.768 (4/0) | 78.299 (3/1) | 47.721 (4/0) |
| HC | T + direct spatial update | 29.907 (4/0) | 60.486 (4/0) | 42.315 (4/0) | 65.382 (3/1) | 42.191 (4/0) |
| HC | T + IoU spatial update | 32.436 (4/0) | 60.486 (4/0) | 44.358 (4/0) | 54.994 (3/1) | 44.324 (4/0) |
| HC | T + reliability-weighted IoU | 32.432 (4/0) | 60.486 (4/0) | 44.346 (4/0) | 55.000 (3/1) | 44.312 (4/0) |
| HC | Frozen source (F) | 22.210 (4/0) | 46.415 (4/0) | 47.895 (4/0) | 60.801 (3/1) | 47.851 (4/0) |
| HC | Direct spatial update | 21.887 (4/0) | 46.415 (4/0) | 42.315 (4/0) | 65.382 (3/1) | 42.191 (4/0) |
| HC | IoU spatial update | 21.840 (4/0) | 46.415 (4/0) | 44.358 (4/0) | 54.994 (3/1) | 44.324 (4/0) |
| HC | Reliability-weighted IoU update | 21.829 (4/0) | 46.415 (4/0) | 44.346 (4/0) | 55.000 (3/1) | 44.312 (4/0) |
| VidSTG | Temporal only (T) | 22.442 (4/0) | 40.905 (4/0) | 48.458 (4/0) | 20.274 (1/3) | 48.478 (4/0) |
| VidSTG | Native teacher direct | 15.787 (2/2) | 24.228 (2/2) | 38.591 (2/2) | 79.336 (1/3) | 38.560 (2/2) |
| VidSTG | T + direct spatial update | 25.939 (4/0) | 40.905 (4/0) | 61.566 (4/0) | 76.249 (1/3) | 61.561 (4/0) |
| VidSTG | T + IoU spatial update | 25.678 (4/0) | 40.905 (4/0) | 60.679 (4/0) | 74.930 (1/3) | 60.658 (4/0) |
| VidSTG | T + reliability-weighted IoU | 25.663 (4/0) | 40.905 (4/0) | 60.619 (4/0) | 74.572 (1/3) | 60.598 (4/0) |
| VidSTG | Frozen source (F) | 14.490 (4/0) | 27.778 (4/0) | 48.458 (4/0) | 20.274 (1/3) | 48.478 (4/0) |
| VidSTG | Direct spatial update | 17.796 (4/0) | 27.778 (4/0) | 61.566 (4/0) | 76.249 (1/3) | 61.561 (4/0) |
| VidSTG | IoU spatial update | 17.550 (4/0) | 27.778 (4/0) | 60.679 (4/0) | 74.930 (1/3) | 60.658 (4/0) |
| VidSTG | Reliability-weighted IoU update | 17.536 (4/0) | 27.778 (4/0) | 60.619 (4/0) | 74.572 (1/3) | 60.598 (4/0) |

## Paired contrasts

Positive, negative, neutral, and unknown counts are source-level pairs. The 95% intervals use 10000 paired source bootstrap replicates with seed 20261014, separately within each domain. With four historically exposed sources per domain these are exploratory descriptive intervals, not a confirmation test, selection gate, or pooled estimate.

| Domain | Contrast | Metric | Paired n | Positive | Negative | Neutral | Unknown | Mean delta (pp) | Median delta (pp) | 95% paired bootstrap CI (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC | T_direct_minus_T | vIoU | 4 | 1 | 3 | 0 | 0 | -1.379 | -1.998 | [-9.507, +7.368] |
| HC | T_direct_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_direct_minus_T | dense sIoU | 4 | 1 | 3 | 0 | 0 | -5.580 | -7.812 | [-15.938, +6.069] |
| HC | T_direct_minus_T | supervised sIoU | 3 | 2 | 1 | 0 | 1 | +4.581 | +1.387 | [-1.815, +14.171] |
| HC | T_direct_minus_T | unsupervised sIoU | 4 | 1 | 3 | 0 | 0 | -5.660 | -7.929 | [-16.046, +5.950] |
| HC | T_iou_minus_T | vIoU | 4 | 1 | 2 | 1 | 0 | +1.150 | -0.765 | [-2.947, +7.163] |
| HC | T_iou_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_iou_minus_T | dense sIoU | 4 | 1 | 2 | 1 | 0 | -3.537 | -0.606 | [-18.268, +8.262] |
| HC | T_iou_minus_T | supervised sIoU | 3 | 2 | 1 | 0 | 1 | -5.806 | +0.518 | [-31.121, +13.184] |
| HC | T_iou_minus_T | unsupervised sIoU | 4 | 1 | 2 | 1 | 0 | -3.527 | -0.610 | [-18.169, +8.197] |
| HC | T_minus_F | vIoU | 4 | 3 | 1 | 0 | 0 | +9.076 | +13.774 | [-3.420, +17.621] |
| HC | T_minus_F | tIoU | 4 | 3 | 1 | 0 | 0 | +14.071 | +26.108 | [-12.053, +28.396] |
| HC | T_minus_F | dense sIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_minus_F | supervised sIoU | 3 | 0 | 0 | 3 | 1 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_minus_F | unsupervised sIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_reliability_weighted_iou_minus_T | vIoU | 4 | 1 | 2 | 1 | 0 | +1.146 | -0.764 | [-2.941, +7.143] |
| HC | T_reliability_weighted_iou_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | T_reliability_weighted_iou_minus_T | dense sIoU | 4 | 1 | 2 | 1 | 0 | -3.549 | -0.605 | [-18.282, +8.239] |
| HC | T_reliability_weighted_iou_minus_T | supervised sIoU | 3 | 2 | 1 | 0 | 1 | -5.801 | +0.522 | [-31.073, +13.147] |
| HC | T_reliability_weighted_iou_minus_T | unsupervised sIoU | 4 | 1 | 2 | 1 | 0 | -3.539 | -0.608 | [-18.183, +8.175] |
| HC | direct_minus_F | vIoU | 4 | 1 | 3 | 0 | 0 | -0.323 | -1.812 | [-4.725, +5.567] |
| HC | direct_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | direct_minus_F | dense sIoU | 4 | 1 | 3 | 0 | 0 | -5.580 | -7.812 | [-15.938, +6.069] |
| HC | direct_minus_F | supervised sIoU | 3 | 2 | 1 | 0 | 1 | +4.581 | +1.387 | [-1.815, +14.171] |
| HC | direct_minus_F | unsupervised sIoU | 4 | 1 | 3 | 0 | 0 | -5.660 | -7.929 | [-16.046, +5.950] |
| HC | iou_minus_F | vIoU | 4 | 1 | 2 | 1 | 0 | -0.371 | -0.716 | [-5.802, +5.406] |
| HC | iou_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | iou_minus_F | dense sIoU | 4 | 1 | 2 | 1 | 0 | -3.537 | -0.606 | [-18.268, +8.262] |
| HC | iou_minus_F | supervised sIoU | 3 | 2 | 1 | 0 | 1 | -5.806 | +0.518 | [-31.121, +13.184] |
| HC | iou_minus_F | unsupervised sIoU | 4 | 1 | 2 | 1 | 0 | -3.527 | -0.610 | [-18.169, +8.197] |
| HC | native_teacher_direct_minus_T | vIoU | 4 | 1 | 3 | 0 | 0 | -3.603 | -2.742 | [-18.233, +10.164] |
| HC | native_teacher_direct_minus_T | tIoU | 4 | 2 | 2 | 0 | 0 | -14.977 | -5.727 | [-47.297, +11.048] |
| HC | native_teacher_direct_minus_T | dense sIoU | 4 | 2 | 2 | 0 | 0 | -0.127 | -2.192 | [-26.657, +26.404] |
| HC | native_teacher_direct_minus_T | supervised sIoU | 3 | 3 | 0 | 0 | 1 | +17.499 | +14.723 | [+8.033, +29.740] |
| HC | native_teacher_direct_minus_T | unsupervised sIoU | 4 | 2 | 2 | 0 | 0 | -0.130 | -2.306 | [-26.711, +26.456] |
| HC | reliability_weighted_iou_minus_F | vIoU | 4 | 1 | 2 | 1 | 0 | -0.381 | -0.715 | [-5.815, +5.387] |
| HC | reliability_weighted_iou_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| HC | reliability_weighted_iou_minus_F | dense sIoU | 4 | 1 | 2 | 1 | 0 | -3.549 | -0.605 | [-18.282, +8.239] |
| HC | reliability_weighted_iou_minus_F | supervised sIoU | 3 | 2 | 1 | 0 | 1 | -5.801 | +0.522 | [-31.073, +13.147] |
| HC | reliability_weighted_iou_minus_F | unsupervised sIoU | 4 | 1 | 2 | 1 | 0 | -3.539 | -0.608 | [-18.183, +8.175] |
| VidSTG | T_direct_minus_T | vIoU | 4 | 1 | 0 | 3 | 0 | +3.497 | +0.000 | [+0.000, +10.491] |
| VidSTG | T_direct_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_direct_minus_T | dense sIoU | 4 | 1 | 0 | 3 | 0 | +13.108 | +0.000 | [+0.000, +39.325] |
| VidSTG | T_direct_minus_T | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +55.975 | +55.975 | [+55.975, +55.975] |
| VidSTG | T_direct_minus_T | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +13.083 | +0.000 | [+0.000, +39.249] |
| VidSTG | T_iou_minus_T | vIoU | 4 | 1 | 0 | 3 | 0 | +3.236 | +0.000 | [+0.000, +9.709] |
| VidSTG | T_iou_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_iou_minus_T | dense sIoU | 4 | 1 | 0 | 3 | 0 | +12.221 | +0.000 | [+0.000, +36.664] |
| VidSTG | T_iou_minus_T | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +54.656 | +54.656 | [+54.656, +54.656] |
| VidSTG | T_iou_minus_T | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +12.180 | +0.000 | [+0.000, +36.540] |
| VidSTG | T_minus_F | vIoU | 4 | 4 | 0 | 0 | 0 | +7.952 | +7.946 | [+0.624, +15.279] |
| VidSTG | T_minus_F | tIoU | 4 | 4 | 0 | 0 | 0 | +13.127 | +12.765 | [+1.574, +24.679] |
| VidSTG | T_minus_F | dense sIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_minus_F | supervised sIoU | 1 | 0 | 0 | 1 | 3 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_minus_F | unsupervised sIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_reliability_weighted_iou_minus_T | vIoU | 4 | 1 | 0 | 3 | 0 | +3.221 | +0.000 | [+0.000, +9.663] |
| VidSTG | T_reliability_weighted_iou_minus_T | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | T_reliability_weighted_iou_minus_T | dense sIoU | 4 | 1 | 0 | 3 | 0 | +12.161 | +0.000 | [+0.000, +36.483] |
| VidSTG | T_reliability_weighted_iou_minus_T | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +54.298 | +54.298 | [+54.298, +54.298] |
| VidSTG | T_reliability_weighted_iou_minus_T | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +12.120 | +0.000 | [+0.000, +36.361] |
| VidSTG | direct_minus_F | vIoU | 4 | 1 | 0 | 3 | 0 | +3.306 | +0.000 | [+0.000, +9.919] |
| VidSTG | direct_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | direct_minus_F | dense sIoU | 4 | 1 | 0 | 3 | 0 | +13.108 | +0.000 | [+0.000, +39.325] |
| VidSTG | direct_minus_F | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +55.975 | +55.975 | [+55.975, +55.975] |
| VidSTG | direct_minus_F | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +13.083 | +0.000 | [+0.000, +39.249] |
| VidSTG | iou_minus_F | vIoU | 4 | 1 | 0 | 3 | 0 | +3.060 | +0.000 | [+0.000, +9.179] |
| VidSTG | iou_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | iou_minus_F | dense sIoU | 4 | 1 | 0 | 3 | 0 | +12.221 | +0.000 | [+0.000, +36.664] |
| VidSTG | iou_minus_F | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +54.656 | +54.656 | [+54.656, +54.656] |
| VidSTG | iou_minus_F | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +12.180 | +0.000 | [+0.000, +36.540] |
| VidSTG | native_teacher_direct_minus_T | vIoU | 2 | 1 | 1 | 0 | 2 | +9.008 | +9.008 | [-6.892, +24.908] |
| VidSTG | native_teacher_direct_minus_T | tIoU | 2 | 1 | 1 | 0 | 2 | +2.450 | +2.450 | [-8.124, +13.024] |
| VidSTG | native_teacher_direct_minus_T | dense sIoU | 2 | 1 | 1 | 0 | 2 | +2.925 | +2.925 | [-48.260, +54.110] |
| VidSTG | native_teacher_direct_minus_T | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +59.062 | +59.062 | [+59.062, +59.062] |
| VidSTG | native_teacher_direct_minus_T | unsupervised sIoU | 2 | 1 | 1 | 0 | 2 | +2.854 | +2.854 | [-48.260, +53.968] |
| VidSTG | reliability_weighted_iou_minus_F | vIoU | 4 | 1 | 0 | 3 | 0 | +3.045 | +0.000 | [+0.000, +9.136] |
| VidSTG | reliability_weighted_iou_minus_F | tIoU | 4 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | [+0.000, +0.000] |
| VidSTG | reliability_weighted_iou_minus_F | dense sIoU | 4 | 1 | 0 | 3 | 0 | +12.161 | +0.000 | [+0.000, +36.483] |
| VidSTG | reliability_weighted_iou_minus_F | supervised sIoU | 1 | 1 | 0 | 0 | 3 | +54.298 | +54.298 | [+54.298, +54.298] |
| VidSTG | reliability_weighted_iou_minus_F | unsupervised sIoU | 4 | 1 | 0 | 3 | 0 | +12.120 | +0.000 | [+0.000, +36.361] |

## All source-level predictions and readouts

Scores are percentages. The operation columns retain support, coverage, optimizer, and final-decode status so an unknown result is distinguishable from an exact no-update branch.

| Domain | Source | Arm | Status | vIoU | tIoU | dense sIoU | supervised sIoU | unsupervised sIoU | Source-grid positions | Supervised anchors | Unsupervised grid positions | GT-known dense / supervised / unsupervised frames | Present frames / total | Interval-present / interval total | Fit status | Steps | Final re-decodes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| HC | HC01 | T | scored | 6.916 | 15.459 | 34.730 | 45.254 | 34.524 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC01 | Native direct | scored | 21.765 | 31.939 | 75.351 | 74.995 | 75.358 | 65 | 4 | 61 | 208 / 4 / 204 | 321 / 500 | 131 / 140 | not_a_fit_arm | 0 | 0 |
| HC | HC01 | T + direct | scored | 5.999 | 15.459 | 21.915 | 43.440 | 21.493 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC01 | T + IoU | scored | 2.986 | 15.459 | 10.373 | 14.134 | 10.299 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC01 | T + weighted IoU | scored | 2.994 | 15.459 | 10.354 | 14.182 | 10.279 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC01 | F | scored | 16.958 | 40.693 | 34.730 | 45.254 | 34.524 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC01 | Direct | scored | 16.557 | 40.693 | 21.915 | 43.440 | 21.493 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC01 | IoU | scored | 9.222 | 40.693 | 10.373 | 14.134 | 10.299 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC01 | Weighted IoU | scored | 9.204 | 40.693 | 10.354 | 14.182 | 10.279 | 65 | 4 | 61 | 208 / 4 / 204 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | T | scored | 52.265 | 82.979 | 64.913 | 62.500 | 65.027 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC02 | Native direct | scored | 50.673 | 65.909 | 77.100 | 77.223 | 77.094 | 65 | 4 | 61 | 88 / 4 / 84 | 216 / 600 | 125 / 132 | not_a_fit_arm | 0 | 0 |
| HC | HC02 | T + direct | scored | 63.115 | 82.979 | 77.276 | 76.672 | 77.305 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | T + IoU | scored | 62.326 | 82.979 | 76.333 | 75.684 | 76.363 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | T + weighted IoU | scored | 62.299 | 82.979 | 76.301 | 75.647 | 76.332 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | F | scored | 35.818 | 58.252 | 64.913 | 62.500 | 65.027 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC02 | Direct | scored | 44.315 | 58.252 | 77.276 | 76.672 | 77.305 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | IoU | scored | 43.503 | 58.252 | 76.333 | 75.684 | 76.363 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC02 | Weighted IoU | scored | 43.478 | 58.252 | 76.301 | 75.647 | 76.332 | 65 | 4 | 61 | 88 / 4 / 84 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC03 | T | scored | 23.780 | 64.935 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC03 | Native direct | scored | 0.000 | 0.000 | 0.000 | unknown | 0.000 | 65 | 1 | 64 | 201 / 0 / 201 | 323 / 600 | 85 / 139 | not_a_fit_arm | 0 | 0 |
| HC | HC03 | T + direct | scored | 11.410 | 64.935 | 17.684 | unknown | 17.684 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC03 | T + IoU | scored | 23.780 | 64.935 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 0 | 1 |
| HC | HC03 | T + weighted IoU | scored | 23.780 | 64.935 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 0 | 1 |
| HC | HC03 | F | scored | 12.677 | 35.632 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC03 | Direct | scored | 6.510 | 35.632 | 17.684 | unknown | 17.684 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC03 | IoU | scored | 12.677 | 35.632 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 0 | 1 |
| HC | HC03 | Weighted IoU | scored | 12.677 | 35.632 | 36.744 | unknown | 36.744 | 65 | 1 | 64 | 201 / 0 / 201 | unknown | unknown | actual_fit | 0 | 1 |
| HC | HC04 | T | scored | 42.184 | 78.571 | 55.193 | 74.647 | 55.109 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC04 | Native direct | scored | 38.292 | 84.188 | 38.622 | 82.680 | 38.432 | 65 | 1 | 64 | 232 / 1 / 231 | 116 / 500 | 116 / 200 | not_a_fit_arm | 0 | 0 |
| HC | HC04 | T + direct | scored | 39.106 | 78.571 | 52.384 | 76.034 | 52.281 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC04 | T + IoU | scored | 40.653 | 78.571 | 53.981 | 75.165 | 53.889 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC04 | T + weighted IoU | scored | 40.656 | 78.571 | 53.984 | 75.170 | 53.892 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC04 | F | scored | 23.388 | 51.082 | 55.193 | 74.647 | 55.109 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| HC | HC04 | Direct | scored | 20.165 | 51.082 | 52.384 | 76.034 | 52.281 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC04 | IoU | scored | 21.956 | 51.082 | 53.981 | 75.165 | 53.889 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| HC | HC04 | Weighted IoU | scored | 21.957 | 51.082 | 53.984 | 75.170 | 53.892 | 65 | 1 | 64 | 232 / 1 / 231 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V01 | T | scored | 51.733 | 78.519 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V01 | Native direct | unknown | unknown | unknown | unknown | unknown | unknown | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V01 | T + direct | scored | 51.733 | 78.519 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V01 | T + IoU | scored | 51.733 | 78.519 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V01 | T + weighted IoU | scored | 51.733 | 78.519 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V01 | F | scored | 36.181 | 54.615 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V01 | Direct | scored | 36.181 | 54.615 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V01 | IoU | scored | 36.181 | 54.615 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V01 | Weighted IoU | scored | 36.181 | 54.615 | 63.491 | unknown | 63.491 | 27 | 0 | 27 | 110 / 0 / 110 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | T | scored | 24.477 | 41.546 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V02 | Native direct | unknown | unknown | unknown | unknown | unknown | unknown | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V02 | T + direct | scored | 24.477 | 41.546 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | T + IoU | scored | 24.477 | 41.546 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | T + weighted IoU | scored | 24.477 | 41.546 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | F | scored | 9.470 | 16.091 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V02 | Direct | scored | 9.470 | 16.091 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | IoU | scored | 9.470 | 16.091 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V02 | Weighted IoU | scored | 9.470 | 16.091 | 59.008 | unknown | 59.008 | 200 | 0 | 200 | 1591 / 0 / 1591 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | T | scored | 6.892 | 15.670 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V03 | Native direct | scored | 0.000 | 7.546 | 0.000 | unknown | 0.000 | 200 | 0 | 200 | 123 / 0 / 123 | 478 / 1647 | 474 / 1631 | not_a_fit_arm | 0 | 0 |
| VidSTG | V03 | T + direct | scored | 6.892 | 15.670 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | T + IoU | scored | 6.892 | 15.670 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | T + weighted IoU | scored | 6.892 | 15.670 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | F | scored | 6.007 | 14.043 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V03 | Direct | scored | 6.007 | 14.043 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | IoU | scored | 6.007 | 14.043 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V03 | Weighted IoU | scored | 6.007 | 14.043 | 48.260 | unknown | 48.260 | 200 | 0 | 200 | 123 / 0 / 123 | unknown | unknown | exact_no_update | 0 | 0 |
| VidSTG | V04 | T | scored | 6.666 | 27.885 | 23.072 | 20.274 | 23.152 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V04 | Native direct | scored | 31.574 | 40.909 | 77.181 | 79.336 | 77.120 | 21 | 4 | 17 | 36 / 1 / 35 | 121 / 121 | 89 / 89 | not_a_fit_arm | 0 | 0 |
| VidSTG | V04 | T + direct | scored | 20.654 | 27.885 | 75.505 | 76.249 | 75.484 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V04 | T + IoU | scored | 19.611 | 27.885 | 71.957 | 74.930 | 71.872 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V04 | T + weighted IoU | scored | 19.551 | 27.885 | 71.715 | 74.572 | 71.634 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V04 | F | scored | 6.303 | 26.364 | 23.072 | 20.274 | 23.152 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | not_a_fit_arm | 0 | 0 |
| VidSTG | V04 | Direct | scored | 19.527 | 26.364 | 75.505 | 76.249 | 75.484 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V04 | IoU | scored | 18.542 | 26.364 | 71.957 | 74.930 | 71.872 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |
| VidSTG | V04 | Weighted IoU | scored | 18.484 | 26.364 | 71.715 | 74.572 | 71.634 | 21 | 4 | 17 | 36 / 1 / 35 | unknown | unknown | actual_fit | 5 | 1 |

## Measured cost and reuse

Tracking wall time for a reused receipt is the actual historical tracking time; it is not free and is not this run's incremental tracking time. Current incremental subtotal excludes reused tracking and includes only current tracking where applicable, teacher worker wall time, and student source capture. Tracking-cache lookup overhead is not separately timed. Algorithm `seconds` covers that arm's fit and full final re-decode. Exact no-update receipts may have no timing, shown as `not measured`; no missing duration is replaced with zero. Domain fit totals distinguish timed fits, exact no-update arms, and untimed fits. Teacher model-load, preprocessing, and generation measurements are components of worker time and must not be added to it. The report does not count a historical cached T reconstruction as free: its retrieval/reinference latency is not measured in this scope and is excluded from all-in deployment latency.

| Domain | Source | Tracking history | Tracking wall (s) | Teacher worker wall (s) | Model load (s) | Preprocess (s) | Generation (s) | Source capture (s) | Direct fit+decode (s) | IoU fit+decode (s) | Weighted-IoU fit+decode (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HC | HC01 | reused historical | 566.408 | 152.778 | 5.876 | 0.576 | 134.384 | 11.448 | 2.873 | 2.701 | 2.740 |
| HC | HC02 | current run | 347.298 | 148.156 | 6.706 | 0.969 | 122.981 | 9.388 | 2.843 | 2.769 | 2.718 |
| HC | HC03 | current run | 331.608 | 218.774 | 6.023 | 1.061 | 194.931 | 9.511 | 2.876 | 2.685 | 2.650 |
| HC | HC04 | current run | 130.864 | 244.738 | 14.666 | 1.506 | 208.969 | 9.406 | 2.808 | 2.754 | 2.778 |
| VidSTG | V01 | current run | 25.941 | 136.458 | 5.640 | 0.226 | 114.839 | 10.806 | not measured | not measured | not measured |
| VidSTG | V02 | current run | 319.449 | 135.259 | 6.361 | 2.354 | 109.781 | 10.305 | not measured | not measured | not measured |
| VidSTG | V03 | reused historical | 1172.331 | 129.693 | 7.655 | 0.801 | 102.676 | 9.825 | not measured | not measured | not measured |
| VidSTG | V04 | current run | 27.530 | 150.531 | 8.446 | 0.180 | 125.414 | 9.113 | 2.738 | 2.688 | 2.611 |

Domain totals follow; timed fit totals include only receipts with an actual measured duration. The current incremental subtotal is reported before spatial fit/redecode time, which remains in its own algorithm columns.

| Domain | Historical tracking total (s) | Reused track count | Current-run tracking total (s) | Teacher worker total (s) | Student source capture total (s) | Direct timed fit+decode (s; timed / missing) | IoU timed fit+decode (s; timed / missing) | Weighted-IoU timed fit+decode (s; timed / missing) | Current incremental subtotal before fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HC | 1376.177 (4/4 timed) | 1 | 809.769 (3/3 timed current runs) | 764.446 (4/4 timed) | 39.753 (4/4 timed) | 11.400 (timed 4; exact no-update 0; untimed fit 0) | 10.909 (timed 4; exact no-update 0; untimed fit 0) | 10.886 (timed 4; exact no-update 0; untimed fit 0) | 1613.967 (4/4) |
| VidSTG | 1545.252 (4/4 timed) | 1 | 372.920 (3/3 timed current runs) | 551.942 (4/4 timed) | 40.050 (4/4 timed) | 2.738 (timed 1; exact no-update 3; untimed fit 0) | 2.688 (timed 1; exact no-update 3; untimed fit 0) | 2.611 (timed 1; exact no-update 3; untimed fit 0) | 964.912 (4/4) |

One initial teacher-worker attempt stopped at the configured 4-GiB free-GPU-memory guard before any teacher result was accepted; this was not a CUDA out-of-memory event. No accepted teacher output was saved for that attempt; any partial generated tokens and its generation-call count are unknown. The later worker configuration changed the placement budgets from 20/4 GiB GPU/CPU to 12/8 GiB while retaining BF16, prompt, visual input, and token cap. Its failed-attempt resource cost is not represented by the successful per-source receipt totals above. Those totals therefore describe the accepted successful path, not every attempted operation.

## Limits and decision boundary

The panel is small and historically exposed. Per-domain means, source-level variation, negative cases, and unknowns must be read together. A favorable arm mean does not establish stable performance or transfer to unseen sources. Detector track availability and R1's response are not a semantic identity verification: errors in detection, mask propagation, event interpretation, or target selection can remain even when the output parses and the mask track is well formed.

The direct native teacher arm measures the selected YOLO+SAM2/R1 output without spatial optimization. The adaptation arms use only their locked reference rule and original optimizer controls; the report does not select a new threshold or promote a method. The reliability proxy is a hand-defined weighting rule, not a learned or calibrated uncertainty estimate. Deployment choice remains governed by the project method registry, not this development report.

The score rows are bound to prediction barrier SHA `bd17360c62c4dcefe9a5708d52560fedcdcf279bbf90c69f57e2b8867424221b`. This report intentionally exposes scalar scores and anonymous aliases only; it does not export raw queries, media, masks, annotation coordinates, model weights, or per-case prediction tensors.


## Additional post-seal coverage and engineering readback

See [filtered instance-pool coverage](STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md) for the complete eight-source candidate geometry diagnostic. It is not raw detector recall or a GT-selected reference for fitting. The direct teacher baseline uses clip-relative R1 seconds mapped to the first source physical frame. A pre-publication reader initially mixed this with absolute source timestamps and stopped; its failed output and original source were retained, and the corrected clock reader leaves every teacher selection, sparse reference and student prediction unchanged. Restricted NumPy dtype and document-level roster-field dispatch corrections likewise affect CPU readback only.
