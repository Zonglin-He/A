# Soft EP-OPD LR×2 matched comparison

Status: LR2 post-seal anonymous report. The 32-source roster is development data with historical exposure; this is not a blind confirmation or a full-cohort result.

The trial fixes `mu=0.25` and doubles the learning rate while keeping the locked references, temporal intervals, sigma, tau, step count, writeback, sample count, and response settings fixed. It includes the historical exact-parent M0 arms (`S_M0`, `TS_M0`) and freshly fitted same-LR validated controls (`S_M0matched`, `TS_M0matched`); these are distinct conditions.

Sealed scoring contains 69 unique fit invocations across 23 non-empty-reference sources, 69 actual full-model readouts, and 23 source captures. There are 64 scorer-labeled exact-parent M0 arm rows; fit/readout counts are not multiplied by paired S/TS score rows. No LR1 M1/M2 fit was reused for LR2. The prior LR1 scores are used only as matched-case comparison baselines.

## Full-tube vIoU by domain

Means are percentages. The complete five-metric rows, all sign tails, unknown cases, and per-source deltas are in the JSON exports.

| Domain | Arm | Valid | Unknown | Mean vIoU (%) |
|---|---:|---:|---:|---:|
| HC-STVG-v2 | F | 16 | 0 | 32.5874 |
| HC-STVG-v2 | T | 16 | 0 | 34.0670 |
| HC-STVG-v2 | S_M0 | 16 | 0 | 33.1177 |
| HC-STVG-v2 | TS_M0 | 16 | 0 | 34.6436 |
| HC-STVG-v2 | S_M1 | 16 | 0 | 33.1177 |
| HC-STVG-v2 | TS_M1 | 16 | 0 | 34.6436 |
| HC-STVG-v2 | S_M2 | 16 | 0 | 33.1700 |
| HC-STVG-v2 | TS_M2 | 16 | 0 | 34.7162 |
| HC-STVG-v2 | S_M0matched | 16 | 0 | 32.8100 |
| HC-STVG-v2 | TS_M0matched | 16 | 0 | 34.3416 |
| VidSTG | F | 16 | 0 | 9.7014 |
| VidSTG | T | 16 | 0 | 9.9667 |
| VidSTG | S_M0 | 16 | 0 | 10.5156 |
| VidSTG | TS_M0 | 16 | 0 | 10.7842 |
| VidSTG | S_M1 | 16 | 0 | 10.2997 |
| VidSTG | TS_M1 | 16 | 0 | 10.5683 |
| VidSTG | S_M2 | 16 | 0 | 10.7281 |
| VidSTG | TS_M2 | 16 | 0 | 10.9868 |
| VidSTG | S_M0matched | 16 | 0 | 10.9737 |
| VidSTG | TS_M0matched | 16 | 0 | 11.2237 |

## Same-LR2 matched controls

These are paired T+S full-tube vIoU contrasts within LR2. `M0matched` is the freshly fitted validated OPD control at the same doubled learning rate. The historical `M0` rows reuse a prior-LR fit, so they do not replace this same-LR control.

| Domain | Contrast | Pairs | + / − / = / unknown | Mean Δ (pp) |
|---|---|---:|---:|---:|
| HC-STVG-v2 | T+S M2 − M1 | 16 | 4 / 5 / 7 / 0 | +0.072594 |
| HC-STVG-v2 | T+S M2 − M0matched | 16 | 7 / 2 / 7 / 0 | +0.374530 |
| VidSTG | T+S M2 − M1 | 16 | 4 / 3 / 9 / 0 | +0.418531 |
| VidSTG | T+S M2 − M0matched | 16 | 4 / 3 / 9 / 0 | -0.236829 |

M2 improves over M0matched in HC-STVG-v2 but declines in VidSTG; this does not establish a dual-domain gain over the same-LR validated OPD control.

## Matched comparison with the prior LR×1 trial

The baseline is the already sealed `mu=0.25` LR×1 trial. Each value is LR2 minus LR1 in percentage points; this is a cross-trial paired comparison, not an additional LR2 arm.

| Domain | Contrast | Metric | Pairs | + / − / = / unknown | Mean Δ (pp) |
|---|---|---|---:|---:|---:|
| HC-STVG-v2 | LR2_M1_vs_LR1_M1_S | m_vIoU | 16 | 5 / 4 / 7 / 0 | +0.1037 |
| HC-STVG-v2 | LR2_M1_vs_LR1_M1_TS | m_vIoU | 16 | 5 / 4 / 7 / 0 | +0.1189 |
| HC-STVG-v2 | LR2_M2_vs_LR1_M2_S | m_vIoU | 16 | 5 / 4 / 7 / 0 | +0.0172 |
| HC-STVG-v2 | LR2_M2_vs_LR1_M2_TS | m_vIoU | 16 | 4 / 5 / 7 / 0 | +0.0385 |
| VidSTG | LR2_M1_vs_LR1_M1_S | m_vIoU | 16 | 4 / 2 / 10 / 0 | +0.1865 |
| VidSTG | LR2_M1_vs_LR1_M1_TS | m_vIoU | 16 | 4 / 3 / 9 / 0 | +0.1917 |
| VidSTG | LR2_M2_vs_LR1_M2_S | m_vIoU | 16 | 4 / 2 / 10 / 0 | +0.4554 |
| VidSTG | LR2_M2_vs_LR1_M2_TS | m_vIoU | 16 | 4 / 3 / 9 / 0 | +0.4496 |

## Limits

All 32 cases are exposed development sources. The LR2 comparison does not establish unseen-source performance, method qualification, or a production change. No bootstrap was run. Unknown metric values remain null. The run uses the already fixed Teacher/Temporal references; no new expert calls were made in this trial, and the cost of the inherited upstream calls is unknown. The separate 128-source coordinate search is not represented here as a completed or scored result.

The inherited finite-difference diagnostic discrepancy range is 7.64–19.58%. `high_accuracy_complete_decoder_FD_qualification=false`: the saved-math audit here checks saved reward/q/loss/Adam/CG/rollback/equal-norm witnesses; it does not qualify a high-accuracy complete-decoder Jacobian.

The source/configuration hashes, exact arm rows, 100 domain-summary cells, 140 within-trial paired cells, 40 LR1 cross-trial cells, and execution counts are preserved in this directory.
Exact per-domain six-scalar settings, response parameters, Adam hyperparameters, active parameter count, and the LR1 M0 versus LR2 M0matched distinction are in `ANONYMOUS_CONFIGURATION.json`.
