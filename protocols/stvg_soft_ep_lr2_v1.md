# Soft EP-OPD LR×2 protocol

## Scope and status

This is the fixed 32-source LR×2 matched comparison under `artifacts/stvg_soft_ep_opd_v1/lr2`. The roster contains 16 HC-STVG-v2 and 16 VidSTG program aliases. These sources have historical development exposure; the trial is not a blind confirmation. This document does not authorize new fitting, Teacher/Temporal inference, bootstrap, or a production-method change.

## Intervention and arms

The trial holds `mu=0.25` and applies a learning-rate multiplier of 2.0. The design lock holds sigma, tau, steps, writeback, sample count, response settings, references, and temporal intervals fixed. The ten arms are F, T, S_M0, TS_M0, S_M1, TS_M1, S_M2, TS_M2, S_M0matched, and TS_M0matched. S/TS denotes spatial-only versus temporal-plus-spatial decoding. `M0` is the exact historical parent reuse; `M0matched` is a fresh validated OPD fit at the LR×2 setting. M1 and M2 are also freshly fitted at LR×2. The LR×1 M1/M2 outputs are comparison baselines only. The T+S contrasts M2−M1 and M2−M0matched are both reported within LR2; historical M0 is a prior-LR reuse and cannot replace the same-LR validated OPD control.

## Outcomes and aggregation

Retain all 320 case-arm rows, 100 domain-by-arm-by-metric summaries, 140 within-LR2 paired cells, and 40 matched LR2-vs-LR1 cross-trial cells. Metrics remain separate: full-tube vIoU, tIoU, dense sIoU, supervised-anchor sIoU, and unsupervised-frame sIoU. Null values remain unknown; positive, negative, neutral, and unknown case tails remain explicit. No bootstrap is used.

## Execution and provenance

The exporter first requires `CPU_COMPLETION.json` with complete status and matching row, summary, and contrast counts. It then verifies the runtime and protected hashes, student/prediction seals, pre-GT readback, GT-exposure binding, saved-math audit, and the LR×1/parent comparison bindings before reading score files. It publishes only program aliases, scalar metrics/counts, and hashes; it drops frame IDs, coordinates, tensors, raw queries, and source payloads. The explicit public manifest lists the seven LR2 execution sources, the independent math auditor, and the reporting/export source.

## Interpretation

This is a finite exposed-development matched comparison, not an unseen-source estimate and not a method qualification. Reused upstream Teacher/Temporal cost is unknown and must not be described as zero deployment cost. The inherited finite-difference diagnostic discrepancy range is 7.64–19.58%; `high_accuracy_complete_decoder_FD_qualification=false`, and the saved-math audit does not qualify a high-accuracy complete-decoder Jacobian. The separate 128-source coordinate search remains outside this result unless its own completed score and seal are published.
