# Finite P0 execution

Use `.conda/tubedetr/bin/python -B` from the workspace.
GPU expert phases `smoke`, `A_ROI`, `GT_ROI` use the original UniversalVTG
environment `.venv-exost/bin/python -B` (mamba/triton/PE dependencies). CPU
preparation, spatial-label extraction, scoring and audit use tubedetr.
The initial tubedetr loader failure occurred before expert construction and
is preserved in `recovery/runtime_environment_001`; revision001 binds the
correct runtime and previously implicit UniversalVTG library dependencies.
Revision002 restores the original HC output-timing decoder binding for the
frame-freeze donor, after the exact old pixel hash blocked that condition.
Six already valid crop outputs are preserved and exactly reused; the failed
attempt and its runtime are counted, with no scientific-pixel change.

1. Run `scripts/test_tastvg_spatial_guided_temporal_p0_v1.py`.
2. `scripts/run_tastvg_spatial_guided_temporal_p0_v1.py prepare` locks cohort, old payloads/caches, weights, code and crop geometry convention.
3. Run `smoke`; root reviews two Full reencoding controls and crop format/resource receipts and writes SMOKE_ROOT_ACCEPTANCE only on pass.
4. Run `A_ROI`, then `prepare_GT` (extract spatial truth only after deployable seal), then `GT_ROI`. Each is bounded; no infinite polling, no other experiment automatically starts.
5. `scripts/score_tastvg_spatial_guided_temporal_p0_v1.py` after GLOBAL_PREDICTION_BARRIER; `scripts/audit_tastvg_spatial_guided_temporal_p0_v1.py root` independently verifies old raw caches, A states, GT box interpolation, features/proposals/selection and teacher metrics, counts, CI and all seals. Public scalar audit is separately runnable with `public <results_dir>`.
6. Report all panels/clean/orders/costs and positive/negative cases. Update RESEARCH_HISTORY and check/snapshot/check, export code/protocol/anonymous scalars/plots to Zonglin-He/A, verify remote bytes and public audit. Keep weights/media/raw crops/boxes/proposals/annotations/private caches out of GitHub.

Do not call the worker exit or smoke a scientific completion. Old experiment files stay immutable. Preserve any engineering failure and lock revision before a repaired retry; do not restart old queues or launch DTA on this audit's result.
