# Authorized native-logit chart revision003: qualified and resumed

See docs/STVG_OPD_P1_CHART_REVISION003.md for the actual qualification,
fixed scientific scope and limitations. The earlier original failure remains
published under results/stvg_opd_p1_chart_failure/2026-10-09.

Only the rounded endpoint inverse-coordinate representation changes.
Interior values/gradients and native box readouts remain unchanged. No clamp,
hyperparameter tuning, skipped query, altered exploration or GT selection.
Six actual GPU fits cover the failed query twice and two ordinary controls
twice (one no-op and one informative 10-round fit). Full repaired fits are
bitwise identical; pre-failure rounds/updates match the retained reproduction.
A native-head-output VJP and independent full 10-round arithmetic pass.
The original dead worker did not serialize its fit, so no equality with its
lost memory or independent full decoder Jacobian is claimed.

Qualification accepted zero predictions. Formal first missing arrival1882
matches the actual qualified fit bitwise before being accepted, with exact
old-prefix linkage and independent writeback/reset checks. HC2's sealed
10,446 predictions are not rerun. VidSTG's suffix continues with the original
fixed configuration and three orders. Progress is a timestamped snapshot,
not full P1 or paper completion. Both directions and all 41,355 arrivals must
seal before P1 GT scoring; EATA remains paused.

Portable synthetic chart/gradient/trace contracts:

```bash
python -B scripts/test_stvg_opd_chart_revision003.py
```

The GPU/controller/readback scripts require the original private locked
research workspace; this export is inspectable code and safe scalar/hash
evidence, not a bundled dataset or model. Private frames, captions, GT,
weights, raw predictions, optimizer/gradient tensors and binary fits are
excluded. Timed qualification fit segments do not include every capture,
model load or replay overhead and are not total recovery or full-run costs.
