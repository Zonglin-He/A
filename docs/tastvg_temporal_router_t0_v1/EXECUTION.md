# CPU-only temporal-router audit execution

This finite authorized audit reuses completed H-lite A. It does not resume an
old queue. Execute sequentially from the project root:

```
.conda/tubedetr/bin/python -B scripts/test_tastvg_temporal_router_t0_v1.py
.conda/tubedetr/bin/python -B scripts/prepare_tastvg_temporal_router_t0_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_temporal_router_t0_v1.py
.conda/tubedetr/bin/python -B scripts/score_tastvg_temporal_router_t0_v1.py
.conda/tubedetr/bin/python -B scripts/audit_tastvg_temporal_router_public_v1.py results/tastvg_temporal_router/2026-10-02
```

Preparation binds the complete 768-cell A trajectory, 192 expert-arrival
temporal inputs and reference receipts before new diagnostic GT use. The raw
cache may be shared across orders; actual unique assets are counted. No model,
weights, media or backbone cache needs loading. Scoring must find a complete
GLOBAL_PREDICTION_BARRIER, validate all hashes, then access already exposed GT.

All CUDA devices are hidden in these processes; assert CUDA never initialized.
No background controller or new recurring schedule is needed for this finite
CPU job. Preserve failed attempts if any, and do not alter locked formulas
after reading the result. Spatial A/old experiments/CURRENT_METHOD stay immutable.

Completion requires root readback, anonymous scalar and bootstrap checks,
actual generated figures visually inspected, GitHub allowlist/remote byte
verification, research archive check/snapshot/check and FINAL_COMPLETION.
Public readback is required; a local plot or score alone is not task closure.
