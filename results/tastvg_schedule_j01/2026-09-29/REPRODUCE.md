# Reproduction

Use authorized original models, source roster and cached H/expert evidence. Existing method code must match frozen J0 hashes.

1. Run tests/test_tastvg_schedule_j01_v1.py with pytest.
2. Run scripts/run_tastvg_schedule_j01_v1.py prepare to seal source-hash orders.
3. Run the same runner with run via bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B.
4. After all predictions seal, run scripts/score_tastvg_schedule_j01_v1.py and scripts/tier0_tastvg_schedule_j01_v1.py.
5. Run scripts/report_tastvg_schedule_j01_v1.py.

Write-once artifacts: preserve old runs and use a new destination for independent repeats. Never resample orders according to metrics. Public scalar-only audit needs Python/NumPy:

    python scripts/audit_tastvg_schedule_j01_public_v1.py results/tastvg_schedule_j01/2026-09-29

Public roster uses Q aliases; original source-hash regeneration requires the authorized private roster. Scalar aggregation and order membership checks need no weights, labels, media or raw states.
