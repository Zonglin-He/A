# Finite frame-acquisition qualification

1. CPU prepare_tastvg_reference_selection_v1.py locks the source-hash roster,
   same A candidate inputs, sampling rule, implementation and specialist receipts.
2. test_tastvg_reference_selection_v1.py analytic contracts.
3. Serial run_tastvg_reference_selection_v1.py using existing tubedetr Python,
   with scripts/with_local_cuda.sh and PYTHONPATH=.runtime/sa2va_deps.
  62 attempts include two uniform bitwise parity checks. No GT or student forward.
4. Global expert-output seal, then score_tastvg_reference_selection_v1.py CPU
   diagnostics and independent geometry/reward/ranking/selection readback.
5. audit_tastvg_reference_public_v1.py on actual anonymous published outputs.
6. Root report/figures, archive/check/snapshot/check; public allowlist push,
   every remote file hash and remote public audit; FINAL_COMPLETION.
   No automatic new trajectory, old queue, method promotion or new grid.
