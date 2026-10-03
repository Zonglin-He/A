# Finite CPU execution

Run `.conda/tubedetr/bin/python -B scripts/run_tastvg_pr_role_v1.py` with
sequential actions `prepare`, `fit`, `readout`, `diagnose`, each a separate
process. Output is isolated under artifacts/tastvg_pr_role_v1 and
results/tastvg_pr_role/2026-10-03. Never overwrite predecessor outputs.

Then run scripts/audit_tastvg_pr_role_v1.py root and public, generate figures
and review. Models stay private. Update research archive, check/snapshot/check,
publish exact export and independently verify remote file bytes before final
completion. No GPU, new candidate/provider/online run or production change.
