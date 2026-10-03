# Finite CPU execution

Entrypoint: `.conda/tubedetr/bin/python -B scripts/run_tastvg_pr_accessibility_v1.py`
with sequential actions `prepare`, `fit`, `readout`, `diagnose`.
Each action is a separate process with immutable output seals; preserve any
failure originals rather than rewriting prior fits or predictions.

Only existing trusted SHA-bound CPU arrays are loaded. New target-search ridge
weights are private under artifacts/tastvg_pr_accessibility_v1, never deployed.
Source-fit predictions and all candidate selections must remain fixed.

Then run `scripts/audit_tastvg_pr_accessibility_v1.py root` and
`scripts/audit_tastvg_pr_accessibility_v1.py public`, write report/figures, archive
check/snapshot/check, publish and verify every remote file. Completion of CPU
scoring alone is `completed_pending_root_audit_report_publication`.
