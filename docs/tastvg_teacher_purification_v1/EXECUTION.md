# Finite CPU execution: teacher purification v1

Run from the workspace using `.conda/tubedetr/bin/python -B`.

1. `scripts/test_tastvg_teacher_purification_v1.py`: meaningful selector and
   guard contracts, no research inputs/model execution.
2. `scripts/run_tastvg_teacher_purification_v1.py`: exclusive prepare,
   guarded select, then postseal score. Private `artifacts/tastvg_teacher_purification_v1`;
   anonymous `results/tastvg_teacher_purification/2026-10-04`.
3. `scripts/audit_tastvg_teacher_purification_v1.py root`: independent scalar
   computation from pinned support/GT and copied old verified controls.
4. `scripts/audit_tastvg_teacher_purification_v1.py public <results_dir>`:
   audit selections, metric rows/aggregates/bootstrap/tails/decision without GT
   files or raw intervals; rerun in the GitHub export checkout as well.
5. `scripts/report_tastvg_teacher_purification_v1.py`: actual measured report
   and scientific PNG/PDF charts; visually inspect the charts.
6. Publish exact allowlisted code/protocol/report/results to Zonglin-He/A,
   verify remote commit/parent/tree/files and public auditor, save receipts;
   update docs/RESEARCH_HISTORY.md, check/snapshot/check, FINAL_COMPLETION.

No old file is mutated. Any failed phase stays recorded; restart of an existing
prepared run is forbidden. A repair needs a saved recovery and explicit runtime
revision; no silent overwriting. Stage completed_pending_root_audit_publication
is not task completion. No recurring worker, monitor, GPU/controller or DTA is
started here; R2c and independent-expert experiments remain unstarted.
