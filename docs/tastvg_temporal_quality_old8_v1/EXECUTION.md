# Execution: fixed Old8 temporal quality

Use `.conda/tubedetr/bin/python -B scripts/run_tastvg_temporal_quality_v1.py`
with `prepare`, `generate`, `score`, `root` in that order, separately. Run
`scripts/test_tastvg_temporal_quality_v1.py` before prepare. Preparation refuses
an existing runtime lock. Generation cannot read GT or score files; it writes
1,152 isolated readouts and one global barrier. Scoring requires that barrier.
Old writers, weights, caches, labels, predictions and CURRENT_METHOD are never
modified. Revisions preserve original code and failed attempts; do not fix
science after examining results.

Then run `scripts/report_tastvg_temporal_quality_v1.py`, export via the runner's
`export`, and run `scripts/audit_tastvg_temporal_quality_public_v1.py` on the
public result directory. Inspect the figures. Publish the explicit manifest to
Zonglin-He/A, verify the remote parent/tree/main and all file bytes and pins,
run the anonymous auditor on the verified public checkout, and save remote
receipts plus FINAL_COMPLETION. Update RESEARCH_HISTORY and check/snapshot/check.
This is a bounded CPU job; do not reactivate the paused hourly monitor or old
GPU queue merely because this execution document exists.
