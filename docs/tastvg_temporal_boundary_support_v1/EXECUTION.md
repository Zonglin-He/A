# Bounded CPU execution

Use .conda/tubedetr/bin/python -B scripts/run_tastvg_boundary_support_v1.py
with prepare, generate, score, root, export sequentially. prepare locks the full
predecessor inputs and all prediction receipts. generate is guarded against
labels/results/reports and seals 1,152 readouts. Only score/root can read labels.
Run scripts/report_tastvg_boundary_support_v1.py after root; visually inspect its
figure, run the standalone public auditor, then export the explicit allowlist.
Verify GitHub commit and all file bytes; save local FINAL_COMPLETION, archive
check/snapshot/check. Old outputs, old locks and CURRENT_METHOD are immutable.
