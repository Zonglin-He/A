# Original LN-only CPU completion count dispatch

The original two-domain, three-order LN-only design produced and sealed all 768
new formal fits before GT. Both original stage CPU readers completed all math,
state, input and dense scoring and saved their rows and completion receipts.
The terminal aggregate assertion expected 36 qualification-to-formal pairs,
although the locked design contains one arm: 2 domains x 3 orders x 2 first
arrivals x 1 LN-only arm = 12. Every one of these 12 accepted receipts is retained.

`resume_stvg_opd_LN_cpu_count001.py` captures the original scorer, locks, completed
stage rows/receipts and failure logs. It compiles a separate dispatch replacing
only the final count assertion with 12, reruns the original seal checks, reuses
the completed stage scores byte-for-byte, writes the original aggregate receipt,
and calls the unchanged original scalar finalizer. No GPU, model, detector,
optimizer or new GT scoring is called. Source/runtime pins stay unchanged.

This resolves the receipt dispatch only. The independent complete fit/source/
state/input/dense/statistical root, actual PDF pixel view, all anonymous outcomes,
remote bytes/SHA/Git blob/mode/tree verification, archive and genuine FINAL remain
required. The original cancelled baseline remains standby. Cost32 and subsequent
decisions remain distinct from LN completion. No manuscript or deployment change.
