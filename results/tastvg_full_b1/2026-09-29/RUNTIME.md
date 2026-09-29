# Full evaluation runtime status and recovery

Full9411query/670source,3order,16condition evaluation is running. Only PhaseA matched ablations are complete; no full metrics are available yet.

The finite pipeline attaches to the current spatial worker, then executes temporal specialists, frozen online streams, global-barrier scoring and the report. Every stage records failures and checkpoints; no failed query/condition is silently skipped. Predictions are completed before any new full-evaluation GT read. Pipeline completion does not itself verify GitHub publication; a remote readback receipt is required separately.

Before full student predictions, the spatial worker was found to omit explicit cudnn.deterministic=True used in original S0. The initial1097input receipts/194unique cached outputs and logs were preserved under excluded_attempt001 and excluded from the final evaluation. Their GPU time remains charged. The appended IMPLEMENTATION_REVISION_001 restores this backend flag; the original execution lock is immutable and the new worker hash is recorded as its one override. Temporal retains its original defaultFalse. The final method, source cohort, conditions and orders are unchanged; no outcome-dependent tuning occurred.

Three CPU contracts cover equal-source weighting, condition/order array axes and missing nonexpert eligibility. The new full capture path also reproduces J0.1 exact H/native/state/output on one old scheduled fixture followed by one inherited nonexpert fixture. These are engineering checks, not full-test performance.
