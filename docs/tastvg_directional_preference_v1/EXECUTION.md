# Finite P0 directional execution

Namespace: artifacts/tastvg_directional_preference_v1. The old proximal results
and all paused historical queues remain immutable. Run only the new controller.

1. prepare_tastvg_directional_v1.py copies the same private input manifest and
   creates A/E/F requests. Seal runtime/code/plans and predecessor barriers.
2. test_tastvg_directional_v1.py; smoke_tastvg_directional_v1.py vidstg/hc2;
   root accepts the actual smoke receipts, then starts the finite controller.
3. continue_tastvg_directional_v1.py serially executes Vid A/E/F then HC A/E/F.
   Six384 prediction barriers form GLOBAL_PREDICTION_BARRIER, before labels.
4. Root scores/audits all six streams on CPU, reads paired/gross/direction
   diagnosis, validates all covered arrivals, exports anonymous results and
   verifies GitHub contents. completed_pending_root means required closeout.
5. Record FINAL_COMPLETION only after actual audits/publication; pause the reused
   monitor. Scientific changes require a separate recorded next experiment.

Workers use own current RKL magnitude; they never copy previous A magnitudes.
Zero teacher rank-direction is an explicit no-op and not hidden as matched
magnitude. Top-1 is expert rank-contrast selection, not high-disagreement gating.
No additional expert inference or backbone optimization is requested.
