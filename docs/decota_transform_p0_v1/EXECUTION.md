# Finite real-input two-P0 execution

1. `scripts/run_decota_transform_p0_v1.py prepare`: independent cohort/runtime lock;
   predecessor and public completion receipts verified; no new prediction.
2. `scripts/run_decota_transform_p0_v1.py all`:12 coordinate/pixel contracts,
   first two development clean native/current full-replay parity per dataset,
  576 unique input transform captures serially on one GPU. No GT/new experts/
   gradients/parameter writes. `LAUNCH.json` and `STATUS.json` locate actual PID.
   Dataset statuses distinguish smoke, capture and sealed completion. The original
   `recovery/smoke_status_001` preserves the status-key failure; revision001 only
   repairs completion metadata and resumes fully sealed stages with hash checks.
3. All576 transformed prediction payloads and fixed interval/consistency rules must
   be in `GLOBAL_PREDICTION_BARRIER.json` before CPU GT exposure.
4. `scripts/score_decota_transform_p0_v1.py`: official dense scoring of the fixed
   readouts/states, source-macro paired10000 bootstrap, fixed-score correlation/AUC;
   immutable CPU_LOCK/GT_EXPOSURE, full anonymous rows and prelocked decision.
5. `scripts/audit_decota_transform_p0_v1.py`: independent private scalar coordinate/
   IoU/state/dense audit and portable independent source-statistics audit.
6. `scripts/report_decota_transform_p0_v1.py`: report, complete CSV, cases and three
   PNG/PDF figures. Root must inspect figures and positive/negative conclusions.
7. Only a passing locked P0 authorizes a separately locked matched conditional
   experiment. No pending conditional may be called final. Failed P0s explicitly
   skip514-head adaptation/acceptance, retain research working point/CURRENT.
8. `scripts/publish_decota_transform_p0_v1.py stage`, reviewed GitHub fast-forward
   export, then `verify <commit>`: exact remote byte checks and public-checkout
   recomputation. Archive check/snapshot/check and FINAL_COMPLETION close the task.

Private full states/H/queries/video/GT are excluded. Old streams and weights are
read-only; no old full query job or monitor is resumed. New input transformations
really require frozen encoder compute; reusing old logits alone is not P0.
