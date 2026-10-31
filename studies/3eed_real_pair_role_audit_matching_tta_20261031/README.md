# Actual-pair subject/reference verification and frozen-box matching TTA

[Report](REPORT.md). Real Waymo-only→Drone/Quad experiments completed:64 A observations,32 newly reserved B observations,223 actual same-class candidate pairs,737 frozen VLM generations and32 calibration-probe forwards. Global exposed-development policy chose20 preference observations. Each target platform actually completed32projection updates (18496parameters), then32updated full-modelBforwards verified unchanged boxes and valid scores.

New B mIoU gains are0for both TTA and the same-information no-grounder-update preference-memory control. TTA changed all32scores but noTop1. No method adopted. All352anonymous task-arm rows,192policy-utility rows,223pair-audit records, negative outcomes and protocol revisions are preserved. Role/mapping gate utility and lack of actionable Top1 feedback are diagnosed without post-B tuning.

All target VAL has historical exposure; global policy/activation uses A task utility for development and is not an unlabeled deployment gate. B is excluded from G/fit and differs in decoded RGB/XYZ; no same-instance/multiview/time-independence claim. Only1Drone/2Quad scenes. Calibration4 is an engineering probe, not a validated role selector.

The executed scorer is research/round17/score_policies.py; research/round17/score.py is an earlier unexecuted prototype retained for history. The current fit is crossplatform/round17/fit_projection.py and the full-model check is closure.py.

No raw RGB, point clouds, captions, coordinates, instance/query IDs, annotations, feature arrays or weights are exported. See DEPENDENCIES.json for prior code and required local assets.
