# 3EED G compositional reference and role-binding diagnostic — R21

Waymo-only source100 → Drone/Quad. All208 observations are historically exposed development/regression data. Source boxes frozen. Three mechanical parser revisions and two full10-arm evaluations retained. Full references add exactly0 mIoU over matched repaired-appearance no-reference control; no TTA launched. Appearance-gate relaxation introduces two Quad harms. No method adoption.

Read REPORT.md, protocol/G_SPEC_LOCK.json, results/v2 and results/v3. Each version contains2080 anonymous task-arm rows; versions reuse208 inputs and are not4160 independent samples. All154 pair diagnostics and failed prerequisite histories retained. Local raw RGB, captions, point clouds, calibration/coordinates, model features, opaque IDs and weights excluded.

Code builds on previous studies in this repository, especially R20 commit a5c8d72ead300f4039989a5e9f918ab8ae448ffb. Reproduction requires the licensed official dataset, original private manifests and frozen source/model assets; the scalar export does not pretend to contain raw data. CPU and GPU environments unchanged from R20.

Execution order: parse21.py → detect21.py → reference21.py (preserved first prerequisite attempt), then *_v2, then *_v3. v2 and v3 assembly precede corresponding score programs. v3 only repairs missing same-sentence reference inputs after v2 exposed diagnostics, without threshold change. No independent confirmation or adaptation result claimed.
