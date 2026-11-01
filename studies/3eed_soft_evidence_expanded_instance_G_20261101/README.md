# R23 — soft evidence ranking and expanded instance pools

Waymo-only source100→Drone/Quad. Latest user criterion is aggregate metric improvement, allowing individual harms. Frozen weights/boxes, direct reranking only.208 exposed development observations and64 fresh current-input content-disjoint observations (historical targetVAL exposure disclosed,2Drone/3Quadscene clusters). Three bounded development grids retained; final globally shared configuration frozen before new64scores.

New64 mean mIoU: Source28.063498%,R21 30.922075%,new31.178995%;net+0.256920pp vsR21,Acc25/50unchanged. Drone2gains/1harm,Quad unchanged. Small effect, scene interval crosses0. Retain as research baseline, not evidence of robust or TTA gain. No-reference has identical taskmetrics;twoD208querychanges bothIoU0,allK64finalchoices identical; reference reasoning contribution is unproven. Bare EDA-style readout does not reproduce EDA dense training.

All15200anonymous scalar rows preserve initial unsuccessful log/linear calibration grids and final selection; they reuse272inputs, not15200independent samples. Scene aliases allow cluster replay. Raw captions/media/pointclouds/calibration/box coordinates/query features/opaque IDs/weights remain private. Own code only; literature audit links official sources rather than redistributing third-party implementation.

Read REPORT.md, protocol/G_SPEC* and SELECTED_v3_LOCK. Reproduce final run with licensed official data and required prior cache/model assets: prepare23.py; source23.py; parse23.py; prepare23.py under crossplatform; detect23.py; evidence23.py; rank23.py; evaluate23.py development; evaluate23_v2.py development; evaluate23_v3.py development then confirm. Old scripts are retained as actual trial history. No independent re-selection after K64scores and no TTA updates.

Dependencies and prior experiments: repository study R22 at commit86cf62fd366ea8d3dc3d919d837236ad833c0ef6. Scalar export is not a raw-data release.
