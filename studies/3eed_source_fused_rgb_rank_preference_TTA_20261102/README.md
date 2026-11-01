# R36 — Source-fused RGB rank preference TTA, fixed A/DEV 2x2

Read REPORT.md and both DECISION and RESOURCE_DECISION. The registered RGB
comparison gate PASSED: DEV TTA exceeds its own frozen and same-information
rank200-memory control by+.212408pp. It is not reclassified as a failure.
The prespecified interaction(deltaRGB-deltaNoRGB) is-1.393059pp, and RGB TTA
absolute22.867746 is below originalR30 25.090904. Resource decision: stop this
fixed RGB rank branch, no extension or new confirmation. This does not refute
allRGB TTA or establish missingRGB as the previous primary failure cause.

Fixed source models are R25 sourceVAL-selected residual_rank seed0 epoch1,
separate RGB/noRGB source weights. Their difference is not pure input causality.
All22original preferences, oldU189, beta1,16steps;4heads64steps update only12929
rank MLP parameters. Source architecture/training changes are separate from
TTA increments relative to each own frozen baseline. No new source training,
Qwen, teacher, EMA, layer/step/seed sweep or regression. Native final scores and
each source own q0 are used, not old XYZ noun-logodds loss. Legacy B is diagnostic
only and not eligibility. Frozen boxes and all nonrank weights unchanged.

Same-info no-update control: A direct groupQP, DEV frozen rankinput200 key
memory, native margin.1, k1 from recordedGPU topk4, raw transfer then currentq0
anchor. Unknown candidates are not new negatives. All native fullvectors and GPU
indices are private and sealed; public scalar tasks/negatives/code are complete.

IndependentCPU audit:544sample-mode rows/2720task values,38independent QPs,
457negative method/reference rows,20scene tables, sourceq0/U/feedback verified.
CPU64 query differences0; actualGPU neighbor indices validated/replayed, not
substituted with CPU topk. OldR25 A208x2=416source rawargmax/boxes replayed.
A TTA raw vs fixed duplicate-group representative differs12rows, all disclosed;
frozen rawargmax exact. These are readout diagnostics, not posthoc extra scores.

Actualfullmodel closure:4registered input-hash observations x2modes x2states=
16point/RGB/fusion forwards, allboxes/query reproduce, nonrank unchanged. CPU
matching-head replays are not fullbackbone forwards. Preparation12.8042s includes
DEV64RGB extraction4.2364s; do not sum nested timings. Fit/QP/prediction6.3461s,
closure11.1426s; these are stage receipts, not optimized deployment latency.

A/DEV are exposed development data, not independent new confirmation; both
modes/controls reuse observations. No U/FINAL row manifests/media/GT accessed.
No broad scene generalization/significance claim. ROI from predicted boxes still
mixes background/neighbors; distortion-zero assumptions and source history remain.
WildRefer/xMUDA are mechanism inspiration, not upstream reproductions. No raw
media/captions/opaqueIDs/coordinates/features/weights/thirdparty source snapshots
exported; private licensed assets and historical caches are needed to reproduce.
