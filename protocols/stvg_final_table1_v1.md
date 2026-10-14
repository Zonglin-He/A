# Frozen final two-branch Table 1 execution

Authorized 2026-10-15. Algorithm exploration ended with the pre-score global A
selection in stvg_cross_frame_validation_v1. Full official results must not
alter this choice. CURRENT and every historical stopped queue remain unchanged.

## Population and declared rows

VidSTG source -> HC-STVG-v2 validation: 3,482 queries / 237 parent movies.
HC-STVG-v2 source -> VidSTG test: 10,303 queries / 732 parent videos.
Use the complete locked official PLAN manifests: 13,785 queries. Preserve
historical exposure flags. Neither domain is an untouched development source.

New episodic predictions: Frozen, Temporal-only, Spatial-only, Temporal+Spatial,
and Temporal+Teacher Geometry-only. Spatial-only and the combined branch use
the same QA-selected observation evidence, matching the development definition;
therefore Spatial-only isolates output-time adaptation, not the cost or absence
of Temporal-QA evidence preparation. Declare that dependency in the ablation.
Teacher geometry uses identical QA output support; missing masks fall back to
the Frozen geometry. It is a dense expert inference control, not parameter TTA.

Reuse the complete sealed Source/DINO-Refine/TENT-STVG/SAR-STVG baseline
evaluation only after exact cohort, checkpoint, input, protocol and payload
verification. Retain its original three online orders and label its setting.
Do not count repeated stateless readouts as independent inference. The previous
Spatial OPD row is unavailable until a complete matching prediction roster is
verified; missing predictions are not filled with development results. EATA
requires both full Fisher and actual per-direction qualification; the old
one-direction result must not be relabeled a complete two-direction row.

## One method, no target-label choice

Temporal: original four Native/Projection/expanded/contracted candidates,
eight sparse marked frames per clip, original caption prompt and 160-token cap,
own-query A/B/C probability feedback, original 512-parameter Adam update.
Spatial: frozen YOLOv12-X + SAM2.1 and STVG-R1 identity. WHO is independent
of R1 WHEN; the temporal branch supplies QA support. Original four quarter-bin
QA centers, original +/-2-frame local visibility support, presence-weighted
1792-parameter Gaussian IoU OPD, five proposed steps and original per-domain
configuration. A rejects reference-agreement deterioration, restores both
parameters and Adam, and stops that fit. No GT enters candidates, identity,
references, visibility, admission, optimizer or rollback.

If four unique eligible observation centers do not exist, record insufficient
support and exact no-update; do not move centers using visibility or labels.
If a temporal clip cannot supply the original eight distinct real frames,
record unavailable feedback under the original no-evidence readout behavior;
never synthesize frames or silently shorten the sampling protocol. New boundary
handling needs explicit source/CPU readback and is engineering qualification,
not a result-driven search. All failures and abstentions remain in the denominator.

## Engineering preflight and acceleration

Lock 32 parent sources per domain outside registered development exposure and
the current eight development parents, by stable metadata hashes only. One
hash-selected representative query per parent gives 64 engineering probes;
the full parent query lists are retained for shared-video cache checks. Do not
read their GT or tune from their performance. Probe source restoration,
prediction consistency, recovery, real throughput, RAM/VRAM and disk growth.

Model residency and exact cache reuse are allowed. Cache tracking by actual
video content, physical input bounds, decoder clock, weights and preprocessing,
never HC parent ID alone. Reuse per-query inference only after receipt, bytes,
hash and all semantic-input bindings match. Stage models serially under one
own-unit GPU worker; do not overlap the old queues. Token caps, prompts, visual
sampling and precision cannot silently change for speed. Pure execution-cache
changes require bounded actual input/output equivalence checks before adoption.

The single two-candidate Temporal deletion was tested and rejected: full-method
development loss exceeded the predeclared 1 pp/domain allowance. No further
component or hyperparameter sweep follows. Exact execution optimization remains
eligible and does not reopen method selection.

Keep host RAM >=6 GiB and GPU >=4 GiB headroom while active, own unit RAM high16/
max20/swap1 GiB, disk >=30 GiB. Forecast retained artifacts from actual preflight
growth. Only newly created, explicitly ephemeral bridge files may be removed
after roundtrip verification and committed receipts. Never delete old evidence,
raw source videos, checkpoints or incomplete-failure assets to make space.
If complete retained evidence cannot fit, pause at a sealed query boundary and
report the storage requirement rather than silently dropping audit material.

## Seal, evaluate and close

Use atomic per-query completed receipts; retries do not create extra counted
queries. Seal all 13,785 new predictions and input/source references before
opening official annotations for new scoring. Keep no-update and failed-input
status explicit; a partially executed cohort is not a complete table.

Report official query-macro m_vIoU and strict vIoU >0.3/>0.5, plus source-macro
means: first aggregate query/order within parent, then parents. Use 10,000 paired
parent bootstrap resamples and preserve pairing. Report tIoU, dense fixed-support
sIoU, negative-source fraction, worst sources, support counts and actual expert/
student end-to-end cost. Distinguish episodic new rows from online old rows.

Root independently checks source/state/readout, metrics and aggregation; actual
figures are pixel-reviewed. Publish anonymous positive, negative, neutral and
unknown results with source/configuration on Zonglin-He/A, verify remote bytes/
SHA/Git blob/mode/tree and refs, then archive check-snapshot-check. Public files
exclude private query/media/GT/weights/caches. No paper-level completion until
the actual full prediction, evaluation and closure barriers exist.

STVG-R1 target-domain training overlap is not fully resolved. State frozen
external-expert-guided TTA and retain this limitation; do not claim an expert
with no target-domain prior.
