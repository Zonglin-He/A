# C1/Scale06 online DeCoTA versus Frozen, within-domain corruption

Registered after the user's explicit confirmation of C1/Scale06 online DeCoTA
on 2026-10-04. This is a new evaluation of an existing locked method; it does
not resume an old queue, search parameters, or change a production registry.

## Question and fixed panel

Does the locked online C1/Scale06 outperform the same checkpoint's Frozen
prediction under the five deployment corruptions at 5% burst severity?
Use the exact `tastvg_current_correction_views_v1` panel: VidSTG and HC-STVG-v2,
each 32 development and 16 disjoint confirmation sources, one query/source,
two previously fixed orders, clean plus frame_drop_5, frame_freeze_5,
motion_blur_5, occlusion_5, exposure_5. There are 576 arrivals/dataset and
1,152 overall (960 corruption and 192 clean). All sources have historical
exposure; confirmation is not a fresh test. A chain resets at each
dataset/split/condition/order boundary. The short 16/32-query chains do not
establish full-stream retention.

## Method identity

Spatial: directly call `vg_tta.c1_luna_tricks_v1.fit(..., 'Scale06')`, pinned by
`methods/C1_FINAL_RESEARCH_CONFIG.json`. Query residual 256 plus the final
spatial block's norm1/norm3/norm4 weight/bias 1,536, total 1,792 parameters.
Fresh Adam per query, lr .03, betas .9/.999, eps 1e-8, weight decay 0,
10 updates; choose the minimum original 5 L1 + 2 GIoU reference loss among
steps 0..10, earliest tie, with denominator four planned observations.
Query residual resets to zero each arrival; only 1/16 of the selected LN
displacement is committed to future queries. Empty admitted reference sets
skip the spatial update. No Rank-RKL, lookahead, new loss, or temporal-head
persistence is added.

Reference acquisition is the unchanged C1 DINO-tiny context/target-token
interface and score/margin gates, using up to four uniform frames in the
Frozen native interval. It occurs at every query rather than A's recent
25% expert schedule. Identical pixel/query/frame requests can be cached
across the two orders. Budget: at most 2,304 new DINO forwards, representing
at most 4,608 planned observations in the logical streams. Sa2VA/UVTG are
not used. This expert-budget difference must be disclosed in any A comparison.

Temporal: original C1 self-actionness projection and NLL + hinge, full
66,306-parameter temp_embed, fresh AdamW and source reset each query,
five updates, last iterate, eta .25 shrink toward source, eps 1e-4,
betas .9/.999, weight decay 0, prior .1, MAD epsilon 1e-6, margin .2.
Preserve the old target-dependent configurations: Vid lr .1 / center .5;
HC lr .001 / center 1. These are transferred unchanged, not selected using
this panel. Scale06 was historically locked on Vid; HC2 is a newly authorized
evaluation of the same spatial rule, not a prior validated HC winner.
The temporal decoder inputs/actions are Frozen and independent of current
spatial fitting. No temporal parameter or optimizer state persists.

## Checkpoint and replay

Use official within-domain EMA checkpoints TASTVG_VidSTG.pth and
TASTVG_HCSTVG2.pth, original Paper48 input sampling and two offsets.
Reuse only hash-bound Frozen H caches from extended_sensitivity_v3,
with checkpoint, pixels, query, frame IDs and metadata matching. H is the
already-normalized frozen encoder output; an isolated SpatialReplay adapter
uses Identity at that normalization boundary and otherwise executes the
original C1 suffix, all six spatial blocks, actual iterative reference
detaches, and original head. No inverse LayerNorm approximation is used.
No new backbone runs except first-clean, per-dataset parity checks against
the original full forward and original SpatialReplay. Require exact native
boxes/logits/intervals, exact Scale06 trajectory on the same anchors,
temporal input parity, and full reinsertion of the selected spatial and
shrunken temporal state. Source model parameters must remain unchanged.

## Timing, sealing and analysis

Current final output includes the current query's selected spatial correction
and episodic temporal correction, matching historical C1. Separately preserve
the inherited-LN/pre-current-update prediction, spatial-only correction and
temporal-only correction. Do not call current after-update gains future TTA.
All 1,152 predictions and update traces seal globally before this worker or
the separate CPU scorer reads GT. No GT is used in references, selection,
optimization, tuning or source/order selection.

Primary: paired source-macro dense vIoU, DeCoTA minus Frozen, corruption,
per dataset and per development/confirmation split. Average both orders and
five conditions within each source; paired 10,000-source bootstrap, seed
20261004. Also report tIoU, dense sIoU, clean, each corruption, both orders,
pre-current-update gains, >5/>20 pp harm, .3/.5 correctness transitions,
gross gain/loss, reference admissions and costs. Use official Vid/HC2 dense
scoring conventions, independently cross-check the vectorized scorer.

Independent audit checks native/cache parity, source/query resets, exact
Scale06 selection/loss/Adam arithmetic, 1/16 state recurrence, temporal
last-state/shrink semantics, coverage, sealing and all metric aggregation.
Keep failures and negative findings. Do not promote a development winner.
Publish implementation, protocol, anonymous results and report to
Zonglin-He/A, verify remote bytes, and update RESEARCH_HISTORY using
check/snapshot/check. Never publish media, captions, GT, weights, H caches,
raw boxes or parameter/gradient tensors. No total time deadline is imposed.

