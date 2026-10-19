# R21 early motion-region multimodal fusion: HC128 capability

User temporal/joint autonomy, spatial implementation/config/output frozen,
no new expert/checkpoint/model training. R20 conditional SA masks were harmful,
not merely unconnected (TV .173/.172); close that mask configuration. Old R1
LateRefer motion-key prior on fused TimeDecoder memory gave−.231pp. B2/query
ROI/attribute heads/event_query all use already-fused representations. Source
modal_encoder concat appearance/text/motion, then six spatial_layers perform
PER-FRAME cross-modal SA; declared temporal_layers are UNUSED here. New factor
moves EXACT old prior to earlier motion keys before representations/query/gates
form. It is not added cross-frame motion reasoning and 'old insertion too late'
is a hypothesis, not a proven cause.

Same current Vid Source224, original65 realframes/33/32 offsets, text/subject,
native ReferDINO eight cached boxes/physical interpolation, full video input.
Early input not in old norm-prefix caches (those are six-layer postfusion).
Immediately seal Source-exact captured inputs before any regional replay; later
regional prefixes are separate immutable files. On failure preserve CPU partials;
no automatic duplicate Source capture or budget reset.
Capture original ground_encoder.encoder INPUT src/pos/bool mask/durations once
per offset during at most128 frozen Source forwards. Same RGB/time/framehash
metadata, no new positions or expert. Native Source boxes/time logits and
replayed early input norm-prefix/time logits must match old Source bitexact.
No posthoc tolerance relaxation. New Source spatial boxes are computed only
for control and discarded; no optimizer/spatial/state update.

Existing prior w=.5+.5coverage/mean_valid_coverage, positive mean1, empty region
neutral, fullframe prior1. Apply logw ONLY to lastL motion keys of all six early
fusion spatial SA layers, all query rows; appearance/text key priors1. This does
NOT preserve their total normalized attention mass. Fold original padding into
encoder-local additive attention bias only; returned encoded_mask originalbool,
no ROI reapplication in TimeDecoder CA. Source positions/weights unchanged.
Native first/second frame gates and both queries recompute from new memory by
original frozen FP32 query/decoder replay. Attribute and bbox heads retain old
weights, no new head/latent fitting. Thus a result belongs to WHOLE early path,
not uniquely improved memory. First2 fullframe branch recovers native exact.

Frozen capability arms Native exact, old LateReferROI EXACT cached outputs,
EarlyRefer, EarlyShift (same existing xcenter+.5 mod1, sizes unchanged), plus
old HiResAuxMix strong static reference. No posterior mixtures, temperature,
layer/modalities/strength/shift search. New EarlyRefer is sole candidate.
Primary final fullv with three frozen spatial seeds parentmean; E auxiliary.
Also evaluate full-support expected candidate GTfullv/t AFTER sealing, including
same controls. Do not gate OPD only on hard readout failure. If expected teacher
advantage >=2pp vs Native with positiveCI AND exceeds LateRefer/Shift/strong
reference, qualifies ONE controlled actual transfer; this is not final gain.
If hard EarlyRefer >=2pp with positiveCI and exceeds LateRefer/Shift/strong reference with positiveCI but no distributional advantage, only
static component confirmation is eligible, no claim useful whole teacher.
Otherwise close this prior/location configuration without strength/mixture scan.
A future successful method still needs independent parents and Direct/Fixed/Live.

All RC128 exposed/one query per independent parent; TC109 not used for selection.
Unchanged E/Live41/42/43 arrays exactly, no new spatial method or registration.
Original65/2080 physical action support and finalMBR unchanged. All128 predictions
and raw early inputs seal BEFORE new GT. Independent CDF posterior checks,
scalar prior geometry and actual attention bias/layout/encodedmask checks;
retain bitexact geometry count diagnostically; geometry GPU/CPU reduction tolerance16eps32 relative to max(1,abs(weight)),
while Native prefix/time/logits remain BITEXACT (no tolerance relaxation),
official/Dense metrics/candidate expected quality/negative tails/pairedCI/LOO.
Preserve all parents/unknown/zero cases, no GT online gating or head selection.

Max900 GPU lease seconds including capture/replay/IO/loading+3cleanupreserve;
first2 noGT conservative maxcase×128+loading+60 qualification, stop if cap fails.
Source capture count<=128,2offsets each; native early reencode256, two regional
early reencodes512; actual native Source/replay and extra decoder calls recorded.
Shared nonblocking gpu.lock, host6GiB/disk30GiB/GPUfree4GiB/allocator.8/5GiBreserve.
No extra DINO/ReferDINO, source weight training, backward, spatial fit, or new
student OPD. Cache upstream cost not free. Preserve failure partials and cost,
no old queues/automations. Publish implementation/protocol/all signed results/
negative findings/verified remote bytes, exclude raw caches/media/GT/text/IDs/
weights. CURRENT/MAIN/PAPER unchanged, goal active.
