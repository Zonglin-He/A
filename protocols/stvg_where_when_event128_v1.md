# HC128 WHERE → WHEN frozen semantic-feedback capability test

Authorized 2026-10-19 after read-only interface audit. One finite test, not temporal
OPD, not a new spatial optimization or a confirmation study. Keep all RC128
parents from the existing conditional-probe roster; all have prior development
exposure. Reuse frozen tau .20 dual-scale Live seeds 41/42/43 and E coordinates.
Keep all eight existing physical positions, Source/E/time initial interval,
ReferDINO ryt_mevis_swinb checkpoint, native full-image preprocessing and original
complete text. No new expert, head, trained network, crop or reference intervention.

New evidence: one native frozen ReferDINO inference per parent (128 video queries,
1024 image frames). Capture final temporally fused class_embed output before token
max with a nonmutating hook; keep all tracked queries and lineage. The module is
shared at other layers: take its last invocation and require token max bitexact
to the final native logits, and native boxes/logits bitexact to old native caches.
Save actual tokenizer IDs/mask/offsets. Existing caches/inputs and config pinned.
No automatic duplicate model forwards. GPU lease cap 900s including loading and
host decoding/writing while leased; GPU free floor4GiB, RAM6GiB, disk30GiB; shared
gpu.lock, no signalling other services, finite 3h orchestration deadline.

Lock readout before any new inference/GT: cached query-only Stanza lexical VERB
tokens seed event groups; absent verbs use copular predicates. Include nonsubject
subtrees for arguments, negation, prepositions and auxiliaries; exclude punctuation.
All lexical verbs include descriptive participles: not guaranteed action-only.
Do not select words using results. Actual normalized/lowercase input character
mapping, word-mean of subword logits then equal event-word mean. Missing predicate,
truncation or span mismatch => unknown, no added temporal correction for event arms.
Native baseline is sigmoid(max token), including native special-token behavior.
No learned calibration, noun subtraction, threshold/temperature/length-prior search.

Association is one global teacher track per video: maximize mean eight-position
IoU against E or each saved Live trajectory, tie lowest query; report zero overlap
but do not GT-gate or remove it. No per-frame reselection. Same associated query
must yield identical semantic curve and interval: any change is a bug. Query
index continuity does not establish physical identity correctness.

Candidate edges: integer rounded midpoints of actual eight observed frame centers,
video start0/end frame_count. All positive contiguous cell unions, original MBR
and full video retained. Every arm uses the same candidates. For each interval
use piecewise-constant nearest-observation cells solely to integrate duration:
mean response inside minus mean outside, weighted by physical frame duration.
This is an explicit sparse-evidence assumption, not dense observation. Full-video
contrast has no outside and is unknown; remains a reported candidate, not an
artificial score0. Original interval is not selected using GT. No valid curve or
constant curve => original interval. Tied maximum within1e-12 => minimum L1
endpoint distance to original MBR, then earlier endpoints. No claims of universal
interval scoring or event probabilities; negative readout does not prove all OPD
on this evidence impossible.

Readouts: NativeConfidence, EventNative (native track, extra attribution control),
EventE, EventLive41/42/43. Compare each interval on exactly E spatial coordinates
first. Separately evaluate each Live's original time and new own EventLive time
for joint results. Primary comparisons: EventE−NativeConfidence, EventLive−EventE
(both E spatial), each temporal variant−E_MBR, joint new time−own old time.
Three seed metrics averaged inside each parent; no box/curve ensembling or best seed.

All semantic files and every readout sealed globally before new GT access.
Then offline only: within-parent inside/outside AUC at actual eight positions;
candidate pair ranking against actual tIoU, strict GT quality ties excluded and
score ties credited .5; complete tIoU/vIoU, actual-node sIoU, severe tails and costs.
Macro parent statistics with paired10000 resamples; unknown AUC/ranking support
counts explicit, all128 retained in deployment endpoint means. No GT annotation
outside the described event implies no verified event-inactive visibility label;
do not label teacher visibility as ground truth. Report this coverage gap and
inside/outside timestamp support separately. Candidate t/v oracle only after
seal, diagnostic not deployed score. Keep all failures, zero-change and harms.

Audit source/legacy equivalence, unit cases for geometry/readout, independent
physical-time score recomputation and official metrics vs DenseTube. Publish code,
this protocol, actual configuration, all signed/unknown per-parent measurements,
negative findings and limits through Zonglin-He/A existing dual-ref workflow.
Exclude media/text/parent IDs/annotations/weights/raw semantic and trajectory caches.
Do not change CURRENT/MAIN/PAPER or frozen spatial package. This test establishes
at most feedback/association capability; no OPD advantage or 2–3pp promised.
