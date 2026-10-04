# Raw UniversalVTG teacher purification audit v1

Authorized on 2026-10-04: a finite CPU proposal-level audit only. The question is
whether confidence-free overlap consensus identifies higher-quality supervision
inside the exact cached raw proposal support. No DTA, forward/backward, model
load, GPU, new expert/view, spatial score, threshold, clustering, gate, PoE,
temperature/LR/step search or persistent-state change is permitted in this run.

## Predecessors and material difference

R1 established a useful temporal gradient channel; R2 confidence top1 and R2b
equal raw mixture failed as deployable teachers. R2b's low average proposal
quality does not uniquely establish the cause of adaptation loss: diffuse
targets and head projection can also matter. This audit isolates teacher
quality before any optimization.

T0 (`docs/TA_TEMPORAL_ROUTER_REVIEW.md`) already tested confidence multiplied by
mean peer overlap, routed observation mass and native-candidate scoring, with
negative results. This reopening changes the mechanism and endpoint: select
the raw proposal with maximum *confidence-free* mean peer overlap and measure
its temporal quality directly. T0's negatives remain evidence; this is not a
new claim that same-model agreement provides independent corroboration.

## Fixed inputs and three teachers

Reuse R2's verified COHORT and EXPERT_SUPPORT, committed in
324b3357dbd67b435d1fb507ae516a44ecf53961; latest R2b publication is
866ec35bd0dc0c41b938e656e9c9e430d18c18af. For each dataset, the parent design has
32 search and 16 confirmation sources, one query/source, two orders, clean and
five transient 5% corruptions, historical exposure, official same-domain
TA-STVG/Paper48 inputs and fixed 25% expert schedule. This audit covers only the
288 scheduled cells (240 corrupt,48 clean), not the 864 nonexpert arrivals.
Independent scheduled expert sources: Vid search16/confirm8; HC search14/confirm7.
Underlying spatial A, states, sampling, pixels and all previous files stay fixed.

Use all already-valid raw rows, preserving order, duplicates and fractional
physical-frame endpoints. Do not clip, round, deduplicate or choose a subset.
Confidence = first raw argmax confidence, exactly the R2 EDeploy rule.
Consensus score = mean interval IoU with the M-1 *other raw rows*, excluding
self. Select first exact argmax, without confidence. Raw duplicates count as
separate votes. Empty/invalid support is an engineering error, no fallback.
Singleton: sole proposal, score0 and undefined-peer-agreement flag; it has no
purification choice. Actual support contains M26..312, no singleton/empty.
Oracle = first argmax continuous proposal/GT tIoU, diagnostic only after seal.

## Boundary, metrics, inference and decision

An isolated subprocess with file-read audit guard fixes *all* Confidence and
Consensus choices and scores before reading GT or old scored controls. Save an
immutable selection barrier binding code, inputs and selection hashes.
Only the subsequent CPU scoring subprocess may open pinned GT spans and the
old R2 scored selection control. No GT endpoint is used for either deployable
teacher, no target selection/tuning, no changes based on scored cases.

Primary metric: continuous half-open physical interval tIoU, matching R2's
support-level teacher metric (no integer truncation or STVG decoder/dense tube
readout). Report source macro, averaging repeated conditions/orders within
source, then equal weighting sources; paired 10000 source bootstrap, seed
20261004. Four primary panels: Vid/HC × search/confirmation, corrupt experts.
Report clean and each order, cell means, source values/leave-one-source-out,
Confidence/Consensus/Oracle levels, Consensus−Confidence, Oracle regrets,
gross gain/loss, raw support quality, P(tIoU>.5) **strict**, and severe wrong
event = tIoU0 (disjoint/touching intervals). Record correct→wrong and rescue,
new disjoint mistakes and recoveries, good and bad paired cases.

Operational continuation requires clear positive paired teacher-quality
evidence in all four primary panels (each lower95% bootstrap bound >0).
Success/disjoint counters are supporting diagnostics, not newly tuned gates.
Failure/inconclusive panels do not trigger R2c. Even if this condition holds,
this finite task ends with the audit; no gradient stage launches automatically.
A negative result stops this medoid/internal-aggregation candidate, not a proof
that every single-expert algorithm is impossible. Independent evidence is a
possible next study only, not authorization to invoke new models/views.
Teacher tIoU gain does not establish adaptation or final-STVG gain.

## Verification and publication

Independent scalar-loop root recomputes all peer-overlap means, first argmax,
proposal/GT qualities, both R2 controls, metric vectors, source bootstrap and
tails. Verify raw-support copy and expert input/cache hashes, temporal seal,
production registry hash and zero model/gradient calls. Public audit recomputes
selections and all statistics from anonymous per-proposal overlap/confidence
scores and postseal GT-quality scalars, without publishing raw intervals, GT,
media, hidden states, raw caches, weights or conversation. Publish full
implementation/protocol/anonymous results/negative cases and figures, verify
remote bytes, then research archive check/snapshot/check and FINAL_COMPLETION.
