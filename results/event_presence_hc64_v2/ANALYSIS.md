# Teacher event information and conditional feedback: HC64

Two fixed cache-only feedback trials are complete. Neither is promoted. Space, paper
configuration and the paused full queue are unchanged. This is the repeatedly exposed
HC64 development set; the teacher audit used its labels before design. New decisions
were nevertheless generated without labels and all64 sealed before their new scores.

## What the teacher actually measures

The frozen ReferDINO `ContrastiveEmbed` produces object-token dot products. Its native
classification path takes the maximum over text tokens; `loss_labels` is binary
referred-object focal loss using annotated valid frames. These are not independently
supervised probabilities that the named event is currently occurring. Temporal fusion
can encode dynamic descriptions for identifying an object without learning its event
start/end. The [ReferDINO paper](https://arxiv.org/abs/2501.14607) describes the temporal
enhancer for motion descriptions; this does not establish event-boundary calibration.
[TGL](https://arxiv.org/abs/2508.11955) explicitly identifies missing moment supervision
in conventional RVOS and adds moment annotations and training. We do not reproduce TGL
or introduce its models, annotations, training or oracle moments.

Across64 queries, predicate-vs-noun correlation median is0.8660; an in-sample linear
noun model explains68.90% of predicate variance on average. This R2 is an audit, not
held-out predictive performance or proof that the noun causally explains the predicate.
Raw predicate event-anchor AUC is0.6622; noun AUC0.6415. Their paired difference0.0206
has95% parent-bootstrap interval[-0.0450,0.0881]. On45 queries having both classes in
the higher-noun-response half, predicate AUC is0.5278[0.4185,0.6352]. Noun response is
only a presence PROXY: no visibility labels have been introduced.

Old relative ownership has AUC0.4352. It measures target-vs-other-slot preference for
the predicate minus that for the noun, not event occurrence. For RC009 and RC016,
raw predicate/noun each have AUC1, but ownership AUC0: normalizing a strong nominal
response away can invert valid temporal evidence. RC042 has raw predicate AUC0.25
while noun AUC0.9375: removing nominal evidence cannot create the missing event signal.
AUC is an input diagnostic, not a surrogate for final temporal localization.

## Fixed change and matched controls

V1 regresses each predicate logit on nominal logit in other14 teacher slots at each
of the original8 observations, evaluating the held-out target-slot residual. Other
slots are calibration data, not known negative identities/actions. This tests a
measured nuisance-response mechanism, with no new expert call. But49/64 queries have
at least one target noun response over3 calibration standard deviations from the
other-slot mean. The estimator extrapolates and its residual AUC0.4435 does not improve
on old ownership. This is a calibration-support failure, not proof that all event
information is absent.

V2 changes ONLY the nuisance estimator: for each time, fit on the same target track
at the other7 times. This avoids cross-slot fitting but32/64 still have an extreme
held-out time; eight observations need not identify appearance and action separately.
The original8 teacher logits were jointly temporally fused: this numerical leave-one
calibration is NOT independent held-out teacher inference or new temporal supervision.
V2 residual AUC0.5554[0.4902,0.6198] remains inconclusive.

All arms use identical temporary ridge1 projections into cached six-layer temporal
hidden, both offsets, full8+8LOO diagnostics; LOO never vetoes an update. Full-view masks
match native expected occupancy. Whole-interval log softIoU subtracts a uniform mask
with the same mass. All legal pairs are reweighted as q0*exp(score), followed by the
original physical MBR. NO eight-anchor membership protection, acceptance threshold,
agreement gate, conditional no-update fallback, step/weight sweep or GT selection.
Unchanged final intervals arise from the same deterministic MBR. Raw predicate,
old ownership and noun controls use this same readout. Shifted feedback moves the
profile by one quarter of the temporal nodes and restores occupancy. No query fitting,
visual encoder, decoder, RGB read, new teacher model or new expert observation is used.

## Actual outcomes

Percent metrics; time improved/harmed/tied counts compare each arm with NativeG6.

| Feedback | complete vIoU | tIoU | unobserved sIoU | changed intervals | time +/−/= | time loss>5pp |
|---|---:|---:|---:|---:|---:|---:|
| NativeG6 | 29.6952 | 55.9613 | 50.5762 | 0 | 0/0/64 | 0 |
| V1 other-slot calibration | 29.8187 | 55.8218 | 50.6460 | 40 | 20/18/26 | 5 |
| V2 same-track LOO calibration | 29.8396 | 56.0578 | 50.7958 | 30 | 13/16/35 | 1 |
| Raw predicate | 29.6278 | 55.8721 | 50.6316 | 34 | 17/15/32 | 2 |
| Old ownership | 30.0351 | 56.2147 | 50.8101 | 36 | 20/15/29 | 2 |
| Noun presence proxy | 29.5164 | 55.9630 | 50.5107 | 37 | 18/17/29 | 3 |
| Shift V2 | 29.6329 | 55.7219 | 50.6501 | 32 | 15/16/33 | 4 |

V2−NativeG6 complete vIoU+0.1444pp[-0.1127,0.4315], tIoU+0.0964pp
[-0.4647,0.6708]. Time13 improve/16 harm/35 ties; complete vIoU12/16/36. Of30 changed
intervals,16 damage tIoU. One tIoU loss>5pp, zero complete-v losses>5pp. Short<=5s
time gain−0.8801pp, long>5s+0.3454pp. Thus the broad-improvement/low-harm objective is
NOT met. Positive mean alone does not rescue the mechanism.

Unobserved fixed-support sIoU+0.2196pp[0.0517,0.4355] is a measured side effect of
running the unchanged spatial DP on changed intervals. It is not a new propagation
method or reliable time improvement. V2−shifted tIoU+0.3359pp[-0.3347,1.0349] does not
establish event specificity. Old ownership scores highest in this matched panel, but
was already tried and is not selected post hoc as a new method.

Concrete cases: RC009 remains at the native final interval under V2, with residual
AUC0.5; RC016 also remains native, AUC0.5. RC042 moves its start8 physical frames later
and loses6.25pp tIoU. The raw-predicate feedback makes the same harmful movement.
A subject-associated high response after an event cannot be relabelled an event-end
signal by nuisance subtraction alone. These examples distinguish target response,
feedback construction and actual discrete readout; agreement of projected views is
not independent confirmation because they share the same eight pseudo-targets.

## Verification, costs and disposition

Both64-query stages sealed before new scoring:1792 independent official/DenseTube
checks max6.66e−16. All320 stage-to-stage control pairs recover bitwise. Eighteen
independent spatial replays and18 dense MBR checks pass (max1.34e−15). All48 paper and
434 archived provider source hashes unchanged. No predictions were edited after seals.
The first stage's scorer was invoked before the seal existed and stopped before any
label read or result; its failed log is retained. It was rerun after successful seal,
without changing code, predictions, fingerprint or scoring definitions.

CPU cached feedback cost is0.306s/query for V1 and0.320s/query for V2 including the
entire control panel and LOO diagnostics. These are NOT end-to-end or isolated-primary
latencies; original expert/capture creation costs are inherited and not relabelled.
There are zero new model forwards and zero optimizer/query steps.

Both feedback estimators remain research negatives. The available eight temporally
fused object-token responses contain correlations with events, but this study does
not establish independent, stable event-vs-presence discrimination. This is a scope
finding, not an impossibility theorem for ReferDINO or query adaptation. Further event
feedback needs an actually verified temporal/event-discriminating source of supervision
or intervention; more pseudo-target fitting cannot repair an unidentified signal.
No default upgrade, no automatic full resume, and no conditional GT-duration mixture.

Complete [V1 signed outcomes](../event_presence_hc64_v1/signed.tsv),
[V2 signed outcomes](signed.tsv), [severe tails](severe_declines.tsv),
[teacher audit](../event_presence_hc64_v1/teacher_audit.json),
[information summary](teacher_information_summary.json), [verification](verification.json).
