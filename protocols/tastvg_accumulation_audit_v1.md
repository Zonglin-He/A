# HC saved-write accumulation audit

User authorization: attachment b620b0e1 (2026-10-02). This is an offline
counterfactual diagnosis, not a newly selected or deployed online method.

## Question and locked population

Does removing all but the most recent **historical** spatial write improve the
same future nonexpert targets? Use both Uniform A and Routed R from
`artifacts/tastvg_routed_online_token_v1/hc2`, with its original 32 search
sources, one query per source, two orders, clean plus the five transient 5%
corruptions. Every nonscheduled arrival is included: 288 targets per arm,
240 corrupted and 48 clean. Targets have 30 unique sources; two sources were
scheduled in both orders. Do not select targets using labels or results.

Keep the HC-trained TA-STVG checkpoint, original sampled frames, captured H,
1792 spatial parameters, and the historical K8/lr/teacher configuration.
There are no new backbone forwards, expert calls, gradients or parameter
optimization. Historical results and CURRENT_METHOD remain unchanged.

## Frozen-delta intervention

Let the common initial spatial state be psi0, and historical write i be
delta_i = saved_post_i - saved_pre_i. A scheduled no-op is a zero write and
remains eligible as the most recent scheduled write.

* Source: psi0.
* All: psi0 + sum of writes strictly before the target arrival.
* Last: psi0 + delta from the latest scheduled expert strictly before target.

Subtract and sum the saved FP32 states in FP64, then cast once to the saved
dtype. Require bitwise equality between All and the historical target state,
and between its new replay and the historical prediction (boxes and temporal
indices). Source must also reproduce the capture baseline exactly. Last is a
transported historical delta applied at psi0, not a write recomputed at psi0.
An exact-refresh online arm is explicitly outside this experiment.

## Inference barrier and records

CPU preparation validates all 768 historical arrival state chains, locks their
receipts plus target captures, and reconstructs all prefix states without GT.
Save delta norms, all 8x8 cosine Gram matrices, consecutive nonzero cosine,
prefix cancellation ratios ||sum delta||/sum ||delta||, and latest-write
provenance. Zero-norm cosine/zero-denominator cancellation is null, not zero.
Verify the runtime code/input pins, use the existing exclusive GPU lease, and
replay Source/All/Last with the cached exact downstream interface. Deduplicate
identical state hashes only within each target, and count actual replays.
Seal every arm/order/condition prediction before opening diagnostic labels.
Outputs are bounded; interruptions/failures are preserved with revision pins.

## Offline endpoints and audit

Primary: Last minus All on original native temporal intervals, source-macro
dense vIoU for corrupted future nonexpert targets, separately A/R. Report Source,
All and Last absolute values, All/Last minus Source, paired 10,000-source
bootstrap intervals (seed 20261001), each order, all five corruptions, and clean.
Use source -> order/condition cell means -> source means, resampling sources.

Also report both spatial tubes on the same Source temporal interval, full-GT
dense sIoU, tIoU, .3/.5 correctness transitions, gross gain/loss and >5pp negative
tail. Include the predeclared >=2 prior writes subgroup and latest nonzero vs
latest no-op; neither replaces the main all-target endpoint. Match the original
public All/Frozen scalars independently. Validate dense metrics with the
independent scorer, every reconstructed state and capture binding, no-op outputs,
and unchanged checkpoint/production hash. Publish anonymous scalar rows, Gram
matrices, all aggregates, positive and negative cases, plots, code and protocol.

A positive Last-All result establishes this frozen-delta counterfactual, not
the benefit of exact-refresh learning or a unique nonlinear-interference cause.
Negative cosine/cancellation alone is not evidence of task-level harm. Retain
negative findings and do not tune a refresh rule on these diagnostic labels.

## Conditional token P2 interface investigation

After launching the main bounded audit, inspect existing UniversalVTG/PE-Core
and TA-STVG video interfaces for spatially indexed, temporally contextualized
tokens in a trained shared text space. P2, if such an interface exists, is only
the previous 60 fixed cells with the previous explicit event phrases and
attention-weighted inside/outside aggregation using its native logit scale;
no new temperature sweep, online update or S+T fusion. Frame-only patch tokens,
pooled temporal vectors, and unaligned video features do not satisfy this
interface. If absent, record the exact code evidence and leave P2 unexecuted
rather than inventing a projection, downloading a new model, or renaming static
CLIP as video-native. ConDA/D2VLM/TF-CADE motivate investigation, not substitute
for the local interface or establish its quality.

## Completion

Independently audit full coverage and scalar claims, update RESEARCH_HISTORY,
run research_archive.py check/snapshot/check, publish to Zonglin-He/A with
remote content verification, then record FINAL_COMPLETION. Exclude private
media, captions, GT, weights, state/H/token tensors and personal conversations.
