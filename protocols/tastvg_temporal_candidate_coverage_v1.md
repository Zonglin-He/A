# Fixed-budget temporal candidate coverage contrast

Authorized by the user's 2026-10-03 inline plan after the oracle/event5 report
at GitHub 9485d3673a7a1320e03026b75b6075856322e4c0. This is a new matched
readout experiment, not a resumption of a paused stream or method promotion.

VidSTG and HC-STVG-v2 retain the preceding 32 development +16 confirmation
sources, one query/source, two orders, clean + five 5% transient corruptions,
25% expert arrivals, same-domain TA-STVG checkpoints and original Paper48
sampling. All sources have historical exposure. All 1,152 A trajectories,
Uniform persistent updates, spatial boxes, parameters, expert schedule and
pixels are immutable. Vid K1/lr .033761698432507946/teacher .34902548789596055;
HC K8/lr .006097133675874025/teacher1; rho .05/student1/D4/1792 unchanged.

Only one replacement candidate allocation is tested, identically on both
datasets. At the 288 existing expert arrivals, compare the old eight versus
eight new intervals. Slot0 is exactly the pre-arrival central native interval,
not A's already reranked final interval. No expert proposal becomes an output
candidate. Slots1-6 are, in order, early-short, early-medium, middle-short,
middle-medium, late-short, late-medium. Slot7 is long, with unrestricted center.
An interval [f_i,f_j+1), i<j, has duration fraction (f_j+1-f_i)/L and center
fraction ((f_i+f_j+1)/2-f_0)/L, L=f_last+1-f_0. Centers are split at1/3,2/3;
short duration <=1/3, medium (1/3,2/3], long >2/3. Boundary comparisons use
integer inequalities to avoid floating boundary drift. Each slot chooses the
highest native endpoint product score in that stratum, excluding already used
intervals. Empty strata fall back to the highest unused score globally and
are explicitly reported, never filled using GT. Ties use start then end index.

The available unlabeled signal is the cached source-checkpoint final temporal
head, separately softmaxed for start/end within each ORIGINAL offset, then
interleaved on the saved merged observed grid. Each offset has weight1/2. For
ranking candidate allocation, use log(p_start[i])+log(p_end[j]), i<j. This is a
candidate-allocation prior, not the native two-offset envelope policy or a
calibrated localization score. Source logits remain frozen; current A native
is retained independently. Source versus A native endpoint parity is checked
and reported rather than assumed. No new model, expert, view, loss, parameter
update, backbone or decoder replay is required.

The new readout permits legal endpoints anywhere on the original merged grid,
including across offsets; it preserves [start_frame,end_frame+1) mapping. It
does not feed a new time interval through the decoder. Old candidates include
their original native, intermediate-layer and final top-pair envelopes. Both
pools use the exact original critic max(confidence*interval-IoU), the same
cached ORIGINAL UniversalVTG proposals and confidences, and np.argmax's
native-first tie behavior. Do not substitute the later two-view min rule or
retune scorer/confidence. Persistent A's full spatial trajectory and all
subsequent states remain sealed. Nonexpert output remains exactly A.

Generation is label-free under an open audit guard; both datasets and both
panels seal globally before any new GT scoring. Confirmation does not select
the allocation, a parameter, a threshold or a per-dataset rule. The current
conversation has already seen historical cohort GT; do not claim fresh/blind
evaluation. Store private input hashes and protected state receipts. Separate
code/protocol pinning from historical asset verification.

CPU evaluation uses the literal official dense scorer, with fixed A full boxes
and physical frame coordinates. It covers GT-time and exhaustive grid oracle
on all1,152 arrivals; old/new candidate oracle and actual selection on the288
expert arrivals. Grid enumeration is ALL i<j, not only spans formed by the
two-offset product heap. Grid includes every old/new candidate and A/native;
GT time is a metric readout only. No GT coordinates enter generation or
expert scoring. Per cell verify:
GT_time-old8_oracle=(grid_oracle-old8_oracle)+(GT_time-grid_oracle),
and analogous new-pool decomposition. Report merged-grid residual rather
than claiming finer unseen frame resolution. Fixed sampled spatial support
and missing event-frame annotations remain unchanged.

Report separate dataset/development/confirmation, corrupt/clean,
expert/nonexpert/full flow, actual vIoU/tIoU, oracle vIoU and paired deltas,
grid and GT limits, unique candidates, fallback/overlap, gross gain/loss,
strict .3/.5 correctness transitions, >5pp harm, order/source influence and
positive/negative cases. Source -> equal condition -> equal order aggregation;
10,000 paired source bootstrap, seed20261003. Expert membership differs by
order, so derive full-flow dilution from complete rows rather than multiplying
by .25. Candidate/grid oracle is diagnostic privilege, not a deployable result.
Do not sum conditional time and spatial upper bounds or interpret a headroom
fraction as total-error fraction. Keep original A and previous reports intact.

Independent checks include brute-force analytic grid controls, cached original
critic/native/indices/boxes/state hashes, prefix-grid versus scalar dense
values, literal official dense readouts, all candidate selection rules and
bootstrap/public anonymous reconstruction. Publish code/protocol/configuration,
all anonymous outcomes including negatives/limits and figures to Zonglin-He/A,
verify remote bytes and runtime pins, archive check/snapshot/check. Raw media,
captions, GT coordinates/annotations, weights, H/state/gradient/cache tensors
and personal conversation records remain private. No third experiment, new
spatial loss, Specific correction, temporal view, automatic full-query restart
or production promotion is authorized.
