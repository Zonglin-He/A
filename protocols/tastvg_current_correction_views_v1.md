# Current query correction, persistent Uniform A, and two temporal views

User authorization: latest inline two-round plan, 2026-10-02. Independent
experiment; historical predictions, checkpoints and CURRENT_METHOD are inputs.
No restoration of paused fullquery, baseline, token or fast-memory experiments.

VidSTG and HC-STVG-v2 each use the fixed v3 32 search sources, one query per
source, two original orders and clean plus five 5% transient corruptions. The
fixed v3 16 confirmation sources are disjoint within this batch, but all inputs
have historical exposure. This is development and matched confirmation, not a
fresh benchmark. Twenty-five percent of arrivals have specialists. Same rules
for both datasets; no dataset-specific correction choice. Vid K1/lr
.033761698432507946/teacher .34902548789596055; HC K8/lr
.006097133675874025/teacher1; rho .05/student1/D4/1792 parameters fixed.

Round1 uses actual A pre-arrival state. Frozen source-fixed nine probe offsets
are identical for every evidence comparison. Historical captured H, A states,
Uniform masks and candidate predictions can be reused only with receipt hashes
and live parity. Confirmation probes not recorded historically are regenerated.
Every persistent state transition remains original Uniform A Rank-RKL, including
HC K8. Its full saved trace and a live two-arrival control are verified. No
selected probe or temporary update changes any future state.

Compare A, U_select, R_select, R_temp and Specific_temp. The first four spatial
alternatives use the same A Fast interval. Direct selection chooses the unique
maximum actual reward; ANY top tie keeps center, including ties excluding center.
Empty evidence uses original boxes. R_temp takes one ordinary SGD step -lr*gR;
Specific_temp takes -lr*(gR-gU), using a single common gradient graph and the
same nine detached targets. Both require nonempty Routed evidence; Specific
additionally requires nonempty Uniform evidence. No evidence therefore cannot
produce an anti-Uniform correction. Flat nonempty rewards retain original RKL
semantics. Temporary states expire immediately. These are current adapt-then-
predict readouts, not future pre-update gains or a native policy distribution.

Round1 winner is a SINGLE correction rule for both datasets: highest equally
weighted dataset corruption-all-source macro delta vIoU over A, requiring
positive pooled mean and nonnegative means on both datasets; otherwise A. Tie
keeps A, then predefined arm order. Orders, clean, expert/nonexpert, gross gain,
gross loss, .3/.5 correctness transitions, >5pp harm and source influence are
reported; the rule is not changed after seeing them. Confirmation is descriptive
and never feeds selection. This finite development rule is not a deployment
promotion and does not guarantee statistical significance.

Round2 compares A, correction C, new temporal output T, and C+T. Also separately
compare C under OLD versus NEW acquisition at the SAME old final interval,
plus C+T with new acquisition. If C does not use Routed evidence, retain Routed
direct-selection as an explicitly labeled acquisition diagnostic. A+two Uniform5
uses original endpoint Uniform5 for persistent learning and a second independent
bin-midpoint Uniform5 for current correction; each has five distinct frames.
This matches two observations without averaging two masks into a new teacher.

Temporal view0 is original 2Hz cached UniversalVTG input, phase0; view1 genuinely
re-extracts images at phase .25 seconds, nearest frozen STVG observed frames.
Both use the same existing video encoder, original text, model, precision and
student temporal candidates (including native). View1 keeps floor(duration*2)
slots, shifts its physical origin .25 seconds, and maps predictions back by
(seconds+.25)*fps+original_first_frame. Duration for its head is duration-.25.
Record actual selected frame IDs, unique and different observation fractions;
duplicate picks remain, and coincident views are reported, not replaced. This
is genuine image resampling on the observed STVG frame grid, not independent
original full-rate media evidence or same-forward proposal consensus.

Scores are max(confidence*interval-IoU). For candidate k, subtract each view's
fixed native score; choose unique positive maximum of min-view differences.
Native wins ties and nonpositive improvements. New observation frames are
original deterministic five weighted quantiles of that ONE selected interval,
with existing deterministic distinct-frame fill. No widening/merging support.
Event-frame precision, sparse valid support, candidate ranking, fixed-interval
spatial gain and final vIoU are separately measured. New acquisition is eligible
only for positive paired fixed-time spatial mean on both datasets and a positive
pooled mean; final combination is highest pooled corrupt-all delta among the
eligible A/C/T/CT alternatives, requiring both dataset means nonnegative. The
same rule selects temporal output. All decisions seal before confirmation.

All predictions in each stage and BOTH datasets seal before that stage's GT
read; confirmation starts only after frozen final selection. GT is exclusively
CPU evaluation and diagnosis, never online admission, mask acquisition, temporal
ranking or test-time target construction. Paired 10,000 source bootstrap with
seed20261001, source→condition→order macro; paired dataset-stratified pooled
intervals are reported. Parameter arithmetic, masks, hashes, candidates, native
mapping and an independent official dense metric are checked. Preserve positive,
negative, null, empty and tied examples. Cost distinguishes logical five-frame
observations, input-identical cache reuse, actual unique model invocations,
suffix replay/backward counts and process wall time (not pure GPU kernels).

No total deadline, no new hyperparameter search, no cross-query fast memory,
extra quality-head training, source annotation fitting, new models or changes
to corruption. Publish implementation/protocol/all anonymous results and
limitations to Zonglin-He/A, verify remote, archive check/snapshot/check.
