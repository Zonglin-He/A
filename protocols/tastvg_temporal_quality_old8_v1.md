# Fixed Old8 temporal localization-quality signal

User authorization: 2026-10-03, after the cbd7d3b candidate-coverage result.
This experiment changes only the temporal candidate scoring signal. Keep A and
its original Old8; do not integrate New8 or restart paused experiments.

## Question and scope

Can text-conditioned contrast between an interval and its immediate temporal
surroundings lower selection regret and improve actual tube quality on **the
same eight student candidates**? This is a new, unvalidated localization proxy,
not an IoU estimator. It does not test richer candidate coverage.

Both datasets retain the current 32 development + 16 confirmation sources,
one query/source, two orders, clean + five fixed 5% transient corruptions, and
25% scheduled expert positions. There are 1,152 arrivals and 288 expert positions
(240 corruption, 48 clean). All sources have historical exposure; confirmation
is source-disjoint within this batch, not fresh. No formula selection or tuning
uses confirmation. Both panels use the single rule specified below.

A is the experimental Uniform5 persistent spatial Rank-RKL trajectory plus the
original temporal Fast critic, not Frozen or the production CURRENT_METHOD.
Vid lr .033761698432507946 / teacher .34902548789596055 / K1; HC lr
.006097133675874025 / teacher1 / K8. rho .05, student1, D4, nine probes, 1,792
parameters, original Paper48 sampling, official same-domain checkpoints remain
fixed. The new selector reads only the arrival's unchanged Old8. Spatial boxes,
pre/post persistent states, corruption pixels and expert schedule are identical
to A. Nonexpert positions are exactly A.

## One predeclared quality signal

UniversalVTG cached PE-Core-L14-336 features contain projected video columns and
a `poolandtoken` text tensor. Column zero is the projected pooled query; use it
without token averaging. Let `s_j = cosine(video_j, text_0)` in float64.
The cached observations are phase-zero 2 Hz; their piecewise constant bins start
at j/2 seconds in the original observed window, with the last bin ending at
the actual window end. Preserve repeated observations; no new frames or view.

For candidate I=[a,b), define adjacent outer bands [a-.25(b-a),a) and
[b,b+.25(b-a)), clipped to the observed window. The score is the overlap-length
weighted **mean of s inside I minus mean of s across both outer bands**. Use
continuous fractional-bin weights, not integer rounding or additional endpoint
generation. No confidence multiplier, proposal voting, clustering, consensus,
cross-view agreement, model fitting or label-derived thresholds.

The geometric outer ratio .25 is fixed from AutoLoc's published OIC recipe,
not selected on this project's GT. AutoLoc trains with video-level action labels
and uses a learned class activation sequence. Here frozen PE query similarities
replace that sequence solely as an inference experiment; this does not inherit
AutoLoc's training protocol or accuracy guarantees.

A full-window candidate has no observed outer band and receives zero contrast
against the same global window mean. It remains in Old8. A zero-norm embedding
or an entirely constant activation curve (range <=1e-12) gives no usable new
evidence, so the entire decision falls back to A. Otherwise choose the maximum
contrast; numerical ties within 1e-12 follow original candidate order, native
first. All eight scores and support diagnostics are retained, including zero
outer-band cases, repeated/invalid inputs and failures. Nonfinite cached values
cause an engineering failure, not a GT-based repair.

References: [AutoLoc, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.pdf),
[official PE implementation](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md).
Local feature contract: `external/UniversalVTG/universal_vtg_inference.py` and
`feature_extraction/extract_text_features.py`, pinned in the execution lock.

## Execution, metrics, and decision

CPU only: re-use exact cache inputs and Old8, reproduce every original critic
score and selected index, then seal new selections for **both datasets and both
panels** before opening diagnostic labels. This process has no target training,
source-side supervised quality-head training, expert inference, student forward,
backward, new temporal view, FPS change or candidate expansion.

Use the official dense scorer with A's fixed full boxes. GT is only offline
scoring and oracle on the fixed Old8; never decoder input. Primary: actual
vIoU and same-support oracle regret, paired by source. Also tIoU, regret
reduction/fraction, harmed/improved/unchanged, >5pp negative tails, strict .3/.5
correct-to-wrong and wrong-to-correct, original rerank gains destroyed, native
replacement errors, clean, both orders, leave-one-source-out and examples.
Report all-flow, expert and nonexpert subsets separately. Do not approximate
all-flow gains by multiplying expert means by .25; membership varies by order.
Use existing equal-source aggregation and 10,000 source-cluster bootstrap,
seed 20261003. No fresh-test or future-transfer claim from this readout-only
intervention. No automatic promotion, new loss, memory, sampling or extra view.
Preserve positive and negative results; changes to the scoring rule require a
new authorization/experiment. The closing deliverable includes report, figures,
independent root and anonymous public audits, archive check/snapshot/check, and
verified code/results synchronization to Zonglin-He/A.
