# Selected student rollout: A/G matched development experiment

Authorization: user attachment 84e8bb80, 2026-10-02, after db4033e P0.
Question: does a selected student tube geometric VJP, instead of geometry-RKL
actuation, improve future nonexpert utility at its own state-local RKL magnitude?
Old E/F negative means have confidence intervals crossing zero. Probe sensitivity
observations motivate this mechanism; they do not exclude all alternatives.

Both original32 development sources/dataset, one query/source, two original
orders, original clean+five5% transient deployment corruption pixels. 25% experts
at positions0,4,...,28. Two arms A/G newly execute384 arrivals each/dataset, total
1536. Historically exposed development sources, not fresh testing. Official
same-domain checkpoints, two offsets, cache H, Sa2VA/UniversalVTG receipts,
Paper48 frame sampling, original nine candidates and1792 spatial parameters stay
fixed. Vid lr=.033761698432507946, teacherT=.34902548789596055,K1; HC2
lr=.006097133675874025,teacherT=1,K8. Both rho=.05,studentT=1,D4. No other tuning.

At each inner step regenerate nine candidates from the arm's current state with
fixed original source-relative probe offsets. Candidate0 is central; then
+u0,-u0,+u1,-u1,... . First exact argmax of raw specialist rewards selects k.
Exact flat spread=0 or k=0 is an exact G no-op. This introduces no margin threshold;
non-central ties retain first original index. The unchanged A rank loss uses its
old average-rank tie tolerance1e-12. Empty specialist support is the original
no-update early break. Nonempty central/flat no-op does not reduce K8 steps.

A uses original FP32 SGD on reverse-KL over geometric candidate compatibility;
bitwise positive control to all384 predecessor A states/predictions/gradients.
G target is detached student-generated B_k, not a specialist box. Geometric
D uses unchanged native bbox and GIoU coefficients across sampled clip. Compute
full1792 parameter VJP gG=grad D(Bpsi,sg B_k); do not restrict to probe subspace
or materialize full Jacobian. G executes -lr*||gA||*gG/(||gG||+1e-12), where gA
is original RKL gradient evaluated at G's own current state, not old A trajectory.
Zero gG or zero RKL magnitude is no-op, logged separately. FP32 rounding is checked.
G is a self-generated hard target chosen by specialist, not a claim of no
pseudo-targeting. Rank/KL remains only as direction-independent magnitude
reference and diagnostic in G, so this is not wholly KL-free or backward-free.

Seal current prediction before spatial write. Expert fetched once/arrival; HC
K8 refreshes candidates, both gradients and chosen target every step. Current
Fast temporal rerank and normal native dynamic routing unchanged. Persistent
space state helps subsequent arrivals. Reset at every condition/order. No
persistent temporal head updates, proximal/cap/anchor/top2/other direction arms.

CPU tests analytic VJP, full-space action, detachment, first-argmax ties, all
no-op reasons and functional movement. Fixed first two scheduled clean order1
inputs/dataset no-GT smoke verifies A bitwise, G algebra, current output,
provider once, K8 refresh, full-model restoration. No new backbone or expert
inference; replay uses cached post-multimodal H. Run finite serial Vid A/G then
HC A/G. All four streams GLOBAL barrier before any new GT score. GT CPU only;
no GT influences selection, updates or sampling.

Primary: corrupt future nonexpert source-macro dense delta vIoU vs Frozen;
paired G-A10000 source-bootstrap seed20261001. Also whole/clean/expert, tIoU,
sIoU, gross gain/loss vsA andvsFrozen, source and arrival negative tails, positive
and negative cases. Log reward/selection/target gradient/full states, both
counterfactual and actual norm, selected D before/after, mean absolute normalized
box movement and cosine(post-central, selected-central). Zero output/target
vector has null cosine and exclusion count, never replaced by zero. Log actual
backward counts (G additional VJP), replays/providers and worker wall, not pure
GPU kernel time. Frozen-target RKL after loss is diagnostic only.

Independent CPU audit checks all input/prediction/state receipts, raw rewards,
ranking, compatibility RKL, selected D, full FP32 actuation coordinate-by-coordinate,
functional scalars and GT utilities, dual dense metrics. GT local utility uses
fixed current pre-update temporal support. Good target with harmful G or negative
cosine does not uniquely isolate decoder: finite step size, nonsmooth geometry,
routing, learned parameterization and critic error remain competing causes.
Preferred development arm G only if strictly higher primary mean; ties within
1e-12 retain A. This is a resource decision, not significance or production
promotion. Neither outcome automatically launches event-conditioned expert
frames, new models, parameter searches, fullquery or historical paused queues.

Single GPU serial, no overall time deadline, minimum8GiB disk. Preserve old and
failed assets. Publish implementation/protocol/anonymous full scalars/negatives/
figures/costs to Zonglin-He/A, remote-byte verification, archive check/snapshot/check,
FINAL_COMPLETION before pausing the reused hourly Luna max monitor.

Literature analogies only: https://arxiv.org/abs/2609.33791 discusses selected
LLM token update directions; https://arxiv.org/abs/2605.09725 selects teacher
trajectories with correctness and even GT recovery. Neither gives guarantees
for unlabeled STVG or authorizes GT filtering of current online feedback.
