# R1: Oracle Distributional Temporal Adaptation

Authorization: user attachment 369ec98e, 2026-10-04. Run only R1; R2 expert
distillation and R3 persistence remain conditional future work. Stop further
frozen quality-head calibration. Production DeCoTA is distinct from scientific
TA-STVG baseline A and is unchanged.

## Question and difference from the old native-head experiment

Does correct, soft temporal supervision produce useful current-query native
parameter-update gain, with A spatial boxes and persistent spatial trajectory
held fixed? The old native-head test used one capped, reverse-KL rank-feedback
step on retained Old8 pairs, and evaluated future arrivals. This uses full legal
per-offset span distributions, GT Gaussian forward KL, three uncapped SGD steps
and episodic current-query readout. A negative result applies to this locked
interface/configuration, not every temporal adaptation route.

## Inputs and parameter scope

Reuse original Paper48 input grids, pixels, queries, checkpoints, A spatial
pre/post states and schedule. Each dataset's original design has 32 development
and 16 historically exposed confirmation sources, one query/source, two orders,
clean plus five 5% transient corruptions, 25% expert arrivals. Available final
temporal hidden covers exactly 288 scheduled cells: Vid 96 search + 48 confirm,
HC 96 + 48. Independent expert sources are Vid 16 + 8 and HC 14 + 7. Do not
extract missing nonexpert hidden, infer new videos or resume old queues.

Official same-domain checkpoints: TASTVG_VidSTG.pth and TASTVG_HCSTVG2.pth, EMA
state as used by the original loader. Reuse source-validation final hidden for
31 Vid and 16 HC official-train-derived validation sources, one query each, with
existing split and media/source-disjoint target binding. Previous exposure is
disclosed. The old 95/48 fitting sources are unused; no quality head is trained.

The temporal MLP has two layers. Recompute its frozen first 256x256 affine and
ReLU on cached decoder hidden in eval mode, then update only the final 2x256
weight and two biases: 514 nominal parameters (bias shifts cancel in normalized
span probability; effective identifiable direction count is at most 512).
No decoder, LN, spatial, routing, encoder or optimizer buffers are inherited.
FP32 head arithmetic, FP64 log-sum-exp/loss, ordinary SGD without clipping,
weight decay, momentum, early stopping or GT best-step selection. Each query
begins from the checkpoint head; readout is always after step 3, then discarded.

## Distribution and native readout

Keep both original offset records. In each offset normalize logits over **all
legal i<j spans**; do not use merged-grid or Old8 restricted likelihoods.
Use physical intervals [frame_i, frame_j+1). This corrects the attachment's
generic s<=e notation to the actual native strict support. It is an explicit
implementation mapping, not a new decoder. The two offset losses are averaged.
Final output keeps official FP32 start/end log-softmax, first-maximum tie rule
and physical envelope of the two offset MAP intervals. Native zero-update
logits and interval parity are checked against existing source/capture metadata.
CPU vs original CUDA arithmetic tolerances are disclosed; interval parity is
exact and checked for every target and source validation query.

For each offset sigma is one observed temporal cell, defined before any labels
by its median consecutive physical-frame spacing. GT boundaries remain in
original physical coordinates, with no clipping to sampled support. Normalize
log q_GT(i,j) = -[(frame_i-s_GT)^2+(frame_j+1-e_GT)^2]/(2 sigma^2) on i<j.
This Gaussian provides correct boundary supervision but is not a perfect dense
vIoU target with fixed spatial errors. Sampling support limitations are retained.

Freeze p0 at incoming checkpoint head. Loss is KL(q_GT||p_theta) +
beta KL(p0||p_theta), beta=1, K=3. The prior KL is a soft anchor penalty, not a
guaranteed trust-region bound. Lambda=.5 concerns future R2 product teacher
only and is unused in R1. No PE, scalar quality scorer, gate or expert involved.

## Source-only learning-rate selection

Before reading target GT for this experiment, evaluate the five locked learning
rates [.0001,.0003,.001,.003,.01] on all existing validation queries, reset each
query. Select per dataset by greatest equal-source after-step-3 physical tIoU,
then smallest learning rate on exact ties. Seal both choices before target
adaptation. No target tuning or automatic expansion of the grid. All five paths
and failures are kept; prior labels/exposure cannot be described as untouched.

## Measurement, audit and decision

R1 intentionally uses target GT in the teacher; it is a supervised capacity and
execution diagnostic, not deployable unsupervised TTA. Seal predictions for both
datasets before separate dense scoring. GT use is explicit, not falsely marked
GT-free. Reuse A boxes; score zero-update native N, after step3 R1, A8, teacher
MAP and GT-time control with official dense scorer, independently cross-check.
Primary expert endpoints: R1-N tIoU and vIoU. Practical secondary: R1-A8.
Report search/confirm separately, corrupt/clean, both orders, source macro and
10,000 paired source-bootstrap intervals, >5pp harms, loss decrease versus task
gain and positive/negative cases. An all-flow emulation keeps the other 864
arrivals exactly A and applies R1 only to cached scheduled positions; it is not
a new complete online execution. Avoid multiplying expert gain by 25%.

Independently recompute all three gradients and SGD updates from joint-marginal
cross-entropy algebra, all objectives/decodes and dense metrics, source choice,
reset semantics and unchanged A/CURRENT hashes. Preserve failed runs/revisions.
Export code, protocol, source scalar paths, all anonymous target scalar traces,
metrics, uncertainty and figures. Exclude weights, hidden, GT coordinates,
private media/annotations/raw caches. GitHub remote contents must be verified.

If GT supervision has substantive gains in both datasets, the gradient channel
is supported within this scope and R2 can be designed with the same frozen
setup. If it is weak, report the observed optimization, discrete MAP and support
limitations; do not automatically add LN/decoder or expand search. Positive
oracle effects do not establish expert quality or persistent online transfer.
