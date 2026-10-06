# Explicit spatial-policy OPD v1 (2026-10-06)

User attachment 5ab425de authorizes this independent research experiment and a
saved interruption of the running paper queue, followed by its exact resumption.
Production CURRENT and all old prediction/configuration/runtime bytes remain
unchanged. The public research branch is research/decota-spatial-opd-v1.

## Hypothesis and mechanism

Sampling and training now use the same explicit frame-aligned output policy.
The final merged normalized cxcywh box b from the original two-offset decoder
is the deterministic central readout. The new exploration policy is
N(logit(b), .25^2 I4), mapped through sigmoid. The original box is returned
directly for exact zero-update equality; sigmoid(logit(b)) has a separately
tested numerical round-trip tolerance. No offset is discarded or interpolated
before training. Exact 0/1 coordinates fail the chart qualification rather
than silently clamping or changing exploration.

Original Native-WHEN/Uniform4, single frozen DINO, admission + per-frame Top1,
1792 query+three last spatial LNs, Adam .03/betas .9,.999/eps1e-8/no decay stay.
Each of ten rounds draws 16 independent Gaussian vectors and their negatives
at each accepted frame, M=32; all actions, scores and weights are detached.
IoU only evaluates sampled student actions, never differentiates into them.
w=softmax(reward/.25); do not multiply by policy density or reward ranks.
The objective is analytic KL(pi_rollout||pi_student) minus
sum((w-1/32) log pi_student(z)), averaged over accepted frames. This estimates
the forward-KL gradient, not its value up to the unknown teacher entropy.
The KL gradient at a fresh on-policy round is zero; it is not a post-update
trust-region guarantee. Skip optimizer.step for globally equal reward, even
with nonzero Adam moments. No dynamic correctness gate is added.

Fixed ten rounds, output ordinary last central boxes; no best sample, best
checkpoint or loss winner. Residual/Adam reset each query; only final LN delta
times1/16 persists, independently for each arm, condition and complete order.
Frozen rollout holds initial samples/weights/pi0 for all ten rounds and is
called frozen-rollout distillation, not an importance-corrected off-policy
algorithm. Shuffled-feedback permutes each frame's reward-to-action mapping;
the same draws and reward multiset are retained. Frozen is zero-update source.

## Finite locked roster and sequence

Reuse paper query-only official metadata/subject parses/canonical input rules.
Deterministically hash-select one query per parent source, without annotations.
Development:32 previously exposed development parents per dataset. Qualification
uses first two development parents per direction, excluded as fresh evidence.
Confirmation:128 parents per dataset from the original outside-development
pool, disjoint from this round's development, never used to choose this method.
The official entire pools have historical exposure; confirmation is not fresh.
No result-driven query exclusions, temperature/scale search or per-dataset
configuration. 128 parents is a larger locked diagnostic, not a power guarantee.

Primary clean cross-domain: Vid source to HC2 validation and HC2 source to
Vid test; development and confirmation each two independently reset complete
parent-blocked orders. Mechanism check: same-domain official checkpoint, the
same 32 development parents per dataset, one fixed order and frame_drop/
frame_freeze/motion_blur/occlusion/exposure at5% physical burst coverage. This new development
roster resolves parent units for this experiment only; it does not answer or
release the old Table2 unit hold.

Three adapted arms share fixed input/evidence per cell: on_policy,
frozen_rollout, shuffled_feedback. Clean:640 logical arrivals/arm. Same-domain
corruption:320 arrivals/arm. Total960/arm,2880 adapted arrivals;640 unique
source/input/condition cells. Inputs/evidence from the old sealed clean
cross cache are reused only after exact pixel/native/interface checks.
Same-domain changed-input cells use the same single DINO and reuse their
newly cached evidence across arms; no second model or extra observations.

CPU contracts first; then real two-offset zero-update parity, gradient
isolation, analytic mean-gradient and full1792 Adam arithmetic qualification.
Require finite nontrivial sampling diagnostics; do not tune using confirmation
GT. Development predictions for all arms seal before development GT audit.
If interface/execution qualifies, confirmation uses the same prelocked config
regardless of development efficacy. If the mechanism qualification fails,
preserve failure and stop that stage for root review, never weaken checks.

## Evidence, scoring and completion

GPU workers have an annotation/result open guard. All three arms and Frozen
of a stage must seal before its CPU GT scoring. Save every round's actions,
original and used rewards, weights, ESS, mean gradient, parameters, central
boxes, observed/unobserved movement and computation. Unknown GT frame stays
unknown; do not invent event-outside identity annotations.
Official dense vIoU/tIoU/sIoU plus an independent DenseTube audit; source-macro
after query/order averaging, paired10000 parent bootstrap; official query-macro
companion. Report After-Frozen, Before-Frozen and After-Before, gross gains and
losses, >5/>20pp harms, strict vIoU>.3/.5 transitions, observed/unobserved GT-frame
changes. Postseal diagnose sample support, DINO weighting versus true sample
quality, and whether final mean realizes distribution improvement. Report
on-policy versus frozen-rollout and true versus shuffled feedback, not just KL.
No per-target winner stitching or automatic promotion. New weights/experts,
temporal adaptation, scalar scorers, gates and memory are outside this run.

Root must independently audit state/LN chain, Gaussian gradient and Adam,
read results/cases, render and inspect figures, publish code/protocol/anonymous
results on the research branch, verify remote bytes/tree, update RESEARCH_HISTORY
check/snapshot/check. After actual OPD completion and publication, resume the
saved paper TENT optimizer state and original queue. Old Ours remains the old
IoU-energy/best-state algorithm, not the new OPD results. Old HC Fisher media
dependency and Table2 unit hold remain unresolved and cannot be omitted.
