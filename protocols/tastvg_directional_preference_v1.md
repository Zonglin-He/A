# P0: state-local magnitude matched directional preference

User authorization: the 2026-10-02 attachment requests A/E/F on the same original
32 development sources per dataset. This is a new actuation experiment, not a
rerun of old queues or promotion of a development winner.

## Question and scope

Do rank-derived probe-axis directions improve future nonexpert task utility over
the geometry-RKL direction when each update retains its own state-local
counterfactual RKL magnitude? Previous proximal/cap contrasts had negative means
with intervals crossing zero. They do not eliminate all magnitude/target
explanations. The directional experiment also restricts the update to four axes;
it cannot uniquely prove that the RKL surrogate is the only cause of failure.

All three arms are executed anew: VidSTG and HC-STVG-v2 each 32 sources, one
query/source, two original orders, clean and five original 5% physical transient
corruptions, 25% expert positions (0,4,...,28). 384 arrivals/arm/dataset, total
2304. The complete inputs/orders/receipts are copied from the preceding proximal
PLAN and hashed before inference. All sources have historical development
exposure. Official same-domain checkpoints, two offsets, existing Sa2VA and
UniversalVTG caches, original Paper48 sampling/pixels and 1792 spatial parameters
are unchanged. H cache replay occurs after the frozen multimodal encoder.

Vid: lr=.033761698432507946, teacherT=.34902548789596055, K=1.
HC2: lr=.006097133675874025, teacherT=1, K=8.
Both rho=.05, studentT=1, four original orthonormal directions, nine candidates.
Temporal remains the same current-arrival Fast selection. No persistent temporal
parameter adaptation. No new learning rate, cap, proximal mixture, expert input,
event-frame sampling, anchor/replay, baseline or full-query job is added.

## Updates

Candidate order is center,+u0,-u0,+u1,-u1,... . The nominal original four directions
and absolute source-relative probe radius are preserved. Rank is descending
zero-based average rank, with existing 1e-12 reward tie tolerance.

At every scheduled inner step each arm observes its own current candidates,
rewards and original geometry-RKL gradient g. Let s=lr*||g|| over all1792
parameters. A executes the original FP32 SGD operation psi += -lr*g bitwise.
E computes c_j=rank(r_j^-)-rank(r_j^+) and v=sum_j c_j u_j.
F picks the first j maximizing |c_j| and v=sign(c_j)u_j.
E/F execute psi += s*v/(||v||+1e-12). Rank-direction zeros produce an exact
no-op; zero RKL magnitude also produces no-op. These cases are counted and
separated because magnitude matching is impossible for a zero direction.

The magnitude reference is the original RKL counterfactual **at E/F's current
state**, never a number copied from the old A trajectory. Actual FP32 displacement
is recorded and independently checked against desired magnitude, with rounding
tolerance. At K8 candidates and RKL gradients refresh at each step; expert cache
is fetched once per arrival. Current output is sealed before any update and
persistent state serves subsequent arrivals. State resets per condition/order.

|c_j| is expert within-pair rank contrast, not teacher/student disagreement.
The aggregate E direction is not one of the evaluated candidate endpoints.
Even F's executed length need not equal the probe radius. Neither sign preference
nor movement within the probe subspace guarantees better GT. No new cap is
silently introduced, because this P0 fixes the magnitude rule.

## Qualification and readout

CPU checks: candidate pair sign/rank ties, top tie deterministic rule, no-op,
SGD bitwise operation, applicable E/F norm and old orthonormal basis. Before
formal execution, fixed first two scheduled clean inputs from order1 per dataset
are used for no-GT smoke. A must match previous A's current output, gradients and
states. E/F must preserve current output, fetch each expert once and satisfy
pairing/direction/norm/no-op on every inner step. Model state must restore.

A additionally matches all384 current outputs, gradients and pre/post states to
the sealed previous A stream. No new full-backbone parity runs are needed: the
unchanged replay/reinsertion interface already has the preceding 96 validation
forwards. They are not counted as new execution here. All optimization steps use
cached H; no new expert inference.

All six new prediction streams complete and globally seal before this batch's
GT scoring. Already exposed development labels are only opened by CPU scoring.
No GT controls direction, sampling, online update or filtering. Primary endpoint:
corrupt future nonexpert source-macro dense delta-vIoU; paired E-A/F-A bootstrap
10000 draws, seed20261001. Whole, clean, expert/nonexpert, both orders, gross gain
and loss, source and arrival negative tails are reported. Current post-update GT
is diagnostic only, not substituted for the pre-update online output.

Each inner step logs raw rewards/ranks, c_j/top pair, RKL gradients/axis projection,
cosine between -g and directional preference, desired/actual step magnitude,
no-op reason, probe-radius ratio, full pre/post state and predictions, frozen
RKL loss before/after, candidate boxes and GT utility after seal. Positive and
negative cases, including good expert top but harmful executed step, are retained.
Independent CPU audit reconstructs rank, geometry objective, both actuation rules
and every FP32 parameter step/state link, then checks dense metrics independently.

The preferred development arm is A unless E/F has strictly higher primary mean;
tie within1e-12 retains A or the earlier arm E. This is a resource/next-method
decision, not fresh significance or automatic production promotion. Even a
negative E/F outcome does not eliminate all possible gradient-direction methods.
Event-conditioned expert sampling and quality-aware temporal reranking are
conditional next questions; neither is automatically executed in this P0.

Single GPU finite serial queue, no overall time deadline. Minimum8GiB free disk;
retained failures and private raw caches stay intact. No concurrent experiment.
Publish code/protocol and all anonymous scalar results, negatives and costs to
Zonglin-He/A with remote byte/hash readback; update research archive check,
snapshot, check. The existing single Luna monitor may be reused until closeout.

## Literature scope

The language-model work motivates checking direction information; it does not
implement these parameter probes or supply STVG guarantees:
[KL-free OPD, 2609.33791](https://arxiv.org/abs/2609.33791).
Likewise [Decomposed OPD](https://proceedings.mlr.press/v306/yoon26f.html) concerns
visual/language gradient components in VLM reasoning. Our E/F are direct
parameter-space preference actuation, not standard reverse-KL OPD or an exact
reproduction of either paper.
