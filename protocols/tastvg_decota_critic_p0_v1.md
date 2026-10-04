# Spatial-DeCoTA: Direct versus distributional energy critic, P0

## Question and scope

Can a differentiable proposal-distribution objective retain current-query
correction when the C1/Scale06 spatial interface is kept? This P0 runs two
episodic spatial arms. The temporal output is the checkpoint's native I0;
there is no temporal adaptation, LN consolidation, new expert, policy
likelihood, preservation term, gate search, or hyperparameter sweep.

The old online spatial-after gain includes inherited LN. It is not the
episodic current-only baseline for this experiment. Current-query benefit
does not establish future online benefit. Neither a differentiable external
coordinate objective nor calling it a critic establishes OPD/novelty or
eliminates weak/pseudo supervision.

## Fixed inputs and budget

Exactly the completed C1 same-domain panel: VidSTG and HC-STVG-v2, each 32
development plus 16 source-disjoint but historically exposed confirmation
sources, one query per source, both original orders, clean plus frame_drop_5,
frame_freeze_5, motion_blur_5, occlusion_5, exposure_5. Official same-domain
EMA checkpoints, original Paper48 pixels/sampling/two offsets and parser.
There are 1,152 logical arrivals per arm; 576 unique inputs are computed
per arm and receipted outputs are reused for the second order. Episodic
order invariance is checked, not described as a persistent stream.

Use the already sealed/hash-bound C1 original four native-I0 observations
and Frozen H. No new DINO forward is needed. Complete original within-input
NMS top-three supports are cached for the context/target-token path. Its
original phrase-score NMS .5 and truncate-then-geometry filtering remain.
The historical fallback saved only the selected single box: retain that
single support, disclose its count, never invent the other two. Use only
valid finite boxes and existing target-token/fallback scores. Inference
and all selection workers deny annotation/GT/scored-result access.

## Matched parameter interface and two arms

Each input begins with checkpoint spatial LN, zero query residual and fresh
Adam. Query256 + final spatial block norm1/norm3/norm4 affine1536 = 1792.
Both use lr .03, betas(.9,.999), eps1e-8, wd0, ten updates, all eleven states
0..10, earliest minimum of their own objective. Empty evidence is an exact
no-op. Source model parameters are restored. No optimizer/state transfer.

**Direct** calls the original Scale06 fit. It preserves DINO admission,
top-one accepted anchor and loss sum(5 L1 + 2 GIoU)/planned4.

**Critic** retains every nonempty valid cached support (M<=3) without the
old .35/.05 score/margin admission. On each observed frame j:

    log w_jm = logsoftmax(existing_score_jm / 1)
    E_j(b) = -logsumexp_m(log w_jm + IoU(b,e_jm)/1)
    L = mean_j E_j(b_theta(j)) over nonempty valid observation frames

Both temperatures are explicitly fixed to one before inference; the
attachment specified no numerical temperatures. No confidence weighting
search is performed. This is softmax of existing scores with temperature
one, not a confidence-as-correctness claim. No GIoU substitute is allowed.
Ordinary IoU can have zero coordinate gradient when all proposals are
disjoint; record this failure mode without changing the objective.

The changed factor is the supervision construction block: retained support,
admission, reduction and differentiable objective change together. Results
cannot uniquely attribute an effect to energy versus L1 or to uncertainty.
Best proxy loss does not ensure GT safety; step zero is an available source
state, not a correctness gate. Prediction uses the selected spatial tube
and the unchanged native physical temporal interval.

## Validation, sealing and metrics

CPU contracts verify single-proposal equivalence, permutation/score-shift
invariance, finite-difference gradients, empty/no-overlap behavior and
finite multimodal gradients. For one original clean input per dataset,
check cached/full-backbone native parity, all-step original Direct parity,
cached/full Critic parity and selected-state full-model reinsertion with
unchanged temporal outputs. These two smoke forwards are the only new
backbone work, disclosed separately from cache-only formal adaptation.

Seal all 576 two-arm payloads globally before CPU GT scoring. Read only
the existing matched labels then; never choose an optimizer step by GT.
Independently recompute NumPy spatial losses/energies, FP32 Adam arithmetic,
source reset, selected state and temporal interval. Compare official dense
scorer with an independent vectorized scorer. All rows remain, including
empty evidence, zero gradients, bad losses and severe harms.

Primary: per-dataset/split source-macro corruption dense vIoU, Critic minus
Frozen, Direct minus Frozen and Critic minus Direct. Average the two orders
and five corruptions within source; paired 10,000-source bootstrap seed
20261004. Also clean, every corruption, tIoU invariance, dense sIoU, gross
gain/loss, >5/>20pp harms, .3/.5 correctness transitions, source concentration,
proxy-improved/GT-worse, step-zero selection, displacement and zero-IoU
gradients. Worker wall time includes loading/IO and is not pure GPU time.

## Decision and delivery

P0 reports whether sizable current correction survives with its paired
uncertainty and negative tails. Near-zero gains or reversal stop this fixed
configuration. Broad positive point estimates alone do not establish
stable benefit; retain evidence as inconclusive if intervals are wide.
This P0 does not automatically promote a method or run LN persistence or
cross-domain. Those are subsequent conditional stages, not completed here.

Bounded work: two smoke inputs, 576 unique paired fits, <=11,520 backward
calls, <=8GiB new artifacts; fail on disk<8GiB, engineering/nonfinite errors.
No total deadline and no restart of historical controllers. Publish new
code/protocol/contracts and anonymous complete results, retain failures,
verify GitHub Zonglin-He/A bytes, update RESEARCH_HISTORY check/snapshot/check.
