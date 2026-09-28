# DESTA-3D v2 auxiliary backflow audit and fixed source contrast

Registered in response to user private authorization (not distributed).
This is a new source-only experiment. Original source-fit checkpoints, predictions,
configuration and optimizer states remain unchanged. No target labels, target TTA,
larger teacher, P1 features, gate multiplier, PCGrad or hyperparameter grid.

## Question and prior evidence

The shared-reference contract is repaired. In the audited early B checkpoint its
198 source-validation outputs were identical to the old path; that repair did not
improve task utility. Eight source training parents gave weighted auxiliary/task
gradient norm ratios 64.24–1123.21 on the 19,968 shared-stem parameters. This
suggests, but does not establish, harmful auxiliary backflow. The expression
`1 + ratio * cosine` concerns the raw infinitesimal SGD direction on this subset;
it is not an AdamW update or a whole-tube result.

## 0. Source label and CPU contracts

Audit all 618 training records, including the six without legal response CE.
Compare actual response time endpoints with sampled event-active endpoints,
physical half-open annotation intervals, and boundary/interior missing boxes.
Keep static-builder counterexamples separate from observed production records.
Select up to 16 distinct training parents by fixed event-duration/box-area strata,
never by scores, gradient sign, or validation results. Report reused exposure.

Gradient scaling has exact identity forward behavior and a detached coefficient.
It applies only on the auxiliary path entering the referent and event heads.
The residual/task path is unchanged. Event-head input scaling after deterministic
spatial pooling gives the same scalar upstream attenuation as scaling before the
pool. Head parameter gradients remain unscaled. Zero gradients have undefined
cosines, not 90-degree angles.

## 1. Real optimizer panel before the new fit

Use the next naturally paused original fit003 checkpoint, including its exact
dual_repaired AdamW state. Save an isolated CPU snapshot, recording epoch, cursor,
adapter/optimizer identity and the original checkpoint file hash. Original LATEST
is not overwritten. On the fixed 16-parent training panel, use four windows of
four queries. Each window starts from the same snapshot and optimizer state;
three counterfactual updates B0/B1/B2 independently start there. They are diagnostic
updates, discarded afterwards, not trajectory training or a selected state.

Measure event/ref-time task gradient gT, spatial task gradient gS, their sum,
weighted auxiliary gradient, raw total, clipped total, and actual AdamW parameter
delta. Report norms and cosines by the real adapter parameter groups, plus
g_task dot delta. Also measure injected residual magnitudes. Do not infer vIoU
from a gradient dot product. First-query real gradient equivalence checks B0
decomposition against the existing combined-loss implementation, with matching
RNG. Backbone parameters remain frozen and gradient-free.

## 2. One common evidence-only start, then three fixed B arms

Original E0 adapter weights were not retained. A partial B state cannot be called
an evidence-only start. Therefore regenerate ONE explicitly new common A stage:
same dual-v2 initialization seed 20260927, architecture, source inputs, query order,
pixel preprocessing, 618 training queries/95 parents, one evidence epoch. Verify
the initial dual hash against the original INITIAL record. This is newly executed
source warmup, not claimed recovery of the missing original E0 weights.

Save A final immutably. B0/B1/B2 all start from this exact state with identical fresh
AdamW states, and see the same shuffled B query order and physical pixels. Run
exactly ONE B epoch, 618 queries, accumulation 4 (last window 2), 155 updates per
arm. Keep the original repaired first-B-epoch LR schedule: reader 3e-5,
head/out/gate 1e-4, 5% warmup/cosine with original planned B horizon 775 updates.
The experiment stops at step 155; no validation-selected state or additional epoch.
Weight decay 0, clip norm 1, auxiliary scalar losses .1 each, joint 0. No loss,
sampling, architecture, decoder or gate changes between arms.

For each complete accumulation window, accumulate g_task=gT+gS and weighted
g_aux before computing ONE detached coefficient on shared_stem:

- B0: c=1, unchanged auxiliary representation gradient.
- B1: c=0, auxiliary heads train but auxiliary representation backflow stops.
- B2: c=min(1, .25 * norm(g_task)/(norm(g_aux)+1e-12)); task norm zero gives c=0.

Apply that scalar to auxiliary contributions of all upstream representation
parameters; auxiliary readout-head gradients are unchanged. The bound is defined
on the registered shared-stem group, not claimed for every parameter group.
Computing c after accumulation avoids an unverified per-query norm bound when
gradients cancel. Tensor gradients are accumulated in FP32; distinguish absent
gradients from real zeros. Log c, group norms, clipping and real update deltas.

Original reference training and original two-pass decoder stay fixed for this
contrast. The final selected source method will still require matched shared-
reference confirmation. Do not attribute a decoder change to auxiliary scaling.

## 3. Prediction seal and source-only readout

For the same 198 source-validation queries/31 parents, retain matched Frozen,
common A start, B0, B1 and B2. All new prediction arms must complete and seal before
reading source-validation GT for scoring. Frozen reuse requires physical-input
and identity checks. Report parent-macro vIoU/sIoU/tIoU, paired parent bootstrap CI,
event readout quality with defined denominators, native-good retention and >5pp
negative tail. Fixed final B states only. Source supervision is not target TTA.

Repeat the fixed training gradient panel at the final B states if useful within
the registered comparison, without selecting or modifying them. Same-video,
different-event query sensitivity is a separate diagnostic using actual source
query pairs and verified identical physical video grids, not arbitrary negatives.

## 4. Later TTA requirements, not authorized shortcuts

The existing user resource gate remains: complete selected dual source vIoU must
reach matched Frozen before target 8-source interface comparison, then 64 sources.
Source fit may establish eligibility, not success under corruption. Keep Frozen /
source-fit-no-TTA / TTA separate. Any target teacher is from the same corrupted
input and a legal view, never clean pixels or target labels. Pre-gate losses keep
the residual gates frozen. Monitor actual post-gate delta magnitudes and real
temporal/coordinate distribution drift: normalized feature consistency alone is
amplitude-blind and parameter anchor gradient is zero at the initial state.

## Resource, preservation and recovery

Cumulative cap is null; all loading, failures, replay and wrapper overhead are
counted with v1/v2 receipts. GPU is serial with the existing flock lease and 8GiB
free-space reserve. Wait for healthy fit003 to pause naturally before using GPU;
the original four-arm run is resumable at the saved cursor, not cancelled or
rewritten. Each new allocation is finite, saves a complete accumulation window
including optimizer/RNG/cursor, and is reviewed before continuation. No target-
score gate, cherry-picked source, or retroactive threshold changes.

## Primary literature boundaries

[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) motivates inspecting
gradient magnitudes but does not validate this head-preserving rule.
[Auxiliary-loss gradient similarity](https://arxiv.org/abs/1812.02224) motivates
direction checks; it does not turn raw SGD algebra into an AdamW/task guarantee.
[TTT++](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html)
and its [official implementation](https://github.com/vita-epfl/ttt-plus-plus) use
specific source auxiliary training/statistical-alignment conditions.
[ViTTA official implementation](https://github.com/wlin-at/ViTTA) addresses video
action recognition; its results do not establish STVG endpoint/tube validity.
These sources support discriminating tests, not an assumed positive outcome.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
