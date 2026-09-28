# Matched source task-signal and calibration-interface control

The source estimator screen is completed and near-same; do not rerun target
TTA for that change. Current unlabeled consistency/alignment has demonstrated
clean/noise tradeoffs but its relation to the main task in the66816 calibration
subspace is unknown. Old auxiliary-training gradients were a different
objective, scope and checkpoint. This is source diagnostic development, not
a new target method or an oracle target result.

Use all16 preselected independent Vid training parents in the immutable
aux_backflow_v1/label_audit/PANEL.json. No outcome-based reselection. All16
source records are CE eligible; sampling lengths7–32 remain unchanged. Same
official frozen PTD4B, B1_FIXED_FINAL adapter, observed clean pixels and
stock multimodal caption features. No new target inputs or target GT.

Three arms: no-update B1, unlabeled original mild-view calib-align, and a
source-labelled task-CE positive control. Each update arm starts from the
same B1 state and fresh AdamW1e-5, wd0, clip1, exactly3 steps; only66816
FiLM/LN parameters may change. Gates/heads/out/textpool/backbone frozen.
Unlabeled teacher is fixed B1 on observed; student brightness1.05/contrast.95,
original population moments, latent/ref/event/mean-parameter-anchor weights1,
alignment.01, joint0, no output anchor. Source supervision never enters this
arm's update function or step selection.

Task control uses the already implemented GT teacher-forced reference/time
CE mean plus box CE mean, weights1+1, no auxiliary BCE. Masks, official MTP
blocks, position/context limits and sampled GT endpoints are unchanged.
Frozen backbone train mode is used only for its existing nonreentrant
gradient checkpointing; assert no active nonzero dropout. This CE path is
not claimed equivalent to native cached generation. Native evaluation uses
the verified shared-reference two-pass cached decoder, event reference/time
and a new spatial KV, with invalid geometry/format failures retained.

At every step in both arms, measure task event/spatial gradients and all
unlabeled objective components in the exact same ordered66816 subspace.
Save finite raw vectors, actually applied gradient, actual Adam delta,
live-Parameter-bound counters, clip statistics, task dot delta and cosine.
Measurement is side-effect free; optimizer uses only the declared arm's
vector. No task-gradient projection or outcome-dependent coefficient. Record
task CE before and after3 steps, full free time/boxes and actual injections.
No-update and both final predictions for all16 are sealed before geometric
scoring. Source training labels are explicitly used during supervised
updates/task diagnostics, so this is not a label-blind worker. Evaluation
labels are source only; unlabeled update has no label input.

CPU: real hidden128 synthetic module, exactly reproduce the prior unlabeled
three-step implementation, scope/actual counters/exact reset, and branch
gradient algebra. Before GPU, lock code, source roster/records, checkpoint,
original moments and scientific protocol. Runtime independently checks
physical pixel/query/grid/time hashes against sealed source moments and
teacher-forced inputs against the same observed pixels. Save raw native
predictions before structural validation. Never reject legal invalid boxes.

Readout: all16 per-source deltas, parent-macro v/s/t and paired descriptive
bootstrap CIs; format failures, existing B1 v/t>.5 retention and >5pp tails.
No score threshold selects updates. Scalar geometry plus a second tensor
implementation; NumPy recomputation of gradient/Adam algebra, CE and scope.
One source query per parent, source-training exposure, one seed; no target
generalization claim. CE descent and first-order proxies do not prove tube
benefit. Supervised CE may itself trade free-decoding accuracy for teacher
forcing; retain that outcome.

If supervised control improves task CE/native output while unlabeled updates
oppose the task, prioritize signal construction. If both fail in native
readout, examine optimization sufficiency or readout before blaming only
self-supervision. If both help, current target tradeoff remains a domain/view
question. Mixed/zero results stay inconclusive; no automatic64/production
promotion or LR/lambda/teacher grid. Report every source and branch.

One3600s serial engineering allocation,8GiB free disk floor, cap=null. Expected
96 fresh Adam steps,48 predictions; failed/replayed steps and wrapper time
count separately. Each completed episode commits a hash seal. On failure
retain partials/receipts, do not resume silently or change samples. Estimated
new storage <=320MB; guard disk throughout. No old scientific pins modified.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
