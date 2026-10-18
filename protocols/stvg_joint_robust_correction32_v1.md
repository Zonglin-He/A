# Fixed four-arm multi-candidate correction, exposed development32

The human explicitly authorizes this finite CPU cache experiment after e95bef20.
No training, network inference, expert call, new visual label, online stream replay,
dense OPD, independent confirmation or production promotion. Original HC16/Vid16
parents, original query-start Before state (primary), Source sensitivity, original
four planned positions, final-distribution MBR, chart(cx,cy,logw,logh), four scalar
correction and whole-trajectory application are fixed. All missing/no-op queries
stay in the domain denominator. Source/Before are never combined into one arm.

## Inputs and first required check

Original rollback32 saved expert probes contain <=3 post-original-NMS candidates
per frame. Use ONLY the ORIGINAL admitted frames, not GT-common positions and not
newly admitting ambiguous/low-score frames. This preserves A's actual DINO support
(56 HC/50 Vid); candidate alternatives may have lower scores than the old winner.
Use all legal boxes in each retained original probe, with no new score reranking,
NMS, pool expansion, label filter or target-tuned threshold. This tests association
within the retained pool, not recovery of boxes discarded before that pool.
Every original Top1 box/candidate ID must replay exactly from its original probe
and match the separately saved raw-recapture OLD probe. Raw recapture is only a
provenance cross-check; it is not substituted for the original expert output.
Original pixel/input receipt hashes and frozen base/MBR cache hashes are bound.
No RGB decode/view or new identity annotation is performed.

## Source-only calibration, fixed before target outputs or scoring

Use ONLY existing source_train36/domain records and source GT cache from the
joint-event/posterior source split, never its source-validation12/domain. For each
source query, read GT ONLY at its original four planned positions with exact GT.
Queries with >=2 such positions provide r_i=z(GT_i)-z(Native_i); subtract their
within-query mean to measure how constant a correct correction is. For coordinate
d set scale_d=sqrt(mean_over_queries(mean_over_anchors((r_i-mean(r))_d^2))).
Only a numerical1e-8 floor is allowed. Cutoff is the source parent-balanced weighted
95th percentile of sum_d((r_i-mean(r))_d/scale_d)^2: each query has total weight1,
its anchors equal shares; first sorted value with cumulative weight>=.95.
This fixed empirical tolerance is an engineering convention, not a theorem.
For HC targets use Vid source-train calibration; for Vid targets use HC source-
train calibration, consistent with original cross-domain Student checkpoints.
No checkpoint training/reselection, heldout validation tuning or target feedback.
Source annotation provenance and no source/target parent overlap are verified.

## Four primary conditions, no additional chosen variant

A: original admitted Top1 arithmetic-mean residual, exact old DINO_full replay.
B: identical Top1 set, global truncated least squares using locked source scales.
C: identical admitted frames, per-frame exclusive candidates plus outlier option,
global truncated least squares. Every frame contributes min(lambda,min_k distance).
D: EXACT C candidate choices, estimates and rejections; apply ONLY its two center
coordinates, setting log-width/height correction to0. Do not refit or reselect D.

For B/C enumerate all <=4^4 candidate/outlier assignments; each nonempty assignment
has the exact conditional LS mean. Evaluate the original truncated objective at
every mean and take the global minimum. Canonical all-outlier proposal delta0;
ties broken by discrete candidate indices, never closeness to Native/Student.
There is no absolute-delta penalty or minimum-support whole-query rejection gate.
One available frame can yield a one-frame-supported correction; save this clearly.
This objective can reject individual frames but cannot infer that all evidence is
wrong merely because the optimal compatible subset has one member.

Save distinct-time support count, rejected count, competing-solution objective gap
and normalized delta separation, and leave-one-frame delta distances. No reliability
score is called calibrated probability or used to introduce another decision rule.
These are four output interventions, not parameter-updating TTA.

## Barrier, evaluation, bounded decision

Lock rules/code/calibration/registries, generate ALL32 A/B/C/D in both fixed bases,
verify A exact replay and D exact shared choices, seal all outputs before any new
target GT evaluation or old identity-label loading. No GT-common availability
enters prediction generation. Target annotations are already development-exposed.

Primary full corrected vIoU and known unobserved actual-Decoder sIoU exclude all
four planned positions; also full t/s/R03/R05, within-MBR/observed/interpolated
supports, all signed per-parent changes, paired10000-parent-bootstrap seed20261018,
severe losses<-5pp, leave-one saved-score sensitivity and positive/negative sums.
Bootstrap describes fixed inherited predictions, not replay of altered streams.
Compare B/C/D with Base and A, C-B and D-C. Report both directional cohort means;
do not turn mean screens into significance or independent-confirmation claims.

After seal, evaluate candidate geometric coverage/selected quality at exact known
GT anchors and old conservative model-assisted labels (role-unknown ->UN). A
candidate IoU>=.5 is geometric quality, NOT certified identity. Unknown anchors
remain unknown. Distinguish absent good retained candidates, available-but-not-
selected geometric candidates and selected-good-but-tube-harmed. Never propagate
an old Top1 identity tag onto a different candidate. No GT-ranked candidate is used
for model output. Report weak-score choices descriptively without posthoc removal.

If C has no extra tube benefit, stop this fixed geometry-only estimator; no kernel,
threshold, source or pool sweep. If B alone benefits, it is a future-confirmation
candidate only. If C passes two-domain positive full-v/unobserved means AND beats
A/B, independent confirmation may be proposed but is not automatically launched.
D is explanatory, not permission to remove size by domain. Distinguish protection
(negative A cases rescued) from active gains. Preserve old positive/negative history.

CPU wall cap3600s, no GPU lease/model, no new downloads/media/weights. Required
closing: independent mathematical/output/metric/count audit, anonymous complete
positive/negative results and limitations, source/checkpoint provenance, verified
Zonglin-He/A public export and archive check/snapshot/check. CURRENT/MAIN/PAPER
unchanged. Do not edit old predictions, labels, protocols or production registry.
