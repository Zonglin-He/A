# Source24 strong-anchor / frozen-propagation isolation

Question: did the previous small observation effect reflect weak anchor
utilization or inability of the frozen prior to help unobserved decoder frames?
Old result 47309488: fixed-time GT mean v gains only .037860/.022777pp. Reopen
only by forcing representable sparse anchors to verified accurate values.

Same original source-validation12/domain, repaired HC30/Vid20 checkpoints and
cached video/query conditions; 0 new models/training/Student/expert/target.
Native time is exact at every stage. GT at available original four planned
physical positions only (HC37/Vid23). No exterior GT invention, quality filtering
or source removal. Oracle solely diagnostic, not deployable nor Direct-vs-OPD.

Minimal probability-space knot projection solves min ||u-u0||^2 subject W u=y,
1e-6<=u<=1-1e-6, using the exact current 32-knot interpolation W of decode_state.
Only knot columns touched by W may change; other logits/time remain byte exact.
Closed-form affine projection when within bounds; otherwise convex SLSQP with
analytic gradient, ftol1e-12/max200. No unobserved labels in this correction.
Verify anchor max coordinate error<2e-6, RMSE/IoU improvements, same anchors in
all positive-control arms. Fail engineering qualification if not representable;
never silently change coordinate definitions or declare oracle utilized.

Four controls, deterministic one output/query (not 4 independent samples):
N = encoded/decoded Native, no model.
A = Native projected to correct sparse anchors + original interpolation.
P = same Native start passed through frozen prior, no anchor constraints.
AP = exactly same A start, frozen prior with identical anchors projected after
all clean/proposal steps. No learned association or observation-likelihood step.
All P/AP use old16 VE steps .213->.03 and old Euler algebra, Native time pinned.
NO added global noise in either arm, to avoid crediting noise recovery as
propagation. This is a deterministic structural propagation probe; model sigma
is its frozen noise-conditioning input, NOT a claim that starting state follows
the corresponding diffusion distribution or calibrated posterior inference.
No sigma/step/solver/guidance search, no retraining, no reselection.
Empty-anchor A=N and AP=P exact. A and AP must have matched final anchor errors.

Primary: AP-A on unobserved ACTUAL decoder frame IDs, inside fixed Native time
and source annotation. Report benchmark interpolated frames separately, unknown
where unavailable; keep all12/query domain denominator for overall v/t/s/R.
Secondary: actual decoder frames outside anchor interpolation footprint (W_eval
has zero columns touching anchor-corrected knots). This excludes improvement
obtainable from local interpolation alone. Also (AP-P)-(A-N), anchor error,
fixed-time/full-support metrics, paired signed source deltas/CI, all negatives.
Postseal full source GT scoring only. Sparse oracle boxes/availability are the
only GT input to inference. All source queries have already been exposed for
auxiliary development, not independent of pretrained Student by implication.

Proceed to cached actual-DINO projection check only if source GT qualification
passes AND AP-A mean sIoU>0 in BOTH domains on BOTH unobserved actual and outside-
footprint actual supports. This is a bounded engineering decision, not statistical
proof. If any fails, stop before DINO; no new observations/targets or dense OPD.
If qualifying, freeze this implementation and test cached DINO without tuning;
never deploy GT controls or select cases/paths by GT.

GPU shared lease<=300s and cumulative old50.91494812397286s+new<=14400s,
16/20GiB memory,1GiB swap,disk30GiB/host6GiB. CPU projection/metrics separate
from cached GPU timing; no E2E claim. Preserve all old artifacts and CURRENT.
Publish code/protocol/source scalar diagnostics including failures, independently
verify remote HTTP bytes/SHA/Gitblob/fetched mode/tree/two refs fresh CAS, then
archive check/snapshot/check. Actual fixed first-source pixel panels private,
no new semantic identity labels. No new agent/chat/monitor/old queue resumption.

Version002 correction: exact decoder scalar physical-time arithmetic replaces
vector float32 conversion when constructing W. v1 outputs and error retained
in artifacts/stvg_joint_anchor_propagation_source24_v1/recovery/interpolation_precision002.
Tolerances/algorithm/model/schedule unchanged. Reuse all v1 unanchored P exactly;
recompute corrected anchor-related paths. Original cumulative GPU cap not reset.
