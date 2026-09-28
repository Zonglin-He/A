# Joint-correction learnability: completed, not qualified

The registered two-seed, one-epoch learned mixer did **not** establish a
reproducible source-privileged native teacher advantage. Both vIoU point
estimates are below B1 and both confidence intervals cross zero. This is a
negative result for the locked configuration, not proof that joint correction
or learnability is impossible. No expert/OPD/target stage is activated.

Frozen PTD4B + B1 + audited union256; only 103,424 mixer parameters trained.
Two seeds each cover the fixed 618 queries / 95 Vid source parents, 155 actual
Adam calls, final fixed checkpoint. Confirmation is 447 queries / 31 newly
locked, optimization-disjoint source parents, three arms, 1,341 predictions.
GT provides privileged evidence in confirmation. The set is disjoint from the
enumerated development manifests, not guaranteed globally/pretraining unseen;
its results are now exposed. No target input/GT or validation state selection.

## Native confirmation, equally weighted source parents

| Arm | tIoU % | sIoU % | vIoU % | Delta vIoU vs B1 (pp), 95% CI |
|---|---:|---:|---:|---|
| B1 | 46.637512 | 48.627481 | 32.407600 | — |
| seed20260928 | 46.887168 | 47.764785 | 32.167599 | -0.240001 [-1.121475, +0.713042] |
| seed20260929 | 46.741185 | 48.246816 | 32.295832 | -0.111768 [-0.915397, +0.750465] |

Within-parent query means precede the parent macro. CIs use 10,000 paired-parent
bootstrap draws, fixed seed 20260927, descriptive and unadjusted. The two mixer
seeds are not independent videos. Do not average them into additional parents.

Seed1 delta t/s/v is +0.249657 / -0.862696 / -0.240001 pp; seed2 is
+0.103673 / -0.380665 / -0.111768 pp. Both have 12 positive, 18 negative and one
zero vIoU parent. Local gains remain: best parent delta vIoU is +6.373210 and
+6.799377 pp. Neither seed passes positive v mean, positive lower CI, spatial
nondegradation or complete native-good retention. Both pass nonnegative temporal
mean and the registered no-parent-v-harm-beyond-5pp condition.

## Harms and retained successes

| Measure | Seed1 | Seed2 |
|---|---:|---:|
| B1 vIoU > .5 queries retained | 127 / 136 | 126 / 136 |
| B1 tIoU > .5 queries retained | 185 / 199 | 184 / 199 |
| Parents with vIoU drop > 5 pp | 0 | 0 |
| Parents with sIoU drop > 5 pp | 5 | 3 |
| Queries with vIoU drop > 5 pp | 38 | 33 |
| Queries with vIoU gain > 5 pp | 35 | 32 |
| Changed native intervals | 139 / 447 | 141 / 447 |
| Same interval, changed boxes on common native positions | 276 | 277 |
| Changed stored reference-token support, comparable spatial pairs | 0 / 447 | 2 / 447 |

Parent-level v tails do not establish absence of query-level or spatial harm.
All 1,341 outputs pass native format. Invalid geometric frames still exist:
B1/seed1/seed2 counts 29/18/27; they are retained and scored, not filtered.
Ten queries without annotated spatial support remain in the registered denominator.
The intervals and box supports change together in 139/141 cases; this readback
does not assign all sIoU differences to an isolated spatial mechanism.

All learned confirmation corrections approach the registered norm bound
0.1354558043: seed1 range [0.1354511552, 0.1354550287], seed2
[0.1354085147, 0.1354387205]. This is an observed magnitude fact, not proof that
saturation causes the lack of utility or an authorization to scan amplitudes.

## Verification and failure disclosure

All 2,235 files and 1,341 prediction identities sealed before metric scoring.
Both final checkpoint SHAs, B1 identity, source video/frame/preprocess and
captured-field hashes, zero Base correction, declared same-field injection,
actual two-pass call records, 0 evaluation optimizer steps, no GT decoder prefix
and locked dependencies pass. No pixel re-decoding is claimed by the CPU audit.

4,023 scalar geometry metrics versus the independent tensor implementation have
maximum absolute discrepancy 4.4408920985e-16. Root's independent 263 summary,
parent CI, retention and tail checks differ by at most 3.3750779949e-14. Every
confirmation case is included in the native support readback. The root reducer
also passes a synthetic +1pp/-20pp, 447-query/31-parent control and rejects an
incorrect reported mean; these CPU controls are distinct from model efficacy.

Both completed training checkpoints pass integer Adam/live Parameter restoration,
exact basis and deterministic 618-query coverage. First CPU auditor v1 wrongly
required two passes even on event failure; v2 fixed that case but wrongly assumed
all early event failures lack time logits. Seed2's one such failure retains a
valid two-endpoint objective and exact replay while spatial is absent. Isolated
v3 validates that original contract; all failed audit scripts/records remain.
Neither repair changed scientific pins, data, loss, predictions or GPU execution.
See the separate both-seed training report for all missing-action counts.

## Decision and cost

This registered configuration receives **NO-GO for privileged teacher
qualification**. Capacity from an analytic oracle has not yet translated into
learned confirmation utility. It does not refute all possible joint corrections,
prove optimization sufficient, or justify returning to independent correction.
Source-only results do not measure expert evidence quality, OPD absorption or TTA.

The current two-seed experiment is closed; preserve both final states, every
prediction, positive/negative result and historical optimizer correction.
No automatic expert, OPD, target, 64-parent expansion, precision/step/LR/lambda
sweep or rerun. Any future reopening needs an explicit discriminating hypothesis
and separately locked protocol; finite component interventions remain a deferred
ablation, not retrospectively a prerequisite. Saved norm behavior and native
case changes can inform a later audit without using this confirmation set to
select a new state.

Confirmation worker 1932.892334826 seconds + non-overlapping measured wrapper
11.308084548 seconds. All historical settled allocation time is
63593.56801247615 seconds, cap=null. Actual research controller and GPU worker
exited; ToDesk remains untouched. No research artifact was deleted.
