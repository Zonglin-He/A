# A0: fixed cached direction screen

Authorized 2026-09-29. This supersedes full618 as the next experiment. Question:
can the already prepared state-aware mixer learn an analytic native correction
direction before paying for a new full-source training/evaluation?

## Locked selection and exposure

SHA256 UTF-8 `DESTA-A0-v1|tag|value`, ascending hex order. Train: original618
queries/95 parents; one minimum `train-query` hash per parent, then the33 lowest
remaining query hashes; order by `train-order`. Dev: original447 exposed diagnosis
queries; take16 smallest `dev-parent` hashes among parents with at least4 queries,
then4 smallest `dev-query` hashes per parent. No metrics used. Retain every selected
query including missing objective support. Fresh31/388 is neither read nor used.

## Cache and controls

Frozen official PTD4B, fixed B1, frozen audited union256. Train128 gets one native
B1 capture and at most two backward calls per query. Native endpoint mean CE and
full152775 coordinate mean CE on B1 reference/interval/anchors define the already
audited analytic joint direction. No GT prefix. Source GT explicitly supplies
privilege and direction supervision. Cache complete FP32 z/qT/qS/evidence8/state33,
oracle coefficients256, physical support hashes and measured FP64 stock norm.
For Dev64 reuse sealed Gap trace, B1 prediction and oracle coefficients. Re-extract
only the frozen z/q features (absent from old artifacts) and verify original
pixel/preprocess/query/grid/time/stock support. No new Dev oracle backward or
baseline generation during caching. Preserve missing support/invalid boxes.
Save both full Train gradients, native trace and projection evidence, allowing
independent projection audit; no target inputs or labels.

## Fixed cached training

Existing StateAwareDirectionMixer, hidden128, state33, union256, radius
0.13545580427763146; seeded output initialization remains normal std1e-3.
Only mixer parameters, seed20260928, AdamW lr0.001, weight_decay0, clip1, constant
schedule,200 actual steps. Batch4 queries accumulated without padding/cropping,
reshuffle Train128 with the fixed generator each traversal;800 query occurrences.
Only loss: mean per-query `1-cos(flat coefficients, flat oracle coefficients)`.
Zero oracle has no direction label, contributes zero while keeping batch divisor4;
retained in all readouts. No CE/KL/amplitude/gate/native objective or best-step.
Training uses cached tensors with PTD unloaded. A coefficients-only execution
uses exactly existing module operations; CPU contract compares original forward.
No network change, no grid. Terminal step200 only; direction means/medians exclude
undefined zero oracle explicitly, counts always reported. Query-global cosine,
not per-token average. Save terminal predicted coefficients for independent NumPy.

## Locked decisions

Dev median cosine<0.1 -> stop without native inference. Train median<0.3 identifies
capacity/optimization as the next hypothesis; train>=0.3 identifies conditioning/
generalization. These are screening classifications, not proven causes. No automatic
capacity or state change in this run.
Dev median>=0.1 ->64 true A0 free-native predictions. Reuse the64 B1 after exact
physical support verification. Same joint field enters both native passes. Seal
all predictions before independent scalar/tensor scoring. Parent macro is primary
(4 queries each parent); report query and parent deltas, descriptive paired parent
CIs, t/s/v, >5pp query harms and B1 v/t>.5 retention. Define systematic branch
collapse before execution as mean delta t or s<=-1pp AND at least12/16 parents
negative for that branch. v<=0: next unique hypothesis trust/no-op gate; positive
v without collapse: eligible for later registered expansion, never automatic here.
Directional alignment is not native utility. Even a pass is exposed development,
not fresh confirmation, expert qualification, OPD or target success.

## Engineering and scope

Serial GPU;8GiB minimum free;max32GiB new storage. One cache allocation up to3600s,
cached fit up to900s, conditional native up to900s; each guard is engineering only,
all loading/failed/replayed/finalization seconds measured and cumulative cap=null.
Baseline ledger70591.88224354811 seconds. Aim30–60min including preparation, not a
guaranteed runtime. Partial failure retained and reviewed in a new version; no
silent replay. No full618/447, fresh388, experts, OPD, target, scalar mask or grids.
