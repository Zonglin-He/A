# PANEL16 native decomposition oracle — one analytic correction

Status: protocol before new GPU. Explicit user authorization: only PANEL16
native T/S gradient geometry and equal-budget joint/decomposed oracle. No scalar
mask, new module training, external TVG/SVG, OPD, target, or additional grid.

## Question and starting state

The previous two outcome-selected cases were mixed: event mask locally adverse,
spatial weakly favorable. They do not establish population-wide conflict.
Use all original sixteen Vid source-training parents/queries in their original
order, official frozen PTD4B and exact B1, same observed pixels and preprocessing.
GT is openly used for this source oracle; it is not an unlabeled TTA result.
No model parameter, optimizer state, query feature, gate, or decoder is updated.

The common variable F is the stock FP32 merger grid [1,T,H,W,2560], with frozen
caption features q and physical-time support. Each branch computes its original
F + R_b(G_b(H(F),q)). A leaf replaces F before the frozen adapter, including the
identity and reader paths. This is a conditional derivative holding q fixed,
not a pixel derivative nor a claim about an upstream q(F) derivative. Both
gradients must have the same F values/hash/order. This differs explicitly from
the earlier post-adapter additive actuation tests. Existing output spans are
used as channel subspaces, not claimed to have already proven this new interface.

## Native objectives and support

First generate baseline B1 reference/time and spatial blocks using independent
official cached prefills. Verify baseline pixels/query/time/grid and native
geometry/endpoint/coordinate logits against original sealed B1. L_T is mean CE
of the two actual endpoint distributions over the N observed time tokens.
L_S is mean CE over valid GT coordinates at BASELINE STUDENT-native anchors,
with full 152775 vocabulary. Reference/interval/anchors and probe cache support
are the student's, never GT teacher-forced. Unannotated anchors remain excluded
from CE and explicitly counted. All16 currently have nonempty valid support;
an unexpected absence is retained as missing, never a replacement source.

Replay each initial branch differentiably, exact logits equal the captured
native distribution, backpropagate only into the shared F leaf. Save complete
unclipped g_T/g_S, full native traces/logits, target support, and initial F.
PTD/B1 parameters remain requires_grad=False and grad=None. No clipping.

## One fixed correction, no state selection

Q_T/Q_S are orthonormal bases of the existing frozen event/spatial output
projection columns (same nonzero gates, absorbed as channel-span coordinates).
P_b g = g Q_b Q_b^T. Set delta_T along -P_T g_T and delta_S along -P_S g_S.
Their radii are .087687 ||F|| and .170316 ||F||, respectively: inherited locked
amplitudes from the earlier source-exposed controls, no new scale search or
independent-validation claim. A zero direction stays zero and is reported.

For Joint, use the union of both column spaces (SVD singular cutoff 1e-10
relative, no restriction to their potentially tiny intersection). Its direction
is -P_union(g_T/||g_T|| + g_S/||g_S||): equal branch importance in local geometry,
not raw unequal-gradient-scale CE summation. Normalize it to
||delta_J||^2 = ||delta_T||^2 + ||delta_S||^2. Report projected retained-gradient
fractions, both gradient norms, full and projected cosines, all self/cross dots.

Six final native conditions, fixed once: Base; T-only; S-only; Decomposed(T,S);
Joint(J,J); Joint-pass-matched(J/sqrt(2),J/sqrt(2)). The last control is required
because the user's parameter-budget equality makes Joint's actual two-pass
injection energy twice Decomposed's. Report both definitions, not one as equal
physical exposure. Joint's union span and branch normalization are explicit
design choices; this is one directional rule and one magnitude, not optimal
joint vs optimal decomposed adaptation. No best-step or loss-selected state.

All deltas replace F by F+delta at the same pre-adapter interface, retain query
features, leave pixels unchanged and keep BF16 PTD/native generation unchanged.
Report actual pre/post-BF16 injection norms/hashes and native reference/interval
changes. T-only and S-only are cross-harm controls. Native final spatial support
may change after event correction; additionally replay each delta against BOTH
fixed baseline branch schedules to measure finite self/cross CE on identical
support. Fixed-support coordinate argmax IoU is a distribution diagnostic,
not an extra native prediction. Native format failures and invalid boxes stay.

## Evidence, analysis, and decision

16 full gradient pairs; 96 final native predictions; 32 branch backward calls;
eight finite fixed-support forward reads per query (T,S,J,J_pass x T,S).
No optimizer. Seal all raw/support/predictions before offline native scoring.
Source GT has already been read for the oracle; do not claim a GT-blind run.
Independent NumPy norms/dots/projections/budgets and scalar versus tensor tube
metrics; parent bootstrap10000 seed20260927, unadjusted descriptive95% intervals.
Report every source: cosine histogram/quantiles/sign counts, self/cross local and
finite objective changes, t/s/v, >5pp parent loss, native-good retention, format
failure, all positive/negative/zero effects and all16 denominators.

Primary native contrasts Decomposed-Joint and Decomposed-Joint-pass-matched
on parent macro vIoU; t/s shown separately. Also vs Base and T-/S-only cross-harm.
Near-orthogonality is descriptive |cos|<=.1, not by itself conflict/necessity.
Negative cosine or cross dot indicates local conflict only, not finite harm.
Support a practical decomposition opportunity here only if Decomposed improves
Base and both Joint versions in macro v, with positive lower descriptive CI for
both Joint contrasts and no extra >5pp severe parent harm relative to Base.
Otherwise label the evidence mixed/inconclusive or adverse for this rule. Even
a pass does not establish generalization, reader learnability, universal need
for decomposition, or OPD success. Later directional qualification requires a
separate protocol; this run never auto-starts it.

## Engineering bounds and recovery

CPU scope/dispatch/restore/full-vocabulary mean/budget/sign tests first, then
write-once CONFIG/INPUTS/LOCK/registration. One serial GPU allocation <=3600s,
new output <=24GiB, free disk >=8GiB, cumulative GPU cap=null. Same deterministic
algorithms/CUBLAS setting as audited controls. All loading/failure/replay and
finalization counted by the allocation receipt; no silent rerun or pin edits.
One existing Luna max watcher checks current actually-started phase, read-only,
at30-minute cadence; exceptions to root. Failures get separate preserved evidence
and isolated recovery versions; no outcome-based sample removal.
