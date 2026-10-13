# Observation-response matched spatial update: fixed candidate cycle

Human authority: 2026-10-13 attachment 09999baf, `按照这个做`.
This is a candidate update-operator experiment, not a production promotion.

## Question and controls

Can the same sparse expert correction be implemented with less unnecessary
change on other decoder frames? Reuse the original P9/P0 previous-P6 32 parents
per domain, 64 queries, and **Inherited-order1** entering LN state. Query
Residual starts at zero; the seven original trainable tensors have 1792
coordinates. Source checkpoint, pixels, physical frame grid, Native WHEN,
Omega0 evidence, admission/Top1, original common antithetic noise and OPD
configuration remain fixed. HC lr/sigma/tau/steps are .01/.025/.05/40; Vid are
.03/.10/.25/10; samples=32. The writeback configuration is retained in custody;
this fixed-state panel does not write across queries. No fresh DINO is called.

A = original OPD (exact old complete fit reference, independently replayed at
qualification); P = minimum parameter change preserving the original proposal's
linear observed response; F = minimum functional change under that response
and original proposal norm; R = OPD plus fixed-reference output consistency;
N = common Before with no update. Formal P/F/R contain 64 fits each, 192 new
fits and 4800 scheduled rounds. A and N are references, not new formal fits.

At current actual state, original OPD gradient updates Adam moments once to
produce d_A. P/F commit a projection or that same d_A fallback. Subsequent
gradients are evaluated at each arm's actual committed state. R updates Adam
once with the sum of the original OPD gradient and consistency gradient.
No-information branches keep the original no-op; no protection-only step.

## Operator

J_O is the Jacobian of the original finite-native logit chart at admitted
observations. J_U is the Jacobian of normalized cxcywh boxes on the complete
decoder grid excluding **all physically visited Omega0 positions**, including
rejected expert observations. GT is absent from both operators. The reference
B_U^0 is the current query's entering output and remains fixed throughout it.
Uniform W gives D_U = ||B_U-B_U^0||^2/(2 |U_run|).

p=J_O^+ J_O d_A is P. F solves minimum ||e_U+J_U d||^2/|U_run| subject to
J_O d=J_O d_A and ||d||<=||d_A||. A thin observed-row SVD and the Gram matrix
of J_U projected into ker(J_O) implement the equivalent trust-region least
squares problem without a full null-space basis. No random subspace or frame
subsample is used. Solve arithmetic is float64 CPU; the actual model and
decoder forward/backward keep their original precision. SVD rank cutoff is
max(1e-12,1e-10*s_max); the nonnegative Lagrange multiplier is bisected for 80
iterations. Record the full matrices/proposals/KKT checks privately.

R lambda is computed once at query entry as
tr(J_O^T J_O/(|O| sigma^2))/tr(J_U^T J_U/|U_run|).
An empty U or zero denominator records a degenerate zero lambda. It is not
tuned by GT or presented as the optimal regularizer.

## Numerical qualification before GT

Each domain has four fixed queries: smallest three indices with nonempty
admitted evidence plus the smallest empty index when an empty exists; otherwise
the smallest four indices. Selection uses IDs and admission metadata only.
Two complete fresh captures/replays per query verify original complete OPD
bitwise, source/state/input restoration and no-op. At fixed original-A
boundaries k=0,floor(K/2),K-1, test one-step P/F/R proposals using that boundary's
exact state and pre-update Adam moments. These are local qualification probes,
not new complete candidate trajectories. R uses the query-entry curvature
scale and is explicitly a local proposal at A's saved state for qualification.

Complete output Jacobians use bounded batched reverse-mode VJPs, with scalar
VJP and double-backward JVP spot checks plus centered actual-decoder finite
differences. Directional checks use deterministic GT-free directions and
h=0.002 and 0.001. Acceptance uses 2% relative plus 1e-4 absolute FD tolerance;
VJP/JVP contractions use 2e-5 relative plus 2e-6 absolute tolerance. Both h
results are retained; no GT-based tolerance selection. Technical absolute
observed-output tolerance is 32*float32 epsilon*sqrt(4|O|); relative observed
response tolerance is 10%. Parameter norm tolerance is 4e-6 to account for
float32 state addition; linear response tolerance is 2e-5 relative plus 2e-6
absolute. F must pass actual fixed-reference D_U non-increase with tolerance
1e-10+64*epsilon32*D_A, and improvement must exceed that numerical tolerance.
P must pass observed response and norm, not F's functional objective.

Failure of solver, response, norm, or actual F improvement commits original
d_A; it never cancels learning. Record each failed check and fallback. Repeated
counterfactuals/replays must agree exactly; measured costs include actual
Jacobian/solver/readout computation. No promise of lightweight operation.

Stage A ends with an explicit resource/technical decision. At least half of
nonempty informative F probes **in each domain** must pass actual checks to
enter the 192-fit panel; equality/FD/original replay contracts must all pass.
This is a viability threshold for this fixed recipe, not an effect threshold.
If it fails, retain and publish all qualification results and stop this recipe;
do not change frames, precision, learning rates, positions, or tolerance based
on GT to obtain a pass. Costs may independently warrant stopping investment,
with mechanism status left unresolved. Candidate episodes have a finite
qualification budget of 8 queries x 2 captures x 3 fixed local boundaries,
alongside 16 exact original complete replays. No unrestricted search.

## Formal evaluation and decision

Only after all 192 complete candidate fits and exact 64 A/N bindings seal may
GT be read. U_eval is U_run intersect genuine GT event annotated decoder frames.
The common Before defines correct (IoU>=.5), incorrect (<.5), and correct plus
Source-advantaged (>=1pp) strata offline, shared across all methods. Report
expert agreement; response errors; original/candidate proposal norms; actual
functional drift; fallback/no-information counts; all parent gains, negative
tails >5/20pp, severe/persistent frame harm, error-to-correct transitions,
complete official dense mvIoU/sIoU; true per-fit costs. Empty/undefined results
remain unknown, not zero. Parent is the unit of 10000 paired bootstrap draws
with seed 20261013; conditional strata disclose their defined denominators.

F supports this candidate only if comparable observed learning accompanies
HC correct-frame retention and preserved Vid incorrect-frame correction. Vid
mvIoU noninferiority margin is .5pp, fixed before formal GT. Compare F-A, F-P,
F-R and F-N; a CI crossing zero is inconclusive, not equivalence. A function
distance improvement alone is insufficient. If P or R explains the gain,
prefer the simpler operator rather than claim functional novelty. If F is
effectively N, or harms Vid, stop this recipe without dataset winner switching.

Conditional stage C is state/source replication and an independently generated
online stream **only after** candidate support. Its concrete immutable cohort
and stream protocol must be recorded before execution; no automatic full main,
CURRENT registration or manuscript. Do not stitch old A Before into new online
results. Existing P7-A and all old FINAL scopes remain frozen. Cancelled
remaining baseline formal stays same-identity T standby; no baseline signals,
unlock, restart or cancelled GT scoring.

## Closure

Independent mathematical/source/state/input/dense/parent-statistical root,
all anonymous positive/negative/neutral/unknown results, actual PDF page views,
finite public export with exact remote byte/SHA256/Gitblob/tree/ref checks,
archive check-snapshot-check and true scoped FINAL are required. A negative
Stage A closure is a completed bounded qualification, not a completed 192-fit
experiment or a universal rejection of projection/functional protection.

Primary context: [EATA](https://proceedings.mlr.press/v162/niu22a.html) protects
important weights; [NTK-overlap analysis](https://proceedings.mlr.press/v130/doan21a.html)
already studies projected-gradient interference. No claim of inventing
functional regularization, projection, or Jacobian overlap.
