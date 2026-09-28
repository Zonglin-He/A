# PANEL16 Joint component attribution — CPU saved-evidence audit

Status at registration: proposed / untested. This is the single audit authorized
by attachment 34496422, following decomposition001. No new model or GPU run.

## Claim, evidence, and scope

Withdraw independent T/S correction as the current principal-method hypothesis:
the locked oracle's Decomposed minus Joint vIoU is -4.401181pp, and minus actual
two-pass-energy-matched Joint is -5.941072pp. Both descriptive CIs are negative.
All positive and negative original cases remain. This does not establish a
universal superiority of Joint or a causal need for coordinated parameter updates.

The candidate is now **Decomposed Evidence, Joint Correction**, with joint OPD
only proposed. The next question is whether Joint's saved direction contains
locally useful components outside an objective's own branch span. Finite native
benefit of individual components has NOT been measured. No mixer, expert,
optimizer, radius/step/lambda/projection sweep, new forward, or target is allowed.

Use all original 16 exposed Vid training parents, frozen PTD4B/B1, same fixed-q
pre-adapter F, saved gT/gS/J/J_pass from decomposition001. Read only sealed
GEOMETRY.pt, input identity/completion records, basis and audited score report.
No videos, raw prediction tensors, label pools, or new GT reads are needed.
Source labels were already used to create these native-objective gradients;
this is a source-supervised diagnostic, not unlabeled TTA or generalization.

## Nonorthogonal spans and attribution definitions

Reorthogonalize saved T/S bases in FP64 without changing their column spaces.
Let PT, PS be the branch projectors and PU the union projector. Report principal
angles, singular values, numerical intersection dimension, union rank, and
conditioning. Relative SVD rank tolerance is 1e-10. No outcome-defined rank.

Two complete orthogonal decompositions are mandatory:

* T-first: J = PT J + (PU-PT) J + (I-PU) J.
* S-first: J = PS J + (PU-PS) J + (I-PU) J.

The second term is the residualized added span, NOT a pure member of the other
original span. Both orders are retained. PT J + PS J generally does not equal
J; report that reconstruction error instead of double-counting overlap.

If numerical intersection is zero, additionally solve the direct sum
J_U = QT cT + QS cS. These two components belong to the original spans but are
NOT orthogonal: retain their signed energy cross term. If intersection is
nonzero, the minimum-norm coefficient convention is disclosed and no unique
branch-ownership claim is made. This is algebraic attribution of the saved J,
not a new correction or a choice of projection for a model run.

For each component and both objectives report energy and -g_b dot component.
Positive means local predicted CE descent only. Outside-union residuals,
reconstruction/Pythagorean errors, both attribution orders, and every signed
parent are retained. Use sign tolerance max(1e-10, 1e-10*sum(abs(contributions)))
for numerical-zero reporting; no scientific effect threshold or success gate.
Report each objective's gradient energy in its own span and the extra union
span, relative both to full gradient and to union-projected gradient.

Also attribute the saved balanced Joint to its two gradient inputs:
-k PU(gT/||gT||) and -k PU(gS/||gS||), with k recovered from the saved J's least
squares scale, and retain the FP32 rounding residual. This distinction prevents
calling a component locally helpful independent evidence: Joint explicitly
contains each objective's supervised gradient by construction.

J_pass uses the stored tensor, not an assumed exact rescale. Verify its saved
1/sqrt(2) relation and retain any rounding difference; component signs should
not be presented as an additional independent experiment.

## Validation and resource limits

Before numerical attribution: CPU controls for nonorthogonal spans, exact
intersection, zero direction, Pythagorean decomposition, cross-task sign and
rotation invariance. Lock code/protocol and consumed input hashes. Verify every
consumed old file against its original seal, and basis against ROOT_BASIS_SEAL.
The existing score report must match its COMPLETE report hash. Query/parent
order and common-F identities must match all16, no replacement or exclusion.

Independent NumPy FP64 attribution and Torch FP64 full-feature projection
readback, atol1e-8 + rtol2e-7 for scalar reductions; reconstruction relative
error <=2e-6. These are engineering tolerances, not claim thresholds.
Complete cases include stored native t/s/v differences and fixed-support CE
changes from the already audited report, clearly labelled inherited outcomes.
No additional source scoring. No significance search or outcome-based subset.

CPU allocation: 900 seconds per primary/readback phase, four BLAS threads,
new outputs <=16MiB, free disk >=8GiB. Record CPU time as CPU_RECEIPT.json;
GPU increment 0, prior cumulative GPU43107.875212573104 seconds, cap=null.
Do not create GPU RECEIPT.json or silently add CPU seconds to the GPU ledger.
Failures/partials remain and a repair uses an isolated version when needed.

## Decision branches

If cross components are locally useful under both ordered and original-span
conventions, support local complementarity only; finite component intervention
and privilege absorption remain untested. If signs depend on attribution order,
retain the ambiguity and do not claim branch ownership. If cross components are
weak/adverse, lower the complementary-span explanation. Existing Joint native
advantage still does not isolate span, gradient mixing, amplitudes, and shared
application, which differ together in the previous comparison.

Stop after the audit, archive check/snapshot/check and public anonymous results.
No automatic joint mixer, TVG/SVG, OPD, new GPU experiment or monitor.
