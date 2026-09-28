# Decomposed Evidence, Joint Correction

Current research candidate after the PANEL16 Decomposition Oracle. The former
claim that temporal/spatial correction must be independently applied is
withdrawn as the principal-method hypothesis. Historical implementations,
protocols, predictions, and positive/negative outcomes remain unchanged.

**Preserve the shared THW representation, distinguish task evidence, coordinate
correction.** This is a proposal to test, not a demonstrated general principle
or a completed TVG/SVG/OPD system. The joint mixer is implemented, CPU checked and real-native interface validated. Fixed source training is running; learned native utility remains untested.

## What the completed oracle establishes

At one frozen PTD4B/B1 state, with fixed query features and the same pre-adapter
THW F, T/S native objectives have heterogeneous local gradients:7/16 negative
cosines,11/16 absolute cosines <=0.1. These are objective gradients; they do not
by themselves establish the optimal sources or independence of visual evidence.

One-step Decomposed vIoU42.867907% is below Joint47.269088% and actual-two-pass-
energy-matched Joint48.808980%. Differences are -4.401181pp and -5.941072pp, with
descriptive unadjusted paired-parent CIs wholly below zero. All16 are repeatedly
developed source-training parents, and severe losses remain in every correction.
Joint-versus-Base CIs still cross zero. This is not a deployment promotion.

The comparison changes span access, direction mixing, and pass sharing together.
T/S each use their128-dimensional branch output span; Joint uses their256-
dimensional union and normalized objective gradients. It does not isolate a
causal benefit of temporal-spatial coordination, an optimal parameter scope, or
the best possible separate correction. Parameter adaptation was not performed.

## Architecture and conditional downstream supervision

Keep frozen PTD visual features F, a shared THW representation Z=H(F,q), and
task-aware evidence acquisition eT and eS. The implemented correction operator forms a256-dimensional coefficient field
from shared features and heterogeneous evidence, then projects it through the
frozen union into one shared delta F. It enters F before both frozen B1 passes.
The exact insertion, norm bound and103424-parameter scope are now fixed in the
learnability protocol. This is a learned operator; the earlier oracle directly
optimized or analytically constructed fields, and did not establish learnability.

TVG and SVG/RVOS are candidate evidence providers, not interchangeable-logit
teachers. Same-PTD privileged policies must first improve actual native utility
using correct evidence and beat matched wrong evidence. A source-GT gradient
oracle is not an available unlabeled test-time evidence provider.

If qualified, a future native-state objective may retain separate endpoint and
coordinate KL terms while allowing both to update shared correction parameters
thetaJ: L=lambdaT*LT+lambdaS*LS. This supersedes mandatory cross-branch gradient
isolation as the current hypothesis. It does not mean simply adding two losses
already solves stability or learnability. The teacher is initially fixed at the
episode-start snapshot; both policies must use the student's current reference,
interval, anchors, parallel block positions, and identical action support.

Temporal-to-spatial conditioning remains part of inference. This is a sum of
native-state conditional losses, not a proven exact KL of the full tube joint
distribution. Measure final ordinary-input utility after removing privilege;
include evidence-provider cost during adaptation. No such OPD has run.

## Historical action: completed component attribution

Read [the audit protocol](../../protocols/desta3d_v3_joint_component_attribution_v1.md).
Use every saved PANEL16 gradient and Joint residual, with both orders of
orthogonal subspace attribution and explicit nonorthogonal direct-sum ownership.
Keep local predicted CE effects separate from the already measured whole-Joint
native outcomes. Individual component finite/native interventions are missing.
Even a positive cross-component dot cannot identify why the final tube improved.

The audit does not change a correction, a radius, a projection, a checkpoint,
or an evaluation rule. No scalar-mask continuation, full-source restart, new
expert, mixer, OPD, target, or hyperparameter grid is part of this action.

## Current authorized construction

The user moved finite-component intervention to later ablation. Implemented
`vg_tta/desta3d_v3_joint_mixer.py` trains only a zero-initialized128-channel
THW mixer into the frozen256-dimensional union. Decomposed source-GT evidence
enters as inputs; one correction enters both frozen B1 passes. Native endpoint
and full-vocabulary coordinate losses share mixer parameters. See
[locked learnability protocol](../../protocols/desta3d_v3_joint_learnability_v1.md).
Train618 queries/95 parents, independently lock447 confirmation queries/31
parents outside enumerated development manifests. Annotation preparation was
previously exposed; no globally untouched claim. Two seeds, one complete epoch
each, fixed final states. Real expert/OPD stages remain conditional.
