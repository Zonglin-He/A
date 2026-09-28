## Current decision: state-aware direction distillation is the next source candidate

本轮447query/31已曝光源父源诊断完成，1788预测、873backward、0optimizer，全部封存、full raw和两级独立评分核验通过。**DECISION=A：优先state-aware方向蒸馏；新训练尚未启动。**

| 父源宏 | tIoU % | sIoU % | vIoU % | Δv vs B1，pp |
|---|---:|---:|---:|---:|
| B1 | 46.637512 | 48.627481 | 32.407600 | — |
| Learned seed1 | 46.887168 | 47.764785 | 32.167599 | −0.240001 |
| Learned seed2 | 46.741185 | 48.246816 | 32.295832 | −0.111768 |
| Analytic Joint oracle | 53.402853 | 60.069317 | 44.679649 | +12.272049 |

Oracle Δv的配对父源95%CI为[+8.979625,+15.496381]pp，29正2负父源；对两seed分别+12.512050/+12.383817pp，CI下界均>9pp。两个learned/oracle方向cos均值仅.001459/.001767，中位.000697/.000833；444定义、3缺支持不定义。Oracle gain/learned harm为128/117query。虽均贴norm cap，但方向接近正交，未满足“方向对、幅度错”的B规则。

**正均值不等于完整teacher资格。** Oracle仍有2个父源v损害>5pp、66个query v损害>5pp，native-good v保持122/136、t保持164/199，未通过尾部和原好保持门。sourceGT诊断、非held-out/target结果；局部对齐不能证明native-state缺失是因果，A仍需单独真实接口/源训练验收。A/B/C只有CPU模块与协议，新31父源388query确认保持metadata-only；无自动expert/OPD/target/网格。

[完整匿名结果与限制](../../results/desta3d_v3/2026-09-29/ORACLE_MIXER_GAP.md)；同目录JSON保留全部31父源和9类结果组合。以下运行中记录保留为历史，不代表仍有GPU任务。

---

## Current execution: frozen gap diagnosis; conditional modules prepared

原两seed训练与447query确认已完成，负结果保留。当前仅进行同447/31已曝光源集合的冻结解析oracle诊断，首例真实接口和独立raw核验通过，尚无全量新效用结论。A方向蒸馏、B trust/no-op gate、C rescue骨架及31父源/388query新metadata确认名单已事先锁定，CPU检查通过，未启动其GPU训练或救援实验。

[当前协议、实现入口和限制](ORACLE_MIXER_GAP.md)。下面旧状态记录保留为历史。

---

# Decomposed Evidence, Joint Correction

Current research candidate after the PANEL16 Decomposition Oracle. The former
claim that temporal/spatial correction must be independently applied is
withdrawn as the principal-method hypothesis. Historical implementations,
protocols, predictions, and positive/negative outcomes remain unchanged.

**Preserve the shared THW representation, distinguish task evidence, coordinate
correction.** This is a proposal to test, not a demonstrated general principle
or a completed TVG/SVG/OPD system. The joint mixer is implemented, CPU checked and real-native interface validated. Both locked seeds completed618queries/155Adam calls and passed their CPU state/coverage audits. All1,341 confirmation predictions are sealed and independently scored: vIoU vs B1 -0.240001pp/-0.111768pp, both CIs crossing zero, spatial means negative and native-good retention incomplete. This configuration did not pass teacher qualification. See [full confirmation and boundaries](../../results/desta3d_v3/2026-09-28/JOINT_LEARNABILITY_CONFIRMATION.md) and [both-seed training record](../../results/desta3d_v3/2026-09-28/JOINT_LEARNABILITY_BOTH_TRAINING.md).

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
