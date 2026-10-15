# E1: same-frame original-DINO reference swap

## Question and scope

This locked development study asks whether replacing the Weight-free spatial reference geometry with an original GroundingDINO expert box, while holding the observed physical frame and reference weight fixed, changes the current five-step spatial adaptation readout. It is a same-frame geometry intervention. It is not a reproduction of the full original DINO or online pipeline, and it does not promote a deployment method.

The roster contains 32 previously exposed sources: 16 HC-STVG-v2 and 16 VidSTG. It is not an untouched confirmation set. The 49 observation attempts are fixed before teacher execution. The original teacher admission and parsing logic are retained, with phrase threshold 0.35, distinct-candidate margin 0.05 and NMS IoU 0.5. The common-support reference keeps only original DINO-accepted observations; each accepted DINO box is paired with the Weight-free box at the same physical observation frame and receives the original Weight-free reliability weight. Rejected and unknown observations remain excluded. There is no interpolation, label repair or GT-based reference selection.

## Locked comparison

The teacher record distinguishes attempts from actual inference. Parser abstentions produce no model forward. The sealed teacher receipt and independent readback report the actual accepted, rejected, unknown and forward counts. Only the 38 accepted common anchors form the paired reference set; these cover 19 sources. The other 13 sources have empty common support and must take exact F/T fallback for the paired arms.

The eight scored arms are `F`, `T`, `S_new_crop`, `TS_new_crop`, `S_WF_common`, `TS_WF_common`, `S_DINO_common` and `TS_DINO_common`. The first four are exact parent reuse. The common-support arms use the locked domain configuration, identical source reset, stable query key, per-query Adam reset, and the same five-step, 32-sample fitter. The domain parameters are:

| Domain | Learning rate | Sigma | Tau | Steps | Writeback | Samples |
|---|---:|---:|---:|---:|---:|---:|
| HC-STVG-v2 | 0.003535533905932738 | 0.0125 | 0.2 | 5 | 0.0625 | 32 |
| VidSTG | 0.0075 | 0.2 | 1.0 | 5 | 0.125 | 32 |

The DINO and Weight-free arms differ only in the locked same-frame reference geometry. An empty reference set causes no fit and exact F/T reuse. There is no LN carry. The locked validation gate is unchanged.

## Readouts and analysis

The post-seal scorer retains 32 sources by eight arms and five separate metrics: mean vIoU, mean tIoU, dense-GT sIoU, supervised-GT sIoU and unsupervised-GT sIoU. Unknown rows remain null. It reports six prespecified paired contrasts per domain, with positive, negative, neutral and unknown counts. Neutral uses 1e-10 percentage points. No bootstrap is run.

A separate post-seal diagnostic reports box IoU for Source, Weight-free and DINO only at the 38 shared anchors with an actual GT annotation. This observed-anchor IoU is not full-tube vIoU, does not select or modify references, and is not an oracle fit.

## Execution accounting and numerical boundary

The teacher accounting separates 49 attempts from 40 actual model forwards; nine unknown/parser-abstain attempts did not run a forward. The Student final barrier retains 38 fit records and 164 optimizer steps. Execution-cost accounting includes 40 actual fit invocations, 40 reported full-model readout invocations, and 174 optimizer steps across the full execution chain. The 38 retained fits comprise 16 student005 prefix fits reused exactly in the final barrier and 22 new student007 fits. Two additional HC11 Weight-free fit attempts in student005 and student006 each completed five optimizer steps before pre-seal failures. The independent math reader verifies those saved partial optimizer chains, but does not independently verify a separately serialized full-model readout artifact for the partial attempts; the readout count is recorded by the independent execution-cost audit.

The worker launch receipts record seven attempts: three teacher and four student. Five failed before completion (two teacher, three student), for 121.561 seconds combined launch-to-exit; the complete execution chain totals 320.600 seconds. The sealed worker intervals are 28.988 seconds for the teacher and 164.852 seconds for the student. Those values use different boundaries from launch-to-exit totals, which include failed attempts and worker setup. Thirty-four source-capture calls were recorded; two failed HC11 source-capture sub-times were not separately serialized. Historical source/temporal/parent Student arms are exact reuses, so these stage times are not fresh end-to-end latency measurements.

The independent CPU optimizer-math reader checked 12,032 reward scalars, 562 loss scalars, 190 Adam parameter-group transitions, 164 accepted steps and two rejected proposals. It did not load a model, run GPU work, replay backward passes, or read GT. The original strict box-validity boundary differed at one float32 edge case (`1 + 2.98e-8`); the frozen worker uses the locked 1e-7 validity tolerance and preserves all 76 paired anchor tensors exactly without clipping.

Stage wall times are not fresh end-to-end method latency: historical source, temporal and parent Student arms are exact reuses. The report must distinguish retained final outputs, work reused from the completed student005 prefix, and work performed in failed attempts. It must also disclose that the independent optimizer reader did not verify serialized partial full-model readout artifacts.

## Claim limits and publication

The result is limited to this fixed, previously exposed 32-source development roster and the admitted same-frame common support. It does not establish behavior on unseen sources, a full-cohort effect, original-DINO pipeline performance, or a production-method change. Whole-tube metrics and observed-anchor IoU answer different questions and must remain separate.

Post-seal scoring is allowed only after all 32 prediction files and receipts are sealed and independently read back. The anonymous report exports fixed aliases, scalar metrics, aggregate counts, cost summaries and artifact-hash bindings. It excludes query text, raw teacher responses, source paths, frame coordinates, images, model weights and raw caches. Publication staging and remote verification are separate from report generation; this file does not itself authorize a commit or push.
