# DESTA-3D v2: limited 8-parent update-interface and alignment comparison

This is the user-authorized next step after source B1 selection and full shared-reference confirmation. It is a development screen, not a new independent confirmation set. Source B1's parent mean barely exceeds matched Frozen (+0.00880 pp, CI crosses zero); the shared-reference contract changes no boxes at that checkpoint. Auxiliary-backflow controls did not establish that dominance was the principal task bottleneck. The next unresolved question is whether small test-time calibration protects task outputs better than convolution updates under the same unlabeled view objective, and whether source feature alignment helps either interface.

## Locked inputs and controls

Select lexical first four query keys in each of the existing historically exposed HC32 and Vid32 metadata cohorts. Eight parents, one 32-frame query each. No score, label, failure case, or gradient-based selection. Source fitting used Vid only; HC remains cross-dataset transfer. Run clean, noise_medium and defocus_extreme, preserving exact established corruption, frame and preprocessing conventions.

Use six arms per episode: actual official Frozen inference, B1 sourcefit without TTA, convolution/view-only, FiLM-LN/view-only, convolution/plus-alignment, FiLM-LN/plus-alignment. The four TTA arms are a matched 2x2 diagnostic, not a LR grid. All start from the same fixed B1 weights with new AdamW, have exactly three steps (LR 1e-5, wd 0, clip 1), and are reset after each episode. No loss-based acceptance or best-step selection. Three steps permit measuring the initially zero parameter anchor after updates; they do not establish sufficient convergence.

The convolution interface updates input projection, shared stem and both branch readers. Calibration updates only branch FiLM and channel LN affine (66,816 parameters). Gates, output projections, heads, text pools and the frozen PTD stay fixed in both. All losses are before the residual gates, so the gates must remain frozen.

Teacher evidence comes from fixed B1 on the same observed corrupted input. The student gets brightness 1.05 / contrast .95 applied to that same observation, without clean reconstruction. Physical time/grid/caption are matched; caption hidden tokens may depend on the view. Latent normalized consistency, referent/event Bernoulli KL and mean parameter anchor each have coefficient 1. Alignment is 0 or .01; joint is 0. Source alignment uses newly recomputed query-conditioned B1 reader moments over all 618 train queries, equal query within parent and equal parent. No v1 statistics are reused.

## Actual interfaces and outputs

Decode each final state with event-only inference followed by an independent spatial KV with the event reference/time and official staged cache schedule. Save full predictions, real gradients by parameter group, actual Adam steps, update scope, frozen assertions, exact resets and input identities. Save post-cast residual magnitudes, free event/box outputs and a read-only fixed-sourcefit-prefix coordinate-logit KL diagnostic. This output diagnostic is not a differentiable training objective. Normalized feature loss can miss magnitude drift; the parameter anchor is initially gradient-free. Neither lower loss nor altered logits establishes useful TTA.

CPU tests cover actual hidden128 three-step updates for each scope, finite gradients, frozen gates, real Adam counters, nonzero post-initial anchor and exact reset, plus malformed prediction retention. Existing real shared-reference zero-gate tests remain the accepted decoder interface; no claim of rerunning them here.

## Seal, score and decision

All 24 episodes and 144 predictions must seal before target GT is read for offline scoring. Audit all file/input/adapter identities and retain failures. The same physical Frozen predictions are generated anew. Primary aggregation is parent-paired TTA minus sourcefit vIoU, averaging noise and blur within each parent; clean is a separate harm audit. Also report TTA-Frozen, domain breakdown, spatial/time metrics, format failures, native-good retention, and >5 pp negative tails with paired parent bootstrap intervals. Use the original P3 sampled-support convention: spatial format failure gives zero spatial contribution but does not automatically zero a legal time interval. Do not interpolate missing boxes or change support.

Eight exposed parents only assess this bounded configuration. Positive means with uncertainty crossing zero do not establish reliable improvement. If updates do not reach outputs, inspect injection/quantization and distributions; if outputs change harmfully, inspect the signal/interface before expanding. No automatic 64-source launch or production promotion.

## Engineering and preservation

Serial GPU lease, 8 GiB disk reserve, 3600-second phase and cap=null cumulatively. All actual loading, failure, replay and wrapper overhead enters the shared v1/v2 receipt accounting. Any failure keeps its partials and receipt; no automatic rerun or silent replacement. The original source arms, raw metrics, optimizer incident and prior negative results remain intact. No label access in registration, update or prediction code; old queues and CURRENT remain unchanged.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
