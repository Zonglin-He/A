# Short event-local tracklets: fixed-optimizer spatial diagnosis

This is the second finite controlled diagnostic, on the original eight historically exposed sources (four per direction). It is not the independent 32-source-per-direction confirmation. Temporal and Spatial remain the intended dual branch. CURRENT and stopped historical queues are unchanged.

## Fixed construction and scope

Four centers per source are chosen inside the sealed QA event on the original student frame grid. Each center receives its actual physical -2 to +2 frame window (20 selected frames per source, 160 total). Each window is initialized separately. The existing Shi-Tomasi / Lucas-Kanade / RANSAC helper and parameters are unchanged. Non-Source seeds copy existing proposal geometry to each center; this is not a newly discovered whole-scene detector and does not guarantee that the object is inside that box at the new time.

152 short-window proposals were constructed. All missing boxes remain missing. 427 valid frame boxes measure coverage, not correctness. Frozen TA-STVG features were newly captured for the same 160 pixels/physical frame IDs; ROI descriptors use area overlap, per-frame L2 normalization, and equal averaging. Adjacent-window association is mutual nearest cosine over all valid candidates, with deterministic ID tie breaking. Appearance association is not proven instance identity. Independent ROI/cosine readback found 99 mutual-nearest candidate edges across 24 boundaries; 15 selected mutual pairs were all Source-to-Source, with zero supported selected non-Source positions.

R1 saw four unmarked event-context frames and five marked local frames per call. There were 32 base calls and 32 ID-rotation controls. The selected reference uses base calls only; control answers do not retroactively select a better result. Timestamps bind physical frames; encoder pacing uses mean gaps and is not an exact irregular video clock. Candidate quality is measured later at common annotated centers, not inferred from tracklet count.

Two prelocked rules are evaluated: local_R1 retains valid selected non-Source center boxes; associated_R1 retains only local_R1 boxes whose selected candidate has selected mutual-nearest support in an adjacent window. Source selection and empty references produce exact no-update F/T. Four local references (one anchor each) triggered four real Gaussian IoU OPD fits; the associated rule retained zero references. The original five declared rounds, 1792 Query/LN coordinates, configuration and stochastic key are retained. Zero-information rounds do not force an Adam update. Final full-model redecodes are required. Original-eight T is exactly reused, not retrained.

## Full-Tube results

S deltas below are versus Frozen. T+S deltas are versus the exact Temporal-only control. T itself is shown at its absolute score; its zero self-delta is not a claim of no temporal benefit. Positive/negative/unchanged count sources, not frames. Scores are postseal official metrics independently recomputed from saved full tubes.

| Domain | Arm | vIoU % | Paired delta pp | Positive / negative / unchanged |
|---|---|---:|---:|---:|
| hc2 | F | 22.210455 | +0.000000 | 0 / 0 / 4 |
| hc2 | T | 31.286031 | +0.000000 | 0 / 0 / 4 |
| hc2 | local_R1 | 22.478144 | +0.267689 | 1 / 0 / 3 |
| hc2 | associated_R1 | 22.210455 | +0.000000 | 0 / 0 / 4 |
| hc2 | T_local_R1 | 31.453047 | +0.167017 | 1 / 0 / 3 |
| hc2 | T_associated_R1 | 31.286031 | +0.000000 | 0 / 0 / 4 |
| vidstg | F | 14.490136 | +0.000000 | 0 / 0 / 4 |
| vidstg | T | 22.441929 | +0.000000 | 0 / 0 / 4 |
| vidstg | local_R1 | 12.401812 | -2.088324 | 1 / 2 / 1 |
| vidstg | associated_R1 | 14.490136 | +0.000000 | 0 / 0 / 4 |
| vidstg | T_local_R1 | 19.341029 | -3.100900 | 1 / 2 / 1 |
| vidstg | T_associated_R1 | 22.441929 | +0.000000 | 0 / 0 / 4 |

## Separate sparse geometry and teacher diagnostics

23 of 32 windows have GT-known common support for Source and every displayed candidate; 9 remain unknown. Pool Oracle: {'positive': 9, 'negative': 0, 'neutral': 14}. Teacher: {'positive': 1, 'negative': 1, 'neutral': 17}, with 13 unknown (including abstention). ID consistency is 21/26 among pairs with two non-abstaining selections; abstain/abstain is not counted as successful identity understanding. All 64 responses parsed legally, with zero truncation; legality is not correctness.

These are sparse center IoUs. They cannot be subtracted from full-Tube vIoU or treated as complete-trajectory quality. Oracle is an offline upper bound and never enters candidate generation or online selection. Source in the pool guarantees a nonnegative maximum and alone does not prove useful improvement. Every candidate/window positive, negative, neutral and unknown is exported.

The short-window construction changes observation support and candidate presentation together. Its contrast with the old long-span experiment is a package-level diagnostic, not isolated proof that one tracking change caused any benefit. The local/associated rules share the same teacher outputs and optimizer. The associated rule's exact no-update result is an abstention/coverage result, not a learned improvement.

## Cost and audit boundary

64 real teacher calls: 105152 input tokens, 81920 visual tokens, 384 output tokens; summed request latency 29.690242 seconds and generation 27.451805 seconds. Shared model loading is excluded from these request timers. Actual stage launch-to-exit wall seconds (including loading and IO) are {'FEATURE': 44.99572253227234, 'TEACHER': 61.192267417907715, 'STUDENT': 60.36665630340576}; these overlap internal request timers and must not be added to them. Eight frozen feature captures and eight student source captures are separate operations. These finite diagnostic costs are not production end-to-end TTA cost.

Saved source hashes, exact F/T reuse, all candidate/feature/teacher bindings, random samples, reward, likelihood, distillation weights, stored gradients and Adam updates are independently read back on CPU. This does not claim an independently repeated decoder backward. Final decoder output equality is checked by the actual worker. All eight private input contact sheets were reviewed at pixel level; raw media, query text, GT coordinates, weights and parameter caches are excluded from the public export.

Preparation/environment/index-binding failures are retained with recovery receipts. They occurred before student execution and do not alter scientific candidates or completed teacher answers. Resource protection is scoped to the current own unit. No observed OOM is not a general guarantee.

## Decision boundary

This finite screen does not freeze a robust Spatial method. No GT-selected amplitude, candidate radius, learning rate, prompt or confidence threshold is introduced. Independent >=32 sources per direction remain conditional on a fixed GT-free rule and verified exclusion from all earlier development. Such confirmation must retain F/T/S/T+S; Temporal-only is a control, not a replacement of the dual-branch objective.
