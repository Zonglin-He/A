# R13: separate language identity and partial-positive geometry

Authorized 2026-10-31. Prior rounds and their exports remain immutable. Run state is separate from measurement validity and scientific claim. All final A/B outputs must be sealed before any new target scoring. Source is the verified Waymo-only epoch100 checkpoint, native model yaw0, fixed strict-FP32 grounder. No source retraining, EMA or teacher-student training.

## Inputs and exposure

Input plan `861976906f0dfa7a349f2fb539cbb7cb502be6ba542126156c8c741e5b90f501` excludes the union of prior700 query paths and all R12 frontend501 contents:914 exact JPEG/point groups. New108 = Drone32 from one remaining scene, Quad32/12/32 from three scenes. No remaining new content in the second Drone scene. All scenes and the dataset have historical research exposure; these are new input contents, not independent fresh domains or scenes. R12's ALL8 actually changed cases form a separate exposed regression set; never pool their outcomes into the new108 efficacy result.

New108 source and mask forward are actual new runs with the unchanged R12 capture engines; their seals retain those engines' R12 state names. Old8 source/masks are reused bitexact with their original seals, not represented as new forward. Original grounder input already contained the full literal caption; its existing firstnoun readout and query pool stay unchanged. The external frozen VLM adds a new association readout, not a newly trained 3D grounder.

## A: association only

Frozen local Qwen2.5-VL receives the full RGB image and a second copy with numbered independent whole-image instance boxes. No source query, source score, detector class/confidence or GT is exposed in the prompt. Candidate ordering is hash-based and reversed in the second presentation. Full literal description, noun-only and fixed same-scene distinct description each receive the same two calls and image/presentation budget. Distinct description does not guarantee contradictory language or a different target. Captions are quoted data, not execution instructions.

Answers are strict candidate ID or UNKNOWN, at most8 new tokens, deterministic greedy decoding; no repair or adaptive retries. Consensus requires the same physical mask under both numberings. UNKNOWN, parse failure, disagreement and 2D-to-3D ambiguity retain source q0 with an explicit status. Preserve the full-description first-order raw readout as a separate no-consensus ablation.

All256 original source proposals are retained. Geometry-only stable greedy AABB IoU>=.5 groups duplicates, retaining the highest original text-score member as the original-box representative. Mapping requires projection IoU>=.1, >=3 nearest-camera-depth points in the same mask and 3D box, and >=.05 top-vs-runnerup gap when multiple eligible groups exist. No averaging/fitting. A outputs exactly the chosen original source box, or original q0 on failure. Mapping is a hypothesis, not literal identity certification.

## B: fixed-q0 geometry only

B uses the fixed source q0, original independent masks and original camera/point inputs. Source-compatible unique mask and source-padded ROI follow R12, but preserve physical raw IDs through float32 model-coordinate deduplication. This class/ROI gate is not full-description identity proof. No A output enters B.

New G uses only point-in-box and foreground-in-projected-box positive inequalities, plus source-centered minimum-change priors. No point mean/extremum or visible plane is equated to a full object boundary. Source already satisfying the inequalities stays exactly unchanged. Centers have fixed bounded shifts, sizes may retain or expand to1.2x source; partial positive observations provide no justification for shrinkage or complete size upper bounds. All6 active coordinates are an optimization domain, not6 independently observed physical quantities. No free-space term: actual firing origin and solid AABB occupancy are not established.

Main full/noRGB coordinate arms share support, gates, optimizer domain and bounds; noRGB removes only the RGB coordinate term. The half control is exact float32 source+0.5(full-source), no refit. Old R12 surface-equality G is a retained legacy control on the same source/mask/raw inputs. Its plane eligibility/search domain also differ, so it is not a one-term causal ablation.

## Final arms and evaluation

Nine arms: source, A_full_consensus, A_full_order0, A_noun_consensus, A_deranged_consensus, B_inequality_full, B_inequality_noRGB, B_inequality_half, B_old_surface_equality. No neural parameter updates in any direct arm. Preserve all unknowns, rejected/missing/ambiguous cases, actual zero changes, finite nonconvergence and negatives in each platform denominator.

Primary readout: nativeyaw0 3D IoU, strict Acc@25/50, center/size and coordinate errors; both sample micro and equal-scene summaries, with signed gain/harm/tie and tails. One Drone scene cannot give a scene-cluster uncertainty estimate. All256 candidate oracle is a postseal diagnostic bound, never a selection method or semantic instance truth. Old8 regression is descriptive and output-conditioned. No post-label changes to thresholds, prompts, support or weights.

## Prerequisite and resource decision

A has useful grounding evidence only if full consensus changes original boxes, improves unconditional task IoU over source and the matched noun-only readout on both platforms, with deranged-language effects reported rather than treated as guaranteed wrong labels. Winning only a deliberately disturbed control is insufficient.

B has useful regression supervision only if direct full changes boxes and improves both unconditional task IoU and relevant center/size errors over source on both platforms, with positive net task gains, and beats the fixed-half control. A claimed RGB coordinate contribution additionally needs the matched noRGB comparison. Evaluate micro and equal-scene consistently; do not choose different arms per platform. This is a finite exploratory resource gate, not a population-significance theorem.

If either identity or geometry prerequisite remains unsupported, do not compose A+B or run regression-head TTA on this G. If both pass, lock a separate finite transfer pilot before updates: supervision from one observation, readout on another observation not used to construct/gate/fit the pseudo-target, and the matched same-information direct-transfer/no-update control. Exact fitting of a pseudo-box on its own input is not additional TTA value. No new pilot is preemptively queued.

Budget: new108 source/mask captures; 116x3x2 frozen-VLM calls (two input-only engineering preview cases may be repeated explicitly); at most116 bounded CPU G solves per fixed arm. No threshold/model/prompt sweep. Root owns the shared GPU lease; other agents perform only bounded CPU implementation/audits.
