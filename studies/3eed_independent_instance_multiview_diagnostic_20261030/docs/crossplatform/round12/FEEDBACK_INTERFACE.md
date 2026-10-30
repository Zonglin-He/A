# R12 eight-arm fixed-q0 direct geometry contract

CPU_READY preparation only. No real G/pair/model/GPU/targetGT execution by this author. Root releases actual input/source/mask stages and scoring separately.

## Pure function and input information

`multiview_feedback.build_geometry_feedback(*, anchor, neighbors, wrong_neighbor, source_box6, q0, caption, seed=0, config=FeedbackConfig())` returns8direct no-parameter boxes and full gate/solver provenance. Each frame dictionary has exactly `{frame_alias,rgb,raw_points,camera,instances}`. `instances` has exactly `{boxes_xyxy,class_ids,scores,masks_bool,instance_indices}` from the sealed independent full-image frontend. RGB is original uint8H×W×3; raw points are actual finiteXYZ, never pre-shifted Drone modelXYZ. Camera is R7CameraSpec with explicit missing-distortion policy. No target annotation/positive-map/GTcategory/index/pose/object-ID field is accepted by this interface.

Q0 is original pure-text source argmax and never reselected. The first lexicon noun regex is identical to R7's fixed lexicon; it is parsed from the literal input caption, not a label. Registered category families are:

- vehicle/car/sedan/van/taxi/suv/jeep/truck/bus → COCO{3,6,8}.
- person/pedestrian/man/woman/people/child/boy/girl/walker/worker → COCO{1}.
- bicycle/bike → COCO{2}.
- cyclist/biker/trailer, absent noun or unknown mapping → explicitNA, no GTfallback.

The motorvehicle family deliberately tolerates detector subclass confusion and supplies only category-family evidence. Neither the firstnoun rule, samefamily track nor independent detector verifies the full referring expression's identity/relations. Independent masks remain full-image observations with no source prompt/ROI seed.

## Anchor association

A candidate independent mask must have a compatible predicted class-family, detector-box IoU>=0.1 with the image-clipped projected source3Dbox AABB and at least3 unique visible foreground3Dpoints in the source ROI. ROI is exactly R10's padded geometric crop (`size/2 + min(.25*size+.1m,.75m)`), recorded as a source-conditioned hypothesis. Continuous FOV/depth>.1 then floor-pixel nearest-depth visibility is used. Source corners crossing/behind the nearplane cannot define this association.

Exactly one candidate is required. Multiple samefamily candidates directly produceNA, regardless of confidence, overlap rank or CLIP. Zero candidate/unsupported noun alsoNA. This preserves a fallible source q0 as a hypothesis and rejects ambiguity; it does not create an independent semantic grounding oracle.

## Same objective, view acceptance and motion policy

GrabCut uses whole actual anchor raw points transformed to model coordinates once. Single-instance geometry uses the same actual anchor points and the independent target mask. Both call the root-owned R10 objective copy `geometry_feedback.py`; without its optional allowed-ID filter it is bitexact to originalR10 in synthetic tests. Same source prior/bounds, camera-facing planes, opposing-size rule, RGB one-sided containment and numerical-status policies apply. No centroid/visibleextent-as-complete-box, yaw change or new neural update is introduced.

Each preselected previous/next retrieval is attempted in fixed order (unknown counters are not seconds). No task result chooses neighbors or transforms. Global registration excludes the union of all independent masks in both images, never the q0ROI. The sole primary backend is registered `one_depth_sensor_PnP_v1` (`sensor_pnp.estimate_pair_registration`): RGB SIFT mutual matches bind depth on A only, fit cameraA3D→cameraB image rays with PnPRANSAC, and validate on spatial-heldout image rays plus unused B-background cloud. Main output mapping remains rawB→rawA. Provided metadata pose has no backend API entry and is never a primary input or fallback. A registration must pass its registered heldout rays/cloud and background-exclusion protocol. Neither original two-depth SIFT nor ICP is tried as a fallback after PnP rejects. GlobalSIFT/pose residual does not itself establish target identity.

Independent all-instance masks are linked by the temporal module's class-family-consistent LK forward/backward-cycle, bilateral ambiguity margin and one-to-one matching. Detector car/bus/truck labels are mapped to a single family only for track compatibility. The selected anchor mask's unique track must yield at least6 distinct nearest-depth3Dcorrespondences inside the respective independent masks and pass static support under **global**T. Fewer than6pairs is that-sideNA, not proven target movement and not an automatic veto of another valid side. With>=6pairs and a failed static check, preserve a contradiction and veto **all** multi-view arms for the case; do not select another favorable neighbor to erase it. Single-frame direct output remains unaffected. No target-motion refit or velocity inference is performed.

When at least one neighbor passes and no verified contradiction exists, transform every passing neighbor's **visible instance foreground raw points only** to anchor raw coordinates by estimatedT. All candidate added observations must lie in anchor cameraFOV, the unchanged anchor independent mask and the same padded sourceROI. Main geometry additionally requires each neighbor point to have an anchor visible instance point within3 projected pixels, camera-depth difference<=0.25m and raw3D distance<=0.25m. These thresholds are the existing depth-binding/RANSAC scales, not retuned. A track with6good matches does not authorize all other mask pixels. Whole neighboring background is never aggregated. The main normal-context cloud retains all anchor raw points plus only qualified neighbor points; actual face-loss support is explicitly limited by allowed_raw_ids to cross-view-supported anchor points and the qualified added points. Combined model points/physicalXYZ are deduplicated before normals/planes. Drone raw→model is one in-place float32z+1.8; the sensor mapping is raw, with the equivalent model conjugation S@T@invS explicitly tested. Added counts and point-set hashes at each filter stage are retained. New neighbor depth may be occluded by existing anchor returns during the unchanged zbuffer; a verified pair can produce no geometric increment.

## Fixed8arms

1. source
2. grabcut_direct
3. single_instance_direct
4. multiview_instance_direct
5. wrongview_instance_direct
6. single_instance_half
7. multiview_instance_half
8. multiview_maskmatch_only_direct

Direct G is the no-parameter control. Half arms are exactly float32 `source + .5*(corresponding_direct-source)` on all6coordinates. No threshold is tuned by task labels. The legacy named `multiview_maskmatch_only_direct` arm is now the **unfiltered track-qualified mask-point aggregation** ablation: it adds all matched neighboring visible instance points surviving anchorFOV/mask/ROI without the new per-point pixel/depth/3D corroboration. It shares the exact instance/static acceptance gate and input views with main. Its name is retained for the fixed8-arm interface, but it is **not anchor-only** and does not isolate a pure acceptance effect. No main fallback to this unfiltered result is allowed when corroborated support lacks a qualified plane. The single arm remains unchanged. This explicit pre-run role revision is preserved in history.

Wrongview is the input-locked observation from another scene on the same platform; run the same registration/mask/depth/static/geometry path, no replacement. It may fail or accidentally pass and is never forced to be a negative task effect. Low wrongview acceptance alone is insufficient for adoption. Both scene-micro and scene-equal platform reporting are required because actual146anchors are Drone50(32+18) and Quad96(32each), not the originally hoped160.

## Actual CPU runner contract

`run_feedback.py --plan ROOT_PLAN.json --plan-sha256 HASH --output NEW_DIRECTORY` only writes a new output. Plan exact fields:

- stage=`multiview_direct_geometry`; GT_read_permitted=false; model_or_GPU_execution_permitted=false; expected_samples=146 (runner permits1..160).
- manifest / manifest_sha256: strict9key ANCHOR146_GPU_INPUTS.
- pairmap / pairmap_sha256: PAIRS_PRIVATE object, `rows_PRIVATE` joined by `anchor_id_PRIVATE`; fixed role aliases previous/next/wrongview, never replace missingnext.
- frame_index / frame_index_sha256: input-only FRAME_INDEX_PRIVATE; K/E/D taken from these already audited fields, not parsed from task annotation containers. Inputpose/counters are not used for primary fit/velocity.
- source_dir / source_seal_sha256; mask_dir / mask_seal_sha256.
- program_sha256: workspace-relative complete code closure.
- feedback_config, temporal_config, pnp_config, geometry_config, RGB_config: exact defaultdataclass dictionaries.

The runner checks all source/mask sealed-file hashes and manifest/input-file bindings; mask image hashes and per-mask hashes; unchanged/repeated source; actual sourcecache sample/checkpoint/q0/argmax/manifest/text bindings. It decodes only source boxes/scores/query/input hashes, never source features/model weights. Whole raw image/point observations are decoded only during this root-authorized actual geometry execution. Runtime uses a6frameLRU to bound memory andOpenCV1thread. Integrity-check time, processing time and whole-run time are distinct receipts. No GPU/librarymodel construction occurs.

NPZ fields: `sample_id`, `source_boxes_raw6[256,6]`, `source_text_scores256[256]`, `q_source`, `arm_names[8]`, `arm_query_index[8]`, `arm_status[8]`, `arm_boxes_raw6/eval6[8,6]`, `acceptance_mask6[8,6]`, `actual_changed_axes[8,6]`, `point_support_sha256[8]`, `provenance[8,6]`, sourcepoint/rawindex/sourcecache hashes and anchor raw-image/point-file hashes. Raw boxes stay unmodified; evaluation sizes alone clamp to1e-6. No GT/numeric labels appear.

Private trace preserves all attempted registration/flow/depth/static rejections; mask IDs/class-family scope; original plane/solver diagnostics; merged foreground counts/hashes; anchor/neighbor aliases and clock limitations. No wrong observation is silently replaced. Seal binds every NPZ/trace/plan/input and source/mask/checkpoint hashes before any scoring release. Root owns actual run and independent CPUscoring.

## Engineering checks and limits

`check_feedback_cpu.py` checks objective-copy r0filter behavior, multicandidate and unsupportednounNA, fixedq0/half8arms, insufficient-depth-vs-contradiction distinction, global-veto/single preservation, unfiltered samegate aggregation role, no substitution when main support is empty, wrongview fallback and foreground-only filtered merge. It stubs verified-neighbor outcomes to isolate orchestration; actual SIFT/LK/depth/static behavior is separately exercised in check_temporal_cpu.py. Neither fixture is a real-data or task result.

Temporal final-consensus degeneracy was independently discovered before any actual pair test. Old module/lock/fixture are retained under history. Only the final rigid refit ValueError is converted to explicitNA; no pose/transform fallback or parameter change occurs. Raw file/hash/unknown implementation errors otherwise stop the batch rather than silently drop cases.

## Pre-execution per-point corroboration correction

Root/independent review found that passing6static depth tracks does not justify adding all foreground mask points. Before any actual G/task run, old G/runner/contract/fixtures were copied to `history/preflight_G_v1_before_perpoint_corroboration/`. Main now uses the pointwise corroboration above. Its allowed support is mapped through exactly the same raw→float32model conversion, `np.unique(..., return_inverse=True)` and canonical ordering as the joint fit context; original raw array row indices are never reinterpreted as model-unique IDs. Diagnostics retain corresponding anchor raw-file indices, stage counts, raw/model support hashes and canonical allowed-ID hashes. Unfiltered samegate aggregation is a separateablation; no threshold/objective/source/q0 change and no target labels are involved.

## Pre-G primary backend registration

Root performed a fixed input-only5-pair engineering probe before G/GT: two-depth SIFT0/5, an independently checked classicalICP0/5, one-depthPnP2/5real and0/5wrong-view acceptance. Those results use different validation rules, only5input-development pairs and the same target-scene acquisition context; they do not establish algorithmic superiority or independent task confirmation. Root explicitly registers PnP as the only primary backend and ends registration variants/parameter searches. Old SIFT G/runner/plan/lock/fixture/documentation are retained under `history/preflight_G_v2_SIFT_before_PnP_backend/`. The selected backend does not relax object static support: it still requires6unique **two-depth** object-track pairs and the existing static/pointwise3px/0.25m checks. Full146coverage may remain low. No task result chose this backend.

PnP's independently discovered empty-B-cloud-holdout edge was corrected by root before actual G: if fewer than20heldout Bcloud points exist, return explicitNA before quantiles. Old module/probe evidence remain in root history. Main fit and thresholds did not change. Import closure includes sensor_pnp.py, sensor_icp.py (its RGB matching helper), temporal_association.py and R7 camera projection. `FeedbackConfig.registration_backend` only permits `one_depth_sensor_PnP_v1`; the runner compares the full defaultPnPConfig to the root plan. A synthetic routing spy patches the actualPnP function and makes any old-SIFT call fail.
