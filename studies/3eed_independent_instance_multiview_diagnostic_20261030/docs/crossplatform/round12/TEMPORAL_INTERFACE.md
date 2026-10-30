# R12 sensor-only registration/independent-mask interface

CPU preparation only. Root owns actual-view input lock and first real execution. No actual RGB/point/metadata/GT/model access occurred in this implementation preparation. Current local pose/timestamp findings are supplied by root's inventory, not independently recomputed here.

## Domain and clocks

Primary mapping is **raw B → raw A**, `p_A = T_raw_B_to_A p_B`, column-vector convention. Current released counters have resets/repeats; nearby folder indices are retrieval cues only. This is a candidate **overlapping-observation/multiview** experiment, not verified continuous video. All outputs mark clock unknown/unverified and never estimate velocity, assume seconds, claim causal online inference or enforce a speed bound.

Provided4x4pose may originate from an offline localization reference. It is diagnostic only: neither RANSAC initialization, random split, correspondence selection, inlier selection nor acceptance uses it. After the sensor estimator has finished, compute heldout/fit residuals for identity, inv(poseA)@poseB and inv(poseB)@poseA. Report every hypothesis rather than choosing a supplied transform as the main mapping. Algebraic agreement cannot prove metadata provenance.

## Core array-only calls

`validate_se3(T)`, `inverse_se3(T)`: finite4x4, homogeneous last row, orthogonal determinant+1 rotation. No automatic repair, unit guess or Euler convention guess.

`estimate_pair_registration(*, rgb_A, raw_points_A, camera_A, rgb_B, raw_points_B, camera_B, excluded_mask_A=None, excluded_mask_B=None, provided_pose_A=None, provided_pose_B=None, seed=0, config=TemporalConfig())`:

- Full uint8RGB H×W×3; raw finiteXYZ N×3, not Drone model-z-shifted inputs; camera is the locked R7CameraSpec.
- Optional original-resolution bool exclusion masks are generated from the **entire independent2D frontend**, never q0/target-ROI prompts. Potentially dynamic COCO foreground is excluded from global estimation, irrespective of the grounding hypothesis. Missing masks are explicitly flagged as preflight-only unqualified background estimation.
- Full-image SIFT, max2000 keypoints, ratio<0.75 in each direction and mutual descriptor match. No caption/q0/model/GT arguments exist.
- Raw points project through the known raw→model→camera convention. Continuous FOV/depth>0.1, floor-pixel nearest-depth zbuffer, then nearest visible return within3px binds each keypoint. Duplicate3D observations on either side cannot create multiple matched pairs: greedily retain distinct physical returns in descriptor-distance order.
- Return candidate T, accepted/status, matched/fit/heldout counts and pair hashes, residual distributions, transform magnitude, source/depth/SIFT diagnostics and supplied-pose comparisons. A candidate transform with accepted=false is **not** authorized geometry support.

`estimate_from_correspondences(*, points_A, points_B, ...)` is the testable sensor estimator. It canonically orders paired coordinates, partitions rawB spatial0.5m cells by deterministic sample seed hash (one in four cells held out), runs5123-point noncollinear RANSAC hypotheses on fit only with0.25m inlier distance, and refits once on winning fit inliers. Minimum fit6/holdout3; fit inlier fraction>=0.5; heldout fraction>=0.6, median<=0.15m, p90<=0.5m. These constants are one pre-run engineering configuration. Heldout points only qualify/reject the fit and never tune/choose T. Pose diagnostics run last. Point-plane/ICP-only locally ambiguous registration is not added as an unregistered fallback.

`raw_pair_to_model_transform(Traw,camera_A,camera_B)` gives `F_A@Traw@inv(F_B)`. For Drone same-platform this is `S(+1.8)@Traw@S(-1.8)`, not Traw itself. Waymo F contains its suffix front rotation; Quad F is identity. Camera extrinsics remain a different map. This conjugation is verified by synthetic rotated transforms where skipping the Drone shift produces a nonzero error.

## Independent instance tracks and static-support qualification

`match_instance_masks(*, rgb_A, rgb_B, masks_A, masks_B, class_ids_A, class_ids_B, config=...)` accepts all independent full-resolution binary instance masks and predicted class indices, neverq0/caption/GT. It uses up to64 Shi-Tomasi points per mask and LK21px window/3pyramidlevels, forward/backward error<=1.5px. Class-compatible mask pairs need at least6 cycle-surviving tracks in each direction. Score is the geometric mean of fractions of tracked endpoints contained in the opposite mask. Retain score>=0.5 with a0.15 margin over every alternative on **both** sides, then one-to-one Hungarian assignment. Equally plausible sameclass duplicates are rejected. Return match/unmatched IDs, full score/valid/margin matrices, cycle diagnostics and tracked original-pixel point arrays for depth binding.

This is a two-frame optical-flow cycle test, not literal referring-expression identity or a3D/motion proof. Any three-view composition must add an explicit third-view cycle check. Matching the wrong similar object or stable background remains possible. Association is independent of sourceq0 initialization; q0 remains a later hypothesis to validate, never a mask/flow seed.

`bind_raw_depth_to_pixel_pairs(*, pixel_A, pixel_B, raw_points_A, camera_A, raw_points_B, camera_B, image_hw_A, image_hw_B, include_mask_A=None, include_mask_B=None)` binds matched original pixel coordinates to unique nearest-depth raw returns<=3px. Object include masks, when supplied, restrict candidate depth returns to the independently generated matched instances. It does not use partial point means/extents as full box center/size.

`check_static_object_support(*, points_A, points_B, T_raw_B_to_A)` tests the object-correspondence support under independently estimated **global** T. It requires at least6 pairs and the same registered residual bounds. Unsupported movement, nonrigidity, occlusion or wrong correspondence yields explicit NA. No object-motion transform is fitted to force this static branch to pass. Unknown clock precludes speed claims. A static-compatible correspondence is not a verified physical target identity.

## Geometry caller contract

The later G must require: accepted sensor registration; external mask frontend with input/config/model/hash binding; unique sameclass association with no unresolved alternatives; visible depth support; and static-compatible target track. If any fail, source fallback/NA remains in the fixed denominator. Do not replace a failed sensor transform with a supplied pose or nearest-index assumption. Mask exclusion+successful residual checks only qualify the declared background-fit protocol; they do not certify that every fitted point is true static background.

Pass independent raw instance point supports per view, intrinsics/extrinsics/distortion policy, estimated raw neighbor→anchor transforms, and current source anchor box/q0 as a separate prior. The sourceprior cannot generate or alter frontend masks. Partial silhouettes are one-sided visible support, not amodal/full-box equality. Fullshape and yaw0 limitations persist; no new center/size fitter is implemented in this module.

## Necessary synthetic checks

CPU_FIXTURE.json binds known-SE3 recovery with fit outliers; supplied-pose independence; heldout rejection with unchanged fit; permutation invariance; inverse/reflection handling; Drone shift conjugation; an actual SIFT+nearest-depth pipeline on artificial texture; real LK mask-cycle uniqueness and sameclass ambiguity rejection; moving/wrong3D support rejection without refitting motion; empty-texture NA. All arrays are generated in memory. These certify implementation behavior, not local data overlap or task correctness.
