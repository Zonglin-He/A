# R12 prerequisite: real independent foreground and sequential observations

Status: method proposal only, awaiting root's actual observation/pose inventory. No new model/GPU inference, target GT/scoring/index access, raw target media/point decoding, downloads or method implementation by this author.

## Why this experiment differs

R11's two point partitions shared sourceq0, RGB, ROI and full-shape prior. Its high consistency could repeat the same wrong wall/tree/background; current formula was not adopted and neural TTA did not execute. A new consistent score on the same source-seeded foreground would not resolve that failure. R12 should first establish **foreground proposals created independently of q0**, then physical cross-frame correspondence, then test whether those observations provide a useful direct geometric target. No parameter update is justified before this direct-target prerequisite.

## What code actually establishes and what it does not

Released `src/joint_det_dataset.py` loads image/point/meta paths, stores `meta['pose']` in annotation entries, and samples16384XYZ. It does not establish temporal identity or use pose to align neighbors. Its Quad virtual transform path is commented; Drone model input adds1.8m in z. Frame enumeration uses unsorted `os.listdir`; numeric directory names and adjacent entries are not verified timing. Waymo suffixes can denote cameras, not different temporal observations.

`utils/align_3eed.convert_points_to_virtual` extracts roll/pitch while dropping yaw and horizontal translation; its optional Drone translation is a height adjustment. That function is not a full body-to-world transform and cannot be silently used to fuse frames. The mere presence of an array named pose does not establish its provenance/direction or whether released XYZ was already virtually aligned.

The locked R7 camera bridge maps **model coordinates to the contemporaneous image**: Waymo `A@inv(E)@undo_front`, Drone `E@undo_z`, Quad `E`. It deliberately does not add world pose. Camera extrinsic E cannot be substituted for ego motion. Missing D remains a declared zero-distortion assumption, not proof that images were undistorted.

The official M3ED source documents separate synchronized data, frame timestamps and pose/depth products; the compact3EED release may not retain these or its raw index mapping. Source documentation is not evidence that local files expose usable neighbor calibration. Localization reference products are auxiliary pose information; if used, their source/offline nature must be disclosed separately from task GT boxes. No semantics/depth GT arrays may be consumed as an unlabelled foreground substitute.

## Required observation inventory before choosing a branch

1. Same actual sensor/sequence and camera identity; image and point hashes show distinct physical observations rather than repeated or renumbered frames. Frame index↔timestamp mapping, units and monotonicity must be explicit; filename adjacency is insufficient.
2. RGB/LiDAR alignment at each timestamp, scan start/end and image time if available. Intrinsics/resolution/distortion and camera-to-lidar/body transformations require traceable conventions. Global/body pose must be distinguished from static sensor extrinsics and virtual/prelevelled frame transforms.
3. Pose shape, rigidity, units, direction, changing/static behavior and provenance. Repeated identity pose can reflect a stationary sensor or missing localization; distinguish these with independent image/background evidence. Do not pick pose direction by GT or silently search transforms until overlap is high.
4. At least a bounded window with valid overlap and measurable view/observation complementarity. A nearby timestamp without visible overlap is not extra target evidence. Large time gaps, strong blur/rolling shutter, unsupported motion, no foreground or ambiguous sameclass neighbors remain explicit NA.
5. Current target validation scenes have historical research exposure. New windows may be new observations but must not be called new scenes or independent heldout population evidence.

A verified full-pose branch can proceed. If poses are unavailable but images are genuinely nearby/synchronized, a separately registered background-only visual/point odometry diagnostic is possible. It is not a free replacement: exclude all independent foreground masks when estimating static ego motion; reject degenerate planes, low overlap, poor condition or conflicting image/point estimates. ICP alone requires a defensible initialization and can align the wrong surfaces. If no branch is verifiable, acquire a small traceable raw sequence subset only under root's bounded plan; do not manufacture multiview from point resampling, source augmentations or camera suffixes.

## Preferred foreground and correspondence branch

Use the single frozen **full-image** COCO instance segmenter specified in MASK_INTERFACE.md. It consumes no q0/prompt/caption/3D crop. Every frame's candidates are independent output masks. Pure text target-noun→class mapping is fixed before any labels; unsupported class or unresolved semantics yields NA. Q0 is only a hypothesis to verify against these candidates. A segmentation class or a unique geometric match does not prove literalcaption identity.

Before geometry, perform one-to-one track correspondence using appearance/optical-flow evidence plus calibrated point depth and ego-compensated spatial consistency. Candidate masks/point observations are generated across the entire image, not grown from source ROI. Source overlap can restrict which independently generated tracks are plausible, but cannot by itself count as independent evidence. Multiple similarly plausible sameclass objects, inconsistent forward/backward mapping or disjoint depth support must stay ambiguous rather than choosing the nearest/highest-score mask.

For optical flow, use image-wide independent features and robust tracking with forward/backward and cycle checks. The apparent motion combines camera and object motion. Verified ego motion predicts static-background flow; residual motion is not automatically a target foreground label because parallax, bad depth, rolling shutter and occlusion can cause residuals. Conversely stationary objects cannot be required to have nonzero motion. A source-seeded track alone is only tracking a hypothesis; it cannot validate the original grounding choice.

## Geometry and motion separation

All transforms must explicitly state source and destination frames. For example let F_i map released rawXYZ→3EED modelXYZ, S_i map rawXYZ→body_i, and Pose_i map body_i→world. For a static physical point, model-i→anchor-model0 is:

`F_0 @ inv(S_0) @ inv(Pose_0) @ Pose_i @ S_i @ inv(F_i)`.

Only use this after F/S/Pose conventions are verified. Drone1.8m and Waymo source-front rotations belong in F, not ego motion. If released point coordinates already incorporate a virtual transform, that transform must be inverted or accounted for exactly once. Camera projection is a separate contemporaneous transform. Same-time different cameras are independent RGB views but may share the same LiDAR scan; disclose the absence of new3D returns.

Static background and moving targets cannot share the same fusion transform. For a moving target, establish depth-supported cross-frame point correspondences within an independently identified instance track, estimate a bounded object rigid/translation motion and its residuals, then motion-compensate to the anchor time. Median matched-point displacement estimates **motion**, not the full object's center. Articulated people, unknown dt, ambiguous correspondences or unconstrained rotation must failclosed; do not erase target motion by fitting it as camera motion. If object motion is unsupported, report the case as unsupported rather than statically accumulating it.

Occlusion uses projected nearest-depth visibility and forward/backward track survival. Absence of a surface in a view does not imply it is a background point or an object's boundary. Full-instance masks are fallible visible silhouettes, not amodal shapes. A sparse/truncated mask may not constrain the opposite face. New geometry should combine actual independently associated point planes/visibility bounds across views with a source shape prior; only identifiable center/size coordinates may change. No visible-point mean or partial extent becomes a full box, and nativeyaw0 mismatch remains a retained limitation.

## Minimal direct-output test and exact information controls

Freeze source checkpoint, anchor queryq0, caption interpretation, camera convention, selected windows, class map, frontend checkpoint and confidence/mask settings. Do not assume query-slot indices or view-relative words refer to the same physical object across frames. The original caption remains anchored to its reference view; tracks propagate the physical hypothesis, not an unchanged left/right relation.

Minimum arms under the same fixed anchorq0:

- Source anchor box.
- Single-frame independent-mask geometry direct output (no neural update).
- Verified multiframe/temporal-correspondence geometry direct output (no neural update).
- Same-information no-update geometric/readout/track aggregation control, sharing every auxiliary mask, pose and neighbor frame used by the proposed method. Keep any difference between target-track evidence and extra mask-model computation explicit.
- Wrong-neighbor/within-sequence temporal shuffle, with the same frame count and frontend compute. Preserve timestamps/actual transforms rather than fabricating a matching camera. For an order-only permutation a static sequence can remain physically valid: such a permutation is not automatically a destructive negative. A registered frame–pose or wrong-track mismatch tests association; correct reversing with correct transforms is a consistency control, not expected failure.
- A missing-pose/motion-compensation control only when physically meaningful and predeclared. Its failure cannot be assumed in a stationary-camera/object sequence.

No GT selects tracks, model variants, shifts or windows. First seal direct boxes, associations, masks, physical observations and all rejection reasons; only then independent CPU scoring/positive and negative tails. Preserve low coverage and all source fallbacks. Compare improvements against the single-frame independent mask, not only the old source-seeded R11 G. Any gain could arise from the external2D model or simple multiview aggregation and must be assigned to that control.

## Conditional unused-observation neural update only after a valid target

Do not implement or run this phase now. If direct independent geometry beats source and same-information controls under the registered resource gate, register a separate support-window A and unused-view B experiment. B's image/points may not enter A's segmentation seeds, identity selection, motion estimation or geometric pseudo fit. Shared sequence and any shared2Dmodel remain disclosed. Hold B's frozen q0 selection fixed and evaluate a prediction not directly fitted to its own pseudo box.

Retain sourceB, A-only geometric target transferred to B with verified object/ego motion, parameter-free same-information/cross-feature-kernel transfer, and episodic last-center/size-head adaptation A→B. Per-view query correspondence must be established geometrically rather than assuming same query index, and caption viewpoint changes must not be ignored. Same-input exact fit cannot be the contribution. Neural weights/gradients/RNG/buffers reset per episode; no teacher/EMA or new source training. If direct target or observation prerequisites fail, retain the failed prerequisite and stop before parameter updates.

## Scope now

Only source/code/documentation inspection and proposal preparation are complete. Actual local timestamps, pose semantics, camera stream continuity, object motion support and frontend coverage are **unverified** by this author. Root's inventory selects the available branch and owns acquisition, input locks, GPU and execution. No thresholds, method implementation or large-asset download are silently added here.

## Root-verified branch update (after the initial proposal)

Root reports actual Drone/Quad metadata with changing4x4pose and pairwise constantK/E, but timestamp counters have resets, repeats and nonmonotonicity. This supports investigating candidate **overlapping views**, not claiming a trustworthy continuous clock. Provided pose provenance remains unresolved and may be an offline localization reference; it is diagnostic only. Root now authorizes implementing sensor-only RGBD/SIFT registration plus independent-mask tracking in new `crossplatform/round12/temporal_association.py`; actual data execution still requires its separate input plan. Its exact APIs/one configuration/synthetic checks are in TEMPORAL_INTERFACE.md. The previously sketched full-pose primary branch is therefore not adopted.

Root selects official torchvision0.22 Mask R-CNN ResNet50-FPN-v2 COCO_V1 as the sole independent2D frontend, replacing the initial model-agnostic small-instance-segmenter example. Mask generation is全图/noq0/no caption, with registered0.5score/mask operating points and native resize. Root owns download/inference; this author does neither.

Primary references: [M3ED data-file timestamp/calibration schema](https://m3ed.io/data_overview/datafiles/), [M3ED processed files and pose products](https://m3ed.io/data_overview/folderst/), [Waymo camera/pose proto](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/dataset.proto), [OpenCV optical-flow assumptions and API](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html), [Open3D ICP initialization and residual definitions](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html). These public descriptions do not prove that local3EED files preserve raw timing/coordinate semantics.
