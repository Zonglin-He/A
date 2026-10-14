# Local motion-residual region candidates — conditional proposal

**Status: proposal only; not implemented or executed.** This note does not authorize candidate generation, RGB decoding, teacher calls, model use, GPU work, or GT scoring. The single proposed mechanism becomes eligible only if the already locked matched-support `Source / R1 / PoolOracle / GT Box Oracle` diagnosis supports candidate insufficiency relative to its GT control. The locked interpretation distinguishes that result from `PoolOracle succeeds but R1 does not` (selection loss) and from `GT Box Oracle also fails` (optimization, reachability, or sparse-to-full transfer remains unresolved). A low PoolOracle fit by itself is not evidence that candidate geometry is the bottleneck. No numeric threshold is added here; apply the existing protocol and root review before any execution.

The registered method in `methods/CURRENT_METHOD.json` remains `decota_spatial_opd_v1`. This proposal does not change the current method, the Temporal branch, the OPD objective, prompts, or optimizer.

## Evidence and available inputs

The only contemplated inputs are the already sealed short-window cache and its fixed eight historically exposed sources. Its no-GT CPU barrier records 8 cases, 32 center windows, 160 unique selected RGB frames, and 152 existing proposal tracklets. The independent tracklet readback counted 427 accepted frame boxes out of 760 possible candidate-frame slots (152 tracklets × five frames). That is 56.18% **tracklet-box support**, not object coverage, localization quality, or identity accuracy.

The independent frozen-feature association readback found 99 mutual-nearest edges among the full candidate pools over 24 adjacent-window boundaries. Fifteen of the 24 selected adjacent candidate pairs were mutual-nearest; all 15 were Source-to-Source. No selected non-Source position received that support. The sealed reference lock consequently has 4 `local_R1` anchors and 0 `associated_R1` anchors. These counts describe a limited local-association interface; they do not establish that the proposed region generator will find the target.

The current matched-support lock contains 19 common center positions and excludes 13 of the 32 centers by availability. This is only the locked support size, not a four-arm effect result. The four-arm results remain the decision gate. In particular, do not conflate “R1 had no usable reference” with “the existing candidate pool lacked the target”: the `PoolOracle` and `GT Box Oracle` arms are needed to separate those cases.

The existing runner saves each case’s already decoded local frames as `private_rgb/{case}.npz`, with `rgb_uint8` and `gray_uint8` arrays. Each case contributes 20 unique frames across its four five-frame windows; decoded frames were resized with `INTER_AREA` to a maximum long edge of 640 pixels. The cache is source/hash/frame bound by `GLOBAL_TRACKLET_BARRIER.json`. This proposal needs no full-video pass or additional decode if that cache remains valid. OpenCV 4.11.0 is present in the captured runtime; the CPU build exposes `goodFeaturesToTrack`, pyramidal LK, `estimateAffinePartial2D`, `calcOpticalFlowFarneback`, and `connectedComponentsWithStats`.

The existing `stvg_adjacent_affine_tracklet_v1.py` only propagates an input box: it detects Shi-Tomasi corners inside that candidate, checks forward/backward LK, estimates an affine transform by RANSAC, and rejects failed/out-of-bounds geometry. It cannot propose a new region. Its locked settings include LK 21×21, pyramid level 3, 30 iterations, forward/backward error 1.5 px, affine reprojection threshold 2 px, and no box clipping. The new proposal below is materially different because it first estimates scene motion over the full frame, segments residual local motion, and forms regions from those components.

## One fixed mechanism, if the gate is met

For each of the 32 existing five-frame windows, use the cached grayscale frames in physical-frame order. Keep the original Source candidate at its existing geometry as candidate 1. Generate at most four additional **window-local** regions; do not copy the old proposal boxes as seeds, do not carry a track identity across windows, and do not retrieve or add other frames.

1. For each consecutive pair, detect full-frame Shi-Tomasi points (maximum 500, quality 0.01, minimum spacing 5 px, 3×3 block). Track them forward and backward using the existing LK settings and retain correspondences with forward/backward error at most 1.5 px. Fit one full-frame `estimateAffinePartial2D` camera transform with the existing RANSAC settings (2 px reprojection threshold, 2,000 iterations, 0.99 confidence, 10 refinement iterations; require at least four inliers and a 0.5 inlier fraction). If the fit fails, mark that transition unavailable; do not fall back to a different motion model.

2. On each valid pair, compute dense forward and backward Farnebäck flow at the cached resolution with the fixed configuration `pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0`. Subtract the camera transform’s predicted flow from each dense flow. Retain pixels whose residual magnitude is at least 1.5 px and whose forward/backward cycle error is at most 1.5 px. The segmentation is therefore based on local motion after dominant scene motion, not on caption, candidate box, GT, or learned features.

3. Apply one fixed 3×3 morphological open followed by a 3×3 close. Extract 8-connected components. Keep components with area from 64 pixels through 25% of the frame area. Preserve empty masks and every rejected component’s reason in metadata. These are fixed proposal constants, not a tuning grid. A failed camera fit or an empty residual mask yields no new region for that transition.

4. Match components between consecutive frames by warping each binary component mask with the corresponding dense flow. Use reciprocal nearest component matches by warped-mask IoU; break exact ties by `(top, left, bottom, right)` row-major order. Require the forward/backward flow cycle check above. Never bridge a missing transition or restart a lost local track from a later frame.

5. At each center, retain only a component trajectory that includes a center-frame component and has at least one accepted match on both the preceding and following side. Its per-frame boxes are the exact component envelopes, normalized by the cached frame dimensions. Missing boxes remain `null`. Rank eligible trajectories by (a) number of supported frames, descending; (b) accepted adjacent matches, descending; (c) median cycle error, ascending; then (d) center component row-major position. Assign local IDs in this order and keep only the first four. Record overflow and dropped tracks. If no trajectory qualifies, keep Source alone.

This gives a fixed maximum of five displayed candidates per window (Source plus four regions), or 160 tracklets and at most 800 candidate-frame slots over the fixed 32 windows. Those are capacity bounds, not predicted or measured output counts. IDs are valid only inside their five-frame window. They do not imply that ID 2 in one window is the same object as ID 2 in another.

## Evaluation and stop conditions

First perform a no-GT CPU coverage readback on the fixed cache: count valid camera transforms, residual components, center components, two-sided trajectories, missing transitions, rejected components, cap overflow, and supported frame boxes. Compare counts with the existing 152-tracklet / 427-valid-box baseline, while labeling both as proposal support rather than semantic coverage. Do not send candidates to R1 or run OPD in this first readback.

If root then authorizes the one conditional experiment, seal every generated box and candidate mapping before any teacher call. Report candidate presence and box IoU on the already locked common support alongside `PoolOracle`; keep the GT Box Oracle strictly offline. Candidate generation and ranking remain GT-free. Preserve all 32 centers and all 8 cases, including empty outputs. Do not condition the denominator on whether the new mechanism produced a region. Do not use GT to choose thresholds, components, step, or candidate cap.

For identity, report only measurable quantities: within-window reciprocal match counts, cycle-error distribution, fraction of retained tracks supported on both sides of center, and (where the sparse center annotation exists) center-box IoU against the locked GT diagnostic. The short-window data do not provide dense identity labels for all 160 frames. Cycle consistency is an internal flow check, not evidence of semantic object identity. No cross-window identity accuracy claim is available from these inputs.

Stop before R1/OPD if the CPU readback yields no two-sided center trajectories, if component output is dominated by failed camera fits or cap overflow, or if the locked four-arm diagnosis does not support candidate insufficiency. A later R1/OPD stage would require a separate root lock after the coverage readback; this proposal does not authorize that stage.

## Known failure modes and limits

- A stationary target with little motion relative to the background produces no residual motion and will be missed. This method cannot replace static-object proposals.
- If the target dominates the frame, the global affine estimate can absorb target motion; if the scene has parallax or a moving background, residual components may represent camera-model error rather than objects.
- Low texture, blur, lighting changes, shadows, reflections, occlusion, and fast motion can create absent, fragmented, or merged components. A five-frame window offers little recovery after occlusion.
- Motion residuals have no category or query semantics. A moving distractor may rank above the referent, even with a stable local trajectory.
- The connected component box may cover only the moving part of an object; it is not a segmentation mask, amodal box, or object detector output.
- The eight sources were previously exposed. A positive result would remain a finite development diagnosis, not independent confirmation. No new sources or backbones are part of this proposal.

## Read-only feasibility pins

This note is based on source and metadata inspection only; no candidate generator was implemented, and no RGB array, GT box, model, teacher response, or GPU state was opened for this proposal.

| Item | SHA-256 / observed value |
|---|---|
| `methods/CURRENT_METHOD.json` | `146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4` |
| `scripts/run_stvg_short_event_tracklets_v1.py` | `27682e8bb7ca8014a3b8e49861af74aefc42c7a505215d4dfada01db0284c401` |
| `scripts/prepare_stvg_short_event_tracklets_v1.py` | `2b113d7997b8e64d00223baed05e732f6615a03c474932e69098e56a335e418a` |
| `scripts/prepare_stvg_short_window_teacher_v1.py` | `aa6b7d76baecfe0f555b3de6c8ee6f001df1d919a4bbead2fca658997bbf6fa9` |
| `scripts/stvg_adjacent_affine_tracklet_v1.py` | `00af1dc7921e539a27993fef1cd0374e9f8e9e4077c81df6007143fbb6910313` |
| `scripts/lock_stvg_short_window_references_v1.py` | `5a75f4c198b60b5dfc982bf814b4c3d3fcb88594783db142c5791973e8647af2` |
| `GLOBAL_TRACKLET_BARRIER.json` | `f4a49f229b270e0f2370307ed8384102eb99d306c0f461df872ad86a7a6f27a0` |
| `PREPARED_MANIFEST.json` | `ffee415b4b2175196978d11de853441d8d3e54161c5fe40b34adf8215900e57f` |
| `ROOT_TRACKLET_READBACK.json` | `be91c5a46d6ca216e236e146b4dd697f7cbce46e7e7b779023c83f37bcf4abe8` |
| `ROOT_REFERENCE_ASSOCIATION_READBACK003.json` | `52d838a8ab3f14f8de035d6cd7deb7cc7533e4757b9bca0f8710b4644ed1b520` |
| `protocols/stvg_matched_support_reference_v1.md` | `d9d834a6e3c8ada86e4e886d5e42ff2089deda245672baee9d8ae205658c227b` |
| OpenCV runtime | `4.11.0`, CPU APIs listed above are present |

The four-arm experiment must finish and its root readback must decide the gate. This proposal supplies one implementable candidate mechanism only; it does not authorize running it.
