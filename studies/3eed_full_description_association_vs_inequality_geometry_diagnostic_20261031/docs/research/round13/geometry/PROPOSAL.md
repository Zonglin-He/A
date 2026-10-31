# R13 fixed-q0 observation geometry — preimplementation proposal

State: proposal; no real sensor inputs, task labels, scoring files, model or GPU accessed. Scope is B only (local observation → complete axis-aligned box); complete-text target association is a separate assay.

## Evidence and competing explanation

Read R12 `crossplatform/round12/geometry_feedback.py` and `research/round12/experiment/REPORT.md`/`DECISION.json`. R12's observed-surface loss penalizes `p_j - (c_j + sign*s_j/2)`. Independent or cross-view-confirmed local surfaces do not prove those surfaces coincide with the amodal AABB boundary. R12 reports 23 eligible multiview anchors but only two changed centers and no sizes; task failures include foreground spatial-support proxies near 100%. These are motivation, not causal proof that the equality term alone caused every failure.

## Proposed bounded observation model

- Fixed source query q0, source six-dimensional box, no candidate reselection/fusion.
- Input target foreground support must be independently associated upstream; this module does not certify literal text identity. De-duplicate physical return IDs and coincident XYZ before counting or loss.
- Point containment: residual is `max(abs(p-c)-s/2-epsilon, 0)`. No attraction to faces, point means or extents. Epsilon=.03m, normalization sigma=.05m. Huber delta=1, mean over distinct points and XYZ.
- Independent foreground silhouette containment: mask boundary samples must be inside the projected eight-corner convex hull. One-sided outside distance only, normalized by `max(3px, .05*min(source projected width,height))`; no mask-box equality, negative-pixel completeness or center-in-mask constraint. Projection is the frozen R7 calibrated convention.
- Source minimum-change prior: center standard deviation clipped `.15*source_size` to [.05,.35]m; log-size std=.1. Fixed weights prior=point=RGB=1.
- Center bounds ±min(.25*source_size,.5m). Size range [source_size, 1.2*source_size]. Size can expand where containment is violated. Shrinkage has no evidence under these one-sided observations, so it remains unidentifiable and is disallowed; this is an explicit prior limitation, not an observed true-size assertion.
- Shared gate: valid positive source dimensions, valid camera and independent nonempty mask, ≥8 unique foreground XYZ. Full/no-RGB share the exact same points, gate, six optimization coordinates and physical bounds. RGB may supply a new lower-bound violation; no-RGB need not move when point containment is already satisfied.
- Null/underconstrained components stay at the source under the minimum-change prior. A moved center remains a prior-regularized solution, never an independently observed amodal center. Source already satisfying all constraints yields exact source without invoking optimization.
- Fixed optimizer: bounded L-BFGS-B, max64 iterations, normalized finite difference 1e-5, fixed ftol/gtol; finite nonconvergence retained with explicit status, numerical invalid returns source.

## Free-space boundary

The AABB is an enclosing volume, not an opaque solid. A ray may traverse empty space inside a valid object's AABB before hitting the object. Therefore `box ray-entry == first return` is invalid and is not used. Camera calibration does not establish the LiDAR firing origin. Optional explicit sensor-ray input may expose certified first-return/ray-origin diagnostics, but neither absence nor a ray/AABB crossing drives this version. A physically justified stronger occupancy model would be a separate protocol.

## Matched controls and interface

Proposed callable `generate_observation_targets(*, points_fit, raw_indices, source_box6, q0, camera_spec, rgb_mask2d, segmentation_valid, config=...)`, no file access and no arbitrary annotation mappings. Four direct/no-parameter arms: `source`, `direct_full`, `direct_no_rgb`, `half_full`. Half is source+.5*(full-source). Return selected boxes, fixed query indices, shared acceptance, actual float32 changes, support hashes, per-coordinate prior/inequality provenance and constraint/prior terms. No neural update in this module.

## Necessary synthetic checks before lock

1. Partial front surface entirely in source: exact identity across visible-area fractions; loss does not force shrinkage.
2. Outside translated foreground support: bounded containment correction can move center/expand size, not replace center with point mean.
3. Center/size tradeoff: residual reduction may improve containment while worsening known synthetic complete-box error; explicit failure retained.
4. Mask one-sided containment does not shrink source to a partial mask; RGB on/off share observations and bounds.
5. Independent mask/background wrong identity can drive wrong correction; no identity claim.
6. Physical-ID/XYZ duplicates and permutations leave outputs/counts invariant.
7. Invalid dimensions, near-plane camera and insufficient support explicit source fallback; nonfinite diagnostics are JSON-safe.
8. Box ray-entry preceding a first return is not itself rejected.

## Decision boundary

First test actual direct boxes under root's separately frozen sensor/model protocol and independent sealed CPU scoring. Do not introduce parameter adaptation unless the preregistered direct-feedback utility gate passes. Loss descent, nonzero coverage and synthetic fitting alone do not establish task correctness.
