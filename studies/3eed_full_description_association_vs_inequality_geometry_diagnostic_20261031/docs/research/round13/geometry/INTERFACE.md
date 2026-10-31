# R13 B CPU interface (no parameter adaptation)

```python
from crossplatform.round13.geometry_observation import generate_observation_targets
r = generate_observation_targets(
    points_support_model=foreground_model_xyz,  # Nx3, physical observations, caller association
    mask_bool=independent_full_resolution_bool_mask,  # HxW
    camera_spec=audited_R7_CameraSpec,
    source6=source_pool[q0], q0=int(q0),
    segmentation_valid=True,
    physical_raw_ids=optional_integer_raw_IDs,
)
```

No file paths, arbitrary metadata/GT mappings, class maps, other candidates or labels are accepted. The input points must already be associated to the independently predicted mask; this module does not reselect points by a new source crop or certify identity. No source proposal ranking changes. The caller owns all actual inputs/seals and task scoring.

## Exact output schema

- `arm_names`: U array[4] = source/direct_full/direct_no_rgb/half_full.
- `arm_query_index`: int64[4], all fixed q0.
- `arm_boxes_raw6`: float32[4,6]; dimensions positive for successful valid source, raw invalid source is preserved without repair.
- `arm_status`: four strings; source, already-satisfied source, converged bounded result, finite nonconverged bounded result, explicit fallback or fixed half.
- `active_mask6`, `acceptance_mask6`: bool[4,6]; these describe shared allowed optimization coordinates, **not six separately identified physical parameters**. If any initial positive-observation inequality is violated, all six coordinates receive the same fixed physical bounds in full/noRGB; otherwise all inactive/source. Prior resolves the center/size and depth ambiguity. Source first row inactive.
- `actual_changed_axes`: exact float32 output != source, bool[4,6]. Do not count active as changed/observed.
- `point_support_sha256`: U64[4], all hash sorted unique float32 XYZ bytes (little endian). Optional raw IDs must have consistent coordinates; coincident XYZ never increase support weight.
- `mask_sha256_by_arm`: U64[4], same independent bool mask including for noRGB, because coverage is controlled. noRGB only disables the coordinate constraint term.
- `provenance`: U192[4,6], source prior vs point containment lower bound + source prior; size is source lower prior with permitted expansion, **not amodal recovery**; RGB one-sided availability additionally marked.
- `diagnostics`: JSON-safe, config/hash, physical dedup, common gate/reason, source constraint terms, point per-axis violation magnitude/rate, boundary violation magnitude/rate, prior scales, normalized bounds, shared active definition, solver original float64 and final delivered float32 terms. Nonfinite diagnostic values are None; delivered boxes always finite.

## Normalization and physical bounds

Centers optimized as delta/source_size; sizes as log(size/source_size). Source center prior std is clip(.15size,.05m,.35m), log-size std=.1. Center absolute bound min(.25size,.5m); log-size range [0,log1.2]. Thus no partial-positive observation is used as a shrinking/amodal upper bound. All six variables share full/noRGB optimizer and bounds, but positive evidence is underdetermined and source prior chooses a regularized solution.

The point loss is mean Huber(max(abs(p−c)−s/2−.03m,0)/.05m) over distinct XYZ and axes. Independent RGB boundary loss is mean Huber(positive outside distance to projected AABB hull / source-fixed pixel sigma). It is foreground containment only. There is no surface equality/nearest-face/PCA, RGB background equality, point mean/extent direct replacement, ray-entry equality or free-space runtime. AABB volume is not assumed occupied solid; actual firing origin is not established.

`direct_no_rgb` has exactly the same input points, mask-derived eligibility, adjustable coordinates and bounds, and sets only the RGB coordinate weight to zero. When its point inequalities are already met, it returns source exactly even if RGB-full changes. `half_full` is computed in float32 as source + float32(.5)*(full−source), never refit. No optimizer updates neural parameters.

CPU synthetic checks are necessary implementation checks, not real target/calibration/identity correctness. They include wrong support improving loss but worsening known synthetic full-box error; all actual task effects remain untested until root's separately sealed experiment.
