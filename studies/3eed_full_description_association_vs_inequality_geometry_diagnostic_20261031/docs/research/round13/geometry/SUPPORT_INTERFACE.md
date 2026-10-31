# Fixed-q0 independent mask support — R13 B only

```python
from crossplatform.round13.support import extract_fixed_q0_support
support = extract_fixed_q0_support(
    frame={
        'frame_alias': input_alias,
        'rgb': original_uint8_rgb, 'raw_points': raw_xyz,
        'camera': audited_R7_CameraSpec,
        'instances': {'boxes_xyxy': boxes, 'class_ids': coco_ids,
                      'scores': frozen_scores, 'masks_bool': fullres_bool_masks,
                      'instance_indices': native_ids},
    },
    source6=source_pool[q0], q0=int(q0), caption=literal_caption,
)
g = generate_observation_targets(
    points_support_model=support['points_support_model'],
    mask_bool=support['mask_bool'], camera_spec=camera,
    source6=source_pool[q0], q0=int(q0),
    segmentation_valid=support['segmentation_valid'],
    physical_raw_ids=support['physical_raw_ids'],
)
```

The strict frame keys and instance keys above are exhaustive. Annotations, GT, metadata mappings or mask-index overrides are rejected. This helper reads no files, model features, pixels as learned descriptors, task labels or GT. RGB only supplies original shape; independent masks are caller-sealed predictions.

## Association scope and matched old conventions

First existing lexicon noun in literal caption selects a COCO category family exactly as R12: motor vehicle `{3,6,8}`, person `{1}`, bicycle `{2}`; composite cyclist/biker/trailer unsupported. No fallback to later relation-object nouns or task class. This is B's fixed-source hypothesis check, **not A's full-sentence identity resolution**. Relationship words do not disambiguate two same-family masks. A unique wrong mask can still pass; explicit negative synthetic fixture preserves that failure.

Exactly R12 eligibility: independent detector box IoU with clipped projected source AABB >=.1, and >=3 visible unique raw XYZ points inside both mask and padded source ROI. ROI margin is min(.25*source size+.1m,.75m), inherited from R10. Zero/multiple eligible masks are NA. Scores do not rank candidates or break ties; source q0 never changes. The geometry module separately requires >=8 unique float32 model XYZ.

Visibility is whole-cloud unique raw XYZ, calibrated R7 projection, continuous FOV/depth>.1, floor pixels, one nearest-depth return per pixel with XYZ tie. Z-buffering precedes mask selection, so excluding a foreground label does not reveal a background return. Visibility uses R12 double raw projection. Geometry coordinates use the original R12 float32 conversion: Drone in-place z+=float32(1.8), Quad unchanged, Waymo declared suffix rotation. This preserves prior coordinate behavior rather than silently changing the precision bridge.

## Output contract

Always: `q0`, `status`, `segmentation_valid`, `selected_mask_index` (mask array position or None), `selected_instance_id` (native instance index or None), `mask_bool` (HxW; all zero on NA), `points_support_model` float32[N,3], `points_support_raw` float64[N,3], `physical_raw_ids` int64[N], `support_raw_ids_before_model_XYZ_merge` int64[M], `visible_raw_ids` int64[V], `diagnostics` JSON-safe.

`physical_raw_ids` are **original raw input array row IDs**, not positions in a model-unique cloud. Raw XYZ duplicate representatives use the first original row. Canonical model float32 collisions use the minimum original ID representative. Exact replay is asserted as both `raw[physical_raw_ids]==points_support_raw` and `model_points(points_support_raw)==points_support_model`. All pre-merge raw IDs and collision counts remain available for trace/audit. A permutation changes row IDs but preserves physical selected model XYZ.

Diagnostics preserve all same-family candidate tests, eligible masks, noun/family scope, source projected box, ROI margins, whole-cloud visibility counts, original-ID/rawXYZ/modelXYZ/mask hashes and coordinate-collision counts. They make no literal identity, amodal box or task-quality claim.

## Synthetic receipt

`SUPPORT_CPU_FIXTURE.json` verifies 13 checks including occluded-background and outside-neighbor isolation, exact original-ID replay, same-family ambiguity NA, unsupported first noun, incompatible classes, strict GT-field/override rejection, R12 visibility XYZ bitexact bridge, float32 Drone shift, continuous-FOV edge cases, and a wrong unique mask that passes q0 geometric compatibility. These are implementation checks, not actual dataset outcomes.
