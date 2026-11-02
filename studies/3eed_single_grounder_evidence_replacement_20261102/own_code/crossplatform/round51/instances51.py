"""R51 CPU point-supported instance hypotheses; no detector/vision-model inputs.

Only raw XYZ, source boxes and a CameraSpec enter this module. Hypotheses are
local connected visible surfaces, not semantic instances or complete object
boundaries. Unsupported, competing and overlapping associations stay UNKNOWN.
No source score, class, caption, GT, external mask or old ownership is accepted.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'round7'))
from camera_projection import project_model_points, raw_to_model_points


@dataclass(frozen=True)
class InstanceConfig:
    minimum_depth_m: float = .1
    minimum_component_points: int = 6
    connectivity_neighbors: int = 8
    radius_nn_multiplier: float = 2.5
    minimum_radius_m: float = .08
    maximum_radius_m: float = .45
    dominant_component_fraction: float = .60
    maximum_runner_fraction: float = .25
    deduplicate_jaccard: float = .70
    deduplicate_each_coverage: float = .75
    maximum_shared_fraction: float = .25
    support_box_tolerance_m: float = 0.
    association_unknown_pseudocount: float = 6.


DEFAULT_CONFIG = InstanceConfig()


def array_sha256(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _validate_config(c):
    if not isinstance(c, InstanceConfig):
        raise TypeError('InstanceConfig required, not untrusted arbitrary options')
    if c.minimum_component_points < 2 or c.connectivity_neighbors < 1:
        raise ValueError('Invalid component configuration')
    if not 0 < c.minimum_radius_m <= c.maximum_radius_m or c.radius_nn_multiplier <= 0:
        raise ValueError('Invalid radius configuration')
    for v in (c.dominant_component_fraction, c.maximum_runner_fraction,
              c.deduplicate_jaccard, c.deduplicate_each_coverage, c.maximum_shared_fraction):
        if not 0 <= v <= 1:
            raise ValueError('Fractions must be in [0,1]')
    if c.minimum_depth_m <= 0 or c.support_box_tolerance_m < 0 or c.association_unknown_pseudocount <= 0:
        raise ValueError('Invalid visibility/support tolerance')


def _pack(parts, width=None, dtype=np.int64):
    offsets = np.r_[0, np.cumsum([len(x) for x in parts])].astype(np.int64)
    shape = (0,) if width is None else (0, width)
    values = np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(shape, dtype)
    return offsets, values


def _local_components(xyz, ids, cfg):
    if len(ids) < cfg.minimum_component_points:
        return [], None
    local = xyz[ids]
    distances, neighbors = cKDTree(local).query(
        local, k=min(len(ids), cfg.connectivity_neighbors + 1), workers=1)
    distances = np.asarray(distances)
    neighbors = np.asarray(neighbors)
    nearest = distances[:, 1]
    positive = nearest[nearest > 1e-8]
    scale = float(np.median(positive)) if len(positive) else cfg.minimum_radius_m
    radius = float(np.clip(cfg.radius_nn_multiplier * scale,
                           cfg.minimum_radius_m, cfg.maximum_radius_m))
    valid = (distances[:, 1:] <= radius) & np.isfinite(distances[:, 1:])
    row = np.broadcast_to(np.arange(len(ids))[:, None], valid.shape)[valid]
    col = neighbors[:, 1:][valid]
    graph = coo_matrix((np.ones(len(row), bool), (row, col)), shape=(len(ids), len(ids))).tocsr()
    _, labels = connected_components(graph, directed=False)
    comps = [np.sort(ids[labels == j]) for j in np.unique(labels)]
    comps = [x for x in comps if len(x) >= cfg.minimum_component_points]
    comps.sort(key=lambda x: (-len(x), int(x[0])))
    return comps, radius


def _compatible(a, b, cfg):
    n = len(np.intersect1d(a, b, assume_unique=True))
    return (n / (len(a) + len(b) - n) >= cfg.deduplicate_jaccard and
            n / len(a) >= cfg.deduplicate_each_coverage and
            n / len(b) >= cfg.deduplicate_each_coverage)


def _visibility(raw_unique, first_raw_ids, camera, image_size_wh, cfg):
    # Independent projection and all-cloud z-buffer; no cached scene/mask input.
    model_double = raw_to_model_points(raw_unique, camera.platform, camera.waymo_view_suffix)
    projection = project_model_points(model_double, camera, image_size_wh,
                                      min_depth=cfg.minimum_depth_m)
    ids = np.flatnonzero(projection['in_image'])
    W, _ = image_size_wh
    pixels = np.floor(projection['uv'][ids]).astype(np.int64)
    keys = pixels[:, 1] * W + pixels[:, 0]
    p = raw_unique[ids]
    order = np.lexsort((first_raw_ids[ids], p[:, 2], p[:, 1], p[:, 0],
                        projection['depth'][ids], keys))
    ids, keys = ids[order], keys[order]
    ids = ids[np.r_[True, keys[1:] != keys[:-1]]] if len(ids) else ids
    return ids, projection


def build_instances(*, raw_points, source_boxes, image_size_wh, camera_spec,
                    config=DEFAULT_CONFIG):
    """Return ``receipt, arrays``. Every point index addresses the original XYZ.

    ``source_boxes`` are original (centerXYZ,sizeXYZ) model-coordinate boxes.
    No box fitting, coordinate change, scores or semantic category is used.
    Sparse crop masks comprise only uniquely owned globally visible pixels.
    They must not be described as complete 2D instance segmentation masks.
    """
    _validate_config(config)
    raw = np.asarray(raw_points)
    boxes = np.asarray(source_boxes)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError('Finite original raw Nx3 XYZ required')
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError('Finite original source Qx6 boxes required')
    W, H = map(int, image_size_wh)
    if W <= 0 or H <= 0:
        raise ValueError('Positive original image dimensions required')
    camera_spec.validated()
    unique, first = np.unique(raw.astype(np.float64), axis=0, return_index=True)
    # Match source XYZ convention separately from physical double projection.
    model = unique.astype(np.float32)
    if camera_spec.platform == 'drone':
        model[:, 2] += np.float32(1.8)
    elif camera_spec.platform == 'waymo':
        model = raw_to_model_points(unique, 'waymo', camera_spec.waymo_view_suffix).astype(np.float32)
    elif camera_spec.platform != 'quad':
        raise ValueError('Unsupported platform')
    vis, projection = _visibility(unique, first, camera_spec, (W, H), config)
    Q = len(boxes)
    support = []
    local_components = []
    radius = np.full(Q, np.nan)
    candidates = []
    for q, box in enumerate(boxes):
        ids = (np.flatnonzero(np.all(np.abs(model.astype(np.float64) - box[:3]) <=
                                    box[3:] / 2 + config.support_box_tolerance_m, axis=1))
               if (box[3:] > 0).all() else np.empty(0, np.int64))
        support.append(ids)
        comps, r = _local_components(model, ids, config)
        local_components.append(comps)
        if r is not None:
            radius[q] = r
        for c, component in enumerate(comps):
            candidates.append((q, c, component))
    # Deterministic complete-link point-support dedup. No source-score order.
    candidates.sort(key=lambda x: (-len(x[2]), int(x[2][0]), x[0], x[1]))
    groups, members, inverted = [], [], {}
    component_group = {}
    ambiguous_dedup = []
    for q, c, component in candidates:
        overlapping = sorted(set(g for point in component for g in inverted.get(int(point), ())))
        compatible = [g for g in overlapping if all(_compatible(component, z, config) for z in members[g])]
        if len(compatible) == 1:
            g = compatible[0]
            members[g].append(component)
        else:
            # Multiple compatible hypotheses are not silently merged transitively.
            g = len(groups)
            groups.append(component)
            members.append([component])
            for point in component:
                inverted.setdefault(int(point), []).append(g)
            if len(compatible) > 1:
                ambiguous_dedup.append([q, c])
        component_group[(q, c)] = g
    I = len(groups)
    point_membership_count = np.zeros(len(unique), np.int32)
    point_owner = np.full(len(unique), -1, np.int64)
    for g, ids in enumerate(groups):
        point_membership_count[ids] += 1
        point_owner[ids] = g
    point_owner[point_membership_count > 1] = -2
    query_instance = np.full(Q, -1, np.int64)
    reasons = []
    fractions = np.zeros((Q, 3), np.float64)  # dominant, runner, shared dominant
    query_component_count = np.array([len(x) for x in local_components], np.int64)
    for q, comps in enumerate(local_components):
        reason = 'UNKNOWN_insufficient_connected_support'
        if not (boxes[q, 3:] > 0).all():
            reason = 'UNKNOWN_nonpositive_source_size'
        elif comps:
            total = len(support[q])
            f0 = len(comps[0]) / total
            f1 = len(comps[1]) / total if len(comps) > 1 else 0.
            shared = float(np.mean(point_membership_count[comps[0]] > 1))
            fractions[q] = [f0, f1, shared]
            if (len(comps) > 1 and len(comps[0]) == len(comps[1])) or f0 < config.dominant_component_fraction or f1 > config.maximum_runner_fraction:
                reason = 'UNKNOWN_multiple_components'
            elif [q, 0] in ambiguous_dedup or shared > config.maximum_shared_fraction:
                reason = 'UNKNOWN_overlapping_instance_hypotheses'
            else:
                query_instance[q] = component_group[(q, 0)]
                reason = 'POINT_SUPPORTED_HYPOTHESIS_not_semantic_identity'
        reasons.append(reason)
    visible_owner = point_owner[vis]
    pixels = np.floor(projection['uv'][vis]).astype(np.int64)
    # All-visible denominator includes points outside every hypothesis and
    # overlapping hypotheses. Six fixed unknown pseudo-points shrink sparse
    # evidence; neither foreground nor accepted queries are renormalized to 1.
    visible_counts = np.zeros((Q, I), np.int64)
    visible_total = np.zeros(Q, np.int64)
    visible_unassigned = np.zeros(Q, np.int64)
    visible_ambiguous = np.zeros(Q, np.int64)
    for q, ids in enumerate(support):
        owners = visible_owner[np.isin(vis, ids, assume_unique=True)]
        visible_total[q] = len(owners)
        visible_unassigned[q] = np.sum(owners == -1)
        visible_ambiguous[q] = np.sum(owners == -2)
        if I:
            visible_counts[q] = np.bincount(owners[owners >= 0], minlength=I)
    subprob_weights = visible_counts / (visible_total[:, None] + config.association_unknown_pseudocount)
    unknown_mass = 1. - subprob_weights.sum(1)
    known = visible_counts.sum(1)
    conditional_weights = np.divide(visible_counts, known[:, None],
                                    out=np.zeros((Q, I), np.float64), where=known[:, None] > 0)
    assert np.all(unknown_mass >= -1e-12)
    assert np.array_equal(visible_counts.sum(1) + visible_unassigned + visible_ambiguous, visible_total)
    instance_points = [first[ids].astype(np.int64) for ids in groups]
    exclusive_points = [first[ids[point_owner[ids] == g]].astype(np.int64) for g, ids in enumerate(groups)]
    vis_positions = [np.flatnonzero(visible_owner == g) for g in range(I)]
    instance_visible = [first[vis[x]].astype(np.int64) for x in vis_positions]
    instance_pixels = [pixels[x] for x in vis_positions]
    rects = np.zeros((I, 4), np.int64)
    centroid_xyz = np.full((I, 3), np.nan)
    centroid_uv = np.full((I, 2), np.nan)
    for g, positions in enumerate(vis_positions):
        if len(positions):
            px = pixels[positions]
            rects[g] = np.r_[px.min(0), px.max(0) + 1]
            centroid_xyz[g] = model[vis[positions]].astype(np.float64).mean(0)
            centroid_uv[g] = projection['uv'][vis[positions]].mean(0)
    arrays = {
        'source_boxes': boxes.copy(), 'query_instance_ids': query_instance,
        # Main propagation preserves conditional known-foreground semantics;
        # subprobability/rho is diagnostic, not a hidden change to the readout.
        'query_instance_weights': conditional_weights,
        'query_instance_conditional_weights': conditional_weights.copy(),
        'query_instance_subprob_weights': subprob_weights,
        'query_unknown_mass': unknown_mass,
        'query_instance_visible_counts': visible_counts,
        'query_allvisible_counts': visible_total,
        'query_visible_unassigned_counts': visible_unassigned,
        'query_visible_ambiguous_counts': visible_ambiguous,
        'query_support_counts': np.array([len(x) for x in support], np.int64),
        'query_component_counts': query_component_count, 'query_component_radius_m': radius,
        'query_dominant_runner_shared_fractions': fractions,
        'visible_raw_point_indices': first[vis].astype(np.int64),
        'visible_pixels_xy': pixels, 'visible_uv': projection['uv'][vis],
        'visible_depth_camera_m': projection['depth'][vis],
        'visible_point_owner_instance_ids': visible_owner,
        'instance_rect_xyxy': rects,
        'instance_visible_centroid_xyz': centroid_xyz,
        'instance_visible_centroid_uv': centroid_uv,
    }
    for prefix, parts, width in (
            ('instance_point', instance_points, None),
            ('instance_exclusive_point', exclusive_points, None),
            ('instance_visible_point', instance_visible, None),
            ('instance_crop_mask_pixel', instance_pixels, 2),
            ('query_support_point', [first[x].astype(np.int64) for x in support], None)):
        offsets, values = _pack(parts, width)
        arrays[prefix + '_offsets'] = offsets
        arrays[prefix + '_indices' if width is None else prefix + '_xy'] = values
    arrays['instance_sparse_mask_values'] = np.ones(len(arrays['instance_crop_mask_pixel_xy']), bool)
    crop_local = [px - rects[g, :2] for g, px in enumerate(instance_pixels)]
    _, arrays['instance_crop_local_mask_pixel_xy'] = _pack(crop_local, 2)
    arrays['instance_crop_shape_hw'] = (rects[:, 2:] - rects[:, :2])[:, ::-1]
    k, e, d, assumed = camera_spec.validated()
    receipt = {
        'schema': 'R51_point_supported_instances_v1', 'config': asdict(config),
        'platform': camera_spec.platform, 'image_size_wh': [W, H],
        'raw_points_sha256': array_sha256(raw), 'source_boxes_sha256': array_sha256(boxes),
        'n_raw_points': len(raw), 'n_unique_raw_points': len(unique), 'n_queries': Q,
        'n_instances': I, 'query_status': reasons,
        'n_mapped_queries': int(np.sum(query_instance >= 0)),
        'n_visible_points': len(vis), 'n_visible_unassigned': int(np.sum(visible_owner == -1)),
        'n_visible_ambiguous': int(np.sum(visible_owner == -2)),
        'projection_diagnostics': projection['diagnostics'],
        'camera_sha256': hashlib.sha256((array_sha256(k) + array_sha256(e) + array_sha256(d) +
                                       camera_spec.platform + str(camera_spec.waymo_view_suffix) +
                                       camera_spec.missing_distortion_policy).encode()).hexdigest(),
        'missing_distortion_assumed_zero': assumed,
        'geometry_uses_no_source_scores_or_classes': True,
        'external_model_outputs_or_ownership_inputs': False,
        'sparse_projected_support_not_complete_object_mask': True,
        'surface_span_not_complete_box_size': True,
        'deduplication_basis': 'complete_link_observed_point_set_overlap_not_box_IoU',
        'query_weight_definition': 'conditional_exclusive_instance_visible_count/known_exclusive_visible_count_or_all_zero',
        'query_subprob_weight_definition': 'exclusive_instance_visible_count/(allvisible_count+fixed_unknown_pseudocount)',
        'subprob_rho_is_diagnostic_not_implicitly_applied_to_main_propagation': True,
        'query_unknown_mass_includes_unassigned_ambiguous_and_pseudocount': True,
        'query_instance_ids_strict_but_weights_are_unthresholded_measured_support': True,
    }
    return receipt, arrays


def dense_support_mask(arrays, instance_id, image_size_wh):
    """Materialize one original-resolution *sparse support* bool mask on demand."""
    W, H = map(int, image_size_wh)
    offsets = arrays['instance_crop_mask_pixel_offsets']
    if not 0 <= instance_id < len(offsets) - 1:
        raise IndexError('Invalid hypothesis index')
    xy = arrays['instance_crop_mask_pixel_xy'][offsets[instance_id]:offsets[instance_id + 1]]
    mask = np.zeros((H, W), bool)
    mask[xy[:, 1], xy[:, 0]] = True
    return mask


def write_instance_bundle(directory, receipt, arrays):
    """Explicit caller-owned I/O; no implicit cache discovery or media reading."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'instances.npz', **arrays)
    (out / 'instances.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
