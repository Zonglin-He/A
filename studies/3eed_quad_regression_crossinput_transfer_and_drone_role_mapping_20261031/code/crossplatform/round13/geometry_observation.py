"""R13 fixed-query positive-observation inequalities, pure CPU arrays.

No task labels, file access, model, candidate reselection, point/face equality,
nearest-face assignment or PCA. An enclosing AABB is not a solid occupancy model;
unknown sensor origins/free-space therefore supply no optimization term here.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'round7'))
from camera_projection import project_model_points

ARM_NAMES = ('source', 'direct_full', 'direct_no_rgb', 'half_full')


@dataclass(frozen=True)
class ObservationConfig:
    min_unique_support: int = 8
    point_containment_tolerance_m: float = .03
    point_sigma_m: float = .05
    huber_delta: float = 1.
    center_bound_fraction: float = .25
    center_bound_cap_m: float = .5
    max_size_expansion: float = 1.2
    center_prior_std_fraction: float = .15
    center_prior_std_min_m: float = .05
    center_prior_std_max_m: float = .35
    log_size_prior_std: float = .1
    prior_weight: float = 1.
    point_weight: float = 1.
    rgb_weight: float = 1.
    min_corner_depth_m: float = .1
    rgb_pixel_sigma_min: float = 3.
    rgb_pixel_sigma_bbox_fraction: float = .05
    max_mask_boundary_samples: int = 128
    solver_max_iterations: int = 64
    solver_ftol: float = 1e-10
    solver_gtol: float = 1e-6
    normalized_finite_difference_eps: float = 1e-5


DEFAULT_CONFIG = ObservationConfig()


def config_sha256(config=DEFAULT_CONFIG):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_support(points_support_model, physical_raw_ids=None):
    """Unique physical observations; coincident float32 XYZ is counted only once.

    IDs are an optional consistency certificate, never a label. Different IDs at
    exactly coincident float32 XYZ do not multiply observation evidence.
    """
    p = np.asarray(points_support_model, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Finite Nx3 model-coordinate foreground support required')
    physical_n = len(p)
    if physical_raw_ids is not None:
        ids = np.asarray(physical_raw_ids)
        if ids.shape != (len(p),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('One integer physical raw ID per support point required')
        unique_ids, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
        for j in np.flatnonzero(counts > 1):
            group = p[inverse == j]
            if not np.array_equal(group, np.broadcast_to(group[0], group.shape)):
                raise ValueError('Repeated physical raw ID has inconsistent XYZ')
        physical_n = len(unique_ids)
    unique = np.unique(p, axis=0)
    digest = hashlib.sha256(np.asarray(unique, dtype='<f4').tobytes()).hexdigest()
    return unique.astype(np.float64), digest, {
        'input_rows': len(p), 'unique_physical_ids_if_supplied': physical_n,
        'physical_ids_supplied': physical_raw_ids is not None,
        'unique_float32_XYZ': len(unique), 'support_sha256_basis': 'sorted unique float32 XYZ little-endian',
        'coincident_returns_do_not_multiply_evidence': True,
    }


def _corners(box):
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    return box[:3] + signs * box[3:] / 2


def _huber(values, delta):
    a = np.abs(values)
    return np.where(a <= delta, .5 * a * a, delta * (a - .5 * delta))


def _boundary(mask, maximum):
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    yx = np.argwhere(boundary)
    if not len(yx):
        return np.empty((0, 2), np.float64)
    index = np.linspace(0, len(yx) - 1, min(len(yx), maximum), dtype=np.int64)
    return yx[index, ::-1].astype(np.float64) + .5


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def generate_observation_targets(*, points_support_model, mask_bool, camera_spec,
                                 source6, q0, segmentation_valid=True,
                                 physical_raw_ids=None, config=DEFAULT_CONFIG):
    """Return direct boxes; caller separately seals and evaluates them.

    Input support/mask association is an upstream hypothesis, not verified target
    identity. Full/off-RGB share support, gate, active coordinates and bounds.
    Center and size are solved in float64 and delivered float32. Size can only
    expand: partial positive observations provide no amodal upper bound.
    """
    cfg = config
    source = np.asarray(source6, np.float32)
    mask = np.asarray(mask_bool)
    if source.shape != (6,) or not np.isfinite(source).all():
        raise ValueError('Finite source6 required; no source repair allowed')
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Boolean full-resolution independent mask required')
    if not isinstance(q0, (int, np.integer)) or int(q0) < 0:
        raise ValueError('Nonnegative fixed source query index required')
    if not isinstance(segmentation_valid, (bool, np.bool_)):
        raise ValueError('Explicit boolean segmentation validity required')
    points, support_sha, dedup = canonical_support(points_support_model, physical_raw_ids)
    mask_sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    result = {
        'arm_names': np.asarray(ARM_NAMES),
        'arm_query_index': np.full(4, int(q0), np.int64),
        'arm_boxes_raw6': np.tile(source, (4, 1)),
        'arm_status': ['source', 'fallback', 'fallback', 'fallback'],
        'active_mask6': np.zeros((4, 6), bool),
        'acceptance_mask6': np.zeros((4, 6), bool),
        'actual_changed_axes': np.zeros((4, 6), bool),
        'point_support_sha256': np.asarray([support_sha] * 4),
        'mask_sha256_by_arm': np.asarray([mask_sha] * 4),
        'provenance': np.full((4, 6), 'source_prior', dtype='U192'),
        'diagnostics': {
            'config': asdict(cfg), 'config_sha256': config_sha256(cfg), 'dedup': dedup,
            'fixed_q0': int(q0), 'gate_passed': False,
            'same_support_gate_active_bounds_full_noRGB': True,
            'free_space': 'NA: firing origin/solid occupancy not established; not used',
            'target_identity': 'upstream association hypothesis; not verified by this module',
            'center_size_identifiability': 'one-sided lower bounds plus source minimum-change prior; not observed amodal center/size',
            'normalization': 'center delta/source size; log size/source size; center prior in meters; point sigma .05m; RGB source-fixed pixel sigma',
            'size_direction': 'source or expansion only; partial foreground supplies no shrinking or amodal upper-bound evidence',
        },
    }

    def finish():
        result['actual_changed_axes'] = result['arm_boxes_raw6'] != source[None]
        result['diagnostics'] = _json_safe(result['diagnostics'])
        json.dumps(result['diagnostics'], allow_nan=False)
        assert np.isfinite(result['arm_boxes_raw6']).all()
        assert np.array_equal(result['arm_boxes_raw6'][0], source)
        return result

    def fallback(reason):
        result['diagnostics']['gate_reason'] = reason
        result['arm_status'][1:] = ['source_fallback:' + reason] * 3
        result['provenance'][1:] = 'fallback:' + reason
        return finish()

    if np.any(source[3:] <= 0):
        return fallback('nonpositive_source_size')
    if not segmentation_valid or not mask.any():
        return fallback('invalid_or_empty_independent_mask')
    if len(points) < cfg.min_unique_support:
        return fallback('insufficient_unique_foreground_support')
    H, W = mask.shape
    source64 = source.astype(np.float64)
    try:
        projection = project_model_points(_corners(source64), camera_spec, (W, H))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return fallback('camera_contract:' + str(exc))
    if np.any(projection['depth'] <= cfg.min_corner_depth_m) or not np.isfinite(projection['uv']).all():
        return fallback('source_box_projection_not_fully_in_front')
    fg = _boundary(mask, cfg.max_mask_boundary_samples)
    pixel_sigma = max(cfg.rgb_pixel_sigma_min,
                      cfg.rgb_pixel_sigma_bbox_fraction * float(np.ptp(projection['uv'], axis=0).min()))
    cstd = np.clip(cfg.center_prior_std_fraction * source64[3:],
                   cfg.center_prior_std_min_m, cfg.center_prior_std_max_m)
    c_bound = np.minimum(cfg.center_bound_fraction * source64[3:], cfg.center_bound_cap_m)
    bounds = [(-c_bound[j] / source64[3+j], c_bound[j] / source64[3+j]) for j in range(3)]
    bounds += [(0., float(np.log(cfg.max_size_expansion)))] * 3

    def decode(z):
        box = source64.copy()
        box[:3] += z[:3] * source64[3:]
        box[3:] *= np.exp(z[3:])
        return box

    def terms(box):
        point_outside = np.maximum(np.abs(points - box[:3]) - box[3:] / 2 - cfg.point_containment_tolerance_m, 0.)
        point_loss = float(_huber(point_outside / cfg.point_sigma_m, cfg.huber_delta).mean())
        prior = float(np.sum(((box[:3] - source64[:3]) / cstd) ** 2)
                      + np.sum((np.log(box[3:] / source64[3:]) / cfg.log_size_prior_std) ** 2))
        projected = project_model_points(_corners(box), camera_spec, (W, H))
        valid = bool(np.all(projected['depth'] > cfg.min_corner_depth_m)
                     and np.isfinite(projected['uv']).all())
        outside = np.empty(0)
        if valid:
            try:
                equations = ConvexHull(projected['uv']).equations
                outside = np.maximum((fg @ equations[:, :2].T + equations[:, 2]).max(axis=1), 0.)
            except QhullError:
                valid = False
        rgb_loss = float(_huber(outside / pixel_sigma, cfg.huber_delta).mean()) if valid else 1e6
        return {
            'prior': prior, 'point_containment': point_loss, 'RGB_positive_containment': rgb_loss,
            'point_violation_fraction': float(np.any(point_outside > 0, axis=1).mean()),
            'point_violation_axis_fraction': (point_outside > 0).mean(axis=0).tolist(),
            'point_violation_mean_m': float(point_outside.mean()),
            'point_violation_max_m': float(point_outside.max()),
            'point_violation_max_axis_m': point_outside.max(axis=0).tolist(),
            'RGB_boundary_violation_fraction': float((outside > 1e-8).mean()) if valid else None,
            'RGB_boundary_outside_mean_px': float(outside.mean()) if valid else None,
            'projection_valid': valid,
        }

    def total(t, use_rgb):
        return cfg.prior_weight*t['prior'] + cfg.point_weight*t['point_containment'] + (cfg.rgb_weight*t['RGB_positive_containment'] if use_rgb else 0.)

    initial = terms(source64)
    if not initial['projection_valid']:
        return fallback('degenerate_source_projection_hull')
    gate_reason = 'valid_unique_foreground_positive_observation'
    result['diagnostics'].update(gate_passed=True, gate_reason=gate_reason,
        initial_terms=initial, pixel_sigma=pixel_sigma, normalized_bounds=bounds,
        prior_center_std_m=cstd.tolist(), RGB_boundary_samples=len(fg),
        point_axis_lower_bound_violation=np.asarray(initial['point_violation_max_axis_m']) > 0,
        projection_lower_bound_violation=initial['RGB_positive_containment'] > 1e-16)
    any_violation = initial['point_containment'] > 0 or initial['RGB_positive_containment'] > 1e-16
    result['active_mask6'][1:] = any_violation
    result['acceptance_mask6'][1:] = any_violation
    result['diagnostics']['active_definition'] = 'all six prior-regularized coordinates share fixed bounds if any source positive inequality violated; not six independently observed parameters'

    def solve(use_rgb):
        if total(initial, use_rgb) <= 1e-16:
            return source.copy(), {'status': 'source_already_satisfies_positive_constraints', 'success': True,
                'iterations': 0, 'initial_terms': initial, 'final_terms': initial,
                'initial_total': total(initial, use_rgb), 'final_total': total(initial, use_rgb),
                'solution_float64': source64.tolist(), 'finite_nonconverged_retained': False}
        fun = lambda z: total(terms(decode(z)), use_rgb)
        sol = minimize(fun, np.zeros(6), method='L-BFGS-B', bounds=bounds,
            options={'maxiter': cfg.solver_max_iterations, 'ftol': cfg.solver_ftol,
                     'gtol': cfg.solver_gtol, 'eps': cfg.normalized_finite_difference_eps})
        box64 = decode(sol.x) if np.isfinite(sol.x).all() else source64.copy()
        final = terms(box64)
        invalid = (not np.isfinite(sol.x).all() or not np.isfinite(box64).all()
                   or not np.isfinite(total(final, use_rgb))
                   or (use_rgb and not final['projection_valid']))
        status = ('numerical_invalid_source_fallback' if invalid else
                  'converged_bounded_result' if sol.success else 'finite_nonconverged_bounded_result')
        delivered = source.copy() if invalid else box64.astype(np.float32)
        delivered_terms = terms(delivered.astype(np.float64))
        return delivered, {'status': status, 'success': bool(sol.success), 'iterations': int(sol.nit),
            'message': str(sol.message), 'numerical_invalid': bool(invalid),
            'finite_nonconverged_retained': bool(not invalid and not sol.success),
            'initial_terms': initial, 'attempted_final_terms': final, 'final_terms': delivered_terms,
            'initial_total': total(initial, use_rgb), 'final_total': total(delivered_terms, use_rgb),
            'solution_float64': box64.tolist(), 'normalized_solution': sol.x.tolist()}

    full, full_diag = solve(True)
    off, off_diag = solve(False)
    result['arm_boxes_raw6'][1] = full
    result['arm_boxes_raw6'][2] = off
    result['arm_boxes_raw6'][3] = source + np.float32(.5) * (full - source)
    result['arm_status'][1:] = [full_diag['status'], off_diag['status'], 'fixed_half_of_direct_full']
    result['diagnostics']['solver'] = {'direct_full': full_diag, 'direct_no_rgb': off_diag}
    for a in (1, 2, 3):
        result['provenance'][a, :3] = ('point_containment_lower_bound_plus_source_prior'
            if initial['point_containment'] > 0 else 'source_prior_unobserved_by_points')
        result['provenance'][a, 3:] = 'source_size_lower_prior;positive_observations_allow_expansion_not_amodal_recovery'
        if a in (1, 3):
            result['provenance'][a] = np.array([x + ';one_sided_RGB_containment' for x in result['provenance'][a]])
    return finish()
