"""Frozen 2D-gated dual-pose readout with a four-scale H fallback."""
from pathlib import Path
import sys

import numpy as np


STUDIES = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(STUDIES/"3eed_pose_consistent_visible_surface_grounding_20261121"),
    str(STUDIES/"3eed_qwen_cray_category_balance_20261121"),
]
import surface_grounding as geometry
from cray import vehicle_reply
from geometry import axial, front_offset, pixel_rays


FOUR_SCALES = geometry.regions.FOUR_LEVEL_INDICES
TWO_SCALES = (FOUR_SCALES[0], FOUR_SCALES[-1])
MIN_RETURNS = 5


def region_catalog(sample, terrain, scales):
    foreground = sample[:, 2] - sample[:, :2] @ terrain[:2] - terrain[2] > .20
    radii, labels = geometry.regions.levels(sample, foreground, indices=scales)
    assigned = np.full_like(labels, -1)
    ids, births = {}, []
    # A component that persists at multiple nested levels keeps its first radius.
    for level, line in enumerate(labels):
        active = np.flatnonzero(line >= 0)
        if not len(active):
            continue
        active = active[np.argsort(line[active], kind="stable")]
        for group in np.split(active, np.flatnonzero(np.diff(line[active])) + 1):
            key = int(group.min()), len(group)
            if key not in ids:
                ids[key] = len(births)
                births.append(level)
            assigned[level, group] = ids[key]
    return radii, assigned, np.asarray(births, dtype=int)


def choose_region(sample, terrain, scales, fit_returns, in_window, selection=None):
    radii, assigned, births = region_catalog(sample, terrain, scales)
    if not len(births) or not in_window.any():
        return None, None
    mapped = geometry.transfer(sample, fit_returns, radii, assigned, births)
    scene, central = mapped[:, :len(in_window)], mapped[:, len(in_window):]
    window = scene[:, in_window]
    exposure = np.bincount(scene[scene >= 0], minlength=len(births))
    overlap = np.bincount(window[window >= 0], minlength=len(births))
    center = np.bincount(central[central >= 0], minlength=len(births))
    candidates = np.flatnonzero((center >= MIN_RETURNS) & (overlap > 0))
    if not len(candidates):
        return None, None
    strength = overlap[candidates].astype(float)**2 / exposure[candidates]
    winner = candidates[np.argmax(strength)]
    level = births[winner]
    region = sample[assigned[level] == winner]
    if selection is None:
        return region, None
    holding = geometry.transfer(sample, selection, radii, assigned, births)
    return region, holding[level] == winner


def horizontal_axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, s), (-s, c)))


def anchor(lower, upper, dimensions, yaw, observer, terrain=None):
    """Anchor the near face on each horizontal axis and the vertical bottom."""
    axes = horizontal_axes(yaw)
    eye = observer[:2] @ axes.T
    face = (eye < lower[:2]).astype(float) - (eye > upper[:2]).astype(float)
    span = upper[:2] - lower[:2]
    middle = (lower[:2] + upper[:2]) / 2
    center = (middle + face * (dimensions[:2] - span) / 2) @ axes
    bottom = lower[2]
    if terrain is not None and np.isfinite(terrain).all():
        bottom = min(bottom, float(center @ terrain[:2] + terrain[2]))
    return np.r_[center, bottom + dimensions[2]/2, dimensions, yaw]


def completed_dimensions(spans, prior):
    aligned = prior.copy()
    aligned[:2] = np.sort(prior[:2])[::-1] if spans[0] >= spans[1] else np.sort(prior[:2])
    return spans + np.maximum(aligned - spans, 0.)


def complete_h(points, terrain, prior):
    measured = geometry.regions.measured_box(points, terrain)
    yaw = measured[6]
    middle = np.r_[measured[:2] @ horizontal_axes(yaw).T, measured[2]]
    span = measured[3:6]
    return anchor(middle-span/2, middle+span/2, completed_dimensions(span, prior), yaw, np.zeros(3))


def pose_box(points, terrain, prior, yaw, eye):
    local = np.c_[points[:, :2] @ horizontal_axes(yaw).T, points[:, 2]]
    return anchor(local.min(0), local.max(0), prior, yaw, eye, terrain)


def ray_score(box, returns, transform, cap):
    hit, residual = geometry.ray_residual(box, returns, transform)
    return float(np.where(hit, np.minimum(residual, cap), cap).sum())


def choose_pose(points, terrain, dimensions, qwen_yaw, eye, returns, transform, frame):
    measured = geometry.regions.measured_box(points, terrain)
    spans = completed_dimensions(measured[3:6], dimensions)
    pca_yaw = axial(measured[6] + (np.pi/2 if spans[1] > spans[0] else 0.))
    candidates = [
        geometry.sensor_frame_box(pose_box(points, terrain, dimensions, yaw, eye), frame["target"], frame["suffix"])
        for yaw in (pca_yaw, qwen_yaw)
    ]
    cap = max(.1, .5*min(dimensions[:2]))
    costs = [ray_score(b, returns, transform, cap) for b in candidates]
    # Preserve the tie rule of the previous normalized mean loss.
    qwen = costs[1] < costs[0] - 1.e-12*len(returns)*cap
    return candidates[int(qwen)], ("qwen" if qwen else "pca")


def a_box(raw, window, central, returns, spec, transform):
    box = raw.copy()
    guide = pixel_rays((window[:2] + window[2:])/2, spec)[0]
    depth = raw[2]
    if len(central) >= MIN_RETURNS:
        depth = np.median(returns[central, 2]) + front_offset(raw, guide)
    box[:3] = guide*depth
    return geometry.to_sensor(box, transform)[0]


def predict(frame, replies, *, vehicle_scales=FOUR_SCALES):
    """Return one upright sensor-frame box and record the route that produced it."""
    if vehicle_scales not in (FOUR_SCALES, TWO_SCALES):
        raise ValueError("Use the locked four-scale readout or its fine/coarse comparison.")
    raw = replies["3d"]["value"]
    pixel = replies["2d"]["value"]
    result = dict(box=None, route="H", applicable=False, readout="missing_3d",
                  selection_rays=0, heading=None)
    if raw is None:
        return result

    raw = np.asarray(raw, float)
    spec, size, transform = geometry.camera(frame)
    original = geometry.to_sensor(raw, transform)[0]
    if pixel is None:
        return dict(result, box=original.tolist(), readout="A")

    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(pixel)*np.tile(size, 2)/1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    if len(central) < MIN_RETURNS:
        return dict(result, box=a_box(raw, window, central, returns, spec, transform).tolist(), readout="A")

    indices, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    parts = geometry.reserve(cells)
    fit_ids = indices[parts == 0]
    in_window = geometry.inside(uv[fit_ids], window)
    model_inverse = np.linalg.inv(geometry.model_to_camera_matrix(spec))
    xyz = geometry.rigid_xyz(returns, model_inverse)
    fit_returns = xyz[np.r_[fit_ids, central]]
    dimensions = original[3:6]

    if vehicle_reply(replies["2d"]):
        fit, _ = geometry.fit_sample(frame, spec, size)
        if len(fit) >= 9:
            terrain = geometry.regions.terrain(fit)
            selector = indices[(parts == 1) & geometry.inside(uv[indices], window)]
            points, support = choose_region(fit, terrain, vehicle_scales, fit_returns, in_window, xyz[selector])
            if points is not None:
                held = returns[selector[support]]
                result["selection_rays"] = len(held)
                if len(held) >= MIN_RETURNS:
                    yaw = original[6]
                    if frame["target"] == "waymo":
                        yaw += np.deg2rad(geometry.WAYMO_ANGLES[frame["suffix"]])
                    box, heading = choose_pose(points, terrain, dimensions, yaw,
                                               model_inverse[:3, 3], held, transform, frame)
                    return dict(result, box=box.tolist(), route="C_ray", readout="C_ray",
                                applicable=True, heading=heading)

    sample = geometry.model_sample(frame)
    terrain = geometry.regions.terrain(sample)
    points, _ = choose_region(sample, terrain, FOUR_SCALES, fit_returns, in_window)
    if points is None:
        return dict(result, box=a_box(raw, window, central, returns, spec, transform).tolist(), readout="A")
    box = geometry.sensor_frame_box(complete_h(points, terrain, dimensions), frame["target"], frame["suffix"])
    return dict(result, box=box.tolist(), readout="H")
