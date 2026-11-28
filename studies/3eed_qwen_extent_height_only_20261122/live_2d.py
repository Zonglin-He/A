"""Rank the locked C_ray pair with the first Qwen rectangle."""
from time import perf_counter

import numpy as np

# Import the historical entry first. It pins the shared geometry modules.
import live
from cray_simple import axial, completed_dimensions, pose_box, ray_score
from geometry import extent_corners, sensor_pixels, z_rotation

geometry = live.geometry


def visible_rectangle(rect, size):
    clipped = np.clip(np.asarray(rect, dtype=float), 0, np.tile(size, 2))
    if not np.isfinite(clipped).all() or np.any(clipped[2:] - clipped[:2] < 2):
        return None
    return clipped


def projection_overlap(box, target, spec, transform, size):
    corners = extent_corners(box[:3], box[3:6], z_rotation(np.rad2deg(box[6])))
    pixels = sensor_pixels(geometry.rigid_xyz(corners, transform), spec)
    if pixels is None:
        return None
    projection = visible_rectangle(np.r_[pixels.min(0), pixels.max(0)], size)
    if projection is None:
        return None
    overlap = np.maximum(0., np.minimum(target[2:], projection[2:]) - np.maximum(target[:2], projection[:2]))
    intersection = np.prod(overlap)
    union = np.prod(target[2:] - target[:2]) + np.prod(projection[2:] - projection[:2]) - intersection
    return float(intersection / union)


def prepare_pair(frame, window, spec, size, uv, returns, transform, raw, central):
    """Construct the two locked poses with the same selected points and rays."""
    sample, _ = geometry.fit_sample(frame, spec, size)
    if len(sample) < 9:
        return None
    ids, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    split = geometry.reserve(cells)
    in_window = geometry.inside(uv[ids], window)
    selectors = ids[(split == 1) & in_window]
    model_camera = geometry.model_to_camera_matrix(spec)
    inverse = np.linalg.inv(model_camera)
    xyz = geometry.rigid_xyz(returns, inverse)
    terrain = geometry.regions.terrain(sample)
    support, associated = live.choose_region(
        sample, terrain, live.FOUR_SCALES, xyz[np.r_[ids[split == 0], central]],
        in_window[split == 0], xyz[selectors],
    )
    if support is None or associated.sum() < live.MIN_RETURNS:
        return None
    selection = returns[selectors[associated]]
    raw = np.asarray(raw, dtype=float)
    upright, rotation = geometry.to_sensor(raw, transform)
    dimensions, yaw = upright[3:6], upright[6]
    if frame["target"] == "waymo":
        yaw += np.deg2rad(geometry.WAYMO_ANGLES[frame["suffix"]])
    measured = geometry.regions.measured_box(support, terrain)
    spans = completed_dimensions(measured[3:6], dimensions)
    pca = axial(measured[6] + (np.pi/2 if spans[1] > spans[0] else 0.))
    pair = [
        geometry.sensor_frame_box(pose_box(support, terrain, dimensions, angle, inverse[:3, 3]),
                                  frame["target"], frame["suffix"])
        for angle in (pca, yaw)
    ]
    cap = max(.1, .5 * min(dimensions[:2]))
    pca_cost, qwen_cost = (ray_score(box, selection, transform, cap) for box in pair)
    ray_winner = int(qwen_cost < pca_cost - 1.e-12 * len(selection) * cap)
    fit = geometry.rigid_xyz(support, np.linalg.inv(transform) @ model_camera)
    return pair, ray_winner, fit, selection, raw[3:6], rotation


def height_fit(box, fit, selection, dimensions, rotation, transform):
    depth = live.first_depth(box, selection, transform)
    witnesses = live.protected_points(box, fit, selection, depth, transform)
    result = live.clip_extent(box, dimensions, rotation, *witnesses, height_only=True)
    return result, bool(result is not box)


def paired(frame, replies, *, diagnostics=False):
    """Return the old and promoted height-only predictions from one physical fit."""
    raw = replies["3d"]["value"]
    pixels = replies["2d"]["value"]

    def fallback():
        out = dict(live.frozen_predict(frame, replies), extent_corrected=False)
        return out, out

    if raw is None or pixels is None or not live.vehicle_reply(replies["2d"]):
        return fallback()
    spec, size, transform = geometry.camera(frame)
    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(pixels, dtype=float) * np.tile(size, 2) / 1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    if len(central) < live.MIN_RETURNS:
        return fallback()
    selected = prepare_pair(frame, window, spec, size, uv, returns, transform, raw, central)
    if selected is None:
        return fallback()
    pair, ray_winner, fit, selection, dims, rotation = selected
    old_box, old_clip = height_fit(pair[ray_winner], fit, selection, dims, rotation, transform)
    names = ("pca", "qwen")
    old = dict(box=old_box.tolist(), route="C_ray", applicable=True, readout="C_ray",
               selection_rays=len(selection), heading=names[ray_winner], extent_corrected=old_clip)
    started = perf_counter() if diagnostics else None
    alternate = 1 - ray_winner
    other_box, other_clip = height_fit(pair[alternate], fit, selection, dims, rotation, transform)
    target = visible_rectangle(window, size)
    scores = (None, None) if target is None else tuple(
        projection_overlap(box, target, spec, transform, size) for box in (old_box, other_box))
    use_alt = scores[1] is not None and scores[1] > (scores[0] if scores[0] is not None else 0.)
    new = dict(old, box=other_box.tolist(), heading=names[alternate], extent_corrected=other_clip) if use_alt else dict(old)
    if diagnostics:
        new["candidate_details"] = dict(old_box=old_box.tolist(), other_box=other_box.tolist(),
            untrimmed_winner=pair[alternate if use_alt else ray_winner].tolist(), overlap=list(scores),
            switched=use_alt, extra_geometry_seconds=perf_counter() - started)
    return old, new


def predict(frame, replies):
    """Predict the registered height-only method with the first 2D pose selection."""
    return paired(frame, replies)[1]
