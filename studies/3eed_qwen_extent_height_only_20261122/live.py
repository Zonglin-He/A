"""Use the frozen Qwen readout followed by ray-equivalent extent clipping."""

from pathlib import Path
import sys

import numpy as np


STUDIES = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(STUDIES / "3eed_qwen_cray_identifiability_audit_20261121"),
    str(STUDIES / "3eed_qwen_cafe_exclusion_audit_20261121"),
]
from trace_pairs import first_depth
from cray_simple import (
    FOUR_SCALES, MIN_RETURNS, choose_pose, choose_region, geometry,
    predict as frozen_predict, vehicle_reply,
)
from extent_clip import clip_extent, protected_points


def incumbent(frame, replies, *, collect_audit=False):
    """Reproduce the locked branch and return its geometry for clipping."""
    raw = replies["3d"]["value"]
    pixels = replies["2d"]["value"]
    if raw is None or pixels is None or not vehicle_reply(replies["2d"]):
        return frozen_predict(frame, replies), None
    spec, size, transform = geometry.camera(frame)
    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(pixels) * np.tile(size, 2) / 1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    if len(central) < MIN_RETURNS:
        return frozen_predict(frame, replies), None
    sample, _ = geometry.fit_sample(frame, spec, size)
    if len(sample) < 9:
        return frozen_predict(frame, replies), None

    ids, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    split = geometry.reserve(cells)
    in_window = geometry.inside(uv[ids], window)
    fit = ids[split == 0]
    selectors = ids[(split == 1) & in_window]
    audit = ids[(split == 2) & in_window] if collect_audit else np.empty(0, dtype=int)
    model_camera = geometry.model_to_camera_matrix(spec)
    inverse = np.linalg.inv(model_camera)
    xyz = geometry.rigid_xyz(returns, inverse)
    terrain = geometry.regions.terrain(sample)
    support, associated = choose_region(
        sample, terrain, FOUR_SCALES, xyz[np.r_[fit, central]],
        in_window[split == 0], xyz[np.r_[selectors, audit]],
    )
    if support is None or associated[:len(selectors)].sum() < MIN_RETURNS:
        return frozen_predict(frame, replies), None

    selection = returns[selectors[associated[:len(selectors)]]]
    audit = returns[audit[associated[len(selectors):]]]
    raw = np.asarray(raw, dtype=float)
    upright, rotation = geometry.to_sensor(raw, transform)
    yaw = upright[6]
    if frame["target"] == "waymo":
        yaw += np.deg2rad(geometry.WAYMO_ANGLES[frame["suffix"]])
    box, heading = choose_pose(support, terrain, upright[3:6], yaw, inverse[:3, 3], selection, transform, frame)
    fit = geometry.rigid_xyz(support, np.linalg.inv(transform) @ model_camera)
    prediction = dict(box=box.tolist(), route="C_ray", applicable=True, readout="C_ray",
                      selection_rays=len(selection), heading=heading)
    return prediction, (box, raw[3:6], rotation, fit, selection, audit, transform)


def predict(frame, replies, *, height_only=True):
    """Return the height-only prediction by default, or the equivalent full clip."""
    prediction, context = incumbent(frame, replies)
    if context is None:
        return dict(prediction, extent_corrected=False)
    box, dimensions, rotation, fit, selection, _, transform = context
    depth = first_depth(box, selection, transform)
    witnesses = protected_points(box, fit, selection, depth, transform)
    result = clip_extent(box, dimensions, rotation, *witnesses, height_only=height_only)
    return dict(prediction, box=result.tolist(), extent_corrected=result is not box)
