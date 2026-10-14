"""CPU-only sparse optical-flow tracklets from the second observed box.

The module depends only on NumPy and OpenCV. It accepts grayscale source
frames and caller-supplied source dimensions; it does not load project data,
models, labels, queries, or media from disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RECEIPT = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1/CPU_TRACKLET_CONTRACTS.json"
SCHEMA = "stvg-motion-tracklet-lk-v1"

# Locked OpenCV pyramidal Lucas-Kanade settings.
MAX_GRAY_LONG_EDGE = 640
MAX_CORNERS = 80
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 5.0
WINDOW_SIZE = (21, 21)
PYRAMID_MAX_LEVEL = 3
ITERATIONS = 30
EPSILON = 0.01
MAX_FORWARD_BACKWARD_ERROR_PX = 1.5
MIN_TRACKED_CORNERS = 4
MIN_RETAINED_RATIO = 0.5


def _finite_box(value: Any, *, name: str) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{name} must be four finite normalized xyxy values")
    x1, y1, x2, y2 = (float(x) for x in box)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{name} is outside the legal normalized xyxy domain")
    return box


def _prepare_gray(frame: np.ndarray, source_wh: tuple[int, int], position: int) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError(f"gray_frames[{position}] must be a two-dimensional uint8 array")
    width, height = source_wh
    if frame.shape != (height, width):
        raise ValueError(f"gray_frames[{position}] shape does not match source_wh")
    longest = max(width, height)
    if longest <= MAX_GRAY_LONG_EDGE:
        return frame.copy()
    scale = MAX_GRAY_LONG_EDGE / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _shape_wh(gray: np.ndarray) -> tuple[int, int]:
    return int(gray.shape[1]), int(gray.shape[0])


def _feature_points(gray: np.ndarray, box_xyxy_norm: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    x1 = max(0, min(width, int(math.floor(float(box_xyxy_norm[0]) * width))))
    y1 = max(0, min(height, int(math.floor(float(box_xyxy_norm[1]) * height))))
    x2 = max(0, min(width, int(math.ceil(float(box_xyxy_norm[2]) * width))))
    y2 = max(0, min(height, int(math.ceil(float(box_xyxy_norm[3]) * height))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 1, 2), dtype=np.float32)
    # Rasterizing the legal normalized box into an image mask is the only
    # pixel-domain clipping here; predicted geometry itself is never clamped.
    mask[y1:y2, x1:x2] = 255
    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        mask=mask,
        blockSize=3,
        useHarrisDetector=False,
    )
    if found is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(found, dtype=np.float32).reshape(-1, 1, 2)


def _scale_points(points: np.ndarray, from_wh: tuple[int, int],
                  to_wh: tuple[int, int]) -> np.ndarray:
    from_width, from_height = from_wh
    to_width, to_height = to_wh
    factors = np.asarray([to_width / from_width, to_height / from_height], dtype=np.float32)
    return np.ascontiguousarray(points * factors.reshape(1, 1, 2), dtype=np.float32)


def _within_image(points: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    width, height = wh
    xy = points.reshape(-1, 2)
    return (np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0.0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0.0) & (xy[:, 1] < height))


def _lk(points: np.ndarray, source: np.ndarray, target: np.ndarray,
        initial_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ITERATIONS, EPSILON)
    moved, status, _error = cv2.calcOpticalFlowPyrLK(
        source,
        target,
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(initial_target, dtype=np.float32),
        winSize=WINDOW_SIZE,
        maxLevel=PYRAMID_MAX_LEVEL,
        criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    if moved is None or status is None:
        return np.empty_like(points), np.zeros((len(points),), dtype=bool)
    return (np.asarray(moved, dtype=np.float32).reshape(-1, 1, 2),
            np.asarray(status).reshape(-1).astype(bool))


def _transition(previous_gray: np.ndarray, next_gray: np.ndarray,
                previous_wh: tuple[int, int], next_wh: tuple[int, int],
                next_source_wh: tuple[int, int],
                previous_points: np.ndarray, current_center: np.ndarray,
                next_box_wh_norm: np.ndarray) -> dict[str, Any]:
    before_count = int(len(previous_points))
    if before_count < MIN_TRACKED_CORNERS:
        return {"ok": False, "reason": "fewer_than_four_input_corners",
                "input_corners": before_count, "kept_corners": 0, "retained_ratio": 0.0}

    initial_next = _scale_points(previous_points, previous_wh, next_wh)
    next_points, forward_status = _lk(previous_points, previous_gray, next_gray, initial_next)
    initial_back = _scale_points(next_points, next_wh, previous_wh)
    back_points, backward_status = _lk(next_points, next_gray, previous_gray, initial_back)

    prev_xy = previous_points.reshape(-1, 2)
    next_xy = next_points.reshape(-1, 2)
    back_xy = back_points.reshape(-1, 2)
    fb_error = np.linalg.norm(back_xy - prev_xy, axis=1)
    keep = (forward_status & backward_status
            & _within_image(next_points, next_wh)
            & _within_image(back_points, previous_wh)
            & np.isfinite(fb_error)
            & (fb_error <= MAX_FORWARD_BACKWARD_ERROR_PX))
    kept_count = int(keep.sum())
    ratio = kept_count / float(before_count)
    median_fb_error = float(np.median(fb_error[keep])) if kept_count else None
    diagnostics = {
        "input_corners": before_count,
        "kept_corners": kept_count,
        "retained_ratio": ratio,
        "median_forward_backward_error_prepared_gray_px": median_fb_error,
    }
    if kept_count < MIN_TRACKED_CORNERS:
        return {"ok": False, "reason": "fewer_than_four_retained_corners", **diagnostics}
    if ratio < MIN_RETAINED_RATIO:
        return {"ok": False, "reason": "retained_ratio_below_half", **diagnostics}

    prev_wh_f = np.asarray(previous_wh, dtype=np.float64)
    next_wh_f = np.asarray(next_wh, dtype=np.float64)
    prev_normalized = prev_xy[keep].astype(np.float64) / prev_wh_f
    next_normalized = next_xy[keep].astype(np.float64) / next_wh_f
    displacement = next_normalized - prev_normalized
    median_dx_norm, median_dy_norm = np.median(displacement, axis=0).tolist()
    shifted_center = current_center + np.asarray([median_dx_norm, median_dy_norm],
                                                 dtype=np.float64)
    half_wh = next_box_wh_norm / 2.0
    shifted_box = np.asarray([
        shifted_center[0] - half_wh[0], shifted_center[1] - half_wh[1],
        shifted_center[0] + half_wh[0], shifted_center[1] + half_wh[1],
    ], dtype=np.float64)
    if not (np.isfinite(shifted_box).all()
            and 0.0 <= shifted_box[0] < shifted_box[2] <= 1.0
            and 0.0 <= shifted_box[1] < shifted_box[3] <= 1.0):
        return {"ok": False, "reason": "box_out_of_bounds", **diagnostics,
                "median_dx_normalized": float(median_dx_norm),
                "median_dy_normalized": float(median_dy_norm),
                "proposed_box_xyxy_norm": shifted_box.tolist()}
    return {
        "ok": True,
        "box_xyxy_norm": shifted_box,
        "center_xy_norm": shifted_center,
        "points": np.ascontiguousarray(next_points[keep], dtype=np.float32),
        **diagnostics,
        "median_dx_normalized": float(median_dx_norm),
        "median_dy_normalized": float(median_dy_norm),
        "median_dx_source_pixels": float(median_dx_norm * next_source_wh[0]),
        "median_dy_source_pixels": float(median_dy_norm * next_source_wh[1]),
    }


def track_second_observation(
    gray_frames: Sequence[np.ndarray],
    source_wh: Sequence[Sequence[int]],
    observation_positions: Sequence[int],
    observation_boxes_xyxy_norm: Sequence[Sequence[float]],
    *,
    normalized_wh_by_frame: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Track from only the second observed anchor, independently each direction.

    ``observation_positions`` index the fixed source-frame sequence. The box
    at ordinal 1 is the sole initialization anchor. ``normalized_wh_by_frame``
    supplies the Source prediction's normalized width and height for every
    frame; the anchor entry must match the anchor box. Other observations are
    never used to restart, correct, or relabel a lost track. A lost branch
    stays ``None`` for all farther frames.
    """
    if not isinstance(gray_frames, Sequence) or len(gray_frames) < 2:
        raise ValueError("gray_frames must contain at least two fixed source frames")
    if not isinstance(source_wh, Sequence) or len(source_wh) != len(gray_frames):
        raise ValueError("source_wh must provide one original (width,height) pair per frame")
    if len(normalized_wh_by_frame) != len(gray_frames):
        raise ValueError("normalized_wh_by_frame must provide one source box width/height per frame")
    if not isinstance(observation_positions, Sequence) or len(observation_positions) < 2:
        raise ValueError("at least two observed positions are required; ordinal 1 is the anchor")
    if not isinstance(observation_boxes_xyxy_norm, Sequence) or len(observation_boxes_xyxy_norm) != len(observation_positions):
        raise ValueError("observation boxes and positions must have equal length")

    fixed_wh: list[tuple[int, int]] = []
    prepared: list[np.ndarray] = []
    for index, (frame, raw_wh) in enumerate(zip(gray_frames, source_wh)):
        if not isinstance(raw_wh, Sequence) or len(raw_wh) != 2:
            raise ValueError(f"source_wh[{index}] must be an (width,height) pair")
        width, height = raw_wh
        if (isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, (int, np.integer))
                or not isinstance(height, (int, np.integer))
                or int(width) <= 0 or int(height) <= 0):
            raise ValueError(f"source_wh[{index}] dimensions must be positive integers")
        wh = (int(width), int(height))
        fixed_wh.append(wh)
        prepared.append(_prepare_gray(frame, wh, index))

    positions: list[int] = []
    for ordinal, raw_position in enumerate(observation_positions):
        if (isinstance(raw_position, bool) or not isinstance(raw_position, (int, np.integer))
                or not 0 <= int(raw_position) < len(gray_frames)):
            raise ValueError(f"observation_positions[{ordinal}] is outside the source frame sequence")
        positions.append(int(raw_position))
    if len(set(positions)) != len(positions):
        raise ValueError("observation positions must be unique")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("observation positions must be strictly increasing")
    anchor_position = positions[1]
    anchor_box = _finite_box(observation_boxes_xyxy_norm[1], name="second observation anchor")
    source_box_wh: list[np.ndarray] = []
    for index, raw_wh in enumerate(normalized_wh_by_frame):
        wh = np.asarray(raw_wh, dtype=np.float64)
        if wh.shape != (2,) or not np.isfinite(wh).all() or not (0.0 < wh[0] <= 1.0
                                                                   and 0.0 < wh[1] <= 1.0):
            raise ValueError(f"normalized_wh_by_frame[{index}] must be finite positive normalized width/height")
        source_box_wh.append(wh.copy())
    anchor_box_wh = anchor_box[2:] - anchor_box[:2]
    if not np.allclose(source_box_wh[anchor_position], anchor_box_wh, rtol=0.0, atol=1e-8):
        raise ValueError("anchor box width/height must match normalized_wh_by_frame at the anchor")

    anchor_gray = prepared[anchor_position]
    anchor_points = _feature_points(anchor_gray, anchor_box)
    trajectory: list[dict[str, Any]] = [
        {"position": index, "source_width": fixed_wh[index][0],
         "source_height": fixed_wh[index][1], "status": "missing",
         "reason": "not_reached_from_second_observation", "box_xyxy_norm": None}
        for index in range(len(prepared))
    ]
    trajectory[anchor_position] = {
        "position": anchor_position,
        "source_width": fixed_wh[anchor_position][0],
        "source_height": fixed_wh[anchor_position][1],
        "status": "anchor",
        "reason": None,
        "box_xyxy_norm": anchor_box.tolist(),
        "corner_count": int(len(anchor_points)),
    }

    directions: dict[str, dict[str, Any]] = {}
    if len(anchor_points) < MIN_TRACKED_CORNERS:
        for direction, indices in (("backward", range(anchor_position - 1, -1, -1)),
                                   ("forward", range(anchor_position + 1, len(prepared)))):
            for index in indices:
                trajectory[index].update(reason="fewer_than_four_anchor_corners")
            directions[direction] = {"status": "missing", "reason": "fewer_than_four_anchor_corners",
                                     "attempted_transitions": 0, "accepted_transitions": 0}
    else:
        directions["backward"] = _track_direction(
            direction="backward", indices=range(anchor_position - 1, -1, -1),
            anchor_position=anchor_position, anchor_box=anchor_box,
            anchor_points=anchor_points, frames=prepared, source_wh=fixed_wh,
            normalized_wh_by_frame=source_box_wh,
            trajectory=trajectory)
        directions["forward"] = _track_direction(
            direction="forward", indices=range(anchor_position + 1, len(prepared)),
            anchor_position=anchor_position, anchor_box=anchor_box,
            anchor_points=anchor_points, frames=prepared, source_wh=fixed_wh,
            normalized_wh_by_frame=source_box_wh,
            trajectory=trajectory)

    return {
        "schema": SCHEMA,
        "status": "complete_with_explicit_missing",
        "anchor_observation_ordinal": 1,
        "anchor_position": anchor_position,
        "anchor_box_xyxy_norm": anchor_box.tolist(),
        "normalized_wh_by_frame": [wh.tolist() for wh in source_box_wh],
        "prepared_gray_wh_by_frame": [list(_shape_wh(frame)) for frame in prepared],
        "anchor_corner_count": int(len(anchor_points)),
        "fixed_settings": {
            "gray_long_edge_max": MAX_GRAY_LONG_EDGE,
            "max_corners": MAX_CORNERS,
            "quality_level": QUALITY_LEVEL,
            "min_distance_px": MIN_DISTANCE,
            "win_size": list(WINDOW_SIZE),
            "pyramid_max_level": PYRAMID_MAX_LEVEL,
            "iterations": ITERATIONS,
            "epsilon": EPSILON,
            "max_forward_backward_error_px": MAX_FORWARD_BACKWARD_ERROR_PX,
            "forward_backward_error_pixel_domain": "per-frame grayscale after longest-edge resize",
            "minimum_tracked_corners": MIN_TRACKED_CORNERS,
            "minimum_retained_ratio": MIN_RETAINED_RATIO,
        },
        "directions": directions,
        "trajectory": trajectory,
        "tracklet_reliability_claim": "none; sparse LK correspondence is diagnostic only",
        "geometry_clamped": False,
        "later_observations_used_to_recover_track": False,
    }


def _track_direction(*, direction: str, indices: Sequence[int], anchor_position: int,
                     anchor_box: np.ndarray, anchor_points: np.ndarray,
                     frames: Sequence[np.ndarray], source_wh: Sequence[tuple[int, int]],
                     normalized_wh_by_frame: Sequence[np.ndarray],
                     trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = list(indices)
    if not ordered:
        return {"status": "empty", "reason": None,
                "attempted_transitions": 0, "accepted_transitions": 0}

    previous_position = anchor_position
    previous_center = (anchor_box[:2] + anchor_box[2:]) / 2.0
    previous_points = anchor_points.copy()
    attempted = 0
    accepted = 0
    for position in ordered:
        attempted += 1
        result = _transition(
            frames[previous_position], frames[position],
            _shape_wh(frames[previous_position]), _shape_wh(frames[position]),
            source_wh[position],
            previous_points, previous_center, normalized_wh_by_frame[position])
        if not result["ok"]:
            trajectory[position].update({
                "reason": result["reason"],
                "transition_diagnostics": {k: v for k, v in result.items()
                                            if k not in {"ok", "reason", "points", "box_xyxy_norm",
                                                         "center_xy_norm"}},
            })
            for lost_position in ordered[ordered.index(position) + 1:]:
                trajectory[lost_position].update(
                    reason=f"track_lost_after_position_{position}")
            return {"status": "lost", "reason": result["reason"],
                    "lost_at_position": position,
                    "attempted_transitions": attempted, "accepted_transitions": accepted}

        box = result["box_xyxy_norm"]
        wh = source_wh[position]
        trajectory[position] = {
            "position": position,
            "source_width": wh[0],
            "source_height": wh[1],
            "status": "tracked",
            "reason": None,
            "box_xyxy_norm": box.tolist(),
            "transition_diagnostics": {k: v for k, v in result.items()
                                        if k not in {"ok", "points", "box_xyxy_norm",
                                                     "center_xy_norm"}},
        }
        accepted += 1
        previous_position = position
        previous_center = result["center_xy_norm"]
        previous_points = result["points"]

    return {"status": "complete", "reason": None,
            "attempted_transitions": attempted, "accepted_transitions": accepted}


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _random_texture(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def _assert_near(value: float, target: float, tolerance: float, label: str) -> None:
    if abs(float(value) - float(target)) > tolerance:
        raise AssertionError(f"{label}: got {value}, expected {target} +/- {tolerance}")


def run_synthetic_contracts() -> dict[str, Any]:
    """Exercise sparse-LK behavior on generated CPU images only."""
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)

    width, height = 128, 96
    base = _random_texture(height, width, 14)
    translation_frames = [_translate(base, -1, -1), base.copy(), _translate(base, 1, 1)]
    source_wh = [(width, height)] * 3
    observations = [0, 1]
    boxes = [[.20, .20, .80, .80], [.20, .20, .80, .80]]
    source_box_sizes = [[.60, .60], [.60, .60], [.60, .60]]
    hashes_before = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in translation_frames]
    observation_copy = json.loads(json.dumps({"positions": observations, "boxes": boxes,
                                              "normalized_wh": source_box_sizes}))
    translated = track_second_observation(translation_frames, source_wh, observations, boxes,
                                          normalized_wh_by_frame=source_box_sizes)
    assert translated["trajectory"][1]["status"] == "anchor"
    assert translated["trajectory"][0]["status"] == "tracked"
    assert translated["trajectory"][2]["status"] == "tracked"
    # Independent forward and reverse propagation around the second observation.
    _assert_near(translated["trajectory"][2]["box_xyxy_norm"][0] - .20,
                 1.0 / width, .012, "forward normalized x displacement")
    _assert_near(translated["trajectory"][0]["box_xyxy_norm"][0] - .20,
                 -1.0 / width, .012, "reverse normalized x displacement")

    # A featureless anchor yields explicit missing values, not zero boxes.
    flat = np.full((height, width), 80, dtype=np.uint8)
    missing = track_second_observation([flat, flat.copy(), flat.copy()], source_wh,
                                       [0, 1], boxes,
                                       normalized_wh_by_frame=source_box_sizes)
    assert missing["trajectory"][1]["status"] == "anchor"
    assert missing["trajectory"][0]["box_xyxy_norm"] is None
    assert missing["trajectory"][2]["box_xyxy_norm"] is None
    assert missing["directions"]["forward"]["reason"] == "fewer_than_four_anchor_corners"

    # The whole box is missing when its proposed translation leaves the image;
    # the geometry is never clamped back into the unit square. A deterministic
    # exact 2px flow isolates the bounds rule from LK texture noise.
    edge_texture = _random_texture(height, width, 31)
    edge_frames = [edge_texture.copy(), edge_texture.copy(), edge_texture.copy()]
    edge_box = [[.005, .20, .995, .80], [.005, .20, .995, .80]]
    edge_sizes = [[.99, .60], [.99, .60], [.99, .60]]
    original_lk = cv2.calcOpticalFlowPyrLK
    calls = {"count": 0}

    def exact_two_pixel_lk(source, target, points, next_points=None, **kwargs):
        calls["count"] += 1
        start = np.asarray(next_points if next_points is not None else points,
                           dtype=np.float32).copy()
        start += np.asarray([[[2.0 if calls["count"] % 2 else -2.0, 0.0]]],
                            dtype=np.float32)
        status = np.ones((len(start), 1), dtype=np.uint8)
        return start, status, np.zeros((len(start), 1), dtype=np.float32)

    cv2.calcOpticalFlowPyrLK = exact_two_pixel_lk
    try:
        out_of_bounds = track_second_observation(
            edge_frames, source_wh, [0, 1], edge_box,
            normalized_wh_by_frame=edge_sizes)
    finally:
        cv2.calcOpticalFlowPyrLK = original_lk
    assert out_of_bounds["trajectory"][2]["box_xyxy_norm"] is None
    assert out_of_bounds["trajectory"][2]["reason"] == "box_out_of_bounds"
    proposed = out_of_bounds["trajectory"][2]["transition_diagnostics"]["proposed_box_xyxy_norm"]
    assert proposed[2] > 1.0

    # Deterministically exercise the FB threshold gate without fabricating a
    # flow field from real content: forward is plausible, reverse is 5px off.
    original_lk = cv2.calcOpticalFlowPyrLK
    calls = {"count": 0}

    def fake_lk(source, target, points, next_points=None, **kwargs):
        calls["count"] += 1
        start = np.asarray(next_points if next_points is not None else points, dtype=np.float32).copy()
        if calls["count"] % 2 == 0:
            start += np.asarray([[[5.0, 0.0]]], dtype=np.float32)
        status = np.ones((len(start), 1), dtype=np.uint8)
        return start, status, np.zeros((len(start), 1), dtype=np.float32)

    cv2.calcOpticalFlowPyrLK = fake_lk
    try:
        fb_rejected = track_second_observation(translation_frames, source_wh, [0, 1], boxes,
                                               normalized_wh_by_frame=source_box_sizes)
    finally:
        cv2.calcOpticalFlowPyrLK = original_lk
    assert fb_rejected["trajectory"][0]["box_xyxy_norm"] is None
    assert fb_rejected["trajectory"][0]["reason"] == "fewer_than_four_retained_corners"

    # Once a direction loses the track at a later observed position, following
    # observations cannot restart it. This mock fails the first forward step
    # after a successful two-step reverse walk from position 2.
    longer_frames = [_random_texture(height, width, 200 + i) for i in range(5)]
    original_lk = cv2.calcOpticalFlowPyrLK
    calls = {"count": 0}

    def lose_then_stop_lk(source, target, points, next_points=None, **kwargs):
        calls["count"] += 1
        start = np.asarray(next_points if next_points is not None else points,
                           dtype=np.float32).copy()
        if calls["count"] == 6:
            start += np.asarray([[[5.0, 0.0]]], dtype=np.float32)
        status = np.ones((len(start), 1), dtype=np.uint8)
        return start, status, np.zeros((len(start), 1), dtype=np.float32)

    cv2.calcOpticalFlowPyrLK = lose_then_stop_lk
    try:
        lost_forward = track_second_observation(
            longer_frames, [(width, height)] * 5, [1, 2, 3, 4],
            [[.20, .20, .80, .80], [.20, .20, .80, .80],
             [.05, .05, .95, .95], [.10, .10, .90, .90]],
            normalized_wh_by_frame=[[.60, .60]] * 5)
    finally:
        cv2.calcOpticalFlowPyrLK = original_lk
    assert lost_forward["trajectory"][3]["status"] == "missing"
    assert lost_forward["trajectory"][3]["reason"] == "fewer_than_four_retained_corners"
    assert lost_forward["trajectory"][4]["status"] == "missing"
    assert lost_forward["trajectory"][4]["reason"] == "track_lost_after_position_3"

    # Distinct local patches move independently; a box around A follows A,
    # not a second patch B moving in the opposite direction.
    scene_h, scene_w = 96, 128
    patch_a = _random_texture(26, 26, 44)
    patch_b = _random_texture(26, 26, 91)

    def independent_frame(a_x: int, b_x: int) -> np.ndarray:
        image = np.zeros((scene_h, scene_w), dtype=np.uint8)
        image[34:60, a_x:a_x + 26] = patch_a
        image[34:60, b_x:b_x + 26] = patch_b
        return image

    independent = [independent_frame(24, 84), independent_frame(30, 78), independent_frame(36, 72)]
    independent_result = track_second_observation(
        independent, source_wh, [0, 1], [[.15, .30, .46, .72], [.15, .30, .52, .72]],
        normalized_wh_by_frame=[[.37, .42], [.37, .42], [.37, .42]])
    _require_status = independent_result["trajectory"][2]
    assert _require_status["status"] == "tracked"
    independent_dx_px = (_require_status["box_xyxy_norm"][0] - .15) * scene_w
    _assert_near(independent_dx_px, 6.0, 2.2, "independent Source patch pixel displacement")
    assert independent_dx_px > 0.0

    # The longest-side resize is 640px; output motion remains normalized by
    # original per-frame width/height, preserving source-pixel units.
    large_w, large_h = 800, 600
    large = _random_texture(large_h, large_w, 12)
    large_result = track_second_observation(
        [_translate(large, -8, -5), large.copy(), _translate(large, 8, 5)],
        [(large_w, large_h)] * 3, [0, 1], [[.20, .20, .80, .80], [.20, .20, .80, .80]],
        normalized_wh_by_frame=source_box_sizes)
    large_row = large_result["trajectory"][2]
    assert large_row["status"] == "tracked"
    assert max(max(size) for size in large_result["prepared_gray_wh_by_frame"]) <= 640
    assert large_result["prepared_gray_wh_by_frame"][1] == [640, 480]
    _assert_near((large_row["box_xyxy_norm"][0] - .20) * large_w,
                 8.0, 2.5, "resized-frame source pixel displacement")
    _assert_near(large_row["transition_diagnostics"]["median_dx_source_pixels"],
                 8.0, 0.35, "reported source pixel units")

    # Box size comes from each frame's explicit Source width/height while its
    # center follows LK. It is not copied from the anchor box.
    varying_sizes = [[.60, .60], [.60, .60], [.40, .50]]
    varying = track_second_observation(translation_frames, source_wh, [0, 1], boxes,
                                       normalized_wh_by_frame=varying_sizes)
    varying_box = np.asarray(varying["trajectory"][2]["box_xyxy_norm"])
    _assert_near(varying_box[2] - varying_box[0], .40, 1e-10,
                 "per-frame Source normalized width")
    _assert_near(varying_box[3] - varying_box[1], .50, 1e-10,
                 "per-frame Source normalized height")
    _assert_near((varying_box[0] + varying_box[2]) / 2.0,
                 .5 + 1.0 / width, .012, "center follows LK with per-frame dimensions")

    # Frame and observation inputs are immutable from the caller's perspective.
    hashes_after = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in translation_frames]
    assert hashes_before == hashes_after
    assert {"positions": observations, "boxes": boxes,
            "normalized_wh": source_box_sizes} == observation_copy

    return {
        "schema": "stvg-motion-tracklet-lk-cpu-contracts-v1",
        "status": "pass",
        "CPU_only": True,
        "GPU_used": False,
        "GT_read": False,
        "model_loaded": False,
        "project_data_imported": False,
        "checks": {
            "translation_forward_and_reverse": True,
            "featureless_anchor_explicit_none": True,
            "out_of_bounds_box_rejected_without_clamp": True,
            "forward_backward_error_gate": True,
            "lost_track_stays_missing_across_later_observation": True,
            "independent_source_patch_motion": True,
            "resized_flow_reports_original_source_pixel_units": True,
            "per_frame_source_width_height_preserved": True,
            "caller_frames_and_observation_metadata_unmodified": True,
            "fixed_contract_parameters": {
                "gray_long_edge_max": MAX_GRAY_LONG_EDGE,
                "max_corners": MAX_CORNERS,
                "quality_level": QUALITY_LEVEL,
                "min_distance_px": MIN_DISTANCE,
                "win_size": list(WINDOW_SIZE),
                "pyramid_max_level": PYRAMID_MAX_LEVEL,
                "iterations": ITERATIONS,
                "epsilon": EPSILON,
                "max_forward_backward_error_px": MAX_FORWARD_BACKWARD_ERROR_PX,
                "forward_backward_error_pixel_domain": "per-frame grayscale after longest-edge resize",
                "minimum_tracked_corners": MIN_TRACKED_CORNERS,
                "minimum_retained_ratio": MIN_RETAINED_RATIO,
            },
        },
    }


def _write_contract_receipt(receipt: Mapping[str, Any]) -> None:
    CONTRACT_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    if CONTRACT_RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite existing contract receipt: {CONTRACT_RECEIPT}")
    payload = dict(receipt)
    payload["helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload["receipt_path"] = str(CONTRACT_RECEIPT.relative_to(ROOT))
    temporary = CONTRACT_RECEIPT.with_suffix(CONTRACT_RECEIPT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(CONTRACT_RECEIPT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run synthetic CPU-only tracklet contracts and write the receipt")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("this helper accepts inputs only through track_second_observation; run --contracts for synthetic checks")
    receipt = run_synthetic_contracts()
    _write_contract_receipt(receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
