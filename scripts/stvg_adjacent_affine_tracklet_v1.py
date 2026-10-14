"""Deterministic adjacent-frame Shi-Tomasi/LK/RANSAC box tracklet helper.

This module consumes only an already-decoded sequence of grayscale frames and
one normalized anchor box.  It has no project-data, model, annotation, or GT
imports.  Every adjacent transition redetects Shi-Tomasi corners inside the
current candidate box; a failed direction remains missing for the rest of that
direction.  Geometry is never clipped to the image.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA = "stvg_adjacent_affine_tracklet_v1"
CONFIG = {
    "corner_detector": "Shi-Tomasi cv2.goodFeaturesToTrack",
    "max_corners": 80,
    "quality_level": 0.01,
    "min_distance_px": 5.0,
    "block_size": 3,
    "lk_win_size": [21, 21],
    "lk_max_level": 3,
    "lk_criteria": ["COUNT|EPS", 30, 0.01],
    "forward_backward_threshold_px": 1.5,
    "minimum_fb_valid_corners": 4,
    "minimum_retention": 0.5,
    "ransac": {
        "method": "estimateAffinePartial2D/RANSAC",
        "reprojection_threshold_px": 2.0,
        "max_iterations": 2000,
        "confidence": 0.99,
        "refine_iterations": 10,
        "minimum_inliers": 4,
        "minimum_inlier_fraction": 0.5,
    },
    "rho": "retention * inlier_fraction * exp(-median_fb_error_px / 1.5)",
    "opencv_rng_seed_per_ransac_transition": 0,
    "opencv_threads": 2,
    "output_box_policy": "transform four corners; axis-aligned envelope; reject out-of-bounds; never clip",
    "retention_denominator": "corners detected in the current candidate box",
    "inlier_fraction_denominator": "forward-backward-valid correspondences",
    "median_fb_population": "forward-backward-valid correspondences",
}


def _as_valid_frames(gray_frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    if not isinstance(gray_frames, (list, tuple)) or not gray_frames:
        raise ValueError("gray_frames must be a non-empty list or tuple")
    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for i, frame in enumerate(gray_frames):
        arr = np.asarray(frame)
        if arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError(f"gray_frames[{i}] must be a 2D uint8 grayscale array")
        if min(arr.shape) < 2 or max(arr.shape) > 640:
            raise ValueError(f"gray_frames[{i}] dimensions must be positive with long edge <= 640")
        if expected_shape is None:
            expected_shape = (int(arr.shape[0]), int(arr.shape[1]))
        elif tuple(arr.shape) != expected_shape:
            raise ValueError("all grayscale frames must have the same shape")
        frames.append(arr)
    assert expected_shape is not None
    return frames, expected_shape[1], expected_shape[0]


def _as_ids(physical_frame_ids: Sequence[int], n: int) -> list[int]:
    if not isinstance(physical_frame_ids, (list, tuple)) or len(physical_frame_ids) != n:
        raise ValueError("physical_frame_ids must have one ID per frame")
    ids: list[int] = []
    for value in physical_frame_ids:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("physical frame IDs must be integers")
        ids.append(int(value))
    if any(b != a + 1 for a, b in zip(ids, ids[1:])):
        raise ValueError("physical_frame_ids must be strictly consecutive with step +1")
    return ids


def _as_anchor_box(anchor_box_xyxy_norm: Sequence[float]) -> list[float]:
    arr = np.asarray(anchor_box_xyxy_norm, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        raise ValueError("anchor_box_xyxy_norm must contain four finite coordinates")
    x1, y1, x2, y2 = (float(v) for v in arr)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("anchor box must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return [x1, y1, x2, y2]


def _empty_transition(from_i: int, to_i: int, ids: list[int], direction: str, reason: str) -> dict[str, Any]:
    return {
        "direction": direction,
        "from_index": int(from_i),
        "to_index": int(to_i),
        "from_physical_frame_id": int(ids[from_i]),
        "to_physical_frame_id": int(ids[to_i]),
        "attempted": False,
        "status": "not_attempted_track_lost",
        "reason": reason,
        "detected_corner_count": 0,
        "forward_status_count": 0,
        "backward_status_count": 0,
        "fb_valid_count": 0,
        "retention": None,
        "fb_error_median_px": None,
        "fb_error_max_px": None,
        "inlier_count": 0,
        "inlier_fraction": None,
        "affine_2x3": None,
        "affine_determinant": None,
        "center_dxdy_px": None,
        "center_dxdy_normalized": None,
        "rho": None,
        "proposed_box_xyxy_norm": None,
        "accepted_box_xyxy_norm": None,
    }


def _candidate_mask(box: Sequence[float], width: int, height: int) -> np.ndarray:
    """Build a clipped *indexing mask*; this does not clamp candidate geometry."""
    x1, y1, x2, y2 = box
    ix1 = max(0, min(width, int(math.floor(x1 * width))))
    iy1 = max(0, min(height, int(math.floor(y1 * height))))
    ix2 = max(0, min(width, int(math.ceil(x2 * width))))
    iy2 = max(0, min(height, int(math.ceil(y2 * height))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 255
    return mask


def _failed_attempt(
    from_i: int,
    to_i: int,
    ids: list[int],
    direction: str,
    reason: str,
    detected: int = 0,
    forward_status: int = 0,
    backward_status: int = 0,
    fb_errors: np.ndarray | None = None,
    retention: float | None = None,
    fb_valid_count: int = 0,
) -> dict[str, Any]:
    err_list = None if fb_errors is None or fb_errors.size == 0 else fb_errors.astype(float).tolist()
    return {
        "direction": direction,
        "from_index": int(from_i),
        "to_index": int(to_i),
        "from_physical_frame_id": int(ids[from_i]),
        "to_physical_frame_id": int(ids[to_i]),
        "attempted": True,
        "status": "rejected",
        "reason": reason,
        "detected_corner_count": int(detected),
        "forward_status_count": int(forward_status),
        "backward_status_count": int(backward_status),
        "fb_valid_count": int(fb_valid_count),
        "retention": None if retention is None else float(retention),
        "fb_error_median_px": None if err_list is None else float(np.median(err_list)),
        "fb_error_max_px": None if err_list is None else float(np.max(err_list)),
        "inlier_count": 0,
        "inlier_fraction": None,
        "affine_2x3": None,
        "affine_determinant": None,
        "center_dxdy_px": None,
        "center_dxdy_normalized": None,
        "rho": None,
        "proposed_box_xyxy_norm": None,
        "accepted_box_xyxy_norm": None,
    }


def _attempt_transition(
    frames: list[np.ndarray],
    ids: list[int],
    from_i: int,
    to_i: int,
    current_box: list[float],
    direction: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    source = frames[from_i]
    target = frames[to_i]
    mask = _candidate_mask(current_box, width, height)
    corners = cv2.goodFeaturesToTrack(
        source,
        maxCorners=int(CONFIG["max_corners"]),
        qualityLevel=float(CONFIG["quality_level"]),
        minDistance=float(CONFIG["min_distance_px"]),
        mask=mask,
        blockSize=int(CONFIG["block_size"]),
        useHarrisDetector=False,
    )
    detected = 0 if corners is None else int(len(corners))
    if detected < int(CONFIG["minimum_fb_valid_corners"]):
        return _failed_attempt(from_i, to_i, ids, direction, "insufficient_detected_corners", detected)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p_next, st_forward, _ = cv2.calcOpticalFlowPyrLK(
        source, target, corners, None, winSize=(21, 21), maxLevel=3, criteria=criteria
    )
    if p_next is None or st_forward is None:
        return _failed_attempt(
            from_i, to_i, ids, direction, "forward_lk_failed", detected
        )
    p_back, st_backward, _ = cv2.calcOpticalFlowPyrLK(
        target, source, p_next, None, winSize=(21, 21), maxLevel=3, criteria=criteria
    )
    if p_back is None or st_backward is None:
        return _failed_attempt(
            from_i,
            to_i,
            ids,
            direction,
            "backward_lk_failed",
            detected,
            int(np.count_nonzero(st_forward)),
        )

    src_pts = corners.reshape(-1, 2).astype(np.float64)
    dst_pts = p_next.reshape(-1, 2).astype(np.float64)
    back_pts = p_back.reshape(-1, 2).astype(np.float64)
    stf = st_forward.reshape(-1).astype(bool)
    stb = st_backward.reshape(-1).astype(bool)
    fb_errors_all = np.linalg.norm(src_pts - back_pts, axis=1)
    finite = np.isfinite(src_pts).all(axis=1) & np.isfinite(dst_pts).all(axis=1) & np.isfinite(fb_errors_all)
    fb_valid = stf & stb & finite & (fb_errors_all <= float(CONFIG["forward_backward_threshold_px"]))
    fb_errors = fb_errors_all[fb_valid]
    valid_count = int(np.count_nonzero(fb_valid))
    retention = float(valid_count / detected)
    forward_count = int(np.count_nonzero(stf))
    backward_count = int(np.count_nonzero(stb))
    if valid_count < int(CONFIG["minimum_fb_valid_corners"]):
        return _failed_attempt(
            from_i,
            to_i,
            ids,
            direction,
            "insufficient_fb_valid_corners",
            detected,
            forward_count,
            backward_count,
            fb_errors,
            retention,
            valid_count,
        )
    if retention < float(CONFIG["minimum_retention"]):
        return _failed_attempt(
            from_i,
            to_i,
            ids,
            direction,
            "fb_retention_below_threshold",
            detected,
            forward_count,
            backward_count,
            fb_errors,
            retention,
            valid_count,
        )

    source_points = src_pts[fb_valid].astype(np.float32)
    target_points = dst_pts[fb_valid].astype(np.float32)
    cv2.setRNGSeed(int(CONFIG["opencv_rng_seed_per_ransac_transition"]))
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(CONFIG["ransac"]["reprojection_threshold_px"]),
        maxIters=int(CONFIG["ransac"]["max_iterations"]),
        confidence=float(CONFIG["ransac"]["confidence"]),
        refineIters=int(CONFIG["ransac"]["refine_iterations"]),
    )
    if affine is None or inlier_mask is None:
        return _failed_attempt(
            from_i,
            to_i,
            ids,
            direction,
            "ransac_affine_failed",
            detected,
            forward_count,
            backward_count,
            fb_errors,
            retention,
            valid_count,
        )
    affine = np.asarray(affine, dtype=np.float64)
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = float(inlier_count / valid_count)
    determinant = float(np.linalg.det(affine[:, :2]))
    base = {
        "direction": direction,
        "from_index": int(from_i),
        "to_index": int(to_i),
        "from_physical_frame_id": int(ids[from_i]),
        "to_physical_frame_id": int(ids[to_i]),
        "attempted": True,
        "status": "rejected",
        "reason": None,
        "detected_corner_count": detected,
        "forward_status_count": forward_count,
        "backward_status_count": backward_count,
        "fb_valid_count": valid_count,
        "retention": retention,
        "fb_error_median_px": float(np.median(fb_errors)),
        "fb_error_max_px": float(np.max(fb_errors)),
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
        "affine_2x3": affine.tolist(),
        "affine_determinant": determinant,
        "center_dxdy_px": None,
        "center_dxdy_normalized": None,
        "rho": None,
        "proposed_box_xyxy_norm": None,
        "accepted_box_xyxy_norm": None,
    }
    if not np.isfinite(affine).all() or not math.isfinite(determinant):
        base["reason"] = "nonfinite_affine"
        return base
    if determinant <= 0.0:
        base["reason"] = "nonpositive_affine_determinant"
        return base
    if inlier_count < int(CONFIG["ransac"]["minimum_inliers"]):
        base["reason"] = "insufficient_ransac_inliers"
        return base
    if inlier_fraction < float(CONFIG["ransac"]["minimum_inlier_fraction"]):
        base["reason"] = "ransac_inlier_fraction_below_threshold"
        return base

    x1, y1, x2, y2 = current_box
    corners_box = np.asarray(
        [
            [x1 * width, y1 * height],
            [x2 * width, y1 * height],
            [x2 * width, y2 * height],
            [x1 * width, y2 * height],
        ],
        dtype=np.float64,
    )
    transformed = corners_box @ affine[:, :2].T + affine[:, 2]
    px_min = transformed.min(axis=0)
    px_max = transformed.max(axis=0)
    proposed = [
        float(px_min[0] / width),
        float(px_min[1] / height),
        float(px_max[0] / width),
        float(px_max[1] / height),
    ]
    base["proposed_box_xyxy_norm"] = proposed
    old_center = np.asarray([(x1 + x2) * width / 2.0, (y1 + y2) * height / 2.0])
    new_center = old_center @ affine[:, :2].T + affine[:, 2]
    center_delta = new_center - old_center
    base["center_dxdy_px"] = [float(center_delta[0]), float(center_delta[1])]
    base["center_dxdy_normalized"] = [float(center_delta[0] / width), float(center_delta[1] / height)]
    rho = retention * inlier_fraction * math.exp(
        -float(np.median(fb_errors)) / float(CONFIG["forward_backward_threshold_px"])
    )
    base["rho"] = float(rho)
    if not (0.0 <= proposed[0] < proposed[2] <= 1.0 and 0.0 <= proposed[1] < proposed[3] <= 1.0):
        base["reason"] = "transformed_box_out_of_bounds"
        return base
    base["status"] = "accepted"
    base["reason"] = None
    base["accepted_box_xyxy_norm"] = proposed.copy()
    return base


def track_adjacent(
    gray_frames: Sequence[np.ndarray],
    physical_frame_ids: Sequence[int],
    anchor_index: int,
    anchor_box_xyxy_norm: Sequence[float],
) -> dict[str, Any]:
    """Track one anchor independently forward and backward over adjacent frames.

    Args:
        gray_frames: Same-shape uint8 grayscale frames, already reduced to a
            longest edge of at most 640 pixels.
        physical_frame_ids: Strictly consecutive physical frame IDs (step +1).
        anchor_index: Index of the frame containing the supplied anchor box.
        anchor_box_xyxy_norm: Legal normalized xyxy anchor coordinates.

    Returns a JSON-safe tracklet. A rejected edge sets that direction's next
    trajectory slot to ``None`` and later edges in that direction are recorded
    as not attempted. No GT, prediction labels, or image arrays are returned.
    """
    frames, width, height = _as_valid_frames(gray_frames)
    ids = _as_ids(physical_frame_ids, len(frames))
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, (int, np.integer)):
        raise ValueError("anchor_index must be an integer")
    anchor_index = int(anchor_index)
    if anchor_index < 0 or anchor_index >= len(frames):
        raise ValueError("anchor_index is outside the frame sequence")
    anchor = _as_anchor_box(anchor_box_xyxy_norm)

    cv2.setNumThreads(int(CONFIG["opencv_threads"]))
    trajectory: list[dict[str, Any] | None] = [None] * len(frames)
    trajectory[anchor_index] = {
        "frame_index": anchor_index,
        "physical_frame_id": ids[anchor_index],
        "status": "anchor",
        "box_xyxy_norm": anchor.copy(),
        "reason": None,
    }
    transitions: list[dict[str, Any]] = []
    direction_stats: dict[str, dict[str, int]] = {}

    for direction in ("backward", "forward"):
        edge_pairs = (
            [(i, i - 1) for i in range(anchor_index, 0, -1)]
            if direction == "backward"
            else [(i, i + 1) for i in range(anchor_index, len(frames) - 1)]
        )
        stats = {
            "attempted_transitions": 0,
            "accepted_transitions": 0,
            "rejected_transitions": 0,
            "not_attempted_after_loss": 0,
            "detected_corner_count": 0,
            "fb_valid_corner_count": 0,
            "ransac_inlier_count": 0,
        }
        current_index = anchor_index
        current_box = anchor.copy()
        lost_reason: str | None = None
        for from_i, to_i in edge_pairs:
            if lost_reason is not None:
                item = _empty_transition(
                    from_i,
                    to_i,
                    ids,
                    direction,
                    f"direction_lost_after:{lost_reason}",
                )
                trajectory[to_i] = {
                    "frame_index": int(to_i),
                    "physical_frame_id": ids[to_i],
                    "status": "missing",
                    "box_xyxy_norm": None,
                    "reason": f"direction_lost_after:{lost_reason}",
                }
                stats["not_attempted_after_loss"] += 1
                transitions.append(item)
                continue
            # The traversal is contiguous from the anchor; these checks defend
            # against accidental reuse of a geometry from the opposite branch.
            if from_i != current_index:
                raise RuntimeError("internal directional traversal lost its current frame")
            item = _attempt_transition(frames, ids, from_i, to_i, current_box, direction, width, height)
            stats["attempted_transitions"] += 1
            stats["detected_corner_count"] += int(item["detected_corner_count"])
            stats["fb_valid_corner_count"] += int(item["fb_valid_count"])
            stats["ransac_inlier_count"] += int(item["inlier_count"])
            if item["status"] == "accepted":
                stats["accepted_transitions"] += 1
                current_index = to_i
                current_box = list(item["accepted_box_xyxy_norm"])
                trajectory[to_i] = {
                    "frame_index": int(to_i),
                    "physical_frame_id": ids[to_i],
                    "status": "tracked",
                    "box_xyxy_norm": current_box.copy(),
                    "reason": None,
                }
            else:
                stats["rejected_transitions"] += 1
                lost_reason = str(item["reason"])
                trajectory[to_i] = {
                    "frame_index": int(to_i),
                    "physical_frame_id": ids[to_i],
                    "status": "missing",
                    "box_xyxy_norm": None,
                    "reason": lost_reason,
                }
            transitions.append(item)
        direction_stats[direction] = stats

    # Present edge diagnostics in physical sequence order, then direction.
    transitions.sort(key=lambda x: (x["from_index"], x["to_index"], x["direction"]))
    total = {
        key: int(sum(v[key] for v in direction_stats.values()))
        for key in (
            "attempted_transitions",
            "accepted_transitions",
            "rejected_transitions",
            "not_attempted_after_loss",
            "detected_corner_count",
            "fb_valid_corner_count",
            "ransac_inlier_count",
        )
    }
    return {
        "schema": SCHEMA,
        "anchor_index": anchor_index,
        "anchor_physical_frame_id": ids[anchor_index],
        "frame_shape_hw": [int(height), int(width)],
        "physical_frame_ids": ids,
        "trajectory": trajectory,
        "transitions": transitions,
        "cost": {"directions": direction_stats, "total": total},
        "config": CONFIG.copy(),
    }


def _synthetic_texture(height: int = 240, width: int = 240, seed: int = 1776) -> np.ndarray:
    """Produce a deterministic textured grayscale image for CPU contracts."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    # Stable high-contrast structure supplements the random corners.
    for y in range(12, height - 12, 24):
        for x in range(12, width - 12, 24):
            color = int((x * 7 + y * 11) % 256)
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), color, -1)
    return image


def _warp(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.asarray(matrix, dtype=np.float32),
        (int(image.shape[1]), int(image.shape[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _jsonable_equal(a: Any, b: Any) -> bool:
    import json

    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )


def synthetic_contracts() -> dict[str, Any]:
    """Run finite synthetic CPU contracts; no project or annotation inputs."""
    base = _synthetic_texture()
    h, w = base.shape
    frames = [_warp(base, np.asarray([[1, 0, 3 * i], [0, 1, 2 * i]], dtype=np.float32)) for i in range(5)]
    moving = track_adjacent(frames, [100, 101, 102, 103, 104], 2, [0.30, 0.30, 0.70, 0.70])
    observed = [
        row for row in moving["trajectory"] if row is not None and row["status"] == "tracked"
    ]
    assert len(observed) == 4, f"translation fixture lost a frame: {moving['trajectory']}"
    first = moving["trajectory"][1]["box_xyxy_norm"]
    last = moving["trajectory"][3]["box_xyxy_norm"]
    assert first[0] < 0.30 and first[1] < 0.30
    assert last[0] > 0.30 and last[1] > 0.30
    assert all(t["rho"] is not None and 0.0 <= t["rho"] <= 1.0 for t in moving["transitions"])
    backward_edges = [t for t in moving["transitions"] if t["direction"] == "backward"]
    forward_edges = [t for t in moving["transitions"] if t["direction"] == "forward"]
    assert len(backward_edges) == len(forward_edges) == 2
    assert all(t["center_dxdy_px"][0] < 0 for t in backward_edges)
    assert all(t["center_dxdy_px"][0] > 0 for t in forward_edges)

    # Scale-only transformations about the image center must alter box size.
    cx, cy = w / 2.0, h / 2.0
    scale_frames = []
    for factor in (0.97, 1.0, 1.03):
        mat = np.asarray(
            [[factor, 0.0, cx * (1.0 - factor)], [0.0, factor, cy * (1.0 - factor)]],
            dtype=np.float32,
        )
        scale_frames.append(_warp(base, mat))
    scaled = track_adjacent(scale_frames, [20, 21, 22], 1, [0.35, 0.35, 0.65, 0.65])
    assert all(
        row is not None and row["status"] in {"tracked", "anchor"}
        for row in scaled["trajectory"]
    )
    left_width = scaled["trajectory"][0]["box_xyxy_norm"][2] - scaled["trajectory"][0]["box_xyxy_norm"][0]
    anchor_width = 0.30
    right_width = scaled["trajectory"][2]["box_xyxy_norm"][2] - scaled["trajectory"][2]["box_xyxy_norm"][0]
    assert left_width < anchor_width < right_width

    # Blank anchor has no Shi-Tomasi corners; later edge is not attempted.
    blank = np.zeros_like(base)
    lost = track_adjacent([blank, base, base], [7, 8, 9], 0, [0.25, 0.25, 0.75, 0.75])
    assert lost["trajectory"][1]["status"] == "missing"
    assert lost["transitions"][0]["reason"] == "insufficient_detected_corners"
    assert lost["transitions"][1]["attempted"] is False
    assert lost["trajectory"][2]["status"] == "missing"

    # Reaching outside the image is retained as a proposal and rejected, never clipped.
    edge_base = _synthetic_texture(320, 320, 991)
    edge_box = [0.69, 0.30, 0.95, 0.60]
    edge_next = _warp(edge_base, np.asarray([[1, 0, 20], [0, 1, 0]], dtype=np.float32))
    oob = track_adjacent([edge_base, edge_next], [40, 41], 0, edge_box)
    oob_transition = next(t for t in oob["transitions"] if t["attempted"])
    assert oob_transition["reason"] == "transformed_box_out_of_bounds", oob_transition
    assert oob_transition["proposed_box_xyxy_norm"][2] > 1.0
    assert oob_transition["accepted_box_xyxy_norm"] is None
    assert oob["trajectory"][1]["box_xyxy_norm"] is None

    # A second anchor is processed independently; the returned path starts at its own box.
    small_anchor = track_adjacent(frames, [100, 101, 102, 103, 104], 2, [0.44, 0.44, 0.56, 0.56])
    assert small_anchor["trajectory"][2]["box_xyxy_norm"] == [0.44, 0.44, 0.56, 0.56]
    assert small_anchor["trajectory"][3]["box_xyxy_norm"] != moving["trajectory"][3]["box_xyxy_norm"]

    repeat = track_adjacent(frames, [100, 101, 102, 103, 104], 2, [0.30, 0.30, 0.70, 0.70])
    assert _jsonable_equal(moving, repeat), "fixed seed/thread settings did not reproduce synthetic result"
    assert moving["schema"] == SCHEMA
    assert not any("gt" in key.lower() or "truth" in key.lower() for key in moving.keys())
    return {
        "status": "pass",
        "cpu_only": True,
        "real_data_read": False,
        "gt_read": False,
        "model_called": False,
        "contracts": {
            "translation_forward_and_backward": "pass",
            "affine_scale_changes_box_size": "pass",
            "missing_after_loss_stops_direction": "pass",
            "blank_anchor_missing": "pass",
            "out_of_bounds_proposal_rejected_without_clipping": "pass",
            "independent_anchor_paths": "pass",
            "fixed_seed_determinism": "pass",
            "no_gt_or_truth_input_or_output": "pass",
        },
        "synthetic_dimensions": {"translation": [int(h), int(w)], "oob": [320, 320]},
        "config": CONFIG,
        "scope": "synthetic generated arrays only; no real frames, GT, models, GPU, or project case files",
    }


__all__ = ["CONFIG", "SCHEMA", "synthetic_contracts", "track_adjacent"]
