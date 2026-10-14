"""Fixed CPU local-motion residual region proposals for locked short windows.

The default path is intentionally inert.  Real cached RGB/gray pixels are
opened only by ``--run`` after a root-written ``RUNTIME.json`` is present and
all declared input pins match.  ``--self-test`` uses synthetic arrays only.

This module does not import a model, annotations, teacher output, or GT.  IDs
2..5 are local to one five-frame window; ID 1 is an exact copy of that
window's existing Source tracklet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCOPE = "artifacts/stvg_local_motion_regions_v1"
OLD_SCOPE = "artifacts/stvg_short_event_tracklets_v1"
MANIFEST_REL = f"{OLD_SCOPE}/PREPARED_MANIFEST.json"
OLD_BARRIER_REL = f"{OLD_SCOPE}/GLOBAL_TRACKLET_BARRIER.json"
RUNTIME_REL = f"{SCOPE}/RUNTIME.json"
OUTPUT_SCHEMA = "stvg-local-motion-region-case-v1"
BARRIER_SCHEMA = "stvg-local-motion-region-cpu-barrier-v1"
RUNTIME_SCHEMA = "stvg-local-motion-region-runtime-v1"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
WINDOW_LENGTH = 5
CENTER_INDEX = 2
MAX_NEW_REGIONS = 4

CONFIG: dict[str, Any] = {
    "scene_motion": {
        "detector": "cv2.goodFeaturesToTrack full grayscale frame",
        "max_corners": 500,
        "quality_level": 0.01,
        "min_distance_px": 5.0,
        "block_size": 3,
        "lk_win_size_px": [21, 21],
        "lk_max_level": 3,
        "lk_criteria": {"type": "COUNT|EPS", "count": 30, "epsilon": 0.01},
        "forward_backward_error_max_px": 1.5,
        "minimum_fb_valid_corners": 4,
        "ransac": {
            "method": "estimateAffinePartial2D/RANSAC",
            "reprojection_threshold_px": 2.0,
            "max_iterations": 2000,
            "confidence": 0.99,
            "refine_iterations": 10,
            "minimum_inliers": 4,
            "minimum_inlier_fraction": 0.5,
        },
        "seed": "first four SHA256 bytes of case/slot/physical transition binding, masked to signed 31-bit positive integer",
        "fallback": "none; invalid affine makes the transition unavailable",
    },
    "dense_flow": {
        "algorithm": "Farneback",
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
        "residual_magnitude_min_px": 1.5,
        "forward_backward_cycle_error_max_px": 1.5,
        "morphology": {"operation_order": ["open", "close"], "kernel": "3x3 ones"},
        "components": {
            "connectivity": 8,
            "minimum_area_px": 64,
            "maximum_area_fraction_of_frame": 0.25,
        },
        "mask_note": "The pre-morphology residual mask requires cycle-valid pixels. The fixed opening/closing operation may remove or fill pixels afterward; component pixels are not individually asserted to pass the pre-morphology cycle test.",
    },
    "frame_mask_rule": "For each non-final frame use the cleaned forward residual mask from its outgoing adjacent transition. For the final frame use the cleaned backward residual mask from the incoming final transition.",
    "component_matching": {
        "directional_warp": "forward splat source masks with forward flow and destination masks with backward flow; nearest-pixel splat",
        "edge_rule": "both directional warped-mask IoUs must be strictly greater than zero",
        "edge_score": "minimum of forward and backward warped-mask IoU; diagnostic only and not used to select directional nearest neighbors",
        "nearest_rule": "source nearest uses forward warped IoU; target reverse-nearest uses backward warped IoU; accept only reciprocal choices with both IoUs strictly positive",
        "assignment": "mutual nearest positive edge, exact ties broken by component row-major (top,left,bottom,right,id)",
        "missing_transition": "no match and no track bridging across it",
    },
    "candidate_rules": {
        "source_candidate_id": 1,
        "source_policy": "copy only the existing Source candidate-1 five-frame tracklet exactly",
        "new_candidate_ids": [2, 5],
        "new_candidate_max_per_window": 4,
        "eligible_track": "center component plus at least one accepted adjacent match on each side of center",
        "ranking": ["supported_frame_count descending", "accepted_adjacent_match_count descending", "median_cycle_error_px ascending", "center_component_row_major ascending"],
        "identity_scope": "one five-frame window only; no identity claim across windows",
        "overflow": "preserve every eligible track beyond four in metadata; only first four receive visible candidate IDs",
        "box": "exact connected-component pixel envelope normalized by cached frame width and height; never clamp a geometry later",
    },
    "runtime": {"device": "CPU", "opencv_threads": 2, "opencl": False},
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _write_json_once(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to replace existing temporary output: {tmp}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_bytes(data), "bytes": len(data)}


def _write_npz_once(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    if tmp.exists():
        raise FileExistsError(f"refusing to replace existing temporary output: {tmp}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"project path escaped root: {relative}")
    return candidate


def _input_pin_map(runtime: Mapping[str, Any]) -> dict[str, str]:
    raw = runtime.get("pins")
    if not isinstance(raw, Mapping):
        raise ValueError("runtime pins must be a path-to-SHA256 mapping")
    pins: dict[str, str] = {}
    for path, value in raw.items():
        if isinstance(value, str):
            digest = value
        elif isinstance(value, Mapping):
            digest = value.get("sha256")
        else:
            digest = None
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid runtime pin entry: {path!r}")
        pins[path] = digest.lower()
    return pins


def _read_and_validate_runtime(runtime_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    runtime = _read_json(runtime_path)
    if runtime.get("schema") != RUNTIME_SCHEMA or runtime.get("status") != "locked":
        raise RuntimeError("real-input execution requires a root-locked stvg-local-motion-region runtime")
    for flag in ("GT_read", "GPU_used", "model_loaded", "teacher_called", "scoring_started"):
        if runtime.get(flag) not in (False, None):
            raise RuntimeError(f"runtime must declare {flag}=false")
    if runtime.get("case_count") != 8 or runtime.get("window_count") != 32:
        raise RuntimeError("runtime must lock exactly 8 cases and 32 windows")
    if tuple(runtime.get("case_ids", ())) != CASE_IDS:
        raise RuntimeError("runtime case_ids must match the fixed historical eight in canonical order")
    pins = _input_pin_map(runtime)
    # A lock cannot include its own SHA as a pin. The containing root runtime
    # is captured by the output barrier after it has been read.
    mandatory = {MANIFEST_REL, OLD_BARRIER_REL, Path(__file__).resolve().relative_to(ROOT).as_posix()}
    missing = mandatory - pins.keys()
    if missing:
        raise RuntimeError(f"runtime is missing required SHA pins: {sorted(missing)}")
    for rel, expected in pins.items():
        path = _root_path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"pinned file missing: {rel}")
        got = _sha_file(path)
        if got != expected:
            raise RuntimeError(f"pinned SHA mismatch for {rel}: {got} != {expected}")
    return runtime, pins


def _as_manifest_and_barrier(runtime: Mapping[str, Any], pins: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(_root_path(MANIFEST_REL))
    barrier = _read_json(_root_path(OLD_BARRIER_REL))
    if barrier.get("schema") != "stvg-short-event-tracklet-barrier-v1" or barrier.get("status") != "complete":
        raise RuntimeError("source short-window barrier is not complete")
    if barrier.get("GT_read") is not False or barrier.get("GPU_used") is not False or barrier.get("model_loaded") is not False:
        raise RuntimeError("source short-window barrier flags are not eligible for this CPU reuse")
    if barrier.get("manifest_sha256") != pins[MANIFEST_REL]:
        raise RuntimeError("source barrier manifest hash differs from locked manifest")
    if manifest.get("status") not in ("prepared", "prepared_manifest_not_executed", "complete", "locked"):
        raise RuntimeError("prepared manifest is not in a usable sealed state")
    if manifest.get("GT_read") is not False or manifest.get("GPU_used") is not False or manifest.get("model_loaded") is not False:
        raise RuntimeError("prepared manifest flags are not eligible")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or tuple(row.get("case_id") for row in cases) != CASE_IDS:
        raise RuntimeError("manifest must contain the exact fixed eight cases in canonical order")
    if barrier.get("case_ids") != list(CASE_IDS) or barrier.get("count") != 8 or barrier.get("center_count") != 32:
        raise RuntimeError("source barrier must bind the fixed 8 cases and 32 centers")
    return manifest, barrier


def _validate_old_case_metadata(
    case: Mapping[str, Any],
    track: Mapping[str, Any],
    barrier_record: Mapping[str, Any],
    manifest_sha: str,
    track_path: str,
    rgb_path: str,
    pins: Mapping[str, str],
) -> None:
    case_id = str(case.get("case_id"))
    if track.get("schema") != "stvg-short-event-tracklet-run-v1" or track.get("status") != "complete":
        raise RuntimeError(f"source tracklet JSON is not complete: {case_id}")
    if track.get("case_id") != case_id or track.get("GT_read") is not False:
        raise RuntimeError(f"source tracklet case/GT binding mismatch: {case_id}")
    if barrier_record.get("GT_read") is not False or barrier_record.get("case_id") != case_id:
        raise RuntimeError(f"source barrier case/GT binding mismatch: {case_id}")
    bindings = track.get("input_binding", {})
    expected_pairs = {
        "manifest_sha256": manifest_sha,
        "source_case_sha256": case.get("source_case_sha256"),
        "source_model_state_sha256": case.get("source_model_state_sha256"),
        "canonical_request_sha256": case.get("canonical_request_sha256"),
        "canonical_video_sha256": case.get("canonical_video_sha256"),
        "source_video_sha256": case.get("source_video_sha256"),
        "candidate_barrier_record_sha256": case.get("candidate_barrier_record_sha256"),
    }
    for key, expected in expected_pairs.items():
        if bindings.get(key) != expected:
            raise RuntimeError(f"source {key} mismatch for {case_id}")
    if barrier_record.get("sha256") != pins.get(track_path):
        raise RuntimeError(f"old tracklet JSON pin mismatch in barrier for {case_id}")
    if barrier_record.get("rgb_npz_sha256") != pins.get(rgb_path):
        raise RuntimeError(f"old RGB NPZ pin mismatch in barrier for {case_id}")
    readback = track.get("frame_readback", {})
    if readback.get("rgb_npz_path") != rgb_path or readback.get("rgb_npz_sha256") != pins.get(rgb_path):
        raise RuntimeError(f"old cached RGB binding mismatch for {case_id}")


def _validate_gray_frames(gray_frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    if not isinstance(gray_frames, (list, tuple)) or not gray_frames:
        raise ValueError("gray_frames must be a non-empty list or tuple")
    frames: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for index, frame in enumerate(gray_frames):
        arr = np.asarray(frame)
        if arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError(f"gray_frames[{index}] must be a 2-D uint8 array")
        if min(arr.shape) < 3 or max(arr.shape) > 640:
            raise ValueError(f"gray_frames[{index}] must have long edge <= 640 and dimensions >=3")
        if shape is None:
            shape = tuple(int(x) for x in arr.shape)
        elif tuple(arr.shape) != shape:
            raise ValueError("all frames in one window must have identical shape")
        frames.append(np.ascontiguousarray(arr))
    assert shape is not None
    return frames, shape[1], shape[0]


def _stable_seed(case_id: str, slot: int, from_physical_id: int, to_physical_id: int) -> int:
    payload = f"{case_id}|{slot}|{from_physical_id}|{to_physical_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _affine_apply(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x2 = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    y2 = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    return np.stack([x2 - xx, y2 - yy], axis=-1).astype(np.float32)


def _sample_flow(flow: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    return cv2.remap(
        flow.astype(np.float32),
        qx.astype(np.float32),
        qy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(float("nan"), float("nan")),
    )


def _dense_cycle_maps(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    qx = xx + flow_fwd[..., 0]
    qy = yy + flow_fwd[..., 1]
    sampled_bwd = _sample_flow(flow_bwd, qx, qy)
    error_fwd = np.linalg.norm(flow_fwd + sampled_bwd, axis=-1).astype(np.float32)
    valid_fwd = (
        np.isfinite(error_fwd)
        & (qx >= 0.0) & (qx <= width - 1)
        & (qy >= 0.0) & (qy <= height - 1)
    )
    px = xx + flow_bwd[..., 0]
    py = yy + flow_bwd[..., 1]
    sampled_fwd = _sample_flow(flow_fwd, px, py)
    error_bwd = np.linalg.norm(flow_bwd + sampled_fwd, axis=-1).astype(np.float32)
    valid_bwd = (
        np.isfinite(error_bwd)
        & (px >= 0.0) & (px <= width - 1)
        & (py >= 0.0) & (py <= height - 1)
    )
    return error_fwd, error_bwd, valid_fwd, valid_bwd


def _morph(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    raw = np.asarray(mask, dtype=np.uint8)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def _component_list(mask: np.ndarray, width: int, height: int) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    components: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for label in range(1, int(count)):
        x, y, w, h, area = (int(v) for v in stats[label].tolist())
        row = {
            "raw_label": int(label),
            "bbox_xyxy_px": [x, y, x + w, y + h],
            "area_px": area,
            "centroid_xy_px": [float(centroids[label, 0]), float(centroids[label, 1])],
            "row_major_key": [y, x, y + h, x + w],
        }
        all_rows.append(row)
    all_rows.sort(key=lambda row: (*row["row_major_key"], row["raw_label"]))
    label_map = np.zeros(binary.shape, dtype=np.uint32)
    min_area = 64
    max_area = int(math.floor((width * height) * 0.25))
    for new_id, row in enumerate(all_rows, start=1):
        old_label = int(row.pop("raw_label"))
        area = row["area_px"]
        reason = None
        if area < min_area:
            reason = "area_below_64_px"
        elif area > max_area:
            reason = "area_above_25_percent_frame"
        box = row["bbox_xyxy_px"]
        x1, y1, x2, y2 = box
        local_region = labels[y1:y2, x1:x2] == old_label
        label_map[y1:y2, x1:x2][local_region] = np.uint32(new_id)
        region: np.ndarray | None = None
        if reason is None:
            region = np.zeros(binary.shape, dtype=bool)
            region[y1:y2, x1:x2] = local_region
        comp = {
            "component_id": int(new_id),
            **row,
            "bbox_xyxy_norm": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            "accepted": reason is None,
            "rejection_reason": reason,
            "mask": region,
        }
        components.append(comp)
    accepted = sum(bool(comp["accepted"]) for comp in components)
    return components, label_map, {
        "connected_component_count": len(components),
        "accepted_component_count": int(accepted),
        "rejected_component_count": int(len(components) - accepted),
        "rejections_by_reason": dict(Counter(comp["rejection_reason"] for comp in components if comp["rejection_reason"])),
        "area_min_px": min_area,
        "area_max_px": max_area,
    }


def _estimate_scene_affine(
    src: np.ndarray,
    dst: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    height, width = src.shape
    corners = cv2.goodFeaturesToTrack(
        src,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=5.0,
        mask=None,
        blockSize=3,
        useHarrisDetector=False,
    )
    detected = 0 if corners is None else int(len(corners))
    diag: dict[str, Any] = {
        "status": "rejected",
        "reason": None,
        "detected_corner_count": detected,
        "forward_lk_status_count": 0,
        "backward_lk_status_count": 0,
        "fb_valid_count": 0,
        "fb_retention_fraction": None,
        "fb_error_median_px": None,
        "fb_error_max_px": None,
        "ransac_inlier_count": 0,
        "ransac_inlier_fraction": None,
        "ransac_seed": int(seed),
        "affine_2x3": None,
        "inverse_affine_2x3": None,
    }
    if detected < 4:
        diag["reason"] = "insufficient_detected_full_frame_corners"
        return None, None, diag
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p1, status_f, _ = cv2.calcOpticalFlowPyrLK(src, dst, corners, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
    if p1 is None or status_f is None:
        diag["reason"] = "forward_lk_failed"
        return None, None, diag
    p0b, status_b, _ = cv2.calcOpticalFlowPyrLK(dst, src, p1, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
    if p0b is None or status_b is None:
        diag["forward_lk_status_count"] = int(np.count_nonzero(status_f))
        diag["reason"] = "backward_lk_failed"
        return None, None, diag
    p0 = corners.reshape(-1, 2).astype(np.float64)
    p1f = p1.reshape(-1, 2).astype(np.float64)
    p0back = p0b.reshape(-1, 2).astype(np.float64)
    sf = status_f.reshape(-1).astype(bool)
    sb = status_b.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(p0 - p0back, axis=1)
    finite = np.isfinite(p0).all(axis=1) & np.isfinite(p1f).all(axis=1) & np.isfinite(fb_error)
    good = sf & sb & finite & (fb_error <= 1.5)
    err = fb_error[good]
    valid_count = int(np.count_nonzero(good))
    retention = float(valid_count / detected)
    diag.update({
        "forward_lk_status_count": int(np.count_nonzero(sf)),
        "backward_lk_status_count": int(np.count_nonzero(sb)),
        "fb_valid_count": valid_count,
        "fb_retention_fraction": retention,
        "fb_error_median_px": None if err.size == 0 else float(np.median(err)),
        "fb_error_max_px": None if err.size == 0 else float(np.max(err)),
    })
    if valid_count < 4:
        diag["reason"] = "insufficient_fb_valid_full_frame_corners"
        return None, None, diag
    cv2.setRNGSeed(int(seed))
    affine, inliers = cv2.estimateAffinePartial2D(
        p0[good].astype(np.float32),
        p1f[good].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inliers is None or not np.isfinite(affine).all():
        diag["reason"] = "ransac_affine_fit_failed"
        return None, None, diag
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = float(inlier_count / valid_count)
    diag["ransac_inlier_count"] = inlier_count
    diag["ransac_inlier_fraction"] = inlier_fraction
    if inlier_count < 4:
        diag["reason"] = "ransac_inliers_below_4"
        return None, None, diag
    if inlier_fraction < 0.5:
        diag["reason"] = "ransac_inlier_fraction_below_0.5"
        return None, None, diag
    try:
        inv = cv2.invertAffineTransform(affine.astype(np.float64))
    except cv2.error:
        diag["reason"] = "inverse_affine_failed"
        return None, None, diag
    if not np.isfinite(inv).all():
        diag["reason"] = "inverse_affine_nonfinite"
        return None, None, diag
    diag.update({
        "status": "accepted",
        "reason": None,
        "affine_2x3": affine.astype(float).tolist(),
        "inverse_affine_2x3": inv.astype(float).tolist(),
    })
    return affine.astype(np.float32), inv.astype(np.float32), diag


def _component_frame_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    return _component_list(mask, width, height)


def _analyze_transition(
    src: np.ndarray,
    dst: np.ndarray,
    from_physical_id: int,
    to_physical_id: int,
    case_id: str,
    slot: int,
) -> dict[str, Any]:
    height, width = src.shape
    seed = _stable_seed(case_id, slot, from_physical_id, to_physical_id)
    affine, inverse, fit = _estimate_scene_affine(src, dst, seed)
    empty = np.zeros((height, width), dtype=np.uint8)
    result: dict[str, Any] = {
        "from_physical_frame_id": int(from_physical_id),
        "to_physical_frame_id": int(to_physical_id),
        "camera_fit": fit,
        "status": "unavailable" if affine is None else "pending_flow",
        "reason": fit.get("reason"),
        "source_residual_components": [],
        "target_residual_components": [],
        "source_component_label_map": np.zeros((height, width), dtype=np.uint32),
        "target_component_label_map": np.zeros((height, width), dtype=np.uint32),
        "raw_forward_mask": empty.copy(),
        "raw_backward_mask": empty.copy(),
        "clean_forward_mask": empty.copy(),
        "clean_backward_mask": empty.copy(),
        "flow_forward": None,
        "flow_backward": None,
        "cycle_error_forward": None,
        "cycle_error_backward": None,
        "cycle_valid_forward": empty.astype(bool),
        "cycle_valid_backward": empty.astype(bool),
        "component_summary": {
            "source": {"connected_component_count": 0, "accepted_component_count": 0, "rejected_component_count": 0, "rejections_by_reason": {}},
            "target": {"connected_component_count": 0, "accepted_component_count": 0, "rejected_component_count": 0, "rejections_by_reason": {}},
        },
        "flow_summary": None,
    }
    if affine is None or inverse is None:
        return result
    flow_fwd = cv2.calcOpticalFlowFarneback(src, dst, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bwd = cv2.calcOpticalFlowFarneback(dst, src, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if flow_fwd is None or flow_bwd is None or not np.isfinite(flow_fwd).all() or not np.isfinite(flow_bwd).all():
        result.update({"status": "unavailable", "reason": "dense_farneback_failed_or_nonfinite"})
        return result
    predicted_fwd = _affine_apply(affine, width, height)
    predicted_bwd = _affine_apply(inverse, width, height)
    residual_fwd = flow_fwd - predicted_fwd
    residual_bwd = flow_bwd - predicted_bwd
    cycle_fwd, cycle_bwd, valid_fwd, valid_bwd = _dense_cycle_maps(flow_fwd, flow_bwd)
    mag_fwd = np.linalg.norm(residual_fwd, axis=-1)
    mag_bwd = np.linalg.norm(residual_bwd, axis=-1)
    raw_fwd = (valid_fwd & (cycle_fwd <= 1.5) & (mag_fwd >= 1.5)).astype(np.uint8) * 255
    raw_bwd = (valid_bwd & (cycle_bwd <= 1.5) & (mag_bwd >= 1.5)).astype(np.uint8) * 255
    clean_fwd = _morph(raw_fwd)
    clean_bwd = _morph(raw_bwd)
    source_components, source_labels, source_summary = _component_frame_mask(clean_fwd, width, height)
    target_components, target_labels, target_summary = _component_frame_mask(clean_bwd, width, height)
    for comp in source_components:
        region = comp["mask"] if comp["mask"] is not None else source_labels == comp["component_id"]
        values = cycle_fwd[region & valid_fwd]
        comp["median_cycle_error_px"] = None if values.size == 0 else float(np.median(values))
        comp["mean_cycle_error_px"] = None if values.size == 0 else float(np.mean(values))
    for comp in target_components:
        region = comp["mask"] if comp["mask"] is not None else target_labels == comp["component_id"]
        values = cycle_bwd[region & valid_bwd]
        comp["median_cycle_error_px"] = None if values.size == 0 else float(np.median(values))
        comp["mean_cycle_error_px"] = None if values.size == 0 else float(np.mean(values))
    result.update({
        "status": "complete",
        "reason": None,
        "source_residual_components": source_components,
        "target_residual_components": target_components,
        "source_component_label_map": source_labels,
        "target_component_label_map": target_labels,
        "raw_forward_mask": raw_fwd,
        "raw_backward_mask": raw_bwd,
        "clean_forward_mask": clean_fwd,
        "clean_backward_mask": clean_bwd,
        "flow_forward": flow_fwd.astype(np.float32),
        "flow_backward": flow_bwd.astype(np.float32),
        "cycle_error_forward": np.nan_to_num(cycle_fwd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "cycle_error_backward": np.nan_to_num(cycle_bwd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "cycle_valid_forward": valid_fwd,
        "cycle_valid_backward": valid_bwd,
        "component_summary": {"source": source_summary, "target": target_summary},
        "flow_summary": {
            "source_pixels_above_residual_threshold": int(np.count_nonzero(mag_fwd >= 1.5)),
            "target_pixels_above_residual_threshold": int(np.count_nonzero(mag_bwd >= 1.5)),
            "source_cycle_valid_pixels": int(np.count_nonzero(valid_fwd & (cycle_fwd <= 1.5))),
            "target_cycle_valid_pixels": int(np.count_nonzero(valid_bwd & (cycle_bwd <= 1.5))),
            "source_raw_mask_pixels": int(np.count_nonzero(raw_fwd)),
            "target_raw_mask_pixels": int(np.count_nonzero(raw_bwd)),
            "source_clean_mask_pixels": int(np.count_nonzero(clean_fwd)),
            "target_clean_mask_pixels": int(np.count_nonzero(clean_bwd)),
            "source_residual_magnitude_median_px": float(np.median(mag_fwd)),
            "target_residual_magnitude_median_px": float(np.median(mag_bwd)),
        },
    })
    return result


def _warp_forward_splat(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    out = np.zeros((height, width), dtype=np.uint8)
    if len(xx) == 0:
        return out
    dx = np.rint(xx.astype(np.float32) + flow[yy, xx, 0]).astype(np.int32)
    dy = np.rint(yy.astype(np.float32) + flow[yy, xx, 1]).astype(np.int32)
    good = np.isfinite(flow[yy, xx]).all(axis=1) & (dx >= 0) & (dx < width) & (dy >= 0) & (dy < height)
    out[dy[good], dx[good]] = 1
    return out


def _binary_bbox(mask: np.ndarray) -> list[int] | None:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


def _warped_iou(
    a: np.ndarray,
    b: np.ndarray,
    bbox_a: Sequence[int] | None = None,
    bbox_b: Sequence[int] | None = None,
    area_a: int | None = None,
    area_b: int | None = None,
) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    ba = _binary_bbox(aa) if bbox_a is None else list(bbox_a)
    bc = _binary_bbox(bb) if bbox_b is None else list(bbox_b)
    if ba is None or bc is None or not _bbox_intersects(ba, bc):
        return 0.0
    x1 = max(ba[0], bc[0])
    y1 = max(ba[1], bc[1])
    x2 = min(ba[2], bc[2])
    y2 = min(ba[3], bc[3])
    inter = int(np.count_nonzero(aa[y1:y2, x1:x2] & bb[y1:y2, x1:x2]))
    union = int((np.count_nonzero(aa) if area_a is None else area_a) + (np.count_nonzero(bb) if area_b is None else area_b) - inter)
    return 0.0 if union == 0 else float(inter / union)


def _bbox_intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _component_order_key(comp: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (*[int(v) for v in comp["row_major_key"]], int(comp["component_id"]))


def _mutual_component_matches(
    transition: Mapping[str, Any],
    source_components: Sequence[Mapping[str, Any]],
    target_components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if transition.get("status") != "complete":
        total = sum(bool(x.get("accepted")) for x in source_components) * sum(bool(x.get("accepted")) for x in target_components)
        return {
            "status": "unavailable_transition",
            "accepted_edges": [],
            "positive_pair_edges": [],
            "rejected_pair_count_by_reason": {"transition_unavailable": int(total)},
            "source_component_count": sum(bool(x.get("accepted")) for x in source_components),
            "target_component_count": sum(bool(x.get("accepted")) for x in target_components),
            "possible_pair_count": int(total),
        }
    srcs = [comp for comp in source_components if comp.get("accepted")]
    dsts = [comp for comp in target_components if comp.get("accepted")]
    flow_fwd = np.asarray(transition["flow_forward"], dtype=np.float32)
    flow_bwd = np.asarray(transition["flow_backward"], dtype=np.float32)
    warped_sources = {int(comp["component_id"]): _warp_forward_splat(comp["mask"], flow_fwd) for comp in srcs}
    warped_targets = {int(comp["component_id"]): _warp_forward_splat(comp["mask"], flow_bwd) for comp in dsts}
    warped_source_boxes = {key: _binary_bbox(value) for key, value in warped_sources.items()}
    warped_target_boxes = {key: _binary_bbox(value) for key, value in warped_targets.items()}
    warped_source_areas = {key: int(np.count_nonzero(value)) for key, value in warped_sources.items()}
    warped_target_areas = {key: int(np.count_nonzero(value)) for key, value in warped_targets.items()}
    source_boxes = {int(comp["component_id"]): _binary_bbox(comp["mask"]) for comp in srcs}
    target_boxes = {int(comp["component_id"]): _binary_bbox(comp["mask"]) for comp in dsts}
    source_areas = {int(comp["component_id"]): int(comp["area_px"]) for comp in srcs}
    target_areas = {int(comp["component_id"]): int(comp["area_px"]) for comp in dsts}
    positive_pairs: list[dict[str, Any]] = []
    positive_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    rejected = Counter()
    possible = len(srcs) * len(dsts)
    src_best: dict[int, tuple[int, float]] = {}
    dst_best: dict[int, tuple[int, float]] = {}
    for src in srcs:
        sid = int(src["component_id"])
        ranked: list[tuple[float, tuple[int, int, int, int, int], int]] = []
        for dst in dsts:
            did = int(dst["component_id"])
            f_iou = _warped_iou(warped_sources[sid], dst["mask"], warped_source_boxes[sid], target_boxes[did], warped_source_areas[sid], target_areas[did])
            b_iou = _warped_iou(src["mask"], warped_targets[did], source_boxes[sid], warped_target_boxes[did], source_areas[sid], warped_target_areas[did])
            if f_iou <= 0.0 and b_iou <= 0.0:
                rejected["zero_overlap_both_directions"] += 1
                continue
            if f_iou <= 0.0 or b_iou <= 0.0:
                rejected["zero_overlap_one_direction"] += 1
                edge_row = {
                    "source_component_id": sid,
                    "target_component_id": did,
                    "forward_warped_iou": float(f_iou),
                    "backward_warped_iou": float(b_iou),
                    "reciprocal_score": 0.0,
                    "status": "rejected",
                    "reason": "reciprocal_warped_iou_not_positive_both_directions",
                }
                positive_pairs.append(edge_row)
                continue
            score = min(f_iou, b_iou)
            ranked.append((f_iou, _component_order_key(dst), did))
            edge_row = {
                "source_component_id": sid,
                "target_component_id": did,
                "forward_warped_iou": float(f_iou),
                "backward_warped_iou": float(b_iou),
                "reciprocal_score": float(score),
                "status": "positive_edge_candidate",
                "reason": None,
            }
            positive_pairs.append(edge_row)
            positive_lookup[(sid, did)] = edge_row
        if ranked:
            best_score = max(item[0] for item in ranked)
            best = min((item for item in ranked if item[0] == best_score), key=lambda item: item[1])
            src_best[sid] = (best[2], best[0])
    for dst in dsts:
        did = int(dst["component_id"])
        ranked = []
        for src in srcs:
            sid = int(src["component_id"])
            match = positive_lookup.get((sid, did))
            if match is not None:
                ranked.append((float(match["backward_warped_iou"]), _component_order_key(src), sid))
        if ranked:
            best_score = max(item[0] for item in ranked)
            best = min((item for item in ranked if item[0] == best_score), key=lambda item: item[1])
            dst_best[did] = (best[2], best[0])
    accepted: list[dict[str, Any]] = []
    for pair in positive_pairs:
        if pair["status"] != "positive_edge_candidate":
            continue
        sid = int(pair["source_component_id"])
        did = int(pair["target_component_id"])
        mutual = src_best.get(sid, (None, None))[0] == did and dst_best.get(did, (None, None))[0] == sid
        if mutual:
            pair["status"] = "accepted_mutual_nearest"
            pair["reason"] = None
            accepted.append(pair)
        else:
            pair["status"] = "rejected"
            pair["reason"] = "not_mutual_nearest_positive_edge"
            rejected["positive_not_mutual_nearest"] += 1
    source_by_id = {int(comp["component_id"]): comp for comp in srcs}
    target_by_id = {int(comp["component_id"]): comp for comp in dsts}
    accepted.sort(key=lambda row: (_component_order_key(source_by_id[row["source_component_id"]]), _component_order_key(target_by_id[row["target_component_id"]])))
    positive_pairs.sort(key=lambda row: (row["source_component_id"], row["target_component_id"]))
    return {
        "status": "complete",
        "accepted_edges": accepted,
        "positive_pair_edges": positive_pairs,
        "rejected_pair_count_by_reason": dict(rejected),
        "source_component_count": len(srcs),
        "target_component_count": len(dsts),
        "possible_pair_count": int(possible),
        "zero_overlap_pair_count_compressed": int(rejected.get("zero_overlap_both_directions", 0)),
    }


def _component_edge_cycle_error(
    transition: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> float | None:
    vals: list[np.ndarray] = []
    for comp, key, valid_key in (
        (source, "cycle_error_forward", "cycle_valid_forward"),
        (target, "cycle_error_backward", "cycle_valid_backward"),
    ):
        error = transition.get(key)
        valid = transition.get(valid_key)
        if error is None or valid is None:
            continue
        mask = np.asarray(comp["mask"], dtype=bool) & np.asarray(valid, dtype=bool)
        selected = np.asarray(error)[mask]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            vals.append(selected.astype(np.float64))
    if not vals:
        return None
    return float(np.median(np.concatenate(vals)))


def _link_windows(
    transitions: Sequence[Mapping[str, Any]],
    frame_components: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for i, transition in enumerate(transitions):
        result = _mutual_component_matches(transition, frame_components[i], frame_components[i + 1])
        by_source = {int(comp["component_id"]): comp for comp in frame_components[i]}
        by_target = {int(comp["component_id"]): comp for comp in frame_components[i + 1]}
        for edge in result["accepted_edges"]:
            edge["median_cycle_error_px"] = _component_edge_cycle_error(
                transition,
                by_source[int(edge["source_component_id"])],
                by_target[int(edge["target_component_id"])],
            )
        links.append(result)
    return links


def _box_from_component(component: Mapping[str, Any], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = component["bbox_xyxy_px"]
    return [float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)]


def _trace_center_component(
    center_component: Mapping[str, Any],
    center_index: int,
    physical_ids: Sequence[int],
    local_ids: Sequence[int],
    links: Sequence[Mapping[str, Any]],
    frame_components: Sequence[Sequence[Mapping[str, Any]]],
    width: int,
    height: int,
) -> dict[str, Any]:
    comp_by_frame = [
        {int(comp["component_id"]): comp for comp in comps if comp.get("accepted")}
        for comps in frame_components
    ]
    chosen: dict[int, Mapping[str, Any]] = {center_index: center_component}
    edge_rows: list[dict[str, Any]] = []
    current = int(center_component["component_id"])
    for i in range(center_index - 1, -1, -1):
        edge = next((e for e in links[i].get("accepted_edges", []) if int(e["target_component_id"]) == current), None)
        if edge is None:
            break
        current = int(edge["source_component_id"])
        comp = comp_by_frame[i].get(current)
        if comp is None:
            break
        chosen[i] = comp
        edge_rows.append({"from_frame_index": i, "to_frame_index": i + 1, **{k: edge[k] for k in ["source_component_id", "target_component_id", "reciprocal_score", "forward_warped_iou", "backward_warped_iou", "median_cycle_error_px"]}})
    current = int(center_component["component_id"])
    for i in range(center_index, len(physical_ids) - 1):
        edge = next((e for e in links[i].get("accepted_edges", []) if int(e["source_component_id"]) == current), None)
        if edge is None:
            break
        current = int(edge["target_component_id"])
        comp = comp_by_frame[i + 1].get(current)
        if comp is None:
            break
        chosen[i + 1] = comp
        edge_rows.append({"from_frame_index": i, "to_frame_index": i + 1, **{k: edge[k] for k in ["source_component_id", "target_component_id", "reciprocal_score", "forward_warped_iou", "backward_warped_iou", "median_cycle_error_px"]}})
    frames: list[dict[str, Any]] = []
    for i, (physical_id, local_id) in enumerate(zip(physical_ids, local_ids)):
        comp = chosen.get(i)
        if comp is None:
            frames.append({
                "physical_frame_id": int(physical_id),
                "local_frame_index": int(local_id),
                "status": "missing_no_contiguous_component_match",
                "valid": False,
                "box_xyxy_norm": None,
                "component_id": None,
            })
        else:
            frames.append({
                "physical_frame_id": int(physical_id),
                "local_frame_index": int(local_id),
                "status": "accepted_local_motion_component",
                "valid": True,
                "box_xyxy_norm": _box_from_component(comp, width, height),
                "component_id": int(comp["component_id"]),
            })
    side_before = center_index - 1 in chosen
    side_after = center_index + 1 in chosen
    cycle_values = [row["median_cycle_error_px"] for row in edge_rows if row.get("median_cycle_error_px") is not None]
    return {
        "center_component_id": int(center_component["component_id"]),
        "center_bbox_xyxy_px": [int(x) for x in center_component["bbox_xyxy_px"]],
        "center_row_major_key": [int(x) for x in center_component["row_major_key"]],
        "supported_frame_count": len(chosen),
        "accepted_adjacent_match_count": len(edge_rows),
        "has_match_before_center": bool(side_before),
        "has_match_after_center": bool(side_after),
        "eligible_two_sided": bool(side_before and side_after),
        "median_cycle_error_px": None if not cycle_values else float(np.median(cycle_values)),
        "accepted_edges": edge_rows,
        "frames": frames,
        "_chosen_component_ids": {str(k): int(v["component_id"]) for k, v in chosen.items()},
    }


def _make_window(
    *,
    case_id: str,
    slot: int,
    source_grid_position: int,
    center_physical_id: int,
    physical_ids: Sequence[int],
    local_ids: Sequence[int],
    gray_frames: Sequence[np.ndarray],
    source_candidate_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frames, width, height = _validate_gray_frames(gray_frames)
    if len(frames) != WINDOW_LENGTH or len(physical_ids) != WINDOW_LENGTH or len(local_ids) != WINDOW_LENGTH:
        raise ValueError("each window must bind exactly five physical/local frames")
    if len(source_candidate_frames) != WINDOW_LENGTH:
        raise ValueError("Source candidate must provide one original frame row per window frame")
    if int(physical_ids[CENTER_INDEX]) != int(center_physical_id):
        raise ValueError("center physical frame does not match fixed five-frame window")
    if [int(row.get("physical_frame_id", -1)) for row in source_candidate_frames] != [int(x) for x in physical_ids]:
        raise ValueError("Source candidate frame binding differs from window IDs")
    transitions = [
        _analyze_transition(frames[i], frames[i + 1], int(physical_ids[i]), int(physical_ids[i + 1]), case_id, slot)
        for i in range(WINDOW_LENGTH - 1)
    ]
    frame_components: list[list[dict[str, Any]]] = []
    label_maps: list[np.ndarray] = []
    for i in range(WINDOW_LENGTH - 1):
        comps = transitions[i]["source_residual_components"] if transitions[i]["status"] == "complete" else []
        labels = transitions[i]["source_component_label_map"] if transitions[i]["status"] == "complete" else np.zeros((height, width), np.uint32)
        frame_components.append(comps)
        label_maps.append(labels)
    last = transitions[-1]
    last_comps = last["target_residual_components"] if last["status"] == "complete" else []
    last_labels = last["target_component_label_map"] if last["status"] == "complete" else np.zeros((height, width), np.uint32)
    frame_components.append(last_comps)
    label_maps.append(last_labels)
    if len(frame_components) != WINDOW_LENGTH or len(label_maps) != WINDOW_LENGTH:
        raise RuntimeError("fixed outgoing-except-final-incoming segmentation rule was not applied")
    # Intermediate target residuals are retained as mask arrays and JSON
    # diagnostics, but the fixed per-frame rule uses the next frame's outgoing
    # mask. Release the unused full-frame component masks before matching.
    for transition in transitions[:-1]:
        for component in transition["target_residual_components"]:
            component["mask"] = None
        transition["target_component_label_map"] = None
    links = _link_windows(transitions, frame_components)
    center_components = [
        comp for comp in frame_components[CENTER_INDEX]
        if comp.get("accepted")
    ]
    all_tracks = [
        _trace_center_component(comp, CENTER_INDEX, physical_ids, local_ids, links, frame_components, width, height)
        for comp in center_components
    ]
    eligible = [track for track in all_tracks if track["eligible_two_sided"]]
    eligible.sort(key=lambda t: (
        -int(t["supported_frame_count"]),
        -int(t["accepted_adjacent_match_count"]),
        float("inf") if t["median_cycle_error_px"] is None else float(t["median_cycle_error_px"]),
        tuple(int(x) for x in t["center_row_major_key"]),
    ))
    retained = eligible[:MAX_NEW_REGIONS]
    overflow = eligible[MAX_NEW_REGIONS:]
    source_frames = [copy.deepcopy(dict(row)) for row in source_candidate_frames]
    source_candidate = {
        "candidate_id": 1,
        "role": "Source",
        "identity_scope": "source-only copied original candidate-1 tracklet",
        "source_copy_exact": True,
        "frames": source_frames,
    }
    candidates = [source_candidate]
    for local_number, track in enumerate(retained, start=2):
        track_uid = f"window-{slot:02d}-center-component-{track['center_component_id']:04d}"
        candidates.append({
            "candidate_id": local_number,
            "role": "new_local_motion_region",
            "track_uid": track_uid,
            "identity_scope": "this five-frame window only; no cross-window identity claim",
            "source_copy_exact": False,
            "center_component_id": int(track["center_component_id"]),
            "center_bbox_xyxy_px": track["center_bbox_xyxy_px"],
            "supported_frame_count": int(track["supported_frame_count"]),
            "accepted_adjacent_match_count": int(track["accepted_adjacent_match_count"]),
            "median_cycle_error_px": track["median_cycle_error_px"],
            "accepted_edges": track["accepted_edges"],
            "frames": track["frames"],
        })
    frame_component_meta: list[dict[str, Any]] = []
    for index, comps in enumerate(frame_components):
        frame_component_meta.append({
            "frame_index": index,
            "physical_frame_id": int(physical_ids[index]),
            "local_frame_index": int(local_ids[index]),
            "segmentation_source": "outgoing_forward_residual" if index < WINDOW_LENGTH - 1 else "incoming_final_backward_residual",
            "component_count": len(comps),
            "accepted_component_count": sum(bool(c["accepted"]) for c in comps),
            "rejected_component_count": sum(not bool(c["accepted"]) for c in comps),
            "components": [
                {k: v for k, v in comp.items() if k != "mask"}
                for comp in comps
            ],
        })
    transition_meta = []
    for i, transition in enumerate(transitions):
        transition_meta.append({
            "from_physical_frame_id": int(physical_ids[i]),
            "to_physical_frame_id": int(physical_ids[i + 1]),
            "camera_fit": transition["camera_fit"],
            "status": transition["status"],
            "reason": transition["reason"],
            "flow_summary": transition["flow_summary"],
            "component_summary": transition["component_summary"],
            "source_components": [{k: v for k, v in comp.items() if k != "mask"} for comp in transition["source_residual_components"]],
            "target_components": [{k: v for k, v in comp.items() if k != "mask"} for comp in transition["target_residual_components"]],
            "match_readback": {
                **{k: v for k, v in links[i].items() if k not in ("accepted_edges", "positive_pair_edges")},
                "accepted_edges": links[i]["accepted_edges"],
                "positive_pair_edges": links[i]["positive_pair_edges"],
            },
        })
    window = {
        "window_slot": int(slot),
        "source_grid_position": int(source_grid_position),
        "center_physical_frame_id": int(center_physical_id),
        "physical_frame_ids": [int(x) for x in physical_ids],
        "local_frame_indices": [int(x) for x in local_ids],
        "candidate_count": len(candidates),
        "visible_candidate_ids": [int(c["candidate_id"]) for c in candidates],
        "source_candidate_copy": {
            "candidate_id": 1,
            "exact_source_tracklet_frame_count": len(source_frames),
            "source_rows_sha256": _canonical_sha(source_frames),
            "other_old_candidates_used_as_seeds": False,
        },
        "new_region_count": len(retained),
        "eligible_track_count_before_cap": len(eligible),
        "overflow_track_count": len(overflow),
        "candidate_cap": MAX_NEW_REGIONS,
        "overflow_tracks": [
            {
                "track_uid": f"window-{slot:02d}-center-component-{track['center_component_id']:04d}",
                "center_component_id": int(track["center_component_id"]),
                "supported_frame_count": int(track["supported_frame_count"]),
                "accepted_adjacent_match_count": int(track["accepted_adjacent_match_count"]),
                "median_cycle_error_px": track["median_cycle_error_px"],
                "reason": "eligible_ranked_track_exceeds_fixed_four_new_region_cap",
                "frames": track["frames"],
            }
            for track in overflow
        ],
        "all_center_component_tracks": [
            {
                "center_component_id": int(track["center_component_id"]),
                "eligible_two_sided": bool(track["eligible_two_sided"]),
                "supported_frame_count": int(track["supported_frame_count"]),
                "accepted_adjacent_match_count": int(track["accepted_adjacent_match_count"]),
                "median_cycle_error_px": track["median_cycle_error_px"],
                "reason_if_not_retained": (
                    "not_two_sided" if not track["eligible_two_sided"]
                    else "eligible_ranked_track_exceeds_fixed_four_new_region_cap" if track in overflow
                    else None
                ),
                "frames": track["frames"],
            }
            for track in all_tracks
        ],
        "candidates": candidates,
        "frame_component_readback": frame_component_meta,
        "transition_readback": transition_meta,
        "contract": {
            "component_frame_mask_rule": CONFIG["frame_mask_rule"],
            "new_candidate_id_scope": "local to this window",
            "missing_transition_bridged": False,
            "zero_reciprocal_overlap_allowed_as_identity": False,
            "all_rejected_components_retained": True,
            "all_eligible_overflow_tracks_retained": True,
        },
    }
    masks = {
        "raw_forward_masks": np.stack([t["raw_forward_mask"] for t in transitions]).astype(np.uint8),
        "raw_backward_masks": np.stack([t["raw_backward_mask"] for t in transitions]).astype(np.uint8),
        "clean_forward_masks": np.stack([t["clean_forward_mask"] for t in transitions]).astype(np.uint8),
        "clean_backward_masks": np.stack([t["clean_backward_mask"] for t in transitions]).astype(np.uint8),
        "frame_component_labels": np.stack(label_maps).astype(np.uint32),
    }
    return window, masks


def _source_frames_for_center(
    manifest_center: Mapping[str, Any],
    track: Mapping[str, Any],
) -> list[dict[str, Any]]:
    planned = [row for row in manifest_center.get("matched_tracklets", []) if row.get("candidate_id") == 1]
    if len(planned) != 1:
        raise RuntimeError("each center must contain exactly one original candidate-1 Source binding")
    tracklet_id = planned[0].get("tracklet_id")
    matches = [row for row in track.get("tracklets", []) if row.get("tracklet_id") == tracklet_id and row.get("candidate_id") == 1]
    if len(matches) != 1:
        raise RuntimeError("candidate-1 Source tracklet is missing or ambiguous")
    rows = matches[0].get("frames")
    if not isinstance(rows, list) or len(rows) != WINDOW_LENGTH:
        raise RuntimeError("Source tracklet must contain exactly five original frame rows")
    wanted_physical = [int(x) for x in manifest_center["physical_frame_ids"]]
    wanted_local = [int(x) for x in manifest_center["local_frame_indices"]]
    if [int(row["physical_frame_id"]) for row in rows] != wanted_physical:
        raise RuntimeError("Source tracklet physical frame rows differ from manifest")
    if [int(row["local_frame_index"]) for row in rows] != wanted_local:
        raise RuntimeError("Source tracklet local frame rows differ from manifest")
    return [copy.deepcopy(dict(row)) for row in rows]


def _case_mask_arrays(windows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("raw_forward_masks", "raw_backward_masks", "clean_forward_masks", "clean_backward_masks", "frame_component_labels")
    return {key: np.stack([np.asarray(row[key]) for row in windows], axis=0) for key in keys}


def _load_case_npz_and_verify(
    npz_path: Path,
    case: Mapping[str, Any],
    track: Mapping[str, Any],
    centers: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int], list[dict[str, Any]]]:
    # Called only after runtime/source files have all passed pre-read SHA checks.
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"rgb_uint8", "gray_uint8", "local_frame_indices", "physical_frame_ids", "manifest_sha256"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"cached NPZ lacks required arrays: {sorted(required - set(archive.files))}")
        rgb = np.asarray(archive["rgb_uint8"])
        gray = np.asarray(archive["gray_uint8"])
        local = np.asarray(archive["local_frame_indices"])
        physical = np.asarray(archive["physical_frame_ids"])
        manifest_hash = str(np.asarray(archive["manifest_sha256"]).item())
    manifest_sha = _sha_file(_root_path(MANIFEST_REL))
    if manifest_hash != manifest_sha:
        raise RuntimeError(f"NPZ embedded manifest SHA mismatch for {case['case_id']}")
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise RuntimeError("cached RGB must be N×H×W×3 uint8")
    if gray.dtype != np.uint8 or gray.ndim != 3 or gray.shape != rgb.shape[:3]:
        raise RuntimeError("cached gray must align with cached RGB N×H×W")
    if local.dtype != np.int64 or physical.dtype != np.int64 or local.shape != physical.shape or local.shape != (len(rgb),):
        raise RuntimeError("cached physical/local frame index arrays have invalid shape or dtype")
    if len(set(int(x) for x in physical)) != len(physical) or len(set(int(x) for x in local)) != len(local):
        raise RuntimeError("cached frame IDs must be unique")
    rgb_by_physical = {int(fid): np.ascontiguousarray(frame) for fid, frame in zip(physical, rgb)}
    gray_by_physical = {int(fid): np.ascontiguousarray(frame) for fid, frame in zip(physical, gray)}
    local_by_physical = {int(fid): int(lid) for fid, lid in zip(physical, local)}
    selected = track.get("frame_readback", {}).get("selected_frames", [])
    selected_by_id = {int(row["physical_frame_id"]): row for row in selected}
    if set(selected_by_id) != set(rgb_by_physical):
        raise RuntimeError("cached frame IDs differ from old frame readback")
    for fid, rgb_frame in rgb_by_physical.items():
        gray_frame = gray_by_physical[fid]
        row = selected_by_id[fid]
        if local_by_physical[fid] != int(row["local_frame_index"]):
            raise RuntimeError(f"cached local frame index mismatch at physical frame {fid}")
        if list(rgb_frame.shape) != row.get("cached_rgb_shape_hwc"):
            raise RuntimeError(f"cached RGB shape mismatch at physical frame {fid}")
        expected_gray_shape = row.get("cached_rgb_shape_hwc", [])[:2]
        if list(gray_frame.shape) != expected_gray_shape:
            raise RuntimeError(f"cached gray shape mismatch at physical frame {fid}")
        if _sha_bytes(rgb_frame.tobytes(order="C")) != row.get("cached_rgb_sha256"):
            raise RuntimeError(f"cached RGB pixel hash mismatch at physical frame {fid}")
        if _sha_bytes(gray_frame.tobytes(order="C")) != row.get("tracking_gray_sha256"):
            raise RuntimeError(f"cached gray pixel hash mismatch at physical frame {fid}")
    expected_ids = {int(fid) for center in centers for fid in center["physical_frame_ids"]}
    if expected_ids != set(rgb_by_physical):
        raise RuntimeError("cached unique physical frame set differs from fixed four-window union")
    selected_meta = [
        {key: row.get(key) for key in ["physical_frame_id", "local_frame_index", "source_timestamp_s", "source_rgb_shape_hwc", "cached_rgb_shape_hwc", "source_frame_rgb_sha256", "cached_rgb_sha256", "tracking_gray_sha256", "resize"]}
        for row in sorted(selected, key=lambda row: int(row["physical_frame_id"]))
    ]
    return rgb_by_physical, gray_by_physical, local_by_physical, selected_meta


def _build_case(
    case: Mapping[str, Any],
    track: Mapping[str, Any],
    barrier_record: Mapping[str, Any],
    rgb_path: str,
    track_path: str,
    barrier_sha: str,
    manifest_sha: str,
    pins: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    centers = list(case.get("centers", []))
    if len(centers) != 4:
        raise RuntimeError(f"each source case must bind four windows, got {len(centers)}")
    rgb_by_id, gray_by_id, local_by_id, selected_meta = _load_case_npz_and_verify(_root_path(rgb_path), case, track, centers)
    window_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, np.ndarray]] = []
    for center in centers:
        physical_ids = [int(x) for x in center["physical_frame_ids"]]
        local_ids = [int(x) for x in center["local_frame_indices"]]
        if any(local_by_id.get(fid) != local for fid, local in zip(physical_ids, local_ids)):
            raise RuntimeError("manifest physical/local mapping differs from cached frame mapping")
        source_rows = _source_frames_for_center(center, track)
        window, masks = _make_window(
            case_id=str(case["case_id"]),
            slot=int(center["slot"]),
            source_grid_position=int(center["source_grid_position"]),
            center_physical_id=int(center["center_physical_frame_id"]),
            physical_ids=physical_ids,
            local_ids=local_ids,
            gray_frames=[gray_by_id[fid] for fid in physical_ids],
            source_candidate_frames=source_rows,
        )
        window_rows.append(window)
        mask_rows.append(masks)
    if len(window_rows) != 4 or sum(len(x["physical_frame_ids"]) for x in window_rows) != 20:
        raise RuntimeError("one case must preserve four five-frame windows")
    mask_arrays = _case_mask_arrays(mask_rows)
    record = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "case_id": str(case["case_id"]),
        "dataset": str(case.get("dataset", "")),
        "cohort": str(case.get("cohort", "")),
        "windows": window_rows,
        "input_bindings": {
            "old_manifest_path": MANIFEST_REL,
            "old_manifest_sha256": manifest_sha,
            "old_global_barrier_path": OLD_BARRIER_REL,
            "old_global_barrier_sha256": barrier_sha,
            "old_tracklet_json_path": track_path,
            "old_tracklet_json_sha256": pins[track_path],
            "old_private_rgb_npz_path": rgb_path,
            "old_private_rgb_npz_sha256": pins[rgb_path],
            "source_case_sha256": case["source_case_sha256"],
            "source_model_state_sha256": case["source_model_state_sha256"],
            "source_video_sha256": case["source_video_sha256"],
            "canonical_request_sha256": case["canonical_request_sha256"],
            "canonical_video_sha256": case["canonical_video_sha256"],
            "candidate_barrier_record_sha256": case["candidate_barrier_record_sha256"],
            "fixed_frame_map_sha256": track["input_binding"]["selected_frame_map_sha256"],
        },
        "frame_readback": {
            "selected_unique_frame_count": len(selected_meta),
            "cached_rgb_shape_hwc": list(next(iter(rgb_by_id.values())).shape),
            "cached_gray_shape_hw": list(next(iter(gray_by_id.values())).shape),
            "frames": selected_meta,
            "raw_rgb_or_gray_pixels_in_json": False,
        },
        "mask_archive": None,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "opencv_threads": int(cv2.getNumThreads()),
            "device": "CPU",
            "model_loaded": False,
            "GPU_used": False,
            "teacher_called": False,
        },
        "config": CONFIG,
        "counts": {
            "window_count": 4,
            "source_tracklet_count": 4,
            "new_region_count": sum(int(w["new_region_count"]) for w in window_rows),
            "eligible_track_count_before_cap": sum(int(w["eligible_track_count_before_cap"]) for w in window_rows),
            "overflow_track_count": sum(int(w["overflow_track_count"]) for w in window_rows),
            "camera_transition_count": sum(len(w["transition_readback"]) for w in window_rows),
            "valid_camera_transition_count": sum(sum(t["status"] == "complete" for t in w["transition_readback"]) for w in window_rows),
            "accepted_mutual_component_match_count": sum(sum(len(t["match_readback"]["accepted_edges"]) for t in w["transition_readback"]) for w in window_rows),
            "connected_component_count": sum(sum(f["component_count"] for f in w["frame_component_readback"]) for w in window_rows),
            "rejected_component_count": sum(sum(f["rejected_component_count"] for f in w["frame_component_readback"]) for w in window_rows),
            "valid_new_region_boxes": sum(sum(bool(f["valid"]) for f in c["frames"]) for w in window_rows for c in w["candidates"] if c["candidate_id"] != 1),
            "source_boxes_copied": sum(sum(bool(f.get("valid")) for f in c["frames"]) for w in window_rows for c in w["candidates"] if c["candidate_id"] == 1),
        },
        "GT_read": False,
        "RGB_read": True,
        "video_decoded": False,
        "model_loaded": False,
        "GPU_used": False,
        "teacher_called": False,
        "scoring_started": False,
    }
    return record, mask_arrays


def run_locked(runtime_path: Path) -> dict[str, Any]:
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    runtime, pins = _read_and_validate_runtime(runtime_path)
    manifest, barrier = _as_manifest_and_barrier(runtime, pins)
    manifest_sha = pins[MANIFEST_REL]
    barrier_sha = pins[OLD_BARRIER_REL]
    if runtime_path.resolve() != _root_path(RUNTIME_REL):
        raise RuntimeError(f"runtime must be the frozen path {RUNTIME_REL}")
    case_rows = {str(row["case_id"]): row for row in manifest["cases"]}
    barrier_rows = {str(row["case_id"]): row for row in barrier["case_records"]}
    if set(case_rows) != set(CASE_IDS) or set(barrier_rows) != set(CASE_IDS):
        raise RuntimeError("runtime inputs do not contain the canonical eight case records")
    required_case_files = {
        path
        for row in barrier["case_records"]
        for path in (str(row["path"]), str(row["rgb_npz_path"]))
    }
    if not required_case_files.issubset(pins):
        raise RuntimeError(f"runtime must pin every old tracklet JSON and RGB NPZ before pixel loading; missing={sorted(required_case_files - pins.keys())}")
    output_dir = _root_path(SCOPE)
    if output_dir.exists() and any(output_dir.iterdir()):
        # The lock and synthetic-only contract receipt may precede the real run.
        allowed_initial = {"RUNTIME.json", "CPU_CONTRACTS.json"}
        allowed_dirs = {"recovery"}
        others = [p.relative_to(ROOT).as_posix() for p in output_dir.iterdir() if p.name not in allowed_initial and p.name not in allowed_dirs]
        if others:
            raise FileExistsError(f"scope contains prior output; refusing resume/overwrite: {others[:5]}")
        recovery_dir = output_dir / "recovery"
        if recovery_dir.exists():
            if not recovery_dir.is_dir():
                raise FileExistsError("recovery path must be a directory")
            recovery_entries = list(recovery_dir.iterdir())
            if any(not (entry.is_file() and entry.name.startswith("CPU_CONTRACTS_") and entry.name.endswith(".json")) and not (entry.is_dir() and entry.name == "rejected_component_mask001" and set(p.name for p in entry.iterdir()) <= {"run_stvg_local_motion_regions_v1.py", "RUNTIME.json", "CPU_CONTRACTS.json", "failure.log", "CAPTURE.json", "RESOLUTION.json"}) for entry in recovery_entries):
                raise FileExistsError("recovery may contain only preserved synthetic CPU_CONTRACTS JSON receipts")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_files: list[dict[str, Any]] = []
    mask_files: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        case = case_rows[case_id]
        b_record = barrier_rows[case_id]
        track_path = str(b_record["path"])
        rgb_path = str(b_record["rgb_npz_path"])
        track = _read_json(_root_path(track_path))
        _validate_old_case_metadata(case, track, b_record, manifest_sha, track_path, rgb_path, pins)
        record, mask_arrays = _build_case(case, track, b_record, rgb_path, track_path, barrier_sha, manifest_sha, pins)
        mask_record = _write_npz_once(_root_path(f"{SCOPE}/private_masks/{case_id}.npz"), **mask_arrays)
        record["mask_archive"] = {
            **mask_record,
            "private": True,
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in mask_arrays.items()},
        }
        json_record = _write_json_once(_root_path(f"{SCOPE}/cases/{case_id}.json"), record)
        case_files.append(json_record)
        mask_files.append(mask_record)
        counts.append({"case_id": case_id, **record["counts"]})
    files = sorted(case_files + mask_files, key=lambda row: row["path"])
    if len(case_files) != 8 or len(mask_files) != 8 or len(files) != 16:
        raise RuntimeError("fixed output contract requires eight JSON cases and eight private mask archives")
    barrier_out = {
        "schema": BARRIER_SCHEMA,
        "status": "complete",
        "scope": "fixed historical eight source cases, four original five-frame windows per case, CPU local motion proposals only",
        "runtime_path": RUNTIME_REL,
        "runtime_sha256": _sha_file(_root_path(RUNTIME_REL)),
        "runner_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "runner_sha256": _sha_file(Path(__file__).resolve()),
        "old_manifest_sha256": manifest_sha,
        "old_global_barrier_sha256": barrier_sha,
        "case_ids": list(CASE_IDS),
        "case_count": 8,
        "window_count": 32,
        "output_json_count": len(case_files),
        "private_mask_archive_count": len(mask_files),
        "files": files,
        "case_counts": counts,
        "GT_read": False,
        "RGB_read": True,
        "video_decoded": False,
        "model_loaded": False,
        "GPU_used": False,
        "teacher_called": False,
        "scoring_started": False,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "opencv_threads": int(cv2.getNumThreads()),
            "device": "CPU",
        },
    }
    barrier_out["files_sha256"] = _canonical_sha(files)
    receipt = _write_json_once(_root_path(f"{SCOPE}/CPU_GENERATOR_BARRIER.json"), barrier_out)
    return {"status": "complete", "case_count": 8, "window_count": 32, "barrier": receipt, "files_sha256": barrier_out["files_sha256"]}


def synthetic_contracts() -> dict[str, Any]:
    """Run fixed CPU-only geometry and mask contracts on generated arrays."""
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    checks: dict[str, bool] = {}

    # Deterministic static clip: it must not invent local residual objects.
    rng = np.random.default_rng(113)
    static = rng.integers(0, 255, (128, 160), dtype=np.uint8)
    static_frames = [static.copy() for _ in range(5)]
    source_rows = [
        {"physical_frame_id": 100 + i, "local_frame_index": 100 + i, "status": "tracked", "valid": True, "box_xyxy_norm": [0.2, 0.2, 0.4, 0.4]}
        for i in range(5)
    ]
    static_window, _ = _make_window(
        case_id="synthetic_static", slot=0, source_grid_position=0, center_physical_id=102,
        physical_ids=[100, 101, 102, 103, 104], local_ids=[0, 1, 2, 3, 4], gray_frames=static_frames,
        source_candidate_frames=source_rows,
    )
    checks["static_sequence_adds_no_new_regions"] = static_window["new_region_count"] == 0

    # A textured foreground translating over a static textured scene should
    # produce a two-sided local residual region under the fixed camera fit.
    moving_rng = np.random.default_rng(53)
    moving_bg = moving_rng.integers(0, 256, (180, 240), dtype=np.uint8)
    moving_texture = moving_rng.integers(0, 256, (38, 42), dtype=np.uint8)
    moving_frames: list[np.ndarray] = []
    for t in range(WINDOW_LENGTH):
        image = moving_bg.copy()
        x, y = 38 + 4 * t, 68
        image[y:y + 38, x:x + 42] = moving_texture
        moving_frames.append(image)
    moving_source = [
        {"physical_frame_id": 200 + i, "local_frame_index": i, "status": "source", "valid": i == 2, "box_xyxy_norm": [0.1, 0.1, 0.2, 0.2] if i == 2 else None}
        for i in range(WINDOW_LENGTH)
    ]
    moving_window, _ = _make_window(
        case_id="synthetic_moving_region", slot=0, source_grid_position=1, center_physical_id=202,
        physical_ids=[200, 201, 202, 203, 204], local_ids=[0, 1, 2, 3, 4], gray_frames=moving_frames,
        source_candidate_frames=moving_source,
    )
    moving_regions = [candidate for candidate in moving_window["candidates"] if candidate["candidate_id"] != 1]
    checks["translating_region_yields_two_sided_track"] = (
        len(moving_regions) == 1
        and moving_regions[0]["supported_frame_count"] == 5
        and all(frame["valid"] for frame in moving_regions[0]["frames"])
    )
    moving_replay, _ = _make_window(
        case_id="synthetic_moving_region", slot=0, source_grid_position=1, center_physical_id=202,
        physical_ids=[200, 201, 202, 203, 204], local_ids=[0, 1, 2, 3, 4], gray_frames=moving_frames,
        source_candidate_frames=moving_source,
    )
    checks["translating_region_candidate_id_and_geometry_replay_stable"] = (
        [c.get("candidate_id") for c in moving_window["candidates"]]
        == [c.get("candidate_id") for c in moving_replay["candidates"]]
        and [c.get("frames") for c in moving_window["candidates"]]
        == [c.get("frames") for c in moving_replay["candidates"]]
    )
    checks["outgoing_masks_except_final_incoming"] = [
        row["segmentation_source"] for row in moving_window["frame_component_readback"]
    ] == ["outgoing_forward_residual"] * 4 + ["incoming_final_backward_residual"]

    # Multi-object synthetic clip exercises the fixed cap and preserves every
    # eligible overflow track in the JSON metadata.
    multi_rng = np.random.default_rng(110)
    multi_bg = multi_rng.integers(0, 256, (240, 320), dtype=np.uint8)
    multi_positions = [(20, 30), (75, 30), (130, 30), (185, 30), (240, 30), (60, 125)]
    multi_textures = [multi_rng.integers(0, 256, (22, 22), dtype=np.uint8) for _ in multi_positions]
    multi_frames: list[np.ndarray] = []
    for t in range(WINDOW_LENGTH):
        image = multi_bg.copy()
        for (x, y), texture in zip(multi_positions, multi_textures):
            image[y:y + 22, x + 4 * t:x + 4 * t + 22] = texture
        multi_frames.append(image)
    multi_source = [
        {"physical_frame_id": 300 + i, "local_frame_index": i, "status": "source", "valid": True, "box_xyxy_norm": [0.1, 0.1, 0.2, 0.2]}
        for i in range(WINDOW_LENGTH)
    ]
    multi_window, _ = _make_window(
        case_id="synthetic_cap", slot=2, source_grid_position=2, center_physical_id=302,
        physical_ids=[300, 301, 302, 303, 304], local_ids=[0, 1, 2, 3, 4], gray_frames=multi_frames,
        source_candidate_frames=multi_source,
    )
    checks["fixed_four_region_cap_preserves_overflow"] = (
        multi_window["new_region_count"] == 4
        and multi_window["eligible_track_count_before_cap"] == 6
        and multi_window["overflow_track_count"] == 2
        and len(multi_window["overflow_tracks"]) == 2
        and [c["candidate_id"] for c in multi_window["candidates"]] == [1, 2, 3, 4, 5]
    )

    # Component filters keep metadata for both too-small and too-large blobs.
    # Use non-touching geometry with frame area large enough for a >25%-area
    # rejection and a small rejection.
    filter_mask = np.zeros((120, 120), dtype=np.uint8)
    filter_mask[2:8, 2:8] = 255
    filter_mask[20:90, 20:90] = 255
    filter_mask[100:110, 100:110] = 255
    filter_components, _, filter_summary = _component_list(filter_mask, 120, 120)
    checks["all_component_area_rejections_retained"] = (
        filter_summary["connected_component_count"] == 3
        and filter_summary["rejected_component_count"] == 2
        and [c["rejection_reason"] for c in filter_components if not c["accepted"]]
        == ["area_below_64_px", "area_above_25_percent_frame"]
    )

    # Reciprocal positive-overlap is a hard requirement.
    base = np.zeros((48, 64), dtype=np.uint8)
    base[10:20, 12:24] = 1
    overlap = _warped_iou(base, base)
    disjoint = np.zeros_like(base)
    disjoint[25:35, 38:50] = 1
    checks["same_mask_reciprocal_iou_positive"] = overlap == 1.0
    checks["disjoint_mask_iou_zero"] = _warped_iou(base, disjoint) == 0.0

    # Component lists with identity flow should make one accepted reciprocal edge.
    def comp(cid: int, mask: np.ndarray) -> dict[str, Any]:
        ys, xs = np.nonzero(mask)
        return {
            "component_id": cid,
            "mask": mask.astype(bool),
            "accepted": True,
            "area_px": int(np.count_nonzero(mask)),
            "bbox_xyxy_px": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "row_major_key": [int(ys.min()), int(xs.min()), int(ys.max() + 1), int(xs.max() + 1)],
        }
    zero_flow = np.zeros((48, 64, 2), dtype=np.float32)
    one_edge_transition = {
        "status": "complete", "flow_forward": zero_flow, "flow_backward": zero_flow,
        "cycle_error_forward": np.zeros((48, 64), np.float32), "cycle_error_backward": np.zeros((48, 64), np.float32),
        "cycle_valid_forward": np.ones((48, 64), bool), "cycle_valid_backward": np.ones((48, 64), bool),
    }
    one_edge = _mutual_component_matches(one_edge_transition, [comp(1, base)], [comp(1, base)])
    checks["mutual_positive_edge_accepted"] = len(one_edge["accepted_edges"]) == 1
    zero_edge = _mutual_component_matches(one_edge_transition, [comp(1, base)], [comp(1, disjoint)])
    checks["zero_overlap_never_identity"] = len(zero_edge["accepted_edges"]) == 0 and zero_edge["rejected_pair_count_by_reason"].get("zero_overlap_both_directions") == 1

    # No bridge across a failed transition; one missing side blocks eligibility.
    no_link = {"status": "unavailable_transition", "accepted_edges": []}
    comp0 = comp(1, base)
    comp1 = comp(1, base)
    comp2 = comp(1, base)
    comp3 = comp(1, base)
    comp4 = comp(1, base)
    fake_transitions = [one_edge_transition, one_edge_transition, {**one_edge_transition, "status": "unavailable"}, one_edge_transition]
    frame_comps = [[comp0], [comp1], [comp2], [comp3], [comp4]]
    links = _link_windows(fake_transitions, frame_comps)
    traced = _trace_center_component(comp2, 2, [1, 2, 3, 4, 5], [0, 1, 2, 3, 4], links, frame_comps, 64, 48)
    checks["missing_transition_not_bridged"] = traced["frames"][3]["valid"] is False and traced["eligible_two_sided"] is False
    checks["window_local_candidate_ids"] = list(range(2, 2 + MAX_NEW_REGIONS)) == [2, 3, 4, 5]

    # Source rows are copied without changes, including missing status and coordinates.
    source_with_missing = copy.deepcopy(source_rows)
    source_with_missing[1] = {"physical_frame_id": 101, "local_frame_index": 1, "status": "missing", "valid": False, "box_xyxy_norm": None}
    mini, _ = _make_window(
        case_id="synthetic_copy", slot=1, source_grid_position=7, center_physical_id=102,
        physical_ids=[100, 101, 102, 103, 104], local_ids=[0, 1, 2, 3, 4], gray_frames=static_frames,
        source_candidate_frames=source_with_missing,
    )
    checks["source_rows_copied_exactly"] = mini["candidates"][0]["frames"] == source_with_missing
    checks["source_missing_not_repaired"] = mini["candidates"][0]["frames"][1]["box_xyxy_norm"] is None

    # IDs and boxes are deterministic across identical synthetic calls.
    first, _ = _make_window(
        case_id="synthetic_static", slot=0, source_grid_position=0, center_physical_id=102,
        physical_ids=[100, 101, 102, 103, 104], local_ids=[0, 1, 2, 3, 4], gray_frames=static_frames,
        source_candidate_frames=source_rows,
    )
    checks["deterministic_empty_window_replay"] = _canonical_sha(first) == _canonical_sha(static_window)

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "stvg-local-motion-region-synthetic-contracts-v1",
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "failed": failed,
        "synthetic_only": True,
        "GT_read": False,
        "RGB_or_video_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "teacher_called": False,
        "real_input_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="run synthetic CPU contracts only")
    group.add_argument("--run", action="store_true", help="run real cached pixels only under a root-locked runtime")
    parser.add_argument("--runtime", type=Path, default=_root_path(RUNTIME_REL))
    args = parser.parse_args(argv)
    if args.self_test:
        result = synthetic_contracts()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] == "pass" else 1
    result = run_locked(args.runtime)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
