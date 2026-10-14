"""Lock local_R1 and appearance-associated_R1 references from sealed inputs.

The executable lock path is intended only after the frozen feature-capture and
64-call teacher barriers are complete.  ``--synthetic-contracts`` exercises
the pooling, deterministic mutual-nearest association, and no-reference rules
without opening any project artifact or model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
MANIFEST_PATH = BASE / "PREPARED_MANIFEST.json"
TRACK_BARRIER_PATH = BASE / "GLOBAL_TRACKLET_BARRIER.json"
FEATURE_BARRIER_PATH = BASE / "FEATURE_CAPTURE_BARRIER.json"
INPUT_BARRIER_PATH = BASE / "CANDIDATE_INPUT_BARRIER.json"
JOB_LOCK_PATH = BASE / "TEACHER_JOB_LOCK.json"
TEACHER_BARRIER_PATH = BASE / "TEACHER_GLOBAL_BARRIER.json"
TEACHER_RUNTIME_PATH = BASE / "RUNTIME.json"
TRACK_DIR = BASE / "tracklets"
FEATURE_DIR = BASE / "features"
TEACHER_DIR = BASE / "teacher"
OUTPUT_PATH = BASE / "METHOD_REFERENCE_LOCK.json"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
SCHEMA = "stvg-short-window-reference-lock-v1"
TOL = 1e-12


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("artifact path escaped the repository root")
    return resolved


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_bytes(raw), "bytes": len(raw)}


def _legal_xyxy(box: Sequence[float]) -> np.ndarray:
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("box must be four finite normalized xyxy values")
    x1, y1, x2, y2 = values.tolist()
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("box must be strictly legal; it is never clipped")
    return values


def xyxy_to_cxcywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = _legal_xyxy(box).tolist()
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def area_overlap_roi_pool(feature_chw: np.ndarray, box_xyxy_norm: Sequence[float]) -> np.ndarray:
    """Area-overlap pool a normalized original-frame box on a CxHxW map."""
    feature = np.asarray(feature_chw, dtype=np.float64)
    if feature.ndim != 3 or min(feature.shape) <= 0 or not np.isfinite(feature).all():
        raise ValueError("appearance map must be a finite CxHxW array")
    x1, y1, x2, y2 = _legal_xyxy(box_xyxy_norm).tolist()
    channels, height, width = feature.shape
    xe = np.arange(width + 1, dtype=np.float64) / width
    ye = np.arange(height + 1, dtype=np.float64) / height
    wx = np.maximum(0.0, np.minimum(x2, xe[1:]) - np.maximum(x1, xe[:-1]))
    wy = np.maximum(0.0, np.minimum(y2, ye[1:]) - np.maximum(y1, ye[:-1]))
    weights = wy[:, None] * wx[None, :]
    denominator = float(weights.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("legal box has no area-overlap support on the feature map")
    pooled = np.einsum("chw,hw->c", feature, weights, optimize=True) / denominator
    if pooled.shape != (channels,) or not np.isfinite(pooled).all():
        raise ValueError("ROI pooled descriptor is not finite")
    return pooled


def tracklet_descriptor(
    frames: Sequence[Mapping[str, Any]], feature_by_physical: Mapping[int, np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Equal-frame mean of per-frame L2-normalized ROI descriptors."""
    unit_rows: list[np.ndarray] = []
    included: list[int] = []
    zero_norm: list[int] = []
    for row in frames:
        if row.get("valid") is not True or row.get("box_xyxy_norm") is None:
            continue
        fid = int(row["physical_frame_id"])
        if fid not in feature_by_physical:
            raise ValueError("valid tracklet frame has no physical-ID-bound appearance map")
        pooled = area_overlap_roi_pool(feature_by_physical[fid], row["box_xyxy_norm"])
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm):
            raise ValueError("pooled descriptor norm is not finite")
        if norm <= TOL:
            zero_norm.append(fid)
            continue
        unit_rows.append(pooled / norm)
        included.append(fid)
    if not unit_rows:
        return None, {"valid_roi_frame_count": 0, "used_physical_frame_ids": [],
                      "zero_norm_physical_frame_ids": zero_norm,
                      "status": "no_nonzero_valid_roi_descriptor"}
    mean = np.mean(np.stack(unit_rows, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or float(np.linalg.norm(mean)) <= TOL:
        return None, {"valid_roi_frame_count": len(unit_rows), "used_physical_frame_ids": included,
                      "zero_norm_physical_frame_ids": zero_norm,
                      "status": "mean_descriptor_canceled_to_zero"}
    return mean, {"valid_roi_frame_count": len(unit_rows), "used_physical_frame_ids": included,
                  "zero_norm_physical_frame_ids": zero_norm, "status": "valid"}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine vectors must have matching finite one-dimensional shapes")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= TOL:
        raise ValueError("zero-norm descriptor cannot participate in cosine matching")
    return float(np.dot(a, b) / denom)


def _nearest_with_id_tie(scores: Mapping[int, float]) -> int:
    if not scores:
        raise ValueError("cannot choose nearest neighbor from an empty pool")
    maximum = max(float(v) for v in scores.values())
    tied = [int(cid) for cid, score in scores.items() if maximum - float(score) <= TOL]
    return min(tied)


def mutual_nearest_edges(
    left_by_candidate: Mapping[int, np.ndarray | None],
    right_by_candidate: Mapping[int, np.ndarray | None],
) -> list[dict[str, Any]]:
    """Mutual maximum cosine edges; ties resolve to the smallest candidate ID."""
    left = {int(k): np.asarray(v, dtype=np.float64) for k, v in left_by_candidate.items() if v is not None}
    right = {int(k): np.asarray(v, dtype=np.float64) for k, v in right_by_candidate.items() if v is not None}
    if not left or not right:
        return []
    scores = {(a, b): _cosine(left[a], right[b]) for a in sorted(left) for b in sorted(right)}
    left_nn = {a: _nearest_with_id_tie({b: scores[a, b] for b in right}) for a in left}
    right_nn = {b: _nearest_with_id_tie({a: scores[a, b] for a in left}) for b in right}
    return [
        {"left_candidate_id": a, "right_candidate_id": b, "cosine": scores[a, b]}
        for a, b in sorted(scores)
        if left_nn[a] == b and right_nn[b] == a
    ]


def _tensor_sha256(tensor: Any, torch_module: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha_bytes(value.view(torch_module.uint8).numpy().tobytes())


def _load_feature_case(
    case_id: str, barrier_entry: Mapping[str, Any], expected: Mapping[str, Any],
    manifest_case: Mapping[str, Any], track_case: Mapping[str, Any], torch_module: Any,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    rel = str(barrier_entry["path"])
    path = _safe_path(rel)
    if path.parent != FEATURE_DIR.resolve() or not path.is_file():
        raise ValueError(f"{case_id}: feature path escaped the frozen features directory")
    if _sha_file(path) != barrier_entry.get("sha256") or path.stat().st_size != int(barrier_entry["bytes"]):
        raise ValueError(f"{case_id}: frozen feature file SHA/size differs from its barrier")
    feature = torch_module.load(path, map_location="cpu", weights_only=True)
    if not isinstance(feature, Mapping) or set(feature) != {"schema", "metadata", "frame_ids", "appearance_maps"}:
        raise ValueError(f"{case_id}: frozen feature payload keys differ from the CPU-safe ABI")
    if feature.get("schema") != "stvg-short-window-frozen-appearance-v1":
        raise ValueError(f"{case_id}: unsupported frozen feature schema")
    metadata = feature["metadata"]
    if not isinstance(metadata, Mapping) or metadata.get("GT_read") is not False:
        raise ValueError(f"{case_id}: feature metadata is not explicitly GT-free")
    if metadata.get("candidate_proposals_created") is not False or metadata.get("backward_calls") != 0:
        raise ValueError(f"{case_id}: feature capture performed a disallowed task")
    for key, expected_value in (
        ("case_id", case_id),
        ("source_case_sha256", expected["source_case_sha256"]),
        ("source_model_state_sha256", expected["source_model_state_sha256"]),
        ("manifest_sha256", expected["manifest_sha256"]),
        ("manifest_case_sha256", _canonical_sha(manifest_case)),
        ("canonical_request_sha256", expected["canonical_request_sha256"]),
        ("canonical_video_sha256", expected["canonical_video_sha256"]),
    ):
        if metadata.get(key) != expected_value:
            raise ValueError(f"{case_id}: frozen feature {key} differs from the track/manifest binding")
    frame_ids = [int(v) for v in feature["frame_ids"]]
    expected_frame_ids = sorted({int(v) for center in track_case["centers"] for v in center["physical_frame_ids"]})
    if frame_ids != expected_frame_ids or frame_ids != sorted(set(frame_ids)):
        raise ValueError(f"{case_id}: feature map physical IDs do not equal the short-window union")
    appearance = feature["appearance_maps"]
    if not isinstance(appearance, torch_module.Tensor) or appearance.device.type != "cpu" or appearance.requires_grad:
        raise ValueError(f"{case_id}: feature maps must be detached CPU tensors")
    if appearance.ndim != 4 or tuple(appearance.shape[:2]) != (len(frame_ids), 256):
        raise ValueError(f"{case_id}: feature maps must have shape N x 256 x h x w")
    if not torch_module.isfinite(appearance).all().item():
        raise ValueError(f"{case_id}: appearance maps contain nonfinite values")
    if list(map(int, metadata.get("feature_maps_shape", []))) != list(appearance.shape):
        raise ValueError(f"{case_id}: feature map shape differs from metadata")
    if metadata.get("feature_maps_sha256") != _tensor_sha256(appearance, torch_module):
        raise ValueError(f"{case_id}: feature tensor bytes differ from their per-case digest")
    offsets = metadata.get("native_offsets")
    if not isinstance(offsets, Mapping) or set(offsets) != {"0", "1"}:
        raise ValueError(f"{case_id}: feature metadata does not bind both native offsets")
    expected_offsets = {"0": frame_ids[::2], "1": frame_ids[1::2]}
    if {str(k): list(map(int, v)) for k, v in offsets.items()} != expected_offsets:
        raise ValueError(f"{case_id}: physical feature merge does not match native-offset frame IDs")
    expected_local_by_physical = {
        int(row["physical_frame_id"]): int(row["local_frame_index"])
        for row in track_case["frame_readback"]["selected_frames"]
    }
    if list(map(int, metadata.get("manifest_local_frame_ids", []))) != [expected_local_by_physical[v] for v in frame_ids]:
        raise ValueError(f"{case_id}: feature local-frame IDs do not align with the physical IDs")
    maps = appearance.detach().cpu().contiguous().numpy().astype(np.float64, copy=False)
    by_physical = {fid: maps[index] for index, fid in enumerate(frame_ids)}
    return by_physical, {"frame_ids": frame_ids, "shape": list(appearance.shape),
                         "feature_sha256": barrier_entry["sha256"],
                         "feature_maps_sha256": metadata["feature_maps_sha256"],
                         "manifest_case_sha256": metadata["manifest_case_sha256"]}


def _load_and_validate_seals() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if OUTPUT_PATH.exists():
        raise FileExistsError("METHOD_REFERENCE_LOCK already exists; refusing overwrite")
    manifest = _read_json(MANIFEST_PATH)
    track_barrier = _read_json(TRACK_BARRIER_PATH)
    feature_barrier = _read_json(FEATURE_BARRIER_PATH)
    input_barrier = _read_json(INPUT_BARRIER_PATH)
    job_lock = _read_json(JOB_LOCK_PATH)
    teacher_barrier = _read_json(TEACHER_BARRIER_PATH)
    runtime = _read_json(TEACHER_RUNTIME_PATH)
    manifest_sha, track_barrier_sha = _sha_file(MANIFEST_PATH), _sha_file(TRACK_BARRIER_PATH)
    feature_barrier_sha, input_barrier_sha = _sha_file(FEATURE_BARRIER_PATH), _sha_file(INPUT_BARRIER_PATH)
    job_lock_sha, teacher_barrier_sha = _sha_file(JOB_LOCK_PATH), _sha_file(TEACHER_BARRIER_PATH)
    runtime_sha = _sha_file(TEACHER_RUNTIME_PATH)

    if manifest.get("schema") != "stvg-short-event-tracklets-plan-v1" or manifest.get("status") != "prepared_manifest_not_executed":
        raise ValueError("short-window prepared manifest is not the fixed frozen plan")
    manifest_cases = manifest.get("cases", [])
    if [str(row.get("case_id")) for row in manifest_cases] != list(CASE_IDS):
        raise ValueError("short-window manifest case order differs from the fixed old8 cohort")
    manifest_by_id = {str(row["case_id"]): row for row in manifest_cases}
    if track_barrier.get("schema") != "stvg-short-event-tracklet-barrier-v1" or track_barrier.get("status") != "complete":
        raise ValueError("complete source-anchored tracklet barrier is required")
    if track_barrier.get("manifest_sha256") != manifest_sha or track_barrier.get("count") != 8 or track_barrier.get("GT_read") is not False:
        raise ValueError("tracklet barrier does not bind the fixed no-GT manifest")
    if feature_barrier.get("schema") != "stvg-short-window-feature-capture-barrier-v1" or feature_barrier.get("status") != "sealed":
        raise ValueError("complete frozen feature barrier is required")
    if feature_barrier.get("count") != 8 or feature_barrier.get("GT_read") is not False:
        raise ValueError("feature barrier must bind eight GT-free cases")
    if input_barrier.get("schema") != "stvg-short-window-teacher-input-v1" or input_barrier.get("status") != "locked":
        raise ValueError("locked two-video teacher input barrier is required")
    if input_barrier.get("tracklet_barrier_sha256") != track_barrier_sha or input_barrier.get("manifest_sha256") != manifest_sha:
        raise ValueError("teacher inputs are not bound to the frozen short-window source manifest")
    if input_barrier.get("native_call_count") != 64 or input_barrier.get("window_count") != 32 or input_barrier.get("GT_read") is not False:
        raise ValueError("teacher input barrier does not describe the fixed 32-window/64-call scope")
    if job_lock.get("schema") != "stvg-short-window-teacher-jobs-v1" or job_lock.get("status") != "locked":
        raise ValueError("locked 64-job teacher input is required")
    jobs = job_lock.get("records", [])
    if job_lock.get("count") != 64 or len(jobs) != 64 or len({str(j.get("job_id")) for j in jobs}) != 64:
        raise ValueError("teacher job lock must contain exactly 64 unique job IDs")
    if job_lock.get("teacher_input_barrier_sha256") != input_barrier_sha:
        raise ValueError("teacher job lock does not bind the exact input barrier bytes")
    if teacher_barrier.get("status") != "complete" or teacher_barrier.get("count") != 64 or teacher_barrier.get("GT_read") is not False:
        raise ValueError("all 64 frozen teacher calls must be sealed before references are created")
    if teacher_barrier.get("runtime_sha256") != runtime_sha or teacher_barrier.get("input_barrier_sha256") != input_barrier_sha or teacher_barrier.get("job_lock_sha256") != job_lock_sha:
        raise ValueError("teacher result barrier runtime/input/job-lock binding differs")
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("teacher runtime must remain locked and GT-free")
    for relative, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        path = _safe_path(relative)
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"teacher runtime pin changed: {relative}")

    feature_entries = feature_barrier.get("files", {})
    if set(map(str, feature_entries)) != set(CASE_IDS):
        raise ValueError("feature barrier must bind exactly the eight fixed case payloads")
    track_rows = {str(row["case_id"]): row for row in track_barrier.get("case_records", [])}
    if list(track_rows) != list(CASE_IDS):
        raise ValueError("tracklet barrier case rows differ from fixed order")
    track_files = {str(row["path"]): row for row in track_barrier.get("files", [])}
    if len(track_files) != 16:
        raise ValueError("tracklet barrier must bind eight JSON files and eight private RGB files")
    # Bind the original per-case RGB cache at the byte level.  The student
    # reader requires this SHA in METHOD_REFERENCE_LOCK; validate it against
    # both the global file manifest and the actual compressed NPZ bytes before
    # copying the value into the reference record.  This hashes bytes only and
    # never decodes or reads RGB pixels.
    for case_id in CASE_IDS:
        row = track_rows[case_id]
        relative = str(row.get("rgb_npz_path", ""))
        rgb_sha = row.get("rgb_npz_sha256")
        file_row = track_files.get(relative)
        if (not relative or not isinstance(rgb_sha, str) or len(rgb_sha) != 64
                or any(char not in "0123456789abcdef" for char in rgb_sha)
                or not isinstance(file_row, Mapping)
                or file_row.get("sha256") != rgb_sha):
            raise ValueError(f"{case_id}: RGB NPZ binding differs from the sealed global file list")
        rgb_path = _safe_path(relative)
        if rgb_path.parent != (BASE / "private_rgb").resolve():
            raise ValueError(f"{case_id}: RGB NPZ path escaped its sealed private cache directory")
        if (not rgb_path.is_file()
                or rgb_path.stat().st_size != int(file_row.get("bytes", -1))
                or _sha_file(rgb_path) != rgb_sha):
            raise ValueError(f"{case_id}: RGB NPZ bytes differ from the sealed global barrier")
    if len(input_barrier.get("windows", [])) != 32:
        raise ValueError("teacher input barrier must contain all 32 fixed windows")
    return {
        "manifest": manifest, "manifest_by_id": manifest_by_id, "manifest_sha256": manifest_sha,
        "track_barrier": track_barrier, "track_barrier_sha256": track_barrier_sha,
        "track_rows": track_rows, "track_files": track_files,
        "feature_barrier": feature_barrier, "feature_barrier_sha256": feature_barrier_sha,
        "feature_entries": feature_entries, "input_barrier": input_barrier,
        "input_barrier_sha256": input_barrier_sha, "job_lock": job_lock,
        "job_lock_sha256": job_lock_sha, "jobs": jobs,
        "teacher_barrier": teacher_barrier, "teacher_barrier_sha256": teacher_barrier_sha,
        "runtime": runtime, "runtime_sha256": runtime_sha,
    }, jobs


def _load_teacher_outputs(context: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs = {str(job["job_id"]): job for job in context["jobs"]}
    files = context["teacher_barrier"].get("files", [])
    if len(files) != 64:
        raise ValueError("teacher result barrier must list exactly 64 immutable result files")
    result_by_id: dict[str, dict[str, Any]] = {}
    output_sha_by_id: dict[str, str] = {}
    for entry in files:
        path = _safe_path(entry["path"])
        if path.parent != TEACHER_DIR.resolve() or not path.is_file():
            raise ValueError("teacher result path escaped its sealed result directory")
        if _sha_file(path) != entry.get("sha256") or path.stat().st_size != int(entry["bytes"]):
            raise ValueError("teacher result SHA/size differs from the global seal")
        # Only the parsed ID is used below. Raw response text is never copied to
        # the reference lock or interpreted as instructions.
        result = _read_json(path)
        job_id = str(result.get("job_id"))
        if job_id not in jobs or job_id in result_by_id:
            raise ValueError("teacher result job ID is missing, duplicated, or unlocked")
        job = jobs[job_id]
        if result.get("GT_read") is not False or result.get("status") != "actual_video_candidate_verification":
            raise ValueError(f"{job_id}: teacher result is not an actual GT-free call")
        for key in ("case_id", "dataset", "kind", "center_slot", "source_case_sha256", "prompt_sha256"):
            if result.get(key) != job.get(key):
                raise ValueError(f"{job_id}: result field {key} differs from its locked job")
        if result.get("input_videos") != job.get("videos") or result.get("candidate_bindings") != job.get("candidate_bindings"):
            raise ValueError(f"{job_id}: sealed video or candidate mapping differs from its lock")
        parsed = result.get("parsed", {})
        valid = parsed.get("status") == "valid"
        selected = parsed.get("selected_candidate_id") if valid else None
        visible = parsed.get("visible_id") if valid else None
        if valid:
            if type(visible) is not int or visible not in job.get("allowed_output_ids", []):
                raise ValueError(f"{job_id}: parsed ID is outside its locked allowed-output set")
            mapped = job.get("visible_id_to_candidate", {}).get(str(visible))
            if selected != mapped:
                raise ValueError(f"{job_id}: parsed candidate does not match the locked visible-ID mapping")
        result_by_id[job_id] = {
            "job_id": job_id, "case_id": job["case_id"], "dataset": job["dataset"],
            "kind": job["kind"], "center_slot": int(job["center_slot"]),
            "status": "valid" if valid else "invalid", "visible_id": visible,
            "selected_candidate_id": selected,
            "output_truncated_at_cap": bool(result.get("output_truncated_at_cap", False)),
            "candidate_count": int(job.get("candidate_count", -1)),
        }
        output_sha_by_id[job_id] = str(entry["sha256"])
    if set(result_by_id) != set(jobs):
        raise ValueError("sealed teacher results do not cover exactly the locked 64 jobs")
    return result_by_id, output_sha_by_id


def _canonical_candidate_binding_signature(job: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    """Candidate identity is canonicalized independently of the visible ID.

    The ID-rotation arm intentionally changes visible_id and its list order;
    those fields must not be compared literally with the base job.
    """
    rows = job.get("candidate_bindings", [])
    if not isinstance(rows, list):
        raise ValueError("locked candidate bindings must be a list")
    signature = []
    seen = set()
    for row in rows:
        candidate_id = int(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError("candidate binding repeats a canonical candidate ID")
        seen.add(candidate_id)
        signature.append((
            candidate_id,
            str(row["tracklet_id"]),
            int(row["window_id"]),
            bool(row["is_source"]),
            tuple(map(int, row["canonical_color_rgb"])),
        ))
    return sorted(signature, key=lambda row: row[0])


def _visible_map_from_order(candidate_order: Sequence[int], control: str) -> dict[str, int]:
    ids = [int(v) for v in candidate_order]
    if len(ids) != len(set(ids)) or any(v < 1 or v > 5 for v in ids):
        raise ValueError("candidate order is not a unique fixed-palette ID list")
    labels = list(range(1, len(ids) + 1))
    if control == "base":
        by_candidate = dict(zip(ids, labels))
    elif control == "id_rotation":
        rotated = {} if not labels else ({ids[0]: 2} if len(ids) == 1 else dict(zip(ids, labels[1:] + labels[:1])))
        by_candidate = rotated
    else:
        raise ValueError("unknown ID control")
    return {str(label): candidate for candidate, label in by_candidate.items()}


def _strip_visible_labels(video: Mapping[str, Any]) -> dict[str, Any]:
    """Return the locked visual metadata after removing only numeric labels."""
    result = {
        key: video.get(key)
        for key in ("actual_timestamps_seconds", "actual_gap_seconds", "mean_actual_gap_seconds",
                    "encoder_uniform_spacing_approximation", "frame_count", "frame_shape_chw",
                    "tensor_layout", "dtype", "clip_role")
    }
    frames = []
    for frame in video.get("frame_metadata", []):
        copy = dict(frame)
        # Rendered pixel hashes necessarily change because the visible glyph
        # changes. Source pixels, geometry, palette, badge placement, and time
        # metadata remain part of the equality check.
        copy.pop("rendered_rgb_sha256", None)
        overlays = []
        for overlay in frame.get("candidate_overlays", []):
            item = dict(overlay)
            item.pop("visible_id", None)
            overlays.append(item)
        copy["candidate_overlays"] = overlays
        frames.append(copy)
    result["frame_metadata"] = frames
    return result


def _validate_base_rotation_pair(base_job: Mapping[str, Any], rotation_job: Mapping[str, Any]) -> None:
    if (base_job.get("case_id"), base_job.get("center_slot")) != (rotation_job.get("case_id"), rotation_job.get("center_slot")):
        raise ValueError("base and ID-rotation jobs do not bind the same window")
    if base_job.get("kind") != "base" or rotation_job.get("kind") != "id_rotation":
        raise ValueError("base and ID-rotation job roles are reversed or malformed")
    if _canonical_candidate_binding_signature(base_job) != _canonical_candidate_binding_signature(rotation_job):
        raise ValueError("ID rotation changed canonical candidate, color, tracklet, or source bindings")
    order_a = list(map(int, base_job.get("candidate_order", [])))
    order_b = list(map(int, rotation_job.get("candidate_order", [])))
    if order_a != order_b:
        raise ValueError("ID rotation changed canonical candidate order")
    expected_base = _visible_map_from_order(order_a, "base")
    expected_rotation = _visible_map_from_order(order_a, "id_rotation")
    if base_job.get("visible_id_to_candidate") != expected_base:
        raise ValueError("base visible-ID assignment differs from the fixed canonical mapping")
    if rotation_job.get("visible_id_to_candidate") != expected_rotation:
        raise ValueError("rotation visible-ID assignment differs from the fixed cyclic mapping")
    if _strip_visible_labels(base_job["local_video"]) != _strip_visible_labels(rotation_job["local_video"]):
        raise ValueError("ID rotation changed local-window content beyond visible numeric labels")
    if base_job.get("context_video", {}).get("sha256") != rotation_job.get("context_video", {}).get("sha256"):
        raise ValueError("ID rotation changed the unmarked context video")


def _window_tracklets(track_case: Mapping[str, Any], center: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_tracklet = {str(row["tracklet_id"]): row for row in track_case.get("tracklets", [])}
    out = []
    for cid, tid in zip(center.get("candidate_ids", []), center.get("tracklet_ids", [])):
        row = by_tracklet.get(str(tid))
        if row is None or int(row.get("candidate_id", -1)) != int(cid):
            raise ValueError("center candidate ID does not bind its immutable tracklet")
        out.append(row)
    return out


def _build_descriptors(track_case: Mapping[str, Any], feature_by_physical: Mapping[int, np.ndarray]) -> tuple[list[dict[int, np.ndarray | None]], list[dict[int, dict[str, Any]]]]:
    descriptor_windows: list[dict[int, np.ndarray | None]] = []
    diagnostic_windows: list[dict[int, dict[str, Any]]] = []
    centers = track_case.get("centers", [])
    tracklet_by_id = {str(row["tracklet_id"]): row for row in track_case.get("tracklets", [])}
    for center in centers:
        descriptors: dict[int, np.ndarray | None] = {}
        diagnostic: dict[int, dict[str, Any]] = {}
        for cid, tid in zip(center["candidate_ids"], center["tracklet_ids"]):
            track = tracklet_by_id[str(tid)]
            desc, info = tracklet_descriptor(track.get("frames", []), feature_by_physical)
            descriptors[int(cid)] = desc
            diagnostic[int(cid)] = info
        descriptor_windows.append(descriptors)
        diagnostic_windows.append(diagnostic)
    if len(descriptor_windows) != 4:
        raise ValueError("each case must contain four QA-center descriptor pools")
    return descriptor_windows, diagnostic_windows


def _selected_candidate(result: Mapping[str, Any]) -> int | None:
    if result.get("status") != "valid":
        return None
    value = result.get("selected_candidate_id")
    return None if value is None else int(value)


def _anchor_for_selection(
    case_id: str, slot: int, center: Mapping[str, Any], tracklets: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any], output_sha: str,
) -> dict[str, Any] | None:
    candidate_id = _selected_candidate(selection)
    # Candidate 1 is the unchanged Source seed. It is selectable, but is not a
    # non-Source teacher reference under this protocol.
    if candidate_id is None or candidate_id == 1:
        return None
    candidate = next((row for row in tracklets if int(row["candidate_id"]) == candidate_id), None)
    if candidate is None:
        raise ValueError(f"{case_id}/window{slot}: teacher selected a candidate not displayed in the locked window")
    center_id = int(center["center_physical_frame_id"])
    center_rows = [row for row in candidate.get("frames", []) if int(row["physical_frame_id"]) == center_id]
    if len(center_rows) != 1 or center_rows[0].get("valid") is not True or center_rows[0].get("box_xyxy_norm") is None:
        raise ValueError(f"{case_id}/window{slot}: selected non-Source candidate has no valid center box")
    box = _legal_xyxy(center_rows[0]["box_xyxy_norm"])
    return {
        "position": int(center["source_grid_position"]),
        "physical_frame_id": center_id,
        "box_cxcywh": xyxy_to_cxcywh(box),
        "selected_candidate_id": candidate_id,
        "candidate_id": candidate_id,
        "tracklet_id": str(candidate["tracklet_id"]),
        "teacher_job_id": f"{case_id}_window{slot:02d}_base",
        "teacher_output_sha256": output_sha,
        "is_Source": False,
    }


def _reference_record(name: str, anchors: Sequence[Mapping[str, Any]], selected_by_position: Mapping[int, int | None], reason_if_empty: str) -> dict[str, Any]:
    ordered = sorted((dict(anchor) for anchor in anchors), key=lambda row: int(row["position"]))
    selected_values = [int(v) for v in selected_by_position.values() if v is not None]
    unique = sorted(set(selected_values))
    if not ordered:
        reason = reason_if_empty
    elif name == "local_R1":
        reason = "selected_non_source_valid_center_boxes"
    else:
        reason = "selected_non_source_boxes_with_mutual_nearest_neighbor_support"
    return {
        "no_update": not bool(ordered),
        "eligible_for_update": bool(ordered),
        "reason": reason,
        "selected_candidate_id": unique[0] if len(unique) == 1 else None,
        "selected_candidate_id_is_case_constant": len(unique) <= 1,
        "selected_candidate_ids_by_position": [
            {"position": int(position), "candidate_id": selected_by_position.get(int(position))}
            for position in sorted(selected_by_position)
        ],
        "anchors": ordered,
    }


def _join_centers_to_manifest(
    track_case: Mapping[str, Any], manifest_case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join QA slots to original Source-grid positions by slot AND physical ID.

    QA slot is only the 0..3 short-window index. ``position`` in student
    anchors is the zero-based index into the original sparse Source grid. The
    two indexes are never assumed equal.
    """
    track_centers = track_case.get("centers", [])
    manifest_centers = manifest_case.get("centers", [])
    if len(track_centers) != 4 or len(manifest_centers) != 4:
        raise ValueError("tracklet and manifest must both bind the four fixed QA centers")
    manifest_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in manifest_centers:
        key = (int(row["slot"]), int(row["center_physical_frame_id"]))
        if key in manifest_by_key:
            raise ValueError("manifest repeats a QA slot/physical-center binding")
        manifest_by_key[key] = row
    joined = []
    seen_slots: set[int] = set()
    seen_source_positions: set[int] = set()
    source_grid_count = int(manifest_case["source_grid_count"])
    for track_center in track_centers:
        slot = int(track_center["slot"])
        physical_id = int(track_center["center_physical_frame_id"])
        key = (slot, physical_id)
        manifest_center = manifest_by_key.get(key)
        if manifest_center is None:
            raise ValueError("tracklet QA center does not exactly match manifest slot and physical ID")
        position = int(manifest_center["source_grid_position"])
        if not (0 <= position < source_grid_count):
            raise ValueError("manifest source-grid position lies outside the original sparse grid")
        if slot in seen_slots or position in seen_source_positions:
            raise ValueError("QA slots or original source-grid positions are duplicated")
        seen_slots.add(slot)
        seen_source_positions.add(position)
        joined.append({**track_center, "source_grid_position": position})
    if seen_slots != {0, 1, 2, 3}:
        raise ValueError("four QA slots are not bound exactly once")
    joined.sort(key=lambda row: int(row["slot"]))
    positions = [int(row["source_grid_position"]) for row in joined]
    physical_ids = [int(row["center_physical_frame_id"]) for row in joined]
    if positions != sorted(positions) or physical_ids != sorted(physical_ids):
        raise ValueError("QA order does not preserve increasing original Source-grid position/time")
    return joined


def _make_case_references(
    case_id: str, dataset: str, cohort: str, source_case_sha: str, source_state_sha: str,
    track_case: Mapping[str, Any], manifest_case: Mapping[str, Any], feature_by_physical: Mapping[int, np.ndarray],
    selection_by_slot: Mapping[int, Mapping[str, Any]], output_sha_by_job: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    centers = _join_centers_to_manifest(track_case, manifest_case)
    tracks_by_slot = {int(center["slot"]): _window_tracklets(track_case, center) for center in centers}
    descriptors, desc_diags = _build_descriptors(track_case, feature_by_physical)
    base_anchors: dict[int, dict[str, Any]] = {}
    selected_by_position: dict[int, int | None] = {}
    selection_diag = []
    for center, tracklets in zip(centers, [tracks_by_slot[i] for i in range(4)]):
        slot = int(center["slot"])
        result = selection_by_slot[slot]
        selected = _selected_candidate(result)
        source_position = int(center["source_grid_position"])
        selected_by_position[source_position] = selected
        anchor = _anchor_for_selection(
            case_id, slot, center, tracklets, result,
            output_sha_by_job[f"{case_id}_window{slot:02d}_base"],
        )
        if anchor is not None:
            base_anchors[slot] = anchor
        selection_diag.append({
            "center_slot": slot,
            "source_grid_position": source_position,
            "base_job_id": result["job_id"],
            "teacher_output_sha256": output_sha_by_job[result["job_id"]],
            "selection_status": result["status"],
            "visible_id": result.get("visible_id"),
            "selected_candidate_id": selected,
            "output_truncated_at_cap": result["output_truncated_at_cap"],
            "reference_excluded_as_Source": selected == 1,
        })

    # Build all adjacent-window mutual-nearest pools; selection is applied only
    # after associations are determined over every valid displayed proposal.
    edge_by_boundary: dict[int, list[dict[str, Any]]] = {}
    for left_slot in range(3):
        edges = mutual_nearest_edges(descriptors[left_slot], descriptors[left_slot + 1])
        edge_by_boundary[left_slot] = edges
    associated_anchors = []
    association_rows = []
    associated_support_by_slot: dict[int, list[dict[str, Any]]] = {}
    for left_slot in range(3):
        left_selected = _selected_candidate(selection_by_slot[left_slot])
        right_selected = _selected_candidate(selection_by_slot[left_slot + 1])
        edges = edge_by_boundary[left_slot]
        selected_pair_edge = next((edge for edge in edges if edge["left_candidate_id"] == left_selected and edge["right_candidate_id"] == right_selected), None)
        association_rows.append({
            "left_center_slot": left_slot,
            "right_center_slot": left_slot + 1,
            "left_selected_candidate_id": left_selected,
            "right_selected_candidate_id": right_selected,
            "mutual_nearest_edge_count": len(edges),
            "mutual_nearest_edges": edges,
            "selected_pair_is_mutual_nearest": selected_pair_edge is not None,
            "selected_pair_cosine": None if selected_pair_edge is None else selected_pair_edge["cosine"],
        })
        if selected_pair_edge is not None:
            for slot in (left_slot, left_slot + 1):
                if slot in base_anchors:
                    associated_support_by_slot.setdefault(slot, []).append({
                        "neighbor_center_slot": left_slot + 1 if slot == left_slot else left_slot,
                        "neighbor_candidate_id": right_selected if slot == left_slot else left_selected,
                        "cosine": float(selected_pair_edge["cosine"]),
                    })
    for slot in sorted(associated_support_by_slot):
        # Position, not window slot, is the student ABI key. A window may be
        # supported from both adjacent centers, but contributes one anchor.
        anchor = dict(base_anchors[slot])
        anchor["association_support"] = associated_support_by_slot[slot]
        associated_anchors.append(anchor)
    associated_selected_by_position = {
        int(anchor["position"]): selected_by_position[int(anchor["position"])]
        for anchor in associated_anchors
    }
    local_selected_by_position = {
        int(center["source_grid_position"]): selected_by_position[int(center["source_grid_position"])]
        for center in centers
    }
    references = {
        "local_R1": _reference_record(
            "local_R1", list(base_anchors.values()), local_selected_by_position,
            "no_valid_selected_non_source_center_box",
        ),
        "associated_R1": _reference_record(
            "associated_R1", associated_anchors, associated_selected_by_position,
            "no_selected_non_source_mutual_nearest_neighbor_pair",
        ),
    }
    descriptor_rows = []
    for slot, by_id in enumerate(desc_diags):
        for cid, diag in sorted(by_id.items()):
            descriptor_rows.append({"center_slot": slot, "candidate_id": cid, **diag})
    case_record = {
        "case_id": case_id,
        "dataset": dataset,
        "cohort": cohort,
        "source_case_sha256": source_case_sha,
        "source_model_state_sha256": source_state_sha,
        "manifest_case_sha256": _canonical_sha(manifest_case),
        "tracklet_case_sha256": str(track_case["_sealed_case_sha256"]),
        "canonical_request_sha256": str(track_case["input_binding"]["canonical_request_sha256"]),
        "canonical_video_sha256": str(track_case["input_binding"]["canonical_video_sha256"]),
        "references": references,
        "qa_center_to_source_grid_binding": [
            {"qa_center_slot": int(center["slot"]),
             "physical_frame_id": int(center["center_physical_frame_id"]),
             "source_grid_position": int(center["source_grid_position"])}
            for center in centers
        ],
        "GT_read": False,
    }
    diagnostics = {
        "case_id": case_id,
        "selection_by_center": selection_diag,
        "descriptor_status_by_center_candidate": descriptor_rows,
        "adjacent_window_mutual_nearest": association_rows,
        "all_displayed_proposals_in_association_pool": True,
        "cross_window_identity_asserted": False,
        "feature_maps_only_used_for_local_association": True,
    }
    return case_record, diagnostics


def lock_references() -> dict[str, Any]:
    context, _ = _load_and_validate_seals()
    import torch

    feature_by_case: dict[str, dict[int, np.ndarray]] = {}
    feature_receipts: dict[str, dict[str, Any]] = {}
    track_case_by_id: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        track_row = context["track_rows"][case_id]
        track_path = _safe_path(track_row["path"])
        if track_path.parent != TRACK_DIR.resolve() or _sha_file(track_path) != track_row["sha256"]:
            raise ValueError(f"{case_id}: tracklet JSON differs from the frozen global barrier")
        track_case = _read_json(track_path)
        if track_case.get("schema") != "stvg-short-event-tracklet-run-v1" or track_case.get("status") != "complete" or track_case.get("GT_read") is not False:
            raise ValueError(f"{case_id}: source-anchored tracklet record is not complete and GT-free")
        if track_case.get("input_binding", {}).get("manifest_sha256") != context["manifest_sha256"]:
            raise ValueError(f"{case_id}: tracklet JSON does not bind the fixed manifest")
        manifest_case = context["manifest_by_id"][case_id]
        input_binding = track_case["input_binding"]
        expected = {
            "case_id": case_id,
            "source_case_sha256": str(input_binding["source_case_sha256"]),
            "source_model_state_sha256": str(input_binding["source_model_state_sha256"]),
            "manifest_sha256": context["manifest_sha256"],
            "canonical_request_sha256": str(input_binding["canonical_request_sha256"]),
            "canonical_video_sha256": str(input_binding["canonical_video_sha256"]),
        }
        manifest_case_bindings = {
            "source_case_sha256": expected["source_case_sha256"],
            "source_model_state_sha256": expected["source_model_state_sha256"],
            "canonical_request_sha256": expected["canonical_request_sha256"],
            "canonical_video_sha256": expected["canonical_video_sha256"],
        }
        for key, value in manifest_case_bindings.items():
            if manifest_case.get(key) != value:
                raise ValueError(f"{case_id}: tracklet and manifest {key} binding differs")
        feature, receipt = _load_feature_case(
            case_id, context["feature_entries"][case_id], expected, manifest_case, track_case, torch
        )
        track_case["_sealed_case_sha256"] = track_row["sha256"]
        feature_by_case[case_id] = feature
        feature_receipts[case_id] = receipt
        track_case_by_id[case_id] = track_case

    output_by_id, output_sha_by_id = _load_teacher_outputs(context)
    job_by_id = {str(row["job_id"]): row for row in context["jobs"]}
    output_receipts: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        for slot in range(4):
            base_id = f"{case_id}_window{slot:02d}_base"
            rotation_id = f"{case_id}_window{slot:02d}_id_rotation"
            base_job, rotation_job = job_by_id[base_id], job_by_id[rotation_id]
            if base_job["kind"] != "base" or rotation_job["kind"] != "id_rotation":
                raise ValueError(f"{case_id}/window{slot}: locked base/rotation job roles changed")
            if int(base_job["center_slot"]) != slot or int(rotation_job["center_slot"]) != slot:
                raise ValueError(f"{case_id}/window{slot}: teacher job slot binding differs")
            _validate_base_rotation_pair(base_job, rotation_job)
            output_receipts[base_id] = output_by_id[base_id]
            output_receipts[rotation_id] = output_by_id[rotation_id]

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        track_case = track_case_by_id[case_id]
        selection_by_slot = {
            slot: output_by_id[f"{case_id}_window{slot:02d}_base"] for slot in range(4)
        }
        case_record, case_diagnostics = _make_case_references(
            case_id,
            str(context["manifest_by_id"][case_id]["dataset"]),
            str(context["manifest_by_id"][case_id]["cohort"]),
            str(track_case["input_binding"]["source_case_sha256"]),
            str(track_case["input_binding"]["source_model_state_sha256"]),
            track_case, context["manifest_by_id"][case_id], feature_by_case[case_id],
            selection_by_slot, output_sha_by_id,
        )
        # The reference is tied to its exact frozen feature payload as well as
        # the per-case source and physical-frame geometry.
        case_record["feature_case_sha256"] = feature_receipts[case_id]["feature_sha256"]
        case_record["feature_maps_sha256"] = feature_receipts[case_id]["feature_maps_sha256"]
        # Preserve a direct binding to the exact cache bytes named by the
        # complete global tracklet barrier.  The hash was verified above.
        case_record["rgb_npz_sha256"] = str(context["track_rows"][case_id]["rgb_npz_sha256"])
        records.append(case_record)
        diagnostics.append(case_diagnostics)

    refs_no_update = {
        name: sum(record["references"][name]["no_update"] for record in records)
        for name in ("local_R1", "associated_R1")
    }
    lock = {
        "schema": SCHEMA,
        "status": "locked",
        "count": len(records),
        "manifest_sha256": context["manifest_sha256"],
        "tracklet_barrier_sha256": context["track_barrier_sha256"],
        "feature_barrier_sha256": context["feature_barrier_sha256"],
        "teacher_input_barrier_sha256": context["input_barrier_sha256"],
        "teacher_job_lock_sha256": context["job_lock_sha256"],
        "teacher_global_barrier_sha256": context["teacher_barrier_sha256"],
        "teacher_runtime_sha256": context["runtime_sha256"],
        "references": ["local_R1", "associated_R1"],
        "expected_fit_counts": {
            "local_R1": len(records) - refs_no_update["local_R1"],
            "associated_R1": len(records) - refs_no_update["associated_R1"],
        },
        "no_update_counts": refs_no_update,
        "records": records,
        "diagnostics": diagnostics,
        "association_rule": "all displayed tracklets; per-frame area-overlap ROI pool; per-frame L2; equal-frame mean; adjacent-window mutual maximum cosine; exact/tolerance ties choose smallest candidate ID; no threshold",
        "reference_rule": "local_R1 uses every selected non-Source valid center; associated_R1 retains only such centers whose selection has a mutual-nearest edge to a selected adjacent-center candidate; no confidence/margin/GT threshold",
        "Source_or_abstain_contributes_reference": False,
        "empty_reference_means_exact_NoUpdate": True,
        "cross_window_identity_claim": False,
        "raw_teacher_text_copied": False,
        "raw_features_copied": False,
        "GT_read": False,
    }
    out = _write_once(OUTPUT_PATH, lock)
    return {"status": "locked", "record_count": len(records), "expected_fit_counts": lock["expected_fit_counts"],
            "no_update_counts": refs_no_update, "output": out, "GT_read": False}


def validate_preteacher_abi() -> dict[str, Any]:
    """Read-only validation of the sealed jobs and frozen CPU feature ABI.

    This deliberately does not open teacher result files, source query payloads,
    RGB caches, annotations, model state, or media decoders.
    """
    import torch

    manifest = _read_json(MANIFEST_PATH)
    track_barrier = _read_json(TRACK_BARRIER_PATH)
    feature_barrier = _read_json(FEATURE_BARRIER_PATH)
    input_barrier = _read_json(INPUT_BARRIER_PATH)
    job_lock = _read_json(JOB_LOCK_PATH)
    manifest_sha = _sha_file(MANIFEST_PATH)
    track_barrier_sha = _sha_file(TRACK_BARRIER_PATH)
    input_barrier_sha = _sha_file(INPUT_BARRIER_PATH)
    feature_barrier_sha = _sha_file(FEATURE_BARRIER_PATH)
    job_lock_sha = _sha_file(JOB_LOCK_PATH)
    assert manifest.get("schema") == "stvg-short-event-tracklets-plan-v1"
    assert manifest.get("status") == "prepared_manifest_not_executed"
    assert track_barrier.get("status") == "complete" and track_barrier.get("manifest_sha256") == manifest_sha
    assert feature_barrier.get("status") == "sealed" and feature_barrier.get("GT_read") is False
    assert input_barrier.get("schema") == "stvg-short-window-teacher-input-v1"
    assert input_barrier.get("status") == "locked" and input_barrier.get("tracklet_barrier_sha256") == track_barrier_sha
    assert input_barrier.get("manifest_sha256") == manifest_sha and input_barrier.get("GT_read") is False
    assert job_lock.get("schema") == "stvg-short-window-teacher-jobs-v1" and job_lock.get("status") == "locked"
    assert job_lock.get("teacher_input_barrier_sha256") == input_barrier_sha
    assert len(job_lock.get("records", [])) == 64 and job_lock.get("count") == 64
    jobs = {str(row["job_id"]): row for row in job_lock["records"]}
    assert len(jobs) == 64

    # Confirm 32 deterministic base/rotation pairs, the immutable unmarked
    # context binding, label-only local control, and every serialized NPY hash.
    video_paths = set()
    frame_counts = {"context": 0, "local": 0}
    for case_id in CASE_IDS:
        for slot in range(4):
            base = jobs[f"{case_id}_window{slot:02d}_base"]
            rotation = jobs[f"{case_id}_window{slot:02d}_id_rotation"]
            _validate_base_rotation_pair(base, rotation)
            for control, job in (("base", base), ("id_rotation", rotation)):
                assert job["teacher_input_barrier_sha256"] == input_barrier_sha
                assert job["case_id"] == case_id and int(job["center_slot"]) == slot and job["kind"] == control
                assert len(job["videos"]) == 2
                context_video, local_video = job["videos"]
                assert int(context_video["frame_count"]) == 4
                assert 1 <= int(local_video["frame_count"]) <= 5
                assert len(context_video["frame_metadata"]) == int(context_video["frame_count"])
                assert len(local_video["frame_metadata"]) == int(local_video["frame_count"])
                frame_counts["context"] += int(context_video["frame_count"])
                frame_counts["local"] += int(local_video["frame_count"])
                for video in job["videos"]:
                    path = _safe_path(video["path"])
                    if path.parent != (BASE / "teacher_inputs" / case_id).resolve() and not (BASE / "teacher_inputs" / case_id).resolve() in path.parents:
                        raise ValueError("locked video array path escaped its per-case teacher-input directory")
                    if not path.is_file() or _sha_file(path) != video["sha256"] or path.stat().st_size != int(video["bytes"]):
                        raise ValueError("locked rendered video file differs from its declared SHA/size")
                    video_paths.add(path)
    if len(video_paths) != 72:
        raise ValueError("fixed input must contain eight shared contexts plus 64 local-control videos")
    if input_barrier.get("teacher_input_video_file_count") != 72 or len(input_barrier.get("teacher_input_video_files", [])) != 72:
        raise ValueError("input barrier does not bind the expected 72 serialized rendered arrays")

    manifest_by_id = {str(row["case_id"]): row for row in manifest["cases"]}
    track_rows = {str(row["case_id"]): row for row in track_barrier.get("case_records", [])}
    feature_entries = feature_barrier.get("files", {})
    if list(manifest_by_id) != list(CASE_IDS) or list(track_rows) != list(CASE_IDS) or set(feature_entries) != set(CASE_IDS):
        raise ValueError("manifest, tracklet, or feature barrier case order/set differs from fixed old8")
    feature_file_count = 0
    feature_map_count = 0
    feature_frame_count = 0
    for case_id in CASE_IDS:
        track_row = track_rows[case_id]
        track_path = _safe_path(track_row["path"])
        if track_path.parent != TRACK_DIR.resolve() or _sha_file(track_path) != track_row["sha256"]:
            raise ValueError(f"{case_id}: tracklet JSON differs from its sealed file record")
        track_case = _read_json(track_path)
        inp = track_case["input_binding"]
        expected = {
            "source_case_sha256": str(inp["source_case_sha256"]),
            "source_model_state_sha256": str(inp["source_model_state_sha256"]),
            "manifest_sha256": manifest_sha,
            "canonical_request_sha256": str(inp["canonical_request_sha256"]),
            "canonical_video_sha256": str(inp["canonical_video_sha256"]),
        }
        for key, value in (
            ("source_case_sha256", expected["source_case_sha256"]),
            ("source_model_state_sha256", expected["source_model_state_sha256"]),
            ("canonical_request_sha256", expected["canonical_request_sha256"]),
            ("canonical_video_sha256", expected["canonical_video_sha256"]),
        ):
            assert manifest_by_id[case_id].get(key) == value
        entry = feature_entries[case_id]
        receipt_info = entry.get("receipt", {})
        receipt_path = _safe_path(receipt_info["path"])
        if receipt_path.parent != FEATURE_DIR.resolve() or not receipt_path.is_file():
            raise ValueError(f"{case_id}: frozen feature receipt escaped its directory")
        if _sha_file(receipt_path) != receipt_info.get("sha256") or receipt_path.stat().st_size != int(receipt_info["bytes"]):
            raise ValueError(f"{case_id}: frozen feature receipt SHA/size mismatch")
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "captured_source_only_feature_maps" or receipt.get("GT_read") is not False or receipt.get("backward_calls") != 0:
            raise ValueError(f"{case_id}: feature receipt does not describe the frozen no-GT forward")
        if receipt.get("feature_file_sha256") != entry.get("sha256") or receipt.get("feature_file_bytes") != entry.get("bytes"):
            raise ValueError(f"{case_id}: feature receipt does not bind its tensor file")
        feature, feature_meta = _load_feature_case(case_id, entry, expected, manifest_by_id[case_id], track_case, torch)
        feature_file_count += 1
        feature_map_count += len(feature)
        feature_frame_count += len(feature_meta["frame_ids"])
    return {
        "status": "pass",
        "scope": "actual_sealed_preteacher_cpu_abi_readback",
        "manifest_sha256": manifest_sha,
        "tracklet_barrier_sha256": track_barrier_sha,
        "candidate_input_barrier_sha256": input_barrier_sha,
        "teacher_job_lock_sha256": job_lock_sha,
        "feature_barrier_sha256": feature_barrier_sha,
        "case_count": 8,
        "window_count": 32,
        "job_count": len(jobs),
        "candidate_rotation_pairs_checked": 32,
        "rendered_video_files_hashed": len(video_paths),
        "feature_files_checked_cpu_weights_only": feature_file_count,
        "feature_physical_frame_maps_checked": feature_map_count,
        "feature_map_rows": feature_frame_count,
        "context_frame_rows": frame_counts["context"],
        "local_frame_rows": frame_counts["local"],
        "teacher_result_files_read": 0,
        "source_queries_read": False,
        "source_RGB_NPZ_read": False,
        "GT_read": False,
        "model_or_gpu_called": False,
    }


def synthetic_contracts() -> dict[str, Any]:
    """Small deterministic CPU-only checks; no repository artifact is opened."""
    # Constant maps must pool to their exact channel value irrespective of box.
    constant = np.zeros((3, 2, 4), dtype=np.float64)
    constant[0, :, :] = 2.0
    constant[1, :, :] = -3.0
    constant[2, :, :] = 0.5
    pooled_full = area_overlap_roi_pool(constant, [0.0, 0.0, 1.0, 1.0])
    pooled_partial = area_overlap_roi_pool(constant, [0.13, 0.12, 0.76, 0.81])
    assert np.array_equal(pooled_full, np.array([2.0, -3.0, 0.5]))
    assert np.allclose(pooled_partial, pooled_full, atol=1e-15, rtol=0)
    # Per-frame normalization followed by an equal-frame mean is distinct from
    # pooling all frames together when box support and magnitude differ.
    maps = {
        10: np.stack([np.ones((2, 2)), np.zeros((2, 2))]),
        11: np.stack([np.zeros((2, 2)), np.full((2, 2), 9.0)]),
    }
    desc, dmeta = tracklet_descriptor([
        {"physical_frame_id": 10, "valid": True, "box_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
        {"physical_frame_id": 11, "valid": True, "box_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
    ], maps)
    assert desc is not None and np.allclose(desc, [0.5, 0.5], atol=1e-15, rtol=0)
    assert dmeta["used_physical_frame_ids"] == [10, 11]
    # Tie on cosine must choose the lowest candidate ID; only reciprocal pairs survive.
    ties = mutual_nearest_edges({7: np.array([1.0, 0.0])}, {
        9: np.array([1.0, 0.0]), 3: np.array([1.0, 0.0])
    })
    assert ties == [{"left_candidate_id": 7, "right_candidate_id": 3, "cosine": 1.0}]
    candidate_order = [1, 2, 4]

    def synthetic_job(control: str) -> dict[str, Any]:
        visible_map = _visible_map_from_order(candidate_order, control)
        bindings = []
        for visible, candidate in sorted(((int(k), int(v)) for k, v in visible_map.items())):
            bindings.append({
                "visible_id": visible, "candidate_id": candidate,
                "tracklet_id": f"syn-{candidate}", "window_id": candidate + 10,
                "is_source": candidate == 1,
                "canonical_color_rgb": [[230, 126, 34], [0, 114, 178], [204, 121, 167], [0, 158, 115]][candidate - 1],
            })
        overlays = [
            {"candidate_id": candidate, "visible_id": int(visible_map_label),
             "canonical_color_rgb": binding["canonical_color_rgb"],
             "box_xyxy_pixels": [candidate, 2, candidate + 10, 20],
             "badge_xyxy_pixels": [candidate, 2, candidate + 8, 18]}
            for binding in sorted(bindings, key=lambda row: row["candidate_id"])
            for visible_map_label, candidate in [(next(k for k, v in visible_map.items() if int(v) == int(binding["candidate_id"])), binding["candidate_id"])]
        ]
        return {
            "case_id": "syn", "center_slot": 0, "kind": control,
            "candidate_order": candidate_order,
            "candidate_bindings": bindings,
            "visible_id_to_candidate": visible_map,
            "local_video": {
                "actual_timestamps_seconds": [1.0, 1.1], "actual_gap_seconds": [0.1],
                "mean_actual_gap_seconds": 0.1, "encoder_uniform_spacing_approximation": False,
                "frame_count": 2, "frame_shape_chw": [3, 448, 448],
                "tensor_layout": "TCHW", "dtype": "uint8", "clip_role": "marked_local_short_window",
                "frame_metadata": [{
                    "slot": 0, "timestamp_seconds": 1.0, "timestamp_overlay": "t=1.000s",
                    "rendered_rgb_sha256": f"pixels-{control}", "source_rgb_sha256": "source",
                    "candidate_overlays": overlays,
                }],
            },
            "context_video": {"sha256": "same-context"},
        }

    _validate_base_rotation_pair(synthetic_job("base"), synthetic_job("id_rotation"))
    synthetic_track = {"centers": [
        {"slot": slot, "center_physical_frame_id": physical, "candidate_ids": [], "tracklet_ids": []}
        for slot, physical in enumerate([10, 20, 30, 40])
    ]}
    synthetic_manifest = {"source_grid_count": 12, "centers": [
        {"slot": slot, "center_physical_frame_id": physical, "source_grid_position": position}
        for slot, (physical, position) in enumerate(zip([10, 20, 30, 40], [1, 3, 7, 9]))
    ]}
    joined = _join_centers_to_manifest(synthetic_track, synthetic_manifest)
    assert [row["slot"] for row in joined] == [0, 1, 2, 3]
    assert [row["source_grid_position"] for row in joined] == [1, 3, 7, 9]
    assert [row["center_physical_frame_id"] for row in joined] == [10, 20, 30, 40]
    mismatch_track = {"centers": [dict(row) for row in synthetic_track["centers"]]}
    mismatch_track["centers"][2]["center_physical_frame_id"] = 31
    try:
        _join_centers_to_manifest(mismatch_track, synthetic_manifest)
    except ValueError:
        center_mismatch_rejected = True
    else:
        center_mismatch_rejected = False
    assert center_mismatch_rejected
    # Empty, Source-only, and abstained selections all yield an exact no-update ref.
    local = _reference_record("local_R1", [], {0: 1, 1: None}, "no_valid_selected_non_source_center_box")
    assoc = _reference_record("associated_R1", [], {0: 1}, "no_selected_non_source_mutual_nearest_neighbor_pair")
    assert local["no_update"] and local["eligible_for_update"] is False and local["anchors"] == []
    assert assoc["no_update"] and assoc["eligible_for_update"] is False and assoc["anchors"] == []
    # The selected pair is retained only if it is an edge in the all-candidate
    # mutual-nearest pool (illustrate a selected pair being excluded).
    all_edges = mutual_nearest_edges(
        {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
        {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
    )
    assert {(e["left_candidate_id"], e["right_candidate_id"]) for e in all_edges} == {(1, 1), (2, 2)}
    try:
        _legal_xyxy([-0.01, 0.1, 0.5, 0.7])
    except ValueError:
        invalid_geometry_rejected = True
    else:
        invalid_geometry_rejected = False
    assert invalid_geometry_rejected
    return {
        "status": "pass", "scope": "synthetic_cpu_contracts_only",
        "actual_artifacts_read": False, "teacher_called": False, "features_loaded": False,
        "model_or_gpu_called": False, "GT_read": False,
        "checks": {
            "normalized_area_overlap_constant_map": True,
            "per_frame_l2_then_equal_frame_mean": True,
            "mutual_nearest_cosine_and_candidate_id_tie_break": True,
            "base_rotation_bindings_compared_after_candidate_id_canonicalization": True,
            "qa_slots_joined_to_distinct_original_source_grid_positions": True,
            "center_join_requires_both_slot_and_physical_id": True,
            "source_abstain_empty_exact_no_update": True,
            "association_uses_all_candidates_then_checks_selected_pair": True,
            "legal_geometry_is_not_clipped": True,
            "raw_teacher_text_and_feature_maps_not_copied_to_reference_lock": True,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-contracts", action="store_true")
    group.add_argument("--preteacher-abi", action="store_true", help="read only the sealed job/feature ABI before teacher results exist")
    group.add_argument("--lock", action="store_true", help="create references after all locked barriers are complete")
    args = parser.parse_args()
    if args.synthetic_contracts:
        result = synthetic_contracts()
    elif args.preteacher_abi:
        result = validate_preteacher_abi()
    else:
        result = lock_references()
    print(json.dumps(result, ensure_ascii=False, indent=2))
