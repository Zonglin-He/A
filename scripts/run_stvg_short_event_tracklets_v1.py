"""Run the root-approved finite CPU-only short-window LK stage for fixed old8.

The runner is deliberately fixed to PREPARED_MANIFEST.json and its eight
source-bound cases. It revalidates all input hashes before decoding selected
frames, never reads labels, and never imports torch or a model runtime.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/stvg_short_event_tracklets_v1"
MANIFEST_PATH = OUT / "PREPARED_MANIFEST.json"
TRACK_DIR = OUT / "tracklets"
RGB_DIR = OUT / "private_rgb"
BARRIER_PATH = OUT / "GLOBAL_TRACKLET_BARRIER.json"
LOCK_PATH = OUT / ".CPU_TRACKLET_RUN004.lock"
ROSTER = ROOT / "artifacts/stvg_adjacent_motion_opd_v1/ROSTER_LOCK.json"
CANDIDATE_BARRIER = ROOT / "artifacts/stvg_adjacent_motion_opd_v1/CANDIDATE_INPUT_BARRIER.json"
QA_BARRIER = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json"
SOURCE_CASES = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1/source_cases"
QA_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
CANONICAL_BASE = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
HELPER_PATH = ROOT / "scripts/stvg_adjacent_affine_tracklet_v1.py"
BUILDER_PATH = ROOT / "scripts/prepare_stvg_short_event_tracklets_v1.py"
PROTOCOL_PATH = ROOT / "protocols/stvg_short_event_tracklets_v1.proposal.json"
CASE_IDS = tuple(
    [f"hc2_{i:03d}" for i in range(4)]
    + [f"vidstg_{i:03d}" for i in range(4)]
)
SCHEMA = "stvg-short-event-tracklet-run-v1"
BARRIER_SCHEMA = "stvg-short-event-tracklet-barrier-v1"
DISK_FREE_FLOOR_BYTES = 30 * (1 << 30)
MAX_TRACK_EDGE = 640


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_root_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"input path escaped project root: {value}")
    return resolved


def _write_once_json(path: Path, value: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output or temporary file: {path}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_bytes(raw), "bytes": len(raw)}


def _write_once_npz(path: Path, **arrays: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output or temporary file: {path}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha_file(path),
        "bytes": path.stat().st_size,
    }


def _acquire_lock_and_preflight(lock_path: Path, preflight_callback):
    """Create the exclusive lock first, then run preflight while holding it.

    The callback can be tested with a synthetic no-video preflight. A lock
    created by this process must not be checked as though it belonged to a
    competing worker; O_EXCL is the single-writer gate.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(fd, "w", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.write(
            json.dumps(
                {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        preflight_result = preflight_callback()
        return stream, preflight_result
    except Exception:
        stream.close()
        raise


def _validate_global_bindings(manifest: Mapping[str, Any], manifest_sha: str) -> None:
    if manifest.get("schema") != "stvg-short-event-tracklets-plan-v1":
        raise ValueError("unsupported prepared manifest schema")
    if manifest.get("status") != "prepared_manifest_not_executed":
        raise ValueError("manifest is not the frozen unexecuted plan")
    if manifest.get("case_count") != 8 or manifest.get("center_count") != 32:
        raise ValueError("prepared plan must contain exactly old8 and 32 centers")
    if [case.get("case_id") for case in manifest.get("cases", [])] != list(CASE_IDS):
        raise ValueError("manifest case list/order differs from fixed old8")
    if manifest.get("GT_read") is not False or manifest.get("GPU_used") is not False:
        raise ValueError("prepared manifest declares a disallowed operation")

    bindings = manifest.get("source_bindings", {})
    fixed_files = {
        "source_roster_lock_path": (ROSTER, "source_roster_lock_sha256"),
        "candidate_input_barrier_path": (CANDIDATE_BARRIER, "candidate_input_barrier_sha256"),
        "qa_student_barrier_path": (QA_BARRIER, "qa_student_barrier_sha256"),
        "track_helper_path": (HELPER_PATH, "track_helper_sha256"),
        "proposal_path": (PROTOCOL_PATH, "proposal_sha256"),
    }
    for path_key, (expected_path, sha_key) in fixed_files.items():
        if bindings.get(path_key) != expected_path.relative_to(ROOT).as_posix():
            raise ValueError(f"manifest fixed source path changed: {path_key}")
        if _sha_file(expected_path) != bindings.get(sha_key):
            raise ValueError(f"manifest source SHA changed: {path_key}")
    if bindings.get("prepared_manifest_sha256") not in (None, manifest_sha):
        raise ValueError("manifest self-binding differs")


def _preflight_inputs(manifest: Mapping[str, Any], manifest_sha: str) -> list[dict[str, Any]]:
    _validate_global_bindings(manifest, manifest_sha)
    if not MANIFEST_PATH.is_file() or _sha_file(MANIFEST_PATH) != manifest_sha:
        raise ValueError("prepared manifest changed during preflight")

    if TRACK_DIR.exists() or RGB_DIR.exists() or BARRIER_PATH.exists():
        raise FileExistsError("track output path already contains artifacts; immutable run will not overwrite")
    free_bytes = __import__("shutil").disk_usage(ROOT).free
    if free_bytes < DISK_FREE_FLOOR_BYTES:
        raise RuntimeError(f"free disk space below locked 30 GiB floor: {free_bytes} bytes")

    candidate_barrier = _read_json(CANDIDATE_BARRIER)
    candidate_records = {str(row["case_id"]): row for row in candidate_barrier["records"]}
    roster = _read_json(ROSTER)
    roster_records = {str(row["case_id"]): row for row in roster["records"]}
    qa_barrier = _read_json(QA_BARRIER)
    qa_files = {Path(str(row["path"])).name: row for row in qa_barrier["files"]}

    result: list[dict[str, Any]] = []
    # Verify every input binding and full encoded-video SHA before opening any
    # decoder. The selected pixel digest remains a per-frame readback; this
    # finite stage does not re-decode every video frame to recompute the full
    # canonical RGB digest.
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        source_path = _safe_root_path(case["source_case_path"])
        if source_path.parent != SOURCE_CASES.resolve() or _sha_file(source_path) != case["source_case_sha256"]:
            raise ValueError(f"source cache binding changed for {case_id}")
        if roster_records[case_id].get("source_case_sha256") != case["source_case_sha256"]:
            raise ValueError(f"source roster binding changed for {case_id}")
        candidate_record = candidate_records.get(case_id)
        if candidate_record is None or _canonical_sha(candidate_record) != case["candidate_barrier_record_sha256"]:
            raise ValueError(f"candidate barrier record changed for {case_id}")
        if candidate_record.get("source_case_sha256") != case["source_case_sha256"]:
            raise ValueError(f"candidate-to-source binding changed for {case_id}")

        qa_path = _safe_root_path(case["qa_student_path"])
        if qa_path.parent != (QA_BASE / "student").resolve() or _sha_file(qa_path) != case["qa_student_sha256"]:
            raise ValueError(f"reused temporal-QA input changed for {case_id}")
        if qa_files.get(qa_path.name, {}).get("sha256") != case["qa_student_sha256"]:
            raise ValueError(f"temporal-QA barrier record changed for {case_id}")

        request_path = _safe_root_path(case["canonical_request_path"])
        canonical_dir = (CANONICAL_BASE / case_id).resolve()
        if request_path.parent != canonical_dir or _sha_file(request_path) != case["canonical_request_sha256"]:
            raise ValueError(f"canonical request changed for {case_id}")
        request = _read_json(request_path)
        if (
            request.get("source_video_sha256") != case["source_video_sha256"]
            or request.get("canonical_video_sha256") != case["canonical_video_sha256"]
            or request.get("canonical_pixel_sha256") != case.get("canonical_pixel_sha256")
        ):
            raise ValueError(f"canonical request SHA fields changed for {case_id}")
        if request.get("video_path") != str(_safe_root_path(case["canonical_video_path"])):
            raise ValueError(f"canonical video path binding changed for {case_id}")
        frame_map = request.get("frame_map")
        if not isinstance(frame_map, list) or len(frame_map) != request.get("local_frame_count"):
            raise ValueError(f"canonical frame map length is invalid for {case_id}")
        physical_to_local: dict[int, int] = {}
        local_to_frame: dict[int, Mapping[str, Any]] = {}
        for row in frame_map:
            physical = int(row["physical_frame_id"])
            local = int(row["local_frame_index"])
            if physical in physical_to_local or local in local_to_frame or local < 0:
                raise ValueError(f"canonical frame map has duplicate indices for {case_id}")
            physical_to_local[physical] = local
            local_to_frame[local] = row

        expected_pairs: set[tuple[int, int]] = set()
        for center in case["centers"]:
            if len(center["matched_tracklets"]) != len(case["fixed_displayed_candidate_ids"]):
                raise ValueError(f"candidate/window count changed for {case_id}")
            for physical, local in zip(center["physical_frame_ids"], center["local_frame_indices"]):
                pair = (int(physical), int(local))
                if pair in expected_pairs:
                    continue
                expected_pairs.add(pair)
        if any(physical_to_local.get(physical) != local for physical, local in expected_pairs):
            raise ValueError(f"manifest physical/local mapping differs from canonical request for {case_id}")

        video_path = _safe_root_path(case["canonical_video_path"])
        if video_path.parent != canonical_dir or not video_path.is_file():
            raise ValueError(f"canonical video escaped its case directory for {case_id}")
        actual_video_sha = _sha_file(video_path)
        if actual_video_sha != case["canonical_video_sha256"]:
            raise ValueError(f"canonical encoded video SHA changed for {case_id}")
        result.append(
            {
                "case": case,
                "request": request,
                "request_path": request_path,
                "video_path": video_path,
                "frame_map_by_local": local_to_frame,
                "physical_to_local": physical_to_local,
                "expected_pairs": sorted(expected_pairs, key=lambda item: item[1]),
                "encoded_video_sha256_verified": actual_video_sha,
            }
        )
        print(f"preflight {case_id}: request/source/QA/candidate/video bindings pass", flush=True)
    return result


def _resize_shape(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, float(MAX_TRACK_EDGE) / float(max(width, height)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _decode_selected_frames(preflight: Mapping[str, Any]) -> dict[str, Any]:
    case = preflight["case"]
    case_id = str(case["case_id"])
    request = preflight["request"]
    frame_map_by_local = preflight["frame_map_by_local"]
    selected_pairs = preflight["expected_pairs"]
    video_path = preflight["video_path"]
    expected_count = int(request["local_frame_count"])

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV/FFmpeg could not open canonical video for {case_id}")
    try:
        backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
        if backend not in {"FFMPEG", "unknown"}:
            raise RuntimeError(f"unexpected decode backend for {case_id}: {backend}")
        reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if reported_count != expected_count:
            raise RuntimeError(f"canonical frame-count mismatch for {case_id}: {reported_count} != {expected_count}")

        rgb_by_local: dict[int, np.ndarray] = {}
        gray_by_local: dict[int, np.ndarray] = {}
        selected_meta: list[dict[str, Any]] = []
        source_shapes: set[tuple[int, int]] = set()
        cached_shapes: set[tuple[int, int]] = set()
        for physical_id, local_index in selected_pairs:
            frame_row = frame_map_by_local[int(local_index)]
            if int(frame_row["physical_frame_id"]) != int(physical_id):
                raise RuntimeError(f"REQUEST frame map identity mismatch for {case_id}/{local_index}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(local_index)):
                raise RuntimeError(f"frame seek failed for {case_id}/{local_index}")
            before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if int(round(before)) != int(local_index):
                raise RuntimeError(f"decoder seek landed on the wrong local frame for {case_id}/{local_index}")
            ok, bgr = capture.read()
            after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok or bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise RuntimeError(f"selected canonical frame decode failed for {case_id}/{local_index}")
            if int(round(after)) != int(local_index) + 1:
                raise RuntimeError(f"decoder post-read index mismatch for {case_id}/{local_index}")

            source_height, source_width = map(int, bgr.shape[:2])
            source_shapes.add((source_height, source_width))
            resize_width, resize_height = _resize_shape(source_width, source_height)
            rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb_small = (
                rgb_full
                if (resize_width, resize_height) == (source_width, source_height)
                else cv2.resize(rgb_full, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            )
            gray_small = (
                gray_full
                if (resize_width, resize_height) == (source_width, source_height)
                else cv2.resize(gray_full, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            )
            rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)
            gray_small = np.ascontiguousarray(gray_small, dtype=np.uint8)
            source_shapes.add(tuple(map(int, rgb_full.shape[:2])))
            cached_shapes.add(tuple(map(int, rgb_small.shape[:2])))
            rgb_by_local[int(local_index)] = rgb_small
            gray_by_local[int(local_index)] = gray_small
            selected_meta.append(
                {
                    "physical_frame_id": int(physical_id),
                    "local_frame_index": int(local_index),
                    "source_timestamp_s": float(frame_row["source_timestamp_s"]),
                    "source_rgb_shape_hwc": [source_height, source_width, 3],
                    "source_frame_rgb_sha256": _sha_bytes(np.ascontiguousarray(rgb_full).tobytes()),
                    "cached_rgb_shape_hwc": [int(rgb_small.shape[0]), int(rgb_small.shape[1]), 3],
                    "cached_rgb_sha256": _sha_bytes(rgb_small.tobytes()),
                    "tracking_gray_sha256": _sha_bytes(gray_small.tobytes()),
                    "resize": {
                        "method": "OpenCV INTER_AREA; identity if long edge <=640",
                        "source_width_height": [source_width, source_height],
                        "cached_width_height": [resize_width, resize_height],
                    },
                    "decoder_frame_position_before": before,
                    "decoder_frame_position_after": after,
                }
            )
            del bgr, rgb_full, gray_full
    finally:
        capture.release()

    if len(rgb_by_local) != len(selected_pairs) or len(rgb_by_local) != len(set(local for _, local in selected_pairs)):
        raise RuntimeError(f"selected physical frames were not decoded exactly once for {case_id}")
    if len(source_shapes) != 1 or len(cached_shapes) != 1:
        raise RuntimeError(f"video frames change dimensions within selected input for {case_id}")
    return {
        "rgb_by_local": rgb_by_local,
        "gray_by_local": gray_by_local,
        "selected_frame_map": selected_meta,
        "source_shape_hw": list(next(iter(source_shapes))),
        "cached_shape_hw": list(next(iter(cached_shapes))),
        "decoder_backend": backend,
        "frame_map_sha256": _canonical_sha(
            [
                {
                    "physical_frame_id": row["physical_frame_id"],
                    "local_frame_index": row["local_frame_index"],
                    "source_timestamp_s": row["source_timestamp_s"],
                }
                for row in selected_meta
            ]
        ),
    }


def _track_one_case(preflight: Mapping[str, Any], decoded: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    case = preflight["case"]
    case_id = str(case["case_id"])
    track_short_window_lk = _load_track_wrapper()

    tracklets: list[dict[str, Any]] = []
    for center in case["centers"]:
        physical_ids = [int(value) for value in center["physical_frame_ids"]]
        local_ids = [int(value) for value in center["local_frame_indices"]]
        if len(physical_ids) != len(local_ids):
            raise ValueError(f"manifest frame mapping length mismatch for {case_id}")
        gray_frames = [decoded["gray_by_local"][local] for local in local_ids]
        for planned in center["matched_tracklets"]:
            tracklet_id = str(planned["tracklet_id"])
            if planned["anchor_box_xyxy_norm"] is None:
                tracklets.append(
                    {
                        **planned,
                        "status": "invalid_anchor_not_run",
                        "result": None,
                        "frames": [
                            {
                                "physical_frame_id": frame_id,
                                "local_frame_index": local_id,
                                "status": "missing",
                                "box_xyxy_norm": None,
                                "valid": False,
                            }
                            for frame_id, local_id in zip(physical_ids, local_ids)
                        ],
                    }
                )
                continue
            result = track_short_window_lk(
                gray_frames=gray_frames,
                physical_frame_ids=physical_ids,
                center_physical_frame_id=int(center["center_physical_frame_id"]),
                anchor_box_xyxy_norm=planned["anchor_box_xyxy_norm"],
                coverage_half_open=case["canonical_physical_coverage_half_open"],
            )
            raw_track = result["tracklet"]
            trajectory = raw_track.get("trajectory", [])
            if len(trajectory) != len(physical_ids):
                raise RuntimeError(f"helper trajectory length mismatch for {tracklet_id}")
            frames: list[dict[str, Any]] = []
            for frame_id, local_id, row in zip(physical_ids, local_ids, trajectory):
                if row is None:
                    frames.append(
                        {
                            "physical_frame_id": frame_id,
                            "local_frame_index": local_id,
                            "status": "missing",
                            "box_xyxy_norm": None,
                            "valid": False,
                        }
                    )
                    continue
                box = row.get("box_xyxy_norm")
                valid = row.get("status") in {"anchor", "tracked"} and box is not None
                if not valid:
                    box = None
                frames.append(
                    {
                        "physical_frame_id": int(row["physical_frame_id"]),
                        "local_frame_index": local_id,
                        "status": str(row.get("status", "missing")),
                        "box_xyxy_norm": box,
                        "valid": bool(valid),
                    }
                )
            if [row["physical_frame_id"] for row in frames] != physical_ids:
                raise RuntimeError(f"helper physical trajectory mapping changed for {tracklet_id}")
            tracklets.append(
                {
                    **planned,
                    "status": "complete",
                    "result": raw_track,
                    "frames": frames,
                }
            )

    expected = sum(len(center["matched_tracklets"]) for center in case["centers"])
    if len(tracklets) != expected:
        raise RuntimeError(f"tracklet count mismatch for {case_id}: {len(tracklets)} != {expected}")
    if len({str(row["tracklet_id"]) for row in tracklets}) != len(tracklets):
        raise RuntimeError(f"tracklet IDs are not unique for {case_id}")

    selected_meta = decoded["selected_frame_map"]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "case_id": case_id,
        "cohort": case["cohort"],
        "input_binding": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": manifest_sha,
            "source_case_path": case["source_case_path"],
            "source_case_sha256": case["source_case_sha256"],
            "source_model_state_sha256": case["source_model_state_sha256"],
            "qa_student_path": case["qa_student_path"],
            "qa_student_sha256": case["qa_student_sha256"],
            "qa_interval_physical_half_open": case["qa_interval_physical_half_open"],
            "candidate_barrier_record_sha256": case["candidate_barrier_record_sha256"],
            "canonical_request_path": case["canonical_request_path"],
            "canonical_request_sha256": case["canonical_request_sha256"],
            "source_video_sha256": case["source_video_sha256"],
            "canonical_video_sha256": case["canonical_video_sha256"],
            "canonical_pixel_sha256_declared": case["canonical_pixel_sha256"],
            "canonical_encoded_video_sha256_recomputed": preflight["encoded_video_sha256_verified"],
            "selected_frame_map_sha256": decoded["frame_map_sha256"],
        },
        "runtime": {
            "runner_sha256": _sha_file(Path(__file__).resolve()),
            "manifest_builder_sha256": _sha_file(BUILDER_PATH),
            "tracker_helper_sha256": _sha_file(HELPER_PATH),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "opencv_decode_backend": decoded["decoder_backend"],
            "opencv_threads": int(cv2.getNumThreads()),
            "cuda_or_model_imported": False,
            "device": "CPU",
        },
        "frame_readback": {
            "selected_unique_frame_count": len(selected_meta),
            "source_shape_hw": decoded["source_shape_hw"],
            "cached_rgb_and_gray_shape_hw": decoded["cached_shape_hw"],
            "maximum_tracking_edge_px": MAX_TRACK_EDGE,
            "full_canonical_pixel_sha256_recomputed": False,
            "note": "canonical_pixel_sha256 is copied as an input binding; each selected full-resolution RGB frame and each cached resized RGB/gray frame has a separately recorded SHA256.",
            "selected_frames": selected_meta,
            "rgb_npz_path": f"artifacts/stvg_short_event_tracklets_v1/private_rgb/{case_id}.npz",
        },
        "centers": [
            {
                "slot": int(center["slot"]),
                "target_physical_frame_id": float(center["target_physical_frame_id"]),
                "center_physical_frame_id": int(center["center_physical_frame_id"]),
                "physical_frame_ids": [int(v) for v in center["physical_frame_ids"]],
                "local_frame_indices": [int(v) for v in center["local_frame_indices"]],
                "candidate_ids": [int(row["candidate_id"]) for row in center["matched_tracklets"]],
                "tracklet_ids": [str(row["tracklet_id"]) for row in center["matched_tracklets"]],
                "matched_pair_key": center["matched_pair_key"],
            }
            for center in case["centers"]
        ],
        "tracklets": tracklets,
        "counts": {
            "candidate_windows": len(tracklets),
            "valid_frame_boxes": sum(frame["valid"] for row in tracklets for frame in row["frames"]),
            "missing_frame_boxes": sum(not frame["valid"] for row in tracklets for frame in row["frames"]),
            "accepted_transitions": sum(
                transition.get("status") == "accepted"
                for row in tracklets
                if row["result"] is not None
                for transition in row["result"].get("transitions", [])
            ),
            "rejected_transitions": sum(
                transition.get("status") == "rejected"
                for row in tracklets
                if row["result"] is not None
                for transition in row["result"].get("transitions", [])
            ),
            "not_attempted_transitions": sum(
                transition.get("attempted") is False
                for row in tracklets
                if row["result"] is not None
                for transition in row["result"].get("transitions", [])
            ),
        },
        "GT_read": False,
        "RGB_read": True,
        "video_decoded": True,
        "model_loaded": False,
        "GPU_used": False,
        "teacher_called": False,
    }


def _load_track_wrapper():
    from scripts.prepare_stvg_short_event_tracklets_v1 import track_short_window_lk

    return track_short_window_lk


def _check_nonoverwrite_paths() -> None:
    if MANIFEST_PATH.exists() is False:
        raise FileNotFoundError("PREPARED_MANIFEST.json must be written before the tracking stage")
    for path in [BARRIER_PATH, LOCK_PATH]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable stage artifact: {path}")
    for directory in [TRACK_DIR, RGB_DIR]:
        if directory.exists():
            raise FileExistsError(f"refusing to reuse existing output directory: {directory}")


def run_once() -> dict[str, Any]:
    _check_nonoverwrite_paths()
    manifest_raw = MANIFEST_PATH.read_bytes()
    manifest_sha = _sha_bytes(manifest_raw)
    manifest = json.loads(manifest_raw.decode("utf-8"))

    lock_stream, preflight = _acquire_lock_and_preflight(
        LOCK_PATH, lambda: _preflight_inputs(manifest, manifest_sha)
    )
    try:
        print(f"preflight complete: cases={len(preflight)} manifest_sha256={manifest_sha}", flush=True)

        cv2.setNumThreads(2)
        decoded_cases: dict[str, dict[str, Any]] = {}
        # Decode only the union of selected +/-2 windows per case, in canonical
        # local-frame order. Keep only resized RGB/gray arrays for the finite
        # render/track stage; raw full-resolution RGB is hashed then discarded.
        for item in preflight:
            case_id = str(item["case"]["case_id"])
            decoded_cases[case_id] = _decode_selected_frames(item)
            print(
                f"decoded {case_id}: unique_frames={len(decoded_cases[case_id]['selected_frame_map'])} backend={decoded_cases[case_id]['decoder_backend']}",
                flush=True,
            )

        case_results: list[dict[str, Any]] = []
        for item in preflight:
            case_id = str(item["case"]["case_id"])
            row = _track_one_case(item, decoded_cases[case_id], manifest_sha)
            case_results.append(row)
            print(
                f"tracked {case_id}: windows={row['counts']['candidate_windows']} valid_boxes={row['counts']['valid_frame_boxes']}",
                flush=True,
            )

        expected_tracklets = int(manifest["tracklet_proposal_count"])
        actual_tracklets = sum(len(row["tracklets"]) for row in case_results)
        if actual_tracklets != expected_tracklets or len(case_results) != 8:
            raise RuntimeError(f"final count mismatch: cases={len(case_results)} tracklets={actual_tracklets}")
        if sum(row["frame_readback"]["selected_unique_frame_count"] for row in case_results) != sum(
            len(decoded_cases[str(item["case"]["case_id"])]["selected_frame_map"]) for item in preflight
        ):
            raise RuntimeError("selected RGB readback counts disagree")

        write_records: list[dict[str, Any]] = []
        case_result_by_id = {row["case_id"]: row for row in case_results}
        for item in preflight:
            case = item["case"]
            case_id = str(case["case_id"])
            decoded = decoded_cases[case_id]
            rgb_meta = decoded["selected_frame_map"]
            local_ids = [int(row["local_frame_index"]) for row in rgb_meta]
            physical_ids = [int(row["physical_frame_id"]) for row in rgb_meta]
            npz_record = _write_once_npz(
                RGB_DIR / f"{case_id}.npz",
                rgb_uint8=np.stack([decoded["rgb_by_local"][local] for local in local_ids], axis=0),
                gray_uint8=np.stack([decoded["gray_by_local"][local] for local in local_ids], axis=0),
                local_frame_indices=np.asarray(local_ids, dtype=np.int64),
                physical_frame_ids=np.asarray(physical_ids, dtype=np.int64),
                manifest_sha256=np.asarray(manifest_sha, dtype="U64"),
            )
            record = case_result_by_id[case_id]
            record["frame_readback"]["rgb_npz_sha256"] = npz_record["sha256"]
            record["frame_readback"]["rgb_npz_bytes"] = npz_record["bytes"]
            write_records.append(_write_once_json(TRACK_DIR / f"{case_id}.json", record))
            print(f"saved {case_id}: track/json + private RGB npz", flush=True)

        all_files = sorted(write_records + [
            {
                "path": (RGB_DIR / f"{case_id}.npz").relative_to(ROOT).as_posix(),
                "sha256": _sha_file(RGB_DIR / f"{case_id}.npz"),
                "bytes": (RGB_DIR / f"{case_id}.npz").stat().st_size,
            }
            for case_id in CASE_IDS
        ], key=lambda row: row["path"])
        if len(all_files) != 16 or len({row["path"] for row in all_files}) != 16:
            raise RuntimeError("global output file set is not exactly eight JSON plus eight RGB npz")

        barrier = {
            "schema": BARRIER_SCHEMA,
            "status": "complete",
            "scope": "fixed eight historically exposed development cases; finite CPU short-window LK only",
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": manifest_sha,
            "runner_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "runner_sha256": _sha_file(Path(__file__).resolve()),
            "tracker_helper_path": HELPER_PATH.relative_to(ROOT).as_posix(),
            "tracker_helper_sha256": _sha_file(HELPER_PATH),
            "case_ids": list(CASE_IDS),
            "count": 8,
            "center_count": 32,
            "tracklet_count": actual_tracklets,
            "decoded_selected_unique_frame_count": sum(
                case["frame_readback"]["selected_unique_frame_count"] for case in case_results
            ),
            "files": all_files,
            "case_records": [
                {
                    "case_id": row["case_id"],
                    "path": f"artifacts/stvg_short_event_tracklets_v1/tracklets/{row['case_id']}.json",
                    "sha256": next(file["sha256"] for file in all_files if file["path"].endswith(f"tracklets/{row['case_id']}.json")),
                    "rgb_npz_path": f"artifacts/stvg_short_event_tracklets_v1/private_rgb/{row['case_id']}.npz",
                    "rgb_npz_sha256": next(file["sha256"] for file in all_files if file["path"].endswith(f"private_rgb/{row['case_id']}.npz")),
                    "tracklet_count": row["counts"]["candidate_windows"],
                    "GT_read": False,
                }
                for row in case_results
            ],
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
            "canonical_pixel_sha256_policy": "source REQUEST digest is recorded as declared metadata; this stage recomputed each selected full-resolution RGB-frame digest but did not decode the entire video to recompute the full-video pixel digest",
            "GT_read": False,
            "RGB_read": True,
            "video_decoded": True,
            "model_loaded": False,
            "GPU_used": False,
            "teacher_called": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_once_json(BARRIER_PATH, barrier)
        print(
            json.dumps(
                {
                    "status": barrier["status"],
                    "case_count": barrier["count"],
                    "center_count": barrier["center_count"],
                    "tracklet_count": barrier["tracklet_count"],
                    "selected_unique_frames": barrier["decoded_selected_unique_frame_count"],
                    "barrier_path": barrier["schema"],
                    "manifest_sha256": manifest_sha,
                    "GT_read": False,
                    "GPU_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return barrier
    finally:
        lock_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the one fixed CPU-only tracking pass")
    args = parser.parse_args()
    if not args.execute:
        print("Execution is gated; pass --execute for the root-approved fixed old8 CPU stage.")
        return 2
    started = time.monotonic()
    barrier = run_once()
    print(f"elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
    if barrier.get("status") != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
