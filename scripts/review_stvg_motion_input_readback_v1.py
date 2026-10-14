"""Independent CPU/no-GT readback of sealed motion-guided inputs.

This reader re-decodes only the canonical source media fields from REQUEST.json,
loads only zero.boxes/frame_ids/observation_positions from each trusted source
case, and replays the already-pinned LK helper. It never reads teacher outputs,
labels, annotations, query text, or model state values.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_stvg_scene_object_id_v1 import _raster_box  # noqa: E402
from scripts.stvg_box_scale_grid_v1 import full_scale_grid  # noqa: E402
from scripts.stvg_motion_tracklet_lk_v1 import track_second_observation  # noqa: E402


BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
NEW = BASE / "display_order_layout001"
CANONICAL_INPUTS = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
SOURCE_CASES = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1/source_cases"
OLD_GRID = ROOT / "artifacts/stvg_box_scale_global_evidence_v1"
OLD_REFERENTS = ROOT / "artifacts/stvg_video_conditioned_global_referent_v1"
REPORT_PATH = BASE / "source_audit/LUNA_ACTUAL_INPUT_READBACK.json"
TRACKLET_CONTRACTS = BASE / "CPU_TRACKLET_CONTRACTS.json"
SCHEMA = "stvg-motion-actual-input-readback-v1"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def root_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("repository-relative receipt path escaped repository root")
    return path


def canonical_json_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))


def as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def load_source_numeric_fields(roster_row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    relative = str(roster_row["source_case_path"])
    path = root_path(relative)
    if not path.is_relative_to(SOURCE_CASES.resolve()):
        raise ValueError("source case path escaped the frozen source_cases directory")
    receipt_path = path.with_suffix(".json")
    receipt = read_json(receipt_path)
    expected = str(roster_row["source_case_sha256"])
    if (receipt.get("status") != "accepted_complete_source_only_case"
            or receipt.get("GT_read") is not False
            or receipt.get("sha256") != expected
            or receipt.get("bytes") != path.stat().st_size
            or sha_file(path) != expected):
        raise ValueError(f"source case bytes/receipt mismatch for {roster_row['case_id']}")

    import torch

    multiarray = getattr(np.core, "multiarray", None)
    if multiarray is None:
        multiarray = np._core.multiarray
    safe_globals: list[Any] = [multiarray._reconstruct, np.ndarray, np.dtype]
    safe_globals.extend(
        type(np.dtype(dtype))
        for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8)
    )
    with torch.serialization.safe_globals(safe_globals):
        case = torch.load(path, map_location="cpu", weights_only=True)
    try:
        # Intentionally access only these three numeric fields.
        boxes = as_numpy(case["zero"]["boxes"], dtype=np.float64)
        frame_ids = as_numpy(case["frame_ids"], dtype=np.int64)
        observation_positions = as_numpy(case["observation_positions"], dtype=np.int64)
    finally:
        del case
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError(f"source zero.boxes is not finite N×4 for {roster_row['case_id']}")
    if frame_ids.shape != (len(boxes),) or len(frame_ids) < 2 or not np.all(np.diff(frame_ids) > 0):
        raise ValueError(f"source frame_ids do not align with zero.boxes for {roster_row['case_id']}")
    if observation_positions.shape != (4,) or not np.all(np.diff(observation_positions) > 0):
        raise ValueError(f"expected four increasing observation positions for {roster_row['case_id']}")
    if np.any(observation_positions < 0) or np.any(observation_positions >= len(frame_ids)):
        raise ValueError(f"source observation position outside frame_ids for {roster_row['case_id']}")
    return boxes, frame_ids, observation_positions, path


def decode_selected_frames(
    case_id: str,
    request: Mapping[str, Any],
    physical_ids: list[int],
    tracking_ids: list[int],
    frame_map: Mapping[int, int],
    expected_rgb: Mapping[str, str],
    expected_gray: Mapping[str, str],
) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[str, str], dict[str, str]]:
    video = Path(str(request["video_path"])).resolve()
    if not video.is_relative_to(CANONICAL_INPUTS.resolve()):
        raise ValueError(f"canonical video path escaped canonical input root for {case_id}")
    if video.parent != (CANONICAL_INPUTS / case_id).resolve():
        raise ValueError(f"canonical video path is not case-scoped for {case_id}")
    if sha_file(video) != request.get("canonical_video_sha256"):
        raise ValueError(f"canonical video SHA mismatch for {case_id}")
    if len(set(physical_ids)) != 4 or any(value not in frame_map for value in tracking_ids):
        raise ValueError(f"invalid selected physical frame mapping for {case_id}")

    targets: dict[int, list[int]] = {}
    for physical in tracking_ids:
        local = int(frame_map[physical])
        targets.setdefault(local, []).append(physical)
    if any(len(physicals) != 1 for physicals in targets.values()):
        raise ValueError(f"selected physical IDs collide at local frame index for {case_id}")

    gray_frames: list[np.ndarray | None] = [None] * len(tracking_ids)
    source_wh: list[tuple[int, int] | None] = [None] * len(tracking_ids)
    rgb_sha: dict[str, str] = {}
    gray_sha: dict[str, str] = {}
    tracking_slot = {physical: index for index, physical in enumerate(tracking_ids)}
    selected = set(physical_ids)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open canonical source video for {case_id}")
    local_index = 0
    try:
        last = max(targets)
        while local_index <= last and targets:
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"canonical source ended before selected frame for {case_id}")
            physicals = targets.pop(local_index, None)
            if physicals:
                physical = physicals[0]
                slot = tracking_slot[physical]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray_frames[slot] = gray
                source_wh[slot] = (int(gray.shape[1]), int(gray.shape[0]))
                gray_hash = sha_bytes(gray.tobytes())
                gray_sha[str(physical)] = gray_hash
                if gray_hash != expected_gray.get(str(physical)):
                    raise ValueError(f"decoded gray SHA differs from saved tracking hash for {case_id}/{physical}")
                if physical in selected:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_hash = sha_bytes(rgb.tobytes())
                    rgb_sha[str(physical)] = rgb_hash
                    if rgb_hash != expected_rgb.get(str(physical)):
                        raise ValueError(f"decoded RGB SHA differs from saved source hash for {case_id}/{physical}")
            local_index += 1
    finally:
        cap.release()
    if targets or any(frame is None for frame in gray_frames) or any(value is None for value in source_wh):
        raise ValueError(f"not all tracking frames decoded for {case_id}")
    return list(gray_frames), list(source_wh), rgb_sha, gray_sha  # type: ignore[arg-type]


def expected_additional_windows(zero_at_anchor: np.ndarray) -> tuple[dict[str, Any], list[int]]:
    grid = full_scale_grid(zero_at_anchor)
    remaining = list(grid["windows"])
    chosen: list[int] = []
    centers = [np.asarray(zero_at_anchor[:2], dtype=np.float64)]
    while remaining and len(chosen) < 4:
        window = min(
            remaining,
            key=lambda row: (
                -min(float(np.sum((np.asarray([row["cx"], row["cy"]]) - point) ** 2))
                     for point in centers),
                int(row["window_id"]),
            ),
        )
        chosen.append(int(window["window_id"]))
        centers.append(np.asarray([window["cx"], window["cy"]], dtype=np.float64))
        remaining.remove(window)
    if len(chosen) != 4:
        raise ValueError("pinned coarse grid did not provide four non-Source windows")
    return grid, chosen


def xyxy_from_cxcywh(box: np.ndarray) -> np.ndarray:
    return np.concatenate((box[:2] - box[2:] / 2.0, box[:2] + box[2:] / 2.0))


def normalized_wh(boxes_cxcywh: np.ndarray, frame_ids: np.ndarray, physical_ids: list[int]) -> np.ndarray:
    return np.stack([np.interp(physical_ids, frame_ids, boxes_cxcywh[:, j]) for j in (2, 3)], axis=1)


def verify_file_manifest(barrier: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = barrier.get("files")
    if not isinstance(rows, list):
        raise ValueError("input barrier lacks a file manifest")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        relative = str(row["path"])
        if relative in out:
            raise ValueError("duplicate path in sealed barrier manifest")
        path = root_path(relative)
        expected_bytes = row.get("bytes")
        if (not path.is_file()
                or (expected_bytes is not None and path.stat().st_size != int(expected_bytes))
                or sha_file(path) != row["sha256"]):
            raise ValueError(f"sealed input file does not match barrier: {relative}")
        out[relative] = row
    return out


def _verify_one_job_tensor(
    job: Mapping[str, Any],
    barrier_record: Mapping[str, Any],
    render_path: Path,
    expected_source_rgb: Mapping[str, str],
    source_candidates: Mapping[int, Mapping[str, Any]],
    expected_times: list[float],
    base_render_path: Path | None = None,
) -> dict[str, Any]:
    if len(job.get("videos", [])) != 1:
        raise ValueError(f"job must contain exactly one input video: {job.get('job_id')}")
    if job.get("source_case_sha256") != barrier_record.get("source_case_sha256"):
        raise ValueError(f"job source-case binding differs from candidate barrier: {job.get('job_id')}")
    video = job["videos"][0]
    path = root_path(video["path"])
    if sha_file(path) != video.get("sha256"):
        raise ValueError(f"job file SHA mismatch: {job.get('job_id')}")
    array = np.load(path, allow_pickle=False)
    if array.shape != (4, 3, 448, 448) or array.dtype != np.uint8:
        raise ValueError(f"unexpected teacher tensor ABI: {job.get('job_id')}")
    tensor_sha = sha_bytes(np.ascontiguousarray(array).tobytes())
    if tensor_sha != video.get("tensor_sha256"):
        raise ValueError(f"job tensor SHA mismatch: {job.get('job_id')}")
    if video.get("actual_times_seconds") != expected_times:
        raise ValueError(f"job timestamps differ from candidate barrier: {job.get('job_id')}")
    expected_fps = 3.0 / (float(video["actual_times_seconds"][-1]) - float(video["actual_times_seconds"][0]))
    if not math.isclose(float(video["fps"]), expected_fps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"job FPS differs from four-frame mean-gap schedule: {job.get('job_id')}")
    if sha_bytes(str(job.get("prompt", "")).encode("utf-8")) != job.get("prompt_sha256"):
        raise ValueError(f"prompt hash mismatch (value withheld): {job.get('job_id')}")

    render = read_json(render_path)
    base_render = read_json(base_render_path) if base_render_path is not None else None
    maps = render.get("id_map", {})
    colors = render.get("color_map", {})
    order = render.get("draw_order", [])
    if set(maps) != set(colors) or sorted(int(x) for x in maps.values()) != list(range(1, len(maps) + 1)):
        raise ValueError(f"invalid control ID/color map: {job.get('job_id')}")
    binding = {int(row["candidate_id"]): row for row in job["candidate_bindings"]}
    if set(binding) != set(map(int, maps.keys())):
        raise ValueError(f"candidate bindings do not align to render map: {job.get('job_id')}")
    if [int(cid) for cid in order] != list(range(1, len(maps) + 1)) and sorted(map(int, order)) != list(range(1, len(maps) + 1)):
        raise ValueError(f"draw order is not a candidate permutation: {job.get('job_id')}")
    for cid, row in binding.items():
        key = str(cid)
        if row.get("visible_id") != maps[key] or row.get("color_index") != colors[key]:
            raise ValueError(f"job/render candidate map mismatch: {job.get('job_id')}")

    frame_hashes: list[str] = []
    for slot, row in enumerate(render.get("records", [])):
        if int(row.get("slot", -1)) != slot:
            raise ValueError(f"render metadata slot order mismatch: {job.get('job_id')}")
        physical = int(barrier_record["physical_frame_ids"][slot])
        if row.get("source_RGB_sha256") != expected_source_rgb.get(str(physical)):
            raise ValueError(f"render source frame does not match canonical decode: {job.get('job_id')}")
        frame_rgb = np.ascontiguousarray(array[slot].transpose(1, 2, 0))
        digest = sha_bytes(frame_rgb.tobytes())
        if digest != row.get("rendered_RGB_sha256"):
            raise ValueError(f"rendered tensor frame SHA mismatch: {job.get('job_id')}")
        frame_hashes.append(digest)
        overlays = {int(x["candidate_id"]): x for x in row.get("overlays", [])}
        if len(overlays) != len(row.get("overlays", [])):
            raise ValueError(f"duplicate rendered candidate overlay: {job.get('job_id')}")
        expected_ids = set(map(int, maps)) - set(map(int, row.get("missing_candidate_ids", [])))
        if set(overlays) != expected_ids:
            raise ValueError(f"render overlay/missing declarations disagree: {job.get('job_id')}")
        for cid, overlay in overlays.items():
            candidate = source_candidates[cid]
            norm_box = candidate["boxes_xyxy"][slot]
            if norm_box is None:
                raise ValueError(f"rendered a missing source candidate: {job.get('job_id')}")
            expected_pixels = list(_raster_box(norm_box, row["letterbox"]))
            if list(map(int, overlay["box_xyxy_pixels"])) != expected_pixels:
                raise ValueError(f"rendered box does not bind to sealed source candidate: {job.get('job_id')}")
            if int(overlay["visible_id"]) != int(maps[str(cid)]) or int(overlay["palette_index"]) != int(colors[str(cid)]):
                raise ValueError(f"overlay identity/color differs from sealed job: {job.get('job_id')}")
        if base_render is not None:
            base_rows = base_render.get("records", [])
            if len(base_rows) != 4:
                raise ValueError(f"base badge source lacks four slots: {job.get('job_id')}")
            base_overlay = {int(x["candidate_id"]): x for x in base_rows[slot].get("overlays", [])}
            for cid, overlay in overlays.items():
                if (cid not in base_overlay
                        or overlay.get("badge_xyxy_pixels") != base_overlay[cid].get("badge_xyxy_pixels")):
                    raise ValueError(f"fixed badge coordinate differs from base layout: {job.get('job_id')}")
    if len(frame_hashes) != 4:
        raise ValueError(f"render metadata frame count is not four: {job.get('job_id')}")
    return {"job_id": job["job_id"], "frame_RGB_hashes": frame_hashes,
            "tensor_sha256": tensor_sha, "prompt_sha256": job["prompt_sha256"],
            "frame_binding_verified": True}


def run_readback() -> dict[str, Any]:
    cv2.setNumThreads(2)
    started = time.perf_counter()
    roster_path = BASE / "ROSTER_LOCK.json"
    old_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    old_jobs_path = BASE / "TEACHER_JOB_LOCK.json"
    new_barrier_path = NEW / "CANDIDATE_INPUT_BARRIER.json"
    new_jobs_path = NEW / "TEACHER_JOB_LOCK.json"
    runtime_path = BASE / "RUNTIME.json"
    design_path = BASE / "DESIGN_LOCK.json"
    protocol_path = ROOT / "protocols/stvg_motion_guided_instance_opd_v1.md"
    helper_path = ROOT / "scripts/stvg_motion_tracklet_lk_v1.py"
    old_prepare_path = ROOT / "scripts/prepare_stvg_motion_guided_v1.py"
    grid_helper_path = ROOT / "scripts/stvg_box_scale_grid_v1.py"
    render_helper_path = ROOT / "scripts/prepare_stvg_scene_object_id_v1.py"

    roster = read_json(roster_path)
    old_barrier = read_json(old_barrier_path)
    old_job_lock = read_json(old_jobs_path)
    new_barrier = read_json(new_barrier_path)
    new_job_lock = read_json(new_jobs_path)
    runtime = read_json(runtime_path)
    design = read_json(design_path)
    old_referents = read_json(OLD_REFERENTS / "CANDIDATE_INPUT_BARRIER.json")
    old_grid_barrier = read_json(OLD_GRID / "GRID_INPUT_BARRIER.json")
    helper_contracts = read_json(TRACKLET_CONTRACTS)
    layout_capture = read_json(NEW / "CAPTURE.json")

    if roster.get("status") != "locked" or roster.get("GT_read") is not False or roster.get("count") != 16:
        raise ValueError("source roster is not the expected locked 16-case no-GT roster")
    if old_barrier.get("status") != "complete" or old_barrier.get("GT_read") is not False or old_barrier.get("count") != 16:
        raise ValueError("original candidate barrier is not sealed no-GT/16")
    if old_job_lock.get("status") != "locked" or old_job_lock.get("GT_read") is not False or old_job_lock.get("count") != 96:
        raise ValueError("original teacher job lock is not 96/locked/no-GT")
    if new_barrier.get("status") != "complete" or new_barrier.get("GT_read") is not False or new_barrier.get("count") != 32:
        raise ValueError("new fixed-layout candidate barrier is not complete 32/no-GT")
    if new_job_lock.get("status") != "locked" or new_job_lock.get("GT_read") is not False or new_job_lock.get("count") != 32:
        raise ValueError("new fixed-layout teacher job lock is not locked 32/no-GT")
    if len(roster.get("records", [])) != 16 or len(old_barrier.get("records", [])) != 16:
        raise ValueError("the roster and old barrier must each contain 16 records")
    if len(new_barrier.get("records", [])) != 32 or len(old_job_lock.get("records", [])) != 96 or len(new_job_lock.get("records", [])) != 32:
        raise ValueError("sealed record counts do not match their declarations")
    old_barrier_sha = sha_file(old_barrier_path)
    old_jobs_sha = sha_file(old_jobs_path)
    if layout_capture.get("source_candidate_barrier_sha256") != old_barrier_sha:
        raise ValueError("fixed-layout capture does not bind the old candidate barrier")
    if layout_capture.get("source_teacher_job_lock_sha256") != old_jobs_sha:
        raise ValueError("fixed-layout capture does not bind the old teacher job lock")
    new_barrier_sha = sha_file(new_barrier_path)
    if new_barrier.get("source_candidate_barrier_sha256") != old_barrier_sha:
        raise ValueError("fixed-layout barrier does not bind the original candidate barrier")
    if new_barrier.get("source_teacher_job_lock_sha256") != old_jobs_sha:
        raise ValueError("fixed-layout barrier does not bind the original teacher job lock")
    if new_job_lock.get("source_candidate_input_barrier_sha256") != new_barrier_sha:
        raise ValueError("fixed-layout teacher job lock does not bind its candidate barrier")
    if new_job_lock.get("source_candidate_barrier_sha256") != old_barrier_sha:
        raise ValueError("fixed-layout teacher job lock does not bind the original candidate barrier")
    if new_job_lock.get("source_teacher_job_lock_sha256") != old_jobs_sha:
        raise ValueError("fixed-layout teacher job lock does not bind the original teacher lock")
    if old_referents.get("old_grid_barrier_sha256") != sha_file(OLD_GRID / "GRID_INPUT_BARRIER.json"):
        raise ValueError("old retrieved-union barrier points to a different coarse-grid barrier")
    if design.get("protocol_sha256") != sha_file(protocol_path):
        raise ValueError("protocol does not match the source design lock")
    if helper_contracts.get("status") != "pass" or helper_contracts.get("helper_sha256") != sha_file(helper_path):
        raise ValueError("LK helper source no longer matches its synthetic CPU contract pin")
    if old_grid_barrier.get("status") != "complete" or old_grid_barrier.get("GT_read") is not False:
        raise ValueError("old coarse grid barrier is not complete/no-GT")
    if old_referents.get("status") != "complete" or old_referents.get("GT_read") is not False:
        raise ValueError("old retrieved-union barrier is not complete/no-GT")

    # Check the source-code and protocol pins only; never inspect model or weight payloads.
    pin_checks: dict[str, str] = {}
    source_paths = ("scripts/run_stvg_box_scale_teacher_v1.py",
                    "scripts/stvg_temporal_caption_query_v1.py",
                    "scripts/run_stvg_motion_guided_teacher_v1.py",
                    "scripts/launch_stvg_motion_guided_teacher_v1.py",
                    "scripts/prepare_stvg_motion_guided_v1.py",
                    "scripts/stvg_motion_tracklet_lk_v1.py",
                    "scripts/stvg_box_scale_grid_v1.py",
                    "scripts/prepare_stvg_scene_object_id_v1.py",
                    "protocols/stvg_motion_guided_instance_opd_v1.md")
    all_runtime_pins = {**runtime.get("pins", {}), **runtime.get("protected", {})}
    for relative in source_paths:
        expected = all_runtime_pins.get(relative)
        if not isinstance(expected, str) or sha_file(root_path(relative)) != expected:
            raise ValueError(f"runtime source pin mismatch: {relative}")
        pin_checks[relative] = expected
    if (layout_capture.get("source_render_helpers_sha256")
            != sha_file(render_helper_path)):
        raise ValueError("fixed-layout render helper does not match capture binding")
    if layout_capture.get("source_helper_sha256") != sha_file(ROOT / "scripts/prepare_stvg_motion_order_layout001.py"):
        raise ValueError("fixed-layout preparation helper does not match capture binding")

    # Verify all listed files for both barriers before loading any job tensors.
    old_manifest = verify_file_manifest(old_barrier)
    new_manifest = verify_file_manifest(new_barrier)
    grid_manifest = verify_file_manifest(old_grid_barrier)
    retrieval_manifest = verify_file_manifest(old_referents)
    if len(old_manifest) != 192:
        raise ValueError("old candidate barrier no longer lists exactly 192 input/render files")
    if len(new_manifest) != 64:
        raise ValueError("fixed-layout barrier no longer lists exactly 64 tensor/render files")
    if len(grid_manifest) != 160 or len(retrieval_manifest) != 34:
        raise ValueError("old seed-source file manifests differ from the sealed expected counts")

    case_record_shas = {
        row["case_id"]: canonical_json_sha(row)
        for row in old_barrier["records"]
    }
    if new_barrier.get("source_candidate_records_sha256") != case_record_shas:
        raise ValueError("fixed-layout barrier does not bind every original candidate record")

    roster_rows = {row["case_id"]: row for row in roster["records"]}
    old_records = {row["case_id"]: row for row in old_barrier["records"]}
    old_referent_records = {row["case_id"]: row for row in old_referents["records"]}
    grid_records = {row["case_id"]: row for row in old_grid_barrier["records"]}
    new_records: dict[str, dict[str, dict[str, Any]]] = {}
    for row in new_barrier["records"]:
        new_records.setdefault(row["case_id"], {})[row["control"]] = row
    old_jobs = {row["job_id"]: row for row in old_job_lock["records"]}
    new_jobs = {row["job_id"]: row for row in new_job_lock["records"]}
    if len(old_jobs) != 96 or len(new_jobs) != 32:
        raise ValueError("duplicate job ID in a sealed teacher lock")

    # Recover expected source tracklet configuration from the trusted source-only
    # checkpoint container, using only zero.boxes, frame_ids, observations.
    per_case: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    old_source_candidate_total = 0
    new_coarse_candidate_total = 0
    old_union_window_total = 0
    total_non_source_replays = 0
    total_track_positions = 0
    total_attempted_transitions = 0
    total_accepted_transitions = 0
    total_explicit_none_positions = 0
    for case_id in sorted(roster_rows):
        roster_row = roster_rows[case_id]
        barrier_record = old_records[case_id]
        if roster_row.get("source_case_sha256") != barrier_record.get("source_case_sha256"):
            raise ValueError(f"roster/source barrier SHA mismatch for {case_id}")
        if roster_row.get("cohort") != barrier_record.get("cohort") or roster_row.get("dataset") != barrier_record.get("dataset"):
            raise ValueError(f"roster/source barrier cohort mismatch for {case_id}")
        source_boxes, source_frame_ids, observation_positions, source_path = load_source_numeric_fields(roster_row)
        source_case_hash = sha_file(source_path)
        physical_obs = [int(source_frame_ids[pos]) for pos in observation_positions]
        if physical_obs != [int(v) for v in barrier_record["physical_frame_ids"]]:
            raise ValueError(f"source observation frames differ from sealed candidate barrier for {case_id}")
        if [int(v) for v in observation_positions] != [int(v) for v in barrier_record["positions"]]:
            raise ValueError(f"source observation positions differ from sealed candidate barrier for {case_id}")
        tracking_ids = [int(v) for v in barrier_record["tracking_physical_frame_ids"]]
        expected_tracking = sorted(set(range(physical_obs[0], physical_obs[-1] + 1, 4)) | set(physical_obs))
        if tracking_ids != expected_tracking:
            raise ValueError(f"tracking frame sequence differs from fixed 4-physical-frame schedule for {case_id}")
        tracking_obs_positions = [tracking_ids.index(value) for value in physical_obs]
        # Decode the REQUEST-selected canonical media fields only. The JSON may
        # contain query text; it is never accessed or emitted.
        request_path = CANONICAL_INPUTS / case_id / "REQUEST.json"
        request = read_json(request_path)
        video = Path(str(request["video_path"])).resolve()
        if not video.is_relative_to(CANONICAL_INPUTS.resolve()) or video.parent != (CANONICAL_INPUTS / case_id).resolve():
            raise ValueError(f"canonical media path escaped its fixed case scope for {case_id}")
        if sha_file(video) != request.get("canonical_video_sha256") or request.get("canonical_video_sha256") != barrier_record.get("canonical_video_sha256"):
            raise ValueError(f"canonical media bytes are not bound to the old input barrier for {case_id}")
        request_sha = sha_file(request_path)
        if request.get("source_video_sha256") is None:
            raise ValueError(f"canonical media request lacks original source-video receipt for {case_id}")
        frame_map: dict[int, int] = {}
        for row in request["frame_map"]:
            physical, local = int(row["physical_frame_id"]), int(row["local_frame_index"])
            if physical in frame_map or local < 0:
                raise ValueError(f"malformed request frame map for {case_id}")
            frame_map[physical] = local

        rgb_hash_map = barrier_record["source_RGB_sha256_by_frame"]
        gray_hash_map = barrier_record["tracking_gray_sha256_by_frame"]
        gray_frames, source_wh, decoded_rgb_hashes, decoded_gray_hashes = decode_selected_frames(
            case_id, request, physical_obs, tracking_ids, frame_map, rgb_hash_map, gray_hash_map
        )
        rgb_binding = {str(fid): decoded_rgb_hashes[str(fid)] for fid in physical_obs}
        gray_binding = {str(fid): decoded_gray_hashes[str(fid)] for fid in tracking_ids}

        grid_record = grid_records.get(case_id)
        old_referent = old_referent_records.get(case_id)
        if barrier_record["cohort"] == "old8":
            if grid_record is None or old_referent is None:
                raise ValueError(f"old8 candidate is missing its locked source grid/retrieval record for {case_id}")
            for key in ("source_case_sha256", "source_pixel_sha256", "positions", "physical_frame_ids"):
                if key in old_referent and key in barrier_record and old_referent[key] != barrier_record[key]:
                    raise ValueError(f"old retrieval seed provenance differs for {case_id}: {key}")
            if grid_record.get("source_case_sha256") != source_case_hash:
                raise ValueError(f"old grid input is not bound to current source case for {case_id}")
            source_second_rgb = rgb_binding[str(physical_obs[1])]
            if grid_record.get("physical_frame_id") != physical_obs[1] or grid_record.get("source_RGB_sha256") != source_second_rgb:
                raise ValueError(f"old grid observation is not bound to the same second source frame for {case_id}")
            grid = grid_record["grid"]
            windows = grid["windows"]
            union = [int(v) for v in old_referent["retrieved_union_window_ids"]]
            if [int(v) for v in barrier_record["retrieved_union_window_ids"]] != union:
                raise ValueError(f"old union window IDs differ from source retrieval barrier for {case_id}")
            if barrier_record.get("seed_rule") != "old_GT_free_marked_masked_top3_union":
                raise ValueError(f"old cohort seed rule changed for {case_id}")
            old_union_window_total += len(union)
        else:
            grid, selected_windows = expected_additional_windows(source_boxes[observation_positions[1]])
            windows = grid["windows"]
            union = [0] + selected_windows
            if barrier_record.get("seed_rule") != "fixed_farthest_center_coarse_grid_not_teacher_retrieval":
                raise ValueError(f"additional cohort is not on the locked source-only coarse seed rule for {case_id}")
            new_coarse_candidate_total += len(union)

        candidates = barrier_record["candidates"]
        expected_window_order = [0] + union if union and union[0] != 0 else union
        actual_windows = [int(row["window_id"]) for row in candidates]
        if actual_windows != expected_window_order:
            raise ValueError(f"candidate window order does not match its locked seed rule for {case_id}")
        if candidates[0].get("is_Source") is not True or int(candidates[0]["window_id"]) != 0:
            raise ValueError(f"Source candidate is not first for {case_id}")

        source_wh_all = normalized_wh(source_boxes, source_frame_ids, tracking_ids)
        if not np.isfinite(source_wh_all).all() or np.any(source_wh_all <= 0.0) or np.any(source_wh_all > 1.0):
            raise ValueError(f"interpolated Source width/height out of legal normalized range for {case_id}")
        source_boxes_obs = [xyxy_from_cxcywh(source_boxes[position]).tolist() for position in observation_positions]
        source_candidate = candidates[0]
        if source_candidate.get("boxes_xyxy") != source_boxes_obs:
            raise ValueError(f"Source candidate readout does not match original zero.boxes for {case_id}")
        if source_candidate.get("positions") != observation_positions.tolist() or source_candidate.get("physical_frame_ids") != physical_obs:
            raise ValueError(f"Source candidate observation binding mismatch for {case_id}")

        window_by_id = {int(row["window_id"]): row for row in windows}
        tracking_results: list[dict[str, Any]] = []
        source_candidate_map = {int(row["candidate_id"]): row for row in candidates}
        obs_box_positions = tracking_obs_positions
        for candidate in candidates[1:]:
            candidate_id = int(candidate["candidate_id"])
            window_id = int(candidate["window_id"])
            if candidate.get("is_Source") is not False or candidate.get("positions") != observation_positions.tolist() or candidate.get("physical_frame_ids") != physical_obs:
                raise ValueError(f"non-Source candidate frame binding mismatch for {case_id}/{candidate_id}")
            window = window_by_id.get(window_id)
            if window is None:
                raise ValueError(f"candidate window ID not in pinned grid for {case_id}/{window_id}")
            anchor = xyxy_from_cxcywh(np.asarray([window["cx"], window["cy"],
                                                 source_boxes[observation_positions[1], 2],
                                                 source_boxes[observation_positions[1], 3]], dtype=np.float64))
            legal = (np.isfinite(anchor).all() and 0.0 <= anchor[0] < anchor[2] <= 1.0
                     and 0.0 <= anchor[1] < anchor[3] <= 1.0)
            if not legal:
                replay = {"status": "invalid_anchor", "geometry_clamped": False}
            else:
                replay = track_second_observation(
                    gray_frames,
                    source_wh,
                    tracking_obs_positions,
                    [anchor.tolist()] * len(tracking_obs_positions),
                    normalized_wh_by_frame=source_wh_all.tolist(),
                )
            saved_track = candidate.get("tracklet")
            saved_hash = canonical_json_sha(saved_track)
            replay_hash = canonical_json_sha(replay)
            if replay != saved_track:
                differing_top_keys = sorted(
                    key for key in set(saved_track or {}) | set(replay)
                    if (saved_track or {}).get(key) != replay.get(key)
                )
                raise ValueError(
                    f"saved tracklet differs from exact pinned-helper replay for {case_id}/{candidate_id}; "
                    f"saved_sha={saved_hash}; replay_sha={replay_hash}; differing_top_keys={differing_top_keys}; "
                    f"anchor_sha={canonical_json_sha(anchor.tolist())}; "
                    f"width_sha={sha_bytes(np.ascontiguousarray(source_wh_all).tobytes())}; "
                    f"gray_sequence_sha={sha_bytes(b''.join(frame.tobytes() for frame in gray_frames))}"
                )
            sampled = [replay.get("trajectory", [])[position].get("box_xyxy_norm")
                       for position in obs_box_positions] if legal else [None] * 4
            if candidate.get("boxes_xyxy") != sampled:
                raise ValueError(f"sampled candidate boxes differ from replay at the four fixed observations for {case_id}/{candidate_id}")
            trajectory = replay.get("trajectory", []) if legal else []
            statuses = Counter(row.get("status") for row in trajectory)
            none_positions = [int(row["position"]) for row in trajectory if row.get("box_xyxy_norm") is None]
            failure_reason_by_position = {
                str(row["position"]): row["reason"]
                for row in trajectory if row.get("reason") not in (None, "not_reached_from_second_observation")
            }
            directions = replay.get("directions", {}) if legal else {}
            attempted = sum(int(value.get("attempted_transitions", 0)) for value in directions.values())
            accepted = sum(int(value.get("accepted_transitions", 0)) for value in directions.values())
            row = {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "window_id": window_id,
                "tracklet_exact_equal": True,
                "saved_tracklet_sha256": saved_hash,
                "replayed_tracklet_sha256": replay_hash,
                "anchor_observation_ordinal": replay.get("anchor_observation_ordinal"),
                "anchor_position_in_tracking_sequence": replay.get("anchor_position"),
                "anchor_corner_count": replay.get("anchor_corner_count"),
                "trajectory_position_count": len(trajectory),
                "trajectory_status_counts": dict(statuses),
                "explicit_none_positions": none_positions,
                "failure_reason_by_position": failure_reason_by_position,
                "attempted_transitions": attempted,
                "accepted_transitions": accepted,
                "rejected_transitions": attempted - accepted,
                "four_observation_sample_matches": True,
                "GT_read": False,
            }
            tracking_results.append(row)
            replay_rows.append(row)
            total_non_source_replays += 1
            total_track_positions += len(trajectory)
            total_attempted_transitions += attempted
            total_accepted_transitions += accepted
            total_explicit_none_positions += len(none_positions)

        if barrier_record["cohort"] == "old8":
            old_source_candidate_total += len(candidates)
        child_controls = new_records.get(case_id, {})
        if set(child_controls) != {"order_swap", "joint_swap"}:
            raise ValueError(f"fixed-layout barrier lacks both controls for {case_id}")
        for control, new_record in child_controls.items():
            if new_record["source_case_sha256"] != source_case_hash or new_record["source_candidate_barrier_sha256"] != old_barrier_sha:
                raise ValueError(f"fixed layout lost its source/source-barrier binding for {case_id}/{control}")
            if new_record["physical_frame_ids"] != physical_obs:
                raise ValueError(f"fixed-layout frames differ from source observations for {case_id}/{control}")
            if new_record["source_RGB_sha256_by_frame"] != rgb_binding:
                raise ValueError(f"fixed-layout RGB hashes differ from fresh canonical decode for {case_id}/{control}")
            if (new_record.get("request_sha256") != request_sha
                    or new_record.get("source_video_sha256") != request.get("source_video_sha256")
                    or new_record.get("canonical_video_sha256") != request.get("canonical_video_sha256")):
                raise ValueError(f"fixed-layout request/video binding differs from canonical REQUEST for {case_id}/{control}")
            expected_dimensions = {str(fid): list(source_wh[i]) for i, fid in enumerate(physical_obs)}
            if new_record.get("source_wh_by_frame") != expected_dimensions:
                raise ValueError(f"fixed-layout source dimensions differ from canonical decode for {case_id}/{control}")

        per_case.append({
            "case_id": case_id,
            "dataset": roster_row["dataset"],
            "cohort": roster_row["cohort"],
            "source_case_sha256": source_case_hash,
            "source_frame_count": int(len(source_frame_ids)),
            "observation_positions": observation_positions.tolist(),
            "physical_observation_frame_ids": physical_obs,
            "tracking_physical_frame_ids": tracking_ids,
            "fresh_source_RGB_sha256_by_frame": rgb_binding,
            "fresh_gray_sha256_by_tracking_frame": gray_binding,
            "source_wh_by_tracking_frame": {str(fid): list(source_wh[i]) for i, fid in enumerate(tracking_ids)},
            "source_wh_interpolation_sha256": sha_bytes(np.ascontiguousarray(source_wh_all).tobytes()),
            "candidate_count": len(candidates),
            "candidate_window_ids": actual_windows,
            "seed_rule": barrier_record["seed_rule"],
            "retrieved_union_window_ids": union,
            "non_source_track_replays": tracking_results,
            "new_fixed_layout_controls": sorted(child_controls),
            "GT_read": False,
        })

    # Re-hash every original 96 input and every new 32 input, and validate all
    # tensor/render frames against their corresponding sealed metadata.
    job_frame_bindings: list[dict[str, Any]] = []
    old_case_candidates = {row["case_id"]: {int(c["candidate_id"]): c for c in row["candidates"]}
                           for row in old_barrier["records"]}
    for job in old_job_lock["records"]:
        case_id = job["case_id"]
        barrier_record = old_records[case_id]
        render_relative = f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{job['kind']}_render.json"
        job_frame_bindings.append(_verify_one_job_tensor(
            job, barrier_record, root_path(render_relative),
            barrier_record["source_RGB_sha256_by_frame"], old_case_candidates[case_id],
            expected_times=barrier_record["actual_times_seconds"],
        ))
    new_case_candidates = old_case_candidates
    new_records_list = new_barrier["records"]
    new_by_control = {(row["case_id"], row["control"]): row for row in new_records_list}
    for job in new_job_lock["records"]:
        case_id, control = job["case_id"], job["kind"]
        if control not in ("order_swap", "joint_swap"):
            raise ValueError("fixed-layout job lock includes an unexpected condition")
        new_record = new_by_control[(case_id, control)]
        render_relative = f"artifacts/stvg_motion_guided_instance_opd_v1/display_order_layout001/private_inputs/{case_id}/{control}_render.json"
        job_frame_bindings.append(_verify_one_job_tensor(
            job, new_record, root_path(render_relative), old_records[case_id]["source_RGB_sha256_by_frame"],
            new_case_candidates[case_id],
            expected_times=old_records[case_id]["actual_times_seconds"],
            base_render_path=BASE / "private_inputs" / case_id / "base_render.json",
        ))
        old_job = old_jobs[f"{case_id}_{control}"]
        if {k: v for k, v in old_job.items() if k != "videos"} != {k: v for k, v in job.items() if k != "videos"}:
            raise ValueError(f"fixed-layout job changed non-input field: {job['job_id']}")
        if (old_job["videos"][0]["fps"] != job["videos"][0]["fps"]
                or old_job["videos"][0]["actual_times_seconds"] != job["videos"][0]["actual_times_seconds"]):
            raise ValueError(f"fixed-layout input changed FPS/timestamps: {job['job_id']}")

    if len(job_frame_bindings) != 128:
        raise AssertionError("expected readback for all 96 original plus 32 fixed-layout jobs")

    old_pool_counts = {
        "old8_cases": sum(row["cohort"] == "old8" for row in per_case),
        "old8_source_inclusive_candidate_total": old_source_candidate_total,
        "old8_non_source_union_total": old_union_window_total - 8,
        "additional8_cases": sum(row["cohort"] == "additional8" for row in per_case),
        "additional8_source_inclusive_coarse_candidate_total": new_coarse_candidate_total,
        "additional8_non_source_coarse_total": new_coarse_candidate_total - 8,
        "combined_candidate_total": sum(row["candidate_count"] for row in per_case),
        "non_source_candidates_replayed": total_non_source_replays,
    }
    if old_pool_counts["old8_source_inclusive_candidate_total"] != 43:
        raise ValueError(f"old8 source-included candidate count changed: {old_pool_counts}")
    if old_pool_counts["additional8_source_inclusive_coarse_candidate_total"] != 40:
        raise ValueError(f"additional8 coarse candidate count changed: {old_pool_counts}")
    if old_pool_counts["non_source_candidates_replayed"] != 67:
        raise ValueError(f"non-Source replay count changed: {old_pool_counts}")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "CPU source/media/hash/input readback and exact replay through the already-pinned LK helper; no second tracking algorithm",
        "case_count": 16,
        "freshly_decoded_keyframe_count": 64,
        "freshly_decoded_tracking_gray_count": sum(len(row["tracking_physical_frame_ids"]) for row in per_case),
        "old_input_job_count_sha_tensor_framebind": 96,
        "new_fixed_layout_job_count_sha_tensor_framebind": 32,
        "total_input_job_frame_bindings_verified": len(job_frame_bindings),
        "all_old_manifest_files_verified": len(old_manifest),
        "all_new_manifest_files_verified": len(new_manifest),
        "candidate_seed_pool_readback": old_pool_counts,
        "tracklet_readback": {
            "method": "same pinned track_second_observation helper, same RGB-to-gray decode, source dimensions, anchor, and width/height interpolation",
            "non_source_candidates_replayed": total_non_source_replays,
            "saved_trajectory_records_compared_exactly": total_non_source_replays,
            "trajectory_positions_compared": total_track_positions,
            "explicit_none_positions": total_explicit_none_positions,
            "attempted_transitions": total_attempted_transitions,
            "accepted_transitions": total_accepted_transitions,
            "rejected_transitions": total_attempted_transitions - total_accepted_transitions,
            "per_candidate_results": replay_rows,
        },
        "source_pins": {
            "runtime_code_and_protocol_pins_sha256": pin_checks,
            "source_case_roster_bound": True,
            "source_case_receipt_sha_verified": True,
            "lk_helper_contract_sha256": helper_contracts["helper_sha256"],
            "old_candidate_barrier_sha256": old_barrier_sha,
            "old_job_lock_sha256": old_jobs_sha,
            "new_candidate_barrier_source_sha256": new_barrier.get("source_candidate_barrier_sha256"),
            "new_candidate_input_barrier_sha256": new_barrier_sha,
            "new_job_lock_sha256": sha_file(new_jobs_path),
            "new_fixed_layout_capture_sha256": sha_file(NEW / "CAPTURE.json"),
            "old_grid_barrier_sha256": sha_file(OLD_GRID / "GRID_INPUT_BARRIER.json"),
            "old_grid_file_count_verified": len(grid_manifest),
            "old_retriever_candidate_barrier_sha256": sha_file(OLD_REFERENTS / "CANDIDATE_INPUT_BARRIER.json"),
            "old_retriever_file_count_verified": len(retrieval_manifest),
            "runtime_sha256": sha_file(runtime_path),
            "design_lock_sha256": sha_file(design_path),
            "protocol_sha256": sha_file(protocol_path),
        },
        "per_case": per_case,
        "elapsed_cpu_seconds": time.perf_counter() - started,
        "GT_read": False,
        "model_calls": 0,
        "GPU_used": False,
        "query_text_read_or_emitted": False,
        "teacher_output_read": False,
        "source_model_state_values_read": False,
    }
    return report


def main() -> None:
    report = run_readback()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise FileExistsError(REPORT_PATH)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "old_input_job_count": report["old_input_job_count_sha_tensor_framebind"],
        "new_input_job_count": report["new_fixed_layout_job_count_sha_tensor_framebind"],
        "non_source_candidates_replayed": report["tracklet_readback"]["non_source_candidates_replayed"],
        "report_sha256": sha_file(REPORT_PATH),
        "GT_read": False,
        "model_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
