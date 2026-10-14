"""Prepare the fixed old-eight adjacent-motion candidate/control inputs.

This is the CPU-only P0/P1 input builder for the locked scope.  It reads only
the eight accepted no-GT source cases, the already sealed 43-seed union, their
canonical source videos, and the existing no-GT preparation locks.  It does not
read annotations, score labels, model weights, teacher outputs, or GT.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stvg_sparse_r1_v2 import inference_guard
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
from scripts.stvg_adjacent_affine_tracklet_v1 import CONFIG as TRACK_CONFIG
from scripts.stvg_adjacent_affine_tracklet_v1 import SCHEMA as TRACK_SCHEMA
from scripts.stvg_adjacent_affine_tracklet_v1 import track_adjacent
from scripts.prepare_stvg_motion_guided_v1 import legal, prompt as build_prompt, render as render_badge_basis
from scripts.prepare_stvg_motion_order_layout001 import _render_order_layout


BASE = ROOT / "artifacts/stvg_adjacent_motion_opd_v1"
SOURCE_BASE = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1"
SOURCE_CASES = SOURCE_BASE / "source_cases"
PREVIOUS_BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
PREVIOUS_GRID = ROOT / "artifacts/stvg_box_scale_global_evidence_v1"
CANONICAL_INPUTS = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
DESIGN_PATH = BASE / "DESIGN_LOCK.json"
PROTOCOL_PATH = ROOT / "protocols/stvg_adjacent_motion_opd_v1.md"
OLD_CANDIDATE_BARRIER = PREVIOUS_BASE / "CANDIDATE_INPUT_BARRIER.json"
OLD_TEACHER_JOB_LOCK = PREVIOUS_BASE / "TEACHER_JOB_LOCK.json"
OLD_GRID_BARRIER = PREVIOUS_GRID / "GRID_INPUT_BARRIER.json"
KINDS = ("base", "id_swap", "color_swap", "order_swap", "joint_swap", "wrong_query")
IMAGE_SIDE = 448
SCHEMA = "stvg-adjacent-motion-opd-v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _root_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("output/input path escaped repository root")
    return resolved.relative_to(ROOT.resolve()).as_posix()


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        if str(value.device) != "cpu" or bool(getattr(value, "requires_grad", False)):
            raise ValueError(f"{name} must be detached and CPU resident")
        value = value.detach().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _cxcywh_to_xyxy(boxes: Any) -> np.ndarray:
    values = _array(boxes, "source zero boxes")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("source zero boxes must have shape [frames,4] normalized cxcywh")
    cx, cy, width, height = values.T
    return np.stack((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0), axis=-1)


def _legal_anchor(box: Sequence[float] | None) -> bool:
    return box is not None and legal(box)


def _safe_request(case_id: str, case: Mapping[str, Any], barrier_record: Mapping[str, Any]):
    request_path = CANONICAL_INPUTS / case_id / "REQUEST.json"
    request = _read_json(request_path)
    video_path = Path(str(request["video_path"])).resolve()
    expected_dir = (CANONICAL_INPUTS / case_id).resolve()
    if video_path.parent != expected_dir or not video_path.is_file():
        raise ValueError(f"canonical video escaped its fixed case directory for {case_id}")
    canonical_sha = _sha_file(video_path)
    if canonical_sha != request.get("canonical_video_sha256"):
        raise ValueError(f"canonical video bytes changed for {case_id}")
    if canonical_sha != barrier_record.get("canonical_video_sha256"):
        raise ValueError(f"canonical video is not the old source-bound video for {case_id}")
    source_video_sha = request.get("source_video_sha256")
    if source_video_sha != case["input"]["video_sha256"]:
        raise ValueError(f"canonical video does not match the source case for {case_id}")
    frame_map_rows = request.get("frame_map")
    if not isinstance(frame_map_rows, list) or not frame_map_rows:
        raise ValueError(f"canonical request has no physical frame map for {case_id}")
    physical_to_local: dict[int, int] = {}
    local_to_physical: dict[int, int] = {}
    for row in frame_map_rows:
        physical = int(row["physical_frame_id"])
        local = int(row["local_frame_index"])
        if physical in physical_to_local or local in local_to_physical or local < 0:
            raise ValueError(f"duplicate/invalid canonical frame mapping for {case_id}")
        physical_to_local[physical] = local
        local_to_physical[local] = physical
    return request, request_path, video_path, canonical_sha, source_video_sha, physical_to_local, local_to_physical


def _source_rgb(case: Mapping[str, Any], physical_id: int) -> Mapping[str, Any] | None:
    table = case.get("source_RGB", {})
    return table.get(physical_id, table.get(str(physical_id)))


def _resize_gray(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= 640:
        return gray
    scale = 640.0 / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if max(resized.shape) > 640:
        raise ValueError("INTER_AREA grayscale resize exceeded the 640-pixel long edge")
    return resized


def _decode_dense_range(
    case_id: str,
    case: Mapping[str, Any],
    barrier_record: Mapping[str, Any],
    positions: list[int],
    physical_ids: list[int],
) -> dict[str, Any]:
    request, request_path, video_path, canonical_sha, source_video_sha, physical_to_local, local_to_physical = _safe_request(
        case_id, case, barrier_record
    )
    dense_ids = list(range(physical_ids[0], physical_ids[-1] + 1))
    if len(physical_ids) != 4 or len(set(physical_ids)) != 4:
        raise ValueError(f"exactly four unique original observations are required for {case_id}")
    if any(b <= a for a, b in zip(physical_ids, physical_ids[1:])):
        raise ValueError(f"source observations must remain in physical time order for {case_id}")
    if any(fid not in physical_to_local for fid in dense_ids):
        raise ValueError(f"physical step-one video coverage is missing for {case_id}")
    wanted_local = [physical_to_local[fid] for fid in dense_ids]
    if wanted_local != list(range(wanted_local[0], wanted_local[0] + len(wanted_local))):
        raise ValueError(f"physical +1 frames do not map to adjacent canonical decoded frames for {case_id}")
    if any(local_to_physical.get(local) != physical for physical, local in zip(dense_ids, wanted_local)):
        raise ValueError(f"canonical local/physical frame map is not bijective for {case_id}")

    frame_to_local = dict(zip(dense_ids, wanted_local))
    local_to_target = {local: physical for physical, local in frame_to_local.items()}
    rgb_needed = set(physical_ids)
    source_hashes_expected = barrier_record["source_RGB_sha256_by_frame"]
    gray_frames: list[np.ndarray] = []
    gray_hashes: dict[str, str] = {}
    rgb_images: dict[int, Image.Image] = {}
    rgb_hashes: dict[str, str] = {}
    raw_wh_by_frame: dict[str, list[int]] = {}
    resized_wh_by_frame: dict[str, list[int]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open the locked canonical video for {case_id}")
    first_local, last_local = wanted_local[0], wanted_local[-1]
    current_local = 0
    try:
        while current_local <= last_local:
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"canonical decode ended before the locked dense source interval for {case_id}")
            physical = local_to_target.get(current_local)
            if physical is not None:
                if physical != dense_ids[len(gray_frames)]:
                    raise ValueError(f"decoded physical frames are not in exact step-one order for {case_id}")
                raw_height, raw_width = map(int, bgr.shape[:2])
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray = _resize_gray(gray)
                if gray_frames and gray.shape != gray_frames[0].shape:
                    raise ValueError(f"resized grayscale frame shape changed inside {case_id}")
                gray_frames.append(gray)
                gray_hashes[str(physical)] = _sha_bytes(gray.tobytes())
                raw_wh_by_frame[str(physical)] = [raw_width, raw_height]
                resized_wh_by_frame[str(physical)] = [int(gray.shape[1]), int(gray.shape[0])]
                if physical in rgb_needed:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_digest = _sha_bytes(rgb.tobytes())
                    expected_source = source_hashes_expected.get(str(physical))
                    if expected_source != rgb_digest:
                        raise ValueError(f"decoded keyframe RGB differs from old source barrier for {case_id}/{physical}")
                    source_receipt = _source_rgb(case, physical)
                    if source_receipt is None or source_receipt.get("RGB_sha256") != rgb_digest:
                        raise ValueError(f"decoded keyframe RGB differs from original source capture for {case_id}/{physical}")
                    rgb_hashes[str(physical)] = rgb_digest
                    rgb_images[physical] = Image.fromarray(rgb)
            current_local += 1
    finally:
        cap.release()
    if len(gray_frames) != len(dense_ids):
        raise ValueError(f"did not process every physical +1 grayscale frame for {case_id}")
    if set(rgb_images) != rgb_needed or len(rgb_images) != 4:
        raise ValueError(f"must retain exactly four original RGB observations for {case_id}")
    if list(map(int, rgb_images)) != sorted(rgb_needed):
        rgb_images = {fid: rgb_images[fid] for fid in sorted(rgb_needed)}
    return {
        "request_sha256": _sha_file(request_path),
        "canonical_video_sha256": canonical_sha,
        "source_video_sha256": source_video_sha,
        "dense_ids": dense_ids,
        "gray_frames": gray_frames,
        "gray_hashes": gray_hashes,
        "gray_shape_hw": [int(gray_frames[0].shape[0]), int(gray_frames[0].shape[1])],
        "rgb_images": rgb_images,
        "rgb_hashes": rgb_hashes,
        "raw_wh_by_frame": raw_wh_by_frame,
        "resized_wh_by_frame": resized_wh_by_frame,
        "video_fps": float(request["local_fps"]),
    }


def _track_one(
    gray_frames: list[np.ndarray],
    physical_ids: list[int],
    anchor_physical_id: int,
    anchor_box: Sequence[float] | None,
) -> tuple[dict[str, Any], list[list[float] | None], list[float]]:
    anchor_index = physical_ids.index(anchor_physical_id)
    if _legal_anchor(anchor_box):
        tracklet = track_adjacent(gray_frames, physical_ids, anchor_index, anchor_box)
    else:
        # Preserve this seed and its invalid-anchor reason; do not clip or replace it.
        tracklet = {
            "schema": TRACK_SCHEMA,
            "status": "not_run_invalid_anchor",
            "anchor_index": anchor_index,
            "anchor_physical_frame_id": int(anchor_physical_id),
            "frame_shape_hw": [int(gray_frames[0].shape[0]), int(gray_frames[0].shape[1])],
            "physical_frame_ids": [int(value) for value in physical_ids],
            "trajectory": [
                {
                    "frame_index": index,
                    "physical_frame_id": int(fid),
                    "status": "missing",
                    "box_xyxy_norm": None,
                    "reason": "invalid_seed_anchor_no_clipping",
                }
                for index, fid in enumerate(physical_ids)
            ],
            "transitions": [],
            "cost": {
                "directions": {
                    name: {
                        "attempted_transitions": 0,
                        "accepted_transitions": 0,
                        "rejected_transitions": 0,
                        "not_attempted_after_loss": 0,
                        "detected_corner_count": 0,
                        "fb_valid_corner_count": 0,
                        "ransac_inlier_count": 0,
                    }
                    for name in ("backward", "forward")
                },
                "total": {
                    "attempted_transitions": 0,
                    "accepted_transitions": 0,
                    "rejected_transitions": 0,
                    "not_attempted_after_loss": 0,
                    "detected_corner_count": 0,
                    "fb_valid_corner_count": 0,
                    "ransac_inlier_count": 0,
                },
            },
            "config": TRACK_CONFIG.copy(),
            "reason": "invalid_seed_anchor_no_clipping",
        }
    rho_by_frame = {int(fid): 0.0 for fid in physical_ids}
    if tracklet.get("status") != "not_run_invalid_anchor":
        rho_by_frame[int(anchor_physical_id)] = 1.0
        for transition in tracklet["transitions"]:
            target = int(transition["to_physical_frame_id"])
            if transition.get("status") == "accepted":
                rho_by_frame[target] = float(transition["rho"])
            elif transition.get("attempted"):
                rho_by_frame[target] = 0.0
        for trajectory in tracklet["trajectory"]:
            if trajectory is not None:
                fid = int(trajectory["physical_frame_id"])
                trajectory["rho"] = float(rho_by_frame[fid])
    boxes = []
    rhos = []
    for fid in physical_ids:
        trajectory = tracklet["trajectory"][physical_ids.index(fid)]
        if trajectory is None:
            boxes.append(None)
        else:
            value = trajectory.get("box_xyxy_norm")
            boxes.append(None if value is None else [float(x) for x in value])
        rhos.append(float(rho_by_frame[int(fid)]))
    return tracklet, boxes, rhos


def _seed_layout(case_id: str, candidate_ids: list[int], n: int):
    seed = int(hashlib.sha256(("presentation|" + case_id).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    visible_values = rng.permutation(np.arange(1, n + 1)).tolist()
    palette_values = rng.permutation(5)[:n].tolist()
    order = rng.permutation(candidate_ids).tolist()
    return seed, dict(zip(candidate_ids, visible_values)), dict(zip(candidate_ids, palette_values)), order


def _control_maps(kind: str, base_ids: Mapping[int, int], base_colors: Mapping[int, int], base_order: list[int]):
    ids = dict(base_ids)
    colors = dict(base_colors)
    order = list(base_order)
    if kind in ("id_swap", "joint_swap"):
        values = [base_ids[cid] for cid in sorted(base_ids)]
        rotated = values[1:] + values[:1]
        ids = dict(zip(sorted(base_ids), rotated))
    if kind in ("color_swap", "joint_swap"):
        colors = {candidate_id: (color + 1) % 5 for candidate_id, color in base_colors.items()}
    if kind in ("order_swap", "joint_swap"):
        order = order[::-1]
    return ids, colors, order


def _file_row(path: Path) -> dict[str, Any]:
    return {"path": _root_relative(path), "sha256": _sha_file(path), "bytes": int(path.stat().st_size)}


def prepare() -> dict[str, Any]:
    sys.addaudithook(inference_guard)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    for path in (BASE / "CANDIDATE_INPUT_BARRIER.json", BASE / "TEACHER_JOB_LOCK.json", BASE / "CAPTURE.json"):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite sealed preparation artifact: {path}")

    roster = _read_json(ROSTER_PATH)
    design = _read_json(DESIGN_PATH)
    if roster.get("status") != "locked" or roster.get("GT_read") is not False or roster.get("count") != 8:
        raise ValueError("new-scope roster lock is not the expected fixed no-GT old8")
    if design.get("status") != "locked" or design.get("GT_read") is not False or design.get("count") != 8:
        raise ValueError("new-scope design lock is not the expected fixed no-GT old8")
    if design.get("teacher_calls_planned") != 48:
        raise ValueError("design lock does not specify exactly 48 teacher calls")
    if _sha_file(PROTOCOL_PATH) != design.get("protocol_sha256"):
        raise ValueError("locked protocol hash changed")
    if _sha_file(ROOT / "methods/CURRENT_METHOD.json") != design.get("CURRENT_sha256"):
        raise ValueError("current-method source binding changed")
    roster_records = roster.get("records")
    if not isinstance(roster_records, list) or len(roster_records) != 8:
        raise ValueError("roster does not contain exactly eight records")
    case_ids = [str(row["case_id"]) for row in roster_records]
    if len(set(case_ids)) != 8 or any(row.get("cohort") != "old8" for row in roster_records):
        raise ValueError("roster is not the exact unique old8 cohort")
    if any(row.get("P0") is not True for row in roster_records):
        raise ValueError("all locked old8 cases must be P0 eligible")

    old_barrier = _read_json(OLD_CANDIDATE_BARRIER)
    old_job_lock = _read_json(OLD_TEACHER_JOB_LOCK)
    grid_barrier = _read_json(OLD_GRID_BARRIER)
    if old_barrier.get("status") != "complete" or old_barrier.get("count") != 16 or old_barrier.get("GT_read") is not False:
        raise ValueError("previous fixed-seed candidate source is not complete and no-GT")
    if old_job_lock.get("status") != "locked" or old_job_lock.get("count") != 96 or old_job_lock.get("GT_read") is not False:
        raise ValueError("previous no-GT teacher lock is not the sealed 96-call source")
    if grid_barrier.get("status") != "complete" or grid_barrier.get("count") != 8 or grid_barrier.get("GT_read") is not False:
        raise ValueError("previous scale-grid source is not the expected no-GT eight-case barrier")

    old_records_all = old_barrier.get("records", [])
    grid_records_all = grid_barrier.get("records", [])
    old_records = {str(row["case_id"]): row for row in old_records_all if row.get("cohort") == "old8"}
    grid_records = {str(row["case_id"]): row for row in grid_records_all}
    if set(old_records) != set(case_ids) or set(grid_records) != set(case_ids):
        raise ValueError("the previous source/grid barriers do not match the locked old8 roster")
    old_jobs = old_job_lock.get("records", [])
    old_jobs_by_id = {str(job.get("job_id")): job for job in old_jobs}
    if len(old_jobs_by_id) != 96:
        raise ValueError("old teacher jobs are not uniquely identified")

    source_case_by_id: dict[str, dict[str, Any]] = {}
    for rec in roster_records:
        case_id = str(rec["case_id"])
        source_path = (ROOT / str(rec["source_case_path"])).resolve()
        receipt_path = source_path.with_suffix(".json")
        if source_path.parent != SOURCE_CASES.resolve() or not source_path.is_file() or not receipt_path.is_file():
            raise ValueError(f"locked source case is unavailable or outside the accepted source directory: {case_id}")
        if _sha_file(source_path) != rec["source_case_sha256"]:
            raise ValueError(f"source-case SHA changed for {case_id}")
        if _sha_file(source_path) != old_records[case_id].get("source_case_sha256"):
            raise ValueError(f"source-case bytes differ from previous fixed seed record for {case_id}")
        if _sha_file(source_path) != grid_records[case_id].get("source_case_sha256"):
            raise ValueError(f"source-case bytes differ from previous grid seed for {case_id}")
        case = load_source_case(source_path)
        if case.get("GT_read") is not False or case.get("original_source_only") is not True or case.get("DINO_read_or_call") is not False:
            raise ValueError(f"source case is not marked as accepted source-only and no-GT for {case_id}")
        if case.get("_source_case_sha256") != rec["source_case_sha256"]:
            raise ValueError(f"source loader did not return the locked source SHA for {case_id}")
        if case.get("source_model_state_sha256") != old_records[case_id].get("source_model_state_sha256"):
            raise ValueError(f"source model state differs from previous candidate record for {case_id}")
        source_case_by_id[case_id] = case

    if sum(len(old_records[cid].get("retrieved_union_window_ids", [])) for cid in case_ids) != 43:
        raise ValueError("old8 retrieval union no longer contains exactly 43 locked seeds")
    if sum(len(old_records[cid].get("candidates", [])) for cid in case_ids) != 43:
        raise ValueError("old8 candidate barrier no longer contains exactly 43 seeds")

    old_candidate_barrier_sha = _sha_file(OLD_CANDIDATE_BARRIER)
    old_teacher_job_lock_sha = _sha_file(OLD_TEACHER_JOB_LOCK)
    old_grid_barrier_sha = _sha_file(OLD_GRID_BARRIER)
    roster_sha = _sha_file(ROSTER_PATH)
    design_sha = _sha_file(DESIGN_PATH)
    protocol_sha = _sha_file(PROTOCOL_PATH)
    current_method_sha = _sha_file(ROOT / "methods/CURRENT_METHOD.json")
    helper_sha = _sha_file(ROOT / "scripts/stvg_adjacent_affine_tracklet_v1.py")
    renderer_sha = _sha_file(ROOT / "scripts/prepare_stvg_scene_object_id_v1.py")
    fixed_badge_helper_sha = _sha_file(ROOT / "scripts/prepare_stvg_motion_order_layout001.py")
    basis_helper_sha = _sha_file(ROOT / "scripts/prepare_stvg_motion_guided_v1.py")
    input_records: list[dict[str, Any]] = []
    job_records: list[dict[str, Any]] = []
    file_manifest: list[dict[str, Any]] = []
    case_request_sha: dict[str, str] = {}
    case_video_sha: dict[str, str] = {}
    count_by_case: dict[str, int] = {}

    for roster_index, rec in enumerate(roster_records):
        case_id = str(rec["case_id"])
        case = source_case_by_id[case_id]
        old_record = old_records[case_id]
        grid_record = grid_records[case_id]
        positions = [int(value) for value in old_record["positions"]]
        physical_ids = [int(value) for value in old_record["physical_frame_ids"]]
        if len(positions) != 4 or len(physical_ids) != 4:
            raise ValueError(f"old candidate barrier does not preserve four source observations for {case_id}")
        source_frame_ids = [int(value) for value in np.asarray(case["frame_ids"]).tolist()]
        if any(pos < 0 or pos >= len(source_frame_ids) for pos in positions):
            raise ValueError(f"source positions are outside the original frame grid for {case_id}")
        if [source_frame_ids[pos] for pos in positions] != physical_ids:
            raise ValueError(f"old positions and physical IDs do not match the source case for {case_id}")
        if positions != [int(value) for value in case["observation_positions"][:4]]:
            raise ValueError(f"old fixed observation positions changed for {case_id}")
        if grid_record.get("observation_position") != positions[1] or grid_record.get("physical_frame_id") != physical_ids[1]:
            raise ValueError(f"scale-grid anchor is not the old second observation for {case_id}")
        if grid_record.get("source_RGB_sha256") != old_record["source_RGB_sha256_by_frame"].get(str(physical_ids[1])):
            raise ValueError(f"grid second-observation RGB is not tied to the old source frame for {case_id}")

        box_cxcywh = _cxcywh_to_xyxy(case["zero"]["boxes"])
        if len(box_cxcywh) != len(source_frame_ids):
            raise ValueError(f"source boxes and frame IDs have different lengths for {case_id}")
        display_source_boxes = [[float(x) for x in box_cxcywh[pos]] for pos in positions]
        if not all(legal(box) for box in display_source_boxes):
            raise ValueError(f"original Source display boxes are not legal for {case_id}")
        source_anchor = [float(x) for x in box_cxcywh[positions[1]]]
        source_wh = [float(box_cxcywh[positions[1], 2] - box_cxcywh[positions[1], 0]),
                     float(box_cxcywh[positions[1], 3] - box_cxcywh[positions[1], 1])]
        # Width and height are derived from the exact source anchor, not from a
        # later Source track, and are used only to initialize the old center seeds.
        source_xyxy = box_cxcywh
        original_cxcywh = _array(case["zero"]["boxes"], "source zero boxes")
        source_width_height = [float(original_cxcywh[positions[1], 2]), float(original_cxcywh[positions[1], 3])]
        if source_width_height[0] <= 0 or source_width_height[1] <= 0:
            raise ValueError(f"Source anchor has a non-positive size for {case_id}")

        decoded = _decode_dense_range(case_id, case, old_record, positions, physical_ids)
        case_request_sha[case_id] = decoded["request_sha256"]
        case_video_sha[case_id] = decoded["canonical_video_sha256"]
        dense_ids = decoded["dense_ids"]
        gray_frames = decoded["gray_frames"]
        anchor_fid = physical_ids[1]
        anchor_index = dense_ids.index(anchor_fid)
        source_tracklet, source_geometry, source_rho = _track_one(
            gray_frames, dense_ids, anchor_fid, source_anchor
        )
        source_candidate = {
            "candidate_id": 1,
            "window_id": 0,
            "is_Source": True,
            "seed_order": 1,
            "positions": positions,
            "physical_frame_ids": physical_ids,
            "boxes_xyxy": display_source_boxes,
            "geometry_boxes_xyxy": source_geometry,
            "rho_by_observation": source_rho,
            "geometry_valid_observation_count": sum(value is not None for value in source_geometry),
            "valid_observation_count": sum(value is not None for value in source_geometry),
            "anchor_box_xyxy": source_anchor,
            "geometry_sha256": _canonical_sha({"geometry_boxes_xyxy": source_geometry, "rho": source_rho}),
            "tracklet": source_tracklet,
        }
        candidates = [source_candidate]

        seed_window_ids = [int(value) for value in old_record["retrieved_union_window_ids"]]
        old_candidates = old_record.get("candidates", [])
        if [int(row["window_id"]) for row in old_candidates] != seed_window_ids:
            raise ValueError(f"old seed order differs from the stored candidate catalog for {case_id}")
        if seed_window_ids[0] != 0 or len(seed_window_ids) != len(set(seed_window_ids)):
            raise ValueError(f"old seed union must start with Source and have unique windows for {case_id}")
        window_by_id = {int(row["window_id"]): row for row in grid_record["grid"]["windows"]}
        for ordinal, window_id in enumerate(seed_window_ids[1:], start=2):
            if window_id not in window_by_id:
                raise ValueError(f"old seed window is absent from the bound coarse grid for {case_id}/{window_id}")
            window = window_by_id[window_id]
            anchor = [
                float(window["cx"] - source_width_height[0] / 2.0),
                float(window["cy"] - source_width_height[1] / 2.0),
                float(window["cx"] + source_width_height[0] / 2.0),
                float(window["cy"] + source_width_height[1] / 2.0),
            ]
            tracklet, geometry_boxes, rho = _track_one(gray_frames, dense_ids, anchor_fid, anchor)
            candidates.append(
                {
                    "candidate_id": ordinal,
                    "window_id": window_id,
                    "is_Source": False,
                    "seed_order": ordinal,
                    "positions": positions,
                    "physical_frame_ids": physical_ids,
                    "boxes_xyxy": geometry_boxes,
                    "geometry_boxes_xyxy": geometry_boxes,
                    "rho_by_observation": rho,
                    "geometry_valid_observation_count": sum(value is not None for value in geometry_boxes),
                    "valid_observation_count": sum(value is not None for value in geometry_boxes),
                    "anchor_box_xyxy": anchor,
                    "geometry_sha256": _canonical_sha({"geometry_boxes_xyxy": geometry_boxes, "rho": rho}),
                    "tracklet": tracklet,
                }
            )
        if len(candidates) != len(seed_window_ids):
            raise AssertionError(f"candidate seed count changed for {case_id}")
        count_by_case[case_id] = len(candidates)

        request = _read_json(CANONICAL_INPUTS / case_id / "REQUEST.json")
        fps = float(request["local_fps"])
        video_bounds = [float(value) for value in case["video_bounds"]]
        times = [(float(fid) - video_bounds[0]) / fps for fid in physical_ids]
        if not all(math.isfinite(value) for value in times) or not all(b > a for a, b in zip(times, times[1:])):
            raise ValueError(f"physical observation times are not strictly increasing for {case_id}")
        mean_gap_fps = 3.0 / (times[-1] - times[0])
        displayed = candidates[:5]
        candidate_ids = [int(row["candidate_id"]) for row in displayed]
        if len(displayed) > 5 or candidate_ids != list(range(1, len(displayed) + 1)):
            raise ValueError(f"visible candidates do not preserve the original first-five stable seed order for {case_id}")
        presentation_seed, base_ids, base_colors, base_order = _seed_layout(case_id, candidate_ids, len(candidate_ids))
        base_tensor_provisional, base_records = render_badge_basis(
            [decoded["rgb_images"][fid] for fid in physical_ids], displayed, base_ids, base_colors, base_order, times
        )
        if base_tensor_provisional.shape != (4, 3, IMAGE_SIDE, IMAGE_SIDE) or base_tensor_provisional.dtype != np.uint8:
            raise ValueError(f"base presentation violates the fixed four-frame tensor ABI for {case_id}")
        base_render = {"records": base_records}
        case_dir = BASE / "private_inputs" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        base_badges_by_slot = {
            str(int(row["slot"])): {
                str(int(overlay["candidate_id"])): list(map(int, overlay["badge_xyxy_pixels"]))
                for overlay in row["overlays"]
            }
            for row in base_records
        }
        job_ids_for_case: list[str] = []
        prompt_sha_by_kind: dict[str, str] = {}
        for kind in KINDS:
            ids, colors, order = _control_maps(kind, base_ids, base_colors, base_order)
            if set(ids) != set(candidate_ids) or set(colors) != set(candidate_ids) or sorted(order) != sorted(candidate_ids):
                raise ValueError(f"control maps are not bijective over the fixed candidate set for {case_id}/{kind}")
            provisional, control_records = render_badge_basis(
                [decoded["rgb_images"][fid] for fid in physical_ids], displayed, ids, colors, order, times
            )
            if provisional.shape != (4, 3, IMAGE_SIDE, IMAGE_SIDE) or provisional.dtype != np.uint8:
                raise ValueError(f"control renderer violates tensor ABI for {case_id}/{kind}")
            control_render = {
                "records": control_records,
                "id_map": {str(key): int(value) for key, value in ids.items()},
                "color_map": {str(key): int(value) for key, value in colors.items()},
                "draw_order": [int(value) for value in order],
            }
            rendered_tensor, rendered_records = _render_order_layout(
                [decoded["rgb_images"][fid] for fid in physical_ids],
                times,
                base_render,
                control_render,
                case_id,
                kind,
            )
            if rendered_tensor.shape != (4, 3, IMAGE_SIDE, IMAGE_SIDE) or rendered_tensor.dtype != np.uint8:
                raise ValueError(f"fixed-badge renderer violated the teacher tensor ABI for {case_id}/{kind}")
            for row in rendered_records:
                slot = str(int(row["slot"]))
                expected_by_candidate = base_badges_by_slot[slot]
                actual_by_candidate = {
                    str(int(overlay["candidate_id"])): list(map(int, overlay["badge_xyxy_pixels"]))
                    for overlay in row["overlays"]
                }
                if actual_by_candidate != expected_by_candidate:
                    raise ValueError(f"control changed base badge positions for {case_id}/{kind}/slot{slot}")
            rendered_metadata = {
                "schema": "stvg-adjacent-motion-fixed-badge-input-v1",
                "GT_read": False,
                "model_calls": 0,
                "case_id": case_id,
                "kind": kind,
                "presentation_seed": int(presentation_seed),
                "id_map": {str(key): int(value) for key, value in ids.items()},
                "color_map": {str(key): int(value) for key, value in colors.items()},
                "draw_order": [int(value) for value in order],
                "badge_layout_source": "base_same_slot_same_candidate",
                "base_badge_positions_by_slot_candidate": base_badges_by_slot,
                "actual_times_seconds": times,
                "mean_gap_encoder_fps": mean_gap_fps,
                "source_RGB_sha256_by_frame": {str(fid): decoded["rgb_hashes"][str(fid)] for fid in physical_ids},
                "records": rendered_records,
            }
            npy_path = case_dir / f"{kind}.npy"
            render_path = case_dir / f"{kind}_render.json"
            if npy_path.exists() or render_path.exists():
                raise FileExistsError(f"refusing to overwrite partial prepared input for {case_id}/{kind}")
            np.save(npy_path, rendered_tensor, allow_pickle=False)
            _write_once(render_path, rendered_metadata)
            tensor_sha = _sha_bytes(rendered_tensor.tobytes())
            file_sha = _sha_file(npy_path)
            prompt_query = case["caption"]
            donor_id = None
            if kind == "wrong_query":
                same_domain = [row for row in roster_records if row["dataset"] == rec["dataset"] and row["cohort"] == rec["cohort"]]
                index = next(i for i, row in enumerate(same_domain) if row["case_id"] == case_id)
                donor_id = str(same_domain[(index + 1) % len(same_domain)]["case_id"])
                prompt_query = source_case_by_id[donor_id]["caption"]
            missing_visible_ids = [
                sorted(int(ids[int(candidate_id)]) for candidate_id in frame_record["missing_candidate_ids"])
                for frame_record in rendered_records
            ]
            query_prompt = build_prompt(prompt_query, list(ids.values()), missing_visible_ids, times)
            prompt_sha = _sha_bytes(query_prompt.encode("utf-8"))
            prompt_sha_by_kind[kind] = prompt_sha
            candidate_bindings = [
                {
                    "candidate_id": int(candidate["candidate_id"]),
                    "visible_id": int(ids[int(candidate["candidate_id"])]),
                    "color_index": int(colors[int(candidate["candidate_id"])]),
                    "window_id": int(candidate["window_id"]),
                    "geometry_sha256": candidate["geometry_sha256"],
                    "is_Source": bool(candidate["is_Source"]),
                }
                for candidate in displayed
            ]
            job_id = f"{case_id}_{kind}"
            job_ids_for_case.append(job_id)
            job_records.append(
                {
                    "job_id": job_id,
                    "case_id": case_id,
                    "dataset": rec["dataset"],
                    "cohort": rec["cohort"],
                    "kind": kind,
                    "videos": [
                        {
                            "path": _root_relative(npy_path),
                            "sha256": file_sha,
                            "tensor_sha256": tensor_sha,
                            "fps": float(mean_gap_fps),
                            "actual_times_seconds": times,
                            "frame_count": 4,
                            "frame_shape_chw": [3, IMAGE_SIDE, IMAGE_SIDE],
                            "source_RGB_sha256_by_frame": {str(fid): decoded["rgb_hashes"][str(fid)] for fid in physical_ids},
                        }
                    ],
                    "prompt": query_prompt,
                    "prompt_sha256": prompt_sha,
                    "candidate_bindings": candidate_bindings,
                    "candidate_count": len(displayed),
                    "allowed_output_ids": [0] + sorted(int(value) for value in ids.values()),
                    "visible_id_to_candidate": {str(value): int(key) for key, value in ids.items()},
                    "presentation_seed": int(presentation_seed),
                    "badge_layout_source": "base_same_slot_same_candidate",
                    "wrong_query_donor_case_id": donor_id,
                    "source_case_sha256": rec["source_case_sha256"],
                    "source_model_state_sha256": old_record["source_model_state_sha256"],
                    "source_candidate_barrier_sha256": old_candidate_barrier_sha,
                    "source_grid_barrier_sha256": old_grid_barrier_sha,
                    "GT_read": False,
                    "model_calls": 0,
                }
            )
            file_manifest.extend([_file_row(npy_path), _file_row(render_path)])
            del rendered_tensor, provisional, control_records

        if not (
            prompt_sha_by_kind["base"]
            == prompt_sha_by_kind["color_swap"]
            == prompt_sha_by_kind["order_swap"]
        ):
            raise ValueError(f"order/color-only controls changed the teacher prompt for {case_id}")

        if len(job_ids_for_case) != 6 or set(job_ids_for_case) != {f"{case_id}_{kind}" for kind in KINDS}:
            raise AssertionError(f"case did not generate six fixed teacher jobs for {case_id}")
        input_records.append(
            {
                "case_id": case_id,
                "dataset": rec["dataset"],
                "cohort": rec["cohort"],
                "source_case_path": rec["source_case_path"],
                "source_case_sha256": rec["source_case_sha256"],
                "source_model_state_sha256": old_record["source_model_state_sha256"],
                "source_pixel_sha256": old_record["source_pixel_sha256"],
                "source_candidate_record_sha256": _canonical_sha(old_record),
                "source_grid_record_sha256": _canonical_sha(grid_record),
                "source_RGB_sha256_by_frame": decoded["rgb_hashes"],
                "canonical_video_sha256": decoded["canonical_video_sha256"],
                "source_video_sha256": decoded["source_video_sha256"],
                "request_sha256": decoded["request_sha256"],
                "positions": positions,
                "physical_frame_ids": physical_ids,
                "tracking_physical_frame_ids": dense_ids,
                "tracking_gray_sha256_by_frame": decoded["gray_hashes"],
                "tracking_gray_shape_hw": decoded["gray_shape_hw"],
                "source_wh_by_frame": decoded["raw_wh_by_frame"],
                "tracking_gray_wh_by_frame": decoded["resized_wh_by_frame"],
                "seed_rule": old_record["seed_rule"],
                "retrieved_union_window_ids": seed_window_ids,
                "candidate_count": len(candidates),
                "displayed_candidate_ids": [int(row["candidate_id"]) for row in displayed],
                "presentation_seed": int(presentation_seed),
                "base_visible_id_by_candidate": {str(key): int(value) for key, value in base_ids.items()},
                "base_color_by_candidate": {str(key): int(value) for key, value in base_colors.items()},
                "base_draw_order": [int(value) for value in base_order],
                "base_badge_positions_by_slot_candidate": base_badges_by_slot,
                "actual_times_seconds": times,
                "source_fps": fps,
                "mean_gap_encoder_fps": mean_gap_fps,
                "candidate_geometry": candidates,
                "teacher_job_ids": job_ids_for_case,
                "GT_read": False,
                "model_calls": 0,
            }
        )
        del gray_frames, decoded["gray_frames"], decoded["rgb_images"], base_tensor_provisional
        import gc
        gc.collect()

    total_candidates = sum(count_by_case.values())
    if total_candidates != 43 or len(input_records) != 8 or len(job_records) != 48:
        raise AssertionError("old8 P0/input preparation must retain exactly 43 candidates and 48 jobs")
    if len(file_manifest) != 96 or len({row["path"] for row in file_manifest}) != 96:
        raise AssertionError("all 48 teacher jobs require one unique tensor and render receipt")

    source_bindings = {
        "roster_lock_path": _root_relative(ROSTER_PATH),
        "roster_lock_sha256": roster_sha,
        "design_lock_path": _root_relative(DESIGN_PATH),
        "design_lock_sha256": design_sha,
        "protocol_path": _root_relative(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "current_method_sha256": current_method_sha,
        "old_candidate_barrier_path": _root_relative(OLD_CANDIDATE_BARRIER),
        "old_candidate_barrier_sha256": old_candidate_barrier_sha,
        "old_teacher_job_lock_path": _root_relative(OLD_TEACHER_JOB_LOCK),
        "old_teacher_job_lock_sha256": old_teacher_job_lock_sha,
        "old_grid_barrier_path": _root_relative(OLD_GRID_BARRIER),
        "old_grid_barrier_sha256": old_grid_barrier_sha,
        "source_tracklet_helper_path": _root_relative(ROOT / "scripts/stvg_adjacent_affine_tracklet_v1.py"),
        "source_tracklet_helper_sha256": helper_sha,
        "source_render_helper_path": _root_relative(ROOT / "scripts/prepare_stvg_scene_object_id_v1.py"),
        "source_render_helper_sha256": renderer_sha,
        "fixed_badge_helper_path": _root_relative(ROOT / "scripts/prepare_stvg_motion_order_layout001.py"),
        "fixed_badge_helper_sha256": fixed_badge_helper_sha,
        "prompt_render_basis_path": _root_relative(ROOT / "scripts/prepare_stvg_motion_guided_v1.py"),
        "prompt_render_basis_sha256": basis_helper_sha,
        "prepare_source_path": _root_relative(Path(__file__)),
        "prepare_source_sha256": _sha_file(Path(__file__)),
    }
    candidate_barrier = {
        "schema": SCHEMA,
        "status": "complete",
        "count": 8,
        "candidate_count": total_candidates,
        "teacher_job_count": 48,
        "records": input_records,
        "files": file_manifest,
        "source_bindings": source_bindings,
        "scope": "fixed historical old8 only; exact pre-existing 43-seed union; no candidate filtering",
        "tracking": {
            "physical_step": 1,
            "gray_long_edge_max": 640,
            "gray_resize_interpolation": "INTER_AREA",
            "source_rgb_retained_per_case": 4,
            "source_rgb_hashes_verified_against_original_source": True,
            "source_display_boxes": "original four source boxes",
            "source_geometry_boxes": "independently tracked Source trajectory",
            "candidate_geometry": "independent adjacent track from each seed at the second original observation",
            "tracklet_schema": TRACK_SCHEMA,
            "tracklet_config": TRACK_CONFIG,
        },
        "presentation": {
            "candidate_limit": 5,
            "controls": list(KINDS),
            "same_deterministic_presentation_seed": True,
            "badge_positions_fixed_to_base_per_slot_and_candidate": True,
            "wrong_query_donor": "next old8 case in the same dataset",
        },
        "elapsed_cpu_seconds": float(time.perf_counter() - started),
        "GT_read": False,
        "model_calls": 0,
        "GPU_used": False,
    }
    candidate_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    _write_once(candidate_path, candidate_barrier)
    candidate_sha = _sha_file(candidate_path)
    teacher_lock = {
        "schema": SCHEMA,
        "status": "locked",
        "count": 48,
        "records": job_records,
        "source_candidate_input_barrier_path": _root_relative(candidate_path),
        "source_candidate_input_barrier_sha256": candidate_sha,
        "source_roster_lock_sha256": roster_sha,
        "source_design_lock_sha256": design_sha,
        "source_protocol_sha256": protocol_sha,
        "source_old_candidate_barrier_sha256": old_candidate_barrier_sha,
        "source_old_teacher_job_lock_sha256": old_teacher_job_lock_sha,
        "source_old_grid_barrier_sha256": old_grid_barrier_sha,
        "GT_read": False,
        "model_calls": 0,
        "teacher_calls_pending": True,
        "created_unix": time.time(),
    }
    teacher_path = BASE / "TEACHER_JOB_LOCK.json"
    _write_once(teacher_path, teacher_lock)
    capture = {
        "schema": SCHEMA,
        "status": "prepared_waiting_for_teacher_calls",
        "scope": "P0 real adjacent-frame tracks plus fixed 48-request old8 input lock",
        "case_count": 8,
        "candidate_count": 43,
        "teacher_call_count": 48,
        "tensor_and_render_file_count": 96,
        "candidate_input_barrier_path": _root_relative(candidate_path),
        "candidate_input_barrier_sha256": candidate_sha,
        "teacher_job_lock_path": _root_relative(teacher_path),
        "teacher_job_lock_sha256": _sha_file(teacher_path),
        "source_bindings": source_bindings,
        "canonical_video_sha256_by_case": case_video_sha,
        "request_sha256_by_case": case_request_sha,
        "source_RGB_sha256_by_case": {
            row["case_id"]: row["source_RGB_sha256_by_frame"] for row in input_records
        },
        "elapsed_cpu_seconds": float(time.perf_counter() - started),
        "GT_read": False,
        "model_calls": 0,
        "GPU_used": False,
    }
    _write_once(BASE / "CAPTURE.json", capture)
    return capture


if __name__ == "__main__":
    result = prepare()
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"canonical_video_sha256_by_case", "request_sha256_by_case", "source_RGB_sha256_by_case"}},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
