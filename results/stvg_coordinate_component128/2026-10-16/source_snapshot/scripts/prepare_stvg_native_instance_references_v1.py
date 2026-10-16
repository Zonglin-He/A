#!/usr/bin/env python3
"""Prepare sparse, label-free native STVG-R1 instance references.

This module is deliberately import-inert.  ``prepare_references`` is the only
filesystem path and is gated on the sealed 8-case tracking and teacher
barriers.  ``--self-test`` exercises only synthetic CPU geometry and never
opens scope artifacts, videos, checkpoints, or annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_native_instance_common_v1 as common

BASE = common.BASE
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
TRACKING_BARRIER_PATH = BASE / "TRACKING_GLOBAL_BARRIER.json"
TEACHER_BARRIER_PATH = BASE / "TEACHER_GLOBAL_BARRIER.json"
TRACKING_RUNTIME_PATH = BASE / "TRACKING_RUNTIME.json"
TEACHER_RUNTIME_PATH = BASE / "TEACHER_RUNTIME.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
DIRECT_PATH = BASE / "DIRECT_TEACHER_READOUT.json"
GEOMETRY_PATH = BASE / "TEACHER_GEOMETRY_READBACK.json"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MASK_NAME_RE = re.compile(r"mask_(\d+)\.json\Z")
EXPECTED_CASES = 8
MAX_NEW_TOKENS = 2048


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _safe_root_file(relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("sealed file path must be a nonempty root-relative string")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("sealed file path must stay relative to the project root")
    path = (ROOT / rel).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("sealed file path escapes the project root")
    if not path.is_file():
        raise FileNotFoundError("sealed path is not a regular file")
    return path


def _xyxy_inclusive_to_normalized(box: Sequence[Any], width: int, height: int) -> tuple[list[float], list[float]]:
    """Convert inclusive integer pixels to continuous normalized xyxy/cxcywh.

    Inclusive xmax/ymax become the exclusive right/bottom edges ``+1`` before
    division.  This is the coordinate convention used by the student decoder's
    normalized continuous boxes; omitting the increment would shrink every box.
    """
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("inclusive box requires positive image dimensions and four coordinates")
    vals: list[int] = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("inclusive pixel bounds must be integers")
        vals.append(int(value))
    x1, y1, xmax, ymax = vals
    if not (0 <= x1 <= xmax < width and 0 <= y1 <= ymax < height):
        raise ValueError("inclusive pixel box is outside its decoded mask")
    x0n, y0n = x1 / width, y1 / height
    x2n, y2n = (xmax + 1) / width, (ymax + 1) / height
    if not (0.0 <= x0n < x2n <= 1.0 and 0.0 <= y0n < y2n <= 1.0):
        raise ValueError("inclusive-to-continuous normalization produced an invalid box")
    return [x0n, y0n, x2n, y2n], [
        (x0n + x2n) / 2.0,
        (y0n + y2n) / 2.0,
        x2n - x0n,
        y2n - y0n,
    ]


def _mask_iou(a: Any, b: Any) -> float:
    """Binary mask IoU for two nonempty same-shape synthetic/decoded masks."""
    import numpy as np

    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("adjacent masks must be two-dimensional and shape-identical")
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    if union <= 0:
        raise ValueError("empty masks cannot form a valid adjacent-pair IoU")
    return intersection / union


def _window_reliability(local_indices: Sequence[int], masks_by_local: Mapping[int, Any], center_local: int) -> dict[str, Any]:
    """Compute presence x mean valid adjacent-mask IoU in the fixed local window."""
    if not local_indices or len(local_indices) > 5:
        raise ValueError("the locked center window must have one to five local frames")
    indices = [int(value) for value in local_indices]
    if indices != sorted(set(indices)):
        raise ValueError("center window local indices must be unique and increasing")
    if center_local not in indices:
        raise ValueError("center local frame is absent from its fixed window")
    present = [index for index in indices if masks_by_local.get(index) is not None]
    presence_fraction = len(present) / len(indices)
    pair_scores = [
        _mask_iou(masks_by_local[left], masks_by_local[right])
        for left, right in zip(indices, indices[1:])
        if right == left + 1
        and masks_by_local.get(left) is not None
        and masks_by_local.get(right) is not None
    ]
    mean_iou = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    return {
        "window_frame_count": len(indices),
        "present_frame_count": len(present),
        "presence_fraction": presence_fraction,
        "valid_adjacent_pair_count": len(pair_scores),
        "mean_adjacent_mask_iou": mean_iou,
        "weight": presence_fraction * mean_iou,
        "center_present": masks_by_local.get(center_local) is not None,
    }


def _present_runs(values: Sequence[bool]) -> tuple[int, int]:
    """Return contiguous present-run count and internal absent-run count."""
    runs = sum(bool(value) and (i == 0 or not bool(values[i - 1])) for i, value in enumerate(values))
    present_positions = [i for i, value in enumerate(values) if value]
    if not present_positions:
        return 0, 0
    first, last = present_positions[0], present_positions[-1]
    internal_gaps = 0
    in_gap = False
    for value in values[first:last + 1]:
        if not value and not in_gap:
            internal_gaps += 1
            in_gap = True
        elif value:
            in_gap = False
    return runs, internal_gaps


def _validate_roster(roster: Mapping[str, Any]) -> list[dict[str, Any]]:
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise RuntimeError("native instance roster must be locked and GT-free")
    rows = roster.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES or roster.get("count") != EXPECTED_CASES:
        raise RuntimeError("native instance roster must contain exactly eight locked records")
    cases: set[str] = set()
    datasets: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("roster rows must be objects")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise RuntimeError("roster case IDs must be nonempty and unique")
        cases.add(case_id)
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise RuntimeError("roster dataset field is missing")
        datasets[dataset] = datasets.get(dataset, 0) + 1
        if not _is_sha(row.get("source_case_sha256")) or not _is_sha(row.get("request_sha256")):
            raise RuntimeError("roster source/request SHA binding is malformed")
        centers = row.get("centers")
        if not isinstance(centers, list) or len(centers) != 4:
            raise RuntimeError("every roster row must bind exactly four fixed observation centers")
        slots: set[int] = set()
        positions: set[int] = set()
        physical_ids: set[int] = set()
        for center in centers:
            if not isinstance(center, Mapping) or set(center) != {
                "slot", "center_physical_frame_id", "source_grid_position", "local_frame_indices"
            }:
                raise RuntimeError("roster center schema differs from the fixed four-center ABI")
            slot = center["slot"]
            physical = center["center_physical_frame_id"]
            position = center["source_grid_position"]
            window = center["local_frame_indices"]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (slot, physical, position)):
                raise RuntimeError("center slot/physical/source-grid positions must be nonnegative integers")
            if slot in slots or position in positions or physical in physical_ids:
                raise RuntimeError("center slots, source-grid positions, and physical IDs must be unique")
            slots.add(slot); positions.add(position); physical_ids.add(physical)
            if (not isinstance(window, list) or len(window) != 5
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in window)
                    or window != sorted(set(window))):
                raise RuntimeError("center local_frame_indices must be five unique increasing integers")
        if slots != {0, 1, 2, 3}:
            raise RuntimeError("center slots must be exactly 0..3")
    if len(datasets) != 2 or sorted(datasets.values()) != [4, 4]:
        raise RuntimeError("locked native roster must retain four cases per dataset")
    return rows


def _verify_global_barrier(path: Path, *, expected_stage: str, case_ids: set[str], runtime_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{expected_stage} global barrier is missing; no reference preparation is allowed")
    barrier = common.read(path)
    if barrier.get("status") != "complete" or barrier.get("count") != EXPECTED_CASES or barrier.get("GT_read") is not False:
        raise RuntimeError(f"{expected_stage} global barrier must be complete, count eight, and GT-free")
    runtime_sha = barrier.get("runtime_sha256")
    if not _is_sha(runtime_sha) or runtime_sha != common.sha(runtime_path):
        raise RuntimeError(f"{expected_stage} barrier is not bound to the current locked runtime")
    files = barrier.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_CASES:
        raise RuntimeError(f"{expected_stage} barrier must list exactly eight case files")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"case_id", "path", "sha256"}:
            raise RuntimeError(f"{expected_stage} barrier file entry has the wrong schema")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id not in case_ids or case_id in seen:
            raise RuntimeError(f"{expected_stage} barrier case set differs from the locked roster")
        if not _is_sha(entry.get("sha256")):
            raise RuntimeError(f"{expected_stage} barrier contains a malformed file digest")
        file_path = _safe_root_file(entry.get("path"))
        if common.sha(file_path) != entry["sha256"]:
            raise RuntimeError(f"{expected_stage} barrier file digest changed")
        seen.add(case_id)
    if seen != case_ids:
        raise RuntimeError(f"{expected_stage} global barrier does not cover the exact roster")
    return barrier


def _verify_tracking_case(case: Mapping[str, Any], request: Mapping[str, Any], request_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[int, Path]]:
    case_id = str(case["case_id"])
    accepted_path = BASE / "TRACKING" / case_id / "ACCEPTED.json"
    inventory_path = BASE / "TRACKING" / case_id / "MASK_INVENTORY.json"
    accepted = common.read(accepted_path)
    inventory = common.read(inventory_path)
    if accepted.get("status") != "pass" or accepted.get("GT_read") is not False:
        raise RuntimeError("tracking acceptance is not a passing GT-free record")
    if accepted.get("target_mode") != "all" or accepted.get("metadata_input") != "empty_mapping":
        raise RuntimeError("tracking acceptance differs from the all-category official policy")
    if accepted.get("input_request_sha256") != request_sha:
        raise RuntimeError("tracking acceptance is not bound to the locked canonical request")
    if not _is_sha(accepted.get("mask_inventory_sha256")) or common.sha(inventory_path) != accepted["mask_inventory_sha256"]:
        raise RuntimeError("tracking acceptance does not bind the case mask inventory")
    if inventory.get("GT_read") is not False or not isinstance(inventory.get("files"), list):
        raise RuntimeError("mask inventory must be a GT-free file list")
    if inventory.get("mask_count") != len(inventory["files"]) or len(inventory["files"]) != request["local_frame_count"]:
        raise RuntimeError("mask inventory must cover the complete canonical local frame clock")
    mask_dir = Path(accepted.get("mask_data_dir", "")).resolve(strict=True)
    if not mask_dir.is_dir() or mask_dir.name != "mask_data_filtered_1_3":
        raise RuntimeError("accepted masks must come from the official filtered mask directory")
    tracking_root = (BASE / "TRACKING" / case_id).resolve()
    reused = accepted.get("reused") is True
    if reused:
        expected_old = Path("artifacts/stvg_target_teacher_distillation_v1/tracking") / case_id / "ACCEPTED.json"
        if accepted.get("old_receipt_path") != str(expected_old):
            raise RuntimeError("reused mask cache does not name the fixed accepted predecessor")
        old_receipt_path = _safe_root_file(str(expected_old))
        if common.sha(old_receipt_path) != accepted.get("old_receipt_sha256"):
            raise RuntimeError("reused tracking receipt lineage digest changed")
        old = common.read(old_receipt_path)
        if (old.get("status") != "pass" or old.get("GT_read") is not False
                or old.get("input_request_sha256") != request_sha
                or old.get("marker_video_sha256") != accepted.get("marker_video_sha256")
                or old.get("mask_data_dir") != accepted.get("mask_data_dir")):
            raise RuntimeError("reused tracking lineage is not the same accepted request/mask source")
        cache_root = old_receipt_path.parent.resolve()
    else:
        cache_root = tracking_root
    if not mask_dir.is_relative_to(cache_root):
        raise RuntimeError("accepted filtered masks escape their tracking receipt directory")
    files: dict[int, Path] = {}
    for entry in inventory["files"]:
        if not isinstance(entry, Mapping) or set(entry) != {"index", "path", "sha256", "bytes"}:
            raise RuntimeError("mask inventory entry has the wrong schema")
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < request["local_frame_count"] or index in files:
            raise RuntimeError("mask inventory local frame indices are invalid or duplicated")
        if not _is_sha(entry.get("sha256")) or isinstance(entry.get("bytes"), bool) or not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise RuntimeError("mask inventory file digest/size is invalid")
        path = _safe_root_file(entry.get("path"))
        match = MASK_NAME_RE.fullmatch(path.name)
        if (match is None or int(match.group(1)) != index or not path.is_relative_to(mask_dir)
                or path.stat().st_size != entry["bytes"] or common.sha(path) != entry["sha256"]):
            raise RuntimeError("generated tracking mask bytes differ from the sealed inventory")
        files[index] = path
    if set(files) != set(range(request["local_frame_count"])):
        raise RuntimeError("tracking mask inventory is not a dense frame-index mapping")
    return accepted, inventory, files


def _teacher_eligibility(record: Mapping[str, Any], receipt: Mapping[str, Any], request: Mapping[str, Any], request_sha: str) -> tuple[bool, str, int | None]:
    """Require a complete uncapped one-call response before exposing references."""
    if receipt.get("status") != "complete" or receipt.get("GT_read") is not False:
        return False, "teacher_receipt_incomplete_or_not_gt_free", None
    if receipt.get("request_sha256") != request_sha:
        return False, "teacher_request_binding_mismatch", None
    if receipt.get("query_sha256") != request.get("query_sha256"):
        return False, "teacher_query_binding_mismatch", None
    if receipt.get("failure") is not None or receipt.get("one_request_one_generation") is not True or receipt.get("reprompt_count") != 0:
        return False, "teacher_generation_receipt_invalid", None
    generation = receipt.get("generation")
    if not isinstance(generation, Mapping):
        return False, "teacher_generation_receipt_missing", None
    output_tokens = generation.get("output_token_count")
    max_tokens = generation.get("max_new_tokens")
    calls = generation.get("generation_call_count")
    truncated = generation.get("output_truncated_at_cap")
    if (isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0
            or max_tokens != MAX_NEW_TOKENS or calls != 1 or not isinstance(truncated, bool)):
        return False, "teacher_truncation_evidence_missing_or_malformed", None
    if truncated or output_tokens >= MAX_NEW_TOKENS:
        return False, "teacher_output_truncated_at_2048_token_cap", output_tokens
    parsed = record.get("parsed")
    if not isinstance(parsed, Mapping) or parsed.get("status") != "valid":
        status = parsed.get("status") if isinstance(parsed, Mapping) else None
        return False, f"native_r1_not_valid:{status or 'missing'}", output_tokens
    candidate_id = parsed.get("candidate_id")
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id <= 0:
        return False, "native_r1_candidate_id_invalid", output_tokens
    interval = record.get("projected_interval")
    tube = record.get("mask_tube")
    if not isinstance(interval, Mapping) or not interval.get("local_frame_indices") or not isinstance(tube, Mapping):
        return False, "native_r1_interval_or_mask_tube_missing", output_tokens
    return True, "valid_uncapped_native_r1_selection", candidate_id


def _validate_teacher_binding(record: Mapping[str, Any], receipt: Mapping[str, Any], request: Mapping[str, Any], request_sha: str) -> None:
    binding = record.get("input_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("teacher record has no input binding")
    for key in ("source_video_sha256", "canonical_video_sha256", "canonical_pixel_sha256", "query_sha256", "frame_clock_sha256"):
        if binding.get(key) != request.get(key):
            raise RuntimeError(f"teacher record input binding differs for {key}")
    if binding.get("local_fps") != request.get("local_fps") or binding.get("local_frame_count") != request.get("local_frame_count"):
        raise RuntimeError("teacher record frame clock differs from the locked request")
    if binding.get("frame_map") != request.get("frame_map"):
        raise RuntimeError("teacher record frame map differs from the full locked physical frame clock")
    if binding.get("query_text") != request.get("query_text"):
        raise RuntimeError("teacher record query bytes differ from the locked request")
    receipt_record_sha = receipt.get("teacher_record_sha256")
    if not _is_sha(receipt_record_sha):
        raise RuntimeError("teacher receipt has no teacher-record digest")
    if receipt.get("request_sha256") != request_sha:
        raise RuntimeError("teacher receipt request digest differs from the locked request")
    for key in ("source_video_sha256", "canonical_video_sha256", "canonical_pixel_sha256"):
        if receipt.get(key) != request.get(key):
            raise RuntimeError(f"teacher receipt source binding differs for {key}")
    if receipt.get("runtime_sha256") != common.sha(TEACHER_RUNTIME_PATH):
        raise RuntimeError("teacher receipt runtime hash differs from the locked teacher runtime")
    if receipt.get("teacher_record_path") != str((BASE / "teacher" / f"{receipt.get('case_id')}.json").relative_to(ROOT)):
        raise RuntimeError("teacher receipt record path differs from the locked output location")


def _normalized_direct_tube(
    record: Mapping[str, Any],
    request: Mapping[str, Any],
    mask_files: Mapping[int, Path],
    candidate_id: int,
    sparse_local_indices: set[int],
) -> tuple[list[dict[str, Any]], dict[int, Any], tuple[int, int] | None]:
    """Read generated RLEs for one selected ID; retain only GT-free geometry."""
    import numpy as np
    from pycocotools import mask as mask_utils
    from scripts.stvg_r1_native_teacher_v1 import bbox_from_mask_inclusive

    tube = record.get("mask_tube")
    rows = tube.get("frames") if isinstance(tube, Mapping) else None
    if not isinstance(rows, list) or len(rows) != request["local_frame_count"]:
        raise RuntimeError("valid teacher output must contain a dense full-video mask tube")
    frame_map = request["frame_map"]
    masks_for_windows: dict[int, Any] = {}
    output: list[dict[str, Any]] = []
    dimensions: tuple[int, int] | None = None
    presence: list[bool] = []
    for index, (teacher_row, frame_clock_row) in enumerate(zip(rows, frame_map)):
        if not isinstance(teacher_row, Mapping):
            raise RuntimeError("teacher mask tube frame is malformed")
        if (teacher_row.get("local_frame_index") != index
                or teacher_row.get("physical_frame_id") != frame_clock_row["physical_frame_id"]
                or teacher_row.get("candidate_id") != candidate_id):
            raise RuntimeError("teacher mask tube differs from the locked dense frame clock or selected ID")
        frame_json = common.read(mask_files[index])
        objects = frame_json.get("objects")
        if not isinstance(objects, Mapping):
            raise RuntimeError("generated filtered mask file has no objects mapping")
        object_row = objects.get(str(candidate_id))
        mask = None
        box = None
        area = 0
        if isinstance(object_row, Mapping):
            rle = object_row.get("rle")
            if not isinstance(rle, Mapping):
                raise RuntimeError("selected generated instance is missing its RLE")
            mask = np.asarray(mask_utils.decode(rle))
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 2:
                raise RuntimeError("decoded selected-ID mask is not a single 2-D raster")
            dims = (int(mask.shape[1]), int(mask.shape[0]))
            if dimensions is None:
                dimensions = dims
            elif dimensions != dims:
                raise RuntimeError("canonical decoded mask dimensions changed across frames")
            box, area = bbox_from_mask_inclusive(mask)
        if bool(teacher_row.get("present")) != (box is not None):
            raise RuntimeError("teacher saved presence flag disagrees with its selected filtered RLE")
        saved_box = teacher_row.get("xyxy_inclusive_pixels")
        if saved_box != box or teacher_row.get("mask_area_px") != area:
            raise RuntimeError("teacher saved inclusive bbox/area differs from the official selected-ID RLE")
        presence.append(box is not None)
        if mask is not None and box is not None and index in sparse_local_indices:
            # Only the fixed four center windows stay resident; the full-video
            # direct readout below is streamed one selected mask at a time.
            masks_for_windows[index] = mask.astype(bool, copy=False)
        normalized_xyxy = normalized_cxcywh = None
        if box is not None:
            if dimensions is None:
                raise RuntimeError("selected mask dimensions are unavailable")
            normalized_xyxy, normalized_cxcywh = _xyxy_inclusive_to_normalized(box, *dimensions)
        output.append({
            "local_frame_index": index,
            "physical_frame_id": frame_clock_row["physical_frame_id"],
            "source_timestamp_s": frame_clock_row["source_timestamp_s"],
            "candidate_id": candidate_id,
            "present": box is not None,
            "xyxy_inclusive_pixels": box,
            "xyxy_normalized_continuous": normalized_xyxy,
            "cxcywh_normalized_continuous": normalized_cxcywh,
            "mask_area_px": area,
        })
    return output, masks_for_windows, dimensions


def _validate_center_bindings(case: Mapping[str, Any], request: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame_map = request["frame_map"]
    by_physical = {row["physical_frame_id"]: row["local_frame_index"] for row in frame_map}
    count = request["local_frame_count"]
    centers = []
    for center in case["centers"]:
        physical_id = center["center_physical_frame_id"]
        if physical_id not in by_physical:
            raise RuntimeError("roster center physical ID is absent from the teacher request frame map")
        center_local = by_physical[physical_id]
        window = list(center["local_frame_indices"])
        expected = list(range(max(0, center_local - 2), min(count, center_local + 3)))
        if window != expected:
            raise RuntimeError("roster local center window differs from the fixed clipped center±2 window")
        centers.append({
            "slot": int(center["slot"]),
            "center_physical_frame_id": int(physical_id),
            "source_grid_position": int(center["source_grid_position"]),
            "center_local_frame_index": int(center_local),
            "local_frame_indices": window,
        })
    return sorted(centers, key=lambda item: item["slot"])


def _prepare_case(case: Mapping[str, Any], *, tracking_entry: Mapping[str, Any], teacher_entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_id = str(case["case_id"])
    request_path = _safe_root_file(case.get("request_path"))
    request_sha = common.sha(request_path)
    if request_sha != case["request_sha256"]:
        raise RuntimeError("locked canonical request bytes differ from roster")
    from scripts.stvg_r1_native_teacher_v1 import validate_request
    request = validate_request(common.read(request_path))
    centers = _validate_center_bindings(case, request)

    accepted, inventory, mask_files = _verify_tracking_case(case, request, request_sha)
    accepted_path = BASE / "TRACKING" / case_id / "ACCEPTED.json"
    inventory_path = BASE / "TRACKING" / case_id / "MASK_INVENTORY.json"
    track_barrier_path = _safe_root_file(tracking_entry["path"])
    if track_barrier_path not in {accepted_path.resolve(), inventory_path.resolve()}:
        raise RuntimeError("tracking global barrier does not bind this case accepted/inventory artifact")
    if common.sha(track_barrier_path) != tracking_entry["sha256"]:
        raise RuntimeError("tracking barrier case digest differs from current accepted artifacts")

    teacher_path = BASE / "teacher" / f"{case_id}.json"
    receipt_path = BASE / "teacher" / f"{case_id}.receipt.json"
    teacher_record = common.read(teacher_path)
    receipt = common.read(receipt_path)
    if (teacher_entry.get("path") not in {
            str(teacher_path.relative_to(ROOT)), str(receipt_path.relative_to(ROOT))
        }):
        raise RuntimeError("teacher global barrier does not bind this case teacher record/receipt")
    teacher_barrier_file = _safe_root_file(teacher_entry["path"])
    if common.sha(teacher_barrier_file) != teacher_entry["sha256"]:
        raise RuntimeError("teacher barrier case digest differs from current teacher artifacts")
    if receipt.get("case_id") != case_id or receipt.get("dataset") != case.get("dataset"):
        raise RuntimeError("teacher receipt identity differs from the locked roster")
    if receipt.get("source_case_sha256") != case["source_case_sha256"]:
        raise RuntimeError("teacher receipt source binding differs from the locked roster")
    if (receipt.get("tracking_acceptance_sha256") != common.sha(BASE / "TRACKING" / case_id / "ACCEPTED.json")
            or receipt.get("mask_inventory_sha256") != accepted.get("mask_inventory_sha256")
            or receipt.get("mask_inventory_receipt_sha256") != common.sha(BASE / "TRACKING" / case_id / "MASK_INVENTORY.json")):
        raise RuntimeError("teacher receipt does not bind the current accepted tracking and mask inventory")
    binding = teacher_record.get("input_binding")
    if not isinstance(binding, Mapping) or binding.get("marker_video_sha256") != accepted.get("marker_video_sha256"):
        raise RuntimeError("teacher input is not bound to the accepted all-category marker video")
    if receipt.get("teacher_record_sha256") != common.sha(teacher_path):
        raise RuntimeError("teacher receipt does not bind the immutable teacher JSON")
    _validate_teacher_binding(teacher_record, receipt, request, request_sha)
    eligible, ineligible_reason, candidate = _teacher_eligibility(teacher_record, receipt, request, request_sha)
    parsed = teacher_record.get("parsed") if isinstance(teacher_record.get("parsed"), Mapping) else {}
    projected = teacher_record.get("projected_interval") if isinstance(teacher_record.get("projected_interval"), Mapping) else {}

    anchors: list[dict[str, Any]] = []
    center_audit: list[dict[str, Any]] = []
    direct_frames: list[dict[str, Any]] = []
    if eligible and candidate is not None:
        sparse_local_indices = {
            index
            for center in centers
            for index in center["local_frame_indices"]
        }
        direct_frames, selected_masks, dimensions = _normalized_direct_tube(
            teacher_record, request, mask_files, candidate, sparse_local_indices
        )
        for center in centers:
            metrics = _window_reliability(center["local_frame_indices"], selected_masks, center["center_local_frame_index"])
            selected_frame = next(row for row in direct_frames if row["local_frame_index"] == center["center_local_frame_index"])
            anchor = None
            if metrics["center_present"]:
                box_xyxy = selected_frame["xyxy_normalized_continuous"]
                box_cxcywh = selected_frame["cxcywh_normalized_continuous"]
                if box_xyxy is None or box_cxcywh is None:
                    raise RuntimeError("center mask is present but normalized geometry is missing")
                anchor = {
                    "slot": center["slot"],
                    "position": center["source_grid_position"],
                    "physical_frame_id": center["center_physical_frame_id"],
                    "local_frame_index": center["center_local_frame_index"],
                    "local_frame_indices": list(center["local_frame_indices"]),
                    "box_xyxy_normalized": box_xyxy,
                    "box_cxcywh": box_cxcywh,
                    "weight": metrics["weight"],
                    "reliability": {
                        "window_frame_count": metrics["window_frame_count"],
                        "present_frame_count": metrics["present_frame_count"],
                        "presence_fraction": metrics["presence_fraction"],
                        "valid_adjacent_pair_count": metrics["valid_adjacent_pair_count"],
                        "mean_adjacent_mask_iou": metrics["mean_adjacent_mask_iou"],
                    },
                    "candidate_id": candidate,
                    "coordinate_convention": "inclusive_pixels_to_exclusive_normalized_xyxy_then_cxcywh",
                }
                anchors.append(anchor)
            center_audit.append({
                "slot": center["slot"],
                "position": center["source_grid_position"],
                "physical_frame_id": center["center_physical_frame_id"],
                "local_frame_index": center["center_local_frame_index"],
                "local_frame_indices": list(center["local_frame_indices"]),
                "center_present": metrics["center_present"],
                "anchor_created": anchor is not None,
                "presence_fraction": metrics["presence_fraction"],
                "valid_adjacent_pair_count": metrics["valid_adjacent_pair_count"],
                "mean_adjacent_mask_iou": metrics["mean_adjacent_mask_iou"],
                "weight": metrics["weight"],
                "selected_candidate_id": candidate,
            })
    else:
        for center in centers:
            center_audit.append({
                "slot": center["slot"],
                "position": center["source_grid_position"],
                "physical_frame_id": center["center_physical_frame_id"],
                "local_frame_index": center["center_local_frame_index"],
                "local_frame_indices": list(center["local_frame_indices"]),
                "center_present": False,
                "anchor_created": False,
                "presence_fraction": 0.0,
                "valid_adjacent_pair_count": 0,
                "mean_adjacent_mask_iou": 0.0,
                "weight": 0.0,
                "selected_candidate_id": None,
            })

    parsed_status = parsed.get("status") if isinstance(parsed, Mapping) else None
    interval = {
        "start_time_s": parsed.get("start_time_s"),
        "end_time_s": parsed.get("end_time_s"),
        "start_local_frame": projected.get("start_local_frame"),
        "end_local_frame": projected.get("end_local_frame"),
        "local_frame_indices": list(projected.get("local_frame_indices", [])),
        "physical_frame_ids": list(projected.get("physical_frame_ids", [])),
    }
    reference_row = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "source_case_sha256": case["source_case_sha256"],
        "request_sha256": request_sha,
        "teacher_record_sha256": common.sha(teacher_path),
        "teacher_receipt_sha256": common.sha(receipt_path),
        "GT_read": False,
        "parsed_status": parsed_status,
        "candidate_id": candidate if eligible else None,
        "native_interval": interval if eligible else None,
        "anchors": anchors,
        "no_update": not bool(anchors),
        "reason": "valid_selected_mask_at_fixed_centers" if anchors else ineligible_reason if not eligible else "no_selected_mask_at_fixed_center",
        "raw_proxy_audit": center_audit,
    }
    direct_row = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "source_case_sha256": case["source_case_sha256"],
        "request_sha256": request_sha,
        "source_video_sha256": request["source_video_sha256"],
        "canonical_video_sha256": request["canonical_video_sha256"],
        "canonical_pixel_sha256": request["canonical_pixel_sha256"],
        "query_sha256": request["query_sha256"],
        "frame_clock_sha256": request["frame_clock_sha256"],
        "teacher_record_sha256": common.sha(teacher_path),
        "teacher_receipt_sha256": common.sha(receipt_path),
        "GT_read": False,
        "valid_uncapped_selection": bool(eligible),
        "selection_status": parsed_status,
        "selection_reason": ineligible_reason,
        "candidate_id": candidate if eligible else None,
        "native_interval": interval if eligible else None,
        "dense_frames": direct_frames if eligible else [],
    }
    present_flags = [row["present"] for row in direct_frames]
    segments, gaps = _present_runs(present_flags)
    interval_indices = set(interval["local_frame_indices"] if eligible else [])
    interval_present = sum(bool(row["present"]) for row in direct_frames if row["local_frame_index"] in interval_indices)
    interval_count = len(interval_indices)
    geometry_row = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "source_case_sha256": case["source_case_sha256"],
        "teacher_record_sha256": common.sha(teacher_path),
        "teacher_receipt_sha256": common.sha(receipt_path),
        "GT_read": False,
        "selection_valid_uncapped": bool(eligible),
        "parsed_status": parsed_status,
        "selected_candidate_id": candidate if eligible else None,
        "full_clock_frame_count": request["local_frame_count"],
        "selected_id_present_frame_count": sum(present_flags),
        "selected_id_full_clock_coverage": sum(present_flags) / len(present_flags) if present_flags else 0.0,
        "selected_id_presence_run_count": segments,
        "selected_id_internal_missing_run_count": gaps,
        "selected_id_missing_frame_count": len(present_flags) - sum(present_flags),
        "native_interval_frame_count": interval_count,
        "native_interval_present_frame_count": interval_present,
        "native_interval_coverage": interval_present / interval_count if interval_count else 0.0,
        "fixed_center_count": len(centers),
        "valid_center_anchor_count": len(anchors),
        "candidate_id_continuity_is_not_semantic_identity": True,
        "semantic_correctness_claimed": False,
    }
    return reference_row, direct_row, geometry_row


def prepare_references() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Prepare all three immutable artifacts after exact sealed-barrier checks.

    This function is provided for the root-controlled post-barrier call.  It is
    intentionally not called by the CPU contract or the module's default CLI.
    """
    roster = common.read(ROSTER_PATH)
    rows = _validate_roster(roster)
    case_ids = {str(row["case_id"]) for row in rows}
    tracking_runtime = common.verify("tracking")
    teacher_runtime = common.verify("teacher")
    if tracking_runtime.get("GT_read") is not False or teacher_runtime.get("GT_read") is not False:
        raise RuntimeError("tracking and teacher runtimes must remain explicitly GT-free")
    tracking_barrier = _verify_global_barrier(
        TRACKING_BARRIER_PATH,
        expected_stage="tracking",
        case_ids=case_ids,
        runtime_path=TRACKING_RUNTIME_PATH,
    )
    teacher_barrier = _verify_global_barrier(
        TEACHER_BARRIER_PATH,
        expected_stage="teacher",
        case_ids=case_ids,
        runtime_path=TEACHER_RUNTIME_PATH,
    )
    if any(path.exists() for path in (REFERENCE_PATH, DIRECT_PATH, GEOMETRY_PATH)):
        raise FileExistsError("native reference artifacts are immutable and will never be overwritten")
    tracking_entries = {item["case_id"]: item for item in tracking_barrier["files"]}
    teacher_entries = {item["case_id"]: item for item in teacher_barrier["files"]}

    # Install the repository's inference guard only after all preflight inputs
    # establish the exact authorized GT-free sources.
    sys.addaudithook(common.guard)
    reference_rows = []
    direct_rows = []
    geometry_rows = []
    for case in rows:
        ref, direct, geometry = _prepare_case(
            case,
            tracking_entry=tracking_entries[str(case["case_id"])],
            teacher_entry=teacher_entries[str(case["case_id"])],
        )
        reference_rows.append(ref)
        direct_rows.append(direct)
        geometry_rows.append(geometry)

    sources = {
        "scripts/prepare_stvg_native_instance_references_v1.py": common.sha(Path(__file__)),
        "scripts/stvg_native_instance_common_v1.py": common.sha(ROOT / "scripts/stvg_native_instance_common_v1.py"),
        "scripts/stvg_r1_native_teacher_v1.py": common.sha(ROOT / "scripts/stvg_r1_native_teacher_v1.py"),
    }
    barrier_bindings = {
        "roster_sha256": common.sha(ROSTER_PATH),
        "tracking_barrier_sha256": common.sha(TRACKING_BARRIER_PATH),
        "teacher_barrier_sha256": common.sha(TEACHER_BARRIER_PATH),
        "tracking_runtime_sha256": common.sha(TRACKING_RUNTIME_PATH),
        "teacher_runtime_sha256": common.sha(TEACHER_RUNTIME_PATH),
    }
    reference_lock = {
        "schema": "stvg-native-instance-reference-lock-v1",
        "status": "locked",
        "count": len(reference_rows),
        "GT_read": False,
        "created_by_sha256": sources["scripts/prepare_stvg_native_instance_references_v1.py"],
        **barrier_bindings,
        "coordinate_conversion": "inclusive_pixels_to_exclusive_normalized_xyxy_then_cxcywh",
        "reliability_definition": "selected_ID_presence_fraction_over_clipped_center_plus_minus_2_times_mean_IoU_of_valid_adjacent_present_mask_pairs; no valid pair gives zero",
        "records": reference_rows,
    }
    direct_readout = {
        "schema": "stvg-native-instance-direct-teacher-readout-v1",
        "status": "complete",
        "count": len(direct_rows),
        "GT_read": False,
        "created_by_sha256": sources["scripts/prepare_stvg_native_instance_references_v1.py"],
        **barrier_bindings,
        "records": direct_rows,
    }
    geometry_readback = {
        "schema": "stvg-native-instance-teacher-geometry-readback-v1",
        "status": "complete",
        "count": len(geometry_rows),
        "GT_read": False,
        "semantic_correctness_claimed": False,
        "created_by_sha256": sources["scripts/prepare_stvg_native_instance_references_v1.py"],
        **barrier_bindings,
        "scope": "label-free selected-ID coverage, run fragmentation, interval coverage, and fixed-center mask-proxy counts; not semantic identity or target correctness",
        "records": geometry_rows,
    }
    # A compact receipt binds source closure without duplicating the roster,
    # query, runtime contents, or raw teacher output into public-facing fields.
    reference_lock["source_bindings"] = sources
    direct_readout["source_bindings"] = sources
    geometry_readback["source_bindings"] = sources

    common.write(REFERENCE_PATH, reference_lock)
    common.write(DIRECT_PATH, direct_readout)
    common.write(GEOMETRY_PATH, geometry_readback)
    return reference_lock, direct_readout, geometry_readback


def run_cpu_contracts() -> dict[str, Any]:
    """Synthetic CPU-only checks; no workspace artifact or media reads."""
    assert _xyxy_inclusive_to_normalized([1, 1, 2, 2], 4, 4) == ([0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 0.5, 0.5])
    assert _xyxy_inclusive_to_normalized([0, 0, 3, 3], 4, 4) == ([0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0])
    try:
        _xyxy_inclusive_to_normalized([1, 1, 4, 4], 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-image inclusive bbox was accepted")

    import numpy as np
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [0, 0]], dtype=bool)
    c = np.array([[0, 1], [0, 0]], dtype=bool)
    empty = np.zeros((2, 2), dtype=bool)
    # [A, A, B, missing, B]: presence=4/5; valid adjacent pair IoUs are 1 and
    # 1/2, giving mean=.75 and reliability=.60.  Missing frame 3 breaks pairs.
    values = {0: a, 1: a, 2: b, 3: None, 4: b}
    reliability = _window_reliability([0, 1, 2, 3, 4], values, 2)
    assert reliability["presence_fraction"] == 0.8
    assert reliability["valid_adjacent_pair_count"] == 2
    assert reliability["mean_adjacent_mask_iou"] == 0.75
    assert math.isclose(reliability["weight"], 0.6, rel_tol=0.0, abs_tol=1e-15)
    assert reliability["center_present"] is True
    no_pairs = _window_reliability([0, 1, 2, 3, 4], {0: a, 1: None, 2: a, 3: None, 4: a}, 2)
    assert no_pairs["valid_adjacent_pair_count"] == 0 and no_pairs["weight"] == 0.0
    absent_center = _window_reliability([0, 1, 2, 3, 4], {0: a, 1: a, 2: None, 3: a, 4: a}, 2)
    assert absent_center["center_present"] is False
    # A present RLE object whose decoded raster is all-zero has no bbox and is
    # treated as absent, never as a valid center or adjacent-pair observation.
    from scripts.stvg_r1_native_teacher_v1 import bbox_from_mask_inclusive
    empty_box, empty_area = bbox_from_mask_inclusive(empty)
    assert empty_box is None and empty_area == 0
    empty_as_absent = _window_reliability([0, 1, 2], {0: a, 1: None if empty_box is None else empty, 2: a}, 1)
    assert empty_as_absent["center_present"] is False and empty_as_absent["valid_adjacent_pair_count"] == 0
    # A boundary window is clipped by roster construction; no extra positions
    # are synthesized.  Reliability uses only the three supplied frames.
    boundary = _window_reliability([0, 1, 2], {0: a, 1: a, 2: None}, 1)
    assert boundary["window_frame_count"] == 3 and boundary["presence_fraction"] == 2 / 3
    segments, gaps = _present_runs([True, True, False, True, False, True])
    assert (segments, gaps) == (3, 2)
    # Explicit schema-level truncation/invalid cases remain ineligible.
    dummy_req = {"query_sha256": "a" * 64}
    valid_record = {"parsed": {"status": "valid", "candidate_id": 7},
                    "projected_interval": {"local_frame_indices": [1]}, "mask_tube": {"frames": []}}
    valid_receipt = {"status": "complete", "GT_read": False, "request_sha256": "b" * 64,
                     "query_sha256": "a" * 64, "failure": None, "one_request_one_generation": True,
                     "reprompt_count": 0, "generation": {"output_token_count": 17, "max_new_tokens": 2048,
                     "generation_call_count": 1, "output_truncated_at_cap": False}}
    assert _teacher_eligibility(valid_record, valid_receipt, dummy_req, "b" * 64)[0] is True
    truncated = json.loads(json.dumps(valid_receipt)); truncated["generation"]["output_truncated_at_cap"] = True
    assert _teacher_eligibility(valid_record, truncated, dummy_req, "b" * 64)[0] is False
    capped = json.loads(json.dumps(valid_receipt)); capped["generation"]["output_token_count"] = 2048
    assert _teacher_eligibility(valid_record, capped, dummy_req, "b" * 64)[0] is False
    invalid = {**valid_record, "parsed": {"status": "unknown_candidate_id", "candidate_id": 7}}
    assert _teacher_eligibility(invalid, valid_receipt, dummy_req, "b" * 64)[0] is False
    bad_call = json.loads(json.dumps(valid_receipt)); bad_call["generation"]["generation_call_count"] = 2
    assert _teacher_eligibility(valid_record, bad_call, dummy_req, "b" * 64)[0] is False
    assert "torch" not in sys.modules and "transformers" not in sys.modules and "pycocotools" not in sys.modules
    return {
        "status": "pass",
        "mode": "synthetic_cpu_only",
        "workspace_reads": 0,
        "media_reads": 0,
        "mask_payload_reads": 0,
        "GT_reads": 0,
        "model_imports": 0,
        "checks": [
            "inclusive xmax/ymax are converted to exclusive normalized right/bottom edges",
            "reliability uses the exact clipped five-frame presence and valid-adjacent-pair IoU rule",
            "no valid adjacent pair yields zero reliability",
            "missing and all-zero center masks create no anchor instead of fallback",
            "boundary windows use only roster-supplied clipped local indices",
            "presence-run and internal-gap counts are separate label-free diagnostics",
            "invalid, missing-truncation-evidence, multi-call, and capped teacher responses fail closed",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run only synthetic CPU checks")
    args = parser.parse_args()
    if args.self_test:
        output = run_cpu_contracts()
    else:
        references, direct, geometry = prepare_references()
        output = {
            "status": "complete",
            "count": references["count"],
            "reference_lock_sha256": common.sha(REFERENCE_PATH),
            "direct_readout_sha256": common.sha(DIRECT_PATH),
            "geometry_readback_sha256": common.sha(GEOMETRY_PATH),
            "GT_read": False,
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
