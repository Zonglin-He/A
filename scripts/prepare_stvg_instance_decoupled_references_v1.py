#!/usr/bin/env python3
"""Prepare the locked eight-case WHO/WHEN-decoupled STVG references.

Importing this module is inert. ``run_cpu_contracts`` uses synthetic masks
only. ``prepare_references`` is a separate root-controlled, CPU-only path that
reads the already sealed source, teacher JSON, and generated filtered-mask
RLEs; it never opens annotations, decodes video, or loads a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_stvg_native_instance_references_v1 as oldprep
from scripts import stvg_native_instance_common_v1 as common

BASE = ROOT / "artifacts/stvg_instance_identity_update_v1"
OLD_BASE = common.BASE
DESIGN_LOCK_PATH = BASE / "DESIGN_LOCK.json"
STEP_RUNTIME_PATH = BASE / "STEP_DIAGNOSTIC_RUNTIME.json"
PREPARATION_RUNTIME_PATH = BASE / "ROOT_PREPARATION_RUNTIME.json"
ROSTER_PATH = OLD_BASE / "ROSTER_LOCK.json"
OLD_REFERENCE_PATH = OLD_BASE / "REFERENCE_LOCK.json"
TRACKING_BARRIER_PATH = OLD_BASE / "TRACKING_GLOBAL_BARRIER.json"
TEACHER_BARRIER_PATH = OLD_BASE / "TEACHER_GLOBAL_BARRIER.json"
TRACKING_RUNTIME_PATH = OLD_BASE / "TRACKING_RUNTIME.json"
TEACHER_RUNTIME_PATH = OLD_BASE / "TEACHER_RUNTIME.json"
PROTOCOL_PATH = ROOT / "protocols/stvg_instance_identity_update_v1.md"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
GEOMETRY_PATH = BASE / "TEACHER_GEOMETRY_ONLY_READOUT.json"
READBACK_PATH = BASE / "REFERENCE_PREPARATION_READBACK.json"
EXPECTED_CASES = 8
MAX_NEW_TOKENS = 2048
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _formatted_json_sha(value: Any) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return _sha_bytes(data)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _safe_root_file(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("locked path must be a nonempty project-relative string")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("locked path must remain project-relative")
    path = (ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("locked path is outside the project or is not a file")
    return path


def _float_list(value: Any, *, width: int, name: str) -> list[list[float]]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite N x {width} array")
    return array.tolist()


def _int_list(value: Any, *, name: str) -> list[int]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    result: list[int] = []
    for item in array.tolist():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or int(item) != item:
            raise ValueError(f"{name} must contain integer-valued frame IDs")
        result.append(int(item))
    if not result or any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} must be nonempty and strictly increasing")
    return result


def _valid_candidate_id(parsed: Mapping[str, Any], inventory_ids: set[int]) -> tuple[int | None, bool, bool]:
    """Validate WHO syntax and inventory independently of the WHEN status."""
    value = parsed.get("candidate_id")
    syntactic = isinstance(value, int) and not isinstance(value, bool) and value > 0
    candidate = int(value) if syntactic else None
    return candidate, syntactic, bool(syntactic and candidate in inventory_ids)


def _valid_qa_interval(value: Any) -> tuple[int, int]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in value)):
        raise ValueError("QA interval must be two integer physical-frame bounds")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start:
        raise ValueError("QA interval must be a nonempty half-open physical-frame interval")
    return start, end


def _qa_mask_presence(frame_map: Sequence[Mapping[str, Any]], boxes_by_local: Mapping[int, Any], interval: tuple[int, int]) -> bool:
    start, end = interval
    return any(
        start <= int(clock["physical_frame_id"]) < end and boxes_by_local.get(int(clock["local_frame_index"])) is not None
        for clock in frame_map
    )


def _make_anchor_sets(
    centers: Sequence[Mapping[str, Any]],
    frames_by_local: Mapping[int, Mapping[str, Any]],
    masks_by_local: Mapping[int, Any],
    candidate_id: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return raw center anchors, structurally supported anchors, and audit rows."""
    import numpy as np

    nonempty_masks: dict[int, Any] = {}
    for local, mask in masks_by_local.items():
        if mask is None:
            continue
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError("center-window masks must be two-dimensional")
        if array.size and bool(array.any()):
            nonempty_masks[int(local)] = array.astype(bool, copy=False)
    raw: list[dict[str, Any]] = []
    supported: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for center in centers:
        local = int(center["center_local_frame_index"])
        metrics = oldprep._window_reliability(center["local_frame_indices"], nonempty_masks, local)
        frame = frames_by_local[local]
        anchor = None
        if metrics["center_present"] and candidate_id is not None:
            box_xyxy = frame.get("xyxy_normalized_continuous")
            box_cxcywh = frame.get("cxcywh_normalized_continuous")
            if box_xyxy is None or box_cxcywh is None:
                raise ValueError("present center mask has no normalized bbox")
            reliability = {
                "window_frame_count": int(metrics["window_frame_count"]),
                "present_frame_count": int(metrics["present_frame_count"]),
                "presence_fraction": float(metrics["presence_fraction"]),
                "valid_adjacent_pair_count": int(metrics["valid_adjacent_pair_count"]),
                "mean_adjacent_mask_iou": float(metrics["mean_adjacent_mask_iou"]),
                "structural_presence_supported": bool(metrics["valid_adjacent_pair_count"] >= 1),
            }
            anchor = {
                "slot": int(center["slot"]),
                "position": int(center["source_grid_position"]),
                "physical_frame_id": int(center["center_physical_frame_id"]),
                "local_frame_index": local,
                "box_xyxy_normalized": list(box_xyxy),
                "box_cxcywh": list(box_cxcywh),
                # Keep the frozen legacy reliability weight unchanged. The new
                # presence fraction is exposed separately for the later P1 lock.
                "weight": float(metrics["weight"]),
                "reliability": reliability,
                "candidate_id": int(candidate_id),
                "coordinate_convention": "inclusive_pixels_to_exclusive_normalized_xyxy_then_cxcywh",
            }
            raw.append(anchor)
            if metrics["valid_adjacent_pair_count"] >= 1:
                supported.append(copy.deepcopy(anchor))
        audit.append({
            "slot": int(center["slot"]),
            "position": int(center["source_grid_position"]),
            "physical_frame_id": int(center["center_physical_frame_id"]),
            "local_frame_index": local,
            "local_frame_indices": list(center["local_frame_indices"]),
            "center_mask_present": bool(metrics["center_present"]),
            "presence_fraction": float(metrics["presence_fraction"]),
            "valid_adjacent_pair_count": int(metrics["valid_adjacent_pair_count"]),
            "mean_adjacent_mask_iou": float(metrics["mean_adjacent_mask_iou"]),
            "legacy_weight": float(metrics["weight"]),
            "raw_anchor_created": anchor is not None,
            "presence_supported_anchor_created": anchor is not None and metrics["valid_adjacent_pair_count"] >= 1,
        })
    return raw, supported, audit


def _geometry_arm(
    *,
    frame_ids: Sequence[int],
    frozen_boxes: Sequence[Sequence[float]],
    teacher_boxes_by_physical: Mapping[int, Sequence[float] | None],
    candidate_id: int | None,
    admitted: bool,
    qa_interval: tuple[int, int],
    abstain_reason: str,
) -> dict[str, Any]:
    """Map real masks over the full source grid and use exact F boxes when absent."""
    if len(frame_ids) != len(frozen_boxes):
        raise ValueError("source frame grid and Frozen F boxes must have equal length")
    output_boxes: list[list[float]] = []
    teacher_boxes: list[list[float] | None] = []
    present: list[bool] = []
    fallback: list[bool] = []
    reasons: list[str | None] = []
    sources: list[str] = []
    for physical, frozen_box in zip(frame_ids, frozen_boxes):
        raw_box = teacher_boxes_by_physical.get(int(physical)) if admitted else None
        has_teacher = raw_box is not None
        frozen = [float(x) for x in frozen_box]
        if len(frozen) != 4 or not all(math.isfinite(x) for x in frozen):
            raise ValueError("Frozen F fallback box must be finite xyxy")
        if has_teacher:
            box = [float(x) for x in raw_box]
            if len(box) != 4 or not all(math.isfinite(x) for x in box):
                raise ValueError("teacher bbox must be finite normalized xyxy")
            output_boxes.append(box)
            teacher_boxes.append(box)
            present.append(True)
            fallback.append(False)
            reasons.append(None)
            sources.append("selected_id_mask_bbox")
        else:
            output_boxes.append(frozen)
            teacher_boxes.append(None)
            present.append(False)
            fallback.append(True)
            reasons.append(abstain_reason if not admitted else "selected_id_mask_absent_at_grid_frame")
            sources.append("frozen_source_fallback")
    return {
        "status": "mask_geometry_with_frozen_fallback" if admitted else "all_frozen_fallback",
        "candidate_id": candidate_id if admitted else None,
        "interval_frame_bounds_half_open": [int(qa_interval[0]), int(qa_interval[1])],
        "frame_ids_physical": [int(value) for value in frame_ids],
        "boxes_xyxy_normalized": output_boxes,
        "teacher_boxes_xyxy_normalized": teacher_boxes,
        "teacher_mask_present": present,
        "teacher_mask_missing": [not item for item in present],
        "fallback_used": fallback,
        "fallback_reason": reasons,
        "box_source": sources,
        "interpolation_used": False,
        "nearest_frame_replacement_used": False,
    }


def _decode_selected_id(
    *,
    candidate_id: int,
    request: Mapping[str, Any],
    mask_files: Mapping[int, Path],
    sparse_local_indices: set[int],
    expected_width: int,
    expected_height: int,
) -> tuple[list[dict[str, Any]], dict[int, Any], tuple[int, int]]:
    """Read only the chosen generated filtered RLE and retain boxes, not masks."""
    import numpy as np
    from pycocotools import mask as mask_utils
    from scripts.stvg_r1_native_teacher_v1 import bbox_from_mask_inclusive

    rows: list[dict[str, Any]] = []
    sparse_masks: dict[int, Any] = {}
    dimensions = (expected_width, expected_height)
    for local_index, clock in enumerate(request["frame_map"]):
        frame_data = common.read(mask_files[local_index])
        objects = frame_data.get("objects")
        if not isinstance(objects, Mapping):
            raise RuntimeError("filtered tracker mask JSON is missing its objects map")
        object_row = objects.get(str(candidate_id))
        mask = None
        box = None
        area = 0
        if object_row is not None:
            if not isinstance(object_row, Mapping) or not isinstance(object_row.get("rle"), Mapping):
                raise RuntimeError("selected filtered instance row has no valid RLE")
            mask = np.asarray(mask_utils.decode(object_row["rle"]))
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 2:
                raise RuntimeError("selected mask RLE did not decode to a 2-D raster")
            actual = (int(mask.shape[1]), int(mask.shape[0]))
            if actual != dimensions:
                raise RuntimeError("selected mask RLE dimensions differ from the frozen source geometry")
            box, area = bbox_from_mask_inclusive(mask)
            if box is not None and local_index in sparse_local_indices:
                sparse_masks[local_index] = mask.astype(bool, copy=False)
        xyxy = cxcywh = None
        if box is not None:
            xyxy, cxcywh = oldprep._xyxy_inclusive_to_normalized(box, expected_width, expected_height)
        rows.append({
            "local_frame_index": local_index,
            "physical_frame_id": int(clock["physical_frame_id"]),
            "xyxy_inclusive_pixels": box,
            "xyxy_normalized_continuous": xyxy,
            "cxcywh_normalized_continuous": cxcywh,
            "mask_area_px": int(area),
            "present": box is not None,
        })
    return rows, sparse_masks, dimensions


def _preflight_locked_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    roster = common.read(ROSTER_PATH)
    roster_rows = oldprep._validate_roster(roster)
    case_ids = {str(row["case_id"]) for row in roster_rows}
    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError("locked source roster must contain exactly eight cases")

    design = common.read(DESIGN_LOCK_PATH)
    if (design.get("status") != "locked" or design.get("GT_read") is not False
            or design.get("count") != EXPECTED_CASES or design.get("historically_exposed") is not True):
        raise RuntimeError("new DESIGN_LOCK must be locked, eight-case, GT-free, and expose-history aware")
    if design.get("protocol_sha256") != common.sha(PROTOCOL_PATH):
        raise RuntimeError("new protocol bytes differ from DESIGN_LOCK")
    if design.get("CURRENT_sha256") != common.sha(ROOT / "methods/CURRENT_METHOD.json"):
        raise RuntimeError("CURRENT_METHOD changed after the new DESIGN_LOCK")

    step_runtime = common.read(STEP_RUNTIME_PATH)
    if (step_runtime.get("status") != "locked" or step_runtime.get("GT_read") is not False
            or step_runtime.get("count") != EXPECTED_CASES or step_runtime.get("design_lock_sha256") != common.sha(DESIGN_LOCK_PATH)):
        raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME is not locked to the current DESIGN_LOCK")

    old_files = {
        "GLOBAL_PREDICTION_BARRIER.json": OLD_BASE / "GLOBAL_PREDICTION_BARRIER.json",
        "REFERENCE_LOCK.json": OLD_REFERENCE_PATH,
        "ROSTER_LOCK.json": ROSTER_PATH,
        "STUDENT_GLOBAL_BARRIER.json": OLD_BASE / "STUDENT_GLOBAL_BARRIER.json",
        "TEACHER_GLOBAL_BARRIER.json": TEACHER_BARRIER_PATH,
        "TRACKING_GLOBAL_BARRIER.json": TRACKING_BARRIER_PATH,
    }
    prior = design.get("prior_bindings")
    if not isinstance(prior, Mapping) or set(prior) != set(old_files):
        raise RuntimeError("DESIGN_LOCK prior binding set differs from the required frozen predecessor set")
    for name, path in old_files.items():
        if not _is_sha(prior.get(name)) or common.sha(path) != prior[name]:
            raise RuntimeError(f"frozen predecessor changed after DESIGN_LOCK: {name}")
    if step_runtime.get("reference_path") != str(OLD_REFERENCE_PATH.relative_to(ROOT)) or step_runtime.get("reference_sha256") != common.sha(OLD_REFERENCE_PATH):
        raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME does not bind the original strict reference lock")

    for stage, path in (
        ("tracking", ROOT / "scripts/stvg_r1_native_teacher_v1.py"),
        ("teacher", ROOT / "scripts/stvg_r1_native_teacher_v1.py"),
    ):
        common.verify(stage)
    tracking_runtime = common.read(TRACKING_RUNTIME_PATH)
    teacher_runtime = common.read(TEACHER_RUNTIME_PATH)
    if tracking_runtime.get("GT_read") is not False or teacher_runtime.get("GT_read") is not False:
        raise RuntimeError("original tracking/teacher runtimes must remain GT-free")
    for path, stage, runtime_path in (
        (TRACKING_BARRIER_PATH, "tracking", TRACKING_RUNTIME_PATH),
        (TEACHER_BARRIER_PATH, "teacher", TEACHER_RUNTIME_PATH),
    ):
        oldprep._verify_global_barrier(path, expected_stage=stage, case_ids=case_ids, runtime_path=runtime_path)

    pins = step_runtime.get("pins")
    expected_pins = {
        "methods/decota_final_simplified_v1/objectives.py",
        "scripts/run_stvg_native_instance_student_v1.py",
        "scripts/stvg_native_instance_spatial_fit_v1.py",
        "scripts/stvg_target_teacher_student_v1.py",
        "vg_tta/decota_spatial_opd_tunable_v1.py",
    }
    if not isinstance(pins, Mapping) or set(pins) != expected_pins:
        raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME code pin set differs from its locked ABI")
    for relative, digest in pins.items():
        if not _is_sha(digest) or common.sha(ROOT / relative) != digest:
            raise RuntimeError(f"STEP_DIAGNOSTIC_RUNTIME code pin changed: {relative}")

    student_records = step_runtime.get("records")
    if not isinstance(student_records, list) or len(student_records) != EXPECTED_CASES:
        raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME must bind exactly eight prior saved student records")
    seen: set[str] = set()
    for row in student_records:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "path", "sha256", "bytes"}:
            raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME student record schema is malformed")
        case_id = row["case_id"]
        if case_id not in case_ids or case_id in seen or not _is_sha(row["sha256"]):
            raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME case set or digest is malformed")
        path = _safe_root_file(row["path"])
        if (path.parent != (OLD_BASE / "student").resolve() or path.name != f"{case_id}.pt"
                or path.stat().st_size != row["bytes"] or common.sha(path) != row["sha256"]):
            raise RuntimeError("prior student saved artifact differs from STEP_DIAGNOSTIC_RUNTIME")
        seen.add(case_id)
    if seen != case_ids:
        raise RuntimeError("STEP_DIAGNOSTIC_RUNTIME record set differs from the locked roster")

    runtime = common.read(PREPARATION_RUNTIME_PATH)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("ROOT_PREPARATION_RUNTIME must be locked and GT-free before preparation")
    runtime_pins = runtime.get("pins")
    script_path = "scripts/prepare_stvg_instance_decoupled_references_v1.py"
    if not isinstance(runtime_pins, Mapping) or runtime_pins.get(script_path) != common.sha(Path(__file__)):
        raise RuntimeError("ROOT_PREPARATION_RUNTIME must pin this exact preparation script")
    if any(path.exists() for path in (REFERENCE_PATH, GEOMETRY_PATH, READBACK_PATH)):
        raise FileExistsError("new-scope preparation outputs already exist; immutable outputs are never overwritten")
    return roster_rows, design, step_runtime, tracking_runtime, teacher_runtime


def _prepare_one(case: Mapping[str, Any], *, tracking_entry: Mapping[str, Any], teacher_entry: Mapping[str, Any], old_reference: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_id = str(case["case_id"])
    request_path = _safe_root_file(case["request_path"])
    request_sha = common.sha(request_path)
    if request_sha != case["request_sha256"]:
        raise RuntimeError("canonical request differs from the locked roster")
    from scripts.stvg_r1_native_teacher_v1 import validate_request, _load_mask_inventory
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case

    request = validate_request(common.read(request_path))
    centers = oldprep._validate_center_bindings(case, request)
    qa_interval = _valid_qa_interval(case.get("qa_interval"))
    if any(not qa_interval[0] <= center["center_physical_frame_id"] < qa_interval[1] for center in centers):
        raise RuntimeError("a fixed observation center lies outside the locked QA half-open interval")

    accepted, inventory, mask_files = oldprep._verify_tracking_case(case, request, request_sha)
    accepted_path = OLD_BASE / "TRACKING" / case_id / "ACCEPTED.json"
    inventory_path = OLD_BASE / "TRACKING" / case_id / "MASK_INVENTORY.json"
    if tracking_entry.get("path") not in {str(accepted_path.relative_to(ROOT)), str(inventory_path.relative_to(ROOT))}:
        raise RuntimeError("tracking global barrier entry does not name this case's accepted or inventory file")
    if common.sha(_safe_root_file(tracking_entry["path"])) != tracking_entry["sha256"]:
        raise RuntimeError("tracking global barrier digest changed")

    teacher_path = OLD_BASE / "teacher" / f"{case_id}.json"
    receipt_path = OLD_BASE / "teacher" / f"{case_id}.receipt.json"
    teacher = common.read(teacher_path)
    receipt = common.read(receipt_path)
    if teacher_entry.get("path") not in {str(teacher_path.relative_to(ROOT)), str(receipt_path.relative_to(ROOT))}:
        raise RuntimeError("teacher global barrier entry does not name this case's teacher file or receipt")
    if common.sha(_safe_root_file(teacher_entry["path"])) != teacher_entry["sha256"]:
        raise RuntimeError("teacher global barrier digest changed")
    if (receipt.get("case_id") != case_id or receipt.get("dataset") != case.get("dataset")
            or receipt.get("source_case_sha256") != case["source_case_sha256"]):
        raise RuntimeError("teacher receipt case/source binding differs from the locked roster")
    if (receipt.get("tracking_acceptance_sha256") != common.sha(accepted_path)
            or receipt.get("mask_inventory_sha256") != accepted.get("mask_inventory_sha256")
            or receipt.get("mask_inventory_receipt_sha256") != common.sha(inventory_path)
            or receipt.get("teacher_record_sha256") != common.sha(teacher_path)):
        raise RuntimeError("teacher receipt does not bind the currently accepted tracking, inventory, and teacher record")
    oldprep._validate_teacher_binding(teacher, receipt, request, request_sha)

    generation = receipt.get("generation")
    generation_ok = (
        receipt.get("status") == "complete" and receipt.get("GT_read") is False
        and receipt.get("failure") is None and receipt.get("one_request_one_generation") is True
        and receipt.get("reprompt_count") == 0 and isinstance(generation, Mapping)
        and generation.get("max_new_tokens") == MAX_NEW_TOKENS
        and generation.get("generation_call_count") == 1
        and isinstance(generation.get("output_token_count"), int)
        and not isinstance(generation.get("output_token_count"), bool)
        and 0 <= generation["output_token_count"] < MAX_NEW_TOKENS
        and generation.get("output_truncated_at_cap") is False
    )
    parsed = teacher.get("parsed")
    if not isinstance(parsed, Mapping):
        parsed = {}

    mask_dir = Path(accepted["mask_data_dir"]).resolve(strict=True)
    indexed_masks, inventory_ids = _load_mask_inventory(mask_dir)
    if set(indexed_masks) != set(mask_files) or any(indexed_masks[i].resolve() != mask_files[i].resolve() for i in mask_files):
        raise RuntimeError("official generated mask index differs from the sealed dense mask inventory")
    candidate_id, id_syntax_valid, id_in_inventory = _valid_candidate_id(parsed, inventory_ids)
    selected_candidate = candidate_id if generation_ok and id_in_inventory else None

    source_path = _safe_root_file(case["source_case_path"])
    if common.sha(source_path) != case["source_case_sha256"]:
        raise RuntimeError("frozen source case SHA differs from the roster")
    # The existing guarded loader validates the sidecar and loads CPU metadata,
    # Frozen boxes, and the locked frame grid. No RGB is indexed or decoded.
    source_case = load_source_case(source_path)
    frame_ids = _int_list(source_case["frame_ids"], name="source frame_ids")
    frozen_boxes = _float_list(source_case["native_boxes"], width=4, name="Frozen native_boxes")
    if len(frame_ids) != len(frozen_boxes):
        raise RuntimeError("Frozen source frame grid and boxes do not align")
    request_physical = [int(row["physical_frame_id"]) for row in request["frame_map"]]
    request_local = {physical: index for index, physical in enumerate(request_physical)}
    if any(physical not in request_local for physical in frame_ids):
        raise RuntimeError("Frozen student source grid is not contained in the canonical teacher frame map")
    for center in centers:
        if frame_ids[center["source_grid_position"]] != center["center_physical_frame_id"]:
            raise RuntimeError("fixed center source-grid position does not match its physical frame ID")
    input_row = source_case["input"]
    width, height = int(input_row["width"]), int(input_row["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError("Frozen source dimensions must be positive")

    direct_frames: list[dict[str, Any]] = []
    masks_by_local: dict[int, Any] = {}
    if selected_candidate is not None:
        sparse_local = {local for center in centers for local in center["local_frame_indices"]}
        direct_frames, masks_by_local, _ = _decode_selected_id(
            candidate_id=selected_candidate,
            request=request,
            mask_files=mask_files,
            sparse_local_indices=sparse_local,
            expected_width=width,
            expected_height=height,
        )
    else:
        direct_frames = [{
            "local_frame_index": i,
            "physical_frame_id": int(clock["physical_frame_id"]),
            "xyxy_inclusive_pixels": None,
            "xyxy_normalized_continuous": None,
            "cxcywh_normalized_continuous": None,
            "mask_area_px": 0,
            "present": False,
        } for i, clock in enumerate(request["frame_map"])]

    boxes_by_local = {int(row["local_frame_index"]): row.get("xyxy_normalized_continuous") for row in direct_frames}
    qa_mask_present = bool(selected_candidate is not None and _qa_mask_presence(request["frame_map"], boxes_by_local, qa_interval))
    all_centers_present = bool(selected_candidate is not None and qa_mask_present)
    raw_anchors: list[dict[str, Any]] = []
    supported_anchors: list[dict[str, Any]] = []
    center_audit: list[dict[str, Any]] = []
    if all_centers_present:
        frames_by_local = {int(row["local_frame_index"]): row for row in direct_frames}
        raw_anchors, supported_anchors, center_audit = _make_anchor_sets(
            centers, frames_by_local, masks_by_local, selected_candidate
        )
    else:
        for center in centers:
            center_audit.append({
                "slot": int(center["slot"]),
                "position": int(center["source_grid_position"]),
                "physical_frame_id": int(center["center_physical_frame_id"]),
                "local_frame_index": int(center["center_local_frame_index"]),
                "local_frame_indices": list(center["local_frame_indices"]),
                "center_mask_present": False,
                "presence_fraction": 0.0,
                "valid_adjacent_pair_count": 0,
                "mean_adjacent_mask_iou": 0.0,
                "legacy_weight": 0.0,
                "raw_anchor_created": False,
                "presence_supported_anchor_created": False,
            })

    # A candidate can exist in the inventory yet have no nonempty QA/center
    # geometry. In that case WHO is recorded diagnostically, but is not admitted.
    raw_candidate_admitted = bool(selected_candidate is not None and qa_mask_present and raw_anchors)
    presence_candidate_admitted = bool(raw_candidate_admitted and supported_anchors)
    raw_reason = (
        "admitted_id_with_nonempty_fixed_center_masks" if raw_candidate_admitted else
        "teacher_generation_incomplete_or_capped" if not generation_ok else
        "candidate_id_invalid" if not id_syntax_valid else
        "candidate_id_absent_from_mask_inventory" if not id_in_inventory else
        "selected_id_absent_in_QA_interval" if not qa_mask_present else
        "no_selected_mask_at_fixed_centers"
    )
    supported_reason = (
        "admitted_with_center_mask_and_adjacent_present_pair" if presence_candidate_admitted else
        "raw_decoupled_not_admitted" if not raw_candidate_admitted else
        "no_fixed_center_has_adjacent_present_pair"
    )

    old_ref_row = old_reference.get(case_id)
    if not isinstance(old_ref_row, Mapping) or old_ref_row.get("source_case_sha256") != case["source_case_sha256"]:
        raise RuntimeError("old strict reference row is missing or no longer binds the source case")
    strict_candidate = old_ref_row.get("candidate_id")
    strict_admitted = bool(old_ref_row.get("no_update") is False and old_ref_row.get("anchors") and
                           isinstance(strict_candidate, int) and not isinstance(strict_candidate, bool))
    if strict_admitted and (not generation_ok or strict_candidate != candidate_id or strict_candidate not in inventory_ids):
        raise RuntimeError("old strict valid reference no longer matches the uncapped teacher WHO selection")

    teacher_boxes_by_physical: dict[int, list[float] | None] = {}
    if selected_candidate is not None:
        for row in direct_frames:
            teacher_boxes_by_physical[int(row["physical_frame_id"])] = row.get("xyxy_normalized_continuous")
    strict_boxes_by_physical: dict[int, list[float] | None] = {}
    if strict_admitted:
        if strict_candidate == selected_candidate:
            strict_boxes_by_physical = teacher_boxes_by_physical
        else:
            strict_frames, _, _ = _decode_selected_id(
                candidate_id=int(strict_candidate), request=request, mask_files=mask_files,
                sparse_local_indices=set(), expected_width=width, expected_height=height,
            )
            strict_boxes_by_physical = {int(row["physical_frame_id"]): row.get("xyxy_normalized_continuous") for row in strict_frames}

    strict_geometry = _geometry_arm(
        frame_ids=frame_ids, frozen_boxes=frozen_boxes, teacher_boxes_by_physical=strict_boxes_by_physical,
        candidate_id=int(strict_candidate) if strict_admitted else None, admitted=strict_admitted,
        qa_interval=qa_interval, abstain_reason="strict_original_invalid_or_no_anchor",
    )
    raw_geometry = _geometry_arm(
        frame_ids=frame_ids, frozen_boxes=frozen_boxes, teacher_boxes_by_physical=teacher_boxes_by_physical,
        candidate_id=selected_candidate, admitted=raw_candidate_admitted,
        qa_interval=qa_interval, abstain_reason=raw_reason,
    )

    parsed_status = parsed.get("status")
    native_interval = {
        "start_time_s": parsed.get("start_time_s"),
        "end_time_s": parsed.get("end_time_s"),
        "projected_interval": copy.deepcopy(teacher.get("projected_interval")),
    }
    reference_record = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "cohort": case.get("cohort"),
        "source_case_sha256": case["source_case_sha256"],
        "canonical_pixel_sha256": request["canonical_pixel_sha256"],
        "frame_ids_sha256": _sha_bytes(_canonical_bytes(frame_ids)),
        "request_sha256": request_sha,
        "qa_student_sha256": case["qa_student_sha256"],
        "qa_interval_frame_bounds_half_open": [qa_interval[0], qa_interval[1]],
        "when": {
            "parsed_status": parsed_status,
            "start_time_s": parsed.get("start_time_s"),
            "end_time_s": parsed.get("end_time_s"),
            "native_interval": native_interval,
            "qa_interval_frame_bounds_half_open": [qa_interval[0], qa_interval[1]],
            "time_validity_used_for_who_admission": False,
        },
        "who": {
            "candidate_id": candidate_id,
            "id_syntax_valid": bool(id_syntax_valid),
            "id_in_mask_inventory": bool(id_in_inventory),
            "generation_complete_uncapped": bool(generation_ok),
            "nonempty_mask_in_qa_interval": bool(qa_mask_present),
            "fixed_center_nonempty_mask_count": len(raw_anchors),
            "semantic_identity_correctness": "unknown_not_verified",
            "center_audit": center_audit,
        },
        "strict_original": copy.deepcopy(dict(old_ref_row)),
        "raw_decoupled": {
            "candidate_id": selected_candidate,
            "selection_status": parsed_status,
            "eligible_for_geometry": raw_candidate_admitted,
            "anchors": raw_anchors,
            "no_update": not bool(raw_anchors),
            "reason": raw_reason,
        },
        "presence_supported": {
            "candidate_id": selected_candidate if presence_candidate_admitted else None,
            "selection_status": parsed_status,
            "eligible_for_update": presence_candidate_admitted,
            "anchors": supported_anchors,
            "no_update": not bool(supported_anchors),
            "reason": supported_reason,
            "support_rule": "fixed_center_mask_and_at_least_one_adjacent_present_same_id_pair_within_original_center_plus_minus_2_window",
        },
    }
    geometry_record = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "source_case_sha256": case["source_case_sha256"],
        "canonical_pixel_sha256": request["canonical_pixel_sha256"],
        "request_sha256": request_sha,
        "qa_student_sha256": case["qa_student_sha256"],
        "frame_ids_physical": frame_ids,
        "qa_interval_frame_bounds_half_open": [qa_interval[0], qa_interval[1]],
        "strict_original": strict_geometry,
        "raw_decoupled": raw_geometry,
    }
    return reference_record, geometry_record, {
        "teacher_status": parsed_status,
        "who_syntax_valid": bool(id_syntax_valid),
        "who_in_inventory": bool(id_in_inventory),
        "who_qa_mask_present": bool(qa_mask_present),
        "raw_anchor_count": len(raw_anchors),
        "presence_supported_anchor_count": len(supported_anchors),
        "strict_geometry_mask_frame_count": sum(strict_geometry["teacher_mask_present"]),
        "raw_geometry_mask_frame_count": sum(raw_geometry["teacher_mask_present"]),
        "raw_geometry_fallback_frame_count": sum(raw_geometry["fallback_used"]),
    }


def prepare_references() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the finite eight-source CPU preparation after the locked preflight."""
    roster_rows, design, step_runtime, tracking_runtime, teacher_runtime = _preflight_locked_inputs()
    old_roster = common.read(ROSTER_PATH)
    old_reference_doc = common.read(OLD_REFERENCE_PATH)
    if old_reference_doc.get("status") != "locked" or old_reference_doc.get("GT_read") is not False:
        raise RuntimeError("old strict reference lock must remain locked and GT-free")
    old_reference_rows = old_reference_doc.get("records")
    if not isinstance(old_reference_rows, list) or len(old_reference_rows) != EXPECTED_CASES:
        raise RuntimeError("old strict reference lock must contain eight rows")
    old_reference_by_case = {row.get("case_id"): row for row in old_reference_rows if isinstance(row, Mapping)}
    if set(old_reference_by_case) != {row["case_id"] for row in roster_rows}:
        raise RuntimeError("old strict reference case set differs from the locked roster")

    case_ids = {str(row["case_id"]) for row in roster_rows}
    tracking_barrier = common.read(TRACKING_BARRIER_PATH)
    teacher_barrier = common.read(TEACHER_BARRIER_PATH)
    tracking_entries = {item["case_id"]: item for item in tracking_barrier["files"]}
    teacher_entries = {item["case_id"]: item for item in teacher_barrier["files"]}
    if set(tracking_entries) != case_ids or set(teacher_entries) != case_ids:
        raise RuntimeError("tracking or teacher barriers do not cover the exact eight-source roster")

    # The existing audit guard is installed only after all immutable lock checks.
    sys.addaudithook(common.guard)
    reference_records: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case in roster_rows:
        reference, geometry, diagnostic = _prepare_one(
            case,
            tracking_entry=tracking_entries[str(case["case_id"])],
            teacher_entry=teacher_entries[str(case["case_id"])],
            old_reference=old_reference_by_case,
        )
        reference_records.append(reference)
        geometry_records.append(geometry)
        diagnostics.append(diagnostic)

    script_sha = common.sha(Path(__file__))
    source_bindings = {
        "scripts/prepare_stvg_instance_decoupled_references_v1.py": script_sha,
        "scripts/prepare_stvg_native_instance_references_v1.py": common.sha(ROOT / "scripts/prepare_stvg_native_instance_references_v1.py"),
        "scripts/stvg_native_instance_common_v1.py": common.sha(ROOT / "scripts/stvg_native_instance_common_v1.py"),
        "scripts/stvg_r1_native_teacher_v1.py": common.sha(ROOT / "scripts/stvg_r1_native_teacher_v1.py"),
        "scripts/stvg_sparse_r1_prompt_v1.py": common.sha(ROOT / "scripts/stvg_sparse_r1_prompt_v1.py"),
    }
    bindings = {
        "design_lock_sha256": common.sha(DESIGN_LOCK_PATH),
        "step_diagnostic_runtime_sha256": common.sha(STEP_RUNTIME_PATH),
        "protocol_sha256": common.sha(PROTOCOL_PATH),
        "roster_sha256": common.sha(ROSTER_PATH),
        "strict_reference_lock_sha256": common.sha(OLD_REFERENCE_PATH),
        "tracking_barrier_sha256": common.sha(TRACKING_BARRIER_PATH),
        "teacher_barrier_sha256": common.sha(TEACHER_BARRIER_PATH),
        "tracking_runtime_sha256": common.sha(TRACKING_RUNTIME_PATH),
        "teacher_runtime_sha256": common.sha(TEACHER_RUNTIME_PATH),
    }
    reference_lock = {
        "schema": "stvg-instance-decoupled-reference-lock-v1",
        "status": "locked",
        "count": len(reference_records),
        "GT_read": False,
        "historically_exposed": True,
        "semantic_identity_correctness_claimed": False,
        "bindings": bindings,
        "source_bindings": source_bindings,
        "coordinate_convention": "inclusive_pixels_to_exclusive_normalized_xyxy_then_cxcywh",
        "legacy_weight_definition": "unchanged old presence_fraction_times_mean_valid_adjacent_mask_iou",
        "presence_support_definition": "center_mask_and_at_least_one_adjacent_same_id_present_pair_in_original_clipped_center_plus_minus_2_window",
        "records": reference_records,
    }
    geometry_readout = {
        "schema": "stvg-instance-teacher-geometry-only-readout-v1",
        "status": "complete",
        "count": len(geometry_records),
        "GT_read": False,
        "semantic_identity_correctness_claimed": False,
        "geometry_grid": "all_locked_student_physical_frame_ids; QA interval only controls final evaluation interval and WHO admission",
        "missing_mask_policy": "exact_original_F_box_per_grid_frame; no interpolation or nearest-frame replacement",
        "bindings": bindings,
        "source_bindings": source_bindings,
        "records": geometry_records,
    }
    status_counts = Counter(str(row["teacher_status"]) for row in diagnostics)
    readback = {
        "schema": "stvg-instance-decoupled-reference-preparation-readback-v1",
        "status": "complete",
        "count": len(reference_records),
        "GT_read": False,
        "model_loaded": False,
        "video_decoded": False,
        "annotation_accessed": False,
        "raw_teacher_text_exported": False,
        "semantic_identity_correctness_claimed": False,
        "teacher_parsed_status_counts": dict(sorted(status_counts.items())),
        "raw_decoupled_anchor_count": sum(row["raw_anchor_count"] for row in diagnostics),
        "presence_supported_anchor_count": sum(row["presence_supported_anchor_count"] for row in diagnostics),
        "raw_decoupled_geometry_mask_frame_count": sum(row["raw_geometry_mask_frame_count"] for row in diagnostics),
        "raw_decoupled_geometry_fallback_frame_count": sum(row["raw_geometry_fallback_frame_count"] for row in diagnostics),
        "strict_geometry_mask_frame_count": sum(row["strict_geometry_mask_frame_count"] for row in diagnostics),
        "source_bindings": source_bindings,
        "bindings": bindings,
    }

    readback["reference_lock_sha256"] = _formatted_json_sha(reference_lock)
    readback["teacher_geometry_readout_sha256"] = _formatted_json_sha(geometry_readout)
    common.write(REFERENCE_PATH, reference_lock)
    common.write(GEOMETRY_PATH, geometry_readout)
    common.write(READBACK_PATH, readback)
    return reference_lock, geometry_readout, readback


def run_cpu_contracts() -> dict[str, Any]:
    """Run synthetic CPU contracts without project artifacts or model imports."""
    import numpy as np

    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    # WHO parser ID extraction is independent of the old WHEN status.
    parsed = {"status": "out_of_clock_interval", "candidate_id": 7}
    candidate, syntax, in_inventory = _valid_candidate_id(parsed, {7})
    check(candidate == 7 and syntax and in_inventory, "out-of-clock WHO ID was incorrectly rejected")
    for bad in (True, 0, -1, "7", None):
        candidate, syntax, in_inventory = _valid_candidate_id({"candidate_id": bad}, {7})
        check(candidate is None and not syntax and not in_inventory, "invalid WHO ID passed syntax validation")
    candidate, syntax, in_inventory = _valid_candidate_id({"candidate_id": 9}, {7})
    check(candidate == 9 and syntax and not in_inventory, "unknown inventory ID was not separated from syntax validity")

    empty = np.zeros((8, 8), dtype=np.uint8)
    m0 = np.zeros((8, 8), dtype=np.uint8); m0[1:4, 1:4] = 1
    m1 = np.zeros((8, 8), dtype=np.uint8); m1[1:4, 2:5] = 1
    window = [0, 1, 2, 3, 4]
    center = {"slot": 0, "source_grid_position": 2, "center_physical_frame_id": 12,
              "center_local_frame_index": 2, "local_frame_indices": window}
    boxes = {
        0: {"xyxy_normalized_continuous": [0.1, 0.1, 0.3, 0.3], "cxcywh_normalized_continuous": [0.2, 0.2, 0.2, 0.2]},
        1: {"xyxy_normalized_continuous": [0.2, 0.1, 0.4, 0.3], "cxcywh_normalized_continuous": [0.3, 0.2, 0.2, 0.2]},
        2: {"xyxy_normalized_continuous": [0.3, 0.1, 0.5, 0.3], "cxcywh_normalized_continuous": [0.4, 0.2, 0.2, 0.2]},
        3: {"xyxy_normalized_continuous": None, "cxcywh_normalized_continuous": None},
        4: {"xyxy_normalized_continuous": None, "cxcywh_normalized_continuous": None},
    }
    masks = {0: m0, 1: m1, 2: m0}
    raw, supported, _ = _make_anchor_sets([center], boxes, masks, 7)
    check(len(raw) == 1 and len(supported) == 1, "center plus adjacent-pair anchor should pass both arms")
    check(raw[0]["box_cxcywh"] == boxes[2]["cxcywh_normalized_continuous"], "anchor cxcywh differs from center mask")
    check(raw[0]["reliability"]["presence_fraction"] == 0.6, "presence fraction was not computed over the full fixed window")
    raw, supported, _ = _make_anchor_sets([center], boxes, {2: m0}, 7)
    check(len(raw) == 1 and len(supported) == 0, "center-only mask must be raw-eligible but structurally unsupported")
    raw, supported, _ = _make_anchor_sets([center], boxes, {0: empty, 2: m0}, 7)
    check(len(raw) == 1 and len(supported) == 0, "empty masks must not count as adjacent presence")
    raw, supported, _ = _make_anchor_sets([center], boxes, {0: m0, 1: m1, 2: empty}, 7)
    check(len(raw) == 0 and len(supported) == 0, "empty fixed center must not create any anchor")

    frame_map = [
        {"local_frame_index": i, "physical_frame_id": 10 + i}
        for i in range(5)
    ]
    check(_qa_mask_presence(frame_map, {0: [0, 0, 1, 1], 1: None, 2: None, 3: None, 4: None}, (10, 11)),
          "QA interval must include its start boundary")
    check(not _qa_mask_presence(frame_map, {0: [0, 0, 1, 1]}, (11, 15)),
          "half-open QA interval must exclude the frame at its end boundary")

    student_ids = [10, 12, 14]
    frozen = [[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3], [0.3, 0.3, 0.4, 0.4]]
    teacher = {10: [0.0, 0.0, 0.5, 0.5], 12: None, 14: [0.4, 0.4, 0.8, 0.8]}
    geom = _geometry_arm(frame_ids=student_ids, frozen_boxes=frozen, teacher_boxes_by_physical=teacher,
                         candidate_id=7, admitted=True, qa_interval=(11, 13), abstain_reason="not_admitted")
    check(geom["boxes_xyxy_normalized"] == [teacher[10], frozen[1], teacher[14]],
          "P2 geometry must map full-grid masks and use exact per-frame F fallback")
    check(geom["teacher_mask_missing"] == [False, True, False] and geom["fallback_used"] == [False, True, False],
          "P2 missing-mask markers or fallback mask is wrong")
    check(geom["interval_frame_bounds_half_open"] == [11, 13], "P2 must keep the exact QA interval")
    invalid = _geometry_arm(frame_ids=student_ids, frozen_boxes=frozen, teacher_boxes_by_physical=teacher,
                            candidate_id=None, admitted=False, qa_interval=(11, 13), abstain_reason="strict_invalid")
    check(invalid["boxes_xyxy_normalized"] == frozen and all(invalid["fallback_used"]),
          "invalid strict reference must remain all-F")
    check(not geom["interpolation_used"] and not geom["nearest_frame_replacement_used"],
          "P2 must not interpolate or replace from a neighboring frame")
    return {"schema": "stvg-instance-decoupled-reference-cpu-contracts-v1", "status": "pass", "GT_read": False,
            "model_loaded": False, "video_decoded": False, "check_count": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run synthetic CPU-only contracts")
    parser.add_argument("--prepare-after-lock", action="store_true", help="run the finite eight-source CPU preparation after root locks this script")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(run_cpu_contracts(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.prepare_after_lock:
        _, _, readback = prepare_references()
        print(json.dumps({k: v for k, v in readback.items() if k not in {"source_bindings", "bindings"}}, ensure_ascii=False, sort_keys=True))
        return 0
    parser.error("choose --self-test or --prepare-after-lock explicitly")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
