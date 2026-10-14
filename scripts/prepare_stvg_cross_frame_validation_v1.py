#!/usr/bin/env python3
"""Freeze disjoint cross-frame validation references from sealed mask geometry.

The 22 training anchors are copied from the already locked
presence-supported reference list. Validation positions are selected from the
locked Student physical-frame grid and QA interval before visibility is
examined. This module is CPU-only and does not open annotations, media, model
checkpoints, or serialized student tensors.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_cross_frame_validation_v1"
IDENTITY = ROOT / "artifacts/stvg_instance_identity_update_v1"
NATIVE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
PROTOCOL = ROOT / "protocols/stvg_cross_frame_validation_v1.md"
CURRENT = ROOT / "methods/CURRENT_METHOD.json"

DESIGN_PATH = BASE / "DESIGN_LOCK.json"
OLD_REFERENCE_PATH = IDENTITY / "REFERENCE_LOCK.json"
OLD_GEOMETRY_PATH = IDENTITY / "TEACHER_GEOMETRY_ONLY_READOUT.json"
OLD_PREPARATION_PATH = IDENTITY / "REFERENCE_PREPARATION_READBACK.json"
OLD_GEOMETRY_AUDIT_PATH = IDENTITY / "ROOT_INDEPENDENT_GEOMETRY_READBACK.json"
OLD_GLOBAL_BARRIER_PATH = IDENTITY / "GLOBAL_PREDICTION_BARRIER.json"
OLD_DESIGN_PATH = IDENTITY / "DESIGN_LOCK.json"
OLD_STEP_RUNTIME_PATH = IDENTITY / "STEP_DIAGNOSTIC_RUNTIME.json"
OLD_P1_LOCK_PATH = IDENTITY / "P1_UPDATE_LOCK.json"
ROSTER_PATH = NATIVE / "ROSTER_LOCK.json"
NATIVE_REFERENCE_PATH = NATIVE / "REFERENCE_LOCK.json"
TRACKING_BARRIER_PATH = NATIVE / "TRACKING_GLOBAL_BARRIER.json"
TEACHER_BARRIER_PATH = NATIVE / "TEACHER_GLOBAL_BARRIER.json"

CANDIDATE_LOCK_PATH = BASE / "CANDIDATE_POSITION_LOCK.json"
REFERENCE_LOCK_PATH = BASE / "REFERENCE_LOCK.json"
READBACK_PATH = BASE / "PREPARATION_READBACK.json"
CONTRACTS_PATH = BASE / "CPU_CONTRACTS.json"
CAPTURE_PATH = BASE / "PREPARATION_ENGINEERING_CAPTURE.json"

EXPECTED_CASES = 8
TRAINING_ANCHORS = 22
MAX_VALIDATION_POSITIONS = 8
MIN_VISIBLE_VALIDATION = 2
SHA256_LEN = 64


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha_json(value: Any) -> str:
    return _sha_bytes(_json_bytes(value))


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == SHA256_LEN
            and all(char in "0123456789abcdef" for char in value))


def _write_immutable(path: Path, value: Any) -> str:
    """Create a prepared artifact once; permit only byte-identical recovery."""
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(f"immutable output differs; refusing overwrite: {path.name}")
        return _sha_bytes(existing)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"stale temporary output requires review: {temp.name}")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return _sha_bytes(data)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _strict_interval(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("QA interval must be two half-open physical-frame bounds")
    start, end = (_strict_int(value[0], "QA start"), _strict_int(value[1], "QA end"))
    if start < 0 or end <= start:
        raise ValueError("QA interval must be nonempty and nonnegative")
    return start, end


def _frame_grid(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("source grid must be a nonempty physical-frame list")
    result = [_strict_int(item, "physical frame ID") for item in value]
    if result[0] < 0 or any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("source-grid physical IDs must be nonnegative and strictly increasing")
    return result


def _quantile_indices(candidate_count: int, limit: int = MAX_VALIDATION_POSITIONS) -> list[int]:
    """Return deterministic integer endpoint quantiles over ordered candidates."""
    n = _strict_int(candidate_count, "candidate_count")
    cap = _strict_int(limit, "limit")
    if n < 0 or cap < 1:
        raise ValueError("candidate count must be nonnegative and limit positive")
    k = min(n, cap)
    if k == 0:
        return []
    if k == 1:
        return [0]
    # Integer half-up rounding of j*(n-1)/(k-1); endpoints are exact.
    selected = [((2 * j * (n - 1) + (k - 1)) // (2 * (k - 1))) for j in range(k)]
    if selected[0] != 0 or selected[-1] != n - 1 or len(set(selected)) != k:
        raise AssertionError("endpoint quantiles must be unique and include both endpoints")
    return selected


def _candidate_row(
    case_id: str,
    dataset: str,
    source_sha: str,
    pixel_sha: str,
    qa_interval: Sequence[int],
    frame_ids: Sequence[int],
    training_anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select candidate positions using clocks only; this function accepts no masks."""
    start, end = _strict_interval(qa_interval)
    ids = _frame_grid(list(frame_ids))
    train_ids: list[int] = []
    for anchor in training_anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError("training anchors must be mappings")
        frame_id = _strict_int(anchor.get("physical_frame_id"), "training physical_frame_id")
        if frame_id not in ids or not start <= frame_id < end:
            raise ValueError("training anchor must be on the source grid and inside the QA interval")
        train_ids.append(frame_id)
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("training anchors must not repeat physical frames")
    train_set = set(train_ids)
    eligible_positions = [i for i, frame_id in enumerate(ids)
                          if start <= frame_id < end and frame_id not in train_set]
    selected_indices = _quantile_indices(len(eligible_positions))
    positions = [eligible_positions[index] for index in selected_indices]
    return {
        "case_id": case_id,
        "dataset": dataset,
        "source_case_sha256": source_sha,
        "canonical_pixel_sha256": pixel_sha,
        "qa_interval_frame_bounds_half_open": [start, end],
        "source_grid_sha256": _sha_json(ids),
        "source_grid_count": len(ids),
        "training_physical_frame_ids": train_ids,
        "training_anchor_count": len(training_anchors),
        "eligible_candidate_positions_source_grid": eligible_positions,
        "eligible_candidate_physical_frame_ids": [ids[position] for position in eligible_positions],
        "selected_positions_source_grid": positions,
        "selected_physical_frame_ids": [ids[position] for position in positions],
        "candidate_count": len(positions),
        "selection_stage": "locked_before_mask_visibility_check",
    }


def _xyxy_to_cxcywh(box: Any) -> tuple[list[float], list[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("visible mask bbox must be a four-value normalized xyxy box")
    vals = [float(value) for value in box]
    if not all(math.isfinite(value) for value in vals):
        raise ValueError("visible mask bbox must be finite")
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("visible mask bbox must be nonempty normalized xyxy geometry")
    return vals, [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]


def _visible_anchor(position: int, physical_frame_id: int, bbox: Any, candidate_id: Any) -> dict[str, Any]:
    xyxy, cxcywh = _xyxy_to_cxcywh(bbox)
    return {
        "position": int(position),
        "physical_frame_id": int(physical_frame_id),
        "box_cxcywh": cxcywh,
        "box": cxcywh.copy(),
        "box_xyxy_normalized": xyxy,
        "weight": 1.0,
        "candidate_id": candidate_id,
        "source": "sealed_selected_id_sam2_mask_bbox",
        "weighting": "uniform_validation",
        "semantic_identity_correctness": "unknown_not_verified",
    }


def _validate_reference_sources() -> dict[str, Any]:
    paths = {
        "design": DESIGN_PATH,
        "protocol": PROTOCOL,
        "current": CURRENT,
        "old_reference": OLD_REFERENCE_PATH,
        "old_geometry": OLD_GEOMETRY_PATH,
        "old_preparation_readback": OLD_PREPARATION_PATH,
        "old_geometry_independent_readback": OLD_GEOMETRY_AUDIT_PATH,
        "old_global_prediction_barrier": OLD_GLOBAL_BARRIER_PATH,
        "old_design": OLD_DESIGN_PATH,
        "old_step_runtime": OLD_STEP_RUNTIME_PATH,
        "old_p1_lock": OLD_P1_LOCK_PATH,
        "source_roster": ROSTER_PATH,
        "native_reference": NATIVE_REFERENCE_PATH,
        "native_tracking_runtime": NATIVE / "TRACKING_RUNTIME.json",
        "native_teacher_runtime": NATIVE / "TEACHER_RUNTIME.json",
        "tracking_barrier": TRACKING_BARRIER_PATH,
        "teacher_barrier": TEACHER_BARRIER_PATH,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required locked input is missing: {label}")
    digests = {str(path.relative_to(ROOT)): _sha(path) for path in paths.values()}
    script_path = str(Path(__file__).resolve().relative_to(ROOT))
    digests[script_path] = _sha(Path(__file__))

    design = _read_json(DESIGN_PATH)
    if (design.get("status") != "locked" or design.get("count") != EXPECTED_CASES
            or design.get("GT_read") is not False or design.get("selection_uses_GT") is not False
            or design.get("validation_max_positions") != MAX_VALIDATION_POSITIONS
            or design.get("validation_min_visible") != MIN_VISIBLE_VALIDATION
            or design.get("protocol_sha256") != digests[str(PROTOCOL.relative_to(ROOT))]
            or design.get("CURRENT_sha256") != digests[str(CURRENT.relative_to(ROOT))]):
        raise RuntimeError("new DESIGN_LOCK is not the fixed eight-case GT-free protocol lock")

    original = _read_json(OLD_REFERENCE_PATH)
    geometry = _read_json(OLD_GEOMETRY_PATH)
    prep = _read_json(OLD_PREPARATION_PATH)
    geometry_audit = _read_json(OLD_GEOMETRY_AUDIT_PATH)
    barrier = _read_json(OLD_GLOBAL_BARRIER_PATH)
    roster = _read_json(ROSTER_PATH)
    old_design = _read_json(OLD_DESIGN_PATH)
    old_runtime = _read_json(OLD_STEP_RUNTIME_PATH)
    old_p1 = _read_json(OLD_P1_LOCK_PATH)
    native_reference = _read_json(NATIVE_REFERENCE_PATH)
    tracking_barrier = _read_json(TRACKING_BARRIER_PATH)
    teacher_barrier = _read_json(TEACHER_BARRIER_PATH)

    for doc, status, label in ((original, "locked", "original reference"),
                               (geometry, "complete", "teacher geometry"),
                               (prep, "complete", "original preparation readback"),
                               (barrier, "complete", "original prediction barrier"),
                               (geometry_audit, "pass", "independent geometry readback"),
                               (roster, "locked", "native source roster"),
                               (old_design, "locked", "identity design lock"),
                               (old_runtime, "locked", "identity step runtime"),
                               (old_p1, "locked", "identity P1 lock"),
                               (native_reference, "locked", "native original reference"),
                               (tracking_barrier, "complete", "native tracking barrier"),
                               (teacher_barrier, "complete", "native teacher barrier")):
        if doc.get("status") != status or doc.get("GT_read") is not False:
            raise RuntimeError(f"{label} is not in its sealed GT-free state")
    if any(doc.get("count") != EXPECTED_CASES for doc in
           (original, geometry, prep, barrier, roster, old_design, old_runtime, native_reference,
            tracking_barrier, teacher_barrier)):
        raise RuntimeError("a sealed predecessor does not cover exactly eight cases")
    if (prep.get("model_loaded") is not False or prep.get("video_decoded") is not False
            or prep.get("annotation_accessed") is not False
            or prep.get("raw_teacher_text_exported") is not False
            or original.get("historically_exposed") is not True):
        raise RuntimeError("predecessor custody or historical-exposure declaration differs")
    if (geometry_audit.get("reference_sha256") != digests[str(OLD_REFERENCE_PATH.relative_to(ROOT))]
            or geometry_audit.get("geometry_sha256") != digests[str(OLD_GEOMETRY_PATH.relative_to(ROOT))]):
        raise RuntimeError("independent geometry readback is bound to different references")
    if (prep.get("reference_lock_sha256") != digests[str(OLD_REFERENCE_PATH.relative_to(ROOT))]
            or prep.get("teacher_geometry_readout_sha256") != digests[str(OLD_GEOMETRY_PATH.relative_to(ROOT))]):
        raise RuntimeError("original preparation readback does not bind the selected source geometry")

    expected_bindings = {
        "design_lock_sha256": OLD_DESIGN_PATH,
        "step_diagnostic_runtime_sha256": OLD_STEP_RUNTIME_PATH,
        "protocol_sha256": ROOT / "protocols/stvg_instance_identity_update_v1.md",
        "roster_sha256": ROSTER_PATH,
        "strict_reference_lock_sha256": NATIVE_REFERENCE_PATH,
        "tracking_barrier_sha256": TRACKING_BARRIER_PATH,
        "teacher_barrier_sha256": TEACHER_BARRIER_PATH,
        "tracking_runtime_sha256": NATIVE / "TRACKING_RUNTIME.json",
        "teacher_runtime_sha256": NATIVE / "TEACHER_RUNTIME.json",
    }
    recorded_bindings = original.get("bindings")
    if not isinstance(recorded_bindings, Mapping) or set(recorded_bindings) != set(expected_bindings):
        raise RuntimeError("original reference lock binding schema differs")
    for key, path in expected_bindings.items():
        if not path.is_file() or recorded_bindings.get(key) != _sha(path):
            raise RuntimeError(f"original reference binding changed: {key}")
    if geometry.get("bindings") != recorded_bindings or prep.get("bindings") != recorded_bindings:
        raise RuntimeError("original geometry/readback binding maps differ from the original reference lock")

    source_bindings = original.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise RuntimeError("original source code binding map is missing")
    for relative, expected_sha in source_bindings.items():
        if (not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts
                or not _is_sha(expected_sha)):
            raise RuntimeError("original source binding entry is malformed")
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected_sha:
            raise RuntimeError(f"original source code binding changed: {relative}")
    if geometry.get("source_bindings") != source_bindings or prep.get("source_bindings") != source_bindings:
        raise RuntimeError("original geometry/readback source bindings differ")

    if len(roster.get("records", [])) != EXPECTED_CASES:
        raise RuntimeError("native roster case count differs")
    if len(original.get("records", [])) != EXPECTED_CASES or len(geometry.get("records", [])) != EXPECTED_CASES:
        raise RuntimeError("sealed source reference/geometry case count differs")
    ids_ref = [row.get("case_id") for row in original["records"]]
    ids_geo = [row.get("case_id") for row in geometry["records"]]
    ids_roster = [row.get("case_id") for row in roster["records"]]
    if len(set(ids_ref)) != EXPECTED_CASES or set(ids_ref) != set(ids_geo) or set(ids_ref) != set(ids_roster):
        raise RuntimeError("original reference, teacher geometry, and source roster case sets differ")

    return {
        "digests": digests,
        "original": original,
        "geometry": geometry,
        "preparation": prep,
        "geometry_audit": geometry_audit,
        "barrier": barrier,
        "roster": roster,
        "source_bindings": dict(source_bindings),
    }


def _build_candidate_lock(inputs: Mapping[str, Any]) -> dict[str, Any]:
    original = inputs["original"]
    geometry = inputs["geometry"]
    roster = inputs["roster"]
    by_geometry = {row["case_id"]: row for row in geometry["records"]}
    by_roster = {row["case_id"]: row for row in roster["records"]}
    records: list[dict[str, Any]] = []
    total_training = 0
    for source in original["records"]:
        case_id = source["case_id"]
        geometry_row = by_geometry[case_id]
        roster_row = by_roster[case_id]
        training = source.get("presence_supported", {}).get("anchors")
        if not isinstance(training, list):
            raise ValueError("original presence-supported training anchors are missing")
        if (source.get("dataset") != roster_row.get("dataset")
                or source.get("source_case_sha256") != roster_row.get("source_case_sha256")
                or source.get("source_case_sha256") != geometry_row.get("source_case_sha256")
                or source.get("canonical_pixel_sha256") != geometry_row.get("canonical_pixel_sha256")
                or source.get("request_sha256") != roster_row.get("request_sha256")
                or source.get("qa_student_sha256") != roster_row.get("qa_student_sha256")
                or source.get("qa_interval_frame_bounds_half_open") != list(roster_row.get("qa_interval", []))
                or source.get("qa_interval_frame_bounds_half_open") != geometry_row.get("qa_interval_frame_bounds_half_open")):
            raise RuntimeError(f"locked source/QA bindings differ for {case_id}")
        raw = source.get("raw_decoupled", {}).get("anchors", [])
        if not isinstance(raw, list) or training != raw:
            raise RuntimeError("presence-supported training list is not the unchanged original 22-anchor list")
        frame_ids = _frame_grid(geometry_row.get("frame_ids_physical"))
        if geometry_row.get("raw_decoupled", {}).get("frame_ids_physical") != frame_ids:
            raise RuntimeError("geometry and case physical frame grids differ")
        for anchor in training:
            position = _strict_int(anchor.get("position"), "training position")
            frame_id = _strict_int(anchor.get("physical_frame_id"), "training physical frame ID")
            if (not 0 <= position < len(frame_ids) or frame_ids[position] != frame_id
                    or anchor.get("box_cxcywh") is None or anchor.get("reliability") is None):
                raise RuntimeError("original training anchor no longer matches the source physical grid")
        total_training += len(training)
        candidate = _candidate_row(
            case_id=str(case_id), dataset=str(source["dataset"]),
            source_sha=str(source["source_case_sha256"]),
            pixel_sha=str(source["canonical_pixel_sha256"]),
            qa_interval=source["qa_interval_frame_bounds_half_open"],
            frame_ids=frame_ids, training_anchors=training,
        )
        candidate["training_anchor_sha256"] = _sha_json(training)
        records.append(candidate)
    if total_training != TRAINING_ANCHORS:
        raise RuntimeError(f"expected exactly {TRAINING_ANCHORS} preserved training anchors; found {total_training}")
    return {
        "schema": "stvg-cross-frame-candidate-position-lock-v1",
        "status": "locked",
        "count": EXPECTED_CASES,
        "GT_read": False,
        "model_loaded": False,
        "video_decoded": False,
        "annotation_accessed": False,
        "selection_uses_GT": False,
        "selection_uses_mask_visibility": False,
        "candidate_selection": "Within the source physical-frame grid and QA half-open interval, exclude original training physical frames, then choose up to eight integer endpoint quantiles over the ordered remaining grid positions using half-up integer rounding; do not replace missing masks.",
        "training_reference_scope": "exact original presence_supported anchors, copied without modification",
        "max_validation_positions": MAX_VALIDATION_POSITIONS,
        "validation_min_visible": MIN_VISIBLE_VALIDATION,
        "bindings": {
            "protocol_sha256": inputs["digests"][str(PROTOCOL.relative_to(ROOT))],
            "design_lock_sha256": inputs["digests"][str(DESIGN_PATH.relative_to(ROOT))],
            "current_method_sha256": inputs["digests"][str(CURRENT.relative_to(ROOT))],
            "original_reference_lock_sha256": inputs["digests"][str(OLD_REFERENCE_PATH.relative_to(ROOT))],
            "teacher_geometry_readout_sha256": inputs["digests"][str(OLD_GEOMETRY_PATH.relative_to(ROOT))],
            "original_preparation_readback_sha256": inputs["digests"][str(OLD_PREPARATION_PATH.relative_to(ROOT))],
            "source_roster_sha256": inputs["digests"][str(ROSTER_PATH.relative_to(ROOT))],
            "preparation_script_sha256": inputs["digests"][str(Path(__file__).resolve().relative_to(ROOT))],
        },
        "records": records,
    }


def _make_reference_lock(inputs: Mapping[str, Any], candidate_lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect selected-ID mask visibility only after candidate lock exists."""
    old_by_case = {row["case_id"]: row for row in inputs["original"]["records"]}
    geometry_by_case = {row["case_id"]: row for row in inputs["geometry"]["records"]}
    candidate_by_case = {row["case_id"]: row for row in candidate_lock["records"]}
    records: list[dict[str, Any]] = []
    visible_total = unknown_total = candidate_total = fallback_cases = ready_cases = 0
    for candidate_row in candidate_lock["records"]:
        case_id = candidate_row["case_id"]
        source = old_by_case[case_id]
        geometry = geometry_by_case[case_id]["raw_decoupled"]
        frame_ids = _frame_grid(geometry["frame_ids_physical"])
        present = geometry.get("teacher_mask_present")
        boxes = geometry.get("teacher_boxes_xyxy_normalized")
        missing = geometry.get("teacher_mask_missing")
        sources = geometry.get("box_source")
        if not all(isinstance(value, list) and len(value) == len(frame_ids)
                   for value in (present, boxes, missing, sources)):
            raise ValueError("sealed geometry visibility arrays must align exactly to the Student grid")
        selected_id = geometry.get("candidate_id")
        if selected_id != source.get("presence_supported", {}).get("candidate_id"):
            raise RuntimeError(f"sealed selected ID differs from original WHO selection for {case_id}")
        visible_observations: list[dict[str, Any]] = []
        unknown_rows: list[dict[str, Any]] = []
        for position, physical_frame_id in zip(candidate_row["selected_positions_source_grid"],
                                                candidate_row["selected_physical_frame_ids"]):
            if frame_ids[position] != physical_frame_id:
                raise RuntimeError("candidate position/physical ID binding changed")
            is_present = present[position]
            if not isinstance(is_present, bool) or not isinstance(missing[position], bool):
                raise ValueError("sealed mask visibility markers must be booleans")
            if missing[position] == is_present:
                raise ValueError("sealed mask presence and missing markers are inconsistent")
            box = boxes[position]
            box_source = sources[position]
            if is_present:
                if box is None or box_source != "selected_id_mask_bbox":
                    raise ValueError("present validation mask lacks its actual selected-ID bbox")
                visible_observations.append(_visible_anchor(position, physical_frame_id, box, selected_id))
            else:
                if box is not None or box_source != "frozen_source_fallback":
                    raise ValueError("missing validation mask must remain missing, with no replacement bbox")
                unknown_rows.append({"position": position, "physical_frame_id": physical_frame_id,
                                     "status": "selected_id_mask_missing"})
        enough_visible = len(visible_observations) >= MIN_VISIBLE_VALIDATION
        effective = copy.deepcopy(visible_observations) if enough_visible else []
        status = "ready" if enough_visible else "fallback_train_only"
        fallback_reason = None if enough_visible else "fewer_than_two_visible_locked_positions"
        training = copy.deepcopy(source["presence_supported"]["anchors"])
        if _sha_json(training) != candidate_row["training_anchor_sha256"]:
            raise RuntimeError("original training anchor list changed while preparing validation")
        records.append({
            "case_id": case_id,
            "dataset": source["dataset"],
            "source_case_sha256": source["source_case_sha256"],
            "canonical_pixel_sha256": source["canonical_pixel_sha256"],
            "request_sha256": source["request_sha256"],
            "qa_student_sha256": source["qa_student_sha256"],
            "qa_interval_frame_bounds_half_open": copy.deepcopy(source["qa_interval_frame_bounds_half_open"]),
            "source_grid_sha256": candidate_row["source_grid_sha256"],
            "candidate_position_lock_sha256": _sha_json(candidate_lock),
            "training_anchors": training,
            "training_anchor_count": len(training),
            "training_anchor_sha256": _sha_json(training),
            "validation_selection": {
                "policy": "integer_endpoint_quantiles_over_candidate_source_grid",
                "selection_stage": "locked_before_mask_visibility_check",
                "eligible_candidate_count": len(candidate_row["eligible_candidate_positions_source_grid"]),
                "candidate_count": len(candidate_row["selected_positions_source_grid"]),
                "positions_source_grid": copy.deepcopy(candidate_row["selected_positions_source_grid"]),
                "physical_frame_ids": copy.deepcopy(candidate_row["selected_physical_frame_ids"]),
            },
            "validation_candidate_count": len(candidate_row["selected_positions_source_grid"]),
            "validation_visible_count": len(visible_observations),
            "validation_unknown_count": len(unknown_rows),
            "validation_unknown_positions": unknown_rows,
            "validation_visible_observations": visible_observations,
            "validation_anchors": effective,
            "validation_anchor_count": len(effective),
            "validation_uniform_weighting": True,
            "validation_status": status,
            "fallback_reason": fallback_reason,
            "semantic_identity_correctness": "unknown_not_verified",
        })
        candidate_total += len(candidate_row["selected_positions_source_grid"])
        visible_total += len(visible_observations)
        unknown_total += len(unknown_rows)
        fallback_cases += int(not enough_visible)
        ready_cases += int(enough_visible)

    lock = {
        "schema": "stvg-cross-frame-validation-reference-lock-v1",
        "status": "locked",
        "count": EXPECTED_CASES,
        "GT_read": False,
        "model_loaded": False,
        "video_decoded": False,
        "annotation_accessed": False,
        "historically_exposed": True,
        "training_anchor_count": TRAINING_ANCHORS,
        "training_reference": "exact prior presence-supported anchor list, byte-equivalent per anchor after JSON serialization",
        "validation_policy": "fixed non-GT integer endpoint quantiles; missing selected-ID masks are unknown and never replaced",
        "validation_max_positions": MAX_VALIDATION_POSITIONS,
        "validation_min_visible": MIN_VISIBLE_VALIDATION,
        "semantic_identity_correctness_claimed": False,
        "bindings": {
            **copy.deepcopy(dict(candidate_lock["bindings"])),
            "candidate_position_lock_sha256": _sha_json(candidate_lock),
            "original_geometry_independent_readback_sha256": inputs["digests"][str(OLD_GEOMETRY_AUDIT_PATH.relative_to(ROOT))],
            "original_global_prediction_barrier_sha256": inputs["digests"][str(OLD_GLOBAL_BARRIER_PATH.relative_to(ROOT))],
            "identity_design_lock_sha256": inputs["digests"][str(OLD_DESIGN_PATH.relative_to(ROOT))],
            "identity_step_runtime_sha256": inputs["digests"][str(OLD_STEP_RUNTIME_PATH.relative_to(ROOT))],
            "identity_p1_update_lock_sha256": inputs["digests"][str(OLD_P1_LOCK_PATH.relative_to(ROOT))],
            "native_reference_lock_sha256": inputs["digests"][str(NATIVE_REFERENCE_PATH.relative_to(ROOT))],
            "tracking_global_barrier_sha256": inputs["digests"][str(TRACKING_BARRIER_PATH.relative_to(ROOT))],
            "teacher_global_barrier_sha256": inputs["digests"][str(TEACHER_BARRIER_PATH.relative_to(ROOT))],
        },
        "source_bindings": copy.deepcopy(inputs["source_bindings"]),
        "records": records,
    }
    readback = {
        "schema": "stvg-cross-frame-reference-preparation-readback-v1",
        "status": "complete",
        "count": EXPECTED_CASES,
        "GT_read": False,
        "model_loaded": False,
        "video_decoded": False,
        "annotation_accessed": False,
        "selected_mask_values_used_for_candidate_selection": False,
        "candidate_positions_locked_before_mask_visibility_check": True,
        "training_anchor_count": TRAINING_ANCHORS,
        "validation_candidate_count": candidate_total,
        "validation_visible_mask_count": visible_total,
        "validation_unknown_mask_count": unknown_total,
        "validation_ready_case_count": ready_cases,
        "validation_fallback_train_only_case_count": fallback_cases,
        "semantic_identity_correctness_claimed": False,
        "bindings": copy.deepcopy(lock["bindings"]),
        "case_counts": [{
            "case_id": row["case_id"], "dataset": row["dataset"],
            "training_anchor_count": row["training_anchor_count"],
            "validation_candidate_count": row["validation_candidate_count"],
            "validation_visible_count": row["validation_visible_count"],
            "validation_unknown_count": row["validation_unknown_count"],
            "validation_status": row["validation_status"],
        } for row in records],
    }
    return lock, readback


def _engineering_capture() -> dict[str, Any]:
    """Record the prior broad-search deviation without importing its values."""
    return {
        "schema": "stvg-cross-frame-preparation-engineering-capture-v1",
        "status": "recorded",
        "type": "bounded_read_only_scope_deviation",
        "artifact": "artifacts/stvg_instance_identity_update_v1/PUBLIC_TREE_PACKET.json",
        "content_class": "preexisting_public_development_summary_export; not raw ground-truth annotations",
        "derived_metric_scalars_visible": True,
        "raw_ground_truth_annotations_accessed": False,
        "raw_media_or_masks_accessed": False,
        "private_case_payload_accessed": False,
        "values_used_for_candidate_or_reference_selection": False,
        "preparation_rules_changed": False,
    }


def _cpu_contracts() -> dict[str, Any]:
    checks = 0

    def check(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    check(_quantile_indices(0) == [], "empty candidate set")
    check(_quantile_indices(1) == [0], "single endpoint")
    check(_quantile_indices(8) == list(range(8)), "at-cap candidate set")
    check(_quantile_indices(9) == [0, 1, 2, 3, 5, 6, 7, 8], "integer endpoint quantiles")
    check(_quantile_indices(17) == [0, 2, 5, 7, 9, 11, 14, 16], "integer endpoint quantiles over a longer grid")
    check(_quantile_indices(80) == [0, 11, 23, 34, 45, 56, 68, 79], "maximum validation set is deterministic")

    anchors = [
        {"position": 2, "physical_frame_id": 102, "box_cxcywh": [0.2, 0.3, 0.1, 0.2],
         "weight": 1.0, "reliability": {"presence_fraction": 1.0}},
        {"position": 8, "physical_frame_id": 108, "box_cxcywh": [0.4, 0.5, 0.2, 0.3],
         "weight": 0.8, "reliability": {"presence_fraction": 0.8}},
    ]
    candidate = _candidate_row("synthetic", "hc2", "a" * 64, "b" * 64,
                               [100, 110], list(range(100, 110)), anchors)
    check(candidate["eligible_candidate_positions_source_grid"] == [0, 1, 3, 4, 5, 6, 7, 9],
          "training physical frames are excluded before quantiles")
    check(candidate["selected_positions_source_grid"] == candidate["eligible_candidate_positions_source_grid"],
          "candidate positions do not depend on mask visibility")
    check(candidate["selected_physical_frame_ids"] == [100, 101, 103, 104, 105, 106, 107, 109],
          "selected positions bind to the original physical clock")

    # Missing masks do not alter selection or pull in a replacement position.
    fixed_positions = list(candidate["selected_positions_source_grid"])
    fixed_ids = list(candidate["selected_physical_frame_ids"])
    synthetic_present = {position: position != fixed_positions[2] for position in fixed_positions}
    visible = [position for position in fixed_positions if synthetic_present[position]]
    unknown = [position for position in fixed_positions if not synthetic_present[position]]
    check(fixed_positions == candidate["selected_positions_source_grid"] and len(unknown) == 1,
          "a missing selected mask remains unknown with no replacement")
    check(visible == [p for p in fixed_positions if p != fixed_positions[2]],
          "visibility is evaluated after positions are locked")

    box1 = _visible_anchor(4, 104, [0.1, 0.2, 0.5, 0.8], 9)
    box2 = _visible_anchor(6, 106, [0.2, 0.1, 0.7, 0.6], 9)
    check(box1["box_cxcywh"] == [0.3, 0.5, 0.4, 0.6000000000000001],
          "bbox-to-cxcywh conversion")
    check(box2["weight"] == 1.0 and box2["weighting"] == "uniform_validation",
          "validation uses uniform weight")
    check(box2["semantic_identity_correctness"] == "unknown_not_verified",
          "mask presence does not imply verified identity")
    check(len(visible) >= MIN_VISIBLE_VALIDATION,
          "two real visible masks enable validation")

    observed_one = [box1]
    effective = observed_one if len(observed_one) >= MIN_VISIBLE_VALIDATION else []
    fallback = "ready" if len(observed_one) >= MIN_VISIBLE_VALIDATION else "fallback_train_only"
    check(effective == [] and fallback == "fallback_train_only",
          "fewer than two visible masks explicitly falls back to training-only")
    check(observed_one == [box1], "fallback retains visible diagnostic observations")
    observed_two = [box1, box2]
    effective_two = observed_two if len(observed_two) >= MIN_VISIBLE_VALIDATION else []
    check(effective_two == observed_two, "two visible references are retained")

    copied = copy.deepcopy(anchors)
    check(copied == anchors and _sha_json(copied) == _sha_json(anchors),
          "training anchors are preserved exactly")
    check(len(anchors) == 2 and all("reliability" in anchor for anchor in copied),
          "training reliability metadata is not rewritten")
    return {
        "schema": "stvg-cross-frame-validation-cpu-contracts-v1",
        "status": "pass",
        "GT_read": False,
        "model_loaded": False,
        "video_decoded": False,
        "check_count": checks,
        "checks": [
            "integer_endpoint_quantiles", "qa_half_open_grid_filter", "training_frame_exclusion",
            "mask_independent_candidate_selection", "no_replacement_for_missing_masks",
            "minimum_two_visible_fallback", "uniform_validation_weights",
            "exact_training_anchor_preservation", "identity_correctness_not_inferred",
        ],
    }


def prepare() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create the immutable position lock, then inspect sealed visibility arrays."""
    inputs = _validate_reference_sources()
    # Build and write a mask-independent candidate lock before this module
    # reads any mask visibility or bbox fields from the in-memory geometry rows.
    candidate_lock = _build_candidate_lock(inputs)
    candidate_sha = _write_immutable(CANDIDATE_LOCK_PATH, candidate_lock)
    if candidate_sha != _sha_json(candidate_lock):
        raise AssertionError("candidate-position lock writeback hash mismatch")

    capture = _engineering_capture()
    _write_immutable(CAPTURE_PATH, capture)
    lock, readback = _make_reference_lock(inputs, candidate_lock)
    readback["candidate_position_lock_sha256"] = candidate_sha
    readback["reference_lock_sha256"] = _sha_json(lock)
    readback["engineering_capture_sha256"] = _sha_json(capture)
    contracts = _cpu_contracts()
    _write_immutable(REFERENCE_LOCK_PATH, lock)
    _write_immutable(CONTRACTS_PATH, contracts)
    _write_immutable(READBACK_PATH, readback)
    return lock, readback, contracts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="run synthetic CPU-only contracts")
    group.add_argument("--prepare", action="store_true", help="prepare immutable references from sealed metadata")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(_cpu_contracts(), sort_keys=True))
        return 0
    if args.prepare:
        _, readback, contracts = prepare()
        print(json.dumps({
            "status": readback["status"],
            "training_anchor_count": readback["training_anchor_count"],
            "validation_candidate_count": readback["validation_candidate_count"],
            "validation_visible_mask_count": readback["validation_visible_mask_count"],
            "validation_unknown_mask_count": readback["validation_unknown_mask_count"],
            "validation_ready_case_count": readback["validation_ready_case_count"],
            "validation_fallback_train_only_case_count": readback["validation_fallback_train_only_case_count"],
            "cpu_contract_check_count": contracts["check_count"],
            "reference_lock_sha256": readback["reference_lock_sha256"],
        }, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
