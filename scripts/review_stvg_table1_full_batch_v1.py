#!/usr/bin/env python3
"""CPU-only custody and math readback for FULL Table 1 student batches.

The reader consumes only sealed source/teacher/student artifacts and their
GT-free locks.  It never opens annotations, RGB/video, checkpoints, or starts
an inference model.  ``--global`` walks the fixed 13,785-query global student
seal; ``--student-barrier`` audits one exact same-video batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FULL = ROOT / "artifacts/stvg_final_table1_v1/FULL"
ROSTER_PATH = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
SOURCE_RUNTIME_PATH = FULL / "SOURCE_RUNTIME.json"
TEACHER_RUNTIME_PATH = FULL / "TEACHER_RUNTIME.json"
STUDENT_RUNTIME_PATH = FULL / "STUDENT_RUNTIME.json"
SOURCE_BARRIER_PATH = FULL / "SOURCE_CAPTURE_BARRIER.json"
SOURCE_REL = "scripts/run_stvg_table1_full_source_v1.py"
TEACHER_REL = "scripts/run_stvg_table1_full_teacher_v1.py"
STUDENT_REL = "scripts/run_stvg_table1_full_student_v1.py"
SCIENCE_REL = "scripts/run_stvg_table1_student003.py"
EXPECTED_COUNT = 13785
EXPECTED_DOMAINS = {"hc2": 3482, "vidstg": 10303}
TOL = 2e-6


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_path(value: Any, *, expected_parent: Path | None = None) -> Path:
    if not isinstance(value, str):
        raise ValueError("sealed artifact path must be a string")
    raw = Path(value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("sealed artifact must be a file inside the repository")
    if expected_parent is not None and path.parent != expected_parent.resolve():
        raise ValueError("sealed artifact is outside its expected case directory")
    return path


def _check_digest(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _verify_runtime_pins(runtime: Mapping[str, Any], label: str) -> int:
    pins, protected = runtime.get("pins"), runtime.get("protected")
    if not isinstance(pins, Mapping) or not isinstance(protected, Mapping):
        raise RuntimeError(f"{label} must contain pins and protected maps")
    count = 0
    for relative, expected in {**pins, **protected}.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"{label} contains an unsafe pin path")
        digest = _check_digest(expected, f"{label}.{relative}")
        path = _safe_path(relative)
        if _sha(path) != digest:
            raise RuntimeError(f"{label} pin mismatch: {relative}")
        count += 1
    return count


def _validate_roster(roster: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if (roster.get("schema") != "stvg-final-table1-full-source-roster-v1"
            or roster.get("status") != "locked"
            or roster.get("GT_read") is not False
            or roster.get("selection_uses_GT_or_scores") is not False
            or roster.get("official_query_count") != EXPECTED_COUNT
            or roster.get("domain_counts") != EXPECTED_DOMAINS):
        raise RuntimeError("FULL roster is not the locked fixed GT-free 13,785-query roster")
    rows = roster.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise RuntimeError("FULL roster records are incomplete")
    index: dict[str, dict[str, Any]] = {}
    ordinals: dict[str, set[int]] = {"hc2": set(), "vidstg": set()}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FULL roster row must be an object")
        alias, dataset, ordinal = row.get("probe_alias"), row.get("dataset"), row.get("query_ordinal")
        prefix = "HC" if dataset == "hc2" else "V" if dataset == "vidstg" else None
        expected_length = 7 if prefix == "HC" else 6 if prefix == "V" else -1
        numeric_start = len(prefix) if prefix is not None else 0
        if (not isinstance(alias, str) or prefix is None or not alias.startswith(prefix)
                or len(alias) != expected_length or not alias[numeric_start:].isdigit()
                or isinstance(ordinal, bool) or not isinstance(ordinal, int)
                or alias != f"{prefix}{ordinal:05d}" or alias in index
                or row.get("GT_read") is not False):
            raise ValueError("FULL roster alias/domain identity is malformed")
        for field in ("query_key_sha256", "safe_record_sha256", "parent_sha256",
                      "plan_sha256", "subject_sha256", "video_sha256"):
            _check_digest(row.get(field), f"roster.{field}")
        fps, duration = float(row.get("fps")), float(row.get("duration"))
        bounds = row.get("video_bounds")
        if (not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0
                or not isinstance(bounds, list) or len(bounds) != 2
                or int(bounds[0]) >= int(bounds[1])):
            raise ValueError("FULL roster video clock/bounds are malformed")
        ordinals[dataset].add(ordinal)
        index[alias] = dict(row)
    for dataset, count in EXPECTED_DOMAINS.items():
        if ordinals[dataset] != set(range(count)):
            raise RuntimeError(f"FULL roster ordinal coverage differs for {dataset}")
    if [r["probe_alias"] for r in rows] != list(index):
        raise RuntimeError("FULL roster ordering is not canonical")
    return [dict(r) for r in rows], index


def _authority() -> dict[str, Any]:
    roster = _read_json(ROSTER_PATH)
    source_runtime = _read_json(SOURCE_RUNTIME_PATH)
    teacher_runtime = _read_json(TEACHER_RUNTIME_PATH)
    student_runtime = _read_json(STUDENT_RUNTIME_PATH)
    source_barrier = _read_json(SOURCE_BARRIER_PATH)
    roster_rows, roster_index = _validate_roster(roster)
    roster_sha = _sha(ROSTER_PATH)
    source_runtime_sha = _sha(SOURCE_RUNTIME_PATH)
    teacher_runtime_sha = _sha(TEACHER_RUNTIME_PATH)
    student_runtime_sha = _sha(STUDENT_RUNTIME_PATH)
    source_barrier_sha = _sha(SOURCE_BARRIER_PATH)
    for label, runtime in (("SOURCE_RUNTIME", source_runtime),
                           ("TEACHER_RUNTIME", teacher_runtime),
                           ("STUDENT_RUNTIME", student_runtime)):
        if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
            raise RuntimeError(f"FULL/{label} must be locked and explicitly GT-free")
    source_pins = _verify_runtime_pins(source_runtime, "SOURCE_RUNTIME")
    teacher_pins = _verify_runtime_pins(teacher_runtime, "TEACHER_RUNTIME")
    student_pins = _verify_runtime_pins(student_runtime, "STUDENT_RUNTIME")
    pins = student_runtime.get("pins", {})
    protected = student_runtime.get("protected", {})
    for relative in (STUDENT_REL, SCIENCE_REL):
        expected = pins.get(relative, protected.get(relative))
        if expected != _sha(ROOT / relative):
            raise RuntimeError(f"STUDENT_RUNTIME does not pin the executed worker: {relative}")
    source_bindings = {
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": source_barrier.get("input_manifest_sha256"),
    }
    if any(source_runtime.get(k) != v for k, v in source_bindings.items()):
        raise RuntimeError("SOURCE_RUNTIME roster/manifest binding differs")
    if (source_barrier.get("schema") != "stvg-final-table1-full-source-capture-barrier-v1"
            or source_barrier.get("status") != "sealed"
            or source_barrier.get("count") != EXPECTED_COUNT
            or source_barrier.get("GT_read") is not False
            or source_barrier.get("full_source_roster_sha256") != roster_sha
            or source_barrier.get("source_runtime_sha256") != source_runtime_sha):
        raise RuntimeError("FULL source capture barrier is incomplete or detached")
    source_files = source_barrier.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != set(roster_index):
        raise RuntimeError("FULL source barrier does not bind exactly the fixed roster")
    teacher_bindings = teacher_runtime.get("bindings")
    if not isinstance(teacher_bindings, Mapping):
        raise RuntimeError("TEACHER_RUNTIME must bind fixed source/tracking authorities")
    expected_teacher_bindings = {
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": source_barrier.get("input_manifest_sha256"),
        "source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
    }
    if any(teacher_bindings.get(k) != v for k, v in expected_teacher_bindings.items()):
        raise RuntimeError("TEACHER_RUNTIME source authority binding differs")
    tracking_runtime_sha = _check_digest(teacher_bindings.get("tracking_runtime_sha256"),
                                         "TEACHER_RUNTIME.bindings.tracking_runtime_sha256")
    student_bindings = {
        "full_source_roster_sha256": roster_sha,
        "source_barrier_sha256": source_barrier_sha,
        "source_runtime_sha256": source_runtime_sha,
        "teacher_runtime_sha256": teacher_runtime_sha,
    }
    if any(student_runtime.get(k) != v for k, v in student_bindings.items()):
        raise RuntimeError("STUDENT_RUNTIME fixed FULL authority binding differs")
    return {
        "roster": roster, "rows": roster_rows, "roster_index": roster_index,
        "roster_sha256": roster_sha,
        "source_runtime_sha256": source_runtime_sha,
        "teacher_runtime_sha256": teacher_runtime_sha,
        "student_runtime_sha256": student_runtime_sha,
        "tracking_runtime_sha256": tracking_runtime_sha,
        "source_barrier": source_barrier,
        "source_barrier_sha256": source_barrier_sha,
        "source_files": source_files,
        "pin_counts": {"source": source_pins, "teacher": teacher_pins, "student": student_pins},
    }


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row["video_sha256"]), float(row["fps"]), int(row["width"]), int(row["height"]),
            int(row["frame_count"]), float(row["duration"]), tuple(map(int, row["video_bounds"])))


def _load_teacher_barrier(path_value: Any, expected_sha: str, ctx: Mapping[str, Any],
                          aliases: Sequence[str]) -> dict[str, Any]:
    path = _safe_path(path_value)
    if not path.is_relative_to(FULL.resolve()):
        raise ValueError("teacher barrier must live in the FULL namespace")
    if _sha(path) != expected_sha:
        raise RuntimeError("teacher barrier SHA differs from the student binding")
    barrier = _read_json(path)
    is_global = "global" in path.name.lower()
    expected_schema = ("stvg-final-table1-full-teacher-global-barrier-v1" if is_global
                       else "stvg-final-table1-full-teacher-batch-barrier-v1")
    if (barrier.get("schema") != expected_schema or barrier.get("status") not in {"sealed", "complete"}
            or barrier.get("GT_read") is not False
            or barrier.get("count") != len(barrier.get("files", []))
            or barrier.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or barrier.get("tracking_runtime_sha256") != ctx["tracking_runtime_sha256"]
            or barrier.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or barrier.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or barrier.get("source_capture_barrier_sha256") != ctx["source_barrier_sha256"]
            or barrier.get("input_manifest_sha256") != ctx["source_barrier"].get("input_manifest_sha256")):
        raise RuntimeError("teacher barrier schema/count/authority bindings differ")
    aliases_rows = barrier.get("aliases")
    files = barrier.get("files")
    if (not isinstance(aliases_rows, list) or not isinstance(files, list)
            or aliases_rows != [r.get("alias") for r in files]
            or len(aliases_rows) != len(set(aliases_rows))):
        raise RuntimeError("teacher barrier alias order does not match immutable files")
    roster_order = {alias: i for i, alias in enumerate(ctx["roster_index"])}
    if aliases_rows != sorted(aliases_rows, key=roster_order.__getitem__):
        raise RuntimeError("teacher barrier aliases are not in fixed roster order")
    index: dict[str, dict[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("teacher barrier file row must be an object")
        alias = row.get("alias")
        if alias not in roster_order or alias in index:
            raise RuntimeError("teacher barrier has an unknown or duplicate alias")
        for key in ("path", "receipt_path"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"teacher barrier row lacks {key}: {alias}")
        _check_digest(row.get("sha256"), f"teacher file {alias} SHA")
        _check_digest(row.get("receipt_sha256"), f"teacher receipt {alias} SHA")
        if isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise ValueError(f"teacher file {alias} has invalid byte count")
        index[alias] = dict(row)
    if not set(aliases).issubset(index):
        raise RuntimeError("teacher barrier does not cover the student batch")
    return {"path": path, "sha256": expected_sha, "barrier": barrier, "files": index}


def _verify_teacher_case(alias: str, source_item: Mapping[str, Any], teacher_barrier: Mapping[str, Any],
                         ctx: Mapping[str, Any], torch: Any) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    item = teacher_barrier["files"].get(alias)
    if not isinstance(item, Mapping):
        raise RuntimeError(f"teacher barrier lacks {alias}")
    case_path = _safe_path(item["path"], expected_parent=FULL / "teacher_cases")
    receipt_path = _safe_path(item["receipt_path"], expected_parent=FULL / "teacher_cases")
    if case_path.name != f"{alias}.pt" or receipt_path.name != f"{alias}.receipt.json":
        raise RuntimeError(f"teacher case path/alias mismatch: {alias}")
    if (_sha(case_path) != item["sha256"] or case_path.stat().st_size != item["bytes"]
            or _sha(receipt_path) != item["receipt_sha256"]):
        raise RuntimeError(f"teacher file or receipt bytes differ from barrier: {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema") != "stvg-final-table1-full-teacher-case-receipt-v1"
            or receipt.get("status") != "sealed" or receipt.get("case_id") != alias
            or receipt.get("path") != str(case_path.relative_to(ROOT))
            or receipt.get("sha256") != item["sha256"] or receipt.get("bytes") != item["bytes"]
            or receipt.get("source_sha256") != source_item.get("sha256")
            or receipt.get("source_case_sha256") != source_item.get("source_case_sha256")
            or receipt.get("source_receipt_sha256") != source_item.get("receipt_sha256")
            or receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or receipt.get("source_capture_barrier_sha256") != ctx["source_barrier_sha256"]
            or receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or receipt.get("tracking_runtime_sha256") != ctx["tracking_runtime_sha256"]
            or receipt.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or receipt.get("GT_read") is not False or receipt.get("query_reset") is not True):
        raise RuntimeError(f"teacher receipt is stale or detached: {alias}")
    value = torch.load(case_path, map_location="cpu", weights_only=False)
    if (not isinstance(value, Mapping) or value.get("case_id") != alias
            or value.get("source_sha256") != source_item.get("sha256")
            or value.get("source_case_sha256") != source_item.get("source_case_sha256")
            or value.get("request_sha256") != receipt.get("request_sha256")
            or value.get("tracking_binding_sha256") != receipt.get("tracking_binding_sha256")
            or value.get("GT_read") is not False or value.get("query_reset") is not True):
        raise RuntimeError(f"teacher payload identity differs from its receipt: {alias}")
    if (value.get("request_sha256") != receipt.get("request_sha256")
            or value.get("tracking_binding_sha256") != receipt.get("tracking_binding_sha256")
            or not isinstance(value.get("reference"), Mapping)
            or value["reference"].get("GT_read") is not False):
        raise RuntimeError(f"teacher request/tracking/reference binding mismatch: {alias}")
    who = value.get("who", {}).get("who", {})
    selected = who.get("candidate_id") if who.get("status") == "valid" else None
    if (value["reference"].get("selected_object_id") != selected
            or bool(value["reference"].get("no_update"))
            != (len(value["reference"].get("anchors", [])) == 0)):
        raise RuntimeError(f"teacher WHO selection and Spatial-A reference differ: {alias}")
    actual_temporal = int(receipt.get("actual_temporal_teacher_calls", -1))
    actual_who = int(receipt.get("actual_who_teacher_calls", -1))
    actual_total = int(receipt.get("actual_teacher_calls", -1))
    if (actual_who != 1 or actual_total != actual_temporal + actual_who
            or actual_temporal not in {0, 8}
            or int(value.get("actual_temporal_teacher_calls", -1)) != actual_temporal
            or int(value.get("actual_who_teacher_calls", -1)) != actual_who
            or int(value.get("actual_teacher_calls", -1)) != actual_total):
        raise RuntimeError(f"teacher inference call accounting differs: {alias}")
    return dict(value), receipt, case_path, receipt_path


def _load_source_case(alias: str, roster_row: Mapping[str, Any], source_item: Mapping[str, Any],
                      ctx: Mapping[str, Any], torch: Any, source_module: Any):
    if not isinstance(source_item, Mapping):
        raise ValueError(f"source barrier row is malformed: {alias}")
    path = _safe_path(source_item.get("path"), expected_parent=FULL / "source_cases")
    receipt_path = _safe_path(source_item.get("receipt_path"), expected_parent=FULL / "source_cases")
    if path.name != f"{alias}.pt" or receipt_path.name != f"{alias}.receipt.json":
        raise RuntimeError(f"source case path/alias mismatch: {alias}")
    if (_sha(path) != source_item.get("sha256") or path.stat().st_size != source_item.get("bytes")
            or _sha(receipt_path) != source_item.get("receipt_sha256")):
        raise RuntimeError(f"source file or receipt bytes differ from global barrier: {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema") != "stvg-final-table1-full-source-receipt-v1"
            or receipt.get("case_id") != alias or receipt.get("sha256") != source_item.get("sha256")
            or receipt.get("bytes") != source_item.get("bytes")
            or receipt.get("source_case_sha256") != source_item.get("source_case_sha256")
            or receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or receipt.get("query_key_sha256") != roster_row.get("query_key_sha256")
            or receipt.get("GT_read") is not False or receipt.get("DINO_read_or_call") is not False):
        raise RuntimeError(f"source receipt is stale or detached: {alias}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(value, Mapping) or value.get("schema") != "stvg-final-table1-source-case-v1"
            or value.get("case_id") != alias or value.get("dataset") != roster_row.get("dataset")
            or value.get("query_key_sha256") != roster_row.get("query_key_sha256")
            or value.get("source_case_sha256") != source_item.get("source_case_sha256")
            or source_module.content_sha256(value) != source_item.get("source_case_sha256")
            or value.get("GT_read") is not False or value.get("DINO_read_or_call") is not False):
        raise RuntimeError(f"source payload semantic identity/hash mismatch: {alias}")
    return dict(value), receipt, path, receipt_path


def _tensor_tree_equal(left: Any, right: Any, torch: Any, np: Any, *, atol: float = 0.0) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        try:
            a = torch.as_tensor(left).detach().cpu()
            b = torch.as_tensor(right).detach().cpu()
        except Exception:
            return False
        if a.shape != b.shape:
            return False
        if a.is_floating_point() or b.is_floating_point():
            return bool(torch.allclose(a.double(), b.double(), rtol=0.0, atol=atol, equal_nan=False))
        return torch.equal(a, b)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            a, b = np.asarray(left), np.asarray(right)
        except Exception:
            return False
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            return bool(np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=0.0,
                                    atol=atol, equal_nan=False))
        return bool(np.array_equal(a, b))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_tensor_tree_equal(left[k], right[k], torch, np, atol=atol) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_tensor_tree_equal(a, b, torch, np, atol=atol) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            return False
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return left == right


def _cxcywh_iou(left: Any, right: Any, np: Any) -> Any:
    # The frozen fit computes IoU on float32 decoder/evidence tensors. Mirror
    # that arithmetic independently in NumPy, including its union floor, so a
    # gate decision is not changed by widening tiny boxes or rounding first.
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] != 4:
        raise ValueError("IoU expects aligned cxcywh arrays with final dimension four")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("IoU input contains nonfinite values")
    half = np.float32(0.5)
    ax1, ay1 = a[..., 0] - half * a[..., 2], a[..., 1] - half * a[..., 3]
    ax2, ay2 = a[..., 0] + half * a[..., 2], a[..., 1] + half * a[..., 3]
    bx1, by1 = b[..., 0] - half * b[..., 2], b[..., 1] - half * b[..., 3]
    bx2, by2 = b[..., 0] + half * b[..., 2], b[..., 1] + half * b[..., 3]
    inter = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1)) * np.maximum(
        0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = np.maximum(area_a + area_b - inter, np.float32(1e-7))
    return inter / union


def _xyxy_from_cxcywh(boxes: Any, np: Any) -> Any:
    # Match the student readout's explicit float64 conversion before geometry.
    value = np.asarray(boxes, dtype=np.float64)
    return np.concatenate((value[..., :2] - value[..., 2:] / 2,
                           value[..., :2] + value[..., 2:] / 2), axis=-1)


def _audit_temporal(alias: str, source_case: Mapping[str, Any], teacher_case: Mapping[str, Any],
                    student_case: Mapping[str, Any], torch: Any, np: Any, temporal: Any) -> dict[str, Any]:
    teacher_temporal = teacher_case.get("temporal")
    if not isinstance(teacher_temporal, Mapping) or teacher_temporal.get("GT_read") is not False:
        raise RuntimeError(f"teacher temporal payload is missing: {alias}")
    saved_summary = student_case.get("temporal_summary")
    if not isinstance(saved_summary, Mapping):
        raise RuntimeError(f"student temporal summary is missing: {alias}")
    responses = teacher_temporal.get("teacher_responses")
    replayed = None
    if responses is not None:
        if not isinstance(responses, list) or len(responses) != 4:
            raise RuntimeError(f"teacher temporal responses are not the four fixed candidates: {alias}")
        replayed = temporal.fit_query(source_case, source_case["temporal_candidates"], responses)
        cached_fit = teacher_temporal.get("temporal_fit")
        if not isinstance(cached_fit, Mapping) or not _tensor_tree_equal(
                replayed, cached_fit, torch, np, atol=1e-12):
            raise RuntimeError(f"independent Temporal CPU replay differs from cached fit: {alias}")
        if list(replayed["final_interval"]) != list(teacher_temporal.get("final_interval", [])):
            raise RuntimeError(f"replayed Temporal interval differs from teacher record: {alias}")
    else:
        native = list(map(float, source_case["Native_interval"]))
        if (teacher_temporal.get("status") != "unavailable_temporal_input_native_fallback"
                or list(map(float, teacher_temporal.get("final_interval", []))) != native
                or int(teacher_temporal.get("optimizer_steps", -1)) != 0):
            raise RuntimeError(f"teacher no-input temporal fallback is not exact Native: {alias}")
    # Independently project only the response fields that the student seals;
    # raw teacher text stays inside the sealed private artifact and is not emitted.
    fit = teacher_temporal.get("temporal_fit") if isinstance(teacher_temporal.get("temporal_fit"), Mapping) else {}
    response_rows = []
    if isinstance(responses, list):
        for row in responses:
            response_rows.append({
                "candidate_id": row.get("candidate_id"), "status": row.get("status"),
                "caption_status": row.get("caption_status"), "reward": row.get("reward"),
                "probabilities": row.get("probabilities"),
                "caption_input_tokens": row.get("caption_input_tokens"),
                "caption_output_tokens": row.get("caption_output_tokens"),
                "input_tokens": row.get("input_tokens"), "output_tokens": row.get("output_tokens"),
                "max_new_tokens": row.get("max_new_tokens"),
                "output_truncated_at_cap": row.get("output_truncated_at_cap"),
                "teacher_call_count": row.get("teacher_call_count", 1),
            })
    expected_summary = {
        "status": teacher_temporal.get("status"),
        "final_interval": teacher_temporal.get("final_interval"),
        "candidate_order": teacher_temporal.get("candidate_order"),
        "optimizer_steps": fit.get("optimizer_steps", 0),
        "active_parameters": fit.get("active_parameters", 0),
        "seed_key": fit.get("seed_key"), "reward": fit.get("reward"),
        "loss": fit.get("loss"), "teacher_responses": response_rows,
        "GT_read": teacher_temporal.get("GT_read"),
        "raw_response_text_exported": False, "prompt_text_exported": False,
    }
    if not _tensor_tree_equal(saved_summary, expected_summary, torch, np, atol=1e-12):
        raise RuntimeError(f"student Temporal summary differs from sealed teacher response scalars: {alias}")
    return {
        "status": "replayed" if replayed is not None else "exact_native_fallback",
        "optimizer_steps": int(replayed["optimizer_steps"]) if replayed is not None else 0,
        "candidate_count": len(responses) if isinstance(responses, list) else 0,
        "final_interval": list(map(float, teacher_temporal["final_interval"])),
    }


def _audit_spatial(alias: str, source_case: Mapping[str, Any], teacher_case: Mapping[str, Any],
                   student_case: Mapping[str, Any], torch: Any, np: Any,
                   science: Any) -> dict[str, Any]:
    arms = student_case.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"F", "T", "S", "TS", "TeacherGeometry"}:
        raise RuntimeError(f"student arm schema differs: {alias}")
    frame_ids = list(map(int, source_case["frame_ids"]))
    native_xyxy = np.asarray(source_case["native_boxes_xyxy_normalized"], dtype=np.float64)
    native_cxcywh = source_case["zero"]["boxes"]
    temporal_interval = list(map(float, teacher_case["temporal"]["final_interval"]))
    native_interval = list(map(float, source_case["Native_interval"]))
    reference = teacher_case["reference"]
    geometry = reference["teacher_geometry"]
    expected_teacher_geometry = np.asarray(geometry["boxes_xyxy_normalized"], dtype=np.float64)
    if (list(map(int, geometry["frame_ids_physical"])) != frame_ids
            or list(map(float, geometry["interval_frame_bounds_half_open"])) != temporal_interval):
        raise RuntimeError(f"TeacherGeometry frame/interval binding differs: {alias}")
    expected = {"F": (native_xyxy, native_interval),
                "T": (native_xyxy, temporal_interval),
                "TeacherGeometry": (expected_teacher_geometry, temporal_interval)}
    for name, (boxes, interval) in expected.items():
        got = arms[name]
        if (not np.array_equal(np.asarray(got["boxes"], dtype=np.float64), boxes)
                or list(map(float, got["interval"])) != interval):
            raise RuntimeError(f"{name} arm composition differs from frozen inputs: {alias}")
    if (not np.array_equal(np.asarray(arms["F"]["boxes"]), np.asarray(arms["T"]["boxes"]))
            or not np.array_equal(np.asarray(arms["S"]["boxes"]), np.asarray(arms["TS"]["boxes"]))
            or list(map(float, arms["F"]["interval"])) != list(map(float, arms["S"]["interval"]))
            or list(map(float, arms["T"]["interval"])) != list(map(float, arms["TS"]["interval"]))):
        raise RuntimeError(f"F/T/S/TS shared prediction composition differs: {alias}")
    for arm_name, arm in arms.items():
        boxes = np.asarray(arm["boxes"], dtype=np.float64)
        if (boxes.shape != native_xyxy.shape or not np.isfinite(boxes).all()
                or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > 1)
                or np.any(boxes[:, 0] >= boxes[:, 2]) or np.any(boxes[:, 1] >= boxes[:, 3])):
            raise RuntimeError(f"illegal arm geometry: {alias}/{arm_name}")
    anchors = reference.get("anchors")
    no_update = reference.get("no_update")
    if (not isinstance(anchors, list) or not isinstance(no_update, bool)
            or no_update != (len(anchors) == 0)
            or student_case.get("reference_record") is None
            or science._reference_sha256(student_case["reference_record"])
            != science._reference_sha256(reference)):
        raise RuntimeError(f"student Spatial-A reference differs from teacher-locked reference: {alias}")
    fits, readouts = student_case.get("fits"), student_case.get("readouts")
    execution = student_case.get("execution", {})
    if not isinstance(fits, Mapping) or not isinstance(readouts, Mapping):
        raise RuntimeError(f"student fit/readout maps are malformed: {alias}")
    if no_update:
        if (fits or readouts or int(execution.get("actual_fit_count", -1)) != 0
                or int(execution.get("optimizer_steps", -1)) != 0
                or int(execution.get("proposed_optimizer_steps", -1)) != 0
                or int(execution.get("rejected_optimizer_steps", -1)) != 0
                or int(execution.get("final_full_model_redecode_calls", -1)) != 0
                or int(execution.get("source_capture_calls", -1)) != 1
                or int(execution.get("source_offset_forwards", -1)) != 2
                or not np.array_equal(arms["F"]["boxes"], arms["S"]["boxes"])
                or not np.array_equal(arms["T"]["boxes"], arms["TS"]["boxes"])):
            raise RuntimeError(f"no-update reference is not exact F/T reuse: {alias}")
        return {"no_update": True, "anchor_count": 0, "proposed_steps": 0,
                "accepted_steps": 0, "rejected_steps": 0, "readout_matches_cache": True}

    fit = fits.get("SpatialA")
    readout = readouts.get("SpatialA")
    if not isinstance(fit, Mapping) or not isinstance(readout, Mapping):
        raise RuntimeError(f"updating reference lacks SpatialA fit/readout: {alias}")
    expected_key = science.stable_fit_key(str(source_case["dataset"]),
                                          str(source_case["query_key_sha256"]))
    from scripts import stvg_target_teacher_student_v1 as core
    expected_cfg = core.load_original_opd_config(str(source_case["dataset"]))
    expected_cfg["steps"] = 5
    if (fit.get("active_parameters") != 1792 or fit.get("declared_steps") != 5
            or int(execution.get("actual_fit_count", -1)) != 1
            or int(execution.get("optimizer_steps", -1)) != fit.get("optimizer_steps")
            or int(execution.get("final_full_model_redecode_calls", -1)) != 1
            or int(execution.get("source_capture_calls", -1)) != 1
            or int(execution.get("source_offset_forwards", -1)) != 2
            or fit.get("seed_key") != expected_key
            or fit.get("reference_sha256") != student_case.get("reference_receipt", {}).get("teacher_file_sha256")
            or fit.get("reference_role") != "teacher"
            or fit.get("algorithm") != "validated_loss_weighted_iou"
            or fit.get("reward") != "original_iou"
            or fit.get("validation") != "weighted_reference_IoU_non_decrease_1e-7_then_stop"
            or student_case.get("configuration", {}).get("fit_key") != expected_key
            or student_case.get("configuration", {}).get("fit_algorithm")
            != "original_loss_weighted_iou_with_reference_non_decrease_gate"):
        raise RuntimeError(f"SpatialA configuration or reference binding differs: {alias}")
    if (not _tensor_tree_equal(fit.get("config"), expected_cfg, torch, np, atol=0.0)
            or not _tensor_tree_equal(student_case.get("configuration", {}).get("config"),
                                     expected_cfg, torch, np, atol=0.0)):
        raise RuntimeError(f"SpatialA config differs from frozen original config: {alias}")
    anchor_positions = [int(row["position"]) for row in anchors]
    anchor_boxes = np.asarray([row["box_cxcywh"] for row in anchors], dtype=np.float64)
    weights = np.asarray([row["weight"] for row in anchors], dtype=np.float64)
    gate_weights = np.asarray(fit.get("reliability"), dtype=np.float64)
    if (not anchor_positions or len(set(anchor_positions)) != len(anchor_positions)
            or any(p < 0 or p >= len(frame_ids) for p in anchor_positions)
            or anchor_boxes.shape != (len(anchors), 4) or not np.isfinite(anchor_boxes).all()
            or np.any(anchor_boxes[:, 2:] <= 0)
            or np.any(anchor_boxes[:, :2] - anchor_boxes[:, 2:] / 2 < 0)
            or np.any(anchor_boxes[:, :2] + anchor_boxes[:, 2:] / 2 > 1)
            or any(int(row["physical_frame_id"]) != frame_ids[position]
                   for row, position in zip(anchors, anchor_positions))
            or any(float(row["weight"]) != float(row["reliability"]["presence_fraction"])
                   for row in anchors)
            or weights.shape != (len(anchors),) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any(weights > 1)
            or gate_weights.shape != weights.shape or not np.isfinite(gate_weights).all()
            or np.any(gate_weights <= 0)
            or not np.allclose(gate_weights, weights, rtol=0.0, atol=1e-7)
            or fit.get("observation_positions") != anchor_positions
            ):
        raise RuntimeError(f"SpatialA anchor support/weights differ from locked reference: {alias}")
    path = fit.get("path")
    if not isinstance(path, list) or len(path) != 5:
        raise RuntimeError(f"SpatialA fit must retain five training slots: {alias}")
    before_gate = accepted_gate = proposed_steps = accepted_steps = rejected_steps = 0
    first_state = None
    last_after_state = None
    stopped_after_rejection = False
    for step, row in enumerate(path):
        if row.get("step") != step:
            raise RuntimeError(f"SpatialA path step index differs: {alias}/{step}")
        if row.get("skipped"):
            if step == 0 or not stopped_after_rejection or row.get("reason") != "stopped_after_rejected_step":
                raise RuntimeError(f"unexpected SpatialA skipped slot: {alias}/{step}")
            continue
        before_boxes = np.asarray(row["boxes"], dtype=np.float64)
        proposed_boxes = np.asarray(row["proposed_boxes"], dtype=np.float64)
        before_iou = _cxcywh_iou(before_boxes[anchor_positions], anchor_boxes, np)
        proposed_iou = _cxcywh_iou(proposed_boxes[anchor_positions], anchor_boxes, np)
        stored_before = np.asarray(row["reference_iou_before"], dtype=np.float64)
        stored_proposed = np.asarray(row["reference_iou_proposed"], dtype=np.float64)
        if (np.max(np.abs(before_iou - stored_before)) > TOL
                or np.max(np.abs(proposed_iou - stored_proposed)) > TOL):
            raise RuntimeError(f"SpatialA independent anchor IoU differs from path receipt: {alias}/{step}")
        before_gate += 1
        proposes = bool(row.get("proposed"))
        accepted = bool(row.get("accepted"))
        # Re-evaluate the acceptance scalar with the same float32 tensor
        # arithmetic as accept_reference_step.  The independently recomputed
        # IoUs above still verify the saved values; this avoids changing a
        # boundary decision through a NumPy float64 reduction.
        gate_weight_tensor = torch.as_tensor(fit.get("reliability"), dtype=torch.float32)
        before_tensor = torch.as_tensor(row["reference_iou_before"], dtype=torch.float32)
        proposed_tensor = torch.as_tensor(row["reference_iou_proposed"], dtype=torch.float32)
        gate_delta = float((gate_weight_tensor * (proposed_tensor - before_tensor)).sum()
                           / gate_weight_tensor.sum())
        gate_before = float((gate_weight_tensor * before_tensor).sum() / gate_weight_tensor.sum())
        gate_proposed = float((gate_weight_tensor * proposed_tensor).sum() / gate_weight_tensor.sum())
        can_accept = bool(torch.isfinite(proposed_tensor).all()) and math.isfinite(gate_delta) and gate_delta >= -1e-7
        if accepted != (proposes and can_accept):
            raise RuntimeError(f"SpatialA reference acceptance gate differs: {alias}/{step}")
        if proposes:
            proposed_steps += 1
        if accepted:
            accepted_gate += 1
            accepted_steps += 1
            if not _tensor_tree_equal(row["update_state"], row["proposed_state"], torch, np):
                raise RuntimeError(f"accepted SpatialA parameters differ from proposal: {alias}/{step}")
            if not _tensor_tree_equal(row["optimizer_after"], row["optimizer_proposed"], torch, np):
                raise RuntimeError(f"accepted SpatialA Adam state differs from proposal: {alias}/{step}")
        else:
            if proposes:
                rejected_steps += 1
                stopped_after_rejection = True
            if (not _tensor_tree_equal(row["update_state"], row["state"], torch, np)
                    or not _tensor_tree_equal(row["optimizer_after"], row["optimizer_before"], torch, np)
                    or not _tensor_tree_equal(row["after_boxes"], row["boxes"], torch, np)):
                raise RuntimeError(f"rejected/no-op SpatialA step did not restore parameters and Adam: {alias}/{step}")
        if first_state is None:
            first_state = row["state"]
        last_after_state = row["update_state"]
        if len(fit.get("rounds", [])) > step:
            round_row = fit["rounds"][step]
            if (round_row.get("step") != step or bool(round_row.get("proposed")) != proposes
                    or bool(round_row.get("accepted")) != accepted
                    or abs(float(round_row.get("reference_mean_before")) - gate_before) > TOL
                    or abs(float(round_row.get("reference_mean_proposed")) - gate_proposed) > TOL):
                raise RuntimeError(f"SpatialA round summary differs from independent IoU: {alias}/{step}")
    if (fit.get("optimizer_steps") != accepted_steps
            or fit.get("proposed_optimizer_steps") != proposed_steps
            or fit.get("rejected_optimizer_steps") != rejected_steps
            or int(execution.get("proposed_optimizer_steps", -1)) != proposed_steps
            or int(execution.get("rejected_optimizer_steps", -1)) != rejected_steps
            or proposed_steps != accepted_steps + rejected_steps
            or len(fit.get("rounds", [])) != before_gate):
        raise RuntimeError(f"SpatialA accepted/proposed/rejected counters differ: {alias}")
    if first_state is None or not _tensor_tree_equal(first_state, fit.get("initial_state"), torch, np):
        raise RuntimeError(f"SpatialA fit initial state differs from first path row: {alias}")
    if last_after_state is None or not _tensor_tree_equal(last_after_state, fit.get("final_state"), torch, np):
        raise RuntimeError(f"SpatialA fit final state differs from last executed path row: {alias}")
    cached = fit.get("final_cached_decoder_boxes")
    full = readout.get("boxes")
    if not _tensor_tree_equal(cached, full, torch, np):
        raise RuntimeError(f"SpatialA full readout differs from cached decoder boxes: {alias}")
    source_logits = source_case["zero"]["logits"]
    if not _tensor_tree_equal(readout.get("logits"), source_logits, torch, np):
        raise RuntimeError(f"SpatialA full readout changed source logits: {alias}")
    expected_xyxy = _xyxy_from_cxcywh(full, np)
    if (not np.array_equal(expected_xyxy, np.asarray(arms["S"]["boxes"], dtype=np.float64))
            or not np.array_equal(expected_xyxy, np.asarray(arms["TS"]["boxes"], dtype=np.float64))):
        raise RuntimeError(f"SpatialA full readout xyxy differs from S/TS cached arms: {alias}")
    return {"no_update": False, "anchor_count": len(anchors),
            "proposed_steps": proposed_steps, "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps, "checked_gate_steps": before_gate,
            "readout_matches_cache": True}


def _verify_one(alias: str, student_item: Mapping[str, Any], batch_ctx: Mapping[str, Any],
                ctx: Mapping[str, Any], torch: Any, np: Any,
                source_module: Any, science: Any, temporal: Any) -> dict[str, Any]:
    roster_row = ctx["roster_index"][alias]
    source_item = ctx["source_files"][alias]
    source_case, source_receipt, source_path, source_receipt_path = _load_source_case(
        alias, roster_row, source_item, ctx, torch, source_module)
    teacher_path_value = student_item.get("teacher_barrier_path")
    teacher_barrier_sha = student_item.get("teacher_barrier_sha256")
    student_barrier = batch_ctx.get("barrier")
    if student_barrier is not None:
        teacher_path_value = student_barrier.get("teacher_barrier_path")
        teacher_barrier_sha = student_barrier.get("teacher_barrier_sha256")
    if not isinstance(teacher_path_value, str):
        raise RuntimeError(f"student case lacks exact teacher barrier path: {alias}")
    teacher_barrier = _load_teacher_barrier(teacher_path_value, str(teacher_barrier_sha),
                                             ctx, [alias])
    teacher_case, teacher_receipt, teacher_path, teacher_receipt_path = _verify_teacher_case(
        alias, source_item, teacher_barrier, ctx, torch)
    student_path = _safe_path(student_item.get("path"), expected_parent=FULL / "student_cases")
    student_receipt_row = student_item.get("receipt")
    if not isinstance(student_receipt_row, Mapping):
        raise RuntimeError(f"student barrier file row lacks receipt metadata: {alias}")
    student_receipt_path = _safe_path(student_receipt_row.get("path"),
                                      expected_parent=FULL / "student_cases")
    if student_path.name != f"{alias}.pt" or student_receipt_path.name != f"{alias}.receipt.json":
        raise RuntimeError(f"student file path/alias mismatch: {alias}")
    if (_sha(student_path) != student_item.get("sha256")
            or student_path.stat().st_size != student_item.get("bytes")
            or _sha(student_receipt_path) != student_receipt_row.get("sha256")
            or student_receipt_path.stat().st_size != student_receipt_row.get("bytes")):
        raise RuntimeError(f"student output or receipt bytes differ from barrier: {alias}")
    student_receipt = _read_json(student_receipt_path)
    if (student_receipt.get("schema") != "stvg-final-table1-full-student-receipt-v1"
            or student_receipt.get("status") != "sealed"
            or student_receipt.get("case_id") != alias
            or student_receipt.get("sha256") != student_item.get("sha256")
            or student_receipt.get("bytes") != student_item.get("bytes")
            or student_receipt.get("source_case_sha256") != source_item.get("source_case_sha256")
            or student_receipt.get("source_file_sha256") != source_item.get("sha256")
            or student_receipt.get("source_receipt_sha256") != source_item.get("receipt_sha256")
            or student_receipt.get("teacher_file_sha256") != teacher_barrier["files"][alias]["sha256"]
            or student_receipt.get("teacher_receipt_sha256") != teacher_barrier["files"][alias]["receipt_sha256"]
            or student_receipt.get("teacher_barrier_path") != str(teacher_barrier["path"].relative_to(ROOT))
            or student_receipt.get("teacher_barrier_sha256") != teacher_barrier["sha256"]
            or student_receipt.get("student_runtime_sha256") != ctx["student_runtime_sha256"]
            or student_receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or student_receipt.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or student_receipt.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or student_receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or student_receipt.get("request_sha256") != teacher_receipt.get("request_sha256")
            or student_receipt.get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")
            or student_receipt.get("GT_read") is not False):
        raise RuntimeError(f"student receipt is stale or detached: {alias}")
    student_case = torch.load(student_path, map_location="cpu", weights_only=False)
    if (student_case.get("schema") != "stvg-final-table1-student-case-v1"
            or student_case.get("case_id") != alias
            or student_case.get("dataset") != roster_row["dataset"]
            or student_case.get("query_key_sha256") != roster_row["query_key_sha256"]
            or student_case.get("source_case_sha256") != source_item.get("source_case_sha256")
            or student_case.get("source_file_sha256") != source_item.get("sha256")
            or student_case.get("source_receipt_sha256") != source_item.get("receipt_sha256")
            or student_case.get("teacher_receipt_sha256") != teacher_barrier["files"][alias]["receipt_sha256"]
            or student_case.get("reference_sha256") != teacher_barrier["files"][alias]["sha256"]
            or student_case.get("request_sha256") != teacher_receipt.get("request_sha256")
            or student_case.get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")
            or student_case.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or student_case.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or student_case.get("student_runtime_sha256") != ctx["student_runtime_sha256"]
            or student_case.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or student_case.get("teacher_barrier_sha256") != teacher_barrier["sha256"]
            or student_case.get("teacher_batch_barrier_path") != str(teacher_barrier["path"].relative_to(ROOT))
            or student_case.get("teacher_batch_barrier_sha256") != teacher_barrier["sha256"]
            or student_case.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or student_case.get("GT_read") is not False
            or student_case.get("online_carry") is not False):
        raise RuntimeError(f"student payload identity/configuration binding differs: {alias}")
    if (student_case.get("reference_receipt", {}).get("teacher_file_sha256") != teacher_barrier["files"][alias]["sha256"]
            or student_case.get("reference_receipt", {}).get("teacher_receipt_sha256") != teacher_barrier["files"][alias]["receipt_sha256"]
            or student_case.get("reference_receipt", {}).get("request_sha256") != teacher_receipt.get("request_sha256")
            or student_case.get("reference_receipt", {}).get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")):
        raise RuntimeError(f"student reference receipt differs from teacher custody: {alias}")
    if (teacher_case.get("dataset") != source_case.get("dataset")
            or teacher_case["reference"].get("case_id") != alias
            or teacher_case["reference"].get("dataset") != source_case.get("dataset")):
        raise RuntimeError(f"teacher dataset/reference identity differs from source: {alias}")
    # Source reset and cold readback: compare every retained source state leaf,
    # original cxcywh/logits, frame grid, pixel receipt and model-state digest.
    cold = student_case.get("source_cold_readback", {})
    for flag in ("frame_ids_match", "initial_state_match", "native_xyxy_match",
                 "source_logits_bitwise", "full_model_state_restored"):
        if cold.get(flag) is not True:
            raise RuntimeError(f"student source readback flag {flag} is false: {alias}")
    if (cold.get("source_case_sha256") != source_item.get("source_case_sha256")
            or cold.get("source_model_state_sha256") != source_case.get("source_model_state_sha256")
            or cold.get("pixel_sha256") != source_case.get("pixel_sha256")
            or list(map(int, student_case.get("frame_ids", []))) != list(map(int, source_case["frame_ids"]))
            or student_case.get("source_model_state_sha256") != source_case.get("source_model_state_sha256")):
        raise RuntimeError(f"student source cold identity differs from sealed source: {alias}")
    native_from_cxcywh = _xyxy_from_cxcywh(source_case["zero"]["boxes"], np)
    if not np.array_equal(native_from_cxcywh,
                          np.asarray(source_case["native_boxes_xyxy_normalized"], dtype=np.float64)):
        raise RuntimeError(f"sealed source xyxy differs from independent native cxcywh conversion: {alias}")
    for key in ("source_light_initial_state", "source_boxes_cxcywh", "source_logits"):
        expected_value = (source_case[key] if key == "source_light_initial_state"
                          else source_case["zero"]["boxes"] if key == "source_boxes_cxcywh"
                          else source_case["zero"]["logits"])
        if not _tensor_tree_equal(student_case.get(key), expected_value, torch, __import__("numpy")):
            raise RuntimeError(f"student retained source reset tensor differs: {alias}/{key}")
    temporal_result = _audit_temporal(alias, source_case, teacher_case, student_case,
                                      torch, __import__("numpy"), temporal)
    spatial_result = _audit_spatial(alias, source_case, teacher_case, student_case,
                                    torch, __import__("numpy"), science)
    teacher_status = teacher_case.get("who", {}).get("who", {}).get("status")
    teacher_selected = (teacher_case.get("who", {}).get("who", {}).get("candidate_id")
                        if teacher_status == "valid" else None)
    if (student_case.get("teacher_selection", {}).get("status") != teacher_status
            or student_case.get("teacher_selection", {}).get("selected_object_id") != teacher_selected):
        raise RuntimeError(f"student Teacher selection status differs from sealed teacher: {alias}")
    receipt_no_update = student_receipt.get("no_update")
    if receipt_no_update is not teacher_case["reference"].get("no_update"):
        raise RuntimeError(f"student receipt no-update flag differs from reference: {alias}")
    expected_fit_count = 0 if spatial_result["no_update"] else 1
    expected_readout_count = 0 if spatial_result["no_update"] else 1
    if (int(student_receipt.get("actual_fit_count", -1)) != expected_fit_count
            or int(student_receipt.get("optimizer_steps", -1)) != spatial_result["accepted_steps"]
            or int(student_receipt.get("final_full_model_redecode_calls", -1)) != expected_readout_count):
        raise RuntimeError(f"student receipt execution counters differ from audited fit: {alias}")
    return {
        "alias": alias,
        "dataset": roster_row["dataset"],
        "source_case_sha256": source_item["source_case_sha256"],
        "source_file_sha256": source_item["sha256"],
        "teacher_file_sha256": teacher_barrier["files"][alias]["sha256"],
        "teacher_receipt_sha256": teacher_barrier["files"][alias]["receipt_sha256"],
        "student_file_sha256": student_item["sha256"],
        "student_receipt_sha256": student_receipt_row["sha256"],
        "temporal": temporal_result,
        "spatial": spatial_result,
        "teacher_status": teacher_status,
        "source_reset": "pass",
        "arm_composition": "pass",
        "readout_matches_cache": spatial_result["readout_matches_cache"],
        "student_batch_barrier_path": (str(batch_ctx["path"].relative_to(ROOT))
                                        if batch_ctx.get("path") else None),
        "student_batch_barrier_sha256": batch_ctx.get("sha256"),
        "reused_in_batch": bool(batch_ctx.get("row", student_item).get("reused", False)),
        "execution": {
            "fit_count": expected_fit_count,
            "optimizer_steps": spatial_result["accepted_steps"],
            "full_model_redecode_calls": expected_readout_count,
            "source_capture_calls": 1,
        },
    }


def _read_barrier(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    barrier = _read_json(path)
    if path.name == "STUDENT_GLOBAL_BARRIER.json":
        if (barrier.get("schema") != "stvg-final-table1-full-student-global-barrier-v1"
                or barrier.get("status") != "sealed" or barrier.get("GT_read") is not False
                or barrier.get("count") != EXPECTED_COUNT
                or barrier.get("domain_counts") != EXPECTED_DOMAINS
                or barrier.get("student_runtime_sha256") != ctx["student_runtime_sha256"]
                or barrier.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
                or barrier.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
                or barrier.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
                or barrier.get("full_source_roster_sha256") != ctx["roster_sha256"]):
            raise RuntimeError("FULL global student barrier is incomplete or detached")
        files = barrier.get("files")
        expected_aliases = [row["probe_alias"] for row in ctx["rows"]]
        if (not isinstance(files, list) or len(files) != EXPECTED_COUNT
                or [row.get("alias") for row in files] != expected_aliases):
            raise RuntimeError("FULL global student files do not exactly cover canonical roster order")
        batches = barrier.get("student_batch_barriers")
        if not isinstance(batches, list) or not batches:
            raise RuntimeError("FULL global student barrier lacks its bounded batch seals")
        batch_index: dict[str, dict[str, Any]] = {}
        batches_read = []
        batch_count_sum = 0
        for batch_row in batches:
            if not isinstance(batch_row, Mapping):
                raise ValueError("global student batch row must be an object")
            batch_path = _safe_path(batch_row.get("path"))
            if not batch_path.is_relative_to(FULL.resolve()) or _sha(batch_path) != batch_row.get("sha256"):
                raise RuntimeError("global student batch barrier hash/path mismatch")
            batch = _read_batch_barrier(batch_path, ctx)
            if batch["barrier"].get("count") != batch_row.get("count"):
                raise RuntimeError("global student batch count differs from referenced batch seal")
            batch_count_sum += batch_row["count"]
            for alias in batch["barrier"]["aliases"]:
                if alias in batch_index:
                    raise RuntimeError(f"student global has duplicate batch membership: {alias}")
                batch_index[alias] = {"path": batch_path, "sha256": batch_row["sha256"],
                                      "barrier": batch["barrier"], "rows": batch["rows"],
                                      "teacher_barrier": batch["teacher_barrier"]}
            batches_read.append(batch_row["sha256"])
        if batch_count_sum != EXPECTED_COUNT or set(batch_index) != set(expected_aliases):
            raise RuntimeError("global student batch seals do not cover the exact roster")
        global_rows = {row["alias"]: row for row in files}
        for alias in expected_aliases:
            row = global_rows[alias]
            membership = batch_index[alias]
            if (row.get("student_batch_barrier_path") != str(membership["path"].relative_to(ROOT))
                    or row.get("student_batch_barrier_sha256") != membership["sha256"]):
                raise RuntimeError(f"global student row has incorrect batch membership: {alias}")
            batch_row = membership["rows"][alias]
            for key in ("path", "sha256", "bytes", "receipt"):
                if row.get(key) != batch_row.get(key):
                    raise RuntimeError(f"global student row differs from batch row: {alias}/{key}")
        return {"path": path, "sha256": _sha(path), "barrier": barrier,
                "rows": {row["alias"]: row for row in files}, "global": True,
                "batch_index": batch_index}
    return _read_batch_barrier(path, ctx)


def _read_batch_barrier(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    barrier = _read_json(path)
    barrier_sha = _sha(path)
    if (barrier.get("schema") != "stvg-final-table1-full-student-batch-barrier-v1"
            or barrier.get("status") != "sealed" or barrier.get("GT_read") is not False
            or barrier.get("student_runtime_sha256") != ctx["student_runtime_sha256"]
            or barrier.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or barrier.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or barrier.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or barrier.get("full_source_roster_sha256") != ctx["roster_sha256"]):
        raise RuntimeError(f"student batch barrier is incomplete or detached: {path.name}")
    aliases, rows = barrier.get("aliases"), barrier.get("files")
    if (not isinstance(aliases, list) or not aliases or len(set(aliases)) != len(aliases)
            or not isinstance(rows, list) or barrier.get("count") != len(rows) or len(rows) != len(aliases)
            or [row.get("alias") for row in rows] != aliases
            or not set(aliases).issubset(ctx["roster_index"])):
        raise RuntimeError(f"student batch aliases/files are inconsistent: {path.name}")
    roster_order = {alias: i for i, alias in enumerate(ctx["roster_index"])}
    if aliases != sorted(aliases, key=roster_order.__getitem__):
        raise RuntimeError(f"student batch aliases are not in canonical roster order: {path.name}")
    groups = {}
    for alias, row in ctx["roster_index"].items():
        groups.setdefault(_group_key(row), set()).add(alias)
    selected = set(aliases)
    for group in {_group_key(ctx["roster_index"][alias]) for alias in aliases}:
        if not groups[group].issubset(selected):
            raise RuntimeError(f"student batch splits an exact video/clock/bounds group: {path.name}")
    if barrier.get("video_group_count") != len({_group_key(ctx["roster_index"][a]) for a in aliases}):
        raise RuntimeError(f"student batch video-group count differs: {path.name}")
    teacher_path = _safe_path(barrier.get("teacher_barrier_path"))
    teacher = _load_teacher_barrier(barrier.get("teacher_barrier_path"),
                                    str(barrier.get("teacher_barrier_sha256")), ctx, aliases)
    rows_index = {row["alias"]: dict(row) for row in rows}
    return {"path": path, "sha256": barrier_sha, "barrier": barrier,
            "rows": rows_index, "teacher_barrier": teacher, "teacher_path": teacher_path}


def audit(*, student_barrier: str | Path | None, global_mode: bool,
          output: str | Path) -> dict[str, Any]:
    import gc
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("FULL readback refuses an initialized CUDA context")
    from scripts.stvg_native_instance_common_v1 import guard
    sys.addaudithook(guard)
    from scripts import run_stvg_table1_source_v1 as source_module
    from scripts import run_stvg_table1_student003 as science
    from scripts import stvg_table1_temporal_adapter_v1 as temporal

    ctx = _authority()
    if global_mode:
        path = FULL / "STUDENT_GLOBAL_BARRIER.json"
    else:
        path = _safe_path(student_barrier)
        if not path.is_relative_to(FULL.resolve()):
            raise ValueError("student batch barrier must be inside FULL namespace")
    barrier_ctx = _read_barrier(path, ctx)
    aliases = list(barrier_ctx["rows"])
    result_rows = []
    total_fit = total_steps = total_proposed = total_accepted = total_rejected = 0
    no_update_count = 0
    started = time.perf_counter()
    for index, alias in enumerate(aliases, 1):
        student_item = barrier_ctx["rows"][alias]
        if barrier_ctx.get("global"):
            membership = barrier_ctx["batch_index"][alias]
            per_case_ctx = {"barrier": membership["barrier"], "path": membership["path"],
                            "sha256": membership["sha256"], "row": membership["rows"][alias]}
            # Actual teacher binding is per query and copied into the student receipt.
            row_for_case = dict(student_item)
            receipt_path = _safe_path(row_for_case["receipt"]["path"])
            student_receipt = _read_json(receipt_path)
            row_for_case["teacher_barrier_path"] = student_receipt.get("teacher_barrier_path")
            row_for_case["teacher_barrier_sha256"] = student_receipt.get("teacher_barrier_sha256")
        else:
            per_case_ctx = {"barrier": barrier_ctx["barrier"], "path": barrier_ctx["path"],
                            "sha256": barrier_ctx["sha256"], "row": student_item}
            row_for_case = dict(student_item)
            row_for_case["teacher_barrier_path"] = barrier_ctx["barrier"].get("teacher_barrier_path")
            row_for_case["teacher_barrier_sha256"] = barrier_ctx["barrier"].get("teacher_barrier_sha256")
        readback = _verify_one(alias, row_for_case, per_case_ctx, ctx, torch, np,
                               source_module, science, temporal)
        total_fit += int(readback["spatial"]["anchor_count"] > 0)
        total_steps += int(readback["spatial"]["accepted_steps"])
        total_proposed += int(readback["spatial"]["proposed_steps"])
        total_accepted += int(readback["spatial"]["accepted_steps"])
        total_rejected += int(readback["spatial"]["rejected_steps"])
        no_update_count += int(readback["spatial"]["no_update"])
        result_rows.append(readback)
        if index % 250 == 0:
            print(f"FULL_CPU_READBACK {index}/{len(aliases)}", flush=True)
        del readback
        gc.collect()
    readbacks_by_batch: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        batch_path_value = row.get("student_batch_barrier_path")
        if not isinstance(batch_path_value, str):
            raise RuntimeError(f"student row has no batch membership: {row['alias']}")
        readbacks_by_batch.setdefault(batch_path_value, []).append(row)
    expected_batch_contexts = {}
    if barrier_ctx.get("global"):
        for membership in barrier_ctx["batch_index"].values():
            batch_rel = str(membership["path"].relative_to(ROOT))
            expected_batch_contexts[batch_rel] = membership
    else:
        expected_batch_contexts[str(barrier_ctx["path"].relative_to(ROOT))] = {
            "path": barrier_ctx["path"], "sha256": barrier_ctx["sha256"],
            "barrier": barrier_ctx["barrier"],
        }
    if set(readbacks_by_batch) != set(expected_batch_contexts):
        raise RuntimeError("audited rows do not match exact student batch membership")
    for batch_rel, membership in expected_batch_contexts.items():
        batch = membership["barrier"]
        rows = readbacks_by_batch[batch_rel]
        reused = [row for row in rows if row["reused_in_batch"]]
        new = [row for row in rows if not row["reused_in_batch"]]
        sum_field = lambda values, key: sum(int(row["execution"][key]) for row in values)
        expected_counts = {
            "actual_source_capture_calls": len(new),
            "reused_student_cases": len(reused),
            "actual_fit_count_this_invocation": sum_field(new, "fit_count"),
            "optimizer_steps_this_invocation": sum_field(new, "optimizer_steps"),
            "final_full_model_redecode_calls_this_invocation": sum_field(new, "full_model_redecode_calls"),
            "reused_case_fit_count": sum_field(reused, "fit_count"),
            "reused_case_optimizer_steps": sum_field(reused, "optimizer_steps"),
            "reused_case_final_full_model_redecode_calls": sum_field(reused, "full_model_redecode_calls"),
            "sealed_case_fit_count_total": sum_field(rows, "fit_count"),
            "sealed_case_optimizer_steps_total": sum_field(rows, "optimizer_steps"),
            "sealed_case_final_full_model_redecode_calls_total": sum_field(rows, "full_model_redecode_calls"),
        }
        for key, expected in expected_counts.items():
            if int(batch.get(key, -1)) != expected:
                raise RuntimeError(f"student batch aggregate {key} differs: {batch_rel}")
        if (batch.get("teacher_calls") != 0 or batch.get("DINO_calls") != 0
                or batch.get("GT_read") is not False):
            raise RuntimeError(f"student batch purity counters differ: {batch_rel}")
    if global_mode and len(aliases) != EXPECTED_COUNT:
        raise RuntimeError("global audit did not cover all 13,785 fixed roster rows")
    if global_mode:
        global_barrier = barrier_ctx["barrier"]
        aggregate = {
            "student_case_receipts_readback": len(result_rows),
            "actual_fit_count": total_fit,
            "optimizer_steps": total_steps,
            "final_full_model_redecode_calls": sum(r["execution"]["full_model_redecode_calls"] for r in result_rows),
        }
        for key, expected in aggregate.items():
            if int(global_barrier.get(key, -1)) != expected:
                raise RuntimeError(f"global student aggregate {key} differs from individual receipts")
        if (global_barrier.get("teacher_calls") != 0 or global_barrier.get("DINO_calls") != 0
                or global_barrier.get("GT_read") is not False):
            raise RuntimeError("global student purity counters differ")
        expected_teacher_barriers = {}
        for membership in barrier_ctx["batch_index"].values():
            tbarrier = membership["teacher_barrier"]
            expected_teacher_barriers[tbarrier["sha256"]] = {
                "path": str(tbarrier["path"].relative_to(ROOT)),
                "sha256": tbarrier["sha256"],
                "count": int(tbarrier["barrier"]["count"]),
            }
        declared = global_barrier.get("teacher_batch_barriers")
        if (not isinstance(declared, list)
                or {r.get("sha256"): {k: r.get(k) for k in ("path", "sha256", "count")} for r in declared}
                != expected_teacher_barriers):
            raise RuntimeError("global student teacher-batch barrier set differs from case memberships")
    barrier_schema = barrier_ctx["barrier"].get("schema")
    out = {
        "schema": "stvg-final-table1-full-student-cpu-readback-v1",
        "status": "pass", "scope": "GT-free source/teacher/student custody, frozen arms, Temporal replay, Spatial-A anchor gate/rollback and full-vs-cached readout",
        "student_barrier_path": str(path.relative_to(ROOT)),
        "student_barrier_sha256": barrier_ctx["sha256"],
        "student_barrier_schema": barrier_schema,
        "count": len(result_rows),
        "global_mode": bool(global_mode),
        "domain_counts": {dataset: sum(row["dataset"] == dataset for row in result_rows)
                          for dataset in EXPECTED_DOMAINS},
        "source_barrier_sha256": ctx["source_barrier_sha256"],
        "full_source_roster_sha256": ctx["roster_sha256"],
        "source_runtime_sha256": ctx["source_runtime_sha256"],
        "teacher_runtime_sha256": ctx["teacher_runtime_sha256"],
        "student_runtime_sha256": ctx["student_runtime_sha256"],
        "runtime_pin_count_readback": ctx["pin_counts"],
        "actual_fit_case_count": total_fit,
        "no_update_case_count": no_update_count,
        "optimizer_steps": total_steps,
        "proposed_optimizer_steps": total_proposed,
        "accepted_optimizer_steps": total_accepted,
        "rejected_optimizer_steps": total_rejected,
        "records": result_rows,
        "GT_read": False, "CUDA_used": False, "model_loaded": False,
        "elapsed_cpu_seconds": time.perf_counter() - started,
    }
    out_path = _safe_path(output) if Path(output).exists() else None
    if out_path is not None:
        raise FileExistsError(f"preserve existing readback: {out_path}")
    target = Path(output)
    target = target if target.is_absolute() else ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("readback output must stay inside repository root")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {key: value for key, value in out.items() if key != "records"}


def contracts() -> dict[str, Any]:
    import numpy as np
    import torch

    exact = _cxcywh_iou(np.asarray([[0.5, 0.5, 0.4, 0.4]]),
                        np.asarray([[0.5, 0.5, 0.4, 0.4]]), np)
    partial = _cxcywh_iou(np.asarray([[0.5, 0.5, 0.4, 0.4]]),
                          np.asarray([[0.6, 0.5, 0.4, 0.4]]), np)
    assert exact.shape == (1,) and exact[0] == 1.0
    assert math.isclose(float(partial[0]), 3.0 / 5.0, abs_tol=1e-7)
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    rng = np.random.default_rng(20261015)
    boxes_a = rng.random((256, 4), dtype=np.float32)
    boxes_b = rng.random((256, 4), dtype=np.float32)
    boxes_a[:, 2:] *= np.float32(0.35)
    boxes_b[:, 2:] *= np.float32(0.35)
    numpy_iou = _cxcywh_iou(boxes_a, boxes_b, np)
    torch_iou = aligned_iou(torch.from_numpy(boxes_a), torch.from_numpy(boxes_b)).cpu().numpy()
    assert np.array_equal(numpy_iou, torch_iou)
    tiny_box = np.asarray([[0.5, 0.5, 1e-4, 1e-4]], dtype=np.float32)
    tiny_iou = _cxcywh_iou(tiny_box, tiny_box, np)
    tiny_torch_iou = aligned_iou(torch.from_numpy(tiny_box), torch.from_numpy(tiny_box)).cpu().numpy()
    assert np.array_equal(tiny_iou, tiny_torch_iou) and 0.09 < float(tiny_iou[0]) < 0.11
    raw_float32 = np.asarray([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
    value64 = raw_float32.astype(np.float64)
    xy = _xyxy_from_cxcywh(raw_float32, np)
    expected_xy = np.concatenate((value64[:, :2] - value64[:, 2:] / 2,
                                  value64[:, :2] + value64[:, 2:] / 2), axis=-1)
    assert np.array_equal(xy, expected_xy) and xy[0, 0] != 0.3
    # A synthetic weighted non-decrease is accepted; a harmful proposal is
    # rejected and both the saved parameter state and Adam state are restored.
    before = np.asarray([0.4, 0.5], dtype=np.float64)
    good = np.asarray([0.5, 0.5], dtype=np.float64)
    bad = np.asarray([0.3, 0.5], dtype=np.float64)
    weights = np.asarray([0.2, 0.8], dtype=np.float64)
    good_mean = float(np.dot(weights, good - before) / weights.sum())
    bad_mean = float(np.dot(weights, bad - before) / weights.sum())
    assert good_mean >= -1e-7 and bad_mean < -1e-7
    before_params = {"w": torch.tensor([1.0, 2.0])}
    before_adam = {"step": torch.tensor(2), "exp_avg": torch.tensor([0.1, 0.2])}
    after_reject_params = {"w": torch.tensor([1.3, 2.4])}
    after_reject_adam = {"step": torch.tensor(3), "exp_avg": torch.tensor([0.2, 0.3])}
    # Model the exact rollback contract with independent copies.
    restored_params = {k: v.clone() for k, v in before_params.items()}
    restored_adam = {k: v.clone() for k, v in before_adam.items()}
    assert _tensor_tree_equal(restored_params, before_params, torch, np)
    assert _tensor_tree_equal(restored_adam, before_adam, torch, np)
    assert not _tensor_tree_equal(after_reject_params, restored_params, torch, np)
    assert not _tensor_tree_equal(after_reject_adam, restored_adam, torch, np)
    # Full decoder readout must match the fit cache before the xyxy arms are
    # materialized; a changed row is detected by the same exact comparator.
    cached = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
    full = cached.clone()
    assert _tensor_tree_equal(cached, full, torch, np)
    full[0, 0] += 1e-3
    assert not _tensor_tree_equal(cached, full, torch, np)
    from scripts.stvg_instance_validated_fit_v1 import accept_reference_step
    gate_good = accept_reference_step(torch.tensor([0.4, 0.5]), torch.tensor([0.5, 0.5]),
                                      torch.tensor([0.2, 0.8]))
    gate_bad = accept_reference_step(torch.tensor([0.4, 0.5]), torch.tensor([0.3, 0.5]),
                                     torch.tensor([0.2, 0.8]))
    assert gate_good is True and gate_bad is False
    return {
        "schema": "stvg-final-table1-full-student-cpu-readback-contracts-v1",
        "status": "pass", "checks": 12,
        "synthetic_checks": {
            "cxcywh_iou_identity_and_partial_overlap": True,
            "numpy_float32_iou_exact_frozen_objective_on_256_boxes": True,
            "union_floor_matches_frozen_objective_for_tiny_legal_box": True,
            "float64_xyxy_conversion": True,
            "weighted_reference_gate_accepts_non_decrease": True,
            "weighted_reference_gate_rejects_harm": True,
            "frozen_accept_reference_step_cpu_contract": True,
            "parameter_rollback_detects_proposal": True,
            "adam_rollback_detects_proposal": True,
            "full_readout_exact_cache_match": True,
            "full_readout_cache_mismatch_detected": True,
            "no_update_is_validated_separately_in_actual_case_reader": True,
        },
        "GT_read": False, "CUDA_used": False, "model_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--student-barrier", help="root-relative FULL student batch barrier")
    choice.add_argument("--global", dest="global_mode", action="store_true",
                        help="audit FULL/STUDENT_GLOBAL_BARRIER.json with 13,785 rows")
    parser.add_argument("--output", required=False, help="new root-relative JSON readback path")
    parser.add_argument("--contracts", action="store_true", help="run synthetic CPU math contracts only")
    args = parser.parse_args()
    if args.contracts:
        if args.student_barrier or args.global_mode or args.output:
            parser.error("--contracts cannot be combined with artifact-reading options")
        print(json.dumps(contracts(), sort_keys=True, ensure_ascii=False))
        return
    if args.output is None or (args.student_barrier is None and not args.global_mode):
        parser.error("provide --student-barrier or --global, and --output")
    if args.global_mode:
        if args.student_barrier:
            parser.error("--global and --student-barrier are mutually exclusive")
    elif not args.student_barrier:
        parser.error("--student-barrier path is required unless --global is selected")
    result = audit(student_barrier=args.student_barrier, global_mode=args.global_mode,
                   output=args.output)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
