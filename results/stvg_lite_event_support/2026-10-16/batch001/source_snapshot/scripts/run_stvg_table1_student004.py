#!/usr/bin/env python3
"""Source-reset Table 1 student over the locked 64-query engineering cohort.

The worker consumes only the sealed source and teacher artifacts.  It replays
each frozen source query, checks the original zero-update readout, fits the
locked Spatial-A arm when its pre-locked reference has at least one supported
anchor, performs one full decoder readout for that fit, and writes F/T/S/TS/
TeacherGeometry rows.  It never reads GT or changes the teacher/reference.

This module is inert on import.  ``--contracts`` runs pure CPU schema/key/arm
contracts and does not import Torch, model packages, media readers, or artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
ALIASES = {*(f"HC{i:02d}" for i in range(1, 33)), *(f"V{i:02d}" for i in range(1, 33))}
DATASETS = {"hc2", "vidstg"}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_BARRIER_REL = "artifacts/stvg_final_table1_v1/SOURCE_CAPTURE_BARRIER.json"
TEACHER_BARRIER_REL = "artifacts/stvg_final_table1_v1/TEACHER_GLOBAL_BARRIER.json"
SOURCE_RUNTIME_REL = "artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json"
TEACHER_RUNTIME_REL = "artifacts/stvg_final_table1_v1/TEACHER_RUNTIME.json"
STUDENT_RUNTIME_REL = "artifacts/stvg_final_table1_v1/STUDENT_RUNTIME.json"
OWN_REL = "scripts/run_stvg_table1_student004.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def stable_fit_key(dataset: str, query_key_sha256: str) -> str:
    """Stable across preflight/full aliases while preserving the original key prefix.

    The legacy fixed-space key parser permits a single separator after the
    prefix, so dataset and query identity are joined with ``_`` in its suffix.
    """
    dataset = str(dataset)
    digest = str(query_key_sha256)
    if dataset not in DATASETS or not SHA_RE.fullmatch(digest):
        raise ValueError("stable OPD key requires a locked dataset and query-key SHA-256")
    return f"caption_query_fixed_space|{dataset}_{digest}"


def _plain(value: Any) -> Any:
    """Convert teacher-reference values to canonical JSON-safe primitives."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _plain(value.detach().cpu().tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value in locked reference")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported locked-reference value: {type(value).__name__}")


def _reference_sha256(reference: Mapping[str, Any]) -> str:
    body = json.dumps(_plain(reference), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _check_interval(interval: Any, bounds: Any, name: str) -> list[float]:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"{name} interval must be a two-value sequence")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("video_bounds must be a two-value sequence")
    values = [float(interval[0]), float(interval[1])]
    limits = [float(bounds[0]), float(bounds[1])]
    if (not all(math.isfinite(x) for x in values + limits)
            or values[0] >= values[1] or limits[0] > values[0]
            or values[1] > limits[1]):
        raise ValueError(f"{name} interval is not finite, ordered, and inside video bounds")
    return values


def _arm(boxes: Any, interval: Sequence[float], *, name: str, expected_count: int) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(boxes, dtype=np.float64)
    if array.shape != (expected_count, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} boxes must be finite N x 4 normalized xyxy")
    if not (np.all(array[:, :2] >= 0.0) and np.all(array[:, 2:] <= 1.0)
            and np.all(array[:, 0] < array[:, 2]) and np.all(array[:, 1] < array[:, 3])):
        raise ValueError(f"{name} boxes contain illegal normalized xyxy geometry")
    if len(interval) != 2 or not float(interval[0]) < float(interval[1]):
        raise ValueError(f"{name} interval is invalid")
    return {"boxes": array.copy(), "interval": [float(interval[0]), float(interval[1])]}


def _compose_arms(native_boxes: Any, native_interval: Sequence[float],
                  temporal_interval: Sequence[float], spatial_boxes: Any,
                  teacher_geometry_boxes: Any, *, expected_count: int) -> dict[str, dict[str, Any]]:
    """Construct the five locked prediction arms without changing their boxes."""
    return {
        "F": _arm(native_boxes, native_interval, name="F", expected_count=expected_count),
        "T": _arm(native_boxes, temporal_interval, name="T", expected_count=expected_count),
        "S": _arm(spatial_boxes, native_interval, name="S", expected_count=expected_count),
        "TS": _arm(spatial_boxes, temporal_interval, name="TS", expected_count=expected_count),
        "TeacherGeometry": _arm(teacher_geometry_boxes, temporal_interval,
                                 name="TeacherGeometry", expected_count=expected_count),
    }


def _safe_temporal_summary(temporal: Mapping[str, Any]) -> dict[str, Any]:
    """Retain inference scalars and IDs, never raw teacher text or prompt data."""
    fit = temporal.get("temporal_fit")
    fit = fit if isinstance(fit, Mapping) else {}
    responses = temporal.get("teacher_responses", [])
    response_rows = []
    if isinstance(responses, list):
        for row in responses:
            if not isinstance(row, Mapping):
                continue
            probabilities = row.get("probabilities")
            response_rows.append({
                "candidate_id": row.get("candidate_id"),
                "status": row.get("status"),
                "caption_status": row.get("caption_status"),
                "reward": row.get("reward"),
                "probabilities": probabilities,
                "caption_input_tokens": row.get("caption_input_tokens"),
                "caption_output_tokens": row.get("caption_output_tokens"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "max_new_tokens": row.get("max_new_tokens"),
                "output_truncated_at_cap": row.get("output_truncated_at_cap"),
                "teacher_call_count": row.get("teacher_call_count", 1),
            })
    return {
        "status": temporal.get("status"),
        "final_interval": temporal.get("final_interval"),
        "candidate_order": temporal.get("candidate_order"),
        "optimizer_steps": fit.get("optimizer_steps", 0),
        "active_parameters": fit.get("active_parameters", 0),
        "seed_key": fit.get("seed_key"),
        "reward": fit.get("reward"),
        "loss": fit.get("loss"),
        "teacher_responses": response_rows,
        "GT_read": temporal.get("GT_read"),
        "raw_response_text_exported": False,
        "prompt_text_exported": False,
    }


def _resolve_runtime_path(runtime_path: str | Path | None) -> Path:
    """Resolve a root-relative (or in-root absolute) student runtime path."""
    if runtime_path is None:
        path = BASE / "STUDENT_RUNTIME.json"
    else:
        supplied = Path(runtime_path)
        path = supplied if supplied.is_absolute() else ROOT / supplied
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("student runtime must be a file inside the repository root")
    return path


def _verify_runtime_and_barriers(*, case_alias: str | None, all_preflight: bool,
                                 runtime_path: str | Path | None = None):
    if all_preflight == (case_alias is not None):
        raise ValueError("choose exactly one of --all-preflight or --case-alias")
    runtime_path = _resolve_runtime_path(runtime_path)
    source_runtime_path = BASE / "SOURCE_RUNTIME.json"
    teacher_runtime_path = BASE / "TEACHER_RUNTIME.json"
    roster_path = BASE / "PREFLIGHT_ROSTER_LOCK.json"
    source_barrier_path = BASE / "SOURCE_CAPTURE_BARRIER.json"
    for path in (runtime_path, source_runtime_path, teacher_runtime_path,
                 roster_path, source_barrier_path):
        if not path.is_file():
            raise FileNotFoundError(f"root-locked Table 1 input is missing: {path.name}")

    runtime = _read_json(runtime_path)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("STUDENT_RUNTIME must be locked and explicitly GT-free")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise RuntimeError("STUDENT_RUNTIME pins and protected maps are required")
    expected_own_sha = pins.get(OWN_REL)
    if expected_own_sha != _sha(ROOT / OWN_REL):
        raise RuntimeError("STUDENT_RUNTIME does not pin this exact student worker")
    for rel, expected in {**pins, **protected}.items():
        if not isinstance(rel, str) or not SHA_RE.fullmatch(str(expected)):
            raise ValueError("runtime pins must be relative paths with lowercase SHA-256 values")
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(ROOT.resolve()) or _sha(path) != expected:
            raise RuntimeError(f"STUDENT_RUNTIME pin mismatch: {rel}")

    source_runtime_sha = _sha(source_runtime_path)
    teacher_runtime_sha = _sha(teacher_runtime_path)
    source_runtime = _read_json(source_runtime_path)
    teacher_runtime = _read_json(teacher_runtime_path)
    for label, value in (("SOURCE_RUNTIME", source_runtime), ("TEACHER_RUNTIME", teacher_runtime)):
        if value.get("status") != "locked" or value.get("GT_read") is not False:
            raise RuntimeError(f"{label} must be locked and explicitly GT-free")
    if runtime.get("source_runtime_sha256", source_runtime_sha) != source_runtime_sha:
        raise RuntimeError("STUDENT_RUNTIME does not bind the exact SOURCE_RUNTIME")
    if runtime.get("teacher_runtime_sha256", teacher_runtime_sha) != teacher_runtime_sha:
        raise RuntimeError("STUDENT_RUNTIME does not bind the exact TEACHER_RUNTIME")

    roster = _read_json(roster_path)
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise RuntimeError("PREFLIGHT_ROSTER_LOCK is not locked and GT-free")
    roster_rows = roster.get("records")
    if not isinstance(roster_rows, list) or len(roster_rows) != 64:
        raise RuntimeError("Table 1 roster must contain the fixed 64 source probes")
    roster_index = {row.get("probe_alias"): row for row in roster_rows
                    if isinstance(row, Mapping) and isinstance(row.get("probe_alias"), str)}
    if set(roster_index) != ALIASES:
        raise RuntimeError("PREFLIGHT_ROSTER_LOCK aliases differ from the fixed 32+32 cohort")
    if case_alias is not None and case_alias not in roster_index:
        raise ValueError(f"unknown locked Table 1 alias: {case_alias}")

    source_barrier = _read_json(source_barrier_path)
    if (source_barrier.get("status") != "sealed" or source_barrier.get("count") != 64
            or source_barrier.get("GT_read") is not False):
        raise RuntimeError("SOURCE_CAPTURE_BARRIER is not the complete sealed 64-case barrier")
    source_barrier_sha = _sha(source_barrier_path)
    if runtime.get("source_barrier_sha256") != source_barrier_sha:
        raise RuntimeError("STUDENT_RUNTIME source_barrier_sha256 differs from the sealed source barrier")
    if source_barrier.get("source_runtime_sha256") != source_runtime_sha:
        raise RuntimeError("source barrier does not bind the current SOURCE_RUNTIME")
    source_files = source_barrier.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != ALIASES:
        raise RuntimeError("SOURCE_CAPTURE_BARRIER does not bind all fixed aliases")
    if source_barrier.get("preflight_roster_sha256") != _sha(roster_path):
        raise RuntimeError("source barrier does not bind PREFLIGHT_ROSTER_LOCK")

    teacher_candidates: list[tuple[Path, dict[str, Any]]] = []
    global_teacher_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    if global_teacher_path.is_file():
        global_doc = _read_json(global_teacher_path)
        if _sha(global_teacher_path) == runtime.get("teacher_barrier_sha256"):
            teacher_candidates.append((global_teacher_path, global_doc))
    if case_alias is not None:
        subset_path = BASE / f"TEACHER_{case_alias}_BARRIER.json"
        if subset_path.is_file() and _sha(subset_path) == runtime.get("teacher_barrier_sha256"):
            teacher_candidates.append((subset_path, _read_json(subset_path)))
    if not teacher_candidates:
        raise RuntimeError("no teacher barrier matches STUDENT_RUNTIME.teacher_barrier_sha256")
    # If a one-case qualification is explicitly requested and the runtime binds
    # that exact subset barrier, prefer it; otherwise the full sealed barrier is
    # used and only the requested alias is processed.
    teacher_path, teacher_barrier = teacher_candidates[-1] if case_alias is not None else teacher_candidates[0]
    if teacher_barrier.get("status") != "sealed" or teacher_barrier.get("GT_read") is not False:
        raise RuntimeError("selected teacher barrier is not sealed and GT-free")
    teacher_files = teacher_barrier.get("files")
    teacher_count = teacher_barrier.get("count")
    if not isinstance(teacher_files, list) or teacher_count != len(teacher_files):
        raise RuntimeError("teacher barrier file list/count is malformed")
    teacher_index = {row.get("alias"): row for row in teacher_files
                     if isinstance(row, Mapping) and isinstance(row.get("alias"), str)}
    if len(teacher_index) != len(teacher_files):
        raise RuntimeError("teacher barrier contains duplicate or invalid aliases")
    if teacher_path.name == "TEACHER_GLOBAL_BARRIER.json":
        if set(teacher_index) != ALIASES or teacher_count != 64:
            raise RuntimeError("full teacher barrier must bind all 64 fixed aliases")
        if all_preflight and set(teacher_index) != ALIASES:
            raise RuntimeError("--all-preflight requires the complete teacher barrier")
    elif case_alias is None or set(teacher_index) != {case_alias} or teacher_count != 1:
        raise RuntimeError("a subset teacher barrier is allowed only for its one requested alias")

    if all_preflight:
        if teacher_path.name != "TEACHER_GLOBAL_BARRIER.json" or teacher_count != 64:
            raise RuntimeError("--all-preflight requires TEACHER_GLOBAL_BARRIER.json with 64 cases")
        selected = sorted(ALIASES)
    else:
        selected = [str(case_alias)]

    # When present, runtime identity fields are checked as well as the file
    # hashes in receipts.  The barrier hashes are the required source of truth.
    if runtime.get("teacher_barrier_sha256") != _sha(teacher_path):
        raise RuntimeError("selected teacher barrier differs from runtime binding")
    if runtime.get("source_runtime_sha256", source_runtime_sha) != source_runtime_sha:
        raise RuntimeError("student runtime source-runtime binding changed")
    if runtime.get("teacher_runtime_sha256", teacher_runtime_sha) != teacher_runtime_sha:
        raise RuntimeError("student runtime teacher-runtime binding changed")

    return {
        "runtime": runtime,
        "runtime_sha256": _sha(runtime_path),
        "source_runtime_sha256": source_runtime_sha,
        "teacher_runtime_sha256": teacher_runtime_sha,
        "roster": roster_index,
        "source_barrier": source_barrier,
        "source_barrier_sha256": source_barrier_sha,
        "source_files": source_files,
        "teacher_barrier": teacher_barrier,
        "teacher_barrier_path": teacher_path,
        "teacher_barrier_sha256": _sha(teacher_path),
        "teacher_files": teacher_index,
        "selected_aliases": selected,
    }


def _safe_locked_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("sealed artifact path must be a nonempty relative string")
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError("sealed artifact path must be safe and root-relative")
    path = (ROOT / raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("sealed artifact path escapes repository root")
    return path


def _verify_source_file(alias: str, item: Mapping[str, Any], roster: Mapping[str, Any]):
    path = _safe_locked_path(item.get("path"))
    if _sha(path) != item.get("sha256") or path.stat().st_size != item.get("bytes"):
        raise RuntimeError(f"source artifact bytes differ from barrier for {alias}")
    receipt_path = _safe_locked_path(item.get("receipt_path"))
    if _sha(receipt_path) != item.get("receipt_sha256"):
        raise RuntimeError(f"source receipt differs from barrier for {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("case_id") != alias or receipt.get("sha256") != item.get("sha256")
            or receipt.get("bytes") != item.get("bytes") or receipt.get("GT_read") is not False
            or receipt.get("source_runtime_sha256") != _sha(BASE / "SOURCE_RUNTIME.json")
            or receipt.get("query_key_sha256") != roster.get("query_key_sha256")):
        raise RuntimeError(f"source receipt binding differs from roster/barrier for {alias}")
    if item.get("source_case_sha256") != receipt.get("source_case_sha256"):
        raise RuntimeError(f"source semantic SHA differs from receipt for {alias}")
    return path, receipt


def _verify_teacher_file(alias: str, item: Mapping[str, Any], source_file_sha256: str,
                         teacher_runtime_sha256: str):
    path = _safe_locked_path(item.get("path"))
    if _sha(path) != item.get("sha256"):
        raise RuntimeError(f"teacher artifact bytes differ from barrier for {alias}")
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"teacher receipt is missing for {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("status") != "sealed" or receipt.get("sha256") != item.get("sha256")
            or receipt.get("bytes") != path.stat().st_size or receipt.get("GT_read") is not False
            or receipt.get("source_sha256") != source_file_sha256
            or receipt.get("runtime_sha256") != teacher_runtime_sha256):
        raise RuntimeError(f"teacher receipt does not bind its sealed source for {alias}")
    return path, receipt


def _load_source_payload(path: Path, alias: str, expected_source_case_sha256: str,
                         roster_record: Mapping[str, Any], torch: Any,
                         source_module: Any) -> dict[str, Any]:
    case = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(case, dict) or case.get("schema") != "stvg-final-table1-source-case-v1":
        raise RuntimeError(f"source payload schema is invalid for {alias}")
    if (case.get("case_id") != alias or case.get("dataset") != roster_record.get("dataset")
            or case.get("query_key_sha256") != roster_record.get("query_key_sha256")
            or case.get("source_case_sha256") != expected_source_case_sha256
            or source_module.content_sha256(case) != expected_source_case_sha256
            or case.get("GT_read") is not False or case.get("DINO_read_or_call") is not False):
        raise RuntimeError(f"source payload semantic binding differs for {alias}")
    return case


def _load_teacher_payload(path: Path, alias: str, source_file_sha256: str,
                          source_case_sha256: str, source_receipt: Mapping[str, Any],
                          teacher_receipt: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"teacher payload is not a mapping for {alias}")
    if (value.get("case_id") != alias or value.get("GT_read") is not False
            or value.get("query_reset") is not True
            or value.get("source_sha256") != source_file_sha256
            or value.get("request_sha256") != teacher_receipt.get("request_sha256")
            or value.get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")):
        raise RuntimeError(f"teacher payload binding differs from its sealed receipt for {alias}")
    temporal = value.get("temporal")
    reference = value.get("reference")
    who = value.get("who")
    if not isinstance(temporal, Mapping) or not isinstance(reference, Mapping) or not isinstance(who, Mapping):
        raise RuntimeError(f"teacher payload lacks temporal/WHO/reference records for {alias}")
    if (reference.get("case_id") != alias or reference.get("dataset") != value.get("dataset")
            or reference.get("GT_read") is not False or temporal.get("GT_read") is not False):
        raise RuntimeError(f"teacher reference is not bound to the expected GT-free query for {alias}")
    if source_receipt.get("source_case_sha256") != source_case_sha256:
        raise RuntimeError(f"teacher/source semantic case binding differs for {alias}")
    return value


def _teacher_identity_summary(teacher_case: Mapping[str, Any]) -> dict[str, Any]:
    who = teacher_case["who"].get("who", {})
    generation = teacher_case["who"].get("generation", {})
    status = who.get("status") if isinstance(who, Mapping) else None
    selected_id = who.get("candidate_id") if status == "valid" else None
    ref = teacher_case["reference"]
    if selected_id != ref.get("selected_object_id"):
        raise RuntimeError("WHO candidate ID differs from the sealed Spatial-A reference")
    return {
        "status": status,
        "selected_object_id": selected_id,
        "reason": who.get("reason") if isinstance(who, Mapping) else None,
        "input_tokens": generation.get("input_tokens"),
        "output_tokens": generation.get("output_tokens"),
        "visual_tokens": generation.get("visual_tokens"),
        "output_truncated_at_cap": generation.get("output_truncated_at_cap"),
        "generation_call_count": generation.get("generation_call_count"),
    }


def _cached_complete_result(alias: str, path: Path, receipt_path: Path,
                            receipt: Mapping[str, Any], bindings: Mapping[str, Any], torch: Any):
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete prior student case must be preserved for recovery: {alias}")
    if _sha(path) != receipt.get("sha256") or _sha(receipt_path) != bindings["receipt_sha256"]:
        raise RuntimeError(f"prior student case/receipt bytes differ for {alias}")
    if (receipt.get("status") != "sealed" or receipt.get("case_id") != alias
            or receipt.get("GT_read") is not False
            or receipt.get("runtime_sha256") != bindings["runtime_sha256"]
            or receipt.get("source_file_sha256") != bindings["source_file_sha256"]
            or receipt.get("teacher_file_sha256") != bindings["teacher_file_sha256"]
            or receipt.get("source_case_sha256") != bindings["source_case_sha256"]):
        raise RuntimeError(f"prior student receipt does not match locked inputs for {alias}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("case_id") != alias or saved.get("GT_read") is not False
            or saved.get("source_case_sha256") != bindings["source_case_sha256"]
            or saved.get("teacher_file_sha256") != bindings["teacher_file_sha256"]
            or saved.get("student_runtime_sha256") != bindings["runtime_sha256"]):
        raise RuntimeError(f"prior student payload binding differs for {alias}")
    return saved


def _make_result(*, alias: str, roster_record: Mapping[str, Any], source_case: Mapping[str, Any],
                 source_file_sha256: str, teacher_case: Mapping[str, Any],
                 teacher_file_sha256: str, teacher_receipt: Mapping[str, Any],
                 student_runtime_sha256: str, source_runtime_sha256: str,
                 teacher_runtime_sha256: str, source_barrier_sha256: str,
                 teacher_barrier_sha256: str, model: Any, current_source: str | None,
                 torch: Any, np: Any, collections_module: Any, model_for: Any,
                 frames_for: Any, raw_row: Any, frozen_forward: Any,
                 prediction: Any, state_hash: Any, core: Any, spatial: Any,
                 fit_validated: Any, transient_native_masks: Any,
                 frozen_visual_host: Any, frozen_attention_storage: Any,
                 inactive_modules_on_host: Any, query_subject: Any,
                 checkpoint_callback: Any = None):
    import copy

    # _capture_source is the already-audited cold source replay.  The adapter
    # catalog adds only transient runtime aliases needed by its legacy reader;
    # the sealed source payload and its semantic hash remain untouched.
    from scripts import run_stvg_adjacent_motion_student_v1 as source_capture
    from scripts import run_stvg_table1_source_v1 as source_module

    model_name = source_module.SOURCE_MODEL_NAME[str(roster_record["dataset"])]
    case_for_capture = dict(source_case)
    case_for_capture["source"] = model_name
    case_for_capture["_source_case_sha256"] = source_case["source_case_sha256"]
    _, official_row = source_module._validate_roster_record(roster_record)
    case_for_capture["parent"] = official_row["source"]

    rec = {
        "case_id": alias,
        "dataset": roster_record["dataset"],
        "source_case_sha256": source_case["source_case_sha256"],
    }

    def catalog(record):
        if record["case_id"] != alias or record["source_case_sha256"] != source_case["source_case_sha256"]:
            raise RuntimeError("transient source catalog request differs from current case")
        return case_for_capture, {}

    source_capture.catalog = catalog
    (model, current_source, case, _catalog, row, frames, frame_ids, records,
     batch, base, native, expected_state, pixel_sha, capture_seconds) = source_capture._capture_source(
        rec, model, current_source, torch, collections_module,
        model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
    )
    if (case["source_case_sha256"] != source_case["source_case_sha256"]
            or case["query_key_sha256"] != roster_record["query_key_sha256"]
            or case["dataset"] != roster_record["dataset"]):
        raise RuntimeError("cold replay source identity differs from the locked roster")
    if expected_state != source_case["source_model_state_sha256"]:
        raise RuntimeError("cold model full-state hash differs from the sealed source case")
    if pixel_sha != source_case["pixel_sha256"] or list(frame_ids) != list(map(int, source_case["frame_ids"])):
        raise RuntimeError("cold decoded frames differ from source pixel/grid bindings")

    native_boxes = source_capture._xyxy(native["boxes"])
    cached_xyxy = np.asarray(source_case["native_boxes_xyxy_normalized"], dtype=np.float64)
    if not np.array_equal(native_boxes, cached_xyxy):
        raise RuntimeError("cold native xyxy readout differs from the sealed source case")
    if not source_capture._same_logits(native["logits"], source_case["zero"]["logits"]):
        raise RuntimeError("cold native source logits differ from the sealed source case")

    temporal = teacher_case["temporal"]
    reference = teacher_case["reference"]
    native_interval = _check_interval(case["Native_interval"], case["video_bounds"], "Native")
    projection_interval = _check_interval(case["Projection_interval"], case["video_bounds"], "Projection")
    temporal_interval = _check_interval(temporal.get("final_interval"), case["video_bounds"], "Temporal")
    if not np.isfinite(native_boxes).all():
        raise RuntimeError("source boxes contain nonfinite values")

    geometry = reference.get("teacher_geometry")
    if not isinstance(geometry, Mapping):
        raise RuntimeError("locked spatial reference lacks the TeacherGeometry arm")
    if list(map(int, geometry.get("frame_ids_physical", []))) != list(map(int, frame_ids)):
        raise RuntimeError("TeacherGeometry frame IDs differ from the exact source grid")
    teacher_geometry_boxes = np.asarray(geometry.get("boxes_xyxy_normalized"), dtype=np.float64)
    if teacher_geometry_boxes.shape != native_boxes.shape:
        raise RuntimeError("TeacherGeometry boxes differ in shape from the source grid")
    geometry_interval = list(map(float, geometry.get("interval_frame_bounds_half_open", [])))
    if geometry_interval != temporal_interval:
        raise RuntimeError("TeacherGeometry interval differs from the frozen Temporal interval")
    no_update = reference.get("no_update")
    anchors_rows = reference.get("anchors")
    if not isinstance(no_update, bool) or not isinstance(anchors_rows, list):
        raise RuntimeError("locked reference no_update/anchors schema is invalid")
    if no_update != (len(anchors_rows) == 0):
        raise RuntimeError("locked reference no_update flag conflicts with supported anchors")
    if not no_update:
        center_slots = [int(row["slot"]) for row in reference.get("centers", [])]
        if center_slots != [0, 1, 2, 3]:
            raise RuntimeError("updating reference must retain exactly the four fixed QA centers")
        if len(anchors_rows) > 4:
            raise RuntimeError("Spatial-A reference exceeds the four fixed center slots")

    from scripts.stvg_native_instance_common_v1 import sha as _sha_common
    # The complete teacher case is the sealed carrier of this reference; its
    # global barrier and receipt bind source/request/tracking inputs.
    reference_sha = teacher_file_sha256
    fit = None
    readout = None
    fit_key = stable_fit_key(str(case["dataset"]), str(case["query_key_sha256"]))
    cfg = core.load_original_opd_config(str(case["dataset"]))
    cfg["steps"] = 5
    initial = {name: value.detach().clone() for name, value in base.initial.items()}
    spatial_boxes = native_boxes.copy()
    fit_elapsed = 0.0
    readout_elapsed = 0.0
    fit_calls = 0
    final_redecode_calls = 0

    if no_update:
        if anchors_rows:
            raise RuntimeError("exact no-update reference unexpectedly contains anchors")
        # Exact Frozen reuse: neither fit nor final model readout is invoked.
        if not source_capture._same_logits(native["logits"], source_case["zero"]["logits"]):
            raise RuntimeError("no-update source logits changed before output sealing")
    else:
        anchors = spatial._anchor_tensors(reference, frame_ids, torch)
        if len(anchors) != len(anchors_rows) or not anchors:
            raise RuntimeError("Spatial-A anchor tensor conversion changed the locked support")
        for cooked, raw in zip(anchors, anchors_rows):
            if (cooked["position"] != int(raw["position"])
                    or cooked["physical_frame_id"] != int(raw["physical_frame_id"])
                    or cooked["weight"] != float(raw["weight"])):
                raise RuntimeError("Spatial-A tensor anchor differs from the sealed reference")
        fit_started = time.perf_counter()
        # The original A math is fit_validated(validate=True).  Its RNG key is
        # query-content stable, not an engineering alias, so preflight and
        # full-query aliases cannot change Gaussian sampling.
        fit = fit_validated(
            base, initial, anchors, frame_ids, fit_key, cfg, validate=True,
            reference_sha256=reference_sha, checkpoint_callback=checkpoint_callback,
        )
        fit_elapsed = time.perf_counter() - fit_started
        if fit.get("seed_key") != fit_key or fit.get("reference_sha256") != reference_sha:
            raise RuntimeError("Spatial-A fit receipt lost its stable key/reference binding")
        if int(fit.get("active_parameters", -1)) != 1792 or int(fit.get("declared_steps", -1)) != 5:
            raise RuntimeError("Spatial-A fit changed the original 1792-parameter/five-step contract")
        if state_hash(model.state_dict()) != expected_state:
            raise RuntimeError("source model state changed during Spatial-A fitting")
        readout_started = time.perf_counter()
        readout = source_capture._readout(
            model, batch, frame_ids, records, fit, row["parses"]["subject"],
            core, torch, transient_native_masks=transient_native_masks,
            frozen_visual_host=frozen_visual_host,
            frozen_attention_storage=frozen_attention_storage,
            inactive_modules_on_host=inactive_modules_on_host,
            query_subject=query_subject,
        )
        readout_elapsed = time.perf_counter() - readout_started
        spatial._assert_readout_matches_fit(readout, fit)
        if not source_capture._same_logits(readout["logits"], native["logits"]):
            raise RuntimeError("Spatial-A full readout changed frozen source logits")
        if state_hash(model.state_dict()) != expected_state:
            raise RuntimeError("source model full state was not restored after Spatial-A readout")
        spatial_boxes = source_capture._xyxy(readout["boxes"])
        fit_calls = 1
        final_redecode_calls = 1

    arms = _compose_arms(
        native_boxes, native_interval, temporal_interval, spatial_boxes,
        teacher_geometry_boxes, expected_count=len(frame_ids),
    )
    if no_update:
        if not np.array_equal(arms["F"]["boxes"], arms["S"]["boxes"]):
            raise RuntimeError("no-update Spatial-only must be exact Frozen reuse")
        if not np.array_equal(arms["T"]["boxes"], arms["TS"]["boxes"]):
            raise RuntimeError("no-update Temporal+Spatial must be exact Temporal reuse")
        if fit is not None or readout is not None or fit_calls or final_redecode_calls:
            raise RuntimeError("no-update reference performed a fit/readout")
    if state_hash(model.state_dict()) != expected_state:
        raise RuntimeError("source model full state is not restored before serialization")
    # Reading the old cache receipt through its existing helper above preserves
    # exact source/reset checks. Projection is recorded for provenance only.
    result = {
        "schema": "stvg-final-table1-student-case-v1",
        "case_id": alias,
        "dataset": str(case["dataset"]),
        "query_key_sha256": str(case["query_key_sha256"]),
        "source_case_sha256": str(case["source_case_sha256"]),
        "source_file_sha256": source_file_sha256,
        "source_model_state_sha256": expected_state,
        "source_checkpoint_sha256": case.get("source_checkpoint_sha256"),
        "pixel_sha256": pixel_sha,
        "frame_ids": list(map(int, frame_ids)),
        "Native_interval": list(case["Native_interval"]),
        "Projection_interval": list(case["Projection_interval"]),
        "Temporal_interval": list(temporal.get("final_interval")),
        "arms": arms,
        "fits": {"SpatialA": fit} if fit is not None else {},
        "readouts": {"SpatialA": core._copy_cpu(readout)} if readout is not None else {},
        "reference_record": _plain(reference),
        "reference_sha256": reference_sha,
        "teacher_selection": _teacher_identity_summary(teacher_case),
        "temporal_summary": _safe_temporal_summary(temporal),
        "source_light_initial_state": core._copy_cpu(base.initial),
        "source_boxes_cxcywh": core._copy_cpu(native["boxes"]),
        "source_logits": core._copy_cpu(native["logits"]),
        "source_cold_readback": {
            "status": "pass",
            "source_case_sha256": case["source_case_sha256"],
            "source_model_state_sha256": expected_state,
            "pixel_sha256": pixel_sha,
            "frame_ids_match": True,
            "initial_state_match": True,
            "native_xyxy_match": True,
            "source_logits_bitwise": True,
            "full_model_state_restored": True,
        },
        "reference_receipt": {
            "teacher_file_sha256": teacher_file_sha256,
            "teacher_receipt_sha256": _sha_common(Path(teacher_receipt["_receipt_path"])),
            "request_sha256": teacher_case.get("request_sha256"),
            "tracking_binding_sha256": teacher_case.get("tracking_binding_sha256"),
            "no_update": no_update,
            "no_update_reason": reference.get("no_update_reason"),
            "anchor_count": len(anchors_rows),
        },
        "source_runtime_sha256": source_runtime_sha256,
        "teacher_runtime_sha256": teacher_runtime_sha256,
        "student_runtime_sha256": student_runtime_sha256,
        "source_barrier_sha256": source_barrier_sha256,
        "teacher_barrier_sha256": teacher_barrier_sha256,
        "configuration": {
            "fit_key": fit_key,
            "fit_key_basis": "dataset plus original query_key_sha256; alias-independent",
            "fit_scope": "joint",
            "fit_algorithm": "original_loss_weighted_iou_with_reference_non_decrease_gate",
            "steps": 5,
            "active_parameters": 1792,
            "config": dict(cfg),
            "validation": "same-reference weighted decoded-box IoU must not decrease by more than 1e-7; reject restores parameters and Adam then stops",
        },
        "execution": {
            "source_capture_calls": 1,
            "source_offset_forwards": 2,
            "actual_fit_count": fit_calls,
            "optimizer_steps": int(fit.get("optimizer_steps", 0)) if fit is not None else 0,
            "proposed_optimizer_steps": int(fit.get("proposed_optimizer_steps", 0)) if fit is not None else 0,
            "rejected_optimizer_steps": int(fit.get("rejected_optimizer_steps", 0)) if fit is not None else 0,
            "final_full_model_redecode_calls": final_redecode_calls,
            "source_capture_seconds": float(capture_seconds),
            "spatial_fit_seconds": float(fit_elapsed),
            "full_readout_seconds": float(readout_elapsed),
            "source_runtime_model_name": model_name,
            "source_runtime_model_hot": True,
            "teacher_model_calls": 0,
            "GT_read": False,
        },
        "GT_read": False,
        "online_carry": False,
    }
    return result, model, model_name


def _contracts() -> dict[str, Any]:
    import numpy as np

    sha_a = "a" * 64
    sha_b = "b" * 64
    key_a = stable_fit_key("hc2", sha_a)
    key_b = stable_fit_key("hc2", sha_a)
    key_c = stable_fit_key("vidstg", sha_a)
    assert key_a == key_b and key_a != key_c
    assert key_a == "caption_query_fixed_space|hc2_" + sha_a
    native = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.4, 0.5]]
    fitted = [[0.11, 0.2, 0.31, 0.4], [0.2, 0.25, 0.4, 0.55]]
    teacher = [[0.15, 0.2, 0.3, 0.4], [0.2, 0.2, 0.4, 0.5]]
    rows = _compose_arms(native, [0, 2], [1, 3], fitted, teacher, expected_count=2)
    assert rows["F"]["interval"] == [0.0, 2.0]
    assert rows["T"]["interval"] == [1.0, 3.0]
    assert rows["S"]["interval"] == [0.0, 2.0]
    assert rows["TS"]["interval"] == [1.0, 3.0]
    assert rows["TeacherGeometry"]["interval"] == [1.0, 3.0]
    assert np.array_equal(rows["S"]["boxes"], rows["TS"]["boxes"])
    assert np.array_equal(rows["F"]["boxes"], rows["T"]["boxes"])
    no_update = _compose_arms(native, [0, 2], [1, 3], native, native,
                              expected_count=2)
    assert np.array_equal(no_update["F"]["boxes"], no_update["S"]["boxes"])
    assert np.array_equal(no_update["T"]["boxes"], no_update["TS"]["boxes"])
    assert _reference_sha256({"case_id": "HC01", "anchors": []}) != _reference_sha256(
        {"case_id": "HC01", "anchors": [{"position": 1}]})
    try:
        _compose_arms(native, [0, 2], [1, 3], fitted, [[0.8, 0.8, 0.2, 0.2]],
                      expected_count=2)
    except ValueError:
        invalid_geometry_rejected = True
    else:
        raise AssertionError("illegal TeacherGeometry box was accepted")
    try:
        stable_fit_key("unknown", sha_b)
    except ValueError:
        invalid_dataset_rejected = True
    else:
        raise AssertionError("unknown dataset fit key was accepted")
    return {
        "status": "pass",
        "synthetic_contracts": {
            "alias_independent_query_key": True,
            "dataset_separates_fit_key": True,
            "five_arm_geometry_and_interval_composition": True,
            "shared_S_TS_spatial_prediction": True,
            "F_T_source_geometry_reuse": True,
            "no_update_exact_arm_reuse": True,
            "reference_content_hash_changes_with_reference": True,
            "illegal_teacher_geometry_rejected": invalid_geometry_rejected,
            "unsupported_dataset_rejected": invalid_dataset_rejected,
        },
        "GT_read": False,
        "CUDA_used": False,
        "model_imported": False,
    }


def _verify_complete_output(path: Path, receipt_path: Path, expected: Mapping[str, Any], torch):
    if not path.is_file() or not receipt_path.is_file():
        if path.exists() or receipt_path.exists():
            raise RuntimeError(f"incomplete prior student output must be preserved: {path.name}")
        return None
    receipt = _read_json(receipt_path)
    if (_sha(path) != receipt.get("sha256") or path.stat().st_size != receipt.get("bytes")
            or _sha(receipt_path) != expected.get("receipt_sha256")):
        raise RuntimeError(f"prior student artifact/receipt differs from its seal: {path.name}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "case_id": expected["case_id"],
        "source_case_sha256": expected["source_case_sha256"],
        "source_file_sha256": expected["source_file_sha256"],
        "teacher_file_sha256": expected["teacher_file_sha256"],
        "student_runtime_sha256": expected["student_runtime_sha256"],
        "GT_read": False,
    }
    if any(saved.get(key) != value for key, value in checks.items()):
        raise RuntimeError(f"prior student payload differs from locked inputs: {path.name}")
    if receipt.get("GT_read") is not False or receipt.get("status") != "sealed":
        raise RuntimeError(f"prior student receipt is not sealed and GT-free: {path.name}")
    return receipt


def run(*, case_alias: str | None = None, all_preflight: bool = False,
        runtime_path: str | Path | None = None) -> dict[str, Any]:
    import collections
    import gc
    import numpy as np
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for, frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward, transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import run_stvg_table1_source_v1 as source_module
    from scripts import run_stvg_adjacent_motion_student_v1 as source_capture
    from scripts import stvg_table1_spatial_adapter_v1 as spatial
    from scripts.stvg_instance_validated_fit_v1 import fit_validated
    from scripts.stvg_temporal_caption_query_v1 import inference_guard

    locked = _verify_runtime_and_barriers(case_alias=case_alias, all_preflight=all_preflight,
                                          runtime_path=runtime_path)
    runtime = locked["runtime"]
    roster_index = locked["roster"]
    source_files = locked["source_files"]
    teacher_files = locked["teacher_files"]
    aliases = locked["selected_aliases"]
    out_dir = BASE / "student_preflight_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    barrier_path = BASE / ("STUDENT_GLOBAL_BARRIER.json" if all_preflight
                           else f"STUDENT_{case_alias}_BARRIER.json")
    if barrier_path.exists():
        raise FileExistsError(f"preserve existing student barrier: {barrier_path}")

    install_clean_loader()
    sys.addaudithook(inference_guard)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The stochastic policy is keyed by dataset + query hash below.  The
    # process seed is fixed only for any original-library code still using the
    # global stream and is identical for qualification/full execution.
    torch.manual_seed(20261015)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(20261015)
    np.random.seed(20261015)

    model = None
    current_source = None
    file_rows = []
    reused_count = 0
    actual_capture_count = 0
    total_fit_count = 0
    total_optimizer_steps = 0
    total_readout_count = 0
    started = time.perf_counter()
    _write_json(BASE / "STUDENT_STATUS.json", {
        "status": "running", "stage": "source_reset_student", "done": 0,
        "total": len(aliases), "case_alias": case_alias,
        "student_runtime_sha256": locked["runtime_sha256"], "GT_read": False,
        "time": time.time(),
    }, replace=(BASE / "STUDENT_STATUS.json").exists())

    try:
        for index, alias in enumerate(aliases, 1):
            roster_record = roster_index[alias]
            source_item = source_files[alias]
            teacher_item = teacher_files[alias]
            source_path, source_receipt = _verify_source_file(alias, source_item, roster_record)
            teacher_path, teacher_receipt = _verify_teacher_file(
                alias, teacher_item, source_item["sha256"], locked["teacher_runtime_sha256"])
            teacher_receipt["_receipt_path"] = str(teacher_path.with_suffix(".receipt.json"))
            destination = out_dir / f"{alias}.pt"
            receipt_path = destination.with_suffix(".receipt.json")
            receipt_binding = {
                "case_id": alias,
                "source_case_sha256": source_item["source_case_sha256"],
                "source_file_sha256": source_item["sha256"],
                "teacher_file_sha256": teacher_item["sha256"],
                "student_runtime_sha256": locked["runtime_sha256"],
                "receipt_sha256": _sha(receipt_path) if receipt_path.is_file() else None,
            }
            if destination.exists() or receipt_path.exists():
                prior = _verify_complete_output(destination, receipt_path, receipt_binding, torch)
                if prior is None:
                    raise AssertionError("partial existing case path was not rejected")
                saved = torch.load(destination, map_location="cpu", weights_only=False)
                file_rows.append({
                    "alias": alias,
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": _sha(destination),
                    "bytes": destination.stat().st_size,
                    "receipt": {"path": str(receipt_path.relative_to(ROOT)),
                                "sha256": _sha(receipt_path), "bytes": receipt_path.stat().st_size},
                    "reused": True,
                })
                reused_count += 1
                total_fit_count += int(saved["execution"]["actual_fit_count"])
                total_optimizer_steps += int(saved["execution"]["optimizer_steps"])
                total_readout_count += int(saved["execution"]["final_full_model_redecode_calls"])
                _write_json(BASE / "STUDENT_STATUS.json", {
                    "status": "running", "stage": "source_reset_student", "done": index,
                    "total": len(aliases), "current_alias": alias,
                    "student_runtime_sha256": locked["runtime_sha256"], "GT_read": False,
                    "time": time.time(),
                }, replace=True)
                continue

            source_case = _load_source_payload(
                source_path, alias, str(source_item["source_case_sha256"]),
                roster_record, torch, source_module,
            )
            teacher_case = _load_teacher_payload(
                teacher_path, alias, source_item["sha256"],
                str(source_item["source_case_sha256"]), source_receipt,
                teacher_receipt, torch,
            )
            if teacher_case.get("dataset") != roster_record.get("dataset"):
                raise RuntimeError(f"teacher dataset differs from locked roster for {alias}")
            result, model, current_source = _make_result(
                alias=alias, roster_record=roster_record, source_case=source_case,
                source_file_sha256=source_item["sha256"], teacher_case=teacher_case,
                teacher_file_sha256=teacher_item["sha256"],
                teacher_receipt=teacher_receipt,
                student_runtime_sha256=locked["runtime_sha256"],
                source_runtime_sha256=locked["source_runtime_sha256"],
                teacher_runtime_sha256=locked["teacher_runtime_sha256"],
                source_barrier_sha256=locked["source_barrier_sha256"],
                teacher_barrier_sha256=locked["teacher_barrier_sha256"],
                model=model, current_source=current_source, torch=torch, np=np,
                collections_module=collections, model_for=model_for,
                frames_for=frames_for, raw_row=raw_row,
                frozen_forward=frozen_forward, prediction=prediction,
                state_hash=state_hash, core=core, spatial=spatial,
                fit_validated=fit_validated,
                transient_native_masks=transient_native_masks,
                frozen_visual_host=frozen_visual_host,
                frozen_attention_storage=frozen_attention_storage,
                inactive_modules_on_host=inactive_modules_on_host,
                query_subject=query_subject,
            )
            # Teacher receipt metadata only; raw response text is never copied
            # into the student artifact or progress log.
            result["request_sha256"] = teacher_case["request_sha256"]
            result["tracking_binding_sha256"] = teacher_case["tracking_binding_sha256"]
            result["teacher_receipt_sha256"] = _sha(teacher_path.with_suffix(".receipt.json"))
            if result.get("GT_read") is not False:
                raise RuntimeError("student result lost its GT-free contract")
            temp = destination.with_name(destination.name + f".{os.getpid()}.tmp")
            torch.save(result, temp)
            os.replace(temp, destination)
            receipt_value = {
                "schema": "stvg-final-table1-student-receipt-v1",
                "status": "sealed",
                "case_id": alias,
                "sha256": _sha(destination),
                "bytes": destination.stat().st_size,
                "source_case_sha256": source_item["source_case_sha256"],
                "source_file_sha256": source_item["sha256"],
                "teacher_file_sha256": teacher_item["sha256"],
                "teacher_receipt_sha256": _sha(teacher_path.with_suffix(".receipt.json")),
                "source_runtime_sha256": locked["source_runtime_sha256"],
                "teacher_runtime_sha256": locked["teacher_runtime_sha256"],
                "student_runtime_sha256": locked["runtime_sha256"],
                "source_barrier_sha256": locked["source_barrier_sha256"],
                "teacher_barrier_sha256": locked["teacher_barrier_sha256"],
                "request_sha256": result["request_sha256"],
                "tracking_binding_sha256": result["tracking_binding_sha256"],
                "source_capture_calls": result["execution"]["source_capture_calls"],
                "actual_fit_count": result["execution"]["actual_fit_count"],
                "optimizer_steps": result["execution"]["optimizer_steps"],
                "final_full_model_redecode_calls": result["execution"]["final_full_model_redecode_calls"],
                "GT_read": False,
                "time": time.time(),
            }
            _write_json(receipt_path, receipt_value)
            file_rows.append({
                "alias": alias,
                "path": str(destination.relative_to(ROOT)),
                "sha256": receipt_value["sha256"],
                "bytes": receipt_value["bytes"],
                "receipt": {"path": str(receipt_path.relative_to(ROOT)),
                            "sha256": _sha(receipt_path), "bytes": receipt_path.stat().st_size},
                "reused": False,
            })
            actual_capture_count += 1
            total_fit_count += int(result["execution"]["actual_fit_count"])
            total_optimizer_steps += int(result["execution"]["optimizer_steps"])
            total_readout_count += int(result["execution"]["final_full_model_redecode_calls"])
            _write_json(BASE / "STUDENT_STATUS.json", {
                "status": "running", "stage": "source_reset_student", "done": index,
                "total": len(aliases), "current_alias": alias,
                "student_runtime_sha256": locked["runtime_sha256"], "GT_read": False,
                "time": time.time(),
            }, replace=True)
            print(f"TABLE1_STUDENT {index}/{len(aliases)} {alias} "
                  f"fit={result['execution']['actual_fit_count']} "
                  f"steps={result['execution']['optimizer_steps']}", flush=True)
            del source_case, teacher_case, result
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

        if model is not None:
            expected_sha = model._fixed_full_state_hash
            if state_hash(model.state_dict()) != expected_sha:
                raise RuntimeError("hot source model state differs after the last query")
        if set(row["alias"] for row in file_rows) != set(aliases):
            raise RuntimeError("student output count differs from the selected locked aliases")
        if all_preflight and len(file_rows) != 64:
            raise RuntimeError("complete student barrier requires all 64 sealed cases")
        barrier = {
            "schema": "stvg-final-table1-student-global-barrier-v1" if all_preflight
                      else "stvg-final-table1-student-case-barrier-v1",
            "status": "sealed",
            "scope": "all_preflight" if all_preflight else "case_qualification",
            "count": len(file_rows),
            "files": file_rows,
            "source_barrier_sha256": locked["source_barrier_sha256"],
            "teacher_barrier_sha256": locked["teacher_barrier_sha256"],
            "source_runtime_sha256": locked["source_runtime_sha256"],
            "teacher_runtime_sha256": locked["teacher_runtime_sha256"],
            "student_runtime_sha256": locked["runtime_sha256"],
            "actual_source_capture_calls": actual_capture_count,
            "reused_student_cases": reused_count,
            "actual_fit_count": total_fit_count,
            "optimizer_steps": total_optimizer_steps,
            "final_full_model_redecode_calls": total_readout_count,
            "teacher_calls": 0,
            "DINO_calls": 0,
            "GT_read": False,
            "wall_seconds": time.perf_counter() - started,
            "time": time.time(),
        }
        _write_json(barrier_path, barrier)
        _write_json(BASE / "STUDENT_STATUS.json", {
            "status": "sealed", "stage": "source_reset_student", "done": len(file_rows),
            "total": len(aliases), "barrier_path": str(barrier_path.relative_to(ROOT)),
            "barrier_sha256": _sha(barrier_path), "GT_read": False, "time": time.time(),
        }, replace=True)
        return barrier
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-alias", choices=sorted(ALIASES))
    group.add_argument("--all-preflight", action="store_true")
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument(
        "--runtime",
        help="optional root-relative locked student runtime JSON (default: artifacts/stvg_final_table1_v1/STUDENT_RUNTIME.json)",
    )
    args = parser.parse_args()
    if args.contracts:
        if args.case_alias is not None or args.all_preflight:
            parser.error("--contracts cannot be combined with execution scope")
        print(json.dumps(_contracts(), ensure_ascii=False, sort_keys=True))
        return
    if args.case_alias is None and not args.all_preflight:
        parser.error("choose --case-alias or --all-preflight")
    result = run(case_alias=args.case_alias, all_preflight=args.all_preflight,
                 runtime_path=args.runtime)
    print(json.dumps({
        "status": result["status"], "scope": result["scope"], "count": result["count"],
        "barrier_sha256": _sha(BASE / ("STUDENT_GLOBAL_BARRIER.json" if args.all_preflight
                                        else f"STUDENT_{args.case_alias}_BARRIER.json")),
        "GT_read": False,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
