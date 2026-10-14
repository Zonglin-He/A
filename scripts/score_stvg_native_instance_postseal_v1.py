#!/usr/bin/env python3
"""Independent post-seal scorer for the native-instance spatial OPD experiment.

This file is deliberately inert at import.  The official scoring path is gated
by ``--score-after-seal`` and verifies the complete 37-file prediction barrier
before loading source cases or calling the official GT reader.  It creates no
student, teacher, or tracking outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
SEAL = BASE / "GLOBAL_PREDICTION_BARRIER.json"
ROSTER = BASE / "ROSTER_LOCK.json"
REFERENCE_LOCK = BASE / "REFERENCE_LOCK.json"
DIRECT_READOUT = BASE / "DIRECT_TEACHER_READOUT.json"
PHASES = ("TRACKING", "TEACHER", "STUDENT")
STUDENT_ALGORITHMS = ("direct", "iou", "reliability_weighted_iou")
ARMS = (
    "F", "T", "native_teacher_direct", "direct", "iou",
    "reliability_weighted_iou", "T_direct", "T_iou",
    "T_reliability_weighted_iou",
)
PRIMARY_ARMS = (
    "T", "native_teacher_direct", "T_direct", "T_iou",
    "T_reliability_weighted_iou",
)
METRICS = (
    "m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261014
SIGN_EPSILON = 1e-12
PREDICTION_BARRIER_SCHEMA = "stvg-native-instance-prediction-barrier-v1"
ROWS_SCHEMA = "stvg-native-instance-postseal-rows-v1"
SUMMARY_SCHEMA = "stvg-native-instance-postseal-summary-v1"


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def _safe_root_path(relative: Any) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "sealed artifact path is malformed")
    path = (ROOT / relative).resolve()
    _require(path.is_relative_to(ROOT.resolve()), "sealed artifact path escapes project root")
    _require(path.is_file(), f"sealed artifact is missing: {Path(relative).name}")
    return path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _same_json(left: Any, right: Any) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError):
        return False


def _verify_file_entry(row: Mapping[str, Any], *, require_bytes: bool) -> Path:
    _require(isinstance(row, Mapping), "barrier file entry must be an object")
    path = _safe_root_path(row.get("path"))
    digest = row.get("sha256")
    _require(_is_sha(digest) and _sha(path) == digest,
             f"sealed file digest differs: {path.name}")
    if require_bytes:
        size = row.get("bytes")
        _require(isinstance(size, int) and not isinstance(size, bool) and
                 size == path.stat().st_size,
                 f"sealed file byte count differs: {path.name}")
    return path


def _expected_seal_paths(case_ids: Sequence[str]) -> set[str]:
    expected = {
        str((BASE / f"{phase}_GLOBAL_BARRIER.json").relative_to(ROOT))
        for phase in PHASES
    }
    expected.update({
        str(REFERENCE_LOCK.relative_to(ROOT)),
        str(DIRECT_READOUT.relative_to(ROOT)),
    })
    for case_id in case_ids:
        _require(case_id and all(ch.isalnum() or ch in "_-" for ch in case_id),
                 "roster has an unsafe case ID")
        expected.add(str((BASE / "teacher" / f"{case_id}.json").relative_to(ROOT)))
        expected.add(str((BASE / "teacher" / f"{case_id}.receipt.json").relative_to(ROOT)))
        expected.add(str((BASE / "student" / f"{case_id}.pt").relative_to(ROOT)))
        expected.add(str((BASE / "student" / f"{case_id}.receipt.json").relative_to(ROOT)))
    return expected


def _verify_runtime(stage: str, barrier: Mapping[str, Any]) -> dict[str, Any]:
    path = BASE / f"{stage}_RUNTIME.json"
    _require(path.is_file(), f"{stage.lower()} runtime lock is missing")
    runtime_hash = _sha(path)
    _require(barrier.get("runtime_sha256") == runtime_hash,
             f"{stage.lower()} barrier runtime binding differs")
    runtime = _read_json(path)
    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is False,
             f"{stage.lower()} runtime is not a GT-free lock")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    _require(isinstance(pins, Mapping) and isinstance(protected, Mapping),
             f"{stage.lower()} runtime pin maps are malformed")
    for relative, digest in {**pins, **protected}.items():
        pin_path = _safe_root_path(relative)
        _require(_is_sha(digest) and _sha(pin_path) == digest,
                 f"{stage.lower()} runtime pin changed: {Path(relative).name}")
    return runtime


def _verify_phase(stage: str, final_paths: Mapping[str, Mapping[str, Any]],
                  expected_case_ids: set[str]) -> tuple[dict[str, Any], str]:
    phase_path = BASE / f"{stage}_GLOBAL_BARRIER.json"
    phase = _read_json(phase_path)
    _require(phase.get("status") == "complete" and phase.get("count") == len(expected_case_ids)
             and phase.get("GT_read") is False,
             f"{stage.lower()} barrier is incomplete or not GT-free")
    entries = phase.get("files")
    _require(isinstance(entries, list) and len(entries) == len(expected_case_ids),
             f"{stage.lower()} barrier must contain one file per case")
    seen_cases: set[str] = set()
    for entry in entries:
        _verify_file_entry(entry, require_bytes=False)
        case_id = entry.get("case_id")
        _require(case_id in expected_case_ids and case_id not in seen_cases,
                 f"{stage.lower()} barrier has an unexpected or duplicate case")
        seen_cases.add(case_id)
        final = final_paths.get(entry.get("path"))
        if final is not None:
            _require(final.get("sha256") == entry.get("sha256"),
                     f"{stage.lower()} barrier and final seal disagree")
    _require(seen_cases == expected_case_ids, f"{stage.lower()} barrier case set differs")
    _verify_runtime(stage, phase)
    return phase, _sha(phase_path)


def _load_locked_inputs() -> dict[str, Any]:
    """Verify every prediction SHA before any source/GT payload is opened."""
    _require(SEAL.is_file(), "global prediction barrier has not been written")
    seal = _read_json(SEAL)
    _require(seal.get("status") == "complete" and seal.get("count") == 8 and
             seal.get("GT_read") is False,
             "global prediction barrier is not a complete eight-case pre-GT seal")

    roster = _read_json(ROSTER)
    roster_rows = roster.get("records")
    _require(roster.get("status") == "locked" and roster.get("count") == 8 and
             roster.get("GT_read") is False and isinstance(roster_rows, list) and
             len(roster_rows) == 8,
             "roster is not the fixed, GT-free eight-case lock")
    case_ids = [row.get("case_id") for row in roster_rows]
    _require(len(set(case_ids)) == 8 and all(isinstance(x, str) for x in case_ids),
             "roster case IDs are missing or duplicated")
    expected_paths = _expected_seal_paths(case_ids)

    entries = seal.get("files")
    _require(isinstance(entries, list) and len(entries) == 37,
             "global seal must contain the three phase barriers, two locks, and 32 case artifacts")
    paths: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        path = _verify_file_entry(entry, require_bytes=True)
        relative = str(path.relative_to(ROOT))
        _require(relative not in paths, "global seal repeats a file path")
        paths[relative] = entry
    _require(set(paths) == expected_paths,
             "global seal path set differs from the locked 37-file schema")

    _require(seal.get("schema", PREDICTION_BARRIER_SCHEMA) == PREDICTION_BARRIER_SCHEMA,
             "global prediction barrier schema differs")
    _require(ROSTER.exists() and _is_sha(_sha(ROSTER)), "roster digest is unavailable")
    ref = _read_json(REFERENCE_LOCK)
    direct = _read_json(DIRECT_READOUT)
    _require(ref.get("schema") == "stvg-native-instance-reference-lock-v1" and
             ref.get("status") == "locked" and ref.get("count") == 8 and
             ref.get("GT_read") is False and ref.get("roster_sha256") == _sha(ROSTER),
             "reference lock is not bound to the fixed roster")
    roster_counts = {"hc2": 0, "vidstg": 0}
    for roster_row in roster_rows:
        dataset = roster_row.get("dataset")
        _require(dataset in roster_counts,
                 "roster dataset is outside the two-domain fixed scope")
        roster_counts[dataset] += 1
    _require(roster_counts == {"hc2": 4, "vidstg": 4},
             "roster is not four historical development sources per domain")
    _require(direct.get("schema") == "stvg-native-instance-direct-teacher-readout-v1" and
             direct.get("status") == "complete" and direct.get("count") == 8 and
             direct.get("GT_read") is False,
             "direct teacher readout is incomplete or not GT-free")

    ref_rows = ref.get("records")
    direct_rows = direct.get("records")
    _require(isinstance(ref_rows, list) and len(ref_rows) == 8 and
             isinstance(direct_rows, list) and len(direct_rows) == 8,
             "reference/direct records must cover all eight cases")
    ref_by_case = {r.get("case_id"): r for r in ref_rows}
    direct_by_case = {r.get("case_id"): r for r in direct_rows}
    _require(set(ref_by_case) == set(case_ids) and set(direct_by_case) == set(case_ids),
             "reference/direct case sets differ from the roster")

    phases: dict[str, Any] = {}
    phase_hashes: dict[str, str] = {}
    for stage in PHASES:
        phases[stage], phase_hashes[stage] = _verify_phase(stage, paths, set(case_ids))

    _require(ref.get("tracking_barrier_sha256") == phase_hashes["TRACKING"] and
             ref.get("teacher_barrier_sha256") == phase_hashes["TEACHER"],
             "reference lock stage-barrier bindings differ")
    teacher_barrier = phases["TEACHER"]
    _require(teacher_barrier.get("GT_read") is False, "teacher stage accessed GT")

    # Read only machine-readable teacher/student records after all 37 file
    # hashes and all phase/runtime pins have passed. Never export raw teacher
    # prose, source query text, private paths, or tensor values.
    teacher_records: dict[str, dict[str, Any]] = {}
    teacher_receipts: dict[str, dict[str, Any]] = {}
    student_receipts: dict[str, dict[str, Any]] = {}
    student_paths: dict[str, Path] = {}
    eligibility_by_case: dict[str, bool] = {}
    for case_id in case_ids:
        record_path = BASE / "teacher" / f"{case_id}.json"
        teacher_receipt_path = BASE / "teacher" / f"{case_id}.receipt.json"
        student_path = BASE / "student" / f"{case_id}.pt"
        student_receipt_path = BASE / "student" / f"{case_id}.receipt.json"
        teacher_records[case_id] = _read_json(record_path)
        teacher_receipts[case_id] = _read_json(teacher_receipt_path)
        student_receipts[case_id] = _read_json(student_receipt_path)
        student_paths[case_id] = student_path

        ref_row = ref_by_case[case_id]
        direct_row = direct_by_case[case_id]
        t_receipt = teacher_receipts[case_id]
        s_receipt = student_receipts[case_id]
        roster_row = next(r for r in roster_rows if r.get("case_id") == case_id)
        _require(ref_row.get("GT_read") is False and direct_row.get("GT_read") is False and
                 teacher_records[case_id].get("GT_read", False) is False and
                 t_receipt.get("GT_read") is False and s_receipt.get("GT_read") is False,
                 f"a sealed case declares GT access: {case_id}")
        _require(ref_row.get("source_case_sha256") == next(
            r.get("source_case_sha256") for r in roster_rows if r.get("case_id") == case_id),
            f"reference source digest differs from roster: {case_id}")
        _require(ref_row.get("teacher_record_sha256") == _sha(record_path) and
                 ref_row.get("teacher_receipt_sha256") == _sha(teacher_receipt_path) and
                 t_receipt.get("teacher_record_sha256") == _sha(record_path),
                 f"teacher record or receipt binding differs: {case_id}")
        _require(direct_row.get("teacher_record_sha256") == _sha(record_path) and
                 direct_row.get("teacher_receipt_sha256") == _sha(teacher_receipt_path) and
                 direct_row.get("source_case_sha256") == ref_row.get("source_case_sha256") and
                 direct_row.get("request_sha256") == ref_row.get("request_sha256") and
                 direct_row.get("dataset") == roster_row.get("dataset"),
                 f"direct teacher readout binding differs: {case_id}")
        _require(s_receipt.get("status") == "sealed" and
                 s_receipt.get("path") == str(student_path.relative_to(ROOT)) and
                 s_receipt.get("sha256") == _sha(student_path) and
                 s_receipt.get("bytes") == student_path.stat().st_size,
                 f"student receipt does not bind the sealed prediction: {case_id}")

        _require(roster_row.get("source_case_sha256") == ref_row.get("source_case_sha256") and
                 roster_row.get("request_sha256") == ref_row.get("request_sha256") and
                 t_receipt.get("case_id") == case_id,
                 f"roster/reference input bindings differ: {case_id}")

        teacher = teacher_records[case_id]
        parsed = teacher.get("parsed") if isinstance(teacher.get("parsed"), Mapping) else {}
        _require(parsed.get("status") in {
            "valid", "invalid_format", "invalid_time_or_id", "unknown_candidate_id",
            "invalid_duration", "out_of_clock_interval", "out_of_duration",
            "candidate_missing_in_interval",
        }, f"teacher parsed status is outside the native parser enum: {case_id}")
        generation = t_receipt.get("generation") if isinstance(t_receipt.get("generation"), Mapping) else {}
        token_count = generation.get("output_token_count")
        eligible = (
            parsed.get("status") == "valid" and
            isinstance(parsed.get("candidate_id"), int) and not isinstance(parsed.get("candidate_id"), bool) and
            parsed.get("candidate_id", 0) > 0 and
            generation.get("output_truncated_at_cap") is False and
            isinstance(token_count, int) and not isinstance(token_count, bool) and
            0 <= token_count < 2048 and generation.get("max_new_tokens") == 2048 and
            generation.get("generation_call_count") == 1
        )
        candidate_id = parsed.get("candidate_id") if eligible else None
        _require(bool(direct_row.get("valid_uncapped_selection")) is eligible and
                 ref_row.get("candidate_id") == candidate_id and
                 direct_row.get("candidate_id") == candidate_id,
                 f"native teacher selection/cap gate differs from the locked parse: {case_id}")
        _require(ref_row.get("no_update") is (len(ref_row.get("anchors", [])) == 0),
                 f"reference no-update flag differs from fixed-center anchors: {case_id}")
        if eligible:
            _require(_same_json(ref_row.get("native_interval"), direct_row.get("native_interval")),
                     f"selected native interval differs between lock and direct readout: {case_id}")
        else:
            _require(ref_row.get("native_interval") is None and
                     direct_row.get("native_interval") is None and
                     direct_row.get("dense_frames") == [],
                     f"invalid/capped teacher output exposes a native mask tube: {case_id}")
        eligibility_by_case[case_id] = bool(eligible)

    return {
        "seal": seal,
        "seal_sha256": _sha(SEAL),
        "roster": roster,
        "roster_sha256": _sha(ROSTER),
        "reference_lock": ref,
        "reference_lock_sha256": _sha(REFERENCE_LOCK),
        "direct_readout": direct,
        "direct_readout_sha256": _sha(DIRECT_READOUT),
        "phases": phases,
        "phase_hashes": phase_hashes,
        "case_ids": case_ids,
        "ref_by_case": ref_by_case,
        "direct_by_case": direct_by_case,
        "teacher_records": teacher_records,
        "teacher_receipts": teacher_receipts,
        "student_receipts": student_receipts,
        "eligibility_by_case": eligibility_by_case,
        "student_paths": student_paths,
        "roster_by_case": {r["case_id"]: r for r in roster_rows},
    }


def _safe_torch_load(path: Path) -> Any:
    """Load tensor/array payloads with PyTorch's restricted unpickler only."""
    import numpy as np
    import torch

    safe: list[Any] = [np.ndarray, np.dtype]
    for module in (getattr(np, "core", None), getattr(np, "_core", None)):
        multiarray = getattr(module, "multiarray", None) if module is not None else None
        if multiarray is not None:
            for name in ("_reconstruct", "scalar"):
                value = getattr(multiarray, name, None)
                if value is not None:
                    safe.append(value)
    for dtype in (np.float16, np.float32, np.float64, np.int8, np.int16,
                  np.int32, np.int64, np.uint8, np.bool_):
        safe.append(np.dtype(dtype).type)
    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    _require(callable(add_safe), "this PyTorch build lacks restricted tensor loading")
    add_safe(safe)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"restricted tensor load failed for {path.name}; unsafe fallback is disabled") from exc


def _load_source_cases(locked: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from scripts import stvg_sparse_r1_prompt_v1 as source_reader

    result: dict[str, dict[str, Any]] = {}
    for case_id in locked["case_ids"]:
        row = locked["roster_by_case"][case_id]
        source_path = _safe_root_path(row.get("source_case_path"))
        _require(_sha(source_path) == row.get("source_case_sha256"),
                 f"source case digest differs from roster: {case_id}")
        receipt_path = source_path.with_suffix(".json")
        receipt = _read_json(receipt_path)
        _require(receipt.get("status") == "accepted_complete_source_only_case" and
                 receipt.get("GT_read") is False and
                 receipt.get("sha256") == row.get("source_case_sha256") and
                 receipt.get("bytes") == source_path.stat().st_size,
                 f"source-case custody receipt differs: {case_id}")
        case = _safe_torch_load(source_path)
        _require(isinstance(case, Mapping), f"source case is not an object: {case_id}")
        source_reader._validate_source_case(case)
        case = dict(case)
        case["_source_case_sha256"] = receipt["sha256"]
        _require(case.get("case_id") == case_id and case.get("dataset") == row.get("dataset") and
                 case.get("_source_case_sha256") == row.get("source_case_sha256"),
                 f"source case identity differs from roster: {case_id}")
        result[case_id] = case
    return result


def _to_numpy(value: Any, *, dtype: Any = None):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(value, dtype=dtype)
    return arr


def _validate_prediction(case_id: str, case: Mapping[str, Any],
                         ref: Mapping[str, Any], direct_row: Mapping[str, Any],
                         student: Mapping[str, Any], receipt: Mapping[str, Any],
                         student_sha: str, student_receipt_sha: str) -> dict[str, Any]:
    import numpy as np

    _require(isinstance(student, Mapping) and student.get("case_id") == case_id and
             student.get("dataset") == case.get("dataset") and
             student.get("GT_read") is False and student.get("online_carry") is False,
             f"student output identity/GT status differs: {case_id}")
    _require(student.get("source_case_sha256") == case.get("_source_case_sha256") and
             student.get("reference_lock_sha256") == _sha(REFERENCE_LOCK) and
             _same_json(student.get("reference_record"), ref),
             f"student output is bound to another source/reference: {case_id}")
    _require(student.get("source_restored") is True and
             student.get("source_cold_readback", {}).get("status") == "pass" and
             student.get("source_cold_readback", {}).get("source_boxes_logits_bitwise") is True and
             student.get("source_cold_readback", {}).get("full_model_state_sha256") ==
             student.get("source_model_state_sha256") and
             student.get("source_cold_readback", {}).get("pixel_sha256") == student.get("pixel_sha256"),
             f"student source restoration receipt failed: {case_id}")
    _require(receipt.get("status") == "sealed" and
             receipt.get("sha256") == student_sha and
             receipt.get("bytes") == (BASE / "student" / f"{case_id}.pt").stat().st_size and
             receipt.get("actual_fit_count") == student.get("actual_fit_count") and
             receipt.get("optimizer_steps") == student.get("optimizer_steps") and
             receipt.get("actual_final_redecode_calls") == student.get("actual_final_redecode_calls"),
             f"student receipt counters differ from prediction: {case_id}")
    _require(_is_sha(student.get("source_model_state_sha256")) and
             _is_sha(student.get("pixel_sha256")),
             f"student source/pixel binding is malformed: {case_id}")

    frame_ids = _to_numpy(case["frame_ids"], dtype=np.int64).reshape(-1)
    native_boxes = _to_numpy(case["native_boxes"], dtype=np.float64)
    _require(native_boxes.shape == (len(frame_ids), 4) and len(frame_ids) >= 2 and
             bool(np.all(np.diff(frame_ids) > 0)),
             f"source frame grid is malformed: {case_id}")
    _require(_to_numpy(student.get("frame_ids"), dtype=np.int64).reshape(-1).tolist() == frame_ids.tolist(),
             f"student frame grid differs from source: {case_id}")
    _require(_to_numpy(student.get("source_boxes"), dtype=np.float64).shape == (len(frame_ids), 4),
             f"student cold source boxes have the wrong shape: {case_id}")
    _require(_same_interval(student.get("Native_interval"), case.get("Native_interval")) and
             _same_interval(student.get("Projection_interval"), case.get("Projection_interval")),
             f"student source/temporal intervals differ: {case_id}")
    qa_interval = next(r for r in _read_json(ROSTER)["records"] if r["case_id"] == case_id).get("qa_interval")
    _require(_same_interval(student.get("arms", {}).get("T", {}).get("interval"), qa_interval),
             f"fixed T interval differs from its locked QA target: {case_id}")

    arms = student.get("arms")
    fits = student.get("fits")
    readouts = student.get("readouts")
    receipts = student.get("receipts")
    student_arms = set(ARMS) - {"native_teacher_direct"}
    _require(isinstance(arms, Mapping) and set(arms) == student_arms and
             isinstance(fits, Mapping) and isinstance(readouts, Mapping) and
             isinstance(receipts, Mapping),
             f"student output arm/fit schema is incomplete: {case_id}")
    _require(_same_boxes(arms["F"].get("boxes"), native_boxes) and
             _same_interval(arms["F"].get("interval"), case.get("Native_interval")) and
             _same_boxes(arms["T"].get("boxes"), native_boxes),
             f"F/T source geometry is not exact reuse: {case_id}")
    _require(_same_boxes(_cxcywh_to_xyxy(student.get("source_boxes")), arms["F"].get("boxes")),
             f"student source-reset boxes differ from F: {case_id}")

    anchors = ref.get("anchors")
    _require(isinstance(anchors, list) and ref.get("no_update") is (len(anchors) == 0),
             f"reference no-update declaration differs from anchors: {case_id}")
    positions: list[int] = []
    for anchor in anchors:
        pos = anchor.get("position")
        _require(isinstance(pos, int) and not isinstance(pos, bool) and 0 <= pos < len(frame_ids),
                 f"reference anchor position is outside the source grid: {case_id}")
        _require(anchor.get("physical_frame_id") == int(frame_ids[pos]),
                 f"reference anchor does not bind its source position: {case_id}")
        positions.append(pos)
    _require(len(set(positions)) == len(positions), f"reference positions are duplicated: {case_id}")

    active_algorithms = []
    if anchors:
        active_algorithms.extend(("direct", "iou"))
        weights = [float(a.get("weight", 0.0)) for a in anchors]
        _require(all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in weights),
                 f"locked reference weights are malformed: {case_id}")
        if any(weight > 0.0 for weight in weights):
            active_algorithms.append("reliability_weighted_iou")
    expected_fits = set(active_algorithms)
    _require(set(fits) == expected_fits and set(readouts) == expected_fits and
             student.get("actual_fit_count") == len(expected_fits) and
             student.get("actual_final_redecode_calls") == len(expected_fits),
             f"fit/readout set differs from the fixed per-arm support rule: {case_id}")
    step_sum = 0
    config = student.get("configuration")
    if expected_fits:
        _require(isinstance(config, Mapping) and config.get("steps") == 5,
                 f"student configuration does not preserve five declared steps: {case_id}")
    for algorithm in STUDENT_ALGORITHMS:
        if algorithm not in expected_fits:
            _require(receipts.get(algorithm, {}).get("status") == "exact_no_update" and
                     int(receipts[algorithm].get("optimizer_steps", -1)) == 0 and
                     int(receipts[algorithm].get("final_redecode_calls", -1)) == 0,
                     f"exact no-update algorithm receipt differs: {case_id}/{algorithm}")
            _require(_same_boxes(arms[algorithm].get("boxes"), arms["F"].get("boxes")) and
                     _same_interval(arms[algorithm].get("interval"), arms["F"].get("interval")) and
                     _same_boxes(arms[f"T_{algorithm}"].get("boxes"), arms["T"].get("boxes")) and
                     _same_interval(arms[f"T_{algorithm}"].get("interval"), arms["T"].get("interval")),
                     f"no-update arm is not exact F/T reuse: {case_id}/{algorithm}")
            continue
        fit = fits[algorithm]
        _require(isinstance(fit, Mapping) and fit.get("algorithm") == algorithm and
                 fit.get("reference_sha256") == _sha(REFERENCE_LOCK) and
                 isinstance(fit.get("optimizer_steps"), int) and
                 0 <= fit["optimizer_steps"] <= 5 and
                 fit.get("selected_step") == 5 and fit.get("declared_steps") == 5 and
                 fit.get("active_parameters") == 1792 and
                 isinstance(fit.get("config"), Mapping) and
                 fit["config"].get("steps") == 5,
                 f"fit audit/configuration differs: {case_id}/{algorithm}")
        _require(_same_json(fit["config"], config),
                 f"fit arm config differs from case config: {case_id}/{algorithm}")
        _require(_tree_equal(fit.get("initial_state"), student.get("initial_state")),
                 f"fit initial state differs from the cold source state: {case_id}/{algorithm}")
        step_sum += fit["optimizer_steps"]
        readout = readouts[algorithm]
        _require(isinstance(readout, Mapping),
                 f"fit final readout is missing: {case_id}/{algorithm}")
        _require(_tree_equal(readout.get("logits"), student.get("source_logits")) and
                 _same_interval(readout.get("physical_interval"), case.get("Native_interval")),
                 f"fit full readout changed source logits or temporal interval: {case_id}/{algorithm}")
        output_xyxy = _cxcywh_to_xyxy(readout.get("boxes"))
        _require(_same_boxes(arms[algorithm].get("boxes"), output_xyxy) and
                 _same_boxes(arms[f"T_{algorithm}"].get("boxes"), output_xyxy),
                 f"arm geometry differs from its actual final readout: {case_id}/{algorithm}")
        if "final_cached_decoder_boxes" in fit:
            _require(_tree_equal(fit["final_cached_decoder_boxes"], readout.get("boxes")),
                     f"fit cached boxes differ from final full-model readout: {case_id}/{algorithm}")
        _require(_same_interval(arms[algorithm].get("interval"), case.get("Native_interval")) and
                 _same_interval(arms[f"T_{algorithm}"].get("interval"),
                                arms["T"].get("interval")),
                 f"fit arm temporal interval is not the locked native/T interval: {case_id}/{algorithm}")
    _require(student.get("optimizer_steps") == step_sum,
             f"student optimizer-step total differs from fit records: {case_id}")

    # Direct teacher output must preserve all native frames and absence flags.
    _validate_direct_teacher(case_id, case, ref, direct_row)
    return {
        "frame_ids": frame_ids.tolist(),
        "native_boxes": native_boxes,
        "arms": arms,
        "anchors": anchors,
        "supervised_positions": set(positions),
        "source_case_sha256": case["_source_case_sha256"],
        "student_sha256": student_sha,
        "student_receipt_sha256": student_receipt_sha,
        "actual_fit_count": int(student.get("actual_fit_count", 0)),
        "optimizer_steps": int(student.get("optimizer_steps", 0)),
        "no_update": bool(ref["no_update"]),
    }


def _same_interval(left: Any, right: Any, *, atol: float = 0.0) -> bool:
    import numpy as np
    try:
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
        return a.shape == (2,) and b.shape == (2,) and bool(np.allclose(a, b, rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def _same_boxes(left: Any, right: Any, *, atol: float = 0.0) -> bool:
    import numpy as np
    try:
        a = _to_numpy(left, dtype=np.float64)
        b = _to_numpy(right, dtype=np.float64)
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _cxcywh_to_xyxy(value: Any):
    import numpy as np
    # The student writer explicitly converts decoder boxes to float64 before
    # XYXY arithmetic (`run_stvg_adjacent_motion_student_v1._xyxy`).
    boxes = _to_numpy(value, dtype=np.float64)
    _require(boxes.ndim == 2 and boxes.shape[1] == 4 and np.isfinite(boxes).all(),
             "final readout boxes are malformed")
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def _tree_equal(left: Any, right: Any) -> bool:
    import numpy as np
    try:
        import torch
        if torch.is_tensor(left) or torch.is_tensor(right):
            return (torch.is_tensor(left) and torch.is_tensor(right) and
                    left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu()))
    except ImportError:
        pass
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and
                left.dtype == right.dtype and np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and
                set(left) == set(right) and
                all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _validate_direct_teacher(case_id: str, case: Mapping[str, Any], ref: Mapping[str, Any],
                             direct_row: Mapping[str, Any]) -> None:
    import numpy as np
    _require(direct_row.get("selection_status") == ref.get("parsed_status") and
             direct_row.get("candidate_id") == ref.get("candidate_id") and
             isinstance(direct_row.get("valid_uncapped_selection"), bool),
             f"native direct/readout selection status differs: {case_id}")
    if not direct_row.get("valid_uncapped_selection"):
        _require(direct_row.get("candidate_id") is None and
                 direct_row.get("native_interval") is None and
                 direct_row.get("dense_frames") == [],
                 f"invalid/capped teacher output unexpectedly contains a selected tube: {case_id}")
        return
    _require(ref.get("candidate_id") is not None,
             f"valid native teacher selection has no candidate ID: {case_id}")
    interval = direct_row.get("native_interval")
    dense = direct_row.get("dense_frames")
    _require(isinstance(interval, Mapping) and isinstance(dense, list) and dense,
             f"selected teacher has no direct full-mask readout: {case_id}")
    teacher_path = BASE / "teacher" / f"{case_id}.json"
    teacher = _read_json(teacher_path)
    frame_map = teacher.get("input_binding", {}).get("frame_map")
    tube_frames = teacher.get("mask_tube", {}).get("frames")
    _require(isinstance(frame_map, list) and isinstance(tube_frames, list) and
             len(frame_map) == len(tube_frames) == len(dense),
             f"native teacher tube/readout are not dense on the locked frame map: {case_id}")
    _require(_same_json(interval, ref.get("native_interval")),
             f"native interval changed between direct readout and reference lock: {case_id}")
    width, height = int(case["input"]["width"]), int(case["input"]["height"])
    for index, (clock, tube, row) in enumerate(zip(frame_map, tube_frames, dense)):
        _require(row.get("local_frame_index") == index == clock.get("local_frame_index") ==
                 tube.get("local_frame_index") and
                 row.get("physical_frame_id") == clock.get("physical_frame_id") ==
                 tube.get("physical_frame_id") and
                 row.get("source_timestamp_s") == clock.get("source_timestamp_s") ==
                 tube.get("source_timestamp_s"),
                 f"native dense frame clock differs: {case_id}/{index}")
        _require(bool(row.get("present")) == bool(tube.get("present")) and
                 row.get("candidate_id") == tube.get("candidate_id") and
                 row.get("mask_area_px") == tube.get("mask_area_px"),
                 f"native dense presence/ID differs: {case_id}/{index}")
        inclusive = tube.get("xyxy_inclusive_pixels")
        _require(_same_json(row.get("xyxy_inclusive_pixels"), inclusive),
                 f"native inclusive mask box differs: {case_id}/{index}")
        normalized = row.get("xyxy_normalized_continuous")
        cxcywh = row.get("cxcywh_normalized_continuous")
        if not row.get("present"):
            _require(inclusive is None and normalized is None and cxcywh is None and
                     int(row.get("mask_area_px", -1)) == 0,
                     f"missing mask frame has fabricated geometry: {case_id}/{index}")
            continue
        _require(isinstance(inclusive, list) and len(inclusive) == 4 and
                 isinstance(normalized, list) and len(normalized) == 4 and
                 isinstance(cxcywh, list) and len(cxcywh) == 4 and
                 isinstance(row.get("mask_area_px"), int) and row["mask_area_px"] > 0,
                 f"present teacher mask frame lacks geometry: {case_id}/{index}")
        x1, y1, x2, y2 = [float(x) for x in inclusive]
        expected_xyxy = [x1 / width, y1 / height, (x2 + 1.0) / width, (y2 + 1.0) / height]
        expected_cxcywh = [(expected_xyxy[0] + expected_xyxy[2]) / 2,
                           (expected_xyxy[1] + expected_xyxy[3]) / 2,
                           expected_xyxy[2] - expected_xyxy[0],
                           expected_xyxy[3] - expected_xyxy[1]]
        _require(np.allclose(normalized, expected_xyxy, rtol=0.0, atol=1e-12) and
                 np.allclose(cxcywh, expected_cxcywh, rtol=0.0, atol=1e-12),
                 f"native mask normalization differs from inclusive pixel geometry: {case_id}/{index}")


def _truth_map(target: Mapping[str, Any]) -> dict[int, list[float]]:
    truth = target.get("truth")
    _require(isinstance(truth, Mapping) and truth,
             "official post-seal scorer returned empty/invalid GT support")
    converted: dict[int, list[float]] = {}
    for frame_id, box in truth.items():
        values = [float(x) for x in box]
        _require(len(values) == 4 and all(math.isfinite(x) for x in values),
                 "official post-seal scorer returned malformed GT coordinates")
        converted[int(frame_id)] = values
    return converted


def _known_positions(frame_ids: Sequence[int], truth: Mapping[int, Any],
                     supervised_positions: set[int]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    position_frames = {position: int(frame_ids[position]) for position in range(len(frame_ids))}
    anchor_frame_ids = {position_frames[p] for p in supervised_positions}
    gt_frame_ids = sorted(int(fid) for fid in truth)
    # The subset split is by physical GT frame, not by whether a GT frame was
    # sampled by the student's sparse source grid.  Thus off-grid GT frames
    # remain part of the unsupervised full-tube metric.
    supervised = sorted(fid for fid in gt_frame_ids if fid in anchor_frame_ids)
    unsupervised = [fid for fid in gt_frame_ids if fid not in anchor_frame_ids]
    supports = {"supervised": supervised, "unsupervised": unsupervised}
    known = {name: list(ids) for name, ids in supports.items()}
    return supports, known


def _score_student_arm(arm: Mapping[str, Any], case: Mapping[str, Any], target: Mapping[str, Any],
                       supervised_positions: set[int], official: Any, independent: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from scripts.review_stvg_sparse_r1_v2 import score_tube_xyxy

    truth = _truth_map(target)
    frame_ids = _to_numpy(case["frame_ids"], dtype=np.int64).reshape(-1).tolist()
    boxes = _to_numpy(arm.get("boxes"), dtype=np.float64)
    interval = arm.get("interval")
    _require(boxes.shape == (len(frame_ids), 4) and np.isfinite(boxes).all(),
             "student arm boxes must be a complete finite source-grid tube")
    span = target.get("span")
    _require(isinstance(span, (list, tuple)) and len(span) == 2,
             "official scorer returned a malformed GT span")
    width, height = int(case["input"]["width"]), int(case["input"]["height"])
    dataset = str(case["dataset"])
    full = score_tube_xyxy(boxes, frame_ids, width, height, interval, truth, span,
                           dataset, official=official, independent=independent)
    supports, known = _known_positions(frame_ids, truth, supervised_positions)
    support_metrics: dict[str, float | None] = {}
    for name in ("supervised", "unsupervised"):
        known_truth = {fid: truth[fid] for fid in known[name]}
        if not known_truth:
            support_metrics[f"m_sIoU_{name}_GT"] = None
        else:
            restricted = score_tube_xyxy(boxes, frame_ids, width, height, interval,
                                         known_truth, span, dataset,
                                         official=official, independent=independent)
            support_metrics[f"m_sIoU_{name}_GT"] = restricted["metrics"]["m_sIoU_dense_GT"]
    metrics = dict(full["metrics"])
    metrics.update(support_metrics)
    counts = {
        "source_grid_positions": len(frame_ids),
        "source_grid_supervised_position_count": len(supervised_positions),
        "source_grid_unsupervised_position_count": len(frame_ids) - len(supervised_positions),
        "supervised_anchor_count": len(supervised_positions),
        "unsupervised_position_count": len(frame_ids) - len(supervised_positions),
        "dense_gt_known_frame_count": len(truth),
        "source_grid_gt_known_frame_count": sum(int(fid) in set(frame_ids) for fid in truth),
        "off_grid_gt_known_frame_count": sum(int(fid) not in set(frame_ids) for fid in truth),
        "supervised_gt_known_frame_count": len(known["supervised"]),
        "unsupervised_gt_known_frame_count": len(known["unsupervised"]),
        "supervised_positions": sorted(supervised_positions),
        "supervised_physical_frame_ids": supports["supervised"],
        "unsupervised_physical_frame_ids": supports["unsupervised"],
        "gt_support_split": "supervised=known GT physical IDs at reference anchors; unsupervised=all other known GT physical IDs, including off-grid frames",
    }
    audit = {"official_independent_max_abs_error": full["official_independent_max_abs_error"],
             "source_grid_frames": full["source_grid_frames"],
             "interval_frame_bounds_half_open": full["interval_frame_bounds_half_open"]}
    return metrics, {"support": counts, "metric_audit": audit}


def _native_time_interval(case: Mapping[str, Any], direct_row: Mapping[str, Any],
                          teacher: Mapping[str, Any]):
    parsed = teacher.get("parsed", {})
    start_s, end_s = parsed.get("start_time_s"), parsed.get("end_time_s")
    _require(isinstance(start_s, (int, float)) and not isinstance(start_s, bool) and
             isinstance(end_s, (int, float)) and not isinstance(end_s, bool) and
             math.isfinite(float(start_s)) and math.isfinite(float(end_s)) and
             0 <= float(start_s) < float(end_s),
             "valid teacher selection has malformed time bounds")
    fps = float(case["input"]["fps"])
    bounds = [float(x) for x in case["video_bounds"]]
    frame_map = teacher.get("input_binding", {}).get("frame_map")
    _require(isinstance(frame_map, list) and len(frame_map) >= 2 and fps > 0,
             "teacher time conversion lacks a source frame clock")
    first = frame_map[0]
    offset = float(first["physical_frame_id"]) - float(first["source_timestamp_s"]) * fps
    start = offset + float(start_s) * fps
    end = offset + float(end_s) * fps
    _require(bounds[0] - 1e-6 <= start < end <= bounds[1] + 1e-6,
             "native R1 interval converted outside the source's half-open physical bounds")
    local_interval = direct_row.get("native_interval")
    projected = teacher.get("projected_interval", {})
    _require(isinstance(local_interval, Mapping) and
             float(local_interval.get("start_time_s", float("nan"))) == float(start_s) and
             float(local_interval.get("end_time_s", float("nan"))) == float(end_s) and
             local_interval.get("start_local_frame") == projected.get("start_local_frame") and
             local_interval.get("end_local_frame") == projected.get("end_local_frame"),
             "direct readout native interval is malformed")
    return [float(start), float(end)]


def _independent_teacher_metrics(record: Mapping[str, Any], width: int, height: int,
                                truth: Mapping[int, Sequence[float]], span: Sequence[int],
                                dataset: str) -> dict[str, float]:
    """Independent arithmetic for the official native full-grid/missing-mask policy."""
    _require(dataset in {"hc2", "vidstg"} and width > 0 and height > 0,
             "native metric dimensions or dataset are malformed")
    frame_ids = [int(value) for value in record["frame_ids"]]
    present = list(record["present_mask"])
    boxes = list(record["boxes_cxcywh_normalized"])
    _require(len(frame_ids) == len(present) == len(boxes) > 0,
             "native metric grid/presence/box lengths differ")
    iou_by_frame: dict[int, float] = {}
    for frame_id, is_present, box in zip(frame_ids, present, boxes):
        if not is_present:
            iou_by_frame[frame_id] = 0.0
            continue
        _require(isinstance(box, (list, tuple)) and len(box) == 4,
                 "present native metric frame has no cxcywh geometry")
        cx, cy, bw, bh = (float(value) for value in box)
        pred = [(cx - bw / 2.0) * width, (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width, (cy + bh / 2.0) * height]
        if dataset == "hc2":
            pred = [max(value, 0.0) for value in pred]
        gt_raw = truth.get(frame_id)
        if gt_raw is None:
            continue
        gt = [float(value) for value in gt_raw]
        _require(len(gt) == 4 and all(math.isfinite(value) for value in gt),
                 "native metric GT box is malformed")
        ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
        ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
        intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        pred_area = max(pred[2] - pred[0], 0.0) * max(pred[3] - pred[1], 0.0)
        gt_area = max(gt[2] - gt[0], 0.0) * max(gt[3] - gt[1], 0.0)
        union = pred_area + gt_area - intersection
        iou_by_frame[frame_id] = intersection / union if union > 0.0 else 0.0

    gt_ids = sorted(int(frame_id) for frame_id in truth)
    _require(bool(gt_ids), "native metric requires nonempty known GT support")
    spatial = math.fsum(iou_by_frame.get(frame_id, 0.0) for frame_id in gt_ids) / len(gt_ids)
    a, z = (int(value) for value in record["interval_frame_bounds"])
    g, h = (int(value) for value in span)
    overlap = max(0, min(z, h) - max(a, g))
    temporal_union = z - a + h - g - overlap
    _require(temporal_union > 0, "native temporal union is empty")
    overlap_ids = [frame_id for frame_id in gt_ids
                   if max(a, g) <= frame_id < min(z, h)]
    tube_denominator = max(max(z, h) - min(a, g), 1)
    tube = math.fsum(iou_by_frame.get(frame_id, 0.0) for frame_id in overlap_ids) / tube_denominator
    return {"m_vIoU": float(tube), "m_tIoU": float(overlap / temporal_union),
            "m_sIoU_dense_GT": float(spatial)}


def _score_native_teacher(case_id: str, case: Mapping[str, Any], target: Mapping[str, Any],
                          ref: Mapping[str, Any], direct_row: Mapping[str, Any],
                          teacher: Mapping[str, Any], official: Any, independent: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.stvg_target_teacher_evaluate_v2 import _teacher_physical_dense_metric

    truth = _truth_map(target)
    frames = direct_row.get("dense_frames")
    interval = direct_row.get("native_interval")
    if (not direct_row.get("valid_uncapped_selection") or
            not isinstance(frames, list) or not frames or not isinstance(interval, Mapping)):
        return ({k: None for k in METRICS}, {
            "status": "invalid_or_abstained_no_teacher_tube",
            "support": _empty_support(case, truth, ref.get("anchors", [])),
            "coverage": None, "selected_track_gt_geometry_coverage": None,
            "metric_audit": None,
        })

    physical_interval = _native_time_interval(case, direct_row, teacher)
    frame_ids = [int(row["physical_frame_id"]) for row in frames]
    present = [bool(row.get("present")) for row in frames]
    boxes = [row.get("cxcywh_normalized_continuous") if row.get("present") else None
             for row in frames]
    record = {"frame_ids": frame_ids, "present_mask": present,
              "boxes_cxcywh_normalized": boxes,
              "interval_frame_bounds": [int(math.floor(physical_interval[0])),
                                         int(math.floor(physical_interval[1]))]}
    span = [int(x) for x in target["span"]]
    metric, _details = _teacher_physical_dense_metric(
        record, {"frame_ids": frame_ids,
                 "input": {"width": int(case["input"]["width"]),
                           "height": int(case["input"]["height"])}},
        truth, span, str(case["dataset"]))
    metrics = {key: float(metric[key]) for key in
               ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")}
    independent_metrics = _independent_teacher_metrics(
        record, int(case["input"]["width"]), int(case["input"]["height"]),
        truth, span, str(case["dataset"]))
    independent_error = max(abs(metrics[key] - independent_metrics[key]) for key in metrics)
    _require(independent_error <= 2e-12,
             f"native teacher full/partial-mask metric differs from independent arithmetic: {case_id}")

    # For a complete mask tube, independently compare the native physical
    # metric against the same official/independent dense path used by students.
    metric_error: float | None = None
    if all(present):
        xyxy = [row["xyxy_normalized_continuous"] for row in frames]
        check = __import__("scripts.review_stvg_sparse_r1_v2", fromlist=["score_tube_xyxy"]).score_tube_xyxy(
            xyxy, frame_ids, int(case["input"]["width"]), int(case["input"]["height"]),
            record["interval_frame_bounds"], truth, span, str(case["dataset"]),
            official=official, independent=independent)
        metric_error = max(abs(metrics[key] - check["metrics"][key]) for key in
                           ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT"))
        _require(metric_error <= 2e-9,
                 f"complete native teacher tube disagrees with official dense metric: {case_id}")

    frame_to_row = {int(row["physical_frame_id"]): row for row in frames}
    source_frame_ids = _to_numpy(case["frame_ids"], dtype="int64").reshape(-1).tolist()
    anchor_positions = {int(a["position"]) for a in ref.get("anchors", [])}
    supports, known = _known_positions(source_frame_ids, truth, anchor_positions)
    for name in ("supervised", "unsupervised"):
        if not known[name]:
            metrics[f"m_sIoU_{name}_GT"] = None
            continue
        sub_truth = {fid: truth[fid] for fid in known[name]}
        sub_metric, _ = _teacher_physical_dense_metric(
            record, {"frame_ids": frame_ids,
                     "input": {"width": int(case["input"]["width"]),
                               "height": int(case["input"]["height"])}},
            sub_truth, span, str(case["dataset"]))
        sub_independent = _independent_teacher_metrics(
            record, int(case["input"]["width"]), int(case["input"]["height"]),
            sub_truth, span, str(case["dataset"]))
        support_error = abs(float(sub_metric["m_sIoU_dense_GT"]) -
                            sub_independent["m_sIoU_dense_GT"])
        independent_error = max(independent_error, support_error)
        _require(support_error <= 2e-12,
                 f"native teacher subset metric differs from independent arithmetic: {case_id}/{name}")
        metrics[f"m_sIoU_{name}_GT"] = float(sub_metric["m_sIoU_dense_GT"])

    selected_in_interval = [row for row in frames if
                            physical_interval[0] <= float(row["source_timestamp_s"]) *
                            float(case["input"]["fps"]) +
                            (int(row["physical_frame_id"]) -
                             float(row["source_timestamp_s"]) * float(case["input"]["fps"]))
                            < physical_interval[1]]
    run_count = _present_run_count(present)
    present_count = sum(present)
    counts = _support_counts(case, truth, anchor_positions)
    coverage = {
        "frame_count": len(frames),
        "present_frame_count": present_count,
        "missing_frame_count": len(frames) - present_count,
        "present_fraction": present_count / len(frames) if frames else 0.0,
        "present_run_count": run_count,
        "fragmentation_gap_count": max(0, run_count - 1),
        "interval_frame_count": len(selected_in_interval),
        "interval_present_frame_count": sum(bool(row.get("present")) for row in selected_in_interval),
    }
    gt_ids = set(truth)
    tube_ids = set(frame_to_row)
    represented_known = sorted(gt_ids & tube_ids)
    present_known = sum(bool(frame_to_row[fid].get("present")) for fid in represented_known)
    geometric = {
        "gt_known_frame_count": len(gt_ids),
        "teacher_tube_frame_overlap_count": len(represented_known),
        "gt_known_frames_with_present_mask_count": present_known,
        "gt_known_frames_without_present_mask_count": len(gt_ids) - present_known,
        "present_mask_fraction_of_gt_known_frames": present_known / len(gt_ids) if gt_ids else None,
    }
    audit = {"official_independent_max_abs_error": metric_error,
             "native_missing_mask_independent_max_abs_error": independent_error,
             "scoring_path": "official_physical_teacher_missing_as_zero"}
    return metrics, {"status": "scored", "support": counts,
                     "coverage": coverage,
                     "selected_track_gt_geometry_coverage": geometric,
                     "metric_audit": audit, "interval": physical_interval}


def _present_run_count(flags: Sequence[bool]) -> int:
    return sum(1 for i, flag in enumerate(flags) if flag and (i == 0 or not flags[i - 1]))


def _support_counts(case: Mapping[str, Any], truth: Mapping[int, Any],
                    supervised_positions: set[int]) -> dict[str, Any]:
    frame_ids = _to_numpy(case["frame_ids"], dtype="int64").reshape(-1).tolist()
    supports, known = _known_positions(frame_ids, truth, supervised_positions)
    source_grid_ids = set(int(fid) for fid in frame_ids)
    return {
        "source_grid_positions": len(frame_ids),
        "source_grid_supervised_position_count": len(supervised_positions),
        "source_grid_unsupervised_position_count": len(frame_ids) - len(supervised_positions),
        "supervised_anchor_count": len(supervised_positions),
        "unsupervised_position_count": len(frame_ids) - len(supervised_positions),
        "dense_gt_known_frame_count": len(truth),
        "source_grid_gt_known_frame_count": sum(int(fid) in source_grid_ids for fid in truth),
        "off_grid_gt_known_frame_count": sum(int(fid) not in source_grid_ids for fid in truth),
        "supervised_gt_known_frame_count": len(known["supervised"]),
        "unsupervised_gt_known_frame_count": len(known["unsupervised"]),
        "supervised_positions": sorted(supervised_positions),
        "supervised_physical_frame_ids": supports["supervised"],
        "unsupervised_physical_frame_ids": supports["unsupervised"],
        "gt_support_split": "supervised=known GT physical IDs at reference anchors; unsupervised=all other known GT physical IDs, including off-grid frames",
    }


def _empty_support(case: Mapping[str, Any], truth: Mapping[int, Any], anchors: Sequence[Mapping[str, Any]]):
    positions = {int(row["position"]) for row in anchors}
    return _support_counts(case, truth, positions)


def _evaluate(locked: Mapping[str, Any], source_cases: Mapping[str, Mapping[str, Any]],
              student_outputs: Mapping[str, Mapping[str, Any]], targets: Mapping[str, Any],
              official: Any, independent: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teacher_status: dict[str, list[dict[str, Any]]] = {"hc2": [], "vidstg": []}
    for case_id in locked["case_ids"]:
        case = source_cases[case_id]
        reference = locked["ref_by_case"][case_id]
        direct = locked["direct_by_case"][case_id]
        prediction = student_outputs[case_id]
        target = targets[case_id]
        truth = _truth_map(target)
        data = prediction["validated"]
        base_info = {
            "case_id": case_id,
            "dataset": str(case["dataset"]),
            "cohort": locked["roster_by_case"][case_id].get("cohort"),
            "support": None,
            "coverage": None,
            "selected_track_gt_geometry_coverage": None,
            "candidate_detection_recall": "not_measured",
            "fit": None,
            "metric_audit": None,
            "bindings": {
                "source_case_sha256": data["source_case_sha256"],
                "student_artifact_sha256": data["student_sha256"],
                "student_receipt_sha256": data["student_receipt_sha256"],
                "teacher_record_sha256": reference.get("teacher_record_sha256"),
                "teacher_receipt_sha256": reference.get("teacher_receipt_sha256"),
                "reference_lock_sha256": locked["reference_lock_sha256"],
            },
        }
        for arm_name in ARMS:
            row = dict(base_info)
            row["arm"] = arm_name
            row["metrics"] = {metric_name: None for metric_name in METRICS}
            row["status"] = "unscored"
            row["reason"] = None
            row["fit"] = {"status": "not_a_fit_arm", "optimizer_steps": 0,
                           "final_redecode_calls": 0}

            if arm_name == "native_teacher_direct":
                metrics, audit = _score_native_teacher(
                    case_id, case, target, reference, direct,
                    locked["teacher_records"][case_id], official, independent)
                row["metrics"] = metrics
                row["status"] = audit["status"]
                row["reason"] = (None if audit["status"] == "scored"
                                  else str(direct.get("selection_reason") or
                                           direct.get("selection_status") or "invalid_teacher_output"))
                row["support"] = audit.get("support")
                row["coverage"] = audit.get("coverage")
                row["selected_track_gt_geometry_coverage"] = audit.get("selected_track_gt_geometry_coverage")
                row["metric_audit"] = audit.get("metric_audit")
                row["native_interval"] = audit.get("interval")
                teacher_status[row["dataset"]].append({
                    "case_id": case_id,
                    "selection_status": direct.get("selection_status"),
                    "valid_uncapped_selection": bool(direct.get("valid_uncapped_selection")),
                    "coverage_fraction": (audit.get("coverage") or {}).get("present_fraction"),
                    "present_run_count": (audit.get("coverage") or {}).get("present_run_count"),
                    "metrics_status": row["status"],
                })
                rows.append(row)
                continue

            arm = data["arms"][arm_name]
            metrics, audit = _score_student_arm(
                arm, case, target, data["supervised_positions"], official, independent)
            row["metrics"] = metrics
            row["status"] = "scored"
            row["support"] = audit["support"]
            row["metric_audit"] = audit["metric_audit"]
            if arm_name in STUDENT_ALGORITHMS:
                algorithm = arm_name
            elif arm_name.startswith("T_"):
                algorithm = arm_name[2:]
            else:
                algorithm = None
            if algorithm is not None:
                fit = prediction["raw"].get("fits", {}).get(algorithm)
                if fit is None:
                    row["fit"] = {"status": "exact_no_update", "optimizer_steps": 0,
                                   "final_redecode_calls": 0}
                else:
                    row["fit"] = {"status": "actual_fit",
                                  "optimizer_steps": int(fit.get("optimizer_steps", 0)),
                                  "final_redecode_calls": 1}
            rows.append(row)

    summary = _summarize(rows, teacher_status)
    return rows, summary


def _summarize(rows: Sequence[Mapping[str, Any]],
               teacher_status: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_key = {(row["case_id"], row["arm"]): row for row in rows}
    datasets = sorted({str(row["dataset"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows}, key=ARMS.index)
    arm_summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    case_ids_by_dataset = {
        dataset: sorted({str(row["case_id"]) for row in rows if row["dataset"] == dataset})
        for dataset in datasets
    }

    for dataset in datasets:
        dataset_cases = case_ids_by_dataset[dataset]
        for arm in arms:
            for metric in METRICS:
                known = [(cid, by_key[(cid, arm)]["metrics"].get(metric)) for cid in dataset_cases]
                valid = [(cid, float(value)) for cid, value in known if _finite_number(value)]
                missing = [cid for cid, value in known if not _finite_number(value)]
                values = [value for _, value in valid]
                arm_summaries.append({
                    "dataset": dataset, "arm": arm, "metric": metric,
                    "n_valid": len(values), "n_unknown": len(missing),
                    "unknown_case_ids": missing,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                })

        for contrast, left, right in _contrast_specs():
            for metric in METRICS:
                deltas: list[tuple[str, float]] = []
                unknown: list[str] = []
                for cid in dataset_cases:
                    a = by_key[(cid, left)]["metrics"].get(metric)
                    b = by_key[(cid, right)]["metrics"].get(metric)
                    if _finite_number(a) and _finite_number(b):
                        deltas.append((cid, float(a) - float(b)))
                    else:
                        unknown.append(cid)
                values = [delta for _, delta in deltas]
                positive = [cid for cid, delta in deltas if delta > SIGN_EPSILON]
                negative = [cid for cid, delta in deltas if delta < -SIGN_EPSILON]
                neutral = [cid for cid, delta in deltas if abs(delta) <= SIGN_EPSILON]
                boot = _paired_bootstrap(values, dataset, contrast, metric)
                contrasts.append({
                    "dataset": dataset, "contrast": contrast,
                    "comparison_arm": left, "baseline_arm": right,
                    "metric": metric, "n_pairs": len(values),
                    "unknown_count": len(unknown), "unknown_case_ids": unknown,
                    "positive_count": len(positive), "positive_case_ids": positive,
                    "negative_count": len(negative), "negative_case_ids": negative,
                    "neutral_count": len(neutral), "neutral_case_ids": neutral,
                    "neutral_tolerance": SIGN_EPSILON,
                    "mean_delta": statistics.fmean(values) if values else None,
                    "mean_delta_pp": statistics.fmean(values) * 100 if values else None,
                    "median_delta_pp": statistics.median(values) * 100 if values else None,
                    "min_delta_pp": min(values) * 100 if values else None,
                    "max_delta_pp": max(values) * 100 if values else None,
                    "delta_le_minus_5pp_count": sum(value <= -0.05 for value in values),
                    "delta_ge_plus_5pp_count": sum(value >= 0.05 for value in values),
                    "bootstrap_ci95_delta_pp": boot,
                })

    teacher_summary: list[dict[str, Any]] = []
    for dataset in datasets:
        items = list(teacher_status.get(dataset, []))
        selected = [item for item in items if item["valid_uncapped_selection"]]
        valid_coverage = [float(item["coverage_fraction"]) for item in selected
                          if _finite_number(item.get("coverage_fraction"))]
        valid_runs = [int(item["present_run_count"]) for item in selected
                      if isinstance(item.get("present_run_count"), int)]
        statuses: dict[str, int] = {}
        for item in items:
            key = str(item.get("selection_status") or "missing_status")
            statuses[key] = statuses.get(key, 0) + 1
        teacher_summary.append({
            "dataset": dataset,
            "case_count": len(items),
            "valid_uncapped_selection_count": len(selected),
            "selection_rate": len(selected) / len(items) if items else None,
            "selection_status_counts": statuses,
            "mean_unlabeled_present_fraction_selected": (
                statistics.fmean(valid_coverage) if valid_coverage else None),
            "mean_present_runs_selected": statistics.fmean(valid_runs) if valid_runs else None,
            "unknown_or_abstained_count": len(items) - len(selected),
        })
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "GT_read": True,
        "metric_scale": "0_to_1",
        "candidate_detection_recall": "not_measured",
        "aggregation": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired source case",
            "scope": "separately within each dataset; four historical development sources per domain",
            "ci_interpretation": "exploratory descriptive only",
            "pooled_cross_domain_gate": False,
            "ci_used_for_arm_or_step_selection": False,
            "limitation": "the fixed eight-source cohort is historically exposed and is not independent confirmation",
        },
        "arm_metric_summaries": arm_summaries,
        "contrasts": contrasts,
        "teacher_selection_and_coverage": teacher_summary,
        "contrast_definitions": [
            {"contrast": name, "comparison_arm": left, "baseline_arm": right}
            for name, left, right in _contrast_specs()
        ],
        "case_count_by_dataset": {dataset: len(case_ids_by_dataset[dataset]) for dataset in datasets},
    }


def _contrast_specs() -> list[tuple[str, str, str]]:
    specs = [
        ("T_minus_F", "T", "F"),
        ("native_teacher_direct_minus_T", "native_teacher_direct", "T"),
        ("T_direct_minus_T", "T_direct", "T"),
        ("T_iou_minus_T", "T_iou", "T"),
        ("T_reliability_weighted_iou_minus_T", "T_reliability_weighted_iou", "T"),
        ("direct_minus_F", "direct", "F"),
        ("iou_minus_F", "iou", "F"),
        ("reliability_weighted_iou_minus_F", "reliability_weighted_iou", "F"),
    ]
    return specs


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _paired_bootstrap(values: Sequence[float], dataset: str, contrast: str,
                      metric: str) -> dict[str, float] | None:
    if not values:
        return None
    seed_material = f"{BOOTSTRAP_SEED}|{dataset}|{contrast}|{metric}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        means.append(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lower = means[int(0.025 * (BOOTSTRAP_REPLICATES - 1))] * 100
    upper = means[int(0.975 * (BOOTSTRAP_REPLICATES - 1))] * 100
    return {"lower": lower, "upper": upper}


def _synthetic_prediction_writer_contract() -> dict[str, Any]:
    """Run the real student validator on one temporary writer-shaped fixture."""
    import numpy as np
    import torch
    from scripts import run_stvg_adjacent_motion_student_v1 as student_writer
    global BASE, REFERENCE_LOCK, ROSTER
    old_base, old_reference, old_roster = BASE, REFERENCE_LOCK, ROSTER
    try:
        with tempfile.TemporaryDirectory(prefix="stvg-native-score-contract-") as tmp:
            root = Path(tmp)
            BASE = root / "artifacts/stvg_native_instance_spatial_opd_v1"
            REFERENCE_LOCK = BASE / "REFERENCE_LOCK.json"
            ROSTER = BASE / "ROSTER_LOCK.json"
            (BASE / "teacher").mkdir(parents=True)
            (BASE / "student").mkdir(parents=True)
            REFERENCE_LOCK.write_text('{"synthetic":true}\n', encoding="utf-8")
            REFERENCE_LOCK_HASH = _sha(REFERENCE_LOCK)
            qa_interval = [100, 102]
            _write_once(ROSTER, {"records": [{"case_id": "fixture", "qa_interval": qa_interval}]})
            student_path = BASE / "student/fixture.pt"
            student_path.write_bytes(b"synthetic writer payload only")
            student_sha = _sha(student_path)

            frame_map = [
                {"local_frame_index": 0, "physical_frame_id": 100, "source_timestamp_s": 0.0},
                {"local_frame_index": 1, "physical_frame_id": 101, "source_timestamp_s": 0.04},
            ]
            native_interval = {
                "start_time_s": 0.0, "end_time_s": 0.04,
                "start_local_frame": 0, "end_local_frame": 1,
                "local_frame_indices": [0, 1], "physical_frame_ids": [100, 101],
            }
            teacher_frames = [
                {"local_frame_index": 0, "physical_frame_id": 100,
                 "source_timestamp_s": 0.0, "candidate_id": 1, "present": True,
                 "xyxy_inclusive_pixels": [0, 0, 0, 0], "mask_area_px": 1},
                {"local_frame_index": 1, "physical_frame_id": 101,
                 "source_timestamp_s": 0.04, "candidate_id": 1, "present": False,
                 "xyxy_inclusive_pixels": None, "mask_area_px": 0},
            ]
            teacher_record = {
                "input_binding": {"frame_map": frame_map},
                "projected_interval": {"start_local_frame": 0, "end_local_frame": 1},
                "mask_tube": {"frames": teacher_frames},
            }
            (BASE / "teacher/fixture.json").write_text(
                json.dumps(teacher_record), encoding="utf-8")
            ref = {
                "case_id": "fixture", "dataset": "hc2", "GT_read": False,
                "parsed_status": "valid", "candidate_id": 1,
                "native_interval": native_interval, "no_update": False,
                "anchors": [{"position": 0, "physical_frame_id": 100, "weight": 0.0}],
            }
            direct = {
                "selection_status": "valid", "valid_uncapped_selection": True,
                "candidate_id": 1, "native_interval": native_interval,
                "dense_frames": [
                    {**teacher_frames[0], "xyxy_normalized_continuous": [0.0, 0.0, 0.5, 0.5],
                     "cxcywh_normalized_continuous": [0.25, 0.25, 0.5, 0.5]},
                    {**teacher_frames[1], "xyxy_normalized_continuous": None,
                     "cxcywh_normalized_continuous": None},
                ],
            }
            source_boxes = torch.tensor([
                [0.12345679, 0.6345679, 0.2345679, 0.3456789],
                [0.436789, 0.523456, 0.111111, 0.222222],
            ], dtype=torch.float32)
            # Use the actual CPU writer conversion to make the fixture reject
            # a float32-arithmetic scorer by exact comparison.
            case_boxes = student_writer._xyxy(source_boxes)
            dtype_probe = torch.tensor(
                [[0.12345679, 0.6345679, 0.2345679, 0.3456789]], dtype=torch.float32)
            writer_probe = student_writer._xyxy(dtype_probe)
            scorer_probe = _cxcywh_to_xyxy(dtype_probe)
            assert writer_probe.dtype == np.float64 and np.array_equal(writer_probe, scorer_probe)
            logits = np.asarray([[0.1, 0.2]], dtype=np.float32)
            readout_boxes = source_boxes.detach().cpu().numpy().copy()
            config = {"steps": 5}
            initial = {"theta": np.asarray([0.0], dtype=np.float32)}
            active_fits = {}
            readouts = {}
            arms = {
                "F": {"boxes": case_boxes.copy(), "interval": [100, 102]},
                "T": {"boxes": case_boxes.copy(), "interval": qa_interval},
                "reliability_weighted_iou": {"boxes": case_boxes.copy(), "interval": [100, 102]},
                "T_reliability_weighted_iou": {"boxes": case_boxes.copy(), "interval": qa_interval},
            }
            for algorithm in ("direct", "iou"):
                active_fits[algorithm] = {
                    "algorithm": algorithm, "reference_sha256": REFERENCE_LOCK_HASH,
                    "optimizer_steps": 0, "selected_step": 5, "declared_steps": 5,
                    "active_parameters": 1792, "config": config,
                    "initial_state": initial,
                    "final_cached_decoder_boxes": readout_boxes,
                }
                readouts[algorithm] = {
                    "logits": logits, "physical_interval": [100, 102],
                    "boxes": readout_boxes,
                }
                out_xyxy = _cxcywh_to_xyxy(readout_boxes)
                arms[algorithm] = {"boxes": out_xyxy, "interval": [100, 102]}
                arms["T_" + algorithm] = {"boxes": out_xyxy, "interval": qa_interval}
            receipts = {
                "direct": {"status": "actual_updated_full_redecode", "optimizer_steps": 0,
                           "final_redecode_calls": 1},
                "iou": {"status": "actual_updated_full_redecode", "optimizer_steps": 0,
                        "final_redecode_calls": 1},
                "reliability_weighted_iou": {"status": "exact_no_update", "optimizer_steps": 0,
                                               "final_redecode_calls": 0},
            }
            student = {
                "case_id": "fixture", "dataset": "hc2", "GT_read": False,
                "online_carry": False, "source_case_sha256": "c" * 64,
                "reference_lock_sha256": REFERENCE_LOCK_HASH, "reference_record": ref,
                "source_model_state_sha256": "a" * 64, "pixel_sha256": "b" * 64,
                "source_restored": True,
                "source_cold_readback": {"status": "pass", "source_boxes_logits_bitwise": True,
                    "full_model_state_sha256": "a" * 64, "pixel_sha256": "b" * 64},
                "frame_ids": [100, 101], "source_boxes": source_boxes,
                "source_logits": logits, "Native_interval": [100, 102],
                "Projection_interval": [100, 102], "arms": arms,
                "fits": active_fits, "readouts": readouts, "receipts": receipts,
                "actual_fit_count": 2, "actual_final_redecode_calls": 2,
                "optimizer_steps": 0, "configuration": config, "initial_state": initial,
            }
            receipt = {"status": "sealed", "sha256": student_sha,
                       "bytes": student_path.stat().st_size,
                       "actual_fit_count": 2, "optimizer_steps": 0,
                       "actual_final_redecode_calls": 2}
            case = {"case_id": "fixture", "dataset": "hc2", "_source_case_sha256": "c" * 64,
                    "frame_ids": [100, 101], "native_boxes": case_boxes,
                    "Native_interval": [100, 102], "Projection_interval": [100, 102],
                    "input": {"width": 2, "height": 2}}
            validated = _validate_prediction(
                "fixture", case, ref, direct, student, receipt, student_sha, "d" * 64)
            assert validated["actual_fit_count"] == 2 and validated["optimizer_steps"] == 0
            assert set(student["arms"]) == set(ARMS) - {"native_teacher_direct"}
            return {"status": "pass", "student_arm_count": len(student["arms"]),
                    "active_fit_count": 2, "optimizer_steps": 0,
                    "partial_teacher_mask_frames": 1,
                    "decoder_xyxy_dtype": str(case_boxes.dtype),
                    "float32_decoder_values_were_converted_before_xyxy": True}
    finally:
        BASE, REFERENCE_LOCK, ROSTER = old_base, old_reference, old_roster


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to replace an existing score artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    _require(not temp.exists(), f"stale temporary output prevents write: {temp.name}")
    temp.write_text(data, encoding="utf-8")
    os.replace(temp, path)


def _run_sealed_score() -> dict[str, Any]:
    output_paths = [BASE / "POSTSEAL_SCORE_ROWS.json",
                    BASE / "POSTSEAL_SCORE_SUMMARY.json",
                    BASE / "CPU_COMPLETION.json"]
    _require(not any(path.exists() for path in output_paths),
             "post-seal score outputs already exist; refusing overwrite")

    locked = _load_locked_inputs()
    source_cases = _load_source_cases(locked)
    student_outputs: dict[str, dict[str, Any]] = {}
    for case_id in locked["case_ids"]:
        path = locked["student_paths"][case_id]
        raw = _safe_torch_load(path)
        _require(isinstance(raw, Mapping), f"student prediction is not an object: {case_id}")
        raw = dict(raw)
        checked = _validate_prediction(
            case_id, source_cases[case_id], locked["ref_by_case"][case_id],
            locked["direct_by_case"][case_id], raw,
            locked["student_receipts"][case_id], _sha(path),
            _sha(BASE / "student" / f"{case_id}.receipt.json"))
        student_outputs[case_id] = {"raw": raw, "validated": checked}

    # This is the first label read. Every source, phase, runtime, teacher,
    # reference, student payload, and receipt has already passed hash checks.
    from scripts import review_stvg_sparse_r1_v2 as scorer
    targets, gt_provenance = scorer._truths_after_seal(source_cases)
    official, independent, metric_contract = scorer._metric_functions()
    rows, summary = _evaluate(locked, source_cases, student_outputs,
                              targets, official, independent)

    seal_relative = str(SEAL.relative_to(ROOT))
    common_bindings = {
        "roster_sha256": locked["roster_sha256"],
        "reference_lock_sha256": locked["reference_lock_sha256"],
        "direct_teacher_readout_sha256": locked["direct_readout_sha256"],
        "tracking_barrier_sha256": locked["phase_hashes"]["TRACKING"],
        "teacher_barrier_sha256": locked["phase_hashes"]["TEACHER"],
        "student_barrier_sha256": locked["phase_hashes"]["STUDENT"],
    }
    rows_document = {
        "schema": ROWS_SCHEMA,
        "status": "complete",
        "GT_read": True,
        "prediction_barrier": {"path": seal_relative, "sha256": locked["seal_sha256"]},
        "bindings": common_bindings,
        "metric_scale": "0_to_1",
        "case_count": 8,
        "arm_count": len(ARMS),
        "rows": rows,
    }
    summary["prediction_barrier"] = {"path": seal_relative, "sha256": locked["seal_sha256"]}
    summary["bindings"] = common_bindings
    summary["official_metric_contract_schema"] = metric_contract.get("schema")
    summary["official_truth_binding_digest"] = hashlib.sha256(
        _canonical_bytes(gt_provenance)).hexdigest() if gt_provenance else None
    completion = {
        "schema": "stvg-native-instance-cpu-completion-v1",
        "status": "pending_root_review_visualization_publication",
        "scoring_status": "complete",
        "experiment_complete": False,
        "GT_read": True,
        "case_count": 8,
        "arm_count": len(ARMS),
        "row_count": len(rows),
        "prediction_barrier_sha256": locked["seal_sha256"],
        "rows_sha256": hashlib.sha256(_canonical_bytes(rows_document)).hexdigest(),
        "summary_sha256": hashlib.sha256(_canonical_bytes(summary)).hexdigest(),
        "pending": ["root independent review", "actual visual inspection", "anonymous publication", "archive update"],
    }
    _write_once(output_paths[0], rows_document)
    _write_once(output_paths[1], summary)
    _write_once(output_paths[2], completion)
    return completion


def _synthetic_contracts() -> dict[str, Any]:
    """Exercise summary/support math with made-up in-memory values only."""
    fake_rows: list[dict[str, Any]] = []
    for dataset in ("hc2", "vidstg"):
        for i in range(4):
            cid = f"{dataset}_synthetic_{i}"
            for arm in ARMS:
                value = None if (dataset == "hc2" and i == 3 and arm == "native_teacher_direct") else .25 + i * .01
                fake_rows.append({"case_id": cid, "dataset": dataset, "arm": arm,
                                  "metrics": {metric: value for metric in METRICS}})
    summary = _summarize(fake_rows, {
        "hc2": [{"case_id": f"hc2_synthetic_{i}",
                 "selection_status": "valid" if i < 2 else "abstain",
                 "valid_uncapped_selection": i < 2,
                 "coverage_fraction": .75 if i < 2 else None,
                 "present_run_count": 1 if i < 2 else None} for i in range(4)],
        "vidstg": [{"case_id": f"vidstg_synthetic_{i}",
                    "selection_status": "valid", "valid_uncapped_selection": True,
                    "coverage_fraction": 1.0, "present_run_count": 1} for i in range(4)],
    })
    missing = next(x for x in summary["contrasts"] if
                   x["dataset"] == "hc2" and x["contrast"] == "native_teacher_direct_minus_T" and
                   x["metric"] == "m_vIoU")
    assert missing["unknown_count"] == 1 and missing["n_pairs"] == 3
    assert len(summary["contrasts"]) == 2 * len(_contrast_specs()) * len(METRICS)
    assert all(x["dataset"] in {"hc2", "vidstg"} for x in summary["arm_metric_summaries"])
    assert summary["aggregation"]["pooled_cross_domain_gate"] is False

    # Off-grid known labels are still part of the unsupervised full-tube
    # support, while source-grid position counts remain a separate quantity.
    grid = [10, 20, 30]
    fake_truth = {10: [0, 0, 1, 1], 15: [0, 0, 1, 1], 20: [0, 0, 1, 1],
                  35: [0, 0, 1, 1]}
    support, known = _known_positions(grid, fake_truth, {1})
    assert support["supervised"] == [20]
    assert support["unsupervised"] == [10, 15, 35]
    assert set(known["supervised"]).isdisjoint(known["unsupervised"])
    assert set(known["supervised"]) | set(known["unsupervised"]) == set(fake_truth)
    empty_support, empty_known = _known_positions(grid, fake_truth, set())
    assert empty_support["supervised"] == [] and empty_known["unsupervised"] == sorted(fake_truth)

    # Recompute the native Teacher physical-grid metric independently on a
    # partial tube with missing masks and an off-grid GT frame.
    from scripts.stvg_target_teacher_evaluate_v2 import _teacher_physical_dense_metric
    partial_record = {
        "frame_ids": [10, 11, 12, 13],
        "present_mask": [True, False, True, False],
        "boxes_cxcywh_normalized": [[0.25, 0.25, 0.5, 0.5], None,
                                    [0.5, 0.5, 0.5, 0.5], None],
        "interval_frame_bounds": [10, 14],
    }
    partial_truth = {
        10: [0, 0, 50, 50], 11: [0, 0, 50, 50], 12: [25, 25, 75, 75],
        13: [0, 0, 50, 50], 14: [0, 0, 50, 50],
    }
    partial_span = [10, 15]
    official_partial, _ = _teacher_physical_dense_metric(
        partial_record, {"frame_ids": partial_record["frame_ids"],
                        "input": {"width": 100, "height": 100}},
        partial_truth, partial_span, "vidstg")
    independent_partial = _independent_teacher_metrics(
        partial_record, 100, 100, partial_truth, partial_span, "vidstg")
    partial_error = max(abs(float(official_partial[key]) - independent_partial[key])
                        for key in independent_partial)
    assert partial_error <= 2e-12

    writer_contract = _synthetic_prediction_writer_contract()

    # Frame-clock conversion and absent-mask bookkeeping are also synthetic.
    clock = [{"physical_frame_id": 100 + i, "source_timestamp_s": i / 25.0}
             for i in range(5)]
    fake_case = {"input": {"fps": 25}, "video_bounds": [100, 105]}
    fake_teacher = {"parsed": {"start_time_s": 1 / 25, "end_time_s": 4 / 25},
                    "projected_interval": {"start_local_frame": 1, "end_local_frame": 4},
                    "input_binding": {"frame_map": clock}}
    fake_interval = _native_time_interval(fake_case,
                                          {"native_interval": {"start_time_s": 1 / 25,
                                                                "end_time_s": 4 / 25,
                                                                "start_local_frame": 1,
                                                                "end_local_frame": 4}},
                                          fake_teacher)
    assert fake_interval == [101.0, 104.0]
    assert _present_run_count([True, False, True, True, False]) == 2
    assert _finite_number(.0) and not _finite_number(None) and not _finite_number(float("nan"))
    return {
        "schema": "stvg-native-instance-score-contract-v1",
        "status": "pass",
        "GT_read": False,
        "model_calls": 0,
        "checks": {
            "all_primary_and_attribution_arm_schema": True,
            "dataset_separate_paired_summaries": True,
            "unknown_metrics_not_zero_filled": True,
            "paired_bootstrap_deterministic": _paired_bootstrap([.1, -.1, .0], "hc2", "x", "m_vIoU") ==
                _paired_bootstrap([.1, -.1, .0], "hc2", "x", "m_vIoU"),
            "native_time_to_physical_frame_interval": fake_interval == [101.0, 104.0],
            "candidate_detection_recall_explicitly_unmeasured": summary["candidate_detection_recall"] == "not_measured",
            "present_run_count_tracks_fragmentation": True,
            "no_cross_domain_pooled_gate": True,
            "off_grid_gt_in_unsupervised_full_tube_support": True,
            "supervised_unsupervised_gt_partition_covers_known_support": True,
            "native_partial_missing_mask_metric_matches_independent_arithmetic": partial_error <= 2e-12,
            "student_writer_schema_passes_real_prediction_validator": writer_contract["status"] == "pass",
        },
        "synthetic_unknown_native_teacher_pairs": missing["unknown_count"],
        "synthetic_native_partial_mask_metric_max_abs_error": partial_error,
        "synthetic_student_writer_fixture": writer_contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run in-memory synthetic CPU contracts only")
    parser.add_argument("--score-after-seal", action="store_true",
                        help="open official GT only after all prediction barrier checks pass")
    args = parser.parse_args(argv)
    if args.contracts:
        print(json.dumps(_synthetic_contracts(), ensure_ascii=False, indent=2))
        return 0
    if not args.score_after_seal:
        parser.error("scoring is disabled unless --score-after-seal is supplied")
    completion = _run_sealed_score()
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
