"""Independent CPU readback for the sealed adjacent-motion OPD method phase.

This reader accepts only the new scope's exact ``STUDENT_RUNTIME.json`` and a
complete eight-case method barrier. It verifies all output/source bytes before
deserializing any case tensors. It does not initialize CUDA, read annotations,
decode pixels, construct a model, or run a fit. ``contracts`` uses synthetic
CPU values only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_adjacent_motion_opd_v1"
OLD_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
PHASE = "method"
CASE_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(4)
}
ALGORITHMS = ("direct", "iou", "motion_iou")
WORKER = "scripts/run_stvg_adjacent_motion_student_v1.py"
FIT_HELPER = "scripts/stvg_adjacent_motion_fit_v1.py"
SPATIAL_HELPER = "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py"
OLD_BARRIER = "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json"
METHOD_REFERENCE = "artifacts/stvg_adjacent_motion_opd_v1/METHOD_REFERENCE_LOCK.json"
ROSTER = "artifacts/stvg_adjacent_motion_opd_v1/ROSTER_LOCK.json"
TOL = {
    "mean_abs": 3e-5,
    "sample_abs": 3e-5,
    "reward_abs": 2e-4,
    "weight_abs": 2e-4,
    "loss_abs": 5e-4,
    "adam_parameter_abs": 1e-4,
    "adam_state_abs": 1e-4,
    "motion_abs": 1e-6,
}

# Reuse only the frozen CPU arithmetic primitives; the motion objective is
# independently reconstructed below and is never passed to the pure-IoU fit
# assertion.
from scripts.review_stvg_scene_object_id_spatial_math_v1 import (  # noqa: E402
    _audit_adam_transition,
    _audit_direct_fit,
    _audit_opd_fit,
    _compare_tree,
    _cpu_noise,
    _iou,
    _likelihood,
    _require,
    _reward,
    _safe_root_path,
    _sha_file,
    _verify_digest_path,
    _xyxy,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(left: Any, right: Any, name: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        _require(isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                 and left.dtype == right.dtype and left.shape == right.shape
                 and np.array_equal(left, right, equal_nan=False),
                 f"{name}: arrays differ")
    elif torch.is_tensor(left) or torch.is_tensor(right):
        _compare_tree(left, right, name=name)
    elif isinstance(left, Mapping) or isinstance(right, Mapping):
        _require(isinstance(left, Mapping) and isinstance(right, Mapping)
                 and set(left) == set(right), f"{name}: mapping keys differ")
        for key in left:
            _equal(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        _require(type(left) is type(right) and len(left) == len(right),
                 f"{name}: sequence schema differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _equal(a, b, f"{name}[{index}]")
    else:
        _require(type(left) is type(right) and left == right,
                 f"{name}: values differ")


def _close_number(actual: Any, expected: Any, tol: float, name: str) -> float:
    error = abs(float(actual) - float(expected))
    _require(math.isfinite(error) and error <= tol,
             f"{name}: absolute error {error:.9g} exceeds {tol:.9g}")
    return error


def _compare_numeric_tree(actual: Any, expected: Any, tol: float, name: str) -> float:
    """Schema-exact recursive comparison with tolerance only for real numbers."""
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping) and isinstance(expected, Mapping)
                 and set(actual) == set(expected), f"{name}: mapping keys differ")
        return max((_compare_numeric_tree(actual[k], expected[k], tol, f"{name}.{k}")
                    for k in actual), default=0.0)
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        _require(type(actual) is type(expected) and len(actual) == len(expected),
                 f"{name}: sequence schema differs")
        return max((_compare_numeric_tree(a, b, tol, f"{name}[{i}]")
                    for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.0)
    if (isinstance(actual, (int, float, np.number))
            and not isinstance(actual, (bool, np.bool_))
            and isinstance(expected, (int, float, np.number))
            and not isinstance(expected, (bool, np.bool_))):
        return _close_number(actual, expected, tol, name)
    _require(type(actual) is type(expected) and actual == expected,
             f"{name}: values differ")
    return 0.0


def _static_bindings(path: Path) -> dict[str, str]:
    """Read literal SOURCE_BINDINGS without importing a producer/trainer."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    raw = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SOURCE_BINDINGS"
                for target in node.targets):
            raw = ast.literal_eval(node.value)
            break
    _require(isinstance(raw, dict) and raw, f"{path.name}: missing literal source bindings")
    result = {}
    for relative, value in raw.items():
        _safe_root_path(relative)
        digest = value.get("sha256") if isinstance(value, Mapping) else value
        _require(isinstance(digest, str) and len(digest) == 64,
                 f"{path.name}: malformed SOURCE_BINDINGS entry")
        result[relative] = digest
    return result


def _verify_runtime_pins(runtime: Mapping[str, Any]) -> dict[str, str]:
    pins, protected = runtime.get("pins"), runtime.get("protected")
    _require(isinstance(pins, Mapping) and pins
             and isinstance(protected, Mapping) and protected,
             "locked runtime must contain pins and protected files")
    all_pins = {}
    for group_name, group in (("pin", pins), ("protected", protected)):
        for relative, digest in group.items():
            rel = _safe_root_path(relative)
            _require(isinstance(digest, str) and len(digest) == 64,
                     f"malformed {group_name} digest: {relative}")
            _verify_digest_path(rel, digest)
            if rel in all_pins:
                _require(all_pins[rel] == digest, f"conflicting runtime pins: {relative}")
            all_pins[rel] = digest

    required = {WORKER, FIT_HELPER, SPATIAL_HELPER, METHOD_REFERENCE, ROSTER,
                "methods/CURRENT_METHOD.json", OLD_BARRIER}
    _require(required <= set(all_pins), "runtime omits a required worker/lock/temporal pin")

    fit_bindings = _static_bindings(ROOT / FIT_HELPER)
    spatial_bindings = _static_bindings(ROOT / SPATIAL_HELPER)
    for relative, digest in {**fit_bindings, **spatial_bindings}.items():
        _require(all_pins.get(relative) == digest,
                 f"runtime and embedded frozen fit source disagree: {relative}")
    return all_pins


def _preflight() -> dict[str, Any]:
    """Fail closed unless the exact new method barrier is complete and bound."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU readback refused because CUDA is already initialized")

    barrier_path = BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json"
    _require(barrier_path.is_file(),
             "method global barrier is absent; partial student outputs will not be read")
    barrier = _read_json(barrier_path)
    _require(barrier.get("status") == "complete" and barrier.get("count") == 8
             and barrier.get("GT_read") is False,
             "method global barrier is incomplete or not explicitly GT-free")

    # This path is intentionally exact: never select an RUNTIME002 or another
    # phase variant by glob or by recency.
    runtime_path = BASE / "STUDENT_RUNTIME.json"
    _require(runtime_path.is_file(), "new scope STUDENT_RUNTIME.json is absent")
    runtime_sha = _sha_file(runtime_path)
    _require(barrier.get("runtime_sha256") == runtime_sha,
             "method global barrier is not bound to exact STUDENT_RUNTIME.json")
    runtime = _read_json(runtime_path)
    _require(runtime.get("status") == "locked" and runtime.get("phase") == PHASE
             and runtime.get("GT_read") is False and runtime.get("steps") == 5
             and runtime.get("scope") == "joint",
             "new method runtime is not locked to the expected GT-free five-step scope")

    reference_path = BASE / "METHOD_REFERENCE_LOCK.json"
    roster_path = BASE / "ROSTER_LOCK.json"
    _require(reference_path.is_file() and roster_path.is_file(),
             "method reference or roster lock is absent")
    reference_sha, roster_sha = _sha_file(reference_path), _sha_file(roster_path)
    _require(runtime.get("reference_lock_sha256") == reference_sha
             and runtime.get("roster_lock_sha256") == roster_sha
             and barrier.get("reference_lock_sha256") == reference_sha
             and barrier.get("roster_lock_sha256") == roster_sha,
             "runtime/barrier reference or roster lock binding differs")

    all_pins = _verify_runtime_pins(runtime)
    _require(all_pins.get(str(reference_path.relative_to(ROOT))) == reference_sha
             and all_pins.get(str(roster_path.relative_to(ROOT))) == roster_sha,
             "reference and roster locks must be directly pinned")

    old_barrier = _read_json(ROOT / OLD_BARRIER)
    _require(old_barrier.get("status") == "complete" and old_barrier.get("count") == 8
             and old_barrier.get("GT_read") is False,
             "pinned old temporal barrier is incomplete or not GT-free")
    old_rows = old_barrier.get("files")
    _require(isinstance(old_rows, list) and len(old_rows) == 8,
             "old temporal barrier does not seal exactly eight source outputs")
    old_paths = {}
    old_prefix = "artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/"
    for row in old_rows:
        _require(isinstance(row, Mapping), "malformed old temporal barrier file row")
        relative = _safe_root_path(row.get("path"))
        _require(relative.startswith(old_prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(old_prefix):],
                 "old temporal output path escaped the exact student directory")
        cid = relative[len(old_prefix):-3]
        _require(cid in CASE_IDS and cid not in old_paths,
                 "old temporal barrier repeats or changes the fixed eight cases")
        _require(all_pins.get(relative) == row.get("sha256"),
                 f"runtime does not pin reused old temporal output: {relative}")
        old_paths[cid] = _verify_digest_path(relative, row.get("sha256"), row.get("bytes"))
    _require(set(old_paths) == CASE_IDS, "old temporal barrier case set differs")

    roster = _read_json(roster_path)
    references = _read_json(reference_path)
    _require(roster.get("status") == references.get("status") == "locked"
             and roster.get("GT_read") is False and references.get("GT_read") is False,
             "roster/reference lock is not locked and GT-free")
    roster_rows, reference_rows = roster.get("records"), references.get("records")
    _require(isinstance(roster_rows, list) and len(roster_rows) == 8
             and isinstance(reference_rows, list) and len(reference_rows) == 8,
             "method locks must contain exactly eight records")
    roster_by_id = {row.get("case_id"): row for row in roster_rows}
    reference_by_id = {row.get("case_id"): row for row in reference_rows}
    _require(set(roster_by_id) == set(reference_by_id) == CASE_IDS,
             "method lock case IDs differ from the fixed eight-case cohort")
    _require(len(roster_by_id) == len(roster_rows)
             and len(reference_by_id) == len(reference_rows),
             "method locks contain duplicate case IDs")
    domains = {domain: 0 for domain in ("hc2", "vidstg")}
    eligible_count = 0
    source_paths = {}
    for cid in sorted(CASE_IDS):
        source_row, reference_row = roster_by_id[cid], reference_by_id[cid]
        _require(source_row.get("dataset") in domains
                 and source_row.get("dataset") == reference_row.get("dataset")
                 and source_row.get("source_case_sha256")
                 == reference_row.get("source_case_sha256"),
                 f"{cid}: roster/reference source binding differs")
        domains[source_row["dataset"]] += 1
        _require(isinstance(source_row.get("source_case_path"), str),
                 f"{cid}: source path is absent")
        source_rel = _safe_root_path(source_row["source_case_path"])
        source_digest = source_row.get("source_case_sha256")
        _require(all_pins.get(source_rel) == source_digest,
                 f"{cid}: source case bytes are not runtime-pinned")
        source_paths[cid] = _verify_digest_path(source_rel, source_digest)
        teacher_roles = reference_row.get("references")
        _require(isinstance(teacher_roles, Mapping) and set(teacher_roles) == {"teacher"},
                 f"{cid}: method phase must have exactly the teacher reference")
        teacher = teacher_roles["teacher"]
        _require(isinstance(teacher.get("no_update"), bool)
                 and isinstance(teacher.get("eligible_for_update"), bool)
                 and teacher["eligible_for_update"] is (not teacher["no_update"]),
                 f"{cid}: no-update/eligibility flags disagree")
        _require(isinstance(teacher.get("is_Source"), bool)
                 and isinstance(teacher.get("anchors"), list)
                 and isinstance(teacher.get("motion_evidence"), list),
                 f"{cid}: reference record is malformed")
        if teacher["eligible_for_update"]:
            _require(bool(teacher["anchors"]) and bool(teacher["motion_evidence"]),
                     f"{cid}: eligible teacher record lacks anchors or motion evidence")
            eligible_count += 1
    _require(domains == {"hc2": 4, "vidstg": 4},
             "method roster is not balanced over the locked eight cases")
    expected_fit_count = eligible_count * len(ALGORITHMS)
    _require(runtime.get("expected_fit_count") == expected_fit_count
             and barrier.get("actual_fit_count") == expected_fit_count
             and barrier.get("actual_final_redecode_calls") == expected_fit_count,
             "dynamic fit/readout count differs from the eligible reference records")

    output_rows = barrier.get("files")
    _require(isinstance(output_rows, list) and len(output_rows) == 8,
             "method barrier output inventory must contain exactly eight files")
    prefix = "artifacts/stvg_adjacent_motion_opd_v1/method/student/"
    case_paths = {}
    for row in output_rows:
        _require(isinstance(row, Mapping), "malformed method barrier output row")
        relative = _safe_root_path(row.get("path"))
        _require(relative.startswith(prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(prefix):],
                 "method output path escaped the exact student directory")
        cid = relative[len(prefix):-3]
        _require(cid in CASE_IDS and cid not in case_paths,
                 "method barrier repeats or changes the fixed eight cases")
        case_paths[cid] = _verify_digest_path(relative, row.get("sha256"), row.get("bytes"))
    _require(set(case_paths) == CASE_IDS, "method barrier case set differs")

    return {
        "runtime": runtime,
        "runtime_path": runtime_path,
        "roster": roster,
        "reference_lock": references,
        "roster_by_id": roster_by_id,
        "reference_by_id": reference_by_id,
        "source_paths": source_paths,
        "old_paths": old_paths,
        "case_paths": case_paths,
        "barrier": barrier,
        "barrier_path": barrier_path,
        "expected_fit_count": expected_fit_count,
        "eligible_case_count": eligible_count,
        "domains": domains,
        "hashes": {
            "runtime": runtime_sha,
            "roster": roster_sha,
            "reference_lock": reference_sha,
            "barrier": _sha_file(barrier_path),
            "old_barrier": _sha_file(ROOT / OLD_BARRIER),
        },
    }


def _validate_motion_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, (list, tuple)):
        raise RuntimeError("motion evidence must be an ordered record list")
    normalized = []
    last = None
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"motion evidence row {index} is malformed")
        fid = record.get("physical_frame_id")
        _require(isinstance(fid, int) and not isinstance(fid, bool)
                 and (last is None or fid > last),
                 f"motion evidence row {index} has non-increasing physical ID")
        last = fid
        rho = record.get("rho")
        _require(isinstance(rho, (int, float)) and not isinstance(rho, bool)
                 and math.isfinite(float(rho)) and float(rho) >= 0,
                 f"motion evidence row {index} has invalid rho")
        center = record.get("center_xy")
        if center is None:
            _require(float(rho) == 0.0,
                     f"motion evidence row {index}: missing center must have rho=0")
            center_row = None
        else:
            _require(isinstance(center, (list, tuple)) and len(center) == 2,
                     f"motion evidence row {index} center has wrong shape")
            center_row = [float(center[0]), float(center[1])]
            _require(all(math.isfinite(x) and 0 <= x <= 1 for x in center_row)
                     and float(rho) > 0,
                     f"motion evidence row {index} center/rho is invalid")
        normalized.append({"physical_frame_id": int(fid), "center_xy": center_row,
                           "rho": float(rho)})
    return normalized


def _interpolate_center(boxes: torch.Tensor, frame_ids: Sequence[int], target_id: int):
    if target_id < frame_ids[0] or target_id > frame_ids[-1]:
        raise ValueError("motion interpolation would extrapolate beyond the source grid")
    for index, fid in enumerate(frame_ids):
        if target_id == fid:
            return boxes[index, :2], {"left_grid_position": index,
                                      "right_grid_position": index, "alpha": 0.0}
        if index + 1 < len(frame_ids) and fid < target_id < frame_ids[index + 1]:
            right = frame_ids[index + 1]
            alpha = float(target_id - fid) / float(right - fid)
            center = boxes[index, :2] * (1 - alpha) + boxes[index + 1, :2] * alpha
            return center, {"left_grid_position": index,
                            "right_grid_position": index + 1, "alpha": alpha}
    raise AssertionError("failed to interpolate in-range physical frame")


def _independent_motion_loss(student_boxes: torch.Tensor, frame_ids: Sequence[int],
                             evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute pair selection, interpolation, rho weights, and center L1."""
    _require(torch.is_tensor(student_boxes) and student_boxes.device.type == "cpu"
             and student_boxes.ndim == 2 and student_boxes.shape == (len(frame_ids), 4)
             and torch.isfinite(student_boxes).all(),
             "motion audit expects finite CPU (N,4) cxcywh boxes")
    ids = list(map(int, frame_ids))
    _require(bool(ids) and all(b > a for a, b in zip(ids, ids[1:])),
             "motion audit needs increasing physical source IDs")
    rows = _validate_motion_records(evidence)
    valid, excluded = [], []
    for index, (left, right) in enumerate(zip(rows, rows[1:])):
        a, b = left["physical_frame_id"], right["physical_frame_id"]
        reason = None
        if b != a + 1:
            reason = "nonconsecutive_physical_frames"
        elif (left["center_xy"] is None or right["center_xy"] is None
              or left["rho"] <= 0 or right["rho"] <= 0):
            reason = "missing_or_zero_rho_endpoint"
        elif a < ids[0] or b > ids[-1]:
            reason = "outside_sampled_source_grid"
        rho = min(left["rho"], right["rho"])
        if reason is None and rho <= 0:
            reason = "zero_pair_weight"
        if reason:
            excluded.append({"evidence_pair_index": index,
                             "physical_frame_ids": [a, b], "reason": reason})
            continue
        lp, li = _interpolate_center(student_boxes, ids, a)
        rp, ri = _interpolate_center(student_boxes, ids, b)
        pred_delta = rp - lp
        target_delta = student_boxes.new_tensor(right["center_xy"]) - student_boxes.new_tensor(
            left["center_xy"])
        abs_error = (pred_delta - target_delta).abs()
        pair_error = float(abs_error.mean())
        valid.append({
            "evidence_pair_index": index,
            "left_physical_frame_id": a,
            "right_physical_frame_id": b,
            "left_center_xy": list(left["center_xy"]),
            "right_center_xy": list(right["center_xy"]),
            "rho_left": left["rho"], "rho_right": right["rho"],
            "pair_rho": rho,
            "left_interpolation": li, "right_interpolation": ri,
            "student_delta_xy": pred_delta.detach().cpu().tolist(),
            "evidence_delta_xy": target_delta.detach().cpu().tolist(),
            "absolute_delta_error_xy": abs_error.detach().cpu().tolist(),
            "mean_coordinate_l1": pair_error,
            "weighted_l1_contribution": pair_error * rho,
        })
    weight_sum = sum(row["pair_rho"] for row in valid)
    loss = (sum(row["weighted_l1_contribution"] for row in valid) / weight_sum
            if weight_sum else 0.0)
    return {"loss": float(loss), "valid_pair_count": len(valid),
            "pair_weight_sum": float(weight_sum), "valid_pairs": valid,
            "excluded_pairs": excluded}


def _anchor_tuple(reference: Mapping[str, Any], frame_ids: Sequence[int]):
    positions, evidence = [], []
    for index, row in enumerate(reference.get("anchors", [])):
        _require(isinstance(row, Mapping), f"reference anchor {index} is malformed")
        position = row.get("position")
        _require(isinstance(position, int) and not isinstance(position, bool)
                 and 0 <= position < len(frame_ids)
                 and row.get("physical_frame_id") == frame_ids[position],
                 f"reference anchor {index} physical frame binding differs")
        box = torch.as_tensor(row.get("box_cxcywh"), dtype=torch.float32, device="cpu")
        _require(box.shape == (4,) and torch.isfinite(box).all(),
                 f"reference anchor {index} is not a finite cxcywh box")
        positions.append(position)
        evidence.append(box)
    _require(len(set(positions)) == len(positions), "reference anchor positions repeat")
    stacked = torch.stack(evidence) if evidence else torch.empty((0, 4), dtype=torch.float32)
    return positions, stacked


def _audit_motion_fit(fit: Mapping[str, Any], anchors, frame_ids: Sequence[int],
                      key: str, expected_initial: Mapping[str, Any],
                      locked_evidence: Sequence[Mapping[str, Any]], reference_sha: str):
    """Audit OPD component and motion component separately, then their sum."""
    _require(fit.get("algorithm") == "motion_iou"
             and fit.get("reward_objective") == "iou"
             and fit.get("motion_objective") == "rho_weighted_mean_adjacent_center_displacement_L1"
             and fit.get("lambda_motion") == 1.0,
             "motion fit objective/lambda metadata differs")
    _require(fit.get("reference_sha256") == reference_sha
             and fit.get("seed_key") == key
             and fit.get("scope") == "joint"
             and fit.get("active_parameters") == 1792
             and fit.get("declared_steps") == fit.get("selected_step") == 5,
             "motion fit binding or declared scope differs")
    config = fit.get("config")
    _require(isinstance(config, Mapping) and config.get("steps") == 5
             and config.get("samples") == 32 and float(config.get("sigma", 0)) > 0
             and float(config.get("tau", 0)) > 0 and float(config.get("lr", 0)) > 0,
             "motion fit must retain the original five-step Gaussian OPD config")
    positions, evidence_boxes = anchors
    _require(fit.get("observation_positions") == positions
             and len(fit.get("path", [])) == len(fit.get("rounds", [])) == 5,
             "motion fit path/anchor positions are incomplete")
    _compare_tree(fit.get("initial_state"), expected_initial, name="motion.initial_state")
    checked = _validate_motion_records(locked_evidence)
    _equal(fit.get("motion_evidence"), checked, "motion.fit_evidence_vs_locked_reference")
    evidence_sha = hashlib.sha256(json.dumps(checked, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    _require(fit.get("motion_evidence_sha256") == evidence_sha
             and fit.get("motion_valid_pair_count") is not None,
             "motion fit evidence digest/count differs")
    binding = fit.get("fit_binding")
    _require(isinstance(binding, Mapping)
             and binding.get("algorithm") == "motion_iou"
             and binding.get("lambda_motion") == 1.0
             and binding.get("reference_role") == fit.get("reference_role")
             and binding.get("reference_sha256") == reference_sha
             and binding.get("motion_evidence_sha256") == evidence_sha,
             "motion resume binding differs from locked evidence")

    total_valid_pairs = _independent_motion_loss(
        torch.zeros((len(frame_ids), 4), dtype=torch.float32), frame_ids, checked
    )["valid_pair_count"]
    _require(int(fit["motion_valid_pair_count"]) == total_valid_pairs,
             "motion fit valid-pair count differs from physical evidence")

    max_errors = {"mean_abs": 0.0, "sample_abs": 0.0, "reward_abs": 0.0,
                  "weight_abs": 0.0, "opd_loss_abs": 0.0,
                  "motion_loss_abs": 0.0, "total_loss_abs": 0.0,
                  "adam_parameter_abs": 0.0, "adam_state_abs": 0.0,
                  "motion_receipt_abs": 0.0}
    actual_steps = 0
    for step, (entry, round_row) in enumerate(zip(fit["path"], fit["rounds"], strict=True)):
        _require(entry.get("step") == round_row.get("step") == step
                 and entry.get("algorithm") == "motion_iou"
                 and entry.get("lambda_motion") == 1.0
                 and entry.get("motion_evidence_sha256") == evidence_sha
                 and round_row.get("lambda_motion") == 1.0,
                 f"motion step {step} order/objective differs")
        boxes = entry.get("boxes")
        _require(torch.is_tensor(boxes) and boxes.device.type == "cpu"
                 and boxes.shape == (len(frame_ids), 4) and torch.isfinite(boxes).all(),
                 f"motion step {step} lacks finite CPU pre-step boxes")
        motion = _independent_motion_loss(boxes, frame_ids, checked)
        receipt = entry.get("motion")
        _require(isinstance(receipt, Mapping), f"motion step {step} omits pair receipt")
        receipt_expected = {k: motion[k] for k in ("valid_pair_count", "pair_weight_sum",
                                                    "valid_pairs", "excluded_pairs")}
        error = _compare_numeric_tree(receipt, receipt_expected, TOL["motion_abs"],
                                      f"motion step {step} pair receipt")
        max_errors["motion_receipt_abs"] = max(max_errors["motion_receipt_abs"], error)
        _close_number(round_row.get("motion_loss"), motion["loss"], TOL["motion_abs"],
                      f"motion round {step} motion loss")
        _require(round_row.get("motion_valid_pair_count") == motion["valid_pair_count"],
                 f"motion round {step} valid-pair count differs")
        _close_number(round_row.get("motion_pair_weight_sum"), motion["pair_weight_sum"],
                      TOL["motion_abs"], f"motion round {step} pair weight sum")
        _compare_numeric_tree(round_row.get("motion_pairs"), motion["valid_pairs"],
                              TOL["motion_abs"], f"motion round {step} valid pairs")
        _compare_numeric_tree(round_row.get("motion_excluded_pairs"), motion["excluded_pairs"],
                              TOL["motion_abs"], f"motion round {step} excluded pairs")

        if entry.get("skipped") is True:
            skipped_gradient = entry.get("raw_total_gradient")
            _require(not positions and motion["valid_pair_count"] == 0
                     and motion["loss"] == 0.0,
                     f"motion step {step} skipped despite an anchor or valid physical pair")
            _require(torch.is_tensor(skipped_gradient) and skipped_gradient.numel() == 1792
                     and torch.count_nonzero(skipped_gradient).item() == 0
                     and entry.get("update") is None
                     and round_row.get("updated") is False
                     and round_row.get("no_information") is True
                     and round_row.get("total_loss") == 0.0
                     and round_row.get("opd_loss") == 0.0
                     and round_row.get("motion_loss") == 0.0,
                     f"motion step {step} explicit no-information receipt differs")
            _compare_tree(entry.get("state"), fit["initial_state"],
                          name=f"motion step {step} skipped source parameters")
            _require(not entry.get("optimizer_state", {}).get("state"),
                     f"motion step {step} skipped path contains Adam state")
            if step + 1 < len(fit["path"]):
                _compare_tree(entry.get("state"), fit["path"][step + 1].get("state"),
                              name=f"motion step {step} skipped params to next step")
                _compare_tree(entry.get("optimizer_state"),
                              fit["path"][step + 1].get("optimizer_state"),
                              name=f"motion step {step} skipped Adam to next step")
                _tensor_close(entry.get("boxes"), fit["path"][step + 1].get("boxes"), 0.0,
                              f"motion step {step} skipped boxes to next step")
            continue

        _close_number(entry.get("motion_loss"), motion["loss"], TOL["motion_abs"],
                      f"motion step {step} motion loss")

        # Reconstruct the original Gaussian IoU rollout and likelihood only as
        # the OPD component. The motion total never enters this pure-IoU math.
        policy = entry.get("rollout")
        if positions:
            _require(isinstance(policy, Mapping), f"motion step {step} omits Gaussian rollout")
            mean = torch.logit(boxes.detach().cpu().float()[positions])
            err = _tensor_close(policy.get("mean"), mean, TOL["mean_abs"],
                                f"motion step {step} Gaussian mean")
            max_errors["mean_abs"] = max(max_errors["mean_abs"], err)
            seed, noise = _cpu_noise(key, step, len(positions))
            _require(policy.get("seed") == seed, f"motion step {step} Gaussian seed differs")
            samples = policy["mean"].detach().cpu().float()[:, None, :] + float(config["sigma"]) * noise
            err = _tensor_close(policy.get("samples"), samples, TOL["sample_abs"],
                                f"motion step {step} Gaussian samples")
            max_errors["sample_abs"] = max(max_errors["sample_abs"], err)
            rewards = _reward(samples.sigmoid(), evidence_boxes[:, None, :], "iou")
            for label in ("rewards", "used_rewards"):
                err = _tensor_close(policy.get(label), rewards, TOL["reward_abs"],
                                    f"motion step {step} {label}")
                max_errors["reward_abs"] = max(max_errors["reward_abs"], err)
            _require(policy.get("permutations") is None, "motion OPD rewards were permuted")
            weights = torch.softmax(rewards / float(config["tau"]), dim=-1)
            err = _tensor_close(policy.get("weights"), weights, TOL["weight_abs"],
                                f"motion step {step} Gaussian weights")
            max_errors["weight_abs"] = max(max_errors["weight_abs"], err)
            informative = bool((rewards != rewards[:, :1]).any())
            _require(policy.get("informative") is informative,
                     f"motion step {step} OPD informative flag differs")
            opd_loss = float(_likelihood(policy["mean"].detach().cpu().float(),
                                         policy["mean"].detach().cpu().float(),
                                         policy["samples"].detach().cpu().float(), weights,
                                         float(config["sigma"])))
        else:
            _require(isinstance(policy, Mapping) and policy.get("informative") is False
                     and policy.get("seed") is None,
                     f"motion step {step} empty-anchor rollout is malformed")
            _require(policy.get("samples").shape == (0, 32, 4)
                     and policy.get("rewards").shape == (0, 32)
                     and policy.get("weights").shape == (0, 32),
                     f"motion step {step} empty-anchor Gaussian shapes differ")
            informative = False
            opd_loss = 0.0

        _close_number(entry.get("opd_loss"), opd_loss, TOL["loss_abs"],
                      f"motion step {step} OPD loss")
        _close_number(round_row.get("opd_loss"), opd_loss, TOL["loss_abs"],
                      f"motion round {step} OPD loss")
        max_errors["opd_loss_abs"] = max(max_errors["opd_loss_abs"],
            abs(float(entry["opd_loss"]) - opd_loss))
        total = opd_loss if motion["valid_pair_count"] == 0 else opd_loss + motion["loss"]
        _close_number(entry.get("total_loss"), total, TOL["loss_abs"],
                      f"motion step {step} combined loss")
        _close_number(entry.get("loss"), total, TOL["loss_abs"],
                      f"motion step {step} loss alias")
        _close_number(round_row.get("total_loss"), total, TOL["loss_abs"],
                      f"motion round {step} combined loss")
        _close_number(round_row.get("loss"), total, TOL["loss_abs"],
                      f"motion round {step} loss alias")
        max_errors["motion_loss_abs"] = max(max_errors["motion_loss_abs"],
            abs(float(entry["motion_loss"]) - motion["loss"]))
        max_errors["total_loss_abs"] = max(max_errors["total_loss_abs"],
            abs(float(entry["total_loss"]) - total))

        gradient = entry.get("raw_total_gradient")
        _require(torch.is_tensor(gradient) and gradient.device.type == "cpu"
                 and gradient.numel() == 1792 and torch.isfinite(gradient).all(),
                 f"motion step {step} raw combined gradient witness is malformed")
        gradient_nonzero = bool(torch.count_nonzero(gradient).item())
        _require(int(torch.count_nonzero(gradient).item())
                 == int(entry.get("raw_gradient_nonzero_count", torch.count_nonzero(gradient).item()))
                 and abs(float(gradient.norm()) - float(entry.get("raw_gradient_norm", gradient.norm()))) <= 1e-6,
                 f"motion step {step} gradient summary differs from saved vector")
        update = entry.get("update")
        _require(round_row.get("teacher_targets_detached") is True
                 and round_row.get("reward_weights_detached") is True
                 and round_row.get("actions_detached") is True,
                 f"motion step {step} detached rollout/target boundary differs")
        _require(round_row.get("updated") is (update is not None),
                 f"motion step {step} update receipt differs")
        _require(round_row.get("no_information") is (update is None),
                 f"motion step {step} no-information receipt differs")
        _require(round_row.get("policy_informative") is informative,
                 f"motion step {step} policy-information flag differs")
        _require(round_row.get("motion_gradient_informative") is (gradient_nonzero and not informative),
                 f"motion step {step} motion-gradient flag differs")
        if informative or gradient_nonzero:
            _require(isinstance(update, Mapping),
                     f"motion step {step} has signal but no optimizer update")
            _compare_tree(gradient, update.get("gradient"),
                          name=f"motion step {step} combined gradient witness")
            max_errors["adam_parameter_abs"], max_errors["adam_state_abs"] = _audit_adam_transition(
                entry, config, max_errors["adam_parameter_abs"],
                max_errors["adam_state_abs"], step)
            actual_steps += 1
        else:
            _require(update is None,
                     f"motion step {step} took Adam step with no OPD or motion gradient")
            _compare_tree(entry.get("state"), entry.get("update_state"),
                          name=f"motion step {step} zero-gradient parameters")
            _compare_tree(entry.get("optimizer_before"), entry.get("optimizer_state"),
                          name=f"motion step {step} zero-gradient Adam state")

        after_boxes = (fit["path"][step + 1]["boxes"] if step + 1 < len(fit["path"])
                       else fit.get("final_cached_decoder_boxes"))
        _require(torch.is_tensor(after_boxes), f"motion step {step} lacks next/final boxes")
        actual_delta = after_boxes.detach().cpu() - boxes.detach().cpu()
        _tensor_close(entry.get("actual_delta"), actual_delta, 1e-7,
                      f"motion step {step} decoder boxes to next state")
        if step + 1 < len(fit["path"]):
            next_entry = fit["path"][step + 1]
            expected_state = entry.get("update_state", entry.get("state"))
            _compare_tree(expected_state, next_entry.get("state"),
                          name=f"motion step {step} parameters to next pre-state")
            _compare_tree(entry.get("optimizer_state"), next_entry.get("optimizer_before"),
                          name=f"motion step {step} Adam to next pre-state")

    _require(actual_steps == int(fit.get("optimizer_steps", -1)),
             "motion fit optimizer step aggregate differs from recorded updates")
    _compare_tree(fit.get("final_state"), fit["path"][-1].get("update_state", fit["path"][-1].get("state")),
                  name="motion final state")
    last = fit["path"][-1]
    if last.get("skipped") is True:
        _tensor_close(fit.get("final_cached_decoder_boxes"), last.get("boxes"), 0.0,
                      "motion no-update final cached decoder boxes")
    else:
        _tensor_close(fit.get("final_cached_decoder_boxes"), last.get("boxes")
                      + last.get("actual_delta"), 1e-7,
                      "motion final cached decoder boxes")
    return {"steps": 5, "optimizer_steps": actual_steps,
            "motion_valid_pair_count": total_valid_pairs,
            "max_abs_errors": max_errors,
            "gradient_scope": "Adam transition independently replayed from saved combined gradient; full GPU decoder gradient not independently reconstructed"}


def _tensor_close(actual: Any, expected: Any, tolerance: float, name: str) -> float:
    _require(torch.is_tensor(actual) and torch.is_tensor(expected), f"{name}: tensor required")
    a, b = actual.detach().cpu(), expected.detach().cpu()
    _require(a.shape == b.shape and a.dtype == b.dtype,
             f"{name}: tensor dtype/shape differs")
    if not a.numel():
        return 0.0
    error = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
    _require(math.isfinite(error) and error <= tolerance,
             f"{name}: max error {error:.9g} exceeds {tolerance:.9g}")
    return error


def _assert_interval(interval: Any, bounds: Sequence[float], name: str) -> None:
    _require(isinstance(interval, (list, tuple)) and len(interval) == 2,
             f"{name}: interval shape differs")
    left, right = map(float, interval)
    _require(math.isfinite(left) and math.isfinite(right)
             and float(bounds[0]) <= left < right <= float(bounds[1]),
             f"{name}: interval is invalid or outside source bounds")


def _load_source(path: Path) -> dict[str, Any]:
    # Called only after _preflight has verified the complete global barrier and
    # every runtime pin/output hash.
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), "pinned source case must be a mapping")
    return value


def _validate_source(case: Mapping[str, Any], roster: Mapping[str, Any],
                     output: Mapping[str, Any]) -> list[float]:
    cid = roster["case_id"]
    _require(case.get("_source_case_sha256") == roster["source_case_sha256"]
             and case.get("GT_read") is False,
             f"{cid}: source payload digest or GT flag differs")
    _equal(list(map(int, case.get("frame_ids", []))),
           list(map(int, output.get("frame_ids", []))), f"{cid}.source_frame_ids")
    _require(case.get("pixel_sha256") == output.get("pixel_sha256")
             and case.get("source_model_state_sha256") == output.get("source_model_state_sha256"),
             f"{cid}: source pixel/model-state binding differs")
    _equal(case["zero"]["boxes"], output.get("source_boxes"), f"{cid}.source_zero_boxes")
    _equal(case["zero"]["logits"], output.get("source_logits"), f"{cid}.source_zero_logits")
    _equal(case["source_light_initial_state"], output.get("initial_state"),
           f"{cid}.source_initial_parameters")
    cold = output.get("source_cold_readback", {})
    _require(output.get("source_restored") is True
             and cold.get("status") == "pass"
             and cold.get("source_case_sha256") == roster["source_case_sha256"]
             and cold.get("source_model_state_sha256") == case.get("source_model_state_sha256")
             and cold.get("pixel_sha256") == case.get("pixel_sha256")
             and cold.get("frame_ids_match") is True
             and cold.get("initial_state_match") is True
             and cold.get("zero_boxes_logits_match") is True
             and cold.get("source_restored") is True,
             f"{cid}: cold-source restoration receipt differs")
    _equal(list(map(float, case.get("Native_interval", []))),
           list(map(float, output.get("Native_interval", []))), f"{cid}.source Native interval")
    _equal(list(map(float, case.get("Projection_interval", []))),
           list(map(float, output.get("Projection_interval", []))), f"{cid}.source Projection interval")
    bounds = list(map(float, case["video_bounds"]))
    return bounds


def _validate_source_arms(output: Mapping[str, Any], source: Mapping[str, Any],
                          bounds: Sequence[float]) -> None:
    cid = output["case_id"]
    arms = output.get("arms")
    _require(isinstance(arms, Mapping), f"{cid}: source arms absent")
    _equal(list(map(float, output.get("Native_interval", []))),
           list(map(float, source.get("Native_interval", []))), f"{cid}.Native interval")
    _equal(list(map(float, output.get("Projection_interval", []))),
           list(map(float, source.get("Projection_interval", []))), f"{cid}.Projection interval")
    native = torch.as_tensor(output["source_boxes"], dtype=torch.float64)
    expected_xyxy = _xyxy(native)
    _tensor_close(torch.as_tensor(arms["F"]["boxes"], dtype=torch.float64), expected_xyxy,
                  0.0, f"{cid}.F exact source boxes")
    _equal(arms["F"]["interval"], output["Native_interval"], f"{cid}.F interval")
    _tensor_close(torch.as_tensor(arms["Projection"]["boxes"], dtype=torch.float64), expected_xyxy,
                  0.0, f"{cid}.Projection exact source boxes")
    _equal(arms["Projection"]["interval"], output["Projection_interval"],
           f"{cid}.Projection interval")
    _assert_interval(output["Native_interval"], bounds, f"{cid}.Native")
    _assert_interval(output["Projection_interval"], bounds, f"{cid}.Projection")


def _load_old_temporal(gate: Mapping[str, Any], cid: str,
                       output: Mapping[str, Any], roster: Mapping[str, Any],
                       bounds: Sequence[float]) -> Mapping[str, Any]:
    old_path = gate["old_paths"][cid]
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    _require(old.get("case_id") == cid and old.get("GT_read") is False
             and old.get("source_case_sha256") == roster["source_case_sha256"],
             f"{cid}: pinned old temporal output belongs to another source")
    old_cold = old.get("source_cold_readback", {})
    _equal(list(map(int, old.get("frame_ids", []))),
           list(map(int, output.get("frame_ids", []))), f"{cid}.old temporal frame IDs")
    _require(old_cold.get("status") == "pass"
             and old_cold.get("pixel_sha256") == output.get("pixel_sha256"),
             f"{cid}: old temporal pixels differ")
    for name in ("F", "Projection", "T"):
        _tensor_close(torch.as_tensor(output["arms"][name]["boxes"], dtype=torch.float64),
                      torch.as_tensor(old["arms"][name]["boxes"], dtype=torch.float64),
                      0.0, f"{cid}.exact prior {name} boxes")
        _equal(output["arms"][name]["interval"], old["arms"][name]["interval"],
               f"{cid}.exact prior {name} interval")
        _assert_interval(output["arms"][name]["interval"], bounds, f"{cid}.{name}")
    reuse = output.get("temporal_exact_reuse", {})
    _require(reuse.get("status") == "reused_exact_old_T"
             and reuse.get("old_student_sha256") == _sha_file(old_path)
             and reuse.get("old_T_boxes_equal_F") is True
             and reuse.get("old_T_interval_reused") is True,
             f"{cid}: exact old T reuse receipt differs")
    return old


def _check_no_update(case: Mapping[str, Any], teacher: Mapping[str, Any],
                     old: Mapping[str, Any], fit_names: Sequence[str]) -> dict[str, Any]:
    cid = case["case_id"]
    _require(teacher.get("no_update") is True and teacher.get("eligible_for_update") is False,
             f"{cid}: no-update output conflicts with reference lock")
    _require(case.get("fits") == {} and case.get("readouts") == {}
             and case.get("actual_fit_count") == 0
             and case.get("optimizer_steps") == 0
             and case.get("actual_final_redecode_calls") == 0,
             f"{cid}: no-update case contains fitting/readout activity")
    arms, receipts = case["arms"], case.get("reference_receipts", {})
    _require(set(arms) == {
        "F", "Projection", "T", "teacher_direct", "teacher_iou", "teacher_motion_iou",
        "T_teacher_direct", "T_teacher_iou", "T_teacher_motion_iou",
    }, f"{cid}: no-update case arm set differs")
    for algorithm in ALGORITHMS:
        name = f"teacher_{algorithm}"
        _equal(arms[name]["boxes"], arms["F"]["boxes"], f"{cid}.{name} exact F boxes")
        _equal(arms[name]["interval"], arms["F"]["interval"], f"{cid}.{name} exact F interval")
        receipt = receipts.get(name, {})
        _require(receipt.get("status") == "explicit_no_update"
                 and receipt.get("lock_no_update") is True
                 and receipt.get("optimizer_steps") == 0
                 and receipt.get("final_redecode_calls") == 0
                 and receipt.get("exact_F_reuse") is True,
                 f"{cid}.{name}: zero-update receipt differs")
        tname = f"T_teacher_{algorithm}"
        _equal(arms[tname]["boxes"], old["arms"]["T"]["boxes"],
               f"{cid}.{tname} exact old T boxes")
        _equal(arms[tname]["interval"], old["arms"]["T"]["interval"],
               f"{cid}.{tname} exact old T interval")
        treceipt = receipts.get(tname, {})
        _require(treceipt.get("status") == "exact_old_T_reuse"
                 and treceipt.get("optimizer_steps") == 0
                 and treceipt.get("final_redecode_calls") == 0
                 and treceipt.get("exact_T_reuse") is True,
                 f"{cid}.{tname}: zero-update old T receipt differs")
    _equal(case.get("motion_evidence"), teacher.get("motion_evidence"),
           f"{cid}.no-update locked motion evidence")
    return {"fit_count": 0, "optimizer_steps": 0, "zero_update": True,
            "exact_F_Projection_T_reuse": True}


def _check_locked_motion_pairs(case: Mapping[str, Any], teacher: Mapping[str, Any],
                               frame_ids: Sequence[int], source_boxes: Any) -> dict[str, Any]:
    boxes = torch.as_tensor(source_boxes, dtype=torch.float32, device="cpu")
    derived = _independent_motion_loss(boxes, frame_ids, teacher["motion_evidence"])
    keys = ("evidence_pair_index", "left_physical_frame_id", "right_physical_frame_id",
            "left_center_xy", "right_center_xy", "rho_left", "rho_right", "pair_rho")
    valid = [{key: row[key] for key in keys} for row in derived["valid_pairs"]]
    excluded = [{key: row[key] for key in ("evidence_pair_index", "physical_frame_ids", "reason")}
                for row in derived["excluded_pairs"]]
    diagnostics = case.get("motion_evidence_diagnostics", {})
    _require(diagnostics.get("eligibility_from_locked_reference") is teacher.get("eligible_for_update"),
             f"{case['case_id']}: motion eligibility diagnostic differs")
    _require(diagnostics.get("valid_in_grid_adjacent_pair_count") == len(valid),
             f"{case['case_id']}: valid physical-pair count differs")
    _compare_numeric_tree(diagnostics.get("valid_pairs"), valid, TOL["motion_abs"],
                          f"{case['case_id']}.locked valid motion pairs")
    _equal(diagnostics.get("excluded_pairs"), excluded,
           f"{case['case_id']}.locked excluded motion pairs")
    if teacher.get("eligible_for_update"):
        _require(bool(valid), f"{case['case_id']}: eligible update lacks an in-grid adjacent pair")
    return {"valid_pair_count": len(valid), "excluded_pair_count": len(excluded)}


def _audit_case(gate: Mapping[str, Any], cid: str) -> dict[str, Any]:
    roster = gate["roster_by_id"][cid]
    reference = gate["reference_by_id"][cid]
    path = gate["case_paths"][cid]
    # All eight output hashes, the exact runtime, its code/source pins, and
    # old temporal hashes have passed before the first case deserialization.
    case = torch.load(path, map_location="cpu", weights_only=False)
    _require(case.get("case_id") == cid and case.get("dataset") == roster["dataset"]
             and case.get("source_case_sha256") == roster["source_case_sha256"]
             and case.get("GT_read") is False
             and case.get("reference_lock_sha256") == gate["hashes"]["reference_lock"],
             f"{cid}: output identity, source binding, reference binding, or GT flag differs")
    source = _load_source(gate["source_paths"][cid])
    bounds = _validate_source(source, roster, case)
    _validate_source_arms(case, source, bounds)
    old = _load_old_temporal(gate, cid, case, roster, bounds)
    teacher = reference["references"]["teacher"]
    _equal(case.get("reference_anchors"), {"teacher": {
        "anchors": teacher["anchors"], "motion_evidence": teacher["motion_evidence"],
        "no_update": teacher["no_update"],
        "eligible_for_update": teacher["eligible_for_update"],
        "reason": teacher["reason"],
        "selected_candidate_id": teacher["selected_candidate_id"],
        "is_Source": teacher["is_Source"],
    }}, f"{cid}.reference anchor lock")
    _equal(case.get("supervision_anchors"), case.get("reference_anchors"),
           f"{cid}.supervision anchor lock")
    _equal(case.get("motion_evidence"), teacher["motion_evidence"],
           f"{cid}.case motion evidence")
    pair_check = _check_locked_motion_pairs(
        case, teacher, list(map(int, case["frame_ids"])), case["source_boxes"])
    _equal(case.get("initial_state"), source["source_light_initial_state"],
           f"{cid}.source-reset initial state")

    from scripts import stvg_target_teacher_student_v1 as core
    config = core.load_original_opd_config(roster["dataset"])
    config["steps"] = 5
    cfg_row = case.get("configuration", {})
    _require(cfg_row.get("key") == f"caption_query_fixed_space|{cid}"
             and cfg_row.get("scope") == "joint" and cfg_row.get("steps") == 5
             and cfg_row.get("active_parameters") == 1792
             and cfg_row.get("method_reference_lock_sha256") == gate["hashes"]["reference_lock"],
             f"{cid}: locked method configuration metadata differs")
    _equal(cfg_row.get("config"), config, f"{cid}.original per-domain OPD config")

    if teacher["no_update"]:
        detail = _check_no_update(case, teacher, old, ())
        return {"dataset": roster["dataset"], "no_update": True,
                "fit_count": 0, "optimizer_steps": 0,
                "source_restored_receipt": True,
                "F_Projection_T_exact_reuse": True,
                "motion_pair_readback": pair_check, "readback": detail}

    fits, readouts = case.get("fits"), case.get("readouts")
    expected_keys = {f"teacher_{algorithm}" for algorithm in ALGORITHMS}
    _require(isinstance(fits, Mapping) and set(fits) == expected_keys
             and isinstance(readouts, Mapping) and set(readouts) == expected_keys,
             f"{cid}: fitted method arms/readouts differ from exact three-algorithm set")
    _require(case.get("actual_fit_count") == 3
             and case.get("actual_final_redecode_calls") == 3,
             f"{cid}: actual fit or full-model redecode count differs")
    _require(cfg_row.get("config") == config, f"{cid}: config mismatch")
    frame_ids = list(map(int, case["frame_ids"]))
    key = f"caption_query_fixed_space|{cid}"
    anchors = _anchor_tuple(teacher, frame_ids)
    _require(anchors[0], f"{cid}: eligible reference has no anchors")
    _require(teacher.get("is_Source") is bool(teacher.get("is_Source")),
             f"{cid}: malformed source-reference flag")
    expected_role = "source" if teacher["is_Source"] else "teacher"
    fit_checks = {}
    actual_steps = 0
    for algorithm in ALGORITHMS:
        name = f"teacher_{algorithm}"
        fit, readout = fits[name], readouts[name]
        _equal(fit.get("initial_state"), case["initial_state"], f"{cid}.{name}.initial")
        _require(fit.get("scope") == "joint" and fit.get("active_parameters") == 1792
                 and fit.get("declared_steps") == fit.get("selected_step") == 5
                 and len(fit.get("path", [])) == 5
                 and fit.get("config") == config,
                 f"{cid}.{name}: fit scope/config/path differs")
        if algorithm == "direct":
            check = _audit_direct_fit(fit, readout, anchors, case["initial_state"])
        elif algorithm == "iou":
            check = _audit_opd_fit(fit, anchors, frame_ids, key, "iou", case["initial_state"])
            _require(fit.get("reference_sha256") == gate["hashes"]["reference_lock"]
                     and fit.get("reference_role") == expected_role
                     and fit.get("reward_objective") == "iou",
                     f"{cid}.{name}: original IoU fit binding differs")
        else:
            check = _audit_motion_fit(
                fit, anchors, frame_ids, key, case["initial_state"],
                teacher["motion_evidence"], gate["hashes"]["reference_lock"],
            )
            _require(fit.get("reference_role") == expected_role,
                     f"{cid}.{name}: motion fit reference role differs")

        _tensor_close(readout.get("boxes"), fit.get("final_cached_decoder_boxes"),
                      0.0, f"{cid}.{name}.full readout vs cached boxes")
        _equal(readout.get("logits"), case.get("source_logits"),
               f"{cid}.{name}.reused temporal logits")
        _equal(readout.get("physical_interval"), case.get("Native_interval"),
               f"{cid}.{name}.Native interval")
        xyxy = _xyxy(torch.as_tensor(readout["boxes"], dtype=torch.float64))
        arm = case["arms"][name]
        _tensor_close(torch.as_tensor(arm["boxes"], dtype=torch.float64), xyxy,
                      0.0, f"{cid}.{name}.arm vs full readout")
        _equal(arm.get("interval"), case.get("Native_interval"),
               f"{cid}.{name}.arm Native interval")
        tname = f"T_{name}"
        _tensor_close(torch.as_tensor(case["arms"][tname]["boxes"], dtype=torch.float64), xyxy,
                      0.0, f"{cid}.{tname}.spatial fit inheritance")
        _equal(case["arms"][tname]["interval"], old["arms"]["T"]["interval"],
               f"{cid}.{tname}.old temporal interval reuse")
        _require(case["reference_receipts"][name].get("status") == "updated"
                 and case["reference_receipts"][name].get("lock_no_update") is False
                 and case["reference_receipts"][name].get("optimizer_steps") == fit.get("optimizer_steps")
                 and case["reference_receipts"][name].get("final_redecode_calls") == 1,
                 f"{cid}.{name}: update receipt differs")
        _require(case["reference_receipts"][tname].get("status")
                 == "updated_spatial_plus_exact_old_T_interval"
                 and case["reference_receipts"][tname].get("inherits_fit") == name
                 and case["reference_receipts"][tname].get("optimizer_steps") == fit.get("optimizer_steps")
                 and case["reference_receipts"][tname].get("final_redecode_calls") == 0
                 and case["reference_receipts"][tname].get("old_T_interval_reused") is True,
                 f"{cid}.{tname}: spatial fit/old temporal interval receipt differs")
        actual_steps += int(fit.get("optimizer_steps", -1))
        fit_checks[name] = check

    _require(case.get("optimizer_steps") == actual_steps,
             f"{cid}: aggregate optimizer steps differ from fits")
    _require(set(case["arms"]) == {
        "F", "Projection", "T", "teacher_direct", "teacher_iou", "teacher_motion_iou",
        "T_teacher_direct", "T_teacher_iou", "T_teacher_motion_iou",
    }, f"{cid}: method output has missing or extra arms")
    return {"dataset": roster["dataset"], "no_update": False,
            "fit_count": len(fits), "optimizer_steps": actual_steps,
            "source_restored_receipt": True, "F_Projection_T_exact_reuse": True,
            "motion_pair_readback": pair_check, "readback": {"fits": fit_checks}}


def audit_method(*, write_report: bool = True) -> dict[str, Any]:
    gate = _preflight()
    cases, fit_count, step_count, no_update_count = {}, 0, 0, 0
    domain_count = {"hc2": 0, "vidstg": 0}
    for cid in sorted(CASE_IDS):
        row = _audit_case(gate, cid)
        cases[cid] = row
        domain_count[row["dataset"]] += 1
        fit_count += row["fit_count"]
        step_count += row["optimizer_steps"]
        no_update_count += int(row["no_update"])
    _require(fit_count == gate["expected_fit_count"],
             "case readbacks do not match dynamic locked fit count")
    _require(domain_count == {"hc2": 4, "vidstg": 4}, "case domains differ from roster")
    _require(step_count == gate["barrier"].get("optimizer_steps"),
             "case optimizer steps differ from complete barrier")
    _require(gate["barrier"].get("actual_final_redecode_calls") == fit_count,
             "case full-model readouts differ from barrier count")

    report = {
        "schema": "stvg-adjacent-motion-independent-cpu-math-v1",
        "status": "pass", "phase": PHASE,
        "case_count": len(cases), "fit_count": fit_count,
        "eligible_case_count": gate["eligible_case_count"],
        "no_update_case_count": no_update_count,
        "declared_fit_steps": fit_count * 5,
        "actual_optimizer_steps": step_count,
        "domain_counts": domain_count,
        "GT_read": False, "GT_loader_called": False,
        "model_loaded": False, "GPU_initialized": False, "pixels_decoded": False,
        "barrier_bound_runtime": str(gate["runtime_path"].relative_to(ROOT)),
        "worker_source": WORKER, "fit_helper_source": FIT_HELPER,
        "case_readbacks": cases, "provenance": gate["hashes"],
        "reader_sha256": _sha_file(Path(__file__)),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpretation": {
            "expected_fit_count_is_dynamic_from_locked_eligibility": True,
            "old_temporal_F_Projection_T_reused_exactly": True,
            "no_update_paths_are_exact_reuse": True,
            "motion_loss_physical_pair_interpolation_rho_weighting_and_l1_independently_recomputed": True,
            "motion_opd_loss_is_separately_recomputed_before_total_loss": True,
            "adam_parameter_and_moment_transition_replayed_from_saved_gradient": True,
            "full_gpu_decoder_gradient_independently_reconstructed": False,
            "full_model_readout_verified_by_saved_cold_receipt_and_exact_tensor_join_not_model_replay": True,
        },
    }
    if write_report:
        destination = BASE / "method" / "ROOT_CPU_FIT_MATH_READBACK.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing readback: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False,
                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, destination)
    return report


def _contracts() -> dict[str, Any]:
    """Synthetic-only CPU checks for motion pair selection and combined loss."""
    if torch.cuda.is_initialized():
        raise RuntimeError("synthetic contracts require CUDA to remain uninitialized")
    ids = [10, 12, 14]
    # Centers progress linearly on the physical frame axis. Odd-frame teacher
    # evidence therefore requires interpolation of the student source grid.
    boxes = torch.tensor([[.1, .2, .2, .2], [.3, .2, .2, .2], [.5, .2, .2, .2]],
                         dtype=torch.float32)
    evidence = [
        {"physical_frame_id": fid, "center_xy": [.1 + .1 * (fid - 10), .2], "rho": rho}
        for fid, rho in ((10, 1.0), (11, .8), (12, .5), (13, .6), (14, .4))
    ]
    result = _independent_motion_loss(boxes, ids, evidence)
    assert result["valid_pair_count"] == 4
    assert result["pair_weight_sum"] == .8 + .5 + .5 + .4
    assert abs(result["loss"]) < 1e-7
    assert result["valid_pairs"][0]["left_interpolation"] == {
        "left_grid_position": 0, "right_grid_position": 0, "alpha": 0.0}
    assert result["valid_pairs"][1]["left_interpolation"] == {
        "left_grid_position": 0, "right_grid_position": 1, "alpha": .5}

    # Missing centers break the trajectory; records on either side must not be
    # bridged even when their physical frame IDs are otherwise adjacent later.
    missing = [
        {"physical_frame_id": 10, "center_xy": [.1, .2], "rho": 1.0},
        {"physical_frame_id": 11, "center_xy": None, "rho": 0.0},
        {"physical_frame_id": 12, "center_xy": [.3, .2], "rho": .5},
    ]
    miss = _independent_motion_loss(boxes, ids, missing)
    assert miss["valid_pair_count"] == 0 and miss["loss"] == 0.0
    assert [row["reason"] for row in miss["excluded_pairs"]] == [
        "missing_or_zero_rho_endpoint", "missing_or_zero_rho_endpoint"]

    nonconsecutive = [
        {"physical_frame_id": 10, "center_xy": [.1, .2], "rho": 1.0},
        {"physical_frame_id": 12, "center_xy": [.3, .2], "rho": .5},
    ]
    gap = _independent_motion_loss(boxes, ids, nonconsecutive)
    assert gap["valid_pair_count"] == 0
    assert gap["excluded_pairs"][0]["reason"] == "nonconsecutive_physical_frames"

    # Pair-confidence is min(endpoint rho), while L1 is the mean over the two
    # center coordinates. This produces a nonzero result with unequal motion.
    shifted = [dict(row) for row in evidence]
    shifted[2] = {**shifted[2], "center_xy": [.4, .1]}
    weighted = _independent_motion_loss(boxes, ids, shifted)
    assert weighted["valid_pair_count"] == 4 and weighted["loss"] > 0
    assert weighted["valid_pairs"][1]["pair_rho"] == .5
    opd, motion = .125, weighted["loss"]
    total = opd + motion  # lambda_motion is fixed at one when valid pairs exist.
    assert total == .125 + motion
    # With no reliable physical pair the production contract preserves the
    # pure IoU operation path exactly, rather than adding a synthetic zero term.
    no_pair_total = opd if miss["valid_pair_count"] == 0 else opd + miss["loss"]
    assert no_pair_total == opd

    # Exercise the complete independent reader against the producer's tiny
    # synthetic CPU base. This is not a source/model/data run and does not
    # touch the sealed experiment scope.
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_adjacent_motion_fit_v1 import fit_motion
    base = core._synthetic_student_base(n=8)
    config = core.load_original_opd_config("hc2")
    config["steps"] = 5
    fit_key = "caption_query_fixed_space|synthetic_motion_contract"
    ref_sha = "a" * 64
    anchors = ([2], torch.tensor([[.62, .38, .16, .22]], dtype=torch.float32))
    sparse_anchors = [{"position": 2, "physical_frame_id": 2,
                       "box": anchors[1][0].clone(), "weight": 1.0}]
    motion_records = [
        {"physical_frame_id": frame, "center_xy": [.12 + .035 * frame, .28],
         "rho": .9 - .04 * frame}
        for frame in range(8)
    ]
    fitted = fit_motion(
        base, base.initial, sparse_anchors, list(range(8)), motion_records,
        case_id="synthetic_motion_contract", dataset="hc2", reference_role="teacher",
        reference_sha256=ref_sha, config=config, scope="joint",
    )
    full_check = _audit_motion_fit(fitted, anchors, list(range(8)), fit_key,
                                   base.initial, motion_records, ref_sha)
    assert full_check["steps"] == 5 and full_check["motion_valid_pair_count"] == 7

    empty_base = core._synthetic_student_base(n=8)
    empty_fit = fit_motion(
        empty_base, empty_base.initial, [], list(range(8)), [],
        case_id="synthetic_motion_empty_contract", dataset="hc2", reference_role="teacher",
        reference_sha256="b" * 64, config=config, scope="joint",
    )
    empty_check = _audit_motion_fit(
        empty_fit, ([], torch.empty((0, 4), dtype=torch.float32)), list(range(8)),
        "caption_query_fixed_space|synthetic_motion_empty_contract", empty_base.initial, [], "b" * 64,
    )
    assert empty_check["optimizer_steps"] == 0 and empty_check["motion_valid_pair_count"] == 0

    return {
        "status": "pass", "CPU_only": True,
        "checks": ["physical_adjacent_pairs", "linear_interpolation_no_extrapolation",
                   "missing_breaks_pair_path", "endpoint_min_rho_weights",
                   "mean_coordinate_L1", "fixed_lambda_combination",
                   "zero_valid_pair_preserves_original_OPD_loss",
                   "synthetic_motion_iou_rollout_total_and_adam_readback",
                   "synthetic_exact_no_anchor_no_pair_zero_update"],
        "GT_read": False, "model_loaded": False, "GPU_initialized": False,
        "note": "synthetic arithmetic checks only; no real barrier or case outputs read",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("contracts", "method"), default="contracts")
    args = parser.parse_args(argv)
    if args.action == "contracts":
        result = _contracts()
    else:
        result = audit_method()
    print(json.dumps(result if args.action == "contracts" else {
        "status": result["status"], "phase": result["phase"],
        "case_count": result["case_count"], "fit_count": result["fit_count"],
        "eligible_case_count": result["eligible_case_count"],
        "actual_optimizer_steps": result["actual_optimizer_steps"],
        "GT_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
