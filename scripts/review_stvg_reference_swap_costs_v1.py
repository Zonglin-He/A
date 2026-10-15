#!/usr/bin/env python3
"""CPU-only cost/readout aggregation for the sealed E1 reference-swap run.

This reader binds the final sealed case and teacher barriers, the two root
readbacks, per-case receipts, and the recorded launch/exit metadata.  It does
not run a model, open annotation/GT files, or score predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"


class AuditError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"invalid JSON metadata: {path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise AuditError(f"{label} is not finite/nonnegative")
    return result


def _root_file(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT.resolve()) or not p.is_file():
        raise AuditError(f"missing or out-of-root bound file: {rel}")
    return p


def _verify_runtime(name: str) -> tuple[dict[str, Any], str]:
    rel = f"artifacts/stvg_reference_recovery_v1/Reference_swap/{name}"
    path = _root_file(rel)
    doc = _json(path)
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise AuditError(f"{name} must be locked and GT-free")
    pins = doc.get("pins")
    protected = doc.get("protected", {})
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise AuditError(f"{name} pins/protected must be mappings")
    if set(pins).intersection(protected) and any(
            pins[k] != protected[k] for k in set(pins).intersection(protected)):
        raise AuditError(f"{name} has conflicting pin entries")
    for pinned_rel, expected in {**pins, **protected}.items():
        if not isinstance(pinned_rel, str) or not isinstance(expected, str):
            raise AuditError(f"{name} contains malformed hash pin")
        if _sha(_root_file(pinned_rel)) != expected:
            raise AuditError(f"{name} pin mismatch: {pinned_rel}")
    return doc, _sha(path)


def _barrier_files(barrier: dict[str, Any], expected_count: int, label: str) -> list[dict[str, Any]]:
    files = barrier.get("files")
    if not isinstance(files, list) or len(files) != expected_count:
        raise AuditError(f"{label} has wrong file count")
    return files


def _validate_launch_exit(stage: str, suffix: str, *, expected_exit: int) -> dict[str, Any]:
    run_rel = f"artifacts/stvg_reference_recovery_v1/Reference_swap/execution_{stage}{suffix}"
    run_dir = _root_file(run_rel + "/LAUNCH.json").parent
    launch_path = run_dir / "LAUNCH.json"
    exit_path = run_dir / "WORKER_EXIT.json"
    limits_path = run_dir / "ACTUAL_LIMITS.json"
    launch, exit_doc, limits = _json(launch_path), _json(exit_path), _json(limits_path)
    if launch.get("status") != "running" or launch.get("GT_read") is not False:
        raise AuditError(f"invalid launch receipt: {run_rel}")
    if exit_doc.get("status") != "exited" or exit_doc.get("GT_read") is not False:
        raise AuditError(f"invalid worker exit receipt: {run_rel}")
    if int(exit_doc.get("exit_code", -999)) != expected_exit:
        raise AuditError(f"unexpected exit code: {run_rel}")
    if limits.get("status") != "pass" or limits.get("GT_read") is not False:
        raise AuditError(f"resource admission receipt missing: {run_rel}")
    start = _finite_number(launch.get("time"), f"{run_rel} launch.time")
    end = _finite_number(exit_doc.get("time"), f"{run_rel} exit.time")
    if end < start:
        raise AuditError(f"worker exit precedes launch: {run_rel}")
    return {
        "run_id": f"{stage}{suffix}",
        "launch_path": str((run_dir / "LAUNCH.json").relative_to(ROOT)),
        "launch_sha256": _sha(launch_path),
        "exit_path": str(exit_path.relative_to(ROOT)),
        "exit_sha256": _sha(exit_path),
        "actual_limits_path": str(limits_path.relative_to(ROOT)),
        "actual_limits_sha256": _sha(limits_path),
        "exit_code": expected_exit,
        "launch_to_exit_seconds": end - start,
        "model_load_seconds": None,
        "substage_seconds": None,
    }


def _partial_step_evidence() -> dict[str, Any]:
    # Both failed HC11 attempts retain the same five-step checkpoint chain in
    # two recovery snapshots.  The copies are byte-identical; they must not be
    # double-counted as separate checkpoint trees.
    roots = [
        BASE / "partial/HC11/WF_common",
        BASE / "recovery/student_float_boundary006/partial/HC11/WF_common",
        BASE / "recovery/student_readout_counter008/partial/HC11/WF_common",
    ]
    expected_names = [f"step{i}.pt" for i in range(1, 6)]
    snapshots: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            raise AuditError(f"missing HC11 partial checkpoint directory: {root.relative_to(ROOT)}")
        files = sorted(root.glob("step*.pt"))
        if [p.name for p in files] != expected_names:
            raise AuditError(f"HC11 partial checkpoint steps differ: {root.relative_to(ROOT)}")
        snapshots.append({
            "directory": str(root.relative_to(ROOT)),
            "step_count": len(files),
            "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": _sha(p)} for p in files],
        })
    signature_sets = [[(x["name"], x["bytes"], x["sha256"]) for x in s["files"]]
                      for s in snapshots]
    if not all(sig == signature_sets[0] for sig in signature_sets[1:]):
        raise AuditError("preserved HC11 partial checkpoint snapshots disagree")
    return {
        "case_alias": "HC11",
        "role": "WF_common",
        "partial_fit_attempts": 2,
        "optimizer_steps_per_attempt": 5,
        "optimizer_steps_across_attempts": 10,
        "full_model_readout_attempts": 2,
        "step_indexes": [1, 2, 3, 4, 5],
        "checkpoint_snapshot_directories": snapshots,
        "snapshot_copies_identical_and_counted_once_as_evidence": True,
        "attempt_multiplicity_source": "separate student005 and student006 launch/exit records plus root-reviewed HC11 partial fit/readout accounting",
    }


def build_cost_readback() -> dict[str, Any]:
    runtime, runtime_sha = _verify_runtime("RUNTIME.json")
    student_runtime, student_runtime_sha = _verify_runtime("STUDENT_RUNTIME.json")
    teacher_barrier_path = _root_file("artifacts/stvg_reference_recovery_v1/Reference_swap/TEACHER_GLOBAL_BARRIER.json")
    student_barrier_path = _root_file("artifacts/stvg_reference_recovery_v1/Reference_swap/STUDENT_GLOBAL_BARRIER.json")
    teacher_barrier, student_barrier = _json(teacher_barrier_path), _json(student_barrier_path)
    if teacher_barrier.get("status") != "complete" or teacher_barrier.get("GT_read") is not False:
        raise AuditError("teacher barrier is not complete and GT-free")
    if student_barrier.get("status") != "complete" or student_barrier.get("GT_read") is not False:
        raise AuditError("student barrier is not complete and GT-free")
    if student_barrier.get("runtime_sha256") != student_runtime_sha:
        raise AuditError("student barrier runtime binding mismatch")
    teacher_files = _barrier_files(teacher_barrier, 49, "teacher barrier")
    student_files = _barrier_files(student_barrier, 32, "student barrier")

    teacher_readback_path = _root_file("artifacts/stvg_reference_recovery_v1/Reference_swap/ROOT_TEACHER_READBACK.json")
    student_readback_path = _root_file("artifacts/stvg_reference_recovery_v1/Reference_swap/ROOT_READBACK.json")
    teacher_readback, student_readback = _json(teacher_readback_path), _json(student_readback_path)
    if teacher_readback.get("status") != "pass" or teacher_readback.get("GT_read") is not False:
        raise AuditError("teacher root readback is not pass/GT-free")
    if student_readback.get("status") != "pass" or student_readback.get("GT_read") is not False:
        raise AuditError("student root readback is not pass/GT-free")
    if teacher_readback.get("bindings", {}).get("teacher_global_barrier_sha256") != _sha(teacher_barrier_path):
        raise AuditError("teacher readback does not bind the current barrier")
    if student_readback.get("bindings", {}).get("student_barrier_sha256") != _sha(student_barrier_path):
        raise AuditError("student readback does not bind the current barrier")

    roster_path = _root_file("artifacts/stvg_reference_recovery_v1/Reference_swap/ROSTER_LOCK.json")
    roster = _json(roster_path)
    roster_rows = roster.get("records")
    if not isinstance(roster_rows, list) or len(roster_rows) != 32:
        raise AuditError("locked E1 roster must contain 32 records")
    aliases = [str(r.get("case_id", "")) for r in roster_rows]
    if len(set(aliases)) != 32 or any(not a for a in aliases):
        raise AuditError("roster aliases are missing or duplicated")

    # Validate all 49 teacher output files without opening their prose/payload.
    teacher_seen: set[str] = set()
    teacher_row_sha = []
    for row in teacher_files:
        rel = row.get("path")
        if not isinstance(rel, str):
            raise AuditError("teacher barrier row lacks path")
        if rel.startswith("teacher/"):
            p = (BASE / rel).resolve()
            if not p.is_relative_to(BASE.resolve()):
                raise AuditError("teacher output escapes its scope")
        else:
            p = _root_file(rel)
        if not p.is_file() or _sha(p) != row.get("sha256"):
            raise AuditError("teacher barrier output hash mismatch")
        if row.get("bytes") is not None and p.stat().st_size != int(row["bytes"]):
            raise AuditError("teacher barrier output byte count mismatch")
        job_id = str(row.get("job_id", ""))
        if not job_id or job_id in teacher_seen:
            raise AuditError("teacher barrier has an empty/duplicate job id")
        teacher_seen.add(job_id)
        teacher_row_sha.append(str(row["sha256"]))
    if len(teacher_seen) != 49:
        raise AuditError("teacher output job ids are incomplete")

    # Load only the locally produced, barrier-hashed student tensors to extract
    # scalar timing/counter fields.  No geometry, labels, query text or logits
    # are read or written to the receipt.
    try:
        import torch
    except Exception as exc:
        raise AuditError("PyTorch is required for locked student cost fields") from exc
    rows_by_alias: dict[str, dict[str, Any]] = {}
    for row in student_files:
        rel = row.get("path")
        if not isinstance(rel, str) or not rel.endswith(".pt"):
            raise AuditError("student barrier row has invalid artifact path")
        p = _root_file(rel)
        if p.parent != (BASE / "student").resolve():
            raise AuditError("student artifact escaped its fixed directory")
        case_id = p.stem
        if case_id in rows_by_alias or case_id not in aliases:
            raise AuditError("student barrier case is duplicate or absent from roster")
        if _sha(p) != row.get("sha256") or p.stat().st_size != int(row.get("bytes", -1)):
            raise AuditError("student barrier artifact hash/bytes mismatch")
        receipt_path = p.with_suffix(".receipt.json")
        receipt = _json(receipt_path)
        if (receipt.get("status") != "complete" or receipt.get("GT_read") is not False
                or receipt.get("case_id") != case_id or receipt.get("sha256") != row.get("sha256")
                or int(receipt.get("bytes", -1)) != p.stat().st_size):
            raise AuditError(f"student per-case receipt does not bind {case_id}")
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("case_id") != case_id or payload.get("GT_read") is not False:
            raise AuditError(f"student payload identity/GT guard mismatch for {case_id}")
        if payload.get("runtime_sha256") != student_runtime_sha:
            raise AuditError(f"student payload runtime mismatch for {case_id}")
        if payload.get("reference_lock_sha256") != student_barrier.get("reference_lock_sha256"):
            raise AuditError(f"student payload reference lock mismatch for {case_id}")
        receipt_counters = {
            "actual_fit_count": int(receipt.get("actual_fit_count", -1)),
            "optimizer_steps": int(receipt.get("optimizer_steps", -1)),
            "readout_call_count": int(receipt.get("readout_call_count", -1)),
        }
        payload_counters = {
            "actual_fit_count": int(payload.get("actual_fit_count", -2)),
            "optimizer_steps": int(payload.get("optimizer_steps", -2)),
            "readout_call_count": int(payload.get("readout_call_count", -2)),
        }
        if receipt_counters != payload_counters:
            raise AuditError(f"student receipt counters differ from payload for {case_id}")
        if payload_counters["actual_fit_count"] not in (0, 2):
            raise AuditError(f"unexpected fit-record count for {case_id}")
        reused = isinstance(receipt.get("reused_prediction_sha256"), str)
        source_capture_seconds = _finite_number(
            payload.get("source_capture_seconds"), f"{case_id}.source_capture_seconds")
        fits = payload.get("fits")
        if not isinstance(fits, dict) or set(fits) != {"WF_common", "DINO_common"}:
            raise AuditError(f"student fit cost schema differs for {case_id}")
        fit_wall = 0.0
        fit_wall_count = 0
        if not reused:
            for fit in fits.values():
                if fit is None:
                    continue
                if not isinstance(fit, dict):
                    raise AuditError(f"malformed fit cost entry for {case_id}")
                if "wall_seconds" in fit:
                    fit_wall += _finite_number(fit["wall_seconds"], f"{case_id}.fit.wall_seconds")
                    fit_wall_count += 1
        rows_by_alias[case_id] = {
            "case_alias": case_id,
            "dataset": str(receipt.get("dataset", "")),
            "student_artifact_sha256": str(row["sha256"]),
            "student_artifact_bytes": p.stat().st_size,
            "receipt_sha256": _sha(receipt_path),
            "complete_case_prediction_reused": reused,
            "fit_record_count": payload_counters["actual_fit_count"],
            "optimizer_steps_in_final_record": payload_counters["optimizer_steps"],
            "full_readout_record_count": payload_counters["readout_call_count"],
            "serialized_source_capture_seconds": source_capture_seconds,
            "new_fit_wall_seconds_if_measured": fit_wall if fit_wall_count else None,
            "new_fit_wall_timed_role_count": fit_wall_count,
        }
    if set(rows_by_alias) != set(aliases):
        raise AuditError("student barrier does not cover the complete locked roster")
    rows = [rows_by_alias[cid] for cid in aliases]

    fit_records = sum(r["fit_record_count"] for r in rows)
    optimizer_steps = sum(r["optimizer_steps_in_final_record"] for r in rows)
    readout_records = sum(r["full_readout_record_count"] for r in rows)
    reused_cases = [r for r in rows if r["complete_case_prediction_reused"]]
    reused_fit_records = sum(r["fit_record_count"] for r in reused_cases)
    reused_readout_records = sum(r["full_readout_record_count"] for r in reused_cases)
    reused_optimizer_steps = sum(r["optimizer_steps_in_final_record"] for r in reused_cases)
    new_fit_executions = fit_records - reused_fit_records
    new_readout_executions = readout_records - reused_readout_records
    if (fit_records != int(student_barrier.get("actual_fit_count", -1))
            or optimizer_steps != int(student_barrier.get("optimizer_steps", -1))
            or readout_records != int(student_barrier.get("full_model_readout_calls", -1))):
        raise AuditError("student per-case counters do not sum to final barrier")
    if (fit_records, len(reused_cases), reused_fit_records, new_fit_executions,
            readout_records, reused_readout_records, new_readout_executions) != (38, 10, 16, 22, 38, 16, 22):
        raise AuditError("student final/reused/new fit accounting differs from reviewed run")

    # Runs 005/006 each performed one full HC11 WF fit and one full-model
    # readout before a later schema/dispatch failure.  Check the preserved
    # five-round serialized checkpoint prefix; recovery copies are duplicates.
    partial = _partial_step_evidence()
    runs: list[dict[str, Any]] = []
    expected_exits = {
        ("teacher", "001"): 1, ("teacher", "002"): 1, ("teacher", "003"): 0,
        ("student", "004"): 1, ("student", "005"): 1, ("student", "006"): 1,
        ("student", "007"): 0,
    }
    for (stage, suffix), exit_code in expected_exits.items():
        runs.append(_validate_launch_exit(stage, suffix, expected_exit=exit_code))
    run_index = {row["run_id"]: row for row in runs}
    # Attempt-level work accounting is tied to the completed barrier/readbacks
    # and the preserved failure records.  Per-run load/forward sub-times were
    # not separately instrumented and remain null.
    run_work = {
        "teacher001": {"model_forwards": 0, "fit_attempts": 0, "full_readout_attempts": 0,
                       "reason_class": "pre-model failure; no teacher forward"},
        "teacher002": {"model_forwards": 0, "fit_attempts": 0, "full_readout_attempts": 0,
                       "reason_class": "subject-binding failure after model setup; no teacher forward"},
        "teacher003": {"model_forwards": 40, "fit_attempts": 0, "full_readout_attempts": 0,
                       "reason_class": "successful sealed teacher generation"},
        "student004": {"model_forwards": 0, "fit_attempts": 0, "full_readout_attempts": 0,
                       "reason_class": "pre-student-schema failure before model work"},
        "student005": {"model_forwards": None, "completed_fit_invocations": 16,
                       "partial_fit_attempts": 1, "fit_attempts": 17,
                       "full_readout_attempts": 17, "optimizer_steps": 74,
                       "completed_prefix_optimizer_steps": 69,
                       "partial_optimizer_steps": 5, "source_capture_calls": 11,
                       "reason_class": "ten-case completed prefix (16 fits/69 steps) plus one HC11 partial WF fit/readout"},
        "student006": {"model_forwards": None, "fit_attempts": 1, "full_readout_attempts": 1,
                       "optimizer_steps": 5, "source_capture_calls": 1,
                       "reason_class": "HC11 partial WF fit/readout; same deterministic checkpoint bytes preserved"},
        "student007": {"model_forwards": None, "completed_fit_invocations": new_fit_executions,
                       "partial_fit_attempts": 0, "fit_attempts": new_fit_executions,
                       "full_readout_attempts": new_readout_executions,
                       "optimizer_steps": optimizer_steps - reused_optimizer_steps,
                       "source_capture_calls": 22,
                       "reason_class": "successful remaining 22 case captures and new fits/readouts"},
    }
    for key, work in run_work.items():
        run_index[key]["work_accounting"] = work
    # 005/006 partial captures had no per-call seconds serialized.  Final case
    # artifacts retain one capture duration per case: first ten from 005, the
    # remaining 22 from 007.  HC11's two failed-attempt durations stay null.
    serialized_capture_sum = sum(r["serialized_source_capture_seconds"] for r in rows)
    reused_capture_sum = sum(r["serialized_source_capture_seconds"] for r in reused_cases)
    new_capture_sum = serialized_capture_sum - reused_capture_sum
    teacher_readback_counts = teacher_readback.get("counts", {})
    if (int(teacher_barrier.get("count", -1)) != 49
            or int(teacher_readback_counts.get("actual_model_forwards", -1)) != 40
            or int(teacher_readback_counts.get("accepted", -1)) != 38
            or int(teacher_readback_counts.get("rejected", -1)) != 2
            or int(teacher_readback_counts.get("unknown", -1)) != 9):
        raise AuditError("teacher attempt/output accounting does not match root readback")
    if (int(student_readback.get("count", -1)) != 32
            or int(student_readback.get("actual_fit_count", -1)) != 38):
        raise AuditError("student root readback does not match final barrier")

    teacher_worker_seconds = _finite_number(teacher_barrier.get("elapsed_seconds"), "teacher.elapsed_seconds")
    student_worker_seconds = _finite_number(student_barrier.get("wall_seconds"), "student.wall_seconds")
    teacher_run_wall = sum(r["launch_to_exit_seconds"] for r in runs if r["run_id"].startswith("teacher"))
    student_run_wall = sum(r["launch_to_exit_seconds"] for r in runs if r["run_id"].startswith("student"))
    failed_wall = sum(r["launch_to_exit_seconds"] for r in runs if r["exit_code"] != 0)
    passed_wall = sum(r["launch_to_exit_seconds"] for r in runs if r["exit_code"] == 0)

    inputs = {
        "teacher_barrier": str(teacher_barrier_path.relative_to(ROOT)),
        "teacher_barrier_sha256": _sha(teacher_barrier_path),
        "student_barrier": str(student_barrier_path.relative_to(ROOT)),
        "student_barrier_sha256": _sha(student_barrier_path),
        "teacher_runtime": "artifacts/stvg_reference_recovery_v1/Reference_swap/RUNTIME.json",
        "teacher_runtime_sha256": runtime_sha,
        "student_runtime": "artifacts/stvg_reference_recovery_v1/Reference_swap/STUDENT_RUNTIME.json",
        "student_runtime_sha256": student_runtime_sha,
        "teacher_root_readback_sha256": _sha(teacher_readback_path),
        "student_root_readback_sha256": _sha(student_readback_path),
        "roster_sha256": _sha(roster_path),
        "teacher_output_file_sha256_count": len(teacher_row_sha),
    }
    launch_exit_by_run = {r["run_id"]: r["launch_to_exit_seconds"] for r in runs}
    actual_fit_invocations = fit_records + partial["partial_fit_attempts"]
    actual_full_readouts = readout_records + partial["full_model_readout_attempts"]
    actual_optimizer_steps = optimizer_steps + partial["optimizer_steps_across_attempts"]
    return {
        "schema": "stvg-reference-swap-actual-execution-cost-v1",
        "status": "pass",
        "scope": "CPU-only aggregation of sealed E1 execution costs and existing launch/exit receipts; no model, GPU, GT, or scoring",
        "GT_read": False,
        "CPU_only": True,
        "model_called": False,
        "inputs": inputs,
        # Stable flat fields consumed by the report builder.
        "retained_fit_count": fit_records,
        "total_fit_invocations": actual_fit_invocations,
        "retained_optimizer_steps": optimizer_steps,
        "total_optimizer_steps": actual_optimizer_steps,
        "source_capture_calls": 34,
        "total_full_readout_calls": actual_full_readouts,
        "teacher_attempt_count": 49,
        "teacher_forward_count": 40,
        "worker_launch_to_exit_seconds": launch_exit_by_run,
        "worker_launch_to_exit_total_seconds": teacher_run_wall + student_run_wall,
        "counts": {
            "teacher_attempts": 49,
            "teacher_actual_forward_count": 40,
            "teacher_accepted": 38,
            "teacher_rejected": 2,
            "teacher_unknown_without_forward": 9,
            "student_final_case_count": 32,
            "student_final_fit_record_count": 38,
            "student_complete_case_reuses": len(reused_cases),
            "student_reused_fit_records": reused_fit_records,
            "student005_completed_prefix_fit_invocations": reused_fit_records,
            "student_reused_empty_cases": sum(r["fit_record_count"] == 0 for r in reused_cases),
            "student_new_fit_executions_007": new_fit_executions,
            "student_new_readout_executions_007": new_readout_executions,
            "student_partial_fit_attempts_005_006": partial["partial_fit_attempts"],
            "student_partial_readout_attempts_005_006": partial["full_model_readout_attempts"],
            "student_total_fit_records_plus_partial_attempts": fit_records + partial["partial_fit_attempts"],
            "student_total_readout_records_plus_partial_attempts": readout_records + partial["full_model_readout_attempts"],
            "source_capture_calls_total": 34,
            "source_capture_calls_by_run": {"student005": 11, "student006": 1, "student007": 22},
            "source_capture_seconds_serialized_for_final_case_payloads": 32,
            "failed_hc11_source_capture_seconds_unavailable": 2,
            "retained_optimizer_steps": optimizer_steps,
            "reused_optimizer_steps_already_in_retained_records": reused_optimizer_steps,
            "student005_completed_prefix_optimizer_steps": reused_optimizer_steps,
            "new_optimizer_steps_007": optimizer_steps - reused_optimizer_steps,
            "partial_optimizer_steps_005_006": partial["optimizer_steps_across_attempts"],
            "total_fit_history_optimizer_steps_including_reused_and_partial": optimizer_steps + partial["optimizer_steps_across_attempts"],
            "remaining_optimizer_steps_after_005_prefix": (optimizer_steps - reused_optimizer_steps) + partial["optimizer_steps_across_attempts"],
        },
        "timings_seconds": {
            "teacher_barrier_worker_elapsed": teacher_worker_seconds,
            "student_barrier_worker_elapsed": student_worker_seconds,
            "teacher_all_launch_to_exit_including_failed_attempts": teacher_run_wall,
            "student_all_launch_to_exit_including_failed_attempts": student_run_wall,
            "all_launch_to_exit_including_failed_attempts": teacher_run_wall + student_run_wall,
            "failed_launch_to_exit_total": failed_wall,
            "successful_launch_to_exit_total": passed_wall,
            "serialized_final_case_source_capture_seconds": serialized_capture_sum,
            "reused_prefix_source_capture_seconds": reused_capture_sum,
            "new_suffix_source_capture_seconds": new_capture_sum,
            "teacher_model_load_seconds": None,
            "student_model_load_seconds": None,
            "partial_hc11_fit_substage_seconds_005_006": None,
            "partial_hc11_source_capture_seconds_005_006": None,
            "readout_substage_seconds": None,
        },
        "partial_hc11_evidence": partial,
        "worker_runs": runs,
        "case_costs": rows,
        "interpretation": {
            "fit_accounting": "38 retained fit records comprise 16 fits executed in the completed student005 ten-case prefix and retained by exact case reuse in student007, plus 22 new fits executed in student007. The ten reused cases include eight with two fits and two empty cases. Two additional HC11 WF partial fit/readout attempts occurred in student005 and student006, making 40 actual E1 fit invocations and 40 full-model readout invocations across the complete execution chain.",
            "time_accounting": "Launch-to-exit is measured from launch/exit receipt timestamps and includes failures, loading, and all worker work. Per-run model-load and partial-attempt substage durations were not separately serialized and remain null. Worker-barrier elapsed fields are retained separately.",
            "optimizer_accounting": "The 164 retained steps include 69 from the student005 prefix replayed into the final barrier and 95 from student007. Adding five steps from each of the two HC11 partial attempts gives 174 actual E1 optimizer steps; the 005 prefix must not be excluded from this chain total.",
            "source_capture_accounting": "34 capture calls are counted across student005/006/007. The 32 final case payloads retain per-case capture seconds; two failed HC11 capture sub-times were not serialized and remain unavailable.",
            "checkpoint_accounting": "The two partial HC11 attempts each retained a five-round checkpoint chain. Recovery copies are byte-identical snapshots of that chain and are not counted as extra attempts.",
            "limits": "This is an execution-cost receipt, not a quality result, end-to-end cold-latency benchmark, or independent model replay.",
        },
    }


def write_receipt(output: Path, *, replace_existing: bool = False) -> dict[str, Any]:
    data = build_cost_readback()
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise AuditError("output must remain inside the repository")
    if output.exists() and not replace_existing:
        raise AuditError("refusing to overwrite an existing execution-cost receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tmp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)
    return {"path": str(output.relative_to(ROOT)), "sha256": _sha(output), "bytes": output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/stvg_reference_recovery_v1/Reference_swap/ROOT_ACTUAL_EXECUTION_COST.json")
    parser.add_argument("--replace", action="store_true", help="atomically replace an existing receipt after preserving it separately")
    args = parser.parse_args()
    result = write_receipt(ROOT / args.output, replace_existing=args.replace)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
