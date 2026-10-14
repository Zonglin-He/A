"""Post-seal CPU readback for the support-aware temporal/spatial comparison.

This module is inert on import.  ``main`` verifies every prediction, source,
runtime and readback-lock hash for the complete eight-case roster before it
calls the established GT reader.  Reports contain anonymous scalar metrics;
model parameters, boxes, intervals, query text and GT coordinates are never
serialized.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_support_aware_temporal_spatial_update_v1"
OLD = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
MODES = ("original_adam", "ridge", "trust", "unconstrained_ridge", "per_offset_trust")
SPATIAL_ARMS = ("S0_source", "S1_direct", "S2_old_opd", "S3_conservative_opd")
TRUST_TIME_MODE = "trust"  # Fixed by this reader before any label is loaded.
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")
TOL = 1e-9


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("pin path must be a nonempty ROOT-relative POSIX path")
    p = Path(value)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError("absolute or traversing path in evidence inventory")
    low = "/" + value.lower().strip("/") + "/"
    forbidden = ("/annotations/", "/annotation/", "/ground_truth/",
                 "/groundtruth/", "/gt_labels/", "/labels/", "/truths/")
    if any(part in low for part in forbidden):
        raise ValueError("GT/annotation path cannot be read during preflight")
    return value


def _verify_file(relpath: str, expected_sha: str, expected_bytes: int | None = None) -> Path:
    relpath = _safe_relpath(relpath)
    path = ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(f"pinned file is missing: {relpath}")
    actual = _sha(path)
    if actual != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {relpath}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"byte-count mismatch: {relpath}")
    return path


def _pin_map(document: Mapping[str, Any]) -> dict[str, str]:
    pins = document.get("pins")
    if isinstance(pins, dict):
        result = {}
        for path, digest in pins.items():
            path = _safe_relpath(path)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"malformed hash for pin {path}")
            result[path] = digest
        return result
    # The post-prediction custody receipt may store the same exact map as rows.
    rows = document.get("files")
    if isinstance(rows, list):
        result = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("malformed readback lock file row")
            path, digest = _safe_relpath(row.get("path")), row.get("sha256")
            if path in result or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("duplicate or malformed readback lock pin")
            result[path] = digest
        return result
    raise ValueError("locked custody document has no explicit pins/files map")


def _verify_runtime(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    runtime = _json(path)
    if runtime.get("status") not in ("locked", "locked_pre_GT", "locked_before_GT",
                                      "locked_pre_student_execution"):
        raise RuntimeError(f"{label} is not a locked runtime")
    if runtime.get("GT_read") is not False:
        raise RuntimeError(f"{label} is not marked pre-GT")
    pins = _pin_map(runtime)
    if not pins:
        raise RuntimeError(f"{label} has no source/code pins")
    for relpath, digest in pins.items():
        _verify_file(relpath, digest)
    return runtime, _sha(path)


def _inventory(barrier: Mapping[str, Any], expected_paths: set[str], *, label: str) -> dict[str, dict[str, Any]]:
    rows = barrier.get("files")
    if not isinstance(rows, list):
        raise ValueError(f"{label} has no sealed file inventory")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} inventory contains a malformed row")
        relpath = _safe_relpath(row.get("path"))
        if relpath in index:
            raise ValueError(f"{label} inventory repeats a path")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{label} inventory has a malformed digest")
        path = _verify_file(relpath, digest, row.get("bytes"))
        if path.suffix == ".pt" and relpath not in expected_paths:
            raise ValueError(f"{label} inventory includes an unexpected tensor payload")
        index[relpath] = row
    if set(index) != expected_paths:
        raise RuntimeError(f"{label} does not seal exactly the fixed eight-case file set")
    return index


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_root_readback_lock(required_paths: Sequence[str]) -> tuple[dict[str, Any], str, dict[str, str]]:
    path = BASE / "ROOT_READBACK_RUNTIME.json"
    lock = _json(path)
    if lock.get("status") != "locked_post_prediction_pre_new_scoring" or lock.get("GT_read") is not False:
        raise RuntimeError("ROOT_READBACK_RUNTIME is not the locked pre-scoring custody receipt")
    if lock.get("historical_cohort_GT_exposed") is not True:
        raise RuntimeError("readback lock must disclose the cohort's historical GT exposure")
    pins = _pin_map(lock)
    for required in required_paths:
        if required not in pins:
            raise RuntimeError(f"ROOT_READBACK_RUNTIME does not bind required source: {required}")
    for relpath, digest in pins.items():
        _verify_file(relpath, digest)
    self_path = "scripts/review_stvg_support_aware_update_v1.py"
    if self_path not in pins or pins[self_path] != _sha(Path(__file__)):
        raise RuntimeError("ROOT_READBACK_RUNTIME does not pin this frozen reader")
    return lock, _sha(path), pins


def _verify_global_inputs() -> dict[str, Any]:
    """Verify the full prediction/source custody graph before any GT access."""
    from scripts.stvg_temporal_caption_query_v1 import roster

    temporal_runtime_path = BASE / "TEMPORAL_RUNTIME.json"
    student_runtime_path = BASE / "STUDENT_RUNTIME.json"
    temporal_barrier_path = BASE / "TEMPORAL_GLOBAL_BARRIER.json"
    student_barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
    old_roster_path = OLD / "ROSTER_LOCK.json"
    old_runtime_path = OLD / "RUNTIME.json"
    old_student_runtime_path = OLD / "STUDENT_RUNTIME.json"
    old_teacher_barrier_path = OLD / "TEACHER_GLOBAL_BARRIER.json"
    old_student_barrier_path = OLD / "STUDENT_GLOBAL_BARRIER.json"
    job_lock_path = OLD / "CAPTION_JOB_LOCK.json"
    reference_lock_path = OLD / "SPATIAL_REFERENCE_LOCK.json"

    # The readback receipt is created after prediction sealing; the original
    # temporal barrier itself predates that receipt and did not record runtime_sha256.
    required_receipt_paths = [
        str(p.relative_to(ROOT)) for p in (
            temporal_runtime_path, temporal_barrier_path,
            student_runtime_path, student_barrier_path,
            old_roster_path, old_runtime_path, old_student_runtime_path,
            old_teacher_barrier_path, old_student_barrier_path,
            job_lock_path, reference_lock_path,
        )
    ]
    readback_lock, readback_lock_sha, readback_pins = _verify_root_readback_lock(required_receipt_paths)

    temporal_runtime, temporal_runtime_sha = _verify_runtime(
        temporal_runtime_path, label="TEMPORAL_RUNTIME")
    student_runtime, student_runtime_sha = _verify_runtime(
        student_runtime_path, label="STUDENT_RUNTIME")
    old_runtime, old_runtime_sha = _verify_runtime(old_runtime_path, label="original RUNTIME")
    old_student_runtime, old_student_runtime_sha = _verify_runtime(
        old_student_runtime_path, label="original STUDENT_RUNTIME")

    # Both actual runtime locks must be bound by the post-prediction receipt.
    for relpath, digest in ((str(temporal_runtime_path.relative_to(ROOT)), temporal_runtime_sha),
                            (str(student_runtime_path.relative_to(ROOT)), student_runtime_sha)):
        _require(readback_pins[relpath] == digest, f"readback receipt runtime pin differs: {relpath}")

    old_roster = _json(old_roster_path)
    jobs = _json(job_lock_path)
    references = _json(reference_lock_path)
    old_teacher_barrier = _json(old_teacher_barrier_path)
    old_student_barrier = _json(old_student_barrier_path)
    temporal_barrier = _json(temporal_barrier_path)
    student_barrier = _json(student_barrier_path)

    _require(old_roster.get("status") == "locked" and old_roster.get("count") == 8
             and old_roster.get("GT_read", False) is False, "original roster lock is not valid")
    _require(jobs.get("status") == "locked" and isinstance(jobs.get("records"), list),
             "original candidate job lock is not valid")
    _require(references.get("status") == "locked" and references.get("count") == 8
             and references.get("GT_read") is False, "spatial-reference lock is not valid")

    records = roster()
    _require(len(records) == 8 and len(old_roster["records"]) == 8,
             "fixed roster does not contain eight records")
    record_by_id = {row["case_id"]: row for row in records}
    locked_by_id = {row["case_id"]: row for row in old_roster["records"]}
    _require(len(record_by_id) == len(locked_by_id) == 8 and set(record_by_id) == set(locked_by_id),
             "roster case IDs are duplicated or differ from the source lock")
    for case_id, row in record_by_id.items():
        locked = locked_by_id[case_id]
        _require(row.get("source_case_sha256") == locked.get("source_case_sha256")
                 and row.get("dataset") == locked.get("dataset"),
                 "source roster metadata changed")

    for label, barrier in (("original teacher barrier", old_teacher_barrier),
                           ("original student barrier", old_student_barrier)):
        _require(barrier.get("status") == "complete" and barrier.get("GT_read") is False,
                 f"{label} is not complete and pre-GT")
    _require(old_teacher_barrier.get("runtime_sha256") == old_runtime_sha,
             "original teacher barrier does not bind its runtime")
    _require(old_student_barrier.get("student_runtime_sha256") == old_student_runtime_sha,
             "original student barrier does not bind its runtime")
    _require(old_student_barrier.get("teacher_barrier_sha256") == _sha(old_teacher_barrier_path),
             "original student barrier does not bind the original teacher seal")

    expected_old_student = {f"{OLD.relative_to(ROOT)}/student/{cid}.pt" for cid in record_by_id}
    old_student_files = _inventory(old_student_barrier, expected_old_student,
                                   label="original student barrier")
    expected_teacher_files = {
        _safe_relpath(row["path"]) for row in old_teacher_barrier.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    _require(len(expected_teacher_files) == len(old_teacher_barrier.get("files", [])),
             "original teacher barrier inventory is malformed")
    _inventory(old_teacher_barrier, expected_teacher_files, label="original teacher barrier")

    # The historical temporal barrier omitted its own runtime hash.  The later
    # readback receipt explicitly binds both files; disclose that boundary.
    _require(temporal_barrier.get("status") == "complete" and temporal_barrier.get("count") == 8
             and temporal_barrier.get("GT_read") is False and temporal_barrier.get("CPU_only") is True,
             "new temporal barrier is not a complete CPU-only pre-GT seal")
    _require(temporal_barrier.get("CPU_fit_modes") == 5
             and temporal_barrier.get("actual_new_teacher_calls") == 0
             and temporal_barrier.get("actual_new_model_forwards") == 0,
             "new temporal barrier reports an unexpected execution scope")
    temporal_paths = {f"{BASE.relative_to(ROOT)}/temporal/{cid}.pt" for cid in record_by_id}
    temporal_files = _inventory(temporal_barrier, temporal_paths, label="new temporal barrier")
    _require(student_barrier.get("schema") == "stvg-support-aware-spatial-barrier-v1"
             and student_barrier.get("status") == "complete" and student_barrier.get("count") == 8
             and student_barrier.get("GT_read") is False,
             "new spatial student barrier is not a complete pre-GT seal")
    _require(student_barrier.get("runtime_sha256") == student_runtime_sha,
             "new spatial student barrier does not bind its runtime")
    student_paths = {f"{BASE.relative_to(ROOT)}/student/{cid}.pt" for cid in record_by_id}
    student_files = _inventory(student_barrier, student_paths, label="new spatial student barrier")

    _require(student_barrier.get("old_roster_sha256") == _sha(old_roster_path)
             and student_barrier.get("old_teacher_barrier_sha256") == _sha(old_teacher_barrier_path)
             and student_barrier.get("old_student_barrier_sha256") == _sha(old_student_barrier_path)
             and student_barrier.get("spatial_reference_lock_sha256") == _sha(reference_lock_path),
             "new student seal is not bound to the exact source/teacher/reference locks")
    root_self_sha = readback_pins["scripts/review_stvg_support_aware_update_v1.py"]
    _require(root_self_sha == _sha(Path(__file__)), "readback reader SHA changed after lock")

    return {
        "records": records, "record_by_id": record_by_id,
        "old_roster": old_roster, "jobs": jobs, "references": references,
        "old_teacher_barrier": old_teacher_barrier,
        "old_student_barrier": old_student_barrier,
        "temporal_barrier": temporal_barrier, "student_barrier": student_barrier,
        "old_student_files": old_student_files,
        "temporal_files": temporal_files, "student_files": student_files,
        "runtimes": {"temporal": temporal_runtime_sha, "student": student_runtime_sha,
                      "original": old_runtime_sha, "original_student": old_student_runtime_sha},
        "barriers": {"temporal": _sha(temporal_barrier_path), "student": _sha(student_barrier_path),
                     "original_teacher": _sha(old_teacher_barrier_path),
                     "original_student": _sha(old_student_barrier_path)},
        "locks": {"root_readback": readback_lock_sha,
                  "roster": _sha(old_roster_path), "jobs": _sha(job_lock_path),
                  "reference": _sha(reference_lock_path)},
        "readback_lock": readback_lock,
        "temporal_runtime_barrier_binding": "post-prediction ROOT_READBACK_RUNTIME pins; original temporal barrier has no runtime_sha256 field",
    }


def _tree_equal(left: Any, right: Any, torch: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_tree_equal(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b, torch) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    return left == right


def _cpu_tree(value: Any, torch: Any, where: str) -> None:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise RuntimeError(f"{where} contains a non-CPU tensor")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _cpu_tree(item, torch, f"{where}.{key}")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _cpu_tree(item, torch, f"{where}[{i}]")


def _finite_tensor(t: Any, torch: Any, where: str) -> None:
    if not torch.is_tensor(t) or t.device.type != "cpu" or not bool(torch.isfinite(t).all()):
        raise RuntimeError(f"{where} is not a finite CPU tensor")


def _validate_temporal_case(case_id: str, case: Mapping[str, Any], temporal: Mapping[str, Any],
                            old_case: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    _require(temporal.get("case_id") == case_id and temporal.get("dataset") == case["dataset"]
             and temporal.get("GT_read") is False and temporal.get("new_teacher_calls") == 0
             and temporal.get("new_model_forwards") == 0,
             f"temporal output is not bound to the fixed case: {case_id}")
    _require(temporal.get("source_student_sha256") is not None,
             "temporal result omits the frozen source-student SHA")
    _require(temporal["source_student_sha256"] == old_case["_sealed_sha256"],
             "temporal result was fitted against a different source-student file")
    results = temporal.get("results")
    _require(isinstance(results, dict) and set(results) == set(MODES),
             "temporal result does not contain all five fixed methods")
    legacy = old_case.get("temporal_fit")
    _require(isinstance(legacy, dict) and len(legacy.get("path", [])) == 6,
             "original Adam source trace is incomplete")
    projection = list(case["Projection_interval"])
    mode_summaries = {}
    all_metrics_inputs = {}
    for mode in MODES:
        fit = results[mode]
        path = fit.get("path")
        _require(fit.get("mode") == mode and isinstance(path, list) and len(path) == 6,
                 f"temporal {mode} path does not have six fixed snapshots")
        _require(fit.get("GT_read") is False and fit.get("active_parameters") == 512
                 and fit.get("original_frozen_features") is True,
                 f"temporal {mode} has invalid scope/provenance")
        _require([int(row.get("step", -1)) for row in path] == list(range(6)),
                 f"temporal {mode} path step order is invalid")
        _finite_tensor(path[0].get("theta"), torch, f"{mode}.theta[0]")
        _require(tuple(path[0]["theta"].shape) == (256, 2)
                 and int(torch.count_nonzero(path[0]["theta"])) == 0,
                 f"temporal {mode} must start from exact zero 512-parameter readout")
        _require(list(path[0].get("physical_interval", [])) == projection,
                 f"temporal {mode} does not start from the frozen Projection interval")
        objectives = []
        legacy_losses = []
        previous_theta = path[0]["theta"].detach().clone()
        subspace_residuals = []
        for index, snap in enumerate(path):
            _finite_tensor(snap.get("theta"), torch, f"{mode}.theta[{index}]")
            if tuple(snap["theta"].shape) != (256, 2):
                raise RuntimeError(f"{mode} has an invalid 512-parameter snapshot")
            loss = float(snap.get("loss"))
            objective_loss = float(snap.get("objective_loss"))
            if not math.isfinite(loss) or not math.isfinite(objective_loss):
                raise RuntimeError(f"{mode} has a non-finite loss at step {index}")
            legacy_losses.append(loss)
            objectives.append(objective_loss)
            diagnostics = snap.get("diagnostics") or {}
            basis = diagnostics.get("evidence_subspace_basis")
            _finite_tensor(basis, torch, f"{mode}.subspace_basis[{index}]")
            if basis.ndim != 2 or basis.shape[0] != 512:
                raise RuntimeError(f"{mode} basis has an invalid shape")
            delta = (snap["theta"] - previous_theta).reshape(512)
            projected = basis @ (basis.T @ delta) if basis.shape[1] else torch.zeros_like(delta)
            residual = float(torch.linalg.vector_norm(delta - projected))
            subspace_residuals.append(residual)
            # The unmodified legacy Adam baseline is an optimizer control, not
            # a ridge-subspace method.  All closed-form/ridge variants must
            # remain in the recorded centered evidence subspace.
            if mode != "original_adam" and residual > 2e-8:
                raise RuntimeError(f"{mode} update leaves the saved centered evidence subspace")
            step_info = diagnostics.get("step") or {}
            outside = step_info.get("outside_subspace_norm")
            if mode != "original_adam" and outside is not None and float(outside) > 2e-8:
                raise RuntimeError(f"{mode} step diagnostic exceeds the subspace tolerance")
            previous_theta = snap["theta"].detach().clone()
            # Trust constraints apply to all full-span distributions, at every snapshot.
            if mode in ("trust", "per_offset_trust"):
                reverse = diagnostics.get("full_reverse_kl_by_offset")
                if not isinstance(reverse, (list, tuple)) or len(reverse) != 2:
                    raise RuntimeError(f"{mode} omits both full-span reverse-KL values")
                if any(not math.isfinite(float(v)) or float(v) > 0.01 + 1e-12 for v in reverse):
                    raise RuntimeError(f"{mode} violates the full-span reverse-KL cap")
        if mode in ("trust", "per_offset_trust"):
            if any(b > a + 1e-12 for a, b in zip(objectives, objectives[1:])):
                raise RuntimeError(f"{mode} optimized objective_loss is not monotonic")
        if mode == "original_adam":
            _require(len(legacy["path"]) == len(path), "original Adam trace length changed")
            for old_snap, new_snap in zip(legacy["path"], path):
                if not torch.equal(old_snap["theta"].cpu(), new_snap["theta"].cpu()):
                    raise RuntimeError("original Adam theta no longer exactly matches the old fit")
                _require(old_snap["loss"] == new_snap["loss"]
                         and list(old_snap["physical_interval"]) == list(new_snap["physical_interval"])
                         and old_snap["parameter_norm"] == new_snap["parameter_norm"]
                         and _tree_equal(old_snap["candidate_probabilities"],
                                         new_snap["candidate_probabilities"], torch),
                         "original Adam loss/interval no longer exactly matches the old fit")
            _require(fit["optimizer_steps"] == old_case["temporal_fit"]["optimizer_steps"],
                     "original Adam optimizer-step count differs from the old fit")
            _require(fit.get("status") == legacy.get("status")
                     and fit.get("active_parameters") == legacy.get("active_parameters")
                     and fit.get("final_interval") == legacy.get("final_interval"),
                     "original Adam fit-level status/readout differs from the old fit")
        _require(list(fit.get("final_interval", [])) == list(path[-1]["physical_interval"]),
                 f"temporal {mode} final interval differs from its sixth snapshot")
        if fit.get("no_information") is True:
            _require(fit.get("optimizer_steps") == 0
                     and all(int(torch.count_nonzero(row["theta"])) == 0 for row in path),
                     f"{mode} no-information fit changed parameters")
        mode_summaries[mode] = {
            "optimizer_steps": int(fit.get("optimizer_steps", -1)),
            "no_information": bool(fit.get("no_information")),
            "legacy_compatibility_loss": legacy_losses,
            "optimized_objective_loss": objectives,
            "subspace_constraint_enforced": mode != "original_adam",
            "objective_monotonic": (all(b <= a + 1e-12 for a, b in zip(objectives, objectives[1:]))
                                     if mode in ("trust", "per_offset_trust") else None),
            "full_reverse_kl_max": (max(float(x) for row in path
                                          for x in row["diagnostics"]["full_reverse_kl_by_offset"])
                                     if mode in ("trust", "per_offset_trust") else None),
            "evidence_subspace_max_residual": (max(subspace_residuals, default=0.0)
                                                if mode != "original_adam" else None),
        }
        all_metrics_inputs[mode] = [(int(row["step"]), list(row["physical_interval"])) for row in path]
    return {"modes": mode_summaries, "intervals": all_metrics_inputs}


def _cxcywh_to_xyxy(boxes: Any, torch: Any):
    x = boxes.detach().cpu().double()
    if x.ndim != 2 or x.shape[1] != 4 or not bool(torch.isfinite(x).all()):
        raise RuntimeError("cached decoder boxes must be a finite [frames,4] CPU tensor")
    return torch.cat((x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2), dim=1)


def _assert_xyxy_equal(left: Any, right: Any, torch: Any, where: str) -> None:
    import numpy as np
    a = left.detach().cpu().double().numpy() if torch.is_tensor(left) else np.asarray(left, dtype=np.float64)
    b = right.detach().cpu().double().numpy() if torch.is_tensor(right) else np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise RuntimeError(f"{where} is not an exact cached-decoder/readout match")


def _tensor_rms(boxes: Any, source_boxes: Any, torch: Any):
    if not torch.is_tensor(boxes) or not torch.is_tensor(source_boxes) or boxes.shape != source_boxes.shape or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError("source-cap boxes are not matching [frames,4] tensors")
    return torch.sqrt(((boxes.detach().double() - source_boxes.detach().double()) ** 2).mean(dim=1))


def _validate_spatial_fit(fit: Mapping[str, Any], case_id: str, arm_name: str,
                          torch: Any, expected_config: Mapping[str, Any]) -> dict[str, Any]:
    _require(fit.get("GT_read") is False and fit.get("scope") == "joint"
             and fit.get("key") == f"caption_query_fixed_space|{case_id}"
             and fit.get("active_parameters") == 1792 and fit.get("declared_steps") == 5,
             f"{arm_name} spatial fit is not the fixed 1,792-parameter five-step fit")
    _require(fit.get("config") == dict(expected_config), f"{arm_name} configuration changed")
    path, rounds = fit.get("path"), fit.get("rounds")
    _require(isinstance(path, list) and len(path) == 5 and isinstance(rounds, list) and len(rounds) == 5,
             f"{arm_name} lacks five optimizer-step readbacks")
    source_boxes = fit.get("source_decoder_boxes")
    _finite_tensor(source_boxes, torch, f"{arm_name}.source_decoder_boxes")
    _require(source_boxes.ndim == 2 and source_boxes.shape[1] == 4,
             f"{arm_name} source grid has invalid shape")
    limit = float(fit.get("source_frame_rms_limit"))
    _require(limit == 0.01 and fit.get("backtracking_alphas") == [2.0 ** (-i) for i in range(21)],
             f"{arm_name} source cap or fixed backtracking schedule changed")
    cap_rows = []
    previous_state = None
    previous_boxes = None
    accepted_steps = 0
    rejected_steps = 0
    for index, (entry, summary) in enumerate(zip(path, rounds)):
        _require(int(entry.get("step", -1)) == index and int(summary.get("step", -1)) == index,
                 f"{arm_name} path order is invalid")
        before_boxes = entry.get("boxes_before")
        after_boxes = entry.get("boxes_after")
        _finite_tensor(before_boxes, torch, f"{arm_name}.boxes_before[{index}]")
        _finite_tensor(after_boxes, torch, f"{arm_name}.boxes_after[{index}]")
        _require(torch.equal(entry.get("source_boxes"), source_boxes),
                 f"{arm_name} step {index} changed the source box reference")
        before_rms = _tensor_rms(before_boxes, source_boxes, torch)
        after_rms = _tensor_rms(after_boxes, source_boxes, torch)
        _require(float(before_rms.max()) <= limit + 1e-12 and float(after_rms.max()) <= limit + 1e-12,
                 f"{arm_name} step {index} violates the source-frame cap")
        _require(entry.get("constraint", {}).get("all_source_frames_within_limit") is True
                 and summary.get("constraint", {}).get("all_source_frames_within_limit") is True,
                 f"{arm_name} step {index} does not record source-cap acceptance")
        if previous_boxes is not None:
            _require(torch.equal(before_boxes, previous_boxes),
                     f"{arm_name} step {index} does not continue the prior accepted box state")
        previous_boxes = after_boxes.detach().clone()
        state_before = entry.get("state_before")
        state_after = entry.get("state_after")
        opt_before = entry.get("optimizer_state_before")
        opt_after = entry.get("optimizer_state_after")
        _require(isinstance(state_before, Mapping) and isinstance(state_after, Mapping)
                 and isinstance(opt_before, Mapping) and isinstance(opt_after, Mapping),
                 f"{arm_name} step {index} omits parameter/Adam restoration evidence")
        if previous_state is not None:
            _require(_tree_equal(state_before, previous_state, torch),
                     f"{arm_name} step {index} state-before differs from the prior state-after")
        previous_state = state_after
        attempts = entry.get("backtracking_attempts", [])
        accepted_alpha = entry.get("accepted_alpha")
        no_info = bool(entry.get("no_information"))
        rejected = False
        if no_info:
            _require(accepted_alpha is None and not attempts
                     and _tree_equal(state_before, state_after, torch)
                     and _tree_equal(opt_before, opt_after, torch),
                     f"{arm_name} no-information step {index} changed state or Adam")
            accepted = False
        else:
            _require(isinstance(attempts, list) and len(attempts) <= 21,
                     f"{arm_name} step {index} has an invalid backtracking trace")
            if accepted_alpha is None:
                rejected_steps += 1
                rejected = True
                _require(len(attempts) == 21 and not any(a.get("all_source_frames_within_limit") for a in attempts),
                         f"{arm_name} rejected step {index} did not test all 21 scales")
                _require(not summary.get("updated") and summary.get("proposal_rejected") is True,
                         f"{arm_name} rejected step {index} status differs from its round")
                _require(_tree_equal(state_before, state_after, torch)
                         and _tree_equal(opt_before, opt_after, torch),
                         f"{arm_name} rejected step {index} failed to restore parameters and Adam")
                accepted = False
            else:
                accepted_steps += 1
                _require(1 <= len(attempts) <= 21 and accepted_alpha in [2.0 ** (-i) for i in range(21)],
                         f"{arm_name} accepted step {index} has an invalid alpha")
                passing = [i for i, a in enumerate(attempts) if a.get("all_source_frames_within_limit") is True]
                _require(passing and passing[0] == len(attempts) - 1
                         and passing[0] == int(round(-math.log2(float(accepted_alpha))))
                         and attempts[passing[0]].get("alpha") == accepted_alpha,
                         f"{arm_name} did not accept the first source-cap-feasible scale")
                for attempt in attempts:
                    rms_values = attempt.get("per_frame_rms")
                    if rms_values is not None:
                        values = [float(x) for x in rms_values]
                        _require(len(values) == len(source_boxes) and all(math.isfinite(x) and x >= 0 for x in values),
                                 f"{arm_name} backtracking step {index} has invalid per-frame RMS")
                        _require(abs(max(values, default=0.0) - float(attempt["max_frame_rms"])) <= 1e-12,
                                 f"{arm_name} backtracking step {index} max-RMS summary differs")
                        within = max(values, default=0.0) <= limit
                        _require(bool(attempt.get("all_source_frames_within_limit")) == within,
                                 f"{arm_name} backtracking step {index} feasibility flag differs from frame RMS")
                proposal_state = entry.get("proposal_state")
                proposal_opt = entry.get("proposal_optimizer_state")
                _require(isinstance(proposal_state, Mapping) and isinstance(proposal_opt, Mapping),
                         f"{arm_name} accepted step {index} omits proposal state")
                alpha = float(accepted_alpha)
                expected_state = {}
                _require(set(state_before) == set(proposal_state) == set(state_after),
                         f"{arm_name} step {index} parameter key set changed")
                for key in state_before:
                    left, right = state_before[key], proposal_state[key]
                    _require(torch.is_tensor(left) and torch.is_tensor(right),
                             f"{arm_name} step {index} parameter state is not tensor-valued")
                    expected_state[key] = left.detach() + alpha * (right.detach().to(left) - left.detach())
                interpolation_error = 0.0
                for key, expected_value in expected_state.items():
                    actual_value = state_after[key]
                    _require(torch.is_tensor(actual_value)
                             and actual_value.shape == expected_value.shape,
                             f"{arm_name} step {index} interpolated state shape differs")
                    error = float((expected_value.detach().double() - actual_value.detach().double()).abs().max())
                    interpolation_error = max(interpolation_error, error)
                    _require(error <= 3e-7,
                             f"{arm_name} step {index} accepted-state interpolation differs by {error:.3g}")
                _require(_tree_equal(proposal_opt, opt_after, torch),
                         f"{arm_name} accepted step {index} does not preserve the original Adam proposal state")
                _require(summary.get("updated") is True and summary.get("proposal_rejected") is False,
                         f"{arm_name} accepted step {index} status differs from its round")
                accepted = True
        cap_rows.append({"step": index, "max_before_rms": float(before_rms.max()),
                         "max_after_rms": float(after_rms.max()),
                         "source_cap": limit, "updated": accepted,
                         "accepted_state_interpolation_max_abs_error": (interpolation_error if accepted else None),
                         "accepted_state_interpolation_atol_per_coordinate": (3e-7 if accepted else None),
                         "accepted_alpha": float(accepted_alpha) if accepted_alpha is not None else None,
                         "proposal_rejected": bool(summary.get("proposal_rejected", False)),
                         "adam_restored_on_rejection": (True if rejected else None)})
    _require(int(fit.get("optimizer_steps", -1)) == accepted_steps,
             f"{arm_name} accepted-step counter differs from its trace")
    _require(int(fit.get("rejected_updates", -1)) == rejected_steps,
             f"{arm_name} rejected-update counter differs from its trace")
    final_rms = _tensor_rms(fit.get("final_cached_decoder_boxes"), source_boxes, torch)
    _require(float(final_rms.max()) <= limit + 1e-12
             and fit.get("final_constraint", {}).get("all_source_frames_within_limit") is True,
             f"{arm_name} final cached readout exceeds its source cap")
    return {"accepted_optimizer_steps": accepted_steps,
            "rejected_proposals": rejected_steps,
            "accepted_state_interpolation_atol_per_coordinate": 3e-7,
            "max_source_rms_across_steps": max([max(x["max_before_rms"], x["max_after_rms"]) for x in cap_rows] + [float(final_rms.max())]),
            "steps": cap_rows}


def _validate_spatial_case(case_id: str, case: Mapping[str, Any], spatial: Mapping[str, Any],
                           old_case: Mapping[str, Any], old_sha: str,
                           reference_lock: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    _require(spatial.get("schema") == "stvg-support-aware-spatial-case-v1"
             and spatial.get("case_id") == case_id and spatial.get("dataset") == case["dataset"]
             and spatial.get("source_case_sha256") == case["_source_case_sha256"]
             and spatial.get("GT_read") is False,
             f"new student output is not source-bound for {case_id}")
    _require(spatial.get("pixel_sha256") == case.get("pixel_sha256")
             and spatial.get("source_model_state_sha256") == case.get("source_model_state_sha256")
             and list(spatial.get("frame_ids", [])) == list(case["frame_ids"]),
             f"new student output changed source pixels/model/frame grid for {case_id}")
    _require(spatial.get("source_model_state_restored") is True,
             "student did not record frozen full-model restoration")
    refs = {r["case_id"]: r for r in reference_lock.get("records", [])}
    ref = refs.get(case_id)
    _require(ref is not None, "spatial pseudoreference is absent from the fixed lock")
    binding = spatial.get("reference_binding", {})
    _require(binding.get("spatial_reference_lock_sha256") == _sha(OLD / "SPATIAL_REFERENCE_LOCK.json")
             and binding.get("response_sha256") == ref.get("actual_response_sha256")
             and binding.get("candidate_id") == (ref.get("candidate").get("candidate_id") if ref.get("candidate") else None)
             and binding.get("reference_is_student_pseudoreference") is True,
             f"spatial pseudoreference custody differs for {case_id}")

    reuse = spatial.get("reuse_binding", {})
    _require(reuse.get("previous_student_path") == f"{OLD.relative_to(ROOT)}/student/{case_id}.pt"
             and reuse.get("previous_student_sha256") == old_sha
             and reuse.get("previous_source_arm_exactly_reused") is True
             and reuse.get("previous_S2_arm_exactly_reused") is True,
             f"prior source/S2 reuse is not exact for {case_id}")
    _require(old_case.get("source_case_sha256") == case["_source_case_sha256"]
             and old_case.get("GT_read") is False,
             f"original student output is not bound to the same source for {case_id}")

    arms = spatial.get("arms", {})
    readouts = spatial.get("readouts", {})
    _require(set(arms) == set(SPATIAL_ARMS) and set(readouts) == set(SPATIAL_ARMS),
             f"spatial output does not contain exactly four locked arms for {case_id}")
    native_interval = list(case["Native_interval"])
    _require(list(spatial.get("native_interval", [])) == native_interval,
             f"native interval changed for {case_id}")
    for arm in SPATIAL_ARMS:
        _require(list(arms[arm].get("interval", [])) == native_interval,
                 f"{arm} is not the Native-time spatial arm")
    _assert_xyxy_equal(arms["S0_source"]["boxes"], old_case["arms"]["F"]["boxes"], torch,
                       "S0 source boxes vs original Frozen boxes")
    _assert_xyxy_equal(arms["S2_old_opd"]["boxes"], old_case["arms"]["S"]["boxes"], torch,
                       "reused S2 boxes vs original OPD boxes")
    _require(_tree_equal(readouts["S2_old_opd"], old_case["spatial_native_full_readout"], torch),
             "reused S2 full-model readout differs from the old sealed fit")

    config = spatial.get("configuration", {}).get("spatial_opd")
    _require(isinstance(config, dict) and config.get("steps") == 5 and config.get("samples") == 32,
             "student output changed the original OPD configuration")
    fits = spatial.get("fits", {})
    direct = fits.get("S1_direct", {})
    conservative = fits.get("S3_conservative_opd", {})
    _require(direct.get("config") == config, "Direct and conservative OPD configurations differ")
    _require(direct.get("GT_read") is False and direct.get("scope") == "joint"
             and spatial.get("configuration", {}).get("key") == f"caption_query_fixed_space|{case_id}"
             and direct.get("active_parameters") == 1792 and direct.get("declared_steps") == 5,
             "Direct spatial fit does not match the fixed five-step joint recipe")
    _require(conservative.get("config") == config, "conservative OPD changed the original per-domain settings")
    conservative_caps = _validate_spatial_fit(conservative, case_id, "S3_conservative_opd",
                                              torch, config)
    direct_path = direct.get("path", [])
    direct_trace = spatial.get("direct_step_readback", [])
    _require(len(direct_path) == len(direct_trace) == 5,
             "Direct fit lacks its five full-grid decoder readbacks")
    for index, trace in enumerate(direct_trace):
        entry = direct_path[index]
        _require(int(trace.get("step", -1)) == index
                 and torch.equal(trace.get("boxes_before").cpu(), entry.get("boxes").cpu())
                 and _tree_equal(trace.get("state_before"), entry.get("state"), torch)
                 and _tree_equal(trace.get("state_after"), entry.get("update_state", entry.get("state")), torch),
                 "Direct pre-step boxes or before/after parameter state differs from its independent replay receipt")
        expected_opt_before = entry.get("optimizer_before", entry.get("optimizer_state"))
        _require(_tree_equal(trace.get("optimizer_state_before"), expected_opt_before, torch)
                 and _tree_equal(trace.get("optimizer_state_after"), entry.get("optimizer_state"), torch),
                 "Direct optimizer-state replay differs from the saved per-step path")
        _require(bool(trace.get("no_update")) == bool(entry.get("skipped", False)),
                 "Direct skipped-step state differs from its independent replay receipt")
        if index + 1 < len(direct_path):
            _require(torch.equal(trace.get("boxes_after").cpu(), direct_path[index + 1]["boxes"].cpu())
                     and _tree_equal(trace.get("state_after"), direct_path[index + 1].get("state"), torch),
                     "Direct post-step replay does not continue at the next pre-step state")
    _require(torch.equal(direct_trace[-1].get("boxes_after").cpu(),
                         direct.get("final_cached_decoder_boxes").cpu()),
             "Direct last replay step differs from the sealed final cached decoder")
    _require(torch.equal(readouts["S1_direct"]["boxes"].cpu(), direct.get("final_cached_decoder_boxes").cpu())
             and torch.equal(readouts["S3_conservative_opd"]["boxes"].cpu(), conservative.get("final_cached_decoder_boxes").cpu()),
             "final full-model student readout differs from cached decoder boxes")
    source_logits = readouts["S0_source"].get("logits")
    _require(source_logits is not None and _tree_equal(source_logits, readouts["S1_direct"].get("logits"), torch)
             and _tree_equal(source_logits, readouts["S3_conservative_opd"].get("logits"), torch),
             "student adaptation changed frozen source temporal logits")
    _require(spatial.get("compute", {}).get("source_capture_calls") == 1
             and spatial.get("compute", {}).get("new_full_model_readouts") == 2
             and spatial.get("compute", {}).get("old_S2_readout_reused") is True,
             "per-case actual capture/readout counters are incomplete")
    barrier = spatial.get("compute", {})
    return {
        "source_cap": conservative_caps,
        "direct_optimizer_steps": int(direct.get("optimizer_steps", -1)),
        "conservative_optimizer_steps": int(conservative.get("optimizer_steps", -1)),
        "source_capture_calls": int(barrier["source_capture_calls"]),
        "new_full_model_readouts": int(barrier["new_full_model_readouts"]),
        "full_model_readout_exact": True,
        "source_logits_bitwise_preserved": True,
    }


def _load_and_validate_predictions(gate: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from scripts.stvg_temporal_caption_query_v1 import catalog
    from scripts.stvg_target_teacher_student_v1 import load_original_opd_config

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only review")
    torch.set_num_threads(1)
    outputs: dict[str, Any] = {}
    refs = gate["references"]
    for rec in gate["records"]:
        case_id = rec["case_id"]
        case, _catalog = catalog(rec)
        source_sha = case.get("_source_case_sha256", case.get("source_case_sha256"))
        _require(source_sha == rec["source_case_sha256"], f"source case SHA differs for {case_id}")
        old_path = OLD / "student" / f"{case_id}.pt"
        old_sha = gate["old_student_files"][str(old_path.relative_to(ROOT))]["sha256"]
        old_case = torch.load(old_path, map_location="cpu", weights_only=False)
        old_case["_sealed_sha256"] = old_sha
        temporal_path = BASE / "temporal" / f"{case_id}.pt"
        spatial_path = BASE / "student" / f"{case_id}.pt"
        temporal = torch.load(temporal_path, map_location="cpu", weights_only=False)
        spatial = torch.load(spatial_path, map_location="cpu", weights_only=False)
        _cpu_tree(old_case, torch, f"original student {case_id}")
        _cpu_tree(temporal, torch, f"temporal result {case_id}")
        _cpu_tree(spatial, torch, f"spatial result {case_id}")
        temporal_check = _validate_temporal_case(case_id, case, temporal, old_case, torch)
        expected_config = load_original_opd_config(rec["dataset"])
        expected_config["steps"] = 5
        _require(spatial.get("configuration", {}).get("spatial_opd") == expected_config,
                 f"spatial recipe differs from original domain configuration for {case_id}")
        spatial_check = _validate_spatial_case(case_id, case, spatial, old_case, old_sha,
                                              gate["references"], torch)
        outputs[case_id] = {"record": rec, "case": case, "old_case": old_case,
                            "temporal": temporal, "spatial": spatial,
                            "temporal_check": temporal_check,
                            "spatial_check": spatial_check}
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only prediction validation initialized CUDA")
    return outputs


def _metric_rows(outputs: Mapping[str, Any], truths: Mapping[str, Any],
                 official: Any, independent: Any):
    import numpy as np
    from scripts.review_stvg_sparse_r1_v2 import score_tube_xyxy

    rows = []
    max_error = 0.0

    def score(alias: str, case: Mapping[str, Any], boxes: Any, interval: Sequence[int],
              *, family: str, mode: str | None = None, step: int | None = None,
              arm: str | None = None, baseline: Mapping[str, float] | None = None,
              second_baseline: Mapping[str, float] | None = None):
        nonlocal max_error
        item = next(v for v in outputs.values() if v["alias"] == alias)
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().double().numpy()
        x = score_tube_xyxy(
            boxes, case["frame_ids"], _dimension(case, "width"), _dimension(case, "height"),
            interval, truths[alias]["truth"], truths[alias]["span"],
            item["record"]["dataset"],
            official=official, independent=independent)
        max_error = max(max_error, float(x["official_independent_max_abs_error"]))
        metrics = {k: float(v) for k, v in x["metrics"].items()}
        row = {"case": alias, "dataset": item["record"]["dataset"],
               "family": family, "time_mode": mode, "step": step, "spatial_arm": arm,
               "metrics": metrics,
               "delta_vs_primary_baseline_pp": ({k: (metrics[k] - float(baseline[k])) * 100 for k in METRICS}
                                                  if baseline is not None else None),
               "delta_vs_native_time_same_arm_pp": ({k: (metrics[k] - float(second_baseline[k])) * 100 for k in METRICS}
                                                      if second_baseline is not None else None),
               "official_independent_max_abs_error": float(x["official_independent_max_abs_error"])}
        rows.append(row)
        return metrics

    for _, item in outputs.items():
        alias, case = item["alias"], item["case"]
        temporal = item["temporal"]["results"]
        frozen = item["old_case"]["arms"]["F"]["boxes"]
        frame_count = len(case["frame_ids"])
        _require(np.asarray(frozen).shape == (frame_count, 4), "Frozen boxes do not cover the full source grid")
        projection_baseline = None
        for mode in MODES:
            path = temporal[mode]["path"]
            for snap in path:
                metrics = score(alias, case, frozen, snap["physical_interval"],
                    family="frozen_box_temporal_path", mode=mode, step=int(snap["step"]),
                    baseline=projection_baseline)
                if projection_baseline is None:
                    # The first original-Adam step is the shared zero-readout
                    # Projection prior and is also the first reported row.
                    projection_baseline = metrics
                    rows[-1]["delta_vs_primary_baseline_pp"] = {k: 0.0 for k in METRICS}
                # Attach only the two scalar loss tracks; never serialize theta/features.
                rows[-1]["legacy_compatibility_loss"] = float(snap["loss"])
                rows[-1]["optimized_objective_loss"] = float(snap["objective_loss"])
                rows[-1]["loss_track_note"] = "loss is legacy compatibility aggregate; objective_loss is the optimized objective"
        native_baselines = {}
        for arm in SPATIAL_ARMS:
            arm_row = item["spatial"]["arms"][arm]
            native_baseline = native_baselines.get("S0_source")
            native_baselines[arm] = score(alias, case, arm_row["boxes"], case["Native_interval"],
                family="spatial_arm_at_native_time", arm=arm,
                baseline=native_baseline)
            if arm == "S0_source":
                rows[-1]["delta_vs_primary_baseline_pp"] = {k: 0.0 for k in METRICS}
        for arm in SPATIAL_ARMS:
            metrics = score(alias, case, item["spatial"]["arms"][arm]["boxes"],
                temporal[TRUST_TIME_MODE]["path"][-1]["physical_interval"],
                family="locked_trust_time_by_spatial_arm", mode=TRUST_TIME_MODE,
                arm=arm, baseline=native_baselines["S0_source"],
                second_baseline=native_baselines[arm])
            if arm == "S0_source":
                rows[-1]["delta_vs_primary_baseline_pp"] = {
                    k: (metrics[k] - native_baselines["S0_source"][k]) * 100 for k in METRICS}
    return rows, max_error


def _dimension(case: Mapping[str, Any], key: str) -> int:
    value = case.get(key)
    if value is None and isinstance(case.get("input"), Mapping):
        value = case["input"].get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) <= 0:
        raise ValueError(f"source case has no valid {key}")
    return int(value)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["family"], row.get("time_mode"), row.get("step"), row.get("spatial_arm"), row["dataset"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: tuple("" if x is None else str(x) for x in item[0])):
        metrics = {m: sum(float(r["metrics"][m]) for r in group) / len(group) for m in METRICS}
        delta_rows = [r for r in group if isinstance(r.get("delta_vs_primary_baseline_pp"), dict)]
        delta = ({m: sum(float(r["delta_vs_primary_baseline_pp"][m]) for r in delta_rows) / len(delta_rows)
                  for m in METRICS} if delta_rows else None)
        viou = [float(r["delta_vs_primary_baseline_pp"]["m_vIoU"]) for r in delta_rows]
        result.append({"family": key[0], "time_mode": key[1], "step": key[2],
                       "spatial_arm": key[3], "dataset": key[4], "n": len(group),
                       "mean_metrics_percent": {m: 100 * metrics[m] for m in METRICS},
                       "mean_delta_vs_primary_baseline_pp": delta,
                       "viou_positive_negative_neutral": {
                           "positive": sum(x > 1e-8 for x in viou),
                           "negative": sum(x < -1e-8 for x in viou),
                           "neutral": sum(abs(x) <= 1e-8 for x in viou),
                       } if viou else None})
    return result


def _costs(gate: Mapping[str, Any], outputs: Mapping[str, Any]) -> dict[str, Any]:
    temporal_barrier, student_barrier = gate["temporal_barrier"], gate["student_barrier"]
    cases = list(outputs.values())
    spatial_checks = [x["spatial_check"] for x in cases]
    return {
        "original_teacher_stage": {
            "sealed_output_file_count": len(gate["old_teacher_barrier"].get("files", [])),
            "additional_calls_for_this_comparison": 0,
            "reported_actual_model_requests": gate["old_teacher_barrier"].get("actual_model_requests"),
        },
        "support_aware_temporal_stage": {
            "sealed_cases": int(temporal_barrier["count"]),
            "fit_modes_per_case": int(temporal_barrier["CPU_fit_modes"]),
            "actual_new_teacher_calls": int(temporal_barrier["actual_new_teacher_calls"]),
            "actual_new_model_forwards": int(temporal_barrier["actual_new_model_forwards"]),
            "CPU_only": temporal_barrier.get("CPU_only") is True,
        },
        "support_aware_spatial_student_stage": {
            "sealed_cases": int(student_barrier["count"]),
            "source_capture_calls": int(student_barrier.get("source_capture_calls", 0)),
            "new_full_model_final_readout_calls": int(student_barrier.get("new_full_model_final_readout_calls", 0)),
            "previous_S0_and_S2_outputs_reused": student_barrier.get("previous_source_and_S2_outputs_reused") is True,
            "direct_optimizer_steps": sum(x["spatial_check"]["direct_optimizer_steps"] for x in cases),
            "conservative_opd_optimizer_steps": sum(x["spatial_check"]["conservative_optimizer_steps"] for x in cases),
            "elapsed_seconds": student_barrier.get("elapsed_seconds"),
            "source_capture_calls_from_cases": sum(x["spatial_check"]["source_capture_calls"] for x in cases),
            "new_full_model_readouts_from_cases": sum(x["spatial_check"]["new_full_model_readouts"] for x in cases),
            "accepted_conservative_steps": sum(x["source_cap"]["accepted_optimizer_steps"] for x in spatial_checks),
            "rejected_conservative_proposals": sum(x["source_cap"]["rejected_proposals"] for x in spatial_checks),
            "max_source_frame_rms": max(x["source_cap"]["max_source_rms_across_steps"] for x in spatial_checks),
        },
        "metric_readouts": {
            "expected_count": 8 * (len(MODES) * 6 + len(SPATIAL_ARMS) + len(SPATIAL_ARMS)),
            "scope": "five six-snapshot Frozen-box temporal paths, four spatial arms at Native time, trust final time crossed with four spatial arms",
        },
    }


def _write_atomic(path: Path, data: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable readback: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    """Run only after root has sealed and pinned every prediction/source input."""
    gate = _verify_global_inputs()
    outputs = _load_and_validate_predictions(gate)
    # Assign stable anonymous aliases after the exact locked 4+4 roster is checked.
    domain_counts = {"hc2": 0, "vidstg": 0}
    for item in outputs.values():
        dataset = item["record"]["dataset"]
        if dataset not in domain_counts:
            raise RuntimeError("unexpected dataset in the fixed roster")
        domain_counts[dataset] += 1
    if domain_counts != {"hc2": 4, "vidstg": 4}:
        raise RuntimeError("the fixed cohort is not the locked 4+4 domain split")
    # Root order is the pre-existing locked roster order, giving aliases HC01..04 and V01..04.
    counters = {"hc2": 0, "vidstg": 0}
    for item in outputs.values():
        dataset = item["record"]["dataset"]
        counters[dataset] += 1
        item["alias"] = ("HC" if dataset == "hc2" else "V") + f"{counters[dataset]:02d}"

    from scripts.review_stvg_sparse_r1_v2 import _metric_functions, _truths_after_seal
    # Construct metric functions before the one and only truth-loading call.
    official, independent, metric_contract = _metric_functions()
    source_cases = {case_id: item["case"] for case_id, item in outputs.items()}
    truths, truth_bindings = _truths_after_seal(source_cases)
    truth_by_alias = {outputs[case_id]["alias"]: truths[case_id] for case_id in outputs}
    # The metric row helper indexes by public alias while retaining private case IDs internally.
    metric_outputs = {case_id: {**item, "alias": item["alias"]} for case_id, item in outputs.items()}
    rows, max_error = _metric_rows(metric_outputs, truth_by_alias, official, independent)
    expected_readouts = 8 * (len(MODES) * 6 + 2 * len(SPATIAL_ARMS))
    if len(rows) != expected_readouts:
        raise RuntimeError(f"metric readback count mismatch: {len(rows)} != {expected_readouts}")
    aggregates = _aggregate(rows)
    cost = _costs(gate, outputs)
    cost["metric_readouts"]["actual_count"] = len(rows)

    anonymous_case_rows = []
    case_root_hashes = {}
    source_bindings = {}
    for case_id, item in outputs.items():
        alias = item["alias"]
        local = [row for row in rows if row["case"] == alias]
        local_summary = {
            "case": alias, "dataset": item["record"]["dataset"],
            "temporal_methods": item["temporal_check"]["modes"],
            "spatial_source_cap": item["spatial_check"]["source_cap"],
            "spatial_optimizer_steps": {
                "Direct": item["spatial_check"]["direct_optimizer_steps"],
                "conservative_OPD": item["spatial_check"]["conservative_optimizer_steps"],
            },
            "spatial_full_model_readout_exact": item["spatial_check"]["full_model_readout_exact"],
            "metric_rows": local,
        }
        anonymous_case_rows.append(local_summary)
        case_root_hashes[alias] = _canonical_sha(local_summary)
        source_bindings[alias] = {
            "source_case_sha256": item["record"]["source_case_sha256"],
            "original_student_sha256": item["old_case"]["_sealed_sha256"],
            "temporal_file_sha256": gate["temporal_files"][str((BASE / "temporal" / f"{case_id}.pt").relative_to(ROOT))]["sha256"],
            "student_file_sha256": gate["student_files"][str((BASE / "student" / f"{case_id}.pt").relative_to(ROOT))]["sha256"],
        }

    body = {
        "schema": "stvg-support-aware-temporal-spatial-postseal-v1",
        "status": "complete_actual_all_eight_case_postseal_readback",
        "scope": "Frozen-box five-mode temporal trajectories; four source-preserving spatial arms at Native time; locked trust-time by all four spatial arms",
        "cases": 8,
        "anonymous_case_order": [row["case"] for row in anonymous_case_rows],
        "domain_counts": {"hc2": 4, "vidstg": 4},
        "historical_exposure": "the locked cohort is historically GT-exposed; this readback is not fresh or confirmation data",
        "prior_label_access": "the root had already completed an earlier post-seal scoring pass; this reader is a subsequent independent readback",
        "GT_read": True,
        "GT_loader_called_only_after_full_prediction_and_source_preflight": True,
        "time_mode_selection": {"trust_cross_mode": TRUST_TIME_MODE,
                                 "selection_rule": "fixed in this reader and reported alongside all five modes; no GT-based mode selection"},
        "loss_semantics": "legacy loss is the compatibility aggregate; objective_loss is the optimized objective, especially for per_offset_trust",
        "metrics": {"primary": "m_vIoU", "additional": ["m_tIoU", "m_sIoU_dense_GT"],
                    "values_are_fraction": True,
                    "delta_units": "percentage points",
                    "official_independent_max_abs_error": max_error,
                    "official_independent_tolerance": 2e-9,
                    "readout_count": len(rows)},
        "aggregates": aggregates,
        "cases_readback": anonymous_case_rows,
        "cost_and_actual_counters": cost,
        "provenance": {
            "root_readback_runtime_sha256": gate["locks"]["root_readback"],
            "temporal_runtime_sha256": gate["runtimes"]["temporal"],
            "student_runtime_sha256": gate["runtimes"]["student"],
            "original_runtime_sha256": gate["runtimes"]["original"],
            "original_student_runtime_sha256": gate["runtimes"]["original_student"],
            "temporal_barrier_sha256": gate["barriers"]["temporal"],
            "student_barrier_sha256": gate["barriers"]["student"],
            "original_teacher_barrier_sha256": gate["barriers"]["original_teacher"],
            "original_student_barrier_sha256": gate["barriers"]["original_student"],
            "roster_lock_sha256": gate["locks"]["roster"],
            "job_lock_sha256": gate["locks"]["jobs"],
            "spatial_reference_lock_sha256": gate["locks"]["reference"],
            "metric_contract_sha256": _sha(Path(metric_contract["path"])) if isinstance(metric_contract, dict) and isinstance(metric_contract.get("path"), str) and Path(metric_contract["path"]).is_file() else None,
            "case_roots": case_root_hashes,
            "source_bindings_by_anonymous_case": source_bindings,
            "temporal_barrier_runtime_boundary": gate["temporal_runtime_barrier_binding"],
            "truth_binding_sha256": _canonical_sha(truth_bindings),
        },
        "diagnostic_scope": {
            "parameter_tensors_serialized": False,
            "box_coordinates_or_physical_intervals_serialized": False,
            "query_or_caption_text_serialized": False,
            "GT_coordinates_or_spans_serialized": False,
            "models_or_GPU_invoked_by_reader": False,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = BASE / "ROOT_POSTSEAL_REPORT.json"
    _write_atomic(report_path, body)
    receipt = {
        "schema": "stvg-support-aware-root-readback-v1",
        "status": "pass",
        "scope": "full eight-case sealed CPU source/runtime/prediction readback and post-seal official metric comparison",
        "count": 8,
        "metric_readouts": len(rows),
        "report_sha256": _sha(report_path),
        "GT_read": True,
        "CPU_only": True,
        "CUDA_initialized": False,
        "official_independent_max_abs_error": max_error,
        "root_readback_runtime_sha256": gate["locks"]["root_readback"],
        "temporal_barrier_runtime_sha_missing_disclosed": True,
        "reader_sha256": _sha(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(BASE / "ROOT_READBACK.json", receipt)
    print(json.dumps({"status": receipt["status"], "count": 8,
                      "metric_readouts": len(rows), "max_metric_error": max_error}, sort_keys=True))


if __name__ == "__main__":
    main()
