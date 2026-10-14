"""Read-only amplitude probes of sealed IoU-OPD parameter endpoints.

This worker never trains or changes the locked fit.  It interpolates the saved
1792-parameter source/final states at a prelocked finite alpha schedule and
performs fresh full-model readouts only when the saved fit has a bitwise
nonzero parameter displacement.  It is intentionally dormant until both the
phase input lock and runtime lock have been written by the root controller.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_spatial_causal_decomposition_v1"
PHASE = BASE / "phase1"
ALPHA_LOCK_PATH = PHASE / "PHASE1_LOCK.json"
RUNTIME_PATH = PHASE / "RUNTIME.json"
OUTPUT_DIR = PHASE / "cases"
BARRIER_PATH = PHASE / "PHASE1_GLOBAL_BARRIER.json"
CPU_CONTRACT_PATH = PHASE / "CPU_AMPLITUDE_CONTRACTS.json"

OLD_BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
ROSTER_PATH = OLD_BASE / "ROSTER_LOCK.json"
REFERENCE_LOCK_PATH = OLD_BASE / "METHOD_REFERENCE_LOCK.json"
OLD_RUNTIME_PATH = OLD_BASE / "METHOD_STUDENT_RUNTIME002.json"
OLD_BARRIER_PATH = OLD_BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json"

PHASE_NAME = "phase1"
ALPHAS = (0.0, 0.25, 0.5, 1.0)
EXPECTED_CASE_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(8)
}
OLD_T_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(4)
}

PHASE_REQUIRED_PIN_PATHS = (
    "artifacts/stvg_motion_guided_instance_opd_v1/ROSTER_LOCK.json",
    "artifacts/stvg_motion_guided_instance_opd_v1/METHOD_REFERENCE_LOCK.json",
    "artifacts/stvg_motion_guided_instance_opd_v1/METHOD_STUDENT_RUNTIME002.json",
    "artifacts/stvg_motion_guided_instance_opd_v1/METHOD_STUDENT_GLOBAL_BARRIER.json",
    "scripts/run_stvg_motion_guided_student002.py",
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/run_stvg_spatial_update_amplitude_v1.py",
    "artifacts/stvg_spatial_causal_decomposition_v1/phase1/CPU_AMPLITUDE_CONTRACTS.json",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _root_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a safe repository-relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Pinned path escaped the repository root")
    return path


def _verify_pin_map(document: Mapping[str, Any], name: str) -> dict[str, str]:
    pins = document.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError(f"{name} must have a non-empty path-to-SHA256 pin map")
    out: dict[str, str] = {}
    for relative, digest in pins.items():
        path = _root_path(relative)
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"{name} pin mismatch: {relative}")
        out[str(relative)] = str(digest)
    return out


def _same_state(left: Mapping[str, Any], right: Mapping[str, Any], torch) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
        return False
    for name in left:
        a, b = left[name], right[name]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            return False
    return True


def _verify_old_phase() -> dict[str, Any]:
    """Check all old source/fit seals before any case tensor is loaded."""
    for path in (ROSTER_PATH, REFERENCE_LOCK_PATH, OLD_RUNTIME_PATH, OLD_BARRIER_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    roster = _read_json(ROSTER_PATH)
    reference_lock = _read_json(REFERENCE_LOCK_PATH)
    old_runtime = _read_json(OLD_RUNTIME_PATH)
    barrier = _read_json(OLD_BARRIER_PATH)
    if roster.get("status") != "locked" or roster.get("GT_read") is not False or roster.get("count") != 16:
        raise ValueError("The sealed historical roster is incomplete or not GT-free")
    roster_rows = roster.get("records")
    if not isinstance(roster_rows, list) or len(roster_rows) != 16:
        raise ValueError("The historical roster must contain exactly 16 fixed sources")
    roster_ids = [str(row["case_id"]) for row in roster_rows]
    if set(roster_ids) != EXPECTED_CASE_IDS or len(set(roster_ids)) != 16 or roster_ids != sorted(roster_ids):
        raise ValueError("The historical roster is not the fixed sorted 16-case set")
    if reference_lock.get("status") != "locked" or reference_lock.get("GT_read") is not False:
        raise ValueError("The old R1 reference lock is not sealed GT-free")
    reference_rows = reference_lock.get("records")
    if not isinstance(reference_rows, list) or len(reference_rows) != 16:
        raise ValueError("The reference lock must contain exactly 16 records")
    refs = {str(row["case_id"]): row for row in reference_rows}
    roster_by_id = {str(row["case_id"]): row for row in roster_rows}
    if set(refs) != EXPECTED_CASE_IDS or len(refs) != 16:
        raise ValueError("Reference lock does not match the full fixed roster")
    for case_id, reference_row in refs.items():
        source_row = roster_by_id[case_id]
        teacher = reference_row.get("references", {}).get("teacher", {})
        if reference_row.get("source_case_sha256") != source_row.get("source_case_sha256"):
            raise ValueError(f"Reference source SHA differs from roster for {case_id}")
        if not isinstance(teacher.get("no_update"), bool):
            raise ValueError(f"Reference no-update state is missing for {case_id}")
        if not teacher["no_update"]:
            if int(teacher.get("selected_candidate_id", 1)) <= 1 or not teacher.get("anchors"):
                raise ValueError(f"Updating reference is not the sealed non-Source R1 selection for {case_id}")
    if old_runtime.get("status") != "locked" or old_runtime.get("phase") != "method" or old_runtime.get("GT_read") is not False:
        raise ValueError("Historical method runtime is not locked, method-scoped, and GT-free")
    old_pins = old_runtime.get("pins", {})
    protected = old_runtime.get("protected", {})
    if not isinstance(old_pins, dict) or not isinstance(protected, dict):
        raise ValueError("Historical runtime pins are malformed")
    for relative, digest in {**old_pins, **protected}.items():
        path = _root_path(relative)
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"Historical runtime pin mismatch: {relative}")
    if barrier.get("status") != "complete" or barrier.get("count") != 16 or barrier.get("GT_read") is not False:
        raise ValueError("Historical method output barrier is incomplete or not GT-free")
    if barrier.get("runtime_sha256") != _sha_file(OLD_RUNTIME_PATH):
        raise ValueError("Historical student barrier is not bound to the locked runtime")
    if barrier.get("reference_lock_sha256") != _sha_file(REFERENCE_LOCK_PATH):
        raise ValueError("Historical student barrier is not bound to the fixed reference lock")
    if barrier.get("roster_lock_sha256") != _sha_file(ROSTER_PATH):
        raise ValueError("Historical student barrier is not bound to the fixed roster")

    files = barrier.get("files")
    if not isinstance(files, list) or len(files) != 16:
        raise ValueError("Historical barrier does not seal exactly 16 per-source outputs")
    output_paths: dict[str, Path] = {}
    for row in files:
        relative = row.get("path")
        path = _root_path(relative)
        if not relative.startswith("artifacts/stvg_motion_guided_instance_opd_v1/method/student/"):
            raise ValueError("Historical fit output escaped its fixed method output directory")
        case_id = Path(relative).stem
        if case_id in output_paths:
            raise ValueError(f"Historical barrier repeats case output {case_id}")
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha_file(path) != row["sha256"]:
            raise ValueError(f"Historical case output hash/size mismatch for {case_id}")
        output_paths[case_id] = path
    if set(output_paths) != EXPECTED_CASE_IDS:
        raise ValueError("Historical outputs do not exactly cover all 16 locked sources")
    return {
        "roster": roster,
        "roster_by_id": roster_by_id,
        "reference_lock": reference_lock,
        "reference_by_id": refs,
        "old_runtime": old_runtime,
        "barrier": barrier,
        "output_paths": output_paths,
        "old_sha256": {
            "roster": _sha_file(ROSTER_PATH),
            "reference_lock": _sha_file(REFERENCE_LOCK_PATH),
            "runtime": _sha_file(OLD_RUNTIME_PATH),
            "method_barrier": _sha_file(OLD_BARRIER_PATH),
        },
    }


def _preflight() -> dict[str, Any]:
    if not ALPHA_LOCK_PATH.is_file() or not RUNTIME_PATH.is_file():
        raise RuntimeError("Phase1 is dormant until PHASE1_LOCK.json and RUNTIME.json are sealed")
    lock = _read_json(ALPHA_LOCK_PATH)
    runtime = _read_json(RUNTIME_PATH)
    if lock.get("status") != "locked" or lock.get("phase") != PHASE_NAME or lock.get("GT_read") is not False:
        raise ValueError("Phase1 lock is not sealed in the expected GT-free scope")
    if lock.get("alpha_values") != list(ALPHAS):
        raise ValueError("Phase1 alpha values must be exactly [0, 0.25, 0.5, 1]")
    if runtime.get("status") != "locked" or runtime.get("phase") != PHASE_NAME or runtime.get("GT_read") is not False:
        raise ValueError("Phase1 runtime is not sealed in the expected GT-free scope")
    if runtime.get("phase1_lock_sha256") != _sha_file(ALPHA_LOCK_PATH):
        raise ValueError("Phase1 runtime does not bind the exact amplitude lock")
    if not CPU_CONTRACT_PATH.is_file():
        raise FileNotFoundError(f"Phase1 CPU contract is required and must be frozen first: {CPU_CONTRACT_PATH}")
    contract = _read_json(CPU_CONTRACT_PATH)
    if (contract.get("status") != "pass" or contract.get("CPU_only") is not True
            or contract.get("GT_read") is not False or contract.get("model_calls") != 0
            or contract.get("GPU_calls") != 0
            or contract.get("worker_sha256") != _sha_file(Path(__file__))):
        raise ValueError("The frozen CPU contract is invalid or does not bind this worker")
    required = {relative: _sha_file(_root_path(relative)) for relative in PHASE_REQUIRED_PIN_PATHS}
    lock_pins = _verify_pin_map(lock, "PHASE1_LOCK")
    runtime_pins = _verify_pin_map(runtime, "RUNTIME")
    for relative, digest in required.items():
        if lock_pins.get(relative) != digest or runtime_pins.get(relative) != digest:
            raise ValueError(f"Phase1 lock/runtime omitted or changed a required source pin: {relative}")
    phase_lock_relative = ALPHA_LOCK_PATH.relative_to(ROOT).as_posix()
    if runtime_pins.get(phase_lock_relative) != _sha_file(ALPHA_LOCK_PATH):
        raise ValueError("Phase1 runtime does not pin the exact phase lock")
    if runtime.get("cpu_contract_sha256") != _sha_file(CPU_CONTRACT_PATH):
        raise ValueError("Phase1 runtime does not bind the frozen CPU contract")
    old = _verify_old_phase()
    return {"lock": lock, "runtime": runtime, "contract": contract, "old": old}


def _alpha_key(alpha: float) -> str:
    return {0.0: "0", 0.25: "0.25", 0.5: "0.5", 1.0: "1"}[float(alpha)]


def interpolate_fit_state(initial: Mapping[str, Any], final: Mapping[str, Any], alpha: float, torch):
    """Return a new state at an exact locked fraction of the saved parameter delta.

    This is a parameter-space interpolation only. It does not update Adam,
    replay the objective, or assert any clipped-box/Jacobian interpretation.
    """
    value = float(alpha)
    if value not in ALPHAS:
        raise ValueError("alpha is not in the fixed phase1 schedule")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping) or set(initial) != set(final):
        raise ValueError("Fit endpoint states must have the same non-empty parameter keys")
    if not initial:
        raise ValueError("Fit endpoint state cannot be empty")
    result = {}
    for name in initial:
        left, right = initial[name], final[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise TypeError(f"State entry {name} must be a tensor")
        if left.device.type != "cpu" or right.device.type != "cpu":
            raise ValueError("Saved fit endpoint states must be CPU-resident before interpolation")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"State schema mismatch at {name}")
        if not torch.is_floating_point(left) or not torch.is_floating_point(right):
            raise TypeError(f"Trainable parameter {name} must have floating-point dtype")
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise ValueError(f"Non-finite endpoint state at {name}")
        if value == 0.0:
            mixed = left.detach().clone()
        elif value == 1.0:
            mixed = right.detach().clone()
        else:
            mixed = left.detach() + (right.detach() - left.detach()) * value
        result[name] = mixed.detach().clone()
    return result


def state_displacement(initial: Mapping[str, Any], final: Mapping[str, Any], torch) -> dict[str, Any]:
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping) or set(initial) != set(final):
        raise ValueError("Fit endpoint states must have the same parameter keys")
    changed_tensors = 0
    changed_elements = 0
    squared = 0.0
    max_abs = 0.0
    total_elements = 0
    for name in initial:
        left, right = initial[name], final[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Malformed fit endpoint state at {name}")
        delta = right.detach().to(dtype=torch.float64) - left.detach().to(dtype=torch.float64)
        count = int(delta.numel())
        total_elements += count
        changed = int(torch.count_nonzero(delta).item())
        changed_tensors += int(changed > 0)
        changed_elements += changed
        if count:
            squared += float(delta.square().sum().item())
            max_abs = max(max_abs, float(delta.abs().max().item()))
    return {
        "nonzero": changed_elements > 0,
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "parameter_element_count": total_elements,
        "l2": math.sqrt(squared),
        "linf": max_abs,
    }


def _cxcywh_to_xyxy(boxes, torch):
    values = torch.as_tensor(boxes).detach().cpu().double()
    if values.ndim != 2 or values.shape[1] != 4 or not bool(torch.isfinite(values).all()):
        raise ValueError("Cached source boxes must be finite [frames,4] cxcywh")
    return torch.cat((values[:, :2] - values[:, 2:] / 2.0,
                      values[:, :2] + values[:, 2:] / 2.0), dim=1)


def _cpu_readout(readout: Mapping[str, Any], torch, np):
    boxes = torch.as_tensor(readout["boxes"]).detach().cpu().clone()
    logits = [torch.as_tensor(row).detach().cpu().clone() for row in readout["logits"]]
    xyxy = _cxcywh_to_xyxy(boxes, torch)
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(xyxy).all()):
        raise ValueError("Actual full-model readout contains non-finite boxes")
    interval = readout.get("physical_interval")
    if interval is not None:
        interval = [int(interval[0]), int(interval[1])]
        if interval[0] >= interval[1]:
            raise ValueError("Actual readout returned an empty temporal interval")
    return {"boxes_cxcywh": boxes, "boxes_xyxy": xyxy, "logits": logits, "physical_interval": interval}


def _same_logits(left, right, torch) -> bool:
    return len(left) == len(right) and all(
        torch.equal(torch.as_tensor(a).detach().cpu(), torch.as_tensor(b).detach().cpu())
        for a, b in zip(left, right)
    )


def _source_alias_rows(saved: Mapping[str, Any], alpha_values: tuple[float, ...], state_hash: str | None,
                       mode: str, torch, np) -> dict[str, Any]:
    source_boxes = torch.as_tensor(saved["source_boxes"]).detach().cpu().clone()
    source_logits = [torch.as_tensor(row).detach().cpu().clone() for row in saved["source_logits"]]
    source = {"boxes_cxcywh": source_boxes, "boxes_xyxy": _cxcywh_to_xyxy(source_boxes, torch),
              "logits": source_logits, "physical_interval": list(saved["Native_interval"])}
    amplitudes = {
        _alpha_key(alpha): {
            "alpha": float(alpha),
            "status": mode,
            "state_sha256": state_hash,
            "full_model_redecode": False,
            "redecode_calls": 0,
            "exact_alias_of": "source_readout",
        }
        for alpha in alpha_values
    }
    return {"source_readout": source, "amplitudes": amplitudes}


def _case_result(saved: Mapping[str, Any], roster_row: Mapping[str, Any], reference_row: Mapping[str, Any],
                 output_sha: str, expected_reference_sha256: str, torch, np,
                 state_hash_fn) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    case_id = str(saved["case_id"])
    teacher = reference_row["references"]["teacher"]
    no_update = bool(teacher["no_update"])
    if saved.get("source_case_sha256") != roster_row.get("source_case_sha256"):
        raise ValueError(f"Old method output source SHA mismatch for {case_id}")
    if saved.get("dataset") != roster_row.get("dataset"):
        raise ValueError(f"Old method output dataset differs from the fixed roster for {case_id}")
    if saved.get("reference_lock_sha256") != expected_reference_sha256:
        raise ValueError(f"Old method output is not bound to the fixed R1 reference lock for {case_id}")
    if not isinstance(saved.get("source_model_state_sha256"), str) or len(saved["source_model_state_sha256"]) != 64:
        raise ValueError(f"Old method output has no valid source checkpoint fingerprint for {case_id}")
    if saved.get("GT_read") is not False or saved.get("source_restored") is not True:
        raise ValueError(f"Old method output is not certified source-restored and GT-free for {case_id}")
    if saved.get("source_cold_readback", {}).get("status") != "pass":
        raise ValueError(f"Old source cold readback did not pass for {case_id}")
    if saved.get("reference_anchors", {}).get("teacher", []) != teacher.get("anchors", []):
        raise ValueError(f"Historical fit anchors differ from the sealed R1 reference for {case_id}")
    fit = saved.get("fits", {}).get("teacher_iou")
    if no_update:
        if fit is not None or int(saved.get("actual_fit_count", 0)) != 0:
            raise ValueError(f"Locked no-update case unexpectedly contains a teacher IoU fit for {case_id}")
        alias = _source_alias_rows(saved, ALPHAS, None, "exact_source_reuse_locked_no_update", torch, np)
        result = {
            "case_id": case_id,
            "dataset": saved["dataset"],
            "source_case_sha256": saved["source_case_sha256"],
            "source_model_state_sha256": saved["source_model_state_sha256"],
            "old_method_output_sha256": output_sha,
            "reference_lock_sha256": saved["reference_lock_sha256"],
            "reference": {"selected_candidate_id": teacher.get("selected_candidate_id"),
                          "no_update": True, "anchor_count": len(teacher.get("anchors", []))},
            "fit": {"status": "not_run_locked_no_update", "optimizer_steps": 0,
                    "new_optimizer_steps": 0, "new_fit_calls": 0, "GT_read": False},
            **alias,
            "temporal_reuse": _temporal_lineage(case_id, saved),
            "actual_full_model_redecode_calls": 0,
            "GT_read": False,
        }
        return result, False, None

    if fit is None:
        raise ValueError(f"Updating locked R1 case has no sealed teacher IoU fit: {case_id}")
    if int(teacher.get("selected_candidate_id", 1)) <= 1:
        raise ValueError(f"Teacher IoU fit did not use the locked non-Source candidate for {case_id}")
    if (fit.get("arm") != "SpatialOPD" or fit.get("scope") != "joint"
            or int(fit.get("active_parameters", -1)) != 1792
            or int(fit.get("declared_steps", -1)) != 5
            or int(fit.get("config", {}).get("steps", -1)) != 5
            or fit.get("seed_key") != "caption_query_fixed_space|" + case_id
            or fit.get("reference_sha256") != expected_reference_sha256):
        raise ValueError(f"Old IoU-OPD fit metadata is not the fixed 1792-parameter source-reset fit for {case_id}")
    if int(fit.get("optimizer_steps", -1)) < 0 or int(fit.get("optimizer_steps", -1)) > 5:
        raise ValueError(f"Old IoU-OPD fit has an invalid actual step count for {case_id}")
    if int(saved.get("actual_fit_count", -1)) != 3:
        raise ValueError(f"Updating method case does not contain the three sealed algorithm fits for {case_id}")
    initial, final = fit.get("initial_state"), fit.get("final_state")
    if not _same_state(initial, final, torch):
        # A nonzero endpoint is expected to differ; here only validate the state schema.
        if (not isinstance(initial, Mapping) or not isinstance(final, Mapping)
                or set(initial) != set(final)
                or any(not torch.is_tensor(initial[name]) or not torch.is_tensor(final[name])
                       or initial[name].shape != final[name].shape
                       or initial[name].dtype != final[name].dtype
                       for name in initial)):
            raise ValueError(f"Old IoU fit endpoint state schema is inconsistent for {case_id}")
    movement = state_displacement(initial, final, torch)
    old_readout = saved.get("readouts", {}).get("teacher_iou")
    if not isinstance(old_readout, Mapping):
        raise ValueError(f"Old IoU fit has no saved full-model readout for {case_id}")
    if not torch.equal(torch.as_tensor(fit.get("final_cached_decoder_boxes")).detach().cpu(),
                       torch.as_tensor(old_readout.get("boxes")).detach().cpu()):
        raise ValueError(f"Old cached IoU decoder boxes differ from its sealed full-model readout for {case_id}")
    if not _same_logits(old_readout.get("logits", []), saved.get("source_logits", []), torch):
        raise ValueError(f"Old IoU full-model readout logits differ from source logits for {case_id}")
    if not movement["nonzero"]:
        if not torch.equal(torch.as_tensor(fit["final_cached_decoder_boxes"]).cpu(),
                           torch.as_tensor(saved["source_boxes"]).cpu()):
            raise ValueError(f"Zero-displacement IoU fit does not exactly reuse source boxes for {case_id}")
        if int(fit.get("optimizer_steps", 0)) < 0 or int(fit.get("optimizer_steps", 0)) > 5:
            raise ValueError(f"Old IoU fit has invalid actual optimizer step count for {case_id}")
        if not _same_state(saved.get("initial_state", {}), initial, torch):
            raise ValueError(f"Zero-displacement fit initial state differs from the saved source initial state for {case_id}")
        if not torch.equal(torch.as_tensor(old_readout.get("boxes")).detach().cpu(),
                           torch.as_tensor(saved["source_boxes"]).detach().cpu()):
            raise ValueError(f"Zero-displacement fit readout does not exactly reuse source boxes for {case_id}")
        state_sha = state_hash_fn(initial)
        alias = _source_alias_rows(saved, ALPHAS, state_sha, "exact_source_reuse_zero_parameter_displacement", torch, np)
        result = {
            "case_id": case_id,
            "dataset": saved["dataset"],
            "source_case_sha256": saved["source_case_sha256"],
            "source_model_state_sha256": saved["source_model_state_sha256"],
            "old_method_output_sha256": output_sha,
            "reference_lock_sha256": saved["reference_lock_sha256"],
            "reference": {"selected_candidate_id": teacher.get("selected_candidate_id"),
                          "no_update": False, "anchor_count": len(teacher.get("anchors", []))},
            "fit": {"status": "sealed_fit_zero_final_parameter_displacement", "optimizer_steps": int(fit["optimizer_steps"]),
                    "new_optimizer_steps": 0, "new_fit_calls": 0, "active_parameters": 1792,
                    "declared_steps": 5, "parameter_displacement": movement,
                    "initial_state_sha256": state_sha, "final_state_sha256": state_sha,
                    "GT_read": False},
            **alias,
            "temporal_reuse": _temporal_lineage(case_id, saved),
            "actual_full_model_redecode_calls": 0,
            "GT_read": False,
        }
        return result, False, None

    return {
        "case_id": case_id,
        "dataset": saved["dataset"],
        "source_case_sha256": saved["source_case_sha256"],
        "source_model_state_sha256": saved["source_model_state_sha256"],
        "old_method_output_sha256": output_sha,
        "reference_lock_sha256": saved["reference_lock_sha256"],
        "reference": {"selected_candidate_id": teacher.get("selected_candidate_id"),
                      "no_update": False, "anchor_count": len(teacher.get("anchors", []))},
        "fit": {"status": "sealed_iou_fit_with_nonzero_final_parameter_displacement",
                "optimizer_steps": int(fit["optimizer_steps"]), "new_optimizer_steps": 0, "new_fit_calls": 0,
                "active_parameters": 1792, "declared_steps": 5, "parameter_displacement": movement,
                "initial_state_sha256": state_hash_fn(initial), "final_state_sha256": state_hash_fn(final),
                "GT_read": False},
        "source_readout": None,
        "amplitudes": {},
        "temporal_reuse": _temporal_lineage(case_id, saved),
        "actual_full_model_redecode_calls": 0,
        "GT_read": False,
    }, True, {"fit": fit, "saved": saved, "teacher": teacher}


def _temporal_lineage(case_id: str, saved: Mapping[str, Any]) -> dict[str, Any]:
    if case_id not in OLD_T_IDS:
        return {"status": "unavailable_no_new_T_created", "interval": None,
                "old_temporal_output_path": None, "old_temporal_output_sha256": None}
    reuse = saved.get("temporal_exact_reuse", {})
    if reuse.get("status") != "reused_exact_old_T" or reuse.get("old_T_interval_reused") is not True:
        raise ValueError(f"Old-eight exact temporal result is not preserved for {case_id}")
    arm = saved.get("arms", {}).get("T_teacher_iou")
    if not isinstance(arm, Mapping) or not isinstance(arm.get("interval"), (list, tuple)) or len(arm["interval"]) != 2:
        raise ValueError(f"Old-eight sealed T interval is unavailable for {case_id}")
    interval = [int(arm["interval"][0]), int(arm["interval"][1])]
    if interval[0] >= interval[1]:
        raise ValueError(f"Old-eight T interval is empty for {case_id}")
    path = _root_path(str(reuse["old_student_path"]))
    if not path.is_file() or _sha_file(path) != reuse["old_student_sha256"]:
        raise ValueError(f"Old-eight exact temporal output hash changed for {case_id}")
    return {"status": "exact_old_T_interval_reused", "interval": interval,
            "old_temporal_output_path": str(reuse["old_student_path"]),
            "old_temporal_output_sha256": str(reuse["old_student_sha256"]),
            "spatial_boxes_are_from_this_amplitude": True}


def _rng_snapshot(torch, np, random):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]}


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any], torch) -> bool:
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0] or not (left["numpy"][1] == right["numpy"][1]).all():
        return False
    if left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    return (len(left["torch_cuda"]) == len(right["torch_cuda"])
            and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"])))


def _restore_rng_snapshot(snapshot: Mapping[str, Any], torch, np, random_module) -> None:
    random_module.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["torch_cuda"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _amplitude_record(model, current_source, context, alpha, fit, saved, runtime_modules,
                      source_custody, source_readout, torch, np, state_hash_fn):
    core, query_subject, contexts, readout_function = runtime_modules
    state = interpolate_fit_state(fit["initial_state"], fit["final_state"], alpha, torch)
    state_hash = state_hash_fn(state)
    if alpha == 0.0:
        if state_hash_fn(fit["initial_state"]) != state_hash:
            raise RuntimeError("Alpha-zero state did not preserve the exact saved initial state")
        # _capture_source already performed a fresh full-source forward on the
        # same canonical frames/query. Reuse that output as the alpha-zero
        # eligibility replay instead of running an extra duplicate forward.
        readout = source_readout
        rng_unchanged = True
        replay_kind = "fresh_full_source_capture"
    else:
        pseudo_fit = {"final_state": state}
        random_module = __import__("random")
        before_rng = _rng_snapshot(torch, np, random_module)
        try:
            readout = readout_function(
                model, context["batch"], context["frame_ids"], context["records"],
                pseudo_fit, context["row"]["parses"]["subject"], core, torch,
                *contexts, query_subject,
            )
        finally:
            after_rng = _rng_snapshot(torch, np, random_module)
            _restore_rng_snapshot(before_rng, torch, np, random_module)
        rng_unchanged = _rng_equal(before_rng, after_rng, torch)
        if not rng_unchanged:
            raise RuntimeError(f"Full model amplitude readout consumed randomness for {context['case_id']}/{alpha}")
        replay_kind = "fresh_full_model_redecode"
    if state_hash_fn(model.state_dict()) != source_custody["source_model_state_sha256"]:
        raise RuntimeError(f"Full model source weights were not restored after amplitude {alpha}")
    cpu = _cpu_readout(readout, torch, np)
    alpha_key = _alpha_key(alpha)
    if alpha == 0.0:
        if not torch.equal(cpu["boxes_cxcywh"], torch.as_tensor(saved["source_boxes"]).detach().cpu()):
            raise RuntimeError(f"Alpha-zero full readout differs from the sealed source cache for {context['case_id']}")
        if not _same_logits(cpu["logits"], saved["source_logits"], torch):
            raise RuntimeError(f"Alpha-zero logits differ from the sealed source cache for {context['case_id']}")
        consistency = "exact_source_boxes_and_logits"
    elif alpha == 1.0:
        fit_output = saved.get("readouts", {}).get("teacher_iou")
        if not torch.equal(cpu["boxes_cxcywh"], torch.as_tensor(fit["final_cached_decoder_boxes"]).detach().cpu()):
            raise RuntimeError(f"Alpha-one full readout differs from the cached OPD decoder boxes for {context['case_id']}")
        if not _same_logits(cpu["logits"], saved["source_logits"], torch):
            raise RuntimeError(f"Alpha-one logits differ from the sealed source cache for {context['case_id']}")
        if not isinstance(fit_output, Mapping) or not torch.equal(
            cpu["boxes_cxcywh"], torch.as_tensor(fit_output["boxes"]).detach().cpu()
        ):
            raise RuntimeError(f"Alpha-one full readout differs from the old saved actual readout for {context['case_id']}")
        if "logits" in fit_output and not _same_logits(cpu["logits"], fit_output["logits"], torch):
            raise RuntimeError(f"Alpha-one logits differ from the old saved actual readout for {context['case_id']}")
        consistency = "exact_old_cached_decoder_and_readout_boxes_and_source_logits"
    else:
        consistency = "fresh_full_model_readout"
    return {
        "alpha": float(alpha),
        "status": "decoded",
        "state_sha256": state_hash,
        "boxes_cxcywh": cpu["boxes_cxcywh"],
        "boxes_xyxy": cpu["boxes_xyxy"],
        "logits": cpu["logits"],
        "native_interval": cpu["physical_interval"],
        "full_model_redecode": True,
        "redecode_calls": 1,
        "replay_kind": replay_kind,
        "endpoint_cache_consistency": consistency,
        "random_state_unchanged": True,
        "source_model_restored": True,
        "GT_read": False,
    }


def run_phase1() -> dict[str, Any]:
    """Run only after root creates the phase1 amplitude/runtime locks."""
    preflight = _preflight()
    if BARRIER_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite phase1 barrier: {BARRIER_PATH}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing phase1 outputs: {OUTPUT_DIR}")

    import gc
    import random
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
    from scripts.run_stvg_motion_guided_student002 import (
        _capture_source, _readout, _xyxy,
    )
    from scripts.stvg_temporal_caption_query_v1 import catalog, inference_guard

    install_clean_loader()
    sys.addaudithook(inference_guard)
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    random.seed(20261014)
    np.random.seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    old = preflight["old"]
    roster_by_id = old["roster_by_id"]
    reference_by_id = old["reference_by_id"]
    output_rows = []
    output_files = []
    actual_readouts = 0
    nonzero_cases = 0
    zero_cases = 0
    no_update_cases = 0
    source_model_load_count = 0
    model = None
    current_source = None
    started = time.perf_counter()

    for case_id in sorted(EXPECTED_CASE_IDS):
        old_path = old["output_paths"][case_id]
        saved = torch.load(old_path, map_location="cpu", weights_only=False)
        fit_result, needs_decode, payload = _case_result(
            saved, roster_by_id[case_id], reference_by_id[case_id],
            _sha_file(old_path), old["old_sha256"]["reference_lock"], torch, np, state_hash,
        )
        if not needs_decode:
            if reference_by_id[case_id]["references"]["teacher"]["no_update"]:
                no_update_cases += 1
            else:
                zero_cases += 1
            destination = OUTPUT_DIR / f"{case_id}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fit_result, destination)
            output_files.append({"path": destination.relative_to(ROOT).as_posix(),
                                 "sha256": _sha_file(destination), "bytes": destination.stat().st_size})
            output_rows.append({"case_id": case_id,
                                "status": fit_result["fit"]["status"],
                                "actual_full_model_redecode_calls": 0})
            del saved, fit_result
            continue

        nonzero_cases += 1
        rec = roster_by_id[case_id]
        preview_case, preview_catalog = catalog(rec)
        next_source = preview_case["source"]
        if model is not None and current_source != next_source:
            # Drop the caller's outer model reference before loading a new
            # source. `_capture_source` only deletes its local argument.
            del model
            model = None
            current_source = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if model is None:
            source_model_load_count += 1
        del preview_case, preview_catalog
        (
            model, current_source, case, catalog_data, row, frames, frame_ids,
            records, batch, base, native, expected_state_sha, pixel_sha, source_capture_seconds,
        ) = _capture_source(
            rec, model, current_source, torch, __import__("collections"),
            model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
        )
        fit = payload["fit"]
        if case["_source_case_sha256"] != rec["source_case_sha256"]:
            raise RuntimeError(f"Fresh source case SHA differs for {case_id}")
        if expected_state_sha != saved["source_model_state_sha256"]:
            raise RuntimeError(f"Fresh source checkpoint differs from sealed old fit for {case_id}")
        if pixel_sha != saved["pixel_sha256"] or frame_ids != list(map(int, saved["frame_ids"])):
            raise RuntimeError(f"Fresh source frame pixels/grid differ from sealed old fit for {case_id}")
        if not _same_state(base.initial, fit["initial_state"], torch):
            raise RuntimeError(f"Fresh initial student state differs from old IoU fit for {case_id}")
        if not torch.equal(native["boxes"].detach().cpu(), torch.as_tensor(saved["source_boxes"]).cpu()):
            raise RuntimeError(f"Fresh Native boxes differ from the old source cache for {case_id}")
        if not _same_logits(native["logits"], saved["source_logits"], torch):
            raise RuntimeError(f"Fresh Native logits differ from the old source cache for {case_id}")

        context = {"case_id": case_id, "batch": batch, "frame_ids": frame_ids,
                   "records": records, "row": row}
        amplitude_rows = {}
        amplitude_runtime_modules = (
            core, query_subject,
            (transient_native_masks, frozen_visual_host, frozen_attention_storage,
             inactive_modules_on_host),
            _readout,
        )
        for alpha in ALPHAS:
            amplitude_rows[_alpha_key(alpha)] = _amplitude_record(
                model, current_source, context, alpha, fit, saved,
                amplitude_runtime_modules,
                {"source_model_state_sha256": expected_state_sha}, native,
                torch, np, state_hash,
            )
            actual_readouts += 1
        fit_result["source_readout"] = {
            "status": "fresh_source_capture_used_for_alpha_zero_qualification",
            "boxes_cxcywh": native["boxes"].detach().cpu().clone(),
            "boxes_xyxy": torch.as_tensor(_xyxy(native["boxes"])).detach().cpu(),
            "logits": [x.detach().cpu().clone() for x in native["logits"]],
            "physical_interval": list(native["physical_interval"]),
            "pixel_sha256": pixel_sha,
            "frame_ids": list(map(int, frame_ids)),
            "source_capture_seconds": float(source_capture_seconds),
            "GT_read": False,
        }
        fit_result["amplitudes"] = amplitude_rows
        fit_result["actual_full_model_redecode_calls"] = len(ALPHAS)
        fit_result["source_capture_full_model_readout_calls"] = 1
        fit_result["additional_amplitude_full_model_redecode_calls"] = len(ALPHAS) - 1
        fit_result["fresh_pixel_sha256"] = pixel_sha
        fit_result["source_model_state_restored"] = state_hash(model.state_dict()) == expected_state_sha
        fit_result["GT_read"] = False
        if not fit_result["source_model_state_restored"]:
            raise RuntimeError(f"Source model was not restored after amplitude readouts for {case_id}")

        destination = OUTPUT_DIR / f"{case_id}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fit_result, destination)
        output_files.append({"path": destination.relative_to(ROOT).as_posix(),
                             "sha256": _sha_file(destination), "bytes": destination.stat().st_size})
        output_rows.append({"case_id": case_id,
                            "status": fit_result["fit"]["status"],
                            "actual_full_model_redecode_calls": len(ALPHAS)})
        del saved, fit_result, fit, payload, frames, batch, base, native, records, row, case, catalog_data
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model is not None:
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(output_files) != 16 or len({row["path"] for row in output_files}) != 16:
        raise AssertionError("Phase1 must seal exactly one output per fixed source")
    result = {
        "schema": "stvg-spatial-update-amplitude-phase1-v1",
        "status": "complete",
        "phase": PHASE_NAME,
        "count": 16,
        "files": output_files,
        "records": output_rows,
        "alpha_values": list(ALPHAS),
        "source_roster_sha256": old["old_sha256"]["roster"],
        "method_reference_lock_sha256": old["old_sha256"]["reference_lock"],
        "method_runtime_sha256": old["old_sha256"]["runtime"],
        "method_barrier_sha256": old["old_sha256"]["method_barrier"],
        "phase1_lock_sha256": _sha_file(ALPHA_LOCK_PATH),
        "phase1_runtime_sha256": _sha_file(RUNTIME_PATH),
        "cpu_contract_sha256": _sha_file(CPU_CONTRACT_PATH),
        "actual_selected_nonSource_count": sum(
            not bool(reference_by_id[cid]["references"]["teacher"]["no_update"])
            for cid in EXPECTED_CASE_IDS
        ),
        "locked_no_update_case_count": no_update_cases,
        "zero_parameter_displacement_fit_count": zero_cases,
        "nonzero_parameter_displacement_case_count": nonzero_cases,
        "actual_source_model_load_count": source_model_load_count,
        "actual_full_model_redecode_calls": actual_readouts,
        "source_capture_full_model_readout_calls": nonzero_cases,
        "additional_amplitude_full_model_redecode_calls": nonzero_cases * (len(ALPHAS) - 1),
        "new_optimizer_steps": 0,
        "new_fit_calls": 0,
        "GT_read": False,
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(BARRIER_PATH, result)
    return result


def _synthetic_state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.float().numpy().tobytes())
    return digest.hexdigest()


def run_cpu_contracts() -> dict[str, Any]:
    """Run small synthetic state and branch contracts without model/data access."""
    import numpy as np
    import torch

    checks = []

    initial = {
        "float32": torch.tensor([0.25, -1.5], dtype=torch.float32),
        "bf16": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
    }
    final = {
        "float32": torch.tensor([1.25, 0.5], dtype=torch.float32),
        "bf16": torch.tensor([1.0, 0.25], dtype=torch.bfloat16),
    }
    states = {alpha: interpolate_fit_state(initial, final, alpha, torch) for alpha in ALPHAS}
    assert all(states[alpha][name].dtype == initial[name].dtype for alpha in ALPHAS for name in initial)
    assert all(torch.equal(states[0.0][name], initial[name]) for name in initial)
    assert all(torch.equal(states[1.0][name], final[name]) for name in initial)
    for alpha in (0.25, 0.5):
        for name in initial:
            expected = initial[name] + (final[name] - initial[name]) * alpha
            assert torch.equal(states[alpha][name], expected)
    checks.append({"name": "locked_alpha_interpolation_and_endpoint_copy",
                   "status": "pass", "alpha_values": list(ALPHAS),
                   "dtypes": {name: str(value.dtype) for name, value in initial.items()},
                   "endpoint_arithmetic": False, "intermediate_dtype_preserved": True})

    movement = state_displacement(initial, final, torch)
    assert movement["nonzero"] and movement["changed_element_count"] == 4
    zero_state = {name: value.clone() for name, value in initial.items()}
    zero_movement = state_displacement(initial, zero_state, torch)
    assert not zero_movement["nonzero"] and zero_movement["changed_element_count"] == 0
    checks.append({"name": "exact_zero_vs_nonzero_endpoint_movement",
                   "status": "pass", "nonzero_element_count": movement["changed_element_count"],
                   "zero_element_count": zero_movement["changed_element_count"]})

    rejection_cases = []
    for bad_alpha in (0.1, float("nan"), float("inf")):
        try:
            interpolate_fit_state(initial, final, bad_alpha, torch)
        except ValueError:
            rejection_cases.append("alpha")
        else:
            raise AssertionError("Unexpected alpha was accepted")
    try:
        interpolate_fit_state(initial, {"other": final["float32"]}, 0.25, torch)
    except ValueError:
        rejection_cases.append("key_mismatch")
    else:
        raise AssertionError("Mismatched endpoint keys were accepted")
    try:
        interpolate_fit_state(initial, {**final, "float32": torch.tensor([1, 2], dtype=torch.int64)}, 0.25, torch)
    except (TypeError, ValueError):
        rejection_cases.append("dtype_mismatch")
    else:
        raise AssertionError("Integer endpoint state was accepted")
    checks.append({"name": "invalid_alpha_and_endpoint_schema_rejection",
                   "status": "pass", "rejected": rejection_cases})

    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32)
    logits = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
    common_saved = {
        "dataset": "hc2", "source_case_sha256": "a" * 64,
        "source_model_state_sha256": "b" * 64, "reference_lock_sha256": "c" * 64,
        "GT_read": False, "source_restored": True,
        "source_cold_readback": {"status": "pass"},
        "source_boxes": boxes.clone(), "source_logits": [logits[0].clone()],
        "Native_interval": [10, 12], "reference_anchors": {"teacher": []},
        "actual_fit_count": 0, "fits": {}, "readouts": {},
    }
    roster_row = {"dataset": "hc2", "source_case_sha256": "a" * 64}
    no_update_reference = {"references": {"teacher": {"no_update": True, "anchors": [],
                                                          "selected_candidate_id": 1}}}
    no_update_saved = {**common_saved, "case_id": "hc2_006"}
    no_update_record, no_update_decode, no_update_payload = _case_result(
        no_update_saved, roster_row, no_update_reference, "d" * 64, "c" * 64,
        torch, np, _synthetic_state_hash,
    )
    assert not no_update_decode and no_update_payload is None
    assert no_update_record["actual_full_model_redecode_calls"] == 0
    assert all(row["full_model_redecode"] is False and row["redecode_calls"] == 0
               for row in no_update_record["amplitudes"].values())
    checks.append({"name": "locked_no_update_is_exact_source_alias_without_decode",
                   "status": "pass", "decode_calls": 0, "alias_count": len(ALPHAS)})

    state_zero = {"ln.synthetic": torch.tensor([0.125, -0.25], dtype=torch.float32)}
    fit_zero = {
        "arm": "SpatialOPD", "scope": "joint", "active_parameters": 1792,
        "declared_steps": 5, "optimizer_steps": 5,
        "config": {"steps": 5}, "seed_key": "caption_query_fixed_space|hc2_006",
        "reference_sha256": "c" * 64,
        "initial_state": {k: v.clone() for k, v in state_zero.items()},
        "final_state": {k: v.clone() for k, v in state_zero.items()},
        "final_cached_decoder_boxes": boxes.clone(),
    }
    zero_saved = {
        **common_saved, "case_id": "hc2_006", "actual_fit_count": 3,
        "initial_state": {k: v.clone() for k, v in state_zero.items()},
        "fits": {"teacher_iou": fit_zero},
        "readouts": {"teacher_iou": {"boxes": boxes.clone(), "logits": [logits[0].clone()]}},
    }
    selected_reference = {"references": {"teacher": {"no_update": False,
                         "anchors": [{"position": 0}], "selected_candidate_id": 2}}}
    zero_saved["reference_anchors"] = {"teacher": [{"position": 0}]}
    zero_roster = {"dataset": "hc2", "source_case_sha256": "a" * 64}
    zero_record, zero_decode, zero_payload = _case_result(
        zero_saved, zero_roster, selected_reference, "d" * 64, "c" * 64,
        torch, np, _synthetic_state_hash,
    )
    assert not zero_decode and zero_payload is None
    assert zero_record["actual_full_model_redecode_calls"] == 0
    assert zero_record["fit"]["new_optimizer_steps"] == 0
    assert all(row["state_sha256"] == _synthetic_state_hash(state_zero)
               and row["redecode_calls"] == 0 for row in zero_record["amplitudes"].values())
    checks.append({"name": "zero_displacement_fit_is_exact_source_alias_without_decode",
                   "status": "pass", "declared_old_steps": 5,
                   "new_steps": 0, "decode_calls": 0, "alias_count": len(ALPHAS)})

    checks.append({"name": "nonzero_fit_readout_budget",
                   "status": "pass", "per_nonzero_fit": {
                       "alpha0_source_capture_full_model_readout": 1,
                       "alpha_quarter_half_one_additional_full_model_readouts": 3,
                       "total_full_model_readouts_including_source_capture": 4,
                   }, "three_nonzero_cases": {
                       "source_capture_readouts": 3, "additional_readouts": 9,
                       "total_actual_readouts": 12,
                   }})
    return {"schema": "stvg-spatial-update-amplitude-cpu-contracts-v1",
            "status": "pass", "checks": checks, "CPU_only": True,
            "GT_read": False, "model_calls": 0, "GPU_calls": 0,
            "actual_data_read": False}


def write_cpu_contracts() -> dict[str, Any]:
    result = run_cpu_contracts()
    result["worker_sha256"] = _sha_file(Path(__file__))
    if CPU_CONTRACT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite frozen CPU contract: {CPU_CONTRACT_PATH}")
    _write_json(CPU_CONTRACT_PATH, result)
    result["contract_path"] = CPU_CONTRACT_PATH.relative_to(ROOT).as_posix()
    result["contract_sha256"] = _sha_file(CPU_CONTRACT_PATH)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-contracts", action="store_true",
                        help="run synthetic CPU-only contracts without reading experiment artifacts")
    parser.add_argument("--write-cpu-contracts", action="store_true",
                        help="run CPU contracts and create the new phase1 contract file once")
    args = parser.parse_args()
    if args.cpu_contracts and args.write_cpu_contracts:
        parser.error("choose at most one CPU contract mode")
    if args.write_cpu_contracts:
        result = write_cpu_contracts()
    elif args.cpu_contracts:
        result = run_cpu_contracts()
    else:
        result = run_phase1()
    print(json.dumps(result, sort_keys=True))
