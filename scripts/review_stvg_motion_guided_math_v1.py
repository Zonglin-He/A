"""Independent CPU arithmetic readback for locked motion-guided student phases.

This reader opens only sealed, GT-free source cases and completed student
barriers. It never creates a model, initializes CUDA, decodes pixels, or reads
teacher response prose. Each phase is refused until its global barrier and all
file hashes are complete.
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
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
OLD_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
WORKER_CANDIDATES = {
    "scripts/run_stvg_motion_guided_student_v1.py",
    "scripts/run_stvg_motion_guided_student002.py",
}
FIT_HELPER = "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py"
ALGORITHMS = ("direct", "iou", "diou")
METHOD_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(8)
}
OLD_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(4)
}
CONTROL_ID = "vidstg_002"
CONTROL_ROLES = ("Source", "oldR1ID2", "oldAlternativeID3")

# Reuse the independently implemented CPU math and tensor comparison routines
# from the preceding scope. This import defines helpers only; its CLI is inert.
from scripts.review_stvg_scene_object_id_spatial_math_v1 import (  # noqa: E402
    _anchors_from_case,
    _audit_direct_fit,
    _audit_opd_fit,
    _compare_tree,
    _require,
    _safe_root_path,
    _sha_file,
    _verified_static_helper_bindings,
    _verify_digest_path,
    _xyxy,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(left: Any, right: Any, name: str) -> None:
    """Strict equality for metadata; exact tensor equality for numeric state."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=False)):
            raise RuntimeError(f"{name}: arrays are not exactly equal")
        return
    if torch.is_tensor(left) or torch.is_tensor(right):
        _compare_tree(left, right, name=name)
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)):
            raise RuntimeError(f"{name}: mapping schema differs")
        for key in left:
            _equal(left[key], right[key], f"{name}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (type(left) is type(right) and len(left) == len(right)):
            raise RuntimeError(f"{name}: sequence schema differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f"{name}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"{name}: values differ ({left!r} != {right!r})")


def _runtime_paths(phase: str) -> list[Path]:
    stem = f"{phase.upper()}_STUDENT_RUNTIME"
    return sorted(BASE.glob(f"{stem}*.json"))


def _phase_lock_path(phase: str) -> Path:
    if phase == "method":
        return BASE / "METHOD_REFERENCE_LOCK.json"
    if phase == "reference_control":
        return BASE / "REFERENCE_CONTROL_REFERENCE_LOCK.json"
    raise ValueError(f"unsupported phase: {phase}")


def _barrier_path(phase: str) -> Path:
    return BASE / f"{phase.upper()}_STUDENT_GLOBAL_BARRIER.json"


def _expected_fit_count(phase: str) -> int:
    return 18 if phase == "method" else 3


def _expected_case_ids(phase: str) -> set[str]:
    return METHOD_IDS if phase == "method" else {CONTROL_ID}


def _select_runtime(phase: str, barrier: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    target = barrier.get("runtime_sha256")
    _require(isinstance(target, str) and len(target) == 64,
             "complete student barrier lacks the runtime SHA")
    matches = []
    for path in _runtime_paths(phase):
        if _sha_file(path) == target:
            matches.append((path, _read_json(path)))
    _require(len(matches) == 1,
             "barrier runtime SHA must resolve to exactly one phase runtime (old or dispatch002)")
    return matches[0]


def _preflight(phase: str) -> dict[str, Any]:
    if phase not in ("method", "reference_control"):
        raise ValueError("phase must be method or reference_control")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU readback refused because CUDA is already initialized")

    barrier_path = _barrier_path(phase)
    _require(barrier_path.is_file(),
             "phase global barrier is absent; partial outputs will not be read")
    barrier = _read_json(barrier_path)
    expected_count = 16 if phase == "method" else 1
    expected_fits = _expected_fit_count(phase)
    expected_readouts = expected_fits
    _require(barrier.get("status") == "complete"
             and barrier.get("count") == expected_count
             and barrier.get("actual_fit_count") == expected_fits
             and barrier.get("actual_final_redecode_calls") == expected_readouts
             and barrier.get("GT_read") is False,
             "student barrier is incomplete or its phase counts/GT flag differ")

    runtime_path, runtime = _select_runtime(phase, barrier)
    reference_path = _phase_lock_path(phase)
    roster_path = BASE / "ROSTER_LOCK.json"
    _require(runtime.get("status") == "locked"
             and runtime.get("phase") == phase
             and runtime.get("GT_read") is False
             and runtime.get("steps") == 5
             and runtime.get("scope") == "joint"
             and runtime.get("expected_fit_count") == expected_fits,
             "barrier-bound runtime is not the expected locked five-step phase")
    _require(runtime.get("reference_lock_sha256") == _sha_file(reference_path)
             and runtime.get("roster_lock_sha256") == _sha_file(roster_path),
             "runtime does not bind the current phase reference and roster locks")
    _require(barrier.get("reference_lock_sha256") == _sha_file(reference_path)
             and barrier.get("roster_lock_sha256") == _sha_file(roster_path),
             "student barrier does not bind current reference and roster locks")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    _require(isinstance(pins, dict) and pins
             and isinstance(protected, dict) and protected,
             "phase runtime must contain source pins and protected files")
    ref_rel = str(reference_path.relative_to(ROOT))
    _require(pins.get(ref_rel) == _sha_file(reference_path),
             "phase reference lock is not pinned by the barrier-bound runtime")
    helper_digest = pins.get(FIT_HELPER)
    _require(isinstance(helper_digest, str), "runtime omits the frozen spatial fit helper")
    _verify_digest_path(FIT_HELPER, helper_digest)
    transitive = _verified_static_helper_bindings(ROOT / FIT_HELPER)
    for relative, digest in transitive.items():
        _require(relative not in pins or pins[relative] == digest,
                 f"runtime and fit-helper SOURCE_BINDINGS disagree: {relative}")
    effective_pins = {**transitive, **pins}
    if runtime_path.stem.endswith("RUNTIME002"):
        worker_path = "scripts/run_stvg_motion_guided_student002.py"
    else:
        worker_path = "scripts/run_stvg_motion_guided_student_v1.py"
    _require(worker_path in effective_pins,
             "barrier-bound runtime omits the worker for its selected dispatch version")
    for relative, digest in effective_pins.items():
        _verify_digest_path(relative, digest)
    for relative, digest in protected.items():
        _verify_digest_path(relative, digest)

    roster_doc = _read_json(roster_path)
    _require(roster_doc.get("status") == "locked"
             and roster_doc.get("GT_read") is False,
             "roster lock is not a locked GT-free roster")
    roster_rows = roster_doc.get("records")
    _require(isinstance(roster_rows, list) and len(roster_rows) == 16,
             "roster lock does not contain all sixteen fixed source cases")
    roster_by_id = {row.get("case_id"): row for row in roster_rows}
    _require(set(roster_by_id) == METHOD_IDS,
             "roster lock case IDs differ from the fixed 16-case cohort")
    ref_doc = _read_json(reference_path)
    _require(ref_doc.get("status") == "locked"
             and ref_doc.get("GT_read") is False
             and ref_doc.get("phase") == phase,
             "phase reference lock is not locked and GT-free")
    reference_rows = ref_doc.get("records")
    expected_ids = _expected_case_ids(phase)
    _require(isinstance(reference_rows, list)
             and {row.get("case_id") for row in reference_rows} == expected_ids
             and len(reference_rows) == len(expected_ids),
             "phase reference lock does not contain the exact expected case IDs")
    reference_by_id = {row["case_id"]: row for row in reference_rows}
    for cid, row in reference_by_id.items():
        source_row = roster_by_id[cid]
        _require(row.get("dataset") == source_row.get("dataset")
                 and row.get("source_case_sha256") == source_row.get("source_case_sha256"),
                 f"reference lock source binding differs from roster for {cid}")
        refs = row.get("references")
        _require(isinstance(refs, Mapping), f"{cid} lacks reference map")
        if phase == "method":
            _require(set(refs) == {"teacher"}
                     and isinstance(refs["teacher"].get("no_update"), bool),
                     f"{cid} has unexpected method reference roles or no_update flag")
            anchors = refs["teacher"].get("anchors", [])
            _require(isinstance(anchors, list) and len(anchors) <= 4,
                     f"{cid} has malformed teacher anchors")
        else:
            _require(cid == CONTROL_ID and set(refs) == set(CONTROL_ROLES),
                     "reference control must contain exactly the three locked roles")
            for role in CONTROL_ROLES:
                _require(isinstance(refs[role].get("anchors"), list)
                         and refs[role]["anchors"],
                         f"reference-control role has no anchors: {role}")

    # Every source case used for independent state comparison must be pinned,
    # and its sidecar must bind the same no-GT bytes.
    source_inputs = {}
    for cid, row in roster_by_id.items():
        relative = _safe_root_path(row["source_case_path"])
        digest = row.get("source_case_sha256")
        _require(pins.get(relative) == digest,
                 f"source-case payload is not pinned to roster SHA for {cid}")
        sidecar = str(Path(relative).with_suffix(".json"))
        _require(sidecar in pins, f"source-case sidecar is not runtime-pinned for {cid}")
        _verify_digest_path(relative, digest)
        source_inputs[cid] = ROOT / relative

    files = barrier.get("files")
    _require(isinstance(files, list) and len(files) == expected_count,
             "global barrier output inventory has the wrong size")
    prefix = f"{BASE.relative_to(ROOT)}/{phase}/student/"
    case_paths = {}
    for row in files:
        _require(isinstance(row, dict), "malformed barrier output row")
        relative = _safe_root_path(row.get("path"))
        _require(relative.startswith(prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(prefix):],
                 "student output path escaped its phase case directory")
        cid = relative[len(prefix):-3]
        _require(cid not in case_paths, "barrier repeats a case output")
        case_paths[cid] = _verify_digest_path(relative, row.get("sha256"), row.get("bytes"))
    _require(set(case_paths) == expected_ids,
             "barrier does not seal exactly the expected phase case outputs")

    old_barrier_path = OLD_BASE / "STUDENT_GLOBAL_BARRIER.json"
    old_barrier_rel = str(old_barrier_path.relative_to(ROOT))
    _require(pins.get(old_barrier_rel) == _sha_file(old_barrier_path),
             "old temporal student barrier is not pinned")
    old_barrier = _read_json(old_barrier_path)
    _require(old_barrier.get("status") == "complete"
             and old_barrier.get("count") == 8
             and old_barrier.get("GT_read") is False,
             "old temporal student barrier is incomplete or not GT-free")
    old_case_paths = {}
    for row in old_barrier.get("files", []):
        relative = _safe_root_path(row.get("path"))
        cid = Path(relative).stem
        _require(pins.get(relative) == row.get("sha256"),
                 f"old temporal output is not runtime-pinned: {relative}")
        old_case_paths[cid] = _verify_digest_path(relative, row["sha256"], row.get("bytes"))
    _require(set(old_case_paths) == OLD_IDS,
             "old temporal barrier does not seal the exact original eight cases")

    return {
        "phase": phase,
        "runtime": runtime,
        "runtime_path": runtime_path,
        "reference_doc": ref_doc,
        "reference_path": reference_path,
        "reference_by_id": reference_by_id,
        "roster_by_id": roster_by_id,
        "source_inputs": source_inputs,
        "case_paths": case_paths,
        "old_case_paths": old_case_paths,
        "barrier": barrier,
        "barrier_path": barrier_path,
        "old_barrier": old_barrier,
        "worker_path": worker_path,
        "transitive_bindings": transitive,
        "expected_cases": expected_count,
        "expected_fits": expected_fits,
        "declared_fit_steps": expected_fits * 5,
        "hashes": {
            "runtime": _sha_file(runtime_path),
            "reference_lock": _sha_file(reference_path),
            "roster_lock": _sha_file(roster_path),
            "student_barrier": _sha_file(barrier_path),
            "old_temporal_barrier": _sha_file(old_barrier_path),
        },
    }


def _load_source_case(path: Path) -> dict[str, Any]:
    """Load a previously sealed source-only case on CPU; return no prose."""
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    return load_source_case(path)


def _validate_source_case(case: Mapping[str, Any], roster: Mapping[str, Any],
                          output: Mapping[str, Any]) -> dict[str, Any]:
    cid = roster["case_id"]
    _require(case.get("_source_case_sha256") == roster["source_case_sha256"],
             f"{cid}: original source-case digest mismatch")
    _require(case.get("GT_read") is False,
             f"{cid}: source case is not explicitly GT-free")
    ids = list(map(int, case["frame_ids"]))
    _require(ids == list(map(int, output["frame_ids"])),
             f"{cid}: source physical frame IDs differ")
    _require(case.get("pixel_sha256") == output.get("pixel_sha256"),
             f"{cid}: decoded source pixel SHA differs")
    _require(case.get("source_model_state_sha256") == output.get("source_model_state_sha256"),
             f"{cid}: source model-state SHA differs")
    _equal(case["zero"]["boxes"], output["source_boxes"], f"{cid}.zero_boxes")
    _equal(case["zero"]["logits"], output["source_logits"], f"{cid}.zero_logits")
    source_initial = case["source_light_initial_state"]
    _equal(source_initial, output["initial_state"], f"{cid}.source_initial_state")
    _require(output.get("source_restored") is True,
             f"{cid}: source model restoration is not receipted")
    cold = output.get("source_cold_readback", {})
    _require(cold.get("status") == "pass"
             and cold.get("source_case_sha256") == roster["source_case_sha256"]
             and cold.get("source_model_state_sha256") == case["source_model_state_sha256"]
             and cold.get("pixel_sha256") == case["pixel_sha256"]
             and cold.get("frame_ids_match") is True
             and cold.get("initial_state_match") is True
             and cold.get("zero_boxes_logits_match") is True
             and cold.get("source_restored") is True,
             f"{cid}: source-cold readback receipt is inconsistent")
    bounds = list(map(float, case["video_bounds"]))
    return {
        "source_case_sha256": roster["source_case_sha256"],
        "source_model_state_sha256": case["source_model_state_sha256"],
        "frame_count": len(ids),
        "frame_ids_sha256": _sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
        "pixel_sha256": case["pixel_sha256"],
        "video_bounds": bounds,
    }


def _assert_interval(interval: Any, bounds: list[float], name: str) -> None:
    _require(isinstance(interval, (list, tuple)) and len(interval) == 2,
             f"{name}: interval schema differs")
    left, right = map(float, interval)
    _require(math.isfinite(left) and math.isfinite(right)
             and bounds[0] <= left < right <= bounds[1],
             f"{name}: interval is nonfinite, empty, or outside source bounds")


def _assert_old_temporal_reuse(output: Mapping[str, Any], old: Mapping[str, Any],
                               cid: str, roster: Mapping[str, Any],
                               bounds: list[float]) -> dict[str, Any]:
    _require(old.get("case_id") == cid and old.get("GT_read") is False
             and old.get("source_case_sha256") == roster["source_case_sha256"],
             f"{cid}: pinned old temporal student belongs to another source")
    # The prior temporal artifact kept the decoded pixel digest inside its
    # source-cold receipt, while the motion-guided artifact promotes it to the
    # top level. Compare the underlying receipts instead of assuming a newer
    # serialization schema for the pinned old output.
    old_cold = old.get("source_cold_readback", {})
    _require(list(map(int, old["frame_ids"])) == list(map(int, output["frame_ids"]))
             and old_cold.get("status") == "pass"
             and old_cold.get("pixel_sha256") == output.get("pixel_sha256"),
             f"{cid}: old temporal pixels or frames differ from the new source readback")
    old_arms, arms = old["arms"], output["arms"]
    for name in ("F", "Projection", "T"):
        _equal(torch.as_tensor(arms[name]["boxes"], dtype=torch.float64),
               torch.as_tensor(old_arms[name]["boxes"], dtype=torch.float64),
               f"{cid}.old_exact_{name}_boxes")
        _equal(arms[name]["interval"], old_arms[name]["interval"],
               f"{cid}.old_exact_{name}_interval")
        _assert_interval(arms[name]["interval"], bounds, f"{cid}.{name}")
    return {
        "status": "exact_old_student_F_Projection_T_reuse",
        "old_student_sha256": _sha_file(OLD_BASE / "student" / f"{cid}.pt"),
        "old_T_boxes_equal_F": True,
        "old_T_interval": list(old_arms["T"]["interval"]),
    }


def _validate_source_arms(output: Mapping[str, Any], cid: str,
                          source: Mapping[str, Any], bounds: list[float]) -> None:
    arms = output.get("arms")
    _require(isinstance(arms, Mapping), f"{cid}: missing result arms")
    _equal(output.get("Native_interval"), source.get("Native_interval"),
           f"{cid}.Native_interval_vs_original_case")
    _equal(output.get("Projection_interval"), source.get("Projection_interval"),
           f"{cid}.Projection_interval_vs_original_case")
    boxes_cxcywh = torch.as_tensor(output["source_boxes"], dtype=torch.float64)
    expected_xyxy = _xyxy(boxes_cxcywh)
    _equal(torch.as_tensor(arms["F"]["boxes"], dtype=torch.float64), expected_xyxy,
           f"{cid}.F_vs_native_source")
    _equal(arms["F"]["interval"], output["Native_interval"], f"{cid}.F_interval_vs_Native")
    _equal(torch.as_tensor(arms["Projection"]["boxes"], dtype=torch.float64), expected_xyxy,
           f"{cid}.Projection_boxes_vs_native_source")
    _equal(arms["Projection"]["interval"], output["Projection_interval"],
           f"{cid}.Projection_interval_vs_source")
    _assert_interval(output["Native_interval"], bounds, f"{cid}.Native")
    _assert_interval(output["Projection_interval"], bounds, f"{cid}.Projection")


def _check_no_update(case: Mapping[str, Any], record: Mapping[str, Any],
                     phase: str, old: Mapping[str, Any] | None) -> dict[str, Any]:
    cid = record["case_id"]
    _require(phase == "method", "explicit no_update reuse is only valid in method phase")
    teacher = record["references"]["teacher"]
    _require(teacher.get("no_update") is True,
             f"{cid}: no-update output has an updating teacher lock")
    fits, readouts = case.get("fits"), case.get("readouts")
    _require(fits == {} and readouts == {}
             and case.get("actual_fit_count") == 0
             and case.get("optimizer_steps") == 0
             and case.get("actual_final_redecode_calls") == 0,
             f"{cid}: no_update case contains a fit, readout, or nonzero count")
    arms = case["arms"]
    receipts = case.get("reference_receipts", {})
    for algorithm in ALGORITHMS:
        name = f"teacher_{algorithm}"
        _equal(arms[name]["boxes"], arms["F"]["boxes"], f"{cid}.{name}_exact_F_boxes")
        _equal(arms[name]["interval"], arms["F"]["interval"], f"{cid}.{name}_exact_F_interval")
        receipt = receipts.get(name, {})
        _require(receipt.get("status") == "explicit_no_update"
                 and receipt.get("lock_no_update") is True
                 and receipt.get("optimizer_steps") == 0
                 and receipt.get("final_redecode_calls") == 0
                 and receipt.get("exact_F_reuse") is True,
                 f"{cid}.{name}: explicit zero-update receipt differs")
        if old is not None:
            tname = f"T_teacher_{algorithm}"
            _equal(arms[tname]["boxes"], old["arms"]["T"]["boxes"],
                   f"{cid}.{tname}_exact_T_boxes")
            _equal(arms[tname]["interval"], old["arms"]["T"]["interval"],
                   f"{cid}.{tname}_exact_T_interval")
            treceipt = receipts.get(tname, {})
            _require(treceipt.get("status") == "exact_old_T_reuse"
                     and treceipt.get("optimizer_steps") == 0
                     and treceipt.get("final_redecode_calls") == 0
                     and treceipt.get("exact_T_reuse") is True,
                     f"{cid}.{tname}: exact temporal reuse receipt differs")
    if old is None:
        _require(not any(name in arms for name in ("T", *(f"T_teacher_{a}" for a in ALGORITHMS)))
                 and case.get("temporal_exact_reuse", {}).get("status")
                 == "unavailable_no_new_T_created",
                 f"{cid}: new case fabricated a T arm")
    else:
        for name in ("T", *(f"T_teacher_{a}" for a in ALGORITHMS)):
            _require(name in arms, f"{cid}: old-eight T arm missing: {name}")
    return {"fit_count": 0, "optimizer_steps": 0,
            "zero_update_arms": list(f"teacher_{a}" for a in ALGORITHMS),
            "old_T_reused": old is not None}


def _audit_fitted_arms(case: Mapping[str, Any], record: Mapping[str, Any],
                       phase: str, reference_sha: str,
                       original_config: Mapping[str, Any],
                       old: Mapping[str, Any] | None) -> dict[str, Any]:
    cid = record["case_id"]
    references = record["references"]
    if phase == "method":
        role_names = ["teacher"]
        arm_prefixes = {"teacher": "teacher"}
        expected_algorithms = set(ALGORITHMS)
    else:
        role_names = list(CONTROL_ROLES)
        arm_prefixes = {role: role for role in role_names}
        expected_algorithms = {"iou"}

    fits = case.get("fits")
    readouts = case.get("readouts")
    _require(isinstance(fits, Mapping) and isinstance(readouts, Mapping),
             f"{cid}: missing fit/readout mappings")
    expected_keys = {
        f"{role}_{algorithm}"
        for role in role_names for algorithm in expected_algorithms
    }
    _require(set(fits) == expected_keys and set(readouts) == expected_keys,
             f"{cid}: exact fit/readout key set differs")
    _require(case.get("actual_fit_count") == len(expected_keys)
             and case.get("actual_final_redecode_calls") == len(expected_keys),
             f"{cid}: fitted case counts differ from saved arms")
    _equal(case.get("supervision_anchors"),
           {role: references[role]["anchors"] for role in role_names},
           f"{cid}.supervision_anchor_lock_binding")

    if phase == "reference_control":
        role_rows = case.get("reference_control_roles")
        _require(isinstance(role_rows, Mapping) and set(role_rows) == set(CONTROL_ROLES),
                 f"{cid}: control role receipts differ")
        for role in CONTROL_ROLES:
            _equal(role_rows[role].get("anchors"), references[role]["anchors"],
                   f"{cid}.{role}.role_anchor_lock_binding")
            _require(role_rows[role].get("final_redecode_calls") == 1
                     and role_rows[role].get("source_role_update_intentionally_allowed")
                     is (role == "Source")
                     and role_rows[role].get("lock_no_update_ignored_for_control")
                     is bool(references[role].get("no_update", False)),
                     f"{cid}.{role}: reference-control role receipt differs")

    frame_ids = list(map(int, case["frame_ids"]))
    key = case.get("configuration", {}).get("key")
    _require(key == f"caption_query_fixed_space|{cid}",
             f"{cid}: common-random-number fit key differs")
    fit_checks = {}
    actual_steps = 0
    for name, fit in fits.items():
        role, algorithm = name.rsplit("_", 1)
        _require(role in role_names and algorithm in expected_algorithms,
                 f"{cid}: malformed fit key {name}")
        anchors = _anchors_from_case(case, role, frame_ids)
        if algorithm == "direct":
            check = _audit_direct_fit(fit, readouts[name], anchors,
                                      case["initial_state"])
            if fit.get("seed_key") is not None:
                _require(fit.get("seed_key") == key,
                         f"{cid}.{name}: Direct fit key differs")
        else:
            check = _audit_opd_fit(fit, anchors, frame_ids, key,
                                   algorithm, case["initial_state"])
            _require(fit.get("reference_sha256") == reference_sha,
                     f"{cid}.{name}: fit reference-lock SHA differs")
            _require(isinstance(fit.get("reference_role"), str)
                     and fit.get("reference_role") in {"source", "teacher", "offline_oracle"},
                     f"{cid}.{name}: fit uses an unsupported internal reference binding")
            _require(fit.get("reward_objective") in {
                algorithm, "diou_lambda_1" if algorithm == "diou" else "iou"},
                f"{cid}.{name}: fit reward objective differs")
        _require(fit.get("config") == original_config
                 and fit.get("scope") == "joint"
                 and fit.get("active_parameters") == 1792
                 and fit.get("declared_steps") == 5
                 and len(fit.get("path", [])) == 5,
                 f"{cid}.{name}: fit differs from original five-step joint configuration")
        _equal(fit["path"][0].get("state"), case["initial_state"],
               f"{cid}.{name}.first_step_state_vs_source_initial")
        if algorithm != "direct":
            _require(fit.get("seed_key") == key,
                     f"{cid}.{name}: Gaussian fit key differs")
        readout = readouts[name]
        _equal(readout.get("boxes"), fit.get("final_cached_decoder_boxes"),
               f"{cid}.{name}.full_readout_vs_cached_boxes")
        _equal(readout.get("logits"), case.get("source_logits"),
               f"{cid}.{name}.temporal_logits_vs_native")
        arm_name = f"{arm_prefixes[role]}_{algorithm}"
        _equal(torch.as_tensor(case["arms"][arm_name]["boxes"], dtype=torch.float64),
               _xyxy(torch.as_tensor(readout["boxes"], dtype=torch.float64)),
               f"{cid}.{arm_name}.saved_arm_vs_readout")
        _equal(case["arms"][arm_name]["interval"], readout.get("physical_interval"),
               f"{cid}.{arm_name}.interval_vs_readout")
        _require(readout.get("physical_interval") == case.get("Native_interval"),
                 f"{cid}.{name}: spatial fit changed the source physical interval")
        actual_steps += int(fit.get("optimizer_steps", -1))
        fit_checks[name] = check
        if phase == "method":
            receipt = case.get("reference_receipts", {}).get(name, {})
            _require(receipt.get("status") == "updated"
                     and receipt.get("lock_no_update") is False
                     and receipt.get("optimizer_steps") == int(fit.get("optimizer_steps", -1))
                     and receipt.get("final_redecode_calls") == 1,
                     f"{cid}.{name}: fit receipt differs from saved arithmetic")
            if cid in OLD_IDS:
                temporal_receipt = case.get("reference_receipts", {}).get(
                    f"T_teacher_{algorithm}", {})
                _require(temporal_receipt.get("inherits_fit") == name
                         and temporal_receipt.get("optimizer_steps")
                         == int(fit.get("optimizer_steps", -1))
                         and temporal_receipt.get("final_redecode_calls") == 0
                         and temporal_receipt.get("old_T_interval_reused") is True,
                         f"{cid}.T_teacher_{algorithm}: inherited T fit receipt differs")
        else:
            _require(case["reference_control_roles"][role].get("optimizer_steps")
                     == int(fit.get("optimizer_steps", -1)),
                     f"{cid}.{role}: role receipt optimizer count differs from fit")
    _require(case.get("optimizer_steps") == actual_steps,
             f"{cid}: case optimizer step aggregate differs from per-fit counts")

    if old is not None:
        for algorithm in ALGORITHMS if phase == "method" else ("iou",):
            role = "teacher" if phase == "method" else None
            if role is not None:
                name, tname = f"teacher_{algorithm}", f"T_teacher_{algorithm}"
            else:
                continue
            _equal(case["arms"][tname]["boxes"], case["arms"][name]["boxes"],
                   f"{cid}.{tname}_inherits_spatial_fit")
            _equal(case["arms"][tname]["interval"], old["arms"]["T"]["interval"],
                   f"{cid}.{tname}_reuses_old_T_interval")
    elif phase == "method":
        _require(not any(f"T_teacher_{a}" in case["arms"] for a in ALGORITHMS),
                 f"{cid}: new case fabricated a temporal teacher arm")

    return {"fit_count": len(fits), "optimizer_steps": actual_steps,
            "declared_steps": len(fits) * 5, "fits": fit_checks}


def _audit_case(gate: Mapping[str, Any], cid: str) -> dict[str, Any]:
    phase = gate["phase"]
    roster = gate["roster_by_id"][cid]
    record = gate["reference_by_id"][cid]
    path = gate["case_paths"][cid]
    # All output bytes and every runtime pin were verified before this first
    # result deserialization.
    case = torch.load(path, map_location="cpu", weights_only=False)
    _require(case.get("case_id") == cid
             and case.get("dataset") == roster["dataset"]
             and case.get("source_case_sha256") == roster["source_case_sha256"],
             f"{cid}: student output source binding differs")
    _require(case.get("GT_read") is False
             and case.get("reference_lock_sha256") == gate["hashes"]["reference_lock"]
             and case.get("source_restored") is True,
             f"{cid}: phase hash, GT flag, or source restoration receipt differs")
    source = _load_source_case(gate["source_inputs"][cid])
    source_summary = _validate_source_case(source, roster, case)
    _validate_source_arms(case, cid, source, source_summary["video_bounds"])
    expected_anchors = {
        role: value.get("anchors", [])
        for role, value in record["references"].items()
    }
    _equal(case.get("supervision_anchors"), expected_anchors,
           f"{cid}.supervision_anchors_vs_reference_lock")
    if phase == "method":
        _equal(case.get("reference_anchors"), expected_anchors,
               f"{cid}.reference_anchors_vs_reference_lock")

    # Every fit begins from the exact source-only trainable state; all optimizer
    # and reward math is rederived from the saved step evidence below.
    initial = source["source_light_initial_state"]
    _equal(case.get("initial_state"), initial, f"{cid}.saved_initial_vs_source_initial")
    for name, fit in case.get("fits", {}).items():
        _equal(fit.get("initial_state"), initial, f"{cid}.{name}.fit_initial_vs_source_initial")

    old = None
    reuse = None
    if cid in OLD_IDS:
        old_path = gate["old_case_paths"][cid]
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        reuse = _assert_old_temporal_reuse(
            case, old, cid, roster, source_summary["video_bounds"])
        reuse["old_temporal_barrier_sha256"] = gate["hashes"]["old_temporal_barrier"]
        _require(case.get("temporal_exact_reuse", {}).get("old_student_sha256")
                 == reuse["old_student_sha256"],
                 f"{cid}: temporal reuse receipt SHA differs")
    else:
        _require(case.get("temporal_exact_reuse", {}).get("status")
                 == "unavailable_no_new_T_created",
                 f"{cid}: new source must not claim a T baseline")
        _require("T" not in case["arms"], f"{cid}: new source contains a fabricated T arm")

    from scripts import stvg_target_teacher_student_v1 as core
    original_config = core.load_original_opd_config(roster["dataset"])
    original_config["steps"] = 5
    config_row = case.get("configuration", {})
    _require(config_row.get("scope") == "joint"
             and config_row.get("key") == f"caption_query_fixed_space|{cid}"
             and config_row.get("steps") == 5
             and config_row.get("active_parameters") == 1792
             and config_row.get("config") == original_config,
             f"{cid}: phase configuration/key differs from frozen original settings")
    if phase == "method":
        _require(config_row.get("method_reference_lock_sha256")
                 == gate["hashes"]["reference_lock"],
                 f"{cid}: method case does not bind its reference lock")
    else:
        _require(config_row.get("reference_lock_sha256")
                 == gate["hashes"]["reference_lock"]
                 and config_row.get("Source_role_updated_for_diagnostic") is True,
                 f"{cid}: reference-control configuration binding differs")

    if phase == "method" and record["references"]["teacher"]["no_update"]:
        detail = _check_no_update(case, record, phase, old)
        if old is not None:
            detail["temporal_reuse"] = reuse
        return {
            "dataset": roster["dataset"],
            "no_update": True,
            "fit_count": 0,
            "optimizer_steps": 0,
            "declared_steps": 0,
            "source": source_summary,
            "temporal_reuse": reuse,
            "readback": detail,
        }

    result = _audit_fitted_arms(case, record, phase,
                                gate["hashes"]["reference_lock"],
                                original_config, old)
    if old is not None:
        result["temporal_reuse"] = reuse
    _require(case.get("actual_fit_count") == result["fit_count"],
             f"{cid}: output fit count differs from audited fit mapping")
    return {"dataset": roster["dataset"],
            "no_update": False,
            "fit_count": result["fit_count"],
            "optimizer_steps": result["optimizer_steps"],
            "declared_steps": result["declared_steps"],
            "source": source_summary,
            "temporal_reuse": reuse,
            "readback": result}


def audit_phase(phase: str, *, write_report: bool = True) -> dict[str, Any]:
    gate = _preflight(phase)
    case_results = {}
    fit_count = optimizer_steps = declared_steps = no_update_count = 0
    domains = {"hc2": 0, "vidstg": 0}
    for cid in sorted(gate["case_paths"]):
        result = _audit_case(gate, cid)
        case_results[cid] = result
        domains[result["dataset"]] += 1
        fit_count += result["fit_count"]
        optimizer_steps += result["optimizer_steps"]
        declared_steps += result["declared_steps"]
        no_update_count += int(result["no_update"])
    _require(fit_count == gate["expected_fits"],
             "audited fit count differs from the locked phase matrix")
    _require(declared_steps == gate["declared_fit_steps"],
             "declared fit-step count differs from the five-step locked matrix")
    _require(optimizer_steps == int(gate["barrier"].get("optimizer_steps", -1)),
             "audited actual optimizer-step total differs from phase barrier")
    if phase == "method":
        _require(set(case_results) == METHOD_IDS
                 and domains == {"hc2": 8, "vidstg": 8}
                 and fit_count == 18
                 and no_update_count == 10,
                 "method phase does not match the fixed 16-case, 18-fit no-update matrix")
    else:
        _require(set(case_results) == {CONTROL_ID}
                 and domains == {"hc2": 0, "vidstg": 1}
                 and fit_count == 3 and no_update_count == 0,
                 "reference-control phase does not match one case and three fitted roles")
    _require(gate["barrier"].get("actual_final_redecode_calls") == fit_count,
             "phase barrier readout count differs from audited fitted arms")

    report = {
        "schema": "stvg-motion-guided-independent-cpu-math-v1",
        "status": "pass",
        "phase": phase,
        "case_count": len(case_results),
        "fit_count": fit_count,
        "declared_fit_steps": declared_steps,
        "actual_optimizer_steps": optimizer_steps,
        "no_update_case_count": no_update_count,
        "domain_counts": domains,
        "GT_read": False,
        "GT_loader_called": False,
        "model_loaded": False,
        "GPU_initialized": False,
        "pixels_decoded": False,
        "barrier_bound_runtime": str(gate["runtime_path"].relative_to(ROOT)),
        "worker_source": gate["worker_path"],
        "fit_helper_source": FIT_HELPER,
        "transitive_source_bindings": gate["transitive_bindings"],
        "case_readbacks": case_results,
        "provenance": gate["hashes"],
        "reader_sha256": _sha_file(Path(__file__)),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpretation": {
            "fit_steps_per_declared_fit": 5,
            "actual_optimizer_steps_may_be_zero_for_uniform_reward": True,
            "new_eight_have_no_temporal_T_arm": True,
            "old_eight_temporal_F_Projection_T_reused_exactly": True,
            "reference_control_source_role_update_is_diagnostic": phase == "reference_control",
        },
    }
    if write_report:
        output = BASE / phase / "ROOT_CPU_FIT_MATH_READBACK.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing readback: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("method", "reference_control"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="check locks, runtime pins, and complete barriers without loading outputs")
    args = parser.parse_args()
    if args.preflight_only:
        gate = _preflight(args.phase)
        print(json.dumps({"status": "preflight_pass", "phase": args.phase,
                          "case_count": gate["expected_cases"],
                          "fit_count": gate["expected_fits"],
                          "declared_fit_steps": gate["declared_fit_steps"],
                          "actual_optimizer_steps": gate["barrier"].get("optimizer_steps"),
                          "GT_read": False}, sort_keys=True))
        return
    report = audit_phase(args.phase)
    print(json.dumps({"status": report["status"], "phase": report["phase"],
                      "case_count": report["case_count"],
                      "fit_count": report["fit_count"],
                      "declared_fit_steps": report["declared_fit_steps"],
                      "actual_optimizer_steps": report["actual_optimizer_steps"],
                      "GT_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
