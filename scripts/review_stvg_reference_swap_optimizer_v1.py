"""CPU readback of saved E1 Reference-swap Gaussian-IoU optimizer artifacts.

This reader is deliberately post-seal. It loads only hash-verified saved source,
parent prediction, student, recovery, and optimizer-checkpoint files. It does not
instantiate a model, initialize CUDA, read annotations, run a scorer, or replay
backward. Final sealed fits, exact reused predictions, and preserved failed
prefix fits are reported as separate quantities.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
PARENT = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
SOURCE_BASE = ROOT / "artifacts/stvg_final_table1_v1"
STUDENT_DIR = BASE / "student"
STUDENT_RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
DESIGN_PATH = BASE / "DESIGN_LOCK.json"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
TEACHER_BARRIER_PATH = BASE / "TEACHER_GLOBAL_BARRIER.json"
STUDENT_BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
PARENT_BARRIER_PATH = PARENT / "STUDENT_GLOBAL_BARRIER.json"
PARENT_REFERENCE_PATH = PARENT / "REFERENCE_LOCK.json"
MATH_HELPER_PATH = ROOT / "scripts/review_stvg_reference_carry_optimizer_v1.py"
MATH_HELPER_SHA256 = "52d8fd784d91a09eb15b8243f93313619751e413875eecc67f9eba1b0b2aea73"
RECOVERY_PREFIXES = (
    BASE / "recovery/student_float_boundary006/student",
    BASE / "recovery/student_readout_counter008/student",
)
PARTIAL_ATTEMPTS = (
    BASE / "recovery/student_float_boundary006/partial/HC11/WF_common",
    BASE / "recovery/student_readout_counter008/partial/HC11/WF_common",
)
PARTIAL_CAPTURE_PATHS = (
    BASE / "recovery/student_float_boundary006/CAPTURE.json",
    BASE / "recovery/student_readout_counter008/CAPTURE.json",
)
EXPECTED_CASES = 32
EXPECTED_FITS = 38
EXPECTED_REUSED_CASES = 10
EXPECTED_REUSED_FITS = 16
EXPECTED_PARTIAL_FITS = 2
DOMAINS = ("hc2", "vidstg")
ROLES = ("WF_common", "DINO_common")
ARMS = (
    "F", "T", "S_new_crop", "TS_new_crop",
    "S_WF_common", "TS_WF_common", "S_DINO_common", "TS_DINO_common",
)
STATE_KEYS = (
    "spatial.query_residual",
    "spatial.layers.5.norm1.weight", "spatial.layers.5.norm1.bias",
    "spatial.layers.5.norm3.weight", "spatial.layers.5.norm3.bias",
    "spatial.layers.5.norm4.weight", "spatial.layers.5.norm4.bias",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("artifact escaped the project root") from exc


def _eq(left: Any, right: Any, torch, np) -> bool:
    """Exact recursive comparison that handles both serialized tensors and ndarrays."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_eq(left[key], right[key], torch, np) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_eq(a, b, torch, np) for a, b in zip(left, right)))
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
    return left == right


def _assert_locked(doc: Mapping[str, Any], label: str, *, count: int | None = None) -> None:
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise ValueError(f"{label} must be locked and explicitly GT-free")
    observed_count = doc.get("count", len(doc.get("records", [])))
    if count is not None and int(observed_count) != count:
        raise ValueError(f"{label} count differs from {count}")


def _verify_runtime(runtime: Mapping[str, Any]) -> None:
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("student runtime must be locked and declare GT_read=false")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise ValueError("student runtime pins/protected must be mappings")
    for rel, expected in {**pins, **protected}.items():
        path = (ROOT / rel).resolve()
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"student runtime source hash mismatch: {rel}")


def _safe_artifact_path(rel: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError("artifact path must be a nonempty project-relative path")
    path = (ROOT / rel).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("artifact path escapes the project root")
    return path


def _verify_barrier_files(barrier: Mapping[str, Any], expected_paths: set[str],
                          label: str) -> dict[str, dict[str, Any]]:
    files = barrier.get("files")
    if not isinstance(files, list):
        raise ValueError(f"{label} files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"{label} contains a malformed file record")
        rel = item.get("path")
        if not isinstance(rel, str) or rel in by_path:
            raise ValueError(f"{label} has a missing or duplicate file path")
        path = _safe_artifact_path(rel)
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise ValueError(f"{label} file missing or byte count differs: {rel}")
        if _sha(path) != item.get("sha256"):
            raise ValueError(f"{label} file SHA mismatch: {rel}")
        by_path[rel] = item
    if set(by_path) != expected_paths:
        raise ValueError(f"{label} paths differ from the locked expected set")
    return by_path


def _load_locks() -> dict[str, Any]:
    runtime = _read(STUDENT_RUNTIME_PATH)
    _verify_runtime(runtime)
    design = _read(DESIGN_PATH)
    roster_doc = _read(ROSTER_PATH)
    ref_doc = _read(REFERENCE_PATH)
    teacher_barrier = _read(TEACHER_BARRIER_PATH)
    student_barrier = _read(STUDENT_BARRIER_PATH)
    parent_barrier = _read(PARENT_BARRIER_PATH)
    parent_ref_doc = _read(PARENT_REFERENCE_PATH)
    for label, doc, count in (
        ("E1 design", design, None), ("E1 roster", roster_doc, EXPECTED_CASES),
        ("E1 reference lock", ref_doc, EXPECTED_CASES),
        ("E1 teacher barrier", teacher_barrier, 49),
        ("E1 student barrier", student_barrier, EXPECTED_CASES),
        ("parent student barrier", parent_barrier, EXPECTED_CASES),
        ("parent reference lock", parent_ref_doc, EXPECTED_CASES),
    ):
        if label.endswith("barrier"):
            if doc.get("status") != "complete" or doc.get("GT_read") is not False:
                raise ValueError(f"{label} must be complete and GT-free")
            if int(doc.get("count", -1)) != count:
                raise ValueError(f"{label} count differs from {count}")
        else:
            _assert_locked(doc, label, count=count)

    if not STUDENT_BARRIER_PATH.is_file():
        raise RuntimeError("all-case audit requires the complete student barrier")
    if int(student_barrier.get("actual_fit_count", -1)) != EXPECTED_FITS:
        raise ValueError("final student barrier does not declare 38 finalized fits")
    if int(student_barrier.get("full_model_readout_calls", -1)) != EXPECTED_FITS:
        raise ValueError("final student barrier does not declare 38 full-model readouts")

    runtime_sha = _sha(STUDENT_RUNTIME_PATH)
    design_sha = _sha(DESIGN_PATH)
    roster_sha = _sha(ROSTER_PATH)
    ref_sha = _sha(REFERENCE_PATH)
    teacher_sha = _sha(TEACHER_BARRIER_PATH)
    parent_sha = _sha(PARENT_BARRIER_PATH)
    parent_ref_sha = _sha(PARENT_REFERENCE_PATH)
    student_sha = _sha(STUDENT_BARRIER_PATH)

    expected_bindings = {
        "design_lock_sha256": design_sha,
        "roster_sha256": roster_sha,
        "teacher_global_barrier_sha256": teacher_sha,
        "parent_wf_reference_lock_sha256": parent_ref_sha,
    }
    ref_bindings = ref_doc.get("bindings", {})
    for key, expected in expected_bindings.items():
        if ref_bindings.get(key) != expected:
            raise ValueError(f"E1 reference lock binding mismatch: {key}")
    if ref_bindings.get("runtime_sha256") != runtime.get("bindings", {}).get("teacher_runtime_sha256",
                                                                         ref_bindings.get("runtime_sha256")):
        # The E1 reference lock is authored by the teacher runtime.  The student
        # runtime has its own digest and must not be conflated with that parent.
        raise ValueError("reference lock teacher-runtime binding is inconsistent")

    student_bindings = runtime.get("bindings", {})
    for key in ("design_lock_sha256", "roster_sha256", "source_roster_sha256",
                "source_capture_barrier_sha256", "teacher_job_lock_sha256",
                "parent_wf_reference_lock_sha256", "expert_checkpoint_sha256",
                "expert_config_sha256", "processor_config_sha256"):
        expected = ref_bindings.get(key)
        if expected is not None and student_bindings.get(key) != expected:
            raise ValueError(f"student runtime binding mismatch: {key}")
    if student_barrier.get("runtime_sha256") != runtime_sha:
        raise ValueError("student barrier runtime SHA mismatch")
    if student_barrier.get("reference_lock_sha256") != ref_sha:
        raise ValueError("student barrier reference-lock SHA mismatch")
    if student_barrier.get("parent_student_barrier_sha256") != parent_sha:
        raise ValueError("student barrier parent-student SHA mismatch")
    if ref_bindings.get("runtime_sha256") != teacher_barrier.get("bindings", {}).get("runtime_sha256"):
        raise ValueError("teacher/reference runtime binding differs")

    roster_rows = roster_doc.get("records")
    ref_rows = ref_doc.get("records")
    parent_ref_rows = parent_ref_doc.get("records")
    if not all(isinstance(x, list) and len(x) == EXPECTED_CASES
               for x in (roster_rows, ref_rows, parent_ref_rows)):
        raise ValueError("locked rosters/references must each contain exactly 32 records")
    roster = {str(row.get("case_id", "")): row for row in roster_rows}
    refs = {str(row.get("case_id", "")): row for row in ref_rows}
    parent_refs = {str(row.get("case_id", "")): row for row in parent_ref_rows}
    if any(not key for key in roster) or len(roster) != 32 or set(roster) != set(refs) or set(refs) != set(parent_refs):
        raise ValueError("E1/parent case membership differs or has duplicates")
    if any(roster[cid].get("dataset") not in DOMAINS or
           refs[cid].get("dataset") != roster[cid].get("dataset") or
           parent_refs[cid].get("dataset") != roster[cid].get("dataset") for cid in roster):
        raise ValueError("case dataset binding differs across the locks")
    domain_counts = {domain: sum(row.get("dataset") == domain for row in roster.values()) for domain in DOMAINS}
    if domain_counts != {"hc2": 16, "vidstg": 16}:
        raise ValueError("E1 cohort is not balanced at 16 sources per domain")

    # Fixed lock-level reference support checks; no annotations are consulted.
    common_count = 0
    common_cases = 0
    empty_cases = 0
    for cid, row in refs.items():
        wf = row.get("common_wf_anchors")
        dino = row.get("common_dino_anchors")
        if not isinstance(wf, list) or not isinstance(dino, list) or len(wf) != len(dino):
            raise ValueError(f"matched common support malformed for {cid}")
        for a, b in zip(wf, dino):
            if any(a.get(key) != b.get(key) for key in ("position", "physical_frame_id", "weight")):
                raise ValueError(f"matched WF/DINO support differs for {cid}")
        parent_anchors = parent_refs[cid].get("new_crop", {}).get("anchors", [])
        parent_by_pos = {int(a["position"]): a for a in parent_anchors}
        for anchor in wf:
            old = parent_by_pos.get(int(anchor["position"]))
            if old is None or any(old.get(key) != anchor.get(key) for key in
                                  ("position", "physical_frame_id", "weight", "box_cxcywh")):
                raise ValueError(f"WF common anchor does not preserve parent geometry for {cid}")
        common_count += len(wf)
        common_cases += int(bool(wf))
        empty_cases += int(not wf)
    if (common_count, common_cases, empty_cases) != (38, 19, 13):
        raise ValueError("reference support counts differ from the locked 38/19/13 design")

    if (int(teacher_barrier.get("accepted_count", -1)), int(teacher_barrier.get("rejected_count", -1)),
            int(teacher_barrier.get("unknown_count", -1))) != (38, 2, 9):
        raise ValueError("teacher barrier selection counts differ from the sealed 38/2/9 readout")
    parent_files = _verify_barrier_files(
        parent_barrier,
        {str((PARENT / "student" / f"{cid}.pt").relative_to(ROOT)) for cid in roster},
        "parent prediction barrier",
    )
    student_files = _verify_barrier_files(
        student_barrier,
        {str((STUDENT_DIR / f"{cid}.pt").relative_to(ROOT)) for cid in roster},
        "E1 student barrier",
    )
    return {
        "runtime": runtime, "design": design, "roster_doc": roster_doc,
        "ref_doc": ref_doc, "teacher_barrier": teacher_barrier,
        "student_barrier": student_barrier, "parent_barrier": parent_barrier,
        "parent_ref_doc": parent_ref_doc, "roster": roster, "refs": refs,
        "parent_refs": parent_refs, "parent_files": parent_files,
        "student_files": student_files, "runtime_sha": runtime_sha,
        "design_sha": design_sha, "roster_sha": roster_sha, "ref_sha": ref_sha,
        "teacher_sha": teacher_sha, "parent_sha": parent_sha,
        "parent_ref_sha": parent_ref_sha, "student_sha": student_sha,
        "domain_counts": domain_counts, "common_count": common_count,
        "common_cases": common_cases, "empty_cases": empty_cases,
    }


def _torch_load(path: Path, torch):
    # Every call is preceded by the exact lock/barrier SHA and byte check.
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_source_state(rec: Mapping[str, Any], ref: Mapping[str, Any], torch):
    rel = rec.get("source_case_path")
    path = _safe_artifact_path(rel)
    if not path.is_file() or _sha(path) != rec.get("source_case_file_sha256"):
        raise ValueError(f"source-case file hash mismatch for {rec.get('case_id')}")
    receipt_path = _safe_artifact_path(str(rec.get("source_receipt_path", "")))
    if not receipt_path.is_file() or _sha(receipt_path) != rec.get("source_receipt_sha256"):
        raise ValueError(f"source-case receipt hash mismatch for {rec.get('case_id')}")
    receipt = _read(receipt_path)
    if (receipt.get("GT_read") is not False
            or receipt.get("case_id") != rec.get("case_id")
            or receipt.get("source_case_sha256") != ref.get("source_case_sha256")
            or receipt.get("sha256") != rec.get("source_case_file_sha256")
            or int(receipt.get("bytes", -1)) != int(rec.get("source_case_bytes", -2))):
        raise ValueError(f"source-case receipt binding mismatch for {rec.get('case_id')}")
    case = _torch_load(path, torch)
    if (case.get("GT_read") is not False or case.get("case_id") != rec.get("case_id")
            or case.get("dataset") != rec.get("dataset")
            or case.get("source_case_sha256") != ref.get("source_case_sha256")):
        raise ValueError(f"source-case payload binding mismatch for {rec.get('case_id')}")
    if not isinstance(case.get("source_light_initial_state"), Mapping):
        raise ValueError(f"source-light initial state absent for {rec.get('case_id')}")
    return case, _relative(path), _sha(receipt_path)


def _xyxy_from_cxcywh(value: Any, np):
    if hasattr(value, "detach"):
        arr = value.detach().cpu().double().numpy()
    else:
        arr = np.asarray(value, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    return np.concatenate((arr[..., :2] - arr[..., 2:] / 2.0,
                           arr[..., :2] + arr[..., 2:] / 2.0), axis=-1)


def _audit_reuse(final_payload: Mapping[str, Any], final_receipt: Mapping[str, Any],
                 cid: str, runtime_sha: str, torch, np) -> dict[str, Any]:
    reused_sha = final_payload.get("reused_prediction_sha256")
    if not isinstance(reused_sha, str) or len(reused_sha) != 64:
        raise ValueError(f"reused prediction hash absent for {cid}")
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for directory in RECOVERY_PREFIXES:
        path = directory / f"{cid}.pt"
        if not path.is_file() or _sha(path) != reused_sha:
            continue
        receipt_path = path.with_suffix(".receipt.json")
        if not receipt_path.is_file():
            raise ValueError(f"recovery receipt missing for reused output {cid}")
        old_receipt = _read(receipt_path)
        if (old_receipt.get("sha256") != reused_sha or
                int(old_receipt.get("bytes", -1)) != path.stat().st_size or
                old_receipt.get("GT_read") is not False):
            raise ValueError(f"preserved recovery receipt mismatch for {cid}")
        old_payload = _torch_load(path, torch)
        matches.append((path, old_receipt, old_payload))
    if not matches:
        raise ValueError(f"reused prediction does not match an allowed immutable recovery file: {cid}")
    if any(not _eq(matches[0][2], row[2], torch, np) for row in matches[1:]):
        raise ValueError(f"multiple recovery copies disagree for reused output {cid}")

    old_path, old_receipt, old_payload = matches[0]
    if final_payload.get("reused_runtime_sha256") != old_payload.get("runtime_sha256"):
        raise ValueError(f"reused runtime hash does not identify the original output for {cid}")
    if final_payload.get("runtime_sha256") != runtime_sha:
        raise ValueError(f"reused payload is not bound to the current runtime for {cid}")
    expected_fit_count = sum(value is not None for value in final_payload.get("fits", {}).values())
    correction = final_payload.get("readout_counter_correction")
    if not isinstance(correction, Mapping):
        raise ValueError(f"reused payload lacks the declared readout-counter correction for {cid}")
    if (int(correction.get("previous", -1)) != int(old_payload.get("readout_call_count", -2))
            or int(correction.get("actual_per_case", -1)) != expected_fit_count
            or int(final_payload.get("readout_call_count", -1)) != expected_fit_count):
        raise ValueError(f"readout-counter correction is not the exact per-case fit count for {cid}")
    if (final_receipt.get("reused_prediction_sha256") != reused_sha
            or final_receipt.get("reused_runtime_sha256", old_payload.get("runtime_sha256")) != old_payload.get("runtime_sha256")
            or final_receipt.get("new_model_execution") is not False
            or final_receipt.get("readout_call_count") != expected_fit_count):
        raise ValueError(f"reused receipt does not disclose exact reuse/counter correction for {cid}")

    # Normalize only the expressly allowed provenance/runtime/counter fields.
    new_normalized = copy.deepcopy(dict(final_payload))
    for key in ("reused_prediction_sha256", "reused_runtime_sha256", "readout_counter_correction"):
        new_normalized.pop(key, None)
    new_normalized["runtime_sha256"] = old_payload.get("runtime_sha256")
    new_normalized["readout_call_count"] = old_payload.get("readout_call_count")
    if not _eq(new_normalized, old_payload, torch, np):
        raise ValueError(f"reused scientific payload differs from its preserved original for {cid}")
    return {
        "case_id": cid,
        "recovery_path": _relative(old_path),
        "recovery_sha256": reused_sha,
        "recovery_bytes": old_path.stat().st_size,
        "recovery_receipt_sha256": _sha(old_path.with_suffix(".receipt.json")),
        "recovery_runtime_sha256": old_payload.get("runtime_sha256"),
        "final_payload_sha256": final_receipt.get("sha256"),
        "final_payload_science_tree_exact": True,
        "final_fit_count": expected_fit_count,
        "prior_readout_count": correction["previous"],
        "corrected_readout_count": expected_fit_count,
    }


def _fit_role_for_math(fit: Mapping[str, Any], role: str) -> dict[str, Any]:
    allowed = {
        "WF_common": "weight_free_common_same_frame_geometry",
        "DINO_common": "original_dino_same_frame_geometry",
    }
    if fit.get("reference_role") != allowed[role]:
        raise ValueError(f"fit role label mismatch for {role}")
    view = dict(fit)
    # The independent arithmetic routine was first validated on the E3 fixed
    # Weight-free role. Only this descriptive field differs in E1; all math,
    # seed, config, state, and reference geometry remain the saved E1 values.
    view["reference_role"] = "fixed_weight_free_top1_teacher"
    return view


def _audit_partial_checkpoint_dir(directory: Path, cid: str, role: str,
                                  rec: Mapping[str, Any], ref: Mapping[str, Any],
                                  config: Mapping[str, Any], ref_sha: str,
                                  torch, np, math_reader) -> dict[str, Any]:
    paths = [directory / f"step{index}.pt" for index in range(1, 6)]
    if not all(path.is_file() for path in paths):
        raise ValueError(f"preserved partial fit is incomplete: {_relative(directory)}")
    snapshots = []
    for expected_next_step, path in enumerate(paths, 1):
        snapshot = _torch_load(path, torch)
        if (snapshot.get("GT_read") is not False or snapshot.get("case_id") != cid
                or snapshot.get("reference_sha256") != ref_sha
                or snapshot.get("reference_role") != "weight_free_common_same_frame_geometry"
                or int(snapshot.get("next_step", -1)) != expected_next_step
                or len(snapshot.get("path", [])) != expected_next_step
                or len(snapshot.get("rounds", [])) != expected_next_step):
            raise ValueError(f"partial fit checkpoint metadata mismatch: {_relative(path)}")
        path_rows = snapshot["path"]
        if (path_rows[-1].get("step") != expected_next_step - 1
                or not _eq(path_rows[-1]["state_after"], snapshot["state"], torch, np)
                or not _eq(path_rows[-1]["optimizer_after"], snapshot["optimizer_state"], torch, np)):
            raise ValueError(f"partial fit checkpoint state does not match its last saved step: {_relative(path)}")
        snapshots.append((path, snapshot))
    # Each later checkpoint must append one step while retaining the prior
    # exact path/round history; this catches overwritten or mixed attempts.
    for prior, current in zip(snapshots, snapshots[1:]):
        if not _eq(prior[1]["path"], current[1]["path"][:-1], torch, np):
            raise ValueError(f"partial fit path changed across checkpoints: {_relative(current[0])}")
        if not _eq(prior[1]["rounds"], current[1]["rounds"][:-1], torch, np):
            raise ValueError(f"partial fit rounds changed across checkpoints: {_relative(current[0])}")

    final_path, final = snapshots[-1]
    path_rows = final["path"]
    fit = {
        "schema": "stvg-reference-recovery-fit-v1",
        "profile": "current5", "scope": "joint", "dataset": rec["dataset"],
        "active_parameters": 1792, "config": dict(config),
        "validation_enabled": True,
        "initial_state": path_rows[0]["state_before"],
        "initial_cached_decoder_boxes": path_rows[0]["boxes_before"],
        "fit_final_state": final["state"], "final_state": final["state"],
        "final_cached_decoder_boxes": path_rows[-1]["boxes_after"],
        "path": path_rows, "rounds": final["rounds"],
        "optimizer_steps": int(final["accepted_optimizer_steps"]),
        "proposed_optimizer_steps": int(final["proposed_optimizer_steps"]),
        "rejected_optimizer_steps": int(final["proposed_optimizer_steps"] - final["accepted_optimizer_steps"]),
        "declared_steps": 5,
        "seed_key": f"caption_query_fixed_space|{rec['dataset']}_{rec['query_key_sha256']}",
        "observation_positions": [int(row["position"]) for row in ref["common_wf_anchors"]],
        "reference_sha256": ref_sha,
        "reference_role": final["reference_role"],
        "optimizer_state_reset_each_query": True,
        "query_residual_reset_at_arrival": bool(torch.count_nonzero(path_rows[0]["state_before"][STATE_KEYS[0]]) == 0),
        "GT_read": False,
    }
    detail = math_reader._audit_fit(_fit_role_for_math(fit, role), ref["common_wf_anchors"],
                                    rec, config, ref_sha, torch, np)
    return {
        "case_id": cid, "role": role,
        "checkpoint_dir": _relative(directory),
        "checkpoint_sha256s": [
            {"path": _relative(path), "sha256": _sha(path), "bytes": path.stat().st_size}
            for path, _ in snapshots
        ],
        "saved_steps": 5,
        "optimizer_steps_accepted": fit["optimizer_steps"],
        "optimizer_steps_proposed": fit["proposed_optimizer_steps"],
        "math_readback": detail,
        "full_model_readout_artifact_present": False,
    }


def _run(output: str | None = None) -> dict[str, Any]:
    # Set before importing torch; the audit must remain CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from scripts import review_stvg_reference_carry_optimizer_v1 as math_reader

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU optimizer audit refuses an initialized CUDA runtime")
    torch.set_num_threads(1)
    if _sha(MATH_HELPER_PATH) != MATH_HELPER_SHA256:
        raise ValueError("previously validated CPU math reader changed")
    locks = _load_locks()
    runtime_sha = locks["runtime_sha"]
    design = locks["design"]
    roster = locks["roster"]
    refs = locks["refs"]
    student_barrier = locks["student_barrier"]
    student_files = locks["student_files"]
    ref_sha = locks["ref_sha"]
    expected_fit_total = 0
    expected_steps_total = 0
    expected_readouts_total = 0
    fit_counts = {role: 0 for role in ROLES}
    errors = {key: 0.0 for key in (
        "sample_abs", "mean_abs", "reward_abs", "q_abs", "per_anchor_loss_abs",
        "loss_abs", "reference_iou_abs", "adam_moment_abs", "adam_delta_abs",
    )}
    count_totals = {key: 0 for key in (
        "reward_scalars", "loss_scalars", "adam_parameter_groups", "state_chain_edges",
        "proposals", "accepted", "rejected", "no_information", "skipped_no_anchor",
        "skipped_after_rejection",
    )}
    per_case: list[dict[str, Any]] = []
    reused_reports: list[dict[str, Any]] = []
    reused_case_ids: set[str] = set()
    parent_files = locks["parent_files"]
    parent_barrier = locks["parent_barrier"]
    parent_barrier_items = {item["path"]: item for item in parent_barrier["files"] if item.get("path", "").endswith(".pt")}
    if len(parent_barrier_items) != EXPECTED_CASES:
        raise ValueError("parent barrier must contain exactly 32 saved student .pt files")

    for cid in sorted(roster):
        rec, ref = roster[cid], refs[cid]
        rel = str((STUDENT_DIR / f"{cid}.pt").relative_to(ROOT))
        item = student_files[rel]
        output_path = _safe_artifact_path(rel)
        receipt_path = output_path.with_suffix(".receipt.json")
        receipt = _read(receipt_path)
        if (receipt.get("status") != "complete" or receipt.get("GT_read") is not False
                or receipt.get("case_id") != cid or receipt.get("dataset") != rec["dataset"]
                or receipt.get("sha256") != item["sha256"] or receipt.get("bytes") != item["bytes"]
                or receipt.get("runtime_sha256") != runtime_sha
                or receipt.get("reference_lock_sha256") != ref_sha
                or receipt.get("source_case_sha256") != ref["source_case_sha256"]):
            raise ValueError(f"student receipt binding mismatch for {cid}")
        payload = _torch_load(output_path, torch)
        if (payload.get("schema") != "stvg-reference-swap-student-case-v1"
                or payload.get("case_id") != cid or payload.get("dataset") != rec["dataset"]
                or payload.get("GT_read") is not False or payload.get("source_restored") is not True
                or payload.get("runtime_sha256") != runtime_sha
                or payload.get("reference_lock_sha256") != ref_sha
                or payload.get("source_case_sha256") != ref["source_case_sha256"]):
            raise ValueError(f"student payload binding mismatch for {cid}")

        case, source_rel, source_receipt_sha = _load_source_state(rec, ref, torch)
        frame_ids = list(map(int, case.get("frame_ids", [])))
        if (not frame_ids or payload.get("frame_ids") != frame_ids
                or payload.get("source_file_sha256") != rec.get("source_case_file_sha256")
                or payload.get("source_model_state_sha256") != case.get("source_model_state_sha256")
                or payload.get("pixel_sha256") != case.get("pixel_sha256")):
            raise ValueError(f"student source/frame/pixel binding mismatch for {cid}")
        source_initial = case["source_light_initial_state"]
        if tuple(source_initial) != STATE_KEYS:
            raise ValueError(f"source-light state schema differs for {cid}")
        if payload.get("source_capture_readback", {}).get("status") != "pass":
            raise ValueError(f"source capture readback is not pass for {cid}")
        if payload["source_capture_readback"].get("GT_read") is not False:
            raise ValueError(f"source capture readback is not explicitly GT-free for {cid}")
        if payload.get("source_restore_status") not in ("pass", "pass_source_capture_no_new_fit"):
            raise ValueError(f"saved source restore status is not accepted for {cid}")

        # Verify the shared historical arms are copied byte-for-byte at array/tensor level.
        parent_rel = str((PARENT / "student" / f"{cid}.pt").relative_to(ROOT))
        parent_item = parent_files.get(parent_rel)
        if parent_item is None:
            raise ValueError(f"sealed parent prediction missing for {cid}")
        parent_path = _safe_artifact_path(parent_rel)
        parent_payload = _torch_load(parent_path, torch)
        if (parent_payload.get("GT_read") is not False or parent_payload.get("case_id") != cid
                or parent_payload.get("source_case_sha256") != ref["source_case_sha256"]):
            raise ValueError(f"parent student identity mismatch for {cid}")
        parent_arms = parent_payload.get("arms", {})
        student_arms = payload.get("arms", {})
        if set(student_arms) != set(ARMS):
            raise ValueError(f"student arm set differs from E1 lock for {cid}")
        for arm in ("F", "T", "S_new_crop", "TS_new_crop"):
            if not _eq(student_arms[arm], parent_arms[arm], torch, np):
                raise ValueError(f"parent arm {arm} was not exact reused for {cid}")

        wf = ref.get("common_wf_anchors")
        dino = ref.get("common_dino_anchors")
        if not isinstance(wf, list) or not isinstance(dino, list) or len(wf) != len(dino):
            raise ValueError(f"matched anchor rows malformed for {cid}")
        expected_positions = [int(a["position"]) for a in wf]
        if expected_positions != [int(a["position"]) for a in dino]:
            raise ValueError(f"matched WF/DINO anchor positions differ for {cid}")
        for anchor in wf + dino:
            pos = int(anchor["position"])
            if not 0 <= pos < len(frame_ids) or int(anchor["physical_frame_id"]) != frame_ids[pos]:
                raise ValueError(f"reference source-grid/frame mapping mismatch for {cid}")

        fits = payload.get("fits")
        readouts = payload.get("readouts")
        if not isinstance(fits, Mapping) or set(fits) != set(ROLES):
            raise ValueError(f"fit roles differ from fixed E1 schema for {cid}")
        if not isinstance(readouts, Mapping) or set(readouts) != set(ROLES):
            raise ValueError(f"readout roles differ from fixed E1 schema for {cid}")
        support_n = len(wf)
        case_fit_count = 0
        case_optimizer_steps = 0
        case_readout_count = 0
        case_math = []
        if support_n == 0:
            if any(fits[role] is not None or readouts[role] is not None for role in ROLES):
                raise ValueError(f"empty common support fabricated fit/readout for {cid}")
            for arm, fallback in (("S_WF_common", "F"), ("TS_WF_common", "T"),
                                  ("S_DINO_common", "F"), ("TS_DINO_common", "T")):
                if not _eq(student_arms[arm], student_arms[fallback], torch, np):
                    raise ValueError(f"empty common support is not exact F/T reuse for {cid}/{arm}")
        else:
            for role, anchors in (("WF_common", wf), ("DINO_common", dino)):
                fit = fits[role]
                readout = readouts[role]
                if not isinstance(fit, Mapping) or not isinstance(readout, Mapping):
                    raise ValueError(f"nonempty common support lacks fit/readout for {cid}/{role}")
                if fit.get("candidate_ids") != [row.get("candidate_id") for row in anchors]:
                    raise ValueError(f"saved candidate IDs differ from the locked anchors for {cid}/{role}")
                if int(fit.get("reference_support_count", -1)) != support_n:
                    raise ValueError(f"fit support count mismatch for {cid}/{role}")
                if fit.get("source_model_state_sha256") != case.get("source_model_state_sha256"):
                    raise ValueError(f"fit is not bound to the captured Source state for {cid}/{role}")
                if fit.get("full_model_readout_calls") != 1:
                    raise ValueError(f"fit/readout receipt missing one full-model reread for {cid}/{role}")
                if not _eq(fit.get("initial_state"), source_initial, torch, np):
                    raise ValueError(f"fit did not start from exact Source light state for {cid}/{role}")
                if tuple(fit["initial_state"]) != STATE_KEYS or int(torch.count_nonzero(fit["initial_state"][STATE_KEYS[0]])) != 0:
                    raise ValueError(f"fit Query/LN state schema or reset differs for {cid}/{role}")
                detail = math_reader._audit_fit(
                    _fit_role_for_math(fit, role), anchors, rec, design["configs"][rec["dataset"]],
                    ref_sha, torch, np,
                )
                if not _eq(readout.get("boxes"), fit["final_cached_decoder_boxes"], torch, np):
                    raise ValueError(f"final full-model readout boxes differ from fit for {cid}/{role}")
                s_arm, ts_arm = (("S_WF_common", "TS_WF_common") if role == "WF_common"
                                  else ("S_DINO_common", "TS_DINO_common"))
                expected_s = _xyxy_from_cxcywh(readout["boxes"], np)
                if not np.array_equal(np.asarray(student_arms[s_arm]["boxes"]), expected_s):
                    raise ValueError(f"full-model S arm differs from its readout for {cid}/{role}")
                if not np.array_equal(np.asarray(student_arms[ts_arm]["boxes"]), expected_s):
                    raise ValueError(f"full-model TS arm differs from its readout for {cid}/{role}")
                if list(student_arms[s_arm]["interval"]) != list(student_arms["F"]["interval"]):
                    raise ValueError(f"S common interval differs from F for {cid}/{role}")
                if list(student_arms[ts_arm]["interval"]) != list(student_arms["T"]["interval"]):
                    raise ValueError(f"TS common interval differs from T for {cid}/{role}")
                fit_counts[role] += 1
                case_fit_count += 1
                case_optimizer_steps += int(fit["optimizer_steps"])
                case_readout_count += 1
                for key, value in detail["max_abs_errors"].items():
                    errors[key] = max(errors[key], float(value))
                for key, value in detail["counts"].items():
                    count_totals[key] += int(value)
                case_math.append({"role": role, "anchor_count": detail["anchor_count"],
                                  "optimizer_steps": int(fit["optimizer_steps"]),
                                  "counts": detail["counts"], "max_abs_errors": detail["max_abs_errors"]})

        expected_readout_count = sum(value is not None for value in fits.values())
        if (int(payload.get("actual_fit_count", -1)) != expected_readout_count
                or int(payload.get("readout_call_count", -1)) != expected_readout_count
                or int(payload.get("optimizer_steps", -1)) != case_optimizer_steps
                or int(receipt.get("actual_fit_count", -1)) != expected_readout_count
                or int(receipt.get("readout_call_count", -1)) != expected_readout_count
                or int(receipt.get("optimizer_steps", -1)) != case_optimizer_steps):
            raise ValueError(f"per-case fit/readout/step counters do not match actual artifacts for {cid}")
        expected_fit_total += expected_readout_count
        expected_steps_total += case_optimizer_steps
        expected_readouts_total += case_readout_count

        reuse_info = None
        if payload.get("reused_prediction_sha256") is not None:
            reused_case_ids.add(cid)
            reuse_info = _audit_reuse(payload, receipt, cid, runtime_sha, torch, np)
            reused_reports.append(reuse_info)

        per_case.append({
            "case_id": cid,
            "dataset": rec["dataset"],
            "source_case_path": source_rel,
            "source_case_file_sha256": rec["source_case_file_sha256"],
            "source_receipt_sha256": source_receipt_sha,
            "source_case_sha256": ref["source_case_sha256"],
            "source_model_state_sha256": payload["source_model_state_sha256"],
            "pixel_sha256": payload["pixel_sha256"],
            "reference_support_count": support_n,
            "final_fit_count": case_fit_count,
            "final_optimizer_steps": case_optimizer_steps,
            "final_full_model_readout_count": case_readout_count,
            "source_state_equal_for_each_fit": support_n > 0,
            "math": case_math,
            "reuse": reuse_info,
            "student_payload_sha256": item["sha256"],
            "student_receipt_sha256": _sha(receipt_path),
        })

    if (len(per_case) != EXPECTED_CASES or expected_fit_total != EXPECTED_FITS
            or expected_readouts_total != EXPECTED_FITS
            or expected_steps_total != int(student_barrier.get("optimizer_steps", -1))):
        raise ValueError("final case/fit/readout/optimizer-step totals differ from the student barrier")
    if fit_counts != {"WF_common": 19, "DINO_common": 19}:
        raise ValueError("finalized fit role counts differ from the expected 19+19")
    if len(reused_case_ids) != EXPECTED_REUSED_CASES:
        raise ValueError("exact reused output count differs from 10")
    reused_fit_count = sum(
        sum(value is not None for value in _torch_load(STUDENT_DIR / f"{cid}.pt", torch)["fits"].values())
        for cid in reused_case_ids
    )
    if reused_fit_count != EXPECTED_REUSED_FITS:
        raise ValueError("the 10 exact-reuse payloads do not contain the expected 16 historical fits")

    # Audit the two preserved failed HC11/WF fit checkpoints as extra attempts,
    # separate from the final 38 fits and the 10 reused case payloads.
    partial_reports = []
    for capture_path, partial_dir in zip(PARTIAL_CAPTURE_PATHS, PARTIAL_ATTEMPTS):
        capture = _read(capture_path)
        if (capture.get("status") != "preserved" or capture.get("GT_read") is not False
                or capture.get("partial_case", capture.get("new_partial_source")) != "HC11"):
            raise ValueError(f"unexpected preserved partial-fit capture: {_relative(capture_path)}")
        if capture.get("partial_fit_retained") is False:
            raise ValueError(f"preserved partial-fit capture does not retain a fit: {_relative(capture_path)}")
        rec, ref = roster["HC11"], refs["HC11"]
        if not ref.get("common_wf_anchors"):
            raise ValueError("HC11/WF partial fit has no locked common anchors")
        partial_reports.append(_audit_partial_checkpoint_dir(
            partial_dir, "HC11", "WF_common", rec, ref, design["configs"][rec["dataset"]],
            ref_sha, torch, np, math_reader,
        ))
    if len(partial_reports) != EXPECTED_PARTIAL_FITS:
        raise ValueError("preserved failed attempts differ from the expected two partial HC11 fits")

    result = {
        "schema": "stvg-reference-swap-independent-optimizer-math-readback-v1",
        "status": "pass",
        "scope": "saved E1 current5 Gaussian-IoU rollout/reward/q/loss/Adam/state-rollback arithmetic; exact Source initial-state binding; parent-arm reuse; preserved recovery-prefix comparison; no model inference, annotation read, scoring, or independent backward replay",
        "finalized_case_count": len(per_case),
        "finalized_fit_count": expected_fit_total,
        "finalized_optimizer_steps": expected_steps_total,
        "finalized_full_model_readout_count": expected_readouts_total,
        "finalized_fit_counts_by_role": fit_counts,
        "common_support_case_count": locks["common_cases"],
        "empty_common_support_case_count": locks["empty_cases"],
        "common_anchor_count": locks["common_count"],
        "exact_reused_case_count": len(reused_case_ids),
        "exact_reused_fit_count": reused_fit_count,
        "reused_cases_are_new_model_execution": False,
        "preserved_partial_fit_attempt_count": len(partial_reports),
        "unique_saved_fit_artifact_count_including_partials": expected_fit_total + len(partial_reports),
        "partial_fit_full_model_readout_artifact_verified_by_this_reader": False,
        "partial_fit_readout_claim_source": "recovery CAPTURE metadata records partial-fit retention; this CPU math reader independently verifies the saved optimizer path, not a separately serialized full-model readout artifact",
        "optimizer_math_counts": count_totals,
        "max_abs_errors": errors,
        "CPU_only": True,
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "backward_replayed": False,
        "runtime_sha256": runtime_sha,
        "design_lock_sha256": locks["design_sha"],
        "roster_lock_sha256": locks["roster_sha"],
        "reference_lock_sha256": ref_sha,
        "teacher_barrier_sha256": locks["teacher_sha"],
        "student_barrier_sha256": locks["student_sha"],
        "parent_student_barrier_sha256": locks["parent_sha"],
        "parent_reference_lock_sha256": locks["parent_ref_sha"],
        "validated_math_reader_sha256": _sha(MATH_HELPER_PATH),
        "auditor_sha256": _sha(Path(__file__)),
        "cases": per_case,
        "exact_reuse_readbacks": reused_reports,
        "preserved_partial_fit_readbacks": partial_reports,
        "time": time.time(),
    }
    out = Path(output) if output else BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json"
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _self_test() -> dict[str, Any]:
    """Synthetic structure checks only; no experiment payloads or tensors loaded."""
    import numpy as np
    import torch
    from scripts import review_stvg_reference_carry_optimizer_v1 as math_reader

    if torch.cuda.is_initialized():
        raise RuntimeError("self-test refuses initialized CUDA")
    fit = {"reference_role": "original_dino_same_frame_geometry", "sentinel": [1, 2]}
    adapted = _fit_role_for_math(fit, "DINO_common")
    assert adapted["reference_role"] == "fixed_weight_free_top1_teacher"
    assert fit["reference_role"] == "original_dino_same_frame_geometry"
    assert _sha(MATH_HELPER_PATH) == MATH_HELPER_SHA256
    left = {"a": np.asarray([1.0, 2.0], dtype=np.float32),
            "b": torch.tensor([3.0]), "meta": {"k": ["x", 4]}}
    right = copy.deepcopy(left)
    assert _eq(left, right, torch, np)
    right["a"][1] = 2.5
    assert not _eq(left, right, torch, np)
    checkpoint = {
        "next_step": 1,
        "path": [{"step": 0, "state_before": {"x": torch.tensor([0.])},
                  "boxes_before": torch.zeros((1, 4)), "state_after": {"x": torch.tensor([0.])},
                  "boxes_after": torch.zeros((1, 4))}],
        "rounds": [{"step": 0}],
    }
    assert checkpoint["next_step"] == len(checkpoint["path"]) == len(checkpoint["rounds"]) == 1
    assert callable(math_reader._audit_fit)
    return {"status": "pass", "checks": 8, "GT_read": False,
            "model_loaded": False, "GPU_used": False,
            "scope": "adapter/tree/checkpoint structure only; no optimizer math executed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", help="new output path; existing files are never overwritten")
    args = parser.parse_args()
    result = _self_test() if args.self_test else _run(args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("cases", "exact_reuse_readbacks", "preserved_partial_fit_readbacks")},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
