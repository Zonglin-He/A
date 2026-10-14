"""Matched-support Source/R1/oracle spatial OPD diagnostic worker.

Importing this module is inert.  The worker consumes the root-sealed
REFERENCE_LOCK.json, which deliberately contains GT-derived offline
diagnostic references.  It never opens annotation files itself.  The GPU
launcher and independent reviewer own execution authorization.
"""
from __future__ import annotations

import collections
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_temporal_caption_query_v1 import catalog, read, sha, write, inference_guard
from scripts import run_stvg_adjacent_motion_student_v1 as frozen_helpers
from scripts import run_stvg_short_window_student_v1 as source_helpers

BASE = ROOT / "artifacts/stvg_matched_support_reference_v1"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
REFERENCE_LOCK_PATH = BASE / "REFERENCE_LOCK.json"
RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
OUT_BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
STATUS_PATH = BASE / "STUDENT_GPU_STATUS.json"

SHORT_BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
OLD_T_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
MANIFEST_PATH = SHORT_BASE / "PREPARED_MANIFEST.json"
TRACKLET_BARRIER_PATH = SHORT_BASE / "GLOBAL_TRACKLET_BARRIER.json"
FEATURE_BARRIER_PATH = SHORT_BASE / "FEATURE_CAPTURE_BARRIER.json"
TEACHER_BARRIER_PATH = SHORT_BASE / "TEACHER_GLOBAL_BARRIER.json"
TEACHER_JOB_LOCK_PATH = SHORT_BASE / "TEACHER_JOB_LOCK.json"
SHORT_STUDENT_BARRIER_PATH = SHORT_BASE / "STUDENT_GLOBAL_BARRIER.json"
OLD_T_BARRIER_PATH = OLD_T_BASE / "STUDENT_GLOBAL_BARRIER.json"

CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
ROLES = ("source", "r1", "pool_oracle", "gt_oracle")
HELPER_ROLES = {"source": "source", "r1": "teacher",
                "pool_oracle": "offline_oracle", "gt_oracle": "offline_oracle"}
STEPS = 5
ACTIVE_PARAMETERS = 1792
SCHEMA = "stvg-matched-support-student-v1"
BARRIER_SCHEMA = "stvg-matched-support-student-barrier-v1"

REQUIRED_STATIC_PINS = (
    "scripts/run_stvg_matched_support_reference_v1.py",
    "scripts/launch_stvg_matched_support_reference_v1.py",
    "scripts/run_stvg_adjacent_motion_student_v1.py",
    "scripts/run_stvg_short_window_student_v1.py",
    "scripts/stvg_temporal_caption_query_v1.py",
    "scripts/run_decota_paper_main_v1.py",
    "scripts/stvg_mllm_feedback_common_v1.py",
    "scripts/run_tastvg_evidence_vulnerability_v2.py",
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py",
    "vg_tta/decota_spatial_opd_tunable_v1.py",
    "vg_tta/stvg_opd_frozen_mask_cache012.py",
    "vg_tta/stvg_opd_frozen_visual_host011.py",
    "vg_tta/stvg_opd_frozen_attention_lifetime010.py",
    "vg_tta/stvg_opd_frozen_module_cpu008.py",
    "vg_tta/c1_enabling_tricks_v1.py",
    "vg_tta/tastvg_decota_critic_p0_v1.py",
    "methods/decota_final_simplified_v1/objectives.py",
    "methods/decota_final_simplified_v1/tensors.py",
    "methods/decota_final_simplified_v1/backbone.py",
    "methods/decota_spatial_opd_v1/configs.json",
    "protocols/stvg_matched_support_reference_v1.md",
    "artifacts/stvg_matched_support_reference_v1/ROSTER_LOCK.json",
    "artifacts/stvg_matched_support_reference_v1/REFERENCE_LOCK.json",
    "artifacts/stvg_short_event_tracklets_v1/ROSTER_LOCK.json",
    "artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json",
    "artifacts/stvg_short_event_tracklets_v1/GLOBAL_TRACKLET_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/FEATURE_CAPTURE_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/TEACHER_GLOBAL_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/TEACHER_JOB_LOCK.json",
    "artifacts/stvg_short_event_tracklets_v1/STUDENT_GLOBAL_BARRIER.json",
    "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json",
)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _records(doc: Mapping[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    rows = doc.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}.{key} must be a list of objects")
    return rows


def _safe_root_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("locked source binding contains an unsafe relative path")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("locked source binding escapes the project root")
    return path


def _require_digest_binding(label: str, actual: Any, expected: Any) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise ValueError(f"locked SHA binding differs: {label}")


def _reference_support_signature(references: Mapping[str, Mapping[str, Any]],
                                 centers: Sequence[Mapping[str, Any]],
                                 *, case_id: str) -> list[tuple[int, int]]:
    """Validate all four reference rows and return their common ordered support."""
    if set(references) != set(ROLES):
        raise ValueError(f"reference role set differs from the fixed four roles: {case_id}")
    center_by_position = {int(c["source_grid_position"]): c for c in centers}
    if len(center_by_position) != len(centers):
        raise ValueError(f"manifest repeats a source-grid position: {case_id}")
    signatures: dict[str, list[tuple[int, int]]] = {}
    for role in ROLES:
        ref = references[role]
        anchors = ref.get("anchors")
        if not isinstance(anchors, list) or len(anchors) > 4:
            raise ValueError(f"{role} must have zero to four locked anchors: {case_id}")
        if ref.get("offline_diagnostic") is not True:
            raise ValueError(f"{role} is not explicitly marked offline diagnostic: {case_id}")
        no_update, eligible = ref.get("no_update"), ref.get("eligible_for_update")
        if (not isinstance(no_update, bool) or not isinstance(eligible, bool)
                or no_update != (not anchors) or eligible != bool(anchors)):
            raise ValueError(f"{role} no-update flags disagree with anchor emptiness: {case_id}")
        if not isinstance(ref.get("reason"), str):
            raise ValueError(f"{role} lacks a locked reason string: {case_id}")
        signature: list[tuple[int, int]] = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError(f"{role} anchor is not an object: {case_id}")
            pos, physical = anchor.get("position"), anchor.get("physical_frame_id")
            if (isinstance(pos, bool) or not isinstance(pos, int)
                    or isinstance(physical, bool) or not isinstance(physical, int)):
                raise ValueError(f"{role} anchor positions must be integer source-grid/physical IDs: {case_id}")
            center = center_by_position.get(pos)
            if center is None or int(center["center_physical_frame_id"]) != physical:
                raise ValueError(f"{role} position/physical-frame mapping differs: {case_id}")
            slot = anchor.get("slot")
            if slot is not None and (isinstance(slot, bool) or not isinstance(slot, int)
                                     or slot != int(center["slot"])):
                raise ValueError(f"{role} QA slot disagrees with its source-grid position: {case_id}")
            box = anchor.get("box_cxcywh")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"{role} anchor needs a normalized cxcywh box: {case_id}")
            if any(isinstance(v, bool) for v in box):
                raise ValueError(f"{role} box coordinates cannot be booleans: {case_id}")
            vals = [float(v) for v in box]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"{role} box contains a nonfinite coordinate: {case_id}")
            cx, cy, width, height = vals
            if not (width > 0 and height > 0 and 0 <= cx - width / 2 < cx + width / 2 <= 1
                    and 0 <= cy - height / 2 < cy + height / 2 <= 1):
                raise ValueError(f"{role} normalized box is outside image bounds: {case_id}")
            if not isinstance(anchor.get("is_Source"), bool):
                raise ValueError(f"{role} anchor lacks an explicit Source flag: {case_id}")
            label_assisted = role in ("pool_oracle", "gt_oracle")
            if anchor.get("offline_label_assisted") is not label_assisted:
                raise ValueError(f"{role} label-assistance marker differs from the locked role: {case_id}")
            if not isinstance(anchor.get("teacher_job_id"), str):
                raise ValueError(f"{role} anchor lacks its frozen teacher job ID: {case_id}")
            if not _is_sha256(anchor.get("teacher_output_sha256")):
                raise ValueError(f"{role} anchor lacks a teacher-output SHA: {case_id}")
            signature.append((pos, physical))
        if signature != sorted(signature) or len({p for p, _ in signature}) != len(signature):
            raise ValueError(f"{role} anchors must have unique increasing source-grid positions: {case_id}")
        signatures[role] = signature
    expected = signatures[ROLES[0]]
    if any(signatures[role] != expected for role in ROLES[1:]):
        raise ValueError(f"four references do not share identical ordered support: {case_id}")
    return expected


def _validate_role_provenance(references: Mapping[str, Mapping[str, Any]],
                              teacher_digest_by_job: Mapping[str, str], *, case_id: str) -> None:
    for role in ROLES:
        for anchor in references[role]["anchors"]:
            slot = int(anchor["slot"])
            job_id = f"{case_id}_window{slot:02d}_base"
            if anchor["teacher_job_id"] != job_id:
                raise ValueError(f"{role} anchor is not bound to the base teacher call: {case_id}")
            if teacher_digest_by_job.get(job_id) != anchor["teacher_output_sha256"]:
                raise ValueError(f"{role} anchor output SHA does not bind its exact teacher job: {case_id}")
            candidate = anchor.get("candidate_id")
            if role in ("source", "r1", "pool_oracle"):
                if isinstance(candidate, bool) or not isinstance(candidate, int):
                    raise ValueError(f"{role} anchor must identify its candidate: {case_id}")
                if anchor.get("selected_candidate_id") != candidate:
                    raise ValueError(f"{role} selected/candidate IDs disagree: {case_id}")
                if anchor.get("is_Source") is not (candidate == 1):
                    raise ValueError(f"{role} Source flag disagrees with candidate ID: {case_id}")
                if not isinstance(anchor.get("tracklet_id"), str):
                    raise ValueError(f"{role} candidate anchor lacks its tracklet binding: {case_id}")
            elif role == "gt_oracle":
                if candidate is not None or anchor.get("selected_candidate_id") is not None:
                    raise ValueError(f"gt_oracle must remain an explicit GT box, not a candidate: {case_id}")
                if anchor.get("is_Source") is not False:
                    raise ValueError(f"gt_oracle must be marked non-Source: {case_id}")


def _fit_spec(case_id: str, role: str, config: Mapping[str, Any],
              reference: Mapping[str, Any]) -> dict[str, Any]:
    if role not in ROLES or not reference.get("anchors"):
        raise ValueError("fit requires one authorized nonempty reference role")
    cfg = dict(config)
    cfg["steps"] = STEPS
    return {
        "key": "caption_query_fixed_space|" + case_id,
        "scope": "joint",
        "reward_kind": "iou",
        "reference_role": HELPER_ROLES[role],
        "reference_sha256": _canonical_sha(reference),
        "config": cfg,
        "steps": STEPS,
        "active_parameters": ACTIVE_PARAMETERS,
    }


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in state.items()}


def _no_update_arms(f_arms: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]):
    """Copy F/T exactly for empty roles without fit or final-readout calls."""
    arms = {"F": copy.deepcopy(f_arms["F"]), "T": copy.deepcopy(f_arms["T"])}
    receipts: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        if references[role]["anchors"]:
            continue
        arms[role] = copy.deepcopy(arms["F"])
        arms[f"T_{role}"] = copy.deepcopy(arms["T"])
        receipts[role] = {"status": "exact_F_no_update", "no_update": True,
                          "optimizer_steps": 0, "final_redecode_calls": 0,
                          "exact_F_reuse": True}
        receipts[f"T_{role}"] = {"status": "exact_T_no_update", "no_update": True,
                                  "optimizer_steps": 0, "final_redecode_calls": 0,
                                  "exact_T_reuse": True}
    return arms, receipts


def _required_fit_pins() -> set[str]:
    from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import SOURCE_BINDINGS
    return set(SOURCE_BINDINGS) | {"scripts/stvg_video_global_spatial_fit_fp32_geometry001.py"}


def _verify_receipt_files(rows: Sequence[Mapping[str, Any]], expected_paths: set[str], label: str) -> None:
    actual_paths: set[str] = set()
    for row in rows:
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or path in actual_paths or not _is_sha256(digest):
            raise ValueError(f"{label} has malformed or duplicate output receipt paths")
        actual_paths.add(path)
        target = _safe_root_path(path)
        if not target.is_file() or sha(target) != digest:
            raise ValueError(f"{label} output receipt SHA differs: {path}")
        nbytes = row.get("bytes")
        if nbytes is not None and (isinstance(nbytes, bool) or not isinstance(nbytes, int) or target.stat().st_size != nbytes):
            raise ValueError(f"{label} output receipt byte count differs: {path}")
    if actual_paths != expected_paths:
        raise ValueError(f"{label} does not contain its exact eight-case output set")


def preflight_locked_inputs() -> dict[str, Any]:
    """Validate only locked JSON/hash metadata; never loads model/GT/media."""
    roster_doc = read(ROSTER_PATH)
    manifest = read(MANIFEST_PATH)
    tracklet_barrier = read(TRACKLET_BARRIER_PATH)
    feature_barrier = read(FEATURE_BARRIER_PATH)
    teacher_barrier = read(TEACHER_BARRIER_PATH)
    short_student_barrier = read(SHORT_STUDENT_BARRIER_PATH)
    reference_lock = read(REFERENCE_LOCK_PATH)
    runtime = read(RUNTIME_PATH)

    if roster_doc.get("status") != "locked" or roster_doc.get("GT_read") is not False:
        raise ValueError("new ROSTER_LOCK is not a locked no-GT source roster")
    roster = _records(roster_doc, "records", "ROSTER_LOCK")
    if [row.get("case_id") for row in roster] != list(CASE_IDS):
        raise ValueError("ROSTER_LOCK must retain the exact stable old-eight order")
    if sha(ROSTER_PATH) != sha(SHORT_BASE / "ROSTER_LOCK.json"):
        raise ValueError("new source roster is not the exact frozen old-eight roster")
    roster_by_id = {row["case_id"]: row for row in roster}

    if (manifest.get("status") != "prepared_manifest_not_executed"
            or manifest.get("GT_read") is not False
            or manifest.get("model_loaded") is not False
            or manifest.get("GPU_used") is not False):
        raise ValueError("source manifest is not the frozen no-GT prepared manifest")
    manifest_rows = _records(manifest, "cases", "PREPARED_MANIFEST")
    if [row.get("case_id") for row in manifest_rows] != list(CASE_IDS):
        raise ValueError("prepared manifest differs from the exact old-eight order")
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}

    if (tracklet_barrier.get("status") != "complete" or tracklet_barrier.get("GT_read") is not False
            or tracklet_barrier.get("count") != 8
            or tracklet_barrier.get("case_ids") != list(CASE_IDS)):
        raise ValueError("short-window tracklet barrier is not complete for the old-eight set")
    track_rows = _records(tracklet_barrier, "case_records", "GLOBAL_TRACKLET_BARRIER")
    if [row.get("case_id") for row in track_rows] != list(CASE_IDS):
        raise ValueError("short-window tracklet case receipts differ from old-eight")
    tracklet_by_id = {row["case_id"]: row for row in track_rows}

    if (feature_barrier.get("status") != "sealed" or feature_barrier.get("GT_read") is not False
            or feature_barrier.get("GPU_used") is not True or feature_barrier.get("count") != 8):
        raise ValueError("frozen feature metadata is not the sealed no-GT old-eight set")
    feature_rows = _records(feature_barrier, "records", "FEATURE_CAPTURE_BARRIER")
    if [row.get("case_id") for row in feature_rows] != list(CASE_IDS):
        raise ValueError("feature receipts differ from the exact old-eight order")
    feature_by_id = {row["case_id"]: row for row in feature_rows}

    if (teacher_barrier.get("status") not in {"complete", "sealed"}
            or teacher_barrier.get("GT_read") is not False or teacher_barrier.get("count") != 64):
        raise ValueError("sealed teacher barrier does not match the fixed no-GT 64-call source")
    teacher_files = _records(teacher_barrier, "files", "TEACHER_GLOBAL_BARRIER")
    if len(teacher_files) != 64:
        raise ValueError("teacher barrier must bind exactly 64 response receipts")
    teacher_digest_by_job = {}
    for row in teacher_files:
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not _is_sha256(digest):
            raise ValueError("teacher output receipts are malformed")
        basename = Path(path).name
        if basename.endswith(".json"):
            job_id = basename[:-5]
            if job_id in teacher_digest_by_job:
                raise ValueError("teacher barrier contains duplicate response basenames")
            teacher_digest_by_job[job_id] = digest

    if (short_student_barrier.get("status") != "complete"
            or short_student_barrier.get("GT_read") is not False
            or short_student_barrier.get("count") != 8):
        raise ValueError("old short-window student barrier is not complete and no-GT")
    short_files = _records(short_student_barrier, "files", "SHORT_STUDENT_GLOBAL_BARRIER")
    _verify_receipt_files(short_files, {
        f"artifacts/stvg_short_event_tracklets_v1/student/{cid}.pt" for cid in CASE_IDS
    }, "old short-window student barrier")

    old_t_barrier = read(OLD_T_BARRIER_PATH)
    if (old_t_barrier.get("status") not in {"complete", "sealed"}
            or old_t_barrier.get("GT_read") is not False or old_t_barrier.get("count") != 8):
        raise ValueError("old T baseline barrier is not complete and no-GT")
    _verify_receipt_files(_records(old_t_barrier, "files", "OLD_T_STUDENT_GLOBAL_BARRIER"), {
        f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
        for cid in CASE_IDS
    }, "old T baseline barrier")

    if (reference_lock.get("schema") != "stvg-matched-support-reference-lock-v1"
            or reference_lock.get("status") != "locked"
            or reference_lock.get("scope") != "offline_diagnostic_only"
            or reference_lock.get("GT_read") is not True
            or reference_lock.get("GT_used_for_oracle_references") is not True
            or reference_lock.get("GT_used_for_hyperparameters") is not False
            or reference_lock.get("count") != 8
            or reference_lock.get("roles") != list(ROLES)):
        raise ValueError("REFERENCE_LOCK is not the fixed offline diagnostic four-role ABI")
    reference_rows = _records(reference_lock, "records", "REFERENCE_LOCK")
    if [row.get("case_id") for row in reference_rows] != list(CASE_IDS):
        raise ValueError("REFERENCE_LOCK must retain the exact stable old-eight order")
    reference_by_id = {row["case_id"]: row for row in reference_rows}

    top_bindings = {
        "roster_lock_sha256": sha(ROSTER_PATH),
        "manifest_sha256": sha(MANIFEST_PATH),
        "tracklet_barrier_sha256": sha(TRACKLET_BARRIER_PATH),
        "old_student_barrier_sha256": sha(SHORT_STUDENT_BARRIER_PATH),
    }
    for name, digest in top_bindings.items():
        _require_digest_binding(name, reference_lock.get(name), digest)
    source_bindings = reference_lock.get("source_bindings")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("REFERENCE_LOCK source_bindings must be a nonempty SHA map")
    for rel, digest in source_bindings.items():
        target = _safe_root_path(rel)
        if not _is_sha256(digest) or not target.is_file() or sha(target) != digest:
            raise ValueError(f"REFERENCE_LOCK source binding differs: {rel}")
    expected_source_files = {
        "artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json",
        "artifacts/stvg_short_event_tracklets_v1/GLOBAL_TRACKLET_BARRIER.json",
        "artifacts/stvg_short_event_tracklets_v1/TEACHER_GLOBAL_BARRIER.json",
        "artifacts/stvg_short_event_tracklets_v1/TEACHER_JOB_LOCK.json",
        "artifacts/stvg_short_event_tracklets_v1/ROOT_POSTSEAL_REPORT.json",
        "artifacts/stvg_short_event_tracklets_v1/ROOT_POSTSEAL_INDEPENDENT_READBACK.json",
        "protocols/stvg_matched_support_reference_v1.md",
    }
    if not expected_source_files <= set(source_bindings):
        raise ValueError("REFERENCE_LOCK omits required source-audit binding paths")

    expected_fits = 0
    for cid in CASE_IDS:
        roster_row, manifest_row = roster_by_id[cid], manifest_by_id[cid]
        track_row, feature_row = tracklet_by_id[cid], feature_by_id[cid]
        ref_row = reference_by_id[cid]
        if (roster_row.get("dataset") not in ("hc2", "vidstg")
                or roster_row.get("cohort") != "old8" or roster_row.get("P0") is not True):
            raise ValueError(f"source roster row is not an old-eight P0 case: {cid}")
        if (manifest_row.get("GT_read") is not False
                or manifest_row.get("source_case_sha256") != roster_row.get("source_case_sha256")):
            raise ValueError(f"source manifest/roster binding differs: {cid}")
        expected_case_bindings = {
            "dataset": roster_row.get("dataset"),
            "source_case_sha256": roster_row.get("source_case_sha256"),
            "source_model_state_sha256": manifest_row.get("source_model_state_sha256"),
            "manifest_case_sha256": _canonical_sha(manifest_row),
            "tracklet_case_sha256": track_row.get("sha256"),
            "canonical_request_sha256": manifest_row.get("canonical_request_sha256"),
            "canonical_video_sha256": manifest_row.get("canonical_video_sha256"),
            "rgb_npz_sha256": track_row.get("rgb_npz_sha256"),
            "feature_case_sha256": feature_row.get("sha256"),
        }
        for name, value in expected_case_bindings.items():
            if ref_row.get(name) != value:
                raise ValueError(f"REFERENCE_LOCK per-case binding differs: {cid}:{name}")
        if not _is_sha256(ref_row.get("feature_maps_sha256")):
            raise ValueError(f"REFERENCE_LOCK feature-map digest is malformed: {cid}")
        if ref_row.get("GT_read") is not True or ref_row.get("scope") != "offline_diagnostic_only":
            raise ValueError(f"case reference provenance is not explicitly offline GT-assisted: {cid}")
        support = _reference_support_signature(
            ref_row.get("references", {}), manifest_row.get("centers", []), case_id=cid,
        )
        _validate_role_provenance(ref_row["references"], teacher_digest_by_job, case_id=cid)
        if support:
            expected_fits += len(ROLES)
    if reference_lock.get("expected_fit_count") != expected_fits:
        raise ValueError("REFERENCE_LOCK expected_fit_count differs from matched nonempty roles")

    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("STUDENT_RUNTIME must be root-locked and declare no direct GT read")
    if (runtime.get("scope") is not None
            and runtime.get("scope") not in ("offline_diagnostic_only", "matched_support_reference")):
        raise ValueError("STUDENT_RUNTIME scope is not the offline matched-reference experiment")
    if runtime.get("offline_diagnostic_only") is not True:
        raise ValueError("STUDENT_RUNTIME cannot promote this reference diagnostic to deployment")
    if runtime.get("reference_lock_sha256") != sha(REFERENCE_LOCK_PATH):
        raise ValueError("STUDENT_RUNTIME does not bind REFERENCE_LOCK")
    if runtime.get("roster_lock_sha256") != sha(ROSTER_PATH):
        raise ValueError("STUDENT_RUNTIME does not bind ROSTER_LOCK")
    fit_scope = runtime.get("fit_scope", runtime.get("scope_fit"))
    reward = runtime.get("reward", runtime.get("reward_objective"))
    if (runtime.get("steps") != STEPS or runtime.get("active_parameters") != ACTIVE_PARAMETERS
            or fit_scope not in (None, "joint")
            or reward not in (None, "original_gaussian_spatial_iou", "iou")):
        raise ValueError("STUDENT_RUNTIME differs from the fixed original five-step IoU OPD contract")
    if runtime.get("expected_fit_count") != expected_fits:
        raise ValueError("STUDENT_RUNTIME expected_fit_count differs from the sealed four-role references")
    pins, protected = runtime.get("pins"), runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise ValueError("STUDENT_RUNTIME pin maps are malformed")
    # The shared capture/readout code was frozen together with its original
    # full source pin set.  Retain that set and add this worker's own bindings.
    required = set(REQUIRED_STATIC_PINS) | set(source_helpers.REQUIRED_STATIC_PINS) | _required_fit_pins()
    required.update(
        f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
        for cid in CASE_IDS
    )
    required.update(
        f"artifacts/stvg_short_event_tracklets_v1/student/{cid}.pt" for cid in CASE_IDS
    )
    missing = sorted(required - set(pins))
    if missing:
        raise ValueError(f"STUDENT_RUNTIME omits reused scientific source pins: {missing}")
    if "methods/CURRENT_METHOD.json" not in protected:
        raise ValueError("STUDENT_RUNTIME must protect the current deployed-method registry")
    for rel, digest in {**pins, **protected}.items():
        target = _safe_root_path(rel)
        if not _is_sha256(digest) or not target.is_file() or sha(target) != digest:
            raise ValueError(f"STUDENT_RUNTIME source pin differs: {rel}")
    if OUT_BARRIER_PATH.exists():
        raise FileExistsError("STUDENT_GLOBAL_BARRIER already exists; refusing a second run")

    return {
        "roster": roster,
        "manifest_by_id": manifest_by_id,
        "tracklet_by_id": tracklet_by_id,
        "reference_by_id": reference_by_id,
        "runtime": runtime,
        "reference_lock": reference_lock,
    }


def _fit_role(*, case_id: str, role: str, reference: Mapping[str, Any],
              anchors: Sequence[Mapping[str, Any]], initial: Mapping[str, Any], base: Any,
              frame_ids: Sequence[int], dataset: str, config: Mapping[str, Any],
              core: Any, fit_spatial_reward_opd: Any, readout: Any, model: Any,
              batch: Any, records: Any, row: Mapping[str, Any],
              expected_source_hash: str, source_logits: Sequence[Any],
              state_hash: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if role not in ROLES or not anchors:
        raise ValueError("_fit_role requires one nonempty locked role")
    spec = _fit_spec(case_id, role, config, reference)
    partial = BASE / "partial" / role / case_id
    partial.mkdir(parents=True, exist_ok=True)
    saved = sorted(partial.glob("step*.pt"), key=lambda p: int(p.stem[4:]))
    if any(not p.stem[4:].isdigit() for p in saved):
        raise ValueError(f"partial fit checkpoint name is malformed: {role}/{case_id}")
    resume = torch.load(saved[-1], map_location="cpu", weights_only=False) if saved else None

    def checkpoint(payload: Mapping[str, Any]) -> None:
        destination = partial / f"step{int(payload['next_step'])}.pt"
        if destination.exists():
            raise FileExistsError(f"refusing to replace partial checkpoint: {destination}")
        torch.save(dict(payload), destination)

    fit = fit_spatial_reward_opd(
        base, _clone_state(initial), anchors, frame_ids, spec["key"],
        dict(spec["config"]), scope=spec["scope"], reward_kind=spec["reward_kind"],
        checkpoint_callback=checkpoint, reference_role=spec["reference_role"],
        reference_sha256=spec["reference_sha256"], resume_state=resume,
    )
    optimizer_steps = source_helpers._validate_fit_result(fit, role=role)
    out = readout(model, batch, frame_ids, records, fit, row["parses"]["subject"], core, torch)
    if not torch.equal(out["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu()):
        raise AssertionError(f"{role} final full readout differs from cached decoder boxes")
    if not frozen_helpers._same_logits(out["logits"], source_logits):
        raise AssertionError(f"{role} changed the frozen source logits")
    if state_hash(model.state_dict()) != expected_source_hash:
        raise AssertionError(f"{role} mutated the full frozen source model")
    if optimizer_steps != int(fit["optimizer_steps"]):
        raise AssertionError(f"{role} optimizer-step validation differs")
    return fit, out


def _validate_saved_result(saved: Mapping[str, Any], *, rec: Mapping[str, Any],
                           references: Mapping[str, Any], lock_sha: str) -> tuple[int, int, int]:
    cid = rec["case_id"]
    if (saved.get("schema") != SCHEMA or saved.get("case_id") != cid
            or saved.get("source_case_sha256") != rec["source_case_sha256"]
            or saved.get("reference_lock_sha256") != lock_sha
            or saved.get("reference_records") != references
            or saved.get("source_restored") is not True
            or saved.get("GT_derived_reference_consumed") is not True
            or saved.get("offline_diagnostic_only") is not True):
        raise ValueError(f"existing student result is not bound to this locked case/reference: {cid}")
    expected_roles = {role for role in ROLES if references[role]["anchors"]}
    if set(saved.get("fits", {})) != expected_roles or set(saved.get("readouts", {})) != expected_roles:
        raise ValueError(f"existing student result fit/readout role set differs: {cid}")
    expected_arms = {"F", "T", *ROLES, *(f"T_{role}" for role in ROLES)}
    if set(saved.get("arms", {})) != expected_arms:
        raise ValueError(f"existing student result arm set differs: {cid}")
    fits = saved["fits"]
    fit_count = len(fits)
    steps = 0
    for role, fit in fits.items():
        steps += source_helpers._validate_fit_result(fit, role=role)
    readouts = len(saved["readouts"])
    if (saved.get("actual_fit_count") != fit_count
            or saved.get("actual_final_redecode_calls") != readouts
            or fit_count != readouts):
        raise ValueError(f"existing student result counters differ: {cid}")
    return fit_count, steps, readouts


def run_locked_worker() -> None:
    inputs = preflight_locked_inputs()
    roster = inputs["roster"]
    manifest_by_id = inputs["manifest_by_id"]
    reference_by_id = inputs["reference_by_id"]
    runtime = inputs["runtime"]

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
    from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import (
        fit_spatial_reward_opd, verify_source_bindings,
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("worker requires a fresh CUDA context from its own launcher")
    verify_source_bindings()
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    install_clean_loader()
    sys.addaudithook(inference_guard)

    lock_sha = sha(REFERENCE_LOCK_PATH)
    model = None
    current_source = None
    output_files: list[dict[str, Any]] = []
    total_fits = total_steps = total_readouts = 0
    started = time.perf_counter()
    write(STATUS_PATH, {"status": "running", "stage": "matched_support_reference",
                        "done": 0, "total": 8, "pid": os.getpid(),
                        "GT_read": False, "GT_derived_reference_consumed": True,
                        "time": time.time()}, mutable=True)

    for number, rec in enumerate(roster, 1):
        cid = str(rec["case_id"])
        manifest_row = manifest_by_id[cid]
        reference_row = reference_by_id[cid]
        references = reference_row["references"]
        destination = BASE / "student" / f"{cid}.pt"
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=False)
            fits_n, steps_n, readouts_n = _validate_saved_result(
                saved, rec=rec, references=references, lock_sha=lock_sha,
            )
            output_files.append({"path": destination.relative_to(ROOT).as_posix(),
                                 "sha256": sha(destination), "bytes": destination.stat().st_size})
            total_fits += fits_n
            total_steps += steps_n
            total_readouts += readouts_n
            continue

        target_source = catalog(rec)[0]["source"]
        if model is not None and current_source != target_source:
            del model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        (model, current_source, case, catalog_data, row, frames, frame_ids,
         records, batch, base, native, expected_source_hash, pixel_sha,
         capture_seconds) = frozen_helpers._capture_source(
            rec, model, current_source, torch, collections, model_for, frames_for,
            raw_row, frozen_forward, prediction, state_hash,
        )
        if (case["_source_case_sha256"] != reference_row["source_case_sha256"]
                or expected_source_hash != reference_row["source_model_state_sha256"]):
            raise AssertionError(f"cold source identity differs from REFERENCE_LOCK: {cid}")
        # The manifest canonical hash covers teacher-video pixels; student source-grid
        # frames are independently bound below and by the sealed old T capture.
        if pixel_sha != case["pixel_sha256"]:
            raise AssertionError(f"decoded RGB SHA differs from frozen source case: {cid}")
        if _canonical_sha(list(map(int, frame_ids))) != manifest_row.get("source_grid_frame_ids_sha256"):
            raise AssertionError(f"captured source-grid frame IDs differ from manifest: {cid}")

        support = _reference_support_signature(
            references, manifest_row["centers"], case_id=cid,
        )
        for role in ROLES:
            for anchor in references[role]["anchors"]:
                pos = int(anchor["position"])
                if not 0 <= pos < len(frame_ids) or int(frame_ids[pos]) != int(anchor["physical_frame_id"]):
                    raise AssertionError(f"{role} anchor does not map to captured physical frame: {cid}")
        native_boxes = frozen_helpers._xyxy(native["boxes"])
        native_interval = list(case["Native_interval"])
        projection_interval = list(case["Projection_interval"])
        bounds = list(case["video_bounds"])
        arms = {"F": frozen_helpers._arm(native_boxes, native_interval)}
        old = frozen_helpers._load_old_temporal_case(
            cid, rec["source_case_sha256"], expected_source_hash, frame_ids,
            pixel_sha, native_boxes, native_interval, projection_interval, bounds, torch,
        )
        t_arm = copy.deepcopy(old["arms"]["T"])
        if not np.array_equal(t_arm["boxes"], native_boxes):
            raise AssertionError("frozen old T boxes must exactly reuse F space")
        arms["T"] = t_arm
        for role in ROLES:
            if references[role]["anchors"]:
                continue
            arms[role] = copy.deepcopy(arms["F"])
            arms[f"T_{role}"] = copy.deepcopy(arms["T"])
        receipts: dict[str, dict[str, Any]] = {}
        fits: dict[str, Any] = {}
        readouts: dict[str, Any] = {}
        source_logits = native["logits"]
        source_initial = _clone_state(base.initial)
        config = core.load_original_opd_config(rec["dataset"])
        config["steps"] = STEPS
        readout = lambda *args: frozen_helpers._readout(
            *args, transient_native_masks=transient_native_masks,
            frozen_visual_host=frozen_visual_host,
            frozen_attention_storage=frozen_attention_storage,
            inactive_modules_on_host=inactive_modules_on_host,
            query_subject=query_subject,
        )

        for role in ROLES:
            reference = references[role]
            if not reference["anchors"]:
                receipts[role] = {"status": "exact_F_no_update", "no_update": True,
                                  "optimizer_steps": 0, "final_redecode_calls": 0,
                                  "exact_F_reuse": True}
                receipts[f"T_{role}"] = {"status": "exact_T_no_update", "no_update": True,
                                          "optimizer_steps": 0, "final_redecode_calls": 0,
                                          "exact_T_reuse": True}
                continue
            anchors = source_helpers._anchors_from_locked_positions(
                reference, manifest_row["centers"], frame_ids, torch,
            )
            if len(anchors) != len(reference["anchors"]):
                raise AssertionError(f"{role} lost a locked reference anchor: {cid}")
            fit, out = _fit_role(
                case_id=cid, role=role, reference=reference, anchors=anchors,
                initial=source_initial, base=base, frame_ids=frame_ids,
                dataset=rec["dataset"], config=config, core=core,
                fit_spatial_reward_opd=fit_spatial_reward_opd, readout=readout,
                model=model, batch=batch, records=records, row=row,
                expected_source_hash=expected_source_hash,
                source_logits=source_logits, state_hash=state_hash, torch=torch,
            )
            boxes = frozen_helpers._xyxy(out["boxes"])
            arms[role] = frozen_helpers._arm(boxes, native_interval)
            arms[f"T_{role}"] = frozen_helpers._arm(boxes, t_arm["interval"])
            fits[role], readouts[role] = fit, out
            steps = int(fit["optimizer_steps"])
            receipts[role] = {
                "status": "updated_full_final_readout", "no_update": False,
                "optimizer_steps": steps, "final_redecode_calls": 1,
                "reference_sha256": _canonical_sha(reference),
                "helper_reference_role": HELPER_ROLES[role],
                "reward": "original_gaussian_spatial_iou",
            }
            receipts[f"T_{role}"] = {
                "status": "updated_spatial_plus_exact_old_T_interval", "no_update": False,
                "optimizer_steps": steps, "final_redecode_calls": 0,
                "inherits_fit": role, "old_T_interval_reused": True,
            }
            total_fits += 1
            total_steps += steps
            total_readouts += 1

        if state_hash(model.state_dict()) != expected_source_hash:
            raise AssertionError(f"full frozen source changed after role fits: {cid}")
        if len(support) == 0 and (fits or readouts):
            raise AssertionError(f"empty matched-support case unexpectedly fit: {cid}")
        for role in ROLES:
            if references[role]["anchors"]:
                continue
            if (not np.array_equal(arms[role]["boxes"], arms["F"]["boxes"])
                    or arms[role]["interval"] != arms["F"]["interval"]
                    or not np.array_equal(arms[f"T_{role}"]["boxes"], arms["T"]["boxes"])
                    or arms[f"T_{role}"]["interval"] != arms["T"]["interval"]):
                raise AssertionError(f"no-update role is not an exact F/T reuse: {cid}:{role}")

        result = {
            "schema": SCHEMA, "case_id": cid, "dataset": rec["dataset"],
            "cohort": rec.get("cohort", "old8"),
            "source_case_sha256": rec["source_case_sha256"],
            "source_model_state_sha256": expected_source_hash,
            "manifest_case_sha256": reference_row["manifest_case_sha256"],
            "tracklet_case_sha256": reference_row["tracklet_case_sha256"],
            "canonical_request_sha256": reference_row["canonical_request_sha256"],
            "canonical_video_sha256": reference_row["canonical_video_sha256"],
            "frame_ids": list(map(int, frame_ids)), "pixel_sha256": pixel_sha,
            "Native_interval": native_interval, "Projection_interval": projection_interval,
            "arms": arms, "fits": fits, "readouts": readouts,
            "reference_records": copy.deepcopy(references),
            "reference_lock_sha256": lock_sha,
            "reference_receipts": receipts,
            "initial_state": core._copy_cpu(source_initial),
            "source_boxes": base.zero["boxes"].detach().cpu(),
            "source_logits": core._copy_cpu(source_logits),
            "source_restored": True,
            "source_cold_readback": {
                "status": "pass", "source_case_sha256": rec["source_case_sha256"],
                "source_model_state_sha256": expected_source_hash,
                "pixel_sha256": pixel_sha, "frame_ids_match_manifest": True,
                "initial_state_match": True, "zero_boxes_logits_match": True,
                "source_restored": True,
            },
            "temporal_exact_reuse": {
                "status": "reused_exact_old_T",
                "source_case_sha256": rec["source_case_sha256"],
                "old_student_path": (OLD_T_BASE / "student" / f"{cid}.pt").relative_to(ROOT).as_posix(),
                "old_student_sha256": sha(OLD_T_BASE / "student" / f"{cid}.pt"),
                "old_T_boxes_equal_F": True, "old_T_interval_reused": True,
            },
            "configuration": {
                "key": "caption_query_fixed_space|" + cid, "scope": "joint",
                "steps": STEPS, "active_parameters": ACTIVE_PARAMETERS,
                "reward": "original_gaussian_spatial_iou", "config": config,
                "reference_lock_sha256": lock_sha,
                "same_initial_state_for_every_role": True,
                "same_stochastic_key_for_every_role": True,
            },
            "actual_fit_count": len(fits),
            "optimizer_steps": sum(int(f["optimizer_steps"]) for f in fits.values()),
            "actual_final_redecode_calls": len(readouts),
            "GT_read": False,
            "GT_derived_reference_consumed": True,
            "annotation_files_loaded_by_worker": False,
            "offline_diagnostic_only": True,
            "deployment_use": False,
            "online_carry": False,
            "source_capture_calls": 1,
            "source_capture_seconds": capture_seconds,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite student output: {destination}")
        torch.save(result, destination)
        output_files.append({"path": destination.relative_to(ROOT).as_posix(),
                             "sha256": sha(destination), "bytes": destination.stat().st_size})
        write(STATUS_PATH, {"status": "running", "stage": "matched_support_reference",
                            "done": number, "total": 8, "pid": os.getpid(),
                            "GT_read": False, "GT_derived_reference_consumed": True,
                            "time": time.time()}, mutable=True)
        del (result, batch, frames, base, old, native, source_logits, source_initial,
             fits, readouts, arms, receipts, case, catalog_data, row, records)
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    expected_fits = int(runtime["expected_fit_count"])
    if (len(output_files) != 8 or total_fits != expected_fits
            or not 0 <= total_steps <= total_fits * STEPS or total_readouts != total_fits):
        raise AssertionError("completed fit/readout counts differ from locked runtime")
    write(OUT_BARRIER_PATH, {
        "schema": BARRIER_SCHEMA, "status": "complete", "count": 8,
        "case_ids": list(CASE_IDS), "files": output_files,
        "actual_fit_count": total_fits, "optimizer_steps": total_steps,
        "actual_final_redecode_calls": total_readouts,
        "roster_lock_sha256": sha(ROSTER_PATH),
        "reference_lock_sha256": lock_sha,
        "runtime_sha256": sha(RUNTIME_PATH),
        "GT_read": False, "GT_derived_reference_consumed": True,
        "annotation_files_loaded_by_worker": False,
        "offline_diagnostic_only": True,
        "end_to_end_seconds": time.perf_counter() - started,
        "time": time.time(),
    })
    write(STATUS_PATH, {"status": "complete", "stage": "matched_support_reference",
                        "done": 8, "total": 8, "pid": os.getpid(),
                        "actual_fit_count": total_fits, "optimizer_steps": total_steps,
                        "actual_final_redecode_calls": total_readouts,
                        "GT_read": False, "GT_derived_reference_consumed": True,
                        "time": time.time()}, mutable=True)


def main() -> None:
    try:
        run_locked_worker()
    except BaseException as exc:
        write(BASE / "STUDENT_FAILURE_CAPTURE.json", {
            "status": "actual_failure", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "GT_read": False, "GT_derived_reference_consumed": True,
            "time": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
