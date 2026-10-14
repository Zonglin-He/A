"""Finite source-reset student fit for the sealed short-window STVG-R1 refs.

This worker only consumes METHOD_REFERENCE_LOCK.json after it is sealed. It
reuses the frozen source-capture/readout helpers and the original five-step
Gaussian IoU OPD fit. Importing this module is inert: no checkpoint, tokenizer,
video, source case, GT, or GPU is opened until main() is explicitly launched by
the separately locked phase supervisor.
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

from scripts.stvg_temporal_caption_query_v1 import read, sha, write, inference_guard
from scripts import run_stvg_adjacent_motion_student_v1 as frozen_helpers

BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
MANIFEST_PATH = BASE / "PREPARED_MANIFEST.json"
TRACKLET_BARRIER_PATH = BASE / "GLOBAL_TRACKLET_BARRIER.json"
FEATURE_BARRIER_PATH = BASE / "FEATURE_CAPTURE_BARRIER.json"
TEACHER_BARRIER_PATH = BASE / "TEACHER_GLOBAL_BARRIER.json"
TEACHER_JOB_LOCK_PATH = BASE / "TEACHER_JOB_LOCK.json"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
REFERENCE_LOCK_PATH = BASE / "METHOD_REFERENCE_LOCK.json"
RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
OUT_BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
STATUS_PATH = BASE / "STUDENT_GPU_STATUS.json"
OLD_T_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"

CASE_IDS = tuple(
    [f"hc2_{i:03d}" for i in range(4)]
    + [f"vidstg_{i:03d}" for i in range(4)]
)
ROLES = ("local_R1", "associated_R1")
STEPS = 5
ACTIVE_PARAMETERS = 1792
SCHEMA = "stvg-short-window-student-v1"
BARRIER_SCHEMA = "stvg-short-window-student-barrier-v1"

# Runtime pins include all code that supplies capture, training, readout, or
# the fixed short-window input construction. The original T receipts are added
# dynamically below so each of the eight exact files is pinned.
REQUIRED_STATIC_PINS = (
    "scripts/run_stvg_short_window_student_v1.py",
    "scripts/launch_stvg_short_window_v1.py",
    "artifacts/stvg_short_event_tracklets_v1/ROSTER_LOCK.json",
    "artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json",
    "artifacts/stvg_short_event_tracklets_v1/GLOBAL_TRACKLET_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/FEATURE_CAPTURE_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/TEACHER_GLOBAL_BARRIER.json",
    "artifacts/stvg_short_event_tracklets_v1/TEACHER_JOB_LOCK.json",
    "artifacts/stvg_short_event_tracklets_v1/METHOD_REFERENCE_LOCK.json",
    "scripts/run_stvg_adjacent_motion_student_v1.py",
    "scripts/stvg_temporal_caption_query_v1.py",
    "scripts/stvg_evidence_controls_inputs_v1.py",
    "scripts/stvg_query_alignment_inputs_v3.py",
    "scripts/stvg_sparse_r1_prompt_v1.py",
    "scripts/stvg_hierarchical_geometry_v1.py",
    "scripts/run_stvg_sparse_r1_v2.py",
    "scripts/run_decota_paper_main_v1.py",
    "scripts/run_tastvg_evidence_vulnerability_v2.py",
    "scripts/stvg_mllm_feedback_common_v1.py",
    "vg_tta/stvg_opd_frozen_mask_cache012.py",
    "vg_tta/stvg_opd_frozen_visual_host011.py",
    "vg_tta/stvg_opd_frozen_attention_lifetime010.py",
    "vg_tta/stvg_opd_frozen_module_cpu008.py",
    "methods/decota_final_simplified_v1/objectives.py",
    "methods/decota_final_simplified_v1/tensors.py",
    "methods/decota_final_simplified_v1/backbone.py",
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py",
    "vg_tta/decota_spatial_opd_tunable_v1.py",
    "vg_tta/tastvg_decota_critic_p0_v1.py",
    "vg_tta/c1_enabling_tricks_v1.py",
    "methods/decota_spatial_opd_v1/configs.json",
    "scripts/prepare_stvg_short_event_tracklets_v1.py",
    "scripts/run_stvg_short_event_tracklets_v1.py",
    "scripts/stvg_adjacent_affine_tracklet_v1.py",
    "protocols/stvg_short_event_tracklets_execution_v1.md",
)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha_path(relative: str) -> str:
    return sha(ROOT / relative)


def _case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["case_id"]) for row in rows]


def _json_records(doc: Mapping[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    rows = doc.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}.{key} must be a list of objects")
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _old_t_pin_paths() -> tuple[str, ...]:
    return (
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json",
        *(
            f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
            for cid in CASE_IDS
        ),
    )


def _fit_spec(case_id: str, role: str, config: Mapping[str, Any],
              reference: Mapping[str, Any]) -> dict[str, Any]:
    if role not in ROLES or not reference.get("anchors"):
        raise ValueError("fit spec requires one authorized nonempty role")
    chosen = dict(config)
    chosen["steps"] = STEPS
    return {
        "key": "caption_query_fixed_space|" + case_id,
        "scope": "joint",
        "reward_kind": "iou",
        "reference_role": "teacher",
        "reference_sha256": _canonical_sha(reference),
        "config": chosen,
        "steps": STEPS,
        "active_parameters": ACTIVE_PARAMETERS,
    }


def _validate_reference_row(
    reference: Mapping[str, Any],
    *,
    case_id: str,
    frame_ids: Sequence[int],
    centers: Sequence[Mapping[str, Any]],
    tracklets: Sequence[Mapping[str, Any]],
    teacher_output_digests: set[str],
    teacher_jobs: Mapping[str, Mapping[str, Any]],
) -> int:
    if reference.get("case_id") != case_id:
        raise ValueError(f"reference case_id mismatch: {case_id}")
    if reference.get("GT_read") is not False:
        raise ValueError(f"reference lock must declare GT_read false: {case_id}")
    refs = reference.get("references")
    if not isinstance(refs, dict) or set(refs) != set(ROLES):
        raise ValueError(f"reference roles differ from locked phase-2 roles: {case_id}")
    center_by_position = {int(row["source_grid_position"]): row for row in centers}
    if {int(row["slot"]) for row in centers} != {0, 1, 2, 3}:
        raise ValueError(f"four QA center slots are required: {case_id}")
    tracklet_by_id = {str(row["tracklet_id"]): row for row in tracklets}
    expected_fits = 0
    for role in ROLES:
        ref = refs[role]
        anchors = ref.get("anchors")
        if not isinstance(anchors, list) or len(anchors) > 4:
            raise ValueError(f"{role} anchors must contain zero to four centers: {case_id}")
        no_update = ref.get("no_update")
        eligible = ref.get("eligible_for_update")
        if not isinstance(no_update, bool) or not isinstance(eligible, bool):
            raise ValueError(f"{role} update flags must be booleans: {case_id}")
        if no_update != (len(anchors) == 0) or eligible != (len(anchors) > 0):
            raise ValueError(f"{role} no-update decision must be exactly anchor emptiness: {case_id}")
        if not isinstance(ref.get("reason"), str):
            raise ValueError(f"{role} needs a locked reason string: {case_id}")
        if "selected_candidate_id" not in ref:
            raise ValueError(f"{role} needs its candidate summary field: {case_id}")
        if (not isinstance(ref.get("selected_candidate_id_is_case_constant"), bool)
                or not isinstance(ref.get("selected_candidate_ids_by_position"), (dict, list))):
            raise ValueError(f"{role} needs its per-position selected-ID summary: {case_id}")
        seen_positions: set[int] = set()
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError(f"{role} anchor must be an object: {case_id}")
            position = anchor.get("position")
            physical_id = anchor.get("physical_frame_id")
            if (isinstance(position, bool) or not isinstance(position, int)
                    or position not in center_by_position or position >= len(frame_ids)):
                raise ValueError(f"{role} anchor position must be a source-grid index: {case_id}")
            if isinstance(physical_id, bool) or not isinstance(physical_id, int):
                raise ValueError(f"{role} physical frame ID must be an integer: {case_id}")
            if position in seen_positions:
                raise ValueError(f"{role} contains duplicate QA center: {case_id}")
            seen_positions.add(position)
            center = center_by_position[position]
            if int(physical_id) != int(center["center_physical_frame_id"]):
                raise ValueError(f"{role} physical-frame/source-grid binding differs: {case_id}")
            box = anchor.get("box_cxcywh")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"{role} needs a normalized cxcywh box: {case_id}")
            if any(isinstance(x, bool) for x in box):
                raise ValueError(f"{role} box coordinates cannot be booleans: {case_id}")
            vals = [float(x) for x in box]
            if not all(math.isfinite(x) for x in vals):
                raise ValueError(f"{role} box has nonfinite values: {case_id}")
            cx, cy, width, height = vals
            if not (width > 0 and height > 0 and 0 <= cx - width / 2 < cx + width / 2 <= 1
                    and 0 <= cy - height / 2 < cy + height / 2 <= 1):
                raise ValueError(f"{role} anchor box is outside normalized image bounds: {case_id}")
            candidate_id = anchor.get("candidate_id")
            if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id == 1:
                raise ValueError(f"{role} must exclude Source and name a non-Source candidate: {case_id}")
            selected_id = anchor.get("selected_candidate_id")
            if (isinstance(selected_id, bool) or not isinstance(selected_id, int)
                    or selected_id != candidate_id):
                raise ValueError(f"{role} selected ID differs from its fixed proposal: {case_id}")
            if anchor.get("is_Source") is not False:
                raise ValueError(f"{role} must explicitly exclude Source references: {case_id}")
            if role == "associated_R1" and "association_support" not in anchor:
                raise ValueError(f"{role} anchor lacks its mutual-nearest support record: {case_id}")
            tracklet_id = anchor.get("tracklet_id")
            if not isinstance(tracklet_id, str) or tracklet_id not in tracklet_by_id:
                raise ValueError(f"{role} anchor does not bind to a fixed manifest tracklet: {case_id}")
            tracklet = tracklet_by_id[tracklet_id]
            if (int(tracklet["candidate_id"]) != candidate_id
                    or int(tracklet["center_physical_frame_id"]) != int(physical_id)):
                raise ValueError(f"{role} anchor candidate/tracklet mapping differs: {case_id}")
            for digest_name in ("teacher_output_sha256",):
                if not _is_sha256(anchor.get(digest_name)):
                    raise ValueError(f"{role} anchor lacks {digest_name}: {case_id}")
                if anchor[digest_name] not in teacher_output_digests:
                    raise ValueError(f"{role} teacher response is absent from the sealed 64-call barrier: {case_id}")
            slot = int(center["slot"])
            expected_job_id = f"{case_id}_window{slot:02d}_base"
            job_id = anchor.get("teacher_job_id")
            job = teacher_jobs.get(job_id) if isinstance(job_id, str) else None
            if (job_id != expected_job_id or job is None
                    or job.get("case_id") != case_id or job.get("kind") != "base"
                    or job.get("center_slot") != slot):
                raise ValueError(f"{role} anchor lacks teacher job binding: {case_id}")
        expected_fits += int(eligible)
    return expected_fits


def _read_locked_inputs():
    """Read only locked JSON metadata and verify all payload SHA bindings."""
    roster_doc = read(ROSTER_PATH)
    manifest = read(MANIFEST_PATH)
    tracklet_barrier = read(TRACKLET_BARRIER_PATH)
    feature_barrier = read(FEATURE_BARRIER_PATH)
    teacher_barrier = read(TEACHER_BARRIER_PATH)
    teacher_job_lock = read(TEACHER_JOB_LOCK_PATH)
    reference_lock = read(REFERENCE_LOCK_PATH)
    runtime = read(RUNTIME_PATH)

    if roster_doc.get("status") != "locked" or roster_doc.get("GT_read") is not False:
        raise ValueError("ROSTER_LOCK is not a locked no-GT input roster")
    roster = _json_records(roster_doc, "records", "ROSTER_LOCK")
    if len(roster) != 8 or _case_ids(roster) != list(CASE_IDS):
        raise ValueError("ROSTER_LOCK must contain the exact stable old-eight order")
    roster_by_id = {row["case_id"]: row for row in roster}

    # The manifest is GT-free by construction; fail closed if its claim is
    # absent or ever changes rather than treating a missing flag as false.
    if (manifest.get("status") != "prepared_manifest_not_executed"
            or manifest.get("GT_read") is not False):
        raise ValueError("prepared short-window manifest is not explicitly no-GT")
    manifest_rows = _json_records(manifest, "cases", "PREPARED_MANIFEST")
    if [row.get("case_id") for row in manifest_rows] != list(CASE_IDS):
        raise ValueError("prepared manifest must contain the fixed old-eight order")
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}

    for path, doc, allowed_statuses in (
        (TRACKLET_BARRIER_PATH, tracklet_barrier, {"complete"}),
        (FEATURE_BARRIER_PATH, feature_barrier, {"sealed"}),
        (TEACHER_BARRIER_PATH, teacher_barrier, {"sealed", "complete"}),
        (TEACHER_JOB_LOCK_PATH, teacher_job_lock, {"locked"}),
    ):
        if doc.get("status") not in allowed_statuses or doc.get("GT_read") is not False:
            raise ValueError(f"{path.name} is not a complete/locked no-GT binding")

    feature_files = feature_barrier.get("files")
    if (feature_barrier.get("count") != 8 or not isinstance(feature_files, dict)
            or set(feature_files) != set(CASE_IDS)):
        raise ValueError("feature barrier must seal exactly one record for each old-eight case")

    if (tracklet_barrier.get("count") != 8
            or tracklet_barrier.get("case_ids") != list(CASE_IDS)
            or tracklet_barrier.get("GPU_used") is not False
            or tracklet_barrier.get("model_loaded") is not False
            or tracklet_barrier.get("teacher_called") is not False):
        raise ValueError("short-window tracklet barrier does not match the frozen CPU old-eight set")
    tracklet_records = _json_records(tracklet_barrier, "case_records", "GLOBAL_TRACKLET_BARRIER")
    if [row.get("case_id") for row in tracklet_records] != list(CASE_IDS):
        raise ValueError("tracklet case receipts do not match old-eight")
    tracklet_by_id = {row["case_id"]: row for row in tracklet_records}
    teacher_files = teacher_barrier.get("files")
    job_rows = _json_records(teacher_job_lock, "records", "TEACHER_JOB_LOCK")
    if (teacher_barrier.get("count") != 64 or not isinstance(teacher_files, list)
            or len(teacher_files) != 64 or teacher_job_lock.get("count") != 64
            or len(job_rows) != 64):
        raise ValueError("teacher barrier must contain the fixed 64 calls")
    if any(not isinstance(row, dict) or not _is_sha256(row.get("sha256"))
           for row in teacher_files):
        raise ValueError("teacher barrier output digests are malformed")
    teacher_output_digests = {str(row["sha256"]) for row in teacher_files}
    teacher_jobs = {
        str(row.get("job_id")): row for row in job_rows
        if isinstance(row.get("job_id"), str) and row.get("job_id")
    }
    if len(teacher_jobs) != len(job_rows):
        raise ValueError("teacher job lock does not have 64 unique stable job IDs")
    for cid in CASE_IDS:
        for slot in range(4):
            for kind, suffix in (("base", "base"), ("id_rotation", "id_rotation")):
                job_id = f"{cid}_window{slot:02d}_{suffix}"
                job = teacher_jobs.get(job_id)
                if (job is None or job.get("case_id") != cid or job.get("center_slot") != slot
                        or job.get("kind") != kind):
                    raise ValueError(f"teacher job lock differs from the fixed 64-call schedule: {job_id}")

    if (reference_lock.get("schema") != "stvg-short-window-reference-lock-v1"
            or reference_lock.get("status") != "locked"
            or reference_lock.get("count") != 8
            or reference_lock.get("GT_read") is not False):
        raise ValueError("METHOD_REFERENCE_LOCK has not sealed the exact no-GT ABI")
    refs = _json_records(reference_lock, "records", "METHOD_REFERENCE_LOCK")
    if _case_ids(refs) != list(CASE_IDS):
        raise ValueError("METHOD_REFERENCE_LOCK must contain the fixed old-eight order")
    reference_by_id = {row["case_id"]: row for row in refs}

    expected_lock_bindings = {
        "manifest_sha256": sha(MANIFEST_PATH),
        "tracklet_barrier_sha256": sha(TRACKLET_BARRIER_PATH),
        "feature_barrier_sha256": sha(FEATURE_BARRIER_PATH),
        "teacher_global_barrier_sha256": sha(TEACHER_BARRIER_PATH),
        "teacher_job_lock_sha256": sha(TEACHER_JOB_LOCK_PATH),
    }
    for key, digest in expected_lock_bindings.items():
        if reference_lock.get(key) != digest:
            raise ValueError(f"METHOD_REFERENCE_LOCK binding differs: {key}")

    expected_fits = 0
    for cid in CASE_IDS:
        roster_row = roster_by_id[cid]
        manifest_row = manifest_by_id[cid]
        tracklet_row = tracklet_by_id[cid]
        reference_row = reference_by_id[cid]
        if manifest_row.get("GT_read") is not False or roster_row.get("source_case_sha256") != manifest_row.get("source_case_sha256"):
            raise ValueError(f"source roster and prepared manifest disagree: {cid}")
        bindings = {
            "manifest_case_sha256": _canonical_sha(manifest_row),
            "tracklet_case_sha256": tracklet_row.get("sha256"),
            "source_case_sha256": roster_row.get("source_case_sha256"),
            "source_model_state_sha256": manifest_row.get("source_model_state_sha256"),
            "canonical_request_sha256": manifest_row.get("canonical_request_sha256"),
            "canonical_video_sha256": manifest_row.get("canonical_video_sha256"),
            "rgb_npz_sha256": tracklet_row.get("rgb_npz_sha256"),
        }
        for key, value in bindings.items():
            if reference_row.get(key) != value:
                raise ValueError(f"METHOD_REFERENCE_LOCK per-case binding differs: {cid}:{key}")
        if reference_row.get("dataset") != roster_row.get("dataset"):
            raise ValueError(f"reference dataset differs from source roster: {cid}")
        manifest_tracklets = [
            {**tracklet, "center_physical_frame_id": int(center["center_physical_frame_id"])}
            for center in manifest_row.get("centers", [])
            for tracklet in center.get("matched_tracklets", [])
        ]
        expected_fits += _validate_reference_row(
            reference_row,
            case_id=cid,
            frame_ids=list(range(int(manifest_row["source_grid_count"]))),
            centers=manifest_row.get("centers", []),
            tracklets=manifest_tracklets,
            teacher_output_digests=teacher_output_digests,
            teacher_jobs=teacher_jobs,
        )
        expected_grid_sha = manifest_row.get("source_grid_frame_ids_sha256")
        if not _is_sha256(expected_grid_sha):
            raise ValueError(f"manifest source-grid frame hash is missing: {cid}")
        tracklet_path = ROOT / tracklet_row["path"]
        if (tracklet_path.parent != (BASE / "tracklets").resolve()
                or sha(tracklet_path) != tracklet_row.get("sha256")):
            raise ValueError(f"sealed tracklet case file differs from its barrier: {cid}")
        rgb_path = ROOT / tracklet_row["rgb_npz_path"]
        if (rgb_path.parent != (BASE / "private_rgb").resolve()
                or sha(rgb_path) != tracklet_row.get("rgb_npz_sha256")):
            raise ValueError(f"selected-frame RGB cache differs from its barrier: {cid}")

    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("STUDENT_RUNTIME is not locked no-GT")
    if runtime.get("reference_lock_sha256") != sha(REFERENCE_LOCK_PATH):
        raise ValueError("STUDENT_RUNTIME does not bind METHOD_REFERENCE_LOCK")
    if runtime.get("steps") != STEPS or runtime.get("active_parameters") != ACTIVE_PARAMETERS:
        raise ValueError("STUDENT_RUNTIME differs from the fixed five-step 1,792-coordinate fit")
    if runtime.get("expected_fit_count") != expected_fits:
        raise ValueError("STUDENT_RUNTIME fit count differs from the sealed sparse refs")
    pins = runtime.get("pins", {})
    protected = runtime.get("protected", {})
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise ValueError("STUDENT_RUNTIME pin maps are malformed")
    required = set(REQUIRED_STATIC_PINS) | set(_old_t_pin_paths())
    try:
        from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import SOURCE_BINDINGS
        required.add("scripts/stvg_video_global_spatial_fit_fp32_geometry001.py")
        required.update(SOURCE_BINDINGS)
    except Exception as exc:
        raise RuntimeError("cannot enumerate the frozen fit source pins") from exc
    missing = sorted(required - set(pins))
    if missing:
        raise ValueError(f"STUDENT_RUNTIME omits reused scientific source pins: {missing}")
    if "methods/CURRENT_METHOD.json" not in protected:
        raise ValueError("STUDENT_RUNTIME must protect the deployed method registry")
    for rel, digest in {**pins, **protected}.items():
        if not _is_sha256(digest) or _sha_path(rel) != digest:
            raise ValueError(f"STUDENT_RUNTIME file pin mismatch: {rel}")
    if OUT_BARRIER_PATH.exists():
        raise FileExistsError("STUDENT_GLOBAL_BARRIER already exists; refusing overwrite")
    return roster, manifest_by_id, tracklet_by_id, reference_by_id, runtime


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in state.items()}


def _anchors_from_locked_positions(reference: Mapping[str, Any], centers: Sequence[Mapping[str, Any]],
                                   frame_ids: Sequence[int], torch) -> list[dict[str, Any]]:
    """Verify locked source-grid positions and pass them to the frozen parser."""
    center_by_position = {int(row["source_grid_position"]): row for row in centers}
    ids = list(map(int, frame_ids))
    for row in reference["anchors"]:
        position = int(row["position"])
        physical = int(row["physical_frame_id"])
        if position not in center_by_position:
            raise ValueError("locked source-grid position does not identify a QA center")
        if physical != int(center_by_position[position]["center_physical_frame_id"]):
            raise ValueError("source-grid position no longer maps to its locked physical frame")
        if not 0 <= position < len(ids) or ids[position] != physical:
            raise ValueError("QA center is absent at its exact captured source-grid position")
    anchors = frozen_helpers._anchors(reference, ids, torch)
    if len({row["position"] for row in anchors}) != len(anchors):
        raise ValueError("mapped OPD anchors collide on a source-grid position")
    return anchors


def _fit_reference(
    *, case_id: str, role: str, reference: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Any], base: Any, frame_ids: Sequence[int], dataset: str,
    config: Mapping[str, Any], core: Any,
    fit_spatial_reward_opd: Any, readout: Any, model: Any, batch: Any,
    records: Any, row: Mapping[str, Any], expected_source_hash: str,
    source_logits: Sequence[Any], state_hash: Any, torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one role from the same source initial with the original IoU OPD."""
    if role not in ROLES or not anchors:
        raise ValueError("_fit_reference only accepts a nonempty locked role")
    fit_spec = _fit_spec(case_id, role, config, reference)
    partial = BASE / "partial" / role / case_id
    partial.mkdir(parents=True, exist_ok=True)
    saved = sorted(partial.glob("step*.pt"), key=lambda path: int(path.stem[4:]))
    resume = torch.load(saved[-1], map_location="cpu", weights_only=False) if saved else None

    def checkpoint(payload: Mapping[str, Any]) -> None:
        destination = partial / f"step{int(payload['next_step'])}.pt"
        if destination.exists():
            raise FileExistsError(f"refusing to replace fit receipt {destination}")
        torch.save(dict(payload), destination)

    cfg = dict(fit_spec["config"])
    fit = fit_spatial_reward_opd(
        base, _clone_state(initial), anchors, frame_ids, fit_spec["key"], cfg,
        scope=fit_spec["scope"], reward_kind=fit_spec["reward_kind"],
        checkpoint_callback=checkpoint,
        reference_role=fit_spec["reference_role"],
        reference_sha256=fit_spec["reference_sha256"],
        resume_state=resume,
    )
    _validate_fit_result(fit, role=role)
    out = readout(
        model, batch, frame_ids, records, fit, row["parses"]["subject"], core, torch,
    )
    if not torch.equal(out["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu()):
        raise AssertionError(f"{role} final full readout differs from its cached final decoder")
    if not frozen_helpers._same_logits(out["logits"], source_logits):
        raise AssertionError(f"{role} changed source logits while applying its fit state")
    if state_hash(model.state_dict()) != expected_source_hash:
        raise AssertionError(f"{role} mutated the frozen full source model")
    return fit, out


def _validate_fit_result(fit: Mapping[str, Any], *, role: str) -> int:
    """Check the original five-round contract without forcing Adam updates."""
    rounds = fit.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != STEPS:
        raise RuntimeError(f"{role} did not preserve all five original OPD rounds")
    if (int(fit.get("active_parameters", -1)) != ACTIVE_PARAMETERS
            or int(fit.get("declared_steps", STEPS)) != STEPS):
        raise RuntimeError(f"{role} did not complete the fixed 5-step/1,792-coordinate fit")
    if int(fit.get("selected_step", STEPS)) != STEPS:
        raise RuntimeError(f"{role} selected a nonfinal checkpoint")
    updated = 0
    for row in rounds:
        if not isinstance(row, Mapping) or not isinstance(row.get("updated"), bool):
            raise RuntimeError(f"{role} round lacks the original updated flag")
        updated += int(row["updated"])
    optimizer_steps = fit.get("optimizer_steps")
    if (isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int)
            or optimizer_steps != updated or not 0 <= optimizer_steps <= STEPS):
        raise RuntimeError(f"{role} Adam step count differs from the five logged round outcomes")
    return optimizer_steps


def _no_update_arms(f_arms: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]):
    """Return exact F/T copies for empty reference sets, with no optimizer call."""
    arms = {"F": copy.deepcopy(f_arms["F"]), "T": copy.deepcopy(f_arms["T"])}
    receipts = {}
    for role in ROLES:
        reference = references[role]
        if reference["anchors"]:
            continue
        arms[role] = copy.deepcopy(arms["F"])
        arms[f"T_{role}"] = copy.deepcopy(arms["T"])
        receipts[role] = {
            "status": "exact_F_no_update",
            "no_update": True,
            "optimizer_steps": 0,
            "final_redecode_calls": 0,
            "exact_F_reuse": True,
        }
        receipts[f"T_{role}"] = {
            "status": "exact_T_no_update",
            "no_update": True,
            "optimizer_steps": 0,
            "final_redecode_calls": 0,
            "exact_T_reuse": True,
        }
    return arms, receipts


def _main_method() -> None:
    roster, manifest_by_id, tracklet_by_id, reference_by_id, runtime = _read_locked_inputs()
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
        SOURCE_BINDINGS, fit_spatial_reward_opd, verify_source_bindings,
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("student worker expected a fresh CUDA context")
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
    output_files = []
    total_fits = 0
    total_steps = 0
    total_final_readouts = 0
    started_all = time.perf_counter()
    write(STATUS_PATH, {
        "status": "running", "stage": "source_reset_short_window_student",
        "done": 0, "total": len(roster), "pid": os.getpid(),
        "GT_read": False, "time": time.time(),
    }, mutable=True)

    for number, rec in enumerate(roster, 1):
        cid = str(rec["case_id"])
        manifest_row = manifest_by_id[cid]
        tracklet_row = tracklet_by_id[cid]
        reference_row = reference_by_id[cid]
        references = reference_row["references"]
        destination = BASE / "student" / f"{cid}.pt"
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=False)
            assert saved["schema"] == SCHEMA and saved["case_id"] == cid
            assert saved["source_case_sha256"] == rec["source_case_sha256"]
            assert saved["reference_lock_sha256"] == lock_sha and saved["source_restored"]
            assert saved["source_cold_readback"]["status"] == "pass"
            assert saved["reference_records"] == references
            assert set(saved["fits"]) == {role for role in ROLES if references[role]["anchors"]}
            output_files.append({
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": sha(destination), "bytes": destination.stat().st_size,
            })
            total_fits += int(saved["actual_fit_count"])
            total_steps += int(saved["optimizer_steps"])
            total_final_readouts += int(saved["actual_final_redecode_calls"])
            continue

        # Drop the outer reference before model_for constructs a different
        # frozen source family; _capture_source also protects its local scope.
        if model is not None:
            target_source = frozen_helpers.catalog(rec)[0]["source"]
            if current_source != target_source:
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
        (
            model, current_source, case, catalog_data, row, frames, frame_ids,
            records, batch, base, native, expected_source_hash, pixel_sha,
            capture_seconds,
        ) = frozen_helpers._capture_source(
            rec, model, current_source, torch, collections, model_for, frames_for,
            raw_row, frozen_forward, prediction, state_hash,
        )
        assert case["_source_case_sha256"] == reference_row["source_case_sha256"]
        assert expected_source_hash == reference_row["source_model_state_sha256"]
        if _canonical_sha(list(map(int, frame_ids))) != manifest_row["source_grid_frame_ids_sha256"]:
            raise AssertionError("captured source-grid frame ids differ from the prepared manifest")
        assert pixel_sha == case["pixel_sha256"]

        native_boxes = frozen_helpers._xyxy(native["boxes"])
        native_interval = list(case["Native_interval"])
        projection_interval = list(case["Projection_interval"])
        bounds = list(case["video_bounds"])
        f_arms = {"F": frozen_helpers._arm(native_boxes, native_interval)}
        old = frozen_helpers._load_old_temporal_case(
            cid, rec["source_case_sha256"], expected_source_hash, frame_ids,
            pixel_sha, native_boxes, native_interval, projection_interval, bounds, torch,
        )
        t_arm = copy.deepcopy(old["arms"]["T"])
        if not np.array_equal(t_arm["boxes"], native_boxes):
            raise AssertionError("frozen old T boxes are not an exact reuse of F")
        f_arms["T"] = t_arm
        arms, receipts = _no_update_arms(f_arms, references)
        fits = {}
        readouts = {}
        source_logits = native["logits"]
        fit = None
        out = None
        anchors = None
        source_initial = _clone_state(base.initial)
        cfg = core.load_original_opd_config(rec["dataset"])
        cfg["steps"] = STEPS

        centers = manifest_row["centers"]
        for role in ROLES:
            reference = references[role]
            if not reference["anchors"]:
                continue
            anchors = _anchors_from_locked_positions(reference, centers, frame_ids, torch)
            if not anchors:
                raise AssertionError("eligible reference unexpectedly mapped to no source-grid anchors")
            fit, out = _fit_reference(
                case_id=cid, role=role, reference=reference, anchors=anchors,
                initial=source_initial, base=base, frame_ids=frame_ids,
                dataset=rec["dataset"], config=cfg, core=core,
                fit_spatial_reward_opd=fit_spatial_reward_opd,
                readout=lambda *args: frozen_helpers._readout(
                    *args, transient_native_masks=transient_native_masks,
                    frozen_visual_host=frozen_visual_host,
                    frozen_attention_storage=frozen_attention_storage,
                    inactive_modules_on_host=inactive_modules_on_host,
                    query_subject=query_subject,
                ),
                model=model, batch=batch, records=records, row=row,
                expected_source_hash=expected_source_hash,
                source_logits=source_logits, state_hash=state_hash, torch=torch,
            )
            if not torch.equal(out["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu()):
                raise AssertionError(f"{role} readout did not reproduce final decoder boxes")
            boxes = frozen_helpers._xyxy(out["boxes"])
            arms[role] = frozen_helpers._arm(boxes, native_interval)
            arms[f"T_{role}"] = frozen_helpers._arm(boxes, t_arm["interval"])
            fits[role] = fit
            readouts[role] = out
            receipts[role] = {
                "status": "updated_full_final_readout",
                "no_update": False,
                "optimizer_steps": int(fit["optimizer_steps"]),
                "final_redecode_calls": 1,
                "reference_sha256": _canonical_sha(reference),
                "reward": "original_gaussian_spatial_iou",
            }
            receipts[f"T_{role}"] = {
                "status": "updated_spatial_plus_exact_old_T_interval",
                "no_update": False,
                "optimizer_steps": int(fit["optimizer_steps"]),
                "final_redecode_calls": 0,
                "inherits_fit": role,
                "old_T_interval_reused": True,
            }
            total_fits += 1
            total_steps += int(fit["optimizer_steps"])
            total_final_readouts += 1

        if state_hash(model.state_dict()) != expected_source_hash:
            raise AssertionError("frozen source state changed after role fits")
        for role in ROLES:
            if not references[role]["anchors"]:
                assert arms[role]["interval"] == f_arms["F"]["interval"]
                assert np.array_equal(arms[role]["boxes"], f_arms["F"]["boxes"])
                assert arms[f"T_{role}"]["interval"] == t_arm["interval"]
                assert np.array_equal(arms[f"T_{role}"]["boxes"], t_arm["boxes"])

        result = {
            "schema": SCHEMA,
            "case_id": cid,
            "dataset": rec["dataset"],
            "cohort": rec.get("cohort", "historically_exposed_old8"),
            "source_case_sha256": rec["source_case_sha256"],
            "source_model_state_sha256": expected_source_hash,
            "manifest_case_sha256": reference_row["manifest_case_sha256"],
            "tracklet_case_sha256": reference_row["tracklet_case_sha256"],
            "frame_ids": list(map(int, frame_ids)),
            "pixel_sha256": pixel_sha,
            "Native_interval": native_interval,
            "Projection_interval": projection_interval,
            "arms": arms,
            "fits": fits,
            "readouts": readouts,
            "reference_records": copy.deepcopy(references),
            "role_semantics": {
                "local_R1": "all valid selected non-Source center boxes from the sealed base calls",
                "associated_R1": "local_R1 anchors retained only when their selected proposal has a sealed mutual-nearest neighbor at an adjacent QA center",
                "appearance_consistency_is_proven_object_identity": False,
                "cross_window_object_identity_claim": False,
            },
            "reference_lock_sha256": lock_sha,
            "reference_receipts": receipts,
            "initial_state": core._copy_cpu(source_initial),
            "source_boxes": base.zero["boxes"].detach().cpu(),
            "source_logits": core._copy_cpu(source_logits),
            "source_restored": True,
            "source_cold_readback": {
                "status": "pass",
                "source_case_sha256": rec["source_case_sha256"],
                "source_model_state_sha256": expected_source_hash,
                "pixel_sha256": pixel_sha,
                "frame_ids_match_manifest": True,
                "initial_state_match": True,
                "zero_boxes_logits_match": True,
                "source_restored": True,
            },
            "temporal_exact_reuse": {
                "status": "reused_exact_old_T",
                "source_case_sha256": rec["source_case_sha256"],
                "old_student_path": (OLD_T_BASE / "student" / f"{cid}.pt").relative_to(ROOT).as_posix(),
                "old_student_sha256": sha(OLD_T_BASE / "student" / f"{cid}.pt"),
                "old_T_boxes_equal_F": True,
                "old_T_interval_reused": True,
            },
            "configuration": {
                "key": "caption_query_fixed_space|" + cid,
                "scope": "joint",
                "steps": STEPS,
                "active_parameters": ACTIVE_PARAMETERS,
                "reward": "original_gaussian_spatial_iou",
                "config": cfg,
                "method_reference_lock_sha256": lock_sha,
            },
            "source_capture_calls": 1,
            "source_capture_seconds": capture_seconds,
            "actual_fit_count": len(fits),
            "optimizer_steps": sum(int(fit["optimizer_steps"]) for fit in fits.values()),
            "actual_final_redecode_calls": len(readouts),
            "GT_read": False,
            "online_carry": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite sealed student result: {destination}")
        torch.save(result, destination)
        output_files.append({
            "path": destination.relative_to(ROOT).as_posix(),
            "sha256": sha(destination), "bytes": destination.stat().st_size,
        })
        write(STATUS_PATH, {
            "status": "running", "stage": "source_reset_short_window_student",
            "done": number, "total": len(roster), "pid": os.getpid(),
            "GT_read": False, "time": time.time(),
        }, mutable=True)
        del (result, batch, frames, base, old, native, source_logits, source_initial,
             fits, readouts, arms, receipts, f_arms, t_arm, fit, out, anchors,
             case, catalog_data, row, records)
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    expected_fits = int(runtime["expected_fit_count"])
    if len(output_files) != 8 or total_fits != expected_fits:
        raise AssertionError("sealed student output count/fit count differs from runtime")
    if not 0 <= total_steps <= total_fits * STEPS or total_final_readouts != total_fits:
        raise AssertionError("nonempty references must preserve five rounds and one final readout")
    write(OUT_BARRIER_PATH, {
        "schema": BARRIER_SCHEMA,
        "status": "complete",
        "count": len(output_files),
        "case_ids": list(CASE_IDS),
        "files": output_files,
        "actual_fit_count": total_fits,
        "optimizer_steps": total_steps,
        "actual_final_redecode_calls": total_final_readouts,
        "roster_lock_sha256": sha(ROSTER_PATH),
        "manifest_sha256": sha(MANIFEST_PATH),
        "tracklet_barrier_sha256": sha(TRACKLET_BARRIER_PATH),
        "feature_barrier_sha256": sha(FEATURE_BARRIER_PATH),
        "teacher_global_barrier_sha256": sha(TEACHER_BARRIER_PATH),
        "teacher_job_lock_sha256": sha(TEACHER_JOB_LOCK_PATH),
        "reference_lock_sha256": lock_sha,
        "runtime_sha256": sha(RUNTIME_PATH),
        "GT_read": False,
        "end_to_end_seconds": time.perf_counter() - started_all,
        "time": time.time(),
    })
    write(STATUS_PATH, {
        "status": "complete", "stage": "source_reset_short_window_student",
        "done": 8, "total": 8, "pid": os.getpid(), "GT_read": False,
        "actual_fit_count": total_fits, "optimizer_steps": total_steps,
        "actual_final_redecode_calls": total_final_readouts, "time": time.time(),
    }, mutable=True)


def main() -> None:
    try:
        _main_method()
    except BaseException as exc:
        write(BASE / "STUDENT_FAILURE_CAPTURE.json", {
            "status": "actual_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "GT_read": False,
            "time": time.time(),
        }, mutable=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("usage: run_stvg_short_window_student_v1.py")
    main()
