#!/usr/bin/env python3
"""CPU-only roster, seal, and post-seal evaluation contracts for STVG-R1 distillation.

This module never runs a teacher or student. Its scoring path accepts only a
complete label-free prediction seal for the declared evaluation stage, a frozen
SourceFrozen cache lock, and a strict inference manifest before it opens target
labels. The teacher screen can reuse sealed Native/Projection source outputs;
student fitting additionally requires a fresh SpatialReplay feature binding.
Original dense metric functions load only after those gates pass. This V2 adds
the matched OPD-minus-Projection contrast without changing any scoring rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "stvg_target_teacher_distillation_v1"
DEV_ROSTER = BASE / "ROSTER_DEVELOPMENT_PROPOSED.json"
CONFIRM_ROSTER = BASE / "ROSTER_CONFIRMATION_PROPOSED.json"
CONTRACTS = BASE / "evaluation_contracts" / "CPU_CONTRACTS.json"

ROSTER_SCHEMA = "stvg_target_teacher_roster_v1"
PREDICTION_SEAL_SCHEMA = "stvg_target_teacher_predictions_barrier_v1"
SOURCE_LOCK_SCHEMA = "stvg_target_teacher_source_cache_lock_v1"
LABELS_SCHEMA = "stvg_target_teacher_labels_manifest_v1"
LABEL_CASE_SCHEMA = "stvg_target_teacher_label_case_v1"
PREDICTION_CASE_SCHEMA = "stvg_target_teacher_case_predictions_v1"

CONDITIONS = (
    "Source",
    "Projection",
    "Teacher",
    "NoParamRefine",
    "Direct",
    "OPD",
)
EVALUATION_CONDITIONS = {
    "teacher_screen": ("Source", "Projection", "Teacher"),
    "student_matrix": CONDITIONS,
}
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT", "R@0.3_strict", "R@0.5_strict")
CONTRASTS = {
    "Projection_minus_Source": ("Projection", "Source"),
    "Teacher_minus_Source": ("Teacher", "Source"),
    "NoParamRefine_minus_Teacher": ("NoParamRefine", "Teacher"),
    "Direct_minus_Projection": ("Direct", "Projection"),
    "OPD_minus_Projection": ("OPD", "Projection"),
    "OPD_minus_Direct": ("OPD", "Direct"),
    "OPD_minus_Source": ("OPD", "Source"),
}
EVALUATION_CONTRASTS = {
    "teacher_screen": {
        key: value for key, value in CONTRASTS.items()
        if key in ("Projection_minus_Source", "Teacher_minus_Source")
    },
    "student_matrix": CONTRASTS,
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261013
GAP_EPSILON = 1e-3
INFERENCE_FIELDS = frozenset({"realvideo", "query", "fps", "frameBounds", "dimensions"})
PREDICTION_OUTPUT_KIND = {
    "Source": "student_decoder",
    "Projection": "student_decoder",
    "Teacher": "teacher_pipeline",
    "NoParamRefine": "teacher_geometry_direct",
    "Direct": "student_decoder",
    "OPD": "student_decoder",
}
TEACHER_COST_FIELDS = (
    "full_pipeline_seconds",
    "roi_extraction_seconds",
    "tracking_seconds",
    "time_reasoning_seconds",
    "student_adaptation_seconds",
    "peak_gpu_bytes",
)
PIXEL_CLOCK_CONTRACT = "root_canonical_ffv1_v1"
PIXEL_CLOCK_POLICY_BY_TARGET = {"hc2": "ffmpeg_default", "vidstg": "vsync0"}


class ContractError(RuntimeError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON literal: {value}")


def read_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_pairs_no_duplicates,
                             parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ContractError("paths must be non-empty repository-relative POSIX strings")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ContractError(f"unsafe repository-relative path: {relative!r}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ContractError(f"path escapes repository: {relative!r}")
    return resolved


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_ground_truth_fields(value: Any, where: str = "prediction") -> None:
    forbidden = {"ground_truth", "truth", "truth_boxes_xyxy", "gt_span", "gt_boxes",
                 "annotations", "annotation", "labels", "label_path", "gt_path"}
    if isinstance(value, dict):
        for key, nested in value.items():
            _require(str(key).lower() not in forbidden,
                     f"forbidden label-derived field in {where}: {key}")
            _reject_ground_truth_fields(nested, f"{where}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_ground_truth_fields(nested, f"{where}[{index}]")


def validate_inference_row(row: Any) -> None:
    """Enforce the only model-visible fields: video, query, fps, bounds, dimensions.

    Dataset query text is treated as data. This validator does not execute it,
    interpret it as instructions, or use it to select a roster row.
    """
    _require(isinstance(row, dict), "inference row must be an object")
    _require(set(row) == INFERENCE_FIELDS,
             f"inference row keys must be exactly {sorted(INFERENCE_FIELDS)}")
    _require(isinstance(row["realvideo"], str) and bool(row["realvideo"].strip()),
             "realvideo must be a non-empty path string")
    _require(isinstance(row["query"], str) and bool(row["query"].strip()),
             "query must be a non-empty string")
    _require(_finite_number(row["fps"]) and float(row["fps"]) > 0,
             "fps must be finite and positive")
    bounds = row["frameBounds"]
    _require(isinstance(bounds, list) and len(bounds) == 2 and all(_int(x) for x in bounds),
             "frameBounds must be two integer physical frame bounds")
    _require(bounds[1] > bounds[0], "frameBounds must be increasing and half-open")
    dims = row["dimensions"]
    _require(isinstance(dims, list) and len(dims) == 2 and all(_int(x) and x > 0 for x in dims),
             "dimensions must be positive integer [width, height]")


def validate_inference_manifest(manifest: Any, expected_case_ids: set[str]) -> None:
    _require(isinstance(manifest, dict), "inference manifest must map case IDs to rows")
    _require(set(manifest) == expected_case_ids, "inference manifest case IDs do not match roster")
    for case_id in sorted(expected_case_ids):
        validate_inference_row(manifest[case_id])


def _case_id(row: dict[str, Any], expected_prefix: str | None = None) -> str:
    value = row.get("case_id")
    _require(isinstance(value, str) and value, "roster row missing case_id")
    if expected_prefix is not None:
        _require(value.startswith(expected_prefix), "unexpected case ID prefix")
    return value


def _source_cache_metadata(source_base: Path, fixed: dict[str, Any], *, verify_bytes: bool) -> dict[str, Any]:
    runtime_path = source_base / "RUNTIME.json"
    fixed_path = source_base / "FIXED_ROSTER.json"
    design_path = source_base / "DESIGN_LOCK.json"
    barrier_path = source_base / "SCREEN_SOURCE_CAPTURE_BARRIER.json"
    readback_path = source_base / "SOURCE_ROOT_READBACK.json"
    runtime = read_json(runtime_path)
    design = read_json(design_path)
    barrier = read_json(barrier_path)
    readback = read_json(readback_path)

    _require(runtime.get("DINO_free") is True, "existing source runtime is not pinned DINO-free")
    _require(runtime.get("only_original_source_initialization") is True,
             "existing source runtime is not pinned to original source initialization")
    _require(runtime.get("roster_sha256") == sha256_file(fixed_path),
             "existing source runtime does not bind FIXED_ROSTER bytes")
    _require(runtime.get("design_sha256") == sha256_file(design_path),
             "existing source runtime does not bind DESIGN_LOCK bytes")
    _require(design.get("DINO_state_or_reference_used") is False and
             design.get("source_checkpoint_state_only") is True,
             "existing source design does not prove original source-only initialization")
    _require(barrier.get("status") == "sealed" and barrier.get("count") == 64 and
             barrier.get("cases") == 64 and barrier.get("GT_read") is False,
             "existing source barrier is not a 64-case label-free seal")
    _require(barrier.get("new_DINO") == 0 and barrier.get("optimizer_steps") == 0 and
             barrier.get("backwards") == 0,
             "existing source barrier records DINO/optimizer activity")
    _require(readback.get("status") == "pass" and readback.get("count") == 64 and
             readback.get("GT_read") is False and readback.get("CPU_only") is True,
             "existing source independent root readback is not a CPU-only 64-case pass")

    fixed_rows = fixed.get("records")
    files = barrier.get("files")
    _require(isinstance(fixed_rows, list) and len(fixed_rows) == 64,
             "fixed development roster must contain 64 rows")
    _require(isinstance(files, dict) and len(files) == 64,
             "existing source barrier must contain 64 file hashes")
    case_ids = {_case_id(row) for row in fixed_rows}
    _require(len(case_ids) == 64, "duplicate development case IDs")
    source_base_relative = source_base.relative_to(ROOT).as_posix()
    expected_paths = {f"{source_base_relative}/source_cases/{case_id}.pt" for case_id in case_ids}
    _require(set(files) == expected_paths, "source cache file set does not match fixed development roster")
    checked = 0
    if verify_bytes:
        for relative, expected_hash in sorted(files.items()):
            _require(_sha(expected_hash), f"invalid source cache SHA for {relative}")
            path = repo_path(relative)
            _require(path.is_file() and sha256_file(path) == expected_hash,
                     f"opaque source cache byte hash mismatch: {relative}")
            checked += 1

    image_views = design.get("image_views")
    return {
        "runtime_path": runtime_path.relative_to(ROOT).as_posix(),
        "runtime_sha256": sha256_file(runtime_path),
        "design_path": design_path.relative_to(ROOT).as_posix(),
        "design_sha256": sha256_file(design_path),
        "fixed_roster_path": fixed_path.relative_to(ROOT).as_posix(),
        "fixed_roster_sha256": sha256_file(fixed_path),
        "barrier_path": barrier_path.relative_to(ROOT).as_posix(),
        "barrier_sha256": sha256_file(barrier_path),
        "root_readback_path": readback_path.relative_to(ROOT).as_posix(),
        "root_readback_sha256": sha256_file(readback_path),
        "case_count": len(case_ids),
        "opaque_payload_hashes_checked": checked,
        "DINO_free": True,
        "original_source_initialization_only": True,
        "GT_read": False,
        "GPU_used_by_this_review": False,
        "existing_design_image_views": image_views,
        "existing_source_root_status": "pass_64_GT_false_CPU_only",
        "root_reviewed_saved_outputs": [
            "Native raw logits", "actions", "records", "Projection_interval"
        ],
        "teacher_screen_reuse": {
            "eligible_for_root_rebinding": True,
            "scope": "Source, independently recomputed Projection, and Teacher only",
            "projection_must_be_recomputed_from_sealed_raw_projection_inputs": True,
            "task_lock_must_rebind_every_case_to_exact_inference_row": True,
            "this_builder_decoded_source_payload": False,
        },
        "required_task_input_clock_contract": PIXEL_CLOCK_CONTRACT,
        "required_pixel_clock_policy_by_target": PIXEL_CLOCK_POLICY_BY_TARGET,
        "physical_source_checkpoint_policy": "one original checkpoint per target; HC2 and VidSTG checkpoints remain separately bound",
        "student_matrix_reuse": {
            "eligible": False,
            "reason": "Direct/OPD student fitting requires a fresh per-case SpatialReplay feature binding",
            "fresh_spatial_replay_required": True,
        },
        "view_count_context_only": image_views,
        "view_count_used_as_rejection_criterion": False,
        "confirmation_cache_availability": "not_verified",
        "confirmation_action": "root_authenticate_a_sealed_source_cache_or_capture_after_roster_lock",
    }


def build_proposed_rosters(*, verify_source_bytes: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create deterministic metadata-only cohort proposals, never opening GT files."""
    source_base = ROOT / "artifacts" / "stvg_mllm_candidate_feedback_v1"
    fixed_path = source_base / "FIXED_ROSTER.json"
    p7_path = ROOT / "artifacts" / "stvg_opd_temporal_guided_spatial_candidate_v1" / "DESIGN_LOCK.json"
    fixed = read_json(fixed_path)
    p7 = read_json(p7_path)
    _require(fixed.get("GT_read") is False and len(fixed.get("records", [])) == 64,
             "fixed old-P6 development roster is not an exact 64-row label-free manifest")
    _require(p7.get("GT_read") is False, "old P7 design roster does not declare GT_read=false")
    captures = p7.get("captures")
    _require(isinstance(captures, list) and len(captures) == 256,
             "old P7 capture roster must contain 256 rows")

    by_identity: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for capture in captures:
        key = (capture["dataset"], capture["source"], capture["query_ordinal"])
        by_identity.setdefault(key, []).append(capture)
    dev_identity: set[tuple[str, str, int]] = set()
    dev_counts: dict[str, int] = {"hc2": 0, "vidstg": 0}
    dev_rows: list[dict[str, Any]] = []
    for row in fixed["records"]:
        case_id = _case_id(row)
        ds, src = row.get("dataset"), row.get("source")
        ordinal = row.get("query_ordinal")
        _require(ds in dev_counts and isinstance(src, str) and _int(ordinal),
                 "invalid old-P6 development cohort metadata")
        _require(row.get("original_source_initialization") is True and
                 row.get("DINO_state_initialization") is False,
                 "old-P6 development row is not SourceFrozen/DINO-free")
        _require(row.get("historical_target_exposure") is True,
                 "old-P6 development exposure must be disclosed")
        key = (ds, src, ordinal)
        _require(key not in dev_identity, "duplicate old-P6 development parent/query identity")
        dev_identity.add(key)
        matches = by_identity.get(key, [])
        _require(len(matches) == 1 and matches[0].get("previous_P6") is True,
                 "development row does not exactly match the old-P6 flag in the P7 capture roster")
        capture = matches[0]
        _require(capture.get("parent_id") is not None and
                 capture.get("original_Full", {}).get("input_sha256"),
                 "P7 capture is missing a source-input identity hash")
        input_meta = row.get("input", {})
        video_hash = input_meta.get("video_sha256") if isinstance(input_meta, dict) else None
        _require(_sha(video_hash), "old-P6 row is missing its source video SHA")
        dev_rows.append({
            "case_id": case_id,
            "dataset": ds,
            "source": src,
            "parent_token": sha256_bytes(f"{ds}|{src}|{row['parent']}".encode())[:20],
            "query_ordinal": ordinal,
            "source_video_sha256": video_hash,
            "source_cohort_sha256": row.get("origin_cohort_sha256"),
            "fixed_plan_sha256": row.get("plan_sha256"),
            "old_p7_capture_index": capture.get("index"),
            "old_p7_source_input_sha256": capture["original_Full"]["input_sha256"],
            "historically_exposed": True,
            "historical_exposure_note": "exact original P6 development parent; also present in the old P7 capture pool",
        })
        dev_counts[ds] += 1
    _require(dev_counts == {"hc2": 32, "vidstg": 32}, "development roster must remain 32+32")

    confirm_rows: list[dict[str, Any]] = []
    confirm_counts: dict[str, int] = {"hc2": 0, "vidstg": 0}
    for ds in ("hc2", "vidstg"):
        eligible = [row for row in captures if row.get("dataset") == ds and row.get("previous_P6") is False]
        _require(len(eligible) == 96, f"expected exactly 96 non-P6 eligible parents for {ds}")
        ranked = []
        for capture in eligible:
            source_hash = capture.get("original_Full", {}).get("input_sha256")
            _require(_sha(source_hash), "confirmation parent has no source-input SHA")
            fields = ["stvg-target-teacher-confirm-v1", ds, str(capture["parent_id"]),
                      capture["source"], str(capture["query_ordinal"]), source_hash]
            selection_hash = sha256_bytes("|".join(fields).encode("utf-8"))
            ranked.append((selection_hash, capture))
        ranked.sort(key=lambda item: (item[0], item[1]["parent_id"], item[1]["query_ordinal"]))
        for rank, (selection_hash, capture) in enumerate(ranked[:64]):
            confirm_rows.append({
                "case_id": f"{ds}_confirm_{rank:03d}",
                "dataset": ds,
                "source": capture["source"],
                "parent_id": capture["parent_id"],
                "query_ordinal": capture["query_ordinal"],
                "source_parent_token": sha256_bytes(
                    f"{ds}|{capture['source']}|{capture['parent_id']}".encode())[:20],
                "source_input_sha256": capture["original_Full"]["input_sha256"],
                "original_source_output_sha256": capture["original_Full"]["sha256"],
                "old_p7_capture_index": capture["index"],
                "selection_sha256": selection_hash,
                "selection_rank_within_target": rank,
                "historically_exposed": True,
                "historical_exposure_note": "selected from the old P7 128-parent target capture pool; not a fresh unseen cohort",
            })
            confirm_counts[ds] += 1
    _require(confirm_counts == {"hc2": 64, "vidstg": 64}, "confirmation roster must be at most 64 per target")
    confirm_identity = {(r["dataset"], r["source"], r["query_ordinal"]) for r in confirm_rows}
    _require(not (dev_identity & confirm_identity), "confirmation identities overlap development")

    cache = _source_cache_metadata(source_base, fixed, verify_bytes=verify_source_bytes)
    source_meta = {
        "fixed_roster_path": fixed_path.relative_to(ROOT).as_posix(),
        "fixed_roster_sha256": sha256_file(fixed_path),
        "old_p7_design_path": p7_path.relative_to(ROOT).as_posix(),
        "old_p7_design_sha256": sha256_file(p7_path),
        "old_p7_capture_count": len(captures),
        "GT_read": False,
    }
    dev_roster = {
        "schema": ROSTER_SCHEMA,
        "status": "proposed_for_root_review",
        "stage": "development",
        "count": len(dev_rows),
        "counts_by_target": dev_counts,
        "selection": "copied exactly from the fixed original P6 32+32 development roster; no reselection",
        "records": sorted(dev_rows, key=lambda row: (row["dataset"], row["query_ordinal"])),
        "source_metadata": source_meta,
        "source_cache_review": cache,
        "historical_exposure": {
            "exposed": True,
            "scope": "historical P6 development and old P7 candidate/source work; broader prior target-GT score exposure scope is unknown",
            "fresh_generalization_claim": False,
        },
        "GT_read": False,
        "real_RGB_read_by_roster_builder": False,
        "GPU_used_by_roster_builder": False,
    }
    confirm_roster = {
        "schema": ROSTER_SCHEMA,
        "status": "proposed_for_root_review",
        "stage": "confirmation",
        "count": len(confirm_rows),
        "counts_by_target": confirm_counts,
        "eligible_pool_by_target": {"hc2": 96, "vidstg": 96},
        "selection": {
            "method": "sort ascending SHA256 of protocol|dataset|parent_id|source|fixed_query_ordinal|source_input_sha256; take first 64 per target",
            "uses_GT": False,
            "uses_teacher_performance": False,
            "uses_effect_or_length": False,
            "fixed_query_ordinal": True,
        },
        "records": sorted(confirm_rows, key=lambda row: (row["dataset"], row["selection_rank_within_target"])),
        "source_metadata": source_meta,
        "disjointness": {
            "disjoint_from_fixed_P6_development": True,
            "disjoint_from_MLLM_source_screen_64": True,
            "current_method_roster_overlap_audited": False,
            "current_method_overlap_note": "CURRENT_METHOD is a method registry, not a cohort roster; no clean-holdout claim is made. Old P7 and broader prior target-GT score exposure scope is unknown.",
        },
        "historical_exposure": {
            "exposed": True,
            "scope": "all candidates came from the old P7 target capture pool; therefore this is a one-shot historical-exposure confirmation, not a fresh unseen cohort",
            "fresh_generalization_claim": False,
        },
        "source_cache_review": {
            "development_teacher_screen_cache_reuse": "root-rebinding-eligible-for-Source-Projection-Teacher-only",
            "development_student_matrix_cache_reuse": False,
            "confirmation_cache_available": "not_verified",
            "student_spatial_replay_feature_required": True,
            "action": "root authenticate a sealed source cache for this exact confirmation manifest or capture after roster lock; create fresh SpatialReplay features before student fitting",
        },
        "GT_read": False,
        "real_RGB_read_by_roster_builder": False,
        "GPU_used_by_roster_builder": False,
    }
    return dev_roster, confirm_roster


def write_new_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def write_proposed_roster(path: Path, value: Any) -> None:
    """Refresh only this evaluator's unapproved proposed-roster drafts."""
    path = Path(path)
    _require(path in (DEV_ROSTER, CONFIRM_ROSTER),
             "refusing to overwrite a non-owned roster path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")) + "\n"
    temp_path = path.with_name(path.name + ".proposed.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _load_roster(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    roster = read_json(path)
    _reject_ground_truth_fields(roster, "roster")
    _require(roster.get("schema") == ROSTER_SCHEMA, "unsupported roster schema")
    _require(roster.get("status") == "locked", "actual scoring requires a root-locked roster copy")
    _require(roster.get("GT_read") is False, "roster must remain label-free")
    rows = roster.get("records")
    _require(isinstance(rows, list) and len(rows) == roster.get("count"), "roster count mismatch")
    by_case = {_case_id(row): row for row in rows}
    _require(len(by_case) == len(rows), "duplicate roster case ID")
    parent_keys = [(row.get("dataset"), row.get("parent_token") or row.get("source_parent_token"))
                   for row in rows]
    _require(all(ds in ("hc2", "vidstg") and isinstance(parent, str) and parent
                 for ds, parent in parent_keys), "roster requires a private parent token per case")
    _require(len(set(parent_keys)) == len(parent_keys), "roster must contain one fixed query per parent")
    return roster, by_case, sha256_file(path)


def _conditions_for_evaluation_stage(evaluation_stage: str) -> tuple[str, ...]:
    _require(evaluation_stage in EVALUATION_CONDITIONS,
             f"unknown evaluation stage: {evaluation_stage}")
    return tuple(EVALUATION_CONDITIONS[evaluation_stage])


def _verify_source_cache_lock(path: Path, roster: dict[str, Any], roster_sha: str,
                              case_ids: set[str], manifest_sha: str,
                              inference_manifest: dict[str, Any],
                              evaluation_stage: str) -> tuple[dict[str, Any], str]:
    lock = read_json(path)
    _reject_ground_truth_fields(lock, "source_lock")
    _require(lock.get("schema") == SOURCE_LOCK_SCHEMA and lock.get("status") == "sealed",
             "task-specific source cache lock is not sealed")
    _require(lock.get("GT_read") is False, "source cache lock must be label-free")
    _require(lock.get("cohort_sha256") == roster_sha, "source lock does not bind locked roster bytes")
    _require(lock.get("input_manifest_sha256") == manifest_sha,
             "source lock does not bind the exact strict inference manifest")
    _require(lock.get("initialization") == "SourceFrozen" and
             lock.get("dino_state_initialized") is False,
             "source lock is not original SourceFrozen DINO-free initialization")
    expected_scope = ("native_projection" if evaluation_stage == "teacher_screen"
                      else "student_spatial_replay")
    _require(lock.get("cache_scope") == expected_scope,
             f"source lock cache_scope must be {expected_scope} for {evaluation_stage}")
    _require(lock.get("view_id") == "I1", "source lock must identify the I1 view")
    _require(lock.get("input_pixel_clock_contract") == PIXEL_CLOCK_CONTRACT,
             "source lock must bind the root-canonical FFV1 pixel-clock contract")
    source_artifacts = lock.get("source_artifacts")
    required_source_artifacts = {"source_logits", "actions", "records", "projection_interval"}
    _require(isinstance(source_artifacts, list) and required_source_artifacts.issubset(set(source_artifacts)),
             "source lock must bind saved Native logits/actions/records and Projection interval inputs")
    _require(_sha(lock.get("source_checkpoint_sha256")) and _sha(lock.get("source_runtime_sha256")),
             "source lock must pin a source-checkpoint custody digest and runtime")
    cases = lock.get("cases")
    _require(isinstance(cases, dict) and set(cases) == case_ids,
             "source lock case set does not match the full roster")
    roster_dataset = {_case_id(row): row.get("dataset") for row in roster.get("records", [])}
    target_checkpoints: dict[str, set[str]] = {"hc2": set(), "vidstg": set()}
    for case_id, row in cases.items():
        _require(isinstance(row, dict) and _sha(row.get("source_case_sha256")),
                 f"invalid source cache case receipt for {case_id}")
        _require(row.get("view_id") == "I1" and row.get("dino_state_initialized") is False,
                 f"source case {case_id} is not bound to DINO-free I1")
        target = roster_dataset[case_id]
        _require(row.get("pixel_clock_policy") == PIXEL_CLOCK_POLICY_BY_TARGET.get(target),
                 f"source case uses the wrong target-specific physical pixel-clock policy at {case_id}")
        checkpoint_sha = row.get("source_checkpoint_sha256")
        _require(target in target_checkpoints and _sha(checkpoint_sha),
                 f"source case lacks its target-specific physical checkpoint SHA at {case_id}")
        target_checkpoints[target].add(checkpoint_sha)
        _require(row.get("input_row_sha256") == sha256_bytes(canonical_json_bytes(inference_manifest[case_id])),
                 f"source cache input does not bind strict inference row for {case_id}")
        rel = row.get("path")
        _require(isinstance(rel, str), f"missing source cache path for {case_id}")
        _require(sha256_file(repo_path(rel)) == row["source_case_sha256"],
                 f"source cache byte hash mismatch for {case_id}")
        if evaluation_stage == "student_matrix":
            feature_rel = row.get("spatial_replay_feature_path")
            feature_sha = row.get("spatial_replay_feature_sha256")
            _require(isinstance(feature_rel, str) and _sha(feature_sha),
                     f"student stage lacks a fresh SpatialReplay feature receipt for {case_id}")
            _require(row.get("spatial_replay_source_case_sha256") == row["source_case_sha256"] and
                     row.get("spatial_replay_input_row_sha256") == row["input_row_sha256"],
                     f"SpatialReplay feature is not bound to source/input for {case_id}")
            _require(sha256_file(repo_path(feature_rel)) == feature_sha,
                     f"SpatialReplay feature byte hash mismatch for {case_id}")
    _require(all(len(values) == 1 for values in target_checkpoints.values()),
             "each target must bind one consistent original source checkpoint")
    _require(len({next(iter(values)) for values in target_checkpoints.values()}) == 2,
             "HC2 and VidSTG must retain their distinct original source checkpoints")
    expected_checkpoint_custody = sha256_bytes(canonical_json_bytes({
        target: next(iter(target_checkpoints[target])) for target in sorted(target_checkpoints)
    }))
    _require(lock.get("source_checkpoint_sha256") == expected_checkpoint_custody,
             "top-level source checkpoint custody digest does not bind the two per-target checkpoints")
    return lock, sha256_file(path)


def _validate_condition_prediction(condition: str, record: Any,
                                   frame_bounds: list[int], source_case_sha: str,
                                   evaluation_stage: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"condition {condition} must be an object")
    status = record.get("status")
    _require(status in ("ok", "parse_failure", "trajectory_missing", "invalid_interval", "input_unavailable"),
             f"invalid prediction status for {condition}")
    if condition in ("Teacher", "NoParamRefine") and status == "ok":
        _require(record.get("teacher_reference_id") is not None,
                 f"{condition} must identify the frozen teacher output it references")
    if condition in ("Source", "Projection"):
        _require(record.get("source_case_sha256") == source_case_sha,
                 f"{condition} prediction is not bound to the sealed source case")
    if condition == "Projection" and status == "ok":
        _require(record.get("projection_recomputed_from_source") is True and
                 record.get("projection_input_sha256") == source_case_sha,
                 "Projection must be recomputed from this case's sealed raw Projection inputs")
    if condition in ("Direct", "OPD"):
        for field in ("teacher_reference_id", "source_state_sha256",
                      "trainable_parameter_names_sha256", "teacher_observation_set_sha256"):
            _require(record.get(field) is not None, f"{condition} missing matched-mechanism field {field}")
        for field in ("source_state_sha256", "trainable_parameter_names_sha256",
                      "teacher_observation_set_sha256"):
            _require(_sha(record.get(field)), f"{condition} has invalid {field}")
        _require(_int(record.get("optimizer_steps")) and record["optimizer_steps"] >= 0,
                 f"{condition} optimizer step count must be recorded")
        budget = _declared_step_budget(record, condition)
        _require(record["optimizer_steps"] <= budget,
                 f"{condition} actual optimizer steps exceed its declared step budget")
        shift = record.get("actual_output_shift_l2")
        _require(shift is None or (_finite_number(shift) and shift >= 0),
                 f"{condition} has invalid actual decoded-output displacement")
    if status != "ok":
        _require(isinstance(record.get("reason"), str) and bool(record["reason"]),
                 f"failure row for {condition} must retain a reason")
        return {"status": status, "reason": record["reason"], "record": record, "metrics": None}
    _require(record.get("output_kind") == PREDICTION_OUTPUT_KIND[condition],
             f"{condition} output does not come from its declared pipeline")
    ids = record.get("frame_ids")
    boxes = record.get("boxes_cxcywh_normalized")
    interval = record.get("interval_frame_bounds")
    _require(isinstance(ids, list) and len(ids) >= 2 and all(_int(x) for x in ids),
             f"invalid physical frame IDs for {condition}")
    _require(ids == sorted(set(ids)), f"frame IDs for {condition} must be strictly increasing")
    _require(all(frame_bounds[0] <= x < frame_bounds[1] for x in ids),
             f"frame IDs for {condition} escape inference frameBounds")
    _require(isinstance(boxes, list) and len(boxes) == len(ids),
             f"box/physical-frame count mismatch for {condition}")
    if condition == "Teacher":
        _require(record.get("trajectory_completeness") in
                 ("complete", "physical_full_grid_with_presence_mask"),
                 "Teacher output must declare its physical full-grid trajectory")
        _require(ids == list(range(frame_bounds[0], frame_bounds[1])),
                 "Teacher must emit the full physical frame grid from canonical frameBounds")
        present = record.get("present_mask")
        _require(isinstance(present, list) and len(present) == len(ids) and
                 all(isinstance(value, bool) for value in present),
                 "Teacher must include one boolean present/known-identity mask per physical frame")
        for is_present, box in zip(present, boxes):
            if is_present:
                _require(isinstance(box, list) and len(box) == 4 and
                         all(_finite_number(value) for value in box),
                         "present Teacher frames require a finite normalized cxcywh box")
            else:
                _require(box is None,
                         "missing/unknown-identity Teacher frames must carry null, not a synthetic box")
    else:
        _require(record.get("trajectory_completeness") == "complete",
                 f"{condition} trajectory is not explicitly complete")
        _require(all(isinstance(box, list) and len(box) == 4 and
                     all(_finite_number(x) for x in box) for box in boxes),
                 f"invalid normalized cxcywh boxes for {condition}")
    _require(isinstance(interval, list) and len(interval) == 2 and all(_int(x) for x in interval)
             and interval[1] > interval[0], f"invalid half-open physical interval for {condition}")
    _require(frame_bounds[0] <= interval[0] < interval[1] <= frame_bounds[1],
             f"predicted interval for {condition} escapes canonical frameBounds")
    if condition in ("Direct", "OPD"):
        _require(_finite_number(record.get("actual_output_shift_l2")) and record["actual_output_shift_l2"] >= 0,
                 f"{condition} must record actual decoded-output displacement")
    return {"status": "ok", "record": record, "metrics": None}


def _declared_step_budget(record: dict[str, Any], condition: str) -> int:
    """Return the planned Direct/OPD update budget, accepting the two wire aliases."""
    present = [field for field in ("step_budget", "declared_steps") if field in record]
    _require(bool(present), f"{condition} must record declared step_budget")
    values = [record[field] for field in present]
    _require(all(_int(value) and value >= 0 for value in values),
             f"{condition} declared step budget must be a nonnegative integer")
    _require(len(set(values)) == 1,
             f"{condition} step_budget and declared_steps aliases disagree")
    return int(values[0])


def _validate_prediction_file(value: Any, case_id: str, roster_row: dict[str, Any],
                              source_case_sha: str, manifest_sha: str,
                              frame_bounds: list[int], evaluation_stage: str,
                              spatial_replay_feature_sha: str | None = None) -> dict[str, Any]:
    _require(isinstance(value, dict) and value.get("schema") == PREDICTION_CASE_SCHEMA,
             f"bad prediction case schema for {case_id}")
    _reject_ground_truth_fields(value, f"prediction.{case_id}")
    _require(value.get("case_id") == case_id and value.get("GT_read") is False,
             f"prediction case identity/GT guard mismatch for {case_id}")
    _require(value.get("evaluation_stage") == evaluation_stage,
             f"prediction case evaluation stage mismatch for {case_id}")
    _require(value.get("input_manifest_sha256") == manifest_sha,
             f"prediction case is not bound to strict input manifest for {case_id}")
    _require(value.get("source_case_sha256") == source_case_sha,
             f"prediction case is not bound to SourceFrozen+I1 source cache for {case_id}")
    if evaluation_stage == "student_matrix":
        _require(_sha(spatial_replay_feature_sha) and
                 value.get("spatial_replay_feature_sha256") == spatial_replay_feature_sha and
                 value.get("spatial_replay_source_case_sha256") == source_case_sha,
                 f"student prediction is not bound to its fresh SpatialReplay feature for {case_id}")
    _require(value.get("dataset") == roster_row.get("dataset") and
             value.get("query_ordinal") == roster_row.get("query_ordinal"),
             f"prediction case identity differs from locked roster for {case_id}")
    conditions = value.get("conditions")
    expected_conditions = set(_conditions_for_evaluation_stage(evaluation_stage))
    _require(isinstance(conditions, dict) and set(conditions) == expected_conditions,
             f"prediction conditions incomplete for {case_id}")
    parsed = {name: _validate_condition_prediction(name, conditions[name], frame_bounds,
                                                   source_case_sha, evaluation_stage)
              for name in expected_conditions}
    if evaluation_stage == "student_matrix":
        direct, opd = conditions["Direct"], conditions["OPD"]
        matched_fields = ("teacher_reference_id", "source_state_sha256",
                          "trainable_parameter_names_sha256", "teacher_observation_set_sha256")
        _require(all(direct.get(field) == opd.get(field) for field in matched_fields),
                 f"Direct and OPD are not matched on teacher, source, parameters, and observations for {case_id}")
        direct_budget = _declared_step_budget(direct, "Direct")
        opd_budget = _declared_step_budget(opd, "OPD")
        _require(direct_budget == opd_budget,
                 f"Direct and OPD declared step budgets differ for {case_id}")
        teacher_record = conditions["Teacher"]
        if teacher_record.get("status") == "ok":
            _require(direct.get("status") != "ok" or
                     direct.get("teacher_reference_id") == teacher_record.get("teacher_reference_id"),
                     f"Direct is not bound to this case's sealed Teacher output for {case_id}")
            _require(opd.get("status") != "ok" or
                     opd.get("teacher_reference_id") == teacher_record.get("teacher_reference_id"),
                     f"OPD is not bound to this case's sealed Teacher output for {case_id}")
            no_param = conditions["NoParamRefine"]
            _require(no_param.get("status") != "ok" or
                     no_param.get("teacher_reference_id") == teacher_record.get("teacher_reference_id"),
                     f"NoParamRefine is not bound to this case's sealed Teacher output for {case_id}")
    cost = value.get("teacher_cost")
    _require(isinstance(cost, dict) and set(cost) == set(TEACHER_COST_FIELDS),
             f"teacher full-pipeline cost schema incomplete for {case_id}")
    for key, val in cost.items():
        _require(val is None or (_finite_number(val) and float(val) >= 0),
                 f"invalid cost field {key} for {case_id}")
    out = dict(value)
    out["conditions"] = parsed
    return out


def _preflight_prediction_seal(*, seal_path: Path, roster: dict[str, Any],
                               roster_rows: dict[str, dict[str, Any]], roster_sha: str,
                               inference_manifest: dict[str, Any], manifest_sha: str,
                               source_lock: dict[str, Any], source_lock_sha: str,
                               evaluation_stage: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    seal = read_json(seal_path)
    _reject_ground_truth_fields(seal, "prediction_seal")
    stage = roster.get("stage")
    _require(seal.get("schema") == PREDICTION_SEAL_SCHEMA and seal.get("status") == "sealed",
             "complete label-free prediction barrier is required")
    _require(seal.get("stage") == stage, "prediction seal stage does not match roster")
    _require(seal.get("evaluation_stage") == evaluation_stage,
             "prediction seal evaluation stage does not match requested score stage")
    _require(seal.get("GT_read") is False, "prediction seal must state GT_read=false")
    _require(seal.get("roster_sha256") == roster_sha, "prediction seal does not bind locked roster")
    _require(seal.get("input_manifest_sha256") == manifest_sha,
             "prediction seal does not bind the strict inference manifest")
    _require(seal.get("source_cache_lock_sha256") == source_lock_sha,
             "prediction seal does not bind the task-specific source cache lock")
    if stage == "confirmation":
        _require(seal.get("config_lock_sha256") and _sha(seal.get("config_lock_sha256")),
                 "confirmation seal must bind a frozen configuration")
    else:
        _require(seal.get("config_lock_sha256") in (None, ""),
                 "development predictions must not claim a confirmation configuration lock")
    expected_conditions = list(_conditions_for_evaluation_stage(evaluation_stage))
    _require(seal.get("conditions") == expected_conditions, "prediction seal condition list mismatch")
    case_files = seal.get("case_files")
    expected_ids = set(roster_rows)
    _require(isinstance(case_files, dict) and set(case_files) == expected_ids,
             "prediction seal does not enumerate every roster case exactly once")
    _require(seal.get("count") == len(expected_ids) and len(case_files) == len(expected_ids),
             "prediction seal case count mismatch")
    parsed: dict[str, dict[str, Any]] = {}
    for case_id in sorted(expected_ids):
        ref = case_files[case_id]
        _require(isinstance(ref, dict) and _sha(ref.get("sha256")),
                 f"invalid sealed prediction receipt for {case_id}")
        path = repo_path(ref.get("path"))
        _require(path.is_file() and sha256_file(path) == ref["sha256"],
                 f"prediction case bytes do not match seal for {case_id}")
        data = read_json(path)
        input_row = inference_manifest[case_id]
        source_case_sha = source_lock["cases"][case_id]["source_case_sha256"]
        feature_sha = source_lock["cases"][case_id].get("spatial_replay_feature_sha256")
        parsed[case_id] = _validate_prediction_file(data, case_id, roster_rows[case_id],
                                                    source_case_sha, manifest_sha,
                                                    input_row["frameBounds"], evaluation_stage,
                                                    feature_sha)
    return seal, parsed


def _load_metric_functions(metric_contract: dict[str, Any]) -> tuple[Any, Any]:
    pins = metric_contract.get("metric_module_pins")
    _require(isinstance(pins, dict) and pins, "missing original metric module pins")
    for rel, expected in pins.items():
        _require(_sha(expected), f"invalid pinned metric SHA for {rel}")
        _require(sha256_file(repo_path(rel)) == expected,
                 f"original metric module changed after lock: {rel}")
    sys.path.insert(0, str(ROOT))
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    return official, dense_metric


def _validate_truth(label: Any, case_id: str) -> tuple[dict[int, list[float]], list[int]] | None:
    if not isinstance(label, dict) or label.get("schema") != LABEL_CASE_SCHEMA or label.get("case_id") != case_id:
        return None
    truth = label.get("truth_boxes_xyxy")
    span = label.get("span")
    if not isinstance(truth, dict) or not truth or not isinstance(span, list) or len(span) != 2:
        return None
    if not all(_int(x) for x in span) or span[1] <= span[0]:
        return None
    result: dict[int, list[float]] = {}
    for frame, box in truth.items():
        try:
            frame_id = int(frame)
        except (TypeError, ValueError):
            return None
        if not isinstance(box, list) or len(box) != 4 or not all(_finite_number(x) for x in box):
            return None
        result[frame_id] = [float(x) for x in box]
    return result, [int(span[0]), int(span[1])]


def _teacher_physical_dense_metric(record: dict[str, Any], row: dict[str, Any],
                                  truth: dict[int, list[float]], span: list[int],
                                  dataset: str) -> tuple[dict[str, float], dict[str, float | int]]:
    """Score Teacher's physical full grid without bridging absent/unknown identities."""
    width, height = float(row["input"]["width"]), float(row["input"]["height"])
    ids = record["frame_ids"]
    present = record["present_mask"]
    boxes = record["boxes_cxcywh_normalized"]
    iou_by_frame: dict[int, float] = {}
    for frame_id, is_present, box in zip(ids, present, boxes):
        if not is_present:
            iou_by_frame[int(frame_id)] = 0.0
            continue
        cx, cy, bw, bh = [float(value) for value in box]
        pred = [(cx - bw / 2.0) * width, (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width, (cy + bh / 2.0) * height]
        if dataset == "hc2":
            pred = [max(value, 0.0) for value in pred]
        gt = [float(value) for value in truth[int(frame_id)]] if int(frame_id) in truth else None
        if gt is None:
            continue
        ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
        ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        pred_area = max(pred[2] - pred[0], 0.0) * max(pred[3] - pred[1], 0.0)
        gt_area = max(gt[2] - gt[0], 0.0) * max(gt[3] - gt[1], 0.0)
        union = pred_area + gt_area - inter
        iou_by_frame[int(frame_id)] = inter / union if union > 0 else 0.0

    truth_ids = sorted(int(frame_id) for frame_id in truth)
    if not truth_ids:
        raise ContractError("Teacher dense metric requires nonempty official GT support")
    support_ious = [iou_by_frame.get(frame_id, 0.0) for frame_id in truth_ids]
    spatial = math.fsum(support_ious) / len(support_ious)
    a, z = map(int, record["interval_frame_bounds"])
    g, h = map(int, span)
    overlap = max(0, min(z, h) - max(a, g))
    temporal_union = z - a + h - g - overlap
    supported_overlap = [iou_by_frame.get(frame_id, 0.0)
                         for frame_id in truth_ids if max(a, g) <= frame_id < min(z, h)]
    tube = math.fsum(supported_overlap) / max(max(z, h) - min(a, g), 1)
    metrics = {"m_vIoU": float(tube), "m_tIoU": float(overlap / temporal_union),
               "m_sIoU_dense_GT": float(spatial),
               "R@0.3_strict": float(tube > 0.30),
               "R@0.5_strict": float(tube > 0.50)}
    present_count = sum(bool(value) for value in present)
    details = {"present_frames": int(present_count), "physical_frames": len(ids),
               "present_fraction": float(present_count / len(ids))}
    return metrics, details


def _load_all_truths_after_gate(labels_manifest_path: Path, expected_case_ids: set[str]) -> tuple[dict[str, Any], str]:
    """This is the sole production label-opening function; call only after all seals pass."""
    manifest = read_json(labels_manifest_path)
    _require(isinstance(manifest, dict) and manifest.get("schema") == LABELS_SCHEMA,
             "unsupported labels manifest schema")
    _require(manifest.get("status") == "sealed" and manifest.get("case_ids") == sorted(expected_case_ids),
             "labels manifest must seal the complete expected case set")
    case_files = manifest.get("case_files")
    _require(isinstance(case_files, dict) and set(case_files) == expected_case_ids,
             "labels manifest case set mismatch")
    truths: dict[str, Any] = {}
    for case_id in sorted(expected_case_ids):
        receipt = case_files[case_id]
        _require(isinstance(receipt, dict) and _sha(receipt.get("sha256")),
                 f"bad label file receipt for {case_id}")
        path = repo_path(receipt.get("path"))
        _require(path.is_file() and sha256_file(path) == receipt["sha256"],
                 f"label bytes do not match label manifest for {case_id}")
        truths[case_id] = read_json(path)
    return truths, sha256_file(labels_manifest_path)


def _score_prediction(condition: str, parsed: dict[str, Any], manifest_row: dict[str, Any], dataset: str,
                      truth_record: Any, case_id: str, official: Any, dense_metric: Any) -> dict[str, Any]:
    unknown = {metric: None for metric in METRICS}
    valid_truth = _validate_truth(truth_record, case_id)
    if valid_truth is None:
        prediction_status = parsed["status"]
        return {"status": "unknown_truth", "prediction_status": prediction_status,
                "failure_flag": prediction_status != "ok",
                "reason": "missing_or_invalid_truth_support", **unknown}
    truth, span = valid_truth
    if parsed["status"] != "ok":
        if condition == "Teacher":
            return {"status": parsed["status"], "prediction_status": parsed["status"],
                    "failure_flag": True,
                    "reason": parsed["reason"],
                    "primary_failure_policy": "known_GT_prediction_failure_scores_zero",
                    **{metric: 0.0 for metric in METRICS}}
        return {"status": parsed["status"], "prediction_status": parsed["status"],
                "failure_flag": True, "reason": parsed["reason"], **unknown}
    record = parsed["record"]
    width, height = manifest_row["dimensions"]
    metric_row = {"frame_ids": record["frame_ids"], "input": {"width": width, "height": height}}
    interval = record["interval_frame_bounds"]
    if condition == "Teacher":
        try:
            metrics, details = _teacher_physical_dense_metric(record, metric_row, truth, span, dataset)
        except Exception as exc:
            return {"status": "metric_failure", "prediction_status": "ok",
                    "failure_flag": True, "reason": type(exc).__name__, **unknown}
        return {"status": "scored", "prediction_status": "ok", "failure_flag": False,
                **metrics, "teacher_frame_support": details,
                "teacher_missing_frame_policy": "each absent or unknown-identity physical frame contributes IoU zero; no box interpolation"}
    try:
        reference = official(record["boxes_cxcywh_normalized"], metric_row,
                            truth, span, interval, dataset, gt_space=False)
        independent = dense_metric(record["boxes_cxcywh_normalized"], metric_row,
                                   truth, span, interval, dataset)[0]
    except Exception as exc:
        return {"status": "metric_failure", "prediction_status": "ok",
                "failure_flag": True, "reason": type(exc).__name__, **unknown}
    expected = {"v": reference["v"], "t": reference["t"], "s": reference["s"]}
    for key in ("v", "t", "s"):
        if not _finite_number(expected[key]) or not math.isfinite(float(independent[key])):
            return {"status": "metric_failure", "prediction_status": "ok",
                    "failure_flag": True, "reason": "nonfinite_metric", **unknown}
        if abs(float(expected[key]) - float(independent[key])) > 2e-9:
            raise ContractError(f"official and independent dense metric disagree: {case_id}/{condition}/{key}")
    v = float(expected["v"])
    return {
        "status": "scored",
        "prediction_status": "ok",
        "failure_flag": False,
        "reason": None,
        "m_vIoU": v,
        "m_tIoU": float(expected["t"]),
        "m_sIoU_dense_GT": float(expected["s"]),
        "R@0.3_strict": float(v > 0.30),
        "R@0.5_strict": float(v > 0.50),
    }


def _parent_values(case_rows: list[dict[str, Any]], condition: str, metric: str) -> list[tuple[str, float | None]]:
    out = []
    for row in case_rows:
        value = row["conditions"][condition].get(metric)
        out.append((row["parent_token"], float(value) if _finite_number(value) else None))
    return out


def _bootstrap_summary(values: list[float | None], *, seed: int) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if _finite_number(value)], dtype=np.float64)
    total = len(values)
    if not len(finite):
        return {"defined_parents": 0, "unknown_parents": total, "mean": None, "ci95": None,
                "min": None, "max": None, "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": seed}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(finite), size=(BOOTSTRAP_DRAWS, len(finite)))
    boot_means = finite[indices].mean(axis=1)
    return {
        "defined_parents": int(len(finite)),
        "unknown_parents": int(total - len(finite)),
        "mean": float(math.fsum(finite.tolist()) / len(finite)),
        "ci95": [float(x) for x in np.quantile(boot_means, [0.025, 0.975], method="linear")],
        "min": float(finite.min()),
        "max": float(finite.max()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def _seed_for(*parts: str) -> int:
    digest = hashlib.sha256((str(BOOTSTRAP_SEED) + "|" + "|".join(parts)).encode()).digest()
    return BOOTSTRAP_SEED + int.from_bytes(digest[:4], "big")


def _paired_vIoU_disposition_counts(rows: list[dict[str, Any]], left: str,
                                    right: str) -> dict[str, int]:
    """Count complete-cohort positive/negative/neutral/unknown paired vIoU tails."""
    counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
    for row in rows:
        left_value = row["conditions"][left].get("m_vIoU")
        right_value = row["conditions"][right].get("m_vIoU")
        if not (_finite_number(left_value) and _finite_number(right_value)):
            counts["unknown"] += 1
        elif float(left_value) > float(right_value):
            counts["positive"] += 1
        elif float(left_value) < float(right_value):
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
    counts["cohort_count"] = len(rows)
    return counts


def _summarize_stage(case_metrics: list[dict[str, Any]], dataset_counts: dict[str, int],
                    evaluation_stage: str) -> dict[str, Any]:
    conditions = _conditions_for_evaluation_stage(evaluation_stage)
    contrasts = EVALUATION_CONTRASTS[evaluation_stage]
    output: dict[str, Any] = {}
    for dataset in ("hc2", "vidstg"):
        rows = [row for row in case_metrics if row["dataset"] == dataset]
        domain: dict[str, Any] = {"case_count": len(rows), "condition_summary": {},
                                  "conditional_success_summary": {},
                                  "paired_contrasts": {}, "failure_counts": {},
                                  "teacher_cost_summary": {},
                                  "primary_score_policy": "Teacher known-GT prediction failures are zero with a separate failure flag; missing/invalid GT remains null; success-conditional summaries are secondary"}
        for condition in conditions:
            domain["condition_summary"][condition] = {
                metric: _bootstrap_summary([row["conditions"][condition].get(metric) for row in rows],
                                           seed=_seed_for(dataset, condition, metric))
                for metric in METRICS
            }
            domain["conditional_success_summary"][condition] = {
                metric: _bootstrap_summary([
                    row["conditions"][condition].get(metric)
                    if row["conditions"][condition].get("status") == "scored" else None
                    for row in rows
                ], seed=_seed_for(dataset, condition, "conditional_success", metric))
                for metric in METRICS
            }
            status_counts: dict[str, int] = {}
            prediction_status_counts: dict[str, int] = {}
            for row in rows:
                metrics_row = row["conditions"][condition]
                status = metrics_row.get("status", "unknown")
                prediction_status = metrics_row.get("prediction_status", status)
                status_counts[status] = status_counts.get(status, 0) + 1
                prediction_status_counts[prediction_status] = prediction_status_counts.get(prediction_status, 0) + 1
            domain["failure_counts"][condition] = {
                "status_counts": status_counts,
                "prediction_status_counts": prediction_status_counts,
                "failure_flag_cases": sum(
                    bool(row["conditions"][condition].get("failure_flag")) for row in rows
                ),
                "primary_numeric_cases": sum(
                    _finite_number(row["conditions"][condition].get("m_vIoU")) for row in rows),
            }
        for contrast, (left, right) in contrasts.items():
            domain["paired_contrasts"][contrast] = {}
            for metric in METRICS:
                deltas: list[float | None] = []
                for row in rows:
                    lv = row["conditions"][left].get(metric)
                    rv = row["conditions"][right].get(metric)
                    deltas.append(float(lv) - float(rv) if _finite_number(lv) and _finite_number(rv) else None)
                domain["paired_contrasts"][contrast][metric] = _bootstrap_summary(
                    deltas, seed=_seed_for(dataset, contrast, metric))
        for cost_field in TEACHER_COST_FIELDS:
            vals = [row["teacher_cost"].get(cost_field) for row in rows]
            known = [float(v) for v in vals if _finite_number(v)]
            domain["teacher_cost_summary"][cost_field] = {
                "known_cases": len(known),
                "missing_cases": len(vals) - len(known),
                "mean": float(math.fsum(known) / len(known)) if known else None,
                "median": float(statistics.median(known)) if known else None,
                "max": max(known) if known else None,
                "extrapolated": False,
            }
        if evaluation_stage != "student_matrix":
            domain["gap_closure"] = {
                "status": "not_applicable_stage",
                "reason": "student arms are not part of the sealed teacher_screen",
            }
            output[dataset] = domain
            continue
        gap_rows = [row for row in rows if all(_finite_number(row["conditions"][name].get("m_vIoU"))
                                               for name in ("Source", "Teacher", "Direct", "OPD"))]
        gap_closure: dict[str, Any] = {"common_defined_parents": len(gap_rows),
                                      "denominator_epsilon": GAP_EPSILON,
                                      "policy": "ratio of parent-macro gaps; no per-query division and no clipping"}
        source_values = [float(row["conditions"]["Source"]["m_vIoU"]) for row in gap_rows]
        teacher_gap = [float(row["conditions"]["Teacher"]["m_vIoU"]) - source
                       for row, source in zip(gap_rows, source_values)]
        mean_teacher_gap = math.fsum(teacher_gap) / len(teacher_gap) if teacher_gap else None
        gap_closure["mean_teacher_minus_source"] = mean_teacher_gap
        if mean_teacher_gap is None or mean_teacher_gap <= GAP_EPSILON:
            gap_closure["status"] = "not_reported_teacher_gap_nonpositive_or_near_zero"
            gap_closure["Direct"] = None
            gap_closure["OPD"] = None
        else:
            gap_closure["status"] = "reported"
            for student in ("Direct", "OPD"):
                numerator = [float(row["conditions"][student]["m_vIoU"]) - source
                             for row, source in zip(gap_rows, source_values)]
                point = (math.fsum(numerator) / len(numerator)) / mean_teacher_gap
                rng = np.random.default_rng(_seed_for(dataset, "gap_closure", student))
                draws = rng.integers(0, len(gap_rows), size=(BOOTSTRAP_DRAWS, len(gap_rows)))
                num_array = np.asarray(numerator, dtype=np.float64)
                den_array = np.asarray(teacher_gap, dtype=np.float64)
                den_draw = den_array[draws].mean(axis=1)
                valid = den_draw > GAP_EPSILON
                ratio_draws = num_array[draws].mean(axis=1)[valid] / den_draw[valid]
                gap_closure[student] = {
                    "ratio_of_parent_macro_gaps": float(point),
                    "ci95": [float(x) for x in np.quantile(ratio_draws, [0.025, 0.975], method="linear")]
                    if len(ratio_draws) else None,
                    "valid_bootstrap_draws": int(len(ratio_draws)),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": _seed_for(dataset, "gap_closure", student),
                }
        domain["gap_closure"] = gap_closure
        output[dataset] = domain
    return output


def score_stage(*, stage: str, evaluation_stage: str,
                roster_path: Path, inference_manifest_path: Path,
                source_lock_path: Path, prediction_seal_path: Path,
                metric_contract_path: Path, labels_manifest_path: Path,
                output_path: Path, config_lock_path: Path | None = None) -> dict[str, Any]:
    """Score only after all label-free stage artifacts have passed preflight."""
    conditions_to_score = _conditions_for_evaluation_stage(evaluation_stage)
    roster, rows_by_case, roster_sha = _load_roster(roster_path)
    _require(roster.get("stage") == stage, "requested stage does not match locked roster")
    case_ids = set(rows_by_case)
    manifest = read_json(inference_manifest_path)
    validate_inference_manifest(manifest, case_ids)
    manifest_sha = sha256_file(inference_manifest_path)
    source_lock, source_lock_sha = _verify_source_cache_lock(
        source_lock_path, roster, roster_sha, case_ids, manifest_sha, manifest,
        evaluation_stage)

    config_lock = None
    config_lock_sha = None
    if stage == "confirmation":
        _require(config_lock_path is not None, "confirmation scoring requires a frozen config lock")
        config_lock = read_json(config_lock_path)
        _require(config_lock.get("status") == "frozen" and
                 config_lock.get("confirmation_readonly") is True and
                 config_lock.get("autotuning") is False,
                 "confirmation requires frozen, read-only, non-autotuning configuration")
        _require(config_lock.get("GT_read") is False,
                 "configuration must have been frozen without reading confirmation GT")
        config_lock_sha = sha256_file(config_lock_path)

    seal, predictions = _preflight_prediction_seal(
        seal_path=prediction_seal_path,
        roster=roster,
        roster_rows=rows_by_case,
        roster_sha=roster_sha,
        inference_manifest=manifest,
        manifest_sha=manifest_sha,
        source_lock=source_lock,
        source_lock_sha=source_lock_sha,
        evaluation_stage=evaluation_stage,
    )
    if stage == "confirmation":
        _require(seal.get("config_lock_sha256") == config_lock_sha,
                 "confirmation prediction seal does not bind frozen config bytes")
        _require(config_lock.get("frozen_before_confirmation_predictions") is True,
                 "configuration freeze must precede confirmation predictions")

    # No labels manifest or annotation payload is opened until the complete
    # selected-stage seal, every selected arm per case, the exact input whitelist,
    # and source/config locks pass.
    metric_contract = read_json(metric_contract_path)
    _require(metric_contract.get("schema") == "stvg_target_teacher_cpu_contract_v1",
             "unsupported CPU evaluation contract")
    official, dense_metric = _load_metric_functions(metric_contract)
    truths, labels_manifest_sha = _load_all_truths_after_gate(labels_manifest_path, case_ids)

    case_metrics: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        roster_row = rows_by_case[case_id]
        pred_case = predictions[case_id]
        input_row = manifest[case_id]
        conditions = {
            name: _score_prediction(name, pred_case["conditions"][name], input_row,
                                    roster_row["dataset"], truths[case_id], case_id,
                                    official, dense_metric)
            for name in conditions_to_score
        }
        if evaluation_stage == "student_matrix":
            for name in ("Direct", "OPD"):
                record = pred_case["conditions"][name]["record"]
                budget = _declared_step_budget(record, name)
                conditions[name]["training_audit"] = {
                    "optimizer_steps_actual": record["optimizer_steps"],
                    "step_budget_declared": budget,
                    "teacher_reference_id": record["teacher_reference_id"],
                    "source_state_sha256": record["source_state_sha256"],
                    "trainable_parameter_names_sha256": record["trainable_parameter_names_sha256"],
                    "teacher_observation_set_sha256": record["teacher_observation_set_sha256"],
                    "actual_output_shift_l2": record.get("actual_output_shift_l2"),
                }
        case_metrics.append({
            "case_token": sha256_bytes(case_id.encode())[:20],
            "parent_token": roster_row.get("parent_token") or roster_row.get("source_parent_token"),
            "dataset": roster_row["dataset"],
            "conditions": conditions,
            "teacher_cost": pred_case.get("teacher_cost"),
        })
    dataset_counts = {ds: sum(row["dataset"] == ds for row in case_metrics) for ds in ("hc2", "vidstg")}
    summaries = _summarize_stage(case_metrics, dataset_counts, evaluation_stage)
    output = {
        "schema": "stvg_target_teacher_postseal_readback_v1",
        "status": "pass",
        "stage": stage,
        "evaluation_stage": evaluation_stage,
        "cohort_count": len(case_metrics),
        "counts_by_target": dataset_counts,
        "condition_names": list(conditions_to_score),
        "metric_names": list(METRICS),
        "metric_semantics": {
            "coordinate_input": "normalized cxcywh; scorer converts with original dimensions through pinned official metric path",
            "spatial_metric": "m_sIoU_dense_GT over the official fixed annotation support",
            "temporal_metric": "m_tIoU over original official interval endpoints",
            "tube_metric": "m_vIoU using original official dense interpolation and dataset-specific evaluator",
            "strict_recall": "R@0.3 and R@0.5 are indicators of vIoU > threshold, not >=",
            "missing_prediction": "null metric with retained failure status; never zero-filled",
        },
        "summary_by_target": summaries,
        "case_metrics": case_metrics,
        "provenance": {
            "roster_sha256": roster_sha,
            "inference_manifest_sha256": manifest_sha,
            "source_cache_lock_sha256": source_lock_sha,
            "prediction_seal_sha256": sha256_file(prediction_seal_path),
            "labels_manifest_sha256": labels_manifest_sha,
            "config_lock_sha256": config_lock_sha,
            "metric_module_pins": metric_contract["metric_module_pins"],
        },
        "teacher_cost_policy": "report only observed fields; missing values remain null and are not extrapolated",
        "confirmation_policy": "read-only after frozen config; this scorer does not select conditions or tune parameters",
        "parent_paired_bootstrap": {"draws": BOOTSTRAP_DRAWS, "base_seed": BOOTSTRAP_SEED,
                                     "unit": "parent within target; only defined matched pairs enter each contrast"},
        "CPU_only": True,
        "GPU_used": False,
        "GT_read": True,
        "time": time.time(),
    }
    write_new_json(output_path, output)
    return output


def selftest() -> dict[str, Any]:
    """Run synthetic CPU contracts plus read-only pinned metric/contract checks."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    metric_contract = read_json(CONTRACTS)
    official, dense_metric = _load_metric_functions(metric_contract)
    validate_inference_row({"realvideo": "synthetic.mp4", "query": "synthetic object",
                            "fps": 25.0, "frameBounds": [0, 4], "dimensions": [100, 80]})
    rejected_extra = False
    try:
        validate_inference_row({"realvideo": "synthetic.mp4", "query": "synthetic object",
                                "fps": 25.0, "frameBounds": [0, 4], "dimensions": [100, 80],
                                "ground_truth": "forbidden"})
    except ContractError:
        rejected_extra = True
    _require(rejected_extra, "inference whitelist accepted a forbidden field")

    # Use the pinned original metric functions with a synthetic in-memory label only.
    row = {"frame_ids": [0, 1, 2, 3], "input": {"width": 100, "height": 80}}
    boxes = [[0.5, 0.5, 0.4, 0.4]] * 4
    truth = {i: [30.0, 24.0, 70.0, 56.0] for i in range(4)}
    perfect = {}
    for dataset in ("vidstg", "hc2"):
        perfect[dataset] = official(boxes, row, truth, [0, 4], [0, 4], dataset, gt_space=False)
        independent = dense_metric(boxes, row, truth, [0, 4], [0, 4], dataset)[0]
        _require(all(abs(float(perfect[dataset][k]) - float(independent[k])) <= 2e-9
                      for k in ("v", "t", "s")),
                 f"synthetic official and independent metrics diverged for {dataset}")
        _require(all(abs(float(perfect[dataset][k]) - 1.0) <= 1e-12 for k in ("v", "t", "s")),
                 f"synthetic perfect control did not score 1 for {dataset}")
    _require(float(perfect["vidstg"]["v"] > 0.30) == 1.0 and float(0.30 > 0.30) == 0.0,
             "strict R threshold contract failed")

    synthetic_teacher_record = {
        "status": "ok",
        "teacher_reference_id": "synthetic-teacher",
        "output_kind": "teacher_pipeline",
        "trajectory_completeness": "physical_full_grid_with_presence_mask",
        "frame_ids": [0, 1, 2],
        "present_mask": [True, False, True],
        "boxes_cxcywh_normalized": [[0.5, 0.5, 0.4, 0.4], None,
                                    [0.5, 0.5, 0.4, 0.4]],
        "interval_frame_bounds": [0, 3],
    }
    checked_teacher = _validate_condition_prediction(
        "Teacher", synthetic_teacher_record, [0, 3], "d" * 64, "teacher_screen")
    synthetic_teacher_truth = {i: [30.0, 30.0, 70.0, 70.0] for i in range(3)}
    teacher_metric, teacher_support = _teacher_physical_dense_metric(
        checked_teacher["record"], {"frame_ids": [0, 1, 2],
                                    "input": {"width": 100, "height": 100}},
        synthetic_teacher_truth, [0, 3], "vidstg")
    _require(abs(teacher_metric["m_vIoU"] - (2.0 / 3.0)) <= 1e-12 and
             abs(teacher_metric["m_sIoU_dense_GT"] - (2.0 / 3.0)) <= 1e-12 and
             teacher_metric["m_tIoU"] == 1.0 and teacher_support["present_frames"] == 2,
             "Teacher missing-frame mask was interpolated or scored as a fabricated box")
    teacher_all_present = dict(synthetic_teacher_record)
    teacher_all_present["present_mask"] = [True, True, True]
    teacher_all_present["boxes_cxcywh_normalized"] = [
        [0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]
    ]
    full_row = {"frame_ids": [0, 1, 2], "input": {"width": 100, "height": 100}}
    for dataset in ("vidstg", "hc2"):
        manual_full = _teacher_physical_dense_metric(
            teacher_all_present, full_row, synthetic_teacher_truth, [0, 3], dataset)[0]
        official_full = official(teacher_all_present["boxes_cxcywh_normalized"], full_row,
                                 synthetic_teacher_truth, [0, 3], [0, 3], dataset,
                                 gt_space=False)
        _require(abs(manual_full["m_vIoU"] - official_full["v"]) <= 2e-9 and
                 abs(manual_full["m_sIoU_dense_GT"] - official_full["s"]) <= 2e-9 and
                 abs(manual_full["m_tIoU"] - official_full["t"]) <= 2e-9,
                 f"Teacher full-grid metric disagrees with {dataset} official metric without missing frames")
    teacher_partial_interval = dict(teacher_all_present)
    teacher_partial_interval["frame_ids"] = [0, 1, 2, 3]
    teacher_partial_interval["boxes_cxcywh_normalized"] = [
        [0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4],
        [0.15, 0.15, 0.1, 0.1], [0.15, 0.15, 0.1, 0.1]
    ]
    teacher_partial_interval["present_mask"] = [True, True, True, True]
    teacher_partial_interval["interval_frame_bounds"] = [1, 4]
    partial_row = {"frame_ids": [0, 1, 2, 3], "input": {"width": 100, "height": 100}}
    partial_manual = _teacher_physical_dense_metric(
        teacher_partial_interval, partial_row, synthetic_teacher_truth, [0, 3], "vidstg")[0]
    for dataset in ("vidstg", "hc2"):
        partial_manual = _teacher_physical_dense_metric(
            teacher_partial_interval, partial_row, synthetic_teacher_truth, [0, 3], dataset)[0]
        partial_official = official(teacher_partial_interval["boxes_cxcywh_normalized"],
                                    partial_row, synthetic_teacher_truth, [0, 3], [1, 4],
                                    dataset, gt_space=False)
        _require(abs(partial_manual["m_vIoU"] - partial_official["v"]) <= 2e-9 and
                 abs(partial_manual["m_sIoU_dense_GT"] - partial_official["s"]) <= 2e-9 and
                 abs(partial_manual["m_tIoU"] - partial_official["t"]) <= 2e-9,
                 f"Teacher physical-grid metric diverges from {dataset} official interval semantics")
    fabricated_teacher = dict(synthetic_teacher_record)
    fabricated_teacher["boxes_cxcywh_normalized"] = [
        [0.5, 0.5, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.4, 0.4]
    ]
    rejected_fake_teacher_box = False
    try:
        _validate_condition_prediction("Teacher", fabricated_teacher, [0, 3],
                                       "d" * 64, "teacher_screen")
    except ContractError:
        rejected_fake_teacher_box = True
    _require(rejected_fake_teacher_box,
             "Teacher missing-frame mask accepted a synthetic zero box")
    failed_teacher = _score_prediction(
        "Teacher", {"status": "parse_failure", "reason": "synthetic parse failure"},
        {"dimensions": [100, 100]}, "vidstg",
        {"schema": LABEL_CASE_SCHEMA, "case_id": "synthetic_teacher",
         "truth_boxes_xyxy": {str(i): [30.0, 30.0, 70.0, 70.0] for i in range(3)},
         "span": [0, 3]}, "synthetic_teacher", official, dense_metric)
    _require(failed_teacher["m_vIoU"] == 0.0 and failed_teacher["failure_flag"] is True,
             "known-GT Teacher parse failure was not retained as primary zero plus failure flag")
    unknown_teacher = _score_prediction(
        "Teacher", {"status": "parse_failure", "reason": "synthetic parse failure"},
        {"dimensions": [100, 100]}, "vidstg", None,
        "synthetic_teacher", official, dense_metric)
    _require(unknown_teacher["status"] == "unknown_truth" and
             unknown_teacher["m_vIoU"] is None and unknown_teacher["failure_flag"] is True,
             "unavailable Teacher GT was incorrectly converted to a numeric zero")

    synthetic_failure = _validate_condition_prediction(
        "OPD", {"status": "parse_failure", "reason": "synthetic parser failure",
                "teacher_reference_id": "synthetic-teacher-ref",
                "source_state_sha256": "a" * 64,
                "trainable_parameter_names_sha256": "b" * 64,
                "teacher_observation_set_sha256": "c" * 64,
                "optimizer_steps": 0,
                "step_budget": 8,
                "actual_output_shift_l2": None}, [0, 4], "d" * 64, "student_matrix")
    _require(synthetic_failure["status"] == "parse_failure" and synthetic_failure["metrics"] is None,
             "prediction failure was not retained as unknown")
    result = _bootstrap_summary([0.1, None, -0.05], seed=7)
    _require(result["defined_parents"] == 2 and result["unknown_parents"] == 1,
             "paired parent bootstrap unknown accounting failed")
    _require(result["bootstrap_draws"] == BOOTSTRAP_DRAWS,
             "bootstrap draw count changed")
    synthetic_rows = []
    for i in range(4):
        conditions = {}
        opd_value = {0: 0.35, 1: 0.35, 2: 0.20, 3: 0.25}[i]
        vals = {"Source": 0.2, "Projection": 0.25, "Teacher": 0.5,
                "NoParamRefine": 0.4, "Direct": 0.3, "OPD": opd_value}
        for name, value in vals.items():
            record = {metric: value for metric in METRICS}
            record["status"] = "scored"
            if name == "OPD" and i == 1:
                record["m_vIoU"] = None
                record["status"] = "trajectory_missing"
            conditions[name] = record
        synthetic_rows.append({"dataset": "hc2" if i < 2 else "vidstg",
                               "conditions": conditions,
                               "teacher_cost": {field: None for field in TEACHER_COST_FIELDS}})
    summary = _summarize_stage(synthetic_rows, {"hc2": 2, "vidstg": 2}, "student_matrix")
    opd_projection = summary["hc2"]["paired_contrasts"]["OPD_minus_Projection"]["m_vIoU"]
    _require(opd_projection["defined_parents"] == 1 and
             opd_projection["unknown_parents"] == 1 and
             abs(opd_projection["mean"] - 0.10) <= 1e-12 and
             opd_projection["bootstrap_draws"] == BOOTSTRAP_DRAWS,
             "OPD-minus-Projection paired parent bootstrap or unknown accounting failed")
    _require(_paired_vIoU_disposition_counts(synthetic_rows, "OPD", "Projection") == {
                 "positive": 1, "negative": 1, "neutral": 1, "unknown": 1,
                 "cohort_count": 4},
             "OPD-minus-Projection paired tail accounting failed")
    _require(summary["hc2"]["paired_contrasts"]["OPD_minus_Direct"]["m_vIoU"]["unknown_parents"] == 1,
             "paired contrast did not preserve a missing trajectory as unknown")
    _require(summary["hc2"]["gap_closure"]["common_defined_parents"] == 1,
             "gap closure did not use the common parent panel")
    teacher_rows = []
    for row in synthetic_rows:
        teacher_rows.append({
            "dataset": row["dataset"],
            "conditions": {name: row["conditions"][name]
                           for name in EVALUATION_CONDITIONS["teacher_screen"]},
            "teacher_cost": row["teacher_cost"],
        })
    teacher_summary = _summarize_stage(teacher_rows, {"hc2": 2, "vidstg": 2}, "teacher_screen")
    _require(set(teacher_summary["hc2"]["condition_summary"]) ==
             set(EVALUATION_CONDITIONS["teacher_screen"]),
             "teacher screen summary incorrectly requires student arms")
    _require(teacher_summary["hc2"]["gap_closure"]["status"] == "not_applicable_stage",
             "teacher screen incorrectly reports student gap closure")
    _require(set(EVALUATION_CONTRASTS["teacher_screen"]) ==
             {"Projection_minus_Source", "Teacher_minus_Source"},
             "teacher screen includes contrasts outside its sealed arms")
    synthetic_source_sha = "d" * 64
    synthetic_manifest_sha = "e" * 64
    synthetic_roster_row = {"dataset": "hc2", "query_ordinal": 3}

    def failure_row(name: str) -> dict[str, Any]:
        row: dict[str, Any] = {"status": "input_unavailable", "reason": "synthetic missing input"}
        if name in ("Source", "Projection"):
            row["source_case_sha256"] = synthetic_source_sha
        if name in ("Direct", "OPD"):
            row.update({
                "teacher_reference_id": "synthetic-teacher",
                "source_state_sha256": "a" * 64,
                "trainable_parameter_names_sha256": "b" * 64,
                "teacher_observation_set_sha256": "c" * 64,
                "optimizer_steps": 0,
                "step_budget": 8,
                "actual_output_shift_l2": None,
            })
        return row

    for evaluation_stage, expected in EVALUATION_CONDITIONS.items():
        synthetic_prediction = {
            "schema": PREDICTION_CASE_SCHEMA,
            "case_id": "synthetic_case",
            "dataset": "hc2",
            "query_ordinal": 3,
            "evaluation_stage": evaluation_stage,
            "input_manifest_sha256": synthetic_manifest_sha,
            "source_case_sha256": synthetic_source_sha,
            "GT_read": False,
            "conditions": {name: failure_row(name) for name in expected},
            "teacher_cost": {field: None for field in TEACHER_COST_FIELDS},
        }
        feature_sha = None
        if evaluation_stage == "student_matrix":
            feature_sha = "f" * 64
            synthetic_prediction["spatial_replay_feature_sha256"] = feature_sha
            synthetic_prediction["spatial_replay_source_case_sha256"] = synthetic_source_sha
            # A no-op OPD run can stop before the shared planned budget; actual
            # step counts are per-arm audit data, while the declared budgets match.
            synthetic_prediction["conditions"]["Direct"]["optimizer_steps"] = 8
            synthetic_prediction["conditions"]["OPD"]["optimizer_steps"] = 2
        validated = _validate_prediction_file(
            synthetic_prediction, "synthetic_case", synthetic_roster_row,
            synthetic_source_sha, synthetic_manifest_sha, [0, 4], evaluation_stage,
            feature_sha)
        _require(set(validated["conditions"]) == set(expected),
                 f"prediction parser changed condition set for {evaluation_stage}")
        if evaluation_stage == "student_matrix":
            direct_record = validated["conditions"]["Direct"]["record"]
            opd_record = validated["conditions"]["OPD"]["record"]
            _require(direct_record["optimizer_steps"] == 8 and opd_record["optimizer_steps"] == 2 and
                     _declared_step_budget(direct_record, "Direct") ==
                     _declared_step_budget(opd_record, "OPD") == 8,
                     "matched-budget Direct/OPD accepted case failed to preserve distinct actual no-op counts")
            mismatched_budget_prediction = json.loads(json.dumps(synthetic_prediction))
            mismatched_budget_prediction["conditions"]["OPD"]["step_budget"] = 7
            rejected_mismatched_budget = False
            try:
                _validate_prediction_file(
                    mismatched_budget_prediction, "synthetic_case", synthetic_roster_row,
                    synthetic_source_sha, synthetic_manifest_sha, [0, 4], evaluation_stage,
                    feature_sha)
            except ContractError:
                rejected_mismatched_budget = True
            _require(rejected_mismatched_budget,
                     "Direct/OPD with different declared step budgets was accepted")
    forbidden_student_arm = dict(synthetic_prediction)
    forbidden_student_arm["evaluation_stage"] = "teacher_screen"
    forbidden_student_arm["conditions"] = {
        name: failure_row(name) for name in EVALUATION_CONDITIONS["student_matrix"]
    }
    rejected_extra_arm = False
    try:
        _validate_prediction_file(forbidden_student_arm, "synthetic_case", synthetic_roster_row,
                                  synthetic_source_sha, synthetic_manifest_sha, [0, 4],
                                  "teacher_screen")
    except ContractError:
        rejected_extra_arm = True
    _require(rejected_extra_arm, "teacher_screen accepted hidden student arms")
    try:
        import torch
        _require(not torch.cuda.is_initialized(), "synthetic contract unexpectedly initialized CUDA")
    except ImportError:
        pass
    return {
        "status": "pass",
        "synthetic_only": True,
        "strict_inference_whitelist": "pass",
        "official_metric_module_pins": "pass",
        "official_vs_independent_dense_metrics": "pass",
        "teacher_full_grid_missing_mask_zero_iou": "pass",
        "teacher_full_grid_all_present_matches_official_both_datasets": "pass",
        "teacher_partial_interval_matches_official_both_datasets": "pass",
        "teacher_parse_failure_zero_plus_flag": "pass",
        "unavailable_teacher_truth_remains_null": "pass",
        "strict_R_threshold": "pass",
        "missing_prediction_null_semantics": "pass",
        "paired_parent_bootstrap_unknown_semantics": "pass",
        "OPD_minus_Projection_bootstrap_unknown_and_tail_contract": "pass",
        "staged_teacher_screen_three_arm_seal": "pass",
        "student_matrix_six_arm_seal": "pass",
        "direct_opd_equal_budget_different_actual_noop_counts": "pass",
        "prediction_seal_stage_whitelist": "pass",
        "GT_read": False,
        "real_RGB_read": False,
        "GPU_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run CPU synthetic contracts only")
    prep = sub.add_parser("prepare-rosters", help="write proposed metadata-only rosters")
    prep.add_argument("--skip-source-byte-hashes", action="store_true",
                      help="validate seals/identities without rehashing opaque source cache bytes")
    score = sub.add_parser("score", help="post-seal CPU-only scoring; labels open after complete preflight")
    score.add_argument("--stage", choices=("development", "confirmation"), required=True)
    score.add_argument("--evaluation-stage", choices=tuple(EVALUATION_CONDITIONS), required=True,
                       help="teacher_screen seals Source/Projection/Teacher; student_matrix seals all six arms")
    score.add_argument("--roster", type=Path, required=True)
    score.add_argument("--inference-manifest", type=Path, required=True)
    score.add_argument("--source-cache-lock", type=Path, required=True)
    score.add_argument("--prediction-seal", type=Path, required=True)
    score.add_argument("--metric-contract", type=Path, default=CONTRACTS)
    score.add_argument("--labels-manifest", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--config-lock", type=Path)
    score.add_argument("--score-after-seal", action="store_true",
                       help="explicitly enter the label-opening evaluation stage")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    if args.command == "prepare-rosters":
        dev, confirm = build_proposed_rosters(verify_source_bytes=not args.skip_source_byte_hashes)
        write_proposed_roster(DEV_ROSTER, dev)
        write_proposed_roster(CONFIRM_ROSTER, confirm)
        print(json.dumps({"development": str(DEV_ROSTER.relative_to(ROOT)),
                          "confirmation": str(CONFIRM_ROSTER.relative_to(ROOT)),
                          "counts": {"development": dev["counts_by_target"],
                                     "confirmation": confirm["counts_by_target"]},
                          "GT_read": False, "GPU_used": False}, sort_keys=True))
        return 0
    if not args.score_after_seal:
        raise ContractError("refusing to open labels without --score-after-seal")
    result = score_stage(stage=args.stage, evaluation_stage=args.evaluation_stage,
                         roster_path=args.roster,
                         inference_manifest_path=args.inference_manifest,
                         source_lock_path=args.source_cache_lock,
                         prediction_seal_path=args.prediction_seal,
                         metric_contract_path=args.metric_contract,
                         labels_manifest_path=args.labels_manifest,
                         output_path=args.output, config_lock_path=args.config_lock)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "evaluation_stage": result["evaluation_stage"],
                      "cohort_count": result["cohort_count"], "GT_read": True,
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"CONTRACT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
