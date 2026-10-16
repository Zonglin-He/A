#!/usr/bin/env python3
"""Prepare selected-Temporal or Native-clock R1 inputs for STVG-Lite.

This file wraps the already-frozen ``new_crop`` candidate/Prompt renderer. It
does not change Source records, candidate generation, model weights, scoring,
or GT access. Cohort/clock-specific locks are checked independently. Real
preparation, the frozen R1 worker, readback, and reference assembly are
explicit separate actions; this implementation turn runs only synthetic
contracts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCOPE = ROOT / "artifacts/stvg_lite_event_support_v1"
BASE = SCOPE
COORD = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1"
EVIDENCE = ROOT / "artifacts/stvg_evidence_constrained_v1"
WEIGHTFREE_COORD = ROOT / "artifacts/stvg_weightfree_coordinate_v1"
WEIGHTFREE_TUNING = ROOT / "artifacts/stvg_weightfree_tuning_v1"
TABLE1 = ROOT / "artifacts/stvg_final_table1_v1"
OLD_COMPONENT = ROOT / "artifacts/stvg_component_ablation128_v1"
DESCRIPTOR_LOCK = OLD_COMPONENT / "FINAL_PARAMETER_LOCK.json"
FINAL_RECIPE_LOCK = SCOPE / "FINAL_RECIPE_LOCK.json"
DESIGN_LOCK = SCOPE / "DESIGN_LOCK.json"
DEV_TEMPORAL_LOCK = SCOPE / "SELECTED_TEMPORAL_LOCK.json"
CONFIRMATION = SCOPE / "confirmation"
OLD_NATIVE_BASE = OLD_COMPONENT / "native_clock" / "revision002"
OLD_TEACHER_RUNTIME = COORD / "confirmation/TEACHER_RUNTIME.json"
DEV_CASES = 128
CONFIRMATION_CASES = 64
SCHEMA = "stvg-lite-reclocked-inputs-v1"
CURRENT = {"cohort": "development", "clock": "temporal"}


def cohort_paths(cohort: str, clock: str) -> dict[str, Any]:
    if cohort not in {"development", "confirmation"}:
        raise ValueError("cohort must be development or confirmation")
    if clock not in {"temporal", "native"}:
        raise ValueError("clock must be temporal or native")
    if cohort == "development":
        return {
            "count": DEV_CASES,
            "roster": COORD / "ROSTER_LOCK.json",
            "source_barrier": COORD / "SOURCE_CAPTURE_BARRIER.json",
            "temporal_lock": DEV_TEMPORAL_LOCK,
            "base": SCOPE / ("reclocked_inputs" if clock == "temporal" else "native_inputs"),
            "cohort_root": COORD,
        }
    return {
        "count": CONFIRMATION_CASES,
        "roster": CONFIRMATION / "ROSTER_LOCK.json",
        "source_barrier": CONFIRMATION / "SOURCE_CAPTURE_BARRIER.json",
        "temporal_lock": CONFIRMATION / "TEMPORAL_LOCK.json",
        "base": CONFIRMATION / ("temporal_inputs" if clock == "temporal" else "native_inputs"),
        "cohort_root": CONFIRMATION,
    }


def _set_context(cohort: str, clock: str) -> dict[str, Any]:
    global BASE, ROSTER_PATH, SOURCE_BARRIER_PATH, MAX_CASES, MAX_JOBS, CURRENT
    paths = cohort_paths(cohort, clock)
    BASE = paths["base"]
    ROSTER_PATH = paths["roster"]
    SOURCE_BARRIER_PATH = paths["source_barrier"]
    MAX_CASES = int(paths["count"])
    MAX_JOBS = MAX_CASES * 4
    CURRENT = {"cohort": cohort, "clock": clock}
    return paths


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temp.open("x", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def root_rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def _sha_string(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _finite_interval(interval: Sequence[Any], name: str) -> list[int]:
    import math
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"{name} must be a two-endpoint half-open interval")
    values = []
    for value in interval:
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value):
            raise ValueError(f"{name} endpoints must be finite integral frame IDs")
        values.append(int(value))
    if values[1] <= values[0]:
        raise ValueError(f"{name} must be nonempty and half-open")
    return values


def _clock_interval(clock: str, native: Sequence[Any], selected: Sequence[Any]) -> list[Any]:
    if clock == "temporal":
        return copy.deepcopy(list(selected))
    if clock == "native":
        return copy.deepcopy(list(native))
    raise ValueError("clock must be temporal or native")


def _validate_original_descriptor(lock: Mapping[str, Any], roster_sha: str,
                                  roster: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the old Coordinate128 descriptor as provenance only.

    This lock is intentionally never called the current final recipe. It proves
    the frozen candidate/Prompt and M2 lineage only; the independently locked
    STVG-Lite recipe is checked by ``_validate_final_recipe``.
    """
    required = {
        "status", "GT_read", "search_finished", "roster_sha256", "configs",
        "response_settings", "actual_baseline_trial", "search_completion_sha256",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise ValueError(f"original FINAL_PARAMETER_LOCK missing required fields: {missing}")
    if lock["status"] != "locked" or lock["GT_read"] is not False or lock["search_finished"] is not True:
        raise ValueError("original descriptor must be locked and explicitly GT-free")
    if lock["roster_sha256"] != roster_sha:
        raise ValueError("original descriptor does not bind the Coordinate128 roster")
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise ValueError("Coordinate128 roster must remain locked and GT-free")
    rows = roster.get("records")
    if roster.get("count") != DEV_CASES or not isinstance(rows, list) or len(rows) != DEV_CASES:
        raise ValueError("original descriptor provenance must bind exactly Coordinate128")
    if len({r.get("probe_alias") for r in rows}) != DEV_CASES:
        raise ValueError("Coordinate128 aliases are not unique")
    counts = {ds: sum(r.get("dataset") == ds for r in rows) for ds in ("hc2", "vidstg")}
    if counts != {"hc2": 64, "vidstg": 64}:
        raise ValueError(f"Expected 64 fixed source cases per domain, found {counts}")
    if not _sha_string(lock["search_completion_sha256"]):
        raise ValueError("FINAL_PARAMETER_LOCK lacks a valid completion receipt hash")
    if set(lock["configs"]) != {"hc2", "vidstg"} or set(lock["response_settings"]) != {"hc2", "vidstg"}:
        raise ValueError("Final configs and response settings must cover exactly hc2 and vidstg")
    if set(lock["actual_baseline_trial"]) != {"hc2", "vidstg"}:
        raise ValueError("The frozen per-domain baseline trials are required")
    for ds in ("hc2", "vidstg"):
        config = lock["configs"][ds]
        settings = lock["response_settings"][ds]
        if not isinstance(config, dict) or not config or not isinstance(settings, dict):
            raise ValueError(f"Final M2 config/response settings are missing for {ds}")
        trial = lock["actual_baseline_trial"][ds]
        if not isinstance(trial, str) or not trial:
            raise ValueError(f"Actual baseline trial is missing for {ds}")
    return {"case_count": DEV_CASES, "domain_counts": counts}


def _verify_parameter_lock_against_trials(lock: Mapping[str, Any]) -> None:
    """Bind old M2 descriptor values to their exact completed trial designs."""
    for ds in ("hc2", "vidstg"):
        trial_id = lock["actual_baseline_trial"][ds]
        path = COORD / "trials" / trial_id / "DESIGN_LOCK.json"
        if not path.is_file():
            raise FileNotFoundError(f"Locked baseline trial design is missing: {path}")
        trial = read_json(path)
        if trial.get("status") != "locked" or trial.get("GT_read") is not False:
            raise ValueError(f"Baseline trial {trial_id} is not a locked GT-free design")
        if trial.get("configs", {}).get(ds, {}).get("M2") != lock["configs"][ds]:
            raise ValueError(f"Final M2 config differs from locked baseline trial {trial_id}/{ds}")
        if trial.get("response_settings", {}).get(ds, {}).get("M2") != lock["response_settings"][ds]:
            raise ValueError(f"Final response settings differ from locked baseline trial {trial_id}/{ds}")


def _validate_locked_roster(roster: Mapping[str, Any], cohort: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    expected = DEV_CASES if cohort == "development" else CONFIRMATION_CASES
    domain_each = expected // 2
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise ValueError(f"{cohort} roster must be locked and explicitly GT-free")
    records = roster.get("records")
    if not isinstance(records, list) or roster.get("count") != expected or len(records) != expected:
        raise ValueError(f"{cohort} roster must contain exactly {expected} fixed records")
    aliases = [str(row.get("probe_alias", "")) for row in records]
    if any(not alias for alias in aliases) or len(set(aliases)) != expected:
        raise ValueError(f"{cohort} roster aliases must be nonempty and unique")
    if any(row.get("GT_read") is not False for row in records):
        raise ValueError(f"{cohort} roster rows must each explicitly declare GT_read=false")
    counts = {ds: sum(row.get("dataset") == ds for row in records) for ds in ("hc2", "vidstg")}
    if counts != {"hc2": domain_each, "vidstg": domain_each}:
        raise ValueError(f"{cohort} roster domain counts differ from the fixed design: {counts}")
    declared = roster.get("domain_counts")
    if declared is not None and declared != counts:
        raise ValueError(f"{cohort} roster domain_counts do not match its records")
    return records, counts


def _validate_final_recipe(path: Path, cohort: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"root-locked FINAL_RECIPE_LOCK is required: {path}")
    recipe = read_json(path)
    if recipe.get("status") != "locked":
        raise ValueError("FINAL_RECIPE_LOCK must be locked")
    if recipe.get("completion_scope") != "pending_reclocked_confirmation":
        raise ValueError("FINAL_RECIPE_LOCK must remain pending the reclocked confirmation loop")
    if recipe.get("suite_final") is not False or recipe.get("CURRENT_promoted") is not False:
        raise ValueError("FINAL_RECIPE_LOCK cannot claim a final suite or production promotion")
    if recipe.get("GT_used_for_online_reference_or_step") is not False:
        raise ValueError("recipe selection must not have used GT for online references or update steps")
    # The selected development recipe was chosen from a previously scored
    # development screen, so GT_read may truthfully be true in this lock. This
    # builder reads only its frozen config/recipe fields and never opens labels.
    if recipe.get("schema") != "stvg-lite-event-final-recipe-v1":
        raise ValueError("FINAL_RECIPE_LOCK schema mismatch")
    if recipe.get("selected_spatial_recipe") not in {"full", "lite_backtrack", "lite_fullstep"}:
        raise ValueError("FINAL_RECIPE_LOCK has an unsupported selected spatial recipe")
    if recipe.get("selected_response_scope") not in {"full_grid", "event_support"}:
        raise ValueError("FINAL_RECIPE_LOCK has an unsupported response scope")
    if recipe.get("selected_temporal_lock_sha256") != sha(DEV_TEMPORAL_LOCK):
        raise ValueError("FINAL_RECIPE_LOCK does not bind the locked development Temporal selection")
    if set(recipe.get("configs", {})) != {"hc2", "vidstg"} or set(recipe.get("response_settings", {})) != {"hc2", "vidstg"}:
        raise ValueError("FINAL_RECIPE_LOCK must bind both target domains")
    if any(not isinstance(recipe["configs"][ds], dict) for ds in ("hc2", "vidstg")):
        raise ValueError("FINAL_RECIPE_LOCK has incomplete final configs")
    if recipe.get("cohort_counts") is not None:
        counts = recipe["cohort_counts"]
        expected = {"development": DEV_CASES, "confirmation": CONFIRMATION_CASES}
        if counts.get(cohort) != expected[cohort]:
            raise ValueError(f"FINAL_RECIPE_LOCK does not bind the fixed {cohort} count")
    return recipe


def _temporal_records(path: Path, roster_rows: Sequence[Mapping[str, Any]], cohort: str) -> tuple[dict[str, list[Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"locked selected Temporal clock is required: {path}")
    doc = read_json(path)
    expected = DEV_CASES if cohort == "development" else CONFIRMATION_CASES
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise ValueError("selected Temporal lock must be locked and explicitly GT-free")
    rows = doc.get("records")
    if doc.get("count") != expected or not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"selected Temporal lock must contain exactly {expected} records")
    expected_ids = {str(row["probe_alias"]) for row in roster_rows}
    by_id = {str(row.get("case_id", row.get("probe_alias", ""))): row for row in rows}
    if len(by_id) != expected or set(by_id) != expected_ids:
        raise ValueError("selected Temporal lock case set differs from the locked source roster")
    intervals: dict[str, list[Any]] = {}
    for cid in expected_ids:
        row = by_id[cid]
        interval = row.get("selected_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"selected_interval missing for {cid}")
        # Validate exact finite integral endpoints, but preserve the original
        # JSON numbers in the emitted metadata and never round/change them.
        _finite_interval(interval, f"selected interval {cid}")
        intervals[cid] = list(interval)
    return intervals, sha(path)


def _validate_source_barrier(path: Path, roster_path: Path, roster_rows: Sequence[Mapping[str, Any]],
                             cohort: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"complete Source capture barrier is required: {path}")
    barrier = read_json(path)
    expected = DEV_CASES if cohort == "development" else CONFIRMATION_CASES
    if barrier.get("status") != "complete" or barrier.get("GT_read") is not False:
        raise ValueError("Source capture barrier must be complete and explicitly GT-free")
    if barrier.get("count") != expected:
        raise ValueError(f"Source capture barrier count must be {expected}")
    if barrier.get("roster_sha256") != sha(roster_path):
        raise ValueError("Source capture barrier does not bind the selected locked roster")
    rows = barrier.get("files")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError("Source capture barrier must bind one Source file per roster case")
    by_case = {str(row.get("case_id", "")): row for row in rows}
    expected_ids = {str(row["probe_alias"]) for row in roster_rows}
    if len(by_case) != expected or set(by_case) != expected_ids:
        raise ValueError("Source capture barrier case set differs from locked roster")
    for rec in roster_rows:
        cid = str(rec["probe_alias"])
        item = by_case[cid]
        if item.get("path") != rec.get("source_case_path"):
            raise ValueError(f"Source path binding differs from roster for {cid}")
        if not _sha_string(item.get("sha256")) or not _sha_string(item.get("source_case_sha256")):
            raise ValueError(f"Source barrier SHA binding is incomplete for {cid}")
    return barrier, by_case


def _job_binding_sha(job: Mapping[str, Any]) -> str:
    # Match run_stvg_evidence_teacher_v1 exactly: json.dumps defaults to
    # ensure_ascii=True, including for Unicode prompts in the locked job.
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode("utf-8")).hexdigest()


def teacher_input_signature(job: Mapping[str, Any]) -> dict[str, Any]:
    """Only fields that reach or determine the frozen R1 request are reusable."""
    return {
        "case_id": job.get("case_id"),
        "dataset": job.get("dataset"),
        "source_case_sha256": job.get("source_case_sha256"),
        "source_request_sha256": job.get("source_request_sha256"),
        "prompt": job.get("prompt"),
        "prompt_sha256": job.get("prompt_sha256"),
        "candidate_bindings": job.get("candidate_bindings"),
        "candidate_count": job.get("candidate_count"),
        "allowed_output_ids": job.get("allowed_output_ids"),
        "visible_id_to_candidate": job.get("visible_id_to_candidate"),
        "physical_frame_ids": job.get("physical_frame_ids"),
        "local_frame_indices": job.get("local_frame_indices"),
        "timestamps_seconds": job.get("timestamps_seconds"),
        "timestamp_gaps_seconds": job.get("timestamp_gaps_seconds"),
        "context_center_physical_frame_ids": job.get("context_center_physical_frame_ids"),
        "videos": [
            {k: v.get(k) for k in (
                "sha256", "tensor_sha256", "fps", "frame_count", "role", "clip_role",
                "actual_timestamps_seconds", "actual_gap_seconds", "encoder_uniform_spacing_approximation",
                "tensor_layout", "dtype", "frame_shape_chw",
            )}
            for v in job.get("videos", [])
        ],
    }


def _npy_bytes(array: Any) -> bytes:
    import io
    import numpy as np
    f = io.BytesIO()
    np.save(f, np.ascontiguousarray(array), allow_pickle=False)
    return f.getvalue()


def _safe_output_start() -> None:
    allowed = {"CPU_CONTRACTS.json"}
    for child in BASE.iterdir():
        if child.name in allowed and child.is_file():
            continue
        raise RuntimeError(f"Native-clock preparation refuses a nonempty/partial output tree: {child}")


def _candidate_request_paths(alias: str, cohort: str | None = None) -> list[Path]:
    cohort = cohort or CURRENT["cohort"]
    legacy_paths = [
        COORD / "confirmation/requests" / f"{alias}.json",
        EVIDENCE / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_COORD / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_TUNING / "confirmation003/requests" / f"{alias}.json",
        TABLE1 / "tracking_cases" / alias / "REQUEST.json",
    ]
    if cohort == "confirmation":
        return [CONFIRMATION / "requests" / f"{alias}.json", *legacy_paths]
    if cohort != "development":
        raise ValueError("cohort must be development or confirmation")
    return legacy_paths


def _request_matches(request: Mapping[str, Any], case: Mapping[str, Any], query: str) -> bool:
    inp = case.get("input", {})
    if request.get("source_video_sha256") != inp.get("video_sha256"):
        return False
    if request.get("query_text") != query:
        return False
    if not isinstance(request.get("frame_map"), list) or not request["frame_map"]:
        return False
    video_path = request.get("video_path")
    video_sha = request.get("canonical_video_sha256")
    if not isinstance(video_path, str) or not _sha_string(video_sha):
        return False
    path = Path(video_path)
    return path.is_file() and sha(path) == video_sha


def _resolve_request(record: Mapping[str, Any], case: Mapping[str, Any], query: str) -> tuple[dict[str, Any], bytes, str, str]:
    """Reuse an exact canonical request or create one through the frozen adapter."""
    for path in _candidate_request_paths(str(record["probe_alias"]), CURRENT["cohort"]):
        if not path.is_file():
            continue
        try:
            request = read_json(path)
            if _request_matches(request, case, query):
                raw = path.read_bytes()
                return request, raw, sha_bytes(raw), root_rel(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    from scripts.run_stvg_table1_native_instance_adapter_v1 import canonicalize
    request = canonicalize(case, query, BASE / "canonical_cache")
    raw = (json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return request, raw, sha_bytes(raw), "fresh_canonicalization"


def _source_and_query(record: Mapping[str, Any]) -> tuple[dict[str, Any], Path, str]:
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    case, source_path = load_case(record)
    dataset, plan_row = source._validate_roster_record(record)
    if dataset != record["dataset"] or case.get("dataset") != dataset:
        raise ValueError("Source case / locked roster dataset mismatch")
    query = plan_row.get("input", {}).get("caption")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The pinned source plan does not provide the request caption")
    if case.get("case_id") != record.get("probe_alias"):
        raise ValueError("Source case alias differs from the original 128-row roster")
    return case, source_path, query


def _load_prior_teacher_jobs() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Index only sealed earlier R1 jobs; no outputs are trusted by filename alone."""
    roots = [
        COORD / "confirmation",
        OLD_NATIVE_BASE,
        OLD_NATIVE_BASE / "trials/N000",
        EVIDENCE / "confirmation",
        EVIDENCE / "round2",
        EVIDENCE / "round2_prompt009",
        WEIGHTFREE_COORD / "confirmation",
        WEIGHTFREE_TUNING / "confirmation003",
    ]
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for old_base in roots:
        job_path = old_base / "TEACHER_JOB_LOCK.json"
        input_path = old_base / "CANDIDATE_INPUT_BARRIER.json"
        barrier_path = old_base / "TEACHER_GLOBAL_BARRIER.json"
        if not all(p.is_file() for p in (job_path, input_path, barrier_path)):
            continue
        job_lock, input_barrier, barrier = map(read_json, (job_path, input_path, barrier_path))
        if (job_lock.get("status") != "locked" or input_barrier.get("status") != "complete"
                or barrier.get("status") != "complete" or barrier.get("GT_read") is not False
                or job_lock.get("GT_read") is not False or input_barrier.get("GT_read") is not False):
            continue
        if barrier.get("input_barrier_sha256") != sha(input_path) or barrier.get("job_lock_sha256") != sha(job_path):
            continue
        runtime_path = old_base / "TEACHER_RUNTIME.json"
        if not runtime_path.is_file() or barrier.get("runtime_sha256") != sha(runtime_path):
            continue
        runtime = read_json(runtime_path)
        if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
            continue
        good_runtime = True
        for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
            p = ROOT / rel
            if not p.is_file() or sha(p) != expected:
                good_runtime = False
                break
        if not good_runtime:
            continue
        outputs = {str(x.get("path")): x for x in barrier.get("files", [])}
        outputs_by_id = {str(x.get("job_id")): x for x in barrier.get("files", []) if x.get("job_id")}
        jobs = job_lock.get("records", [])
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job.get("variant") not in {"new_crop", "old_crop"}:
                continue
            expected_rel = (old_base / "teacher" / f"{job.get('job_id')}.json").relative_to(ROOT).as_posix()
            out = outputs_by_id.get(str(job.get("job_id"))) or outputs.get(expected_rel)
            if not out:
                continue
            response_path = ROOT / out["path"]
            if not response_path.is_file() or sha(response_path) != out.get("sha256"):
                continue
            response = read_json(response_path)
            if (response.get("GT_read") is not False
                    or response.get("job_binding_sha256") != _job_binding_sha(job)
                    or response.get("source_case_sha256") != job.get("source_case_sha256")):
                continue
            req_sha = job.get("source_request_sha256")
            if not _sha_string(req_sha):
                continue
            candidate = {
                "job": job, "response": response, "response_path": str(response_path),
                "response_sha256": sha(response_path), "origin_base": str(old_base),
            }
            by_source.setdefault((str(job.get("case_id")), str(job.get("source_case_sha256"))), []).append(candidate)
    return by_source


def _find_exact_reuse(job: Mapping[str, Any], video_bytes: Mapping[str, bytes], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    target_sig = teacher_input_signature(job)
    for candidate in candidates:
        old_job = candidate["job"]
        if teacher_input_signature(old_job) != target_sig:
            continue
        # Re-hash the prior NPYs at point of reuse. Metadata equality alone is
        # insufficient evidence that the old teacher saw the same pixels.
        old_videos = old_job.get("videos", [])
        if len(old_videos) != len(job.get("videos", [])) or len(old_videos) != len(video_bytes):
            continue
        ok = True
        for i, old_video in enumerate(old_videos):
            old_path = ROOT / old_video.get("path", "")
            if not old_path.is_file():
                ok = False
                break
            raw = old_path.read_bytes()
            if sha_bytes(raw) != old_video.get("sha256") or raw != video_bytes[i]:
                ok = False
                break
            try:
                import io
                import numpy as np
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
            except Exception:
                ok = False
                break
            if sha_bytes(np.ascontiguousarray(arr).tobytes()) != old_video.get("tensor_sha256"):
                ok = False
                break
        if ok:
            return candidate
    return None


def _copy_reused_response(job: Mapping[str, Any], candidate: Mapping[str, Any], dest: Path) -> dict[str, Any]:
    response = copy.deepcopy(dict(candidate["response"]))
    response.update({
        "job_id": job["job_id"], "case_id": job["case_id"], "dataset": job["dataset"],
        "variant": job["variant"], "center_slot": job["center_slot"],
        "source_case_sha256": job["source_case_sha256"],
        "source_request_sha256": job["source_request_sha256"],
        "prompt_sha256": job["prompt_sha256"], "input_videos": copy.deepcopy(job["videos"]),
        "candidate_bindings": copy.deepcopy(job["candidate_bindings"]),
        "candidate_count": job["candidate_count"],
        "job_binding_sha256": _job_binding_sha(job),
        "exact_teacher_input_reuse": True, "new_teacher_call": False,
        "reuse_source_response_path": candidate["response_path"],
        "reuse_source_response_sha256": candidate["response_sha256"],
        "reuse_source_job_id": candidate["job"].get("job_id"),
        "GT_read": False,
    })
    write_json(dest, response)
    return {"path": root_rel(dest), "sha256": sha(dest), "bytes": dest.stat().st_size,
            "source_response_path": candidate["response_path"],
            "source_response_sha256": candidate["response_sha256"],
            "source_job_id": candidate["job"].get("job_id"), "GT_read": False}


def _write_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return {"path": root_rel(path), "sha256": sha(path), "bytes": len(raw)}


def _make_teacher_runtime(job_count: int, input_barrier_path: Path, job_lock_path: Path,
                          *, cohort: str, clock: str, recipe_path: Path,
                          temporal_lock_path: Path, source_barrier_path: Path,
                          roster_path: Path, roster_sha: str,
                          descriptor_path: Path) -> dict[str, Any]:
    template = read_json(OLD_TEACHER_RUNTIME)
    if template.get("status") != "locked" or template.get("GT_read") is not False:
        raise ValueError("The original locked R1 teacher runtime is unavailable")
    for rel, expected in {**template.get("pins", {}), **template.get("protected", {})}.items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            raise ValueError(f"Original frozen R1 runtime pin changed: {rel}")
    runtime = copy.deepcopy(template)
    runtime.update({
        "schema": "stvg-lite-reclocked-r1-teacher-runtime-v1",
        "status": "locked",
        "GT_read": False,
        "execution_authorized": False,
        "study_id": "stvg_lite_event_support_v1",
        "cohort": cohort,
        "clock": clock,
        "bindings": {
            "final_recipe_lock_sha256": sha(recipe_path),
            "original_descriptor_sha256": sha(descriptor_path),
            "source_roster_sha256": roster_sha,
            "source_capture_barrier_sha256": sha(source_barrier_path),
            "selected_temporal_lock_sha256": sha(temporal_lock_path),
            "candidate_input_barrier_sha256": sha(input_barrier_path),
            "teacher_job_lock_sha256": sha(job_lock_path),
            "official_model_receipt_sha256": runtime["model_readback_sha256"],
            "teacher_job_count": int(job_count),
            "case_count": MAX_CASES,
            "clock_field": ("selected_temporal_lock.records[].selected_interval" if clock == "temporal"
                            else "source_case.Native_interval"),
            "temporal_selection_lock_used_for_centers": clock == "temporal",
        },
    })
    pins = dict(runtime.get("pins", {}))
    explicit = [
        Path(__file__), DESIGN_LOCK, recipe_path, descriptor_path, roster_path,
        source_barrier_path, temporal_lock_path, input_barrier_path, job_lock_path,
        ROOT / "scripts/run_stvg_evidence_teacher_v1.py",
        ROOT / "scripts/stvg_evidence_spatial_inputs005.py",
        ROOT / "scripts/stvg_evidence_spatial_inputs009.py",
        ROOT / "scripts/prepare_stvg_short_window_teacher_v1.py",
        ROOT / "scripts/stvg_weightfree_spatial_v1.py",
        ROOT / "scripts/stvg_adjacent_affine_tracklet_v1.py",
    ]
    for path in explicit:
        pins[root_rel(path)] = sha(path)
    runtime["pins"] = pins
    return runtime


def _recipe_and_lineage(cohort: str, clock: str) -> tuple[
        dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, int],
        dict[str, Any], dict[str, dict[str, Any]], dict[str, list[Any]], str]:
    if not DESIGN_LOCK.is_file():
        raise FileNotFoundError(DESIGN_LOCK)
    design = read_json(DESIGN_LOCK)
    if (design.get("status") != "locked" or design.get("study_id") != "stvg_lite_event_support_v1"
            or design.get("GT_read") is not False):
        raise ValueError("STVG-Lite DESIGN_LOCK must be locked, correctly scoped, and explicitly GT-free")
    recipe = _validate_final_recipe(FINAL_RECIPE_LOCK, cohort)
    if not ROSTER_PATH.is_file():
        raise FileNotFoundError(ROSTER_PATH)
    roster = read_json(ROSTER_PATH)
    roster_rows, domain_counts = _validate_locked_roster(roster, cohort)
    barrier, source_by_case = _validate_source_barrier(
        SOURCE_BARRIER_PATH, ROSTER_PATH, roster_rows, cohort)

    dev_roster_path = COORD / "ROSTER_LOCK.json"
    dev_roster = read_json(dev_roster_path)
    descriptor = read_json(DESCRIPTOR_LOCK)
    _validate_original_descriptor(descriptor, sha(dev_roster_path), dev_roster)
    _verify_parameter_lock_against_trials(descriptor)
    frozen_path = COORD / "FROZEN_METHOD_LOCK.json"
    frozen = read_json(frozen_path)
    expected_frozen = {**descriptor.get("pins", {}), **descriptor.get("protected", {})}.get(root_rel(frozen_path))
    if (expected_frozen != sha(frozen_path) or frozen.get("status") != "locked"
            or frozen.get("spatial_variant") != "new_crop"
            or ("GT_read" in frozen and frozen["GT_read"] is not False)):
        raise ValueError("Original descriptor does not pin the frozen new_crop candidate/Prompt lineage")

    temporal_path = cohort_paths(cohort, clock)["temporal_lock"]
    temporal_intervals, temporal_sha = _temporal_records(temporal_path, roster_rows, cohort)
    return (recipe, descriptor, roster_rows, domain_counts, barrier, source_by_case,
            temporal_intervals, temporal_sha)


def _source_and_query_for_cohort(record: Mapping[str, Any], cohort: str) -> tuple[dict[str, Any], Path, str]:
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    case, source_path = load_case(record)
    dataset, plan_row = source._validate_roster_record(record)
    if dataset != record.get("dataset") or case.get("dataset") != dataset:
        raise ValueError("Source case / locked roster dataset mismatch")
    query = plan_row.get("input", {}).get("caption")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The pinned source plan does not provide the request caption")
    if case.get("case_id") != record.get("probe_alias"):
        raise ValueError(f"Source case alias differs from the locked {cohort} roster")
    return case, source_path, query


def prepare(cohort: str = "development", clock: str = "temporal") -> dict[str, Any]:
    paths = _set_context(cohort, clock)
    import scripts.stvg_evidence_opencv004
    import numpy as np
    from PIL import Image
    from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
    from scripts import prepare_stvg_short_window_teacher_v1 as render
    from scripts import stvg_evidence_spatial_inputs009 as wrapper
    builder = wrapper._base

    (recipe, descriptor, roster_rows, domain_counts, source_barrier, source_by_case,
     temporal_intervals, temporal_sha) = _recipe_and_lineage(cohort, clock)
    roster_sha = sha(ROSTER_PATH)
    descriptor_sha = sha(DESCRIPTOR_LOCK)
    recipe_sha = sha(FINAL_RECIPE_LOCK)
    source_barrier_sha = sha(SOURCE_BARRIER_PATH)
    source_interval_key = ("selected_temporal_lock.records[].selected_interval"
                           if clock == "temporal" else "source_case.Native_interval")
    BASE.mkdir(parents=True, exist_ok=True)
    _safe_output_start()

    # Candidate windows, crop renderer, prompts, and R1 settings remain the
    # original frozen new_crop implementation. Only the clock source changes.
    old_job_index = _load_prior_teacher_jobs()
    all_jobs: list[dict[str, Any]] = []
    file_records: dict[str, dict[str, Any]] = {}
    case_records: list[dict[str, Any]] = []
    reuse_records: list[dict[str, Any]] = []
    unknown_records: list[dict[str, Any]] = []
    status_path = BASE / "CANDIDATE_STATUS.json"
    write_json(status_path, {"status": "running", "cohort": cohort, "clock": clock,
                             "done": 0, "total": MAX_CASES, "GT_read": False})

    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    for index, rec in enumerate(roster_rows, 1):
        cid = str(rec["probe_alias"])
        case, case_path = load_case(rec)
        barrier_row = source_by_case[cid]
        if (case.get("GT_read") is not False or rec.get("GT_read") is not False
                or sha(case_path) != barrier_row.get("sha256")
                or case.get("source_case_sha256") != barrier_row.get("source_case_sha256")):
            raise ValueError(f"Source payload differs from its sealed GT-free barrier: {cid}")
        dataset, plan_row = source._validate_roster_record(rec)
        if dataset != rec["dataset"] or case.get("dataset") != dataset:
            raise ValueError(f"Source dataset binding differs for {cid}")
        query = plan_row.get("input", {}).get("caption")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Pinned source query is empty for {cid}")
        if case.get("case_id") != cid:
            raise ValueError(f"Source alias differs from roster for {cid}")

        request, request_raw, request_sha, request_origin = _resolve_request(rec, case, query)
        if not _request_matches(request, case, query):
            raise ValueError(f"Request does not bind exact source video/query for {cid}")
        request_path = BASE / "requests" / f"{cid}.json"
        req_file = _write_bytes(request_path, request_raw)
        native_interval = case.get("Native_interval")
        if not isinstance(native_interval, (list, tuple)) or len(native_interval) != 2:
            raise ValueError(f"Missing exact Source Native_interval for {cid}")
        native_interval = copy.deepcopy(list(native_interval))
        selected_interval = copy.deepcopy(temporal_intervals[cid])
        input_interval = _clock_interval(clock, native_interval, selected_interval)
        _finite_interval(input_interval, f"{clock} clock interval for {cid}")
        source_grid = [int(x) for x in case["frame_ids"]]
        canonical_ids = sorted(int(x["physical_frame_id"]) for x in request["frame_map"])
        try:
            centers = builder.nearest_unique_grid_centers(input_interval, source_grid, canonical_ids)
        except ValueError as exc:
            if str(exc) != "unknown_fewer_than_four_unique_QA_grid_centers_with_two_adjacent_frames":
                raise
            row = {
                "schema": SCHEMA, "case_id": cid, "dataset": dataset, "clock": clock,
                "source_case_path": root_rel(case_path), "source_case_sha256": case["source_case_sha256"],
                "source_request_path": req_file["path"], "source_request_sha256": request_sha,
                "request_origin": request_origin, "clock_source": source_interval_key,
                "clock_interval_half_open": input_interval,
                "source_native_interval_half_open": native_interval,
                "selected_interval_half_open": selected_interval,
                "selected_temporal_lock_sha256": temporal_sha,
                "status": "unknown", "reason": str(exc), "center_count": 0,
                "teacher_job_count": 0, "GT_read": False,
            }
            path = BASE / "candidate_inputs" / f"{cid}.json"
            write_json(path, row)
            entry = {"path": root_rel(path), "sha256": sha(path), "bytes": path.stat().st_size}
            file_records[entry["path"]] = entry
            file_records[req_file["path"]] = req_file
            case_records.append({**row, "candidate_path": entry["path"], "candidate_sha256": entry["sha256"]})
            unknown_records.append({"case_id": cid, "reason": str(exc), "GT_read": False})
            write_json(status_path, {"status": "running", "cohort": cohort, "clock": clock,
                                     "done": index, "total": MAX_CASES, "GT_read": False}, replace=True)
            continue

        required = {int(center["center_physical_frame_id"]) + k for center in centers for k in range(-2, 3)}
        rgb = selected_rgb(request, required)
        pool = builder.build_candidate_pool(
            case, request, input_interval, cid, rgb,
            source_case_sha256=case["source_case_sha256"], request_sha256=request_sha,
        )
        windows = sorted(pool["windows"], key=lambda row: int(row["slot"]))
        expected_centers = [int(row["center_physical_frame_id"]) for row in centers]
        if len(windows) != 4 or [int(row["center_physical_frame_id"]) for row in windows] != expected_centers:
            raise ValueError(f"{clock} candidate pool did not preserve the exact four locked centers for {cid}")
        context_ids = expected_centers
        frame_map = {int(row["physical_frame_id"]): row for row in request["frame_map"]}
        context_times = [float(frame_map[fid]["source_timestamp_s"]) for fid in context_ids]
        context_hwc, context_meta = render.render_context_frames(
            [Image.fromarray(rgb[fid], mode="RGB") for fid in context_ids])
        context_chw = np.ascontiguousarray(context_hwc.transpose(0, 3, 1, 2))
        context_raw = builder._npy_bytes(context_chw)
        context_path = BASE / "new_crop" / "teacher_inputs" / cid / "context.npy"
        context_record = _write_bytes(context_path, context_raw)
        file_records[context_record["path"]] = context_record

        masks_by_slot: dict[int, np.ndarray] = {}
        for window in windows:
            slot = int(window["slot"])
            window["interval_half_open"] = copy.deepcopy(input_interval)
            window["source_case_sha256"] = case["source_case_sha256"]
            window["request_sha256"] = request_sha
            window["context_center_physical_frame_ids"] = copy.deepcopy(context_ids)
            window["context_timestamps_seconds"] = copy.deepcopy(context_times)
            masks = np.ascontiguousarray(window.pop("_masks"), dtype=np.uint8)
            masks_by_slot[slot] = masks
            mask_raw = builder._npz_bytes(masks=masks)
            mask_path = BASE / "new_crop" / "proposals" / f"{cid}_masks_{slot}.npz"
            window["mask_archive_path"] = root_rel(mask_path)
            window["mask_archive_sha256"] = sha_bytes(mask_raw)
            window["mask_tensor_sha256"] = sha_bytes(masks.tobytes())
            window["mask_tensor_shape"] = list(masks.shape)
            mask_record = _write_bytes(mask_path, mask_raw)
            file_records[mask_record["path"]] = mask_record

        pool["context_center_physical_frame_ids"] = copy.deepcopy(context_ids)
        pool["context_timestamps_seconds"] = copy.deepcopy(context_times)
        pool["context_renderer_metadata"] = context_meta
        pool["source_clock"] = source_interval_key
        pool["clock_interval_half_open"] = copy.deepcopy(input_interval)
        pool["source_native_interval_half_open"] = copy.deepcopy(native_interval)
        pool["selected_interval_half_open"] = copy.deepcopy(selected_interval)
        pool["selected_temporal_lock_sha256"] = temporal_sha
        pool["temporal_selection_lock_used_for_centers"] = clock == "temporal"
        proposal_rel, proposal_raw = builder._proposal_file_record("new_crop", cid, pool)
        proposal_path = BASE / proposal_rel
        proposal_record = _write_bytes(proposal_path, proposal_raw)
        file_records[proposal_record["path"]] = proposal_record

        jobs, input_bytes = builder._build_jobs_for_condition(
            condition="new_crop", condition_base=BASE, case_id=cid, dataset=dataset,
            window_rows=windows, context_chw=context_chw, context_raw=context_raw,
            rgb_by_physical_id=rgb, query_text=query, masks_by_slot=masks_by_slot,
        )
        if len(jobs) != 4:
            raise ValueError(f"Expected exactly four {clock}-clock jobs for valid case {cid}")
        case_job_rows = []
        for job in jobs:
            slot = int(job["center_slot"])
            job["job_id"] = f"{cid}_{clock}_new_crop_{slot}"
            job["clock_source"] = source_interval_key
            job["clock_interval_half_open"] = copy.deepcopy(input_interval)
            job["source_native_interval_half_open"] = copy.deepcopy(native_interval)
            job["selected_interval_half_open"] = copy.deepcopy(selected_interval)
            job["selected_temporal_lock_sha256"] = temporal_sha
            job["source_case_path"] = root_rel(case_path)
            job["source_request_path"] = req_file["path"]
            job["source_roster_sha256"] = roster_sha
            job["source_capture_barrier_sha256"] = source_barrier_sha
            job["original_descriptor_sha256"] = descriptor_sha
            job["final_recipe_lock_sha256"] = recipe_sha
            job["cohort"] = cohort
            job["clock"] = clock
            for rel, raw in input_bytes.items():
                if rel not in file_records:
                    file_records[rel] = _write_bytes(ROOT / rel, raw)
            video_bytes = []
            for video in job["videos"]:
                video_path = ROOT / video["path"]
                raw = video_path.read_bytes()
                if sha_bytes(raw) != video["sha256"]:
                    raise ValueError(f"New R1 input bytes failed their own SHA: {video['path']}")
                video_bytes.append(raw)
            prior = old_job_index.get((cid, case["source_case_sha256"]), [])
            reuse = _find_exact_reuse(job, video_bytes, prior)
            if reuse is not None:
                response_path = BASE / "teacher" / f"{job['job_id']}.json"
                reuse_receipt = _copy_reused_response(job, reuse, response_path)
                reuse_records.append({"job_id": job["job_id"], **reuse_receipt})
            case_job_rows.append({
                "job_id": job["job_id"], "center_slot": slot,
                "center_physical_frame_id": int(job["center_physical_frame_id"]),
                "candidate_count": int(job["candidate_count"]),
                "prompt_sha256": job["prompt_sha256"],
                "video_bindings": copy.deepcopy(job["videos"]),
                "exact_teacher_response_reused": reuse is not None,
                "GT_read": False,
            })
            all_jobs.append(job)

        if len(all_jobs) > MAX_JOBS:
            raise ValueError(f"R1 jobs exceed the fixed {MAX_CASES}×4 cohort bound")
        row = {
            "schema": SCHEMA, "status": "complete", "center_count": 4,
            "teacher_job_count": 4, "case_id": cid, "dataset": dataset, "clock": clock,
            "source_case_path": root_rel(case_path), "source_case_sha256": case["source_case_sha256"],
            "source_request_path": req_file["path"], "source_request_sha256": request_sha,
            "request_origin": request_origin, "clock_source": source_interval_key,
            "clock_interval_half_open": copy.deepcopy(input_interval),
            "source_native_interval_half_open": copy.deepcopy(native_interval),
            "selected_interval_half_open": copy.deepcopy(selected_interval),
            "selected_temporal_lock_sha256": temporal_sha,
            "source_grid_positions": [int(window["source_grid_position"]) for window in windows],
            "center_physical_frame_ids": context_ids,
            "center_timestamps_seconds": context_times,
            "candidate_pool_origin": pool.get("candidate_source", "frozen_selective_search_grabcut_dis"),
            "candidate_count_by_center": [len(window["tracks"]) for window in windows],
            "teacher_jobs": case_job_rows, "teacher_job_count": 4,
            "temporal_selection_lock_used_for_centers": clock == "temporal",
            "GT_read": False,
        }
        record_path = BASE / "candidate_inputs" / f"{cid}.json"
        write_json(record_path, row)
        record_file = {"path": root_rel(record_path), "sha256": sha(record_path), "bytes": record_path.stat().st_size}
        file_records[record_file["path"]] = record_file
        file_records[req_file["path"]] = req_file
        case_records.append({**row, "candidate_path": record_file["path"], "candidate_sha256": record_file["sha256"]})
        write_json(status_path, {"status": "running", "cohort": cohort, "clock": clock,
                                 "done": index, "total": MAX_CASES, "GT_read": False}, replace=True)
        print(f"reclocked-inputs {cohort}/{clock} {index}/{MAX_CASES} {cid} jobs=4", flush=True)
        del rgb, pool, windows, masks_by_slot, context_hwc, context_chw

    if len({job["job_id"] for job in all_jobs}) != len(all_jobs) or len(all_jobs) > MAX_JOBS:
        raise ValueError("R1 job lock has duplicate IDs or exceeds the fixed four-jobs-per-case bound")
    job_lock_path = BASE / "TEACHER_JOB_LOCK.json"
    input_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    write_json(job_lock_path, {
        "schema": "stvg-lite-reclocked-teacher-job-lock-v1", "status": "locked",
        "count": len(all_jobs), "case_count": MAX_CASES, "max_count": MAX_JOBS,
        "cohort": cohort, "clock": clock, "roster_sha256": roster_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
        "final_recipe_lock_sha256": recipe_sha, "original_descriptor_sha256": descriptor_sha,
        "selected_temporal_lock_sha256": temporal_sha, "clock_source": source_interval_key,
        "variant": "new_crop", "records": all_jobs, "unknown": unknown_records, "GT_read": False,
    })
    job_file = {"path": root_rel(job_lock_path), "sha256": sha(job_lock_path), "bytes": job_lock_path.stat().st_size}
    file_records[job_file["path"]] = job_file
    write_json(input_barrier_path, {
        "schema": "stvg-lite-reclocked-candidate-input-barrier-v1", "status": "complete",
        "cohort": cohort, "clock": clock, "count": MAX_CASES,
        "teacher_job_count": len(all_jobs), "max_teacher_job_count": MAX_JOBS,
        "unknown_count": len(unknown_records), "exact_teacher_response_reuse_count": len(reuse_records),
        "case_records": case_records,
        "files": sorted(file_records.values(), key=lambda row: row["path"]),
        "reused_teacher_responses": reuse_records,
        "bindings": {
            "roster_sha256": roster_sha, "source_capture_barrier_sha256": source_barrier_sha,
            "final_recipe_lock_sha256": recipe_sha, "original_descriptor_sha256": descriptor_sha,
            "selected_temporal_lock_sha256": temporal_sha,
            "teacher_job_lock_sha256": sha(job_lock_path), "builder_sha256": sha(Path(__file__)),
            "frozen_spatial_variant": "new_crop", "clock_source": source_interval_key,
            "temporal_selection_lock_used_for_centers": clock == "temporal",
        },
        "GT_read": False, "model_called": False, "GPU_used": False,
    })
    runtime = _make_teacher_runtime(
        len(all_jobs), input_barrier_path, job_lock_path, cohort=cohort, clock=clock,
        recipe_path=FINAL_RECIPE_LOCK, temporal_lock_path=paths["temporal_lock"],
        source_barrier_path=SOURCE_BARRIER_PATH, roster_path=ROSTER_PATH,
        roster_sha=roster_sha, descriptor_path=DESCRIPTOR_LOCK,
    )
    write_json(BASE / "TEACHER_RUNTIME.json", runtime)
    write_json(status_path, {
        "status": "complete", "cohort": cohort, "clock": clock,
        "done": MAX_CASES, "total": MAX_CASES, "teacher_job_count": len(all_jobs),
        "unknown_count": len(unknown_records),
        "exact_teacher_response_reuse_count": len(reuse_records), "GT_read": False,
    }, replace=True)
    return {
        "status": "complete", "cohort": cohort, "clock": clock,
        "case_count": MAX_CASES, "teacher_job_count": len(all_jobs),
        "unknown_count": len(unknown_records), "max_teacher_job_count": MAX_JOBS,
        "exact_teacher_response_reuse_count": len(reuse_records),
        "final_recipe_lock_sha256": recipe_sha, "original_descriptor_sha256": descriptor_sha,
        "source_roster_sha256": roster_sha, "source_capture_barrier_sha256": source_barrier_sha,
        "selected_temporal_lock_sha256": temporal_sha, "GT_read": False,
    }


def _safe_relative_file(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"artifact path escapes workspace root: {path_text}")
    return path


def _verify_video_binding(job: Mapping[str, Any], video: Mapping[str, Any],
                          expected_timestamps: Sequence[Any]) -> None:
    import io
    import numpy as np
    path = _safe_relative_file(str(video["path"]))
    raw = path.read_bytes()
    if sha_bytes(raw) != video.get("sha256"):
        raise ValueError(f"R1 NPY byte SHA mismatch: {video['path']}")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if array.ndim != 4 or array.dtype != np.uint8 or video.get("tensor_layout") != "TCHW":
        raise ValueError(f"R1 tensor layout/dtype mismatch: {video['path']}")
    if list(array.shape[1:]) != list(video.get("frame_shape_chw", [])):
        raise ValueError(f"R1 tensor frame shape mismatch: {video['path']}")
    if sha_bytes(np.ascontiguousarray(array).tobytes()) != video.get("tensor_sha256"):
        raise ValueError(f"R1 tensor SHA mismatch: {video['path']}")
    timestamps = video.get("actual_timestamps_seconds")
    gaps = video.get("actual_gap_seconds")
    if not isinstance(timestamps, list) or len(timestamps) != int(array.shape[0]):
        raise ValueError(f"R1 timestamp count mismatch: {video['path']}")
    if len(gaps or []) != max(0, len(timestamps) - 1):
        raise ValueError(f"R1 gap count mismatch: {video['path']}")
    if any(float(b) - float(a) != float(g) for a, b, g in zip(timestamps, timestamps[1:], gaps or [])):
        raise ValueError(f"R1 exact timestamps/gaps mismatch: {video['path']}")
    if timestamps != list(expected_timestamps):
        raise ValueError(f"R1 timestamps do not bind the job clock: {video['path']}")
    if len(timestamps) > 1:
        expected_fps = 1.0 / (sum(float(g) for g in gaps) / len(gaps))
    else:
        expected_fps = 1.0
    if float(video.get("fps")) != expected_fps:
        raise ValueError(f"R1 mean-gap encoder clock mismatch: {video['path']}")


def readback_inputs(cohort: str = "development", clock: str = "temporal",
                    *, write_receipt: bool = True) -> dict[str, Any]:
    """Full no-GT readback of Source, clock, job, Prompt and exact R1 tensors."""
    paths = _set_context(cohort, clock)
    (recipe, descriptor, roster_rows, _domain_counts, _source_barrier, source_by_case,
     temporal_intervals, temporal_sha) = _recipe_and_lineage(cohort, clock)
    roster_sha, recipe_sha = sha(ROSTER_PATH), sha(FINAL_RECIPE_LOCK)
    descriptor_sha, source_barrier_sha = sha(DESCRIPTOR_LOCK), sha(SOURCE_BARRIER_PATH)
    input_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    job_lock_path = BASE / "TEACHER_JOB_LOCK.json"
    runtime_path = BASE / "TEACHER_RUNTIME.json"
    for path in (input_barrier_path, job_lock_path, runtime_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    barrier, job_lock, runtime = map(read_json, (input_barrier_path, job_lock_path, runtime_path))
    for doc, label in ((barrier, "candidate barrier"), (job_lock, "teacher job lock"),
                       (runtime, "teacher runtime")):
        if doc.get("GT_read") is not False:
            raise ValueError(f"{label} must remain explicitly GT-free")
    if barrier.get("status") != "complete" or job_lock.get("status") != "locked" or runtime.get("status") != "locked":
        raise ValueError("input artifacts are not all complete/locked")
    if (barrier.get("cohort"), barrier.get("clock")) != (cohort, clock):
        raise ValueError("candidate barrier cohort/clock mismatch")
    expected_bindings = {
        "roster_sha256": roster_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
        "final_recipe_lock_sha256": recipe_sha,
        "original_descriptor_sha256": descriptor_sha,
        "selected_temporal_lock_sha256": temporal_sha,
        "teacher_job_lock_sha256": sha(job_lock_path),
    }
    for key, expected in expected_bindings.items():
        if barrier.get("bindings", {}).get(key) != expected:
            raise ValueError(f"candidate barrier does not bind the locked {key}")
        if key != "teacher_job_lock_sha256" and job_lock.get(key) != expected:
            raise ValueError(f"teacher job lock does not bind the locked {key}")
    runtime_bindings = {
        "final_recipe_lock_sha256": recipe_sha,
        "original_descriptor_sha256": descriptor_sha,
        "source_roster_sha256": roster_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
        "selected_temporal_lock_sha256": temporal_sha,
        "candidate_input_barrier_sha256": sha(input_barrier_path),
        "teacher_job_lock_sha256": sha(job_lock_path),
        "official_model_receipt_sha256": runtime.get("model_readback_sha256"),
        "teacher_job_count": job_lock.get("count"),
        "case_count": MAX_CASES,
        "clock_field": ("selected_temporal_lock.records[].selected_interval" if clock == "temporal"
                        else "source_case.Native_interval"),
        "temporal_selection_lock_used_for_centers": clock == "temporal",
    }
    if runtime.get("bindings", {}) != runtime_bindings:
        raise ValueError("teacher runtime bindings do not match current locked inputs")
    if not isinstance(runtime.get("execution_authorized"), bool):
        raise ValueError("teacher runtime authorization flag must be an explicit boolean")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        pinned = _safe_relative_file(rel)
        if not pinned.is_file() or sha(pinned) != expected:
            raise ValueError(f"teacher runtime pin changed: {rel}")
    if job_lock.get("count") != len(job_lock.get("records", [])) or job_lock.get("case_count") != MAX_CASES:
        raise ValueError("teacher job count metadata mismatch")
    if barrier.get("count") != MAX_CASES or barrier.get("teacher_job_count") != job_lock.get("count"):
        raise ValueError("candidate barrier counts mismatch")
    jobs = job_lock["records"]
    if len({str(job.get("job_id")) for job in jobs}) != len(jobs):
        raise ValueError("teacher job IDs are not unique")
    job_by_id = {str(job["job_id"]): job for job in jobs}
    files = barrier.get("files")
    if not isinstance(files, list):
        raise ValueError("candidate barrier file receipt list is missing")
    file_map = {str(row.get("path")): row for row in files}
    if len(file_map) != len(files):
        raise ValueError("candidate barrier has duplicate file receipts")
    for rel, row in file_map.items():
        artifact = _safe_relative_file(rel)
        if not artifact.is_file() or sha(artifact) != row.get("sha256") or artifact.stat().st_size != row.get("bytes"):
            raise ValueError(f"candidate barrier file receipt mismatch: {rel}")
    if not isinstance(barrier.get("case_records"), list) or len(barrier["case_records"]) != MAX_CASES:
        raise ValueError("candidate barrier must contain one case record per locked roster row")
    case_rows = {str(row.get("case_id")): row for row in barrier["case_records"]}
    if len(case_rows) != MAX_CASES or set(case_rows) != {str(r["probe_alias"]) for r in roster_rows}:
        raise ValueError("candidate barrier case set differs from roster")
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import stvg_evidence_spatial_inputs009 as wrapper
    builder = wrapper._base
    verified_job_count = 0
    for rec in roster_rows:
        cid = str(rec["probe_alias"])
        source_row = source_by_case[cid]
        case, source_path = load_case(rec)
        if (case.get("GT_read") is not False or rec.get("GT_read") is not False
                or root_rel(source_path) != source_row.get("path")
                or sha(source_path) != source_row.get("sha256")
                or case.get("source_case_sha256") != source_row.get("source_case_sha256")):
            raise ValueError(f"sealed Source payload does not match its exact barrier row: {cid}")
        _, plan_row = source._validate_roster_record(rec)
        query = plan_row.get("input", {}).get("caption")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Source query is empty for {cid}")
        item = case_rows[cid]
        candidate_path = _safe_relative_file(item["candidate_path"])
        expected_candidate = BASE / "candidate_inputs" / f"{cid}.json"
        if (candidate_path != expected_candidate.resolve()
                or sha(candidate_path) != item.get("candidate_sha256")):
            raise ValueError(f"candidate input receipt mismatch for {cid}")
        candidate = read_json(candidate_path)
        if (candidate.get("case_id") != cid or candidate.get("source_case_sha256") != case["source_case_sha256"]
                or candidate.get("source_case_path") != root_rel(source_path)
                or candidate.get("selected_temporal_lock_sha256") != temporal_sha
                or candidate.get("selected_interval_half_open") != temporal_intervals[cid]
                or candidate.get("source_native_interval_half_open") != case.get("Native_interval")
                or candidate.get("clock") != clock):
            raise ValueError(f"candidate input Source/clock binding mismatch for {cid}")
        expected_interval = temporal_intervals[cid] if clock == "temporal" else case["Native_interval"]
        if candidate.get("clock_interval_half_open") != expected_interval:
            raise ValueError(f"clock interval does not match its locked source for {cid}")
        request_path = _safe_relative_file(candidate["source_request_path"])
        request = read_json(request_path)
        if (candidate.get("source_request_sha256") != sha(request_path)
                or not _request_matches(request, case, query)):
            raise ValueError(f"request binding mismatch for {cid}")
        frame_map = {int(row["physical_frame_id"]): row for row in request["frame_map"]}
        if (candidate.get("source_request_path") != item.get("source_request_path")
                or candidate.get("source_request_sha256") != item.get("source_request_sha256")):
            raise ValueError(f"candidate request metadata differs from barrier row for {cid}")
        try:
            centers = builder.nearest_unique_grid_centers(expected_interval,
                                                           [int(x) for x in case["frame_ids"]],
                                                           sorted(frame_map))
        except ValueError as exc:
            if str(exc) != "unknown_fewer_than_four_unique_QA_grid_centers_with_two_adjacent_frames":
                raise
            if candidate.get("status") != "unknown" or candidate.get("center_count") != 0 or candidate.get("teacher_job_count") != 0:
                raise ValueError(f"invalid unknown/fallback record for {cid}")
            if any(job.get("case_id") == cid for job in jobs):
                raise ValueError(f"unknown candidate case unexpectedly has teacher jobs: {cid}")
            continue
        if candidate.get("status") != "complete":
            raise ValueError(f"valid four-center case is not complete: {cid}")
        centers_by_slot = sorted(centers, key=lambda row: int(row["slot"]))
        if candidate.get("center_physical_frame_ids") != [int(x["center_physical_frame_id"]) for x in centers_by_slot]:
            raise ValueError(f"candidate centers differ from exact clock mapping for {cid}")
        if candidate.get("source_grid_positions") != [
                [int(x) for x in case["frame_ids"]].index(int(row["center_physical_frame_id"]))
                for row in centers_by_slot]:
            raise ValueError(f"candidate source-grid positions differ for {cid}")
        proposal_path = BASE / "new_crop" / "proposals" / f"{cid}.json"
        proposal_rel = root_rel(proposal_path)
        if proposal_rel not in file_map:
            raise ValueError(f"candidate proposal is missing from the complete input barrier: {cid}")
        proposal = read_json(proposal_path)
        if (proposal.get("status") != "complete" or proposal.get("condition") != "new_crop"
                or proposal.get("GT_read") is not False
                or proposal.get("case_id") != cid
                or proposal.get("source_case_sha256") != case["source_case_sha256"]
                or proposal.get("request_sha256") != candidate["source_request_sha256"]
                or proposal.get("interval_half_open") != expected_interval):
            raise ValueError(f"proposal does not bind the exact Source/request/clock for {cid}")
        proposal_windows = proposal.get("windows")
        if not isinstance(proposal_windows, list) or len(proposal_windows) != 4:
            raise ValueError(f"proposal must contain exactly four locked windows for {cid}")
        proposal_by_slot = {int(row.get("slot", -1)): row for row in proposal_windows}
        if set(proposal_by_slot) != {0, 1, 2, 3}:
            raise ValueError(f"proposal window slots are not exactly 0..3 for {cid}")
        case_jobs = [job for job in jobs if job.get("case_id") == cid]
        if len(case_jobs) != 4 or sorted(int(job["center_slot"]) for job in case_jobs) != [0, 1, 2, 3]:
            raise ValueError(f"each valid case must have exactly four clock-bound R1 jobs: {cid}")
        case_jobs.sort(key=lambda job: int(job["center_slot"]))
        declared_case_jobs = candidate.get("teacher_jobs")
        if not isinstance(declared_case_jobs, list) or len(declared_case_jobs) != 4:
            raise ValueError(f"candidate row must bind exactly four teacher jobs for {cid}")
        declared_by_slot = {int(row.get("center_slot", -1)): row for row in declared_case_jobs}
        if set(declared_by_slot) != {0, 1, 2, 3}:
            raise ValueError(f"candidate teacher-job slots are not exactly 0..3 for {cid}")
        for slot, job in enumerate(case_jobs):
            expected_center = int(centers_by_slot[slot]["center_physical_frame_id"])
            window = proposal_by_slot[slot]
            declared_job = declared_by_slot[slot]
            expected_job_id = f"{cid}_{clock}_new_crop_{slot}"
            if (job.get("clock") != clock or job.get("clock_interval_half_open") != expected_interval
                    or job.get("GT_read") is not False
                    or job.get("event_interval_half_open") != expected_interval
                    or job.get("cohort") != cohort
                    or job.get("job_id") != expected_job_id
                    or job.get("clock_source") != ("selected_temporal_lock.records[].selected_interval"
                                                    if clock == "temporal" else "source_case.Native_interval")
                    or job.get("source_case_path") != root_rel(source_path)
                    or job.get("source_request_path") != candidate.get("source_request_path")
                    or job.get("source_request_sha256") != candidate.get("source_request_sha256")
                    or job.get("source_roster_sha256") != roster_sha
                    or job.get("source_capture_barrier_sha256") != source_barrier_sha
                    or job.get("original_descriptor_sha256") != descriptor_sha
                    or job.get("final_recipe_lock_sha256") != recipe_sha
                    or job.get("source_native_interval_half_open") != case.get("Native_interval")
                    or job.get("selected_interval_half_open") != temporal_intervals[cid]
                    or job.get("selected_temporal_lock_sha256") != temporal_sha
                    or job.get("source_case_sha256") != case["source_case_sha256"]
                    or int(job.get("center_physical_frame_id", -1)) != expected_center
                    or int(window.get("center_physical_frame_id", -1)) != expected_center
                    or int(window.get("source_grid_position", -1)) != candidate["source_grid_positions"][slot]
                    or job.get("candidate_bindings") != window.get("tracks")
                    or job.get("physical_frame_ids") != window.get("physical_frame_ids")
                    or job.get("local_frame_indices") != window.get("local_frame_indices")
                    or job.get("timestamps_seconds") != window.get("timestamps_seconds")
                    or job.get("timestamp_gaps_seconds") != window.get("timestamp_gaps_seconds")
                    or job.get("candidate_count") != len(window.get("tracks", []))
                    or declared_job.get("job_id") != job.get("job_id")
                    or declared_job.get("candidate_count") != job.get("candidate_count")
                    or declared_job.get("prompt_sha256") != job.get("prompt_sha256")
                    or declared_job.get("video_bindings") != job.get("videos")):
                raise ValueError(f"R1 job Source/clock/center binding mismatch for {cid}/{slot}")
            expected_local_ids = [int(fid) for fid in job.get("physical_frame_ids", [])]
            mapped_local = [int(frame_map[fid]["local_frame_index"]) for fid in expected_local_ids]
            mapped_local_times = [float(frame_map[fid]["source_timestamp_s"]) for fid in expected_local_ids]
            context_ids = [int(fid) for fid in job.get("context_center_physical_frame_ids", [])]
            context_times = [float(frame_map[fid]["source_timestamp_s"]) for fid in context_ids]
            if (job.get("local_frame_indices") != mapped_local
                    or job.get("timestamps_seconds") != mapped_local_times
                    or context_ids != [int(fid) for fid in candidate["center_physical_frame_ids"]]
                    or context_ids != [int(row["center_physical_frame_id"]) for row in centers_by_slot]
                    or context_ids != [int(fid) for fid in proposal.get("context_center_physical_frame_ids", [])]
                    or len(context_ids) != 4 or expected_center not in context_ids):
                raise ValueError(f"R1 local/context physical clock map mismatch for {cid}/{slot}")
            bindings = job.get("candidate_bindings")
            ids = [int(row.get("id", row.get("candidate_id", -1))) for row in bindings or []]
            if (job.get("candidate_count") != len(ids) or len(set(ids)) != len(ids)
                    or job.get("allowed_output_ids") != [0, *ids]
                    or job.get("visible_id_to_candidate") != {str(i): i for i in ids}):
                raise ValueError(f"candidate ID/output mapping mismatch for {cid}/{slot}")
            expected_prompt = builder._prompt(
                query, context_times, job.get("timestamps_seconds", []),
                ids, "crop")
            if (job.get("prompt") != expected_prompt
                    or hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest() != job.get("prompt_sha256")):
                raise ValueError(f"frozen Prompt binding mismatch for {cid}/{slot}")
            if len(job.get("videos", [])) != 2:
                raise ValueError(f"R1 job must bind context and local crop tensors: {cid}/{slot}")
            if (job["videos"][0].get("clip_role") != "preserved_unmarked_event_context"
                    or job["videos"][1].get("clip_role") != "candidate_crop_contact"):
                raise ValueError(f"R1 job video roles differ from the frozen context/crop protocol: {cid}/{slot}")
            context_video_times = [float(frame_map[fid]["source_timestamp_s"]) for fid in context_ids]
            _verify_video_binding(job, job["videos"][0], context_video_times)
            _verify_video_binding(job, job["videos"][1], mapped_local_times)
            verified_job_count += 1
    if verified_job_count != len(jobs):
        raise ValueError("teacher job lock contains jobs outside complete case rows")
    receipt = {
        "schema": "stvg-lite-reclocked-root-input-readback-v1",
        "status": "pass", "study_id": "stvg_lite_event_support_v1",
        "cohort": cohort, "clock": clock, "count": MAX_CASES,
        "teacher_job_count": len(jobs), "verified_job_count": verified_job_count,
        "bindings": {
            **expected_bindings,
            "candidate_input_barrier_sha256": sha(input_barrier_path),
            "teacher_job_lock_sha256": sha(job_lock_path),
            "teacher_runtime_sha256": sha(runtime_path),
        },
        "source_interval_unchanged": True, "selected_clock_exact": True,
        "all_job_prompt_tensor_bindings_verified": True,
        "GT_read": False, "GPU_used": False, "model_called": False,
    }
    if write_receipt:
        write_json(BASE / "ROOT_INPUT_READBACK.json", receipt,
                   replace=(BASE / "ROOT_INPUT_READBACK.json").exists())
    return receipt


def _anchor_from_teacher_selection(job: Mapping[str, Any], response: Mapping[str, Any],
                                   frame_ids: Sequence[int], source_grid_position: int,
                                   *, teacher_output_sha256: str,
                                   presence_threshold: float = 0.6) -> tuple[dict[str, Any] | None, str]:
    parsed = response.get("parsed", {})
    if not isinstance(parsed, Mapping) or parsed.get("status") != "valid":
        return None, "invalid_or_unparsed_teacher_response"
    visible = parsed.get("visible_id")
    if isinstance(visible, bool) or not isinstance(visible, int):
        return None, "invalid_visible_id"
    mapped = job.get("visible_id_to_candidate", {}).get(str(visible))
    selected = parsed.get("selected_candidate_id")
    if mapped is None or selected is None or int(mapped) != int(selected):
        return None, "visible_to_candidate_mapping_mismatch"
    binding = next((row for row in job.get("candidate_bindings", [])
                    if int(row.get("id", row.get("candidate_id", -1))) == int(selected)), None)
    if binding is None:
        return None, "selected_candidate_missing"
    if int(selected) == 1 or binding.get("role") == "Source":
        return None, "source_candidate_forbidden"
    try:
        presence = float(binding["presence"])
        box = [float(v) for v in binding["center_box"]]
    except (KeyError, TypeError, ValueError):
        return None, "candidate_geometry_or_presence_missing"
    import math
    if not math.isfinite(presence) or not 0.0 <= presence <= 1.0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None, "candidate_geometry_or_presence_invalid"
    if presence < presence_threshold:
        return None, "presence_below_0_6"
    if not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
        return None, "candidate_center_box_outside_normalized_bounds"
    center = int(job["center_physical_frame_id"])
    position = int(source_grid_position)
    if position < 0 or position >= len(frame_ids) or int(frame_ids[position]) != center:
        return None, "center_to_source_grid_mapping_mismatch"
    if not _sha_string(teacher_output_sha256):
        return None, "teacher_output_file_sha_invalid"
    anchor = {
        "position": position,
        "physical_frame_id": center,
        "box_cxcywh": [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0,
                       box[2] - box[0], box[3] - box[1]],
        "weight": presence,
        "candidate_id": int(selected),
        "teacher_output_sha256": teacher_output_sha256,
        "GT_read": False,
    }
    return anchor, "accepted"


def _teacher_outputs_by_job(jobs: Sequence[Mapping[str, Any]], files: Sequence[Mapping[str, Any]],
                           base: Path) -> dict[str, Mapping[str, Any]]:
    if len(files) != len(jobs):
        raise ValueError("R1 barrier must contain exactly one output per teacher job")
    outputs_by_path = {str(row.get("path")): row for row in files}
    if len(outputs_by_path) != len(files):
        raise ValueError("R1 barrier contains duplicate output paths")
    expected_paths = {
        root_rel(base / "teacher" / f"{job['job_id']}.json"): str(job["job_id"])
        for job in jobs
    }
    if set(outputs_by_path) != set(expected_paths):
        raise ValueError("R1 barrier file paths differ from the exact teacher/<job_id>.json set")
    return {job_id: outputs_by_path[path] for path, job_id in expected_paths.items()}


def _teacher_barrier(cohort: str, clock: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    barrier_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    job_lock_path = BASE / "TEACHER_JOB_LOCK.json"
    input_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    runtime_path = BASE / "TEACHER_RUNTIME.json"
    barrier = read_json(barrier_path)
    if (barrier.get("status") != "complete" or barrier.get("GT_read") is not False
            or barrier.get("count") != len(read_json(job_lock_path).get("records", []))
            or barrier.get("runtime_sha256") != sha(runtime_path)
            or barrier.get("input_barrier_sha256") != sha(input_barrier_path)
            or barrier.get("job_lock_sha256") != sha(job_lock_path)):
        raise ValueError("R1 global barrier does not bind this locked runtime and inputs")
    jobs = read_json(job_lock_path)["records"]
    files = barrier.get("files")
    if not isinstance(files, list):
        raise ValueError("R1 barrier output list is missing")
    outputs = _teacher_outputs_by_job(jobs, files, BASE)
    responses = {}
    for job in jobs:
        job_id = str(job["job_id"])
        item = outputs.get(job_id)
        if item is None:
            raise ValueError(f"R1 barrier omits the exact output path for {job_id}")
        path = _safe_relative_file(item["path"])
        if not path.is_file() or sha(path) != item.get("sha256") or path.stat().st_size != item.get("bytes"):
            raise ValueError(f"R1 response receipt mismatch: {item['path']}")
        response = read_json(path)
        if (response.get("GT_read") is not False
                or response.get("teacher_frozen") is not True
                or response.get("job_id") != job_id
                or response.get("case_id") != job.get("case_id")
                or response.get("variant") != job.get("variant")
                or response.get("job_binding_sha256") != _job_binding_sha(job)
                or response.get("source_case_sha256") != job.get("source_case_sha256")
                or response.get("prompt_sha256") != job.get("prompt_sha256")
                or response.get("candidate_bindings") != job.get("candidate_bindings")
                or response.get("candidate_count") != job.get("candidate_count")
                or response.get("input_videos") != job.get("videos")):
            raise ValueError(f"R1 response does not bind exact job input: {job['job_id']}")
        responses[str(job["job_id"])] = {
            "response": response, "path": root_rel(path), "sha256": sha(path),
        }
    return barrier, responses


def _validate_teacher_authorization(runtime: Mapping[str, Any], authorization: Mapping[str, Any],
                                    runtime_sha256: str, cohort: str, clock: str) -> None:
    if (runtime.get("execution_authorized") is not True
            or runtime.get("study_id") != "stvg_lite_event_support_v1"
            or runtime.get("cohort") != cohort or runtime.get("clock") != clock
            or runtime.get("GT_read") is not False):
        raise RuntimeError("R1 execution requires an explicitly root-authorized, cohort-bound runtime")
    if (authorization.get("status") != "pass" or authorization.get("GT_read") is not False
            or authorization.get("cohort", cohort) != cohort
            or authorization.get("clock", clock) != clock
            or not _sha_string(authorization.get("runtime_sha256_before"))
            or authorization.get("runtime_sha256_after") != runtime_sha256):
        raise RuntimeError("Root authorization receipt does not bind the current authorized runtime")


def run_teacher(cohort: str = "development", clock: str = "temporal") -> dict[str, Any]:
    """Run only the frozen R1 worker after root has explicitly authorized it."""
    _set_context(cohort, clock)
    runtime_path = BASE / "TEACHER_RUNTIME.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(runtime_path)
    runtime = read_json(runtime_path)
    authorization_path = BASE / "ROOT_TEACHER_AUTHORIZATION.json"
    if not authorization_path.is_file():
        raise RuntimeError("R1 execution requires the root authorization receipt")
    authorization = read_json(authorization_path)
    _validate_teacher_authorization(runtime, authorization, sha(runtime_path), cohort, clock)
    input_readback = readback_inputs(cohort, clock, write_receipt=True)
    if input_readback.get("status") != "pass":
        raise RuntimeError("R1 input readback did not pass")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        path = _safe_relative_file(rel)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"root-authorized R1 runtime pin mismatch: {rel}")
    import scripts.run_stvg_evidence_teacher_v1 as frozen_teacher
    frozen_teacher.BASE = BASE
    frozen_teacher.main()
    barrier, responses = _teacher_barrier(cohort, clock)
    exit_receipt = {
        "schema": "stvg-lite-reclocked-teacher-worker-exit-v1",
        "status": "complete", "cohort": cohort, "clock": clock,
        "case_count": MAX_CASES, "teacher_job_count": len(responses),
        "worker_returned_cleanly": True,
        "bindings": {
            "runtime_sha256": sha(runtime_path),
            "input_readback_sha256": sha(BASE / "ROOT_INPUT_READBACK.json"),
            "teacher_global_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
            "candidate_input_barrier_sha256": sha(BASE / "CANDIDATE_INPUT_BARRIER.json"),
            "teacher_job_lock_sha256": sha(BASE / "TEACHER_JOB_LOCK.json"),
        },
        "GT_read": False,
    }
    write_json(BASE / "TEACHER_WORKER_EXIT.json", exit_receipt)
    return exit_receipt


def _selected_temporal_document(cohort: str, clock: str, roster_rows: Sequence[Mapping[str, Any]],
                                source_by_case: Mapping[str, Mapping[str, Any]],
                                clock_intervals: Mapping[str, Sequence[Any]],
                                temporal_sha: str) -> dict[str, Any]:
    records = []
    for rec in roster_rows:
        cid = str(rec["probe_alias"])
        src = source_by_case[cid]
        records.append({
            "case_id": cid, "dataset": rec["dataset"],
            "source_case_sha256": src["source_case_sha256"],
            "selected_interval": copy.deepcopy(list(clock_intervals[cid])),
            "clock_source": ("selected_temporal_lock.records[].selected_interval"
                             if clock == "temporal" else "source_case.Native_interval"),
            "source_temporal_sha256": temporal_sha, "GT_read": False,
        })
    return {"status": "locked", "count": len(records), "records": records,
            "source_temporal_lock_sha256": temporal_sha, "cohort": cohort,
            "clock": clock, "GT_read": False}


def _reference_source_binding_row(case_id: str, source_row: Mapping[str, Any], clock: str,
                                 clock_interval: Sequence[Any],
                                 selected_temporal_interval: Sequence[Any],
                                 reference_sha256: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_file_sha256": source_row["sha256"],
        "source_case_sha256": source_row["source_case_sha256"],
        "clock_source": ("selected_temporal_lock.records[].selected_interval"
                         if clock == "temporal" else "source_case.Native_interval"),
        "selected_interval": copy.deepcopy(list(clock_interval)),
        "selected_original_interval": copy.deepcopy(list(selected_temporal_interval)),
        "reference_document_sha256": reference_sha256,
        "GT_read": False,
    }


def build_references(cohort: str = "development", clock: str = "temporal") -> dict[str, Any]:
    """Create GT-free reference/temporal locks after input and R1 output seals."""
    paths = _set_context(cohort, clock)
    receipt = readback_inputs(cohort, clock, write_receipt=True)
    if receipt.get("status") != "pass":
        raise RuntimeError("full input readback is required before reference assembly")
    (recipe, descriptor, roster_rows, _domain_counts, _source_barrier, source_by_case,
     temporal_intervals, temporal_sha) = _recipe_and_lineage(cohort, clock)
    clock_intervals: dict[str, list[Any]] = {}
    for rec in roster_rows:
        cid = str(rec["probe_alias"])
        candidate_path = BASE / "candidate_inputs" / f"{cid}.json"
        candidate = read_json(candidate_path)
        interval = candidate.get("clock_interval_half_open")
        expected_interval = temporal_intervals[cid] if clock == "temporal" else None
        if clock == "temporal" and interval != expected_interval:
            raise ValueError(f"candidate interval changed after input readback for {cid}")
        if clock == "native":
            source_case, _source_path, _query = _source_and_query_for_cohort(rec, cohort)
            expected_interval = source_case.get("Native_interval")
            if interval != expected_interval:
                raise ValueError(f"candidate Native interval changed after input readback for {cid}")
        _finite_interval(interval, f"final {clock} interval for {cid}")
        clock_intervals[cid] = copy.deepcopy(list(interval))
    if cohort == "development" and clock == "native":
        # N000 is the already locked exact Native reference document. Preserve
        # its bytes/rows; never regenerate its old refs under a new alias.
        old_ref = OLD_NATIVE_BASE / "trials" / "N000" / "REFERENCE_LOCK.json"
        if not old_ref.is_file():
            raise FileNotFoundError(old_ref)
        old_doc = read_json(old_ref)
        if (old_doc.get("status") != "locked" or old_doc.get("count") != DEV_CASES
                or old_doc.get("GT_read") is not False):
            raise ValueError("N000 Native reference lock is not the fixed GT-free 128-row artifact")
        old_rows = old_doc.get("records", [])
        if len(old_rows) != DEV_CASES:
            raise ValueError("N000 Native reference row count mismatch")
        old_by_id = {str(row.get("case_id")): row for row in old_rows}
        if len(old_by_id) != DEV_CASES or set(old_by_id) != {str(r["probe_alias"]) for r in roster_rows}:
            raise ValueError("N000 Native reference case set differs from Coordinate128")
        for rec in roster_rows:
            cid = str(rec["probe_alias"])
            old = old_by_id[cid]
            if (old.get("source_case_sha256") != source_by_case[cid]["source_case_sha256"]
                    or old.get("GT_read") is not False
                    or not isinstance(old.get("anchors"), list)):
                raise ValueError(f"N000 Native reference does not bind the exact Source row for {cid}")
            for anchor in old["anchors"]:
                if (int(anchor.get("candidate_id", 1)) == 1
                        or not 0.6 <= float(anchor.get("weight", 0.0)) <= 1.0):
                    raise ValueError(f"N000 Native reference violates current non-Source/presence threshold for {cid}")
        (BASE / "REFERENCE_LOCK.json").write_bytes(old_ref.read_bytes())
        ref_doc = old_doc
        old_ref_sha = sha(old_ref)
    else:
        exit_path = BASE / "TEACHER_WORKER_EXIT.json"
        if not exit_path.is_file():
            raise RuntimeError("reference assembly requires the post-worker-exit receipt")
        exit_doc = read_json(exit_path)
        if (exit_doc.get("status") != "complete" or exit_doc.get("worker_returned_cleanly") is not True
                or exit_doc.get("GT_read") is not False
                or exit_doc.get("bindings", {}).get("runtime_sha256") != sha(BASE / "TEACHER_RUNTIME.json")
                or exit_doc.get("bindings", {}).get("teacher_global_barrier_sha256") != sha(BASE / "TEACHER_GLOBAL_BARRIER.json")):
            raise ValueError("R1 worker-exit receipt does not bind the completed worker")
        _barrier, responses = _teacher_barrier(cohort, clock)
        jobs = read_json(BASE / "TEACHER_JOB_LOCK.json")["records"]
        by_case: dict[str, list[Mapping[str, Any]]] = {}
        for job in jobs:
            by_case.setdefault(str(job["case_id"]), []).append(job)
        records = []
        for rec in roster_rows:
            cid = str(rec["probe_alias"])
            src = source_by_case[cid]
            case, source_path = _source_and_query_for_cohort(rec, cohort)
            candidate_path = BASE / "candidate_inputs" / f"{cid}.json"
            candidate = read_json(candidate_path)
            anchors, outcomes = [], []
            for job in sorted(by_case.get(cid, []), key=lambda row: int(row["center_slot"])):
                result = responses[str(job["job_id"])]
                anchor, reason = _anchor_from_teacher_selection(
                    job, result["response"], case["frame_ids"],
                    int(candidate["source_grid_positions"][int(job["center_slot"])]),
                    teacher_output_sha256=result["sha256"],
                )
                outcomes.append({"job_id": job["job_id"], "teacher_output_path": result["path"],
                                 "teacher_output_sha256": result["sha256"], "status": reason,
                                 "GT_read": False})
                if anchor is not None:
                    anchors.append(anchor)
            records.append({
                "case_id": cid, "dataset": rec["dataset"],
                "source_case_sha256": case["source_case_sha256"],
                "anchors": anchors, "teacher_job_outcomes": outcomes, "GT_read": False,
            })
        ref_doc = {
            "status": "locked", "count": MAX_CASES,
            "anchor_count": sum(len(row["anchors"]) for row in records),
            "records": records, "cohort": cohort, "clock": clock,
            "bindings": {
                "root_input_readback_sha256": sha(BASE / "ROOT_INPUT_READBACK.json"),
                "candidate_input_barrier_sha256": sha(BASE / "CANDIDATE_INPUT_BARRIER.json"),
                "teacher_job_lock_sha256": sha(BASE / "TEACHER_JOB_LOCK.json"),
                "teacher_global_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
                "teacher_worker_exit_sha256": sha(exit_path),
                "selected_temporal_lock_sha256": temporal_sha,
                "final_recipe_lock_sha256": sha(FINAL_RECIPE_LOCK),
                "original_descriptor_sha256": sha(DESCRIPTOR_LOCK),
            },
            "no_source_candidate_anchors": True, "minimum_presence": 0.6, "GT_read": False,
        }
        write_json(BASE / "REFERENCE_LOCK.json", ref_doc)
        old_ref_sha = None
    temporal_doc = _selected_temporal_document(cohort, clock, roster_rows, source_by_case,
                                               clock_intervals, temporal_sha)
    write_json(BASE / "TEMPORAL_LOCK.json", temporal_doc)
    reference_sha = sha(BASE / "REFERENCE_LOCK.json")
    source_rows = [
        _reference_source_binding_row(
            str(rec["probe_alias"]), source_by_case[str(rec["probe_alias"])], clock,
            clock_intervals[str(rec["probe_alias"])],
            temporal_intervals[str(rec["probe_alias"])], reference_sha,
        )
        for rec in roster_rows
    ]
    root_ref = {
        "schema": "stvg-lite-reclocked-root-reference-readback-v1",
        "status": "pass", "cohort": cohort, "clock": clock, "count": MAX_CASES,
        "anchor_count": ref_doc.get("anchor_count"),
        "reference_sha256": sha(BASE / "REFERENCE_LOCK.json"),
        "temporal_sha256": sha(BASE / "TEMPORAL_LOCK.json"),
        "reused_exact_N000_reference_sha256": old_ref_sha,
        "source_bindings": source_rows,
        "GT_read": False, "GPU_used": False,
    }
    write_json(BASE / "ROOT_REFERENCE_READBACK.json", root_ref)
    return root_ref


def run_contracts() -> dict[str, Any]:
    """Small CPU-only schema/math contracts; no source, media, or GT access."""
    from scripts.stvg_evidence_spatial_inputs005 import nearest_unique_grid_centers

    checks = 0

    def must_raise(fn, message: str) -> None:
        nonlocal checks
        try:
            fn()
        except (ValueError, RuntimeError):
            checks += 1
            return
        raise AssertionError(message)

    rows = nearest_unique_grid_centers([10, 30], list(range(40)), list(range(40)))
    assert [r["center_physical_frame_id"] for r in rows] == [12, 17, 22, 27]
    assert [r["slot"] for r in rows] == [0, 1, 2, 3]
    checks += 1
    try:
        nearest_unique_grid_centers([10, 13], list(range(40)), list(range(40)))
    except ValueError as exc:
        assert str(exc) == "unknown_fewer_than_four_unique_QA_grid_centers_with_two_adjacent_frames"
        checks += 1
    else:
        raise AssertionError("Insufficient-clock centers must remain unknown")

    qa_interval, native_interval = [10, 13], [20, 25]
    assert _clock_interval("temporal", native_interval, qa_interval) == qa_interval
    assert _clock_interval("native", native_interval, qa_interval) == native_interval
    checks += 1
    must_raise(lambda: _finite_interval([1, float("nan")], "synthetic"),
               "non-finite clock endpoints must fail")
    alias = "SYNTHETIC"
    expected_development_request_paths = [
        COORD / "confirmation/requests" / f"{alias}.json",
        EVIDENCE / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_COORD / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_TUNING / "confirmation003/requests" / f"{alias}.json",
        TABLE1 / "tracking_cases" / alias / "REQUEST.json",
    ]
    development_paths = _candidate_request_paths(alias, "development")
    confirmation_paths = _candidate_request_paths(alias, "confirmation")
    assert development_paths == expected_development_request_paths
    assert confirmation_paths == [CONFIRMATION / "requests" / f"{alias}.json",
                                  *expected_development_request_paths]
    from unittest.mock import patch
    fake_request = {
        "source_video_sha256": "1" * 64,
        "query_text": "synthetic query",
        "frame_map": [{"physical_frame_id": 1, "source_timestamp_s": 0.0}],
        "video_path": "/synthetic/no-read/video.bin",
        "canonical_video_sha256": "2" * 64,
    }
    fake_case = {"input": {"video_sha256": "1" * 64}}
    with patch.object(Path, "is_file", return_value=True), \
            patch.object(sys.modules[__name__], "sha", return_value="2" * 64):
        assert _request_matches(fake_request, fake_case, "synthetic query")
        assert not _request_matches(fake_request, fake_case, "different query")
        assert not _request_matches(dict(fake_request, canonical_video_sha256="3" * 64),
                                   fake_case, "synthetic query")
        assert not _request_matches(fake_request, {"input": {"video_sha256": "4" * 64}},
                                   "synthetic query")
    checks += 1

    def job(prompt="p", frames=(1, 2, 3, 4, 5), box=(0.1, 0.2, 0.3, 0.4),
            *, presence=0.7, candidate_id=2, video_sha="d" * 64):
        ids = [candidate_id]
        return {
            "case_id": "synthetic", "dataset": "hc2", "source_case_sha256": "a" * 64,
            "source_request_sha256": "b" * 64, "prompt": prompt, "prompt_sha256": "c" * 64,
            "candidate_bindings": [{"id": candidate_id, "candidate_id": candidate_id,
                                    "role": "appearance_region", "presence": presence,
                                    "center_box": list(box)}], "candidate_count": 1,
            "allowed_output_ids": [0, *ids], "visible_id_to_candidate": {str(i): i for i in ids},
            "center_physical_frame_id": 12, "center_slot": 0, "variant": "new_crop",
            "physical_frame_ids": list(frames), "local_frame_indices": list(range(len(frames))),
            "timestamps_seconds": [float(x) for x in frames], "timestamp_gaps_seconds": [1.0] * (len(frames)-1),
            "context_center_physical_frame_ids": [12, 17, 22, 27],
            "videos": [{"sha256": video_sha, "tensor_sha256": "e" * 64,
                        "fps": 1.0, "frame_count": 5, "clip_role": "window"}],
        }

    base = job()
    assert teacher_input_signature(base) == teacher_input_signature(copy.deepcopy(base))
    assert teacher_input_signature(base) != teacher_input_signature(job(prompt="different"))
    assert teacher_input_signature(base) != teacher_input_signature(job(frames=(1, 2, 3, 4, 6)))
    assert teacher_input_signature(base) != teacher_input_signature(job(box=(0.2, 0.2, 0.3, 0.4)))
    assert teacher_input_signature(base) != teacher_input_signature(job(video_sha="f" * 64))
    checks += 5
    unicode_job = job(prompt="出租车")
    expected_unicode_sha = hashlib.sha256(json.dumps(unicode_job, sort_keys=True).encode("utf-8")).hexdigest()
    assert _job_binding_sha(unicode_job) == expected_unicode_sha
    assert _job_binding_sha(unicode_job) != hashlib.sha256(
        json.dumps(unicode_job, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    checks += 1
    synthetic_base = SCOPE / "synthetic-contract"
    synthetic_jobs = [{"job_id": "A_temporal_new_crop_0"}, {"job_id": "B_temporal_new_crop_3"}]
    synthetic_files = [{
        "path": root_rel(synthetic_base / "teacher" / f"{job_row['job_id']}.json"),
        "sha256": "a" * 64, "bytes": 1,
    } for job_row in synthetic_jobs]
    output_map = _teacher_outputs_by_job(synthetic_jobs, synthetic_files, synthetic_base)
    assert set(output_map) == {row["job_id"] for row in synthetic_jobs}
    must_raise(lambda: _teacher_outputs_by_job(
        synthetic_jobs, [dict(synthetic_files[0], path=root_rel(synthetic_base / "teacher/wrong.json")),
                         synthetic_files[1]], synthetic_base),
        "R1 output paths must map exactly to teacher/<job_id>.json")
    checks += 1

    dev_roster = {"status": "locked", "GT_read": False, "count": 128,
                  "records": [{"probe_alias": f"d{i}", "dataset": "hc2" if i < 64 else "vidstg",
                               "GT_read": False} for i in range(128)]}
    confirm_roster = {"status": "locked", "GT_read": False, "count": 64,
                      "records": [{"probe_alias": f"c{i}", "dataset": "hc2" if i < 32 else "vidstg",
                                   "GT_read": False} for i in range(64)]}
    assert len(_validate_locked_roster(dev_roster, "development")[0]) == DEV_CASES
    assert len(_validate_locked_roster(confirm_roster, "confirmation")[0]) == CONFIRMATION_CASES
    bad_roster = copy.deepcopy(confirm_roster)
    bad_roster["records"][0]["GT_read"] = True
    must_raise(lambda: _validate_locked_roster(bad_roster, "confirmation"),
               "confirmation roster rows must explicitly remain GT-free")
    checks += 2
    fake_lock = {
        "status": "locked", "GT_read": False, "search_finished": True,
        "roster_sha256": "f" * 64, "search_completion_sha256": "1" * 64,
        "configs": {"hc2": {"lr": .1}, "vidstg": {"lr": .1}},
        "response_settings": {"hc2": {"mu": .1}, "vidstg": {"mu": .1}},
        "actual_baseline_trial": {"hc2": "D001", "vidstg": "D002"},
    }
    assert _validate_original_descriptor(fake_lock, "f" * 64, dev_roster)["case_count"] == DEV_CASES
    checks += 1
    must_raise(lambda: _validate_original_descriptor(dict(fake_lock, GT_read=True), "f" * 64, dev_roster),
               "GT-marked original descriptor must fail closed")
    must_raise(lambda: _validate_original_descriptor(dict(fake_lock, roster_sha256="0" * 64),
                                                       "f" * 64, dev_roster),
               "a descriptor bound to another roster must fail closed")

    def temporal_document(clock: str, interval: Sequence[int]) -> dict[str, Any]:
        return _selected_temporal_document(
            "confirmation", clock,
            [{"probe_alias": "synthetic", "dataset": "hc2"}],
            {"synthetic": {"source_case_sha256": "a" * 64}},
            {"synthetic": list(interval)}, "b" * 64,
        )

    qa_doc = temporal_document("temporal", qa_interval)
    native_doc = temporal_document("native", native_interval)
    assert qa_doc["records"][0]["selected_interval"] == qa_interval
    assert native_doc["records"][0]["selected_interval"] == native_interval
    assert native_doc["records"][0]["selected_interval"] != qa_doc["records"][0]["selected_interval"]
    source_binding = _reference_source_binding_row(
        "synthetic", {"sha256": "c" * 64, "source_case_sha256": "a" * 64},
        "native", native_interval, qa_interval, "d" * 64,
    )
    assert source_binding["selected_interval"] == native_interval
    assert source_binding["selected_original_interval"] == qa_interval
    assert source_binding["clock_source"] == "source_case.Native_interval"
    checks += 1

    selected_job = job()
    good_response = {"parsed": {"status": "valid", "visible_id": 2, "selected_candidate_id": 2}}
    anchor, reason = _anchor_from_teacher_selection(
        selected_job, good_response, list(range(40)), 12, teacher_output_sha256="c" * 64)
    assert reason == "accepted" and anchor["physical_frame_id"] == 12
    assert anchor["teacher_output_sha256"] == "c" * 64 and anchor["candidate_id"] == 2
    below, below_reason = _anchor_from_teacher_selection(
        job(presence=0.5999), good_response, list(range(40)), 12, teacher_output_sha256="c" * 64)
    assert below is None and below_reason == "presence_below_0_6"
    source_job = job(candidate_id=1)
    source_response = {"parsed": {"status": "valid", "visible_id": 1, "selected_candidate_id": 1}}
    forbidden, source_reason = _anchor_from_teacher_selection(
        source_job, source_response, list(range(40)), 12, teacher_output_sha256="c" * 64)
    assert forbidden is None and source_reason == "source_candidate_forbidden"
    wrong_position, position_reason = _anchor_from_teacher_selection(
        selected_job, good_response, list(range(40)), 11, teacher_output_sha256="c" * 64)
    assert wrong_position is None and position_reason == "center_to_source_grid_mapping_mismatch"
    bad_hash, hash_reason = _anchor_from_teacher_selection(
        selected_job, good_response, list(range(40)), 12, teacher_output_sha256="invalid")
    assert bad_hash is None and hash_reason == "teacher_output_file_sha_invalid"
    outside, outside_reason = _anchor_from_teacher_selection(
        job(box=(-0.01, 0.2, 0.3, 0.4)), good_response, list(range(40)), 12,
        teacher_output_sha256="c" * 64)
    assert outside is None and outside_reason == "candidate_center_box_outside_normalized_bounds"
    outside, outside_reason = _anchor_from_teacher_selection(
        job(box=(0.1, 0.2, 1.01, 0.4)), good_response, list(range(40)), 12,
        teacher_output_sha256="c" * 64)
    assert outside is None and outside_reason == "candidate_center_box_outside_normalized_bounds"
    checks += 7

    authorized_runtime = {"execution_authorized": True, "study_id": "stvg_lite_event_support_v1",
                          "cohort": "development", "clock": "temporal", "GT_read": False}
    auth = {"status": "pass", "runtime_sha256_before": "d" * 64,
            "runtime_sha256_after": "e" * 64, "GT_read": False}
    _validate_teacher_authorization(authorized_runtime, auth, "e" * 64, "development", "temporal")
    must_raise(lambda: _validate_teacher_authorization(
        dict(authorized_runtime, execution_authorized=False), auth, "e" * 64, "development", "temporal"),
        "false execution authorization must stop the R1 worker")
    must_raise(lambda: _validate_teacher_authorization(
        authorized_runtime, auth, "f" * 64, "development", "temporal"),
        "stale authorization receipt must stop the R1 worker")
    checks += 1
    return {
        "schema": "stvg-lite-reclocked-inputs-cpu-contracts-v1", "status": "pass",
        "contract_count": checks, "development_roster_rows": DEV_CASES,
        "confirmation_roster_rows": CONFIRMATION_CASES,
        "per_case_job_limit": 4, "development_max_teacher_jobs": DEV_CASES * 4,
        "confirmation_max_teacher_jobs": CONFIRMATION_CASES * 4,
        "native_interval_distinct_from_selected_temporal_contract": True,
        "verified_contracts": [
            "exact four center selection", "insufficient support fallback",
            "exact Native/selected Temporal clock mapping", "finite integral clock endpoints",
            "confirmation request cache is first while development order is unchanged",
            "confirmation request reuse still validates Source video/query/frame-map/canonical SHA",
            "R1 input signature includes prompt/candidate/video bytes",
            "unicode frozen worker job hash semantics", "128 and 64 fixed roster counts",
            "per-row GT false roster requirement", "old descriptor remains provenance only",
            "selected Interval exact output mapping for Temporal and Native",
            "non-Source presence >= 0.6 and exact center mapping",
            "normalized candidate box bounds match old Native admission",
            "root authorization and latest runtime SHA guard",
        ],
        "GT_read": False, "model_called": False, "GPU_used": False, "media_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contracts", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--readback", action="store_true")
    group.add_argument("--teacher", action="store_true")
    group.add_argument("--references", action="store_true")
    parser.add_argument("--cohort", choices=("development", "confirmation"), default="development")
    parser.add_argument("--clock", choices=("temporal", "native"), default="temporal")
    args = parser.parse_args()
    if args.contracts:
        result = run_contracts()
        print(json.dumps(result, sort_keys=True))
    elif args.prepare:
        print(json.dumps(prepare(args.cohort, args.clock), sort_keys=True))
    elif args.readback:
        print(json.dumps(readback_inputs(args.cohort, args.clock), sort_keys=True))
    elif args.teacher:
        print(json.dumps(run_teacher(args.cohort, args.clock), sort_keys=True))
    elif args.references:
        print(json.dumps(build_references(args.cohort, args.clock), sort_keys=True))


if __name__ == "__main__":
    main()
