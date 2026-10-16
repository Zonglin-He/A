#!/usr/bin/env python3
"""Prepare Native-clock R1 spatial references for the bounded component ablation.

This is a CPU input builder. It deliberately does not score labels or load a
model. The only supported execution modes are synthetic contracts and
``--prepare`` (explicit media/input preparation). Teacher execution is handed
to the root-controlled launcher after this builder seals its inputs.
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

SCOPE = ROOT / "artifacts/stvg_component_ablation128_v1"
BASE = SCOPE / "native_clock"
COORD = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1"
EVIDENCE = ROOT / "artifacts/stvg_evidence_constrained_v1"
WEIGHTFREE_COORD = ROOT / "artifacts/stvg_weightfree_coordinate_v1"
WEIGHTFREE_TUNING = ROOT / "artifacts/stvg_weightfree_tuning_v1"
TABLE1 = ROOT / "artifacts/stvg_final_table1_v1"
FINAL_LOCK = SCOPE / "FINAL_PARAMETER_LOCK.json"
ROSTER_PATH = COORD / "ROSTER_LOCK.json"
OLD_TEACHER_RUNTIME = COORD / "confirmation/TEACHER_RUNTIME.json"
MAX_CASES = 128
MAX_JOBS = MAX_CASES * 4
SCHEMA = "stvg-component-native-clock-inputs-v1"


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


def _validate_final_parameter_lock(lock: Mapping[str, Any], roster_sha: str,
                                   roster: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the parameter-search completion is explicitly locked."""
    required = {
        "status", "GT_read", "search_finished", "roster_sha256", "configs",
        "response_settings", "actual_baseline_trial", "search_completion_sha256",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise ValueError(f"FINAL_PARAMETER_LOCK missing required fields: {missing}")
    if lock["status"] != "locked" or lock["GT_read"] is not False or lock["search_finished"] is not True:
        raise ValueError("Native-clock reference construction requires a completed, GT-free parameter lock")
    if lock["roster_sha256"] != roster_sha:
        raise ValueError("FINAL_PARAMETER_LOCK does not bind the original Coordinate128 roster")
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise ValueError("Coordinate128 roster must remain locked and GT-free")
    rows = roster.get("records")
    if roster.get("count") != MAX_CASES or not isinstance(rows, list) or len(rows) != MAX_CASES:
        raise ValueError("The fixed ablation cohort must be exactly the original 128 source cases")
    if len({r.get("probe_alias") for r in rows}) != MAX_CASES:
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
    return {"case_count": MAX_CASES, "domain_counts": counts}


def _verify_parameter_lock_against_trials(lock: Mapping[str, Any]) -> None:
    """Bind final M2 values to the exact completed trial designs named by the lock."""
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
            {k: v.get(k) for k in ("sha256", "tensor_sha256", "fps", "frame_count", "role")}
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


def _candidate_request_paths(alias: str) -> list[Path]:
    return [
        COORD / "confirmation/requests" / f"{alias}.json",
        EVIDENCE / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_COORD / "confirmation/requests" / f"{alias}.json",
        WEIGHTFREE_TUNING / "confirmation003/requests" / f"{alias}.json",
        TABLE1 / "tracking_cases" / alias / "REQUEST.json",
    ]


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
    for path in _candidate_request_paths(str(record["probe_alias"])):
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
        outputs = {x.get("job_id"): x for x in barrier.get("files", [])}
        jobs = job_lock.get("records", [])
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job.get("variant") not in {"new_crop", "old_crop"}:
                continue
            out = outputs.get(job.get("job_id"))
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
                          final_lock_path: Path, roster_sha: str) -> dict[str, Any]:
    template = read_json(OLD_TEACHER_RUNTIME)
    if template.get("status") != "locked" or template.get("GT_read") is not False:
        raise ValueError("The original locked R1 teacher runtime is unavailable")
    for rel, expected in {**template.get("pins", {}), **template.get("protected", {})}.items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            raise ValueError(f"Original R1 runtime pin changed: {rel}")
    runtime = copy.deepcopy(template)
    runtime["schema"] = "stvg-component-native-clock-teacher-runtime-v1"
    runtime["status"] = "locked"
    runtime["GT_read"] = False
    runtime["parent_bindings"] = {
        "final_parameter_lock_sha256": sha(final_lock_path),
        "source_roster_sha256": roster_sha,
        "candidate_input_barrier_sha256": sha(input_barrier_path),
        "teacher_job_lock_sha256": sha(job_lock_path),
        "official_model_receipt_sha256": runtime["model_readback_sha256"],
        "teacher_job_count": int(job_count), "case_count": MAX_CASES,
        "native_clock_field": "source_case.Native_interval",
        "temporal_selection_lock_used_for_centers": False,
    }
    pins = dict(runtime.get("pins", {}))
    explicit = [
        Path(__file__), ROOT / "scripts/run_stvg_evidence_teacher_v1.py",
        ROOT / "scripts/stvg_evidence_spatial_inputs005.py",
        ROOT / "scripts/stvg_evidence_spatial_inputs009.py",
        ROOT / "scripts/prepare_stvg_short_window_teacher_v1.py",
        ROOT / "scripts/stvg_weightfree_spatial_v1.py",
        ROOT / "scripts/stvg_adjacent_affine_tracklet_v1.py",
        ROSTER_PATH, final_lock_path, input_barrier_path, job_lock_path,
    ]
    for path in explicit:
        pins[root_rel(path)] = sha(path)
    runtime["pins"] = pins
    return runtime


def prepare() -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source
    from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
    from scripts import prepare_stvg_short_window_teacher_v1 as render
    from scripts import stvg_evidence_spatial_inputs009 as builder

    BASE.mkdir(parents=True, exist_ok=True)
    _safe_output_start()
    roster = read_json(ROSTER_PATH)
    roster_sha = sha(ROSTER_PATH)
    final_lock = read_json(FINAL_LOCK)
    _validate_final_parameter_lock(final_lock, roster_sha, roster)
    _verify_parameter_lock_against_trials(final_lock)
    frozen = read_json(COORD / "FROZEN_METHOD_LOCK.json")
    if frozen.get("status") != "locked" or frozen.get("GT_read") is not False or frozen.get("spatial_variant") != "new_crop":
        raise ValueError("The matched reference presentation must be the frozen new_crop R1 protocol")

    # Bind exact source request/case inputs, then generate one Native-clock pool
    # and one new_crop teacher job per valid QA center.
    old_job_index = _load_prior_teacher_jobs()
    all_jobs: list[dict[str, Any]] = []
    file_records: dict[str, dict[str, Any]] = {}
    case_records: list[dict[str, Any]] = []
    reuse_records: list[dict[str, Any]] = []
    unknown_records: list[dict[str, Any]] = []
    status_path = BASE / "CANDIDATE_STATUS.json"
    write_json(status_path, {"status": "running", "done": 0, "total": MAX_CASES, "GT_read": False})

    for index, rec in enumerate(roster["records"], 1):
        cid = str(rec["probe_alias"])
        case, case_path = load_case(rec)
        if case.get("GT_read") is not False or rec.get("GT_read") is not False:
            raise ValueError(f"GT-read source record is forbidden: {cid}")
        _, plan_row = source._validate_roster_record(rec)
        query = plan_row["input"]["caption"]
        request, request_raw, request_sha, request_origin = _resolve_request(rec, case, query)
        if not _request_matches(request, case, query):
            raise ValueError(f"Request does not bind the exact source video/query for {cid}")
        request_path = BASE / "requests" / f"{cid}.json"
        req_file = _write_bytes(request_path, request_raw)

        native_interval = case.get("Native_interval")
        if not isinstance(native_interval, (list, tuple)) or len(native_interval) != 2:
            raise ValueError(f"Missing exact Source Native_interval for {cid}")
        source_grid = [int(x) for x in case["frame_ids"]]
        canonical_ids = sorted(int(x["physical_frame_id"]) for x in request["frame_map"])
        try:
            centers = builder.nearest_unique_grid_centers(native_interval, source_grid, canonical_ids)
        except ValueError as exc:
            if str(exc) != "unknown_fewer_than_four_unique_QA_grid_centers_with_two_adjacent_frames":
                raise
            row = {
                "case_id": cid, "dataset": rec["dataset"],
                "source_case_path": root_rel(case_path), "source_case_sha256": case["source_case_sha256"],
                "source_request_path": req_file["path"], "source_request_sha256": request_sha,
                "request_origin": request_origin, "native_interval_half_open": list(native_interval),
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
            write_json(status_path, {"status": "running", "done": index, "total": MAX_CASES, "GT_read": False}, replace=True)
            continue

        needed = {int(center["center_physical_frame_id"]) + k for center in centers for k in range(-2, 3)}
        rgb = selected_rgb(request, needed)
        pool = builder.build_candidate_pool(
            case, request, native_interval, cid, rgb,
            source_case_sha256=case["source_case_sha256"], request_sha256=request_sha,
        )
        windows = sorted(pool["windows"], key=lambda x: int(x["slot"]))
        if len(windows) != 4 or [int(w["center_physical_frame_id"]) for w in windows] != [int(c["center_physical_frame_id"]) for c in centers]:
            raise ValueError(f"Native-clock candidate pool did not preserve the four locked centers for {cid}")
        context_ids = [int(w["center_physical_frame_id"]) for w in windows]
        frame_map = {int(row["physical_frame_id"]): row for row in request["frame_map"]}
        context_times = [float(frame_map[fid]["source_timestamp_s"]) for fid in context_ids]
        context_hwc, context_meta = render.render_context_frames([Image.fromarray(rgb[fid], mode="RGB") for fid in context_ids])
        context_chw = np.ascontiguousarray(context_hwc.transpose(0, 3, 1, 2))
        context_raw = builder._npy_bytes(context_chw)
        context_path = BASE / "new_crop" / "teacher_inputs" / cid / "context.npy"
        context_record = _write_bytes(context_path, context_raw)
        file_records[context_record["path"]] = context_record

        masks_by_slot: dict[int, np.ndarray] = {}
        for window in windows:
            slot = int(window["slot"])
            window["interval_half_open"] = [int(native_interval[0]), int(native_interval[1])]
            window["source_case_sha256"] = case["source_case_sha256"]
            window["request_sha256"] = request_sha
            window["context_center_physical_frame_ids"] = context_ids
            window["context_timestamps_seconds"] = context_times
            masks = np.ascontiguousarray(window.pop("_masks"), dtype=np.uint8)
            masks_by_slot[slot] = masks
            raw_mask = builder._npz_bytes(masks=masks)
            mask_path = BASE / "new_crop" / "proposals" / f"{cid}_masks_{slot}.npz"
            window["mask_archive_path"] = root_rel(mask_path)
            window["mask_archive_sha256"] = sha_bytes(raw_mask)
            window["mask_tensor_sha256"] = sha_bytes(masks.tobytes())
            window["mask_tensor_shape"] = list(masks.shape)
            mask_record = _write_bytes(mask_path, raw_mask)
            file_records[mask_record["path"]] = mask_record

        pool["context_center_physical_frame_ids"] = context_ids
        pool["context_timestamps_seconds"] = context_times
        pool["context_renderer_metadata"] = context_meta
        pool["source_clock"] = "Native_interval"
        pool["temporal_selection_lock_used_for_centers"] = False
        proposal_rel, proposal_raw = builder._proposal_file_record("new_crop", cid, pool)
        proposal_path = ROOT / proposal_rel
        proposal_record = _write_bytes(proposal_path, proposal_raw)
        file_records[proposal_record["path"]] = proposal_record

        jobs, input_bytes = builder._build_jobs_for_condition(
            condition="new_crop", condition_base=BASE, case_id=cid, dataset=rec["dataset"],
            window_rows=windows, context_chw=context_chw, context_raw=context_raw,
            rgb_by_physical_id=rgb, query_text=query, masks_by_slot=masks_by_slot,
        )
        if len(jobs) != 4:
            raise ValueError(f"Expected exactly four Native-clock jobs for valid case {cid}")
        request_line = {"path": req_file["path"], "sha256": request_sha}
        case_job_rows = []
        for job in jobs:
            job["job_id"] = f"{cid}_native_new_crop_{int(job['center_slot'])}"
            job["clock_source"] = "source_case.Native_interval"
            job["source_case_path"] = root_rel(case_path)
            job["source_request_path"] = request_line["path"]
            job["source_roster_sha256"] = roster_sha
            job["final_parameter_lock_sha256"] = sha(FINAL_LOCK)
            for rel, raw in input_bytes.items():
                path = ROOT / rel
                if rel not in file_records:
                    record = _write_bytes(path, raw)
                    file_records[rel] = record
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
                "job_id": job["job_id"], "center_slot": int(job["center_slot"]),
                "center_physical_frame_id": int(job["center_physical_frame_id"]),
                "candidate_count": int(job["candidate_count"]),
                "prompt_sha256": job["prompt_sha256"],
                "video_bindings": copy.deepcopy(job["videos"]),
                "exact_teacher_response_reused": reuse is not None,
                "GT_read": False,
            })
            all_jobs.append(job)

        if len(all_jobs) > MAX_JOBS:
            raise ValueError("Native-clock R1 jobs exceed the fixed 128×4 bound")
        record = {
            "schema": SCHEMA, "case_id": cid, "dataset": rec["dataset"],
            "source_case_path": root_rel(case_path), "source_case_sha256": case["source_case_sha256"],
            "source_request_path": req_file["path"], "source_request_sha256": request_sha,
            "request_origin": request_origin, "native_interval_half_open": list(native_interval),
            "source_grid_positions": [int(w["source_grid_position"]) for w in windows],
            "center_physical_frame_ids": context_ids, "center_timestamps_seconds": context_times,
            "candidate_pool_origin": pool.get("candidate_source", "frozen_selective_search_grabcut_dis"),
            "candidate_count_by_center": [len(w["tracks"]) for w in windows],
            "teacher_jobs": case_job_rows, "teacher_job_count": 4,
            "temporal_selection_lock_used_for_centers": False,
            "GT_read": False,
        }
        record_path = BASE / "candidate_inputs" / f"{cid}.json"
        write_json(record_path, record)
        record_file = {"path": root_rel(record_path), "sha256": sha(record_path), "bytes": record_path.stat().st_size}
        file_records[record_file["path"]] = record_file
        file_records[req_file["path"]] = req_file
        case_records.append({**record, "candidate_path": record_file["path"], "candidate_sha256": record_file["sha256"]})
        write_json(status_path, {"status": "running", "done": index, "total": MAX_CASES, "GT_read": False}, replace=True)
        print(f"native-inputs {index}/{MAX_CASES} {cid} jobs=4", flush=True)
        del rgb, pool, windows, masks_by_slot, context_hwc, context_chw

    if len({job["job_id"] for job in all_jobs}) != len(all_jobs) or len(all_jobs) > MAX_JOBS:
        raise ValueError("Native-clock job lock contains duplicate IDs or exceeds its bound")
    job_lock_path = BASE / "TEACHER_JOB_LOCK.json"
    input_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    write_json(job_lock_path, {
        "schema": "stvg-component-native-clock-teacher-job-lock-v1", "status": "locked",
        "count": len(all_jobs), "case_count": MAX_CASES, "max_count": MAX_JOBS,
        "roster_sha256": roster_sha, "final_parameter_lock_sha256": sha(FINAL_LOCK),
        "clock_source": "source_case.Native_interval", "variant": "new_crop",
        "records": all_jobs, "unknown": unknown_records, "GT_read": False,
    })
    lock_file = {"path": root_rel(job_lock_path), "sha256": sha(job_lock_path), "bytes": job_lock_path.stat().st_size}
    file_records[lock_file["path"]] = lock_file
    write_json(input_barrier_path, {
        "schema": "stvg-component-native-clock-candidate-input-barrier-v1", "status": "complete",
        "count": MAX_CASES, "teacher_job_count": len(all_jobs), "max_teacher_job_count": MAX_JOBS,
        "unknown_count": len(unknown_records), "exact_teacher_response_reuse_count": len(reuse_records),
        "case_records": case_records, "files": sorted(file_records.values(), key=lambda x: x["path"]),
        "reused_teacher_responses": reuse_records,
        "bindings": {
            "roster_sha256": roster_sha, "final_parameter_lock_sha256": sha(FINAL_LOCK),
            "teacher_job_lock_sha256": sha(job_lock_path),
            "builder_sha256": sha(Path(__file__)),
            "frozen_spatial_variant": frozen["spatial_variant"],
            "native_clock_source": "source_case.Native_interval",
        },
        "GT_read": False, "model_called": False, "GPU_used": False,
    })
    runtime = _make_teacher_runtime(len(all_jobs), input_barrier_path, job_lock_path, FINAL_LOCK, roster_sha)
    write_json(BASE / "TEACHER_RUNTIME.json", runtime)
    write_json(status_path, {
        "status": "complete", "done": MAX_CASES, "total": MAX_CASES,
        "teacher_job_count": len(all_jobs), "unknown_count": len(unknown_records),
        "exact_teacher_response_reuse_count": len(reuse_records), "GT_read": False,
    }, replace=True)
    return {
        "status": "complete", "case_count": MAX_CASES, "teacher_job_count": len(all_jobs),
        "unknown_count": len(unknown_records), "max_teacher_job_count": MAX_JOBS,
        "exact_teacher_response_reuse_count": len(reuse_records), "GT_read": False,
    }


def run_contracts() -> dict[str, Any]:
    """Small CPU-only schema/math contracts; no source, media, or GT access."""
    from scripts.stvg_evidence_spatial_inputs005 import nearest_unique_grid_centers

    rows = nearest_unique_grid_centers([10, 30], list(range(40)), list(range(40)))
    assert [r["center_physical_frame_id"] for r in rows] == [12, 17, 22, 27]
    assert [r["slot"] for r in rows] == [0, 1, 2, 3]
    try:
        nearest_unique_grid_centers([10, 13], list(range(40)), list(range(40)))
    except ValueError as exc:
        assert str(exc) == "unknown_fewer_than_four_unique_QA_grid_centers_with_two_adjacent_frames"
    else:
        raise AssertionError("Insufficient Native-clock centers must remain unknown")

    def job(prompt="p", frames=(1, 2, 3, 4, 5), box=(0.1, 0.2, 0.3, 0.4)):
        return {
            "case_id": "synthetic", "dataset": "hc2", "source_case_sha256": "a" * 64,
            "source_request_sha256": "b" * 64, "prompt": prompt, "prompt_sha256": "c" * 64,
            "candidate_bindings": [{"id": 2, "center_box": list(box)}], "candidate_count": 1,
            "allowed_output_ids": [0, 2], "visible_id_to_candidate": {"2": 2},
            "physical_frame_ids": list(frames), "local_frame_indices": list(range(len(frames))),
            "timestamps_seconds": [float(x) for x in frames], "timestamp_gaps_seconds": [1.0] * (len(frames)-1),
            "context_center_physical_frame_ids": [12, 17, 22, 27],
            "videos": [{"sha256": "d" * 64, "tensor_sha256": "e" * 64,
                        "fps": 1.0, "frame_count": 5, "role": "window"}],
        }
    base = job()
    assert teacher_input_signature(base) == teacher_input_signature(copy.deepcopy(base))
    assert teacher_input_signature(base) != teacher_input_signature(job(prompt="different"))
    assert teacher_input_signature(base) != teacher_input_signature(job(frames=(1, 2, 3, 4, 6)))
    assert teacher_input_signature(base) != teacher_input_signature(job(box=(0.2, 0.2, 0.3, 0.4)))
    unicode_job = job(prompt="出租车")
    expected_unicode_sha = hashlib.sha256(json.dumps(unicode_job, sort_keys=True).encode("utf-8")).hexdigest()
    assert _job_binding_sha(unicode_job) == expected_unicode_sha
    assert _job_binding_sha(unicode_job) != hashlib.sha256(
        json.dumps(unicode_job, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    fake_roster = {"status": "locked", "GT_read": False, "count": 128,
                   "records": [{"probe_alias": f"x{i}", "dataset": "hc2" if i < 64 else "vidstg"}
                               for i in range(128)]}
    fake_lock = {
        "status": "locked", "GT_read": False, "search_finished": True,
        "roster_sha256": "f" * 64, "search_completion_sha256": "1" * 64,
        "configs": {"hc2": {"lr": .1}, "vidstg": {"lr": .1}},
        "response_settings": {"hc2": {"mu": .1}, "vidstg": {"mu": .1}},
        "actual_baseline_trial": {"hc2": "D001", "vidstg": "D002"},
    }
    assert _validate_final_parameter_lock(fake_lock, "f" * 64, fake_roster)["case_count"] == 128
    invalid = dict(fake_lock, GT_read=True)
    try:
        _validate_final_parameter_lock(invalid, "f" * 64, fake_roster)
    except ValueError:
        pass
    else:
        raise AssertionError("GT-marked final parameter lock must fail closed")
    invalid = dict(fake_lock, roster_sha256="0" * 64)
    try:
        _validate_final_parameter_lock(invalid, "f" * 64, fake_roster)
    except ValueError:
        pass
    else:
        raise AssertionError("A different roster must fail closed")
    return {
        "schema": "stvg-component-native-clock-cpu-contracts-v1", "status": "pass",
        "contract_count": 12, "native_center_checks": 2, "teacher_input_signature_checks": 5,
        "final_lock_checks": 3, "case_limit": MAX_CASES, "per_case_job_limit": 4,
        "max_teacher_jobs": MAX_JOBS, "GT_read": False, "model_called": False,
        "GPU_used": False, "media_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contracts", action="store_true")
    group.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        result = run_contracts()
        BASE.mkdir(parents=True, exist_ok=True)
        write_json(BASE / "CPU_CONTRACTS.json", result, replace=(BASE / "CPU_CONTRACTS.json").exists())
        print(json.dumps(result, sort_keys=True))
    elif args.prepare:
        print(json.dumps(prepare(), sort_keys=True))


if __name__ == "__main__":
    main()
