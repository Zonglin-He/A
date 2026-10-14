#!/usr/bin/env python3
"""Read-only 64-query cost forecast for the locked full Table 1 run.

No labels, annotations, image/video bytes, or model weights are read. The
case payloads are opened only after the 64-query independent preflight and all
four sealed stage barriers pass. Estimates are deliberately descriptive, not
an execution guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
OUT = FULL / "PREFLIGHT_COST_FORECAST001.json"
PREFLIGHT_READBACK = BASE / "ROOT_PREFLIGHT64_READBACK.json"
PREFLIGHT_ROSTER = BASE / "PREFLIGHT_ROSTER_LOCK.json"
FULL_ROSTER = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
SOURCE_BARRIER = BASE / "SOURCE_CAPTURE_BARRIER.json"
TRACKING_BARRIER = BASE / "TRACKING_GLOBAL_BARRIER.json"
TEACHER_BARRIER = BASE / "TEACHER_GLOBAL_BARRIER.json"
STUDENT_BARRIER = BASE / "STUDENT_GLOBAL_BARRIER.json"
STUDENT_PREFLIGHT_RUNTIME = BASE / "STUDENT_RUNTIME004.json"
PREFLIGHT_EXEC = BASE / "preflight_execution005"
EXPECTED_DOMAIN_COUNTS = {"hc2": 3482, "vidstg": 10303}
EXPECTED_QUERY_COUNT = 13785
EXPECTED_VIDEO_CLOCK_GROUPS = 6009
RESERVE_BYTES = 30 * (1 << 30)


class ForecastError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ForecastError(f"required evidence is missing: {path.relative_to(ROOT)}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastError(f"invalid JSON evidence: {path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise ForecastError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(value: str | Path) -> Path:
    raw = Path(value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ForecastError("a sealed evidence path escaped the repository or is not a file")
    return path


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastError(f"missing/non-numeric measured value: {label}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ForecastError(f"invalid measured value: {label}")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ForecastError(f"missing/invalid integer work counter: {label}")
    return value


def _tracking_optional_fields_match(row: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    """Check duplicated tracking fields only when the barrier row publishes them."""
    return all(row[key] == binding.get(key) for key in (
        "tracking_cache_key", "accepted_path", "accepted_sha256",
        "inventory_path", "inventory_sha256") if key in row)


def _teacher_token_totals(generation: Mapping[str, Any],
                          responses: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    """Return actual WHO+temporal token totals, or None if any counter is absent."""
    who_in, who_out = generation.get("input_token_count"), generation.get("output_token_count")
    if (isinstance(who_in, bool) or not isinstance(who_in, int) or who_in < 0
            or isinstance(who_out, bool) or not isinstance(who_out, int) or who_out < 0):
        return None
    temporal_in = temporal_out = 0
    for response in responses:
        if not isinstance(response, Mapping):
            return None
        values = (response.get("input_tokens"), response.get("caption_input_tokens"),
                  response.get("caption_output_tokens"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in values):
            return None
        temporal_in += values[0] + values[1]
        temporal_out += values[2]
    return who_in + temporal_in, who_out + temporal_out


def _file_record_map(rows: Any, *, key: str, label: str) -> dict[str, dict[str, Any]]:
    if isinstance(rows, Mapping):
        items = []
        for alias, row in rows.items():
            if not isinstance(row, Mapping):
                raise ForecastError(f"malformed {label} file row")
            items.append(dict(row, **({key: alias} if key not in row else {})))
    elif isinstance(rows, list):
        items = rows
    else:
        raise ForecastError(f"missing {label} file rows")
    result: dict[str, dict[str, Any]] = {}
    for row in items:
        if not isinstance(row, Mapping) or not isinstance(row.get(key), str):
            raise ForecastError(f"malformed {label} file identity")
        alias = row[key]
        if alias in result:
            raise ForecastError(f"duplicate {label} file identity")
        result[alias] = dict(row)
    return result


def _require_gate(path: Path, *, statuses: set[str], count: int,
                  label: str, count_keys: Sequence[str] = ("count",)) -> tuple[dict[str, Any], str]:
    value = read_json(path)
    if value.get("status") not in statuses:
        raise ForecastError(f"{label} is not complete/sealed")
    found = next((value.get(k) for k in count_keys if value.get(k) is not None), None)
    if found != count:
        raise ForecastError(f"{label} does not contain the complete {count}-case cohort")
    if value.get("GT_read") is not False:
        raise ForecastError(f"{label} is not explicitly GT-free")
    return value, sha256_file(path)


def _preflight_gates() -> dict[str, Any]:
    """Read only gate metadata. Do not inspect case payloads before success."""
    readback, readback_sha = _require_gate(
        PREFLIGHT_READBACK, statuses={"pass"}, count=64,
        label="ROOT_PREFLIGHT64_READBACK", count_keys=("count", "cases"))
    source, source_sha = _require_gate(
        SOURCE_BARRIER, statuses={"sealed"}, count=64, label="64-query source barrier")
    tracking, tracking_sha = _require_gate(
        TRACKING_BARRIER, statuses={"complete", "sealed"}, count=64,
        label="64-query tracking barrier")
    teacher, teacher_sha = _require_gate(
        TEACHER_BARRIER, statuses={"sealed", "complete"}, count=64,
        label="64-query teacher barrier")
    student, student_sha = _require_gate(
        STUDENT_BARRIER, statuses={"sealed", "complete"}, count=64,
        label="64-query student barrier")

    runtime_paths = {
        "source": BASE / "SOURCE_RUNTIME.json",
        "tracking": BASE / "TRACKING_RUNTIME.json",
        "teacher": BASE / "TEACHER_RUNTIME.json",
        # The preflight005 launcher materializes the runtime from its reviewed
        # template as STUDENT_RUNTIME004.json and passes that exact path to
        # run_stvg_table1_student004.py.  STUDENT_RUNTIME.json belongs to an
        # earlier qualification run and is not the executed preflight runtime.
        "student": STUDENT_PREFLIGHT_RUNTIME,
    }
    runtime_sha256: dict[str, str] = {}
    for name, path in runtime_paths.items():
        runtime = read_json(path)
        if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
            raise ForecastError(f"{name} runtime is not locked and explicitly GT-free")
        runtime_sha256[name] = sha256_file(path)
    if source.get("source_runtime_sha256") != runtime_sha256["source"]:
        raise ForecastError("source barrier does not bind the locked source runtime")
    if any(tracking.get(k) != v for k, v in (
            ("source_runtime_sha256", runtime_sha256["source"]),
            ("source_capture_barrier_sha256", source_sha),
            ("tracking_runtime_sha256", runtime_sha256["tracking"]),
            )):
        raise ForecastError("tracking barrier/runtime/source lineage is incomplete")
    student_bindings = {
        "source_barrier_sha256": source_sha,
        "teacher_barrier_sha256": teacher_sha,
        "source_runtime_sha256": runtime_sha256["source"],
        "teacher_runtime_sha256": runtime_sha256["teacher"],
        "student_runtime_sha256": runtime_sha256["student"],
    }
    if any(student.get(key) != value for key, value in student_bindings.items()):
        raise ForecastError("student barrier is detached from the sealed source/teacher runtimes")

    # Respect explicit barrier bindings when the root receipt publishes them.
    for key, expected in (
        ("source_barrier_sha256", source_sha),
        ("tracking_barrier_sha256", tracking_sha),
        ("teacher_barrier_sha256", teacher_sha),
        ("student_barrier_sha256", student_sha),
    ):
        if key in readback and readback[key] != expected:
            raise ForecastError(f"independent 64-query readback is detached from {key}")
    if readback.get("barrier_sha256") not in (None, student_sha):
        raise ForecastError("independent 64-query readback binds a different student barrier")

    roster = read_json(PREFLIGHT_ROSTER)
    records = roster.get("records")
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise ForecastError("preflight roster is not locked and explicitly GT-free")
    if not isinstance(records, list) or len(records) != 64:
        raise ForecastError("preflight roster is not the fixed 64-query roster")
    counts = {domain: sum(r.get("dataset") == domain for r in records
                          if isinstance(r, Mapping)) for domain in EXPECTED_DOMAIN_COUNTS}
    if counts != {"hc2": 32, "vidstg": 32}:
        raise ForecastError("preflight roster domain split differs from the fixed 32+32 design")
    aliases = [r.get("probe_alias") for r in records if isinstance(r, Mapping)]
    if len(aliases) != 64 or len(set(aliases)) != 64 or any(not isinstance(x, str) for x in aliases):
        raise ForecastError("preflight roster aliases are malformed or duplicated")
    roster_sha = sha256_file(PREFLIGHT_ROSTER)
    if (source.get("preflight_roster_sha256") != roster_sha
            or tracking.get("preflight_roster_sha256") != roster_sha):
        raise ForecastError("source/tracking barriers do not bind the fixed preflight roster")

    files = {
        "source": _file_record_map(source.get("source_files"), key="alias", label="source"),
        "tracking": _file_record_map(tracking.get("records"), key="case_alias", label="tracking"),
        "teacher": _file_record_map(teacher.get("files"), key="alias", label="teacher"),
        "student": _file_record_map(student.get("files"), key="alias", label="student"),
    }
    expected_aliases = set(aliases)
    for label, rows in files.items():
        if set(rows) != expected_aliases or len(rows) != 64:
            raise ForecastError(f"{label} barrier does not bind each fixed preflight alias exactly once")

    return {
        "readback": readback,
        "readback_sha256": readback_sha,
        "barriers": {"source": source, "tracking": tracking,
                     "teacher": teacher, "student": student},
        "barrier_sha256": {"source": source_sha, "tracking": tracking_sha,
                            "teacher": teacher_sha, "student": student_sha},
        "runtime_sha256": runtime_sha256,
        "roster": roster,
        "roster_sha256": roster_sha,
        "records": records,
        "aliases": aliases,
        "files": files,
    }


def _load_pt_cpu(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ForecastError("CPU torch is required to read sealed cost payload metadata") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def _verify_file_row(row: Mapping[str, Any], label: str) -> tuple[Path, int, str]:
    path = safe_path(row.get("path", ""))
    digest = row.get("sha256")
    expected_size = row.get("bytes")
    size = path.stat().st_size
    if (not isinstance(digest, str) or len(digest) != 64
            or (expected_size is not None and size != expected_size)):
        raise ForecastError(f"{label} path size/hash metadata is malformed")
    actual_sha = sha256_file(path)
    if actual_sha != digest:
        raise ForecastError(f"{label} bytes differ from the sealed barrier")
    return path, int(size), actual_sha


def _verify_receipt(path: Path, expected_sha: Any, *, expected_size: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ForecastError(f"sealed case receipt missing: {path.relative_to(ROOT)}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ForecastError("case receipt bytes differ from their sealed barrier reference")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ForecastError("case receipt byte count differs from its sealed reference")
    return read_json(path)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos)); high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0]
    if not finite:
        return {"count": 0, "p10": None, "median": None, "p90": None,
                "mean": None, "sum": 0.0}
    return {
        "count": len(finite), "p10": _quantile(finite, 0.10),
        "median": _quantile(finite, 0.50), "p90": _quantile(finite, 0.90),
        "mean": statistics.fmean(finite), "sum": math.fsum(finite),
    }


def _stage_elapsed(stage: str) -> dict[str, Any]:
    launch_path = PREFLIGHT_EXEC / f"{stage}_LAUNCH.json"
    exit_path = PREFLIGHT_EXEC / f"{stage}_EXIT.json"
    if not launch_path.is_file() or not exit_path.is_file():
        return {"available": False, "seconds": None, "launch_sha256": None,
                "exit_sha256": None, "reason": "actual launch/exit receipt unavailable"}
    launch = read_json(launch_path); ending = read_json(exit_path)
    if launch.get("GT_read") is not False or ending.get("GT_read") is not False:
        raise ForecastError(f"{stage} stage launch/exit receipts are not GT-free")
    if ending.get("status") != "actual_worker_exited" or ending.get("exit_code") != 0:
        raise ForecastError(f"{stage} stage does not have a successful actual worker exit")
    start = _finite_nonnegative(launch.get("time"), f"{stage} launch time")
    finish = _finite_nonnegative(ending.get("time"), f"{stage} exit time")
    if finish < start:
        raise ForecastError(f"{stage} launch/exit timestamps are reversed")
    return {"available": True, "seconds": finish - start,
            "launch_sha256": sha256_file(launch_path), "exit_sha256": sha256_file(exit_path)}


def _video_clock_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    domain = record.get("dataset")
    if domain not in EXPECTED_DOMAIN_COUNTS:
        raise ForecastError("full source roster has an unsupported domain")
    bounds = record.get("video_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ForecastError("full source roster has malformed half-open video bounds")
    start, end = bounds
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start):
        raise ForecastError("full source roster has invalid half-open physical bounds")
    fps = _finite_nonnegative(record.get("fps"), "full roster fps")
    if fps <= 0:
        raise ForecastError("full roster fps must be positive")
    video_sha = record.get("video_sha256")
    if not isinstance(video_sha, str) or len(video_sha) != 64:
        raise ForecastError("full roster source video digest is malformed")
    width, height = record.get("width"), record.get("height")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ForecastError("full roster dimensions are malformed")
    return {
        "dataset_decode_policy": "ffmpeg_default_timing" if domain == "hc2" else "ffmpeg_vsync0_original_grid",
        "source_video_sha256": video_sha,
        "physical_bounds_half_open": [start, end],
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_clock_policy": "contiguous_physical_frame_ids_v1",
    }


def _full_roster_counts() -> tuple[dict[str, int], dict[str, int], str]:
    full = read_json(FULL_ROSTER)
    rows = full.get("records")
    if (full.get("status") not in {"prepared_for_root_runtime_lock", "locked"}
            or full.get("GT_read") is not False):
        raise ForecastError("full roster metadata is not locked/prepared and explicitly GT-free")
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUERY_COUNT:
        raise ForecastError("locked full roster is not the expected 13,785-query metadata roster")
    domains = {d: sum(r.get("dataset") == d for r in rows if isinstance(r, Mapping))
               for d in EXPECTED_DOMAIN_COUNTS}
    if domains != EXPECTED_DOMAIN_COUNTS:
        raise ForecastError("locked full roster domain counts differ from the authorized cohort")
    groups: dict[str, set[str]] = {d: set() for d in EXPECTED_DOMAIN_COUNTS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ForecastError("malformed full source roster record")
        identity = _video_clock_identity(row)
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        groups[row["dataset"]].add(hashlib.sha256(encoded).hexdigest())
    group_counts = {d: len(groups[d]) for d in groups}
    if sum(group_counts.values()) != EXPECTED_VIDEO_CLOCK_GROUPS:
        raise ForecastError("full roster video-clock groups differ from the locked 6,009 denominator")
    return domains, group_counts, sha256_file(FULL_ROSTER)


def _case_costs(gates: Mapping[str, Any]) -> dict[str, Any]:
    """Verify and read actual case cost metadata, called only after all gates."""
    roster_by_alias = {r["probe_alias"]: r for r in gates["records"]}
    source_rows = gates["files"]["source"]
    track_rows = gates["files"]["tracking"]
    teacher_rows = gates["files"]["teacher"]
    student_rows = gates["files"]["student"]

    costs: dict[str, Any] = {
        d: {"source_seconds": [], "source_bytes": [], "teacher_case_wall_seconds": [],
            "teacher_calls": [], "teacher_bytes": [], "teacher_input_tokens": [],
            "teacher_output_tokens": [], "student_source_capture_seconds": [],
            "student_fit_seconds": [], "student_readout_seconds": [],
            "student_bytes": [], "student_fit_count": [], "student_optimizer_steps": [],
            "student_redecode_calls": [], "tracking_request_binding_bytes": []}
        for d in EXPECTED_DOMAIN_COUNTS
    }
    unique_tracking: dict[str, dict[str, Any]] = {}
    actual_totals: dict[str, Any] = {
        "source_case_bytes": 0, "teacher_case_bytes": 0, "student_case_bytes": 0,
        "tracking_query_metadata_bytes": 0,
        "source_case_seconds": 0.0, "student_case_component_seconds": 0.0,
        "teacher_calls": 0, "teacher_input_tokens": 0, "teacher_output_tokens": 0,
        "teacher_calls_in_fresh_outputs": 0,
        "student_source_capture_calls": 0, "student_fit_count": 0, "student_optimizer_steps": 0,
        "student_final_full_model_redecode_calls": 0,
    }

    # Source: validate sidecar receipt and read only compute/provenance metadata.
    for alias, row in source_rows.items():
        domain = roster_by_alias[alias]["dataset"]
        path, size, digest = _verify_file_row(row, "source case")
        receipt_path = safe_path(row.get("receipt_path", ""))
        receipt = _verify_receipt(receipt_path, row.get("receipt_sha256"), expected_size=None)
        if (receipt.get("sha256") != digest or receipt.get("bytes") != size
                or receipt.get("case_id") != alias or receipt.get("GT_read") is not False):
            raise ForecastError("source payload and sidecar receipt disagree")
        if receipt.get("source_runtime_sha256") != gates["runtime_sha256"]["source"]:
            raise ForecastError("source receipt is detached from the locked source runtime")
        case = _load_pt_cpu(path)
        if not isinstance(case, Mapping) or case.get("case_id") != alias or case.get("GT_read") is not False:
            raise ForecastError("source payload is not the sealed GT-free preflight case")
        compute = case.get("compute")
        if not isinstance(compute, Mapping):
            raise ForecastError("source case has no recorded compute metadata")
        seconds = _finite_nonnegative(compute.get("source_capture_seconds"), "source_capture_seconds")
        costs[domain]["source_seconds"].append(seconds)
        receipt_size = receipt_path.stat().st_size
        costs[domain]["source_bytes"].append(float(size + receipt_size))
        actual_totals["source_case_seconds"] += seconds
        actual_totals["source_case_bytes"] += size + receipt_size
        del case

    # Tracking is per unique content-addressed cache key, not per query alias.
    for alias, row in track_rows.items():
        domain = roster_by_alias[alias]["dataset"]
        binding_path = safe_path(row.get("tracking_binding_path", ""))
        binding_sha = row.get("tracking_binding_sha256")
        if not isinstance(binding_sha, str) or sha256_file(binding_path) != binding_sha:
            raise ForecastError("tracking binding bytes differ from the sealed global barrier")
        binding = read_json(binding_path)
        request_path = safe_path(row.get("request_path", ""))
        request_sha = row.get("request_sha256")
        if (not isinstance(request_sha, str) or len(request_sha) != 64
                or binding.get("request_sha256") != request_sha
                or binding.get("request_path") != row.get("request_path")
                or binding.get("case_alias") != alias
                or binding.get("dataset") != domain
                or binding.get("GT_read") is not False
                or binding.get("query_used_by_tracker") is not False):
            raise ForecastError("tracking request/binding metadata is inconsistent or not query-independent")
        # The tracking global barrier's required per-query linkage is the
        # request and binding path/hash. Some valid barrier writers also copy
        # cache/accepted/inventory fields to the row; when present, verify
        # them, but use the SHA-pinned binding as the authority when omitted.
        if not _tracking_optional_fields_match(row, binding):
            raise ForecastError("tracking barrier duplicates cache/accepted/inventory values that disagree with its binding")
        for key, expected in (
                ("source_runtime_sha256", gates["runtime_sha256"]["source"]),
                ("source_capture_barrier_sha256", gates["barrier_sha256"]["source"]),
                ("tracking_runtime_sha256", gates["runtime_sha256"]["tracking"])):
            if binding.get(key) != expected:
                raise ForecastError("tracking binding is detached from its locked runtime lineage")
        if request_path.stat().st_size <= 0:
            raise ForecastError("tracking request file is empty")
        request_binding_bytes = request_path.stat().st_size + binding_path.stat().st_size
        costs[domain]["tracking_request_binding_bytes"].append(float(request_binding_bytes))
        actual_totals["tracking_query_metadata_bytes"] += request_binding_bytes

        cache_key = binding.get("tracking_cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ForecastError("tracking row has no stable content-addressed cache key")
        accepted_path = safe_path(binding.get("accepted_path", ""))
        inventory_path = safe_path(binding.get("inventory_path", ""))
        accepted = read_json(accepted_path)
        inventory = read_json(inventory_path)
        if (accepted.get("tracking_cache_key") != cache_key
                or accepted.get("GT_read") is not False
                or accepted.get("query_independent") is not True
                or inventory.get("tracking_cache_key") != cache_key
                or inventory.get("GT_read") is not False
                or inventory.get("mask_count") != len(inventory.get("files", []))
                or binding.get("accepted_sha256") != sha256_file(accepted_path)
                or binding.get("inventory_sha256") != sha256_file(inventory_path)
                or accepted.get("mask_inventory_sha256") != sha256_file(inventory_path)):
            raise ForecastError("tracking accepted/inventory metadata is inconsistent")
        for expected_key, value in (("accepted_sha256", accepted_path),
                                    ("inventory_sha256", inventory_path)):
            expected = row.get(expected_key)
            if expected and sha256_file(value) != expected:
                raise ForecastError(f"tracking {expected_key} differs from sealed barrier")
        mask_files = inventory.get("files")
        if not isinstance(mask_files, list):
            raise ForecastError("tracking inventory lacks mask byte metadata")
        mask_bytes = 0
        for item in mask_files:
            if (not isinstance(item, Mapping) or isinstance(item.get("bytes"), bool)
                    or not isinstance(item.get("bytes"), int) or item["bytes"] < 0):
                raise ForecastError("malformed lossless-mask byte receipt")
            mask_bytes += int(item["bytes"])
        measurement = {
            "dataset": domain,
            "seconds": _finite_nonnegative(accepted.get("actual_tracking_wall_s"),
                                            "actual_tracking_wall_s"),
            "mask_bytes": mask_bytes,
            "mask_count": len(mask_files),
            "marker_bytes": safe_path(binding.get("marker_video_path", "")).stat().st_size,
            "canonical_video_bytes": int(binding.get("canonical_video_bytes", 0)),
            "accepted_bytes": accepted_path.stat().st_size,
            "inventory_bytes": inventory_path.stat().st_size,
            "canonical_receipt_bytes": safe_path(binding.get("canonical_receipt_path", "")).stat().st_size,
            "accepted_sha256": binding.get("accepted_sha256"),
            "inventory_sha256": binding.get("inventory_sha256"),
        }
        if any(not isinstance(measurement[k], int) or measurement[k] < 0 for k in (
                "mask_bytes", "mask_count", "marker_bytes", "canonical_video_bytes",
                "accepted_bytes", "inventory_bytes", "canonical_receipt_bytes")):
            raise ForecastError("tracking byte/count metadata is invalid")
        old = unique_tracking.get(cache_key)
        if old is not None:
            compare = ("dataset", "seconds", "mask_bytes", "mask_count", "marker_bytes",
                       "canonical_video_bytes", "canonical_receipt_bytes", "accepted_bytes",
                       "inventory_bytes", "accepted_sha256", "inventory_sha256")
            if any(old[k] != measurement[k] for k in compare):
                raise ForecastError("same tracking cache key has inconsistent group assets")
        else:
            unique_tracking[cache_key] = measurement

    # Teacher payload: wall_seconds is a sequential per-case inference interval;
    # the shared model load is reported separately from the global barrier.
    teacher_batch = gates["barriers"]["teacher"]
    teacher_stage_wall = _finite_nonnegative(teacher_batch.get("wall_seconds"),
                                              "teacher barrier wall_seconds")
    teacher_load = teacher_batch.get("model_load_s", teacher_batch.get("model_load_seconds"))
    teacher_load = (_finite_nonnegative(teacher_load, "teacher model load seconds")
                    if teacher_load is not None else None)
    fresh_teacher_calls = 0
    teacher_token_complete_cases = 0
    for alias, row in teacher_rows.items():
        domain = roster_by_alias[alias]["dataset"]
        path, size, digest = _verify_file_row(row, "teacher case")
        receipt_path = path.with_suffix(".receipt.json")
        receipt = _verify_receipt(receipt_path, None)
        if (receipt.get("sha256") != digest or receipt.get("bytes") != size
                or receipt.get("source_sha256") != sha256_file(
                    safe_path(source_rows[alias].get("path", "")))
                or receipt.get("runtime_sha256") != gates["runtime_sha256"]["teacher"]
                or receipt.get("request_sha256") != track_rows[alias].get("request_sha256")
                or receipt.get("tracking_binding_sha256") != track_rows[alias].get("tracking_binding_sha256")
                or receipt.get("GT_read") is not False):
            raise ForecastError("teacher payload and receipt do not bind their sealed source")
        case = _load_pt_cpu(path)
        if not isinstance(case, Mapping) or case.get("case_id") != alias or case.get("GT_read") is not False:
            raise ForecastError("teacher payload is not the sealed GT-free preflight case")
        seconds = _finite_nonnegative(case.get("wall_seconds"), "teacher.wall_seconds")
        temporal = case.get("temporal", {})
        responses = temporal.get("teacher_responses", []) if isinstance(temporal, Mapping) else []
        if not isinstance(responses, list):
            raise ForecastError("teacher temporal response list is malformed")
        who = case.get("who", {})
        generation = who.get("generation", {}) if isinstance(who, Mapping) else {}
        generation_calls = generation.get("generation_call_count", 0) if isinstance(generation, Mapping) else 0
        if isinstance(generation_calls, bool) or not isinstance(generation_calls, int) or generation_calls < 0:
            raise ForecastError("teacher WHO generation-call counter is malformed")
        # Each valid temporal caption/alignment response pair performs two
        # model calls; an unavailable temporal arm has an empty response list.
        calls = 2 * len(responses) + generation_calls
        if row.get("reused") is False:
            fresh_teacher_calls += calls
        elif row.get("reused") is not True:
            raise ForecastError("teacher barrier omits the per-case reuse flag")
        # Native WHO generation records use *_token_count, while temporal
        # response rows use input_tokens/caption_input_tokens/caption_output_tokens.
        token_totals = (_teacher_token_totals(generation, responses)
                        if isinstance(generation, Mapping) else None)
        costs[domain]["teacher_case_wall_seconds"].append(seconds)
        costs[domain]["teacher_calls"].append(float(calls))
        receipt_size = receipt_path.stat().st_size
        costs[domain]["teacher_bytes"].append(float(size + receipt_size))
        if token_totals is not None:
            in_tokens, out_tokens = token_totals
            costs[domain]["teacher_input_tokens"].append(float(in_tokens))
            costs[domain]["teacher_output_tokens"].append(float(out_tokens))
            actual_totals["teacher_input_tokens"] += in_tokens
            actual_totals["teacher_output_tokens"] += out_tokens
            teacher_token_complete_cases += 1
        actual_totals["teacher_case_bytes"] += size + receipt_size
        actual_totals["teacher_calls"] += calls
        del case
    if teacher_token_complete_cases != len(teacher_rows):
        # Never report a partial sum as if it were the cohort total.
        actual_totals["teacher_input_tokens"] = None
        actual_totals["teacher_output_tokens"] = None
    barrier_teacher_calls = teacher_batch.get("actual_teacher_calls")
    if (isinstance(barrier_teacher_calls, bool) or not isinstance(barrier_teacher_calls, int)
            or barrier_teacher_calls != fresh_teacher_calls):
        raise ForecastError("teacher per-case calls do not sum to the actual fresh-call barrier counter")
    actual_totals["teacher_calls_in_fresh_outputs"] = fresh_teacher_calls
    actual_totals["teacher_model_loads"] = _nonnegative_int(
        teacher_batch.get("actual_model_loads"), "actual_model_loads")

    # Student payload: these fields are nonoverlapping instrumented phases.
    for alias, row in student_rows.items():
        domain = roster_by_alias[alias]["dataset"]
        path, size, digest = _verify_file_row(row, "student case")
        receipt_obj = row.get("receipt")
        if not isinstance(receipt_obj, Mapping):
            raise ForecastError("student barrier omits its nested receipt reference")
        receipt_path = safe_path(receipt_obj.get("path", ""))
        receipt = _verify_receipt(receipt_path, receipt_obj.get("sha256"),
                                  expected_size=receipt_obj.get("bytes"))
        if (receipt.get("sha256") != digest or receipt.get("bytes") != size
                or receipt.get("case_id") != alias or receipt.get("GT_read") is not False):
            raise ForecastError("student payload and receipt disagree")
        if any(receipt.get(key) != gates["runtime_sha256"][runtime_key] for key, runtime_key in (
                ("source_runtime_sha256", "source"),
                ("teacher_runtime_sha256", "teacher"),
                ("student_runtime_sha256", "student"))):
            raise ForecastError("student receipt is detached from its locked runtime lineage")
        teacher_row = teacher_rows[alias]
        teacher_receipt_path = safe_path(teacher_row.get("path", "")).with_suffix(".receipt.json")
        if any(receipt.get(key) != value for key, value in (
                ("source_file_sha256", source_rows[alias].get("sha256")),
                ("teacher_file_sha256", teacher_row.get("sha256")),
                ("teacher_receipt_sha256", sha256_file(teacher_receipt_path)),
                ("request_sha256", track_rows[alias].get("request_sha256")),
                ("tracking_binding_sha256", track_rows[alias].get("tracking_binding_sha256")))):
            raise ForecastError("student receipt is detached from its actual source/teacher/tracking files")
        case = _load_pt_cpu(path)
        if not isinstance(case, Mapping) or case.get("case_id") != alias or case.get("GT_read") is not False:
            raise ForecastError("student payload is not the sealed GT-free preflight case")
        execution = case.get("execution")
        if not isinstance(execution, Mapping):
            raise ForecastError("student case omits execution timing counters")
        capture = _finite_nonnegative(execution.get("source_capture_seconds"), "student source capture seconds")
        fit = _finite_nonnegative(execution.get("spatial_fit_seconds"), "student spatial fit seconds")
        readout = _finite_nonnegative(execution.get("full_readout_seconds"), "student full readout seconds")
        source_captures = _nonnegative_int(execution.get("source_capture_calls"), "source_capture_calls")
        if row.get("reused") is False:
            actual_totals["fresh_student_source_capture_calls"] = (
                actual_totals.get("fresh_student_source_capture_calls", 0) + source_captures)
        elif row.get("reused") is not True:
            raise ForecastError("student barrier omits the per-case reuse flag")
        fit_count = _nonnegative_int(execution.get("actual_fit_count"), "actual_fit_count")
        steps = _nonnegative_int(execution.get("optimizer_steps"), "optimizer_steps")
        redecodes = _nonnegative_int(execution.get("final_full_model_redecode_calls"),
                                     "final_full_model_redecode_calls")
        costs[domain]["student_source_capture_seconds"].append(capture)
        costs[domain]["student_fit_seconds"].append(fit)
        costs[domain]["student_readout_seconds"].append(readout)
        receipt_size = receipt_path.stat().st_size
        costs[domain]["student_bytes"].append(float(size + receipt_size))
        costs[domain]["student_fit_count"].append(float(fit_count))
        costs[domain]["student_optimizer_steps"].append(float(steps))
        costs[domain]["student_redecode_calls"].append(float(redecodes))
        actual_totals["student_case_bytes"] += size + receipt_size
        actual_totals["student_case_component_seconds"] += capture + fit + readout
        actual_totals["student_source_capture_calls"] += source_captures
        actual_totals["student_fit_count"] += fit_count
        actual_totals["student_optimizer_steps"] += steps
        actual_totals["student_final_full_model_redecode_calls"] += redecodes
        del case

    source_barrier = gates["barriers"]["source"]
    tracking_barrier = gates["barriers"]["tracking"]
    student_barrier = gates["barriers"]["student"]
    if actual_totals["student_fit_count"] != student_barrier.get("actual_fit_count"):
        raise ForecastError("student per-case fit counts differ from the sealed barrier")
    if actual_totals.get("fresh_student_source_capture_calls", 0) != student_barrier.get("actual_source_capture_calls"):
        raise ForecastError("student per-case source-capture calls differ from the sealed barrier")
    if actual_totals["student_optimizer_steps"] != student_barrier.get("optimizer_steps"):
        raise ForecastError("student per-case optimizer steps differ from the sealed barrier")
    if actual_totals["student_final_full_model_redecode_calls"] != student_barrier.get("final_full_model_redecode_calls"):
        raise ForecastError("student per-case full-model readouts differ from the sealed barrier")
    actual_totals["source_model_loads"] = _nonnegative_int(
        source_barrier.get("actual_source_model_loads"), "actual_source_model_loads")
    actual_totals["source_capture_calls"] = _nonnegative_int(
        source_barrier.get("actual_source_capture_calls"), "actual_source_capture_calls")
    actual_totals["source_offset_forwards"] = _nonnegative_int(
        source_barrier.get("actual_source_offset_forwards"), "actual_source_offset_forwards")
    actual_totals["reused_source_cases"] = _nonnegative_int(
        source_barrier.get("reused_source_cases"), "reused_source_cases")
    actual_totals["tracking_unique_groups"] = len(unique_tracking)
    if tracking_barrier.get("DINO_calls") != 0 or tracking_barrier.get("optimizer_steps") != 0:
        raise ForecastError("tracking barrier reports an unexpected optimization/DINO workload")

    return {"by_domain": costs, "tracking_groups": unique_tracking,
            "actual_totals": actual_totals,
            "teacher_stage_wall_seconds": teacher_stage_wall,
            "teacher_model_load_seconds": teacher_load}


def _forecast(gates: Mapping[str, Any], case_data: Mapping[str, Any]) -> dict[str, Any]:
    domain_counts, group_counts, full_roster_sha = _full_roster_counts()
    per_domain = {}
    size_projection = {"source": {}, "teacher": {}, "student": {}, "lossless_masks": {},
                       "canonical_video_and_marker": {}, "tracking_metadata": {},
                       "tracking_query_metadata": {}}
    time_projection: dict[str, Any] = {}
    for domain in EXPECTED_DOMAIN_COUNTS:
        unit = case_data["by_domain"][domain]
        group_rows = [v for v in case_data["tracking_groups"].values() if v["dataset"] == domain]
        t = len(group_rows)
        if t == 0:
            raise ForecastError(f"64-case preflight contains no accepted tracking groups for {domain}")
        per_domain[domain] = {
            "full_query_count": domain_counts[domain],
            "full_video_clock_group_count": group_counts[domain],
            "preflight_query_count": 32,
            "preflight_unique_tracking_groups": t,
            "source_capture_seconds_per_query": _stats(unit["source_seconds"]),
            "teacher_per_case_inference_wall_seconds": _stats(unit["teacher_case_wall_seconds"]),
            "teacher_actual_calls_per_query": _stats(unit["teacher_calls"]),
            "teacher_input_tokens_per_query_if_recorded": _stats(unit["teacher_input_tokens"]),
            "teacher_output_tokens_per_query_if_recorded": _stats(unit["teacher_output_tokens"]),
            "student_capture_seconds_per_query": _stats(unit["student_source_capture_seconds"]),
            "student_fit_seconds_per_query": _stats(unit["student_fit_seconds"]),
            "student_readout_seconds_per_query": _stats(unit["student_readout_seconds"]),
        }
        for stage, values in (("source", unit["source_bytes"]),
                              ("teacher", unit["teacher_bytes"]),
                              ("student", unit["student_bytes"])):
            st = _stats(values)
            size_projection[stage][domain] = {
                "per_query_bytes_p10_median_p90": {k: st[k] for k in ("p10", "median", "p90")},
                "full_domain_bytes_range": {
                    q: int(round(float(st[q]) * domain_counts[domain])) if st[q] is not None else None
                    for q in ("p10", "median", "p90")},
            }
        masks = _stats([float(row["mask_bytes"]) for row in group_rows])
        videos = _stats([float(row["canonical_video_bytes"] + row["marker_bytes"])
                         for row in group_rows])
        metadata = _stats([float(row["accepted_bytes"] + row["inventory_bytes"]
                                + row["canonical_receipt_bytes"])
                           for row in group_rows])
        size_projection["lossless_masks"][domain] = {
            "sampled_groups": t, "mask_file_count": sum(row["mask_count"] for row in group_rows),
            "per_group_bytes_p10_median_p90": {k: masks[k] for k in ("p10", "median", "p90")},
            "full_domain_bytes_range": {
                q: int(round(float(masks[q]) * group_counts[domain])) if masks[q] is not None else None
                for q in ("p10", "median", "p90")},
        }
        size_projection["canonical_video_and_marker"][domain] = {
            "per_group_bytes_p10_median_p90": {k: videos[k] for k in ("p10", "median", "p90")},
            "full_domain_bytes_range": {
                q: int(round(float(videos[q]) * group_counts[domain])) if videos[q] is not None else None
                for q in ("p10", "median", "p90")},
        }
        size_projection["tracking_metadata"][domain] = {
            "per_group_bytes_p10_median_p90": {k: metadata[k] for k in ("p10", "median", "p90")},
            "full_domain_bytes_range": {
                q: int(round(float(metadata[q]) * group_counts[domain])) if metadata[q] is not None else None
                for q in ("p10", "median", "p90")},
        }
        request_binding = _stats(unit["tracking_request_binding_bytes"])
        size_projection["tracking_query_metadata"][domain] = {
            "per_query_bytes_p10_median_p90": {
                k: request_binding[k] for k in ("p10", "median", "p90")},
            "full_domain_bytes_range": {
                q: int(round(float(request_binding[q]) * domain_counts[domain]))
                if request_binding[q] is not None else None
                for q in ("p10", "median", "p90")},
        }

    stages = {name: _stage_elapsed(name) for name in (
        "source", "tracking", "teacher", "student", "root")}
    # Unbatched measured-unit projections. The teacher/student batch restart
    # overhead is stated separately because the full batch count is controlled
    # by the root lock and may not yet exist at forecast time.
    source_case_low = source_case_mid = source_case_high = 0.0
    tracking_low = tracking_mid = tracking_high = 0.0
    teacher_low = teacher_mid = teacher_high = 0.0
    student_low = student_mid = student_high = 0.0
    for domain, n in domain_counts.items():
        values = case_data["by_domain"][domain]
        s = _stats(values["source_seconds"])
        source_case_low += float(s["p10"] or 0.0) * n
        source_case_mid += float(s["median"] or 0.0) * n
        source_case_high += float(s["p90"] or 0.0) * n
        track_rows = [v for v in case_data["tracking_groups"].values() if v["dataset"] == domain]
        ts = _stats([v["seconds"] for v in track_rows])
        gn = group_counts[domain]
        tracking_low += float(ts["p10"] or 0.0) * gn
        tracking_mid += float(ts["median"] or 0.0) * gn
        tracking_high += float(ts["p90"] or 0.0) * gn

        # The preflight teacher's per-case wall_seconds starts immediately
        # before that case's calls, so unlike full-batch plan timestamps these
        # are disjoint per-query timings. Model loading is separate.
        tw = _stats(values["teacher_case_wall_seconds"])
        teacher_low += float(tw["p10"] or 0.0) * n
        teacher_mid += float(tw["median"] or 0.0) * n
        teacher_high += float(tw["p90"] or 0.0) * n
        tc = _stats(values["teacher_calls"])
        student_components = [_stats(values[k]) for k in (
            "student_source_capture_seconds", "student_fit_seconds", "student_readout_seconds")]
        srates = [float(x["median"] or 0.0) for x in student_components]
        student_mid += sum(srates) * n
        student_low += sum(float(x["p10"] or 0.0) for x in student_components) * n
        student_high += sum(float(x["p90"] or 0.0) for x in student_components) * n

    overhead = {}
    source_stage = stages["source"]["seconds"]
    source_measured_sum = case_data["actual_totals"]["source_case_seconds"]
    overhead["source_worker_one_time_seconds"] = (
        max(0.0, source_stage - source_measured_sum) if source_stage is not None else None)
    track_actual_sum = math.fsum(row["seconds"] for row in case_data["tracking_groups"].values())
    track_stage = stages["tracking"]["seconds"]
    overhead["tracking_preflight_overhead_seconds"] = (
        max(0.0, track_stage - track_actual_sum) if track_stage is not None else None)

    # Student per-case phase timers exclude some worker setup/model loads.
    student_stage = stages["student"]["seconds"]
    student_component_sum = case_data["actual_totals"]["student_case_component_seconds"]
    student_one_invocation_overhead = (
        student_stage - student_component_sum if student_stage is not None else None)
    batch_plan_path = FULL / "BATCH_PLAN007.json"
    batch_plan = read_json(batch_plan_path) if batch_plan_path.is_file() else None
    batch_count = None
    groups_per_batch = None
    if batch_plan is not None:
        if (batch_plan.get("status") != "locked" or batch_plan.get("GT_read") is not False
                or batch_plan.get("count") != EXPECTED_QUERY_COUNT
                or batch_plan.get("video_clock_group_count") != EXPECTED_VIDEO_CLOCK_GROUPS):
            raise ForecastError("full batch plan is present but does not bind the locked full cohort")
        batch_count = batch_plan.get("batch_count")
        groups_per_batch = batch_plan.get("groups_per_batch")
        if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count <= 0:
            raise ForecastError("full batch plan has invalid batch count")

    batch_load_s = case_data["teacher_model_load_seconds"]
    teacher_batch_load_total = (batch_load_s * batch_count
                                if batch_load_s is not None and batch_count is not None else None)
    student_batch_overhead_total = (student_one_invocation_overhead * batch_count
                                    if student_one_invocation_overhead is not None and batch_count is not None
                                    else None)

    # A lower/central/upper operational range. Teacher's per-query body uses the
    # measured total after one model load; the repeated load term is explicit.
    teacher_load_for_range = float(teacher_batch_load_total or 0.0)
    student_overhead_for_range = float(student_batch_overhead_total or 0.0)
    source_start = float(overhead["source_worker_one_time_seconds"] or 0.0)
    track_start = float(overhead["tracking_preflight_overhead_seconds"] or 0.0)
    time_projection = {
        "source_total_seconds_p10_median_p90": {
            "p10": source_case_low + source_start,
            "median": source_case_mid + source_start,
            "p90": source_case_high + source_start,
        },
        "tracking_total_seconds_p10_median_p90": {
            "p10": tracking_low + track_start,
            "median": tracking_mid + track_start,
            "p90": tracking_high + track_start,
        },
        "teacher_query_compute_seconds_p10_median_p90": {
            "p10": teacher_low, "median": teacher_mid, "p90": teacher_high,
        },
        "teacher_total_seconds_p10_median_p90": {
            "p10": (teacher_low + teacher_load_for_range) if batch_count is not None else None,
            "median": (teacher_mid + teacher_load_for_range) if batch_count is not None else None,
            "p90": (teacher_high + teacher_load_for_range) if batch_count is not None else None,
        },
        "student_per_query_components_seconds_p10_median_p90": {
            "p10": student_low, "median": student_mid, "p90": student_high,
        },
        "student_total_seconds_p10_median_p90": {
            "p10": (student_low + student_overhead_for_range) if batch_count is not None else None,
            "median": (student_mid + student_overhead_for_range) if batch_count is not None else None,
            "p90": (student_high + student_overhead_for_range) if batch_count is not None else None,
        },
        "known_stage_sum_seconds_p10_median_p90": (
            {"p10": source_case_low + tracking_low + teacher_low + student_low + source_start + track_start + teacher_load_for_range + student_overhead_for_range,
             "median": source_case_mid + tracking_mid + teacher_mid + student_mid + source_start + track_start + teacher_load_for_range + student_overhead_for_range,
             "p90": source_case_high + tracking_high + teacher_high + student_high + source_start + track_start + teacher_load_for_range + student_overhead_for_range}
            if batch_count is not None else {"p10": None, "median": None, "p90": None}),
        "method": "per-domain p10/median/p90 from 32 preflight queries; tracking uses unique cache groups plus per-query request/binding metadata; teacher per-case inference timers are measured serially and exclude the separately recorded shared load, repeated model loads are included when a batch plan exists; student uses instrumented per-case phases",
        "repeated_batch_load_seconds": teacher_batch_load_total,
        "student_repeated_batch_overhead_seconds": student_batch_overhead_total,
        "batch_count": batch_count,
        "groups_per_batch": groups_per_batch,
        "full_video_clock_group_count": EXPECTED_VIDEO_CLOCK_GROUPS,
        "reliable_eta": False,
        "limitations": [
            "64 preflight queries are a small 32+32 engineering sample; p10/p90 are descriptive ranges, not confidence bounds.",
            "Preflight per-case teacher.wall_seconds are sequential per-case inference intervals, with shared model load reported separately; the full batch wall_seconds is retained as an independent whole-stage measurement.",
            "Teacher batch decoding/generation mix and input lengths may differ across the full roster.",
            "Batch restart overhead is only projected when a locked BATCH_PLAN007 is available; absent batch totals remain null rather than assuming one load.",
            "Tracking times are measured per unique preflight cache key; filesystem pressure and tracker throughput may change at full scale.",
            "The projection excludes unmeasured interruptions, retries, resource admission waits, and any new failure recovery.",
        ],
    }

    # Size projections sum per-query and per-video-group distributions. It is a
    # forecast of final logical outputs, not a claim that files already exist.
    size_totals = {}
    for q in ("p10", "median", "p90"):
        vals = []
        for category in ("source", "teacher", "student", "lossless_masks",
                         "canonical_video_and_marker", "tracking_metadata",
                         "tracking_query_metadata"):
            for domain in EXPECTED_DOMAIN_COUNTS:
                val = size_projection[category][domain]["full_domain_bytes_range"][q]
                if val is not None:
                    vals.append(val)
        size_totals[q] = int(sum(vals))
    disk = shutil.disk_usage(ROOT)
    available_growth = max(0, disk.free - RESERVE_BYTES)
    # Canonical and marker videos are temporary tracker-cache assets and are
    # reported separately; do not count them as retained final outputs. Lossless
    # masks use the uncompressed per-file byte receipts as a conservative
    # storage estimate; eventual archive compression is not extrapolated from
    # one sample. Query/group receipts are included, but global logs, runtimes,
    # and barriers are not.
    for q in ("p10", "median", "p90"):
        size_totals[q] -= sum(
            size_projection["canonical_video_and_marker"][domain]["full_domain_bytes_range"][q] or 0
            for domain in EXPECTED_DOMAIN_COUNTS)
    size_projection["permanent_outputs_plus_lossless_mask_bytes_p10_median_p90"] = size_totals
    size_projection["transient_canonical_video_and_marker_bytes_p10_median_p90"] = {
        q: sum(size_projection["canonical_video_and_marker"][domain]["full_domain_bytes_range"][q] or 0
               for domain in EXPECTED_DOMAIN_COUNTS) for q in ("p10", "median", "p90")}
    size_projection["combined_final_storage_bytes_p10_median_p90"] = size_totals
    size_projection["disk_budget"] = {
        "current_free_bytes": disk.free,
        "minimum_reserve_bytes": RESERVE_BYTES,
        "available_growth_before_reserve_bytes": available_growth,
        "median_projection_fits_budget": size_totals["median"] <= available_growth,
        "p90_projection_fits_budget": size_totals["p90"] <= available_growth,
        "note": "Current free space already reflects existing partial work; forecast adds full-roster outputs conservatively and does not subtract any overlapping preflight assets.",
    }

    return {
        "schema": "stvg-final-table1-preflight-cost-forecast-v1",
        "status": "pass",
        "scope": "64-case GT-free preflight cost extrapolation to the fixed full roster; no labels, model weights, or media read",
        "GT_read": False,
        "CPU_only": True,
        "preflight": {
            "count": 64, "domain_counts": {"hc2": 32, "vidstg": 32},
            "readback_sha256": gates["readback_sha256"],
            "barrier_sha256": gates["barrier_sha256"],
            "runtime_sha256": gates["runtime_sha256"],
            "stage_launch_exit": stages,
        },
        "full_roster": {
            "query_count": EXPECTED_QUERY_COUNT,
            "domain_counts": domain_counts,
            "video_clock_group_count": EXPECTED_VIDEO_CLOCK_GROUPS,
            "video_clock_group_counts_by_domain": group_counts,
            "roster_sha256": full_roster_sha,
        },
        "measured_preflight": {
            "actual_totals": case_data["actual_totals"],
            "unique_tracking_groups": len(case_data["tracking_groups"]),
            "tracking_group_costs": {
                domain: {
                    "actual_tracking_wall_seconds": _stats([
                        v["seconds"] for v in case_data["tracking_groups"].values()
                        if v["dataset"] == domain]),
                    "lossless_mask_png_bytes": _stats([
                        float(v["mask_bytes"]) for v in case_data["tracking_groups"].values()
                        if v["dataset"] == domain]),
                    "mask_count": sum(v["mask_count"] for v in case_data["tracking_groups"].values()
                                      if v["dataset"] == domain),
                } for domain in EXPECTED_DOMAIN_COUNTS
            },
            "teacher_batch_wall_seconds": case_data["teacher_stage_wall_seconds"],
            "teacher_model_load_seconds": case_data["teacher_model_load_seconds"],
            "source_one_time_overhead_seconds": overhead["source_worker_one_time_seconds"],
            "tracking_preflight_overhead_seconds": overhead["tracking_preflight_overhead_seconds"],
            "student_one_invocation_overhead_seconds": student_one_invocation_overhead,
            "per_domain": per_domain,
        },
        "forecast": {"time": time_projection, "storage": size_projection},
        "provenance": {
            "preflight_roster_sha256": gates["roster_sha256"],
            "full_roster_sha256": full_roster_sha,
            "source_barrier_sha256": gates["barrier_sha256"]["source"],
            "tracking_barrier_sha256": gates["barrier_sha256"]["tracking"],
            "teacher_barrier_sha256": gates["barrier_sha256"]["teacher"],
            "student_barrier_sha256": gates["barrier_sha256"]["student"],
            "runtime_sha256": gates["runtime_sha256"],
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "created_unix_time": time.time(),
    }


def contracts() -> dict[str, Any]:
    # In-memory synthetic checks only: no artifact reads or project data.
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _stats([])["median"] is None
    assert _stats([0.0, 0.0, 2.0])["sum"] == 2.0
    try:
        _finite_nonnegative(True, "bool")
    except ForecastError:
        bool_rejected = True
    else:
        raise AssertionError("boolean timing was accepted as numeric cost")
    # Match the actual saved HC01 cost ABI: WHO uses input_token_count and
    # output_token_count; each temporal row has three distinct token fields.
    generation = {"input_token_count": 5, "output_token_count": 2}
    responses = [{"input_tokens": 7, "caption_input_tokens": 11,
                  "caption_output_tokens": 3}]
    assert _teacher_token_totals(generation, responses) == (23, 5)
    assert _teacher_token_totals({"input_tokens": 5, "output_tokens": 2}, responses) is None
    assert STUDENT_PREFLIGHT_RUNTIME.name == "STUDENT_RUNTIME004.json"
    binding_row = {"request_path": "request.json", "tracking_binding_path": "binding.json"}
    assert _tracking_optional_fields_match(binding_row, {
        **binding_row, "tracking_cache_key": "k", "accepted_path": "a", "accepted_sha256": "h",
        "inventory_path": "i", "inventory_sha256": "j"})
    assert not _tracking_optional_fields_match({**binding_row, "accepted_path": "wrong"}, {
        **binding_row, "accepted_path": "a"})
    return {"status": "pass", "synthetic_contracts": {
        "quantile_interpolation": True, "empty_cost_is_null": True,
        "zero_cost_preserved": True, "boolean_cost_rejected": bool_rejected,
        "metadata_only_group_identity": True,
        "who_and_temporal_token_fields_match_saved_abi": True,
        "partial_token_totals_not_reported_as_cohort_total": True,
        "tracking_binding_is_authoritative_when_global_row_omits_cache_fields": True,
        "preflight_student_runtime_is_student004": True,
    }, "GT_read": False, "CUDA_used": False, "model_imported": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run synthetic CPU checks without reading project artifacts")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True))
        return 0

    # The gate phase is intentionally first. If it fails, no case payload,
    # receipt, full roster, or cache metadata is opened.
    gates = _preflight_gates()
    case_data = _case_costs(gates)
    report = _forecast(gates, case_data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with OUT.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ForecastError("forecast output already exists; preserve it and use a new numbered name") from exc
    print(json.dumps({"status": report["status"], "output": str(OUT.relative_to(ROOT)),
                      "sha256": sha256_file(OUT), "queries": EXPECTED_QUERY_COUNT,
                      "video_clock_groups": EXPECTED_VIDEO_CLOCK_GROUPS,
                      "GT_read": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ForecastError as exc:
        print(f"COST_FORECAST_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
