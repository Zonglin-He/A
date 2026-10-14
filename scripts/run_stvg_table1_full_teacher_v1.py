"""Hot, frozen STVG-R1 teacher adapter for bounded FULL Table 1 batches.

This adapter consumes a root-locked full source barrier and one bounded
query-independent tracking batch barrier.  It reuses the frozen 64-query
teacher functions and adapters without changing their scientific behavior.
Each query is written once; a partial teacher attempt is never silently
replayed.  ``--finalize`` is a CPU-only seal of all 13,785 immutable cases.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
MANIFEST = BASE / "INPUT_MANIFEST.json"
ROSTER = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
SOURCE_RUNTIME = FULL / "SOURCE_RUNTIME.json"
SOURCE_BARRIER = FULL / "SOURCE_CAPTURE_BARRIER.json"
TRACKING_RUNTIME = FULL / "TRACKING_RUNTIME.json"
R1_LOCK = BASE / "R1_EXECUTION_LOCK.json"
TEACHER_RUNTIME = FULL / "TEACHER_RUNTIME.json"
SOURCE_CASES = FULL / "source_cases"
TRACKING_CASES = FULL / "tracking_cases"
TEACHER_CASES = FULL / "teacher_cases"
ATTEMPTS = FULL / "teacher_attempts"
BATCH_BARRIERS = FULL / "TEACHER_BATCH_BARRIERS"
GLOBAL_BARRIER = FULL / "TEACHER_GLOBAL_BARRIER.json"
STATUS = FULL / "TEACHER_STATUS.json"
CONTRACTS = FULL / "teacher_contracts/CPU_CONTRACTS.json"

WORKER_REL = "scripts/run_stvg_table1_full_teacher_v1.py"
FROZEN_64_TEACHER_REL = "scripts/run_stvg_table1_teacher_v1.py"
FROZEN_64_TEACHER_SHA256 = "0024a453ffa68fee00aeeb43f4c90c0b1f7b2cfb0e858a89584a40900957c327"
FULL_SOURCE_WORKER_REL = "scripts/run_stvg_table1_full_source_v1.py"
SOURCE_ADAPTER_REL = "scripts/run_stvg_table1_source_v1.py"
TRACKING_WORKER_REL = "scripts/run_stvg_table1_tracking_v1.py"
FULL_TRACKING_WORKER_REL = "scripts/run_stvg_table1_full_tracking_v1.py"
NATIVE_ADAPTER_REL = "scripts/run_stvg_table1_native_instance_adapter_v1.py"
TEMPORAL_ADAPTER_REL = "scripts/stvg_table1_temporal_adapter_v1.py"
SPATIAL_ADAPTER_REL = "scripts/stvg_table1_spatial_adapter_v1.py"
NATIVE_R1_REL = "scripts/stvg_r1_native_teacher_v1.py"
MODEL_WORKER_REL = "scripts/run_stvg_native_instance_teacher_storage002.py"
COMMON_REL = "scripts/stvg_native_instance_common_v1.py"
MODEL_RECEIPT_REL = "artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json"
PREFLIGHT_ROSTER = BASE / "PREFLIGHT_ROSTER_LOCK.json"
PREFLIGHT_SOURCE_RUNTIME = BASE / "SOURCE_RUNTIME.json"
PREFLIGHT_TRACKING_RUNTIME = BASE / "TRACKING_RUNTIME.json"

FULL_SOURCE_BARRIER_SCHEMA = "stvg-final-table1-full-source-capture-barrier-v1"
FULL_TRACKING_BARRIER_SCHEMA = "stvg-final-table1-full-tracking-batch-barrier-v1"
TEACHER_BATCH_BARRIER_SCHEMA = "stvg-final-table1-full-teacher-batch-barrier-v1"
TEACHER_GLOBAL_BARRIER_SCHEMA = "stvg-final-table1-full-teacher-global-barrier-v1"
ALIAS_FILE_SCHEMA = "stvg-final-table1-alias-batch-v1"
DOMAIN_COUNTS = {"hc2": 3482, "vidstg": 10303}
TOTAL_CASES = sum(DOMAIN_COUNTS.values())
SEED = 20261015
EXPECTED_EXECUTION = {
    "BF16": True,
    "quantized": False,
    "attention": "sdpa",
    "device_map": "auto",
    "use_cache": True,
    "max_new_tokens": 2048,
    "max_memory": {"GPU0": "22GiB", "cpu": "4GiB"},
    "seed": SEED,
}

from scripts.stvg_native_instance_common_v1 import guard, read, sha, write


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _file_sha(path: Path) -> str:
    return sha(path)


def _safe_project_file(relative: Any, *, expected_base: Path | None = None) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("locked metadata contains an empty project-relative path")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise RuntimeError("locked metadata contains an unsafe project-relative path")
    path = (ROOT / rel).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve(strict=True)):
        raise RuntimeError("locked path is not a project file")
    if expected_base is not None and not path.is_relative_to(expected_base.resolve(strict=True)):
        raise RuntimeError("locked file escaped the expected Table 1 subtree")
    return path


def _safe_json_file(relative: Any, *, expected_base: Path | None = None) -> tuple[Path, dict[str, Any]]:
    path = _safe_project_file(relative, expected_base=expected_base)
    value = read(path)
    if not isinstance(value, dict):
        raise RuntimeError("locked JSON file must contain an object")
    return path, value


def _canonical_alias_sha(aliases: Sequence[str]) -> str:
    return _sha_bytes(_canonical_json_bytes(list(aliases)))


def _domain_decode_policy(dataset: str) -> str:
    if dataset == "hc2":
        return "ffmpeg_default_timing"
    if dataset == "vidstg":
        return "ffmpeg_vsync0_original_grid"
    raise ValueError("unknown Table 1 domain")


def video_clock_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen source-video/clock group identity from roster metadata."""
    dataset = record.get("dataset")
    required = ("video_sha256", "fps", "video_bounds", "width", "height")
    if any(key not in record for key in required):
        raise ValueError("full roster row lacks canonical video/clock identity")
    bounds = record["video_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("physical bounds must be a half-open pair")
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)
            or bounds[0] < 0 or bounds[1] <= bounds[0]):
        raise ValueError("physical bounds must be an increasing integral half-open pair")
    if dataset not in DOMAIN_COUNTS:
        raise ValueError("unsupported Table 1 dataset identity")
    if (not isinstance(record["video_sha256"], str) or len(record["video_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["video_sha256"])):
        raise ValueError("source video hash must be lowercase SHA-256")
    fps = record["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0:
        raise ValueError("source fps must be positive")
    if any(isinstance(record[key], bool) or not isinstance(record[key], int) or record[key] <= 0
           for key in ("width", "height")):
        raise ValueError("source dimensions must be positive integers")
    return {
        "dataset_decode_policy": _domain_decode_policy(str(dataset)),
        "source_video_sha256": record["video_sha256"],
        "physical_bounds_half_open": list(bounds),
        "fps": float(fps),
        "width": record["width"],
        "height": record["height"],
        "frame_clock_policy": "contiguous_physical_frame_ids_v1",
    }


def video_clock_group_sha256(record: Mapping[str, Any]) -> str:
    return _sha_bytes(_canonical_json_bytes(video_clock_identity(record)))


def _roster_aliases(
    records: Sequence[Mapping[str, Any]],
    *, expected_domain_counts: Mapping[str, int] = DOMAIN_COUNTS,
) -> tuple[list[str], dict[str, int], dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    expected_total = sum(expected_domain_counts.values())
    if len(records) != expected_total:
        raise RuntimeError(f"source roster must contain all {expected_total} expected queries")
    aliases: list[str] = []
    by_alias: dict[str, Mapping[str, Any]] = {}
    group_members: dict[str, list[str]] = collections.defaultdict(list)
    domain_counts: collections.Counter[str] = collections.Counter()
    for row in records:
        alias = row.get("probe_alias")
        dataset = row.get("dataset")
        ordinal = row.get("query_ordinal")
        prefix = "HC" if dataset == "hc2" else "V" if dataset == "vidstg" else None
        expected = f"{prefix}{ordinal:05d}" if prefix and isinstance(ordinal, int) else None
        if alias != expected or alias in by_alias:
            raise RuntimeError("full roster alias/ordinal identity is duplicated or malformed")
        if row.get("GT_read") is not False:
            raise RuntimeError("full roster must remain explicitly GT-free")
        aliases.append(alias)
        by_alias[alias] = row
        group_members[video_clock_group_sha256(row)].append(alias)
        domain_counts[str(dataset)] += 1
    expected_aliases = [f"HC{i:05d}" for i in range(expected_domain_counts.get("hc2", 0))]
    expected_aliases.extend(f"V{i:05d}" for i in range(expected_domain_counts.get("vidstg", 0)))
    if dict(domain_counts) != dict(expected_domain_counts) or aliases != expected_aliases:
        raise RuntimeError("full roster domain counts/order differ from the locked query sequence")
    index = {alias: i for i, alias in enumerate(aliases)}
    return aliases, index, by_alias, dict(group_members)


def _group_descriptors(aliases: Sequence[str], group_members: Mapping[str, Sequence[str]],
                       by_alias: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_set = set(aliases)
    selected_group_ids = list(dict.fromkeys(video_clock_group_sha256(by_alias[a]) for a in aliases))
    descriptors = []
    for group_sha in selected_group_ids:
        full_aliases = list(group_members[group_sha])
        if not set(full_aliases).issubset(selected_set):
            raise ValueError("batch splits a complete FULL-roster video/clock group")
        descriptors.append({
            "video_clock_group_sha256": group_sha,
            "identity": video_clock_identity(by_alias[full_aliases[0]]),
            "aliases": full_aliases,
        })
    return descriptors


def _parse_alias_file_bytes(data: bytes) -> list[str]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("aliases file must be UTF-8 JSON") from exc
    if (not isinstance(value, dict) or not {"schema", "aliases"}.issubset(value)
            or set(value) - {"schema", "aliases", "video_clock_groups"}):
        raise ValueError("aliases file has an unexpected schema")
    if value.get("schema") != ALIAS_FILE_SCHEMA or not isinstance(value.get("aliases"), list):
        raise ValueError("aliases file schema is not the frozen Table 1 batch ABI")
    aliases = value["aliases"]
    if not aliases or any(not isinstance(alias, str) or not alias for alias in aliases):
        raise ValueError("aliases file must contain a nonempty string list")
    if len(set(aliases)) != len(aliases):
        raise ValueError("aliases file contains duplicate rows")
    if data != _canonical_json_bytes(value):
        raise ValueError("aliases file must use canonical JSON bytes")
    return list(aliases)


def select_aliases(
    records: Sequence[Mapping[str, Any]], *, aliases_file_bytes: bytes | None = None,
    all_mode: bool = False, limit: int | None = None,
    expected_domain_counts: Mapping[str, int] = DOMAIN_COUNTS,
) -> tuple[list[str], str | None]:
    """Select a batch without splitting a locked video/clock identity group."""
    roster_order, _, by_alias, groups = _roster_aliases(
        records, expected_domain_counts=expected_domain_counts)
    if (aliases_file_bytes is None) == (not all_mode):
        raise ValueError("choose exactly one of aliases-file input or --all")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("--limit must be a positive hard cap")
    if all_mode:
        aliases = list(roster_order)
        aliases_file_sha = None
    else:
        assert aliases_file_bytes is not None
        aliases = _parse_alias_file_bytes(aliases_file_bytes)
        aliases_file_sha = _sha_bytes(aliases_file_bytes)
        unknown = [alias for alias in aliases if alias not in by_alias]
        if unknown:
            raise ValueError("aliases file contains a non-roster alias")
        roster_index = {alias: i for i, alias in enumerate(roster_order)}
        if aliases != sorted(aliases, key=roster_index.__getitem__):
            raise ValueError("aliases file must preserve FULL roster order")
    if limit is not None and len(aliases) > limit:
        raise ValueError("batch exceeds --limit; use complete video/clock groups in a smaller aliases file")
    descriptors = _group_descriptors(aliases, groups, by_alias)
    if not all_mode and aliases_file_bytes is not None:
        declared = json.loads(aliases_file_bytes.decode("utf-8")).get("video_clock_groups")
        if declared is not None and declared != descriptors:
            raise ValueError("aliases file declares video/clock groups that differ from FULL roster")
    return aliases, aliases_file_sha


def _assert_runtime_pins(runtime: Mapping[str, Any]) -> None:
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict) or not pins:
        raise RuntimeError("FULL TEACHER_RUNTIME requires source pins and protected artifacts")
    required = {
        WORKER_REL, FROZEN_64_TEACHER_REL, FULL_SOURCE_WORKER_REL, SOURCE_ADAPTER_REL,
        TRACKING_WORKER_REL, FULL_TRACKING_WORKER_REL, NATIVE_ADAPTER_REL,
        TEMPORAL_ADAPTER_REL, SPATIAL_ADAPTER_REL,
        NATIVE_R1_REL, MODEL_WORKER_REL, COMMON_REL,
    }
    if not required.issubset(pins):
        raise RuntimeError(f"TEACHER_RUNTIME omits required source pins: {sorted(required - set(pins))}")
    if pins[FROZEN_64_TEACHER_REL] != FROZEN_64_TEACHER_SHA256:
        raise RuntimeError("the frozen 64-query teacher source differs from its authorized SHA")
    if MODEL_RECEIPT_REL not in pins and MODEL_RECEIPT_REL not in protected:
        raise RuntimeError("TEACHER_RUNTIME must pin the official local model-file readback")
    for rel, digest in {**pins, **protected}.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError("runtime pin contains an unsafe path")
        path = ROOT / rel_path
        if not path.is_file() or not isinstance(digest, str) or _file_sha(path) != digest:
            raise RuntimeError(f"FULL TEACHER_RUNTIME pin mismatch: {rel}")
    execution = runtime.get("execution")
    if not isinstance(execution, Mapping):
        raise RuntimeError("TEACHER_RUNTIME must pin the fixed original teacher execution settings")
    for key, expected in EXPECTED_EXECUTION.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"teacher execution setting differs from the frozen protocol: {key}")
    r1_lock = read(R1_LOCK)
    r1_expected = {
        "status": "accepted_bounded_execution_variant",
        "GT_read": False,
        "BF16": True,
        "quantized": False,
        "attention": "sdpa",
        "max_memory": {"GPU0": "22GiB", "cpu": "4GiB"},
        "max_new_tokens": 2048,
        "use_cache": True,
        "manual_seed_per_query": SEED,
        "model_residency": True,
    }
    if any(r1_lock.get(key) != value for key, value in r1_expected.items()):
        raise RuntimeError("R1_EXECUTION_LOCK differs from the original bounded teacher configuration")
    max_video_bytes = execution.get("max_video_group_cache_bytes")
    if isinstance(max_video_bytes, bool) or not isinstance(max_video_bytes, int) or max_video_bytes <= 0:
        raise RuntimeError("TEACHER_RUNTIME must set a positive max_video_group_cache_bytes")


def _verify_full_source_authority() -> dict[str, Any]:
    from scripts import run_stvg_table1_full_source_v1 as source_full

    runtime, roster, records = source_full.verify_runtime()
    source_runtime_sha = _file_sha(SOURCE_RUNTIME)
    roster_sha = _file_sha(ROSTER)
    manifest_sha = _file_sha(MANIFEST)
    barrier = read(SOURCE_BARRIER)
    aliases, _, record_by_alias, _ = _roster_aliases(records)
    expected = {
        "schema": FULL_SOURCE_BARRIER_SCHEMA,
        "status": "sealed",
        "count": TOTAL_CASES,
        "source_runtime_sha256": source_runtime_sha,
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
        "GT_read": False,
    }
    if any(barrier.get(key) != value for key, value in expected.items()):
        raise RuntimeError("FULL SOURCE_CAPTURE_BARRIER is not the completed locked source run")
    source_files = barrier.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(aliases):
        raise RuntimeError("FULL source barrier does not bind every locked source query")
    return {
        "runtime": runtime,
        "roster": roster,
        "records": records,
        "aliases": aliases,
        "record_by_alias": {row["probe_alias"]: row for row in records},
        "source_files": source_files,
        "source_runtime_sha256": source_runtime_sha,
        "source_barrier_sha256": _file_sha(SOURCE_BARRIER),
        "roster_sha256": roster_sha,
        "manifest_sha256": manifest_sha,
        "preflight_roster_sha256": _file_sha(PREFLIGHT_ROSTER),
        "preflight_source_runtime_sha256": _file_sha(PREFLIGHT_SOURCE_RUNTIME),
        "preflight_tracking_runtime_sha256": _file_sha(PREFLIGHT_TRACKING_RUNTIME),
    }


def _verify_teacher_runtime(source_locked: Mapping[str, Any]) -> dict[str, Any]:
    if not TEACHER_RUNTIME.is_file():
        raise RuntimeError("root must create FULL/TEACHER_RUNTIME.json before teacher inference")
    runtime = read(TEACHER_RUNTIME)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("FULL TEACHER_RUNTIME must be locked with GT_read=false")
    _assert_runtime_pins(runtime)
    bindings = runtime.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("FULL TEACHER_RUNTIME must bind its fixed source and tracking inputs")
    expected = {
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "tracking_runtime_sha256": _file_sha(TRACKING_RUNTIME),
        "r1_execution_lock_sha256": _file_sha(R1_LOCK),
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise RuntimeError("FULL TEACHER_RUNTIME is stale or does not bind the locked source/tracking authorities")
    return runtime


def _verify_tracking_runtime(source_locked: Mapping[str, Any]) -> str:
    if not TRACKING_RUNTIME.is_file():
        raise RuntimeError("FULL TRACKING_RUNTIME is missing")
    runtime = read(TRACKING_RUNTIME)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("FULL TRACKING_RUNTIME must be locked and GT-free")
    for rel, digest in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError("tracking runtime contains an unsafe path")
        if not (ROOT / rel_path).is_file() or _file_sha(ROOT / rel_path) != digest:
            raise RuntimeError(f"FULL TRACKING_RUNTIME pin mismatch: {rel}")
    bindings = runtime.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("FULL TRACKING_RUNTIME source bindings are required")
    expected = {
        "full_source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "preflight_source_runtime_sha256": source_locked["preflight_source_runtime_sha256"],
        "preflight_tracking_runtime_sha256": source_locked["preflight_tracking_runtime_sha256"],
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise RuntimeError(f"FULL TRACKING_RUNTIME does not bind {key}")
    return _file_sha(TRACKING_RUNTIME)


def _load_batch_tracking_barrier(path_arg: str, aliases: Sequence[str],
                                 aliases_file_sha256: str | None,
                                 source_locked: Mapping[str, Any],
                                 tracking_runtime_sha256: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(path_arg)
    if not path.is_absolute():
        path = (ROOT / path).resolve(strict=True)
    else:
        path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(FULL.resolve(strict=True)):
        raise RuntimeError("tracking batch barrier must be a file within the FULL artifact tree")
    barrier = read(path)
    if (barrier.get("schema") != FULL_TRACKING_BARRIER_SCHEMA
            or barrier.get("status") != "complete"
            or barrier.get("count") != len(aliases)
            or barrier.get("GT_read") is not False
            or barrier.get("query_used_by_tracker") is not False
            or barrier.get("selection_sha256") != _canonical_alias_sha(aliases)
            or barrier.get("aliases") != list(aliases)):
        raise RuntimeError("tracking batch barrier does not bind this ordered alias batch")
    expected_bindings = {
        "full_source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "tracking_runtime_sha256": tracking_runtime_sha256,
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "preflight_source_runtime_sha256": source_locked["preflight_source_runtime_sha256"],
        "preflight_tracking_runtime_sha256": source_locked["preflight_tracking_runtime_sha256"],
    }
    bindings = barrier.get("bindings")
    if not isinstance(bindings, Mapping) or any(bindings.get(k) != v for k, v in expected_bindings.items()):
        raise RuntimeError("tracking batch barrier has stale source/runtime bindings")
    submitted_alias_sha = barrier.get("submitted_alias_file_sha256")
    if aliases_file_sha256 is not None and submitted_alias_sha != aliases_file_sha256:
        raise RuntimeError("tracking batch barrier does not bind the exact submitted alias-file bytes")
    alias_path, alias_doc = _safe_json_file(barrier.get("alias_file_path"), expected_base=FULL)
    if (alias_path.stat().st_size != barrier.get("alias_file_bytes")
            or _file_sha(alias_path) != barrier.get("alias_file_sha256")
            or alias_doc.get("schema") != ALIAS_FILE_SCHEMA
            or alias_doc.get("aliases") != list(aliases)):
        raise RuntimeError("tracking barrier's saved alias file does not match this exact batch")
    groups = barrier.get("video_clock_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("tracking batch barrier must enumerate complete video/clock groups")
    group_by_alias: dict[str, str] = {}
    group_aliases: list[str] = []
    group_order = []
    _, _, record_by_alias, roster_groups = _roster_aliases(source_locked["records"])
    roster_index = {alias: i for i, alias in enumerate(source_locked["aliases"])}
    expected_descriptors = _group_descriptors(aliases, roster_groups, record_by_alias)
    if groups != expected_descriptors or alias_doc.get("video_clock_groups") != expected_descriptors:
        raise RuntimeError("tracking batch video/clock groups differ from complete FULL roster groups")
    group_aliases_by_sha = {d["video_clock_group_sha256"]: d["aliases"] for d in expected_descriptors}
    for group in groups:
        if not isinstance(group, Mapping):
            raise RuntimeError("video_clock_groups entries must be objects")
        group_sha = group.get("video_clock_group_sha256")
        identity = group.get("identity")
        group_members = group.get("aliases")
        if (not isinstance(group_sha, str) or not isinstance(identity, Mapping)
                or not isinstance(group_members, list) or not group_members):
            raise RuntimeError("video_clock_groups entry is incomplete")
        expected_group_aliases = group_aliases_by_sha.get(group_sha)
        if not expected_group_aliases:
            raise RuntimeError("tracking barrier includes an unknown video/clock group")
        if group.get("identity") != video_clock_identity(record_by_alias[expected_group_aliases[0]]):
            raise RuntimeError("tracking video/clock identity metadata differs from the full source roster")
        if group_sha != video_clock_group_sha256(record_by_alias[expected_group_aliases[0]]):
            raise RuntimeError("tracking video/clock identity digest differs from the full source roster")
        if list(group_members) != expected_group_aliases:
            raise RuntimeError("tracking batch splits or reorders a complete FULL-roster video/clock group")
        group_order.append((roster_index[group_members[0]], group_sha, list(group_members)))
        for alias in group_members:
            if alias not in aliases or alias in group_by_alias:
                raise RuntimeError("tracking batch group list overlaps or includes an unselected alias")
            group_by_alias[alias] = group_sha
            group_aliases.append(alias)
    if sorted(group_aliases, key=roster_index.__getitem__) != list(aliases):
        raise RuntimeError("tracking batch video groups do not exactly cover aliases in roster order")
    records = barrier.get("records")
    if not isinstance(records, list) or len(records) != len(aliases):
        raise RuntimeError("tracking batch barrier records must contain every selected alias")
    row_by_alias = {}
    required_case_fields = {"alias", "dataset", "request_path", "request_sha256", "request_bytes",
                            "tracking_binding_path", "tracking_binding_sha256", "tracking_binding_bytes",
                            "accepted_path", "accepted_sha256", "accepted_bytes", "inventory_path",
                            "inventory_sha256", "inventory_bytes", "tracking_cache_key",
                            "video_clock_group_sha256", "canonical_video_sha256",
                            "canonical_pixel_sha256", "frame_clock_sha256"}
    for row in records:
        if not isinstance(row, Mapping) or not required_case_fields <= set(row):
            raise RuntimeError("tracking per-alias barrier row has an unexpected schema")
        alias = row["alias"]
        if (alias not in aliases or alias in row_by_alias
                or row["video_clock_group_sha256"] != group_by_alias[alias]
                or row.get("dataset") != record_by_alias[alias]["dataset"]):
            raise RuntimeError("tracking per-alias row has a duplicate or wrong video group")
        row_by_alias[alias] = dict(row)
    if list(row_by_alias) != list(aliases):
        raise RuntimeError("tracking per-alias records must follow locked roster order")
    return path, barrier, row_by_alias


def _source_case_paths(alias: str, source_locked: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    entry = source_locked["source_files"].get(alias)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"source global barrier has no row for {alias}")
    path = _safe_project_file(entry.get("path"), expected_base=SOURCE_CASES)
    receipt_path = _safe_project_file(entry.get("receipt_path"), expected_base=SOURCE_CASES)
    if (path.stat().st_size != entry.get("bytes")
            or receipt_path.stat().st_size != entry.get("receipt_bytes")
            or _file_sha(path) != entry.get("sha256")
            or _file_sha(receipt_path) != entry.get("receipt_sha256")):
        raise RuntimeError(f"source payload or receipt bytes differ from FULL source barrier: {alias}")
    receipt = read(receipt_path)
    record = source_locked["record_by_alias"][alias]
    expected = {
        "schema": "stvg-final-table1-full-source-receipt-v1",
        "case_id": alias,
        "sha256": entry.get("sha256"),
        "bytes": entry.get("bytes"),
        "source_case_sha256": entry.get("source_case_sha256"),
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "query_key_sha256": record["query_key_sha256"],
        "GT_read": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"FULL source receipt binding differs for {alias}")
    return path, receipt_path, dict(entry)


def _tracking_case_paths(alias: str, row: Mapping[str, Any], source_locked: Mapping[str, Any],
                         tracking_runtime_sha256: str) -> tuple[Path, dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    folder = TRACKING_CASES / alias
    request_path = _safe_project_file(row["request_path"], expected_base=folder)
    binding_path = _safe_project_file(row["tracking_binding_path"], expected_base=folder)
    if (request_path != (folder / "REQUEST.json").resolve(strict=True)
            or binding_path != (folder / "TRACKING_BINDING.json").resolve(strict=True)):
        raise RuntimeError("tracking request/binding paths must use the locked per-alias FULL layout")
    if (_file_sha(request_path) != row["request_sha256"]
            or _file_sha(binding_path) != row["tracking_binding_sha256"]):
        raise RuntimeError(f"tracking request or binding bytes differ from the input batch barrier: {alias}")
    if (request_path.stat().st_size != row.get("request_bytes")
            or binding_path.stat().st_size != row.get("tracking_binding_bytes")):
        raise RuntimeError(f"tracking request or binding size differs from the input batch barrier: {alias}")
    request = read(request_path)
    binding = read(binding_path)
    if (binding.get("schema") != "stvg-final-table1-tracking-binding-v1"
            or binding.get("status") != "pass"
            or binding.get("case_alias") != alias
            or binding.get("request_path") != row["request_path"]
            or binding.get("request_sha256") != row["request_sha256"]
            or binding.get("dataset") != source_locked["record_by_alias"][alias]["dataset"]
            or binding.get("source_case_sha256") != source_locked["source_files"][alias]["source_case_sha256"]
            or binding.get("accepted_path") != row["accepted_path"]
            or binding.get("accepted_sha256") != row["accepted_sha256"]
            or binding.get("inventory_path") != row["inventory_path"]
            or binding.get("inventory_sha256") != row["inventory_sha256"]
            or binding.get("tracking_cache_key") != row["tracking_cache_key"]
            or binding.get("source_runtime_sha256") != source_locked["source_runtime_sha256"]
            or binding.get("source_capture_barrier_sha256") != source_locked["source_barrier_sha256"]
            or binding.get("tracking_runtime_sha256") != tracking_runtime_sha256
            or binding.get("preflight_roster_sha256") != source_locked["preflight_roster_sha256"]
            or binding.get("input_manifest_sha256") != source_locked["manifest_sha256"]
            or binding.get("query_key_sha256") != source_locked["record_by_alias"][alias]["query_key_sha256"]
            or binding.get("query_used_by_tracker") is not False
            or binding.get("GT_read") is not False):
        raise RuntimeError(f"tracking binding is stale or changed for {alias}")
    for key in ("request_sha256", "accepted_sha256", "inventory_sha256"):
        digest = row[key]
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"tracking barrier has an invalid {key}: {alias}")
    accepted_path = _safe_project_file(row["accepted_path"])
    inventory_path = _safe_project_file(row["inventory_path"])
    if (_file_sha(accepted_path) != row["accepted_sha256"]
            or _file_sha(inventory_path) != row["inventory_sha256"]):
        raise RuntimeError(f"accepted tracking or inventory bytes differ from barrier: {alias}")
    if (accepted_path.stat().st_size != row.get("accepted_bytes")
            or inventory_path.stat().st_size != row.get("inventory_bytes")):
        raise RuntimeError(f"accepted tracking or inventory size differs from barrier: {alias}")
    accepted = read(accepted_path)
    inventory = read(inventory_path)
    if (accepted.get("status") != "pass" or accepted.get("GT_read") is not False
            or accepted.get("query_independent") is not True
            or accepted.get("tracking_cache_key") != row["tracking_cache_key"]
            or accepted.get("mask_inventory_sha256") != row["inventory_sha256"]
            or accepted.get("tracking_binding_sha256") != binding.get("tracking_binding_sha256")
            or accepted.get("marker_video_path") != binding.get("marker_video_path")
            or accepted.get("marker_video_sha256") != binding.get("marker_video_sha256")
            or inventory.get("GT_read") is not False
            or inventory.get("schema") != "stvg-table1-query-independent-mask-inventory-v1"
            or inventory.get("query_independent") is not True
            or inventory.get("tracking_cache_key") != row["tracking_cache_key"]):
        raise RuntimeError(f"accepted tracking cache is not query-independent and GT-free: {alias}")
    _validate_mask_inventory(inventory, request_count=int(request.get("local_frame_count", -1)),
                              expected_dir=accepted.get("mask_data_dir"))
    marker_path = _safe_project_file(accepted.get("marker_video_path"))
    if _file_sha(marker_path) != accepted.get("marker_video_sha256"):
        raise RuntimeError(f"accepted marker video bytes changed: {alias}")
    native = _native_adapter()
    normalized = native._native().validate_request(request)
    request_canonical_sha = _file_sha(request_path)
    if (normalized.get("query_sha256") != binding.get("query_text_sha256")
            or normalized.get("source_video_sha256") != binding.get("source_video_sha256")
            or normalized.get("canonical_video_sha256") != binding.get("canonical_video_sha256")
            or normalized.get("canonical_pixel_sha256") != binding.get("canonical_pixel_sha256")
            or normalized.get("frame_clock_sha256") != binding.get("frame_clock_sha256")):
        raise RuntimeError(f"tracking request identity differs from binding: {alias}")
    roster_record = source_locked["record_by_alias"][alias]
    if normalized.get("source_video_sha256") != roster_record.get("video_sha256"):
        raise RuntimeError(f"tracking request video differs from the locked FULL source roster: {alias}")
    canonical_receipt_path = _safe_project_file(binding.get("canonical_receipt_path"))
    if _file_sha(canonical_receipt_path) != binding.get("canonical_receipt_sha256"):
        raise RuntimeError(f"canonical video receipt bytes changed: {alias}")
    canonical_receipt = read(canonical_receipt_path)
    video_path = Path(str(normalized.get("video_path", ""))).resolve(strict=True)
    if (not video_path.is_file() or _file_sha(video_path) != normalized.get("canonical_video_sha256")
            or canonical_receipt.get("schema") != "stvg-table1-canonical-input-v1"
            or canonical_receipt.get("status") != "complete"
            or canonical_receipt.get("declared_physical_bounds_half_open") != roster_record.get("video_bounds")
            or canonical_receipt.get("local_fps") != float(roster_record.get("fps"))
            or canonical_receipt.get("canonical_video_sha256") != normalized.get("canonical_video_sha256")
            or canonical_receipt.get("canonical_pixel_sha256") != normalized.get("canonical_pixel_sha256")
            or canonical_receipt.get("source_video_sha256") != normalized.get("source_video_sha256")
            or canonical_receipt.get("GT_read") is not False
            or canonical_receipt.get("query_independent") is not True):
        raise RuntimeError(f"canonical request media or receipt changed: {alias}")
    return request_path, request, binding, accepted_path, {
        "accepted": accepted, "mask_inventory": inventory, "inventory_path": inventory_path,
        "marker_video_path": marker_path, "canonical_receipt": canonical_receipt,
        "canonical_receipt_path": canonical_receipt_path, "normalized": normalized,
        "request_sha256": request_canonical_sha,
    }


def _validate_mask_inventory(inventory: Mapping[str, Any], *, request_count: int,
                             expected_dir: Any) -> None:
    files = inventory.get("files")
    known = inventory.get("known_ids")
    if (not isinstance(files, list) or inventory.get("mask_count") != len(files)
            or not isinstance(known, list) or known != sorted(set(known))
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in known)):
        raise RuntimeError("tracking mask inventory schema is malformed")
    if isinstance(request_count, bool) or request_count <= 0:
        raise RuntimeError("canonical request frame count is invalid")
    expected_mask_dir = _safe_project_file(expected_dir) if isinstance(expected_dir, str) else None
    if expected_mask_dir is None or not expected_mask_dir.is_dir():
        raise RuntimeError("accepted tracking receipt has no valid mask directory")
    indexes = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"index", "path", "sha256", "bytes"}:
            raise RuntimeError("tracking mask inventory entry has an unexpected schema")
        index = row["index"]
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or index >= request_count or index in indexes):
            raise RuntimeError("tracking mask index is invalid")
        indexes.add(index)
        path = _safe_project_file(row["path"])
        if (not path.is_file() or not path.is_relative_to(expected_mask_dir.resolve(strict=True))
                or path.stat().st_size != row["bytes"]
                or _file_sha(path) != row["sha256"]):
            raise RuntimeError("tracking mask file bytes differ from their inventory binding")
        digest = row["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise RuntimeError("tracking mask digest is malformed")


def _native_adapter():
    from scripts import run_stvg_table1_native_instance_adapter_v1 as native
    return native


def _load_source_payload(alias: str, source_locked: Mapping[str, Any], torch: Any) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    path, receipt_path, entry = _source_case_paths(alias, source_locked)
    case = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(case, dict):
        raise RuntimeError(f"FULL source case is not an object: {alias}")
    from scripts import run_stvg_table1_source_v1 as source
    record = source_locked["record_by_alias"][alias]
    receipt = read(receipt_path)
    expected = {
        "schema": "stvg-final-table1-source-case-v1",
        "case_id": alias,
        "dataset": record["dataset"],
        "query_ordinal": record["query_ordinal"],
        "query_key_sha256": record["query_key_sha256"],
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_case_sha256": entry["source_case_sha256"],
        "GT_read": False,
        "DINO_read_or_call": False,
        "optimizer_steps": 0,
        "backwards": 0,
    }
    if any(case.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"FULL source case identity/custody differs from its roster: {alias}")
    if source.content_sha256(case) != entry["source_case_sha256"]:
        raise RuntimeError(f"FULL source case canonical payload hash changed: {alias}")
    inp = case.get("input")
    if not isinstance(inp, Mapping):
        raise RuntimeError(f"FULL source case input metadata is missing: {alias}")
    for key in ("video_sha256", "fps", "width", "height", "frame_count"):
        if inp.get(key) != record.get(key):
            raise RuntimeError(f"FULL source case differs from locked input metadata ({key}): {alias}")
    if list(case.get("video_bounds", [])) != list(record.get("video_bounds", [])):
        raise RuntimeError(f"FULL source case physical bounds differ from roster: {alias}")
    if list(map(int, case.get("frame_ids", []))) != list(map(int, record.get("frame_ids", []))):
        raise RuntimeError(f"FULL source case source grid differs from roster: {alias}")
    if receipt.get("source_case_sha256") != source.content_sha256(case):
        raise RuntimeError(f"FULL source receipt and case canonical payload disagree: {alias}")
    return case, path, receipt_path, entry


def _tracking_row_matches(row: Mapping[str, Any], request_path: Path, request_sha256: str,
                          binding_path: Path, binding_sha256: str,
                          accepted_path: Path, accepted_sha256: str,
                          inventory_path: Path, inventory_sha256: str,
                          group_sha: str) -> bool:
    return all((
        row.get("request_path") == request_path.relative_to(ROOT).as_posix(),
        row.get("request_sha256") == request_sha256,
        row.get("tracking_binding_path") == binding_path.relative_to(ROOT).as_posix(),
        row.get("tracking_binding_sha256") == binding_sha256,
        row.get("accepted_path") == accepted_path.relative_to(ROOT).as_posix(),
        row.get("accepted_sha256") == accepted_sha256,
        row.get("inventory_path") == inventory_path.relative_to(ROOT).as_posix(),
        row.get("inventory_sha256") == inventory_sha256,
        row.get("video_clock_group_sha256") == group_sha,
    ))


def _existing_teacher_case(alias: str, *, runtime_sha: str, source_locked: Mapping[str, Any],
                           tracking_sha: str, tracking_row: Mapping[str, Any],
                           group_sha: str) -> dict[str, Any] | None:
    path = TEACHER_CASES / f"{alias}.pt"
    receipt_path = TEACHER_CASES / f"{alias}.receipt.json"
    if not path.exists() and not receipt_path.exists():
        started = ATTEMPTS / f"{alias}.started.json"
        completed = ATTEMPTS / f"{alias}.complete.json"
        if started.exists() or completed.exists():
            raise RuntimeError(f"teacher attempt marker exists without a sealed teacher output; preserve it: {alias}")
        return None
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"partial immutable teacher output exists; preserve it: {alias}")
    source_entry = source_locked["source_files"][alias]
    request_path = _safe_project_file(tracking_row["request_path"], expected_base=TRACKING_CASES / alias)
    binding_path = _safe_project_file(tracking_row["tracking_binding_path"], expected_base=TRACKING_CASES / alias)
    receipt = read(receipt_path)
    expected = {
        "schema": "stvg-final-table1-full-teacher-case-receipt-v1",
        "status": "sealed",
        "case_id": alias,
        "sha256": _file_sha(path),
        "bytes": path.stat().st_size,
        "source_sha256": source_entry["sha256"],
        "source_case_sha256": source_entry["source_case_sha256"],
        "source_receipt_sha256": source_entry["receipt_sha256"],
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "request_path": tracking_row["request_path"],
        "request_sha256": tracking_row["request_sha256"],
        "tracking_binding_path": tracking_row["tracking_binding_path"],
        "tracking_binding_sha256": tracking_row["tracking_binding_sha256"],
        "accepted_path": tracking_row["accepted_path"],
        "accepted_sha256": tracking_row["accepted_sha256"],
        "inventory_path": tracking_row["inventory_path"],
        "inventory_sha256": tracking_row["inventory_sha256"],
        "video_clock_group_sha256": group_sha,
        "tracking_runtime_sha256": tracking_sha,
        "teacher_runtime_sha256": runtime_sha,
        "GT_read": False,
        "query_reset": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"sealed teacher receipt is stale or corrupt: {alias}")
    started_path = ATTEMPTS / f"{alias}.started.json"
    complete_path = ATTEMPTS / f"{alias}.complete.json"
    if not started_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"sealed teacher output has no complete attempt custody: {alias}")
    started = read(started_path)
    complete = read(complete_path)
    if (started.get("case_id") != alias or started.get("teacher_runtime_sha256") != runtime_sha
            or complete.get("case_id") != alias or complete.get("teacher_output_sha256") != receipt["sha256"]
            or complete.get("teacher_receipt_sha256") != _file_sha(receipt_path)
            or complete.get("GT_read") is not False):
        raise RuntimeError(f"teacher attempt custody mismatch: {alias}")
    return {"alias": alias, "path": str(path.relative_to(ROOT)), "sha256": receipt["sha256"],
            "bytes": receipt["bytes"], "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": _file_sha(receipt_path), "reused": True,
            "actual_teacher_calls": int(receipt["actual_teacher_calls"]),
            "actual_temporal_teacher_calls": int(receipt["actual_temporal_teacher_calls"]),
            "actual_who_teacher_calls": int(receipt["actual_who_teacher_calls"])}


def _query_plan(alias: str, source_locked: Mapping[str, Any], tracking_row: Mapping[str, Any],
                group_sha: str, tracking_sha: str, torch: Any) -> dict[str, Any]:
    case, source_path, source_receipt_path, source_entry = _load_source_payload(alias, source_locked, torch)
    from scripts import run_stvg_table1_source_v1 as source
    record = source_locked["record_by_alias"][alias]
    _, official_row = source._validate_roster_record(record)
    query_text = official_row.get("input", {}).get("caption")
    if not isinstance(query_text, str) or not query_text.strip():
        raise RuntimeError(f"official query caption is absent for {alias}")
    req_path = _safe_project_file(tracking_row["request_path"], expected_base=TRACKING_CASES / alias)
    bind_path = _safe_project_file(tracking_row["tracking_binding_path"], expected_base=TRACKING_CASES / alias)
    request = read(req_path)
    binding = read(bind_path)
    if request.get("query_text") != query_text:
        raise RuntimeError(f"canonical request text does not match the sealed PLAN row: {alias}")
    if (binding.get("query_key_sha256") != record["query_key_sha256"]
            or binding.get("source_case_sha256") != source_entry["source_case_sha256"]):
        raise RuntimeError(f"tracking request does not bind the full roster query/source: {alias}")
    from scripts import run_stvg_table1_teacher_v1 as frozen
    temporal_ok, fallback_reason = frozen.temporal_clip_eligibility(case, request)
    needed: list[int] = []
    if temporal_ok:
        temporal = sys.modules.get("scripts.stvg_table1_temporal_adapter_v1")
        if temporal is None:
            from scripts import stvg_table1_temporal_adapter_v1 as temporal
        available = sorted(int(row["physical_frame_id"]) for row in request["frame_map"])
        for candidate in case["temporal_candidates"]:
            fids, _ = temporal._sample_candidate_ids(candidate, available)
            needed.extend(fids)
    request_group = {
        "source_video_sha256": request.get("source_video_sha256"),
        "canonical_video_sha256": request.get("canonical_video_sha256"),
        "canonical_pixel_sha256": request.get("canonical_pixel_sha256"),
        "frame_clock_sha256": binding.get("frame_clock_sha256"),
        "local_fps": request.get("local_fps"),
        "local_frame_count": request.get("local_frame_count"),
        "frame_map": request.get("frame_map"),
    }
    group_key = _sha_bytes(_canonical_json_bytes(request_group))
    result = {
        "alias": alias,
        "record": dict(record),
        "source_path": source_path,
        "source_receipt_path": source_receipt_path,
        "source_entry": source_entry,
        "request_path": req_path,
        "binding_path": bind_path,
        "request": request,
        "binding": binding,
        "temporal_valid": temporal_ok,
        "fallback_reason": fallback_reason,
        "wanted_physical_ids": sorted(set(needed)),
        "group_sha": group_sha,
        "request_group_key": group_key,
        "source_case_sha256": source_entry["source_case_sha256"],
        "width": int(case["input"]["width"]),
        "height": int(case["input"]["height"]),
    }
    del case
    return result


def _validate_batch_aliases(barrier: Mapping[str, Any], aliases: Sequence[str],
                            aliases_file_sha256: str | None) -> None:
    expected_sha = _canonical_alias_sha(aliases)
    if (barrier.get("schema") != FULL_TRACKING_BARRIER_SCHEMA
            or barrier.get("status") != "complete"
            or barrier.get("GT_read") is not False
            or barrier.get("count") != len(aliases)
            or barrier.get("selection_sha256") != expected_sha
            or barrier.get("aliases") != list(aliases)):
        raise RuntimeError("tracking barrier alias selection does not match this teacher batch")
    if aliases_file_sha256 is not None and barrier.get("submitted_alias_file_sha256") != aliases_file_sha256:
        raise RuntimeError("tracking barrier submitted-alias-file SHA does not match the requested file")


def _decode_group_rgb(plans: Sequence[Mapping[str, Any]], max_bytes: int) -> dict[int, Any]:
    """Decode one canonical video/clock group once for all temporal requests."""
    if not plans:
        return {}
    import cv2
    import numpy as np

    first = plans[0]
    req = first["request"]
    frame_map = req["frame_map"]
    physical_to_local = {int(row["physical_frame_id"]): int(row["local_frame_index"])
                         for row in frame_map}
    all_wanted = sorted({int(fid) for plan in plans for fid in plan["wanted_physical_ids"]})
    if not all_wanted:
        return {}
    if any(plan["request_group_key"] != first["request_group_key"] for plan in plans):
        raise RuntimeError("video cache group contains different canonical clocks or pixel bindings")
    estimate = len(all_wanted) * first["width"] * first["height"] * 3
    if estimate > max_bytes:
        raise RuntimeError("video/clock group exceeds locked RGB cache cap; choose a smaller complete-group batch")
    targets = {physical_to_local[fid]: fid for fid in all_wanted if fid in physical_to_local}
    if len(targets) != len(all_wanted):
        raise RuntimeError("temporal sample frame is absent from canonical request frame_map")
    path = Path(req["video_path"]).resolve(strict=True)
    if _file_sha(path) != req["canonical_video_sha256"]:
        raise RuntimeError("canonical video bytes differ from the tracking request")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the canonical FFV1 video")
    rgb: dict[int, Any] = {}
    local = 0
    try:
        while targets:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("canonical video ended before all locked sampled frames were decoded")
            if local in targets:
                physical = targets.pop(local)
                if bgr.shape[:2] != (first["height"], first["width"]):
                    raise RuntimeError("canonical decoded frame dimensions differ from full source metadata")
                rgb[physical] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            local += 1
    finally:
        cap.release()
    actual_bytes = sum(int(array.nbytes) for array in rgb.values())
    if actual_bytes > max_bytes or set(rgb) != set(all_wanted):
        raise RuntimeError("canonical RGB group cache violates its locked size or coverage")
    return rgb


def _save_tensor_immutable(path: Path, value: Any, torch: Any) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable teacher output: {path.name}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)
    return _file_sha(path), path.stat().st_size


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable JSON already exists with different bytes: {path.name}")
        return _sha_bytes(payload)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)
    return _sha_bytes(payload)


def _write_attempt_start(alias: str, plan: Mapping[str, Any], *, runtime_sha: str,
                          expected_calls: int) -> None:
    start_path = ATTEMPTS / f"{alias}.started.json"
    complete_path = ATTEMPTS / f"{alias}.complete.json"
    if start_path.exists() or complete_path.exists():
        raise RuntimeError(f"a teacher attempt already exists for {alias}; refusing duplicate generation")
    _write_immutable_json(start_path, {
        "schema": "stvg-final-table1-full-teacher-attempt-v1",
        "status": "started",
        "case_id": alias,
        "expected_teacher_calls": expected_calls,
        "teacher_runtime_sha256": runtime_sha,
        "source_sha256": _file_sha(plan["source_path"]),
        "source_case_sha256": plan["source_case_sha256"],
        "request_sha256": _file_sha(plan["request_path"]),
        "tracking_binding_sha256": _file_sha(plan["binding_path"]),
        "GT_read": False,
        "time": time.time(),
    })


def _write_attempt_complete(alias: str, output_sha: str, receipt_sha: str,
                            temporal_calls: int, who_calls: int) -> None:
    _write_immutable_json(ATTEMPTS / f"{alias}.complete.json", {
        "schema": "stvg-final-table1-full-teacher-attempt-v1",
        "status": "complete",
        "case_id": alias,
        "teacher_output_sha256": output_sha,
        "teacher_receipt_sha256": receipt_sha,
        "temporal_teacher_calls": temporal_calls,
        "who_teacher_calls": who_calls,
        "actual_teacher_calls": temporal_calls + who_calls,
        "GT_read": False,
        "time": time.time(),
    })


def _prepare_tracking_objects(alias: str, tracking_row: Mapping[str, Any], source_locked: Mapping[str, Any],
                              tracking_runtime_sha: str, group_sha: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_path, request, binding, _, tracking = _tracking_case_paths(
        alias, tracking_row, source_locked, tracking_runtime_sha)
    if tracking_row.get("video_clock_group_sha256") != group_sha:
        raise RuntimeError(f"tracking batch row group hash differs from the locked batch: {alias}")
    if not _tracking_row_matches(
            tracking_row, request_path, _file_sha(request_path),
            TRACKING_CASES / alias / "TRACKING_BINDING.json", _file_sha(TRACKING_CASES / alias / "TRACKING_BINDING.json"),
            _safe_project_file(tracking_row["accepted_path"]), tracking_row["accepted_sha256"],
            tracking["inventory_path"], tracking_row["inventory_sha256"], group_sha):
        raise RuntimeError(f"tracking batch row differs from actual files: {alias}")
    return request_path, request, binding, tracking


def _execute_query(plan: Mapping[str, Any], source_locked: Mapping[str, Any], tracking_row: Mapping[str, Any],
                   group_sha: str, tracking_runtime_sha: str, *, model: Any, who_processor: Any,
                   temporal_processor: Any, torch: Any, device: str, rgb_cache: Mapping[int, Any],
                   runtime_sha: str) -> dict[str, Any]:
    alias = str(plan["alias"])
    case, source_path, source_receipt_path, source_entry = _load_source_payload(alias, source_locked, torch)
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import stvg_table1_temporal_adapter_v1 as temporal
    from scripts import run_stvg_table1_native_instance_adapter_v1 as native
    from scripts import stvg_table1_spatial_adapter_v1 as spatial

    request_path, request, binding, _, tracking = _prepare_tracking_objects(
        alias, tracking_row, source_locked, tracking_runtime_sha, group_sha)
    _, official_row = source._validate_roster_record(source_locked["record_by_alias"][alias])
    query = official_row["input"]["caption"]
    if request.get("query_text") != query:
        raise RuntimeError(f"request query no longer matches locked official caption: {alias}")
    np = __import__("numpy")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    temporal_calls = 0
    if plan["temporal_valid"]:
        _write_attempt_start(alias, plan, runtime_sha=runtime_sha, expected_calls=9)
        result = temporal.adapt_hot_query(
            case, query, case["temporal_candidates"], model=model,
            processor=temporal_processor, device=device,
            decoded_rgb_by_physical_id=rgb_cache, frame_map=request["frame_map"],
        )
        temporal_calls = 8
    else:
        _write_attempt_start(alias, plan, runtime_sha=runtime_sha, expected_calls=1)
        result = dict(
            status="unavailable_temporal_input_native_fallback",
            reason=case["candidate_validity"],
            request_clock_fallback_reason=plan["fallback_reason"],
            final_interval=case["Native_interval"],
            GT_read=False,
            optimizer_steps=0,
        )
    who = native.infer_who(request, tracking, who_processor, model, torch, device)
    who_calls = 1
    selected = who["who"].get("candidate_id") if who["who"].get("status") == "valid" else None
    reference = spatial.build_reference(case, result["final_interval"], request,
                                        selected, tracking["mask_inventory"])
    value = {
        "case_id": alias,
        "dataset": case["dataset"],
        "source_sha256": _file_sha(source_path),
        "source_case_sha256": source_entry["source_case_sha256"],
        "source_receipt_sha256": _file_sha(source_receipt_path),
        "request_sha256": _file_sha(request_path),
        "tracking_binding_sha256": _file_sha(TRACKING_CASES / alias / "TRACKING_BINDING.json"),
        "accepted_sha256": tracking_row["accepted_sha256"],
        "inventory_sha256": tracking_row["inventory_sha256"],
        "video_clock_group_sha256": group_sha,
        "temporal": result,
        "who": who,
        "reference": reference,
        "GT_read": False,
        "query_reset": True,
        "actual_temporal_teacher_calls": temporal_calls,
        "actual_who_teacher_calls": who_calls,
        "actual_teacher_calls": temporal_calls + who_calls,
        "wall_seconds": float(time.perf_counter() - plan["started_at"]),
    }
    dst = TEACHER_CASES / f"{alias}.pt"
    receipt_path = TEACHER_CASES / f"{alias}.receipt.json"
    output_sha, output_bytes = _save_tensor_immutable(dst, value, torch)
    receipt = {
        "schema": "stvg-final-table1-full-teacher-case-receipt-v1",
        "status": "sealed",
        "case_id": alias,
        "path": str(dst.relative_to(ROOT)),
        "sha256": output_sha,
        "bytes": output_bytes,
        "source_sha256": _file_sha(source_path),
        "source_case_sha256": source_entry["source_case_sha256"],
        "source_receipt_sha256": _file_sha(source_receipt_path),
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "request_path": str(request_path.relative_to(ROOT)),
        "request_sha256": _file_sha(request_path),
        "tracking_binding_path": str((TRACKING_CASES / alias / "TRACKING_BINDING.json").relative_to(ROOT)),
        "tracking_binding_sha256": _file_sha(TRACKING_CASES / alias / "TRACKING_BINDING.json"),
        "accepted_path": tracking_row["accepted_path"],
        "accepted_sha256": tracking_row["accepted_sha256"],
        "inventory_path": tracking_row["inventory_path"],
        "inventory_sha256": tracking_row["inventory_sha256"],
        "video_clock_group_sha256": group_sha,
        "tracking_runtime_sha256": tracking_runtime_sha,
        "teacher_runtime_sha256": runtime_sha,
        "actual_temporal_teacher_calls": temporal_calls,
        "actual_who_teacher_calls": who_calls,
        "actual_teacher_calls": temporal_calls + who_calls,
        "GT_read": False,
        "query_reset": True,
        "time": time.time(),
    }
    _write_immutable_json(receipt_path, receipt)
    receipt_sha = _file_sha(receipt_path)
    _write_attempt_complete(alias, output_sha, receipt_sha, temporal_calls, who_calls)
    del value, case, result, who, reference, request, tracking
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return {
        "alias": alias,
        "path": str(dst.relative_to(ROOT)),
        "sha256": output_sha,
        "bytes": output_bytes,
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": receipt_sha,
        "reused": False,
        "actual_temporal_teacher_calls": temporal_calls,
        "actual_who_teacher_calls": who_calls,
        "actual_teacher_calls": temporal_calls + who_calls,
    }


def _make_video_groups(plans: Sequence[dict[str, Any]], max_bytes: int) -> list[list[dict[str, Any]]]:
    """Return ordered complete video/clock groups, never mixing frame clocks."""
    groups: dict[str, list[dict[str, Any]]] = collections.OrderedDict()
    for plan in plans:
        groups.setdefault(plan["group_sha"], []).append(plan)
    result = []
    for group_sha, rows in groups.items():
        clock_keys = {row["request_group_key"] for row in rows}
        if len(clock_keys) != 1:
            raise RuntimeError("one locked video/clock identity produced inconsistent canonical requests")
        union = {fid for row in rows if row["temporal_valid"] for fid in row["wanted_physical_ids"]}
        estimate = len(union) * rows[0]["width"] * rows[0]["height"] * 3
        if estimate > max_bytes:
            raise RuntimeError("complete video/clock group exceeds the pinned frame-cache cap")
        result.append(rows)
    return result


def _write_batch_barrier(path: Path, barrier: Mapping[str, Any]) -> str:
    return _write_immutable_json(path, barrier)


def _validate_batch_output(alias: str, output_row: Mapping[str, Any]) -> None:
    path = _safe_project_file(output_row["path"], expected_base=TEACHER_CASES)
    receipt_path = _safe_project_file(output_row["receipt_path"], expected_base=TEACHER_CASES)
    if (path.stat().st_size != output_row.get("bytes") or _file_sha(path) != output_row.get("sha256")
            or _file_sha(receipt_path) != output_row.get("receipt_sha256")):
        raise RuntimeError(f"teacher output changed before batch seal: {alias}")


def _verify_final_batch_file(path: Path, source_locked: Mapping[str, Any], runtime_sha: str,
                             tracking_runtime_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    barrier = read(path)
    if (barrier.get("schema") != TEACHER_BATCH_BARRIER_SCHEMA
            or barrier.get("status") not in {"complete", "sealed"}
            or barrier.get("GT_read") is not False
            or barrier.get("teacher_runtime_sha256") != runtime_sha
            or barrier.get("tracking_runtime_sha256") != tracking_runtime_sha
            or barrier.get("full_source_roster_sha256") != source_locked["roster_sha256"]
            or barrier.get("source_capture_barrier_sha256") != source_locked["source_barrier_sha256"]):
        raise RuntimeError(f"teacher batch barrier is stale: {path.name}")
    files = barrier.get("files")
    aliases = barrier.get("aliases")
    if (not isinstance(files, list) or not isinstance(aliases, list)
            or len(files) != barrier.get("count") or len(files) != len(aliases)):
        raise RuntimeError(f"teacher batch barrier has inconsistent file count: {path.name}")
    if aliases != sorted(aliases, key=lambda a: source_locked["aliases"].index(a)):
        raise RuntimeError(f"teacher batch aliases are not in roster order: {path.name}")
    seen = set()
    rows = []
    for item in files:
        if not isinstance(item, Mapping) or not {"alias", "path", "sha256", "bytes", "receipt_path", "receipt_sha256"} <= set(item):
            raise RuntimeError("teacher batch file row has an unexpected schema")
        alias = item["alias"]
        if alias not in aliases or alias in seen:
            raise RuntimeError("teacher batch contains a duplicate or unselected alias")
        seen.add(alias)
        output_path = _safe_project_file(item["path"], expected_base=TEACHER_CASES)
        receipt_path = _safe_project_file(item["receipt_path"], expected_base=TEACHER_CASES)
        receipt = read(receipt_path)
        if (output_path.stat().st_size != item["bytes"] or _file_sha(output_path) != item["sha256"]
                or _file_sha(receipt_path) != item["receipt_sha256"]
                or receipt.get("case_id") != alias or receipt.get("sha256") != item["sha256"]
                or receipt.get("teacher_runtime_sha256") != runtime_sha
                or receipt.get("GT_read") is not False):
            raise RuntimeError(f"teacher batch file is not backed by an immutable per-query receipt: {alias}")
        rows.append(dict(item))
    if [item["alias"] for item in files] != aliases:
        raise RuntimeError("teacher batch file order differs from aliases order")
    tracking_barrier_path = _safe_project_file(barrier.get("tracking_batch_barrier_path"), expected_base=FULL)
    if _file_sha(tracking_barrier_path) != barrier.get("tracking_batch_barrier_sha256"):
        raise RuntimeError("teacher batch barrier's tracking input lock bytes changed")
    aliases_file_sha = barrier.get("aliases_file_sha256")
    checked_path, checked_tracking_barrier, _ = _load_batch_tracking_barrier(
        str(tracking_barrier_path), aliases, aliases_file_sha,
        source_locked, tracking_runtime_sha)
    if (checked_path != tracking_barrier_path
            or checked_tracking_barrier.get("selection_sha256") != barrier.get("selection_sha256")):
        raise RuntimeError("teacher batch barrier is detached from its exact tracking batch")
    return barrier, rows


def finalize() -> dict[str, Any]:
    """CPU-only full seal after every locked query has one immutable teacher result."""
    source_locked = _verify_full_source_authority()
    runtime = _verify_teacher_runtime(source_locked)
    tracking_sha = _verify_tracking_runtime(source_locked)
    runtime_sha = _file_sha(TEACHER_RUNTIME)
    existing_batches = sorted(BATCH_BARRIERS.glob("TEACHER_BATCH_*_BARRIER.json"))
    batch_info = []
    alias_rows: dict[str, dict[str, Any]] = {}
    model_loads = 0
    for path in existing_batches:
        barrier, rows = _verify_final_batch_file(path, source_locked, runtime_sha, tracking_sha)
        model_loads += int(barrier.get("actual_model_loads", 0))
        batch_info.append({"path": str(path.relative_to(ROOT)), "sha256": _file_sha(path),
                           "count": len(rows), "selection_sha256": barrier.get("selection_sha256")})
        for row in rows:
            alias = row["alias"]
            prior = alias_rows.get(alias)
            if prior is not None and prior["sha256"] != row["sha256"]:
                raise RuntimeError(f"different teacher outputs are bound for the same query: {alias}")
            alias_rows[alias] = row
    missing = [alias for alias in source_locked["aliases"] if alias not in alias_rows]
    if missing:
        raise RuntimeError(f"full teacher seal is incomplete: {len(missing)} aliases have no sealed batch receipt")
    files = [alias_rows[alias] for alias in source_locked["aliases"]]
    calls = temporal_calls = who_calls = 0
    for alias, row in zip(source_locked["aliases"], files):
        receipt = read(_safe_project_file(row["receipt_path"], expected_base=TEACHER_CASES))
        if (receipt.get("source_runtime_sha256") != source_locked["source_runtime_sha256"]
                or receipt.get("source_capture_barrier_sha256") != source_locked["source_barrier_sha256"]
                or receipt.get("full_source_roster_sha256") != source_locked["roster_sha256"]
                or receipt.get("input_manifest_sha256") != source_locked["manifest_sha256"]):
            raise RuntimeError(f"teacher receipt is detached from full source authority: {alias}")
        calls += int(receipt["actual_teacher_calls"])
        temporal_calls += int(receipt["actual_temporal_teacher_calls"])
        who_calls += int(receipt["actual_who_teacher_calls"])
    if calls != temporal_calls + who_calls or who_calls != TOTAL_CASES:
        raise RuntimeError("unique per-query teacher call totals are inconsistent")
    barrier = {
        "schema": TEACHER_GLOBAL_BARRIER_SCHEMA,
        "status": "sealed",
        "count": TOTAL_CASES,
        "files": files,
        "batch_barriers": batch_info,
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "tracking_runtime_sha256": tracking_sha,
        "teacher_runtime_sha256": runtime_sha,
        "actual_model_loads_across_batch_runs": model_loads,
        "actual_teacher_calls_unique_queries": calls,
        "actual_temporal_teacher_calls_unique_queries": temporal_calls,
        "actual_who_teacher_calls_unique_queries": who_calls,
        "GT_read": False,
        "finalized_cpu_only": True,
        "time": time.time(),
    }
    if GLOBAL_BARRIER.exists():
        prior = read(GLOBAL_BARRIER)
        if prior != barrier:
            # Timestamp differences are not a scientific change; preserve the
            # existing seal and verify all source/row hashes instead.
            stable = {k: v for k, v in barrier.items() if k != "time"}
            prior_stable = {k: v for k, v in prior.items() if k != "time"}
            if prior_stable != stable:
                raise RuntimeError("existing full teacher global barrier differs; preserve it")
        return {"status": "already_sealed", "count": TOTAL_CASES,
                "barrier_sha256": _file_sha(GLOBAL_BARRIER), "GT_read": False}
    digest = _write_immutable_json(GLOBAL_BARRIER, barrier)
    return {"status": "sealed", "count": TOTAL_CASES,
            "barrier_sha256": digest, "GT_read": False,
            "actual_teacher_calls": calls, "actual_model_loads_across_batches": model_loads}


def run_batch(*, aliases_file: str | None, all_mode: bool, limit: int | None,
              tracking_barrier_arg: str) -> dict[str, Any]:
    source_locked = _verify_full_source_authority()
    runtime = _verify_teacher_runtime(source_locked)
    tracking_sha = _verify_tracking_runtime(source_locked)
    runtime_sha = _file_sha(TEACHER_RUNTIME)
    if GLOBAL_BARRIER.exists():
        raise RuntimeError("FULL teacher global barrier is already sealed; inference is closed")
    if aliases_file is None:
        alias_bytes = None
        aliases_file_sha = None
    else:
        alias_path = Path(aliases_file).resolve(strict=True)
        if not alias_path.is_file():
            raise RuntimeError("aliases-file must be a regular file")
        alias_bytes = alias_path.read_bytes()
        aliases_file_sha = _sha_bytes(alias_bytes)
    aliases, verified_alias_file_sha = select_aliases(
        source_locked["records"], aliases_file_bytes=alias_bytes,
        all_mode=all_mode, limit=limit)
    if verified_alias_file_sha != aliases_file_sha:
        raise RuntimeError("aliases-file digest changed during selection")
    tracking_path, tracking_barrier, tracking_rows = _load_batch_tracking_barrier(
        tracking_barrier_arg, aliases, aliases_file_sha,
        source_locked, tracking_sha)
    tracking_barrier_sha = _file_sha(tracking_path)
    selection_sha = _canonical_alias_sha(aliases)
    batch_path = BATCH_BARRIERS / f"TEACHER_BATCH_{selection_sha}_BARRIER.json"
    if batch_path.is_file():
        prior_barrier, prior_rows = _verify_final_batch_file(
            batch_path, source_locked, runtime_sha, tracking_sha)
        if (prior_barrier.get("aliases") != list(aliases)
                or prior_barrier.get("aliases_file_sha256") != aliases_file_sha
                or prior_barrier.get("tracking_batch_barrier_path") != str(tracking_path.relative_to(ROOT))
                or prior_barrier.get("tracking_batch_barrier_sha256") != tracking_barrier_sha):
            raise RuntimeError("an existing teacher batch seal does not match this exact input batch")
        return {
            "status": "already_complete",
            "count": len(aliases),
            "new_case_count": 0,
            "reused_case_count": len(prior_rows),
            "actual_teacher_calls_new_cases": 0,
            "actual_model_loads": 0,
            "batch_barrier_path": str(batch_path.relative_to(ROOT)),
            "batch_barrier_sha256": _file_sha(batch_path),
            "GT_read": False,
        }

    # Validate each selected immutable source and tracking input before loading
    # the model.  Plans retain only one group of image IDs plus request metadata.
    import torch
    plans = []
    existing_rows = {}
    for alias in aliases:
        row = tracking_rows[alias]
        group_sha = row["video_clock_group_sha256"]
        old = _existing_teacher_case(alias, runtime_sha=runtime_sha,
                                     source_locked=source_locked,
                                     tracking_sha=tracking_sha,
                                     tracking_row=row, group_sha=group_sha)
        if old is not None:
            existing_rows[alias] = old
            continue
        source_path, source_receipt_path, source_entry = _source_case_paths(alias, source_locked)
        req_path, request, binding, _, track = _tracking_case_paths(alias, row,
                                                                    source_locked,
                                                                    tracking_sha)
        # Load and independently validate full source tensors on CPU for the
        # original candidate/clock fallback decision; no GT/model is touched.
        case, _, _, _ = _load_source_payload(alias, source_locked, torch)
        from scripts import run_stvg_table1_source_v1 as source
        _, official_row = source._validate_roster_record(source_locked["record_by_alias"][alias])
        if request.get("query_text") != official_row.get("input", {}).get("caption"):
            raise RuntimeError(f"tracking request does not preserve official query text: {alias}")
        from scripts import run_stvg_table1_teacher_v1 as frozen
        valid, fallback_reason = frozen.temporal_clip_eligibility(case, request)
        needed = []
        if valid:
            from scripts import stvg_table1_temporal_adapter_v1 as temporal
            available = sorted(int(x["physical_frame_id"]) for x in request["frame_map"])
            for candidate in case["temporal_candidates"]:
                fids, _ = temporal._sample_candidate_ids(candidate, available)
                needed.extend(fids)
        normalized = track["normalized"]
        request_group = {
            "source_video_sha256": normalized["source_video_sha256"],
            "canonical_video_sha256": normalized["canonical_video_sha256"],
            "canonical_pixel_sha256": normalized["canonical_pixel_sha256"],
            "frame_clock_sha256": binding["frame_clock_sha256"],
            "local_fps": normalized["local_fps"],
            "local_frame_count": normalized["local_frame_count"],
            "frame_map": normalized["frame_map"],
        }
        plans.append({
            "alias": alias,
            "record": dict(source_locked["record_by_alias"][alias]),
            "source_path": source_path,
            "source_receipt_path": source_receipt_path,
            "source_entry": source_entry,
            "request_path": req_path,
            "binding_path": TRACKING_CASES / alias / "TRACKING_BINDING.json",
            "request": request,
            "binding": binding,
            "tracking_row": row,
            "tracking_objects": track,
            "temporal_valid": valid,
            "fallback_reason": fallback_reason,
            "wanted_physical_ids": sorted(set(needed)),
            "group_sha": group_sha,
            "request_group_key": _sha_bytes(_canonical_json_bytes(request_group)),
            "source_case_sha256": source_entry["source_case_sha256"],
            "width": int(case["input"]["width"]),
            "height": int(case["input"]["height"]),
            "started_at": time.perf_counter(),
        })
        del case

    barrier_groups = {g["video_clock_group_sha256"]: g for g in tracking_barrier["video_clock_groups"]}
    expected_group_set = {row["video_clock_group_sha256"] for row in tracking_rows.values()}
    if expected_group_set != set(barrier_groups):
        raise RuntimeError("tracking barrier group declarations differ from per-query rows")
    if any(p["group_sha"] not in expected_group_set for p in plans):
        raise RuntimeError("teacher work plan references a non-barrier video clock group")

    execution = runtime["execution"]
    max_cache_bytes = int(execution["max_video_group_cache_bytes"])
    work_groups = _make_video_groups(plans, max_cache_bytes)
    # Model loading happens after every selected source, tracking row, and
    # clock/group binding passed CPU preflight. All queries then share one hot
    # model instance for this invocation.
    model_loaded = False
    model = who_processor = temporal_processor = torch_obj = device = mapping = None
    load_s = 0.0
    peak = 0
    new_rows: dict[str, dict[str, Any]] = {}
    total_calls = temporal_calls = who_calls = 0
    started = time.perf_counter()
    if plans:
        from transformers import AutoProcessor
        from scripts import run_stvg_native_instance_teacher_storage002 as teacher
        teacher.verify_local_model_receipt()
        teacher.MAX_MEMORY = {0: "22GiB", "cpu": "4GiB"}
        who_processor, model, torch_obj, device, mapping, load_s, peak = teacher._load_qwen(
            ROOT / teacher.CHECKPOINT_REL, FULL / "teacher_offload")
        temporal_processor = AutoProcessor.from_pretrained(
            str(ROOT / teacher.CHECKPOINT_REL), local_files_only=True, trust_remote_code=False)
        model.generation_config.use_cache = True
        model_loaded = True
        # The entire selected alias list is a roster-ordered union of complete
        # clock groups.  Decode each group union once, then call all its query
        # teachers serially with the same resident model.
        for group_plans in work_groups:
            temporal_plans = [p for p in group_plans if p["temporal_valid"]]
            rgb_cache = _decode_group_rgb(temporal_plans, max_cache_bytes) if temporal_plans else {}
            for plan in group_plans:
                alias = plan["alias"]
                result_row = _execute_query(
                    plan, source_locked, plan["tracking_row"], plan["group_sha"], tracking_sha,
                    model=model, who_processor=who_processor,
                    temporal_processor=temporal_processor, torch=torch_obj, device=device,
                    rgb_cache=rgb_cache, runtime_sha=runtime_sha)
                new_rows[alias] = result_row
                calls = int(result_row["actual_teacher_calls"])
                t_calls = int(result_row["actual_temporal_teacher_calls"])
                w_calls = int(result_row["actual_who_teacher_calls"])
                total_calls += calls; temporal_calls += t_calls; who_calls += w_calls
                write(STATUS, {
                    "schema": "stvg-final-table1-full-teacher-progress-v1",
                    "status": "running",
                    "selected_count": len(aliases),
                    "selected_complete_count": len(existing_rows) + len(new_rows),
                    "new_complete_count": len(new_rows),
                    "actual_teacher_calls_this_invocation": total_calls,
                    "GT_read": False,
                    "time": time.time(),
                }, mutable=True)
                print(f"TABLE1_FULL_TEACHER {len(existing_rows)+len(new_rows)}/{len(aliases)} {alias} calls={calls}", flush=True)
            del rgb_cache
            gc.collect()
            if torch_obj.cuda.is_initialized():
                torch_obj.cuda.empty_cache()

    combined_rows = []
    for alias in aliases:
        row = new_rows.get(alias) or existing_rows.get(alias)
        if row is None:
            raise RuntimeError(f"batch output is incomplete after inference: {alias}")
        _validate_batch_output(alias, row)
        combined_rows.append({key: row[key] for key in (
            "alias", "path", "sha256", "bytes", "receipt_path", "receipt_sha256")})
    tracking_barrier_rel = str(tracking_path.relative_to(ROOT))
    batch_barrier = {
        "schema": TEACHER_BATCH_BARRIER_SCHEMA,
        "status": "complete",
        "count": len(aliases),
        "aliases": list(aliases),
        "selection_sha256": selection_sha,
        "aliases_file_sha256": aliases_file_sha,
        "video_clock_groups": tracking_barrier["video_clock_groups"],
        "tracking_batch_barrier_path": tracking_barrier_rel,
        "tracking_batch_barrier_sha256": tracking_barrier_sha,
        "full_source_roster_sha256": source_locked["roster_sha256"],
        "input_manifest_sha256": source_locked["manifest_sha256"],
        "source_runtime_sha256": source_locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": source_locked["source_barrier_sha256"],
        "tracking_runtime_sha256": tracking_sha,
        "teacher_runtime_sha256": runtime_sha,
        "files": combined_rows,
        "actual_model_loads": int(model_loaded),
        "actual_teacher_calls_new_cases": total_calls,
        "actual_temporal_teacher_calls_new_cases": temporal_calls,
        "actual_who_teacher_calls_new_cases": who_calls,
        "new_case_count": len(new_rows),
        "reused_case_count": len(existing_rows),
        "model_load_seconds": float(load_s) if model_loaded else 0.0,
        "peak_gpu_allocated_bytes": int(peak) if model_loaded else 0,
        "actual_device_map": mapping if model_loaded else None,
        "wall_seconds": float(time.perf_counter() - started),
        "GT_read": False,
        "query_reset_seed": SEED,
    }
    batch_sha = _write_batch_barrier(batch_path, batch_barrier)
    return {
        "status": "complete",
        "count": len(aliases),
        "new_case_count": len(new_rows),
        "reused_case_count": len(existing_rows),
        "actual_teacher_calls_new_cases": total_calls,
        "actual_model_loads": int(model_loaded),
        "batch_barrier_path": str(batch_path.relative_to(ROOT)),
        "batch_barrier_sha256": batch_sha,
        "GT_read": False,
    }


def contracts() -> dict[str, Any]:
    checks = 0
    rows = []
    for ds, prefix, count in (("hc2", "HC", 4), ("vidstg", "V", 5)):
        for i in range(count):
            rows.append({
                "dataset": ds, "probe_alias": f"{prefix}{i:05d}", "query_ordinal": i,
                "video_sha256": hashlib.sha256(f"video-{ds}-{i % 2}".encode()).hexdigest(),
                "fps": 25.0, "video_bounds": [0, 40], "width": 640, "height": 360,
                "GT_read": False,
            })
    small_counts = {"hc2": 4, "vidstg": 5}
    aliases, _, by_alias, group_members = _roster_aliases(
        rows, expected_domain_counts=small_counts)
    assert len(aliases) == 9
    checks += 1
    group = video_clock_group_sha256(rows[0])
    assert group == video_clock_group_sha256(rows[2])
    assert group != video_clock_group_sha256(rows[1])
    checks += 1
    batch = [rows[0]["probe_alias"], rows[2]["probe_alias"]]
    descriptors = _group_descriptors(batch, group_members, by_alias)
    raw = _canonical_json_bytes({"schema": ALIAS_FILE_SCHEMA, "aliases": batch,
                                 "video_clock_groups": descriptors})
    selected, file_sha = select_aliases(rows, aliases_file_bytes=raw, limit=2,
                                        expected_domain_counts=small_counts)
    assert selected == batch and file_sha == _sha_bytes(raw)
    checks += 1
    for bad in (
        _canonical_json_bytes({"schema": ALIAS_FILE_SCHEMA, "aliases": [batch[0], batch[0]]}),
        _canonical_json_bytes({"schema": ALIAS_FILE_SCHEMA, "aliases": [batch[0]]}),
    ):
        try:
            select_aliases(rows, aliases_file_bytes=bad, expected_domain_counts=small_counts)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("duplicate or split video group was accepted")
    union = [f"HC{i:05d}" for i in range(4)]
    union_desc = _group_descriptors(union, group_members, by_alias)
    union_raw = _canonical_json_bytes({"schema": ALIAS_FILE_SCHEMA, "aliases": union,
                                       "video_clock_groups": union_desc})
    assert select_aliases(rows, aliases_file_bytes=union_raw, limit=4,
                          expected_domain_counts=small_counts)[0] == union
    checks += 1
    try:
        select_aliases(rows, aliases_file_bytes=raw, limit=1,
                       expected_domain_counts=small_counts)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("--limit silently truncated a complete group")
    try:
        select_aliases(rows, aliases_file_bytes=raw, all_mode=True,
                       expected_domain_counts=small_counts)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("alias-file and --all selectors were accepted together")
    assert len(batch_call_counts(True, reused=False)) == 3
    assert batch_call_counts(True, reused=False) == (8, 1, 9)
    assert batch_call_counts(False, reused=False) == (0, 1, 1)
    assert batch_call_counts(True, reused=True) == (0, 0, 0)
    checks += 1
    return {
        "schema": "stvg-final-table1-full-teacher-cpu-contracts-v1",
        "status": "pass",
        "checks": checks,
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "scope": ["alias-file ABI", "roster order", "complete video/clock groups",
                  "hard batch cap", "stable group identity", "no duplicate call accounting"],
        "contract_sha256": _file_sha(Path(__file__)),
    }


def batch_call_counts(temporal_valid: bool, *, reused: bool) -> tuple[int, int, int]:
    if reused:
        return 0, 0, 0
    temporal = 8 if temporal_valid else 0
    who = 1
    return temporal, who, temporal + who


def write_contracts() -> dict[str, Any]:
    result = contracts()
    CONTRACTS.parent.mkdir(parents=True, exist_ok=True)
    if CONTRACTS.exists():
        prior = read(CONTRACTS)
        if prior != result:
            raise RuntimeError("existing FULL teacher CPU contracts differ; preserve them")
    else:
        write(CONTRACTS, result)
    result["path"] = str(CONTRACTS.relative_to(ROOT))
    result["sha256"] = _file_sha(CONTRACTS)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--aliases-file", help="canonical JSON alias batch locked by FULL tracking")
    selector.add_argument("--all", action="store_true", help="all 13,785 aliases; requires full tracking input barrier")
    parser.add_argument("--limit", type=int, help="hard cap; refuses to truncate video/clock groups")
    parser.add_argument("--tracking-barrier", help="root-relative FULL tracking batch barrier path")
    parser.add_argument("--finalize", action="store_true", help="CPU-only full 13,785-case teacher seal")
    parser.add_argument("--contracts", action="store_true", help="run deterministic synthetic CPU contracts only")
    parser.add_argument("--write-contracts", action="store_true", help="run and save CPU-only synthetic contracts")
    args = parser.parse_args()
    if args.finalize:
        if args.aliases_file or args.all or args.limit is not None or args.tracking_barrier:
            parser.error("--finalize cannot be combined with inference batch arguments")
    elif args.contracts or args.write_contracts:
        if args.aliases_file or args.all or args.limit is not None or args.tracking_barrier:
            parser.error("contracts cannot be combined with inference batch arguments")
    else:
        if not (args.aliases_file or args.all) or not args.tracking_barrier:
            parser.error("inference requires --aliases-file or --all plus --tracking-barrier")
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit is a positive hard cap")
    return args


def main() -> None:
    args = _parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True))
    elif args.write_contracts:
        print(json.dumps(write_contracts(), sort_keys=True))
    elif args.finalize:
        sys.addaudithook(guard)
        print(json.dumps(finalize(), sort_keys=True))
    else:
        sys.addaudithook(guard)
        print(json.dumps(run_batch(aliases_file=args.aliases_file, all_mode=bool(args.all),
                                  limit=args.limit, tracking_barrier_arg=args.tracking_barrier),
                         sort_keys=True))


if __name__ == "__main__":
    main()
