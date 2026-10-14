#!/usr/bin/env python3
"""Compare FULL Table 1 frozen Source captures to sealed original Source NPZs.

With no arguments this performs the CPU-only comparison, but fails closed
unless the 13,785-case FULL source seal and original baseline custody readback
are already complete. ``--contracts`` runs synthetic checks only. This reader
does not open annotations, videos, model weights, or GPU contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
PAPER = ROOT / "artifacts/decota_paper_experiments_v1"
SCORING = ROOT / "artifacts/decota_paper_baseline_scoring_v1"
FULL_ROSTER = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
FULL_SOURCE_RUNTIME = FULL / "SOURCE_RUNTIME.json"
FULL_SOURCE_BARRIER = FULL / "SOURCE_CAPTURE_BARRIER.json"
OLD_SOURCE_BARRIER = BASE / "SOURCE_CAPTURE_BARRIER.json"
OLD_SOURCE_READBACK = BASE / "ROOT_SOURCE_READBACK.json"
BASELINE_AVAILABILITY = FULL / "BASELINE_AVAILABILITY.json"
BASELINE_CUSTODY = FULL / "ROOT_BASELINE_CUSTODY_READBACK.json"
OUTPUT = FULL / "ROOT_SOURCE_BASELINE_READBACK.json"
EXPECTED_TOTAL = 13785
DOMAIN_COUNTS = {"hc2": 3482, "vidstg": 10303}
LEGACY_PROBE_COUNTS = {"hc2": 32, "vidstg": 32}
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_SHA.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _repo_file(value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("sealed path must be a string")
    raw = Path(value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("sealed path must resolve to a project file")
    return path


def _verify_pins(runtime: Mapping[str, Any], label: str) -> int:
    pins, protected = runtime.get("pins"), runtime.get("protected", {})
    if not isinstance(pins, Mapping) or not isinstance(protected, Mapping):
        raise RuntimeError(f"{label} must contain pins and protected maps")
    count = 0
    for relative, expected in {**pins, **protected}.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise RuntimeError(f"{label} contains an unsafe pin path")
        path = _repo_file(relative)
        if _sha(path) != _digest(expected, f"{label}.{relative}"):
            raise RuntimeError(f"{label} pin mismatch: {relative}")
        count += 1
    return count


def _alias(dataset: str, ordinal: int) -> str:
    prefix = "HC" if dataset == "hc2" else "V" if dataset == "vidstg" else None
    if prefix is None or ordinal < 0:
        raise ValueError("unsupported dataset/ordinal")
    return f"{prefix}{ordinal:05d}"


def _legacy_job(dataset: str) -> str:
    if dataset == "hc2":
        return "t1_ours_hc2"
    if dataset == "vidstg":
        return "t1_ours_vid"
    raise ValueError(f"unexpected dataset: {dataset}")


def _legacy_probe_identity(alias: Any, pre_index: Mapping[str, Any]) -> tuple[str, int]:
    if not isinstance(alias, str) or re.fullmatch(r"(?:HC|V)\d{2}", alias) is None:
        raise RuntimeError("64-case legacy alias format differs")
    probe = pre_index.get(alias)
    if not isinstance(probe, Mapping):
        raise RuntimeError(f"64-case readback alias is absent from preflight roster: {alias}")
    dataset, ordinal = probe.get("dataset"), probe.get("query_ordinal")
    if (dataset not in LEGACY_PROBE_COUNTS or isinstance(ordinal, bool)
            or not isinstance(ordinal, int) or not 0 <= ordinal < DOMAIN_COUNTS[dataset]):
        raise RuntimeError(f"64-case preflight alias has malformed dataset/ordinal: {alias}")
    if (alias.startswith("HC") and dataset != "hc2") or (alias.startswith("V") and dataset != "vidstg"):
        raise RuntimeError(f"64-case alias prefix disagrees with preflight dataset: {alias}")
    return str(dataset), ordinal


def _validate_prior_64() -> dict[str, Any]:
    readback = _read_json(OLD_SOURCE_READBACK)
    old_barrier = _read_json(OLD_SOURCE_BARRIER)
    preflight = _read_json(BASE / "PREFLIGHT_ROSTER_LOCK.json")
    if (readback.get("status") != "pass" or readback.get("count") != 64
            or readback.get("all_exact") is not True or readback.get("GT_read") is not False
            or readback.get("new_GPU") is not False
            or old_barrier.get("schema") != "stvg-final-table1-source-capture-barrier-v1"
            or old_barrier.get("status") != "sealed" or old_barrier.get("count") != 64
            or old_barrier.get("GT_read") is not False
            or readback.get("barrier_sha256") != _sha(OLD_SOURCE_BARRIER)):
        raise RuntimeError("the 64-case original Frozen Source readback is not sealed exact")
    rows = readback.get("records")
    source_files = old_barrier.get("source_files")
    if not isinstance(rows, list) or len(rows) != 64 or not isinstance(source_files, Mapping):
        raise RuntimeError("the 64-case readback/barrier schema is incomplete")
    pre_rows = preflight.get("records")
    if (preflight.get("status") != "locked" or preflight.get("GT_read") is not False
            or not isinstance(pre_rows, list) or len(pre_rows) != 64):
        raise RuntimeError("the 64-case preflight roster is missing or not locked")
    pre_index = {r.get("probe_alias"): r for r in pre_rows if isinstance(r, Mapping)}
    if len(pre_index) != 64:
        raise RuntimeError("the 64-case preflight roster has duplicate/malformed aliases")
    seen: set[str] = set()
    probes_by_domain = {name: 0 for name in LEGACY_PROBE_COUNTS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("64-case readback row is malformed")
        alias = row.get("alias")
        if not isinstance(alias, str) or alias in seen:
            raise RuntimeError("64-case readback has an invalid or duplicate alias")
        seen.add(alias)
        # Legacy probe aliases are sample labels, not query ordinals. Bind the
        # NPZ index through the frozen preflight roster (for example, HC01 is
        # not assumed to mean ordinal zero).
        dataset, ordinal = _legacy_probe_identity(alias, pre_index)
        probes_by_domain[dataset] += 1
        for field in ("source_sha256", "prior_prediction_sha256"):
            _digest(row.get(field), f"ROOT_SOURCE_READBACK.{alias}.{field}")
        source_item = source_files.get(alias)
        if not isinstance(source_item, Mapping):
            raise RuntimeError(f"64-case source barrier lacks the legacy alias: {alias}")
        source_path = _repo_file(source_item.get("path"))
        source_receipt_path = _repo_file(source_item.get("receipt_path"))
        if (source_path != (BASE / "source_cases" / f"{alias}.pt").resolve()
                or source_receipt_path != (BASE / "source_cases" / f"{alias}.receipt.json").resolve()
                or source_item.get("sha256") != row.get("source_sha256")
                or _sha(source_path) != source_item.get("sha256")
                or source_path.stat().st_size != source_item.get("bytes")
                or _sha(source_receipt_path) != source_item.get("receipt_sha256")):
            raise RuntimeError(f"64-case source case/receipt custody differs: {alias}")
        source_receipt = _read_json(source_receipt_path)
        if (source_receipt.get("schema") != "stvg-final-table1-source-receipt-v1"
                or source_receipt.get("case_id") != alias
                or source_receipt.get("sha256") != source_item.get("sha256")
                or source_receipt.get("bytes") != source_item.get("bytes")
                or source_receipt.get("GT_read") is not False):
            raise RuntimeError(f"64-case source receipt is stale or detached: {alias}")
        if (row.get("boxes_exact") is not True or row.get("interval_exact") is not True
                or row.get("grid_exact") is not True):
            raise RuntimeError(f"prior exact source-parity check failed: {alias}")
    if probes_by_domain != LEGACY_PROBE_COUNTS:
        raise RuntimeError("64-case readback no longer covers the fixed 32+32 probes")
    return {"readback_sha256": _sha(OLD_SOURCE_READBACK),
            "old_64_source_barrier_sha256": _sha(OLD_SOURCE_BARRIER),
            "count": 64, "all_exact": True}


def _validate_baseline_custody() -> dict[str, Any]:
    availability = _read_json(BASELINE_AVAILABILITY)
    custody = _read_json(BASELINE_CUSTODY)
    if (availability.get("schema") != "stvg-table1-baseline-availability-v1"
            or availability.get("status") != "metadata_audit_complete"
            or custody.get("status") != "pass" or custody.get("GT_read") is not False
            or custody.get("new_predictions") != 0
            or custody.get("total_files") != 120726
            or custody.get("evaluation_barrier_sha256") != availability.get("prediction_custody", {}).get("evaluation_barrier_sha256")):
        raise RuntimeError("original baseline availability/custody readback is incomplete")
    pred_custody = availability.get("prediction_custody", {})
    if (pred_custody.get("evaluation_status") != "sealed"
            or pred_custody.get("GT_read_at_barrier") is not False
            or pred_custody.get("original_barriers_unchanged") is not True
            or pred_custody.get("all_group_current_stat_counts_and_bytes_match") is not True
            or pred_custody.get("declared_prediction_files") != custody.get("total_files")):
        raise RuntimeError("baseline prediction custody metadata is not complete")
    evaluation_meta = availability.get("evidence", {}).get("evaluation_barrier")
    if not isinstance(evaluation_meta, Mapping):
        raise RuntimeError("baseline evaluation barrier evidence is missing")
    evaluation_path = _repo_file(evaluation_meta.get("path"))
    evaluation_sha = _digest(evaluation_meta.get("sha256"), "baseline evaluation barrier SHA")
    if (_sha(evaluation_path) != evaluation_sha
            or evaluation_sha != custody.get("evaluation_barrier_sha256")):
        raise RuntimeError("baseline evaluation barrier hash differs from custody readback")
    evaluation = _read_json(evaluation_path)
    if (evaluation.get("status") != "sealed" or evaluation.get("GT_read") is not False
            or evaluation.get("original_barriers_unchanged") is not True):
        raise RuntimeError("original baseline evaluation seal is not a clean pre-GT seal")

    availability_groups = pred_custody.get("groups")
    custody_groups = custody.get("groups")
    evaluation_groups = evaluation.get("groups")
    if not all(isinstance(x, list) for x in (availability_groups, custody_groups, evaluation_groups)):
        raise RuntimeError("baseline custody group lists are malformed")
    if len(availability_groups) != 9 or len(custody_groups) != 9 or len(evaluation_groups) != 9:
        raise RuntimeError("baseline custody must retain all nine original prediction groups")
    custody_by_key = {(r.get("dataset"), r.get("kind")): r for r in custody_groups}
    evaluation_by_barrier = {r.get("barrier"): r for r in evaluation_groups}
    source_groups: dict[str, dict[str, Any]] = {}
    for group in availability_groups:
        if not isinstance(group, Mapping):
            raise ValueError("baseline availability group is malformed")
        if group.get("kind") != "stateless":
            continue
        dataset = group.get("dataset")
        if dataset not in DOMAIN_COUNTS or dataset in source_groups:
            raise RuntimeError("baseline stateless Source group has unexpected domain/duplicate")
        expected_job = _legacy_job(dataset)
        expected_rel = f"artifacts/decota_paper_experiments_v1/table1_stateless/{expected_job}/PREDICTION_BARRIER.json"
        path = _repo_file(group.get("prediction_barrier"))
        barrier_sha = _digest(group.get("prediction_barrier_sha256"), "old Source barrier SHA")
        custody_group = custody_by_key.get((dataset, "stateless"))
        eval_group = evaluation_by_barrier.get(str(Path(expected_rel).relative_to("artifacts/decota_paper_experiments_v1")))
        if (str(path.relative_to(ROOT)) != expected_rel or _sha(path) != barrier_sha
                or group.get("barrier_sha_matches") is not True
                or group.get("current_stat_count_and_bytes_match_seal") is not True
                or group.get("GT_read") is not False
                or group.get("prediction_files_currently_present_by_stat") != DOMAIN_COUNTS[dataset]
                or not isinstance(custody_group, Mapping)
                or custody_group.get("all_bytes_sha256_exact") is not True
                or custody_group.get("barrier_sha256") != barrier_sha
                or not isinstance(eval_group, Mapping)
                or eval_group.get("prediction_barrier_sha256") != barrier_sha):
            raise RuntimeError(f"original {dataset} stateless Source group failed custody")
        barrier = _read_json(path)
        if (barrier.get("status") != "sealed" or barrier.get("GT_read") is not False
                or "Source Only" not in barrier.get("arms", [])
                or barrier.get("unique_readouts") != DOMAIN_COUNTS[dataset]
                or len(barrier.get("files", {})) != DOMAIN_COUNTS[dataset]):
            raise RuntimeError(f"original {dataset} Source NPZ barrier is incomplete")
        source_groups[dataset] = {"path": str(path.relative_to(ROOT),),
                                  "sha256": barrier_sha,
                                  "barrier": barrier}
    if set(source_groups) != set(DOMAIN_COUNTS):
        raise RuntimeError("both original Source NPZ cohorts are required")
    return {
        "availability_sha256": _sha(BASELINE_AVAILABILITY),
        "custody_sha256": _sha(BASELINE_CUSTODY),
        "evaluation_barrier_sha256": evaluation_sha,
        "source_groups": source_groups,
        "all_original_groups_custody_pass": True,
    }


def _source_context() -> dict[str, Any]:
    roster_path = FULL_ROSTER
    runtime_path = FULL_SOURCE_RUNTIME
    barrier_path = FULL_SOURCE_BARRIER
    roster = _read_json(roster_path)
    runtime = _read_json(runtime_path)
    barrier = _read_json(barrier_path)
    roster_sha, runtime_sha, barrier_sha = _sha(roster_path), _sha(runtime_path), _sha(barrier_path)
    if (roster.get("schema") != "stvg-final-table1-full-source-roster-v1"
            or roster.get("status") != "locked" or roster.get("GT_read") is not False
            or roster.get("selection_uses_GT_or_scores") is not False
            or roster.get("official_query_count") != EXPECTED_TOTAL
            or roster.get("domain_counts") != DOMAIN_COUNTS
            or runtime.get("status") != "locked" or runtime.get("GT_read") is not False
            or _verify_pins(runtime, "FULL SOURCE_RUNTIME") < 1
            or barrier.get("schema") != "stvg-final-table1-full-source-capture-barrier-v1"
            or barrier.get("status") != "sealed" or barrier.get("count") != EXPECTED_TOTAL
            or barrier.get("GT_read") is not False
            or barrier.get("source_runtime_sha256") != runtime_sha
            or barrier.get("full_source_roster_sha256") != roster_sha
            or barrier.get("input_manifest_sha256") != roster.get("manifest_sha256")):
        raise RuntimeError("FULL frozen source roster/runtime/capture barrier is not sealed")
    rows = roster.get("records")
    files = barrier.get("source_files")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TOTAL:
        raise RuntimeError("FULL source roster count is incomplete")
    if not isinstance(files, Mapping) or len(files) != EXPECTED_TOTAL:
        raise RuntimeError("FULL source barrier does not bind exactly 13,785 cases")
    index: dict[str, dict[str, Any]] = {}
    ordinals = {dataset: set() for dataset in DOMAIN_COUNTS}
    order: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FULL source roster row is malformed")
        dataset, ordinal, alias = row.get("dataset"), row.get("query_ordinal"), row.get("probe_alias")
        if (dataset not in DOMAIN_COUNTS or isinstance(ordinal, bool) or not isinstance(ordinal, int)
                or ordinal < 0 or ordinal >= DOMAIN_COUNTS[dataset] or alias != _alias(dataset, ordinal)
                or alias in index or row.get("GT_read") is not False):
            raise RuntimeError("FULL source roster identity/alias mapping is malformed")
        ordinals[dataset].add(ordinal)
        for field in ("query_key_sha256", "source_case_sha256", "parent_sha256", "safe_record_sha256",
                      "plan_sha256", "subject_sha256", "video_sha256"):
            # source_case_sha256 belongs to the receipt/barrier, not the roster;
            # the roster row intentionally binds query/source identity only.
            if field == "source_case_sha256":
                continue
            _digest(row.get(field), f"roster.{alias}.{field}")
        source_item = files.get(alias)
        if not isinstance(source_item, Mapping):
            raise RuntimeError(f"FULL source barrier is missing {alias}")
        index[alias] = {"roster": dict(row), "source_item": dict(source_item)}
        order.append(alias)
    if any(ordinals[d] != set(range(n)) for d, n in DOMAIN_COUNTS.items()):
        raise RuntimeError("FULL source roster ordinal coverage is incomplete")
    if order != sorted(order, key=lambda alias: (0 if alias.startswith("HC") else 1,
                                                 int(alias[2:] if alias.startswith("HC") else alias[1:]))):
        raise RuntimeError("FULL source roster is not in canonical dataset/ordinal order")
    return {"roster": roster, "runtime": runtime, "barrier": barrier,
            "roster_sha256": roster_sha, "runtime_sha256": runtime_sha,
            "barrier_sha256": barrier_sha, "case_index": index, "aliases": order,
            "pin_count": _verify_pins(runtime, "FULL SOURCE_RUNTIME")}


def _source_file_payload(alias: str, entry: Mapping[str, Any], torch: Any,
                         source_adapter: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    item = entry["source_item"]
    case_path = _repo_file(item.get("path"))
    receipt_path = _repo_file(item.get("receipt_path"))
    if (case_path.parent != (FULL / "source_cases").resolve()
            or receipt_path.parent != (FULL / "source_cases").resolve()
            or case_path.name != f"{alias}.pt"
            or receipt_path.name != f"{alias}.receipt.json"):
        raise RuntimeError(f"FULL source case path differs from expected alias: {alias}")
    case_sha = _digest(item.get("sha256"), f"FULL source {alias} file SHA")
    receipt_sha = _digest(item.get("receipt_sha256"), f"FULL source {alias} receipt SHA")
    if (_sha(case_path) != case_sha or case_path.stat().st_size != item.get("bytes")
            or _sha(receipt_path) != receipt_sha):
        raise RuntimeError(f"FULL source file/receipt byte hash mismatch: {alias}")
    receipt = _read_json(receipt_path)
    row = entry["roster"]
    if (receipt.get("schema") != "stvg-final-table1-full-source-receipt-v1"
            or receipt.get("status") not in {None, "sealed"}
            or receipt.get("case_id") != alias or receipt.get("sha256") != case_sha
            or receipt.get("bytes") != item.get("bytes")
            or receipt.get("source_runtime_sha256") != FULL_SOURCE_RUNTIME_SHA
            or receipt.get("full_source_roster_sha256") != FULL_SOURCE_ROSTER_SHA
            or receipt.get("input_manifest_sha256") != FULL_SOURCE_MANIFEST_SHA
            or receipt.get("query_key_sha256") != row.get("query_key_sha256")
            or receipt.get("source_case_sha256") != item.get("source_case_sha256")
            or receipt.get("GT_read") is not False):
        raise RuntimeError(f"FULL source receipt is stale or detached: {alias}")
    value = torch.load(case_path, map_location="cpu", weights_only=False)
    if (not isinstance(value, Mapping)
            or value.get("schema") != "stvg-final-table1-source-case-v1"
            or value.get("case_id") != alias or value.get("dataset") != row.get("dataset")
            or value.get("query_ordinal") != row.get("query_ordinal")
            or value.get("query_key_sha256") != row.get("query_key_sha256")
            or value.get("parent_sha256") != row.get("parent_sha256")
            or value.get("source_runtime_sha256") != FULL_SOURCE_RUNTIME_SHA
            or value.get("source_case_sha256") != item.get("source_case_sha256")
            or source_adapter.content_sha256(value) != item.get("source_case_sha256")
            or value.get("GT_read") is not False or value.get("DINO_read_or_call") is not False
            or int(value.get("optimizer_steps", -1)) != 0 or int(value.get("backwards", -1)) != 0):
        raise RuntimeError(f"FULL source case semantic identity differs: {alias}")
    return dict(value), receipt, case_sha


def _compare_payload(old_boxes: Any, old_meta: Mapping[str, Any],
                     new_case: Mapping[str, Any], np: Any,
                     ordinal: int) -> dict[str, Any]:
    """Compare Source-only normalized cxcywh, native clock, and sampled grid."""
    old = np.asarray(old_boxes)
    new = np.asarray(new_case.get("zero", {}).get("boxes"))
    old_frames = old_meta.get("frame_ids")
    new_frames = new_case.get("frame_ids")
    old_interval = old_meta.get("interval")
    new_interval = new_case.get("Native_interval")
    if (old.ndim != 2 or old.shape[-1] != 4 or new.ndim != 2 or new.shape[-1] != 4
            or old.shape != new.shape or len(old) == 0 or not np.isfinite(old).all()
            or not np.isfinite(new).all()):
        raise ValueError("Source-only cxcywh must be aligned finite [frames,4] arrays")
    if (not isinstance(old_frames, list) or not isinstance(new_frames, (list, tuple))
            or old_meta.get("parent") != ordinal):
        raise ValueError("old Source NPZ metadata is missing the exact parent/grid binding")
    old_frame_tuple = tuple(int(x) for x in old_frames)
    new_frame_tuple = tuple(int(x) for x in new_frames)
    old_iv = np.asarray(old_interval, dtype=np.float64)
    new_iv = np.asarray(new_interval, dtype=np.float64)
    if old_iv.shape != (2,) or new_iv.shape != (2,) or not np.isfinite(old_iv).all() or not np.isfinite(new_iv).all():
        raise ValueError("Native intervals must contain two finite endpoints")
    if len(old_frame_tuple) != len(old) or len(new_frame_tuple) != len(new):
        raise ValueError("Source frame grid and box arrays have different lengths")
    return {
        "boxes_exact": old.dtype == new.dtype and np.array_equal(old, new),
        "interval_exact": np.array_equal(old_iv, new_iv),
        "grid_exact": old_frame_tuple == new_frame_tuple,
        "frame_count": len(new_frame_tuple),
        "box_dtype": str(new.dtype),
    }


def _validate_legacy_probe_join(full_cases: Mapping[str, Any], old_barriers: Mapping[str, Any],
                                prior: Mapping[str, Any], preflight_roster: Mapping[str, Any]) -> dict[str, Any]:
    pre_rows = preflight_roster.get("records")
    if not isinstance(pre_rows, list) or len(pre_rows) != 64:
        raise RuntimeError("64-case preflight roster is missing or incomplete")
    pre_index = {row.get("probe_alias"): row for row in pre_rows if isinstance(row, Mapping)}
    if len(pre_index) != 64:
        raise RuntimeError("64-case preflight roster contains duplicate/malformed aliases")
    checked = 0
    for row in prior.get("records", []):
        alias = row["alias"]
        pre = pre_index.get(alias)
        if not isinstance(pre, Mapping):
            raise RuntimeError(f"64-case alias is absent from the preflight roster: {alias}")
        dataset, ordinal = _legacy_probe_identity(alias, pre_index)
        full_alias = _alias(str(dataset), int(ordinal))
        full = full_cases.get(full_alias)
        if not isinstance(full, Mapping):
            raise RuntimeError(f"legacy probe does not join to the FULL roster: {alias}")
        full_row = full["roster"]
        for key in ("dataset", "query_ordinal", "parent_sha256", "query_key_sha256",
                    "safe_record_sha256", "plan_path", "plan_sha256", "plan_row_pointer",
                    "subject_sha256", "video_sha256", "frame_ids"):
            if pre.get(key) != full_row.get(key):
                raise RuntimeError(f"64-case probe and FULL roster identity differ: {alias}/{key}")
        old_group = old_barriers.get(dataset)
        old_barrier = old_group.get("barrier") if isinstance(old_group, Mapping) else None
        old_files = old_barrier.get("files") if isinstance(old_barrier, Mapping) else None
        if (not isinstance(old_files, Mapping)
                or row.get("prior_prediction_sha256") != old_files.get(
                    f"table1_stateless/{_legacy_job(str(dataset))}/{int(ordinal):05d}.npz")):
            raise RuntimeError(f"64-case old Source prediction mapping differs: {alias}")
        checked += 1
    return {"matched_legacy_source_probes": checked, "exact_probes_per_domain": LEGACY_PROBE_COUNTS}


def compare_after_seal() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"preserve existing source baseline readback: {OUTPUT}")
    if not FULL_SOURCE_BARRIER.is_file():
        raise RuntimeError("FULL source baseline readback requires the sealed 13,785-case source barrier")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("source baseline readback refuses an initialized CUDA context")
    from scripts import run_stvg_table1_source_v1 as source_adapter

    # Verify all prior authorities before opening any new Source payload.
    prior = _validate_prior_64()
    baseline = _validate_baseline_custody()
    full = _source_context()
    global FULL_SOURCE_RUNTIME_SHA, FULL_SOURCE_ROSTER_SHA, FULL_SOURCE_MANIFEST_SHA
    FULL_SOURCE_RUNTIME_SHA = full["runtime_sha256"]
    FULL_SOURCE_ROSTER_SHA = full["roster_sha256"]
    FULL_SOURCE_MANIFEST_SHA = str(full["roster"].get("manifest_sha256"))
    if not HEX_SHA.fullmatch(FULL_SOURCE_MANIFEST_SHA):
        raise RuntimeError("FULL source roster manifest SHA is malformed")

    preflight_roster = _read_json(BASE / "PREFLIGHT_ROSTER_LOCK.json")
    legacy_join = _validate_legacy_probe_join(full["case_index"], baseline["source_groups"],
                                               _read_json(OLD_SOURCE_READBACK), preflight_roster)
    domain_counts = {dataset: 0 for dataset in DOMAIN_COUNTS}
    output_rows: list[dict[str, Any]] = []
    files_by_domain = {dataset: 0 for dataset in DOMAIN_COUNTS}
    exact_box_count = exact_interval_count = exact_grid_count = 0
    for alias in full["aliases"]:
        entry = full["case_index"][alias]
        roster_row = entry["roster"]
        dataset, ordinal = str(roster_row["dataset"]), int(roster_row["query_ordinal"])
        source_case, source_receipt, source_file_sha = _source_file_payload(
            alias, entry, torch, source_adapter)
        group = baseline["source_groups"][dataset]
        relative_npz = f"table1_stateless/{_legacy_job(dataset)}/{ordinal:05d}.npz"
        expected_old_sha = group["barrier"].get("files", {}).get(relative_npz)
        if not isinstance(expected_old_sha, str):
            raise RuntimeError(f"original Source NPZ barrier lacks ordinal {dataset}/{ordinal}")
        old_path = _repo_file(str(PAPER / relative_npz))
        if _sha(old_path) != expected_old_sha:
            raise RuntimeError(f"original Source NPZ byte SHA differs from its barrier: {dataset}/{ordinal}")
        old_receipt_path = old_path.with_suffix(".json")
        old_receipt = _read_json(old_receipt_path)
        if (old_receipt.get("sha256") != expected_old_sha
                or old_receipt.get("bytes") != old_path.stat().st_size
                or old_receipt.get("GT_read") is not False):
            raise RuntimeError(f"original Source NPZ receipt mismatch: {dataset}/{ordinal}")
        with np.load(old_path, allow_pickle=False) as archive:
            if "Source" not in archive.files or "metadata" not in archive.files:
                raise RuntimeError(f"old source prediction has an unexpected NPZ schema: {dataset}/{ordinal}")
            # The old Table 1 scorer defines Source as the frozen cxcywh array
            # and reads frame_ids/interval/parent from the NPZ metadata.
            old_boxes = archive["Source"].copy()
            old_metadata = json.loads(archive["metadata"].tobytes())
        comparison = _compare_payload(old_boxes, old_metadata, source_case, np, ordinal)
        if not all(comparison[k] for k in ("boxes_exact", "interval_exact", "grid_exact")):
            raise RuntimeError(f"FULL frozen Source differs from original prediction: {dataset}/{ordinal}")
        exact_box_count += int(comparison["boxes_exact"])
        exact_interval_count += int(comparison["interval_exact"])
        exact_grid_count += int(comparison["grid_exact"])
        domain_counts[dataset] += 1
        files_by_domain[dataset] += 1
        output_rows.append({
            "alias": alias, "dataset": dataset, "query_ordinal": ordinal,
            "new_source_case_sha256": source_file_sha,
            "new_source_receipt_sha256": entry["source_item"]["receipt_sha256"],
            "original_source_npz_sha256": expected_old_sha,
            "original_source_receipt_sha256": _sha(old_receipt_path),
            "boxes_exact": comparison["boxes_exact"],
            "interval_exact": comparison["interval_exact"],
            "grid_exact": comparison["grid_exact"],
            "source_grid_count": comparison["frame_count"],
            "coordinate_representation": "normalized_cxcywh",
        })
        if (len(output_rows) % 500 == 0):
            print(f"FULL_SOURCE_BASELINE_READBACK {len(output_rows)}/{EXPECTED_TOTAL}", flush=True)

    if domain_counts != DOMAIN_COUNTS or len(output_rows) != EXPECTED_TOTAL:
        raise RuntimeError("FULL source baseline comparison does not cover exact fixed cohort")
    if (exact_box_count != EXPECTED_TOTAL or exact_interval_count != EXPECTED_TOTAL
            or exact_grid_count != EXPECTED_TOTAL):
        raise RuntimeError("at least one FULL source prediction differs from the original Source NPZ")
    out = {
        "schema": "stvg-final-table1-full-source-baseline-readback-v1",
        "status": "pass", "scope": "FULL frozen source vs original sealed Table 1 Source NPZ; exact native prediction parity",
        "count": EXPECTED_TOTAL, "domain_counts": domain_counts,
        "exact_boxes": exact_box_count, "exact_native_intervals": exact_interval_count,
        "exact_source_frame_grids": exact_grid_count,
        "legacy_64_probe_join": legacy_join,
        "full_source_barrier_path": str(FULL_SOURCE_BARRIER.relative_to(ROOT)),
        "full_source_barrier_sha256": full["barrier_sha256"],
        "full_source_runtime_sha256": full["runtime_sha256"],
        "full_source_roster_sha256": full["roster_sha256"],
        "full_source_pin_count_checked": full["pin_count"],
        "old_source_barriers": {
            dataset: {"path": value["path"], "sha256": value["sha256"],
                      "file_count": len(value["barrier"].get("files", {}))}
            for dataset, value in baseline["source_groups"].items()},
        "baseline_availability_sha256": baseline["availability_sha256"],
        "baseline_custody_readback_sha256": baseline["custody_sha256"],
        "baseline_evaluation_barrier_sha256": baseline["evaluation_barrier_sha256"],
        "prior_64_source_readback_sha256": prior["readback_sha256"],
        "prior_64_source_capture_barrier_sha256": prior["old_64_source_barrier_sha256"],
        "reader_sha256": _sha(Path(__file__)),
        "records": output_rows,
        "GT_read": False, "CUDA_used": False, "model_loaded": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {key: value for key, value in out.items() if key != "records"}


def contracts() -> dict[str, Any]:
    import numpy as np

    base = {
        "zero": {"boxes": np.asarray([[0.35, 0.5, 0.2, 0.4],
                                       [0.62, 0.4, 0.3, 0.2]], dtype=np.float32)},
        "Native_interval": [2.0, 8.0], "frame_ids": [1, 4],
    }
    metadata = {"parent": 7, "interval": [2.0, 8.0], "frame_ids": [1, 4]}
    good = _compare_payload(base["zero"]["boxes"].copy(), metadata, base, np, 7)
    assert good == {"boxes_exact": True, "interval_exact": True, "grid_exact": True,
                    "frame_count": 2, "box_dtype": "float32"}
    changed_box = base["zero"]["boxes"].copy()
    changed_box[0, 0] += np.float32(1e-5)
    box_diff = _compare_payload(changed_box, metadata, base, np, 7)
    interval_diff = _compare_payload(base["zero"]["boxes"],
                                     {**metadata, "interval": [2.0, 9.0]}, base, np, 7)
    grid_diff = _compare_payload(base["zero"]["boxes"],
                                 {**metadata, "frame_ids": [1, 5]}, base, np, 7)
    assert box_diff["boxes_exact"] is False and box_diff["interval_exact"] is True and box_diff["grid_exact"] is True
    assert interval_diff["interval_exact"] is False
    assert grid_diff["grid_exact"] is False
    alias_checks = [_alias("hc2", 0) == "HC00000", _alias("hc2", 3481) == "HC03481",
                    _alias("vidstg", 0) == "V00000", _alias("vidstg", 10302) == "V10302"]
    assert all(alias_checks)
    synthetic_probe_index = {"HC01": {"dataset": "hc2", "query_ordinal": 839},
                             "V01": {"dataset": "vidstg", "query_ordinal": 420}}
    probe_bindings = (
        _legacy_probe_identity("HC01", synthetic_probe_index) == ("hc2", 839)
        and _legacy_probe_identity("V01", synthetic_probe_index) == ("vidstg", 420))
    assert probe_bindings
    return {
        "schema": "stvg-final-table1-full-source-baseline-cpu-contracts-v1",
        "status": "pass", "checks": 9,
        "synthetic_checks": {
            "same_normalized_cxcywh_matches": True,
            "box_difference_detected": True,
            "native_interval_difference_detected": True,
            "source_frame_grid_difference_detected": True,
            "legacy_full_roster_alias_boundaries": True,
            "legacy_probe_alias_joins_via_frozen_roster_not_suffix_ordinal": True,
            "metadata_parent_required_for_ordinal_binding": True,
            "source_only_npz_array_name_fixed_to_Source": True,
            "full_source_semantic_hash_and_receipt_checked_only_after_seal": True,
        },
        "old_npz_schema": {
            "prediction_array": "Source",
            "coordinate_representation": "normalized_cxcywh",
            "metadata_fields": ["parent", "frame_ids", "interval"],
            "sha_source": "stateless Source Only PREDICTION_BARRIER.json files map",
        },
        "GT_read": False, "CUDA_used": False, "model_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true", help="synthetic CPU-only schema/math contracts")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True, ensure_ascii=False))
        return
    result = compare_after_seal()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
