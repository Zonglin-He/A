#!/usr/bin/env python3
"""GT-free Table 1 source canonicalization and official tracking orchestration.

The root launcher owns GPU/resource admission.  This program starts only when
the frozen 64-case source barrier and a root-authored TRACKING_RUNTIME are
already present.  It calls the separately frozen adapter explicitly and
stores query-bearing requests only in the private experiment directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
ROSTER = BASE / "PREFLIGHT_ROSTER_LOCK.json"
MANIFEST = BASE / "INPUT_MANIFEST.json"
SOURCE_RUNTIME = BASE / "SOURCE_RUNTIME.json"
SOURCE_BARRIER = BASE / "SOURCE_CAPTURE_BARRIER.json"
TRACKING_RUNTIME = BASE / "TRACKING_RUNTIME.json"
REQUESTS = BASE / "tracking_cases"
CACHE_ROOT = BASE / "native_cache"
TRACKING_BARRIER = BASE / "TRACKING_GLOBAL_BARRIER.json"
ADAPTER_PATH = "scripts/run_stvg_table1_native_instance_adapter_v1.py"
ADAPTER_SHA256 = "4047966499c413329cb0d727969c20bd7ef486b5fc317bbca46c8d7554b3be53"
SCHEMA = "stvg-final-table1-query-tracking-v1"
TRACKING_RUNTIME_SCHEMA = "stvg-final-table1-tracking-runtime-v1"
SOURCE_BARRIER_SCHEMA = "stvg-final-table1-source-capture-barrier-v1"
TRACKING_BINDING_SCHEMA = "stvg-final-table1-tracking-binding-v1"
TRACKING_BARRIER_SCHEMA = "stvg-final-table1-tracking-input-barrier-v1"
MIN_FREE_BYTES = 30 * (1 << 30)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {Path(path).name}")
    return value


def _safe_root_file(relative: Any, *, base: Path = ROOT) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("sealed metadata contains an empty file path")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise RuntimeError("sealed metadata contains an unsafe relative path")
    resolved = (ROOT / rel).resolve(strict=True)
    if not resolved.is_relative_to(base.resolve(strict=True)):
        raise RuntimeError("sealed file escapes its approved experiment directory")
    if not resolved.is_file():
        raise RuntimeError("sealed path is not a regular file")
    return resolved


def _ensure_in_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(ROOT.resolve(strict=True)):
        raise RuntimeError("output or cache path escapes the project workspace")
    return resolved


def _immutable_bytes(path: Path, payload: bytes) -> tuple[str, int, bool]:
    """Install bytes atomically without replacing an existing artifact."""
    path = _ensure_in_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs; preserve it for review: {path.name}")
        return sha256_bytes(payload), len(payload), True

    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link creation is atomic and refuses to replace a concurrent
        # receipt; os.replace would silently overwrite it.
        try:
            os.link(temp, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"concurrent immutable output differs: {path.name}")
            return sha256_bytes(payload), len(payload), True
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(missing_ok=True)
    return sha256_bytes(payload), len(payload), False


def _immutable_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int, bool]:
    return _immutable_bytes(path, canonical_json_bytes(value) + b"\n")


def _alias_set(records: list[dict[str, Any]]) -> set[str]:
    aliases = {f"HC{i:02d}" for i in range(1, 33)} | {f"V{i:02d}" for i in range(1, 33)}
    if len(records) != 64 or {row.get("probe_alias") for row in records} != aliases:
        raise RuntimeError("the fixed Table 1 roster must contain exactly 32 HC and 32 VidSTG aliases")
    return aliases


def _verify_runtime_and_source_barrier() -> dict[str, Any]:
    """Validate source capture, source pins, and this tracking stage's pins."""
    from scripts import run_stvg_table1_source_v1 as source

    source_runtime = source.verify_runtime()
    if source_runtime.get("status") != "locked" or source_runtime.get("GT_read") is not False:
        raise RuntimeError("SOURCE_RUNTIME must remain locked and GT-free")
    records = source._roster_records()
    _alias_set(records)
    roster_sha = sha256_file(ROSTER)
    manifest_sha = sha256_file(MANIFEST)
    source_runtime_sha = sha256_file(SOURCE_RUNTIME)

    if not SOURCE_BARRIER.is_file():
        status_path = BASE / "SOURCE_CAPTURE_STATUS.json"
        status = read_json(status_path) if status_path.is_file() else {}
        raise RuntimeError(
            "SOURCE_CAPTURE_BARRIER.json is absent; tracking starts only after the source worker seals all 64 cases "
            f"(current source status={status.get('status', 'unknown')})"
        )
    barrier = read_json(SOURCE_BARRIER)
    expected = {
        "schema": SOURCE_BARRIER_SCHEMA,
        "status": "sealed",
        "count": 64,
        "source_runtime_sha256": source_runtime_sha,
        "preflight_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "GT_read": False,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
    }
    if any(barrier.get(key) != value for key, value in expected.items()):
        raise RuntimeError("SOURCE_CAPTURE_BARRIER is incomplete or does not bind the current frozen source runtime")
    source_files = barrier.get("source_files")
    aliases = _alias_set(records)
    if not isinstance(source_files, dict) or set(source_files) != aliases:
        raise RuntimeError("source barrier must bind one case and receipt for every fixed alias")
    # Verify all 64 payload and receipt byte bindings before selecting a query.
    # This is CPU/file I/O only; source payloads are not deserialized here.
    source_base = (BASE / "source_cases").resolve(strict=True)
    for alias in sorted(aliases):
        entry = source_files[alias]
        if not isinstance(entry, dict):
            raise RuntimeError("source barrier entry must be an object")
        path = _safe_root_file(entry.get("path"), base=source_base)
        receipt_path = _safe_root_file(entry.get("receipt_path"), base=source_base)
        if (path.stat().st_size != entry.get("bytes")
                or sha256_file(path) != entry.get("sha256")
                or sha256_file(receipt_path) != entry.get("receipt_sha256")):
            raise RuntimeError(f"source payload or receipt bytes differ from the global barrier: {alias}")
        receipt = read_json(receipt_path)
        if (receipt.get("schema") != "stvg-final-table1-source-receipt-v1"
                or receipt.get("case_id") != alias
                or receipt.get("sha256") != entry.get("sha256")
                or receipt.get("bytes") != entry.get("bytes")
                or receipt.get("source_case_sha256") != entry.get("source_case_sha256")
                or receipt.get("source_runtime_sha256") != source_runtime_sha
                or receipt.get("GT_read") is not False
                or receipt.get("DINO_read_or_call") is not False):
            raise RuntimeError(f"source-case receipt does not match the sealed barrier: {alias}")

    if not TRACKING_RUNTIME.is_file():
        raise RuntimeError("Root must create TRACKING_RUNTIME.json after reviewing the tracking orchestrator")
    tracking_runtime = read_json(TRACKING_RUNTIME)
    if (tracking_runtime.get("schema") != TRACKING_RUNTIME_SCHEMA
            or tracking_runtime.get("status") != "locked"
            or tracking_runtime.get("GT_read") is not False):
        raise RuntimeError("TRACKING_RUNTIME must use the locked, GT-free Table 1 tracking schema")
    pins = tracking_runtime.get("pins")
    if not isinstance(pins, dict):
        raise RuntimeError("TRACKING_RUNTIME pins are required")
    expected_pins = {
        "scripts/run_stvg_table1_tracking_v1.py": sha256_file(__file__),
        ADAPTER_PATH: ADAPTER_SHA256,
    }
    for rel, digest in expected_pins.items():
        if pins.get(rel) != digest:
            raise RuntimeError(f"TRACKING_RUNTIME does not pin the reviewed tracking source: {rel}")
    for rel, digest in {**pins, **tracking_runtime.get("protected", {})}.items():
        path = (ROOT / rel).resolve(strict=True)
        if (not path.is_file() or not isinstance(digest, str)
                or sha256_file(path) != digest):
            raise RuntimeError(f"tracking runtime pin mismatch: {rel}")
    bindings = tracking_runtime.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("TRACKING_RUNTIME source/barrier bindings are required")
    required_bindings = {
        "source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": sha256_file(SOURCE_BARRIER),
        "preflight_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
    }
    if any(bindings.get(key) != value for key, value in required_bindings.items()):
        raise RuntimeError("TRACKING_RUNTIME does not bind the current sealed source capture and roster")
    return {
        "records": records,
        "source_files": source_files,
        "source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": sha256_file(SOURCE_BARRIER),
        "tracking_runtime_sha256": sha256_file(TRACKING_RUNTIME),
        "preflight_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "tracking_runtime": tracking_runtime,
    }


def _load_source_case(alias: str, record: Mapping[str, Any], locked: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from scripts import run_stvg_table1_source_v1 as source

    entry = locked["source_files"][alias]
    source_base = (BASE / "source_cases").resolve(strict=True)
    path = _safe_root_file(entry["path"], base=source_base)
    receipt_path = _safe_root_file(entry["receipt_path"], base=source_base)
    case = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(case, dict):
        raise RuntimeError(f"source case is not an object: {alias}")
    if source.content_sha256(case) != entry["source_case_sha256"]:
        raise RuntimeError(f"source case canonical hash differs from the sealed receipt: {alias}")
    checks = {
        "schema": "stvg-final-table1-source-case-v1",
        "case_id": alias,
        "dataset": record["dataset"],
        "query_ordinal": int(record["query_ordinal"]),
        "query_key_sha256": record["query_key_sha256"],
        "source_runtime_sha256": locked["source_runtime_sha256"],
        "GT_read": False,
        "DINO_read_or_call": False,
        "optimizer_steps": 0,
        "backwards": 0,
    }
    if any(case.get(key) != value for key, value in checks.items()):
        raise RuntimeError(f"source case no longer matches its locked query/domain/runtime: {alias}")
    inp = case.get("input")
    if not isinstance(inp, Mapping):
        raise RuntimeError(f"source case input metadata is missing: {alias}")
    for key in ("video_sha256", "fps", "width", "height", "frame_count"):
        if inp.get(key) != record.get(key):
            raise RuntimeError(f"source case differs from its roster input metadata ({key}): {alias}")
    if list(case.get("video_bounds", [])) != list(record.get("video_bounds", [])):
        raise RuntimeError(f"source case physical bounds differ from the locked roster: {alias}")
    if list(map(int, case.get("frame_ids", []))) != list(map(int, record.get("frame_ids", []))):
        raise RuntimeError(f"source case source-grid differs from the locked roster: {alias}")
    receipt = read_json(receipt_path)
    if receipt.get("source_case_sha256") != source.content_sha256(case):
        raise RuntimeError(f"source receipt and case payload disagree: {alias}")
    return case


def _source_row(record: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    from scripts import run_stvg_table1_source_v1 as source

    dataset, row = source._validate_roster_record(record)
    if dataset != record.get("dataset"):
        raise RuntimeError("source row dataset differs from the sealed roster")
    query_text = row.get("input", {}).get("caption")
    if not isinstance(query_text, str) or not query_text.strip():
        raise RuntimeError("the official query row has no nonempty caption")
    # Exact PLAN bytes, ordinal, subject and input fields were verified by
    # source._validate_roster_record above. query_key_sha256 is the manifest
    # join identifier, distinct from the private caption content hash.
    return row, query_text


def _request_paths(alias: str) -> tuple[Path, Path]:
    folder = _ensure_in_root(REQUESTS / alias)
    if folder.exists():
        allowed = {"REQUEST.json", "TRACKING_BINDING.json"}
        unexpected = [p.name for p in folder.iterdir() if p.name not in allowed]
        if unexpected:
            raise RuntimeError(f"preserve partial or unexpected tracking files for {alias}")
    return folder / "REQUEST.json", folder / "TRACKING_BINDING.json"


def _verify_inventory_bytes(accepted: Mapping[str, Any], inventory: Mapping[str, Any]) -> tuple[Path, Path]:
    cache_base = CACHE_ROOT.resolve(strict=False)
    marker = _safe_root_file(accepted.get("marker_video_path"), base=cache_base)
    inventory_path = _safe_root_file(accepted.get("mask_inventory_path"), base=cache_base)
    mask_dir_value = accepted.get("mask_data_dir")
    if not isinstance(mask_dir_value, str):
        raise RuntimeError("accepted tracking record has no mask directory")
    mask_dir_rel = Path(mask_dir_value)
    if mask_dir_rel.is_absolute() or ".." in mask_dir_rel.parts:
        raise RuntimeError("accepted tracking record contains an unsafe mask directory")
    mask_dir = (ROOT / mask_dir_rel).resolve(strict=True)
    if not mask_dir.is_dir() or not mask_dir.is_relative_to(cache_base.resolve(strict=True)):
        raise RuntimeError("accepted mask directory escapes the content-addressed cache")
    if sha256_file(marker) != accepted.get("marker_video_sha256"):
        raise RuntimeError("marker video differs from accepted tracking bytes")
    if sha256_file(inventory_path) != accepted.get("mask_inventory_sha256"):
        raise RuntimeError("mask inventory bytes differ from accepted tracking receipt")
    if inventory.get("GT_read") is not False or inventory.get("query_independent") is not True:
        raise RuntimeError("tracking inventory is not explicitly query-independent and GT-free")
    files = inventory.get("files")
    if not isinstance(files, list) or inventory.get("mask_count") != len(files):
        raise RuntimeError("tracking inventory file count is inconsistent")
    ids: set[int] = set()
    indices: set[int] = set()
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"index", "path", "sha256", "bytes"}:
            raise RuntimeError("tracking inventory entry has an unexpected schema")
        mask_path = _safe_root_file(item["path"], base=cache_base)
        if (not mask_path.is_relative_to(mask_dir)
                or mask_path.stat().st_size != item["bytes"]
                or sha256_file(mask_path) != item["sha256"]):
            raise RuntimeError("a saved SAM2 mask differs from its inventory byte binding")
        index = item["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indices:
            raise RuntimeError("tracking mask index is invalid or duplicated")
        indices.add(index)
    known = inventory.get("known_ids")
    if (not isinstance(known, list) or known != sorted(set(known))
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in known)):
        raise RuntimeError("tracking inventory candidate IDs are malformed")
    return marker, inventory_path


def _verify_canonical_video_bytes(request: Mapping[str, Any]) -> tuple[Path, Path]:
    cache_base = CACHE_ROOT.resolve(strict=True)
    video_path = Path(str(request.get("video_path", ""))).resolve(strict=True)
    if not video_path.is_file() or not video_path.is_relative_to(cache_base):
        raise RuntimeError("canonical request video is missing or outside native_cache")
    if sha256_file(video_path) != request.get("canonical_video_sha256"):
        raise RuntimeError("canonical video bytes changed after lossless roundtrip")
    receipt_path = video_path.parent / "CANONICAL_RECEIPT.json"
    if not receipt_path.is_file():
        raise RuntimeError("canonical video has no immutable roundtrip receipt")
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "stvg-table1-canonical-input-v1"
            or receipt.get("status") != "complete"
            or receipt.get("canonical_video_sha256") != request.get("canonical_video_sha256")
            or receipt.get("canonical_pixel_sha256") != request.get("canonical_pixel_sha256")
            or receipt.get("source_video_sha256") != request.get("source_video_sha256")
            or receipt.get("canonical_video_bytes") != video_path.stat().st_size
            or receipt.get("GT_read") is not False
            or receipt.get("CPU_only") is not True
            or receipt.get("query_independent") is not True):
        raise RuntimeError("canonical clip receipt differs from the query request")
    return video_path, receipt_path


def _build_tracking_binding(alias: str, record: Mapping[str, Any], request: Mapping[str, Any],
                            request_path: Path, accepted: Mapping[str, Any], inventory: Mapping[str, Any],
                            accepted_path: Path, inventory_path: Path, locked: Mapping[str, Any]) -> dict[str, Any]:
    marker_path, bound_inventory_path = _verify_inventory_bytes(accepted, inventory)
    if bound_inventory_path != inventory_path.resolve(strict=True):
        raise RuntimeError("returned mask inventory path differs from the accepted receipt")
    canonical_video, canonical_receipt = _verify_canonical_video_bytes(request)
    return {
        "schema": TRACKING_BINDING_SCHEMA,
        "status": "pass",
        "case_alias": alias,
        "dataset": record["dataset"],
        "query_key_sha256": record["query_key_sha256"],
        "query_text_sha256": request["query_sha256"],
        "source_case_sha256": locked["source_files"][alias]["source_case_sha256"],
        "request_path": request_path.relative_to(ROOT).as_posix(),
        "request_sha256": sha256_file(request_path),
        "source_video_sha256": request["source_video_sha256"],
        "canonical_video_sha256": request["canonical_video_sha256"],
        "canonical_pixel_sha256": request["canonical_pixel_sha256"],
        "frame_clock_sha256": request["frame_clock_sha256"],
        "canonical_video_bytes": canonical_video.stat().st_size,
        "canonical_receipt_path": canonical_receipt.relative_to(ROOT).as_posix(),
        "canonical_receipt_sha256": sha256_file(canonical_receipt),
        "tracking_cache_key": accepted["tracking_cache_key"],
        "tracking_binding_sha256": accepted["tracking_binding_sha256"],
        "tracking_fingerprints": accepted["fingerprints"],
        "target_mode": accepted["target_mode"],
        "marker_video_path": marker_path.relative_to(ROOT).as_posix(),
        "marker_video_sha256": accepted["marker_video_sha256"],
        "inventory_path": bound_inventory_path.relative_to(ROOT).as_posix(),
        "inventory_sha256": accepted["mask_inventory_sha256"],
        "mask_count": inventory["mask_count"],
        "mask_ids": inventory["known_ids"],
        "accepted_path": accepted_path.relative_to(ROOT).as_posix(),
        "accepted_sha256": sha256_file(accepted_path),
        "accepted_bytes": accepted_path.stat().st_size,
        "source_runtime_sha256": locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "preflight_roster_sha256": locked["preflight_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
        "GT_read": False,
        "query_used_by_tracker": False,
    }


def _verify_existing_case(alias: str, record: Mapping[str, Any], locked: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate a prior request/binding pair without starting a tracker."""
    request_path, binding_path = _request_paths(alias)
    present = [path.is_file() for path in (request_path, binding_path)]
    if not any(present):
        if any(path.exists() for path in (request_path, binding_path)):
            raise RuntimeError(f"preserve non-file tracking output for {alias}")
        return None
    if not request_path.is_file():
        raise RuntimeError(f"tracking binding exists without its request; preserve it for review: {alias}")
    request = read_json(request_path)
    # validate_request checks the exact official request structure without
    # decoding source media or loading tracker models.
    from scripts import run_stvg_table1_native_instance_adapter_v1 as adapter
    normalized = adapter._native().validate_request(request)
    raw_request = canonical_json_bytes(request) + b"\n"
    request_sha = sha256_bytes(raw_request)
    if (request_path.stat().st_size != len(raw_request)
            or sha256_file(request_path) != request_sha
            or normalized.get("query_sha256") != sha256_bytes(_source_row(record)[1].encode("utf-8"))
            or normalized.get("source_video_sha256") != record.get("video_sha256")):
        raise RuntimeError(f"saved request bytes or roster binding differ: {alias}")
    canonical_video, canonical_receipt = _verify_canonical_video_bytes(request)
    if not binding_path.exists():
        # A crash after the atomic REQUEST commit but before tracking completed
        # is resumable only after the next pass reconstructs and byte-compares
        # the request from its source case and transient read_row result.
        return None
    if binding_path.exists():
        binding = read_json(binding_path)
        if (binding.get("schema") != TRACKING_BINDING_SCHEMA
                or binding.get("status") != "pass"
                or binding.get("case_alias") != alias
                or binding.get("request_sha256") != request_sha
                or binding.get("request_path") != request_path.relative_to(ROOT).as_posix()
                or binding.get("query_key_sha256") != record.get("query_key_sha256")
                or binding.get("dataset") != record.get("dataset")
                or binding.get("source_case_sha256") != locked["source_files"][alias]["source_case_sha256"]
                or binding.get("source_runtime_sha256") != locked["source_runtime_sha256"]
                or binding.get("source_capture_barrier_sha256") != locked["source_capture_barrier_sha256"]
                or binding.get("tracking_runtime_sha256") != locked["tracking_runtime_sha256"]
                or binding.get("preflight_roster_sha256") != locked["preflight_roster_sha256"]
                or binding.get("input_manifest_sha256") != locked["input_manifest_sha256"]
                or binding.get("canonical_video_sha256") != normalized.get("canonical_video_sha256")
                or binding.get("canonical_pixel_sha256") != normalized.get("canonical_pixel_sha256")
                or binding.get("frame_clock_sha256") != normalized.get("frame_clock_sha256")
                or binding.get("canonical_video_bytes") != canonical_video.stat().st_size
                or binding.get("canonical_receipt_path") != canonical_receipt.relative_to(ROOT).as_posix()
                or binding.get("canonical_receipt_sha256") != sha256_file(canonical_receipt)
                or binding.get("GT_read") is not False):
            raise RuntimeError(f"saved tracking binding is stale or malformed: {alias}")
        raw_binding = canonical_json_bytes(binding) + b"\n"
        if binding_path.stat().st_size != len(raw_binding) or sha256_file(binding_path) != sha256_bytes(raw_binding):
            raise RuntimeError(f"saved tracking binding bytes changed: {alias}")
        _verify_cache_from_binding(binding)
    return {
        "case_alias": alias,
        "dataset": record["dataset"],
        "request_path": request_path.relative_to(ROOT).as_posix(),
        "request_sha256": request_sha,
        "tracking_binding_path": binding_path.relative_to(ROOT).as_posix() if binding_path.is_file() else None,
        "tracking_binding_sha256": sha256_file(binding_path) if binding_path.is_file() else None,
        "tracking_cache_key": (read_json(binding_path).get("tracking_cache_key") if binding_path.is_file() else None),
        "accepted_path": (read_json(binding_path).get("accepted_path") if binding_path.is_file() else None),
        "accepted_sha256": (read_json(binding_path).get("accepted_sha256") if binding_path.is_file() else None),
        "inventory_path": (read_json(binding_path).get("inventory_path") if binding_path.is_file() else None),
        "inventory_sha256": (read_json(binding_path).get("inventory_sha256") if binding_path.is_file() else None),
        "GT_read": False,
    }


def _verify_cache_from_binding(binding: Mapping[str, Any]) -> None:
    cache_base = CACHE_ROOT.resolve(strict=False)
    accepted_path = _safe_root_file(binding.get("accepted_path"), base=cache_base)
    inventory_path = _safe_root_file(binding.get("inventory_path"), base=cache_base)
    if (accepted_path.stat().st_size != binding.get("accepted_bytes")
            or sha256_file(accepted_path) != binding.get("accepted_sha256")
            or sha256_file(inventory_path) != binding.get("inventory_sha256")):
        raise RuntimeError("saved tracking cache receipt bytes changed")
    accepted = read_json(accepted_path)
    inventory = read_json(inventory_path)
    if (accepted.get("tracking_cache_key") != binding.get("tracking_cache_key")
            or accepted.get("tracking_binding_sha256") != binding.get("tracking_binding_sha256")
            or accepted.get("mask_inventory_sha256") != binding.get("inventory_sha256")
            or accepted.get("GT_read") is not False
            or accepted.get("query_independent") is not True):
        raise RuntimeError("saved accepted tracking receipt disagrees with its binding")
    marker, _ = _verify_inventory_bytes(accepted, inventory)
    if (marker.relative_to(ROOT).as_posix() != binding.get("marker_video_path")
            or sha256_file(marker) != binding.get("marker_video_sha256")
            or inventory.get("mask_count") != binding.get("mask_count")
            or inventory.get("known_ids") != binding.get("mask_ids")):
        raise RuntimeError("saved marker or mask inventory differs from its query binding")


def _release_tracker_runtime() -> None:
    """Drop per-video Python references and return unused CUDA blocks if loaded."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception as exc:
            raise RuntimeError("failed to release per-video tracker CUDA cache") from exc
    gc.collect()


def _disk_guard() -> None:
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError("tracking stopped before this video: free disk space is below the 30 GiB floor")


def _process_one(record: Mapping[str, Any], locked: Mapping[str, Any]) -> dict[str, Any]:
    alias = str(record["probe_alias"])
    _disk_guard()
    from scripts import run_stvg_table1_native_instance_adapter_v1 as adapter
    from scripts import run_stvg_table1_source_v1 as source

    _, query_text = _source_row(record)
    source_case = _load_source_case(alias, record, locked)
    request = adapter.canonicalize(source_case, query_text, CACHE_ROOT)
    normalized = adapter._native().validate_request(request)
    if normalized.get("query_sha256") != sha256_bytes(query_text.encode("utf-8")):
        raise RuntimeError(f"canonical request query hash differs from sealed roster: {alias}")
    request_path, binding_path = _request_paths(alias)
    request_bytes = canonical_json_bytes(request) + b"\n"
    _immutable_bytes(request_path, request_bytes)

    # The adapter's content-addressed cache is query-independent. A duplicate
    # source video/clock reuses its canonical clip and tracker cache while
    # retaining a separate private request and binding for this query.
    _disk_guard()
    tracking = None
    try:
        tracking = adapter.track(request, CACHE_ROOT)
        accepted = tracking.get("accepted")
        inventory = tracking.get("mask_inventory")
        if not isinstance(accepted, Mapping) or not isinstance(inventory, Mapping):
            raise RuntimeError(f"tracking adapter returned an incomplete result: {alias}")
        accepted_path = _safe_root_file(
            (CACHE_ROOT / "tracking" / accepted["tracking_cache_key"] / "ACCEPTED.json").relative_to(ROOT).as_posix(),
            base=CACHE_ROOT.resolve(strict=False),
        )
        inventory_path = _safe_root_file(accepted.get("mask_inventory_path"), base=CACHE_ROOT.resolve(strict=False))
        binding = _build_tracking_binding(alias, record, normalized, request_path, accepted,
                                          inventory, accepted_path, inventory_path, locked)
        _immutable_json(binding_path, binding)
    finally:
        _release_tracker_runtime()
    row_out = _verify_existing_case(alias, record, locked)
    if row_out is None:
        raise RuntimeError("tracking output was not readable after atomic receipt creation")
    del tracking, accepted, inventory, request, normalized, source_case, query_text
    return row_out


def _verify_final_barrier(locked: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if not TRACKING_BARRIER.is_file():
        raise RuntimeError("tracking global barrier does not exist")
    barrier = read_json(TRACKING_BARRIER)
    expected = {
        "schema": TRACKING_BARRIER_SCHEMA,
        "status": "complete",
        "count": 64,
        "GT_read": False,
        "source_runtime_sha256": locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "preflight_roster_sha256": locked["preflight_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
    }
    if any(barrier.get(k) != v for k, v in expected.items()):
        raise RuntimeError("existing tracking barrier does not match the current immutable source/runtime")
    rows = barrier.get("records")
    aliases = _alias_set(records)
    if not isinstance(rows, list) or len(rows) != 64 or {r.get("case_alias") for r in rows} != aliases:
        raise RuntimeError("tracking barrier does not bind all 64 unique case aliases")
    by_alias = {r["case_alias"]: r for r in rows}
    record_by_alias = {r["probe_alias"]: r for r in records}
    verified = []
    for alias in sorted(aliases):
        row_out = _verify_existing_case(alias, record_by_alias[alias], locked)
        if row_out is None:
            raise RuntimeError(f"sealed tracking barrier references a missing per-query receipt: {alias}")
        expected_row = by_alias[alias]
        for key in ("request_path", "request_sha256", "tracking_binding_path", "tracking_binding_sha256",
                    "tracking_cache_key", "accepted_path", "accepted_sha256", "inventory_path", "inventory_sha256"):
            if row_out.get(key) != expected_row.get(key):
                raise RuntimeError(f"global tracking barrier entry changed for {alias}: {key}")
        verified.append(row_out)
    return {"status": "already_complete", "count": len(verified), "GT_read": False,
            "barrier_sha256": sha256_file(TRACKING_BARRIER)}


def run(case_alias: str | None = None, all_preflight: bool = False) -> dict[str, Any]:
    if all_preflight == (case_alias is not None):
        raise ValueError("Choose exactly one of --all-preflight or --case-alias")
    locked = _verify_runtime_and_source_barrier()
    records = locked["records"]
    aliases = _alias_set(records)
    if case_alias is not None and case_alias not in aliases:
        raise ValueError("case alias is not in the fixed Table 1 roster")

    # Reuse the project's GT-denial audit hook before reading any query row.
    from scripts.stvg_native_instance_common_v1 import guard
    sys.addaudithook(guard)

    if TRACKING_BARRIER.is_file():
        return _verify_final_barrier(locked, records)
    selected = records if all_preflight else [r for r in records if r["probe_alias"] == case_alias]
    if len(selected) != (64 if all_preflight else 1):
        raise RuntimeError("requested case selection does not match the fixed roster")

    completed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(selected, 1):
        alias = record["probe_alias"]
        existing = _verify_existing_case(alias, record, locked)
        if existing is None or existing.get("tracking_binding_path") is None:
            existing = _process_one(record, locked)
        completed[alias] = existing
        print(f"TABLE1_TRACKING {index}/{len(selected)} {alias} status=pass", flush=True)

    if not all_preflight:
        return {"status": "case_complete", "count": 1, "case_alias": case_alias, "GT_read": False}
    if set(completed) != aliases:
        raise RuntimeError("all-preflight did not complete every fixed case alias")
    output_rows = [completed[alias] for alias in sorted(aliases)]
    barrier = {
        "schema": TRACKING_BARRIER_SCHEMA,
        "status": "complete",
        "count": 64,
        "records": output_rows,
        "source_runtime_sha256": locked["source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "preflight_roster_sha256": locked["preflight_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
        "GT_read": False,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
        "time": time.time(),
    }
    _immutable_json(TRACKING_BARRIER, barrier)
    return {"status": "complete", "count": 64, "GT_read": False,
            "barrier_sha256": sha256_file(TRACKING_BARRIER)}


def contracts() -> dict[str, Any]:
    """Synthetic-only tests for lock selection, byte receipts, and privacy."""
    import tempfile

    checks = 0
    root = Path(tempfile.mkdtemp(prefix="table1-tracking-contract-", dir=ROOT))
    try:
        # Exact immutable replay is accepted; any changed byte is rejected.
        payload = canonical_json_bytes({"schema": "synthetic-request", "query_text": "PRIVATE"}) + b"\n"
        path = root / "HC01" / "REQUEST.json"
        digest, size, reused = _immutable_bytes(path, payload)
        assert digest == sha256_bytes(payload) and size == len(payload) and not reused
        checks += 1
        digest2, size2, reused2 = _immutable_bytes(path, payload)
        assert (digest2, size2, reused2) == (digest, size, True)
        checks += 1
        try:
            _immutable_bytes(path, payload + b" ")
        except RuntimeError:
            checks += 1
        else:
            raise AssertionError("immutable receipt accepted changed bytes")

        # Tracking bindings must use hashes/aliases only and never include the
        # transient query string saved inside the private REQUEST.json.
        binding = {"case_alias": "HC01", "query_key_sha256": "a" * 64,
                   "tracking_cache_key": "b" * 64, "GT_read": False}
        encoded = canonical_json_bytes(binding)
        assert b"PRIVATE" not in encoded and "query_text" not in binding
        checks += 1

        # Alias roster checks are exact and reject duplicates/missing cases.
        fake = [{"probe_alias": f"HC{i:02d}"} for i in range(1, 33)] + [
            {"probe_alias": f"V{i:02d}"} for i in range(1, 33)]
        assert len(_alias_set(fake)) == 64
        checks += 1
        try:
            _alias_set(fake[:-1] + [fake[0]])
        except RuntimeError:
            checks += 1
        else:
            raise AssertionError("duplicate alias roster was accepted")

        return {
            "schema": "stvg-final-table1-tracking-orchestrator-contracts-v1",
            "status": "pass",
            "checks": checks,
            "source_capture_or_runtime_read": False,
            "source_case_or_query_read": False,
            "media_or_tracker_called": False,
            "model_or_GPU_called": False,
            "GT_read": False,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-alias", choices=[f"HC{i:02d}" for i in range(1, 33)] +
                       [f"V{i:02d}" for i in range(1, 33)])
    group.add_argument("--all-preflight", action="store_true")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(contracts(), sort_keys=True))
        return 0
    result = run(case_alias=args.case_alias, all_preflight=args.all_preflight)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
