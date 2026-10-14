#!/usr/bin/env python3
"""GT-free full-roster canonicalization/tracking using the frozen Table 1 adapter.

The root-owned FULL/TRACKING_RUNTIME.json is required before tracking starts.
This wrapper reuses the 64-case tracking engine in a process-local namespace,
redirecting only its paths and its source-case loader. It never edits the
frozen engine or adapter and never reads labels or scores.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tarfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
MANIFEST = BASE / "INPUT_MANIFEST.json"
PREFLIGHT_RUNTIME = BASE / "SOURCE_RUNTIME.json"
PREFLIGHT_TRACKING_RUNTIME = BASE / "TRACKING_RUNTIME.json"
PREFLIGHT_TRACKING_BARRIER = BASE / "TRACKING_GLOBAL_BARRIER.json"
ROSTER = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
SOURCE_RUNTIME = FULL / "SOURCE_RUNTIME.json"
SOURCE_BARRIER = FULL / "SOURCE_CAPTURE_BARRIER.json"
TRACKING_RUNTIME = FULL / "TRACKING_RUNTIME.json"
REQUESTS = FULL / "tracking_cases"
CACHE_ROOT = BASE / "native_cache"
TRACKING_BARRIER = FULL / "TRACKING_GLOBAL_BARRIER.json"
TRACKING_STATUS = FULL / "TRACKING_STATUS.json"
RETIREMENTS = FULL / "CACHE_RETIREMENTS"
BATCH_BARRIER_PREFIX = "TRACKING_BATCH_"
BATCH_BARRIER_SUFFIX = "_BARRIER.json"

SELF_REL = "scripts/run_stvg_table1_full_tracking_v1.py"
FULL_SOURCE_WORKER_REL = "scripts/run_stvg_table1_full_source_v1.py"
SOURCE_WORKER_REL = "scripts/run_stvg_table1_source_v1.py"
TRACKING_ENGINE_REL = "scripts/run_stvg_table1_tracking_v1.py"
ADAPTER_REL = "scripts/run_stvg_table1_native_instance_adapter_v1.py"
TRACKING_ENGINE_SHA256 = "92ff5f00d200a258ab931407c16eb477c010c48a7b10aacc5958604fa5068d75"
ADAPTER_SHA256 = "4047966499c413329cb0d727969c20bd7ef486b5fc317bbca46c8d7554b3be53"
FULL_RUNTIME_SCHEMA = "stvg-final-table1-full-tracking-runtime-v1"
FULL_BARRIER_SCHEMA = "stvg-final-table1-full-tracking-barrier-v1"
FULL_BATCH_SCHEMA = "stvg-final-table1-full-tracking-batch-barrier-v1"
ALIAS_BATCH_SCHEMA = "stvg-final-table1-alias-batch-v1"
RETIREMENT_SCHEMA = "stvg-final-table1-cache-retirement-v1"
FULL_BINDING_SCHEMA = "stvg-final-table1-full-tracking-run-binding-v1"
REQUEST_SCHEMA = "stvg-final-table1-query-tracking-v1"
TRACKING_BINDING_SCHEMA = "stvg-final-table1-tracking-binding-v1"
SOURCE_CAPTURE_SCHEMA = "stvg-final-table1-full-source-capture-barrier-v1"
FULL_SOURCE_RECEIPT_SCHEMA = "stvg-final-table1-full-source-receipt-v1"
MIN_FREE_BYTES = 30 * (1 << 30)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

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
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


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
    if not resolved.is_file() or not resolved.is_relative_to(base.resolve(strict=True)):
        raise RuntimeError("sealed file is missing or outside its approved directory")
    return resolved


def _safe_cache_path(value: Any, *, require_exists: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("tracking metadata contains an empty cache path")
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=require_exists)
    else:
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise RuntimeError("tracking metadata contains an unsafe relative cache path")
        resolved = (ROOT / candidate).resolve(strict=require_exists)
    cache_root = CACHE_ROOT.resolve(strict=False)
    if not resolved.is_relative_to(cache_root) or (require_exists and not resolved.is_file()):
        raise RuntimeError("tracking cache path is missing or outside native_cache")
    return resolved


def _cache_path_is_symlink(value: Any) -> bool:
    if not isinstance(value, (str, Path)) or not str(value):
        return False
    candidate = Path(value)
    lexical = candidate if candidate.is_absolute() else ROOT / candidate
    return lexical.is_symlink()


def _ensure_root(path: Path) -> Path:
    result = path.resolve(strict=False)
    if not result.is_relative_to(ROOT.resolve(strict=True)):
        raise RuntimeError("output/cache path escapes the workspace")
    return result


def _immutable_bytes(path: Path, payload: bytes) -> tuple[str, int, bool]:
    path = _ensure_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs; preserve it: {path.name}")
        return sha256_bytes(payload), len(payload), True
    temp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"concurrent immutable output differs: {path.name}")
            return sha256_bytes(payload), len(payload), True
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)
    return sha256_bytes(payload), len(payload), False


def _immutable_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int, bool]:
    return _immutable_bytes(path, canonical_json_bytes(dict(value)))


def _write_status(value: Mapping[str, Any]) -> None:
    path = _ensure_root(TRACKING_STATUS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _expected_aliases() -> list[str]:
    return ([f"HC{i:05d}" for i in range(3482)] +
            [f"V{i:05d}" for i in range(10303)])


def _validate_roster_metadata(*, allow_prepared: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scripts import run_stvg_table1_full_source_v1 as full_source

    manifest = read_json(MANIFEST)
    manifest_sha = sha256_file(MANIFEST)
    expected = full_source.project_manifest(manifest, manifest_sha)
    roster = read_json(ROSTER)
    allowed_states = {"locked"} | ({"prepared_for_root_runtime_lock"} if allow_prepared else set())
    if roster.get("status") not in allowed_states or roster.get("GT_read") is not False:
        raise RuntimeError("FULL source roster must be locked and remain GT-free")
    if roster.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("FULL source roster no longer binds INPUT_MANIFEST")
    stable_keys = (
        "schema", "manifest_sha256", "official_query_count", "domain_counts",
        "all_manifest_rows_preserved", "selection_uses_GT_or_scores", "records", "GT_read",
    )
    if any(roster.get(key) != expected.get(key) for key in stable_keys):
        raise RuntimeError("FULL source roster differs from the exact manifest projection")
    records = roster.get("records")
    if (not isinstance(records, list) or len(records) != 13785
            or [r.get("probe_alias") for r in records] != _expected_aliases()):
        raise RuntimeError("FULL roster is not the fixed 13,785-row HC/V alias order")
    return roster, records


def _verify_pin_map(owner: Mapping[str, Any], field: str) -> None:
    mapping = owner.get(field, {})
    if not isinstance(mapping, dict):
        raise RuntimeError(f"runtime {field} must be an object")
    for rel, digest in mapping.items():
        path = ROOT / rel
        if (not isinstance(rel, str) or not SHA_RE.fullmatch(str(digest))
                or not path.is_file() or sha256_file(path) != digest):
            raise RuntimeError(f"frozen dependency changed: {rel}")


def _verify_locks() -> dict[str, Any]:
    from scripts import run_stvg_table1_full_source_v1 as full_source

    roster, records = _validate_roster_metadata()
    manifest_sha = sha256_file(MANIFEST)
    roster_sha = sha256_file(ROSTER)
    source_runtime = read_json(SOURCE_RUNTIME)
    if source_runtime.get("status") != "locked" or source_runtime.get("GT_read") is not False:
        raise RuntimeError("FULL/SOURCE_RUNTIME.json must be locked and GT-free")
    preflight_runtime = read_json(PREFLIGHT_RUNTIME)
    preflight_tracking = read_json(PREFLIGHT_TRACKING_RUNTIME)
    identity_path = FULL / "MANIFEST_QUERY_IDENTITY_READBACK.json"
    identity = read_json(identity_path)
    identity_sha = sha256_file(identity_path)
    source_runtime_sha = sha256_file(SOURCE_RUNTIME)
    preflight_runtime_sha = sha256_file(PREFLIGHT_RUNTIME)
    preflight_tracking_sha = sha256_file(PREFLIGHT_TRACKING_RUNTIME)
    source_bindings = {
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "preflight_source_runtime_sha256": preflight_runtime_sha,
        "query_identity_readback_sha256": identity_sha,
    }
    if any(source_runtime.get(key) != value for key, value in source_bindings.items()):
        raise RuntimeError("FULL/SOURCE_RUNTIME.json does not bind the current roster/manifest/readback")
    if (source_runtime.get("official_query_count") != 13785
            or source_runtime.get("domain_counts") != {"hc2": 3482, "vidstg": 10303}
            or identity.get("status") != "pass" or identity.get("count") != 13785
            or identity.get("identity_matches") != 13785
            or identity.get("GT_read") is not False):
        raise RuntimeError("FULL source runtime or query-identity readback is incomplete")
    if preflight_runtime.get("status") != "locked" or preflight_runtime.get("GT_read") is not False:
        raise RuntimeError("The original 64-case source runtime must remain locked and GT-free")
    if preflight_tracking.get("status") != "locked" or preflight_tracking.get("GT_read") is not False:
        raise RuntimeError("The original 64-case tracking runtime must remain locked and GT-free")

    # Validate source pipeline code pins, not source model weights: this worker
    # consumes sealed source cases and never loads their source checkpoints.
    _verify_pin_map(source_runtime, "pins")
    _verify_pin_map(source_runtime, "protected")
    for rel, digest in {**preflight_runtime.get("pins", {}),
                        **preflight_runtime.get("protected", {})}.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"original frozen source dependency changed: {rel}")
    _verify_pin_map(preflight_tracking, "pins")
    _verify_pin_map(preflight_tracking, "protected")

    source_barrier = read_json(SOURCE_BARRIER)
    barrier_sha = sha256_file(SOURCE_BARRIER)
    if (source_barrier.get("schema") != SOURCE_CAPTURE_SCHEMA
            or source_barrier.get("status") != "sealed"
            or source_barrier.get("count") != 13785
            or source_barrier.get("GT_read") is not False
            or source_barrier.get("source_runtime_sha256") != source_runtime_sha
            or source_barrier.get("full_source_roster_sha256") != roster_sha
            or source_barrier.get("input_manifest_sha256") != manifest_sha
            or source_barrier.get("DINO_calls") != 0
            or source_barrier.get("optimizer_steps") != 0
            or source_barrier.get("backwards") != 0):
        raise RuntimeError("FULL source capture barrier is absent, incomplete, or mismatched")
    source_files = source_barrier.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(_expected_aliases()):
        raise RuntimeError("FULL source barrier does not bind exactly all 13,785 source cases")
    for alias, entry in source_files.items():
        if (not isinstance(entry, dict) or entry.get("case_id", alias) != alias
                or not SHA_RE.fullmatch(str(entry.get("sha256", "")))
                or not SHA_RE.fullmatch(str(entry.get("source_case_sha256", "")))
                or not SHA_RE.fullmatch(str(entry.get("receipt_sha256", "")))):
            raise RuntimeError(f"FULL source barrier has a malformed file entry: {alias}")

    runtime = read_json(TRACKING_RUNTIME)
    if (runtime.get("schema") != FULL_RUNTIME_SCHEMA or runtime.get("status") != "locked"
            or runtime.get("GT_read") is not False):
        raise RuntimeError("Root must lock FULL/TRACKING_RUNTIME.json before tracking")
    expected_bindings = {
        "full_source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": barrier_sha,
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "preflight_source_runtime_sha256": preflight_runtime_sha,
        "preflight_tracking_runtime_sha256": preflight_tracking_sha,
    }
    bindings = runtime.get("bindings")
    if not isinstance(bindings, dict) or any(bindings.get(k) != v for k, v in expected_bindings.items()):
        raise RuntimeError("FULL/TRACKING_RUNTIME.json does not bind all source and tracking locks")
    pins = runtime.get("pins")
    if not isinstance(pins, dict):
        raise RuntimeError("FULL/TRACKING_RUNTIME.json must pin frozen worker dependencies")
    required_pins = {
        SELF_REL: sha256_file(Path(__file__)),
        FULL_SOURCE_WORKER_REL: sha256_file(ROOT / FULL_SOURCE_WORKER_REL),
        SOURCE_WORKER_REL: sha256_file(ROOT / SOURCE_WORKER_REL),
        TRACKING_ENGINE_REL: TRACKING_ENGINE_SHA256,
        ADAPTER_REL: ADAPTER_SHA256,
    }
    if any(pins.get(rel) != value for rel, value in required_pins.items()):
        raise RuntimeError("FULL tracking runtime is missing a required exact worker/adapter pin")
    _verify_pin_map(runtime, "pins")
    _verify_pin_map(runtime, "protected")

    return {
        "records": records,
        "source_files": source_files,
        # The frozen 64-case engine uses these legacy key names internally.
        # Keep the FULL-named values above for the batch ABI and expose exact
        # aliases only in this process-local lock mapping.
        "source_runtime_sha256": source_runtime_sha,
        "full_source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": barrier_sha,
        "tracking_runtime_sha256": sha256_file(TRACKING_RUNTIME),
        "preflight_source_runtime_sha256": preflight_runtime_sha,
        "preflight_tracking_runtime_sha256": preflight_tracking_sha,
        "full_source_roster_sha256": roster_sha,
        "preflight_roster_sha256": sha256_file(BASE / "PREFLIGHT_ROSTER_LOCK.json"),
        "input_manifest_sha256": manifest_sha,
        "tracking_runtime": runtime,
        "source_runtime": source_runtime,
        "roster": roster,
    }


def _load_full_source_case(alias: str, record: Mapping[str, Any], locked: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from scripts import run_stvg_table1_source_v1 as source_adapter

    entry = locked["source_files"][alias]
    source_base = (FULL / "source_cases").resolve(strict=True)
    path = _safe_root_file(entry.get("path"), base=source_base)
    receipt_path = _safe_root_file(entry.get("receipt_path"), base=source_base)
    if (path.stat().st_size != entry.get("bytes")
            or sha256_file(path) != entry.get("sha256")
            or receipt_path.stat().st_size != entry.get("receipt_bytes")
            or sha256_file(receipt_path) != entry.get("receipt_sha256")):
        raise RuntimeError(f"FULL source case/receipt bytes differ from the sealed barrier: {alias}")
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != FULL_SOURCE_RECEIPT_SCHEMA
            or receipt.get("case_id") != alias
            or receipt.get("sha256") != entry.get("sha256")
            or receipt.get("bytes") != entry.get("bytes")
            or receipt.get("source_case_sha256") != entry.get("source_case_sha256")
            or receipt.get("source_runtime_sha256") != locked["full_source_runtime_sha256"]
            or receipt.get("full_source_roster_sha256") != locked["full_source_roster_sha256"]
            or receipt.get("input_manifest_sha256") != locked["input_manifest_sha256"]
            or receipt.get("query_key_sha256") != record.get("query_key_sha256")
            or receipt.get("GT_read") is not False):
        raise RuntimeError(f"FULL source receipt metadata is inconsistent: {alias}")
    case = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(case, dict):
        raise RuntimeError(f"FULL source case is not a mapping: {alias}")
    if (case.get("schema") != "stvg-final-table1-source-case-v1"
            or case.get("case_id") != alias
            or case.get("dataset") != record.get("dataset")
            or case.get("query_ordinal") != record.get("query_ordinal")
            or case.get("query_key_sha256") != record.get("query_key_sha256")
            or case.get("parent_sha256") != record.get("parent_sha256")
            or case.get("source_runtime_sha256") != locked["full_source_runtime_sha256"]
            or case.get("source_case_sha256") != entry.get("source_case_sha256")
            or source_adapter.content_sha256(case) != entry.get("source_case_sha256")
            or case.get("input", {}).get("video_sha256") != record.get("video_sha256")
            or case.get("GT_read") is not False
            or case.get("DINO_read_or_call") is not False
            or int(case.get("optimizer_steps", -1)) != 0
            or int(case.get("backwards", -1)) != 0
            or list(case.get("video_bounds", [])) != list(record.get("video_bounds", []))
            or list(map(int, case.get("frame_ids", []))) != list(map(int, record.get("frame_ids", [])))):
        raise RuntimeError(f"FULL source payload differs from its locked roster entry: {alias}")
    for key in ("fps", "width", "height", "frame_count"):
        if case.get("input", {}).get(key) != record.get(key):
            raise RuntimeError(f"FULL source input metadata differs from roster ({key}): {alias}")
    return case


class _CanonicalRequestCache:
    """Reuse native canonical requests by exact video/decode/physical-clock key."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self._original = adapter.canonicalize
        self._ffmpeg: tuple[Path, str] | None = None
        self._base_requests: dict[str, dict[str, Any]] = {}

    def __call__(self, source_case: Mapping[str, Any], query_text: str,
                 cache_root: str | os.PathLike[str]) -> dict[str, Any]:
        cache_path = Path(cache_root).resolve()
        if cache_path != CACHE_ROOT.resolve():
            raise RuntimeError("tracking engine requested a noncanonical cache root")
        native_adapter = self.adapter._native()
        identity = self.adapter._source_identity(source_case)
        if self._ffmpeg is None:
            self._ffmpeg = self.adapter._ffmpeg_identity()
        _, ffmpeg_sha = self._ffmpeg
        key = self.adapter._canonical_key(identity, ffmpeg_sha)
        base = self._base_requests.get(key)
        if base is None:
            request = self._original(source_case, query_text, cache_path)
            base = {k: v for k, v in request.items() if k != "query_text"}
            self._base_requests[key] = base
            return request
        request = dict(base)
        request["query_text"] = query_text
        native_adapter.validate_request(request)
        return request


class _VerificationMemo:
    """Hash each shared clip/cache object once per bounded batch process."""

    def __init__(self, engine: Any):
        self._verify_video = engine._verify_canonical_video_bytes
        self._verify_cache = engine._verify_cache_from_binding
        self._videos: dict[str, tuple[tuple[int, int, int, int, int, int], Path, Path]] = {}
        self._caches: dict[tuple[str, str], tuple[tuple[int, int, int, int, int, int, int, int, int], Path | None]] = {}

    @staticmethod
    def _sig(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def verify_video(self, request: Mapping[str, Any]) -> tuple[Path, Path]:
        video = Path(str(request.get("video_path", ""))).resolve(strict=True)
        receipt = video.parent / "CANONICAL_RECEIPT.json"
        vs, rs = self._sig(video), self._sig(receipt)
        signature = (*vs, *rs)
        old = self._videos.get(str(video))
        if old is not None and old[0] == signature:
            return video, receipt
        checked_video, checked_receipt = self._verify_video(request)
        after = (*self._sig(checked_video), *self._sig(checked_receipt))
        self._videos[str(video)] = (after, checked_video, checked_receipt)
        return checked_video, checked_receipt

    def verify_cache(self, binding: Mapping[str, Any]) -> None:
        accepted = _safe_root_file(binding.get("accepted_path"), base=CACHE_ROOT.resolve(strict=True))
        inventory = _safe_root_file(binding.get("inventory_path"), base=CACHE_ROOT.resolve(strict=True))
        accepted_sig, inventory_sig = self._sig(accepted), self._sig(inventory)
        key = (str(accepted), str(inventory))
        old = self._caches.get(key)
        if old is not None:
            marker = old[1]
            marker_sig = self._sig(marker) if marker is not None else (0, 0, 0)
            signature = (*accepted_sig, *inventory_sig, *marker_sig)
            if old[0] == signature:
                return
        self._verify_cache(binding)
        accepted_json = read_json(accepted)
        marker = _safe_root_file(accepted_json.get("marker_video_path"), base=CACHE_ROOT.resolve(strict=True))
        signature = (*self._sig(accepted), *self._sig(inventory), *self._sig(marker))
        self._caches[key] = (signature, marker)


@contextmanager
def _tracking_engine(locked: Mapping[str, Any]):
    """Process-local path/source-loader overrides around the frozen 64-case engine."""
    engine = importlib.import_module("scripts.run_stvg_table1_tracking_v1")
    adapter = importlib.import_module("scripts.run_stvg_table1_native_instance_adapter_v1")
    if sha256_file(ROOT / TRACKING_ENGINE_REL) != TRACKING_ENGINE_SHA256:
        raise RuntimeError("frozen 64-case tracking engine revision changed")
    if sha256_file(ROOT / ADAPTER_REL) != ADAPTER_SHA256:
        raise RuntimeError("frozen native tracking adapter revision changed")
    replacements = {
        "BASE": FULL,
        "ROSTER": ROSTER,
        "MANIFEST": MANIFEST,
        "SOURCE_RUNTIME": SOURCE_RUNTIME,
        "SOURCE_BARRIER": SOURCE_BARRIER,
        "TRACKING_RUNTIME": TRACKING_RUNTIME,
        "REQUESTS": REQUESTS,
        "CACHE_ROOT": CACHE_ROOT,
        "TRACKING_BARRIER": TRACKING_BARRIER,
        "_load_source_case": _load_full_source_case,
    }
    original_values = {key: getattr(engine, key) for key in replacements}
    original_video_verifier = engine._verify_canonical_video_bytes
    original_cache_verifier = engine._verify_cache_from_binding
    canonicalizer = _CanonicalRequestCache(adapter)
    verification_memo = _VerificationMemo(engine)
    original_canonicalize = adapter.canonicalize
    try:
        for key, value in replacements.items():
            setattr(engine, key, value)
        adapter.canonicalize = canonicalizer
        engine._verify_canonical_video_bytes = verification_memo.verify_video
        engine._verify_cache_from_binding = verification_memo.verify_cache
        yield engine
    finally:
        adapter.canonicalize = original_canonicalize
        engine._verify_canonical_video_bytes = original_video_verifier
        engine._verify_cache_from_binding = original_cache_verifier
        for key, value in original_values.items():
            setattr(engine, key, value)
        gc.collect()


def _input_clock_key(record: Mapping[str, Any]) -> str:
    # This mirrors the adapter's canonical cache identity fields. FFmpeg and
    # codec revisions are fixed by the runtime and are added by the adapter.
    value = {
        "dataset": record["dataset"],
        "source_video_sha256": record["video_sha256"],
        "physical_bounds_half_open": record["video_bounds"],
        "fps": float(record["fps"]),
        "width": int(record["width"]),
        "height": int(record["height"]),
    }
    return sha256_bytes(canonical_json_bytes(value))


def _request_paths(alias: str) -> tuple[Path, Path]:
    folder = _ensure_root(REQUESTS / alias)
    if folder.exists():
        allowed = {"REQUEST.json", "TRACKING_BINDING.json"}
        if any(path.name not in allowed for path in folder.iterdir()):
            raise RuntimeError(f"preserve unexpected/partial tracking artifacts for {alias}")
    return folder / "REQUEST.json", folder / "TRACKING_BINDING.json"


def _disk_guard() -> None:
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("tracking stopped before the next query: free disk space is below 30 GiB")


def _video_clock_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    dataset = record.get("dataset")
    if dataset not in {"hc2", "vidstg"}:
        raise RuntimeError("FULL roster contains an unsupported tracking domain")
    bounds = record.get("video_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise RuntimeError("FULL roster has malformed physical video bounds")
    start, end = bounds
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start):
        raise RuntimeError("FULL roster has malformed half-open physical bounds")
    video_sha = record.get("video_sha256")
    if not SHA_RE.fullmatch(str(video_sha)):
        raise RuntimeError("FULL roster has a malformed source video SHA-256")
    fps = record.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise RuntimeError("FULL roster has an invalid source frame rate")
    width, height = record.get("width"), record.get("height")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise RuntimeError("FULL roster has invalid frame dimensions")
    return {
        "dataset_decode_policy": "ffmpeg_default_timing" if dataset == "hc2" else "ffmpeg_vsync0_original_grid",
        "source_video_sha256": video_sha,
        "physical_bounds_half_open": [start, end],
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_clock_policy": "contiguous_physical_frame_ids_v1",
    }


def _video_clock_group_sha256(record: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_video_clock_identity(record)))


def _selection_sha256(aliases: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(aliases))


def _describe_selected_groups(aliases: list[str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alias = {row["probe_alias"]: row for row in records}
    if any(alias not in by_alias for alias in aliases):
        raise RuntimeError("alias batch contains a name outside the locked FULL roster")
    roster_order = [row["probe_alias"] for row in records]
    selected_set = set(aliases)
    if aliases != [alias for alias in roster_order if alias in selected_set]:
        raise RuntimeError("alias batch order must match the locked FULL roster")
    identities: dict[str, dict[str, Any]] = {}
    roster_groups: dict[str, list[str]] = {}
    alias_groups: dict[str, str] = {}
    for row in records:
        group_sha = _video_clock_group_sha256(row)
        identities.setdefault(group_sha, _video_clock_identity(row))
        roster_groups.setdefault(group_sha, []).append(row["probe_alias"])
        alias_groups[row["probe_alias"]] = group_sha
    selected_groups = list(dict.fromkeys(alias_groups[alias] for alias in aliases))
    descriptors = []
    for group_sha in selected_groups:
        complete_group = roster_groups[group_sha]
        if not set(complete_group).issubset(selected_set):
            raise RuntimeError("alias batch splits a source-video/physical-clock group")
        descriptors.append({
            "video_clock_group_sha256": group_sha,
            "identity": identities[group_sha],
            "aliases": complete_group,
        })
    return descriptors


def _read_alias_batch(path: str | Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alias_path = Path(path).expanduser().resolve(strict=True)
    if not alias_path.is_relative_to(ROOT.resolve(strict=True)) or not alias_path.is_file():
        raise RuntimeError("--aliases-file must be a file inside the workspace")
    raw = json.loads(alias_path.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or raw.get("schema") != ALIAS_BATCH_SCHEMA
            or set(raw) - {"schema", "aliases", "video_clock_groups"}):
        raise RuntimeError("alias batch file has an unsupported or unsafe schema")
    aliases = raw.get("aliases")
    if (not isinstance(aliases, list) or not aliases
            or any(not isinstance(alias, str) for alias in aliases)
            or len(set(aliases)) != len(aliases)):
        raise RuntimeError("alias batch must contain a nonempty unique list of string aliases")
    by_alias = {row["probe_alias"]: row for row in records}
    video_clock_groups = _describe_selected_groups(aliases, records)
    declared_groups = raw.get("video_clock_groups")
    if declared_groups is not None and declared_groups != video_clock_groups:
        raise RuntimeError("alias batch declared video/clock groups differ from FULL roster metadata")
    return [by_alias[alias] for alias in aliases], {
        "path": alias_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(alias_path),
        "bytes": alias_path.stat().st_size,
        "aliases": aliases,
        "selection_sha256": _selection_sha256(aliases),
        "video_clock_groups": video_clock_groups,
    }


def _batch_barrier_path(selection_sha256: str) -> Path:
    if not SHA_RE.fullmatch(selection_sha256):
        raise RuntimeError("tracking batch selection key is malformed")
    return FULL / f"{BATCH_BARRIER_PREFIX}{selection_sha256}{BATCH_BARRIER_SUFFIX}"


def _batch_case_record(alias: str, current: Mapping[str, Any], record: Mapping[str, Any],
                       saved: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if current.get("GT_read") is not False or current.get("case_alias") != alias:
        raise RuntimeError(f"tracking case receipt is not cleanly bound: {alias}")
    required = ("request_path", "tracking_binding_path", "accepted_path", "inventory_path")
    if any(not isinstance(current.get(key), str) for key in required):
        raise RuntimeError(f"tracking case is missing request/binding/cache paths: {alias}")
    request = _safe_root_file(current["request_path"], base=REQUESTS.resolve(strict=True))
    binding = _safe_root_file(current["tracking_binding_path"], base=REQUESTS.resolve(strict=True))
    if (request != (REQUESTS / alias / "REQUEST.json").resolve(strict=False)
            or binding != (REQUESTS / alias / "TRACKING_BINDING.json").resolve(strict=False)):
        raise RuntimeError(f"tracking engine returned a request/binding path for a different alias: {alias}")
    accepted = _safe_cache_path(current["accepted_path"], require_exists=saved is None)
    inventory = _safe_cache_path(current["inventory_path"], require_exists=saved is None)
    request_value = read_json(request)
    binding_value = read_json(binding)
    canonical_path = _safe_cache_path(request_value.get("video_path"), require_exists=saved is None)
    marker_path = _safe_cache_path(binding_value.get("marker_video_path"), require_exists=saved is None)
    if saved is None:
        accepted_bytes, inventory_bytes = accepted.stat().st_size, inventory.stat().st_size
        accepted_sha, inventory_sha = current.get("accepted_sha256"), current.get("inventory_sha256")
        if not SHA_RE.fullmatch(str(accepted_sha)) or not SHA_RE.fullmatch(str(inventory_sha)):
            raise RuntimeError(f"tracking engine returned malformed cache hashes for {alias}")
        canonical_bytes, marker_bytes = canonical_path.stat().st_size, marker_path.stat().st_size
    else:
        accepted_bytes, inventory_bytes = saved["accepted_bytes"], saved["inventory_bytes"]
        accepted_sha, inventory_sha = saved["accepted_sha256"], saved["inventory_sha256"]
        canonical_bytes, marker_bytes = saved["canonical_video_bytes"], saved["marker_video_bytes"]
        if canonical_path.is_file() and canonical_path.stat().st_size != canonical_bytes:
            raise RuntimeError(f"canonical video size changed for {alias}")
        if marker_path.is_file() and marker_path.stat().st_size != marker_bytes:
            raise RuntimeError(f"marker video size changed for {alias}")
        if accepted.is_file() and (accepted.stat().st_size != accepted_bytes or sha256_file(accepted) != accepted_sha):
            raise RuntimeError(f"accepted tracking receipt changed for {alias}")
        if inventory.is_file() and (inventory.stat().st_size != inventory_bytes or sha256_file(inventory) != inventory_sha):
            raise RuntimeError(f"mask inventory changed for {alias}")
    return {
        "alias": alias,
        "dataset": record["dataset"],
        "video_clock_group_sha256": _video_clock_group_sha256(record),
        "request_path": request.relative_to(ROOT).as_posix(),
        "request_sha256": current["request_sha256"],
        "request_bytes": request.stat().st_size,
        "tracking_binding_path": binding.relative_to(ROOT).as_posix(),
        "tracking_binding_sha256": current["tracking_binding_sha256"],
        "tracking_binding_bytes": binding.stat().st_size,
        "accepted_path": accepted.relative_to(ROOT).as_posix(),
        "accepted_sha256": accepted_sha,
        "accepted_bytes": accepted_bytes,
        "inventory_path": inventory.relative_to(ROOT).as_posix(),
        "inventory_sha256": inventory_sha,
        "inventory_bytes": inventory_bytes,
        "tracking_cache_key": current["tracking_cache_key"],
        "canonical_video_sha256": binding_value["canonical_video_sha256"],
        "canonical_pixel_sha256": binding_value["canonical_pixel_sha256"],
        "canonical_video_path": canonical_path.relative_to(ROOT).as_posix(),
        "canonical_video_bytes": canonical_bytes,
        "frame_clock_sha256": binding_value["frame_clock_sha256"],
        "marker_video_path": marker_path.relative_to(ROOT).as_posix(),
        "marker_video_sha256": binding_value["marker_video_sha256"],
        "marker_video_bytes": marker_bytes,
    }


def _batch_bindings(locked: Mapping[str, Any]) -> dict[str, str]:
    return {
        "full_source_runtime_sha256": locked["full_source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "full_source_roster_sha256": locked["full_source_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
        "preflight_source_runtime_sha256": locked["preflight_source_runtime_sha256"],
        "preflight_tracking_runtime_sha256": locked["preflight_tracking_runtime_sha256"],
    }


def _persist_alias_batch(batch: Mapping[str, Any]) -> tuple[str, int]:
    path = FULL / "tracking_batch_inputs" / f"{batch['selection_sha256']}.json"
    payload = {
        "schema": ALIAS_BATCH_SCHEMA,
        "aliases": batch["aliases"],
        "video_clock_groups": batch["video_clock_groups"],
    }
    digest, size, _ = _immutable_json(path, payload)
    return path.relative_to(ROOT).as_posix(), size


def _hash_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(8 << 20), b""):
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def _preflight_protected_cache_roots() -> tuple[set[Path], set[Path]]:
    """Recover old 64-case cache ownership without opening query-bearing requests."""
    if not PREFLIGHT_TRACKING_BARRIER.is_file():
        raise RuntimeError("cannot authorize cache retirement without the sealed preflight tracking barrier")
    barrier = read_json(PREFLIGHT_TRACKING_BARRIER)
    if (barrier.get("schema") != "stvg-final-table1-tracking-input-barrier-v1"
            or barrier.get("status") != "complete" or barrier.get("count") != 64
            or barrier.get("GT_read") is not False or not isinstance(barrier.get("records"), list)
            or len(barrier["records"]) != 64):
        raise RuntimeError("preflight tracking barrier is incomplete; FULL cache retirement is unsafe")
    preflight_source = read_json(PREFLIGHT_RUNTIME)
    preflight_tracker = read_json(PREFLIGHT_TRACKING_RUNTIME)
    if (barrier.get("source_runtime_sha256") != sha256_file(PREFLIGHT_RUNTIME)
            or barrier.get("tracking_runtime_sha256") != sha256_file(PREFLIGHT_TRACKING_RUNTIME)
            or barrier.get("source_capture_barrier_sha256") != sha256_file(BASE / "SOURCE_CAPTURE_BARRIER.json")
            or barrier.get("preflight_roster_sha256") != sha256_file(BASE / "PREFLIGHT_ROSTER_LOCK.json")
            or preflight_source.get("status") != "locked" or preflight_source.get("GT_read") is not False
            or preflight_tracker.get("status") != "locked" or preflight_tracker.get("GT_read") is not False):
        raise RuntimeError("64-case preflight barrier no longer binds the locked source/tracking runtime")
    protected_files: set[Path] = set()
    protected_dirs: set[Path] = set()
    aliases = {row.get("case_alias") for row in barrier["records"]}
    if len(aliases) != 64 or None in aliases:
        raise RuntimeError("preflight tracking barrier aliases are malformed")
    for row in barrier["records"]:
        binding_value = row.get("tracking_binding_path")
        if not isinstance(binding_value, str):
            raise RuntimeError("preflight tracking barrier lacks a binding path")
        binding_path = _safe_root_file(binding_value, base=BASE.resolve(strict=True))
        if (not SHA_RE.fullmatch(str(row.get("tracking_binding_sha256", "")))
                or sha256_file(binding_path) != row["tracking_binding_sha256"]):
            raise RuntimeError("preflight tracking binding bytes differ from the 64-case barrier")
        binding = read_json(binding_path)
        if binding.get("GT_read") is not False or binding.get("case_alias") != row.get("case_alias"):
            raise RuntimeError("preflight tracking binding is malformed")
        for barrier_key, binding_key in (
            ("tracking_cache_key", "tracking_cache_key"),
            ("accepted_path", "accepted_path"),
            ("accepted_sha256", "accepted_sha256"),
            ("inventory_path", "inventory_path"),
            ("inventory_sha256", "inventory_sha256"),
        ):
            if row.get(barrier_key) != binding.get(binding_key):
                raise RuntimeError("preflight binding differs from its sealed 64-case barrier entry")
        for key in ("accepted_path", "inventory_path", "marker_video_path", "mask_data_dir"):
            value = binding.get(key)
            if not isinstance(value, str):
                raise RuntimeError(f"preflight binding omits protected cache path {key}")
            p = _safe_cache_path(value, require_exists=False)
            protected_files.add(p)
            protected_dirs.add(p.parent if key != "mask_data_dir" else p)
        receipt_value = binding.get("canonical_receipt_path")
        if not isinstance(receipt_value, str):
            raise RuntimeError("preflight binding omits its canonical receipt")
        receipt_path = _safe_root_file(receipt_value, base=CACHE_ROOT.resolve(strict=True))
        if (not SHA_RE.fullmatch(str(binding.get("canonical_receipt_sha256", "")))
                or sha256_file(receipt_path) != binding["canonical_receipt_sha256"]):
            raise RuntimeError("preflight canonical receipt differs from its sealed binding")
        canonical_dir = receipt_path.parent
        protected_dirs.add(canonical_dir)
        protected_files.add(canonical_dir / "input.mkv")
        protected_files.add(receipt_path)
    return protected_files, protected_dirs


def _safe_relative_workspace_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("retirement receipt contains an empty path")
    if ("\\" in value or "\x00" in value or re.match(r"^[A-Za-z]:", value)
            or value.startswith("/")):
        raise RuntimeError("retirement receipt contains a nonportable or absolute path")
    rel = Path(value)
    if (rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts)
            or rel.as_posix() != value):
        raise RuntimeError("retirement receipt contains an unsafe path")
    return rel


def _read_archive_members(archive_value: Mapping[str, Any] | None) -> dict[str, tuple[int, str]]:
    if archive_value is None:
        return {}
    if set(archive_value) != {"path", "bytes", "sha256", "members"}:
        raise RuntimeError("lossless evidence archive metadata has an unexpected schema")
    if (not isinstance(archive_value.get("path"), str)
            or (ROOT / archive_value["path"]).is_symlink()):
        raise RuntimeError("lossless evidence archive path must be a regular workspace file")
    archive_path = _safe_root_file(archive_value.get("path"), base=FULL.resolve(strict=True))
    archive_bytes = archive_value.get("bytes")
    if (isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0
            or not SHA_RE.fullmatch(str(archive_value.get("sha256", "")))
            or archive_path.is_symlink() or archive_path.stat().st_size != archive_bytes
            or sha256_file(archive_path) != archive_value.get("sha256")):
        raise RuntimeError("lossless evidence archive bytes do not match its receipt")
    declared = archive_value.get("members")
    if not isinstance(declared, list):
        raise RuntimeError("lossless evidence archive must list exact regular-file members")
    declared_by_path: dict[str, tuple[int, str]] = {}
    for row in declared:
        if (not isinstance(row, dict) or set(row) != {"original_path", "bytes", "sha256"}
                or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int)
                or row["bytes"] < 0 or not SHA_RE.fullmatch(str(row.get("sha256", "")))):
            raise RuntimeError("lossless evidence archive member receipt is malformed")
        rel = _safe_relative_workspace_path(row["original_path"])
        path = _safe_cache_path(rel.as_posix(), require_exists=False)
        if _cache_path_is_symlink(row["original_path"]) or path.suffix.lower() != ".json":
            raise RuntimeError("lossless evidence archive may contain only generated cache JSON files")
        key = rel.as_posix()
        if key in declared_by_path:
            raise RuntimeError("lossless evidence archive member path is duplicated")
        declared_by_path[key] = (row["bytes"], row["sha256"])
    if list(declared_by_path) != sorted(declared_by_path):
        raise RuntimeError("lossless evidence archive members must be sorted by root-relative path")

    observed: dict[str, tuple[int, str]] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            rel = _safe_relative_workspace_path(member.name)
            key = rel.as_posix()
            if not member.isfile() or member.issym() or member.islnk() or key in observed:
                raise RuntimeError("lossless evidence archive contains a non-regular or duplicate member")
            if key not in declared_by_path or member.size != declared_by_path[key][0]:
                raise RuntimeError("lossless evidence archive contains an undeclared or wrong-size member")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("cannot read a declared evidence archive member")
            with stream:
                digest, size = _hash_stream(stream)
            if (size, digest) != declared_by_path[key]:
                raise RuntimeError("lossless evidence archive member bytes differ from its receipt")
            observed[key] = (size, digest)
    if observed != declared_by_path:
        raise RuntimeError("lossless evidence archive omitted or added declared members")
    return observed


def _load_inventory_for_retirement(path: str, expected: Mapping[str, Any],
                                   archive_path: Path | None,
                                   archived_members: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
    cache_path = _safe_cache_path(path, require_exists=False)
    if cache_path.suffix.lower() != ".json" or _cache_path_is_symlink(path):
        raise RuntimeError("mask inventory must be a regular generated JSON file")
    if cache_path.is_file():
        if (cache_path.stat().st_size != expected["inventory_bytes"]
                or sha256_file(cache_path) != expected["inventory_sha256"]):
            raise RuntimeError("retained mask inventory bytes differ from batch barrier")
        return read_json(cache_path)
    rel = cache_path.relative_to(ROOT).as_posix()
    if archive_path is None or rel not in archived_members:
        raise RuntimeError("missing mask inventory is not preserved in the lossless archive")
    if archived_members[rel] != (expected["inventory_bytes"], expected["inventory_sha256"]):
        raise RuntimeError("archived mask inventory does not match the tracking batch binding")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        stream = archive.extractfile(rel)
        if stream is None:
            raise RuntimeError("archived mask inventory is unreadable")
        with stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("archived mask inventory is not a JSON object")
    return value


def _load_json_evidence_for_retirement(path: str, expected_bytes: int, expected_sha: str,
                                      archive_path: Path | None,
                                      archived_members: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
    cache_path = _safe_cache_path(path, require_exists=False)
    if cache_path.suffix.lower() != ".json" or _cache_path_is_symlink(path):
        raise RuntimeError("tracking evidence must be a regular generated JSON file")
    if cache_path.is_file():
        if cache_path.stat().st_size != expected_bytes or sha256_file(cache_path) != expected_sha:
            raise RuntimeError("retained tracking JSON differs from the immutable batch record")
        return read_json(cache_path)
    if cache_path.exists():
        raise RuntimeError("retired tracking JSON path is not a regular file")
    rel = cache_path.relative_to(ROOT).as_posix()
    if (archive_path is None or rel not in archived_members
            or archived_members[rel] != (expected_bytes, expected_sha)):
        raise RuntimeError("missing tracking JSON is not byte-preserved in the lossless archive")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        stream = archive.extractfile(rel)
        if stream is None:
            raise RuntimeError("archived tracking JSON is unreadable")
        with stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("archived tracking evidence is not a JSON object")
    return value


def _register_expected_evidence(target: dict[str, tuple[int, str]], path: str,
                                byte_count: int, digest: str) -> None:
    cache_path = _safe_cache_path(path, require_exists=False)
    rel = cache_path.relative_to(ROOT).as_posix()
    item = (byte_count, digest)
    old = target.get(rel)
    if old is not None and old != item:
        raise RuntimeError("one cache path has conflicting batch/inventory byte bindings")
    target[rel] = item


def _validate_inventory_record(row: Mapping[str, Any], inventory_value: Mapping[str, Any],
                               accepted_value: Mapping[str, Any]) -> None:
    if (inventory_value.get("schema") != "stvg-table1-query-independent-mask-inventory-v1"
            or inventory_value.get("GT_read") is not False
            or inventory_value.get("query_independent") is not True
            or inventory_value.get("tracking_cache_key") != row["tracking_cache_key"]
            or accepted_value.get("mask_inventory_path") != row["inventory_path"]
            or accepted_value.get("mask_inventory_sha256") != row["inventory_sha256"]
            or accepted_value.get("tracking_cache_key") != row["tracking_cache_key"]):
        raise RuntimeError("accepted receipt and generated mask inventory do not bind the same cache")
    files = inventory_value.get("files")
    known_ids = inventory_value.get("known_ids")
    if (not isinstance(files, list) or inventory_value.get("mask_count") != len(files)
            or not isinstance(known_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in known_ids)
            or known_ids != sorted(set(known_ids))):
        raise RuntimeError("retained/archived mask inventory row counts or IDs are malformed")
    mask_dir = _safe_cache_path(accepted_value.get("mask_data_dir"), require_exists=False)
    seen_indices: set[int] = set()
    seen_paths: set[str] = set()
    for mask in files:
        if (not isinstance(mask, dict) or set(mask) != {"index", "path", "sha256", "bytes"}
                or isinstance(mask.get("index"), bool) or not isinstance(mask.get("index"), int)
                or mask["index"] < 0
                or isinstance(mask.get("bytes"), bool) or not isinstance(mask.get("bytes"), int)
                or mask["bytes"] < 0 or not SHA_RE.fullmatch(str(mask.get("sha256", "")))):
            raise RuntimeError("mask inventory contains a malformed generated-mask binding")
        mask_path = _safe_cache_path(mask["path"], require_exists=False)
        rel = mask_path.relative_to(ROOT).as_posix()
        if (mask_path.suffix.lower() != ".json" or _cache_path_is_symlink(mask["path"])
                or not mask_path.is_relative_to(mask_dir)
                or mask["index"] in seen_indices or rel in seen_paths):
            raise RuntimeError("mask inventory contains an unsafe or duplicate mask path/index")
        if mask_path.name != f"mask_{mask['index']}.json":
            raise RuntimeError("generated mask filename does not match its local frame index")
        seen_indices.add(mask["index"])
        seen_paths.add(rel)


def _validate_retirement(path: Path, barrier: Mapping[str, Any], locked: Mapping[str, Any]
                         ) -> dict[str, Any] | None:
    retirement_path = RETIREMENTS / f"{barrier['selection_sha256']}.json"
    if not retirement_path.is_file():
        return None
    if retirement_path.is_symlink():
        raise RuntimeError("cache retirement receipt must be a regular file")
    retirement = read_json(retirement_path)
    if (retirement.get("schema") != RETIREMENT_SCHEMA or retirement.get("status") != "complete"
            or retirement.get("GT_read") is not False
            or retirement.get("selection_sha256") != barrier.get("selection_sha256")
            or retirement.get("tracking_batch_barrier_sha256") != sha256_file(path)
            or retirement.get("tracking_runtime_sha256") != locked["tracking_runtime_sha256"]):
        raise RuntimeError("cache retirement receipt is stale or malformed")
    seals = retirement.get("downstream_seals")
    expected_seal_kinds = {"teacher_batch", "student_batch", "root_batch"}
    if not isinstance(seals, dict) or set(seals) != expected_seal_kinds:
        raise RuntimeError("cache retirement receipt must bind teacher, student, and root batch seals")
    for seal in seals.values():
        if (not isinstance(seal, dict) or set(seal) != {"path", "sha256"}
                or not SHA_RE.fullmatch(str(seal.get("sha256", "")))):
            raise RuntimeError("downstream seal binding is malformed")
        seal_file = _safe_root_file(seal["path"], base=FULL.resolve(strict=True))
        if sha256_file(seal_file) != seal["sha256"]:
            raise RuntimeError("downstream batch seal bytes changed")

    archive_value = retirement.get("lossless_evidence_archive")
    archive_members = _read_archive_members(archive_value)
    archive_path = None if archive_value is None else _safe_root_file(archive_value["path"], base=FULL.resolve(strict=True))
    saved_rows = barrier.get("records", [])
    expected_media: dict[str, tuple[str, int, str]] = {}
    expected_evidence: dict[str, tuple[int, str]] = {}
    for row in saved_rows:
        if not isinstance(row, dict):
            raise RuntimeError("tracking batch contains a malformed saved row")
        for media_path, kind, size_key, sha_key in (
            (row.get("canonical_video_path"), "canonical_video", "canonical_video_bytes", "canonical_video_sha256"),
            (row.get("marker_video_path"), "marker_video", "marker_video_bytes", "marker_video_sha256"),
        ):
            cache_path = _safe_cache_path(media_path, require_exists=False)
            rel = cache_path.relative_to(ROOT).as_posix()
            size, digest = row.get(size_key), row.get(sha_key)
            if (isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    or not SHA_RE.fullmatch(str(digest))):
                raise RuntimeError("tracking batch media binding is malformed")
            item = (kind, size, digest)
            old = expected_media.get(rel)
            if old is not None and old != item:
                raise RuntimeError("duplicate batch media path has conflicting immutable bindings")
            expected_media[rel] = item
        for key in ("accepted", "inventory"):
            p_key = f"{key}_path"
            size, digest = row.get(f"{key}_bytes"), row.get(f"{key}_sha256")
            if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                    or not SHA_RE.fullmatch(str(digest))):
                raise RuntimeError("tracking batch evidence binding is malformed")
            _register_expected_evidence(expected_evidence, row.get(p_key), size, digest)

    missing_media: set[str] = set()
    for rel, (kind, expected_bytes, expected_sha) in expected_media.items():
        cache_path = _safe_cache_path(rel, require_exists=False)
        if _cache_path_is_symlink(rel):
            raise RuntimeError("canonical/marker cache path must not be a symbolic link")
        if cache_path.is_file():
            if cache_path.stat().st_size != expected_bytes or sha256_file(cache_path) != expected_sha:
                raise RuntimeError("retained canonical/marker bytes differ from the tracking batch")
        else:
            if cache_path.exists():
                raise RuntimeError("retired media path is not a regular file")
            missing_media.add(rel)
    retired_rows = retirement.get("retired_files")
    if not isinstance(retired_rows, list):
        raise RuntimeError("cache retirement receipt lacks retired_files")
    retired_by_path: dict[str, dict[str, Any]] = {}
    for row in retired_rows:
        if (not isinstance(row, dict) or set(row) != {"path", "kind", "bytes", "sha256"}
                or row.get("kind") not in {"canonical_video", "marker_video"}
                or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int)
                or not SHA_RE.fullmatch(str(row.get("sha256", "")))):
            raise RuntimeError("retired media file entry is malformed")
        rel = _safe_relative_workspace_path(row["path"]).as_posix()
        if rel in retired_by_path:
            raise RuntimeError("retired media path is duplicated")
        retired_by_path[rel] = row
        if rel not in expected_media:
            raise RuntimeError("retirement receipt lists a file outside the exact batch media set")
        kind, expected_bytes, expected_sha = expected_media[rel]
        if (row["kind"], row["bytes"], row["sha256"]) != (kind, expected_bytes, expected_sha):
            raise RuntimeError("retirement receipt media binding differs from tracking output")
    if set(retired_by_path) != missing_media:
        raise RuntimeError("retirement receipt does not exactly match missing canonical/marker files")

    missing_evidence: set[str] = set()
    for rel, (expected_bytes, expected_sha) in expected_evidence.items():
        evidence_path = _safe_cache_path(rel, require_exists=False)
        if _cache_path_is_symlink(rel):
            raise RuntimeError("tracking evidence path must not be a symbolic link")
        if evidence_path.is_file():
            if evidence_path.stat().st_size != expected_bytes or sha256_file(evidence_path) != expected_sha:
                raise RuntimeError("retained tracking evidence differs from batch binding")
        else:
            if evidence_path.exists():
                raise RuntimeError("retired evidence path is not a regular file")
            missing_evidence.add(rel)

    protected_files, protected_dirs = _preflight_protected_cache_roots()
    all_retired = set(retired_by_path) | set(archive_members)
    for rel in all_retired:
        candidate = _safe_cache_path(rel, require_exists=False)
        if candidate in protected_files or any(candidate == parent or candidate.is_relative_to(parent)
                                               for parent in protected_dirs):
            raise RuntimeError("retirement/archive includes a protected 64-case preflight cache path")

    # Inventory contents define the exact model-generated mask JSON set, which
    # may be preserved plain or through the lossless archive. Accepted and
    # inventory JSON themselves are parsed only after their exact bytes have
    # been verified above (from disk or from the archive).
    for row in saved_rows:
        accepted_value = _load_json_evidence_for_retirement(
            row["accepted_path"], row["accepted_bytes"], row["accepted_sha256"],
            archive_path, archive_members)
        inventory_value = _load_inventory_for_retirement(
            row["inventory_path"], row, archive_path, archive_members)
        if (accepted_value.get("schema") != "stvg-table1-query-independent-tracking-v1"
                or accepted_value.get("status") != "pass"
                or accepted_value.get("GT_read") is not False
                or accepted_value.get("query_independent") is not True
                or accepted_value.get("tracking_cache_key") != row["tracking_cache_key"]
                or accepted_value.get("canonical_video_sha256") != row["canonical_video_sha256"]
                or accepted_value.get("canonical_pixel_sha256") != row["canonical_pixel_sha256"]
                or accepted_value.get("frame_clock_sha256") != row["frame_clock_sha256"]
                or accepted_value.get("marker_video_path") != row["marker_video_path"]
                or accepted_value.get("marker_video_sha256") != row["marker_video_sha256"]):
            raise RuntimeError("retained/archived accepted receipt has an invalid query-free cache binding")
        _validate_inventory_record(row, inventory_value, accepted_value)
        for mask in inventory_value["files"]:
            mask_path = _safe_cache_path(mask["path"], require_exists=False)
            rel = mask_path.relative_to(ROOT).as_posix()
            _register_expected_evidence(expected_evidence, rel, mask["bytes"], mask["sha256"])
            if _cache_path_is_symlink(mask["path"]):
                raise RuntimeError("generated mask JSON path must not be a symbolic link")
            if mask_path.is_file():
                if mask_path.stat().st_size != mask["bytes"] or sha256_file(mask_path) != mask["sha256"]:
                    raise RuntimeError("retained generated mask JSON differs from inventory")
            else:
                if mask_path.exists():
                    raise RuntimeError("generated mask path is not a regular file")
                missing_evidence.add(rel)

    if set(archive_members) != missing_evidence:
        raise RuntimeError("lossless evidence archive does not exactly cover missing accepted/inventory/mask JSON")
    if any(archive_members[rel] != expected_evidence[rel] for rel in archive_members):
        raise RuntimeError("lossless evidence archive member hashes differ from tracking inventory")
    if archive_members and archive_value is None:
        raise RuntimeError("missing generated tracking evidence has no lossless archive")
    retirement["_archive_members"] = archive_members
    retirement["_archive_path"] = archive_path
    return retirement


def _verify_retired_case(alias: str, saved: Mapping[str, Any], record: Mapping[str, Any],
                         locked: Mapping[str, Any], retirement: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck one query receipt without opening its query-bearing REQUEST body."""
    expected_request_path = (REQUESTS / alias / "REQUEST.json").relative_to(ROOT).as_posix()
    expected_binding_path = (REQUESTS / alias / "TRACKING_BINDING.json").relative_to(ROOT).as_posix()
    if (saved.get("request_path") != expected_request_path
            or saved.get("tracking_binding_path") != expected_binding_path):
        raise RuntimeError(f"retired batch has a noncanonical per-query receipt path: {alias}")
    request = _safe_root_file(saved["request_path"], base=REQUESTS.resolve(strict=True))
    binding_path = _safe_root_file(saved["tracking_binding_path"], base=REQUESTS.resolve(strict=True))
    if (request.stat().st_size != saved.get("request_bytes")
            or sha256_file(request) != saved.get("request_sha256")
            or binding_path.stat().st_size != saved.get("tracking_binding_bytes")
            or sha256_file(binding_path) != saved.get("tracking_binding_sha256")):
        raise RuntimeError(f"retired batch request/binding bytes changed: {alias}")
    binding = read_json(binding_path)
    # canonical_json_bytes already includes exactly one trailing newline.
    # Appending another LF makes every writer-produced TRACKING_BINDING fail
    # its own byte-count/SHA check after retirement.
    binding_bytes = canonical_json_bytes(binding)
    if (len(binding_bytes) != binding_path.stat().st_size
            or sha256_bytes(binding_bytes) != saved.get("tracking_binding_sha256")):
        raise RuntimeError(f"retired batch binding is not canonical JSON: {alias}")
    source_entry = locked["source_files"].get(alias)
    if not isinstance(source_entry, Mapping):
        raise RuntimeError(f"retired batch alias has no sealed FULL source entry: {alias}")
    expected_binding = {
        "schema": TRACKING_BINDING_SCHEMA,
        "status": "pass",
        "case_alias": alias,
        "dataset": record["dataset"],
        "query_key_sha256": record["query_key_sha256"],
        "source_case_sha256": source_entry["source_case_sha256"],
        "request_path": expected_request_path,
        "request_sha256": saved["request_sha256"],
        "source_video_sha256": record["video_sha256"],
        "canonical_video_sha256": saved["canonical_video_sha256"],
        "canonical_pixel_sha256": saved["canonical_pixel_sha256"],
        "frame_clock_sha256": saved["frame_clock_sha256"],
        "canonical_video_bytes": saved["canonical_video_bytes"],
        "tracking_cache_key": saved["tracking_cache_key"],
        "marker_video_path": saved["marker_video_path"],
        "marker_video_sha256": saved["marker_video_sha256"],
        "inventory_path": saved["inventory_path"],
        "inventory_sha256": saved["inventory_sha256"],
        "accepted_path": saved["accepted_path"],
        "accepted_sha256": saved["accepted_sha256"],
        "accepted_bytes": saved["accepted_bytes"],
        "source_runtime_sha256": locked["full_source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "preflight_roster_sha256": locked["preflight_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
        "GT_read": False,
        "query_used_by_tracker": False,
    }
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise RuntimeError(f"retired batch tracking binding is stale or malformed: {alias}")
    if not SHA_RE.fullmatch(str(binding.get("query_text_sha256", ""))):
        raise RuntimeError(f"retired batch has a malformed query-text digest: {alias}")
    if (binding.get("canonical_receipt_sha256") is None
            or not SHA_RE.fullmatch(str(binding.get("canonical_receipt_sha256")))):
        raise RuntimeError(f"retired batch has no canonical roundtrip receipt hash: {alias}")
    accepted_value = _load_json_evidence_for_retirement(
        saved["accepted_path"], saved["accepted_bytes"], saved["accepted_sha256"],
        retirement.get("_archive_path"), retirement.get("_archive_members", {}))
    inventory_value = _load_inventory_for_retirement(
        saved["inventory_path"], saved, retirement.get("_archive_path"),
        retirement.get("_archive_members", {}))
    if (accepted_value.get("tracking_binding_sha256") != binding.get("tracking_binding_sha256")
            or accepted_value.get("schema") != "stvg-table1-query-independent-tracking-v1"
            or accepted_value.get("status") != "pass"
            or accepted_value.get("GT_read") is not False
            or accepted_value.get("query_independent") is not True
            or accepted_value.get("target_mode") != "all"
            or accepted_value.get("fingerprints") != binding.get("tracking_fingerprints")
            or accepted_value.get("mask_inventory_sha256") != saved["inventory_sha256"]
            or accepted_value.get("canonical_video_sha256") != saved["canonical_video_sha256"]
            or accepted_value.get("canonical_pixel_sha256") != saved["canonical_pixel_sha256"]
            or accepted_value.get("frame_clock_sha256") != saved["frame_clock_sha256"]
            or accepted_value.get("marker_video_path") != saved["marker_video_path"]
            or accepted_value.get("marker_video_sha256") != saved["marker_video_sha256"]
            or binding.get("mask_count") != inventory_value.get("mask_count")
            or binding.get("mask_ids") != inventory_value.get("known_ids")):
        raise RuntimeError(f"retired batch accepted receipt differs from its query binding: {alias}")
    canonical_receipt = _safe_root_file(binding.get("canonical_receipt_path"), base=CACHE_ROOT.resolve(strict=True))
    if sha256_file(canonical_receipt) != binding["canonical_receipt_sha256"]:
        raise RuntimeError(f"retired batch canonical receipt changed: {alias}")
    canonical_value = read_json(canonical_receipt)
    if (canonical_value.get("schema") != "stvg-table1-canonical-input-v1"
            or canonical_value.get("status") != "complete"
            or canonical_value.get("canonical_video_sha256") != saved["canonical_video_sha256"]
            or canonical_value.get("canonical_pixel_sha256") != saved["canonical_pixel_sha256"]
            or canonical_value.get("source_video_sha256") != record["video_sha256"]
            or canonical_value.get("canonical_video_bytes") != saved["canonical_video_bytes"]
            or canonical_value.get("GT_read") is not False
            or canonical_value.get("query_independent") is not True):
        raise RuntimeError(f"retired batch canonical receipt does not bind its sealed frame clock: {alias}")
    return dict(saved)


def _validate_batch_barrier(path: Path, locked: Mapping[str, Any],
                            engine: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    barrier = read_json(path)
    aliases = barrier.get("aliases")
    records = locked["records"]
    if (barrier.get("schema") != FULL_BATCH_SCHEMA or barrier.get("status") != "complete"
            or barrier.get("GT_read") is not False or barrier.get("query_used_by_tracker") is not False
            or not isinstance(aliases, list) or not aliases
            or any(not isinstance(alias, str) for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or barrier.get("count") != len(aliases)
            or barrier.get("selection_sha256") != _selection_sha256(aliases)
            or path != _batch_barrier_path(barrier.get("selection_sha256", ""))):
        raise RuntimeError(f"tracking batch barrier metadata is malformed: {path.name}")
    expected_groups = _describe_selected_groups(aliases, records)
    if barrier.get("video_clock_groups") != expected_groups:
        raise RuntimeError(f"tracking batch splits or misbinds a video/clock group: {path.name}")
    if barrier.get("bindings") != _batch_bindings(locked):
        raise RuntimeError(f"tracking batch runtime/source bindings are stale: {path.name}")
    alias_file_path = _safe_root_file(barrier.get("alias_file_path"), base=FULL.resolve(strict=True))
    expected_internal_alias = FULL / "tracking_batch_inputs" / f"{barrier['selection_sha256']}.json"
    if alias_file_path != expected_internal_alias.resolve(strict=True):
        raise RuntimeError(f"tracking batch internal alias file is outside its content-addressed path: {path.name}")
    submitted_alias_path = _safe_root_file(barrier.get("submitted_alias_file_path"), base=ROOT.resolve(strict=True))
    if (submitted_alias_path.stat().st_size != barrier.get("submitted_alias_file_bytes")
            or sha256_file(submitted_alias_path) != barrier.get("submitted_alias_file_sha256")
            or not SHA_RE.fullmatch(str(barrier.get("submitted_alias_file_sha256", "")))
            or alias_file_path.stat().st_size != barrier.get("alias_file_bytes")
            or sha256_file(alias_file_path) != barrier.get("alias_file_sha256")):
        raise RuntimeError(f"tracking batch alias file changed: {path.name}")
    alias_file = read_json(alias_file_path)
    submitted_alias_file = read_json(submitted_alias_path)
    if (alias_file.get("schema") != ALIAS_BATCH_SCHEMA
            or alias_file.get("aliases") != aliases
            or alias_file.get("video_clock_groups") != expected_groups
            or submitted_alias_file.get("schema") != ALIAS_BATCH_SCHEMA
            or submitted_alias_file.get("aliases") != aliases
            or submitted_alias_file.get("video_clock_groups") not in (None, expected_groups)):
        raise RuntimeError(f"tracking batch alias file contents changed: {path.name}")
    barrier_rows = barrier.get("records")
    if not isinstance(barrier_rows, list) or len(barrier_rows) != len(aliases):
        raise RuntimeError(f"tracking batch lacks per-alias file bindings: {path.name}")
    retirement = _validate_retirement(path, barrier, locked)
    by_roster = {row["probe_alias"]: row for row in records}
    verified_rows = []
    for alias, saved in zip(aliases, barrier_rows):
        if not isinstance(saved, dict) or saved.get("alias") != alias:
            raise RuntimeError(f"tracking batch alias order changed: {path.name}")
        if retirement is None:
            current = engine._verify_existing_case(alias, by_roster[alias], locked)
            if current is None:
                raise RuntimeError(f"tracking batch references a missing per-query receipt: {alias}")
            expected_row = _batch_case_record(alias, current, by_roster[alias])
        else:
            expected_row = _verify_retired_case(
                alias, saved, by_roster[alias], locked, retirement)
        if saved != expected_row:
            raise RuntimeError(f"tracking batch per-query file hashes changed: {alias}")
        verified_rows.append(expected_row)
    return barrier, verified_rows


def run_batch(aliases_file: str | Path) -> dict[str, Any]:
    locked = _verify_locks()
    records = locked["records"]
    selected, batch = _read_alias_batch(aliases_file, records)
    alias_file_path, alias_file_bytes = _persist_alias_batch(batch)
    barrier_path = _batch_barrier_path(batch["selection_sha256"])

    # Install the existing GT-denial audit hook before any PLAN query is read.
    from scripts.stvg_native_instance_common_v1 import guard
    sys.addaudithook(guard)

    processed = []
    with _tracking_engine(locked) as engine:
        if barrier_path.exists():
            barrier, _ = _validate_batch_barrier(barrier_path, locked, engine)
            return {"status": "already_complete", "count": barrier["count"],
                    "selection_sha256": barrier["selection_sha256"],
                    "barrier_path": barrier_path.relative_to(ROOT).as_posix(),
                    "barrier_sha256": sha256_file(barrier_path), "GT_read": False}
        by_alias = {row["probe_alias"]: row for row in records}
        group_by_alias = {alias: group["video_clock_group_sha256"]
                          for group in batch["video_clock_groups"] for alias in group["aliases"]}
        for index, record in enumerate(selected, 1):
            alias = record["probe_alias"]
            current = engine._verify_existing_case(alias, record, locked)
            if current is None or current.get("tracking_binding_path") is None:
                _disk_guard()
                current = engine._process_one(record, locked)
            if current is None or current.get("tracking_binding_path") is None:
                raise RuntimeError(f"tracking did not produce a complete receipt for {alias}")
            processed.append(_batch_case_record(alias, current, by_alias[alias]))
            _write_status({
                "schema": "stvg-final-table1-full-tracking-progress-v1",
                "status": "running" if index < len(selected) else "sealing_batch",
                "done_this_invocation": index,
                "total_this_invocation": len(selected),
                "current_alias": alias,
                "current_dataset": record["dataset"],
                "video_clock_group_sha256": group_by_alias[alias],
                "selection_sha256": batch["selection_sha256"],
                "GT_read": False,
                "query_used_by_tracker": False,
                "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
                "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
                "updated_at_unix": time.time(),
            })
            if index == 1 or index % 25 == 0 or index == len(selected):
                print(f"TABLE1_FULL_TRACKING_BATCH {index}/{len(selected)} {alias} status=pass", flush=True)

    if len(processed) != len(selected):
        raise RuntimeError("tracking batch did not finish every selected FULL alias")
    bindings = _batch_bindings(locked)
    barrier = {
        "schema": FULL_BATCH_SCHEMA,
        "status": "complete",
        "count": len(selected),
        "aliases": batch["aliases"],
        "selection_sha256": batch["selection_sha256"],
        "submitted_alias_file_path": batch["path"],
        "submitted_alias_file_sha256": batch["sha256"],
        "submitted_alias_file_bytes": batch["bytes"],
        "alias_file_path": alias_file_path,
        "alias_file_sha256": sha256_file(ROOT / alias_file_path),
        "alias_file_bytes": alias_file_bytes,
        "video_clock_groups": batch["video_clock_groups"],
        "records": processed,
        "bindings": bindings,
        "query_independent_tracker_cache": True,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
        "GT_read": False,
        "query_used_by_tracker": False,
        "time": time.time(),
    }
    _immutable_json(barrier_path, barrier)
    _write_status({
        "schema": "stvg-final-table1-full-tracking-progress-v1",
        "status": "batch_complete", "count": len(selected),
        "selection_sha256": batch["selection_sha256"],
        "barrier_path": barrier_path.relative_to(ROOT).as_posix(),
        "barrier_sha256": sha256_file(barrier_path), "GT_read": False,
        "updated_at_unix": time.time(),
    })
    return {"status": "batch_complete", "count": len(selected),
            "selection_sha256": batch["selection_sha256"],
            "barrier_path": barrier_path.relative_to(ROOT).as_posix(),
            "barrier_sha256": sha256_file(barrier_path), "GT_read": False}


def finalize() -> dict[str, Any]:
    locked = _verify_locks()
    records = locked["records"]
    from scripts.stvg_native_instance_common_v1 import guard
    sys.addaudithook(guard)
    batch_paths = sorted(FULL.glob(f"{BATCH_BARRIER_PREFIX}*{BATCH_BARRIER_SUFFIX}"))
    if not batch_paths:
        raise RuntimeError("no completed tracking batch barriers exist")
    roster_by_alias = {row["probe_alias"]: row for row in records}
    all_rows: dict[str, dict[str, Any]] = {}
    group_owner: dict[str, str] = {}
    batch_entries = []
    with _tracking_engine(locked) as engine:
        for path in batch_paths:
            barrier, rows = _validate_batch_barrier(path, locked, engine)
            for group in barrier["video_clock_groups"]:
                group_sha = group["video_clock_group_sha256"]
                if group_sha in group_owner:
                    raise RuntimeError("a video/clock group is duplicated across tracking batches")
                group_owner[group_sha] = path.name
            for row in rows:
                alias = row["alias"]
                if alias in all_rows:
                    raise RuntimeError(f"FULL alias appears in multiple tracking batches: {alias}")
                all_rows[alias] = row
            batch_entries.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "selection_sha256": barrier["selection_sha256"],
                "count": barrier["count"],
            })
        if set(all_rows) != set(roster_by_alias):
            missing = len(set(roster_by_alias) - set(all_rows))
            extra = len(set(all_rows) - set(roster_by_alias))
            raise RuntimeError(f"FULL tracking batches do not cover the exact roster (missing={missing}, extra={extra})")
        # Independently require every physical-clock identity to occur in one
        # and only one batch, with the complete roster membership.
        expected_groups = _describe_selected_groups(list(roster_by_alias), records)
        if set(group_owner) != {g["video_clock_group_sha256"] for g in expected_groups}:
            raise RuntimeError("FULL tracking batches omit or add video/clock groups")
        for group in expected_groups:
            owner = group_owner[group["video_clock_group_sha256"]]
            owned = next(item for item in batch_entries if item["path"].endswith(owner))
            barrier_value = read_json(ROOT / owned["path"])
            actual_aliases = [row["alias"] for row in barrier_value["records"]]
            if not set(group["aliases"]).issubset(set(actual_aliases)):
                raise RuntimeError("FULL tracking batch omitted aliases within a video/clock group")

    barrier = {
        "schema": FULL_BARRIER_SCHEMA,
        "status": "complete",
        "scope": "full official Table 1 query-independent native YOLO+SAM2 tracking; no scoring or fitting",
        "count": 13785,
        "domain_counts": {"hc2": 3482, "vidstg": 10303},
        "records": [all_rows[row["probe_alias"]] for row in records],
        "batch_barriers": batch_entries,
        "video_clock_group_count": len(group_owner),
        "full_source_runtime_sha256": locked["full_source_runtime_sha256"],
        "source_capture_barrier_sha256": locked["source_capture_barrier_sha256"],
        "tracking_runtime_sha256": locked["tracking_runtime_sha256"],
        "full_source_roster_sha256": locked["full_source_roster_sha256"],
        "input_manifest_sha256": locked["input_manifest_sha256"],
        "preflight_source_runtime_sha256": locked["preflight_source_runtime_sha256"],
        "preflight_tracking_runtime_sha256": locked["preflight_tracking_runtime_sha256"],
        "canonical_cache_root": "artifacts/stvg_final_table1_v1/native_cache",
        "query_independent_tracker_cache": True,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
        "GT_read": False,
        "query_used_by_tracker": False,
    }
    _immutable_json(TRACKING_BARRIER, barrier)
    _write_status({
        "schema": "stvg-final-table1-full-tracking-progress-v1",
        "status": "complete", "count": 13785, "GT_read": False,
        "barrier_sha256": sha256_file(TRACKING_BARRIER), "updated_at_unix": time.time(),
    })
    return {"status": "complete", "count": 13785,
            "batch_barrier_count": len(batch_entries),
            "video_clock_group_count": len(group_owner),
            "barrier_path": TRACKING_BARRIER.relative_to(ROOT).as_posix(),
            "barrier_sha256": sha256_file(TRACKING_BARRIER), "GT_read": False}


def metadata_contract() -> dict[str, Any]:
    """Read actual FULL manifest/roster metadata only; no locks or payloads."""
    roster, records = _validate_roster_metadata(allow_prepared=True)
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(_video_clock_group_sha256(record), []).append(record["probe_alias"])
    return {
        "schema": "stvg-final-table1-full-tracking-metadata-contracts-v1",
        "status": "pass",
        "roster_count": len(records),
        "hc2_count": sum(row["dataset"] == "hc2" for row in records),
        "vidstg_count": sum(row["dataset"] == "vidstg" for row in records),
        "unique_video_clock_groups": len(grouped),
        "largest_video_clock_group": max(map(len, grouped.values())),
        "full_source_roster_status": roster.get("status"),
        "GT_read": False,
        "source_case_payloads_read": False,
        "media_opened": False,
        "GPU_used": False,
    }


def contracts() -> dict[str, Any]:
    """Synthetic CPU contracts for grouping, cache reuse, and retirement archives."""
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU tracking contracts refuse an initialized CUDA context")

    rows = [
        {"probe_alias": "HC00000", "dataset": "hc2", "video_sha256": "a" * 64,
         "video_bounds": [10, 20], "fps": 25.0, "frame_count": 10, "width": 640, "height": 360},
        {"probe_alias": "HC00001", "dataset": "hc2", "video_sha256": "a" * 64,
         "video_bounds": [10, 20], "fps": 25.0, "frame_count": 10, "width": 640, "height": 360},
        {"probe_alias": "HC00002", "dataset": "hc2", "video_sha256": "a" * 64,
         "video_bounds": [11, 20], "fps": 25.0, "frame_count": 9, "width": 640, "height": 360},
        {"probe_alias": "V00000", "dataset": "vidstg", "video_sha256": "b" * 64,
         "video_bounds": [0, 12], "fps": 30.0, "frame_count": 12, "width": 1280, "height": 720},
    ]
    checks = 0
    assert _input_clock_key(rows[0]) == _input_clock_key(rows[1])
    checks += 1
    assert _input_clock_key(rows[0]) != _input_clock_key(rows[2])
    checks += 1
    assert _input_clock_key(rows[0]) != _input_clock_key(rows[3])
    checks += 1
    selected_aliases = ["HC00000", "HC00001", "HC00002"]
    selected_groups = _describe_selected_groups(selected_aliases, rows)
    assert len(selected_groups) == 2
    assert selected_groups[0]["aliases"] == ["HC00000", "HC00001"]
    assert selected_groups[1]["aliases"] == ["HC00002"]
    checks += 1
    assert _selection_sha256(selected_aliases) == sha256_bytes(canonical_json_bytes(selected_aliases))
    checks += 1
    try:
        _describe_selected_groups(["HC00000", "HC00002"], rows)
    except RuntimeError:
        checks += 1
    else:
        raise AssertionError("split video/clock group was accepted")
    try:
        _describe_selected_groups(["HC00001", "HC00000"], rows)
    except RuntimeError:
        checks += 1
    else:
        raise AssertionError("roster-reordered alias batch was accepted")

    # Official native request validation proves query-bearing requests differ
    # while the frozen tracking key remains query-independent.
    from scripts import run_stvg_table1_native_instance_adapter_v1 as adapter
    fingerprints = {
        "stvg_r1_revision": "rev",
        "grounded_sam2_revision": "rev",
        "dettrack_sha256": "1" * 64,
        "mask2marker_sha256": "2" * 64,
        "sam2_config_sha256": "3" * 64,
        "sam2_build_sha256": "4" * 64,
        "sam2_misc_sha256": "5" * 64,
        "sam2_transforms_sha256": "6" * 64,
        "mask_dictionary_sha256": "7" * 64,
        "detector_asset_sha256": "8" * 64,
        "sam2_asset_sha256": "9" * 64,
        "bounded_frame_cache_sha256": "a" * 64,
        "qkv_compat_sha256": "b" * 64,
        "adapter_sha256": ADAPTER_SHA256,
    }
    common = {
        "video_path": "/synthetic/canonical.mkv",
        "source_video_sha256": "c" * 64,
        "canonical_video_sha256": "d" * 64,
        "canonical_pixel_sha256": "e" * 64,
        "local_fps": 25.0,
        "local_frame_count": 2,
        "frame_map": [
            {"local_frame_index": 0, "physical_frame_id": 100, "source_timestamp_s": 4.0},
            {"local_frame_index": 1, "physical_frame_id": 101, "source_timestamp_s": 4.04},
        ],
    }
    req_a = dict(common, query_text="synthetic query A")
    req_b = dict(common, query_text="synthetic query B")
    norm_a = adapter._native().validate_request(req_a)
    norm_b = adapter._native().validate_request(req_b)
    assert norm_a["query_sha256"] != norm_b["query_sha256"]
    checks += 1
    assert adapter.tracking_key(req_a, fingerprints) == adapter.tracking_key(req_b, fingerprints)
    checks += 1

    # Exercise the new lossless archive reader in an isolated temporary tree.
    # No project cache, source-case payload, media, or annotations are touched.
    try:
        _safe_relative_workspace_path("../escape.json")
    except RuntimeError:
        checks += 1
    else:
        raise AssertionError("retirement path traversal was accepted")
    old_roots = (ROOT, FULL, CACHE_ROOT)
    try:
        with tempfile.TemporaryDirectory(prefix="stvg-full-tracking-archive-contract-") as temp:
            temp_root = Path(temp).resolve(strict=True)
            temp_full = temp_root / "artifacts/stvg_final_table1_v1/FULL"
            temp_cache = temp_root / "artifacts/stvg_final_table1_v1/native_cache"
            archive_dir = temp_full / "CACHE_RETIREMENTS"
            archive_dir.mkdir(parents=True)
            temp_cache.mkdir(parents=True)
            globals()["ROOT"], globals()["FULL"], globals()["CACHE_ROOT"] = temp_root, temp_full, temp_cache
            member_rel = (temp_cache / "tracking/synthetic/mask_0.json").relative_to(temp_root).as_posix()
            member_content = b'{"schema":"synthetic-cache-json"}\n'
            archive_path = archive_dir / "evidence.tar.gz"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                info = tarfile.TarInfo(member_rel)
                info.size = len(member_content)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(member_content))
            member_sha = sha256_bytes(member_content)
            archive_value = {
                "path": archive_path.relative_to(temp_root).as_posix(),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "members": [{"original_path": member_rel, "bytes": len(member_content), "sha256": member_sha}],
            }
            assert _read_archive_members(archive_value) == {member_rel: (len(member_content), member_sha)}
            checks += 1
            bad_receipt = dict(archive_value, members=[dict(archive_value["members"][0], sha256="0" * 64)])
            try:
                _read_archive_members(bad_receipt)
            except RuntimeError:
                checks += 1
            else:
                raise AssertionError("archive member byte mismatch was accepted")
            bad_members_archive = archive_dir / "symlink.tar.gz"
            with tarfile.open(bad_members_archive, mode="w:gz") as archive:
                info = tarfile.TarInfo(member_rel)
                info.type = tarfile.SYMTYPE
                info.linkname = "../escape"
                archive.addfile(info)
            symlink_receipt = {
                "path": bad_members_archive.relative_to(temp_root).as_posix(),
                "bytes": bad_members_archive.stat().st_size,
                "sha256": sha256_file(bad_members_archive),
                "members": [{"original_path": member_rel, "bytes": len(member_content), "sha256": member_sha}],
            }
            try:
                _read_archive_members(symlink_receipt)
            except RuntimeError:
                checks += 1
            else:
                raise AssertionError("archive symlink member was accepted")
            expected = {}
            _register_expected_evidence(expected, member_rel, len(member_content), member_sha)
            try:
                _register_expected_evidence(expected, member_rel, len(member_content) + 1, member_sha)
            except RuntimeError:
                checks += 1
            else:
                raise AssertionError("conflicting cache member bindings were accepted")
    finally:
        globals()["ROOT"], globals()["FULL"], globals()["CACHE_ROOT"] = old_roots

    return {
        "schema": "stvg-final-table1-full-tracking-cpu-contracts-v1",
        "status": "pass",
        "checks": checks,
        "retirement_archive_integrity": "pass",
        "query_free_cache_key": True,
        "source_group_includes_clock": True,
        "GT_read": False,
        "media_opened": False,
        "tracker_called": False,
        "model_loaded": False,
        "GPU_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--aliases-file", help="process one ordered bounded union of complete video/clock groups")
    choice.add_argument("--finalize", action="store_true", help="CPU-verify all batches and seal the 13,785-row barrier")
    choice.add_argument("--self-test", action="store_true", help="run synthetic CPU-only contracts")
    choice.add_argument("--metadata-contract", action="store_true",
                        help="validate the actual manifest/roster metadata only; no source payload/media")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(contracts(), sort_keys=True))
        return 0
    if args.metadata_contract:
        print(json.dumps(metadata_contract(), sort_keys=True))
        return 0
    result = finalize() if args.finalize else run_batch(args.aliases_file)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
