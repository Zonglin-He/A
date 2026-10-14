"""Full-roster, GT-free Table 1 source capture.

This is a separate extension around the frozen 64-query source adapter.  It
derives a deterministic roster from INPUT_MANIFEST.json, imports the frozen
capture_one implementation without editing it, and writes only below FULL/.
No labels, DINO, R1, fitting, or optimizer are used here.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
MANIFEST = BASE / "INPUT_MANIFEST.json"
PREFLIGHT_RUNTIME = BASE / "SOURCE_RUNTIME.json"
ROSTER = FULL / "FULL_SOURCE_ROSTER_LOCK.json"
RUNTIME = FULL / "SOURCE_RUNTIME.json"
OUT = FULL / "source_cases"
BARRIER = FULL / "SOURCE_CAPTURE_BARRIER.json"
STATUS = FULL / "SOURCE_CAPTURE_STATUS.json"
CONTRACTS = FULL / "source_contracts/CPU_CONTRACTS.json"
IDENTITY_READBACK = FULL / "MANIFEST_QUERY_IDENTITY_READBACK.json"

MANIFEST_REL = "artifacts/stvg_final_table1_v1/INPUT_MANIFEST.json"
PREFLIGHT_RUNTIME_REL = "artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json"
WORKER_REL = "scripts/run_stvg_table1_full_source_v1.py"
FROZEN_ADAPTER_REL = "scripts/run_stvg_table1_source_v1.py"
DOMAIN_ORDER = ("hc2", "vidstg")
DOMAIN_COUNTS = {"hc2": 3482, "vidstg": 10303}
EXPECTED_CHECKPOINTS = {
    "hc2": "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83",
    "vidstg": "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036",
}

sys.path.insert(0, str(ROOT))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: str | Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def query_identity_sha256(dataset: str, key: str) -> str:
    """Match the manifest's domain-namespaced PLAN query-key digest."""
    if dataset not in DOMAIN_COUNTS or not isinstance(key, str):
        raise ValueError("query identity requires a supported dataset and string PLAN key")
    return hashlib.sha256((dataset + "\0" + key).encode("utf-8")).hexdigest()


def project_manifest(manifest: Mapping[str, Any], manifest_sha256: str,
                     expected_counts: Mapping[str, int] = DOMAIN_COUNTS) -> dict[str, Any]:
    """Project every manifest row into an ordered, opaque-ID source roster."""
    if manifest.get("status") != "metadata_preflight_locked" or manifest.get("GT_read") is not False:
        raise ValueError("The full roster requires the sealed GT-free input manifest")
    if manifest.get("selection_uses_GT_or_scores") is not False:
        raise ValueError("The manifest must explicitly exclude GT/score-based selection")
    rows = manifest.get("records")
    if not isinstance(rows, list):
        raise ValueError("Manifest records must be a list")
    counts: dict[str, int] = collections.Counter()
    seen_ordinals: dict[str, set[int]] = {name: set() for name in DOMAIN_ORDER}
    seen_keys: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    required = (
        "dataset", "query_ordinal", "parent_sha256", "query_key_sha256",
        "safe_record_sha256", "plan_path", "plan_sha256", "plan_row_pointer",
        "subject_path", "subject_sha256", "video_sha256", "frame_ids", "fps",
        "width", "height", "frame_count", "duration", "video_bounds", "query_type",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each manifest record must be an object")
        ds = row.get("dataset")
        if ds not in expected_counts:
            raise ValueError(f"Unexpected Table 1 dataset: {ds!r}")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Manifest row is missing source metadata fields: {missing}")
        ordinal = row["query_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("query_ordinal must be a nonnegative integer")
        if ordinal in seen_ordinals[ds]:
            raise ValueError(f"Duplicate query ordinal in {ds}")
        seen_ordinals[ds].add(ordinal)
        key = (ds, row["query_key_sha256"])
        if key in seen_keys:
            raise ValueError(f"Duplicate query identity in {ds}")
        seen_keys.add(key)
        for digest_key in ("parent_sha256", "query_key_sha256", "safe_record_sha256",
                           "plan_sha256", "subject_sha256", "video_sha256"):
            if not _is_sha256(row[digest_key]):
                raise ValueError(f"Invalid manifest {digest_key}")
        frame_ids = row["frame_ids"]
        bounds = row["video_bounds"]
        if (not isinstance(frame_ids, list) or not frame_ids
                or any(isinstance(x, bool) or not isinstance(x, int) for x in frame_ids)
                or frame_ids != sorted(set(frame_ids))):
            raise ValueError("Source frame IDs must be a nonempty, sorted unique integer list")
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)
                or bounds[0] >= bounds[1]
                or frame_ids[0] < bounds[0] or frame_ids[-1] >= bounds[1]):
            raise ValueError("video_bounds must contain the source-grid frame IDs")
        if (isinstance(row["width"], bool) or not isinstance(row["width"], int) or row["width"] <= 0
                or isinstance(row["height"], bool) or not isinstance(row["height"], int) or row["height"] <= 0
                or isinstance(row["frame_count"], bool) or not isinstance(row["frame_count"], int)
                or row["frame_count"] <= 0
                or not isinstance(row["fps"], (int, float)) or not math.isfinite(row["fps"])
                or row["fps"] <= 0
                or not isinstance(row["duration"], (int, float)) or not math.isfinite(row["duration"])
                or row["duration"] <= 0):
            raise ValueError("Invalid source clock or image dimensions")
        records.append({key: row[key] for key in required})
        counts[ds] += 1

    if dict(counts) != dict(expected_counts):
        raise ValueError(f"Manifest domain counts differ: {dict(counts)}")
    for ds, count in expected_counts.items():
        if seen_ordinals[ds] != set(range(count)):
            raise ValueError(f"{ds} query ordinals are not the exact 0..N-1 roster")
    prefix = {"hc2": "HC", "vidstg": "V"}
    records.sort(key=lambda row: (DOMAIN_ORDER.index(row["dataset"]), row["query_ordinal"]))
    for record in records:
        record["probe_alias"] = f"{prefix[record['dataset']]}{record['query_ordinal']:05d}"
        record["GT_read"] = False
    aliases = [row["probe_alias"] for row in records]
    if len(set(aliases)) != len(records):
        raise ValueError("Generated opaque aliases are not unique")

    return {
        "schema": "stvg-final-table1-full-source-roster-v1",
        "status": "prepared_for_root_runtime_lock",
        "selection_rule": "All and only the locked INPUT_MANIFEST records; no filtering, selection, or outcome ordering. Sort by fixed domain order (hc2, vidstg), then query_ordinal. Alias is HC or V plus the zero-padded five-digit original domain ordinal.",
        "historical_exposure": True,
        "fresh_confirmation_claim": False,
        "manifest_path": MANIFEST_REL,
        "manifest_sha256": manifest_sha256,
        "preflight_roster_path": "artifacts/stvg_final_table1_v1/PREFLIGHT_ROSTER_LOCK.json",
        "preflight_source_runtime_path": PREFLIGHT_RUNTIME_REL,
        "official_query_count": len(records),
        "domain_counts": dict(counts),
        "all_manifest_rows_preserved": True,
        "selection_uses_GT_or_scores": False,
        "records": records,
        "GT_read": False,
    }


def prepare_roster() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    identity = verify_manifest_query_identity(manifest)
    roster = project_manifest(manifest, sha256_file(MANIFEST))
    if ROSTER.exists():
        existing = read_json(ROSTER)
        stable_keys = ("schema", "selection_rule", "historical_exposure", "fresh_confirmation_claim",
                       "manifest_path", "manifest_sha256", "official_query_count", "domain_counts",
                       "all_manifest_rows_preserved", "selection_uses_GT_or_scores", "records", "GT_read")
        if any(existing.get(key) != roster.get(key) for key in stable_keys):
            raise RuntimeError("Existing full source roster differs; preserve and investigate")
        return {"path": str(ROSTER.relative_to(ROOT)), "sha256": sha256_file(ROSTER),
                "count": len(roster["records"]), "reused": True,
                "query_identity_readback_path": str(IDENTITY_READBACK.relative_to(ROOT)),
                "query_identity_readback_sha256": sha256_file(IDENTITY_READBACK),
                "query_identity_count": identity["count"]}
    write_json(ROSTER, roster)
    return {"path": str(ROSTER.relative_to(ROOT)), "sha256": sha256_file(ROSTER),
            "count": len(roster["records"]), "reused": False,
            "query_identity_readback_path": str(IDENTITY_READBACK.relative_to(ROOT)),
            "query_identity_readback_sha256": sha256_file(IDENTITY_READBACK),
            "query_identity_count": identity["count"]}


def verify_manifest_query_identity(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check all opaque manifest query hashes against official PLAN key fields.

    Only the query key is read from each PLAN row. Raw keys/captions and any
    annotation fields are never copied to the receipt.
    """
    import hashlib as _hashlib

    manifest_obj = read_json(MANIFEST) if manifest is None else manifest
    manifest_sha = sha256_file(MANIFEST)
    projected = project_manifest(manifest_obj, manifest_sha)
    records = projected["records"]
    plan_cache: dict[str, dict[str, Any]] = {}
    plan_paths: dict[str, str] = {}
    plan_hashes: dict[str, str] = {}
    readback_rows: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = collections.Counter()
    for record in records:
        ds = record["dataset"]
        path = record["plan_path"]
        if ds not in plan_cache:
            plan_file = ROOT / path
            if not plan_file.is_file() or sha256_file(plan_file) != record["plan_sha256"]:
                raise RuntimeError(f"Official PLAN bytes differ from manifest for {ds}")
            plan = read_json(plan_file)
            if plan.get("dataset") != ds or not isinstance(plan.get("rows"), list):
                raise RuntimeError(f"Official PLAN schema differs for {ds}")
            if len(plan["rows"]) != DOMAIN_COUNTS[ds]:
                raise RuntimeError(f"Official PLAN row count differs for {ds}")
            plan_cache[ds] = plan
            plan_paths[ds] = path
            plan_hashes[ds] = record["plan_sha256"]
        elif (record["plan_path"] != plan_paths[ds]
              or record["plan_sha256"] != plan_hashes[ds]):
            raise RuntimeError(f"Manifest changes official PLAN binding within {ds}")
        plan = plan_cache[ds]
        ordinal = int(record["query_ordinal"])
        row = plan["rows"][ordinal]
        key = row.get("key")
        if (row.get("ordinal") != ordinal or not isinstance(key, str)
                or record.get("plan_row_pointer") != f"{path}#rows[{ordinal}]"
                or record.get("query_key_sha256") != query_identity_sha256(ds, key)):
            raise RuntimeError(f"Opaque manifest query identity failed PLAN join: {record['probe_alias']}")
        source = row.get("input", {})
        expected_bounds = [int(source.get("start_frame", 0)),
                           int(source.get("end_frame", source.get("frame_count", -1)))]
        if (source.get("video_sha256") != record["video_sha256"]
                or list(source.get("frame_ids", [])) != record["frame_ids"]
                or source.get("fps") != record["fps"]
                or source.get("width") != record["width"]
                or source.get("height") != record["height"]
                or source.get("frame_count") != record["frame_count"]
                or source.get("duration") != record["duration"]
                or expected_bounds != record["video_bounds"]):
            raise RuntimeError(f"Manifest source clock/input mismatch: {record['probe_alias']}")
        computed = _hashlib.sha256((ds + "\0" + key).encode("utf-8")).hexdigest()
        readback_rows.append({
            "probe_alias": record["probe_alias"], "dataset": ds,
            "query_ordinal": ordinal,
            "manifest_query_key_sha256": record["query_key_sha256"],
            "plan_key_sha256": computed,
            "match": computed == record["query_key_sha256"],
        })
        domain_counts[ds] += 1
    receipt = {
        "schema": "stvg-final-table1-manifest-query-identity-readback-v1",
        "status": "pass",
        "scope": "CPU metadata-only PLAN-key to INPUT_MANIFEST query hash and source-clock join; no media, annotation, label, or model reads",
        "manifest_path": MANIFEST_REL,
        "manifest_sha256": manifest_sha,
        "query_key_hash": "sha256((dataset + NUL + PLAN.rows[query_ordinal].key).encode('utf-8'))",
        "plan_bindings": {
            ds: {"path": plan_paths[ds], "sha256": plan_hashes[ds],
                 "rows": domain_counts[ds]} for ds in DOMAIN_ORDER
        },
        "count": len(readback_rows),
        "domain_counts": dict(domain_counts),
        "identity_matches": sum(row["match"] for row in readback_rows),
        "source_clock_input_matches": len(readback_rows),
        "records": readback_rows,
        "GT_read": False,
        "media_opened": False,
        "models_loaded": False,
        "GPU_used": False,
    }
    if receipt["identity_matches"] != len(readback_rows):
        raise RuntimeError("Not all manifest query identity hashes match")
    if IDENTITY_READBACK.exists():
        old = read_json(IDENTITY_READBACK)
        if old != receipt:
            raise RuntimeError("Existing query identity receipt differs; preserve and investigate")
    else:
        write_json(IDENTITY_READBACK, receipt)
    return receipt


def _load_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(MANIFEST)
    manifest_sha = sha256_file(MANIFEST)
    expected = project_manifest(manifest, manifest_sha)
    roster = read_json(ROSTER)
    if roster.get("status") not in {"prepared_for_root_runtime_lock", "locked"}:
        raise RuntimeError("Full source roster is not prepared or locked")
    for key in ("schema", "manifest_sha256", "official_query_count", "domain_counts",
                "all_manifest_rows_preserved", "selection_uses_GT_or_scores", "records", "GT_read"):
        if roster.get(key) != expected.get(key):
            raise RuntimeError(f"Full source roster no longer matches INPUT_MANIFEST: {key}")
    if roster.get("GT_read") is not False:
        raise RuntimeError("Full source roster must remain GT-free")
    preflight = read_json(PREFLIGHT_RUNTIME)
    if preflight.get("status") != "locked" or preflight.get("GT_read") is not False:
        raise RuntimeError("Original 64-query source runtime is unavailable or not GT-free")
    return roster, roster["records"]


def verify_runtime() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not RUNTIME.is_file():
        raise RuntimeError("Root must create FULL/SOURCE_RUNTIME.json after reviewing this worker")
    runtime = read_json(RUNTIME)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("Full SOURCE_RUNTIME must be locked with GT_read=false")
    roster, records = _load_records()
    manifest_sha = sha256_file(MANIFEST)
    roster_sha = sha256_file(ROSTER)
    preflight_runtime_sha = sha256_file(PREFLIGHT_RUNTIME)
    if runtime.get("full_source_roster_sha256") != roster_sha:
        raise RuntimeError("Full SOURCE_RUNTIME does not bind FULL_SOURCE_ROSTER_LOCK")
    if runtime.get("input_manifest_sha256") != manifest_sha:
        raise RuntimeError("Full SOURCE_RUNTIME does not bind INPUT_MANIFEST")
    if runtime.get("preflight_source_runtime_sha256") != preflight_runtime_sha:
        raise RuntimeError("Full SOURCE_RUNTIME does not bind the prior 64-query source runtime")
    identity = verify_manifest_query_identity()
    if runtime.get("query_identity_readback_sha256") != sha256_file(IDENTITY_READBACK):
        raise RuntimeError("Full SOURCE_RUNTIME does not bind the 13,785-row identity readback")
    if identity.get("count") != 13785 or identity.get("identity_matches") != 13785:
        raise RuntimeError("Full manifest query identity readback is incomplete")
    if runtime.get("official_query_count") != len(records):
        raise RuntimeError("Full SOURCE_RUNTIME query count differs from the fixed full roster")
    if runtime.get("domain_counts") != roster["domain_counts"]:
        raise RuntimeError("Full SOURCE_RUNTIME domain counts differ from the locked roster")

    pins = runtime.get("pins")
    protected = runtime.get("protected", {})
    if not isinstance(pins, dict) or not pins or not isinstance(protected, dict):
        raise RuntimeError("Full SOURCE_RUNTIME pins and protected mappings are required")
    required_pins = {WORKER_REL, FROZEN_ADAPTER_REL}
    if not required_pins.issubset(pins):
        raise RuntimeError("Full SOURCE_RUNTIME must pin both the extension and frozen capture adapter")
    for rel, expected_sha in {**pins, **protected}.items():
        path = ROOT / rel
        if not _is_sha256(expected_sha) or not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Full runtime source/protected binding mismatch: {rel}")

    checkpoint_rows = runtime.get("source_checkpoints")
    if not isinstance(checkpoint_rows, dict):
        raise RuntimeError("Full SOURCE_RUNTIME must bind both frozen source checkpoints")
    preflight_runtime = read_json(PREFLIGHT_RUNTIME)
    for domain, digest in EXPECTED_CHECKPOINTS.items():
        row = checkpoint_rows.get(domain)
        if not isinstance(row, dict):
            raise RuntimeError(f"Full runtime checkpoint row must include path and SHA for {domain}")
        path_value = row.get("path")
        actual = row.get("sha256")
        old_row = preflight_runtime.get("source_checkpoints", {}).get(domain)
        old_sha = old_row.get("sha256") if isinstance(old_row, dict) else old_row
        if actual != digest or old_sha != digest:
            raise RuntimeError(f"Frozen source checkpoint binding mismatch for {domain}")
        old_path = old_row.get("path") if isinstance(old_row, dict) else None
        if (not isinstance(path_value, str) or path_value != old_path
                or not (ROOT / path_value).is_file()):
            raise RuntimeError(f"Checkpoint path differs from original runtime for {domain}")
        if sha256_file(ROOT / path_value) != digest:
            raise RuntimeError(f"Checkpoint bytes changed for {domain}")

    # The old preflight runtime is an independent immutable reference, not the
    # runtime used by this capture.  Its code/dependency pins must still hold.
    preflight = preflight_runtime
    for rel, digest in {**preflight.get("pins", {}), **preflight.get("protected", {})}.items():
        full_digest = runtime.get("pins", {}).get(rel, runtime.get("protected", {}).get(rel))
        if full_digest != digest:
            raise RuntimeError(f"Full runtime omits an original source preflight pin: {rel}")
    for rel, expected_sha in {**preflight.get("pins", {}), **preflight.get("protected", {})}.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Original source preflight binding changed: {rel}")
    return runtime, roster, records


def _load_frozen_adapter():
    from scripts import run_stvg_table1_source_v1 as adapter
    # capture_one reads this module global to bind each output to its runtime.
    # Redirect only in-memory for this separate process; its frozen file and
    # the original 64-query outputs remain untouched.
    adapter.RUNTIME = RUNTIME
    return adapter


def _validate_saved_case(adapter: Any, record: Mapping[str, Any], runtime_sha: str,
                         roster_sha: str, manifest_sha: str, torch: Any) -> dict[str, Any]:
    alias = record["probe_alias"]
    path = OUT / f"{alias}.pt"
    receipt_path = path.with_suffix(".receipt.json")
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Output is missing or unsealed; preserve it: {alias}")
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "stvg-final-table1-full-source-receipt-v1"
            or receipt.get("case_id") != alias
            or receipt.get("sha256") != sha256_file(path)
            or receipt.get("bytes") != path.stat().st_size
            or receipt.get("source_runtime_sha256") != runtime_sha
            or receipt.get("full_source_roster_sha256") != roster_sha
            or receipt.get("input_manifest_sha256") != manifest_sha
            or receipt.get("query_key_sha256") != record["query_key_sha256"]
            or receipt.get("GT_read") is not False):
        raise RuntimeError(f"Sealed receipt binding mismatch: {alias}")
    case = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(case, dict)
            or case.get("case_id") != alias
            or case.get("dataset") != record["dataset"]
            or case.get("query_ordinal") != record["query_ordinal"]
            or case.get("query_key_sha256") != record["query_key_sha256"]
            or case.get("parent_sha256") != record["parent_sha256"]
            or case.get("source_runtime_sha256") != runtime_sha
            or case.get("source_case_sha256") != receipt.get("source_case_sha256")
            or case.get("source_case_sha256") != adapter.content_sha256(case)
            or case.get("input", {}).get("video_sha256") != record["video_sha256"]
            or case.get("GT_read") is not False
            or case.get("DINO_read_or_call") is not False
            or int(case.get("optimizer_steps", -1)) != 0
            or int(case.get("backwards", -1)) != 0):
        raise RuntimeError(f"Saved source case canonical binding mismatch: {alias}")
    summary = {
        "path": str(path.relative_to(ROOT)), "sha256": receipt["sha256"],
        "bytes": receipt["bytes"], "source_case_sha256": receipt["source_case_sha256"],
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha256_file(receipt_path), "reused": True,
    }
    del case
    return summary


def _save_new_case(adapter: Any, record: Mapping[str, Any], runtime_sha: str,
                   roster_sha: str, manifest_sha: str, model: Any,
                   frame_cache: Any, torch: Any) -> dict[str, Any]:
    alias = record["probe_alias"]
    path = OUT / f"{alias}.pt"
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        raise RuntimeError(f"Existing partial output is preserved; refuse overwrite: {alias}")
    case = adapter.capture_one(model, record, frame_cache)
    if (case.get("case_id") != alias
            or case.get("source_runtime_sha256") != runtime_sha
            or case.get("query_key_sha256") != record["query_key_sha256"]
            or case.get("input", {}).get("video_sha256") != record["video_sha256"]
            or case.get("GT_read") is not False
            or case.get("DINO_read_or_call") is not False
            or adapter.content_sha256(case) != case.get("source_case_sha256")):
        raise RuntimeError(f"Frozen capture returned an unexpected or non-GT-free source case: {alias}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(dict(case), temp)
    os.replace(temp, path)
    payload_sha = sha256_file(path)
    receipt = {
        "schema": "stvg-final-table1-full-source-receipt-v1",
        "case_id": alias,
        "sha256": payload_sha,
        "bytes": path.stat().st_size,
        "source_case_sha256": case["source_case_sha256"],
        "source_runtime_sha256": runtime_sha,
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "query_key_sha256": record["query_key_sha256"],
        "GT_read": False,
        "DINO_read_or_call": False,
        "time": time.time(),
    }
    # If interrupted between payload and receipt, the payload is intentionally
    # preserved and the next invocation refuses to overwrite it.
    write_json(receipt_path, receipt)
    return {
        "path": str(path.relative_to(ROOT)), "sha256": payload_sha,
        "bytes": path.stat().st_size, "source_case_sha256": case["source_case_sha256"],
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha256_file(receipt_path), "reused": False,
        "actual_source_offset_forwards": int(case["compute"]["actual_source_offset_forwards"]),
    }


def _validate_existing_barrier(runtime_sha: str, roster_sha: str, manifest_sha: str,
                               expected_aliases: list[str]) -> dict[str, Any] | None:
    if not BARRIER.exists():
        return None
    barrier = read_json(BARRIER)
    if (barrier.get("schema") != "stvg-final-table1-full-source-capture-barrier-v1"
            or barrier.get("status") != "sealed"
            or barrier.get("GT_read") is not False
            or barrier.get("source_runtime_sha256") != runtime_sha
            or barrier.get("full_source_roster_sha256") != roster_sha
            or barrier.get("input_manifest_sha256") != manifest_sha
            or barrier.get("count") != len(expected_aliases)
            or set(barrier.get("source_files", {})) != set(expected_aliases)):
        raise RuntimeError("Existing full source barrier is incompatible; preserve and investigate")
    return barrier


def run_all() -> dict[str, Any]:
    runtime, roster, records = verify_runtime()
    runtime_sha, roster_sha, manifest_sha = sha256_file(RUNTIME), sha256_file(ROSTER), sha256_file(MANIFEST)
    aliases = [row["probe_alias"] for row in records]
    if len(records) != 13785 or len(set(aliases)) != 13785:
        raise RuntimeError("The full Table 1 run requires all 13,785 unique official rows")
    existing_barrier = _validate_existing_barrier(runtime_sha, roster_sha, manifest_sha, aliases)
    if existing_barrier is not None:
        import torch
        adapter = _load_frozen_adapter()
        for record in records:
            alias = record["probe_alias"]
            actual = _validate_saved_case(adapter, record, runtime_sha, roster_sha,
                                          manifest_sha, torch)
            bound = existing_barrier["source_files"][alias]
            for key in ("path", "sha256", "bytes", "source_case_sha256",
                        "receipt_path", "receipt_sha256"):
                if actual.get(key) != bound.get(key):
                    raise RuntimeError(f"Existing sealed barrier file binding differs: {alias}/{key}")
        return {"status": "already_sealed", "count": 13785,
                "barrier_sha256": sha256_file(BARRIER), "GT_read": False}

    import torch
    adapter = _load_frozen_adapter()
    source_cases: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for record in records:
        path = OUT / f"{record['probe_alias']}.pt"
        receipt_path = path.with_suffix(".receipt.json")
        if path.exists() or receipt_path.exists():
            source_cases[record["probe_alias"]] = _validate_saved_case(
                adapter, record, runtime_sha, roster_sha, manifest_sha, torch)
        else:
            missing.append(record)

    if not missing:
        _seal(runtime, records, source_cases, runtime_sha, roster_sha, manifest_sha,
              model_loads=0, captures=0, resumed=len(records), offset_forwards=0)
        return {"status": "sealed_from_existing_receipts", "count": 13785,
                "barrier_sha256": sha256_file(BARRIER), "GT_read": False}

    from methods.decota_final_simplified_v1.config import MethodConfig
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_decota_paper_main_v1 import gpu, model_for
    lease = gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    try:
        import numpy as np
        np.random.seed(20261015)
    except ImportError:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = None
    loaded_source = None
    model_sha = None
    model_loads = 0
    capture_calls = 0
    offset_forwards = 0
    resumed_count = len(source_cases)
    started = time.time()
    try:
        # records are fixed domain/ordinal ordered, so source model lifetime is
        # one hot load per domain (two loads total), never one load per query.
        for index, record in enumerate(missing, 1):
            ds = record["dataset"]
            source_name = adapter.SOURCE_MODEL_NAME[ds]
            source_direction = adapter.SOURCE_DIRECTION[ds]
            cfg = MethodConfig.for_direction(source_direction)
            checkpoint_row = runtime["source_checkpoints"][ds]
            checkpoint_sha = checkpoint_row.get("sha256") if isinstance(checkpoint_row, dict) else checkpoint_row
            if checkpoint_sha != cfg.checkpoint_sha256 or checkpoint_sha != EXPECTED_CHECKPOINTS[ds]:
                raise RuntimeError(f"Runtime checkpoint does not match frozen config for {ds}")
            if loaded_source != source_name:
                if model is not None:
                    if state_hash(model.state_dict()) != model_sha:
                        raise RuntimeError("Frozen source model changed across full-roster captures")
                    del model
                    model = None
                    gc.collect()
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                model = model_for(source_name)
                loaded_source = source_name
                model_sha = model._fixed_full_state_hash
                model_loads += 1
            frame_cache = collections.OrderedDict()
            torch.cuda.reset_peak_memory_stats()
            result = _save_new_case(adapter, record, runtime_sha, roster_sha,
                                    manifest_sha, model, frame_cache, torch)
            source_cases[record["probe_alias"]] = {key: result[key] for key in (
                "path", "sha256", "bytes", "source_case_sha256", "receipt_path", "receipt_sha256", "reused")}
            capture_calls += 1
            offset_forwards += result["actual_source_offset_forwards"]
            frame_cache.clear()
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            write_json(STATUS, {
                "schema": "stvg-final-table1-full-source-progress-v1",
                "status": "running",
                "done_this_invocation": index,
                "total_this_invocation": len(missing),
                "sealed_receipts_total": len(source_cases),
                "official_total": len(records),
                "current_alias": record["probe_alias"],
                "current_domain": ds,
                "source_model": source_name,
                "source_model_loads_this_invocation": model_loads,
                "source_capture_calls_this_invocation": capture_calls,
                "DINO_calls": 0,
                "optimizer_steps": 0,
                "backwards": 0,
                "GT_read": False,
                "source_runtime_sha256": runtime_sha,
                "full_source_roster_sha256": roster_sha,
                "input_manifest_sha256": manifest_sha,
                "started_at_unix": started,
                "updated_at_unix": time.time(),
            }, replace=True)
            print(f"TABLE1_FULL_SOURCE {index}/{len(missing)} {record['probe_alias']} "
                  f"resumed={resumed_count} loads={model_loads}", flush=True)
        if model is not None and state_hash(model.state_dict()) != model_sha:
            raise RuntimeError("Frozen source model state changed during full capture")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        lease.close()
    if len(source_cases) != 13785:
        raise RuntimeError("Not every full-roster source receipt is sealed")
    _seal(runtime, records, source_cases, runtime_sha, roster_sha, manifest_sha,
          model_loads=model_loads, captures=capture_calls,
          resumed=resumed_count, offset_forwards=offset_forwards)
    return {"status": "sealed", "count": 13785,
            "barrier_sha256": sha256_file(BARRIER), "GT_read": False}


def _seal(runtime: Mapping[str, Any], records: list[dict[str, Any]], source_cases: dict[str, Any],
          runtime_sha: str, roster_sha: str, manifest_sha: str, *, model_loads: int,
          captures: int, resumed: int, offset_forwards: int) -> None:
    if BARRIER.exists():
        raise FileExistsError(f"Preserve existing barrier: {BARRIER}")
    if (len(records) != 13785 or len(source_cases) != len(records)
            or set(source_cases) != {row["probe_alias"] for row in records}):
        raise RuntimeError("Cannot seal an incomplete or noncanonical full source roster")
    barrier = {
        "schema": "stvg-final-table1-full-source-capture-barrier-v1",
        "status": "sealed",
        "scope": "full official GT-free DINO-free frozen source capture",
        "count": len(records),
        "domain_counts": runtime["domain_counts"],
        "source_runtime_sha256": runtime_sha,
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": manifest_sha,
        "preflight_source_runtime_sha256": sha256_file(PREFLIGHT_RUNTIME),
        "source_files": source_cases,
        "actual_source_model_loads_this_invocation": model_loads,
        "actual_source_capture_calls_this_invocation": captures,
        "reused_sealed_receipts_this_invocation": resumed,
        "actual_source_offset_forwards_this_invocation": offset_forwards,
        "DINO_calls": 0,
        "optimizer_steps": 0,
        "backwards": 0,
        "GT_read": False,
        "time": time.time(),
    }
    write_json(BARRIER, barrier)
    write_json(STATUS, {
        "schema": "stvg-final-table1-full-source-progress-v1",
        "status": "sealed", "count": len(records), "GT_read": False,
        "barrier_sha256": sha256_file(BARRIER), "time": time.time(),
    }, replace=True)


def contracts() -> dict[str, Any]:
    """Small synthetic manifest ABI tests; no project rows, models, or CUDA."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("CPU manifest contracts refuse an initialized CUDA context")

    def record(ds: str, ordinal: int) -> dict[str, Any]:
        digest = hashlib.sha256(f"{ds}:{ordinal}".encode()).hexdigest()
        return {
            "dataset": ds, "query_ordinal": ordinal,
            "parent_sha256": digest, "query_key_sha256": hashlib.sha256((digest+"q").encode()).hexdigest(),
            "safe_record_sha256": hashlib.sha256((digest+"r").encode()).hexdigest(),
            "plan_path": f"plan/{ds}.json", "plan_sha256": hashlib.sha256((digest+"p").encode()).hexdigest(),
            "plan_row_pointer": f"plan/{ds}.json#rows[{ordinal}]",
            "subject_path": f"subjects/{ds}/{ordinal}.json", "subject_sha256": hashlib.sha256((digest+"s").encode()).hexdigest(),
            "video_sha256": hashlib.sha256((digest+"v").encode()).hexdigest(),
            "frame_ids": [0, 2, 4], "fps": 25.0, "width": 640, "height": 480,
            "frame_count": 5, "duration": 0.2, "video_bounds": [0, 5], "query_type": "synthetic",
        }

    rows = [record(ds, i) for ds, n in (("hc2", 2), ("vidstg", 3)) for i in range(n)]
    fake = {"status": "metadata_preflight_locked", "GT_read": False,
            "selection_uses_GT_or_scores": False, "records": rows}
    projected = project_manifest(fake, "a" * 64, {"hc2": 2, "vidstg": 3})
    aliases = [row["probe_alias"] for row in projected["records"]]
    assert aliases == ["HC00000", "HC00001", "V00000", "V00001", "V00002"]
    assert query_identity_sha256("hc2", "full_hc2:synthetic") == hashlib.sha256(
        b"hc2\0full_hc2:synthetic").hexdigest()
    assert [(row["dataset"], row["query_ordinal"]) for row in projected["records"]] == [
        ("hc2", 0), ("hc2", 1), ("vidstg", 0), ("vidstg", 1), ("vidstg", 2)]
    # Input order can vary; canonical aliases and row order do not.
    shuffled = dict(fake, records=[rows[4], rows[0], rows[3], rows[1], rows[2]])
    again = project_manifest(shuffled, "a" * 64, {"hc2": 2, "vidstg": 3})
    assert projected["records"] == again["records"]
    for bad in (
        dict(fake, GT_read=True),
        dict(fake, selection_uses_GT_or_scores=True),
        dict(fake, records=rows[:-1]),
        dict(fake, records=rows + [rows[0]]),
        dict(fake, records=[dict(rows[0], query_ordinal=7), *rows[1:]]),
        dict(fake, records=[dict(rows[0], frame_ids=[4, 2, 0]), *rows[1:]]),
        dict(fake, records=[dict(rows[0], video_sha256="not-a-hash"), *rows[1:]]),
    ):
        try:
            project_manifest(bad, "a" * 64, {"hc2": 2, "vidstg": 3})
        except (ValueError, TypeError):
            continue
        raise AssertionError("Malformed manifest fixture was accepted")
    return {
        "schema": "stvg-final-table1-full-source-cpu-contracts-v1",
        "status": "pass",
        "contracts": 9,
        "manifest_rows": 5,
        "fixtures": ["canonical domain ordinals", "zero-padded aliases", "stable ordering",
                     "permutation invariance", "GT-read rejection", "outcome-selection rejection",
                     "count/duplicate rejection", "source-grid rejection", "digest rejection"],
        "GPU_used": False, "models_loaded": False, "media_opened": False, "GT_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-roster", action="store_true")
    group.add_argument("--contracts", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.prepare_roster:
        print(json.dumps(prepare_roster(), sort_keys=True), flush=True)
    elif args.contracts:
        result = contracts()
        write_json(CONTRACTS, result, replace=True)
        print(f"TABLE1_FULL_SOURCE_CPU_CONTRACTS_PASS {result['contracts']}", flush=True)
    else:
        print(json.dumps(run_all(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
