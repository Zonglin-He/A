#!/usr/bin/env python3
"""Synthetic end-to-end CPU test for FULL tracking cache retirement.

The fixture monkeypatches the retirement modules to a temporary directory.
It contains only small fabricated JSON/text bytes and never opens real cache,
source-case, media, model, or label payloads.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/stvg_final_table1_v1/FULL/engineering/cache_retirement_synthetic_cpu_contracts_v1.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha_bytes(value), len(value)


def _write_json(path: Path, value: Any) -> tuple[str, int]:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return _write_bytes(path, payload)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _make_cache_case(root: Path, cache_root: Path, cache_name: str,
                     source_video_sha: str, dataset: str = "hc2") -> dict[str, Any]:
    cache_dir = cache_root / "tracking" / cache_name
    mask_dir = cache_dir / "masks"
    canonical_dir = cache_root / "canonical" / cache_name
    marker_path = cache_dir / "marked.mp4"
    canonical_path = canonical_dir / "input.mkv"
    canonical_receipt_path = canonical_dir / "CANONICAL_RECEIPT.json"
    mask_path = mask_dir / "mask_0.json"
    accepted_path = cache_dir / "ACCEPTED.json"
    inventory_path = cache_dir / "MASK_INVENTORY.json"

    canonical_video_sha, canonical_video_bytes = _write_bytes(canonical_path, b"synthetic-canonical-video")
    pixel_sha = _sha_bytes(b"synthetic-rgb-frame-clock")
    frame_clock_sha = _sha_bytes(b"synthetic-frame-clock")
    marker_sha, marker_bytes = _write_bytes(marker_path, b"synthetic-marked-video")
    mask_sha, mask_bytes = _write_json(mask_path, {"synthetic": True, "frame_index": 0})
    tracking_key = _sha_bytes(f"tracking:{cache_name}".encode("utf-8"))

    inventory = {
        "schema": "stvg-table1-query-independent-mask-inventory-v1",
        "mask_count": 1,
        "known_ids": [7],
        "files": [{"index": 0, "path": _rel(mask_path, root), "sha256": mask_sha, "bytes": mask_bytes}],
        "tracking_cache_key": tracking_key,
        "GT_read": False,
        "query_independent": True,
    }
    inventory_sha, inventory_bytes = _write_json(inventory_path, inventory)
    accepted = {
        "schema": "stvg-table1-query-independent-tracking-v1",
        "status": "pass",
        "GT_read": False,
        "query_independent": True,
        "tracking_cache_key": tracking_key,
        "tracking_binding_sha256": _sha_bytes(f"query-free-binding:{cache_name}".encode("utf-8")),
        "fingerprints": {"synthetic_cpu_fixture": _sha_bytes(b"fixed")},
        "source_video_sha256": source_video_sha,
        "canonical_video_sha256": canonical_video_sha,
        "canonical_pixel_sha256": pixel_sha,
        "frame_clock_sha256": frame_clock_sha,
        "local_fps": 25.0,
        "local_frame_count": 1,
        "target_mode": "all",
        "marker_video_path": _rel(marker_path, root),
        "marker_video_sha256": marker_sha,
        "mask_data_dir": _rel(mask_dir, root),
        "mask_inventory_path": _rel(inventory_path, root),
        "mask_inventory_sha256": inventory_sha,
    }
    accepted_sha, accepted_bytes = _write_json(accepted_path, accepted)
    canonical_receipt = {
        "schema": "stvg-table1-canonical-input-v1",
        "status": "complete",
        "canonical_video_sha256": canonical_video_sha,
        "canonical_pixel_sha256": pixel_sha,
        "source_video_sha256": source_video_sha,
        "canonical_video_bytes": canonical_video_bytes,
        "GT_read": False,
        "CPU_only": True,
        "query_independent": True,
    }
    receipt_sha, _ = _write_json(canonical_receipt_path, canonical_receipt)
    return {
        "cache_name": cache_name,
        "tracking_key": tracking_key,
        "accepted": accepted,
        "inventory": inventory,
        "canonical_video_sha256": canonical_video_sha,
        "canonical_pixel_sha256": pixel_sha,
        "frame_clock_sha256": frame_clock_sha,
        "canonical_video_bytes": canonical_video_bytes,
        "canonical_video_path": _rel(canonical_path, root),
        "canonical_receipt_path": _rel(canonical_receipt_path, root),
        "canonical_receipt_sha256": receipt_sha,
        "marker_video_path": _rel(marker_path, root),
        "marker_video_sha256": marker_sha,
        "marker_video_bytes": marker_bytes,
        "mask_data_dir": _rel(mask_dir, root),
        "mask_path": _rel(mask_path, root),
        "mask_sha256": mask_sha,
        "mask_bytes": mask_bytes,
        "accepted_path": _rel(accepted_path, root),
        "accepted_sha256": accepted_sha,
        "accepted_bytes": accepted_bytes,
        "inventory_path": _rel(inventory_path, root),
        "inventory_sha256": inventory_sha,
        "inventory_bytes": inventory_bytes,
    }


def _make_full_row(root: Path, full: Path, case: dict[str, Any], alias: str,
                   source_video_sha: str, runtime: dict[str, str]) -> dict[str, Any]:
    request_path = full / "tracking_cases" / alias / "REQUEST.json"
    request_sha, request_bytes = _write_json(request_path, {"synthetic_private_query_fixture": True})
    binding_path = full / "tracking_cases" / alias / "TRACKING_BINDING.json"
    query_key_sha = _sha_bytes(f"query-key:{alias}".encode())
    source_case_sha = _sha_bytes(f"source-case:{alias}".encode())
    binding = {
        "schema": "stvg-final-table1-tracking-binding-v1",
        "status": "pass",
        "case_alias": alias,
        "dataset": "hc2",
        "query_key_sha256": query_key_sha,
        "query_text_sha256": _sha_bytes(f"synthetic-caption:{alias}".encode()),
        "source_case_sha256": source_case_sha,
        "request_path": _rel(request_path, root),
        "request_sha256": request_sha,
        "source_video_sha256": source_video_sha,
        "canonical_video_sha256": case["canonical_video_sha256"],
        "canonical_pixel_sha256": case["canonical_pixel_sha256"],
        "frame_clock_sha256": case["frame_clock_sha256"],
        "canonical_video_bytes": case["canonical_video_bytes"],
        "canonical_receipt_path": case["canonical_receipt_path"],
        "canonical_receipt_sha256": case["canonical_receipt_sha256"],
        "tracking_cache_key": case["tracking_key"],
        "tracking_binding_sha256": case["accepted"]["tracking_binding_sha256"],
        "tracking_fingerprints": case["accepted"]["fingerprints"],
        "target_mode": "all",
        "marker_video_path": case["marker_video_path"],
        "marker_video_sha256": case["marker_video_sha256"],
        "inventory_path": case["inventory_path"],
        "inventory_sha256": case["inventory_sha256"],
        "mask_count": 1,
        "mask_ids": [7],
        "accepted_path": case["accepted_path"],
        "accepted_sha256": case["accepted_sha256"],
        "accepted_bytes": case["accepted_bytes"],
        "source_runtime_sha256": runtime["source"],
        "source_capture_barrier_sha256": runtime["source_barrier"],
        "tracking_runtime_sha256": runtime["tracking"],
        "preflight_roster_sha256": runtime["preflight_roster"],
        "input_manifest_sha256": runtime["manifest"],
        "GT_read": False,
        "query_used_by_tracker": False,
    }
    binding_sha, binding_bytes = _write_json(binding_path, binding)
    return {
        "alias": alias,
        "dataset": "hc2",
        "video_clock_group_sha256": _sha_bytes(f"clock:{alias}".encode()),
        "request_path": _rel(request_path, root),
        "request_sha256": request_sha,
        "request_bytes": request_bytes,
        "tracking_binding_path": _rel(binding_path, root),
        "tracking_binding_sha256": binding_sha,
        "tracking_binding_bytes": binding_bytes,
        "accepted_path": case["accepted_path"],
        "accepted_sha256": case["accepted_sha256"],
        "accepted_bytes": case["accepted_bytes"],
        "inventory_path": case["inventory_path"],
        "inventory_sha256": case["inventory_sha256"],
        "inventory_bytes": case["inventory_bytes"],
        "tracking_cache_key": case["tracking_key"],
        "canonical_video_sha256": case["canonical_video_sha256"],
        "canonical_pixel_sha256": case["canonical_pixel_sha256"],
        "canonical_video_path": case["canonical_video_path"],
        "canonical_video_bytes": case["canonical_video_bytes"],
        "frame_clock_sha256": case["frame_clock_sha256"],
        "marker_video_path": case["marker_video_path"],
        "marker_video_sha256": case["marker_video_sha256"],
        "marker_video_bytes": case["marker_video_bytes"],
    }


def run_contract() -> dict[str, Any]:
    tracker = importlib.import_module("scripts.run_stvg_table1_full_tracking_v1")
    writer = importlib.import_module("scripts.stvg_table1_full_cache_retirement_v1")
    names = ("ROOT", "BASE", "FULL", "REQUESTS", "CACHE_ROOT", "RETIREMENTS", "PREFLIGHT_RUNTIME",
             "PREFLIGHT_TRACKING_RUNTIME", "PREFLIGHT_TRACKING_BARRIER")
    old_tracker = {name: getattr(tracker, name) for name in names}
    old_writer = {"ROOT": writer.ROOT, "FULL": writer.FULL}
    try:
        with tempfile.TemporaryDirectory(prefix="stvg-full-retirement-e2e-") as temporary:
            temp_root = Path(temporary).resolve(strict=True)
            base = temp_root / "artifacts/stvg_final_table1_v1"
            full = base / "FULL"
            cache = base / "native_cache"
            retirements = full / "CACHE_RETIREMENTS"
            preflight_runtime = base / "SOURCE_RUNTIME.json"
            preflight_tracking_runtime = base / "TRACKING_RUNTIME.json"
            preflight_barrier_path = base / "TRACKING_GLOBAL_BARRIER.json"
            full.mkdir(parents=True)
            cache.mkdir(parents=True)
            retirements.mkdir(parents=True)
            for key, value in {
                "ROOT": temp_root, "BASE": base, "FULL": full, "CACHE_ROOT": cache,
                "REQUESTS": full / "tracking_cases", "RETIREMENTS": retirements,
                "PREFLIGHT_RUNTIME": preflight_runtime,
                "PREFLIGHT_TRACKING_RUNTIME": preflight_tracking_runtime,
                "PREFLIGHT_TRACKING_BARRIER": preflight_barrier_path,
            }.items():
                setattr(tracker, key, value)
            writer.ROOT, writer.FULL = temp_root, full

            source_runtime_sha, _ = _write_json(preflight_runtime, {"status": "locked", "GT_read": False})
            tracking_runtime_sha, _ = _write_json(preflight_tracking_runtime, {"status": "locked", "GT_read": False})
            source_barrier = base / "SOURCE_CAPTURE_BARRIER.json"
            source_barrier_sha, _ = _write_bytes(source_barrier, b"synthetic-preflight-source-barrier")
            preflight_roster = base / "PREFLIGHT_ROSTER_LOCK.json"
            preflight_roster_sha, _ = _write_bytes(preflight_roster, b"synthetic-preflight-roster")

            protected_video_sha = _sha_bytes(b"synthetic-protected-source-video")
            protected = _make_cache_case(temp_root, cache, "preflight-protected", protected_video_sha)
            protected_record_rows = []
            for index in range(64):
                alias = f"HC{index:05d}"
                binding_path = base / "tracking_cases" / alias / "TRACKING_BINDING.json"
                binding = {
                    "case_alias": alias,
                    "GT_read": False,
                    "accepted_path": protected["accepted_path"],
                    "accepted_sha256": protected["accepted_sha256"],
                    "inventory_path": protected["inventory_path"],
                    "inventory_sha256": protected["inventory_sha256"],
                    "marker_video_path": protected["marker_video_path"],
                    "mask_data_dir": protected["mask_data_dir"],
                    "canonical_receipt_path": protected["canonical_receipt_path"],
                    "canonical_receipt_sha256": protected["canonical_receipt_sha256"],
                    "tracking_cache_key": protected["tracking_key"],
                }
                binding_sha, _ = _write_json(binding_path, binding)
                protected_record_rows.append({
                    "case_alias": alias,
                    "tracking_binding_path": _rel(binding_path, temp_root),
                    "tracking_binding_sha256": binding_sha,
                    "tracking_cache_key": protected["tracking_key"],
                    "accepted_path": protected["accepted_path"],
                    "accepted_sha256": protected["accepted_sha256"],
                    "inventory_path": protected["inventory_path"],
                    "inventory_sha256": protected["inventory_sha256"],
                })
            _write_json(preflight_barrier_path, {
                "schema": "stvg-final-table1-tracking-input-barrier-v1",
                "status": "complete", "count": 64, "records": protected_record_rows,
                "source_runtime_sha256": source_runtime_sha,
                "tracking_runtime_sha256": tracking_runtime_sha,
                "source_capture_barrier_sha256": source_barrier_sha,
                "preflight_roster_sha256": preflight_roster_sha,
                "GT_read": False,
            })

            full_runtime = full / "TRACKING_RUNTIME.json"
            full_tracking_sha, _ = _write_json(full_runtime, {"status": "locked", "GT_read": False})
            runtime = {
                "source": _sha_bytes(b"full-source-runtime"),
                "source_barrier": _sha_bytes(b"full-source-barrier"),
                "tracking": full_tracking_sha,
                "preflight_roster": preflight_roster_sha,
                "manifest": _sha_bytes(b"full-input-manifest"),
            }
            fresh_video_sha = _sha_bytes(b"synthetic-new-source-video")
            fresh = _make_cache_case(temp_root, cache, "full-new", fresh_video_sha)
            row_new = _make_full_row(temp_root, full, fresh, "HC00000", fresh_video_sha, runtime)
            row_reused = _make_full_row(temp_root, full, protected, "HC00001", protected_video_sha, runtime)
            aliases = ["HC00000", "HC00001"]
            selection = tracker._selection_sha256(aliases)
            batch_path = full / f"TRACKING_BATCH_{selection}_BARRIER.json"
            batch = {
                "schema": tracker.FULL_BATCH_SCHEMA,
                "status": "complete",
                "count": len(aliases),
                "aliases": aliases,
                "selection_sha256": selection,
                "records": [row_new, row_reused],
                "GT_read": False,
                "query_used_by_tracker": False,
            }
            _write_json(batch_path, batch)

            teacher_path = full / "TEACHER_BATCH_BARRIERS" / "teacher.json"
            student_path = full / "STUDENT_BATCH_BARRIER_student.json"
            root_path = full / "ROOT_BATCH_READBACKS" / "root.json"
            _write_json(teacher_path, {"status": "complete", "count": 2, "aliases": aliases, "GT_read": False})
            _write_json(student_path, {"status": "complete", "count": 2, "aliases": aliases, "GT_read": False})
            _write_json(root_path, {"status": "pass", "count": 2,
                                    "student_barrier_sha256": _sha_file(student_path), "GT_read": False})

            old_protected_files = {
                path.relative_to(temp_root).as_posix(): _sha_file(path)
                for path in (cache / "tracking/preflight-protected/ACCEPTED.json",
                             cache / "tracking/preflight-protected/MASK_INVENTORY.json",
                             cache / "tracking/preflight-protected/marked.mp4",
                             cache / "tracking/preflight-protected/masks/mask_0.json",
                             cache / "canonical/preflight-protected/input.mkv",
                             cache / "canonical/preflight-protected/CANONICAL_RECEIPT.json")
            }

            result = writer.retire(batch_path, teacher_path, student_path, root_path)
            if result.get("status") != "pass" or result.get("GT_read") is not False:
                raise AssertionError("retirement writer did not return a GT-free pass")
            retirement_path = retirements / f"{selection}.json"
            retirement = json.loads(retirement_path.read_text(encoding="utf-8"))
            reader_result = tracker._validate_retirement(
                batch_path, batch, {"tracking_runtime_sha256": full_tracking_sha})
            if reader_result is None:
                raise AssertionError("retirement reader did not find the completed receipt")
            archive = retirement.get("lossless_evidence_archive")
            if not isinstance(archive, dict):
                raise AssertionError("synthetic new evidence was not archived")
            if len(archive["members"]) != 3:
                raise AssertionError("archive must contain the new ACCEPTED, inventory, and one mask JSON")
            for relative in (fresh["canonical_video_path"], fresh["marker_video_path"],
                             fresh["accepted_path"], fresh["inventory_path"], fresh["mask_path"]):
                if (temp_root / relative).exists():
                    raise AssertionError("new-batch media/plain evidence survived the synthetic retirement")
            for relative, digest in old_protected_files.items():
                path = temp_root / relative
                if not path.is_file() or _sha_file(path) != digest:
                    raise AssertionError("a protected preflight cache file was deleted or changed")
            if len(retirement["retired_files"]) != 2:
                raise AssertionError("new unprotected canonical/marker pair was not exactly retired")

            # Exercise the exact retired-case reader after deletion, including
            # its canonical binding byte check and archived JSON recovery.
            source_bindings = {
                alias: json.loads((temp_root / row["tracking_binding_path"]).read_text(encoding="utf-8"))
                for alias, row in (("HC00000", row_new), ("HC00001", row_reused))
            }
            locked = {
                "source_files": {
                    alias: {"source_case_sha256": binding["source_case_sha256"]}
                    for alias, binding in source_bindings.items()
                },
                "full_source_runtime_sha256": runtime["source"],
                "source_capture_barrier_sha256": runtime["source_barrier"],
                "tracking_runtime_sha256": runtime["tracking"],
                "preflight_roster_sha256": runtime["preflight_roster"],
                "input_manifest_sha256": runtime["manifest"],
            }
            query_records = {
                "HC00000": {"dataset": "hc2", "query_key_sha256": _sha_bytes(b"query-key:HC00000"),
                            "video_sha256": fresh_video_sha},
                "HC00001": {"dataset": "hc2", "query_key_sha256": _sha_bytes(b"query-key:HC00001"),
                            "video_sha256": protected_video_sha},
            }
            for alias, saved in (("HC00000", row_new), ("HC00001", row_reused)):
                tracker._verify_retired_case(alias, saved, query_records[alias], locked, reader_result)

            actual_binding = ROOT / "artifacts/stvg_final_table1_v1/tracking_cases/HC01/TRACKING_BINDING.json"
            actual_metadata = {"status": "not_available", "GT_read": False,
                               "request_opened": False, "media_opened": False}
            if actual_binding.is_file():
                actual_bytes = actual_binding.read_bytes()
                actual_value = json.loads(actual_bytes)
                canonical_actual = tracker.canonical_json_bytes(actual_value)
                actual_metadata = {
                    "status": "pass" if canonical_actual == actual_bytes else "fail",
                    "binding_bytes": len(actual_bytes),
                    "canonical_bytes": len(canonical_actual),
                    "legacy_double_lf_bytes": len(canonical_actual + b"\n"),
                    "legacy_double_lf_would_match": canonical_actual + b"\n" == actual_bytes,
                    "GT_read": False,
                    "request_opened": False,
                    "media_opened": False,
                }
                if actual_metadata["status"] != "pass":
                    raise AssertionError("actual HC01 tracking binding is not encoded by the writer's canonical JSON helper")
            return {
                "schema": "stvg-final-table1-cache-retirement-synthetic-e2e-v1",
                "status": "pass",
                "checks": 11,
                "archive_member_count": len(archive["members"]),
                "retired_media_count": len(retirement["retired_files"]),
                "protected_preflight_file_count": len(old_protected_files),
                "protected_preflight_files_retained": True,
                "GT_read": False,
                "media_opened": False,
                "model_loaded": False,
                "GPU_used": False,
                "real_native_cache_touched": False,
                "writer_script_sha256": _sha_file(ROOT / "scripts/stvg_table1_full_cache_retirement_v1.py"),
                "tracking_verifier_sha256": _sha_file(ROOT / "scripts/run_stvg_table1_full_tracking_v1.py"),
                "retirement_selection_sha256": selection,
                "retirement_archive_sha256": archive["sha256"],
                "synthetic_retired_case_reader_count": 2,
                "actual_HC01_binding_metadata_contract": actual_metadata,
            }
    finally:
        for name, value in old_tracker.items():
            setattr(tracker, name, value)
        writer.ROOT, writer.FULL = old_writer["ROOT"], old_writer["FULL"]


def _write_report(report: dict[str, Any]) -> None:
    payload = (json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        if REPORT.read_bytes() != payload:
            raise RuntimeError(f"preserve existing synthetic contract report: {REPORT}")
        return
    temp = REPORT.with_name(f".{REPORT.name}.{os.getpid()}.partial")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, REPORT)
    except FileExistsError:
        if REPORT.read_bytes() != payload:
            raise RuntimeError(f"concurrent synthetic report differs: {REPORT}")
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    result = run_contract()
    _write_report(result)
    print(json.dumps(result, sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
