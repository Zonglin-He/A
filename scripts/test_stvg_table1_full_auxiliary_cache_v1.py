#!/usr/bin/env python3
"""Temporary-directory E2E contracts for FULL auxiliary cache retirement."""
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
REPORT = ROOT / "artifacts/stvg_final_table1_v1/FULL/engineering/auxiliary_cache_synthetic_cpu_contracts_v5.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, tracker: Any) -> tuple[str, int]:
    payload = tracker.canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha_bytes(payload), len(payload)


def _rel(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=True)).as_posix()


def _make_cache(cache_root: Path, root: Path, name: str, mask_count: int) -> dict[str, Any]:
    track_root = cache_root / "tracking" / name
    mask_dir = track_root / "official/dettrack/input/mask_data"
    preview = mask_dir.parent / "input_all_targets_masks.mp4"
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(f"synthetic-unused-preview-{name}".encode())
    members = {}
    for index in range(mask_count):
        path = mask_dir / f"mask_{index}.json"
        path.write_bytes(json.dumps({"fixture": name, "index": index}, sort_keys=True).encode() + b"\n")
        members[path.name] = {"path": path, "sha256": _sha(path), "bytes": path.stat().st_size}
    keep = mask_dir.parent / "keep.txt"
    keep.write_bytes(b"outside-the-retirement-allowlist")
    return {"mask_dir": mask_dir, "preview": preview, "members": members,
            "keep": keep, "tracking_cache_key": _sha_bytes(f"cache-key:{name}".encode())}


def _write_old_preflight(temp_root: Path, tracker: Any, base: Path,
                         cache_root: Path, protected: dict[str, Any]) -> None:
    source_runtime = base / "SOURCE_RUNTIME.json"
    tracking_runtime = base / "TRACKING_RUNTIME.json"
    source_barrier = base / "SOURCE_CAPTURE_BARRIER.json"
    roster = base / "PREFLIGHT_ROSTER_LOCK.json"
    sr_sha, _ = _write_json(source_runtime, {"status": "locked", "GT_read": False}, tracker)
    tr_sha, _ = _write_json(tracking_runtime, {"status": "locked", "GT_read": False}, tracker)
    sb_sha, _ = _write_json(source_barrier, {"synthetic": True}, tracker)
    roster_sha, _ = _write_json(roster, {"synthetic": True}, tracker)

    accepted_path = cache_root / "tracking/preflight/ACCEPTED.json"
    inventory_path = cache_root / "tracking/preflight/MASK_INVENTORY.json"
    marker_path = cache_root / "tracking/preflight/marked.mp4"
    canonical_path = cache_root / "canonical/preflight/input.mkv"
    canonical_receipt_path = cache_root / "canonical/preflight/CANONICAL_RECEIPT.json"
    accepted_sha, _ = _write_json(accepted_path, {"GT_read": False}, tracker)
    inventory_sha, _ = _write_json(inventory_path, {"GT_read": False}, tracker)
    marker_path.write_bytes(b"protected-marker")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_bytes(b"protected-canonical")
    receipt_sha, _ = _write_json(canonical_receipt_path, {
        "schema": "stvg-table1-canonical-input-v1", "status": "complete",
        "canonical_video_sha256": _sha(canonical_path),
        "canonical_pixel_sha256": _sha_bytes(b"protected-pixels"),
        "source_video_sha256": _sha_bytes(b"protected-source"),
        "canonical_video_bytes": canonical_path.stat().st_size,
        "GT_read": False, "query_independent": True,
    }, tracker)

    protected_binding_path = base / "tracking_cases/preflight/unused.json"
    protected_binding_sha, _ = _write_json(protected_binding_path, {"unused": True}, tracker)
    records = []
    for index in range(64):
        alias = f"P{index:02d}"
        binding_path = base / "tracking_cases" / alias / "TRACKING_BINDING.json"
        binding = {
            "case_alias": alias, "GT_read": False,
            "accepted_path": _rel(accepted_path, temp_root), "accepted_sha256": accepted_sha,
            "inventory_path": _rel(inventory_path, temp_root), "inventory_sha256": inventory_sha,
            "marker_video_path": _rel(marker_path, temp_root),
            "mask_data_dir": _rel(protected["mask_dir"], temp_root),
            "canonical_receipt_path": _rel(canonical_receipt_path, temp_root),
            "canonical_receipt_sha256": receipt_sha,
            "tracking_cache_key": protected["tracking_cache_key"],
        }
        binding_sha, _ = _write_json(binding_path, binding, tracker)
        records.append({
            "case_alias": alias,
            "tracking_binding_path": _rel(binding_path, temp_root),
            "tracking_binding_sha256": binding_sha,
            "tracking_cache_key": protected["tracking_cache_key"],
            "accepted_path": _rel(accepted_path, temp_root), "accepted_sha256": accepted_sha,
            "inventory_path": _rel(inventory_path, temp_root), "inventory_sha256": inventory_sha,
        })
    preflight = base / "TRACKING_GLOBAL_BARRIER.json"
    _write_json(preflight, {
        "schema": "stvg-final-table1-tracking-input-barrier-v1",
        "status": "complete", "count": 64, "records": records,
        "source_runtime_sha256": sr_sha, "tracking_runtime_sha256": tr_sha,
        "source_capture_barrier_sha256": sb_sha,
        "preflight_roster_sha256": roster_sha, "GT_read": False,
    }, tracker)
    assert protected_binding_sha


def _make_batch(temp_root: Path, tracker: Any, full: Path, cache: dict[str, Any],
                alias: str, count: int = 1) -> tuple[Path, Path, Path, Path]:
    records = []
    aliases = [f"HC{int(alias[2:])+i:05d}" for i in range(count)]
    for case_alias in aliases:
        binding_path = full / "tracking_cases" / case_alias / "TRACKING_BINDING.json"
        binding = {
            "case_alias": case_alias,
            "tracking_cache_key": cache["tracking_cache_key"],
            "mask_data_dir": _rel(cache["mask_dir"], temp_root),
            "GT_read": False,
        }
        binding_sha, binding_bytes = _write_json(binding_path, binding, tracker)
        records.append({
            "alias": case_alias,
            "tracking_binding_path": _rel(binding_path, temp_root),
            "tracking_binding_sha256": binding_sha,
            "tracking_binding_bytes": binding_bytes,
            "tracking_cache_key": cache["tracking_cache_key"],
        })
    selection = tracker._selection_sha256(aliases)
    tracking_path = full / f"TRACKING_BATCH_{selection}_BARRIER.json"
    _write_json(tracking_path, {
        "schema": tracker.FULL_BATCH_SCHEMA, "status": "complete", "count": count,
        "aliases": aliases, "selection_sha256": selection, "records": records,
        "bindings": {
            "full_source_runtime_sha256": _sha_bytes(b"full-source-runtime"),
            "source_capture_barrier_sha256": _sha_bytes(b"full-source-barrier"),
            "tracking_runtime_sha256": _sha(full / "TRACKING_RUNTIME.json"),
            "full_source_roster_sha256": _sha_bytes(b"full-roster"),
            "input_manifest_sha256": _sha_bytes(b"input-manifest"),
            "preflight_source_runtime_sha256": _sha_bytes(b"preflight-source-runtime"),
            "preflight_tracking_runtime_sha256": _sha_bytes(b"preflight-tracking-runtime"),
        },
        "query_independent_tracker_cache": True, "query_used_by_tracker": False,
        "DINO_calls": 0, "optimizer_steps": 0, "backwards": 0, "GT_read": False,
    }, tracker)
    teacher = full / f"TEACHER_BATCH_{selection}.json"
    student = full / f"STUDENT_BATCH_{selection}.json"
    root = full / f"ROOT_BATCH_{selection}.json"
    _write_json(teacher, {"status": "complete", "count": count, "aliases": aliases, "GT_read": False}, tracker)
    _write_json(student, {"status": "complete", "count": count, "aliases": aliases, "GT_read": False}, tracker)
    _write_json(root, {"status": "pass", "count": count,
                       "student_barrier_sha256": _sha(student), "GT_read": False}, tracker)
    return tracking_path, teacher, student, root


def run_contract() -> dict[str, Any]:
    tracker = importlib.import_module("scripts.run_stvg_table1_full_tracking_v1")
    auxiliary = importlib.import_module("scripts.stvg_table1_full_auxiliary_cache_v1")
    tracker_names = ("ROOT", "BASE", "FULL", "REQUESTS", "CACHE_ROOT", "RETIREMENTS",
                     "PREFLIGHT_RUNTIME", "PREFLIGHT_TRACKING_RUNTIME", "PREFLIGHT_TRACKING_BARRIER")
    old_tracker = {key: getattr(tracker, key) for key in tracker_names}
    aux_names = ("ROOT", "FULL", "AUX_RETIREMENTS", "PREPARED", "ARCHIVES",
                 "BATCH_PLAN", "CUSTODY_READBACK")
    old_aux = {key: getattr(auxiliary, key) for key in aux_names}
    try:
        with tempfile.TemporaryDirectory(prefix="stvg-full-aux-retirement-e2e-") as temporary:
            temp_root = Path(temporary).resolve(strict=True)
            base = temp_root / "artifacts/stvg_final_table1_v1"
            full = base / "FULL"
            cache_root = base / "native_cache"
            full.mkdir(parents=True)
            cache_root.mkdir(parents=True)
            values = {
                "ROOT": temp_root, "BASE": base, "FULL": full,
                "REQUESTS": full / "tracking_cases", "CACHE_ROOT": cache_root,
                "RETIREMENTS": full / "CACHE_RETIREMENTS",
                "PREFLIGHT_RUNTIME": base / "SOURCE_RUNTIME.json",
                "PREFLIGHT_TRACKING_RUNTIME": base / "TRACKING_RUNTIME.json",
                "PREFLIGHT_TRACKING_BARRIER": base / "TRACKING_GLOBAL_BARRIER.json",
            }
            for key, value in values.items():
                setattr(tracker, key, value)
            for key, value in {
                "ROOT": temp_root, "FULL": full,
                "AUX_RETIREMENTS": full / "CACHE_AUXILIARY_RETIREMENTS",
                "PREPARED": full / "CACHE_AUXILIARY_RETIREMENT_PREPARED",
                "ARCHIVES": full / "auxiliary_evidence_archives",
                "BATCH_PLAN": full / "BATCH_PLAN007.json",
                "CUSTODY_READBACK": full / "ROOT_AUXILIARY_CUSTODY_READBACK.json",
            }.items():
                setattr(auxiliary, key, value)

            _write_json(full / "TRACKING_RUNTIME.json", {"status": "locked", "GT_read": False}, tracker)

            empty_receipts = full / "empty-receipts"
            empty_receipts.mkdir()
            previous_receipts = auxiliary.AUX_RETIREMENTS
            auxiliary.AUX_RETIREMENTS = empty_receipts
            try:
                auxiliary.verify_all()
            except RuntimeError as exc:
                if "no auxiliary retirement receipts" not in str(exc):
                    raise
            else:
                raise AssertionError("verify-all incorrectly passed with zero receipts")
            finally:
                auxiliary.AUX_RETIREMENTS = previous_receipts

            old_cache = _make_cache(cache_root, temp_root, "protected-old", 2)
            _write_old_preflight(temp_root, tracker, base, cache_root, old_cache)
            new_cache = _make_cache(cache_root, temp_root, "full-new", 3)
            unrelated = new_cache["keep"]
            unrelated_sha = _sha(unrelated)
            tracking_path, teacher, student, root_path = _make_batch(
                temp_root, tracker, full, new_cache, "HC00000", count=2)
            # The second alias deliberately points to the protected preflight cache.
            batch = json.loads(tracking_path.read_text(encoding="utf-8"))
            aliases = batch["aliases"]
            protected_binding_path = full / "tracking_cases" / aliases[1] / "TRACKING_BINDING.json"
            protected_binding = {
                "case_alias": aliases[1], "tracking_cache_key": old_cache["tracking_cache_key"],
                "mask_data_dir": _rel(old_cache["mask_dir"], temp_root), "GT_read": False,
            }
            protected_binding_sha, protected_binding_bytes = _write_json(protected_binding_path, protected_binding, tracker)
            batch["records"][1].update({
                "tracking_binding_path": _rel(protected_binding_path, temp_root),
                "tracking_binding_sha256": protected_binding_sha,
                "tracking_binding_bytes": protected_binding_bytes,
                "tracking_cache_key": old_cache["tracking_cache_key"],
            })
            _write_json(tracking_path, batch, tracker)
            _write_json(teacher, {"status": "complete", "count": 2, "aliases": aliases, "GT_read": False}, tracker)
            _write_json(student, {"status": "complete", "count": 2, "aliases": aliases, "GT_read": False}, tracker)
            _write_json(root_path, {"status": "pass", "count": 2,
                                    "student_barrier_sha256": _sha(student), "GT_read": False}, tracker)
            # Recompute teacher/student/root seals after finalizing the synthetic batch rows.

            protected_snapshot = {
                path.relative_to(temp_root).as_posix(): _sha(path)
                for path in (old_cache["preview"], *[row["path"] for row in old_cache["members"].values()],
                             old_cache["keep"])
            }
            first = auxiliary.retire(tracking_path, teacher, student, root_path)
            if first.get("status") != "pass" or first.get("GT_read") is not False:
                raise AssertionError("auxiliary retirement did not pass")
            if new_cache["preview"].exists() or any(row["path"].exists() for row in new_cache["members"].values()):
                raise AssertionError("eligible FULL preview or mask_data JSON survived retirement")
            if not unrelated.is_file() or _sha(unrelated) != unrelated_sha:
                raise AssertionError("file outside the exact auxiliary allowlist changed")
            for relative, digest in protected_snapshot.items():
                path = temp_root / relative
                if not path.is_file() or _sha(path) != digest:
                    raise AssertionError("protected preflight cache changed")
            receipt_path = full / "CACHE_AUXILIARY_RETIREMENTS" / f"{batch['selection_sha256']}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            archive = receipt["lossless_mask_data_archive"]
            if len(archive["members"]) != 3:
                raise AssertionError("all three synthetic mask_data JSON files must be archived")
            # Reintroducing any retired path must invalidate custody. These
            # checks exercise the production verifier without touching any
            # non-temporary cache.
            try:
                new_cache["preview"].write_bytes(b"unexpected-preview")
                try:
                    auxiliary._validate_receipt(receipt_path, tracker)
                except RuntimeError as exc:
                    if "preview still exists" not in str(exc):
                        raise
                else:
                    raise AssertionError("reintroduced preview passed retirement verification")
            finally:
                new_cache["preview"].unlink(missing_ok=True)
            first_member = next(iter(new_cache["members"].values()))["path"]
            try:
                first_member.write_bytes(b"unexpected-mask-json")
                try:
                    auxiliary._validate_receipt(receipt_path, tracker)
                except RuntimeError as exc:
                    if "mask_data JSON still exists" not in str(exc):
                        raise
                else:
                    raise AssertionError("reintroduced mask_data JSON passed retirement verification")
            finally:
                first_member.unlink(missing_ok=True)
            verification = auxiliary.verify_all()
            if verification.get("verified_receipts") != 1:
                raise AssertionError("verify-all did not validate the completed auxiliary receipt")

            # Test prepared-journal recovery after deleting inputs but before
            # installing the final receipt. A forced failure is scoped to the
            # temporary fixture and cannot affect workspace artifacts.
            recovery_cache = _make_cache(cache_root, temp_root, "full-recovery", 1)
            tracking2, teacher2, student2, root2 = _make_batch(
                temp_root, tracker, full, recovery_cache, "HC00002", count=1)
            original_write = auxiliary._write_immutable
            def fail_final_receipt(path: Path, value: Mapping[str, Any], module: Any) -> None:
                if Path(path).parent == auxiliary.AUX_RETIREMENTS:
                    raise RuntimeError("synthetic interruption after source unlink")
                original_write(path, value, module)
            auxiliary._write_immutable = fail_final_receipt
            try:
                try:
                    auxiliary.retire(tracking2, teacher2, student2, root2)
                except RuntimeError as exc:
                    if "synthetic interruption" not in str(exc):
                        raise
                else:
                    raise AssertionError("synthetic interruption did not occur")
            finally:
                auxiliary._write_immutable = original_write
            selection2 = json.loads(tracking2.read_text(encoding="utf-8"))["selection_sha256"]
            journal2 = full / "CACHE_AUXILIARY_RETIREMENT_PREPARED" / f"{selection2}.json"
            receipt2 = full / "CACHE_AUXILIARY_RETIREMENTS" / f"{selection2}.json"
            if not journal2.is_file() or receipt2.exists() or recovery_cache["preview"].exists():
                raise AssertionError("interruption did not leave the expected prepared recovery state")
            resumed = auxiliary.retire(tracking2, teacher2, student2, root2)
            if resumed.get("status") != "pass" or recovery_cache["preview"].exists():
                raise AssertionError("prepared journal did not resume safely")
            final_verification = auxiliary.verify_all()
            if final_verification.get("verified_receipts") != 2:
                raise AssertionError("verify-all did not validate both completed synthetic receipts")

            # A production custody readback must prove exact coverage of all
            # 752 fixed selections; two valid receipts cannot masquerade as a
            # completed full-run verification.
            all_aliases = [f"X{i:05d}" for i in range(13785)]
            planned_batches = []
            offset = 0
            for batch_index in range(752):
                batch_size = 19 if batch_index < 249 else 18
                batch_aliases = all_aliases[offset:offset + batch_size]
                offset += batch_size
                planned_batches.append({
                    "index": batch_index,
                    "aliases": batch_aliases,
                    "selection_sha256": tracker._selection_sha256(batch_aliases),
                })
            if offset != len(all_aliases):
                raise AssertionError("synthetic full plan partition is malformed")
            _write_json(auxiliary.BATCH_PLAN, {
                "schema": "stvg-final-table1-fixed-batch-plan-v1", "status": "locked",
                "count": 13785, "batch_count": 752, "GT_read": False,
                "batches": planned_batches,
            }, tracker)
            try:
                auxiliary.verify_all_and_write(auxiliary.CUSTODY_READBACK)
            except RuntimeError as exc:
                if "do not exactly cover" not in str(exc):
                    raise
            else:
                raise AssertionError("partial auxiliary receipts were accepted as full-plan custody")
            if auxiliary.CUSTODY_READBACK.exists():
                raise AssertionError("failed full-plan readback was written")
            return {
                "schema": "stvg-final-table1-auxiliary-cache-synthetic-e2e-v1",
                "status": "pass",
                "checks": 18,
                "reintroduced_retired_paths_rejected": True,
                "completed_receipts": final_verification["verified_receipts"],
                "archived_mask_data_json_count": 4,
                "retired_preview_count": 2,
                "protected_preflight_cache_files_retained": len(protected_snapshot),
                "outside_allowlist_sentinel_preserved": True,
                "prepared_journal_recovery": "pass",
                "verify_all_archive_bytes": "pass",
                "zero_receipts_fail_closed": True,
                "partial_752_batch_plan_rejected": True,
                "GT_read": False,
                "GPU_used": False,
                "model_loaded": False,
                "real_native_cache_touched": False,
                "auxiliary_writer_sha256": _sha(ROOT / "scripts/stvg_table1_full_auxiliary_cache_v1.py"),
                "tracking_verifier_sha256": _sha(ROOT / "scripts/run_stvg_table1_full_tracking_v1.py"),
            }
    finally:
        for key, value in old_tracker.items():
            setattr(tracker, key, value)
        for key, value in old_aux.items():
            setattr(auxiliary, key, value)


def _read(path: Path, tracker: Any) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(report: dict[str, Any]) -> None:
    payload = (json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        if REPORT.read_bytes() != payload:
            raise RuntimeError(f"preserve existing auxiliary synthetic report: {REPORT}")
        return
    temporary = REPORT.with_name(f".{REPORT.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, REPORT)
    except FileExistsError:
        if REPORT.read_bytes() != payload:
            raise RuntimeError(f"concurrent auxiliary report differs: {REPORT}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    report = run_contract()
    _write_report(report)
    print(json.dumps(report, sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
