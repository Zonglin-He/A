#!/usr/bin/env python3
"""Retire only the FULL run's unused official preview and mask_data JSON.

This is separate from the frozen main cache-retirement path. It may run only
after tracking, teacher, student, and root batch seals exist. Raw mask_data JSON
is retained byte-for-byte in a gzip-compressed tar archive; the all-targets
preview is recorded by hash and treated as reproducible scratch media.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "artifacts/stvg_final_table1_v1/FULL"
AUX_RETIREMENTS = FULL / "CACHE_AUXILIARY_RETIREMENTS"
PREPARED = FULL / "CACHE_AUXILIARY_RETIREMENT_PREPARED"
ARCHIVES = FULL / "auxiliary_evidence_archives"
BATCH_PLAN = FULL / "BATCH_PLAN007.json"
CUSTODY_READBACK = FULL / "ROOT_AUXILIARY_CUSTODY_READBACK.json"
SCHEMA = "stvg-final-table1-full-auxiliary-cache-retirement-v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {Path(path).name}")
    return value


def _write_immutable(path: Path, value: Mapping[str, Any], tracker: Any) -> None:
    path = Path(path)
    payload = tracker.canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"prepared auxiliary receipt already exists with different bytes: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"concurrent auxiliary receipt differs: {path.name}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_workspace_rel(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or value.startswith("/"):
        raise RuntimeError("auxiliary receipt has an unsafe workspace-relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts) or rel.as_posix() != value:
        raise RuntimeError("auxiliary receipt has a noncanonical relative path")
    return value


def _relative(path: Path) -> str:
    return Path(path).resolve(strict=False).relative_to(ROOT.resolve(strict=True)).as_posix()


def _validate_cache_layout(mask_dir: Path, cache_root: Path,
                           *, require_exists: bool = True) -> tuple[Path, Path]:
    if mask_dir.is_symlink():
        raise RuntimeError("official mask_data path must not be a symbolic link")
    mask_dir = mask_dir.resolve(strict=require_exists)
    cache_root = cache_root.resolve(strict=True)
    if not mask_dir.is_relative_to(cache_root):
        raise RuntimeError("auxiliary path escapes the native cache")
    if mask_dir.name != "mask_data" or tuple(mask_dir.parts[-4:]) != ("official", "dettrack", "input", "mask_data"):
        raise RuntimeError("only official/dettrack/input/mask_data/*.json is eligible")
    if require_exists and not mask_dir.is_dir():
        raise RuntimeError("official mask_data path must be a regular directory")
    preview = mask_dir.parent / "input_all_targets_masks.mp4"
    if preview.parent != mask_dir.parent or preview.name != "input_all_targets_masks.mp4":
        raise RuntimeError("invalid all-targets preview path")
    return mask_dir, preview


def _is_protected(path: Path, protected_files: set[Path], protected_dirs: set[Path]) -> bool:
    path = path.resolve(strict=False)
    return path in protected_files or any(path == root or path.is_relative_to(root) for root in protected_dirs)


def _assert_regular_cache_file(path: Path, tracker: Any, *, suffix: str | None = None) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"eligible auxiliary cache path is not a regular file: {path.name}")
    cache_path = tracker._safe_cache_path(_relative(path), require_exists=True)
    if cache_path != path.resolve(strict=True):
        raise RuntimeError("auxiliary file resolved outside its cache path")
    if suffix is not None and path.suffix.lower() != suffix:
        raise RuntimeError("unexpected auxiliary cache file extension")
    return path.stat().st_size, _sha(path)


def _seal_downstream(tracking_barrier: Mapping[str, Any], tracking_path: Path,
                     teacher_path: Path, student_path: Path, root_path: Path,
                     tracker: Any) -> dict[str, dict[str, str]]:
    seals: dict[str, dict[str, str]] = {}
    batch_aliases = tracking_barrier["aliases"]
    for kind, path in (("teacher_batch", teacher_path), ("student_batch", student_path), ("root_batch", root_path)):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(FULL.resolve(strict=True)):
            raise RuntimeError(f"{kind} seal is outside the FULL scope")
        value = _read(path)
        if (value.get("status") not in {"complete", "sealed", "pass"}
                or value.get("GT_read") is not False
                or value.get("count") != tracking_barrier["count"]):
            raise RuntimeError(f"{kind} downstream seal is incomplete or not GT-free")
        if kind != "root_batch":
            if value.get("aliases") != batch_aliases:
                raise RuntimeError(f"{kind} downstream seal does not bind selected aliases")
        elif value.get("student_barrier_sha256", value.get("barrier_sha256")) != _sha(student_path):
            raise RuntimeError("root batch receipt does not bind the selected student barrier")
        seals[kind] = {"path": _relative(path), "sha256": _sha(path)}
    return seals


def _validate_tracking_barrier(path: Path, tracker: Any) -> tuple[dict[str, Any], str, str]:
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(FULL.resolve(strict=True)):
        raise RuntimeError("tracking batch barrier is outside the FULL scope")
    barrier = _read(path)
    aliases = barrier.get("aliases")
    if (barrier.get("schema") != tracker.FULL_BATCH_SCHEMA
            or barrier.get("status") != "complete"
            or barrier.get("GT_read") is not False
            or not isinstance(aliases, list) or not aliases
            or len(set(aliases)) != len(aliases)
            or barrier.get("count") != len(aliases)
            or barrier.get("selection_sha256") != tracker._selection_sha256(aliases)
            or not isinstance(barrier.get("records"), list)
            or len(barrier["records"]) != len(aliases)):
        raise RuntimeError("tracking batch barrier is incomplete or malformed")
    runtime_path = FULL / "TRACKING_RUNTIME.json"
    if not runtime_path.is_file():
        raise RuntimeError("FULL tracking runtime is missing")
    runtime_sha = _sha(runtime_path)
    bindings = barrier.get("bindings")
    expected_binding_keys = {
        "full_source_runtime_sha256", "source_capture_barrier_sha256",
        "tracking_runtime_sha256", "full_source_roster_sha256", "input_manifest_sha256",
        "preflight_source_runtime_sha256", "preflight_tracking_runtime_sha256",
    }
    if (not isinstance(bindings, dict) or set(bindings) != expected_binding_keys
            or bindings.get("tracking_runtime_sha256") != runtime_sha
            or any(not SHA_RE.fullmatch(str(value)) for value in bindings.values())
            or barrier.get("query_independent_tracker_cache") is not True
            or barrier.get("query_used_by_tracker") is not False
            or barrier.get("DINO_calls") != 0
            or barrier.get("optimizer_steps") != 0
            or barrier.get("backwards") != 0):
        raise RuntimeError("tracking batch runtime or query-free evidence bindings are malformed")
    return barrier, _relative(path), runtime_sha


def _archive_members(archive_value: Mapping[str, Any] | None, tracker: Any) -> dict[str, tuple[int, str]]:
    if archive_value is None:
        return {}
    if set(archive_value) != {"path", "bytes", "sha256", "members"}:
        raise RuntimeError("auxiliary archive metadata has an unexpected schema")
    relative = _safe_workspace_rel(archive_value["path"])
    archive_path = tracker._safe_root_file(relative, base=FULL.resolve(strict=True))
    if (archive_path.is_symlink() or isinstance(archive_value["bytes"], bool)
            or not isinstance(archive_value["bytes"], int) or archive_value["bytes"] <= 0
            or archive_path.stat().st_size != archive_value["bytes"]
            or _sha(archive_path) != archive_value["sha256"]
            or not SHA_RE.fullmatch(str(archive_value["sha256"]))):
        raise RuntimeError("auxiliary evidence archive bytes differ from receipt")
    declared: dict[str, tuple[int, str]] = {}
    members = archive_value["members"]
    if not isinstance(members, list):
        raise RuntimeError("auxiliary archive members must be a list")
    for row in members:
        if (not isinstance(row, dict) or set(row) != {"original_path", "bytes", "sha256"}
                or isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int)
                or row["bytes"] < 0 or not SHA_RE.fullmatch(str(row["sha256"]))):
            raise RuntimeError("auxiliary archive member receipt is malformed")
        member_path = _safe_workspace_rel(row["original_path"])
        if tracker._cache_path_is_symlink(member_path):
            raise RuntimeError("auxiliary archive member is a symbolic link in the cache")
        safe = tracker._safe_cache_path(member_path, require_exists=False)
        if safe.suffix.lower() != ".json" or safe.name == "":
            raise RuntimeError("auxiliary archive contains a file outside official/dettrack/input/mask_data")
        mask_dir, _ = _validate_cache_layout(safe.parent, tracker.CACHE_ROOT, require_exists=False)
        if mask_dir != safe.parent or _relative(safe) in declared:
            raise RuntimeError("auxiliary archive member path is invalid or duplicated")
        declared[_relative(safe)] = (row["bytes"], row["sha256"])
    if list(declared) != sorted(declared):
        raise RuntimeError("auxiliary archive members are not sorted")
    observed: dict[str, tuple[int, str]] = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = _safe_workspace_rel(member.name)
            if not member.isfile() or member.issym() or member.islnk() or name in observed:
                raise RuntimeError("auxiliary archive contains a nonregular or duplicate tar member")
            if name not in declared or member.size != declared[name][0]:
                raise RuntimeError("auxiliary archive contains undeclared or wrong-size bytes")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError("cannot read auxiliary tar member")
            digest = hashlib.sha256()
            size = 0
            with stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
            observed[name] = (size, digest.hexdigest())
    if observed != declared:
        raise RuntimeError("auxiliary archive bytes or membership differ from the receipt")
    return observed


def _make_plan(tracking_path: Path, barrier: Mapping[str, Any], tracking_sha: str,
               seals: Mapping[str, Any], tracker: Any) -> dict[str, Any]:
    protected_files, protected_dirs = tracker._preflight_protected_cache_roots()
    cache_root = tracker.CACHE_ROOT.resolve(strict=True)
    batch_records = barrier["records"]
    cases: list[dict[str, Any]] = []
    previews: dict[str, dict[str, Any]] = {}
    mask_jsons: dict[str, dict[str, Any]] = {}
    for alias, row in zip(barrier["aliases"], batch_records):
        if row.get("alias") != alias:
            raise RuntimeError("tracking batch alias order differs from records")
        binding_path = tracker._safe_root_file(row.get("tracking_binding_path"), base=FULL / "tracking_cases")
        if (binding_path.stat().st_size != row.get("tracking_binding_bytes")
                or _sha(binding_path) != row.get("tracking_binding_sha256")):
            raise RuntimeError(f"tracking binding bytes differ for {alias}")
        binding = _read(binding_path)
        if (binding.get("case_alias") != alias or binding.get("tracking_cache_key") != row.get("tracking_cache_key")
                or binding.get("GT_read") is not False):
            raise RuntimeError(f"tracking binding does not match batch for {alias}")
        mask_value = binding.get("mask_data_dir")
        if not isinstance(mask_value, str):
            raise RuntimeError(f"tracking binding has no mask_data directory for {alias}")
        mask_dir = tracker._safe_cache_path(mask_value, require_exists=False)
        mask_dir, preview_path = _validate_cache_layout(mask_dir, cache_root)
        mask_rel = _relative(mask_dir)
        preview_rel = _relative(preview_path)
        member_paths: list[str] = []
        protected = _is_protected(mask_dir, protected_files, protected_dirs)
        if protected:
            cases.append({
                "alias": alias,
                "status": "protected_preflight_excluded",
                "tracking_binding_path": _relative(binding_path),
                "tracking_binding_sha256": row["tracking_binding_sha256"],
                "tracking_cache_key": row["tracking_cache_key"],
                "mask_data_dir": mask_rel,
                "preview_path": preview_rel,
                "archive_members": member_paths,
            })
            continue
        if not mask_dir.is_dir() or mask_dir.is_symlink():
            raise RuntimeError(f"eligible mask_data directory is missing or unsafe: {alias}")
        if not preview_path.is_file() or preview_path.is_symlink():
            raise RuntimeError(f"eligible all-targets preview is missing or unsafe: {alias}")
        size, digest = _assert_regular_cache_file(preview_path, tracker)
        preview_item = {"path": preview_rel, "kind": "unused_all_targets_preview", "bytes": size, "sha256": digest}
        old_preview = previews.get(preview_rel)
        if old_preview is not None and old_preview != preview_item:
            raise RuntimeError("duplicate preview path has conflicting bindings")
        previews[preview_rel] = preview_item
        for candidate in sorted(mask_dir.iterdir(), key=lambda p: p.name):
            if candidate.suffix.lower() != ".json":
                continue
            member_size, member_sha = _assert_regular_cache_file(candidate, tracker, suffix=".json")
            member_rel = _relative(candidate)
            member_item = {"original_path": member_rel, "bytes": member_size, "sha256": member_sha}
            old = mask_jsons.get(member_rel)
            if old is not None and old != member_item:
                raise RuntimeError("duplicate mask_data JSON path has conflicting bindings")
            mask_jsons[member_rel] = member_item
            member_paths.append(member_rel)
        cases.append({
            "alias": alias,
            "status": "eligible_retirement",
            "tracking_binding_path": _relative(binding_path),
            "tracking_binding_sha256": row["tracking_binding_sha256"],
            "tracking_cache_key": row["tracking_cache_key"],
            "mask_data_dir": mask_rel,
            "preview_path": preview_rel,
            "preview_bytes": size,
            "preview_sha256": digest,
            "archive_members": member_paths,
        })
    preview_rows = [previews[path] for path in sorted(previews)]
    archive_members = [mask_jsons[path] for path in sorted(mask_jsons)]
    selection = barrier["selection_sha256"]
    archive_value = None
    if archive_members:
        archive_path = ARCHIVES / f"{selection}.tar.gz"
        archive_value = {"path": _relative(archive_path), "bytes": 0, "sha256": "0" * 64,
                         "members": archive_members}
    return {
        "schema": SCHEMA,
        "status": "prepared",
        "selection_sha256": selection,
        "tracking_batch_barrier_path": _relative(tracking_path),
        "tracking_batch_barrier_sha256": _sha(tracking_path),
        "tracking_runtime_sha256": tracking_sha,
        "downstream_seals": dict(seals),
        "count": barrier["count"],
        "aliases": list(barrier["aliases"]),
        "cases": cases,
        "retired_preview_files": preview_rows,
        "lossless_mask_data_archive": archive_value,
        "GT_read": False,
        "scope": "FULL-only unused all-targets previews retired; official mask_data JSON retained losslessly; 64-case preflight caches excluded",
    }


def _make_archive(plan: dict[str, Any], tracker: Any) -> None:
    archive_value = plan["lossless_mask_data_archive"]
    if archive_value is None:
        return
    relative = archive_value["path"]
    path = (ROOT / relative).resolve(strict=False)
    if not path.is_relative_to(FULL.resolve(strict=True)) or path.is_symlink():
        raise RuntimeError("mask_data archive path is outside FULL or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("auxiliary archive exists without a matching prepared journal; refusing overwrite")
    members = archive_value["members"]
    with tarfile.open(path, mode="x:gz", compresslevel=6) as tar:
        for row in members:
            member_path = tracker._safe_cache_path(row["original_path"], require_exists=True)
            size, digest = _assert_regular_cache_file(member_path, tracker, suffix=".json")
            if (size, digest) != (row["bytes"], row["sha256"]):
                raise RuntimeError("mask_data JSON changed before archival")
            tar.add(member_path, arcname=row["original_path"], recursive=False)
    archive_value["bytes"] = path.stat().st_size
    archive_value["sha256"] = _sha(path)
    _archive_members(archive_value, tracker)


def _validate_receipt(path: Path, tracker: Any, *, require_retired: bool = True) -> dict[str, Any]:
    receipt = _read(path)
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "complete"
            or receipt.get("GT_read") is not False
            or path.resolve(strict=True) != (AUX_RETIREMENTS / f"{receipt.get('selection_sha256')}.json").resolve(strict=True)):
        raise RuntimeError("auxiliary retirement receipt is malformed or misplaced")
    tracking_path = tracker._safe_root_file(receipt.get("tracking_batch_barrier_path"), base=FULL.resolve(strict=True))
    if (_sha(tracking_path) != receipt.get("tracking_batch_barrier_sha256")
            or receipt.get("tracking_runtime_sha256") != _sha(FULL / "TRACKING_RUNTIME.json")):
        raise RuntimeError("auxiliary receipt no longer binds the main tracking barrier/runtime")
    barrier, _, _ = _validate_tracking_barrier(tracking_path, tracker)
    if (barrier["selection_sha256"] != receipt["selection_sha256"]
            or barrier["aliases"] != receipt.get("aliases")
            or barrier["count"] != receipt.get("count")):
        raise RuntimeError("auxiliary receipt alias selection differs from main tracking barrier")
    for seal in receipt.get("downstream_seals", {}).values():
        if not isinstance(seal, dict) or set(seal) != {"path", "sha256"}:
            raise RuntimeError("auxiliary downstream seal metadata is malformed")
        seal_path = tracker._safe_root_file(seal["path"], base=FULL.resolve(strict=True))
        if _sha(seal_path) != seal["sha256"]:
            raise RuntimeError("auxiliary downstream seal bytes changed")
    if set(receipt.get("downstream_seals", {})) != {"teacher_batch", "student_batch", "root_batch"}:
        raise RuntimeError("auxiliary receipt does not bind all three downstream seals")
    protected_files, protected_dirs = tracker._preflight_protected_cache_roots()
    rows = receipt.get("cases")
    if not isinstance(rows, list) or len(rows) != barrier["count"]:
        raise RuntimeError("auxiliary receipt does not have one row per selected alias")
    by_alias = {row.get("alias"): row for row in barrier["records"]}
    if len(by_alias) != barrier["count"]:
        raise RuntimeError("main tracking batch aliases are duplicated")
    expected_preview: dict[str, dict[str, Any]] = {}
    expected_member_paths: set[str] = set()
    for alias, case in zip(barrier["aliases"], rows):
        source = by_alias.get(alias)
        if (case.get("alias") != alias or source is None
                or case.get("tracking_binding_path") != source.get("tracking_binding_path")
                or case.get("tracking_binding_sha256") != source.get("tracking_binding_sha256")
                or case.get("tracking_cache_key") != source.get("tracking_cache_key")):
            raise RuntimeError("auxiliary case metadata differs from its tracking barrier row")
        binding_path = tracker._safe_root_file(case["tracking_binding_path"], base=FULL / "tracking_cases")
        if _sha(binding_path) != case["tracking_binding_sha256"]:
            raise RuntimeError("auxiliary tracking binding changed")
        binding = _read(binding_path)
        if (binding.get("case_alias") != alias or binding.get("tracking_cache_key") != case["tracking_cache_key"]
                or binding.get("mask_data_dir") is None or binding.get("GT_read") is not False):
            raise RuntimeError("auxiliary source tracking binding is malformed")
        mask_dir, preview_path = _validate_cache_layout(
            tracker._safe_cache_path(binding["mask_data_dir"], require_exists=False),
            tracker.CACHE_ROOT, require_exists=False)
        mask_rel, preview_rel = _relative(mask_dir), _relative(preview_path)
        if case.get("mask_data_dir") != mask_rel or case.get("preview_path") != preview_rel:
            raise RuntimeError("auxiliary paths differ from exact official cache layout")
        protected = _is_protected(mask_dir, protected_files, protected_dirs)
        if protected:
            if case.get("status") != "protected_preflight_excluded" or case.get("archive_members") != []:
                raise RuntimeError("a protected 64-case mask_data directory was included for retirement")
            continue
        if case.get("status") != "eligible_retirement":
            raise RuntimeError("an unprotected FULL auxiliary cache was incorrectly excluded")
        preview_item = {"path": preview_rel, "kind": "unused_all_targets_preview",
                        "bytes": case.get("preview_bytes"), "sha256": case.get("preview_sha256")}
        if (isinstance(preview_item["bytes"], bool) or not isinstance(preview_item["bytes"], int)
                or preview_item["bytes"] <= 0 or not SHA_RE.fullmatch(str(preview_item["sha256"]))):
            raise RuntimeError("auxiliary preview metadata is malformed")
        previous = expected_preview.get(preview_rel)
        if previous is not None and previous != preview_item:
            raise RuntimeError("duplicate preview has conflicting retirement metadata")
        expected_preview[preview_rel] = preview_item
        members = case.get("archive_members")
        if not isinstance(members, list) or len(set(members)) != len(members):
            raise RuntimeError("auxiliary per-case archive member list is malformed")
        for member_path in members:
            member_path = _safe_workspace_rel(member_path)
            member = tracker._safe_cache_path(member_path, require_exists=False)
            if member.parent != mask_dir or member.suffix.lower() != ".json":
                raise RuntimeError("auxiliary archive member escapes direct mask_data/*.json scope")
            expected_member_paths.add(member_path)

    preview_rows = receipt.get("retired_preview_files")
    if not isinstance(preview_rows, list):
        raise RuntimeError("auxiliary receipt lacks retired preview file rows")
    observed_preview: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        if not isinstance(row, dict) or set(row) != {"path", "kind", "bytes", "sha256"}:
            raise RuntimeError("auxiliary preview row has an unexpected schema")
        path_value = _safe_workspace_rel(row["path"])
        if row["kind"] != "unused_all_targets_preview" or path_value in observed_preview:
            raise RuntimeError("auxiliary preview path is invalid or duplicated")
        observed_preview[path_value] = row
    if observed_preview != expected_preview:
        raise RuntimeError("retired preview set differs from exact per-case auxiliary cache set")
    archive_value = receipt.get("lossless_mask_data_archive")
    archive_members = _archive_members(archive_value, tracker)
    if set(archive_members) != expected_member_paths:
        raise RuntimeError("lossless mask_data archive does not exactly cover selected raw JSON files")
    if require_retired:
        for rel in observed_preview:
            if tracker._cache_path_is_symlink(rel):
                raise RuntimeError("retired preview path was replaced by a symbolic link")
            path = tracker._safe_cache_path(rel, require_exists=False)
            if path.exists():
                raise RuntimeError("retired all-targets preview still exists")
        for rel in expected_member_paths:
            if tracker._cache_path_is_symlink(rel):
                raise RuntimeError("archived mask_data JSON path was replaced by a symbolic link")
            path = tracker._safe_cache_path(rel, require_exists=False)
            if path.exists():
                raise RuntimeError("archived mask_data JSON still exists in the cache")
    return receipt


def retire(tracking_path: str | Path, teacher_path: str | Path,
           student_path: str | Path, root_path: str | Path) -> dict[str, Any]:
    from scripts import run_stvg_table1_full_tracking_v1 as tracker

    tracking_path = Path(tracking_path).resolve(strict=True)
    barrier, tracking_rel, tracking_runtime_sha = _validate_tracking_barrier(tracking_path, tracker)
    selection = barrier["selection_sha256"]
    _ = tracking_rel
    seals = _seal_downstream(barrier, tracking_path, Path(teacher_path), Path(student_path), Path(root_path), tracker)
    AUX_RETIREMENTS.mkdir(parents=True, exist_ok=True)
    receipt_path = AUX_RETIREMENTS / f"{selection}.json"
    if receipt_path.exists():
        receipt = _validate_receipt(receipt_path, tracker)
        if receipt.get("downstream_seals") != seals:
            raise RuntimeError("existing auxiliary retirement receipt has different downstream seals")
        return {"status": "already_verified", "selection_sha256": selection, "GT_read": False}

    journal_path = PREPARED / f"{selection}.json"
    if journal_path.exists():
        plan = _read(journal_path)
        if (plan.get("schema") != SCHEMA or plan.get("status") != "prepared"
                or plan.get("selection_sha256") != selection
                or plan.get("tracking_batch_barrier_sha256") != _sha(tracking_path)
                or plan.get("tracking_runtime_sha256") != tracking_runtime_sha
                or plan.get("downstream_seals") != seals):
            raise RuntimeError("prepared auxiliary journal does not match current sealed inputs")
        _archive_members(plan.get("lossless_mask_data_archive"), tracker)
    else:
        plan = _make_plan(tracking_path, barrier, tracking_runtime_sha, seals, tracker)
        archive_value = plan["lossless_mask_data_archive"]
        if archive_value is not None:
            _make_archive(plan, tracker)
        _write_immutable(journal_path, plan, tracker)

    # The prepared journal is durable and all JSON archive members were read
    # back before any source file is removed. Missing files are accepted only
    # during this journaled recovery path.
    protected_files, protected_dirs = tracker._preflight_protected_cache_roots()
    for case in plan["cases"]:
        if case["status"] == "protected_preflight_excluded":
            continue
        for value in [
            next(row for row in plan["retired_preview_files"] if row["path"] == case["preview_path"]),
        ]:
            if _is_protected(tracker._safe_cache_path(value["path"], require_exists=False),
                             protected_files, protected_dirs):
                raise RuntimeError("prepared retirement now resolves into the protected 64-case cache")
            if tracker._cache_path_is_symlink(value["path"]):
                raise RuntimeError("all-targets preview path became a symbolic link")
            path = tracker._safe_cache_path(value["path"], require_exists=False)
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.stat().st_size != value["bytes"] or _sha(path) != value["sha256"]:
                    raise RuntimeError("all-targets preview changed since prepared journal")
                path.unlink()
        for rel in case["archive_members"]:
            if _is_protected(tracker._safe_cache_path(rel, require_exists=False),
                             protected_files, protected_dirs):
                raise RuntimeError("prepared archive member now resolves into the protected 64-case cache")
            if tracker._cache_path_is_symlink(rel):
                raise RuntimeError("mask_data JSON path became a symbolic link")
            path = tracker._safe_cache_path(rel, require_exists=False)
            expected = next(row for row in plan["lossless_mask_data_archive"]["members"]
                            if row["original_path"] == rel)
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
                    raise RuntimeError("mask_data JSON changed since prepared archive")
                path.unlink()
    receipt = dict(plan)
    receipt["status"] = "complete"
    receipt["completed_at_unix"] = time.time()
    _write_immutable(receipt_path, receipt, tracker)
    _validate_receipt(receipt_path, tracker)
    return {"status": "pass", "selection_sha256": selection,
            "preview_files_retired": len(plan["retired_preview_files"]),
            "mask_data_json_files_archived": len((plan["lossless_mask_data_archive"] or {}).get("members", [])),
            "GT_read": False}


def verify_all() -> dict[str, Any]:
    from scripts import run_stvg_table1_full_tracking_v1 as tracker
    if not AUX_RETIREMENTS.is_dir():
        raise RuntimeError("no auxiliary retirement receipts exist")
    files = sorted(AUX_RETIREMENTS.glob("*.json"))
    if not files:
        raise RuntimeError("no auxiliary retirement receipts exist")
    for path in files:
        _validate_receipt(path, tracker)
    return {"status": "pass", "verified_receipts": len(files), "GT_read": False}


def verify_all_and_write(output_path: str | Path) -> dict[str, Any]:
    """Verify the exact locked 752-batch plan, then write root custody readback."""
    from scripts import run_stvg_table1_full_tracking_v1 as tracker
    output = Path(output_path).resolve(strict=False)
    if output != CUSTODY_READBACK.resolve(strict=False):
        raise RuntimeError("custody readback must use FULL/ROOT_AUXILIARY_CUSTODY_READBACK.json")
    if not BATCH_PLAN.is_file():
        raise RuntimeError("BATCH_PLAN007.json is required for full auxiliary custody verification")
    plan = _read(BATCH_PLAN)
    batches = plan.get("batches")
    if (plan.get("schema") != "stvg-final-table1-fixed-batch-plan-v1"
            or plan.get("status") != "locked" or plan.get("GT_read") is not False
            or plan.get("count") != 13785 or plan.get("batch_count") != 752
            or not isinstance(batches, list) or len(batches) != 752):
        raise RuntimeError("BATCH_PLAN007 is not the locked 13,785-alias/752-batch plan")
    expected = [row.get("selection_sha256") for row in batches]
    if (any(not SHA_RE.fullmatch(str(value)) for value in expected)
            or len(set(expected)) != len(expected)):
        raise RuntimeError("BATCH_PLAN007 selection hashes are malformed or duplicated")
    plan_aliases: dict[str, list[str]] = {}
    seen_aliases: set[str] = set()
    for row in batches:
        aliases = row.get("aliases")
        if (not isinstance(aliases, list) or not aliases or len(set(aliases)) != len(aliases)
                or tracker._selection_sha256(aliases) != row["selection_sha256"]
                or seen_aliases.intersection(aliases)):
            raise RuntimeError("BATCH_PLAN007 contains a malformed or overlapping alias batch")
        seen_aliases.update(aliases)
        plan_aliases[row["selection_sha256"]] = aliases
    if len(seen_aliases) != plan["count"]:
        raise RuntimeError("BATCH_PLAN007 aliases do not cover the locked roster count")
    result = verify_all()
    files = sorted(AUX_RETIREMENTS.glob("*.json"))
    observed = {path.stem for path in files}
    if observed != set(expected):
        raise RuntimeError("auxiliary retirement receipts do not exactly cover BATCH_PLAN007 selections")
    rows = []
    for path in files:
        receipt = _validate_receipt(path, tracker)
        if receipt.get("aliases") != plan_aliases[receipt["selection_sha256"]]:
            raise RuntimeError("auxiliary receipt aliases differ from BATCH_PLAN007")
        rows.append({
            "selection_sha256": receipt["selection_sha256"],
            "receipt_path": _relative(path),
            "receipt_sha256": _sha(path),
            "archive_sha256": (receipt["lossless_mask_data_archive"] or {}).get("sha256"),
            "mask_data_member_count": len((receipt["lossless_mask_data_archive"] or {}).get("members", [])),
            "retired_preview_count": len(receipt["retired_preview_files"]),
        })
    rows.sort(key=lambda row: row["selection_sha256"])
    readback = {
        "schema": "stvg-final-table1-auxiliary-custody-readback-v1",
        "status": "pass",
        "GT_read": False,
        "CPU_only": True,
        "batch_plan_path": _relative(BATCH_PLAN),
        "batch_plan_sha256": _sha(BATCH_PLAN),
        "planned_batch_count": len(expected),
        "verified_receipt_count": result["verified_receipts"],
        "selection_set_exact": True,
        "archive_member_bytes_verified": True,
        "deleted_preview_files_reopened": False,
        "records": rows,
    }
    _write_immutable(output, readback, tracker)
    return readback


def main() -> int:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--retire", action="store_true")
    mode.add_argument("--verify-all", action="store_true")
    parser.add_argument("--tracking")
    parser.add_argument("--teacher")
    parser.add_argument("--student")
    parser.add_argument("--root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.verify_all:
        if not args.output:
            parser.error("--verify-all requires --output FULL/ROOT_AUXILIARY_CUSTODY_READBACK.json")
        result = verify_all_and_write(args.output)
    else:
        if not all((args.tracking, args.teacher, args.student, args.root)):
            parser.error("--retire requires --tracking, --teacher, --student, and --root")
        result = retire(args.tracking, args.teacher, args.student, args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
