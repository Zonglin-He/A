"""Finite 128-query official-clean dispatcher.

This module coordinates the existing frozen pipeline stages. It does not define
models, losses, metrics, or a new scoring path. Batch prediction barriers are
sealed independently; official scoring is dispatched once, after all batch
roots have been merged and verified.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any

from . import pipeline
from .runtime import BASE, ROOT, read, sha


BATCH_SIZE = 128
TOTAL_COUNT = 13_785
DOMAIN_COUNTS = {"hc2": 3_482, "vidstg": 10_303}
PARENT_COUNTS = {"hc2": 237, "vidstg": 732}
DOMAIN_ORDER = ("hc2", "vidstg")
PIPELINE_STAGES = (
    "source",
    "inputs",
    "temporal_teacher",
    "temporal_fit",
    "temporal_readout",
    "spatial_inputs",
    "spatial_teacher",
    "spatial_student",
)
STAGE_BARRIERS = {
    "source": "SOURCE_BARRIER.json",
    "inputs": "TEMPORAL_INPUT_BARRIER.json",
    "temporal_teacher": "TEMPORAL_TEACHER_BARRIER.json",
    "temporal_fit": "TEMPORAL_FIT_BARRIER.json",
    "temporal_readout": "TEMPORAL_PREDICTION_BARRIER.json",
    "spatial_inputs": "SPATIAL_INPUT_BARRIER.json",
    "spatial_teacher": "SPATIAL_TEACHER_BARRIER.json",
    "spatial_student": "GLOBAL_PREDICTION_BARRIER.json",
}
GPU_STAGES = {"source", "temporal_teacher", "temporal_readout", "spatial_teacher", "spatial_student"}
TEACHER_INPUT_DIRS = ("temporal_inputs", "spatial_inputs")
MIB = 1 << 20
GIB = 1 << 30


def _official_root() -> Path:
    return pipeline.path("official_clean")


def _scope(batch_id: str) -> str:
    if len(batch_id) != 5 or batch_id[0] != "B" or not batch_id[1:].isdigit():
        raise ValueError(f"invalid official batch id: {batch_id!r}")
    return f"official_clean/batches/{batch_id}"


def _batch_root(batch_id: str) -> Path:
    return pipeline.path(_scope(batch_id))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    """Write metadata atomically; immutable receipts accept only exact replay."""
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() == data:
            return {"path": str(path.relative_to(ROOT)), "sha256": _digest_bytes(data), "bytes": len(data)}
        raise FileExistsError(f"refusing to replace a non-identical lock/receipt: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": _digest_bytes(data), "bytes": len(data)}


def _canonical_record_hash(record: dict[str, Any]) -> str:
    return _digest_bytes(_json_bytes(record))


def _read_roster() -> tuple[list[dict[str, Any]], str, int]:
    path = _official_root() / "ROSTER_LOCK.json"
    raw = path.read_bytes()
    lock = json.loads(raw)
    rows = lock.get("records")
    if lock.get("status") != "locked" or lock.get("count") != TOTAL_COUNT or not isinstance(rows, list) or len(rows) != TOTAL_COUNT:
        raise RuntimeError("official-clean ROSTER_LOCK is absent or not the complete 13,785-query lock")
    return rows, _digest_bytes(raw), len(raw)


def _validate_original_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    parents: dict[str, set[str]] = {domain: set() for domain in DOMAIN_ORDER}
    last_ordinal: dict[str, int] = {domain: -1 for domain in DOMAIN_ORDER}
    seen_domains: list[str] = []
    aliases: set[str] = set()
    for index, row in enumerate(rows):
        domain = row.get("dataset")
        if domain not in DOMAIN_ORDER:
            raise RuntimeError(f"unexpected domain at roster index {index}")
        if not seen_domains or seen_domains[-1] != domain:
            if domain in seen_domains:
                raise RuntimeError("official roster interleaves domains; refusing to reorder it")
            seen_domains.append(domain)
        ordinal = row.get("query_ordinal")
        if not isinstance(ordinal, int) or ordinal <= last_ordinal[domain]:
            raise RuntimeError(f"official roster ordinal order is not strictly increasing for {domain}")
        last_ordinal[domain] = ordinal
        counts[domain] += 1
        parent = row.get("parent_sha256")
        if not isinstance(parent, str) or len(parent) != 64:
            raise RuntimeError(f"missing parent identity at roster index {index}")
        parents[domain].add(parent)
        alias = row.get("probe_alias")
        if not isinstance(alias, str) or alias in aliases:
            raise RuntimeError(f"missing or duplicate alias at roster index {index}")
        aliases.add(alias)
    if seen_domains != list(DOMAIN_ORDER):
        raise RuntimeError(f"roster domain order changed: {seen_domains}")
    if counts != DOMAIN_COUNTS:
        raise RuntimeError(f"official roster domain counts changed: {counts}")
    observed_parents = {domain: len(parents[domain]) for domain in DOMAIN_ORDER}
    if observed_parents != PARENT_COUNTS:
        raise RuntimeError(f"official roster parent counts changed: {observed_parents}")
    return {"query_counts": counts, "parent_counts": observed_parents, "domain_order": seen_domains}


def _record_plan_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Expose only identifiers and hashes in the metadata plan, never query text."""
    return {
        "official_index": index,
        "probe_alias": row["probe_alias"],
        "dataset": row["dataset"],
        "query_ordinal": row["query_ordinal"],
        "parent_sha256": row["parent_sha256"],
        "query_key_sha256": row.get("query_key_sha256"),
        "record_sha256": _canonical_record_hash(row),
    }


def _make_plan(rows: list[dict[str, Any]], roster_sha256: str, roster_bytes: int) -> dict[str, Any]:
    validation = _validate_original_order(rows)
    batches: list[dict[str, Any]] = []
    cursor = 0
    for domain in DOMAIN_ORDER:
        domain_rows = [i for i in range(cursor, len(rows)) if rows[i]["dataset"] == domain]
        if not domain_rows:
            raise RuntimeError(f"empty domain block: {domain}")
        start, end = domain_rows[0], domain_rows[-1] + 1
        if domain_rows != list(range(start, end)):
            raise RuntimeError(f"domain block is not contiguous: {domain}")
        if start != cursor:
            raise RuntimeError("roster order changed while building batches")
        for block_start in range(start, end, BATCH_SIZE):
            block_end = min(block_start + BATCH_SIZE, end)
            selected = rows[block_start:block_end]
            batch_index = len(batches)
            batches.append({
                "batch_id": f"B{batch_index:04d}",
                "dataset": domain,
                "official_start": block_start,
                "official_stop_exclusive": block_end,
                "count": len(selected),
                "ordinal_first": selected[0]["query_ordinal"],
                "ordinal_last": selected[-1]["query_ordinal"],
                "records_sha256": _digest_bytes(_json_bytes(selected)),
                "records": [_record_plan_row(row, block_start + j) for j, row in enumerate(selected)],
            })
        cursor = end
    # Keep each domain in its own contiguous block.  The split is therefore
    # ceil(3482/128) + ceil(10303/128) = 28 + 81, rather than batching across
    # the domain boundary into 108 mixed/domain-crossing batches.
    if cursor != TOTAL_COUNT or len(batches) != 109:
        raise RuntimeError(f"batch plan size mismatch: cursor={cursor}, batches={len(batches)}")
    return {
        "schema": "stvg-main-official-clean-batch-plan-v1",
        "status": "locked",
        "batch_size": BATCH_SIZE,
        "total_count": TOTAL_COUNT,
        "source_roster": {"path": str((_official_root() / "ROSTER_LOCK.json").relative_to(ROOT)), "sha256": roster_sha256, "bytes": roster_bytes},
        "query_counts": validation["query_counts"],
        "parent_counts": validation["parent_counts"],
        "domain_order": validation["domain_order"],
        "ordering": "preserve locked roster order; each domain is a contiguous block in original query_ordinal order",
        "batches": batches,
        "GT_read": False,
    }


def build_plan() -> dict[str, Any]:
    rows, roster_sha256, roster_bytes = _read_roster()
    plan = _make_plan(rows, roster_sha256, roster_bytes)
    root = _official_root()
    plan_ref = _write_json(root / "BATCH_PLAN.json", plan)
    receipt = {
        "status": "pass",
        "count": len(plan["batches"]),
        "query_count": TOTAL_COUNT,
        "batch_size": BATCH_SIZE,
        "plan": plan_ref,
        "GT_read": False,
    }
    _write_json(root / "BATCH_PLAN_RECEIPT.json", receipt)
    return plan


def _load_plan() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, roster_sha256, roster_bytes = _read_roster()
    path = _official_root() / "BATCH_PLAN.json"
    if not path.is_file():
        plan = build_plan()
    else:
        plan = read(path)
    expected = _make_plan(rows, roster_sha256, roster_bytes)
    if plan != expected:
        raise RuntimeError("BATCH_PLAN no longer matches the locked official roster")
    return plan, rows


def _require_engineering_publication_gate() -> None:
    engineering = pipeline.path("engineering32")
    root_path = engineering / "ROOT_READBACK.json"
    cpu_path = engineering / "CPU_COMPLETION.json"
    publication_path = engineering / "GROUP_PUBLICATION_RECEIPT.json"
    missing = [str(p.relative_to(ROOT)) for p in (root_path, cpu_path, publication_path) if not p.is_file()]
    if missing:
        raise RuntimeError("official run blocked until engineering32 ROOT/CPU/publication receipts exist: " + ", ".join(missing))
    root_receipt = read(root_path)
    cpu_receipt = read(cpu_path)
    publication = read(publication_path)
    if root_receipt.get("status") != "pass" or root_receipt.get("count") != 32:
        raise RuntimeError("engineering32 ROOT_READBACK is not pass/32")
    if cpu_receipt.get("status") != "complete" or cpu_receipt.get("count") != 32:
        raise RuntimeError("engineering32 CPU_COMPLETION is not complete/32")
    if publication.get("status") != "pass":
        raise RuntimeError("engineering32 GROUP_PUBLICATION_RECEIPT is not pass")


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key == "MemAvailable":
            return int(value.split()[0]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _gpu0_free_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == "0":
            return int(parts[1]) * MIB
    raise RuntimeError("GPU 0 memory status unavailable")


def _resource_guard(*, needs_gpu: bool) -> dict[str, int]:
    disk_free = shutil.disk_usage(ROOT).free
    host_available = _mem_available()
    if disk_free < 30 * GIB:
        raise RuntimeError("disk free space fell below the 30 GiB floor")
    if host_available < 6 * GIB:
        raise RuntimeError("host MemAvailable fell below the 6 GiB floor")
    gpu_free = _gpu0_free_bytes() if needs_gpu else 0
    if needs_gpu and gpu_free < 4 * GIB:
        raise RuntimeError("GPU 0 free memory fell below the 4 GiB floor")
    return {"disk_free_bytes": disk_free, "host_available_bytes": host_available, "gpu0_free_bytes": gpu_free}


def _barrier(path: Path, expected_count: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read(path)
    if value.get("status") != "complete" or value.get("count") != expected_count:
        return None
    files = value.get("files")
    if files is not None and (not isinstance(files, list) or len(files) != expected_count):
        return None
    return value


def _verify_ref(ref: dict[str, Any], scope_root: Path) -> None:
    rel = ref.get("path")
    expected_sha = ref.get("sha256")
    expected_bytes = ref.get("bytes")
    if not isinstance(rel, str) or not isinstance(expected_sha, str) or not isinstance(expected_bytes, int):
        raise RuntimeError("barrier file reference is missing path/SHA/bytes metadata")
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("barrier file reference has an unsafe relative path")
    candidate = ROOT / relative
    if candidate.is_symlink():
        raise RuntimeError("barrier file reference is a symlink")
    scope_resolved = scope_root.resolve()
    current = candidate.parent
    while current != scope_resolved and scope_resolved in current.parents:
        if current.is_symlink():
            raise RuntimeError("barrier file reference traverses a symlink directory")
        current = current.parent
    target = candidate.resolve()
    try:
        target.relative_to(scope_resolved)
    except ValueError as exc:
        raise RuntimeError("barrier references a file outside its batch scope") from exc
    if not target.is_file() or target.is_symlink():
        raise RuntimeError("barrier file is absent or not a regular file")
    if target.stat().st_size != expected_bytes or sha(target) != expected_sha:
        raise RuntimeError("barrier file bytes no longer match its sealed SHA")


def _validate_id_file_refs(barrier: dict[str, Any], rows: list[dict[str, Any]], scope_root: Path, dirname: str) -> list[dict[str, Any]]:
    refs = barrier.get("files")
    if not isinstance(refs, list) or len(refs) != len(rows):
        raise RuntimeError(f"{dirname} barrier does not cover the batch roster")
    expected = [row["probe_alias"] for row in rows]
    by_alias: dict[str, dict[str, Any]] = {}
    for ref in refs:
        _verify_ref(ref, scope_root)
        p = Path(ref["path"])
        if p.suffix != ".pt" or p.parent.name != dirname:
            raise RuntimeError(f"unexpected file directory in {dirname} barrier")
        alias = p.stem
        if alias in by_alias:
            raise RuntimeError(f"duplicate {dirname} case reference: {alias}")
        by_alias[alias] = ref
    if set(by_alias) != set(expected):
        raise RuntimeError(f"{dirname} barrier case-id set differs from the locked roster")
    return [by_alias[alias] for alias in expected]


def _validate_batch_root(batch_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scope_root = _batch_root(batch_id)
    count = len(rows)
    source = _barrier(scope_root / "SOURCE_BARRIER.json", count)
    predictions = _barrier(scope_root / "GLOBAL_PREDICTION_BARRIER.json", count)
    root = scope_root / "ROOT_READBACK.json"
    if source is None or predictions is None or not root.is_file():
        raise RuntimeError(f"{batch_id} does not yet have complete source/prediction/root barriers")
    source_files = _validate_id_file_refs(source, rows, scope_root, "source")
    prediction_files = _validate_id_file_refs(predictions, rows, scope_root, "predictions")
    root_receipt = read(root)
    if root_receipt.get("status") != "pass" or root_receipt.get("count") != count:
        raise RuntimeError(f"{batch_id} ROOT_READBACK is not pass for the full batch")
    if root_receipt.get("actual_complete_decoder") is not True or root_receipt.get("temporal_invariants_exact") is not True:
        raise RuntimeError(f"{batch_id} ROOT_READBACK lacks decoder/temporal invariant confirmation")
    return {"source": source, "source_files": source_files, "predictions": predictions,
            "prediction_files": prediction_files, "root": root_receipt}


def _stage_complete(scope_root: Path, stage: str, expected_count: int) -> bool:
    barrier = _barrier(scope_root / STAGE_BARRIERS[stage], expected_count)
    if barrier is None:
        return False
    if stage == "spatial_student":
        try:
            _validate_batch_root(scope_root.name, read(scope_root / "ROSTER_LOCK.json")["records"])
        except (OSError, ValueError, RuntimeError, KeyError):
            return False
    return True


def _attempt_number(attempt_dir: Path) -> int:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in attempt_dir.glob("attempt*.LAUNCH.json"):
        try:
            numbers.append(int(path.name.split("attempt", 1)[1].split(".", 1)[0]))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def _existing_unfinished_attempt(attempt_dir: Path, scope: str, stage: str) -> None:
    for launch_path in sorted(attempt_dir.glob("attempt*.LAUNCH.json")):
        suffix = launch_path.name.removesuffix(".LAUNCH.json")
        exit_path = attempt_dir / f"{suffix}.EXIT.json"
        if exit_path.exists():
            continue
        launch = read(launch_path)
        pid = launch.get("pid")
        if not isinstance(pid, int):
            raise RuntimeError(f"unfinished launch record has no PID: {launch_path}")
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            raise RuntimeError(f"PID {pid} ended without an EXIT receipt; refusing to guess its status")
        cmdline = proc / "cmdline"
        argv = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace") if cmdline.exists() else ""
        if "-m main_method.run" not in argv or stage not in argv or scope not in argv:
            raise RuntimeError(f"PID {pid} no longer matches the recorded pipeline worker; refusing to signal or replace it")
        raise RuntimeError(f"pipeline worker PID {pid} is still active; this dispatcher will not start a second worker")


def _run_stage(scope: str, stage: str, *, expected_count: int, log_root: Path) -> dict[str, Any]:
    if stage not in (*PIPELINE_STAGES, "score"):
        raise ValueError(f"unsupported pipeline stage: {stage}")
    scope_root = pipeline.path(scope)
    if stage in PIPELINE_STAGES and _stage_complete(scope_root, stage, expected_count):
        return {"status": "already_complete", "stage": stage, "scope": scope, "count": expected_count}
    if stage == "score":
        cpu_path = scope_root / "CPU_COMPLETION.json"
        if cpu_path.is_file():
            receipt = read(cpu_path)
            if receipt.get("status") == "complete" and receipt.get("count") == expected_count:
                return {"status": "already_complete", "stage": stage, "scope": scope, "count": expected_count}

    attempt_dir = log_root / "attempts" / stage
    _existing_unfinished_attempt(attempt_dir, scope, stage)
    number = _attempt_number(attempt_dir)
    stem = f"attempt{number:04d}"
    launch_path = attempt_dir / f"{stem}.LAUNCH.json"
    exit_path = attempt_dir / f"{stem}.EXIT.json"
    log_path = attempt_dir / f"{stem}.log"
    argv = [sys.executable, "-m", "main_method.run", stage, "--scope", scope]
    resources = _resource_guard(needs_gpu=stage in GPU_STAGES)
    started = time.time()
    with log_path.open("ab") as log:
        try:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException as exc:
            _write_json(launch_path, {
                "status": "spawn_failed", "stage": stage, "scope": scope,
                "argv": ["<sys.executable>", *argv[1:]], "pid": None,
                "log_path": str(log_path.relative_to(ROOT)), "started_epoch": started,
                "resources_before_launch": resources, "GT_read": False,
            })
            _write_json(exit_path, {"status": "spawn_failed", "stage": stage, "scope": scope,
                                    "exit_code": None, "error_type": type(exc).__name__,
                                    "finished_epoch": time.time(), "GT_read": False})
            raise
        _write_json(launch_path, {
            "status": "launched", "stage": stage, "scope": scope,
            "argv": argv, "pid": child.pid,
            "log_path": str(log_path.relative_to(ROOT)), "started_epoch": started,
            "resources_before_launch": resources, "GT_read": stage == "score",
        })
        log.flush()
        try:
            exit_code = child.wait()
        except KeyboardInterrupt:
            # Do not signal a worker that owns the shared GPU lease. Its PID and
            # LAUNCH receipt remain truthful; a later dispatcher will refuse to
            # duplicate it until an EXIT receipt is available.
            raise
    _write_json(exit_path, {
        "status": "complete" if exit_code == 0 else "failed",
        "stage": stage, "scope": scope, "pid": child.pid,
        "exit_code": exit_code, "started_epoch": started, "finished_epoch": time.time(),
        "log_path": str(log_path.relative_to(ROOT)), "GT_read": stage == "score",
    })
    if exit_code != 0:
        raise RuntimeError(f"pipeline stage {stage} exited with code {exit_code}; see saved log")
    if stage in PIPELINE_STAGES and not _stage_complete(scope_root, stage, expected_count):
        raise RuntimeError(f"pipeline stage {stage} exited zero without its complete barrier")
    if stage == "score":
        result = read(scope_root / "CPU_COMPLETION.json")
        if result.get("status") != "complete" or result.get("count") != expected_count:
            raise RuntimeError("score stage exited zero without complete CPU_COMPLETION")
    return {"status": "complete", "stage": stage, "scope": scope, "pid": child.pid,
            "exit_code": exit_code, "log_path": str(log_path.relative_to(ROOT))}


def _write_batch_roster(batch: dict[str, Any], all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start, stop = batch["official_start"], batch["official_stop_exclusive"]
    rows = all_rows[start:stop]
    if len(rows) != batch["count"] or _digest_bytes(_json_bytes(rows)) != batch["records_sha256"]:
        raise RuntimeError(f"{batch['batch_id']} slice no longer matches the plan")
    batch_root = _batch_root(batch["batch_id"])
    roster = {"status": "locked", "count": len(rows), "records": rows, "GT_read": False,
              "batch_id": batch["batch_id"], "official_start": start,
              "official_stop_exclusive": stop, "source_roster_sha256": batch["records_sha256"]}
    _write_json(batch_root / "ROSTER_LOCK.json", roster)
    return rows


def _make_file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular file: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}


def _prefix_retirement(batch_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch_root = _batch_root(batch_id)
    _validate_batch_root(batch_id, rows)
    prefix = batch_root / "prefix"
    files = sorted(prefix.glob("*.pt")) if prefix.exists() else []
    expected = {r["probe_alias"] for r in rows}
    receipt_path = batch_root / "PREFIX_RETIREMENT_RECEIPT.json"
    if receipt_path.exists():
        current = read(receipt_path)
        if current.get("status") == "retired":
            if any((ROOT / x["path"]).exists() for x in current.get("files", [])):
                raise RuntimeError("retired prefix file unexpectedly reappeared")
            return current
        if current.get("status") != "verified_before_retirement" or current.get("batch_id") != batch_id:
            raise RuntimeError("existing prefix retirement receipt is not recoverable")
        refs = current.get("files")
        if not isinstance(refs, list) or len(refs) != len(expected):
            raise RuntimeError("prefix retirement receipt does not cover the batch roster")
        by_alias = {Path(x.get("path", "")).stem: x for x in refs}
        if set(by_alias) != expected:
            raise RuntimeError("prefix retirement receipt case set differs from the batch roster")
        existing = {p.stem: p for p in files}
        if not set(existing).issubset(expected):
            raise RuntimeError("unindexed prefix cache appeared after the retirement receipt")
        for alias, path in existing.items():
            ref = by_alias[alias]
            if path.stat().st_size != ref.get("bytes") or sha(path) != ref.get("sha256"):
                raise RuntimeError("remaining prefix cache differs from the retirement receipt")
    else:
        if {p.stem for p in files} != expected:
            raise RuntimeError("new prefix cache set does not match the complete batch roster")
        refs = [_make_file_receipt(p) for p in files]
        intent = {
            "status": "verified_before_retirement", "batch_id": batch_id,
            "count": len(refs), "bytes": sum(f["bytes"] for f in refs), "files": refs,
            "preserved": ["source captures", "Source caches", "predictions", "fits", "Teacher responses", "job metadata"],
            "GT_read": False,
        }
        _write_json(receipt_path, intent)
    for ref in refs:
        p = ROOT / ref["path"]
        if not p.exists():
            continue
        if not p.is_file() or p.is_symlink() or p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
            raise RuntimeError("prefix cache changed after retirement receipt")
    for ref in refs:
        p = ROOT / ref["path"]
        if p.exists():
            p.unlink()
    if any((ROOT / x["path"]).exists() for x in refs):
        raise RuntimeError("prefix cache retirement was incomplete")
    complete = {**intent, "status": "retired", "retired_count": len(refs), "retired_bytes": sum(f["bytes"] for f in refs)}
    _write_json(receipt_path, complete, replace=True)
    return complete


def _teacher_input_files(batch_root: Path) -> list[Path]:
    files: list[Path] = []
    for name in TEACHER_INPUT_DIRS:
        base = batch_root / name
        if not base.exists():
            continue
        for path in base.rglob("*.npy"):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("teacher input archive rejects symlinks/non-regular files")
            current = path.parent
            while current != base and base in current.parents:
                if current.is_symlink():
                    raise RuntimeError("teacher input archive rejects symlink directories")
                current = current.parent
            path.resolve().relative_to(batch_root.resolve())
            files.append(path)
    return sorted(files)


def _tar_add_file(tf: tarfile.TarFile, source: Path, batch_root: Path) -> None:
    relative = source.relative_to(batch_root).as_posix()
    info = tarfile.TarInfo(relative)
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as f:
        tf.addfile(info, f)


def _verify_input_archive(archive: Path, batch_root: Path, expected: list[dict[str, Any]]) -> None:
    expected_by_name = {x["member"]: x for x in expected}
    if len(expected_by_name) != len(expected):
        raise RuntimeError("duplicate input member names in archive manifest")
    with tarfile.open(archive, mode="r:gz") as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        if set(names) != set(expected_by_name) or len(names) != len(expected_by_name):
            raise RuntimeError("tar member set differs from the input archive index")
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.isdir() or not member.isfile():
                raise RuntimeError("unsafe member type or path in teacher input archive")
            ref = expected_by_name[member.name]
            if member.size != ref["bytes"]:
                raise RuntimeError("teacher input archive member byte count changed")
            stream = tf.extractfile(member)
            if stream is None:
                raise RuntimeError("teacher input archive member cannot be read")
            h = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(8 << 20), b""):
                size += len(block)
                h.update(block)
            if size != ref["bytes"] or h.hexdigest() != ref["sha256"]:
                raise RuntimeError("teacher input archive member SHA verification failed")


def archive_and_retire_teacher_inputs(batch_id: str) -> dict[str, Any]:
    """Archive only generated Teacher .npy inputs after a complete batch ROOT."""
    plan, all_rows = _load_plan()
    batch = next((x for x in plan["batches"] if x["batch_id"] == batch_id), None)
    if batch is None:
        raise ValueError(f"batch absent from fixed plan: {batch_id}")
    rows = all_rows[batch["official_start"]:batch["official_stop_exclusive"]]
    _validate_batch_root(batch_id, rows)
    batch_root = _batch_root(batch_id)
    files = _teacher_input_files(batch_root)
    index_path = batch_root / "TEACHER_INPUT_ARCHIVE_INDEX.json"
    archive_path = batch_root / "teacher_inputs.tar.gz"
    prior = read(index_path) if index_path.is_file() else None
    if prior is None and archive_path.exists():
        raise RuntimeError("Teacher archive exists without its reversible index")
    if not files and prior is None:
        empty = {"status": "empty", "batch_id": batch_id, "members": [], "GT_read": False}
        _write_json(index_path, empty)
        return empty

    # If a prior archive exists, its index is the authoritative expected file
    # set.  This makes a crash midway through deleting verified originals
    # resumable without treating the remaining subset as a changed input set.
    if prior is not None:
        if prior.get("batch_id") != batch_id or not isinstance(prior.get("members"), list):
            raise RuntimeError("existing Teacher archive index has an invalid batch identity")
        members = prior["members"]
        for item in members:
            member_path = PurePosixPath(item.get("member", ""))
            if (member_path.is_absolute() or not member_path.parts or ".." in member_path.parts
                    or "\\" in item.get("member", "") or member_path.parts[0] not in TEACHER_INPUT_DIRS
                    or member_path.suffix != ".npy"):
                raise RuntimeError("Teacher archive index contains an unsafe member path")
            expected_path = batch_root.joinpath(*member_path.parts)
            if item.get("original_path") != str(expected_path.relative_to(ROOT)):
                raise RuntimeError("Teacher archive member path does not match its original path")
        if archive_path.is_file():
            if not isinstance(prior.get("archive"), dict):
                raise RuntimeError("archive exists without its reversible archive receipt")
            if archive_path.stat().st_size != prior["archive"].get("bytes") or sha(archive_path) != prior["archive"].get("sha256"):
                raise RuntimeError("existing Teacher archive SHA/bytes changed")
            _verify_input_archive(archive_path, batch_root, members)
        elif prior.get("status") in {"verified_not_retired", "retired"}:
            raise RuntimeError("Teacher archive index exists but the archive is missing")
        expected_originals = {x["original_path"]: x for x in members}
        actual_originals = {str(p.relative_to(ROOT)): p for p in files}
        if not set(actual_originals).issubset(expected_originals):
            raise RuntimeError("unindexed Teacher input files appeared after archive creation")
        for rel, source in actual_originals.items():
            ref = expected_originals[rel]
            if source.stat().st_size != ref["bytes"] or sha(source) != ref["sha256"]:
                raise RuntimeError("remaining Teacher input differs from the verified archive index")
        if prior.get("status") == "retired":
            return prior
        if prior.get("status") != "verified_not_retired":
            raise RuntimeError("existing Teacher archive index is not recoverable")
    else:
        members = []
        for source in files:
            ref = _make_file_receipt(source)
            members.append({"member": source.relative_to(batch_root).as_posix(),
                            "original_path": ref["path"], "sha256": ref["sha256"], "bytes": ref["bytes"]})

    if archive_path.exists():
        # The archive and index were verified above; only verified originals
        # still present are eligible for retirement below.
        pass
    else:
        total_source_bytes = sum(x["bytes"] for x in members)
        conservative_bound = total_source_bytes + 1024 * len(members) + MIB
        resources = _resource_guard(needs_gpu=False)
        if resources["disk_free_bytes"] - conservative_bound < 30 * GIB:
            receipt = {"status": "not_archived_resource_floor", "batch_id": batch_id,
                       "count": len(members), "source_bytes": total_source_bytes,
                       "required_free_floor_bytes": 30 * GIB, "GT_read": False}
            _write_json(batch_root / "TEACHER_INPUT_ARCHIVE_DEFERRED.json", receipt, replace=True)
            return receipt
        temp = archive_path.with_name(archive_path.name + f".{os.getpid()}.tmp")
        with temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as tf:
                    for source in files:
                        _tar_add_file(tf, source, batch_root)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp, archive_path)
        _verify_input_archive(archive_path, batch_root, members)
        archive_ref = _make_file_receipt(archive_path)
        _resource_guard(needs_gpu=False)
        index = {"status": "verified_not_retired", "batch_id": batch_id,
                 "archive": archive_ref, "count": len(members),
                 "source_bytes": total_source_bytes, "members": members,
                 "restore_rule": "extract only exact indexed .npy members; preserve JOB/JOBS metadata",
                 "GT_read": False}
        _write_json(index_path, index)

    # Recheck every original that remains after archive verification and before
    # unlinking it.  Missing originals are allowed only for a prior verified
    # archive, covering interruption during the retirement loop.
    for item in members:
        source = ROOT / item["original_path"]
        if not source.exists():
            continue
        if not source.is_file() or source.is_symlink() or source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]:
            raise RuntimeError("Teacher input changed after archive verification; no originals retired")
    for item in members:
        source = ROOT / item["original_path"]
        if source.exists():
            source.unlink()
    if any((ROOT / x["original_path"]).exists() for x in members):
        raise RuntimeError("Teacher input retirement was incomplete")
    index = read(index_path)
    complete = {**index, "status": "retired", "retired_count": len(members),
                "retired_bytes": sum(x["bytes"] for x in members)}
    _write_json(index_path, complete, replace=True)
    return complete


def restore_teacher_inputs(batch_id: str) -> dict[str, Any]:
    """Restore archived .npy inputs byte-for-byte without changing JOB metadata."""
    batch_root = _batch_root(batch_id)
    index_path = batch_root / "TEACHER_INPUT_ARCHIVE_INDEX.json"
    if not index_path.is_file():
        raise RuntimeError("teacher input archive index missing")
    index = read(index_path)
    if index.get("status") not in {"retired", "verified_not_retired"}:
        raise RuntimeError("teacher input archive is not in a restorable state")
    archive = ROOT / index["archive"]["path"]
    _verify_ref(index["archive"], batch_root)
    if sha(archive) != index["archive"]["sha256"] or archive.stat().st_size != index["archive"]["bytes"]:
        raise RuntimeError("teacher input archive hash mismatch")
    _verify_input_archive(archive, batch_root, index["members"])
    restored = 0
    with tarfile.open(archive, mode="r:gz") as tf:
        member_map = {m.name: m for m in tf.getmembers()}
        for ref in index["members"]:
            name = ref["member"]
            relative = PurePosixPath(name)
            if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                    or "\\" in name or relative.parts[0] not in TEACHER_INPUT_DIRS
                    or relative.suffix != ".npy"):
                raise RuntimeError("archive index contains a path outside Teacher input directories")
            target = batch_root.joinpath(*relative.parts)
            if target.is_symlink():
                raise RuntimeError("refusing to restore a Teacher input through a symlink")
            current = target.parent
            while current != batch_root and batch_root in current.parents:
                if current.exists() and current.is_symlink():
                    raise RuntimeError("refusing to restore a Teacher input through a symlink directory")
                current = current.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != ref["bytes"] or sha(target) != ref["sha256"]:
                    raise FileExistsError(f"refusing to overwrite non-matching restored input: {target}")
                continue
            stream = tf.extractfile(member_map[name])
            if stream is None:
                raise RuntimeError("archive member missing during restore")
            temp = target.with_name(target.name + f".{os.getpid()}.restore.tmp")
            h = hashlib.sha256()
            size = 0
            with temp.open("wb") as out:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    size += len(block)
                    h.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if size != ref["bytes"] or h.hexdigest() != ref["sha256"]:
                temp.unlink(missing_ok=True)
                raise RuntimeError("restored Teacher input SHA/bytes mismatch")
            os.replace(temp, target)
            restored += 1
    receipt = {"status": "restored", "batch_id": batch_id, "restored_count": restored,
               "indexed_count": index["count"], "archive_sha256": index["archive"]["sha256"], "GT_read": False}
    _write_json(batch_root / "TEACHER_INPUT_RESTORE_RECEIPT.json", receipt, replace=True)
    return receipt


def _run_batch_locked(batch: dict[str, Any], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require_engineering_publication_gate()
    batch_id = batch["batch_id"]
    rows = _write_batch_roster(batch, all_rows)
    count = len(rows)
    scope = _scope(batch_id)
    batch_root = _batch_root(batch_id)
    log_root = batch_root / "driver"
    status_path = log_root / "STATUS.json"
    _write_json(status_path, {"status": "running", "batch_id": batch_id,
                              "count": count, "completed_stages": [], "GT_read": False}, replace=True)
    results = []
    try:
        for stage in PIPELINE_STAGES:
            if _stage_complete(batch_root, stage, count):
                results.append({"status": "already_complete", "stage": stage, "count": count})
                continue
            resources = _resource_guard(needs_gpu=stage in GPU_STAGES)
            result = _run_stage(scope, stage, expected_count=count, log_root=log_root)
            results.append(result)
            _write_json(status_path, {"status": "running", "batch_id": batch_id,
                                      "count": count, "completed_stages": [x["stage"] for x in results],
                                      "last_resources": resources, "GT_read": False}, replace=True)
        root = _validate_batch_root(batch_id, rows)
        prefix = _prefix_retirement(batch_id, rows)
        archive = archive_and_retire_teacher_inputs(batch_id)
        receipt = {"status": "complete", "batch_id": batch_id, "dataset": batch["dataset"],
                   "official_start": batch["official_start"], "official_stop_exclusive": batch["official_stop_exclusive"],
                   "count": count, "source_count": len(root["source_files"]),
                   "prediction_count": len(root["prediction_files"]),
                   "root_status": root["root"].get("status"),
                   "root_readback_sha256": sha(batch_root / "ROOT_READBACK.json"),
                   "source_barrier_sha256": sha(batch_root / "SOURCE_BARRIER.json"),
                   "global_prediction_barrier_sha256": sha(batch_root / "GLOBAL_PREDICTION_BARRIER.json"),
                   "GT_read": False}
        receipt_path = batch_root / "BATCH_ROOT_RECEIPT.json"
        _write_json(receipt_path, receipt)
        _write_json(status_path, {"status": "complete", "batch_id": batch_id,
                                  "count": count, "completed_stages": [x["stage"] for x in results],
                                  "batch_root_receipt": str(receipt_path.relative_to(ROOT)), "GT_read": False}, replace=True)
        return receipt
    except BaseException as exc:
        _write_json(status_path, {"status": "failed", "batch_id": batch_id, "count": count,
                                  "completed_stages": [x.get("stage") for x in results],
                                  "error_type": type(exc).__name__, "GT_read": False}, replace=True)
        raise


def _acquire_dispatch_lock():
    lock_path = _official_root() / "driver" / "DISPATCH.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError("another official dispatcher holds the CPU coordinator lock") from exc
    return lock_file


def run_batch(batch_id: str) -> dict[str, Any]:
    plan, rows = _load_plan()
    batch_index = next((i for i, x in enumerate(plan["batches"]) if x["batch_id"] == batch_id), None)
    if batch_index is None:
        raise ValueError(f"batch absent from fixed plan: {batch_id}")
    lock = _acquire_dispatch_lock()
    try:
        for earlier in plan["batches"][:batch_index]:
            receipt = _batch_root(earlier["batch_id"]) / "BATCH_ROOT_RECEIPT.json"
            if not receipt.is_file() or read(receipt).get("status") != "complete":
                raise RuntimeError(f"cannot start {batch_id} before earlier batch {earlier['batch_id']} is complete")
        batch = plan["batches"][batch_index]
        return _run_batch_locked(batch, rows)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def run_all_batches() -> dict[str, Any]:
    plan, rows = _load_plan()
    lock = _acquire_dispatch_lock()
    completed = []
    try:
        for batch in plan["batches"]:
            receipt = _run_batch_locked(batch, rows)
            completed.append({"batch_id": batch["batch_id"], "count": batch["count"],
                              "status": receipt["status"]})
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    aggregate_score = aggregate_and_score()
    return {"status": "all_batches_complete", "count": TOTAL_COUNT,
            "batch_count": len(completed), "batches": completed,
            "aggregate_and_score": aggregate_score}


def aggregate_and_score() -> dict[str, Any]:
    plan, all_rows = _load_plan()
    _require_engineering_publication_gate()
    lock = _acquire_dispatch_lock()
    try:
        batch_roots = []
        source_by_alias: dict[str, dict[str, Any]] = {}
        predictions_by_alias: dict[str, dict[str, Any]] = {}
        for batch in plan["batches"]:
            batch_id = batch["batch_id"]
            rows = all_rows[batch["official_start"]:batch["official_stop_exclusive"]]
            receipt_path = _batch_root(batch_id) / "BATCH_ROOT_RECEIPT.json"
            if not receipt_path.is_file() or read(receipt_path).get("status") != "complete":
                raise RuntimeError(f"{batch_id} has no complete BATCH_ROOT_RECEIPT")
            checked = _validate_batch_root(batch_id, rows)
            batch_roots.append({"batch_id": batch_id, "count": len(rows),
                                "root_sha256": sha(_batch_root(batch_id) / "ROOT_READBACK.json")})
            for row, ref in zip(rows, checked["source_files"]):
                alias = row["probe_alias"]
                if alias in source_by_alias:
                    raise RuntimeError("duplicate source alias across batches")
                source_by_alias[alias] = ref
            for row, ref in zip(rows, checked["prediction_files"]):
                alias = row["probe_alias"]
                if alias in predictions_by_alias:
                    raise RuntimeError("duplicate prediction alias across batches")
                predictions_by_alias[alias] = ref

        expected_aliases = [r["probe_alias"] for r in all_rows]
        if len(set(expected_aliases)) != TOTAL_COUNT or set(source_by_alias) != set(expected_aliases) or set(predictions_by_alias) != set(expected_aliases):
            raise RuntimeError("batch roots do not cover the exact unique official roster")
        source_files = [source_by_alias[a] for a in expected_aliases]
        prediction_files = [predictions_by_alias[a] for a in expected_aliases]
        for ref in source_files:
            _verify_ref(ref, _official_root())
        for ref in prediction_files:
            _verify_ref(ref, _official_root())
        aggregate_source = {"status": "complete", "count": TOTAL_COUNT,
                            "files": source_files, "batches": batch_roots, "GT_read": False}
        aggregate_predictions = {"status": "complete", "count": TOTAL_COUNT,
                                 "files": prediction_files, "batches": batch_roots,
                                 "all_predictions_sealed_before_GT": True, "GT_read": False}
        root_readback = {"status": "pass", "count": TOTAL_COUNT,
                         "actual_complete_decoder": True, "temporal_invariants_exact": True,
                         "unique_roster_case_ids": TOTAL_COUNT, "batches": batch_roots, "GT_read": False}
        official_root = _official_root()
        source_ref = _write_json(official_root / "SOURCE_BARRIER.json", aggregate_source)
        prediction_ref = _write_json(official_root / "GLOBAL_PREDICTION_BARRIER.json", aggregate_predictions)
        root_ref = _write_json(official_root / "ROOT_READBACK.json", root_readback)
        aggregate_receipt = {"status": "pass", "count": TOTAL_COUNT,
                             "batch_count": len(plan["batches"]),
                             "source_barrier": source_ref,
                             "global_prediction_barrier": prediction_ref,
                             "root_readback": root_ref,
                             "GT_read": False}
        _write_json(official_root / "BATCH_AGGREGATE_ROOT_RECEIPT.json", aggregate_receipt)
        score_log_root = official_root / "driver"
        result = _run_stage("official_clean", "score", expected_count=TOTAL_COUNT, log_root=score_log_root)
        return {"status": "complete", "count": TOTAL_COUNT,
                "aggregate_receipt": str((official_root / "BATCH_AGGREGATE_ROOT_RECEIPT.json").relative_to(ROOT)),
                "score_stage": result}
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="write the fixed metadata-only 128-query batch plan")
    run_one = sub.add_parser("run-batch", help="run one locked batch through prediction ROOT; never score")
    run_one.add_argument("batch_id")
    sub.add_parser("run-all", help="run all batches serially through prediction ROOT; never score")
    sub.add_parser("aggregate-score", help="merge all verified batch roots, then dispatch official score")
    restore = sub.add_parser("restore-inputs", help="restore archived Teacher .npy inputs exactly")
    restore.add_argument("batch_id")
    args = parser.parse_args()
    if args.command == "plan":
        plan = build_plan()
        plan_path = _official_root() / "BATCH_PLAN.json"
        result = {"status": "locked", "count": plan["total_count"],
                  "query_count": plan["total_count"],
                  "batch_count": len(plan["batches"]), "batch_size": plan["batch_size"],
                  "plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": sha(plan_path),
                           "bytes": plan_path.stat().st_size}, "GT_read": False}
    elif args.command == "run-batch":
        result = run_batch(args.batch_id)
    elif args.command == "run-all":
        result = run_all_batches()
    elif args.command == "aggregate-score":
        result = aggregate_and_score()
    elif args.command == "restore-inputs":
        result = restore_teacher_inputs(args.batch_id)
    else:
        parser.error("unknown command")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
