#!/usr/bin/env python3
"""Bounded full-roster Student adapter for Table 1.

The scientific source replay, reference checks, five-arm construction, and
Spatial-A fit are delegated to the frozen, qualified student implementation.
This adapter changes only the FULL artifact namespace and roster/barrier ABI.
It supports same-video groups as bounded batches and a CPU-only finalizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_final_table1_v1"
FULL = BASE / "FULL"
ROSTER_REL = "artifacts/stvg_final_table1_v1/FULL/FULL_SOURCE_ROSTER_LOCK.json"
SOURCE_RUNTIME_REL = "artifacts/stvg_final_table1_v1/FULL/SOURCE_RUNTIME.json"
TEACHER_RUNTIME_REL = "artifacts/stvg_final_table1_v1/FULL/TEACHER_RUNTIME.json"
STUDENT_RUNTIME_REL = "artifacts/stvg_final_table1_v1/FULL/STUDENT_RUNTIME.json"
SOURCE_BARRIER_REL = "artifacts/stvg_final_table1_v1/FULL/SOURCE_CAPTURE_BARRIER.json"
STUDENT_REL = "scripts/run_stvg_table1_full_student_v1.py"
SCIENCE_REL = "scripts/run_stvg_table1_student003.py"
EXPECTED_COUNT = 13785
EXPECTED_DOMAINS = {"hc2": 3482, "vidstg": 10303}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ALIAS_RE = re.compile(r"(?:HC|V)[0-9]{5}\Z")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _safe_repo_path(value: str | Path) -> Path:
    raw = Path(value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("locked artifact must be a file inside repository root")
    return path


def _group_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """A batch key is the exact canonical video plus its clock/bounds.

    Sparse source-grid frame IDs are query-level sampling metadata, not a new
    video identity, so they must not split cases from one canonical clip.
    """
    fps = float(record["fps"])
    duration = float(record["duration"])
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("full-roster video clock must be finite and positive")
    return (
        str(record["video_sha256"]), fps, int(record["width"]), int(record["height"]),
        int(record["frame_count"]), duration,
        tuple(int(x) for x in record["video_bounds"]),
    )


def _parse_aliases_text(text: str) -> list[str]:
    """Accept a JSON alias array/object or one alias per nonblank text line."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("aliases file is empty")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, list):
            values = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("aliases"), list):
            values = parsed["aliases"]
        else:
            raise ValueError("aliases JSON must be a list or an object with an aliases list")
    if not values or any(not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias)
                         for alias in values):
        raise ValueError("aliases file must contain nonempty HC/V plus five-digit IDs")
    if len(values) != len(set(values)):
        raise ValueError("aliases file contains duplicate IDs")
    return list(values)


def _normalize_teacher_files(barrier: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = barrier.get("files")
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = []
        for alias, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("teacher barrier files mapping must contain objects")
            rows.append({"alias": alias, **dict(value)})
    else:
        raise ValueError("teacher barrier must contain a files list or mapping")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("teacher barrier file rows must be objects")
        alias = row.get("alias", row.get("probe_alias", row.get("case_id")))
        if not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias) or alias in result:
            raise ValueError("teacher barrier has an invalid or duplicate alias")
        path = row.get("path")
        digest = row.get("sha256")
        receipt_path = row.get("receipt_path")
        receipt_digest = row.get("receipt_sha256")
        size = row.get("bytes")
        if (not isinstance(path, str) or not isinstance(digest, str)
                or not SHA_RE.fullmatch(digest)
                or not isinstance(receipt_path, str)
                or not isinstance(receipt_digest, str) or not SHA_RE.fullmatch(receipt_digest)
                or isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(f"teacher barrier row is missing immutable output/receipt binding for {alias}")
        result[alias] = dict(row, alias=alias)
    if len(result) != len(rows):
        raise ValueError("teacher barrier aliases are not unique")
    return result


def _validate_roster(roster: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if (roster.get("schema") != "stvg-final-table1-full-source-roster-v1"
            or roster.get("status") not in {"prepared_for_root_runtime_lock", "locked"}
            or roster.get("GT_read") is not False
            or roster.get("selection_uses_GT_or_scores") is not False
            or roster.get("official_query_count") != EXPECTED_COUNT
            or roster.get("domain_counts") != EXPECTED_DOMAINS):
        raise RuntimeError("FULL_SOURCE_ROSTER_LOCK is not the exact 13,785-query GT-free roster")
    rows = roster.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise RuntimeError("FULL_SOURCE_ROSTER_LOCK records are incomplete")
    index: dict[str, dict[str, Any]] = {}
    seen_domain_ordinals: dict[str, set[int]] = {"hc2": set(), "vidstg": set()}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FULL_SOURCE_ROSTER_LOCK records must be objects")
        alias, dataset = row.get("probe_alias"), row.get("dataset")
        ordinal = row.get("query_ordinal")
        expected_prefix = "HC" if dataset == "hc2" else "V" if dataset == "vidstg" else None
        if (not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias) or alias in index
                or expected_prefix is None or not alias.startswith(expected_prefix)
                or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
                or alias != f"{expected_prefix}{ordinal:05d}"
                or row.get("GT_read") is not False):
            raise ValueError("FULL_SOURCE_ROSTER_LOCK has inconsistent alias/domain identity")
        seen_domain_ordinals[dataset].add(ordinal)
        for key in ("query_key_sha256", "safe_record_sha256", "parent_sha256",
                    "plan_sha256", "subject_sha256", "video_sha256"):
            if not isinstance(row.get(key), str) or not SHA_RE.fullmatch(row[key]):
                raise ValueError(f"FULL_SOURCE_ROSTER_LOCK has invalid {key}")
        _group_key(row)
        index[alias] = dict(row)
    if len(index) != EXPECTED_COUNT:
        raise RuntimeError("FULL_SOURCE_ROSTER_LOCK aliases are not unique")
    for dataset, count in EXPECTED_DOMAINS.items():
        if seen_domain_ordinals[dataset] != set(range(count)):
            raise RuntimeError(f"FULL_SOURCE_ROSTER_LOCK ordinal coverage differs for {dataset}")
    if [row["probe_alias"] for row in rows] != list(index):
        raise RuntimeError("FULL_SOURCE_ROSTER_LOCK order is unstable")
    return [dict(row) for row in rows], index


def _select_aliases(rows: Sequence[Mapping[str, Any]], index: Mapping[str, Mapping[str, Any]],
                    *, all_cases: bool, aliases: Sequence[str] | None) -> list[str]:
    if all_cases == (aliases is not None):
        raise ValueError("choose exactly one of --all or --aliases-file")
    if all_cases:
        selected = set(index)
    else:
        assert aliases is not None
        selected = set(aliases)
        unknown = selected - set(index)
        if unknown:
            raise ValueError(f"aliases file contains IDs outside FULL_SOURCE_ROSTER_LOCK: {sorted(unknown)[:3]}")
        group_members: dict[tuple[Any, ...], set[str]] = {}
        for row in rows:
            group_members.setdefault(_group_key(row), set()).add(str(row["probe_alias"]))
        for group in {_group_key(index[alias]) for alias in selected}:
            if not group_members[group].issubset(selected):
                raise ValueError("aliases file splits an exact video_sha256+clock/bounds group")
    return [str(row["probe_alias"]) for row in rows if row["probe_alias"] in selected]


def _verify_runtime_and_source(*, runtime_path: str | Path | None = None) -> dict[str, Any]:
    runtime_path = _safe_repo_path(runtime_path or FULL / "STUDENT_RUNTIME.json")
    source_runtime_path = _safe_repo_path(FULL / "SOURCE_RUNTIME.json")
    teacher_runtime_path = _safe_repo_path(FULL / "TEACHER_RUNTIME.json")
    roster_path = _safe_repo_path(FULL / "FULL_SOURCE_ROSTER_LOCK.json")
    source_barrier_path = _safe_repo_path(FULL / "SOURCE_CAPTURE_BARRIER.json")
    runtime = _read_json(runtime_path)
    source_runtime = _read_json(source_runtime_path)
    teacher_runtime = _read_json(teacher_runtime_path)
    roster = _read_json(roster_path)
    rows, roster_index = _validate_roster(roster)
    roster_sha = _sha(roster_path)
    source_runtime_sha = _sha(source_runtime_path)
    teacher_runtime_sha = _sha(teacher_runtime_path)
    source_barrier = _read_json(source_barrier_path)
    source_barrier_sha = _sha(source_barrier_path)

    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("FULL/STUDENT_RUNTIME.json must be locked and GT-free")
    for label, value in (("FULL/SOURCE_RUNTIME", source_runtime),
                         ("FULL/TEACHER_RUNTIME", teacher_runtime)):
        if value.get("status") != "locked" or value.get("GT_read") is not False:
            raise RuntimeError(f"{label} must be locked and GT-free")
    if runtime.get("full_source_roster_sha256") != roster_sha:
        raise RuntimeError("FULL/STUDENT_RUNTIME does not bind FULL_SOURCE_ROSTER_LOCK")
    if runtime.get("source_barrier_sha256") != source_barrier_sha:
        raise RuntimeError("FULL/STUDENT_RUNTIME does not bind the sealed full source barrier")
    if runtime.get("source_runtime_sha256") != source_runtime_sha:
        raise RuntimeError("FULL/STUDENT_RUNTIME source runtime binding differs")
    if runtime.get("teacher_runtime_sha256") != teacher_runtime_sha:
        raise RuntimeError("FULL/STUDENT_RUNTIME teacher runtime binding differs")
    if source_runtime.get("full_source_roster_sha256") != roster_sha:
        raise RuntimeError("FULL/SOURCE_RUNTIME does not bind FULL_SOURCE_ROSTER_LOCK")
    if teacher_runtime.get("full_source_roster_sha256", roster_sha) != roster_sha:
        raise RuntimeError("FULL/TEACHER_RUNTIME roster binding differs")
    teacher_bindings = teacher_runtime.get("bindings")
    if not isinstance(teacher_bindings, Mapping):
        raise RuntimeError("FULL/TEACHER_RUNTIME must bind its fixed source/tracking authorities")
    expected_teacher_bindings = {
        "full_source_roster_sha256": roster_sha,
        "input_manifest_sha256": source_barrier.get("input_manifest_sha256"),
        "source_runtime_sha256": source_runtime_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
    }
    if any(teacher_bindings.get(key) != value for key, value in expected_teacher_bindings.items()):
        raise RuntimeError("FULL/TEACHER_RUNTIME source authority bindings differ")
    tracking_runtime_sha = teacher_bindings.get("tracking_runtime_sha256")
    if not isinstance(tracking_runtime_sha, str) or not SHA_RE.fullmatch(tracking_runtime_sha):
        raise RuntimeError("FULL/TEACHER_RUNTIME lacks the fixed tracking runtime SHA")
    if (source_barrier.get("status") != "sealed" or source_barrier.get("GT_read") is not False
            or source_barrier.get("count") != EXPECTED_COUNT
            or source_barrier.get("full_source_roster_sha256") != roster_sha
            or source_barrier.get("source_runtime_sha256") != source_runtime_sha):
        raise RuntimeError("FULL source barrier is incomplete or not bound to its roster/runtime")
    source_files = source_barrier.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(roster_index):
        raise RuntimeError("FULL source barrier must bind every fixed alias exactly once")

    pins = runtime.get("pins")
    protected = runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise RuntimeError("FULL/STUDENT_RUNTIME requires pins and protected maps")
    for required in (STUDENT_REL, SCIENCE_REL):
        expected = pins.get(required, protected.get(required))
        if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
            raise RuntimeError(f"FULL/STUDENT_RUNTIME must bind {required}")
        if _sha(ROOT / required) != expected:
            raise RuntimeError(f"FULL/STUDENT_RUNTIME pin mismatch: {required}")
    for relative, digest in {**pins, **protected}.items():
        if not isinstance(relative, str) or not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            raise ValueError("FULL/STUDENT_RUNTIME pins must be root-relative SHA-256 entries")
        path = _safe_repo_path(relative)
        if _sha(path) != digest:
            raise RuntimeError(f"FULL/STUDENT_RUNTIME pin mismatch: {relative}")

    return {
        "runtime": runtime, "runtime_path": runtime_path, "runtime_sha256": _sha(runtime_path),
        "source_runtime_sha256": source_runtime_sha,
        "teacher_runtime_sha256": teacher_runtime_sha,
        "tracking_runtime_sha256": tracking_runtime_sha,
        "roster": roster, "rows": rows, "roster_index": roster_index,
        "roster_sha256": roster_sha,
        "source_barrier": source_barrier,
        "source_barrier_sha256": source_barrier_sha,
        "source_files": source_files,
    }


def _load_teacher_barrier(path_value: str | Path, ctx: Mapping[str, Any],
                          expected_aliases: Sequence[str], *, allow_superset: bool = False) -> dict[str, Any]:
    path = _safe_repo_path(path_value)
    if not path.is_relative_to(FULL.resolve()):
        raise ValueError("teacher batch barrier must live inside the FULL namespace")
    barrier = _read_json(path)
    expected_schema = ("stvg-final-table1-full-teacher-global-barrier-v1"
                       if "global" in path.name.lower()
                       else "stvg-final-table1-full-teacher-batch-barrier-v1")
    if (barrier.get("schema") != expected_schema or barrier.get("status") not in {"sealed", "complete"}
            or barrier.get("GT_read") is not False):
        raise RuntimeError("selected teacher barrier is incomplete or not GT-free")
    if barrier.get("teacher_runtime_sha256", barrier.get("runtime_sha256")) != ctx["teacher_runtime_sha256"]:
        raise RuntimeError("teacher batch barrier does not bind the locked FULL/TEACHER_RUNTIME")
    source_barrier_values = [barrier[key] for key in
                             ("source_capture_barrier_sha256", "source_barrier_sha256")
                             if key in barrier]
    if (barrier.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or not source_barrier_values
            or len(set(source_barrier_values)) != 1
            or source_barrier_values[0] != ctx["source_barrier_sha256"]
            or barrier.get("input_manifest_sha256") != ctx["source_barrier"].get("input_manifest_sha256")
            or barrier.get("tracking_runtime_sha256") != ctx["tracking_runtime_sha256"]):
        raise RuntimeError("teacher batch barrier source/roster binding differs")
    files = _normalize_teacher_files(barrier)
    if not set(files).issubset(ctx["roster_index"]):
        raise RuntimeError("teacher barrier contains aliases outside FULL_SOURCE_ROSTER_LOCK")
    barrier_aliases = barrier.get("aliases")
    file_aliases = [row["alias"] for row in barrier.get("files", [])] if isinstance(
        barrier.get("files"), list) else list(files)
    if (not isinstance(barrier_aliases, list) or barrier_aliases != file_aliases
            or len(barrier_aliases) != len(set(barrier_aliases))):
        raise RuntimeError("teacher barrier aliases must exactly match file rows in stable order")
    expected = set(expected_aliases)
    if barrier.get("count") != len(files):
        raise RuntimeError("teacher barrier count differs from its file rows")
    if allow_superset:
        if not expected.issubset(files):
            raise RuntimeError("teacher barrier does not cover all requested student aliases")
    elif set(files) != expected:
        raise RuntimeError("teacher barrier aliases must equal this exact student batch")
    return {"path": path, "sha256": _sha(path), "barrier": barrier, "files": files}


def _artifact_path(value: Any, expected_dir: Path, alias: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("locked artifact path must be a string")
    path = _safe_repo_path(value)
    if path.parent != expected_dir.resolve() or path.name != f"{alias}.pt":
        raise ValueError(f"artifact is outside its locked case directory: {alias}")
    return path


def _verify_source_input(alias: str, row: Mapping[str, Any],
                         item: Mapping[str, Any], ctx: Mapping[str, Any]):
    path = _artifact_path(item.get("path"), FULL / "source_cases", alias)
    if _sha(path) != item.get("sha256") or path.stat().st_size != item.get("bytes"):
        raise RuntimeError(f"sealed full source payload hash/size mismatch: {alias}")
    receipt_rel = item.get("receipt_path")
    receipt_path = _safe_repo_path(receipt_rel)
    if receipt_path.parent != path.parent or receipt_path.name != f"{alias}.receipt.json":
        raise ValueError(f"sealed full source receipt path differs: {alias}")
    if _sha(receipt_path) != item.get("receipt_sha256"):
        raise RuntimeError(f"sealed full source receipt hash mismatch: {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema") != "stvg-final-table1-full-source-receipt-v1"
            or receipt.get("case_id") != alias
            or receipt.get("sha256") != item.get("sha256")
            or receipt.get("bytes") != path.stat().st_size
            or receipt.get("source_case_sha256") != item.get("source_case_sha256")
            or receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or receipt.get("query_key_sha256") != row.get("query_key_sha256")
            or receipt.get("GT_read") is not False
            or receipt.get("DINO_read_or_call") is not False):
        raise RuntimeError(f"sealed full source receipt binding mismatch: {alias}")
    return path, receipt_path, receipt


def _verify_teacher_input(alias: str, source_item: Mapping[str, Any],
                          item: Mapping[str, Any], ctx: Mapping[str, Any]):
    path = _artifact_path(item.get("path"), FULL / "teacher_cases", alias)
    if _sha(path) != item.get("sha256"):
        raise RuntimeError(f"sealed full teacher payload hash mismatch: {alias}")
    if item.get("bytes") is not None and path.stat().st_size != item.get("bytes"):
        raise RuntimeError(f"sealed full teacher payload size mismatch: {alias}")
    receipt_value = item.get("receipt_path")
    receipt_path = _safe_repo_path(receipt_value) if receipt_value else path.with_suffix(".receipt.json")
    if receipt_path.parent != path.parent or receipt_path.name != f"{alias}.receipt.json":
        raise ValueError(f"sealed full teacher receipt path differs: {alias}")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"sealed full teacher receipt is missing: {alias}")
    receipt_sha = _sha(receipt_path)
    if item.get("receipt_sha256") and receipt_sha != item.get("receipt_sha256"):
        raise RuntimeError(f"sealed full teacher receipt hash mismatch: {alias}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema") != "stvg-final-table1-full-teacher-case-receipt-v1"
            or receipt.get("status") != "sealed"
            or receipt.get("case_id") != alias
            or receipt.get("path") != str(path.relative_to(ROOT))
            or receipt.get("sha256") != item.get("sha256")
            or receipt.get("bytes") != path.stat().st_size
            or receipt.get("source_sha256") != source_item.get("sha256")
            or receipt.get("source_case_sha256") != source_item.get("source_case_sha256")
            or receipt.get("source_receipt_sha256") != source_item.get("receipt_sha256")
            or (receipt.get("runtime_sha256") or receipt.get("teacher_runtime_sha256")) != ctx["teacher_runtime_sha256"]
            or receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or receipt.get("source_capture_barrier_sha256") != ctx["source_barrier_sha256"]
            or receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or receipt.get("tracking_runtime_sha256") != ctx["tracking_runtime_sha256"]
            or receipt.get("GT_read") is not False
            or not isinstance(receipt.get("request_sha256"), str)
            or not SHA_RE.fullmatch(receipt["request_sha256"])
            or not isinstance(receipt.get("tracking_binding_sha256"), str)
            or not SHA_RE.fullmatch(receipt["tracking_binding_sha256"])):
        raise RuntimeError(f"sealed full teacher receipt binding mismatch: {alias}")
    return path, receipt_path, receipt, receipt_sha


def _student_paths(alias: str) -> tuple[Path, Path]:
    directory = FULL / "student_cases"
    return directory / f"{alias}.pt", directory / f"{alias}.receipt.json"


def _validate_saved_student(alias: str, source_item: Mapping[str, Any],
                            teacher_item: Mapping[str, Any], teacher_receipt: Mapping[str, Any],
                            teacher_receipt_sha: str, teacher_batch: Mapping[str, Any],
                            ctx: Mapping[str, Any], torch: Any,
                            *, load_payload: bool = True) -> dict[str, Any]:
    path, receipt_path = _student_paths(alias)
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"sealed full student output is missing or partial: {alias}")
    receipt_sha = _sha(receipt_path)
    receipt = _read_json(receipt_path)
    if (_sha(path) != receipt.get("sha256") or path.stat().st_size != receipt.get("bytes")
            or receipt.get("schema") != "stvg-final-table1-full-student-receipt-v1"
            or receipt.get("status") != "sealed" or receipt.get("case_id") != alias
            or receipt.get("GT_read") is not False
            or receipt.get("source_file_sha256") != source_item.get("sha256")
            or receipt.get("source_case_sha256") != source_item.get("source_case_sha256")
            or receipt.get("teacher_file_sha256") != teacher_item.get("sha256")
            or receipt.get("teacher_receipt_sha256") != teacher_receipt_sha
            or receipt.get("student_runtime_sha256") != ctx["runtime_sha256"]
            or receipt.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or receipt.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or receipt.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or receipt.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or receipt.get("teacher_barrier_sha256") != teacher_batch["sha256"]
            or receipt.get("teacher_barrier_path") != str(teacher_batch["path"].relative_to(ROOT))
            or receipt.get("request_sha256") != teacher_receipt.get("request_sha256")
            or receipt.get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")):
        raise RuntimeError(f"full student receipt binding differs from sealed inputs: {alias}")
    if load_payload:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        _validate_student_payload(saved, alias, source_item, teacher_item, teacher_receipt,
                                  teacher_receipt_sha, teacher_batch, ctx)
        del saved
    return {
        "alias": alias,
        "path": str(path.relative_to(ROOT)),
        "sha256": receipt["sha256"],
        "bytes": receipt["bytes"],
        "receipt": {"path": str(receipt_path.relative_to(ROOT)),
                    "sha256": receipt_sha, "bytes": receipt_path.stat().st_size},
        "reused": True,
        "execution": {
            "actual_fit_count": int(receipt.get("actual_fit_count", 0)),
            "optimizer_steps": int(receipt.get("optimizer_steps", 0)),
            "final_full_model_redecode_calls": int(receipt.get("final_full_model_redecode_calls", 0)),
        },
    }


def _validate_student_payload(saved: Mapping[str, Any], alias: str,
                              source_item: Mapping[str, Any], teacher_item: Mapping[str, Any],
                              teacher_receipt: Mapping[str, Any], teacher_receipt_sha: str,
                              teacher_batch: Mapping[str, Any],
                              ctx: Mapping[str, Any]) -> None:
    if (saved.get("schema") != "stvg-final-table1-student-case-v1"
            or saved.get("case_id") != alias
            or saved.get("source_file_sha256") != source_item.get("sha256")
            or saved.get("source_case_sha256") != source_item.get("source_case_sha256")
            or saved.get("teacher_file_sha256") != teacher_item.get("sha256")
            or saved.get("reference_receipt", {}).get("teacher_receipt_sha256") != teacher_receipt_sha
            or saved.get("student_runtime_sha256") != ctx["runtime_sha256"]
            or saved.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or saved.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]
            or saved.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or saved.get("teacher_barrier_sha256") != teacher_batch["sha256"]
            or saved.get("teacher_batch_barrier_path") != str(teacher_batch["path"].relative_to(ROOT))
            or saved.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or saved.get("request_sha256") != teacher_receipt.get("request_sha256")
            or saved.get("tracking_binding_sha256") != teacher_receipt.get("tracking_binding_sha256")
            or saved.get("GT_read") is not False):
        raise RuntimeError(f"full student case does not match its locked inputs: {alias}")
    arms = saved.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"F", "T", "S", "TS", "TeacherGeometry"}:
        raise RuntimeError(f"full student case has wrong Table 1 arms: {alias}")
    if saved.get("source_cold_readback", {}).get("full_model_state_restored") is not True:
        raise RuntimeError(f"full student case lacks complete source reset/readback: {alias}")
    no_update = bool(saved.get("reference_receipt", {}).get("no_update"))
    fits, readouts = saved.get("fits"), saved.get("readouts")
    if not isinstance(fits, Mapping) or not isinstance(readouts, Mapping):
        raise RuntimeError(f"full student fit/readout schema malformed: {alias}")
    if no_update:
        if fits or readouts or saved.get("execution", {}).get("actual_fit_count") != 0:
            raise RuntimeError(f"no-update full student row contains a fit: {alias}")
    else:
        fit = fits.get("SpatialA")
        if not isinstance(fit, Mapping) or not isinstance(readouts.get("SpatialA"), Mapping):
            raise RuntimeError(f"updating full student row lacks the Spatial-A fit/readout: {alias}")
        if (fit.get("seed_key") != _science_module().stable_fit_key(
                str(saved.get("dataset")), str(saved.get("query_key_sha256")))
                or fit.get("active_parameters") != 1792 or fit.get("declared_steps") != 5
                or fit.get("reference_sha256") != saved.get("teacher_file_sha256")):
            raise RuntimeError(f"full student fit key or locked parameter count differs: {alias}")


def _science_module():
    from scripts import run_stvg_table1_student003 as science
    return science


def _load_case_payloads(alias: str, roster_record: Mapping[str, Any],
                        source_item: Mapping[str, Any], teacher_item: Mapping[str, Any],
                        ctx: Mapping[str, Any], teacher_batch: Mapping[str, Any], torch: Any,
                        source_module: Any, source_receipt: Mapping[str, Any],
                        teacher_receipt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = _artifact_path(source_item["path"], FULL / "source_cases", alias)
    teacher_path = _artifact_path(teacher_item["path"], FULL / "teacher_cases", alias)
    source_case = _science_module()._load_source_payload(
        source_path, alias, str(source_item["source_case_sha256"]), roster_record,
        torch, source_module,
    )
    if source_case.get("parent_sha256") != roster_record.get("parent_sha256"):
        raise RuntimeError(f"full source case parent hash differs from roster: {alias}")
    teacher_case = _science_module()._load_teacher_payload(
        teacher_path, alias, str(source_item["sha256"]),
        str(source_item["source_case_sha256"]), source_receipt, teacher_receipt, torch,
    )
    if teacher_case.get("dataset") != roster_record.get("dataset"):
        raise RuntimeError(f"full teacher dataset differs from roster: {alias}")
    # Ensure the Teacher barrier being consumed contains this exact payload
    # and receipt; the caller's selected batch SHA is stored with the student.
    if teacher_batch["files"][alias].get("sha256") != teacher_item.get("sha256"):
        raise RuntimeError(f"teacher batch row differs from loaded teacher input: {alias}")
    return source_case, teacher_case


def _runtime_entry(runtime: Mapping[str, Any], key: str, fallback: str | None = None) -> Any:
    return runtime.get(key, runtime.get(fallback) if fallback else None)


def _teacher_batch_id(aliases: Sequence[str], aliases_sha256: str | None,
                      teacher_barrier_sha256: str) -> str:
    body = json.dumps({"aliases": list(aliases), "aliases_file_sha256": aliases_sha256,
                       "teacher_barrier_sha256": teacher_barrier_sha256},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def _verify_source_barrier_files(ctx: Mapping[str, Any], aliases: Sequence[str]) -> None:
    for alias in aliases:
        item = ctx["source_files"][alias]
        row = ctx["roster_index"][alias]
        _verify_source_input(alias, row, item, ctx)


def _write_student_case(result: dict[str, Any], *, alias: str, source_item: Mapping[str, Any],
                        source_receipt: Mapping[str, Any], teacher_item: Mapping[str, Any],
                        teacher_receipt: Mapping[str, Any], teacher_receipt_sha: str,
                        teacher_batch: Mapping[str, Any], ctx: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    out_dir = FULL / "student_cases"
    path, receipt_path = _student_paths(alias)
    if path.exists() or receipt_path.exists():
        raise RuntimeError(f"preserve partial/existing full student output; do not overwrite: {alias}")
    result["teacher_batch_barrier_path"] = str(teacher_batch["path"].relative_to(ROOT))
    result["teacher_batch_barrier_sha256"] = teacher_batch["sha256"]
    result["full_source_roster_sha256"] = ctx["roster_sha256"]
    result["source_receipt_sha256"] = source_item.get("receipt_sha256")
    result["teacher_receipt_sha256"] = teacher_receipt_sha
    if result.get("GT_read") is not False:
        raise RuntimeError("refusing to write a student payload without an explicit GT-free flag")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(result, temp)
    os.replace(temp, path)
    receipt = {
        "schema": "stvg-final-table1-full-student-receipt-v1",
        "status": "sealed", "case_id": alias,
        "sha256": _sha(path), "bytes": path.stat().st_size,
        "source_case_sha256": source_item["source_case_sha256"],
        "source_file_sha256": source_item["sha256"],
        "source_receipt_sha256": source_item["receipt_sha256"],
        "teacher_file_sha256": teacher_item["sha256"],
        "teacher_receipt_sha256": teacher_receipt_sha,
        "teacher_barrier_path": str(teacher_batch["path"].relative_to(ROOT)),
        "teacher_barrier_sha256": teacher_batch["sha256"],
        "request_sha256": teacher_receipt["request_sha256"],
        "tracking_binding_sha256": teacher_receipt["tracking_binding_sha256"],
        "source_runtime_sha256": ctx["source_runtime_sha256"],
        "teacher_runtime_sha256": ctx["teacher_runtime_sha256"],
        "student_runtime_sha256": ctx["runtime_sha256"],
        "source_barrier_sha256": ctx["source_barrier_sha256"],
        "full_source_roster_sha256": ctx["roster_sha256"],
        "actual_fit_count": result["execution"]["actual_fit_count"],
        "optimizer_steps": result["execution"]["optimizer_steps"],
        "final_full_model_redecode_calls": result["execution"]["final_full_model_redecode_calls"],
        "no_update": result["reference_receipt"]["no_update"],
        "GT_read": False, "time": time.time(),
    }
    _write_json(receipt_path, receipt)
    return {
        "alias": alias, "path": str(path.relative_to(ROOT)),
        "sha256": receipt["sha256"], "bytes": receipt["bytes"],
        "receipt": {"path": str(receipt_path.relative_to(ROOT)),
                    "sha256": _sha(receipt_path), "bytes": receipt_path.stat().st_size},
        "reused": False, "execution": result["execution"],
    }


def _load_full_runtime_context(runtime_path: str | Path | None) -> dict[str, Any]:
    ctx = _verify_runtime_and_source(runtime_path=runtime_path)
    # The scientific module is imported inertly; redirect only its process-local
    # BASE so any reused helper that consults it remains inside FULL/.
    science = _science_module()
    science.BASE = FULL
    return ctx


def run_batch(*, aliases_file: str | Path | None, all_cases: bool,
              teacher_barrier_path: str | Path | None,
              runtime_path: str | Path | None = None) -> dict[str, Any]:
    import collections
    import gc
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("FULL student requires a fresh process before model initialization")
    ctx = _load_full_runtime_context(runtime_path)
    alias_file_path = _safe_repo_path(aliases_file) if aliases_file is not None else None
    alias_file_sha = _sha(alias_file_path) if alias_file_path is not None else None
    requested = _parse_aliases_text(alias_file_path.read_text(encoding="utf-8")) if alias_file_path else None
    aliases = _select_aliases(ctx["rows"], ctx["roster_index"],
                              all_cases=all_cases, aliases=requested)
    if not aliases:
        raise ValueError("FULL student batch selected no aliases")
    if teacher_barrier_path is None:
        if not all_cases:
            raise ValueError("bounded batch mode requires --teacher-barrier for this exact batch")
        teacher_barrier_path = FULL / "TEACHER_GLOBAL_BARRIER.json"
    teacher_batch = _load_teacher_barrier(teacher_barrier_path, ctx, aliases,
                                          allow_superset=all_cases)
    if all_cases and len(aliases) != EXPECTED_COUNT:
        raise RuntimeError("--all requires the exact 13,785 aliases")
    # A later --all invocation can continue after complete group batches.  It
    # processes only aliases that do not already belong to a sealed student
    # batch, preserving the one-fit-per-query rule.
    prior_batch_aliases: set[str] = set()
    prior_batch_ids: dict[str, str] = {}
    for prior_path in FULL.glob("STUDENT_BATCH_BARRIER_*.json"):
        prior = _read_json(prior_path)
        if prior.get("status") != "sealed":
            continue
        for prior_alias in prior.get("aliases", []):
            if prior_alias in prior_batch_aliases:
                raise RuntimeError(f"duplicate alias across prior student batch barriers: {prior_alias}")
            prior_batch_aliases.add(prior_alias)
            prior_batch_ids[prior_alias] = prior_path.name
    if all_cases:
        aliases = [alias for alias in aliases if alias not in prior_batch_aliases]
        if aliases:
            # Already completed rows must be whole video groups, so the
            # remaining aliases are also a union of complete video groups.
            _select_aliases(ctx["rows"], ctx["roster_index"], all_cases=False, aliases=aliases)
        if not aliases:
            return {"status": "no_unsealed_aliases", "count": 0,
                    "already_batched_count": len(prior_batch_aliases),
                    "teacher_barrier_sha256": teacher_batch["sha256"], "GT_read": False}
    else:
        overlap = set(aliases) & prior_batch_aliases
        if overlap:
            raise RuntimeError(f"aliases already belong to a sealed batch: {sorted(overlap)[:3]}")
    batch_id = _teacher_batch_id(aliases, alias_file_sha, teacher_batch["sha256"])
    batch_barrier_path = FULL / f"STUDENT_BATCH_BARRIER_{batch_id}.json"
    global_barrier_path = FULL / "STUDENT_GLOBAL_BARRIER.json"
    if global_barrier_path.exists():
        raise RuntimeError("a final full student barrier already exists; refuse later batches")

    # A completed exact batch is a readback-only operation.  Do not fit again.
    if batch_barrier_path.exists():
        batch_barrier = _read_json(batch_barrier_path)
        batch_sha = _sha(batch_barrier_path)
        if (batch_barrier.get("status") != "sealed"
                or batch_barrier.get("aliases") != aliases
                or batch_barrier.get("teacher_barrier_sha256") != teacher_batch["sha256"]
                or batch_barrier.get("student_runtime_sha256") != ctx["runtime_sha256"]):
            raise RuntimeError("existing student batch barrier differs; preserve and investigate")
        for alias in aliases:
            item = ctx["source_files"][alias]
            t_item = teacher_batch["files"][alias]
            _, _, t_receipt, t_receipt_sha = _verify_teacher_input(alias, item, t_item, ctx)
            _validate_saved_student(alias, item, t_item, t_receipt, t_receipt_sha,
                                    teacher_batch, ctx, torch, load_payload=True)
        return {"status": "already_sealed", "count": len(aliases),
                "batch_id": batch_id, "batch_barrier_sha256": batch_sha,
                "GT_read": False}

    _verify_source_barrier_files(ctx, aliases)
    FULL.joinpath("student_cases").mkdir(parents=True, exist_ok=True)
    # Case outputs sealed by a prior interrupted batch can be reused only if all
    # hashes and runtime/reference bindings are exactly the same.
    existing: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for alias in aliases:
        source_item = ctx["source_files"][alias]
        teacher_item = teacher_batch["files"][alias]
        _, _, teacher_receipt, teacher_receipt_sha = _verify_teacher_input(alias, source_item,
                                                                            teacher_item, ctx)
        path, receipt_path = _student_paths(alias)
        if path.exists() or receipt_path.exists():
            existing[alias] = _validate_saved_student(
                alias, source_item, teacher_item, teacher_receipt, teacher_receipt_sha,
                teacher_batch, ctx, torch, load_payload=True,
            )
        else:
            pending.append(alias)

    file_rows: dict[str, dict[str, Any]] = dict(existing)
    reused_fit_count = sum(int(row.get("execution", {}).get("actual_fit_count", 0)) for row in existing.values())
    reused_optimizer_steps = sum(int(row.get("execution", {}).get("optimizer_steps", 0)) for row in existing.values())
    reused_readouts = sum(int(row.get("execution", {}).get("final_full_model_redecode_calls", 0)) for row in existing.values())
    fit_count = optimizer_steps = readouts = 0
    capture_count = 0
    reused_count = len(existing)
    model = None
    current_source = None
    started = time.perf_counter()
    status_path = FULL / "STUDENT_BATCH_STATUS.json"
    _write_json(status_path, {
        "status": "running", "scope": "bounded_video_group_batch",
        "batch_id": batch_id, "done": 0, "total": len(aliases),
        "pending": len(pending), "student_runtime_sha256": ctx["runtime_sha256"],
        "teacher_barrier_sha256": teacher_batch["sha256"], "GT_read": False,
        "time": time.time(),
    }, replace=status_path.exists())

    if pending:
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_decota_paper_main_v1 import model_for, frames_for
        from scripts.stvg_mllm_feedback_common_v1 import raw_row
        from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward, transient_native_masks
        from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
        from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
        from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
        from methods.decota_final_simplified_v1.objectives import prediction
        from methods.decota_final_simplified_v1.tensors import state_hash
        from methods.decota_final_simplified_v1.backbone import query_subject
        from scripts import stvg_target_teacher_student_v1 as core
        from scripts import run_stvg_table1_source_v1 as source_module
        from scripts import run_stvg_adjacent_motion_student_v1 as source_capture
        from scripts import stvg_table1_spatial_adapter_v1 as spatial
        from scripts.stvg_instance_validated_fit_v1 import fit_validated
        from scripts.stvg_temporal_caption_query_v1 import inference_guard

        install_clean_loader()
        sys.addaudithook(inference_guard)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(20261015)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(20261015)
        np.random.seed(20261015)
        science = _science_module()
        for index, alias in enumerate(pending, 1):
            roster_record = ctx["roster_index"][alias]
            source_item = ctx["source_files"][alias]
            teacher_item = teacher_batch["files"][alias]
            source_path, source_receipt_path, source_receipt = _verify_source_input(
                alias, roster_record, source_item, ctx)
            teacher_path, teacher_receipt_path, teacher_receipt, teacher_receipt_sha = _verify_teacher_input(
                alias, source_item, teacher_item, ctx)
            teacher_receipt = dict(teacher_receipt)
            teacher_receipt["_receipt_path"] = str(teacher_receipt_path)
            source_case, teacher_case = _load_case_payloads(
                alias, roster_record, source_item, teacher_item, ctx, teacher_batch,
                torch, source_module, source_receipt, teacher_receipt,
            )
            result, model, current_source = science._make_result(
                alias=alias, roster_record=roster_record, source_case=source_case,
                source_file_sha256=source_item["sha256"], teacher_case=teacher_case,
                teacher_file_sha256=teacher_item["sha256"], teacher_receipt=teacher_receipt,
                student_runtime_sha256=ctx["runtime_sha256"],
                source_runtime_sha256=ctx["source_runtime_sha256"],
                teacher_runtime_sha256=ctx["teacher_runtime_sha256"],
                source_barrier_sha256=ctx["source_barrier_sha256"],
                teacher_barrier_sha256=teacher_batch["sha256"],
                model=model, current_source=current_source, torch=torch, np=np,
                collections_module=collections, model_for=model_for, frames_for=frames_for,
                raw_row=raw_row, frozen_forward=frozen_forward, prediction=prediction,
                state_hash=state_hash, core=core, spatial=spatial,
                fit_validated=fit_validated,
                transient_native_masks=transient_native_masks,
                frozen_visual_host=frozen_visual_host,
                frozen_attention_storage=frozen_attention_storage,
                inactive_modules_on_host=inactive_modules_on_host,
                query_subject=query_subject,
            )
            result["request_sha256"] = teacher_receipt["request_sha256"]
            result["tracking_binding_sha256"] = teacher_receipt["tracking_binding_sha256"]
            row = _write_student_case(
                result, alias=alias, source_item=source_item, source_receipt=source_receipt,
                teacher_item=teacher_item, teacher_receipt=teacher_receipt,
                teacher_receipt_sha=teacher_receipt_sha, teacher_batch=teacher_batch,
                ctx=ctx, torch=torch,
            )
            file_rows[alias] = row
            capture_count += int(result["execution"]["source_capture_calls"])
            fit_count += int(result["execution"]["actual_fit_count"])
            optimizer_steps += int(result["execution"]["optimizer_steps"])
            readouts += int(result["execution"]["final_full_model_redecode_calls"])
            _write_json(status_path, {
                "status": "running", "scope": "bounded_video_group_batch",
                "batch_id": batch_id, "done": index, "total": len(pending),
                "completed_alias": alias, "student_runtime_sha256": ctx["runtime_sha256"],
                "teacher_barrier_sha256": teacher_batch["sha256"], "GT_read": False,
                "time": time.time(),
            }, replace=True)
            print(f"TABLE1_FULL_STUDENT {index}/{len(pending)} {alias} "
                  f"fit={result['execution']['actual_fit_count']} "
                  f"steps={result['execution']['optimizer_steps']}", flush=True)
            del source_case, teacher_case, result
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        if model is not None and state_hash(model.state_dict()) != model._fixed_full_state_hash:
            raise RuntimeError("hot frozen source model state changed during full batch")
        if model is not None:
            del model
            model = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    # Re-read every selected payload/receipt after the last fit before sealing
    # the bounded batch, including rows reused after an earlier interruption.
    checked_rows = []
    for alias in aliases:
        item = ctx["source_files"][alias]
        t_item = teacher_batch["files"][alias]
        _verify_source_input(alias, ctx["roster_index"][alias], item, ctx)
        _, _, t_receipt, t_receipt_sha = _verify_teacher_input(alias, item, t_item, ctx)
        checked = _validate_saved_student(alias, item, t_item, t_receipt, t_receipt_sha,
                                          teacher_batch, ctx, torch, load_payload=True)
        checked["reused"] = alias in existing
        checked_rows.append(checked)
    barrier = {
        "schema": "stvg-final-table1-full-student-batch-barrier-v1",
        "status": "sealed", "scope": "exact_video_sha256_clock_bounds_batch",
        "batch_id": batch_id, "count": len(checked_rows),
        "aliases": list(aliases), "aliases_file_sha256": alias_file_sha,
        "video_group_count": len({_group_key(ctx["roster_index"][a]) for a in aliases}),
        "teacher_barrier_path": str(teacher_batch["path"].relative_to(ROOT)),
        "teacher_barrier_sha256": teacher_batch["sha256"],
        "source_barrier_sha256": ctx["source_barrier_sha256"],
        "full_source_roster_sha256": ctx["roster_sha256"],
        "source_runtime_sha256": ctx["source_runtime_sha256"],
        "teacher_runtime_sha256": ctx["teacher_runtime_sha256"],
        "student_runtime_sha256": ctx["runtime_sha256"],
        "files": [{key: row[key] for key in ("alias", "path", "sha256", "bytes", "receipt", "reused")}
                  for row in checked_rows],
        "actual_source_capture_calls": capture_count,
        "reused_student_cases": reused_count,
        "actual_fit_count_this_invocation": fit_count,
        "optimizer_steps_this_invocation": optimizer_steps,
        "final_full_model_redecode_calls_this_invocation": readouts,
        "reused_case_fit_count": reused_fit_count,
        "reused_case_optimizer_steps": reused_optimizer_steps,
        "reused_case_final_full_model_redecode_calls": reused_readouts,
        "sealed_case_fit_count_total": fit_count + reused_fit_count,
        "sealed_case_optimizer_steps_total": optimizer_steps + reused_optimizer_steps,
        "sealed_case_final_full_model_redecode_calls_total": readouts + reused_readouts,
        "teacher_calls": 0, "DINO_calls": 0, "GT_read": False,
        "wall_seconds": time.perf_counter() - started, "time": time.time(),
    }
    _write_json(batch_barrier_path, barrier)
    _write_json(status_path, {
        "status": "sealed", "scope": "bounded_video_group_batch",
        "batch_id": batch_id, "count": len(checked_rows),
        "barrier_path": str(batch_barrier_path.relative_to(ROOT)),
        "barrier_sha256": _sha(batch_barrier_path),
        "student_runtime_sha256": ctx["runtime_sha256"], "GT_read": False,
        "time": time.time(),
    }, replace=True)
    return {"status": "sealed", "count": len(checked_rows), "batch_id": batch_id,
            "barrier_path": str(batch_barrier_path.relative_to(ROOT)),
            "barrier_sha256": _sha(batch_barrier_path), "GT_read": False}


def _read_student_batch_barrier(path: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    barrier = _read_json(path)
    if (barrier.get("schema") != "stvg-final-table1-full-student-batch-barrier-v1"
            or barrier.get("status") != "sealed" or barrier.get("GT_read") is not False
            or barrier.get("student_runtime_sha256") != ctx["runtime_sha256"]
            or barrier.get("source_barrier_sha256") != ctx["source_barrier_sha256"]
            or barrier.get("full_source_roster_sha256") != ctx["roster_sha256"]
            or barrier.get("source_runtime_sha256") != ctx["source_runtime_sha256"]
            or barrier.get("teacher_runtime_sha256") != ctx["teacher_runtime_sha256"]):
        raise RuntimeError(f"student batch barrier bindings differ: {path.name}")
    aliases = barrier.get("aliases")
    rows = barrier.get("files")
    if (not isinstance(aliases, list) or not aliases or len(aliases) != len(set(aliases))
            or not isinstance(rows, list) or len(rows) != len(aliases)
            or barrier.get("count") != len(aliases)):
        raise RuntimeError(f"student batch barrier count/alias schema differs: {path.name}")
    if not set(aliases).issubset(ctx["roster_index"]):
        raise RuntimeError(f"student batch contains aliases outside full roster: {path.name}")
    # Batch aliases must be a union of complete same-video groups.
    selected = set(aliases)
    group_members: dict[tuple[Any, ...], set[str]] = {}
    for alias, row in ctx["roster_index"].items():
        group_members.setdefault(_group_key(row), set()).add(alias)
    for group in {_group_key(ctx["roster_index"][alias]) for alias in aliases}:
        if not group_members[group].issubset(selected):
            raise RuntimeError(f"student batch splits a locked video group: {path.name}")
    row_index = {str(row.get("alias")): row for row in rows if isinstance(row, Mapping)}
    if set(row_index) != selected:
        raise RuntimeError(f"student batch file rows differ from aliases: {path.name}")
    teacher_path = _safe_repo_path(barrier.get("teacher_barrier_path"))
    if not teacher_path.is_relative_to(FULL.resolve()) or _sha(teacher_path) != barrier.get("teacher_barrier_sha256"):
        raise RuntimeError(f"student batch's teacher barrier binding differs: {path.name}")
    teacher = _load_teacher_barrier(teacher_path, ctx, aliases, allow_superset=True)
    if teacher["sha256"] != barrier.get("teacher_barrier_sha256"):
        raise RuntimeError(f"student batch teacher barrier changed: {path.name}")
    return {"barrier": barrier, "rows": row_index, "teacher_batch": teacher}


def finalize(*, runtime_path: str | Path | None = None) -> dict[str, Any]:
    """CPU-only final seal after every group batch has completed."""
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("--finalize must run before CUDA/model initialization")
    ctx = _load_full_runtime_context(runtime_path)
    rows, roster_index = ctx["rows"], ctx["roster_index"]
    global_path = FULL / "STUDENT_GLOBAL_BARRIER.json"
    if global_path.exists():
        raise FileExistsError(f"preserve existing final student barrier: {global_path}")
    batch_paths = sorted(FULL.glob("STUDENT_BATCH_BARRIER_*.json"))
    if not batch_paths:
        raise RuntimeError("cannot finalize before any sealed student batch exists")
    memberships: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for path in batch_paths:
        readback = _read_student_batch_barrier(path, ctx)
        barrier = readback["barrier"]
        for alias in barrier["aliases"]:
            if alias in memberships:
                raise RuntimeError(f"alias appears in multiple student batch barriers: {alias}")
            memberships[alias] = (path, readback, readback["rows"][alias])
    if set(memberships) != set(roster_index):
        missing = len(set(roster_index) - set(memberships))
        extra = len(set(memberships) - set(roster_index))
        raise RuntimeError(f"student batches do not cover the exact full roster (missing={missing}, extra={extra})")

    final_files = []
    actual_fits = optimizer_steps = readouts = source_captures = 0
    teacher_barriers: dict[str, dict[str, Any]] = {}
    domain_counts = {"hc2": 0, "vidstg": 0}
    science = _science_module()
    for alias in [str(row["probe_alias"]) for row in rows]:
        batch_path, batch_readback, batch_row = memberships[alias]
        teacher_batch = batch_readback["teacher_batch"]
        source_item = ctx["source_files"][alias]
        teacher_item = teacher_batch["files"][alias]
        _verify_source_input(alias, roster_index[alias], source_item, ctx)
        _, _, teacher_receipt, teacher_receipt_sha = _verify_teacher_input(
            alias, source_item, teacher_item, ctx)
        result_row = _validate_saved_student(alias, source_item, teacher_item,
                                             teacher_receipt, teacher_receipt_sha,
                                             teacher_batch, ctx, torch, load_payload=True)
        for key in ("path", "sha256", "bytes", "receipt"):
            if batch_row.get(key) != result_row.get(key):
                raise RuntimeError(f"student batch row differs from final case receipt: {alias}/{key}")
        receipt = _read_json(FULL / "student_cases" / f"{alias}.receipt.json")
        if receipt.get("teacher_barrier_path") != str(teacher_batch["path"].relative_to(ROOT)):
            raise RuntimeError(f"student case receipt references another teacher barrier: {alias}")
        final_files.append({
            **{key: result_row[key] for key in ("alias", "path", "sha256", "bytes", "receipt")},
            "student_batch_barrier_path": str(batch_path.relative_to(ROOT)),
            "student_batch_barrier_sha256": _sha(batch_path),
        })
        execution = receipt
        actual_fits += int(execution.get("actual_fit_count", 0))
        optimizer_steps += int(execution.get("optimizer_steps", 0))
        readouts += int(execution.get("final_full_model_redecode_calls", 0))
        source_captures += 1
        domain_counts[roster_index[alias]["dataset"]] += 1
        teacher_barriers[teacher_batch["sha256"]] = {
            "path": str(teacher_batch["path"].relative_to(ROOT)),
            "sha256": teacher_batch["sha256"],
            "count": teacher_batch["barrier"]["count"],
        }
        del teacher_receipt, result_row
    if domain_counts != EXPECTED_DOMAINS or len(final_files) != EXPECTED_COUNT:
        raise RuntimeError("final student counts differ from fixed full-roster domain totals")
    global_barrier = {
        "schema": "stvg-final-table1-full-student-global-barrier-v1",
        "status": "sealed", "scope": "all 13,785 fixed official source queries",
        "count": EXPECTED_COUNT, "domain_counts": domain_counts,
        "files": final_files,
        "student_batch_barriers": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha(path),
             "count": _read_json(path)["count"]}
            for path in batch_paths
        ],
        "teacher_batch_barriers": sorted(teacher_barriers.values(), key=lambda x: x["path"]),
        "source_barrier_sha256": ctx["source_barrier_sha256"],
        "full_source_roster_sha256": ctx["roster_sha256"],
        "source_runtime_sha256": ctx["source_runtime_sha256"],
        "teacher_runtime_sha256": ctx["teacher_runtime_sha256"],
        "student_runtime_sha256": ctx["runtime_sha256"],
        "student_case_receipts_readback": source_captures,
        "actual_fit_count": actual_fits,
        "optimizer_steps": optimizer_steps,
        "final_full_model_redecode_calls": readouts,
        "teacher_calls": 0, "DINO_calls": 0, "GT_read": False,
        "time": time.time(),
    }
    _write_json(global_path, global_barrier)
    return {"status": "sealed", "count": EXPECTED_COUNT,
            "domain_counts": domain_counts, "barrier_path": str(global_path.relative_to(ROOT)),
            "barrier_sha256": _sha(global_path), "GT_read": False}


def contracts() -> dict[str, Any]:
    # Contracts use only synthetic aliases/records; no roster, model, CUDA, or GT.
    from scripts import run_stvg_table1_student003 as science

    digest = "a" * 64
    key_a = science.stable_fit_key("hc2", digest)
    key_b = science.stable_fit_key("hc2", digest)
    assert key_a == key_b and key_a == f"caption_query_fixed_space|hc2_{digest}"

    def row(alias: str, ds: str, video: str, parent: str, ordinal: int) -> dict[str, Any]:
        return {
            "probe_alias": alias, "dataset": ds, "query_ordinal": ordinal,
            "query_key_sha256": hashlib.sha256(f"query:{alias}".encode()).hexdigest(),
            "safe_record_sha256": hashlib.sha256(f"safe:{alias}".encode()).hexdigest(),
            "parent_sha256": hashlib.sha256(parent.encode()).hexdigest(),
            "plan_sha256": digest, "subject_sha256": digest,
            "video_sha256": hashlib.sha256(video.encode()).hexdigest(),
            "fps": 25.0, "width": 640, "height": 360,
            "frame_count": 20, "duration": .8,
            "video_bounds": [0, 20], "frame_ids": [0, 5, 10, 15],
            "GT_read": False,
        }
    synthetic = [
        row("HC00000", "hc2", "video-a", "parent-shared", 0),
        row("HC00001", "hc2", "video-a", "parent-shared", 1),
        row("HC00002", "hc2", "video-b", "parent-shared", 2),
        row("V00000", "vidstg", "video-v", "parent-v", 0),
    ]
    index = {r["probe_alias"]: r for r in synthetic}
    index["HC00001"]["frame_ids"] = [1, 6, 11, 16]
    complete_group = _select_aliases(synthetic, index, all_cases=False,
                                     aliases=["HC00000", "HC00001"])
    assert complete_group == ["HC00000", "HC00001"]
    sparse_grid_does_not_split_video = _group_key(index["HC00000"]) == _group_key(index["HC00001"])
    assert sparse_grid_does_not_split_video
    parent_does_not_merge_distinct_video = _group_key(index["HC00000"]) != _group_key(index["HC00002"])
    assert parent_does_not_merge_distinct_video
    split_rejected = False
    try:
        _select_aliases(synthetic, index, all_cases=False, aliases=["HC00000"])
    except ValueError:
        split_rejected = True
    assert split_rejected
    assert _parse_aliases_text('["HC00000", "HC00001"]') == ["HC00000", "HC00001"]
    assert _parse_aliases_text("HC00000\nHC00001\n") == ["HC00000", "HC00001"]
    barrier_rows = _normalize_teacher_files({
        "files": [{"alias": "HC00000", "path": "p0", "sha256": "0" * 64,
                   "bytes": 1, "receipt_path": "r0", "receipt_sha256": "2" * 64},
                  {"alias": "HC00001", "path": "p1", "sha256": "1" * 64,
                   "bytes": 2, "receipt_path": "r1", "receipt_sha256": "3" * 64}]
    })
    assert set(barrier_rows) == {"HC00000", "HC00001"}
    map_rows = _normalize_teacher_files({
        "files": {"HC00000": {"path": "p0", "sha256": "0" * 64,
                               "bytes": 1, "receipt_path": "r0", "receipt_sha256": "2" * 64}}
    })
    assert list(map_rows) == ["HC00000"]
    invalid_alias_rejected = False
    try:
        _parse_aliases_text('["HC00000", "bad"]')
    except ValueError:
        invalid_alias_rejected = True
    assert invalid_alias_rejected
    incomplete_barrier_rejected = False
    try:
        _normalize_teacher_files({"files": [{"alias": "HC00000", "path": "p0", "sha256": "0" * 64}]})
    except ValueError:
        incomplete_barrier_rejected = True
    assert incomplete_barrier_rejected
    return {
        "schema": "stvg-final-table1-full-student-cpu-contracts-v1",
        "status": "pass", "synthetic_contracts": {
            "stable_query_hash_fit_key": True,
            "exact_video_clock_grouping": True,
            "query_sparse_frame_ids_do_not_split_video": sparse_grid_does_not_split_video,
            "shared_parent_does_not_merge_distinct_video": parent_does_not_merge_distinct_video,
            "split_video_group_rejected": split_rejected,
            "json_and_lines_alias_file": True,
            "teacher_barrier_list_and_mapping_abi": True,
            "malformed_alias_and_teacher_rows_rejected": invalid_alias_rejected and incomplete_barrier_rejected,
        },
        "synthetic_aliases": len(synthetic), "GT_read": False,
        "CUDA_used": False, "model_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aliases-file", help="JSON aliases list/object or one alias per line")
    mode.add_argument("--all", action="store_true", help="run the full 13,785 alias set")
    mode.add_argument("--finalize", action="store_true", help="CPU-only exact 13,785 receipt readback and global seal")
    parser.add_argument("--teacher-barrier", help="root-relative FULL teacher batch/global barrier JSON")
    parser.add_argument("--runtime", help="optional root-relative locked FULL/STUDENT_RUNTIME.json")
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        if args.aliases_file or args.all or args.finalize or args.teacher_barrier or args.runtime:
            parser.error("--contracts cannot be combined with execution arguments")
        print(json.dumps(contracts(), sort_keys=True, ensure_ascii=False))
        return
    if args.finalize:
        if args.aliases_file or args.all or args.teacher_barrier:
            parser.error("--finalize cannot be combined with batch arguments")
        result = finalize(runtime_path=args.runtime)
    else:
        if not args.all and not args.aliases_file:
            parser.error("choose --aliases-file, --all, or --finalize")
        result = run_batch(aliases_file=args.aliases_file, all_cases=args.all,
                           teacher_barrier_path=args.teacher_barrier,
                           runtime_path=args.runtime)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
