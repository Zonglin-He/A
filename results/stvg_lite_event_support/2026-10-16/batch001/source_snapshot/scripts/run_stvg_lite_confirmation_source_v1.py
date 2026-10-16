"""Source-only capture launcher for the fixed lite confirmation roster.

This file is an execution interface, not an authorization to execute. It
requires a root-reviewed ROSTER_LOCK.json and a root-authored confirmation
SOURCE_RUNTIME.json. Captured source payloads are written only below
artifacts/stvg_lite_event_support_v1/confirmation/source_cases/.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/stvg_lite_event_support_v1"
CONFIRMATION = STUDY / "confirmation"
ROSTER = CONFIRMATION / "ROSTER_LOCK.json"
RUNTIME = CONFIRMATION / "SOURCE_RUNTIME.json"
BASE_SOURCE_RUNTIME = ROOT / "artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json"
CURRENT_DEV_ROSTER = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json"
OUT = CONFIRMATION / "source_cases"
BARRIER = CONFIRMATION / "SOURCE_CAPTURE_BARRIER.json"
STATUS = CONFIRMATION / "SOURCE_CAPTURE_STATUS.json"
ADAPTER_PATH = ROOT / "scripts/run_stvg_table1_source_v1.py"

EXPECTED_ALIASES = {
    *(f"HC{i:02d}" for i in range(1, 33)),
    *(f"V{i:02d}" for i in range(1, 33)),
}
SOURCE_MODEL_NAME = {"hc2": "vidstg", "vidstg": "hcstvg2"}
SOURCE_CHECKPOINT_SHA256 = {
    "hc2": "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83",
    "vidstg": "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.relative_to(ROOT)}")
    return value


def write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def verify() -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    if not ROSTER.is_file() or not RUNTIME.is_file():
        raise RuntimeError("Root must review and lock confirmation/ROSTER_LOCK.json and create confirmation/SOURCE_RUNTIME.json first")
    roster = read_json(ROSTER)
    runtime = read_json(RUNTIME)
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise RuntimeError("Confirmation roster must be root-reviewed, locked, and GT-free")
    records = roster.get("records")
    if not isinstance(records, list) or len(records) != 64 or roster.get("count") != 64:
        raise RuntimeError("The locked confirmation roster must contain exactly 64 rows")
    aliases = [row.get("probe_alias") for row in records]
    if set(aliases) != EXPECTED_ALIASES or len(aliases) != len(set(aliases)):
        raise RuntimeError("Confirmation aliases must be exactly HC01-HC32 and V01-V32")
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("Root-authored confirmation SOURCE_RUNTIME must be locked and GT-free")
    roster_sha = sha256_file(ROSTER)
    base_runtime_sha = sha256_file(BASE_SOURCE_RUNTIME)
    if runtime.get("confirmation_roster_sha256") != roster_sha:
        raise RuntimeError("Confirmation SOURCE_RUNTIME does not bind the reviewed roster")
    if runtime.get("preflight_source_runtime_sha256") != base_runtime_sha:
        raise RuntimeError("Confirmation SOURCE_RUNTIME does not bind the original locked Source runtime")
    expected_workers = {
        "scripts/run_stvg_lite_confirmation_source_v1.py": sha256_file(Path(__file__).resolve()),
        "scripts/run_stvg_table1_source_v1.py": sha256_file(ADAPTER_PATH),
    }
    pins = runtime.get("pins")
    if not isinstance(pins, dict):
        raise RuntimeError("Confirmation SOURCE_RUNTIME must contain file pins")
    for relative_path, digest in expected_workers.items():
        if pins.get(relative_path) != digest:
            raise RuntimeError(f"Confirmation SOURCE_RUNTIME does not pin {relative_path}")
    for relative_path, expected in {**pins, **runtime.get("protected", {})}.items():
        path = ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Confirmation SOURCE_RUNTIME pin mismatch: {relative_path}")
    if runtime.get("source_checkpoints") != read_json(BASE_SOURCE_RUNTIME).get("source_checkpoints"):
        raise RuntimeError("Confirmation SOURCE_RUNTIME must preserve the original source checkpoint bindings")
    for domain, expected in SOURCE_CHECKPOINT_SHA256.items():
        checkpoint = runtime["source_checkpoints"].get(domain)
        actual = checkpoint.get("sha256") if isinstance(checkpoint, dict) else checkpoint
        if actual != expected:
            raise RuntimeError(f"Source checkpoint hash changed for {domain}")

    excluded = {
        domain: {row["parent_sha256"] for row in read_json(CURRENT_DEV_ROSTER)["records"]
                if row.get("dataset") == domain}
        for domain in ("hc2", "vidstg")
    }
    domain_counts = {domain: sum(row.get("dataset") == domain for row in records)
                     for domain in ("hc2", "vidstg")}
    if domain_counts != {"hc2": 32, "vidstg": 32}:
        raise RuntimeError("Locked confirmation roster must contain 32 queries per target domain")
    for row in records:
        alias = row["probe_alias"]
        domain = "hc2" if alias.startswith("HC") else "vidstg"
        if row.get("dataset") != domain:
            raise RuntimeError("Confirmation alias does not match its target domain")
        if row.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256[domain]:
            raise RuntimeError("Confirmation row does not bind the expected source checkpoint")
    if any(row.get("parent_sha256") in excluded[row["dataset"]] for row in records):
        raise RuntimeError("Locked confirmation roster overlaps a current Coordinate128 source parent")
    if len({(row["dataset"], row["parent_sha256"]) for row in records}) != 64:
        raise RuntimeError("Confirmation roster repeats a source parent")
    if any(not row.get("source_case_path", "").startswith(
            "artifacts/stvg_lite_event_support_v1/confirmation/source_cases/") for row in records):
        raise RuntimeError("Confirmation source outputs must remain in the confirmation/source_cases directory")

    sys.path.insert(0, str(ROOT))
    from scripts import run_stvg_table1_source_v1 as adapter

    # The frozen adapter's own runtime remains the preflight runtime. This
    # preserves the original model/checkpoint/decoder binding; the separate
    # confirmation runtime and barrier bind the new roster and this run.
    if adapter.RUNTIME.resolve() != BASE_SOURCE_RUNTIME.resolve():
        raise RuntimeError("Frozen source adapter no longer points to the original locked Source runtime")
    adapter.verify_runtime()
    return records, runtime, adapter


def _save_case(case: Mapping[str, Any], adapter: Any) -> dict[str, Any]:
    import torch
    alias = str(case["case_id"])
    path = OUT / f"{alias}.pt"
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"Preserve incomplete or unreceipted source output: {alias}")
        receipt = read_json(receipt_path)
        if (sha256_file(path) != receipt.get("sha256")
                or path.stat().st_size != receipt.get("bytes")
                or receipt.get("case_id") != alias
                or receipt.get("query_key_sha256") != case.get("query_key_sha256")
                or receipt.get("source_runtime_sha256") != case.get("source_runtime_sha256")
                or receipt.get("source_case_sha256") != case.get("source_case_sha256")):
            raise RuntimeError(f"Existing confirmation source output binding mismatch: {alias}")
        return {
            "case_id": alias,
            "path": str(path.relative_to(ROOT)),
            "sha256": receipt["sha256"],
            "bytes": path.stat().st_size,
            "source_case_sha256": receipt["source_case_sha256"],
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "reused": True,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(dict(case), temp)
    os.replace(temp, path)
    receipt = {
        "schema": "stvg-lite-confirmation-source-receipt-v1",
        "case_id": alias,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_case_sha256": case["source_case_sha256"],
        "source_runtime_sha256": case["source_runtime_sha256"],
        "confirmation_source_runtime_sha256": sha256_file(RUNTIME),
        "query_key_sha256": case["query_key_sha256"],
        "dataset": case["dataset"],
        "query_ordinal": int(case["query_ordinal"]),
        "parent_sha256": case["parent_sha256"],
        "video_sha256": case["input"]["video_sha256"],
        "GT_read": False,
        "DINO_read_or_call": False,
        "time": time.time(),
    }
    write_json(receipt_path, receipt)
    return {
        "case_id": alias,
        "path": str(path.relative_to(ROOT)),
        "sha256": receipt["sha256"],
        "bytes": receipt["bytes"],
        "source_case_sha256": receipt["source_case_sha256"],
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "reused": False,
    }


def run(case_alias: str | None = None, all_confirmation: bool = False) -> None:
    records, runtime, adapter = verify()
    if all_confirmation == (case_alias is not None):
        raise ValueError("Choose exactly one of --all-confirmation or --case-alias")
    selected = records if all_confirmation else [row for row in records if row["probe_alias"] == case_alias]
    if len(selected) != (64 if all_confirmation else 1):
        raise ValueError("Requested alias is missing from the locked confirmation roster")
    if BARRIER.exists():
        raise FileExistsError("A source barrier already exists; preserve it and do not rerun over a completed capture")

    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_decota_paper_main_v1 import gpu, model_for

    lease = gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    np.random.seed(20261015)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.time()
    model = None
    loaded_source = None
    frozen_sha = None
    files: list[dict[str, Any]] = []
    captures = 0
    reused = 0
    model_loads = 0
    try:
        for index, rec in enumerate(selected, start=1):
            alias = rec["probe_alias"]
            source_name = SOURCE_MODEL_NAME[rec["dataset"]]
            frame_cache = None
            dst = OUT / f"{alias}.pt"
            receipt_dst = dst.with_suffix(".receipt.json")
            if dst.exists() or receipt_dst.exists():
                if not dst.is_file() or not receipt_dst.is_file():
                    raise RuntimeError(f"Preserve incomplete/unreceipted source output: {alias}")
                receipt = read_json(receipt_dst)
                if (receipt.get("case_id") != alias
                        or receipt.get("query_key_sha256") != rec["query_key_sha256"]
                        or receipt.get("dataset") != rec["dataset"]
                        or receipt.get("parent_sha256") != rec["parent_sha256"]
                        or receipt.get("video_sha256") != rec["video_sha256"]
                        or receipt.get("confirmation_source_runtime_sha256") != sha256_file(RUNTIME)
                        or dst.stat().st_size != receipt.get("bytes")
                        or sha256_file(dst) != receipt.get("sha256")):
                    raise RuntimeError(f"Existing source cache does not bind the locked roster: {alias}")
                item = {
                    "case_id": alias,
                    "path": str(dst.relative_to(ROOT)),
                    "sha256": receipt["sha256"],
                    "bytes": receipt["bytes"],
                    "source_case_sha256": receipt["source_case_sha256"],
                    "receipt_path": str(receipt_dst.relative_to(ROOT)),
                    "reused": True,
                }
                reused += 1
            else:
                if loaded_source != source_name:
                    if model is not None:
                        if state_hash(model.state_dict()) != frozen_sha:
                            raise RuntimeError("Frozen source model changed between domains")
                        del model
                        model = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    model = model_for(source_name)
                    frozen_sha = model._fixed_full_state_hash
                    loaded_source = source_name
                    model_loads += 1
                frame_cache = collections.OrderedDict()
                torch.cuda.reset_peak_memory_stats()
                case = adapter.capture_one(model, rec, frame_cache)
                if state_hash(model.state_dict()) != frozen_sha:
                    raise RuntimeError("Frozen source model state changed during capture")
                item = _save_case(case, adapter)
                captures += 1
                del case
            files.append(item)
            write_json(STATUS, {
                "schema": "stvg-lite-confirmation-source-progress-v1",
                "status": "running",
                "done": index,
                "total": len(selected),
                "current_alias": alias,
                "confirmation_roster_sha256": sha256_file(ROSTER),
                "confirmation_source_runtime_sha256": sha256_file(RUNTIME),
                "base_source_runtime_sha256": sha256_file(BASE_SOURCE_RUNTIME),
                "GT_read": False,
                "DINO_read_or_call": False,
                "optimizer_steps": 0,
                "backwards": 0,
                "time": time.time(),
            }, replace=True)
            print(f"LITE_CONFIRM_SOURCE {index}/{len(selected)} {alias}", flush=True)
            if frame_cache is not None:
                frame_cache.clear()
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        if all_confirmation and len(files) != 64:
            raise RuntimeError("Complete confirmation source capture did not produce all 64 cases")
        if all_confirmation:
            barrier = {
                "schema": "stvg-lite-confirmation-source-barrier-v1",
                "status": "complete",
                "count": len(files),
                "roster_sha256": sha256_file(ROSTER),
                "confirmation_source_runtime_sha256": sha256_file(RUNTIME),
                "base_source_runtime_sha256": sha256_file(BASE_SOURCE_RUNTIME),
                "files": files,
                "actual_source_model_loads": model_loads,
                "actual_source_capture_calls": captures,
                "reused_exact_confirmation_cases": reused,
                "DINO_calls": 0,
                "optimizer_steps": 0,
                "backwards": 0,
                "GT_read": False,
                "time": time.time(),
                "wall_seconds": time.time() - started,
            }
            write_json(BARRIER, barrier)
            write_json(STATUS, {
                "schema": "stvg-lite-confirmation-source-progress-v1",
                "status": barrier["status"],
                "done": len(files),
                "total": 64,
                "barrier_sha256": sha256_file(BARRIER),
                "GT_read": False,
                "time": time.time(),
            }, replace=True)
        else:
            write_json(STATUS, {
                "schema": "stvg-lite-confirmation-source-progress-v1",
                "status": "single_case_ready",
                "done": len(files),
                "total": 64,
                "case_alias": case_alias,
                "case_file": files[0],
                "GT_read": False,
                "time": time.time(),
            }, replace=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        lease.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-confirmation", action="store_true")
    group.add_argument("--case-alias", choices=sorted(EXPECTED_ALIASES))
    args = parser.parse_args()
    run(case_alias=args.case_alias, all_confirmation=args.all_confirmation)


if __name__ == "__main__":
    main()
