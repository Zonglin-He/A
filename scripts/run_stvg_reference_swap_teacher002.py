#!/usr/bin/env python3
"""Same-frame frozen GroundingDINO reference-swap teacher for Table 1.

`--prepare` is CPU-only: it binds the existing 32-source roster, source
captures, and the 49 already-locked WF Top-1 observations. `--run` is the only
model path. It uses the original local GroundingDINO processor, parser output,
FP32 inference, and admission gates, then seals paired references only after
all 49 attempts have a result. It does not read GT or create proposals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
DESIGN = BASE / "DESIGN_LOCK.json"
ROSTER_IN = ROOT / "artifacts/stvg_reference_recovery_v1/Optimizer_replay/ROSTER_LOCK.json"
SOURCE_BARRIER = ROOT / "artifacts/stvg_final_table1_v1/SOURCE_CAPTURE_BARRIER.json"
PARENT_REFERENCE = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009/REFERENCE_LOCK.json"
ROSTER = BASE / "ROSTER_LOCK.json"
JOB_LOCK = BASE / "TEACHER_JOB_LOCK.json"
RUNTIME = BASE / "RUNTIME.json"
TEACHER_DIR = BASE / "teacher"
GLOBAL_BARRIER = BASE / "TEACHER_GLOBAL_BARRIER.json"
REFERENCE_LOCK = BASE / "REFERENCE_LOCK.json"
STATUS = BASE / "STATUS.json"

SOURCE_MODULE = "scripts/run_stvg_table1_source_v1.py"
PAPER_MODULE = "scripts/run_decota_paper_main_v1.py"
OBSERVATIONS_MODULE = "methods/decota_final_simplified_v1/observations.py"
CONFIG_MODULE = "methods/decota_final_simplified_v1/config.py"
TENSOR_MODULE = "methods/decota_final_simplified_v1/tensors.py"
HC2_DECODER = "vg_tta/tastvg_paper48_hc2_decode_v1.py"
VIDSTG_DECODER = "vg_tta/exact_frame_decode_audit_v2.py"
DECOTA_COMMON = "scripts/decota_paper_common_v1.py"
LEASE_MODULE = "scripts/run_final_simplification_v1.py"
REQUIRED_PIN_FILES = (SOURCE_MODULE, PAPER_MODULE, DECOTA_COMMON, OBSERVATIONS_MODULE,
                      CONFIG_MODULE, TENSOR_MODULE, HC2_DECODER, VIDSTG_DECODER,
                      LEASE_MODULE,
                      "artifacts/stvg_reference_recovery_v1/Reference_swap/CPU_CONTRACTS.json")

SNAPSHOT_REL = ".cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71"
EXPERT_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
ASSET_FILES = {
    "model.safetensors": EXPERT_SHA256,
    "config.json": "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a",
    "preprocessor_config.json": "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",
}
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "vocab.txt",
                   "special_tokens_map.json", "added_tokens.json")
GATE = {"phrase_threshold": 0.35, "distinct_margin": 0.05, "nms_iou": 0.5}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_ALIASES = {*(f"HC{i:02d}" for i in range(1, 17)), *(f"V{i:02d}" for i in range(1, 17))}
DATASET_BY_ALIAS = {**{f"HC{i:02d}": "hc2" for i in range(1, 17)},
                    **{f"V{i:02d}": "vidstg" for i in range(1, 17)}}


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_json(value: Any) -> str:
    return _sha_bytes(_json_bytes(value))


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def _root_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("locked relative path escapes the workspace")
    return path


def _safe_torch_load(path: Path) -> Any:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _source_case_digest(case: Mapping[str, Any]) -> str:
    from scripts.run_stvg_table1_source_v1 import content_sha256
    return content_sha256(case)


def _read_parent_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    roster_in = _read_json(ROSTER_IN)
    source_barrier = _read_json(SOURCE_BARRIER)
    parent_ref = _read_json(PARENT_REFERENCE)
    if roster_in.get("status") != "locked" or roster_in.get("GT_read") is not False:
        raise ValueError("E2 source roster must be locked and GT-free")
    if roster_in.get("count") != 32 or len(roster_in.get("records", [])) != 32:
        raise ValueError("E1 requires the fixed 32-query E2 roster")
    if source_barrier.get("status") != "sealed" or source_barrier.get("GT_read") is not False:
        raise ValueError("source capture barrier must be sealed and GT-free")
    if source_barrier.get("DINO_calls") != 0:
        raise ValueError("source barrier must prove no previous DINO calls")
    if parent_ref.get("status") != "locked" or parent_ref.get("GT_read") is not False:
        raise ValueError("parent WF reference lock must be locked and GT-free")
    aliases = [str(row.get("probe_alias")) for row in roster_in["records"]]
    if len(set(aliases)) != 32 or set(aliases) != EXPECTED_ALIASES:
        raise ValueError("E1 roster must contain the fixed 16+16 source aliases")
    if len(parent_ref.get("records", [])) != 32:
        raise ValueError("parent WF reference lock must contain all 32 source records")
    return roster_in, source_barrier, parent_ref


def _parent_record_map(parent_ref: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = parent_ref.get("records")
    if not isinstance(rows, list):
        raise ValueError("parent reference records missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        alias = row.get("case_id")
        if alias in result or alias not in EXPECTED_ALIASES:
            raise ValueError("duplicate or unexpected parent reference case")
        result[str(alias)] = row
    if set(result) != EXPECTED_ALIASES:
        raise ValueError("parent reference cases do not match the E1 roster")
    return result


def _asset_bindings() -> dict[str, Any]:
    snapshot = _root_path(SNAPSHOT_REL)
    assets: dict[str, str] = {}
    for name, expected in ASSET_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"required local GroundingDINO asset missing: {name}")
        actual = _sha_file(path)
        if actual != expected:
            raise ValueError(f"local GroundingDINO asset hash mismatch: {name}")
        assets[name] = actual
    tokenizer: dict[str, str] = {}
    for name in TOKENIZER_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"required local tokenizer asset missing: {name}")
        tokenizer[name] = _sha_file(path)
    from methods.decota_final_simplified_v1.config import (
        EXPERT_SHA256 as CONFIG_EXPERT_SHA256,
        EXPERT_SNAPSHOT as CONFIG_EXPERT_SNAPSHOT,
    )
    if CONFIG_EXPERT_SHA256 != EXPERT_SHA256 or str(CONFIG_EXPERT_SNAPSHOT) != SNAPSHOT_REL:
        raise AssertionError("frozen GroundingDINO config constants changed")
    return {
        "expert_checkpoint_sha256": assets["model.safetensors"],
        "expert_config_sha256": assets["config.json"],
        "processor_config_sha256": assets["preprocessor_config.json"],
        "tokenizer_files": tokenizer,
        "asset_files": assets,
        "expert_snapshot_rel": SNAPSHOT_REL,
    }


def _validate_source_payload(rec: Mapping[str, Any], barrier_entry: Mapping[str, Any]) -> dict[str, Any]:
    source_path = _root_path(str(barrier_entry["path"]))
    receipt_path = _root_path(str(barrier_entry["receipt_path"]))
    if not source_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("sealed source payload/receipt is missing")
    if source_path.stat().st_size != int(barrier_entry["bytes"]):
        raise ValueError("source case byte count differs from barrier")
    if _sha_file(source_path) != barrier_entry.get("sha256"):
        raise ValueError("source case file hash differs from barrier")
    if _sha_file(receipt_path) != barrier_entry.get("receipt_sha256"):
        raise ValueError("source receipt hash differs from barrier")
    case = _safe_torch_load(source_path)
    if not isinstance(case, dict):
        raise TypeError("source payload must be a first-party mapping")
    semantic_sha = _source_case_digest(case)
    if semantic_sha != barrier_entry.get("source_case_sha256"):
        raise ValueError("source case semantic hash differs from barrier")
    receipt = _read_json(receipt_path)
    alias = str(rec["probe_alias"])
    if (receipt.get("case_id") != alias or receipt.get("GT_read") is not False
            or receipt.get("DINO_read_or_call") is not False
            or receipt.get("source_case_sha256") != semantic_sha):
        raise ValueError("source receipt identity or no-DINO/GT provenance mismatch")
    if (case.get("case_id") != alias or case.get("dataset") != rec.get("dataset")
            or case.get("GT_read") is not False or case.get("DINO_read_or_call") is not False):
        raise ValueError("source payload identity or no-DINO/GT provenance mismatch")
    if case.get("source_case_sha256") != semantic_sha:
        raise ValueError("source payload self-hash mismatch")
    if case.get("query_key_sha256") != rec.get("query_key_sha256"):
        raise ValueError("source query-key binding mismatch")
    if case.get("source_model_state_sha256") is None or case.get("source_checkpoint_sha256") is None:
        raise ValueError("source model identity hashes are required")
    inp = case.get("input")
    if not isinstance(inp, Mapping):
        raise ValueError("source payload input metadata missing")
    for key in ("video_sha256", "fps", "width", "height", "frame_count", "duration"):
        value = inp.get(key)
        expected = rec.get(key)
        if key == "video_sha256":
            value = inp.get(key)
        if value != expected:
            raise ValueError(f"source payload and roster differ in {key}")
    if list(map(int, case.get("frame_ids", []))) != list(map(int, rec.get("frame_ids", []))):
        raise ValueError("source grid frame IDs differ from roster")
    if list(map(int, case.get("video_bounds", []))) != list(map(int, rec.get("video_bounds", []))):
        raise ValueError("source video bounds differ from roster")
    if case.get("subject_sha256") != rec.get("subject_sha256"):
        raise ValueError("source payload subject hash differs from roster")
    if case.get("plan_sha256") != rec.get("plan_sha256"):
        raise ValueError("source payload plan hash differs from roster")
    return case


def _make_prepared_locks() -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    design = _read_json(DESIGN)
    if design.get("status") != "locked" or design.get("GT_read") is not False:
        raise ValueError("DESIGN_LOCK must be locked and GT-free")
    if design.get("source_count") != 32 or design.get("teacher_observation_attempts") != 49:
        raise ValueError("DESIGN_LOCK must fix 32 sources and 49 observations")
    expected_gate = {k: design.get(k) for k in GATE}
    if expected_gate != GATE or design.get("teacher_original_text_and_gate") is not True:
        raise ValueError("DINO gate must match original frozen configuration")
    roster_in, source_barrier, parent_ref = _read_parent_inputs()
    source_rows = {str(k): v for k, v in source_barrier["source_files"].items()}
    parent_rows = _parent_record_map(parent_ref)
    source_barrier_sha = _sha_file(SOURCE_BARRIER)
    parent_ref_sha = _sha_file(PARENT_REFERENCE)
    source_roster_sha = _sha_file(ROSTER_IN)
    design_sha = _sha_file(DESIGN)
    asset = _asset_bindings()

    local_roster_records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    expected_parent_count = 0
    for roster_row in roster_in["records"]:
        alias = str(roster_row["probe_alias"])
        if roster_row.get("GT_read") is not False:
            raise ValueError("source roster record is not explicitly GT-free")
        if roster_row.get("dataset") != DATASET_BY_ALIAS[alias]:
            raise ValueError("source roster alias/dataset mismatch")
        entry = source_rows.get(alias)
        if not isinstance(entry, Mapping):
            raise ValueError("source capture barrier lacks a selected case")
        case = _validate_source_payload(roster_row, entry)
        if case.get("pixel_sha256") is None:
            raise ValueError("source pixel digest is required")
        parent = parent_rows[alias]
        if parent.get("dataset") != roster_row["dataset"]:
            raise ValueError("parent reference dataset differs from source roster")
        if parent.get("source_case_sha256") != case["source_case_sha256"]:
            raise ValueError("parent WF reference binds a different source case")
        crop = parent.get("new_crop")
        wf_anchors = crop.get("anchors") if isinstance(crop, Mapping) else None
        if not isinstance(wf_anchors, list):
            raise ValueError("parent WF Top-1 anchors missing")
        alias_jobs: list[dict[str, Any]] = []
        positions_seen: set[int] = set()
        frame_ids = np.asarray(case["frame_ids"], dtype=np.int64)
        source_rgb = case.get("source_RGB")
        if not isinstance(source_rgb, Mapping):
            raise ValueError("source frame RGB receipts missing")
        for anchor_index, anchor in enumerate(wf_anchors):
            if not isinstance(anchor, Mapping):
                raise TypeError("WF reference anchor must be a mapping")
            pos = anchor.get("position")
            fid = anchor.get("physical_frame_id")
            if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < len(frame_ids):
                raise ValueError("WF anchor position falls outside the original source grid")
            if pos in positions_seen:
                raise ValueError("duplicate Top-1 observation positions are not permitted")
            positions_seen.add(pos)
            if int(frame_ids[pos]) != int(fid):
                raise ValueError("WF anchor physical frame does not match source-grid position")
            rgb_row = source_rgb.get(int(fid), source_rgb.get(str(int(fid))))
            if not isinstance(rgb_row, Mapping):
                raise ValueError("source per-frame RGB receipt is missing for anchor")
            if rgb_row.get("dtype") != "uint8" or len(rgb_row.get("shape", [])) != 3:
                raise ValueError("source frame RGB receipt has invalid shape or dtype")
            box = np.asarray(anchor.get("box_cxcywh"), dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("WF anchor geometry must be a finite normalized cxcywh box")
            if not (0 <= box[0] <= 1 and 0 <= box[1] <= 1 and 0 < box[2] <= 1
                    and 0 < box[3] <= 1 and box[0] - box[2] / 2 >= -1e-7
                    and box[1] - box[3] / 2 >= -1e-7
                    and box[0] + box[2] / 2 <= 1 + 1e-7
                    and box[1] + box[3] / 2 <= 1 + 1e-7):
                raise ValueError("WF anchor normalized geometry is invalid")
            weight = float(anchor.get("weight"))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("WF anchor weight must be finite and nonnegative")
            anchor_plain = {
                "position": int(pos),
                "physical_frame_id": int(fid),
                "box_cxcywh": [float(x) for x in box],
                "weight": weight,
                "candidate_id": anchor.get("candidate_id"),
                "teacher_output_sha256": anchor.get("teacher_output_sha256"),
            }
            if not SHA_RE.fullmatch(str(anchor_plain["teacher_output_sha256"])):
                raise ValueError("original WF teacher output hash missing")
            job_id = f"{alias}_{int(pos):03d}_{int(fid)}"
            identity = {
                "job_id": job_id,
                "case_id": alias,
                "dataset": str(roster_row["dataset"]),
                "query_ordinal": int(roster_row["query_ordinal"]),
                "query_key_sha256": str(roster_row["query_key_sha256"]),
                "subject_sha256": str(roster_row["subject_sha256"]),
                "plan_sha256": str(roster_row["plan_sha256"]),
                "source_case_path": str(entry["path"]),
                "source_case_file_sha256": str(entry["sha256"]),
                "source_case_bytes": int(entry["bytes"]),
                "source_case_sha256": str(case["source_case_sha256"]),
                "source_receipt_path": str(entry["receipt_path"]),
                "source_receipt_sha256": str(entry["receipt_sha256"]),
                "source_pixel_sha256": str(case["pixel_sha256"]),
                "source_model_state_sha256": str(case["source_model_state_sha256"]),
                "source_checkpoint_sha256": str(case["source_checkpoint_sha256"]),
                "video_sha256": str(case["input"]["video_sha256"]),
                "frame_ids_sha256": _sha_json([int(x) for x in frame_ids]),
                "position": int(pos),
                "physical_frame_id": int(fid),
                "frame_rgb_sha256": str(rgb_row["RGB_sha256"]),
                "frame_shape": [int(x) for x in rgb_row["shape"]],
                "frame_dtype": str(rgb_row["dtype"]),
                "parent_wf_reference_lock_sha256": parent_ref_sha,
                "parent_wf_anchor_index": anchor_index,
                "parent_wf_anchor_sha256": _sha_json(anchor_plain),
                "parent_wf_anchor": anchor_plain,
                "source_capture_barrier_sha256": source_barrier_sha,
                "source_roster_sha256": source_roster_sha,
                "design_lock_sha256": design_sha,
                "gate": GATE,
                "GT_read": False,
            }
            identity["job_binding_sha256"] = _sha_json(identity)
            jobs.append(identity)
            alias_jobs.append(identity)
            expected_parent_count += 1
        local_roster_records.append({
            "case_id": alias,
            "dataset": str(roster_row["dataset"]),
            "query_ordinal": int(roster_row["query_ordinal"]),
            "query_key_sha256": str(roster_row["query_key_sha256"]),
            "source_case_path": str(entry["path"]),
            "source_case_file_sha256": str(entry["sha256"]),
            "source_case_bytes": int(entry["bytes"]),
            "source_case_sha256": str(case["source_case_sha256"]),
            "source_receipt_path": str(entry["receipt_path"]),
            "source_receipt_sha256": str(entry["receipt_sha256"]),
            "subject_sha256": str(roster_row["subject_sha256"]),
            "plan_sha256": str(roster_row["plan_sha256"]),
            "video_sha256": str(case["input"]["video_sha256"]),
            "fps": float(roster_row["fps"]),
            "width": int(roster_row["width"]),
            "height": int(roster_row["height"]),
            "frame_count": int(roster_row["frame_count"]),
            "duration": float(roster_row["duration"]),
            "video_bounds": [int(x) for x in case["video_bounds"]],
            "frame_ids": [int(x) for x in frame_ids],
            "frame_ids_sha256": _sha_json([int(x) for x in frame_ids]),
            "pixel_sha256": str(case["pixel_sha256"]),
            "source_model_state_sha256": str(case["source_model_state_sha256"]),
            "source_checkpoint_sha256": str(case["source_checkpoint_sha256"]),
            "attempted_anchor_count": len(alias_jobs),
            "attempt_job_ids": [row["job_id"] for row in alias_jobs],
            "GT_read": False,
        })
    if expected_parent_count != 49 or len(jobs) != 49:
        raise ValueError(f"expected exactly 49 frozen WF observations; got {len(jobs)}")
    jobs.sort(key=lambda row: (row["case_id"], row["position"], row["physical_frame_id"]))
    if len({row["job_id"] for row in jobs}) != 49:
        raise ValueError("teacher job IDs are not unique")
    bindings = {
        "design_lock_sha256": design_sha,
        "source_roster_sha256": source_roster_sha,
        "source_capture_barrier_sha256": source_barrier_sha,
        "parent_wf_reference_lock_sha256": parent_ref_sha,
        **asset,
    }
    roster_doc = {
        "schema": "stvg-reference-swap-roster-v1",
        "status": "locked",
        "count": 32,
        "attempt_count": 49,
        "GT_read": False,
        "bindings": dict(bindings),
        "records": local_roster_records,
    }
    first_preview_by_dataset: dict[str, str] = {}
    for row in jobs:
        first_preview_by_dataset.setdefault(row["dataset"], row["job_id"])
    job_doc = {
        "schema": "stvg-reference-swap-teacher-job-lock-v1",
        "status": "locked",
        "count": 49,
        "source_count": 32,
        "GT_read": False,
        "bindings": {
            "design_lock_sha256": design_sha,
            "roster_sha256": None,
            "source_roster_sha256": source_roster_sha,
            "source_capture_barrier_sha256": source_barrier_sha,
            "parent_wf_reference_lock_sha256": parent_ref_sha,
            **asset,
        },
        "preview_job_ids": [first_preview_by_dataset[k] for k in ("hc2", "vidstg")],
        "records": jobs,
    }
    return roster_doc, job_doc


def prepare() -> None:
    if ROSTER.exists() or JOB_LOCK.exists() or RUNTIME.exists():
        raise FileExistsError("preserve existing E1 preparation/runtime artifacts; no overwrite")
    allowed = {"DESIGN_LOCK.json", "CPU_CONTRACTS.json"}
    existing = {p.name for p in BASE.iterdir()} if BASE.exists() else set()
    if existing - allowed:
        raise RuntimeError(f"unexpected pre-existing E1 outputs: {sorted(existing - allowed)}")
    roster_doc, job_doc = _make_prepared_locks()
    _write_json(ROSTER, roster_doc)
    job_doc["bindings"]["roster_sha256"] = _sha_file(ROSTER)
    _write_json(JOB_LOCK, job_doc)
    _write_json(STATUS, {
        "schema": "stvg-reference-swap-teacher-status-v1",
        "status": "prepared_waiting_for_root_runtime",
        "source_count": 32,
        "attempt_count": 49,
        "GT_read": False,
        "roster_sha256": _sha_file(ROSTER),
        "teacher_job_lock_sha256": _sha_file(JOB_LOCK),
    })


def _runtime_bindings() -> dict[str, str]:
    if not RUNTIME.is_file():
        raise RuntimeError("root must create Reference_swap/RUNTIME.json before --run")
    runtime = _read_json(RUNTIME)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("RUNTIME.json must be locked and explicitly GT-free")
    if runtime.get("pins") is None or runtime.get("protected") is None:
        raise ValueError("RUNTIME.json must contain pins and protected hash maps")
    roster_doc = _read_json(ROSTER)
    job_doc = _read_json(JOB_LOCK)
    if roster_doc.get("status") != "locked" or roster_doc.get("GT_read") is not False:
        raise ValueError("local ROSTER_LOCK is not sealed")
    if job_doc.get("status") != "locked" or job_doc.get("GT_read") is not False:
        raise ValueError("TEACHER_JOB_LOCK is not sealed")
    asset = _asset_bindings()
    required = {
        "design_lock_sha256": _sha_file(DESIGN),
        "roster_sha256": _sha_file(ROSTER),
        "source_roster_sha256": _sha_file(ROSTER_IN),
        "teacher_job_lock_sha256": _sha_file(JOB_LOCK),
        "cpu_contracts_sha256": _sha_file(BASE / "CPU_CONTRACTS.json"),
        "source_capture_barrier_sha256": _sha_file(SOURCE_BARRIER),
        "parent_wf_reference_lock_sha256": _sha_file(PARENT_REFERENCE),
        "expert_checkpoint_sha256": asset["expert_checkpoint_sha256"],
        "expert_config_sha256": asset["expert_config_sha256"],
        "processor_config_sha256": asset["processor_config_sha256"],
    }
    bindings = runtime.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("RUNTIME.json bindings object missing")
    for key, expected in required.items():
        if bindings.get(key) != expected:
            raise ValueError(f"RUNTIME.json does not bind the locked {key}")
    missing_pins = sorted(set(REQUIRED_PIN_FILES) - set(runtime["pins"]) - set(runtime["protected"]))
    if missing_pins:
        raise ValueError(f"RUNTIME is missing required source-code pins: {missing_pins}")
    pins = {**runtime["pins"], **runtime["protected"]}
    for rel, expected in pins.items():
        path = _root_path(str(rel))
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"RUNTIME source/protected hash mismatch: {rel}")
    own_rel = str(Path(__file__).resolve().relative_to(ROOT))
    if runtime["pins"].get(own_rel) != _sha_file(__file__):
        raise ValueError("RUNTIME does not pin this worker version")
    return {
        **required,
        "runtime_sha256": _sha_file(RUNTIME),
        "worker_sha256": _sha_file(__file__),
    }


def _check_runtime_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    import numpy
    import torch
    import torchvision
    import transformers
    import PIL

    actual = {
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "transformers": str(transformers.__version__),
        "numpy": str(numpy.__version__),
        "pillow": str(PIL.__version__),
    }
    expected = runtime.get("environment")
    if not isinstance(expected, Mapping):
        raise ValueError("RUNTIME environment version map is required")
    for name, version in actual.items():
        if expected.get(name) != version:
            raise ValueError(f"runtime package version mismatch for {name}")
    return actual


def _install_gt_read_guard() -> None:
    """Block explicit annotation/label reads without disabling DINO assets."""
    import sys

    blocked_parts = {"annotations", "annotation", "groundtruth", "ground_truth", "gt",
                     "labels_scorer_only", "gt_exposure"}
    blocked_files = {"postseal_rows.json", "postseal_summary.json"}

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        target = args[0]
        if isinstance(target, (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(target)).replace("\\", "/").lower()
            parts = {part for part in path.split("/") if part}
            if parts & blocked_parts or Path(path).name in blocked_files:
                raise PermissionError("GT read prohibited by E1 teacher guard")

    sys.addaudithook(guard)


def _validate_clean_frames(case: Mapping[str, Any], frames: Any, ids: Any) -> tuple[Any, dict[int, str]]:
    import numpy as np

    rgb = np.asarray(frames)
    frame_ids = [int(x) for x in ids]
    expected_ids = [int(x) for x in case["frame_ids"]]
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("clean decoder must return uint8 N x H x W x RGB")
    if frame_ids != expected_ids or rgb.shape[0] != len(frame_ids):
        raise ValueError("clean decoder frame map differs from the locked source grid")
    if rgb.shape[1:3] != (int(case["height"]), int(case["width"])):
        raise ValueError("clean decoder frame shape differs from original source dimensions")
    contiguous = np.ascontiguousarray(rgb)
    whole_sha = _sha_bytes(contiguous.tobytes())
    if whole_sha != case["pixel_sha256"]:
        raise ValueError("full clean-RGB pixel hash differs from sealed source capture")
    expected = case.get("source_RGB")
    if not isinstance(expected, Mapping):
        raise ValueError("source per-frame clean RGB receipt missing")
    per_frame: dict[int, str] = {}
    for pos, fid in enumerate(frame_ids):
        row = expected.get(fid, expected.get(str(fid)))
        if not isinstance(row, Mapping):
            raise ValueError("source per-frame RGB receipt lacks a decoded frame")
        frame = np.ascontiguousarray(contiguous[pos])
        digest = _sha_bytes(frame.tobytes())
        if digest != row.get("RGB_sha256"):
            raise ValueError("decoded clean RGB frame SHA differs from original capture")
        if list(frame.shape) != list(row.get("shape", [])) or str(frame.dtype) != row.get("dtype"):
            raise ValueError("decoded clean RGB frame shape/dtype differs from source receipt")
        per_frame[fid] = digest
    return contiguous, per_frame


def _tensor_meta(tensor: Any) -> dict[str, Any]:
    import torch
    value = tensor.detach().contiguous().cpu()
    return {"sha256": _sha_bytes(value.numpy().tobytes()),
            "shape": [int(x) for x in value.shape], "dtype": str(value.dtype)}


def _plain(value: Any) -> Any:
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite teacher output")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported teacher output type {type(value).__name__}")


def _is_valid_box(box: Any) -> bool:
    import numpy as np
    x = np.asarray(box, dtype=np.float64)
    return bool(x.shape == (4,) and np.isfinite(x).all()
                and 0 <= x[0] <= 1 and 0 <= x[1] <= 1
                and 0 < x[2] <= 1 and 0 < x[3] <= 1
                and x[0] - x[2] / 2 >= -1e-7
                and x[1] - x[3] / 2 >= -1e-7
                and x[0] + x[2] / 2 <= 1 + 1e-7
                and x[1] + x[3] / 2 <= 1 + 1e-7)


def _box_pixels(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = map(float, box)
    x1 = max(0, min(width - 1, round((cx - bw / 2) * width)))
    y1 = max(0, min(height - 1, round((cy - bh / 2) * height)))
    x2 = max(0, min(width - 1, round((cx + bw / 2) * width)))
    y2 = max(0, min(height - 1, round((cy + bh / 2) * height)))
    return x1, y1, x2, y2


def _save_private_preview(directory: Path, job: Mapping[str, Any], rgb: Any,
                          result: Mapping[str, Any], worker_sha: str) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("private preview requires original uint8 RGB frame")
    height, width = array.shape[:2]
    image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wf_box = job["parent_wf_anchor"]["box_cxcywh"]
    wf_xyxy = _box_pixels(wf_box, width, height)
    draw.rectangle(wf_xyxy, outline=(0, 255, 0), width=3)
    draw.text((4, 4), "WF locked", fill=(0, 255, 0), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    dino_box = result.get("selected_box_cxcywh")
    if result.get("decision_status") == "accepted" and _is_valid_box(dino_box):
        dino_xyxy = _box_pixels(dino_box, width, height)
        draw.rectangle(dino_xyxy, outline=(255, 0, 0), width=3)
        draw.text((4, 18), "DINO accepted", fill=(255, 0, 0), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    else:
        dino_xyxy = None
        draw.text((4, 18), f"DINO {result['decision_status']}: {result['reason']}",
                  fill=(255, 255, 0), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    path = directory / f"{job['job_id']}.png"
    if path.exists():
        raise FileExistsError("pre-existing private preview is preserved")
    image.save(path, format="PNG", optimize=False)
    return {
        "path": str(path.relative_to(BASE)),
        "sha256": _sha_file(path),
        "bytes": path.stat().st_size,
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "physical_frame_id": int(job["physical_frame_id"]),
        "source_image_sha256": job["frame_rgb_sha256"],
        "source_image_shape": [int(height), int(width), 3],
        "wf_box_cxcywh": [float(x) for x in wf_box],
        "wf_drawn_xyxy_inclusive_pixels": list(wf_xyxy),
        "dino_decision_status": result["decision_status"],
        "dino_box_cxcywh": [float(x) for x in dino_box] if dino_box is not None else None,
        "dino_drawn_xyxy_inclusive_pixels": list(dino_xyxy) if dino_xyxy is not None else None,
        "worker_sha256": worker_sha,
        "GT_read": False,
    }


def _observe_locked_position(expert: Any, rgb: Any, position: int, frame_id: int,
                             parses: Mapping[str, Any]) -> dict[str, Any]:
    """Exact one-position equivalent of the original frozen observation path."""
    import torch
    from methods.decota_final_simplified_v1.observations import (
        ContextView, probe_from_detection,
    )

    context = parses["context"]
    fallback = parses["old"]
    context_eligible = bool(context.get("eligible"))
    tokenizer_count = None
    if context_eligible:
        tokenizer_count = len(expert.processor.tokenizer(context["context"])["input_ids"])
    use_context = context_eligible and tokenizer_count <= 256
    if not use_context and not fallback.get("phrase"):
        return {
            "decision_status": "unknown",
            "reason": "query_parser_abstention",
            "parser_path": "context" if context_eligible else "old_visual_query",
            "context_eligible": context_eligible,
            "context_token_count": tokenizer_count,
            "selected_box_cxcywh": None,
            "selected_candidate_id": None,
            "selected_model_candidate_index": None,
            "selected_score": None,
            "margin": None,
            "target_span": context.get("span") if use_context else None,
            "target_token_positions": None,
            "phrase_token_positions": None,
            "model_inputs": {},
            "raw_boxes_cxcywh": [],
            "raw_phrase_scores": [],
            "processed_probe": None,
            "forward_seconds": 0.0,
        }
    input_meta: dict[str, Any] = {}

    def capture_inputs(module: Any, args: Any, kwargs: Mapping[str, Any]) -> None:
        for key in ("pixel_values", "pixel_mask", "input_ids", "attention_mask"):
            value = kwargs.get(key)
            if torch.is_tensor(value):
                input_meta[key] = _tensor_meta(value)

    handle = expert.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        with torch.no_grad():
            if use_context:
                from methods.decota_final_simplified_v1.observations import ContextView
                detection = ContextView(expert, context)(rgb, context["context"], context["entity"])
                probe = probe_from_detection(detection, position, frame_id)
                accepted = bool(probe["accepted"])
                reason = str(probe["reason"])
                score_values = probe.get("target_scores", [])
                boxes = probe.get("boxes")
                candidate_ids = probe.get("candidate_ids", [])
                selected_box = None
                selected_candidate_id = "DINO_same_frame_expert"
                selected_model_candidate_index = None
                selected_score = None
                if accepted:
                    selected_ix = int(torch.as_tensor(score_values).argmax().item())
                    selected_box = _plain(torch.as_tensor(boxes)[selected_ix])
                    selected_model_candidate_index = int(candidate_ids[selected_ix])
                    selected_score = float(score_values[selected_ix])
                all_boxes = detection.get("all_boxes")
                all_scores = detection.get("all_phrase_scores")
                phrase_positions = detection.get("phrase_tokens")
                target_positions = detection.get("entity_tokens")
                processed_probe = {
                    "accepted": accepted,
                    "reason": reason,
                    "margin": float(probe["margin"]),
                    "boxes_cxcywh": _plain(boxes),
                    "target_scores": _plain(score_values),
                    "candidate_ids": _plain(candidate_ids),
                    "admission_winner": probe.get("admission_winner"),
                }
                margin = float(probe["margin"])
            else:
                from methods.decota_final_simplified_v1.observations import SpatialExpert
                detection = expert(rgb, fallback["phrase"], fallback["entity"])
                accepted = bool(detection["accepted"])
                reason = str(detection["reason"])
                selected_box = _plain(detection.get("box")) if accepted else None
                selected_candidate_id = "DINO_same_frame_expert" if accepted else None
                selected_model_candidate_index = None
                selected_score = float(detection.get("score", 0.0)) if accepted else None
                all_boxes = detection.get("all_boxes")
                all_scores = detection.get("all_phrase_scores")
                target_positions = detection.get("entity_tokens")
                phrase_positions = detection.get("phrase_tokens")
                processed_probe = {
                    "accepted": accepted,
                    "reason": reason,
                    "margin": float(detection.get("margin", 0.0)),
                    "top_boxes_cxcywh": _plain(detection.get("top_boxes", [])),
                    "top_scores": _plain(detection.get("top_scores", [])),
                    "nms_iou": GATE["nms_iou"],
                }
                margin = float(detection.get("margin", 0.0))
        if accepted and not _is_valid_box(selected_box):
            raise ValueError("original DINO gate accepted invalid normalized geometry")
        return {
            "decision_status": "accepted" if accepted else "rejected",
            "reason": reason,
            "parser_path": "context" if use_context else "old_visual_query",
            "context_eligible": context_eligible,
            "context_token_count": tokenizer_count,
            "selected_box_cxcywh": selected_box,
            "selected_candidate_id": selected_candidate_id,
            "selected_model_candidate_index": selected_model_candidate_index,
            "selected_score": selected_score,
            "margin": margin,
            "target_span": context.get("span") if use_context else None,
            "target_token_positions": _plain(target_positions),
            "phrase_token_positions": _plain(phrase_positions),
            "model_inputs": input_meta,
            "raw_boxes_cxcywh": _plain(all_boxes) if all_boxes is not None else [],
            "raw_phrase_scores": _plain(all_scores) if all_scores is not None else [],
            "processed_probe": processed_probe,
            "forward_seconds": float(detection.get("seconds", 0.0)),
        }
    finally:
        handle.remove()


def _output_path(job_id: str) -> Path:
    if not re.fullmatch(r"(?:HC|V)\d{2}_\d{3}_-?\d+", job_id):
        raise ValueError("invalid deterministic job ID")
    return TEACHER_DIR / f"{job_id}.json"


def _make_reference_lock(job_doc: Mapping[str, Any], outputs: Mapping[str, Mapping[str, Any]],
                         files: list[dict[str, Any]], barrier_sha: str,
                         runtime_bindings: Mapping[str, str]) -> dict[str, Any]:
    roster = _read_json(ROSTER)
    roster_rows = roster["records"]
    by_case = {row["case_id"]: row for row in roster_rows}
    records = []
    file_by_job = {row["job_id"]: row for row in files}
    jobs_by_case: dict[str, list[Mapping[str, Any]]] = {alias: [] for alias in EXPECTED_ALIASES}
    for job in job_doc["records"]:
        jobs_by_case[job["case_id"]].append(job)
    if sum(len(v) for v in jobs_by_case.values()) != 49:
        raise ValueError("reference assembly requires all locked 49 jobs")
    for case_id in sorted(EXPECTED_ALIASES):
        rr = by_case.get(case_id)
        if not isinstance(rr, Mapping):
            raise ValueError("local roster is missing a source case")
        dino_anchors, wf_anchors, decisions = [], [], []
        rejected, unknown = [], []
        for job in sorted(jobs_by_case[case_id], key=lambda row: row["position"]):
            output = outputs[job["job_id"]]
            status = output["decision_status"]
            item = file_by_job[job["job_id"]]
            decision = {
                "job_id": job["job_id"],
                "position": int(job["position"]),
                "physical_frame_id": int(job["physical_frame_id"]),
                "status": status,
                "reason": output["reason"],
                "teacher_output_path": item["path"],
                "teacher_output_sha256": item["sha256"],
            }
            decisions.append(decision)
            if status == "rejected":
                rejected.append(int(job["position"]))
                continue
            if status == "unknown":
                unknown.append(int(job["position"]))
                continue
            if status != "accepted":
                raise ValueError("unknown DINO decision status")
            box = output.get("selected_box_cxcywh")
            if not _is_valid_box(box):
                raise ValueError("accepted DINO teacher result lacks valid normalized geometry")
            wf = job["parent_wf_anchor"]
            if (int(wf["position"]) != int(job["position"])
                    or int(wf["physical_frame_id"]) != int(job["physical_frame_id"])):
                raise ValueError("WF/DINO same-frame anchor binding mismatch")
            dino_anchor = {
                "position": int(job["position"]),
                "physical_frame_id": int(job["physical_frame_id"]),
                "box_cxcywh": [float(x) for x in box],
                "weight": float(wf["weight"]),
                "candidate_id": "DINO_same_frame_expert",
                "teacher_job_id": job["job_id"],
                "teacher_output_sha256": item["sha256"],
                "original_wf_candidate_id": wf.get("candidate_id"),
                "original_wf_teacher_output_sha256": wf["teacher_output_sha256"],
            }
            wf_anchor = {
                "position": int(wf["position"]),
                "physical_frame_id": int(wf["physical_frame_id"]),
                "box_cxcywh": [float(x) for x in wf["box_cxcywh"]],
                "weight": float(wf["weight"]),
                "candidate_id": wf.get("candidate_id"),
                "teacher_job_id": job["job_id"],
                "teacher_output_sha256": item["sha256"],
                "original_wf_candidate_id": wf.get("candidate_id"),
                "original_wf_teacher_output_sha256": wf["teacher_output_sha256"],
            }
            if (dino_anchor["position"], dino_anchor["physical_frame_id"], dino_anchor["weight"]) != (
                    wf_anchor["position"], wf_anchor["physical_frame_id"], wf_anchor["weight"]):
                raise ValueError("DINO/WF common-support pairing differs in position/frame/weight")
            dino_anchors.append(dino_anchor)
            wf_anchors.append(wf_anchor)
        if len(decisions) != int(rr["attempted_anchor_count"]):
            raise ValueError("case-level reference omitted a locked DINO attempt")
        records.append({
            "case_id": case_id,
            "dataset": rr["dataset"],
            "source_case_sha256": rr["source_case_sha256"],
            "query_key_sha256": rr["query_key_sha256"],
            "attempt_count": len(decisions),
            "accepted_count": len(dino_anchors),
            "rejected_count": len(rejected),
            "unknown_count": len(unknown),
            "dino_decisions": decisions,
            "dino_rejected_positions": rejected,
            "dino_unknown_positions": unknown,
            "common_dino_anchors": dino_anchors,
            "common_wf_anchors": wf_anchors,
            "no_common_support": len(dino_anchors) == 0,
            "GT_read": False,
        })
    return {
        "schema": "stvg-reference-swap-reference-lock-v1",
        "status": "locked",
        "count": 32,
        "attempt_count": 49,
        "common_anchor_count": sum(len(row["common_dino_anchors"]) for row in records),
        "GT_read": False,
        "bindings": {
            "design_lock_sha256": _sha_file(DESIGN),
            "roster_sha256": _sha_file(ROSTER),
            "teacher_job_lock_sha256": _sha_file(JOB_LOCK),
            "source_capture_barrier_sha256": _sha_file(SOURCE_BARRIER),
            "parent_wf_reference_lock_sha256": _sha_file(PARENT_REFERENCE),
            "teacher_global_barrier_sha256": barrier_sha,
            **dict(runtime_bindings),
        },
        "records": records,
    }


def _run_teacher() -> None:
    if GLOBAL_BARRIER.exists() or REFERENCE_LOCK.exists() or STATUS.exists() and _read_json(STATUS).get("status") == "running":
        raise FileExistsError("existing E1 run/seal state is preserved; automatic resume is disabled")
    if not JOB_LOCK.is_file() or not ROSTER.is_file():
        raise RuntimeError("run CPU --prepare first, then let root review and lock RUNTIME.json")
    runtime_binding = _runtime_bindings()
    runtime_doc = _read_json(RUNTIME)
    job_doc = _read_json(JOB_LOCK)
    roster_doc = _read_json(ROSTER)
    if job_doc.get("count") != 49 or len(job_doc.get("records", [])) != 49:
        raise ValueError("teacher job lock does not contain exactly 49 attempts")
    if roster_doc.get("count") != 32 or len(roster_doc.get("records", [])) != 32:
        raise ValueError("teacher roster does not contain exactly 32 sources")
    if job_doc.get("bindings", {}).get("roster_sha256") != _sha_file(ROSTER):
        raise ValueError("teacher job lock does not bind local roster")
    if not (ROOT / SOURCE_BARRIER.relative_to(ROOT)).is_file():
        raise FileNotFoundError("sealed source capture barrier missing")

    _install_gt_read_guard()
    import numpy as np
    import torch
    from collections import OrderedDict
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.run_stvg_table1_source_v1 import _validate_roster_record
    from scripts.run_final_simplification_v1 import lease

    environment = _check_runtime_environment(runtime_doc)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    lock = lease()
    files: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    call_count = 0
    accepted_count = rejected_count = unknown_count = 0
    started = time.time()
    expert = None
    source_rows = {row["case_id"]: row for row in roster_doc["records"]}
    raw_roster_rows = {r["probe_alias"]: r for r in _read_json(ROSTER_IN)["records"]}
    jobs_by_case: dict[str, list[dict[str, Any]]] = {alias: [] for alias in EXPECTED_ALIASES}
    for row in job_doc["records"]:
        jobs_by_case[row["case_id"]].append(row)
    try:
        if TEACHER_DIR.exists() and any(TEACHER_DIR.iterdir()):
            raise FileExistsError("unsealed teacher output directory is nonempty; no automatic resume")
        TEACHER_DIR.mkdir(parents=True, exist_ok=True)
        preview_dir = BASE / "private_previews"
        if preview_dir.exists() and any(preview_dir.iterdir()):
            raise FileExistsError("pre-existing private previews are preserved")
        preview_dir.mkdir(parents=True, exist_ok=True)
        for stale in (GLOBAL_BARRIER, REFERENCE_LOCK):
            if stale.exists():
                raise FileExistsError(f"preserve existing output: {stale}")
        before_state = None
        preview_job_ids = set(job_doc.get("preview_job_ids", []))
        if (len(preview_job_ids) != 2
                or any(j not in {r["job_id"] for r in job_doc["records"]} for j in preview_job_ids)):
            raise ValueError("exactly two prelocked preview job IDs are required")
        previews: list[dict[str, Any]] = []
        # Group by fixed roster order so each source video is decoded once.
        for case_id in sorted(EXPECTED_ALIASES):
            if not jobs_by_case[case_id]:
                continue
            local_roster = source_rows[case_id]
            raw_roster = raw_roster_rows[case_id]
            ds, query_row = _validate_roster_record(raw_roster)
            if ds != local_roster["dataset"]:
                raise ValueError("source query validation returned a different dataset")
            subject = _read_json(_root_path(str(raw_roster["subject_path"])))
            if (subject.get("GT_read") is not False
                    or subject.get("ordinal") != int(local_roster["query_ordinal"])
                    or subject.get("caption_sha256") != _sha_bytes(query_row["input"]["caption"].encode())):
                raise ValueError("source query parser sidecar binding mismatch")
            parses = subject.get("parses")
            if not isinstance(parses, Mapping) or not isinstance(parses.get("context"), Mapping) or not isinstance(parses.get("old"), Mapping):
                raise ValueError("source query parser sidecar lacks original parse structures")
            frame_cache: OrderedDict = OrderedDict()
            frames, ids, _reuse = frames_for(ds, query_row, frame_cache)
            # Source payload itself is reloaded only after the sealed file/receipt hashes were checked in prepare.
            source_case = _safe_torch_load(_root_path(local_roster["source_case_path"]))
            if _source_case_digest(source_case) != local_roster["source_case_sha256"]:
                raise ValueError("source payload changed since CPU preparation")
            source_path = _root_path(local_roster["source_case_path"])
            source_receipt_path = _root_path(local_roster["source_receipt_path"])
            if _sha_file(source_path) != local_roster["source_case_file_sha256"]:
                raise ValueError("source payload file hash changed since CPU preparation")
            if _sha_file(source_receipt_path) != local_roster["source_receipt_sha256"]:
                raise ValueError("source receipt hash changed since CPU preparation")
            source_receipt = _read_json(source_receipt_path)
            if (source_receipt.get("case_id") != case_id or source_receipt.get("GT_read") is not False
                    or source_receipt.get("DINO_read_or_call") is not False
                    or source_receipt.get("source_case_sha256") != local_roster["source_case_sha256"]):
                raise ValueError("source receipt provenance changed since CPU preparation")
            clean, per_frame_hashes = _validate_clean_frames(source_case, frames, ids)
            if expert is None:
                expert = SpatialExpert(ROOT / EXPERT_SNAPSHOT)
                before_state = state_hash(expert.model.state_dict())
            for job in sorted(jobs_by_case[case_id], key=lambda row: row["position"]):
                if job["source_case_sha256"] != local_roster["source_case_sha256"]:
                    raise ValueError("job/source case semantic binding mismatch")
                pos = int(job["position"])
                fid = int(job["physical_frame_id"])
                if int(ids[pos]) != fid or per_frame_hashes.get(fid) != job["frame_rgb_sha256"]:
                    raise ValueError("locked source observation frame does not match clean RGB bytes")
                call_started = time.perf_counter()
                result = _observe_locked_position(expert, clean[pos], pos, fid, parses)
                if result["decision_status"] == "accepted":
                    accepted_count += 1
                elif result["decision_status"] == "rejected":
                    rejected_count += 1
                elif result["decision_status"] == "unknown":
                    unknown_count += 1
                else:
                    raise ValueError("invalid teacher decision status")
                call_count += 1
                preview_info = None
                if job["job_id"] in preview_job_ids:
                    preview_info = _save_private_preview(preview_dir, job, clean[pos], result,
                                                         runtime_binding["worker_sha256"])
                    previews.append(preview_info)
                output = {
                    "schema": "stvg-reference-swap-teacher-output-v1",
                    "status": "complete",
                    "job_id": job["job_id"],
                    "case_id": case_id,
                    "dataset": ds,
                    "position": pos,
                    "physical_frame_id": fid,
                    "decision_status": result["decision_status"],
                    "reason": result["reason"],
                    "selected_box_cxcywh": result["selected_box_cxcywh"],
                    "selected_candidate_id": result["selected_candidate_id"],
                    "selected_score": result["selected_score"],
                    "margin": result["margin"],
                    "input_audit": {
                        "source_image_sha256": job["frame_rgb_sha256"],
                        "source_image_shape": job["frame_shape"],
                        "source_image_dtype": job["frame_dtype"],
                        "parser_path": result["parser_path"],
                        "context_eligible": result["context_eligible"],
                        "context_token_count": result["context_token_count"],
                        "target_span": result["target_span"],
                        "target_token_positions": result["target_token_positions"],
                        "phrase_token_positions": result["phrase_token_positions"],
                        "model_inputs": result["model_inputs"],
                    },
                    "raw_boxes_cxcywh": result["raw_boxes_cxcywh"],
                    "raw_phrase_scores": result["raw_phrase_scores"],
                    "processed_probe": result["processed_probe"],
                    "private_preview": preview_info,
                    "gate": GATE,
                    "bindings": {
                        "design_lock_sha256": job["design_lock_sha256"],
                        "roster_sha256": _sha_file(ROSTER),
                        "source_roster_sha256": job["source_roster_sha256"],
                        "teacher_job_lock_sha256": runtime_binding["teacher_job_lock_sha256"],
                        "source_capture_barrier_sha256": job["source_capture_barrier_sha256"],
                        "parent_wf_reference_lock_sha256": job["parent_wf_reference_lock_sha256"],
                        "source_case_file_sha256": job["source_case_file_sha256"],
                        "source_case_sha256": job["source_case_sha256"],
                        "source_receipt_sha256": job["source_receipt_sha256"],
                        "subject_sha256": job["subject_sha256"],
                        "plan_sha256": job["plan_sha256"],
                        "query_key_sha256": job["query_key_sha256"],
                        "source_pixel_sha256": job["source_pixel_sha256"],
                        "source_model_state_sha256": job["source_model_state_sha256"],
                        "source_checkpoint_sha256": job["source_checkpoint_sha256"],
                        "parent_wf_anchor_sha256": job["parent_wf_anchor_sha256"],
                        "job_binding_sha256": job["job_binding_sha256"],
                        "runtime_sha256": runtime_binding["runtime_sha256"],
                        "worker_sha256": runtime_binding["worker_sha256"],
                        "expert_checkpoint_sha256": runtime_binding["expert_checkpoint_sha256"],
                        "expert_config_sha256": runtime_binding["expert_config_sha256"],
                        "processor_config_sha256": runtime_binding["processor_config_sha256"],
                    },
                    "compute": {
                        "forward_seconds": result["forward_seconds"],
                        "attempt_seconds": time.perf_counter() - call_started,
                    },
                    "GT_read": False,
                }
                path = _output_path(job["job_id"])
                _write_json(path, output)
                relative = str(path.relative_to(BASE))
                file_row = {
                    "job_id": job["job_id"],
                    "path": relative,
                    "sha256": _sha_file(path),
                    "bytes": path.stat().st_size,
                    "case_id": case_id,
                    "position": pos,
                    "physical_frame_id": fid,
                    "decision_status": result["decision_status"],
                }
                files.append(file_row)
                outputs[job["job_id"]] = output
                _write_json(STATUS, {
                    "schema": "stvg-reference-swap-teacher-status-v1",
                    "status": "running",
                    "completed_attempts": call_count,
                    "total_attempts": 49,
                    "accepted": accepted_count,
                    "rejected": rejected_count,
                    "unknown": unknown_count,
                    "last_job_id": job["job_id"],
                    "GT_read": False,
                }, replace=True)
            del clean, frames, source_case, frame_cache
        if call_count != 49 or len(files) != 49 or len(outputs) != 49:
            raise ValueError("teacher did not produce one output for every locked attempt")
        if expert is None or before_state is None:
            raise ValueError("no DINO expert was loaded for the locked jobs")
        after_state = state_hash(expert.model.state_dict())
        if after_state != before_state:
            raise ValueError("frozen DINO expert state changed during inference")
        binding_doc = {
            "design_lock_sha256": _sha_file(DESIGN),
            "roster_sha256": _sha_file(ROSTER),
            "source_roster_sha256": _sha_file(ROSTER_IN),
            "cpu_contracts_sha256": _sha_file(BASE / "CPU_CONTRACTS.json"),
            "teacher_job_lock_sha256": _sha_file(JOB_LOCK),
            "source_capture_barrier_sha256": _sha_file(SOURCE_BARRIER),
            "parent_wf_reference_lock_sha256": _sha_file(PARENT_REFERENCE),
            "runtime_sha256": runtime_binding["runtime_sha256"],
            "worker_sha256": runtime_binding["worker_sha256"],
            "expert_checkpoint_sha256": runtime_binding["expert_checkpoint_sha256"],
            "expert_config_sha256": runtime_binding["expert_config_sha256"],
            "processor_config_sha256": runtime_binding["processor_config_sha256"],
        }
        barrier = {
            "schema": "stvg-reference-swap-teacher-global-barrier-v1",
            "status": "complete",
            "count": 49,
            "source_count": 32,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "unknown_count": unknown_count,
            "files": files,
            "bindings": binding_doc,
            "dino_state_before_sha256": before_state,
            "dino_state_after_sha256": after_state,
            "dino_state_unchanged": True,
            "environment": environment,
            "private_previews": previews,
            "elapsed_seconds": time.time() - started,
            "GT_read": False,
        }
        _write_json(GLOBAL_BARRIER, barrier)
        barrier_sha = _sha_file(GLOBAL_BARRIER)
        ref_lock = _make_reference_lock(job_doc, outputs, files, barrier_sha, runtime_binding)
        _write_json(REFERENCE_LOCK, ref_lock)
        _write_json(STATUS, {
            "schema": "stvg-reference-swap-teacher-status-v1",
            "status": "complete",
            "completed_attempts": 49,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "unknown": unknown_count,
            "teacher_global_barrier_sha256": barrier_sha,
            "reference_lock_sha256": _sha_file(REFERENCE_LOCK),
            "GT_read": False,
        }, replace=True)
    except BaseException as exc:
        # Preserve all completed per-attempt files and do not resume automatically.
        _write_json(STATUS, {
            "schema": "stvg-reference-swap-teacher-status-v1",
            "status": "failed_partial_preserved",
            "completed_attempts": len(files),
            "total_attempts": 49,
            "last_job_id": files[-1]["job_id"] if files else None,
            "error_type": type(exc).__name__,
            "GT_read": False,
        }, replace=True)
        raise
    finally:
        if expert is not None:
            del expert
        lock.close()


def _pair_common_anchors(jobs: list[dict[str, Any]], outputs: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Small pure helper used by CPU contracts and final reference builder."""
    dino, wf = [], []
    for job in jobs:
        output = outputs[job["job_id"]]
        if output["decision_status"] != "accepted":
            continue
        box = output.get("selected_box_cxcywh")
        if not _is_valid_box(box):
            raise ValueError("accepted outputs need valid normalized geometry")
        original = job["parent_wf_anchor"]
        common = {
            "position": int(job["position"]),
            "physical_frame_id": int(job["physical_frame_id"]),
            "weight": float(original["weight"]),
        }
        dino.append({**common, "box_cxcywh": [float(x) for x in box],
                     "candidate_id": "DINO_same_frame_expert",
                     "teacher_job_id": job["job_id"],
                     "teacher_output_sha256": output.get("_output_sha256", "0" * 64),
                     "original_wf_candidate_id": original.get("candidate_id"),
                     "original_wf_teacher_output_sha256": original["teacher_output_sha256"]})
        wf.append({**common, "box_cxcywh": [float(x) for x in original["box_cxcywh"]],
                   "candidate_id": original.get("candidate_id"),
                   "teacher_job_id": job["job_id"],
                   "teacher_output_sha256": output.get("_output_sha256", "0" * 64),
                   "original_wf_candidate_id": original.get("candidate_id"),
                   "original_wf_teacher_output_sha256": original["teacher_output_sha256"]})
    for left, right in zip(dino, wf):
        if (left["position"], left["physical_frame_id"], left["weight"]) != (
                right["position"], right["physical_frame_id"], right["weight"]):
            raise AssertionError("paired supports differ")
    return dino, wf


def _contracts() -> dict[str, Any]:
    import numpy as np

    jobs = [
        {"job_id": "HC01_001_10", "case_id": "HC01", "position": 1,
         "physical_frame_id": 10,
         "parent_wf_anchor": {"position": 1, "physical_frame_id": 10,
                              "box_cxcywh": [0.4, 0.5, 0.2, 0.3], "weight": 0.625,
                              "candidate_id": 7, "teacher_output_sha256": "a" * 64}},
        {"job_id": "HC01_003_14", "case_id": "HC01", "position": 3,
         "physical_frame_id": 14,
         "parent_wf_anchor": {"position": 3, "physical_frame_id": 14,
                              "box_cxcywh": [0.6, 0.5, 0.2, 0.3], "weight": 0.25,
                              "candidate_id": 8, "teacher_output_sha256": "b" * 64}},
        {"job_id": "HC01_004_16", "case_id": "HC01", "position": 4,
         "physical_frame_id": 16,
         "parent_wf_anchor": {"position": 4, "physical_frame_id": 16,
                              "box_cxcywh": [0.5, 0.4, 0.1, 0.1], "weight": 1.0,
                              "candidate_id": 9, "teacher_output_sha256": "c" * 64}},
    ]
    out = {
        jobs[0]["job_id"]: {"decision_status": "accepted", "selected_box_cxcywh": [0.41, 0.51, 0.2, 0.3], "reason": "accepted"},
        jobs[1]["job_id"]: {"decision_status": "rejected", "selected_box_cxcywh": None, "reason": "low_target_score"},
        jobs[2]["job_id"]: {"decision_status": "unknown", "selected_box_cxcywh": None, "reason": "query_parser_abstention"},
    }
    dino, wf = _pair_common_anchors(jobs, out)
    assert len(dino) == len(wf) == 1
    assert dino[0]["position"] == wf[0]["position"] == 1
    assert dino[0]["physical_frame_id"] == wf[0]["physical_frame_id"] == 10
    assert dino[0]["weight"] == wf[0]["weight"] == 0.625
    assert dino[0]["box_cxcywh"] != wf[0]["box_cxcywh"]
    empty, empty_wf = _pair_common_anchors(jobs[1:], out)
    assert empty == empty_wf == []
    pixel = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    assert _sha_bytes(np.ascontiguousarray(pixel).tobytes()) == _sha_bytes(pixel.tobytes())
    assert _is_valid_box([0.5, 0.5, 0.2, 0.3])
    assert not _is_valid_box([0.99, 0.5, 0.2, 0.3])
    assert not _is_valid_box([0.5, 0.5, -0.2, 0.3])
    return {
        "schema": "stvg-reference-swap-teacher-cpu-contracts-v1",
        "status": "pass",
        "checks": 12,
        "checks_list": [
            "accepted-only common support",
            "rejected attempt retained but excluded",
            "unknown attempt retained but excluded",
            "empty support remains empty",
            "paired physical position equality",
            "paired physical frame equality",
            "exact WF weight copied to both arms",
            "geometry is the only scientific input difference",
            "normalized valid box accepted",
            "out-of-frame box rejected",
            "nonpositive box size rejected",
            "deterministic RGB-byte SHA helper",
        ],
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--contracts", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        result = _contracts()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.prepare:
        prepare()
        print(f"PREPARED {ROSTER.relative_to(ROOT)} {JOB_LOCK.relative_to(ROOT)}")
    else:
        _run_teacher()
        print(f"SEALED {GLOBAL_BARRIER.relative_to(ROOT)} {REFERENCE_LOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
