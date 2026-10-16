"""DINO-free cold TA-STVG source capture for the fixed Table 1 probe roster.

This file defines the source adapter and execution entrypoint only. It does not
read labels, run R1, fit either learner, or select candidates from outcomes.
The root-controlled launcher must first pin SOURCE_RUNTIME.json.
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
ROSTER = BASE / "PREFLIGHT_ROSTER_LOCK.json"
MANIFEST = BASE / "INPUT_MANIFEST.json"
RUNTIME = BASE / "SOURCE_RUNTIME.json"
OUT = BASE / "source_cases"
BARRIER = BASE / "SOURCE_CAPTURE_BARRIER.json"
STATUS = BASE / "SOURCE_CAPTURE_STATUS.json"
CONTRACTS = BASE / "source_contracts/CPU_CONTRACTS.json"

sys.path.insert(0, str(ROOT))

PINNED_SOURCE_FILES = (
    "scripts/run_stvg_table1_source_v1.py",
    "scripts/run_decota_paper_main_v1.py",
    "scripts/run_spatial_regression_alignment_v1.py",
    "scripts/run_spatial_ssl_gpu_v1.py",
    "scripts/stvg_opd_p6_existing_common001.py",
    "scripts/stvg_mllm_candidates_v1.py",
    "scripts/stvg_table1_temporal_adapter_v1.py",
    "vg_tta/stvg_opd_frozen_mask_cache012.py",
    "vg_tta/stvg_opd_frozen_visual_host011.py",
    "vg_tta/stvg_opd_frozen_attention_lifetime010.py",
    "vg_tta/stvg_opd_frozen_capture_cpu007.py",
    "vg_tta/tastvg_paper48_hc2_decode_v1.py",
    "vg_tta/exact_frame_decode_audit_v2.py",
    "methods/decota_final_simplified_v1/backbone.py",
    "methods/decota_final_simplified_v1/replay.py",
    "methods/decota_final_simplified_v1/objectives.py",
    "methods/decota_final_simplified_v1/tensors.py",
    "methods/decota_final_simplified_v1/config.py",
)

SOURCE_DIRECTION = {"hc2": "vid_to_hc1", "vidstg": "hc2_to_vid"}
SOURCE_MODEL_NAME = {"hc2": "vidstg", "vidstg": "hcstvg2"}
CLIP_FRAMES = 8


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> Any:
    """Hash tensors/arrays by dtype, shape and bytes; never serialize payloads as JSON."""
    import numpy as np
    try:
        import torch
    except ImportError:  # CPU metadata contracts do not require torch.
        torch = None
    if torch is not None and torch.is_tensor(value):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Source case hashing accepts detached CPU tensors only")
        value = value.detach().contiguous().numpy()
    if isinstance(value, np.ndarray):
        x = np.ascontiguousarray(value)
        return {"__array__": True, "dtype": str(x.dtype), "shape": list(x.shape),
                "sha256": _sha_bytes(x.tobytes())}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite value in source case")
    return value


def content_sha256(value: Mapping[str, Any]) -> str:
    body = {k: v for k, v in value.items()
            if k not in {"source_case_sha256", "compute", "source_runtime_sha256"}}
    raw = json.dumps(_canonical(body), sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_atomic(path: str | Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def verify_runtime() -> dict[str, Any]:
    if not RUNTIME.is_file():
        raise RuntimeError("Root must create SOURCE_RUNTIME.json after reviewing this worker")
    runtime = read_json(RUNTIME)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("SOURCE_RUNTIME must be locked with GT_read=false")
    roster_sha = sha256_file(ROSTER)
    manifest_sha = sha256_file(MANIFEST)
    if runtime.get("preflight_roster_sha256") != roster_sha:
        raise RuntimeError("runtime does not bind PREFLIGHT_ROSTER_LOCK")
    if runtime.get("input_manifest_sha256") != manifest_sha:
        raise RuntimeError("runtime does not bind INPUT_MANIFEST")
    pins = runtime.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise RuntimeError("runtime pins are required")
    if pins.get("scripts/run_stvg_table1_source_v1.py") != sha256_file(__file__):
        raise RuntimeError("runtime does not pin this exact source worker")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"runtime source/protected binding mismatch: {rel}")
    checkpoints = runtime.get("source_checkpoints")
    if not isinstance(checkpoints, dict):
        raise RuntimeError("runtime must bind the two cold TA-STVG source checkpoints")
    expected_checkpoint_sha = {
        "hc2": "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83",
        "vidstg": "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036",
    }
    for target, expected in expected_checkpoint_sha.items():
        row = checkpoints.get(target)
        actual = row.get("sha256") if isinstance(row, dict) else row
        if actual != expected:
            raise RuntimeError(f"runtime checkpoint binding mismatch for {target}")
    return runtime


def _roster_records() -> list[dict[str, Any]]:
    roster = read_json(ROSTER)
    if (roster.get("status") != "locked" or roster.get("GT_read") is not False
            or roster.get("representative_probe_count") != 64
            or roster.get("parents_per_domain") != 32):
        raise RuntimeError("Unexpected or unsealed Table 1 preflight roster")
    if roster.get("query_manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("preflight roster no longer binds the full input manifest")
    records = roster.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("Expected exactly 64 fixed source probes")
    aliases = [row.get("probe_alias") for row in records]
    if len(set(aliases)) != 64 or set(aliases) != {
            *(f"HC{i:02d}" for i in range(1, 33)), *(f"V{i:02d}" for i in range(1, 33))}:
        raise RuntimeError("Roster aliases do not match the fixed 32+32 cohort")
    return records


def _validate_roster_record(rec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    ds = rec.get("dataset")
    if ds not in SOURCE_DIRECTION:
        raise ValueError("Only the locked HC2 and VidSTG domains are supported")
    plan_path = ROOT / rec["plan_path"]
    subject_path = ROOT / rec["subject_path"]
    if sha256_file(plan_path) != rec["plan_sha256"]:
        raise ValueError("Official source plan changed after preflight")
    if sha256_file(subject_path) != rec["subject_sha256"]:
        raise ValueError("Clean subject parse receipt changed after preflight")
    from scripts.run_decota_paper_main_v1 import read_row
    row = read_row(ds, int(rec["query_ordinal"]))
    source = row["input"]
    expected_bounds = [int(source.get("start_frame", 0)),
                       int(source.get("end_frame", source["frame_count"]))]
    comparisons = {
        "video_sha256": source.get("video_sha256"),
        "fps": source.get("fps"),
        "width": source.get("width"),
        "height": source.get("height"),
        "frame_count": source.get("frame_count"),
        "duration": source.get("duration"),
    }
    for key, value in comparisons.items():
        if rec.get(key) != value:
            raise ValueError(f"roster/PLAN input metadata mismatch for {key}")
    if list(source.get("frame_ids", [])) != list(rec.get("frame_ids", [])):
        raise ValueError("roster and PLAN source-grid frame IDs differ")
    if list(rec.get("video_bounds", [])) != expected_bounds:
        raise ValueError("roster true video bounds differ from the original PLAN row")
    if row.get("subject_sha256") != rec.get("subject_sha256"):
        raise ValueError("clean subject parse binding mismatch")
    # Caption/query is held only in this transient row, used by native capture,
    # and is intentionally omitted from the serialized source_case.input.
    return ds, row


def _clip_spec(interval: Any, bounds: Any) -> dict[str, Any]:
    """Match the frozen 8-frame half-open sampler without decoding or padding."""
    if interval is None or bounds is None or len(interval) != 2 or len(bounds) != 2:
        return {"status": "unavailable", "reason": "invalid_interval_or_bounds"}
    lo, hi = map(float, interval)
    blo, bhi = map(int, bounds)
    if not all(math.isfinite(x) for x in (lo, hi)) or not blo < bhi or not lo < hi:
        return {"status": "unavailable", "reason": "invalid_interval_or_bounds"}
    start = max(blo, int(lo))
    end = min(bhi, int(hi))
    available_count = max(0, end - start)
    if available_count < CLIP_FRAMES:
        return {"status": "unavailable", "reason": "fewer_than_8_original_physical_frames",
                "clipped_physical_bounds": [start, end], "available_physical_frame_count": available_count}
    stride = max(1, (available_count - 1) // (CLIP_FRAMES - 1))
    sampled = [start + k * stride for k in range(CLIP_FRAMES)]
    if sampled[-1] >= end or len(set(sampled)) != CLIP_FRAMES:
        raise AssertionError("fixed 8-frame physical sampler produced an invalid clip")
    return {"status": "ready", "reason": None,
            "clipped_physical_bounds": [start, end],
            "available_physical_frame_count": available_count,
            "sampled_physical_frame_ids": sampled,
            "uniform_stride": stride}


def _candidate_edge_records(candidate_bundle: Mapping[str, Any], frame_ids: Any,
                            native_interval: Any, bounds: Any, fps: float) -> dict[str, Any]:
    import numpy as np
    ids = np.asarray(frame_ids, dtype=np.int64)
    temporal = candidate_bundle["temporal"]
    unique = list(temporal.get("candidates", []))
    by_role = {row["role"]: row for row in unique}
    requests = []
    role_rows = []
    for request in temporal.get("requests", []):
        role = request["role"]
        candidate = by_role.get(role)
        if not request.get("valid"):
            row = {"role": role, "candidate_id": None, "status": "unavailable",
                   "reason": request.get("reason") or "invalid_after_original_bounds_clamp",
                   "requested_interval": request.get("requested_interval"),
                   "interval_frames": request.get("interval")}
        elif request.get("duplicate_of") is not None:
            row = {"role": role, "candidate_id": None, "status": "duplicate_not_replaced",
                   "reason": "exact_interval_duplicate_of_prior_role",
                   "duplicate_of": request["duplicate_of"],
                   "requested_interval": request.get("requested_interval"),
                   "interval_frames": request.get("interval")}
        elif candidate is None:
            row = {"role": role, "candidate_id": None, "status": "unavailable",
                   "reason": "candidate_missing_from_original_deduplicated_pool",
                   "requested_interval": request.get("requested_interval"),
                   "interval_frames": request.get("interval")}
        else:
            clip = _clip_spec(candidate["interval"], bounds)
            lo, hi = map(float, candidate["interval"])
            positions = np.flatnonzero((ids >= lo) & (ids < hi)).astype(np.int64).tolist()
            grid_count = len(set(int(ids[i]) for i in positions))
            eligible_update = clip["status"] == "ready" and grid_count >= 4
            reason = None
            if clip["status"] != "ready":
                reason = clip["reason"]
            elif grid_count < 4:
                reason = "fewer_than_4_distinct_original_source_grid_centers"
            row = {
                "role": role,
                "candidate_id": candidate["candidate_id"],
                "status": "ready" if eligible_update else "unknown_or_native_fallback_required",
                "reason": reason,
                "requested_interval": candidate.get("requested_interval"),
                "interval_frames": list(candidate["interval"]),
                "clip": clip,
                "clip_fps": float(fps / clip["uniform_stride"]) if clip["status"] == "ready" else None,
                "start_time_s": ((clip["sampled_physical_frame_ids"][0] - int(bounds[0])) / fps
                                 if clip["status"] == "ready" else None),
                "source_grid_positions_in_interval": positions,
                "distinct_source_grid_center_count": grid_count,
                "clip_request_eligible": clip["status"] == "ready",
                "temporal_update_eligible": eligible_update,
                "visual_ID": "1",
            }
            role_rows.append(row)
        requests.append(row)
    valid = [row for row in role_rows if row["temporal_update_eligible"]]
    return {
        "candidate_roles": requests,
        "unique_candidate_count": len(unique),
        "valid_8frame_candidate_count": sum(x["clip_request_eligible"] for x in role_rows),
        "update_eligible_candidate_count": len(valid),
        "native_support_source_grid_count": int(sum(
            int(native_interval[0]) <= int(fid) < int(native_interval[1]) for fid in ids)),
        "requires_unknown_or_native_fallback": len(valid) != 4,
        "fit_allowed_only_if_all_four_unique_and_eligible": len(unique) == 4 and len(valid) == 4,
        "candidate_algorithm_changed": False,
        "no_frame_padding_or_interval_radius_change": True,
        "source_grid_positions_are_not_full_video_frame_map": True,
    }


def _input_for_source_case(input_row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("video_path", "video_sha256", "fps", "width", "height", "frame_ids",
               "frame_count", "duration", "start_frame", "end_frame", "index", "kind")
    return {key: input_row[key] for key in allowed if key in input_row}


def capture_one(model: Any, record: Mapping[str, Any], frame_cache: Any = None) -> dict[str, Any]:
    """Capture one cold frozen source. Caller owns model lifetime; no DINO/GT/fit."""
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.objectives import prediction, project
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.stvg_mllm_candidates_v1 import generate_candidates
    from scripts.stvg_opd_p6_existing_common001 import config, check_projection
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward

    runtime = read_json(RUNTIME)
    runtime_sha = sha256_file(RUNTIME)
    ds, row = _validate_roster_record(record)
    before_model_sha = state_hash(model.state_dict())
    if before_model_sha != getattr(model, "_fixed_full_state_hash", None):
        raise RuntimeError("loaded source state differs from model_for() frozen-state receipt")
    video_path = Path(row["input"]["video_path"])
    if not video_path.is_file() or sha256_file(video_path) != row["input"]["video_sha256"]:
        raise RuntimeError("source video bytes do not match the official input row")
    if frame_cache is None:
        frame_cache = collections.OrderedDict()
    frames, ids, reused = frames_for(ds, row, frame_cache)
    if list(map(int, ids)) != list(map(int, record["frame_ids"])):
        raise RuntimeError("official decoder output frame IDs differ from locked source grid")
    if tuple(frames.shape[1:3]) != (int(record["height"]), int(record["width"])):
        raise RuntimeError("decoded source frame geometry differs from roster metadata")
    pixel_sha = _sha_bytes(np.ascontiguousarray(frames).tobytes())
    started = time.perf_counter()
    with torch.no_grad():
        batch, records, base = frozen_forward(model, frames, row)
    native = prediction(base.zero["logits"], base.zero["boxes"], records, ids)
    if int(base.initial["spatial.query_residual"].count_nonzero()) != 0:
        raise RuntimeError("cold source query residual is not exactly zero")

    source_lns: dict[str, Any] = {}
    for name, value in base.initial.items():
        if name == "spatial.query_residual":
            continue
        _, _, layer, norm_name, parameter_name = name.split(".")
        original = getattr(getattr(model.ground_decoder.decoder.layers[int(layer)], norm_name), parameter_name)
        if not torch.equal(value.detach().cpu(), original.detach().float().cpu()):
            raise RuntimeError("captured source LayerNorm differs from the frozen model")
        source_lns[name] = original.detach().float().cpu().clone()

    cfg = config(ds)
    evidence = project(base.zero["logits"], base.zero["actions"], records, cfg)
    projection_error = check_projection(base.zero["logits"], base.zero["actions"], records,
                                        evidence, cfg)
    intervals = [item["interval"] for item in evidence["offsets"]]
    projection_interval = [min(item[0] for item in intervals), max(item[1] for item in intervals)]
    native_cxcywh = native["boxes"].detach().cpu().numpy().copy()
    chart = native_cxcywh.astype(np.float64)
    native_xyxy = np.concatenate((chart[:, :2] - chart[:, 2:] / 2,
                                  chart[:, :2] + chart[:, 2:] / 2), axis=1)
    seed = 20261013 + int(record["query_ordinal"]) + (0 if ds == "hc2" else 100000)
    candidates = generate_candidates(
        native_xyxy, np.asarray(ids), native["physical_interval"], projection_interval,
        ds, seed, cfg={"video_bounds": record["video_bounds"]})
    edge_records = _candidate_edge_records(
        candidates, ids, native["physical_interval"], record["video_bounds"],
        float(row["input"]["fps"]))

    rgb_receipts = {
        int(fid): {"RGB_sha256": _sha_bytes(np.ascontiguousarray(frame).tobytes()),
                   "shape": list(frame.shape), "dtype": str(frame.dtype)}
        for fid, frame in zip(ids, frames)
    }
    temporal_inputs = [detached(value, "cpu") for value in base.temporal_inputs]
    temporal_hidden = [value[-1, 0].detach().cpu().clone() for value in base.temporal_inputs]
    source_input = _input_for_source_case(row["input"])
    method_config = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    if state_hash(model.state_dict()) != before_model_sha:
        raise RuntimeError("frozen source model state changed during capture")
    if before_model_sha != model._fixed_full_state_hash:
        raise RuntimeError("source frozen-state receipt changed during capture")

    source_case: dict[str, Any] = {
        "schema": "stvg-final-table1-source-case-v1",
        "case_id": record["probe_alias"],
        "dataset": ds,
        "query_ordinal": int(record["query_ordinal"]),
        "query_key_sha256": record["query_key_sha256"],
        "parent_sha256": record["parent_sha256"],
        "subject_sha256": record["subject_sha256"],
        "plan_sha256": record["plan_sha256"],
        "input": source_input,
        "width": int(record["width"]),
        "height": int(record["height"]),
        "video_bounds": list(map(int, record["video_bounds"])),
        "frame_ids": np.asarray(ids, dtype=np.int64),
        "Native_interval": list(native["physical_interval"]),
        "Projection_interval": list(projection_interval),
        "native_boxes_cxcywh": native_cxcywh,
        "native_boxes_xyxy_normalized": native_xyxy,
        "native_box_coordinate_system": "normalized continuous xyxy, float64 conversion from original native cxcywh; no clamp",
        "pixel_sha256": pixel_sha,
        "source_RGB": rgb_receipts,
        "zero": detached({key: base.zero[key] for key in ("boxes", "logits", "actions")}, "cpu"),
        "records": detached(records, "cpu"),
        "temporal_inputs": temporal_inputs,
        "temporal_hidden": temporal_hidden,
        "source_light_initial_state": detached(base.initial, "cpu"),
        "source_original_LN_values": source_lns,
        "source_model_state_sha256": before_model_sha,
        "source_checkpoint_sha256": cfg.checkpoint_sha256,
        "source_direction": cfg.direction,
        "source_model_config": method_config,
        "projection_independent_max_error": float(projection_error),
        "candidates": candidates,
        "temporal_candidates": [
            {**candidate, "canonical_id": candidate["candidate_id"],
             "interval_frames": list(candidate["interval"])}
            for candidate in candidates["temporal"].get("candidates", [])
        ],
        "candidate_validity": edge_records,
        "mark_requests": edge_records["candidate_roles"],
        "frame_clock": {
            "fps": float(row["input"]["fps"]),
            "fps_source": "official PLAN input metadata, unchanged",
            "physical_frame_ids_source": "official dataset decoder, exact roster frame grid",
            "video_bounds_source": "sealed preflight roster matched to original PLAN row",
            "interval_semantics": "continuous physical frame axis, half-open",
            "marked_clip_contract": "8 distinct original physical frame IDs; no temporal padding or frame fabrication",
        },
        "source_case_sha256": None,
        "source_runtime_sha256": runtime_sha,
        "GT_read": False,
        "DINO_read_or_call": False,
        "old_adapted_state_loaded": False,
        "optimizer_steps": 0,
        "backwards": 0,
        "compute": {
            "actual_source_capture_calls": 1,
            "actual_source_offset_forwards": 2,
            "source_capture_seconds": float(time.perf_counter() - started),
            "decoded_frame_cache_reused": bool(reused),
            "CUDA_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "candidate_count_unique": int(edge_records["unique_candidate_count"]),
            "valid_8frame_candidate_count": int(edge_records["valid_8frame_candidate_count"]),
            "update_eligible_candidate_count": int(edge_records["update_eligible_candidate_count"]),
        },
    }
    source_case["source_case_sha256"] = content_sha256(source_case)
    del frames, batch, base, native, evidence
    frame_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return source_case


def _save_case(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    alias = case["case_id"]
    path = OUT / f"{alias}.pt"
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"Preserve incomplete/unreceipted source output: {alias}")
        receipt = read_json(receipt_path)
        if (sha256_file(path) != receipt.get("sha256")
                or receipt.get("source_case_sha256") != case.get("source_case_sha256")
                or receipt.get("source_runtime_sha256") != sha256_file(RUNTIME)):
            raise RuntimeError(f"Existing source output differs; do not overwrite: {alias}")
        return {"path": str(path.relative_to(ROOT)), "sha256": receipt["sha256"],
                "bytes": path.stat().st_size, "source_case_sha256": receipt["source_case_sha256"],
                "receipt_path": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": sha256_file(receipt_path), "reused": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(dict(case), temp)
    os.replace(temp, path)
    receipt = {
        "schema": "stvg-final-table1-source-receipt-v1",
        "case_id": alias,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_case_sha256": case["source_case_sha256"],
        "source_runtime_sha256": case["source_runtime_sha256"],
        "query_key_sha256": case["query_key_sha256"],
        "GT_read": False,
        "DINO_read_or_call": False,
        "time": time.time(),
    }
    _write_atomic(receipt_path, receipt)
    return {"path": str(path.relative_to(ROOT)), "sha256": receipt["sha256"],
            "bytes": receipt["bytes"], "source_case_sha256": receipt["source_case_sha256"],
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": sha256_file(receipt_path), "reused": False}


def run(case_alias: str | None = None, all_preflight: bool = False) -> None:
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.config import MethodConfig
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_decota_paper_main_v1 import gpu, model_for

    runtime = verify_runtime()
    records = _roster_records()
    if all_preflight == (case_alias is not None):
        raise ValueError("Choose exactly one of --all-preflight or --case-alias")
    selected = records if all_preflight else [r for r in records if r["probe_alias"] == case_alias]
    if len(selected) != (64 if all_preflight else 1):
        raise ValueError("Requested alias is absent from the sealed Table 1 roster")

    install_lease = gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    np.random.seed(20261015)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = None
    loaded_source = None
    model_sha = None
    files: dict[str, Any] = {}
    edge_counts = {
        "candidate_roles": 0,
        "unique_candidates": 0,
        "valid_8frame_clips": 0,
        "update_eligible_candidates": 0,
        "fewer_than_8_original_physical_frames": 0,
        "fewer_than_4_distinct_source_grid_centers": 0,
        "duplicate_not_replaced": 0,
        "cases_requiring_unknown_or_native_fallback": 0,
    }
    actual_source_capture_calls = 0
    actual_source_offset_forwards = 0
    reused_source_cases = 0
    started = time.time()
    try:
        for index, rec in enumerate(selected, 1):
            ds = rec["dataset"]
            source_name = SOURCE_MODEL_NAME[ds]
            expected = MethodConfig.for_direction(SOURCE_DIRECTION[ds])
            binding = runtime["source_checkpoints"][ds]
            binding_sha = binding.get("sha256") if isinstance(binding, dict) else binding
            if binding_sha != expected.checkpoint_sha256:
                raise RuntimeError("locked checkpoint differs from the source MethodConfig")
            if loaded_source != source_name:
                if model is not None:
                    if state_hash(model.state_dict()) != model_sha:
                        raise RuntimeError("source model state changed between records")
                    del model
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                model = model_for(source_name)
                loaded_source = source_name
                model_sha = model._fixed_full_state_hash
            frame_cache = collections.OrderedDict()
            dst = OUT / f"{rec['probe_alias']}.pt"
            receipt_dst = dst.with_suffix(".receipt.json")
            if dst.exists() and receipt_dst.exists():
                rc = read_json(receipt_dst)
                if (sha256_file(dst) != rc.get("sha256")
                        or rc.get("source_runtime_sha256") != sha256_file(RUNTIME)
                        or rc.get("query_key_sha256") != rec["query_key_sha256"]):
                    raise RuntimeError(f"Existing output binding mismatch: {rec['probe_alias']}")
                index_record = {
                    "path": str(dst.relative_to(ROOT)), "sha256": rc["sha256"],
                    "bytes": dst.stat().st_size, "source_case_sha256": rc["source_case_sha256"],
                    "receipt_path": str(receipt_dst.relative_to(ROOT)),
                    "receipt_sha256": sha256_file(receipt_dst), "reused": True,
                }
                import torch as _torch
                case = _torch.load(dst, map_location="cpu", weights_only=False)
                if (case.get("case_id") != rec["probe_alias"]
                        or case.get("source_case_sha256") != content_sha256(case)
                        or case.get("source_case_sha256") != rc.get("source_case_sha256")
                        or case.get("source_runtime_sha256") != sha256_file(RUNTIME)
                        or case.get("query_key_sha256") != rec["query_key_sha256"]
                        or case.get("input", {}).get("video_sha256") != rec["video_sha256"]):
                    raise RuntimeError(f"Existing source payload canonical binding mismatch: {rec['probe_alias']}")
                reused_source_cases += 1
            else:
                if dst.exists() or receipt_dst.exists():
                    raise RuntimeError(f"Preserve incomplete/unreceipted source output: {rec['probe_alias']}")
                torch.cuda.reset_peak_memory_stats()
                case = capture_one(model, rec, frame_cache)
                index_record = _save_case(case)
                actual_source_capture_calls += 1
                actual_source_offset_forwards += int(case["compute"]["actual_source_offset_forwards"])
            files[rec["probe_alias"]] = index_record
            edge = case["candidate_validity"]
            edge_counts["candidate_roles"] += len(edge["candidate_roles"])
            edge_counts["unique_candidates"] += edge["unique_candidate_count"]
            edge_counts["valid_8frame_clips"] += edge["valid_8frame_candidate_count"]
            edge_counts["update_eligible_candidates"] += edge["update_eligible_candidate_count"]
            for role in edge["candidate_roles"]:
                if role.get("reason") == "fewer_than_8_original_physical_frames":
                    edge_counts["fewer_than_8_original_physical_frames"] += 1
                if role.get("reason") == "fewer_than_4_distinct_original_source_grid_centers":
                    edge_counts["fewer_than_4_distinct_source_grid_centers"] += 1
                if role.get("status") == "duplicate_not_replaced":
                    edge_counts["duplicate_not_replaced"] += 1
            if edge["requires_unknown_or_native_fallback"]:
                edge_counts["cases_requiring_unknown_or_native_fallback"] += 1
            del case
            frame_cache.clear()
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            _write_atomic(STATUS, {
                "status": "source_capture_running",
                "done": index,
                "total": len(selected),
                "current_alias": rec["probe_alias"],
                "source_model": source_name,
                "source_model_loads": len({SOURCE_MODEL_NAME[r["dataset"]] for r in selected[:index]}),
                "GT_read": False,
                "DINO_read_or_call": False,
                "optimizer_steps": 0,
                "backwards": 0,
                "runtime_sha256": sha256_file(RUNTIME),
                "time": time.time(),
            }, replace=True)
            print(f"TABLE1_SOURCE_CASE {index}/{len(selected)} {rec['probe_alias']} "
                  f"reused={bool(index_record['reused'])}", flush=True)
        if model is not None and state_hash(model.state_dict()) != model_sha:
            raise RuntimeError("source model state changed across frozen query captures")
        if all_preflight and len(files) != 64:
            raise RuntimeError("complete preflight requires all 64 source cases")
        if all_preflight:
            barrier = {
                "schema": "stvg-final-table1-source-capture-barrier-v1",
                "status": "sealed",
                "count": 64,
                "source_runtime_sha256": sha256_file(RUNTIME),
                "preflight_roster_sha256": sha256_file(ROSTER),
                "input_manifest_sha256": sha256_file(MANIFEST),
                "source_files": files,
                "edge_counts": edge_counts,
                "actual_source_model_loads": len({SOURCE_MODEL_NAME[r["dataset"]] for r in selected}),
                "actual_source_capture_calls": actual_source_capture_calls,
                "actual_source_offset_forwards": actual_source_offset_forwards,
                "reused_source_cases": reused_source_cases,
                "DINO_calls": 0,
                "optimizer_steps": 0,
                "backwards": 0,
                "GT_read": False,
                "time": time.time(),
            }
            _write_atomic(BARRIER, barrier)
            _write_atomic(STATUS, {"status": "sealed", "count": 64,
                                   "barrier_sha256": sha256_file(BARRIER),
                                   "GT_read": False, "time": time.time()}, replace=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        install_lease.close()


def contracts() -> dict[str, Any]:
    """Synthetic candidate/clock boundary checks; no CUDA context or project data."""
    import numpy as np
    from scripts.stvg_mllm_candidates_v1 import generate_candidates
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("CPU source contracts refuse an initialized CUDA context")
    good = _clip_spec([0, 64], [0, 64])
    short_clip = _clip_spec([4, 11], [0, 64])
    assert good["status"] == "ready" and len(good["sampled_physical_frame_ids"]) == 8
    assert short_clip["status"] == "unavailable"
    ids = np.arange(0, 64, 4, dtype=np.int64)
    chart = np.tile(np.asarray([.5, .5, .2, .3], np.float32), (len(ids), 1)).astype(np.float64)
    xyxy = np.concatenate((chart[:, :2] - chart[:, 2:] / 2,
                           chart[:, :2] + chart[:, 2:] / 2), axis=1)
    bundle = generate_candidates(xyxy, ids, [8, 56], [16, 48], "hc2", 20261015,
                                 cfg={"video_bounds": [0, 64]})
    assert bundle["temporal"]["requested_count"] == 4
    assert len(bundle["temporal"]["requests"]) == 4
    assert bundle["temporal"]["candidates"]
    edges = _candidate_edge_records(bundle, ids, [8, 56], [0, 64], 25.)
    assert edges["candidate_algorithm_changed"] is False
    assert edges["no_frame_padding_or_interval_radius_change"] is True
    # Four or more original centers are required; a shorter grid is never padded.
    narrow = _candidate_edge_records(bundle, ids[:3], [8, 56], [0, 64], 25.)
    assert narrow["requires_unknown_or_native_fallback"] is True
    assert not any(row.get("temporal_update_eligible") for row in narrow["candidate_roles"])
    # Exact duplicate candidate roles remain duplicates; no replacement is made.
    duplicate_bundle = generate_candidates(xyxy, ids, [8, 56], [8, 56], "hc2", 4,
                                           cfg={"video_bounds": [0, 64]})
    duplicate_edges = _candidate_edge_records(duplicate_bundle, ids, [8, 56], [0, 64], 25.)
    assert any(row["status"] == "duplicate_not_replaced" for row in duplicate_edges["candidate_roles"])
    return {
        "schema": "stvg-final-table1-source-cpu-contracts-v1",
        "status": "pass",
        "contracts": 8,
        "fixtures": ["8 distinct physical frames", "short clip unavailable", "fixed four-role requests",
                     "source-grid support edge", "duplicate interval not replaced", "no frame padding",
                     "half-open source intervals", "CPU-only no model/data access"],
        "GT_read": False,
        "media_opened": False,
        "models_loaded": False,
        "GPU_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-preflight", action="store_true")
    group.add_argument("--case-alias", choices=[*(f"HC{i:02d}" for i in range(1, 33)),
                                                  *(f"V{i:02d}" for i in range(1, 33))])
    group.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        value = contracts()
        _write_atomic(CONTRACTS, value, replace=True)
        print("TABLE1_SOURCE_CPU_CONTRACTS_PASS", value["contracts"], flush=True)
    else:
        run(args.case_alias, args.all_preflight)


if __name__ == "__main__":
    main()
