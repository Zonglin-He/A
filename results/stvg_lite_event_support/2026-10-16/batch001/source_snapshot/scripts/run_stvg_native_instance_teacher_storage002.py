#!/usr/bin/env python3
"""Run one frozen native STVG-R1 teacher request for the instance scope.

This worker is intentionally one-case/one-process: it verifies the locked
runtime and local model receipt, consumes only the accepted GT-free tracking
cache, makes exactly one official R1 generation call, and writes a private
teacher row plus a hash-bound receipt. Importing this file is CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_native_instance_common_v1 as common
from scripts import stvg_r1_native_teacher_v1 as native


BASE = common.BASE
TEACHER_RECEIPT = ROOT / "artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json"
CHECKPOINT_REL = Path("checkpoints/STVG-R1-7B")
MAX_NEW_TOKENS = 2048
MAX_MEMORY = {0: "12GiB", "cpu": "8GiB"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _safe_relative_file(relative: str, *, base: Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"unsafe relative path in receipt: {relative!r}")
    path = (base / rel).resolve(strict=True)
    if not path.is_relative_to(base.resolve()):
        raise ValueError("receipt path escapes its bound directory")
    if not path.is_file():
        raise FileNotFoundError("receipt entry is not a regular file")
    return path


def verify_local_model_receipt() -> dict[str, Any]:
    """Check the locked readback, local receipt, and every listed local file."""
    checkpoint = ROOT / CHECKPOINT_REL
    if not checkpoint.is_dir():
        raise FileNotFoundError("the locked local STVG-R1 checkpoint directory is missing")
    root_receipt = common.read(TEACHER_RECEIPT)
    local_receipt_path = checkpoint / "OFFICIAL_RECEIPT.json"
    local_receipt = common.read(local_receipt_path)
    for name, receipt in (("root readback", root_receipt), ("local receipt", local_receipt)):
        if receipt.get("status") != "pass" or receipt.get("GT_read") is not False:
            raise RuntimeError(f"{name} is not a passing GT-free receipt")
        if receipt.get("repository") != native.STVG_R1_MODEL_REPO or receipt.get("revision") != native.STVG_R1_MODEL_REVISION:
            raise RuntimeError(f"{name} does not name the pinned official model revision")
    if root_receipt.get("CPU_only") is not True or root_receipt.get("model_loaded") is not False:
        raise RuntimeError("root model-file readback must be CPU-only and must not claim model loading")
    root_files = root_receipt.get("files")
    local_files = local_receipt.get("files")
    if not isinstance(root_files, Mapping) or not isinstance(local_files, Mapping) or not root_files:
        raise RuntimeError("official model file receipt has no file inventory")
    if dict(root_files) != dict(local_files):
        raise RuntimeError("local official receipt differs from the independent root byte readback")
    if root_receipt.get("count") != len(root_files) or local_receipt.get("weight_bytes") != root_receipt.get("weight_bytes"):
        raise RuntimeError("official model receipt count/byte total is inconsistent")

    total_bytes = 0
    weight_bytes = 0
    for relative, entry in sorted(root_files.items()):
        if not isinstance(relative, str) or not isinstance(entry, Mapping) or set(entry) != {"sha256", "bytes"}:
            raise RuntimeError("malformed official model file receipt entry")
        if not _is_sha(entry["sha256"]) or isinstance(entry["bytes"], bool) or not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
            raise RuntimeError("malformed official model file receipt digest/size")
        path = _safe_relative_file(relative, base=checkpoint)
        if path.stat().st_size != entry["bytes"] or common.sha(path) != entry["sha256"]:
            raise RuntimeError(f"local model file differs from pinned byte receipt: {relative}")
        total_bytes += entry["bytes"]
        if relative.endswith(".safetensors"):
            weight_bytes += entry["bytes"]
    if weight_bytes != root_receipt.get("weight_bytes"):
        raise RuntimeError("summed local safetensors bytes do not match the declared weight total")
    return {
        "checkpoint_path": str(CHECKPOINT_REL),
        "official_model_repository": native.STVG_R1_MODEL_REPO,
        "official_model_revision": native.STVG_R1_MODEL_REVISION,
        "root_readback_sha256": common.sha(TEACHER_RECEIPT),
        "local_official_receipt_sha256": common.sha(local_receipt_path),
        "files_verified": len(root_files),
        "all_receipt_file_bytes_verified": total_bytes,
        "safetensors_weight_bytes_verified": weight_bytes,
    }


def _load_locked_request(case_id: str, roster_row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    relative = roster_row.get("request_path")
    expected_hash = roster_row.get("request_sha256")
    if not isinstance(relative, str) or not _is_sha(expected_hash):
        raise RuntimeError("locked roster lacks its request path/hash")
    path = _safe_relative_file(relative, base=ROOT)
    actual_hash = common.sha(path)
    if actual_hash != expected_hash:
        raise RuntimeError("request bytes differ from the locked roster")
    request = native.validate_request(common.read(path))
    if roster_row.get("case_id") != case_id:
        raise RuntimeError("roster case identity changed")
    return request, actual_hash


def _load_accepted_tracking(case_id: str, request: Mapping[str, Any], request_hash: str) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    accepted_path = BASE / "TRACKING" / case_id / "ACCEPTED.json"
    if not accepted_path.is_file():
        raise FileNotFoundError("official accepted tracking record is not present for this case")
    accepted = common.read(accepted_path)
    required = {"status", "GT_read", "target_mode", "metadata_input", "input_request_sha256",
                "marker_video_path", "marker_video_sha256", "mask_data_dir", "mask_inventory_sha256",
                "dettrack_s", "filter_marker_s", "actual_tracking_wall_s", "tracker_precision",
                "preprocess_warnings"}
    if not required <= set(accepted):
        raise RuntimeError(f"accepted tracking record is missing fields: {sorted(required - set(accepted))}")
    if accepted["status"] != "pass" or accepted["GT_read"] is not False:
        raise RuntimeError("tracking record is not an accepted GT-free output")
    if accepted["target_mode"] != native.TARGET_MODE or accepted["metadata_input"] != "empty_mapping":
        raise RuntimeError("tracking record does not certify the fixed native target mode/input metadata")
    if accepted["input_request_sha256"] != request_hash:
        raise RuntimeError("tracking output is not bound to this roster request")
    if not _is_sha(accepted["marker_video_sha256"]) or not _is_sha(accepted["mask_inventory_sha256"]):
        raise RuntimeError("tracking output has invalid marker/inventory hashes")

    tracking_root = (BASE / "TRACKING" / case_id).resolve()
    reused = accepted.get("reused") is True
    if reused:
        expected_old_receipt = Path("artifacts/stvg_target_teacher_distillation_v1/tracking") / case_id / "ACCEPTED.json"
        if accepted.get("old_receipt_path") != str(expected_old_receipt):
            raise RuntimeError("reused tracking cache is not from the explicitly allowed prior accepted path")
        old_receipt_path = (ROOT / expected_old_receipt).resolve(strict=True)
        if common.sha(old_receipt_path) != accepted.get("old_receipt_sha256"):
            raise RuntimeError("reused tracking cache receipt hash does not match its lineage field")
        old_accepted = common.read(old_receipt_path)
        if (old_accepted.get("status") != "pass" or old_accepted.get("GT_read") is not False
                or old_accepted.get("input_request_sha256") != request_hash
                or old_accepted.get("marker_video_sha256") != accepted["marker_video_sha256"]
                or old_accepted.get("marker_video_path") != accepted["marker_video_path"]
                or old_accepted.get("mask_data_dir") != accepted["mask_data_dir"]):
            raise RuntimeError("reused prior tracking cache does not match the new accepted receipt")
        cache_root = old_receipt_path.parent.resolve()
        reused_receipt_path = old_receipt_path
    else:
        cache_root = tracking_root
        reused_receipt_path = None
    marker_path = Path(accepted["marker_video_path"]).resolve(strict=True)
    mask_path = Path(accepted["mask_data_dir"]).resolve(strict=True)
    if not marker_path.is_relative_to(cache_root) or not marker_path.is_file():
        raise RuntimeError("accepted marker video must remain within its accepted tracking cache")
    if not mask_path.is_relative_to(cache_root) or not mask_path.is_dir() or mask_path.name != "mask_data_filtered_1_3":
        raise RuntimeError("accepted masks must be the official filtered mask directory inside its accepted tracking cache")
    if common.sha(marker_path) != accepted["marker_video_sha256"]:
        raise RuntimeError("accepted marker video bytes do not match their receipt")
    inventory_receipt = tracking_root / "MASK_INVENTORY.json"
    if not inventory_receipt.is_file() or common.sha(inventory_receipt) != accepted["mask_inventory_sha256"]:
        raise RuntimeError("tracking mask inventory bytes do not match their accepted receipt")
    inventory = common.read(inventory_receipt)
    if inventory.get("GT_read") is not False or not isinstance(inventory.get("files"), list):
        raise RuntimeError("tracking mask inventory is not a GT-free file inventory")
    if inventory.get("mask_count") != len(inventory["files"]):
        raise RuntimeError("tracking mask inventory count is inconsistent")
    inventory_indices: set[int] = set()
    inventory_paths: set[Path] = set()
    for item in inventory["files"]:
        if not isinstance(item, Mapping) or set(item) != {"index", "path", "sha256", "bytes"}:
            raise RuntimeError("tracking mask inventory row has an unexpected schema")
        if (isinstance(item["index"], bool) or not isinstance(item["index"], int) or item["index"] < 0
                or item["index"] in inventory_indices or not _is_sha(item["sha256"])
                or isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] <= 0):
            raise RuntimeError("tracking mask inventory row has an invalid byte binding")
        mask_file = _safe_relative_file(item["path"], base=ROOT)
        filename_match = re.fullmatch(r"mask_(\d+)\.json", mask_file.name)
        if (not mask_file.is_relative_to(mask_path) or filename_match is None
                or int(filename_match.group(1)) != item["index"]
                or mask_file.stat().st_size != item["bytes"] or common.sha(mask_file) != item["sha256"]):
            raise RuntimeError("tracking mask bytes/path/index differ from their inventory readback")
        inventory_indices.add(item["index"])
        inventory_paths.add(mask_file)
    inventoried_files, inventoried_ids = native._load_mask_inventory(mask_path)
    if set(inventoried_files.values()) != inventory_paths or sorted(inventoried_ids) != inventory.get("known_ids"):
        raise RuntimeError("accepted mask inventory is incomplete or its known-ID index differs")

    for field in ("dettrack_s", "filter_marker_s", "actual_tracking_wall_s"):
        value = accepted[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise RuntimeError(f"accepted tracking cost {field} is invalid")
    precision = accepted["tracker_precision"]
    if not isinstance(precision, Mapping) or set(precision) != {
        "autocast", "cuda_compute_capability_major", "allow_tf32_matmul", "allow_tf32_cudnn"
    }:
        raise RuntimeError("accepted tracking precision receipt has the wrong schema")
    warnings = accepted["preprocess_warnings"]
    if not isinstance(warnings, list) or any(not isinstance(x, Mapping) or set(x) != {"category", "message"} for x in warnings):
        raise RuntimeError("accepted tracking warnings have the wrong schema")
    if any(not isinstance(x["category"], str) or not isinstance(x["message"], str) for x in warnings):
        raise RuntimeError("accepted tracking warning fields must be strings")
    # The official tracker can repeat identical warnings once per frame. Keep
    # their complete multiset custody in the worker receipt, while the frozen
    # native row carries one copy of each exact warning for compactness.
    unique_warnings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (str(warning["category"]), str(warning["message"]))
        if key not in seen:
            seen.add(key)
            unique_warnings.append({"category": key[0], "message": key[1]})
    warning_histogram = Counter(str(x["category"]) for x in warnings)
    tracking = {
        "accepted_path": str(accepted_path.relative_to(ROOT)),
        "accepted_sha256": common.sha(accepted_path),
        "marker_video_path": marker_path,
        "marker_video_sha256": accepted["marker_video_sha256"],
        "mask_data_dir": mask_path,
        "mask_inventory_sha256": accepted["mask_inventory_sha256"],
        "mask_inventory_receipt_sha256": common.sha(inventory_receipt),
        "reused_prior_cache": reused,
        "prior_accepted_receipt_sha256": common.sha(reused_receipt_path) if reused_receipt_path is not None else None,
        "checkpoint_code_compatibility": accepted.get("checkpoint_code_compatibility"),
        "dettrack_s": float(accepted["dettrack_s"]),
        "filter_marker_s": float(accepted["filter_marker_s"]),
        "actual_tracking_wall_s": float(accepted["actual_tracking_wall_s"]),
        "tracker_precision": dict(precision),
        "preprocess_warnings_for_native_record": unique_warnings,
        "preprocess_warning_count": len(warnings),
        "preprocess_warning_category_counts": dict(sorted(warning_histogram.items())),
        "preprocess_warnings_sha256": _sha_bytes(_canonical_bytes(warnings)),
    }
    # Keep request binding explicit without copying the query text to telemetry.
    tracking["query_sha256"] = request["query_sha256"]
    return accepted, marker_path, mask_path, tracking


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return _sha_bytes(_canonical_bytes({"dtype": str(array.dtype), "shape": list(array.shape)}) + array.tobytes())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite processor metadata cannot be written")
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    raise TypeError(f"unsupported JSON metadata value: {type(value).__name__}")


def _grid_rows(grid: Any) -> list[list[int]]:
    if grid is None:
        raise RuntimeError("processor did not return video_grid_thw; visual token count cannot be verified")
    if hasattr(grid, "detach"):
        grid = grid.detach().to(device="cpu").tolist()
    elif hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, list):
        raise RuntimeError("video_grid_thw has an unexpected representation")
    if len(grid) == 3 and all(isinstance(v, (int, float)) for v in grid):
        grid = [grid]
    rows: list[list[int]] = []
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise RuntimeError("video_grid_thw rows must be [temporal,height,width]")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) <= 0:
                raise RuntimeError("video_grid_thw must contain positive integer dimensions")
            values.append(int(value))
        rows.append(values)
    if not rows:
        raise RuntimeError("video_grid_thw is empty")
    return rows


def visual_token_count_from_grid(grid: Any, merge_size: int) -> tuple[int, list[list[int]]]:
    """Count exact Qwen visual patch tokens from processor-returned video grids."""
    if isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("spatial merge size must be a positive integer")
    rows = _grid_rows(grid)
    divisor = merge_size * merge_size
    count = 0
    for temporal, height, width in rows:
        patches = temporal * height * width
        if patches % divisor:
            raise RuntimeError("video grid does not divide exactly by the configured spatial merge factor")
        count += patches // divisor
    return count, rows


def _merge_size(processor: Any, model: Any) -> int:
    candidates = []
    config = getattr(getattr(model, "config", None), "vision_config", None)
    if config is not None and getattr(config, "spatial_merge_size", None) is not None:
        candidates.append(int(config.spatial_merge_size))
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None and getattr(video_processor, "merge_size", None) is not None:
        candidates.append(int(video_processor.merge_size))
    if not candidates or any(v <= 0 for v in candidates) or len(set(candidates)) != 1:
        raise RuntimeError("model and processor spatial merge sizes are absent or inconsistent")
    return candidates[0]


def _load_qwen(checkpoint: Path, offload_dir: Path) -> tuple[Any, Any, Any, str, dict[str, str], float, int]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("the teacher stage requires its assigned CUDA device")
    torch.set_num_threads(2)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=dict(MAX_MEMORY),
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        offload_buffers=True,
        attn_implementation="sdpa",
    ).eval().requires_grad_(False)
    load_seconds = time.perf_counter() - load_started
    if getattr(model, "is_quantized", False):
        raise RuntimeError("the teacher checkpoint must remain unquantized")
    if sum(p.numel() for p in model.parameters()) != 8_292_166_656:
        raise RuntimeError("loaded official model parameter count differs from the pinned 7B checkpoint")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("teacher model is not frozen in eval mode")
    device = str(model.get_input_embeddings().weight.device)
    device_map = {str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
    if not device_map:
        raise RuntimeError("loaded model did not expose its actual device map")
    model_peak = int(torch.cuda.max_memory_allocated(0))
    return processor, model, torch, device, device_map, load_seconds, model_peak


def _one_generation(request: Mapping[str, Any], marker_video: Path, *, processor: Any, model: Any, torch: Any, device: str) -> dict[str, Any]:
    """Mirror the frozen helper's processor/generate/decode path and count actual tokens."""
    from qwen_vl_utils import process_vision_info

    messages = native.build_native_messages(str(request["query_text"]), str(marker_video))
    preprocess_started = time.perf_counter()
    rendered_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if "fps" not in video_kwargs:
        raise RuntimeError("Qwen vision processor omitted video fps metadata")
    inputs = processor(
        text=[rendered_prompt],
        images=image_inputs,
        videos=video_inputs,
        fps=video_kwargs["fps"],
        padding=True,
        return_tensors="pt",
    )
    if len(inputs["input_ids"]) != 1:
        raise RuntimeError("native teacher worker must process exactly one query")
    input_token_count = int(inputs["input_ids"].shape[1])
    input_ids_sha256 = _tensor_sha256(inputs["input_ids"])
    grids = inputs.get("video_grid_thw")
    merge_size = _merge_size(processor, model)
    visual_tokens, grid_rows = visual_token_count_from_grid(grids, merge_size)
    inputs = inputs.to(device)
    model_preprocess_s = time.perf_counter() - preprocess_started

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    generation_started = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize(0)
    model_generate_s = time.perf_counter() - generation_started
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    if len(generated_ids) != 1:
        raise RuntimeError("native teacher worker expects exactly one generated sequence")
    output_token_count = int(generated_ids[0].numel())
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if len(output_text) != 1:
        raise RuntimeError("native teacher worker expects exactly one decoded output")
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    return {
        "raw_output": output_text[0],
        "model_preprocess_s": model_preprocess_s,
        "model_generate_s": model_generate_s,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "output_truncated_at_cap": output_token_count >= MAX_NEW_TOKENS,
        "visual_token_count": visual_tokens,
        "visual_token_count_method": "sum(video_grid_thw[t,h,w]) / spatial_merge_size^2",
        "video_grid_thw": grid_rows,
        "spatial_merge_size": merge_size,
        "input_ids_sha256": input_ids_sha256,
        "rendered_prompt_sha256": _sha_bytes(rendered_prompt.encode("utf-8")),
        "vision_video_fps_metadata": _json_safe(video_kwargs["fps"]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_call_count": 1,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def run_case(case_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    records = common.read(BASE / "ROSTER_LOCK.json")
    if records.get("status") != "locked" or records.get("GT_read") is not False or records.get("count") != len(records.get("records", [])):
        raise RuntimeError("native instance roster is not complete and locked")
    roster_row = next((row for row in records["records"] if row.get("case_id") == case_id), None)
    if roster_row is None:
        raise ValueError("case is not in the fixed native-instance roster")
    if roster_row.get("GT_read") is True or roster_row.get("cohort") not in {"old8", "new8"}:
        raise RuntimeError("roster row has an invalid cohort/GT marker")

    request, request_hash = _load_locked_request(case_id, roster_row)
    accepted, marker_video, mask_data_dir, tracking = _load_accepted_tracking(case_id, request, request_hash)
    teacher_record_path = BASE / "teacher" / f"{case_id}.json"
    receipt_path = BASE / "teacher" / f"{case_id}.receipt.json"
    if teacher_record_path.exists() or receipt_path.exists():
        raise FileExistsError("teacher outputs already exist; this one-shot worker never overwrites or reprompts")

    weight_check_started = time.perf_counter()
    weight_readback = verify_local_model_receipt()
    weight_receipt_validation_s = time.perf_counter() - weight_check_started
    checkpoint = ROOT / CHECKPOINT_REL
    offload_parent = BASE / "offload"
    offload_parent.mkdir(parents=True, exist_ok=True)
    offload_dir = Path(tempfile.mkdtemp(prefix=f"{case_id}-", dir=offload_parent))
    model = processor = torch = None
    model_load_s = None
    model_device_map: dict[str, str] = {}
    model_load_peak_bytes = None
    generation: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        processor, model, torch, device, model_device_map, model_load_s, model_load_peak_bytes = _load_qwen(checkpoint, offload_dir)
        generation = _one_generation(request, marker_video, processor=processor, model=model, torch=torch, device=device)
        costs = {
            "dettrack_s": tracking["dettrack_s"],
            "filter_marker_s": tracking["filter_marker_s"],
            "model_preprocess_s": generation["model_preprocess_s"],
            "model_generate_s": generation["model_generate_s"],
            # This staged pipeline's total is the measured case-level wall:
            # accepted tracking wall plus this isolated process (load included).
            "total_s": tracking["actual_tracking_wall_s"] + (time.perf_counter() - started),
        }
        teacher_record = native.build_teacher_record(
            request,
            generation["raw_output"],
            mask_data_dir=str(mask_data_dir),
            marker_video_sha256=tracking["marker_video_sha256"],
            costs=costs,
            tracker_precision=tracking["tracker_precision"],
            preprocess_warnings=tracking["preprocess_warnings_for_native_record"],
        )
        teacher_record_path.parent.mkdir(parents=True, exist_ok=True)
        common.write(teacher_record_path, teacher_record)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        teacher_record = None
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        if torch is not None:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Offloaded tensors are process-local spill files, not experiment data.
        import shutil
        shutil.rmtree(offload_dir, ignore_errors=True)

    worker_wall = time.perf_counter() - started
    receipt: dict[str, Any] = {
        "schema": "stvg-native-instance-teacher-receipt-v1",
        "status": "failed" if failure else "complete",
        "case_id": case_id,
        "dataset": roster_row.get("dataset"),
        "cohort": roster_row.get("cohort"),
        "source_case_sha256": roster_row.get("source_case_sha256"),
        "GT_read": False,
        "query_text_in_receipt": False,
        "request_sha256": request_hash,
        "query_sha256": request["query_sha256"],
        "runtime_sha256": common.sha(BASE / "TEACHER_RUNTIME.json"),
        "tracking_acceptance_path": tracking["accepted_path"],
        "tracking_acceptance_sha256": tracking["accepted_sha256"],
        "marker_video_sha256": tracking["marker_video_sha256"],
        "mask_inventory_sha256": tracking["mask_inventory_sha256"],
        "mask_inventory_receipt_sha256": tracking["mask_inventory_receipt_sha256"],
        "source_video_sha256": request["source_video_sha256"],
        "canonical_video_sha256": request["canonical_video_sha256"],
        "canonical_pixel_sha256": request["canonical_pixel_sha256"],
        "official_model": weight_readback,
        "model_configuration": {
            "precision": "bfloat16",
            "quantization": False,
            "attention_backend": "sdpa",
            "device_map_policy": "auto",
            "max_memory": {str(k): v for k, v in MAX_MEMORY.items()},
            "actual_device_map": model_device_map,
            "parameter_count": 8_292_166_656 if model_load_s is not None else None,
            "frozen_eval": True if model_load_s is not None else None,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "costs": {
            "dettrack_s": tracking["dettrack_s"],
            "filter_marker_s": tracking["filter_marker_s"],
            "accepted_tracking_wall_s": tracking["actual_tracking_wall_s"],
            "model_receipt_validation_s": weight_receipt_validation_s,
            "model_load_s": model_load_s,
            "model_preprocess_s": generation.get("model_preprocess_s") if generation else None,
            "model_generate_s": generation.get("model_generate_s") if generation else None,
            "worker_wall_s": worker_wall,
            "combined_staged_case_wall_s": tracking["actual_tracking_wall_s"] + worker_wall,
        },
        "preprocess_warning_count": tracking["preprocess_warning_count"],
        "preprocess_warning_category_counts": tracking["preprocess_warning_category_counts"],
        "preprocess_warnings_sha256": tracking["preprocess_warnings_sha256"],
        "preprocess_warnings_deduplicated_in_teacher_record": True,
        "generation": None if generation is None else {
            key: generation[key] for key in (
                "input_token_count", "output_token_count", "output_truncated_at_cap",
                "visual_token_count", "visual_token_count_method", "video_grid_thw",
                "spatial_merge_size", "input_ids_sha256", "rendered_prompt_sha256",
                "vision_video_fps_metadata", "max_new_tokens", "generation_call_count",
                "peak_allocated_bytes", "peak_reserved_bytes",
            )
        },
        "teacher_record_path": str(teacher_record_path.relative_to(ROOT)) if teacher_record is not None else None,
        "teacher_record_sha256": common.sha(teacher_record_path) if teacher_record is not None else None,
        "parsed_status": teacher_record["parsed"]["status"] if teacher_record is not None else None,
        "one_request_one_generation": True,
        "reprompt_count": 0,
        "failure": failure,
    }
    if model_load_peak_bytes is not None:
        receipt["costs"]["model_load_peak_allocated_bytes"] = model_load_peak_bytes
    common.write(receipt_path, receipt)
    if failure:
        raise RuntimeError(f"native teacher failed for {case_id}: {failure['type']}: {failure['message']}")
    return receipt


def run_cpu_contracts() -> dict[str, Any]:
    """Synthetic math/ABI checks; does not read scope files or import torch."""
    forbidden = ("torch", "transformers", "qwen_vl_utils", "cv2", "pycocotools", "ultralytics", "sam2")
    loaded = [name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in forbidden)]
    if loaded:
        raise RuntimeError(f"CPU contract process must not have model/runtime packages loaded: {loaded}")
    grid = [[1, 8, 12], [2, 6, 10]]
    count, rows = visual_token_count_from_grid(grid, merge_size=2)
    assert rows == grid and count == (1 * 8 * 12 + 2 * 6 * 10) // 4
    for invalid in (None, [[1, 7, 7]], [[0, 8, 8]], [[1, 8]], "bad"):
        try:
            visual_token_count_from_grid(invalid, merge_size=2)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid video grid was accepted: {invalid!r}")
    try:
        visual_token_count_from_grid([[1, 8, 8]], merge_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid merge size was accepted")
    assert MAX_NEW_TOKENS == 2048 and MAX_MEMORY == {0: "20GiB", "cpu": "4GiB"}
    assert native.GROUND_TEMPLATE.count("[EVENT]") == 1
    assert native.build_native_messages("synthetic query", "/synthetic/marker.mp4")[0]["content"][1]["total_pixels"] == 2048 * 28 * 28
    return {
        "status": "pass",
        "mode": "synthetic_cpu_only",
        "model_imports": 0,
        "checkpoint_reads": 0,
        "video_reads": 0,
        "GT_reads": 0,
        "generation_calls": 0,
        "checks": [
            "processor video_grid_thw rows are strictly validated",
            "visual patch tokens use the loaded model/processor spatial merge factor",
            "invalid/empty/nondivisible video grids fail closed",
            "fixed 2048-token output cap and 20GiB/4GiB dispatch limit remain explicit",
            "official GROUND_TEMPLATE and total_pixels budget are consumed unchanged",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="one locked case ID")
    parser.add_argument("--cpu-contract", action="store_true", help="run synthetic-only CPU checks")
    args = parser.parse_args()
    if args.cpu_contract:
        if args.case is not None:
            parser.error("--cpu-contract cannot be combined with --case")
        print(json.dumps(run_cpu_contracts(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.case:
        parser.error("--case CASE is required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runtime = common.verify("teacher")
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        parser.error("TEACHER_RUNTIME must be locked and GT-free")
    sys.addaudithook(common.guard)
    receipt = run_case(args.case)
    print(json.dumps({
        "status": receipt["status"],
        "case_id": receipt["case_id"],
        "teacher_record_path": receipt["teacher_record_path"],
        "teacher_record_sha256": receipt["teacher_record_sha256"],
        "receipt_path": str((BASE / "teacher" / f"{args.case}.receipt.json").relative_to(ROOT)),
        "parsed_status": receipt["parsed_status"],
        "input_token_count": (receipt["generation"] or {}).get("input_token_count"),
        "output_token_count": (receipt["generation"] or {}).get("output_token_count"),
        "visual_token_count": (receipt["generation"] or {}).get("visual_token_count"),
        "GT_read": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
