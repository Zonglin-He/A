#!/usr/bin/env python3
"""Draft, label-free source-fit DESTA-3D episodic TTA runner.

This runner is an integration draft. GPU actions require a caller-provided
configuration and completed source-fit checkpoints. It has no label/scoring
API, does not choose TTA hyperparameters, and never writes adapted weights.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vg_tta.desta3d_v1 import ARCHITECTURES, Desta3DAdapter, reshape_merged_video_tokens
from vg_tta.desta3d_tta_v1 import (
    MODES,
    SourceFeatureStatsAccumulator,
    TTAWeights,
    candidate_update_parameters,
    compute_tta_losses,
    forward_with_features,
    validate_aligned_views,
)

PARENT = ROOT / "artifacts/desta3d_v1"
OUT = PARENT / "tta_run"
SOURCE_FIT = PARENT / "source_fit"
SOURCE_CAPTURE_LOCK = PARENT / "SOURCE_CAPTURE_LOCK.json"
INITIAL_SOURCE_INPUTS = PARENT / "SOURCE_INITIAL_INPUTS.json"
FULL_SOURCE_INPUTS = PARENT / "SOURCE_INPUTS.json"
PTD_DIR = ROOT / "checkpoints/ParallelTubeDecoding-Qwen3-VL-4B"
PTD_WEIGHTS = PTD_DIR / "model.safetensors"
GLOBAL_GPU_LOCK = ROOT / "artifacts/spatial_tta_research_v2/gpu.lock"
BUDGET_AUTHORITY = PARENT / "BUDGET_AMENDMENT_20260927_UNCAPPED.json"
MIN_FREE_BYTES = 8 * 2**30
MAX_TTA_STEPS = 3
UPDATE_GROUPS = ("input_proj", "stem", "readers")
MILD_BRIGHTNESS = 1.05
MILD_CONTRAST = 0.95


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return sha256_bytes(payload)


def tensor_sha256(value: Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    return sha256_bytes(str(cpu.dtype).encode() + repr(tuple(cpu.shape)).encode()
                        + cpu.view(torch.uint8).numpy().tobytes())


def adapter_sha256(adapter: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(adapter.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(repr(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json_once(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed JSON: {path}")
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + "\n")
    os.replace(tmp, path)


def scan_nested_gpu_receipts(root: str | Path) -> tuple[float, list[dict[str, Any]]]:
    """Sum every nested receipts/worker_receipts seconds field; fail closed."""
    root = Path(root)
    rows = []
    total = 0.0
    if not root.exists():
        return 0.0, rows
    for path in sorted(root.rglob("*.json")):
        if not any(part in {"receipts", "worker_receipts"} for part in path.parts):
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot safely account GPU receipt {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GPU receipt must be a JSON object: {path}")
        if "seconds" not in payload:
            raise RuntimeError(f"GPU receipt is missing required seconds field: {path}")
        value = payload["seconds"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0):
            raise RuntimeError(f"invalid seconds field in nested GPU receipt: {path}")
        total += float(value)
        rows.append({"path": str(path), "seconds": float(value), "stage": payload.get("stage"),
                     "status": payload.get("status")})
    return total, rows


def atomic_torch_save_once(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed prediction: {path}")
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(k): cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return value


def digest_numpy_frames(frames: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(frames).tobytes())


def make_mild_photometric_view(
    frames: np.ndarray,
    *,
    brightness: float = MILD_BRIGHTNESS,
    contrast: float = MILD_CONTRAST,
) -> np.ndarray:
    """Return one mild RGB view: contrast around each frame mean, then brightness.

    The exact order is ``1.05 * (mean + 0.95 * (pixel - mean))`` followed by
    round-to-nearest and uint8 clamp. No clean target frame or second video is
    read; the view is made from the same observed input array.
    """
    x = np.asarray(frames)
    if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8 or x.shape[0] < 1:
        raise ValueError("frames must be nonempty RGB uint8 [T,H,W,3]")
    if not math.isfinite(float(brightness)) or not math.isfinite(float(contrast)):
        raise ValueError("photometric factors must be finite")
    f = x.astype(np.float32)
    mean = f.mean(axis=(1, 2), keepdims=True)
    changed = (mean + float(contrast) * (f - mean)) * float(brightness)
    return np.rint(changed).clip(0, 255).astype(np.uint8)


def freeze_stock_model(model: nn.Module) -> None:
    """Set the PTD backbone to deterministic eval and freeze every parameter."""
    model.eval()
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("frozen PTD model still has trainable parameters")


def adapter_sha256_from_state(state: Mapping[str, Tensor]) -> str:
    """Match the source-fit state-dict digest without constructing a module."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, Tensor):
            raise ValueError(f"adapter state entry is not a tensor: {name}")
        tensor = value.detach().contiguous().cpu()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_best_checkpoint_payload(
    payload: Mapping[str, Any], architecture: str, expected_identity: Mapping[str, str],
    fit_complete: Mapping[str, Any],
) -> tuple[str, int, float]:
    """Validate the final source-fit BEST contract before loading it for TTA."""
    if payload.get("architecture") != architecture or not isinstance(payload.get("adapter"), dict):
        raise ValueError("unexpected source-fit BEST checkpoint payload")
    if payload.get("source_fit_identity") != dict(expected_identity):
        raise ValueError("BEST source_fit_identity does not match current lock/seals")
    state_sha = adapter_sha256_from_state(payload["adapter"])
    if payload.get("adapter_sha256") != state_sha:
        raise ValueError("BEST adapter_sha256 does not match serialized adapter state")
    if fit_complete.get("best_adapter_sha256", {}).get(architecture) != state_sha:
        raise ValueError("BEST adapter hash does not match FIT_COMPLETE")
    try:
        best_epoch = int(payload["best_epoch"])
        score = float(payload["source_parent_macro_vIoU"])
        fit_arch = fit_complete["architectures"][architecture]
        fit_epoch = int(fit_arch["best_epoch"])
        fit_score = float(fit_arch["source_parent_macro_vIoU"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BEST metadata is incomplete or malformed") from exc
    if best_epoch != fit_epoch or not math.isfinite(score) or not math.isfinite(fit_score) or score != fit_score:
        raise ValueError("BEST epoch/score does not match FIT_COMPLETE")
    return state_sha, best_epoch, score


def validate_source_fit_query_manifest(query_manifest: Mapping[str, Any],
                                       train_config: Mapping[str, Any]) -> int:
    """Check the real source-fit 618-train + 198-validation query seal."""
    split = train_config.get("source_split")
    if not isinstance(split, Mapping):
        raise ValueError("source-fit TRAIN_CONFIG has no source_split contract")
    try:
        expected_count = int(split["train_queries"]) + int(split["validation_queries"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source-fit TRAIN_CONFIG query counts are malformed") from exc
    files = query_manifest.get("files")
    keys = [str(row.get("key")) for row in files if isinstance(row, Mapping)] if isinstance(files, list) else []
    if (query_manifest.get("status") != "complete_gt_free_exact_query_capture"
            or query_manifest.get("GT_read") is not False
            or int(query_manifest.get("count", -1)) != expected_count
            or not isinstance(files, list) or len(files) != expected_count
            or len(keys) != expected_count or len(set(keys)) != expected_count):
        raise ValueError("source-fit clean-query seal is not a complete GT-free configured source capture")
    return expected_count


def set_adapter_groups(adapter: Desta3DAdapter, update: bool) -> dict[str, int]:
    groups = {
        "input_proj": tuple(adapter.input_proj.parameters()),
        "stem": tuple(adapter.stem.parameters()) if hasattr(adapter, "stem") else (),
        "readers": tuple(p for name in ("factorized_reader", "shared_reader", "short_reader", "long_reader")
                         for module in (getattr(adapter, name, None),) if module is not None
                         for p in module.parameters()),
        "query_proj": tuple(adapter.query_proj.parameters()),
        "film": tuple(adapter.film.parameters()),
        "out_proj": tuple(adapter.out_proj.parameters()),
        "scalar_heads": tuple(adapter.referent_head.parameters()) + tuple(adapter.event_head.parameters()),
    }
    for name, params in groups.items():
        active = update and name in UPDATE_GROUPS
        for p in params:
            p.requires_grad_(active)
    adapter.eval()
    if update:
        selected = set(id(p) for p in candidate_update_parameters(adapter))
        permitted = {id(p) for name in UPDATE_GROUPS for p in groups[name]}
        if not selected or not selected.issubset(permitted):
            raise RuntimeError("adapter update selection escaped input_proj/stem/readers")
    else:
        if any(p.requires_grad for p in adapter.parameters()):
            raise RuntimeError("no-TTA source-fit baseline adapter is not fully frozen")
    return {name: sum(p.numel() for p in params if p.requires_grad) for name, params in groups.items()}


def load_adapter_checkpoint(architecture: str) -> tuple[Desta3DAdapter, dict[str, Any], Path, str]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported source-fit architecture: {architecture}")
    fit_complete_path = SOURCE_FIT / "FIT_COMPLETE.json"
    train_lock_path = SOURCE_FIT / "TRAIN_LOCK.json"
    train_config_path = SOURCE_FIT / "TRAIN_CONFIG.json"
    query_manifest_path = SOURCE_FIT / "QUERY_CLEAN_SEAL.json"
    train_records_path = SOURCE_FIT / "SOURCE_TRAIN_RECORDS.json"
    checkpoint_path = SOURCE_FIT / "checkpoints" / f"{architecture}_BEST.pt"
    for path in (fit_complete_path, train_lock_path, train_config_path,
                 query_manifest_path, train_records_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed source-fit artifact is required: {path}")
    fit_complete = read_json(fit_complete_path)
    if fit_complete.get("status") != "completed_source_only_fit_and_clean_validation_selection":
        raise ValueError("source-fit checkpoint is not marked completed")
    checkpoint_sha = sha256_file(checkpoint_path)
    if fit_complete.get("best_checkpoint_hashes", {}).get(architecture) != checkpoint_sha:
        raise ValueError("BEST checkpoint hash does not match FIT_COMPLETE.json")
    train_lock = read_json(train_lock_path)
    train_config = read_json(train_config_path)
    if train_lock.get("config_sha256") != sha256_file(train_config_path):
        raise ValueError("source-fit TRAIN_CONFIG changed after its lock")
    if fit_complete.get("train_lock_sha256") != sha256_file(train_lock_path):
        raise ValueError("FIT_COMPLETE source-fit TRAIN_LOCK hash mismatch")
    train_records_sha = sha256_file(train_records_path)  # hash-only provenance check; never parse source labels here
    if fit_complete.get("train_records_sha256") != train_records_sha:
        raise ValueError("FIT_COMPLETE source training-record hash mismatch")
    pinned_train_records_sha = train_lock.get("pins", {}).get(str(train_records_path))
    if pinned_train_records_sha != train_records_sha:
        raise ValueError("source training-record hash does not match TRAIN_LOCK pin")
    query_manifest_sha = sha256_file(query_manifest_path)
    if fit_complete.get("query_cache_sha256") != query_manifest_sha:
        raise ValueError("FIT_COMPLETE clean-query manifest hash mismatch")
    query_manifest = read_json(query_manifest_path)
    query_count = validate_source_fit_query_manifest(query_manifest, train_config)
    lock_script_sha = train_lock.get("script_sha256")
    source_fit_script = ROOT / "scripts/desta3d_source_fit_v1.py"
    if lock_script_sha and source_fit_script.is_file() and sha256_file(source_fit_script) != lock_script_sha:
        raise ValueError("source-fit code differs from its TRAIN_LOCK")
    locked_ptd_receipt_sha = train_lock.get("pins", {}).get(str(PTD_DIR / "OFFICIAL_RECEIPT.json"))
    if not locked_ptd_receipt_sha:
        raise ValueError("source-fit TRAIN_LOCK has no official PTD checkpoint receipt pin")
    if locked_ptd_receipt_sha != sha256_file(PTD_DIR / "OFFICIAL_RECEIPT.json"):
        raise ValueError("source-fit and current PTD official checkpoint receipts differ")
    source_inputs_sha = train_lock.get("pins", {}).get(str(FULL_SOURCE_INPUTS))
    if not source_inputs_sha:
        raise ValueError("source-fit TRAIN_LOCK has no SOURCE_INPUTS manifest pin")
    arch_cfg = train_config.get("architectures", {}).get(architecture)
    if not isinstance(arch_cfg, dict) or "hidden_dim" not in arch_cfg:
        raise ValueError(f"source-fit architecture config is missing for {architecture}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_identity = {
        "train_lock_sha256": sha256_file(train_lock_path),
        "query_manifest_sha256": query_manifest_sha,
        "train_records_sha256": train_records_sha,
    }
    state_sha, best_epoch, source_score = validate_best_checkpoint_payload(
        payload, architecture, expected_identity, fit_complete)
    adapter = Desta3DAdapter(in_channels=2560, query_dim=2560,
                             hidden_dim=int(arch_cfg["hidden_dim"]), architecture=architecture)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    if adapter_sha256(adapter) != state_sha:
        raise ValueError("loaded BEST adapter state digest changed")
    metadata = {
        "architecture": architecture,
        "hidden_dim": int(arch_cfg["hidden_dim"]),
        "best_epoch": best_epoch,
        "source_parent_macro_vIoU": source_score,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "fit_complete_sha256": sha256_file(fit_complete_path),
        "train_lock_sha256": sha256_file(train_lock_path),
        "train_config_sha256": sha256_file(train_config_path),
        "query_manifest_sha256": query_manifest_sha,
        "query_manifest_count": query_count,
        "train_records_sha256": train_records_sha,
        "source_inputs_sha256": source_inputs_sha,
        "adapter_sha256": state_sha,
        "ptd_train_lock_identity": train_lock.get("pins", {}).get(str(PTD_DIR / "OFFICIAL_RECEIPT.json")),
    }
    if not math.isfinite(metadata["source_parent_macro_vIoU"]):
        raise ValueError("source-fit score metadata is not finite")
    return adapter, metadata, checkpoint_path, checkpoint_sha


def verify_ptd_identity() -> dict[str, Any]:
    receipt_path = PTD_DIR / "OFFICIAL_RECEIPT.json"
    receipt = read_json(receipt_path)
    if str(receipt.get("path")) != str(PTD_WEIGHTS) or not PTD_WEIGHTS.is_file():
        raise ValueError("official frozen PTD checkpoint path does not match its receipt")
    actual = sha256_file(PTD_WEIGHTS)
    if actual != receipt.get("sha256"):
        raise ValueError("official PTD-4B weights differ from their receipt")
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
            "weights_path": str(PTD_WEIGHTS), "weights_sha256": actual,
            "revision": receipt.get("revision")}


def _processor_prompt_kwargs(inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    keep = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos",
            "video_grid_thw", "second_per_grid_ts"}
    return {k: v for k, v in inputs.items() if k in keep}


def capture_prompt_fields_once(
    model: nn.Module,
    inputs: Mapping[str, Tensor],
    frame_ids: Sequence[int],
    fps: float,
) -> dict[str, Any]:
    """One frozen stock-prompt forward captures raw merger THW and exact query.

    Query pooling mirrors ``capture_exact_query`` from the source pilot. The
    merger and language hooks observe the same no-response prompt forward; no
    adapted token, generated response, or GT enters either representation.
    """
    core = model.model
    visual = core.visual
    language = core.language_model
    model_inputs = _processor_prompt_kwargs(inputs)
    if "input_ids" not in model_inputs or "attention_mask" not in model_inputs:
        raise ValueError("PTD stock prompt inputs lack input_ids/attention_mask")
    grid_thw = inputs["video_grid_thw"][0]
    n_frames = int(grid_thw[0])
    if n_frames != len(frame_ids) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("prompt grid, exact frame ids and fps are inconsistent")
    input_ids = model_inputs["input_ids"]
    video_token_id = int(model.config.video_token_id)
    video_positions = input_ids[0].eq(video_token_id).nonzero(as_tuple=False).flatten()
    if video_positions.numel() == 0:
        raise ValueError("video token missing from frozen PTD prompt")
    query_mask = torch.arange(input_ids.shape[1], device=input_ids.device) > video_positions[-1]
    query_mask &= model_inputs["attention_mask"][0].bool()
    if not torch.any(query_mask):
        raise ValueError("stock prompt has no instruction tokens after the final video token")

    captured: dict[str, Any] = {}
    handles = []

    def merger_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("visual.merger output is not a tensor")
        if "merger_tokens" in captured:
            raise RuntimeError("visual.merger was called more than once in a single-view prefill")
        captured["merger_tokens"] = output.detach()

    def language_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, Tensor) or hidden.shape[1] != input_ids.shape[1]:
            raise RuntimeError("language prefill hidden states do not match prompt token positions")
        captured["query"] = hidden[0, query_mask].float().mean(0).detach()

    handles.append(visual.merger.register_forward_hook(merger_hook))
    handles.append(language.register_forward_hook(language_hook))
    try:
        with torch.no_grad():
            core(**model_inputs, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()
    tokens = captured.get("merger_tokens")
    query = captured.get("query")
    if not isinstance(tokens, Tensor) or not isinstance(query, Tensor):
        raise RuntimeError("one-pass prompt prefill failed to capture merger tokens and query")
    if query.numel() != 2560 or not torch.isfinite(query).all():
        raise ValueError(f"expected a finite 2560-D stock query, got {tuple(query.shape)}")
    grid = reshape_merged_video_tokens(tokens.float(), grid_thw)
    if grid.shape[0] != 1 or grid.shape[1] != len(frame_ids) or grid.shape[-1] != 2560:
        raise ValueError(f"unexpected merger THW/query dimensions: {tuple(grid.shape)}")
    return {
        "visual_grid": grid.detach(),
        "query": query.reshape(1, 2560).detach(),
        "frame_times": torch.tensor(frame_ids, dtype=torch.float32, device=grid.device)[None] / float(fps),
        "video_grid_thw": torch.as_tensor(grid_thw).detach().cpu().tolist(),
        "query_definition": "exact stock PTD prompt-prefill hidden mean after final video token; no response/GT",
    }


def make_view_inputs(
    row: Mapping[str, Any],
    processor: Any,
    condition: str,
    *,
    frames_for: Callable[..., tuple[np.ndarray, Sequence[int]]],
    inputs_for: Callable[..., tuple[dict[str, Tensor], Mapping[str, Any]]],
    guard: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if guard is not None:
        guard("before_decode_observed_input")
    frames, frame_ids = frames_for(row, condition)
    frame_ids = [int(v) for v in frame_ids]
    if frame_ids != [int(v) for v in row["input"]["frame_ids"]]:
        raise ValueError("decoded physical frame IDs differ from the input manifest")
    mild = make_mild_photometric_view(frames)
    if guard is not None:
        guard("before_observed_view_preprocess")
    inputs_base, preprocess_base = inputs_for(row, processor, frames)
    if guard is not None:
        guard("before_mild_view_preprocess")
    inputs_mild, preprocess_mild = inputs_for(row, processor, mild)
    grid_a = inputs_base["video_grid_thw"].detach().cpu()
    grid_b = inputs_mild["video_grid_thw"].detach().cpu()
    if not torch.equal(grid_a, grid_b):
        raise ValueError("photometric view changed the PTD video grid")
    base_sha, mild_sha = digest_numpy_frames(frames), digest_numpy_frames(mild)
    if preprocess_base.get("pixel_sha") != base_sha or preprocess_mild.get("pixel_sha") != mild_sha:
        raise ValueError("inputs_for pixel SHA does not match exact RGB bytes")
    ids_sha = json_sha256(frame_ids)
    common = {"frame_ids": frame_ids, "frame_ids_sha256": ids_sha,
              "fps": float(row["input"]["fps"]), "video_grid_thw": grid_a.tolist(),
              "mild_transform": {"brightness": MILD_BRIGHTNESS, "contrast_about_per_frame_mean": MILD_CONTRAST,
                                 "order": "contrast_about_mean_then_brightness", "round": "numpy.rint", "clamp": [0, 255]}}
    def view_entry(name: str, input_values: dict[str, Tensor], preprocess: Mapping[str, Any], pixel_sha: str) -> dict[str, Any]:
        in_sig = {
            "pixel_sha256": pixel_sha,
            "frame_ids": frame_ids,
            "preprocess": cpu_copy(dict(preprocess)),
            "input_ids_sha256": tensor_sha256(input_values["input_ids"]),
            "video_grid_thw": input_values["video_grid_thw"].detach().cpu().tolist(),
        }
        return {"name": name, "inputs": input_values, "preprocess": dict(preprocess),
                "frame_ids": list(frame_ids),
                "video_grid_thw": input_values["video_grid_thw"].detach().cpu().tolist(),
                "pixel_sha256": pixel_sha, "preprocess_sha256": json_sha256(in_sig),
                "input_signature_sha256": json_sha256(in_sig)}
    return {
        **common,
        "base": view_entry("observed", inputs_base, preprocess_base, base_sha),
        "mild": view_entry("mild_photometric", inputs_mild, preprocess_mild, mild_sha),
        "frame_array_shape": list(frames.shape),
        "frame_dtype": str(frames.dtype),
    }


def assert_same_physical_grid(view_a: Mapping[str, Any], view_b: Mapping[str, Any]) -> None:
    if list(view_a["frame_ids"]) != list(view_b["frame_ids"]):
        raise ValueError("views use different physical frame IDs")
    if view_a["video_grid_thw"] != view_b["video_grid_thw"]:
        raise ValueError("views use different PTD video grids")
    if tuple(view_a["capture"]["visual_grid"].shape) != tuple(view_b["capture"]["visual_grid"].shape):
        raise ValueError("views have different actual merger THW shapes")
    features_a, features_b = view_a.get("features"), view_b.get("features")
    if (features_a is None) != (features_b is None):
        raise ValueError("views must either both carry precomputed features or neither")
    if features_a is not None:
        validate_aligned_views(features_a, features_b)
    else:
        times_a = view_a["capture"]["frame_times"].detach().cpu()
        times_b = view_b["capture"]["frame_times"].detach().cpu()
        if not torch.equal(times_a, times_b):
            raise ValueError("views use different physical times")


def _captured_features(adapter: Desta3DAdapter, view: Mapping[str, Any]) -> dict[str, Any]:
    cap = view["capture"]
    return forward_with_features(adapter, cap["visual_grid"], cap["query"], frame_times=cap["frame_times"])


def source_state_cpu(adapter: Desta3DAdapter) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}


def reset_episode_adapter(adapter: Desta3DAdapter, source_state: Mapping[str, Tensor]) -> str:
    adapter.load_state_dict(source_state, strict=True)
    adapter.zero_grad(set_to_none=True)
    set_adapter_groups(adapter, update=True)
    return adapter_sha256(adapter)


def _optimizer_from_config(params: Sequence[nn.Parameter], cfg: Mapping[str, Any]):
    name = cfg.get("name")
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer.name must be explicitly set to AdamW or Adam")


def _require_finite_active_gradients(params: Sequence[nn.Parameter], step_index: int) -> None:
    nonfinite = [i for i, p in enumerate(params)
                 if p.requires_grad and p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    if nonfinite:
        raise FloatingPointError(
            f"nonfinite active adapter gradients at fixed step {step_index}; parameter indices={nonfinite[:16]}"
        )


def _require_finite_active_parameters(params: Sequence[nn.Parameter], step_index: int) -> None:
    nonfinite = [i for i, p in enumerate(params)
                 if p.requires_grad and not bool(torch.isfinite(p.detach()).all())]
    if nonfinite:
        raise FloatingPointError(
            f"nonfinite updated adapter parameters at fixed step {step_index}; parameter indices={nonfinite[:16]}"
        )


def validate_tta_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"architecture", "mode", "steps", "seed", "optimizer", "weights"}
    normalized = {"update_groups", "state_selection"}
    if set(config) == required | normalized:
        if (config["update_groups"] != list(UPDATE_GROUPS)
                or config["state_selection"] != "fixed configured terminal step; no loss- or GT-based best-step selection"):
            raise ValueError("normalized TTA config carries unexpected update/state-selection contract")
        config = {k: config[k] for k in required}
    elif set(config) != required:
        raise ValueError(f"run config must contain exactly {sorted(required)}")
    if config["architecture"] not in ARCHITECTURES or config["mode"] not in MODES:
        raise ValueError("run config has unsupported architecture/mode")
    if type(config["steps"]) is not int:
        raise ValueError("TTA steps must be an explicitly supplied integer")
    steps = config["steps"]
    if steps < 1 or steps > MAX_TTA_STEPS:
        raise ValueError(f"TTA steps must be explicitly 1..{MAX_TTA_STEPS}")
    if type(config["seed"]) is not int:
        raise ValueError("TTA seed must be an explicitly supplied integer")
    seed = config["seed"]
    opt = config["optimizer"]
    if set(opt) != {"name", "lr", "weight_decay"}:
        raise ValueError("optimizer config must explicitly give name, lr and weight_decay")
    if any(isinstance(opt[k], bool) or not isinstance(opt[k], (int, float)) for k in ("lr", "weight_decay")):
        raise ValueError("optimizer lr and weight_decay must be JSON numbers")
    lr, weight_decay = float(opt["lr"]), float(opt["weight_decay"])
    if (opt["name"] not in ("AdamW", "Adam") or not math.isfinite(lr)
            or not math.isfinite(weight_decay) or lr <= 0 or weight_decay < 0):
        raise ValueError("invalid explicitly supplied optimizer values")
    expected_weights = {"feature_alignment", "latent_consistency", "referent_consistency",
                        "event_consistency", "asymmetric_joint"}
    if set(config["weights"]) != expected_weights:
        raise ValueError(f"weights must explicitly supply exactly {sorted(expected_weights)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in config["weights"].values()):
        raise ValueError("loss weights must be JSON numbers")
    weights = {k: float(v) for k, v in config["weights"].items()}
    TTAWeights(**weights)
    return {"architecture": str(config["architecture"]), "mode": str(config["mode"]),
            "steps": steps, "seed": seed,
            "optimizer": {"name": str(opt["name"]), "lr": lr,
                          "weight_decay": weight_decay},
            "weights": weights,
            "update_groups": list(UPDATE_GROUPS),
            "state_selection": "fixed configured terminal step; no loss- or GT-based best-step selection"}


def tta_weight_object(config: Mapping[str, Any]) -> TTAWeights:
    return TTAWeights(**{k: float(v) for k, v in config["weights"].items()})


def perform_fixed_tta_steps(
    adapter: Desta3DAdapter,
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    source_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    guard: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly the configured 1–3 steps and return a digest/loss trace."""
    cfg = validate_tta_config(config)
    if adapter.architecture != cfg["architecture"]:
        raise ValueError("run config and loaded source-fit adapter architectures differ")
    assert_same_physical_grid(view_a, view_b)
    params = candidate_update_parameters(adapter, UPDATE_GROUPS)
    optimizer = _optimizer_from_config(params, cfg["optimizer"])
    weights = tta_weight_object(cfg)
    trace: list[dict[str, Any]] = []
    for step_index in range(1, cfg["steps"] + 1):
        if guard is not None:
            guard(f"tta_update_{step_index}")
        optimizer.zero_grad(set_to_none=True)
        out_a = _captured_features(adapter, view_a)
        out_b = _captured_features(adapter, view_b)
        losses = compute_tta_losses(out_a, out_b, source_stats, mode=cfg["mode"], weights=weights)
        loss = losses["total"]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"nonfinite unsupervised TTA loss at fixed step {step_index}")
        loss.backward()
        gradient_l2: dict[str, float] = {}
        for name, module in (("input_proj", adapter.input_proj),
                             ("stem", adapter.stem if hasattr(adapter, "stem") else None),
                             ("readers", nn.ModuleList([getattr(adapter, n) for n in
                                                        ("factorized_reader", "shared_reader", "short_reader", "long_reader")
                                                        if getattr(adapter, n, None) is not None]))):
            if module is None:
                gradient_l2[name] = 0.0
                continue
            sq = sum(float(p.grad.detach().float().square().sum()) for p in module.parameters()
                     if p.requires_grad and p.grad is not None)
            gradient_l2[name] = math.sqrt(sq)
        frozen_gradients = any(p.grad is not None for name in ("query_proj", "film", "out_proj", "referent_head", "event_head")
                               for p in getattr(adapter, name).parameters())
        if frozen_gradients:
            raise RuntimeError("frozen adapter heads/query projection unexpectedly received gradients")
        active_params = [p for p in params if p.requires_grad]
        _require_finite_active_gradients(active_params, step_index)
        if any(not math.isfinite(value) for value in gradient_l2.values()):
            raise FloatingPointError(f"nonfinite active gradient norm at fixed step {step_index}")
        optimizer.step()
        _require_finite_active_parameters(active_params, step_index)
        trace.append({
            "step": step_index,
            "total_loss_before_update": float(loss.detach().cpu()),
            "components_before_update": {k: float(v.detach().cpu()) for k, v in losses["components"].items()},
            "active_weights": dict(losses["weights"]),
            "gradient_l2_by_group": gradient_l2,
            "frozen_heads_no_grad": True,
            "adapter_state_sha256_after_update": adapter_sha256(adapter),
        })
    optimizer.zero_grad(set_to_none=True)
    return trace


def inference_prediction_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("GT_used") is True:
        raise ValueError("PTD inference reports GT use")
    positions = result.get("positions")
    if positions is None:
        positions = []
    boxes = result.get("boxes")
    if boxes is None:
        boxes = torch.empty((0, 4), dtype=torch.float32)
    valid = result.get("geometry_valid")
    if valid is None:
        valid = torch.empty((0,), dtype=torch.bool)
    return {
        "positions": [int(v) for v in positions],
        "boxes_cxcywh": torch.as_tensor(boxes).detach().float().cpu(),
        "geometry_valid": torch.as_tensor(valid).detach().bool().cpu(),
        "interval": None if result.get("interval") is None else [int(v) for v in result["interval"]],
        "format_ok": bool(result.get("format_ok", False)),
        "completion": str(result.get("completion", "")),
        "GT_used": False,
    }


def full_ptd_replay_with_merger_adapter(
    model: nn.Module,
    processor: Any,
    inputs: Mapping[str, Tensor],
    adapter: Desta3DAdapter,
    query: Tensor,
    frame_ids: Sequence[int],
    fps: float,
    *,
    infer_fn: Callable[..., Mapping[str, Any]] | None = None,
    expected_visual_grid: Tensor | None = None,
) -> dict[str, Any]:
    """Run stock full PTD generation with one temporary visual.merger hook."""
    if infer_fn is None:
        from scripts.ptd_spatial_adapter_ab_v1 import infer as infer_fn
    grid_thw = inputs["video_grid_thw"][0]
    frame_times = torch.tensor(frame_ids, device=query.device, dtype=torch.float32)[None] / float(fps)
    capture: dict[str, Any] = {"calls": 0}

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        capture["calls"] += 1
        if capture["calls"] != 1:
            raise RuntimeError("visual.merger hook fired more than once during a single full PTD replay")
        visual_grid = reshape_merged_video_tokens(output.float(), grid_thw)
        if visual_grid.shape[1] != len(frame_ids):
            raise ValueError("full replay merger grid does not match exact sampled frame IDs")
        if expected_visual_grid is not None:
            expected = expected_visual_grid.to(device=visual_grid.device, dtype=visual_grid.dtype)
            if tuple(expected.shape) != tuple(visual_grid.shape) or not torch.allclose(visual_grid, expected, rtol=1e-3, atol=1e-3):
                raise ValueError("full replay visual.merger differs from the captured same-view prompt grid")
        adapted = adapter(visual_grid, query.to(device=visual_grid.device, dtype=torch.float32),
                          frame_times=frame_times.to(visual_grid.device))
        capture["updated_tokens"] = adapted["updated_tokens"].detach()
        return adapted["updated_tokens"].reshape_as(output).to(output.dtype)

    merger = model.model.visual.merger
    handle = merger.register_forward_hook(hook)
    try:
        result = infer_fn(model, processor, dict(inputs))
    finally:
        handle.remove()
    if capture["calls"] != 1:
        raise RuntimeError(f"full PTD generation called visual.merger {capture['calls']} times; expected once")
    summary = inference_prediction_summary(result)
    summary["merger_hook_calls"] = int(capture["calls"])
    summary["updated_tokens_sha256"] = tensor_sha256(capture["updated_tokens"])
    return summary


def sanitized_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Pass only original, non-label input metadata to the PTD reader."""
    if "key" not in row or "source" not in row or not isinstance(row.get("input"), Mapping):
        raise ValueError("input manifest row needs key/source/input")
    input_fields = ("caption", "end_frame", "fps", "frame_count", "frame_ids", "height", "index",
                    "kind", "original_video_id", "source", "start_frame", "video_id", "video_path",
                    "video_sha256", "width")
    inp = {k: row["input"][k] for k in input_fields if k in row["input"]}
    needed = {"caption", "fps", "frame_count", "frame_ids", "video_path", "video_sha256", "width", "height"}
    if not needed.issubset(inp):
        raise ValueError(f"input manifest is missing required no-label fields: {sorted(needed-set(inp))}")
    ids = [int(x) for x in inp["frame_ids"]]
    if not ids or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("manifest frame_ids must be nonempty and strictly increasing")
    inp["frame_ids"] = ids
    return {"key": str(row["key"]), "source": str(row["source"]),
            "split": str(row.get("split", "unspecified")),
            "cohort": str(row.get("cohort", row.get("domain", "unspecified"))), "input": inp}


def load_input_manifest(path: str | Path, split: str) -> tuple[list[dict[str, Any]], str]:
    path = Path(path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("input manifest must be inside the project workspace")
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("input manifest must be a JSON list of unlabeled inputs")
    rows = [sanitized_input_row(r) for r in payload if str(r.get("split", "unspecified")) == split]
    if not rows:
        raise ValueError(f"input manifest has no rows for split={split!r}")
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("input manifest has duplicate query keys in selected split")
    return rows, sha256_file(path)


def source_statistics_rows(input_manifest: str | Path, split: str,
                           source_fit_metadata: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Load an explicitly selected, immutable source-train image-only pool."""
    if split != "train":
        raise ValueError("source statistics require explicit --split train; validation/target rows are refused")
    path = Path(input_manifest).resolve()
    full_path = FULL_SOURCE_INPUTS.resolve()
    initial_path = INITIAL_SOURCE_INPUTS.resolve()
    actual = sha256_file(path) if path.is_file() else None
    if path == full_path:
        expected = source_fit_metadata.get("source_inputs_sha256")
        pool_name, expected_queries = "SOURCE_INPUTS_full_source_train", 618
        if not expected or actual != expected:
            raise ValueError("full SOURCE_INPUTS differs from the completed source-fit TRAIN_LOCK pin")
    elif path == initial_path:
        lock = read_json(SOURCE_CAPTURE_LOCK)
        expected = lock.get("pins", {}).get(str(INITIAL_SOURCE_INPUTS))
        pool_name, expected_queries = "SOURCE_INITIAL_INPUTS_legacy_initial_subset", 95
        if not expected or actual != expected:
            raise ValueError("SOURCE_INITIAL_INPUTS differs from the existing E2 input lock")
    else:
        raise ValueError("choose the locked SOURCE_INPUTS.json full train pool or the locked SOURCE_INITIAL_INPUTS.json legacy subset explicitly")

    rows, manifest_sha = load_input_manifest(path, split)
    pool_meta = summarize_source_statistics_pool(rows, path, manifest_sha,
                                                 pool_name=pool_name,
                                                 expected_queries=expected_queries)
    return rows, manifest_sha, pool_meta


def summarize_source_statistics_pool(rows: Sequence[Mapping[str, Any]], input_manifest: str | Path,
                                     manifest_sha: str, *, pool_name: str,
                                     expected_queries: int) -> dict[str, Any]:
    """Validate exact source-train denominators and parent-varying visual inputs."""
    parents = {r["source"] for r in rows}
    videos = {r["input"]["video_sha256"] for r in rows}
    if len(rows) != expected_queries or len(parents) != 95 or len(videos) != 95:
        raise ValueError(f"unexpected source-statistics pool size: queries={len(rows)} parents={len(parents)} media={len(videos)}; expected queries={expected_queries}, parents=95, media=95")
    signatures_by_parent: dict[str, set[str]] = {}
    for row in rows:
        inp = row["input"]
        signature = json_sha256({"video_sha256": inp["video_sha256"], "frame_ids": inp["frame_ids"],
                                 "fps": float(inp["fps"]), "width": int(inp["width"]),
                                 "height": int(inp["height"])})
        signatures_by_parent.setdefault(row["source"], set()).add(signature)
    varying_parents = sum(len(signatures) > 1 for signatures in signatures_by_parent.values())
    return {
        "pool_id": pool_name, "input_manifest": str(Path(input_manifest).resolve()), "input_manifest_sha256": manifest_sha,
        "split": "train", "queries": len(rows), "parents": len(parents),
        "unique_video_sha256": len(videos),
        "parents_with_query_varying_visual_signature": varying_parents,
        "visual_signature_fields": ["video_sha256", "frame_ids", "fps", "width", "height"],
        "aggregation_unit": "each query contributes its own query-averaged channel moments; equal queries within parent then equal parents",
        "legacy_95_not_equated_to_full_618_mean": expected_queries == 95,
    }


def validate_source_stats(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if OUT.resolve() not in path.parents:
        raise ValueError("source statistics must be inside this runner's tta_run artifact directory")
    if not path.is_file():
        raise FileNotFoundError(f"source-train statistics JSON is required: {path}")
    meta_path = path.with_suffix(path.suffix + ".receipt.json")
    if not meta_path.is_file():
        raise FileNotFoundError("source statistics lack their immutable receipt")
    meta = read_json(meta_path)
    if meta.get("stats_sha256") != sha256_file(path) or meta.get("source_split") != "train" or meta.get("gt_read") is not False:
        raise ValueError("source stats receipt/hash/split/GT marker failed")
    stats = read_json(path)
    if stats.get("source_split") != "train" or stats.get("gt_read") is not False:
        raise ValueError("TTA requires source-train moments marked gt_read=false")
    pool = meta.get("pool", {})
    if (pool.get("queries") not in (95, 618) or pool.get("parents") != 95
            or pool.get("unique_video_sha256") != 95 or pool.get("split") != "train"
            or meta.get("no_target_or_source_validation_rows_used") is not True):
        raise ValueError("TTA moments must come from an explicitly locked 95-query or full 618-query source-train pool")
    if meta.get("input_manifest_sha256") != pool.get("input_manifest_sha256"):
        raise ValueError("source-statistics receipt does not bind the selected input manifest")
    if meta.get("target_data_read") is not False:
        raise ValueError("source statistics receipt reports target data access")
    for branch in ("referent", "event"):
        try:
            global_stats = stats["parents"][branch]["__global__"]
            mean = np.asarray(global_stats["mean"], dtype=np.float64)
            std = np.asarray(global_stats["std"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed source moments for {branch}") from exc
        if mean.ndim != 1 or std.shape != mean.shape or not mean.size or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"source moments for {branch} must be finite channel vectors")
        global_stats = stats["parents"][branch]["__global__"]
        if global_stats.get("query_count") != pool["queries"] or global_stats.get("parent_count") != pool["parents"]:
            raise ValueError(f"source moments for {branch} do not match receipt query/parent counts")
    return {"stats": stats, "stats_sha256": meta["stats_sha256"], "receipt": meta,
            "receipt_sha256": sha256_file(meta_path)}


def capture_both_views(
    model: nn.Module,
    processor: Any,
    row: Mapping[str, Any],
    condition: str,
    *,
    frames_for: Callable[..., tuple[np.ndarray, Sequence[int]]],
    inputs_for: Callable[..., tuple[dict[str, Tensor], Mapping[str, Any]]],
    guard: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    view = make_view_inputs(row, processor, condition, frames_for=frames_for,
                            inputs_for=inputs_for, guard=guard)
    frame_ids = view["frame_ids"]
    captures = {}
    for name in ("base", "mild"):
        one = view[name]
        if guard is not None:
            guard(f"before_{name}_stock_prompt_prefill")
        captures[name] = capture_prompt_fields_once(model, one["inputs"], frame_ids, view["fps"])
        captures[name]["query_sha256"] = tensor_sha256(captures[name]["query"])
        captures[name]["visual_grid_sha256"] = tensor_sha256(captures[name]["visual_grid"])
        one["inputs"] = one["inputs"]
        one["capture"] = captures[name]
        one["features"] = None
    # The caller supplies an adapter to calculate the two internal feature fields.
    view["captures"] = captures
    return view


def stats_command(args: argparse.Namespace) -> dict[str, Any]:
    adapter, fit_meta, checkpoint_path, checkpoint_sha = load_adapter_checkpoint(args.architecture)
    set_adapter_groups(adapter, update=False)
    source_state = source_state_cpu(adapter)
    source_digest = adapter_sha256(adapter)
    rows, inputs_sha, pool_meta = source_statistics_rows(args.input_manifest, args.split, fit_meta)
    output = Path(args.output).resolve() if args.output else (
        OUT / "source_stats" / f"{args.architecture}_{pool_meta['queries']}q_{checkpoint_sha[:12]}.json")
    if OUT.resolve() not in output.parents:
        raise ValueError("source statistics output must remain inside artifacts/desta3d_v1/tta_run")
    if output.exists() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError(f"source statistics output or receipt already exists: {output}")
    args.session.check("before_source_stats_official_ptd_hash")
    ptd_meta = verify_ptd_identity()
    args.session.check("after_source_stats_official_ptd_hash")
    args.session.check("before_source_stats_model_load")
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, frames_for, inputs_for
    processor = processor_load()
    args.session.check("before_source_stats_ptd_weights_load")
    model = model_load()
    freeze_stock_model(model)
    accumulator = SourceFeatureStatsAccumulator(source_split="train")
    manifests: list[dict[str, Any]] = []
    started = time.monotonic()
    def guard(phase: str) -> None:
        args.session.check(phase)
    for index, row in enumerate(rows, 1):
        guard(f"source_stats_{index}_before_decode")
        frames, ids = frames_for(row, "clean")
        if [int(x) for x in ids] != row["input"]["frame_ids"]:
            raise ValueError(f"source exact frame IDs changed for {row['key']}")
        guard(f"source_stats_{index}_before_preprocess")
        inputs, preprocess = inputs_for(row, processor, frames)
        if preprocess.get("pixel_sha") != digest_numpy_frames(frames):
            raise ValueError(f"source pixel SHA mismatch for {row['key']}")
        guard(f"source_stats_{index}_before_stock_prompt_prefill")
        captured = capture_prompt_fields_once(model, inputs, ids, float(row["input"]["fps"]))
        guard(f"source_stats_{index}_after_stock_prompt_prefill")
        with torch.no_grad():
            fields = forward_with_features(adapter.to(captured["visual_grid"].device),
                                           captured["visual_grid"], captured["query"],
                                           frame_times=captured["frame_times"])
        accumulator.add(parent_id=row["source"], query_id=row["key"],
                        branch_features={"referent": fields["referent_features"],
                                         "event": fields["event_features"]})
        manifests.append({"key": row["key"], "source_parent": row["source"],
                          "source_video_sha256": row["input"]["video_sha256"],
                          "frame_ids": [int(v) for v in ids],
                          "pixel_sha256": preprocess["pixel_sha"],
                          "preprocess_sha256": json_sha256({"preprocess": cpu_copy(dict(preprocess)),
                                                              "frame_ids": list(ids), "video_grid_thw": inputs["video_grid_thw"].detach().cpu().tolist()}),
                          "stock_query_sha256": tensor_sha256(captured["query"]),
                          "merger_grid_sha256": tensor_sha256(captured["visual_grid"])})
        if adapter_sha256(adapter) != source_digest:
            raise RuntimeError("source statistics pass mutated the source-fit adapter")
        if index % 10 == 0 or index == len(rows):
            print("SOURCE_STATS", index, len(rows), row["key"], flush=True)
        del inputs, frames, captured, fields
    if adapter_sha256(adapter) != source_digest or any(p.requires_grad for p in adapter.parameters()):
        raise RuntimeError("source-statistics pass changed/froze the wrong source adapter state")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("PTD model became trainable during source statistics")
    stats = accumulator.finalize()
    if stats["parents"]["referent"]["__global__"]["parent_count"] != pool_meta["parents"]:
        raise ValueError("source statistics parent aggregation count mismatch")
    accumulator.save_json(output)
    receipt = {
        "status": "completed_source_train_statistics_only",
        "stats_path": str(output), "stats_sha256": sha256_file(output),
        "stats_schema": "vg_tta.desta3d_tta_v1.SourceFeatureStatsAccumulator",
        "source_split": "train", "gt_read": False, "target_data_read": False,
        "input_manifest": str(Path(args.input_manifest).resolve()), "input_manifest_sha256": inputs_sha,
        "pool": pool_meta, "per_parent_query_pixel_preprocess_hashes": manifests,
        "aggregation": stats["aggregation"], "branch_channels": {
            branch: len(stats["parents"][branch]["__global__"]["mean"]) for branch in ("referent", "event")},
        "source_fit": fit_meta, "source_fit_adapter_state_sha256_before_after": source_digest,
        "source_checkpoint_sha256": checkpoint_sha, "ptd_identity": ptd_meta,
        "query_count": pool_meta["queries"], "parent_count": pool_meta["parents"],
        "wall_seconds_including_model_load": time.monotonic() - args.session.started,
        "not_equated_to_618_query_mean": pool_meta["queries"] != 618,
        "full_618_query_train_pool_used": pool_meta["queries"] == 618,
        "no_target_or_source_validation_rows_used": True,
    }
    write_json_once(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return {"stats_path": str(output), "stats_sha256": receipt["stats_sha256"],
            "query_count": pool_meta["queries"], "parent_count": pool_meta["parents"],
            "GT_read": False, "gpu_started": True}


def load_ptd_runtime() -> tuple[Any, nn.Module, Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, frames_for, inputs_for, infer
    processor = processor_load()
    model = model_load()
    freeze_stock_model(model)
    return processor, model, infer, frames_for, inputs_for


def prediction_path(run_dir: Path, row: Mapping[str, Any], condition: str) -> Path:
    key_hash = hashlib.sha256(str(row["key"]).encode()).hexdigest()
    return run_dir / "predictions" / condition / f"{key_hash}.pt"


def validate_prediction_file(path: Path, context_sha: str) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"partial/unsealed prediction must be inspected manually: {path}")
    receipt = read_json(receipt_path)
    actual = sha256_file(path)
    if receipt.get("prediction_sha256") != actual or receipt.get("context_sha256") != context_sha:
        raise RuntimeError(f"prediction SHA/context mismatch: {path}")
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record.get("target_GT_read") is not False or record.get("GT_used") is not False:
        raise RuntimeError(f"prediction record has unsafe GT metadata: {path}")
    return receipt


def write_prediction_seal_once(path: Path, payload: Mapping[str, Any]) -> None:
    files = payload.get("files")
    if (payload.get("status") != "complete_gt_free_e5_tta_predictions"
            or not isinstance(payload.get("run_context_sha256"), str)
            or not isinstance(files, list) or payload.get("count") != len(files)):
        raise ValueError("malformed label-free prediction seal payload")
    write_json_once(path, dict(payload))


def read_prediction_seal(path: Path, context_sha: str, expected_count: int) -> dict[str, Any]:
    seal = read_json(path)
    files = seal.get("files")
    if (seal.get("status") != "complete_gt_free_e5_tta_predictions"
            or seal.get("run_context_sha256") != context_sha
            or seal.get("count") != expected_count
            or not isinstance(files, list) or len(files) != expected_count):
        raise RuntimeError("existing prediction seal does not match this exact run context")
    keys = [str(r.get("key")) for r in files if isinstance(r, dict)]
    if len(keys) != expected_count or len(set(keys)) != expected_count:
        raise RuntimeError("existing prediction seal has invalid or duplicate query entries")
    for record in files:
        receipt = validate_prediction_file(Path(record["path"]), context_sha)
        if record.get("prediction_sha256") != receipt.get("prediction_sha256"):
            raise RuntimeError(f"prediction seal entry hash mismatch: {record.get('path')}")
    return seal


def commit_prediction(path: Path, record: Mapping[str, Any], context_sha: str,
                      key: str, source: str) -> dict[str, Any]:
    atomic_torch_save_once(path, cpu_copy(dict(record)))
    receipt = {"status": "sealed_label_free_prediction",
               "key": key, "source": source, "path": str(path),
               "prediction_sha256": sha256_file(path), "context_sha256": context_sha,
               "GT_used": False, "target_GT_read": False, "completed_unix": time.time()}
    write_json_once(path.with_suffix(path.suffix + ".json"), receipt)
    return receipt


def run_row(
    row: Mapping[str, Any],
    *,
    architecture: str,
    source_metadata: Mapping[str, Any],
    source_state: Mapping[str, Tensor],
    source_digest: str,
    source_stats: Mapping[str, Any],
    source_stats_sha: str,
    config: Mapping[str, Any],
    processor: Any,
    model: nn.Module,
    infer_fn: Callable[..., Any],
    frames_for: Callable[..., tuple[np.ndarray, Sequence[int]]],
    inputs_for: Callable[..., tuple[dict[str, Tensor], Mapping[str, Any]]],
    condition: str,
    guard: Callable[[str], None],
) -> dict[str, Any]:
    adapter = Desta3DAdapter(in_channels=2560, query_dim=2560,
                             hidden_dim=int(source_metadata["hidden_dim"]), architecture=architecture)
    adapter.load_state_dict(source_state, strict=True)
    adapter.to("cuda")
    set_adapter_groups(adapter, update=False)
    initial_sha = adapter_sha256(adapter)
    if initial_sha != source_digest:
        raise RuntimeError("episode did not reset exactly to the selected source-fit BEST state")

    pair = capture_both_views(model, processor, row, condition,
                              frames_for=frames_for, inputs_for=inputs_for, guard=guard)
    base, mild = pair["base"], pair["mild"]
    guard("source_fit_no_tta_baseline_before_full_ptd")
    baseline = full_ptd_replay_with_merger_adapter(
        model, processor, base["inputs"], adapter, base["capture"]["query"],
        pair["frame_ids"], pair["fps"], infer_fn=infer_fn,
        expected_visual_grid=base["capture"]["visual_grid"])
    baseline_state_sha = adapter_sha256(adapter)
    if baseline_state_sha != initial_sha:
        raise RuntimeError("no-TTA baseline inference changed the source-fit adapter")
    guard("after_source_fit_no_tta_full_ptd")

    reset_sha = reset_episode_adapter(adapter, source_state)
    if reset_sha != initial_sha:
        raise RuntimeError("episodic reset after no-TTA baseline did not restore source-fit weights")
    set_adapter_groups(adapter, update=True)
    trace = perform_fixed_tta_steps(adapter, pair["base"], pair["mild"], source_stats,
                                    config, guard=guard)
    final_state_sha = adapter_sha256(adapter)
    if not trace or trace[-1]["adapter_state_sha256_after_update"] != final_state_sha:
        raise RuntimeError("fixed TTA terminal state does not match last update receipt")

    guard("fixed_tta_terminal_state_before_full_ptd_replay")
    adapted = full_ptd_replay_with_merger_adapter(
        model, processor, base["inputs"], adapter, base["capture"]["query"],
        pair["frame_ids"], pair["fps"], infer_fn=infer_fn,
        expected_visual_grid=base["capture"]["visual_grid"])
    if adapter_sha256(adapter) != final_state_sha:
        raise RuntimeError("full PTD replay changed the fixed terminal adapter state")
    guard("after_fixed_tta_terminal_full_ptd_replay")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("frozen PTD backbone became trainable during episodic TTA")

    view_records = {}
    for name in ("base", "mild"):
        view = pair[name]
        view_records[name] = {k: view[k] for k in ("name", "pixel_sha256", "preprocess_sha256",
                                                     "input_signature_sha256", "preprocess")}
        view_records[name].update({"query_sha256": view["capture"]["query_sha256"],
                                   "visual_grid_sha256": view["capture"]["visual_grid_sha256"]})
    return {
        "key": row["key"], "source": row["source"], "cohort": row["cohort"],
        "split": row["split"], "condition": condition,
        "frame_ids": pair["frame_ids"], "frame_ids_sha256": pair["frame_ids_sha256"],
        "fps": pair["fps"], "video_grid_thw": pair["video_grid_thw"],
        "mild_transform": pair["mild_transform"], "views": view_records,
        "source_fit_adapter_sha256": source_digest,
        "adapter_sha256_before_tta": initial_sha,
        "adapter_sha256_after_tta": final_state_sha,
        "update_groups": list(UPDATE_GROUPS),
        "updates": trace,
        "state_selection": "last fixed configured step; no loss-based checkpoint selection",
        "source_stats_sha256": source_stats_sha,
        "source_stats_parent_count": source_stats["parents"]["referent"]["__global__"]["parent_count"],
        "baseline_source_fit_frozen_no_tta": baseline,
        "adapted_fixed_terminal": adapted,
        "GT_used": False, "target_GT_read": False,
        "loss_decrease_is_not_utility_success": True,
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    config_raw = read_json(args.config)
    config = validate_tta_config(config_raw)
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    if args.architecture and args.architecture != config["architecture"]:
        raise ValueError("CLI architecture differs from explicit run config")
    architecture = config["architecture"]
    rows, input_manifest_sha = load_input_manifest(args.input_manifest, args.split)
    stats_ref = validate_source_stats(args.source_stats)
    source_stats, stats_sha = stats_ref["stats"], stats_ref["stats_sha256"]
    adapter, source_meta, checkpoint_path, checkpoint_sha = load_adapter_checkpoint(architecture)
    stats_source_fit = stats_ref["receipt"].get("source_fit", {})
    if (stats_source_fit.get("architecture") != architecture
            or stats_source_fit.get("checkpoint_sha256") != checkpoint_sha
            or stats_source_fit.get("adapter_sha256") != source_meta["adapter_sha256"]
            or stats_source_fit.get("train_lock_sha256") != source_meta["train_lock_sha256"]
            or stats_source_fit.get("query_manifest_sha256") != source_meta["query_manifest_sha256"]
            or stats_source_fit.get("train_records_sha256") != source_meta["train_records_sha256"]
            or stats_ref["receipt"].get("source_checkpoint_sha256") != checkpoint_sha):
        raise ValueError("source statistics were computed from a different source-fit checkpoint")
    set_adapter_groups(adapter, update=False)
    source_state = source_state_cpu(adapter)
    source_digest = adapter_sha256(adapter)
    if checkpoint_sha != source_meta["checkpoint_sha256"]:
        raise RuntimeError("source fit checkpoint changed after loading")
    run_id = args.run_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a simple 1-80 character filename label")
    run_dir = OUT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha = json_sha256(config)
    condition = args.condition
    run_context = {"run_id": run_id, "config": config, "config_sha256": config_sha,
                   "architecture": architecture, "source_fit_checkpoint_sha256": checkpoint_sha,
                   "source_fit_adapter_sha256": source_digest, "source_stats_sha256": stats_sha,
                   "source_fit_identity": {"train_lock_sha256": source_meta["train_lock_sha256"],
                     "query_manifest_sha256": source_meta["query_manifest_sha256"],
                     "train_records_sha256": source_meta["train_records_sha256"]},
                   "source_stats_receipt_sha256": stats_ref["receipt_sha256"],
                   "input_manifest": str(Path(args.input_manifest).resolve()),
                   "input_manifest_sha256": input_manifest_sha, "split": args.split,
                   "condition": condition, "mild_transform": {"brightness": MILD_BRIGHTNESS,
                     "contrast_about_per_frame_mean": MILD_CONTRAST,
                     "order": "contrast_about_mean_then_brightness", "round": "numpy.rint", "clamp": [0, 255]},
                   "gt_read": False, "target_gt_used": False}
    context_sha = json_sha256(run_context)
    context_path = run_dir / "RUN_CONTEXT.json"
    if context_path.exists():
        existing = read_json(context_path)
        if existing != {**run_context, "context_sha256": context_sha}:
            raise RuntimeError("run_id already belongs to different config/input/checkpoints; choose another run_id")
    else:
        write_json_once(context_path, {**run_context, "context_sha256": context_sha})
    seal_path = run_dir / "PREDICTION_SEAL.json"
    if seal_path.exists():
        seal = read_prediction_seal(seal_path, context_sha, len(rows))
        return {"status": "existing_seal_verified", "run_dir": str(run_dir),
                "count": seal["count"], "gpu_started": False}

    args.session.check("before_run_official_ptd_hash")
    ptd_meta = verify_ptd_identity()
    args.session.check("after_run_official_ptd_hash")
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, infer as ptd_infer, frames_for, inputs_for
    processor = processor_load()
    args.session.check("before_ptd_weights_load")
    model = model_load()
    freeze_stock_model(model)
    args.session.check("after_model_load_before_row_loop")
    if ptd_infer is None:
        raise RuntimeError("PTD inference function unavailable")
    started = time.monotonic()
    file_receipts = []
    for index, row in enumerate(rows, 1):
        args.session.check(f"query_{index}_before_forward")
        output_path = prediction_path(run_dir, row, condition)
        if output_path.exists() or output_path.with_suffix(output_path.suffix + ".json").exists():
            receipt = validate_prediction_file(output_path, context_sha)
            file_receipts.append({"key": row["key"], "path": receipt["path"],
                                  "prediction_sha256": receipt["prediction_sha256"]})
            continue
        result = run_row(row, architecture=architecture, source_metadata=source_meta,
                         source_state=source_state, source_digest=source_digest,
                         source_stats=source_stats, source_stats_sha=stats_sha,
                         config=config, processor=processor, model=model,
                         infer_fn=ptd_infer, frames_for=frames_for, inputs_for=inputs_for,
                         condition=condition, guard=args.session.check)
        receipt = commit_prediction(output_path, result, context_sha, row["key"], row["source"])
        file_receipts.append({"key": row["key"], "path": receipt["path"],
                              "prediction_sha256": receipt["prediction_sha256"]})
        print("TTA_PREDICTION", index, len(rows), row["key"],
              result["baseline_source_fit_frozen_no_tta"]["format_ok"],
              result["adapted_fixed_terminal"]["format_ok"], flush=True)
    if len(file_receipts) != len(rows):
        raise RuntimeError("not all input rows have a saved prediction record")
    sealed_files = []
    for item in file_receipts:
        p = Path(item["path"])
        receipt = validate_prediction_file(p, context_sha)
        sealed_files.append({"key": item["key"], "path": str(p),
                             "prediction_sha256": receipt["prediction_sha256"]})
    seal = {
        "status": "complete_gt_free_e5_tta_predictions",
        "created_unix": time.time(), "run_id": run_id,
        "run_context_sha256": context_sha, "count": len(sealed_files),
        "files": sorted(sealed_files, key=lambda x: x["key"]),
        "ptd_identity": ptd_meta, "source_fit_checkpoint_sha256": checkpoint_sha,
        "source_stats_sha256": stats_sha, "input_manifest_sha256": input_manifest_sha,
        "GPU_prediction_worker": True, "target_GT_read": False,
        "uses_stock_ptd_decoder_and_merger_hook": True,
        "source_fit_no_tta_baseline_saved_per_query": True,
        "scoring_and_method_selection": "deferred to root scorer; none performed by this runner",
    }
    write_prediction_seal_once(seal_path, seal)
    return {"status": "sealed", "run_dir": str(run_dir), "count": len(sealed_files),
            "worker_seconds": time.monotonic() - started, "target_GT_read": False,
            "scoring_performed": False, "gpu_started": True}


class InvocationSession:
    """Nested-receipt cumulative wall-time and disk guard with failure receipts."""
    def __init__(self, action: str, run_id: str) -> None:
        authorization = read_json(BUDGET_AUTHORITY)
        if authorization.get("status") != "active" or authorization.get("GPU_seconds_cap", "missing") is not None:
            raise RuntimeError("explicit current uncapped authorization is required")
        self.cumulative_cap_seconds = None
        self.budget_authority_sha256 = sha256_file(BUDGET_AUTHORITY)
        self.action, self.run_id = action, run_id
        self.started = time.monotonic()
        self.started_unix = time.time()
        self.session_id = f"{time.time_ns()}_{os.getpid()}"
        self.marker = OUT / "running" / f"{self.session_id}.json"
        self.receipt = OUT / "receipts" / f"{self.session_id}.json"
        self.failure = OUT / "failures" / f"{self.session_id}.json"
        self.lease_file = None
        self.prior_seconds = 0.0
        self.prior_receipts: list[dict[str, Any]] = []
        self.phase = "initializing"
        self.status = "running"
        self.error: str | None = None

    def _gpu_receipts(self) -> tuple[float, list[dict[str, Any]]]:
        return scan_nested_gpu_receipts(PARENT)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _recover_orphaned_markers(self) -> None:
        now = time.time()
        for marker in sorted((OUT / "running").glob("*.json")):
            if marker == self.marker:
                continue
            payload = read_json(marker)
            pid = int(payload.get("pid", -1))
            if pid > 0 and self._pid_alive(pid):
                raise RuntimeError(f"another TTA invocation has a live marker despite the GPU lease: {marker}")
            elapsed = max(0.0, now - float(payload.get("started_unix", now)))
            receipt = OUT / "receipts" / f"interrupted_{marker.stem}.json"
            record = {"stage": f"interrupted_{payload.get('action', 'unknown')}",
                      "run_id": payload.get("run_id"), "seconds": elapsed,
                      "status": "interrupted_process_walltime_conservatively_counted",
                      "phase": payload.get("phase"), "GT_read": False,
                      "target_GT_read": False, "gpu_started": True,
                      "note": "orphaned running marker recovered after nonblocking shared GPU lease was acquired"}
            if not receipt.exists():
                write_json_once(receipt, record)
            failure = OUT / "failures" / f"interrupted_{marker.stem}.json"
            if not failure.exists():
                write_json_once(failure, record)
            marker.unlink()

    def __enter__(self) -> "InvocationSession":
        for directory in (OUT / "running", OUT / "receipts", OUT / "failures"):
            directory.mkdir(parents=True, exist_ok=True)
        write_json_once(self.marker, {"status": "running", "action": self.action,
                                      "run_id": self.run_id, "pid": os.getpid(),
                                      "started_unix": self.started_unix})
        try:
            GLOBAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
            self.lease_file = GLOBAL_GPU_LOCK.open("a")
            fcntl.flock(self.lease_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.status = "failed_gpu_lease_unavailable"
            self.error = traceback.format_exc()
            self.close()
            raise
        try:
            self._recover_orphaned_markers()
            free = shutil.disk_usage(ROOT).free
            if free < MIN_FREE_BYTES:
                self.status = "failed_disk_floor"
                self.error = f"free disk {free} is below required {MIN_FREE_BYTES} bytes"
                self.close()
                raise RuntimeError(self.error)
            self.prior_seconds, self.prior_receipts = self._gpu_receipts()
            if self.cumulative_cap_seconds is not None and self.prior_seconds >= self.cumulative_cap_seconds:
                self.status = "failed_cumulative_cap_preflight"
                self.error = f"prior nested GPU receipts total {self.prior_seconds:.3f}s >= {self.cumulative_cap_seconds:.0f}s"
                self.close()
                raise RuntimeError(self.error)
        except BaseException:
            if self.lease_file is not None:
                self.status = self.status if self.status != "running" else "failed_preflight"
                self.error = self.error or traceback.format_exc()
                if self.marker.exists():
                    self.close()
            raise
        self.phase = "preflight_passed"
        self.write_marker()
        return self

    def write_marker(self) -> None:
        tmp = self.marker.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": "running", "action": self.action,
                                   "run_id": self.run_id, "pid": os.getpid(),
                                   "started_unix": self.started_unix, "phase": self.phase,
                                   "prior_receipt_seconds": self.prior_seconds}, indent=2) + "\n")
        os.replace(tmp, self.marker)

    def check(self, phase: str) -> None:
        self.phase = phase
        self.write_marker()
        free = shutil.disk_usage(ROOT).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"disk floor reached at {phase}: {free} < {MIN_FREE_BYTES}")
        current_receipts, _ = self._gpu_receipts()
        elapsed = time.monotonic() - self.started
        used = max(self.prior_seconds, current_receipts) + elapsed
        if self.cumulative_cap_seconds is not None and used >= self.cumulative_cap_seconds:
            raise RuntimeError(f"nested cumulative GPU cap reached at {phase}: {used:.3f} >= {self.cumulative_cap_seconds:.0f}")

    def close(self) -> None:
        elapsed = max(0.0, time.monotonic() - self.started)
        try:
            final_receipts, _ = self._gpu_receipts()
        except BaseException as exc:
            final_receipts = self.prior_seconds
            self.error = (self.error or "") + f"\nreceipt audit failed during close: {exc!r}"
            self.status = "failed_receipt_audit"
        record = {"action": self.action, "run_id": self.run_id,
                  "seconds": elapsed, "prior_seconds": self.prior_seconds,
                  "nested_receipt_seconds_at_start": self.prior_seconds,
                  "nested_receipt_seconds_at_end": final_receipts,
                  "nested_receipts_at_start": self.prior_receipts,
                  "phase": self.phase, "status": self.status,
                  "error": self.error, "GT_read": False,
                  "target_GT_read": False, "gpu_started": self.status != "failed_gpu_lease_unavailable",
                  "cumulative_gpu_cap_seconds": self.cumulative_cap_seconds,
                  "budget_authority": str(BUDGET_AUTHORITY),
                  "budget_authority_sha256": self.budget_authority_sha256,
                  "free_disk_floor_bytes": MIN_FREE_BYTES,
                  "finished_unix": time.time()}
        if self.status != "completed":
            self.failure.parent.mkdir(parents=True, exist_ok=True)
            write_json_once(self.failure, record)
        if not self.receipt.exists():
            write_json_once(self.receipt, record)
        try:
            self.marker.unlink(missing_ok=True)
        finally:
            if self.lease_file is not None:
                try:
                    fcntl.flock(self.lease_file, fcntl.LOCK_UN)
                finally:
                    self.lease_file.close()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.status = "failed"
            self.error = "".join(traceback.format_exception(exc_type, exc, tb))
        elif self.status == "running":
            self.status = "completed"
        self.close()
        return False


def run_with_session(action: str, run_id: str, fn: Callable[[InvocationSession], Any]) -> Any:
    session = InvocationSession(action, run_id)
    with session:
        # Functions needing the session guard receive it through args.session.
        class Args:  # tiny carrier avoids mutable global state
            pass
        args = Args()
        args.session = session
        result = fn(args)
        session.status = "completed"
        return result


def _dummy_stats(channels: int) -> dict[str, Any]:
    zero = torch.zeros(channels)
    one = torch.ones(channels)
    return {"source_split": "train", "gt_read": False,
            "parents": {name: {"__global__": {"mean": zero.clone(), "std": one.clone()}}
                        for name in ("referent", "event")}}


def _toy_prompt_model() -> nn.Module:
    from types import SimpleNamespace

    class ToyVision(nn.Module):
        def __init__(self):
            super().__init__(); self.merger = nn.Identity(); self.mock_parameter = nn.Parameter(torch.ones(()))

    class ToyLanguage(nn.Module):
        def forward(self, input_ids: Tensor, **_kwargs: Any):
            b, length = input_ids.shape
            base = torch.arange(length, device=input_ids.device, dtype=torch.float32)[None, :, None]
            hidden = base.expand(b, length, 2560).contiguous()
            return SimpleNamespace(last_hidden_state=hidden)

    class ToyCore(nn.Module):
        def __init__(self):
            super().__init__(); self.visual = ToyVision(); self.language_model = ToyLanguage()
        def forward(self, input_ids: Tensor, attention_mask: Tensor, video_grid_thw: Tensor,
                    pixel_values_videos: Tensor, **kwargs: Any):
            merger = self.visual.merger(pixel_values_videos)
            language = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
            return merger, language

    class ToyFull(nn.Module):
        def __init__(self):
            super().__init__(); self.model = ToyCore(); self.config = SimpleNamespace(video_token_id=99)
    return ToyFull().eval()


def run_cpu_checks() -> dict[str, Any]:
    torch.manual_seed(20260927)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    pixels = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    identity = make_mild_photometric_view(pixels, brightness=1.0, contrast=1.0)
    mild = make_mild_photometric_view(pixels)
    checks["same_video_mild_view"] = {"identity_factors_are_exact_noop": bool(np.array_equal(identity, pixels)),
                                       "fixed_transform_keeps_shape_dtype": mild.shape == pixels.shape and mild.dtype == np.uint8,
                                       "different_pixel_sha": digest_numpy_frames(mild) != digest_numpy_frames(pixels)}
    if not np.array_equal(identity, pixels) or mild.shape != pixels.shape or mild.dtype != np.uint8:
        failures.append("same-video photometric transform identity/shape check failed")

    model = _toy_prompt_model()
    model.model.visual.merger.requires_grad_(True)
    freeze_stock_model(model)
    mock_inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 10, 11, 12]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "pixel_values_videos": torch.arange(2 * 2560, dtype=torch.float32).reshape(2, 2560),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
        "unused_gt": torch.tensor([987]),
    }
    hooks_before = len(model.model.visual.merger._forward_hooks) + len(model.model.language_model._forward_hooks)
    prompt_capture = capture_prompt_fields_once(model, mock_inputs, [3, 8], 2.0)
    hooks_after = len(model.model.visual.merger._forward_hooks) + len(model.model.language_model._forward_hooks)
    exact_query = torch.full((1, 2560), 4.0)
    capture_ok = (prompt_capture["query"].shape == (1, 2560)
                  and torch.equal(prompt_capture["query"], exact_query)
                  and tuple(prompt_capture["visual_grid"].shape) == (1, 2, 1, 1, 2560)
                  and hooks_before == hooks_after == 0)
    checks["one_stock_forward_captures_grid_and_query"] = {"query_after_last_video_token": capture_ok,
                                                            "grid_shape": list(prompt_capture["visual_grid"].shape),
                                                            "temporary_hooks_removed": hooks_before == hooks_after}
    if not capture_ok:
        failures.append("one-pass PTD mock did not capture exact stock query/grid")

    small = Desta3DAdapter(in_channels=8, query_dim=3, hidden_dim=8, architecture="dual3d").cpu()
    small_source = source_state_cpu(small)
    original_digest = adapter_sha256(small)
    with torch.no_grad():
        next(small.input_proj.parameters()).add_(0.25)
    mutated_digest = adapter_sha256(small)
    reset_digest = reset_episode_adapter(small, small_source)
    groups = set_adapter_groups(small, update=True)
    group_freeze_ok = (reset_digest == original_digest and mutated_digest != original_digest
                       and groups["input_proj"] > 0 and groups["stem"] > 0 and groups["readers"] > 0
                       and all(not p.requires_grad for name in ("query_proj", "film", "out_proj", "referent_head", "event_head")
                               for p in getattr(small, name).parameters()))
    checks["source_checkpoint_reset_and_freeze"] = {"mutated_state_reset_exactly": reset_digest == original_digest,
                                                     "only_input_stem_readers_trainable": group_freeze_ok}
    if not group_freeze_ok:
        failures.append("source-fit adapter reset/freeze group mock failed")

    # The mock full inference calls the exact visual merger under the actual hook
    # helper; zero-initialized out_proj must preserve its source tokens exactly.
    noop_adapter = Desta3DAdapter(in_channels=8, query_dim=3, hidden_dim=8,
                                  architecture="dual3d").cpu().eval()
    noop_q = torch.ones((1, 3))
    mock_full_inputs = {"video_grid_thw": torch.tensor([[2, 2, 2]]),
                        "mock_tokens": torch.randn(2, 8)}
    frozen_input_tokens = mock_full_inputs["mock_tokens"].clone()
    def mock_infer(mock_model: nn.Module, _processor: Any, values: Mapping[str, Any]) -> dict[str, Any]:
        returned = mock_model.model.visual.merger(values["mock_tokens"])
        return {"positions": [0, 1], "boxes": torch.zeros(2, 4),
                "geometry_valid": torch.ones(2, dtype=torch.bool), "interval": [0, 1],
                "format_ok": True, "completion": "mock", "GT_used": False,
                "raw_merger_output": returned}
    merger_hooks_before = len(model.model.visual.merger._forward_hooks)
    replay = full_ptd_replay_with_merger_adapter(
        model, None, mock_full_inputs, noop_adapter, noop_q, [3, 8], 2.0,
        infer_fn=mock_infer,
    )
    replay_tokens = torch.as_tensor(mock_infer(model, None, mock_full_inputs)["raw_merger_output"])
    expected_noop_grid = noop_adapter(
        reshape_merged_video_tokens(mock_full_inputs["mock_tokens"], mock_full_inputs["video_grid_thw"][0]),
        noop_q, frame_times=torch.tensor([[1.5, 4.0]])
    )["updated_tokens"].detach().reshape_as(mock_full_inputs["mock_tokens"])
    expected_noop_grid = expected_noop_grid.reshape(1, 2, 1, 1, 8)
    merger_hooks_after = len(model.model.visual.merger._forward_hooks)
    # The toy merger input must match the supplied THW grid's merged token count.
    # The second mock call above is outside the hook and must remain unchanged.
    no_tta_hook_ok = (replay["merger_hook_calls"] == 1 and replay["format_ok"]
                      and torch.equal(replay_tokens, frozen_input_tokens)
                      and torch.equal(expected_noop_grid.reshape_as(frozen_input_tokens), frozen_input_tokens)
                      and replay["updated_tokens_sha256"] == tensor_sha256(expected_noop_grid)
                      and merger_hooks_before == merger_hooks_after == 0)
    checks["merger_noop_full_replay_and_hook_cleanup"] = {"zero_adapter_exact_merger_output": no_tta_hook_ok,
                                                           "temporary_hook_removed": merger_hooks_before == merger_hooks_after}
    if not no_tta_hook_ok:
        failures.append("zero adapter merger replay/mock hook cleanup failed")

    frozen_sourcefit_mock = Desta3DAdapter(in_channels=8, query_dim=3, hidden_dim=8,
                                           architecture="dual3d").cpu().eval()
    with torch.no_grad():
        frozen_sourcefit_mock.out_proj.weight.fill_(0.002)
    set_adapter_groups(frozen_sourcefit_mock, update=False)
    frozen_sourcefit_sha = adapter_sha256(frozen_sourcefit_mock)
    mock_raw_grid = reshape_merged_video_tokens(mock_full_inputs["mock_tokens"],
                                                mock_full_inputs["video_grid_thw"][0])
    expected_sourcefit_output = frozen_sourcefit_mock(
        mock_raw_grid, noop_q, frame_times=torch.tensor([[1.5, 4.0]])
    )["updated_tokens"].detach()
    sourcefit_baseline_replay = full_ptd_replay_with_merger_adapter(
        model, None, mock_full_inputs, frozen_sourcefit_mock, noop_q, [3, 8], 2.0,
        infer_fn=mock_infer,
    )
    sourcefit_no_tta_frozen = (adapter_sha256(frozen_sourcefit_mock) == frozen_sourcefit_sha
                               and sourcefit_baseline_replay["updated_tokens_sha256"]
                               == tensor_sha256(expected_sourcefit_output)
                               and all(not p.requires_grad for p in frozen_sourcefit_mock.parameters()))
    checks["sourcefit_no_tta_baseline_is_frozen"] = {"state_unchanged_during_full_replay": sourcefit_no_tta_frozen,
                                                      "baseline_is_adapter_injected_not_native": True}
    if not sourcefit_no_tta_frozen:
        failures.append("frozen source-fit no-TTA baseline changed state during PTD replay")

    # Deterministic synthetic same-grid views verify fixed 3-step terminal
    # selection. This uses only mock tensors and never scores utility.
    t = torch.tensor([[0.0, 0.4, 1.0]])
    visual_a = torch.randn(1, 3, 4, 4, 8)
    visual_b = visual_a * 0.98 + 0.01
    q = torch.randn(1, 3)
    out_a = forward_with_features(small, visual_a, q, frame_times=t)
    out_b = forward_with_features(small, visual_b, q, frame_times=t.clone())
    va = {"frame_ids": [0, 2, 5], "video_grid_thw": [3, 4, 4], "capture": {"visual_grid": visual_a, "query": q, "frame_times": t}, "features": out_a}
    vb = {"frame_ids": [0, 2, 5], "video_grid_thw": [3, 4, 4], "capture": {"visual_grid": visual_b, "query": q, "frame_times": t.clone()}, "features": out_b}
    mock_config = {"architecture": "dual3d", "mode": "dual_nojoint", "steps": 3, "seed": 17,
                   "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
                   "weights": {"feature_alignment": 1.0, "latent_consistency": 1.0,
                               "referent_consistency": 1.0, "event_consistency": 1.0,
                               "asymmetric_joint": 1.0}}
    test_stats = _dummy_stats(8)
    normalized_config = validate_tta_config(mock_config)
    config_validation_is_idempotent = normalized_config == validate_tta_config(normalized_config)
    step_trace = perform_fixed_tta_steps(small, va, vb, test_stats, normalized_config)
    fixed_steps_ok = (len(step_trace) == 3 and [r["step"] for r in step_trace] == [1, 2, 3]
                      and adapter_sha256(small) == step_trace[-1]["adapter_state_sha256_after_update"]
                      and all("adapter_state_sha256_after_update" in r for r in step_trace)
                      and config_validation_is_idempotent)
    checks["fixed_three_step_terminal_no_best_loss_selection"] = {"steps": len(step_trace),
                                                                  "terminal_digest_is_step3": fixed_steps_ok,
                                                                  "validated_config_is_idempotent": config_validation_is_idempotent,
                                                                  "selection": "configured final step, not min loss"}
    if not fixed_steps_ok:
        failures.append("fixed configured 3-step terminal state test failed")

    with tempfile.TemporaryDirectory(prefix="desta3d-tta-seal-") as tmp:
        test_dir = Path(tmp)
        pred_path = test_dir / "prediction.pt"
        seal_path = test_dir / "PREDICTION_SEAL.json"
        context_sha = json_sha256({"fixture": "no-gt"})
        prediction_receipt = commit_prediction(
            pred_path, {"GT_used": False, "target_GT_read": False, "boxes": torch.zeros(1, 4)},
            context_sha, "mock:key", "mock-source")
        receipt_ok = validate_prediction_file(pred_path, context_sha)["prediction_sha256"] == sha256_file(pred_path)
        seal_payload = {"status": "complete_gt_free_e5_tta_predictions",
                        "run_context_sha256": context_sha, "count": 1,
                        "files": [{"key": "mock:key", "path": str(pred_path),
                                   "prediction_sha256": prediction_receipt["prediction_sha256"]}]}
        write_prediction_seal_once(seal_path, seal_payload)
        roundtrip_ok = read_prediction_seal(seal_path, context_sha, 1)["count"] == 1
        overwrite_rejected = False
        try:
            commit_prediction(pred_path, {"GT_used": False, "target_GT_read": False}, context_sha,
                              "mock:key", "mock-source")
        except FileExistsError:
            overwrite_rejected = True
    checks["prediction_files_are_hash_sealed_once"] = {"sidecar_hash_verified": receipt_ok,
                                                        "writer_reader_seal_roundtrip": roundtrip_ok,
                                                        "overwrite_rejected": overwrite_rejected}
    if not receipt_ok or not roundtrip_ok or not overwrite_rejected:
        failures.append("prediction receipt/seal roundtrip/overwrite guard failed")

    with tempfile.TemporaryDirectory(prefix="desta3d-tta-budget-") as tmp:
        receipt_root = Path(tmp)
        write_json_once(receipt_root / "a" / "receipts" / "one.json", {"seconds": 2.0, "stage": "one"})
        write_json_once(receipt_root / "b" / "worker_receipts" / "two.json", {"seconds": 3.5, "stage": "two"})
        write_json_once(receipt_root / "not_a_receipt.json", {"seconds": 99.0})
        total, entries = scan_nested_gpu_receipts(receipt_root)
    nested_receipts_ok = total == 5.5 and len(entries) == 2
    checks["nested_gpu_receipts_accounted"] = {"seconds": total, "receipt_count": len(entries),
                                               "nonreceipt_json_ignored": nested_receipts_ok}
    if not nested_receipts_ok:
        failures.append("nested GPU receipt scan did not sum only receipts/worker_receipts")

    malformed_receipt_rejected = missing_seconds_receipt_rejected = False
    with tempfile.TemporaryDirectory(prefix="desta3d-tta-bad-receipts-") as tmp:
        receipt_root = Path(tmp)
        malformed_path = receipt_root / "nested" / "receipts" / "malformed.json"
        missing_path = receipt_root / "nested" / "worker_receipts" / "missing.json"
        malformed_path.parent.mkdir(parents=True)
        missing_path.parent.mkdir(parents=True)
        malformed_path.write_text("[]\n")
        missing_path.write_text('{"stage":"worker_without_duration"}\n')
        try:
            scan_nested_gpu_receipts(receipt_root)
        except RuntimeError as exc:
            malformed_receipt_rejected = "must be a JSON object" in str(exc)
        malformed_path.unlink()
        try:
            scan_nested_gpu_receipts(receipt_root)
        except RuntimeError as exc:
            missing_seconds_receipt_rejected = "missing required seconds" in str(exc)
    checks["malformed_and_missing_gpu_receipts_fail_closed"] = {
        "non_object_receipt_rejected": malformed_receipt_rejected,
        "missing_seconds_receipt_rejected": missing_seconds_receipt_rejected,
    }
    if not malformed_receipt_rejected or not missing_seconds_receipt_rejected:
        failures.append("malformed/missing-seconds GPU receipt was not rejected")

    bad_grad_rejected = bad_parameter_rejected = bad_config_rejected = False
    finite_fixture = nn.Parameter(torch.tensor([1.0]))
    finite_fixture.grad = torch.tensor([float("nan")])
    try:
        _require_finite_active_gradients([finite_fixture], 1)
    except FloatingPointError:
        bad_grad_rejected = True
    with torch.no_grad():
        finite_fixture.fill_(float("inf"))
    try:
        _require_finite_active_parameters([finite_fixture], 1)
    except FloatingPointError:
        bad_parameter_rejected = True
    bad_config = copy.deepcopy(mock_config)
    bad_config["optimizer"]["lr"] = float("nan")
    try:
        validate_tta_config(bad_config)
    except ValueError:
        bad_config_rejected = True
    from scripts.desta3d_source_fit_v1 import _state_dict_digest
    digest_fixture = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                      "b": torch.tensor([0.25, -1.0], dtype=torch.bfloat16)}
    sourcefit_digest_matches = adapter_sha256_from_state(digest_fixture) == _state_dict_digest(digest_fixture)
    source_identity_fixture = {"train_lock_sha256": "1" * 64,
                               "query_manifest_sha256": "2" * 64,
                               "train_records_sha256": "3" * 64}
    source_best_sha = adapter_sha256_from_state(digest_fixture)
    source_best_payload = {"architecture": "dual3d", "best_epoch": 1,
                           "source_parent_macro_vIoU": 0.375,
                           "source_fit_identity": source_identity_fixture,
                           "adapter_sha256": source_best_sha, "adapter": digest_fixture}
    source_fit_complete_fixture = {
        "best_adapter_sha256": {"dual3d": source_best_sha},
        "architectures": {"dual3d": {"best_epoch": 1, "source_parent_macro_vIoU": 0.375}},
    }
    best_payload_contract_ok = validate_best_checkpoint_payload(
        source_best_payload, "dual3d", source_identity_fixture, source_fit_complete_fixture
    ) == (source_best_sha, 1, 0.375)
    query_manifest_fixture = {"status": "complete_gt_free_exact_query_capture",
                              "count": 816, "GT_read": False,
                              "files": [{"key": f"fixture:{i:04d}"} for i in range(816)]}
    query_manifest_contract_ok = validate_source_fit_query_manifest(
        query_manifest_fixture,
        {"source_split": {"train_queries": 618, "validation_queries": 198}},
    ) == 816
    pool_fixture = []
    for parent_index in range(95):
        query_n = 7 if parent_index < 48 else 6
        for query_index in range(query_n):
            frame_ids = [0, 2, 4]
            if parent_index < 57 and query_index == query_n - 1:
                frame_ids = [1, 3, 5]
            pool_fixture.append({
                "key": f"fixture:{parent_index}:{query_index}",
                "source": f"parent-{parent_index}",
                "input": {"video_sha256": f"{parent_index + 1:064x}",
                          "frame_ids": frame_ids, "fps": 24.0,
                          "width": 320, "height": 240},
            })
    pool_summary = summarize_source_statistics_pool(
        pool_fixture, "/fixture/SOURCE_INPUTS.json", "4" * 64,
        pool_name="SOURCE_INPUTS_full_source_train", expected_queries=618,
    )
    full_train_pool_contract_ok = (pool_summary["queries"] == 618
                                   and pool_summary["parents"] == 95
                                   and pool_summary["unique_video_sha256"] == 95
                                   and pool_summary["parents_with_query_varying_visual_signature"] == 57)
    checks["nonfinite_update_inputs_rejected"] = {
        "nonfinite_gradient_rejected_before_step": bad_grad_rejected,
        "nonfinite_updated_parameter_rejected": bad_parameter_rejected,
        "nonfinite_learning_rate_rejected": bad_config_rejected,
        "source_fit_adapter_digest_matches": sourcefit_digest_matches,
        "final_source_fit_best_identity_contract": best_payload_contract_ok,
        "source_fit_query_manifest_618_plus_198_contract": query_manifest_contract_ok,
        "full_source_train_pool_618_queries_95_parents": full_train_pool_contract_ok,
    }
    if (not bad_grad_rejected or not bad_parameter_rejected or not bad_config_rejected
            or not sourcefit_digest_matches or not best_payload_contract_ok or not query_manifest_contract_ok
            or not full_train_pool_contract_ok):
        failures.append("numeric guard or final source-fit cross-stage contract failed")

    if failures:
        raise AssertionError("; ".join(failures))
    return {"status": "passed", "device": "cpu", "checks": checks,
            "target_data_read": False, "GT_read": False, "source_statistics_computed": False,
            "GPU_started": False,
            "interpretation": "CPU mock/interface engineering only; no 4B model, project video, source fit or target inference executed"}


def source_stats_main(args: argparse.Namespace) -> None:
    run_id = args.run_id
    def action(carrier):
        args.session = carrier.session
        return stats_command(args)
    result = run_with_session("source_stats", run_id, action)
    print(json.dumps(result, indent=2))


def run_main(args: argparse.Namespace) -> None:
    run_id = args.run_id
    def action(carrier):
        args.session = carrier.session
        return run_command(args)
    result = run_with_session("target_or_sourceval_unlabeled_inference", run_id, action)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-check", help="CPU mocks only; no dataset or GPU")
    check.add_argument("--output", type=Path, default=OUT / "CPU_CHECK.json")
    stats = sub.add_parser("source-stats", help="capture source-train query moments from an explicitly selected locked manifest; GPU action")
    stats.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    stats.add_argument("--input-manifest", type=Path, required=True,
                       help="explicitly choose locked SOURCE_INPUTS.json (618 train queries) or SOURCE_INITIAL_INPUTS.json (legacy 95-query subset)")
    stats.add_argument("--split", choices=("train",), required=True)
    stats.add_argument("--run-id", required=True)
    stats.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="shared unlabeled sourceval/target inference driver; GPU action")
    run.add_argument("--input-manifest", type=Path, required=True)
    run.add_argument("--split", required=True)
    run.add_argument("--condition", choices=("clean", "noise_medium", "defocus_extreme"), required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--source-stats", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--architecture", choices=ARCHITECTURES)
    args = parser.parse_args()
    if args.command == "self-check":
        result = run_cpu_checks()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "checks": sorted(result["checks"]), "GPU_started": False}, indent=2))
    elif args.command == "source-stats":
        source_stats_main(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
