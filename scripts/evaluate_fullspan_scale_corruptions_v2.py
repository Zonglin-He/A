#!/usr/bin/env python3
"""Evaluate clean-selected Fullspan+scale TTA on three HC corruptions (new runner).

This is a source-disjoint re-evaluation of the 47-query HC roster in
``spatial_scale_replication_v1``.  Parameters are read only from the clean
development ``hc_to_hc/selected.json``; no corruption-specific selection is
performed.  Legal fits see frozen model features and source statistics only.
GT/annotations are stored separately and are passed to the metric evaluator
only after every prediction for a query has been produced.

The historical TENT/MEMO/SAR rows are copied (not rerun) from the complete
baseline-expansion cells on exactly the same 47 indices.  The independent
non-blocking GPU guard is the shared research GPU lock. ``prepare`` and
``analyze`` are CPU-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analyze_baseline_expansion_v1 import summarize
from scripts.run_feasibility import load_full_video, prediction_record
from vg_tta.baseline_expansion_data import lift_prediction, make_dataset, shift_pixels
from vg_tta.fullspan_tta import fit_fullspan_head, replay_temporal_head
from vg_tta.geometric_video_io import normalize_raw_view
from vg_tta.metrics import compute_stvg_metrics, interval_from_logits
from vg_tta.spatial_affine import AffineBoxHead, AffineConfig, boxes_from_features, cached_features
from vg_tta.spatial_affine_variance import fit_variance_affine
from vg_tta.tubedetr_runtime import (
    add_repo_to_path,
    build_model,
    decode_video,
    encode_video,
    load_official_checkpoint,
)


TUNE = ROOT / "artifacts/fullspan_scale_optuna_v2"
REPLICATION = ROOT / "artifacts/spatial_scale_replication_v1"
SOURCE = ROOT / "artifacts/spatial_affine_v1"
BASELINE = ROOT / "artifacts/baseline_expansion_v1"
OUT = ROOT / "artifacts/fullspan_scale_corruptions_v2"
GPU_LOCK = ROOT / "artifacts/spatial_tta_research_v2/gpu.lock"
CONDITIONS = ("blur3", "low_light3", "subsample2")
NEW_METHODS = (
    "frozen",
    "temporal_only_selected",
    "spatial_only_selected",
    "fullspan_scale_selected",
    "direct_fullspan_native_boxes",
    "direct_fullspan_adapted_boxes",
)
SELECTED_KEYS = (
    "spatial_lr", "spatial_steps", "spatial_gamma", "spatial_rank",
    "temporal_lr", "temporal_steps", "temporal_gamma",
)
METRICS = ("vIoU_corrected", "sIoU", "tIoU", "vIoU_legacy")


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def write(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delta_stats(records: list[dict[str, Any]], reference: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, base in zip(records, reference):
        if (row["source"], row["sample_index"]) != (base["source"], base["sample_index"]):
            raise AssertionError("delta rows are not query/source aligned")
        grouped[row["source"]].append(100.0 * (float(row[metric]) - float(base[metric])))
    values = np.asarray([np.mean(grouped[source]) for source in sorted(grouped)], dtype=float)
    if not len(values):
        raise ValueError("cannot bootstrap an empty source set")
    rng = np.random.default_rng(20260907)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    trimmed = np.sort(values)[int(.1 * len(values)):len(values) - int(.1 * len(values))]
    return {"mean_pp": float(values.mean()), "median_pp": float(np.median(values)),
            "trimmed_mean_pp": float(trimmed.mean()), "ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
            "win_neutral_loss_sources": [int((values > .1).sum()), int((np.abs(values) <= .1).sum()), int((values < -.1).sum())]}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _raw_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(json.dumps(list(value.shape)).encode()); h.update(value.tobytes())
    return h.hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"selected {name} is non-finite")
    return result


def selected_config() -> tuple[dict[str, Any], Path]:
    path = TUNE / "hc_to_hc" / "selected.json"
    if not path.is_file():
        raise FileNotFoundError(f"clean selected configuration is missing: {path}")
    result = read(path)
    if result.get("status") != "complete" or result.get("development_only") is not True:
        raise RuntimeError("hc_to_hc selected.json is not a complete development result")
    if result.get("test_used_for_tuning") is not False:
        raise RuntimeError("selected configuration does not prove test-free selection")
    tune_lock = TUNE / "lock.json"
    if not tune_lock.is_file() or result.get("lock_sha256") != sha(tune_lock):
        raise RuntimeError("selected.json is stale or does not match the frozen Optuna lock")
    raw = result.get("selected_config")
    if not isinstance(raw, dict) or set(raw) != set(SELECTED_KEYS):
        raise ValueError(f"selected_config must contain exactly {SELECTED_KEYS}")
    config: dict[str, Any] = {}
    for key in SELECTED_KEYS:
        value = raw[key]
        if key.endswith("_steps") or key.endswith("_rank"):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"invalid selected integer {key}: {value!r}")
            config[key] = int(value)
        else:
            value = _finite(value, key)
            if value < 0:
                raise ValueError(f"invalid selected non-negative value {key}: {value!r}")
            config[key] = value
    if config["spatial_steps"] == 0 or config["temporal_steps"] == 0:
        raise RuntimeError("selected clean configuration is a no-op; refusing corruption evaluation")
    return config, path


def _resolve_video(spec: dict[str, Any], row: dict[str, Any]) -> Path:
    value = Path(str(row["video_path"]))
    candidates = [value] if value.is_absolute() else []
    candidates += [Path(spec["root"]) / "video" / value, Path(spec["root"]) / value]
    for path in candidates:
        if path.is_file():
            return path  # manifest keys intentionally preserve literal symlink paths
    raise FileNotFoundError(f"cannot resolve video_path {row['video_path']!r}")


def _roster(rep_lock: dict[str, Any]) -> dict[str, Any]:
    spec = rep_lock["datasets"]["hc"]
    indices = [int(x) for x in spec["indices"]]
    sources = [spec["sources"][str(i)] for i in indices]
    if len(indices) != 47 or len(set(indices)) != 47 or len(set(sources)) != 47:
        raise AssertionError("replication HC roster is not the locked 47 distinct-source roster")
    return {"indices": indices, "sources": {str(i): spec["sources"][str(i)] for i in indices}, "source_count": 47,
            "source_stat_overlap_already_removed_by_replication_lock": True}


def _validate_old(rep_lock: dict[str, Any], base_lock: dict[str, Any]) -> dict[str, Any]:
    if base_lock.get("checkpoint_sha256", {}).get("hc") != rep_lock["checkpoint_sha256"]:
        raise AssertionError("old baseline HC checkpoint differs from replication checkpoint")
    rep_spec = rep_lock["datasets"]["hc"]
    base_indices = {int(x) for x in base_lock["datasets"]["hc"]["indices"]}
    indices = [int(x) for x in rep_spec["indices"]]
    if not set(indices).issubset(base_indices):
        raise AssertionError("old baseline does not contain all replication HC indices")
    if len(base_indices) != 64 or len(base_indices - set(indices)) != 17:
        raise AssertionError("the 47-query roster is not the declared 64-minus-17 source-stat exclusion")
    old_lock_sha = sha(BASELINE / "lock.json")
    for condition in CONDITIONS:
        cell = BASELINE / f"hc_to_hc_{condition}"
        complete = cell / "complete.json"
        if not complete.is_file():
            raise FileNotFoundError(f"historical baseline cell is incomplete: {complete}")
        for index in indices:
            path = cell / "episodes" / f"{index:06d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"historical baseline episode missing: {path}")
            episode = read(path)
            if episode.get("lock_sha256") != old_lock_sha or episode.get("index") != index:
                raise AssertionError(f"historical baseline lock/index mismatch: {path}")
            if episode.get("source") != rep_spec["sources"][str(index)]:
                raise AssertionError(f"historical baseline source mismatch: {path}")
            if len([r for r in episode.get("records", []) if r.get("method") == "frozen"]) != 1:
                raise AssertionError(f"historical frozen record missing: {path}")
            if any(r.get("gt_used") is not False for r in episode["records"]):
                raise AssertionError(f"historical record is marked GT-used: {path}")
    return {"lock_sha256": old_lock_sha, "aligned_queries_per_cell": len(indices),
            "cells": [f"hc_to_hc_{c}" for c in CONDITIONS],
            "policy": "historical rows copied, not rerun or retuned"}


def initialize(out: Path = OUT) -> dict[str, Any]:
    """Write/validate a CPU manifest; this never initializes CUDA."""
    out = Path(out).resolve()
    config, selected_path = selected_config()
    rep_lock = read(REPLICATION / "lock.json")
    base_lock = read(BASELINE / "lock.json")
    roster = _roster(rep_lock)
    historical = _validate_old(rep_lock, base_lock)
    if sha(rep_lock["checkpoint"]) != rep_lock["checkpoint_sha256"]:
        raise AssertionError("HC checkpoint hash changed")
    source_complete = read(SOURCE / "source_complete.json")
    if sha(SOURCE / "source_statistics.pt") != source_complete["stats_sha256"] or sha(SOURCE / "source_bbox_head.pt") != source_complete["head_sha256"]:
        raise AssertionError("source statistics/head hash changed")
    spec = rep_lock["datasets"]["hc"]
    if sha(spec["annotation"]) != spec["annotation_sha256"]:
        raise AssertionError("HC annotation hash changed")
    dependencies = [
        Path(__file__), ROOT / "vg_tta/baseline_expansion_data.py", ROOT / "vg_tta/geometric_video_io.py",
        ROOT / "vg_tta/tubedetr_runtime.py", ROOT / "vg_tta/fullspan_tta.py", ROOT / "vg_tta/spatial_affine.py",
        ROOT / "vg_tta/spatial_affine_variance.py", ROOT / "vg_tta/metrics.py", ROOT / "scripts/run_feasibility.py",
    ]
    lock = {
        "status": "prepared",
        "protocol": {
            "name": "fullspan_scale_corruptions_v2", "checkpoint": "HC -> HC",
            "conditions": list(CONDITIONS), "query_roster": "replication HC47; source-stat overlap 17 already excluded",
            "selection": "single clean-selected hc_to_hc config; no corruption tuning",
            "adaptation": "single private original view; source-stat variance affine boxes + fullspan temporal head",
            "gt_usage": "separate labels; metrics only after all legal predictions",
            "historical_baselines": "baseline_expansion rows copied on the same 47-query subset",
        },
        "selected_config": config, "selected_json": str(selected_path), "selected_json_sha256": sha(selected_path),
        "replication_lock_sha256": sha(REPLICATION / "lock.json"), "baseline_lock_sha256": sha(BASELINE / "lock.json"),
        "source_complete_sha256": sha(SOURCE / "source_complete.json"),
        "checkpoint": rep_lock["checkpoint"], "checkpoint_sha256": rep_lock["checkpoint_sha256"],
        "dataset": {**copy.deepcopy(spec), **roster},
        "historical_baseline": historical, "code_sha256": {str(path): sha(path) for path in dependencies},
        "seed": 20260907, "bootstrap_samples": 10000,
        "timing": {
            "runtime_sec_includes": "shared encoder+frozen decoder plus method feature extraction, private head fit and replay",
            "runtime_sec_excludes": "video loading, CPU normalization, timestamp lifting, metrics, cache I/O and audit overhead; not end-to-end deployment latency",
            "shared_encoder_disclosure": "shared encoder/decoder cost is charged in each method runtime and reported once per episode",
            "native_reinsertion": "first query of each corruption only; extra decode is separate",
            "cache_not_free": True,
        },
        "shared_gpu_lock": str(GPU_LOCK),
    }
    path = out / "lock.json"
    if path.exists():
        if read(path) != lock:
            raise RuntimeError(f"sealed manifest changed; use a new output directory: {path}")
    else:
        write(path, lock)
    return lock


def _sync(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _timed(device: torch.device | str, function):
    _sync(device)
    start = time.perf_counter()
    value = function()
    _sync(device)
    return value, time.perf_counter() - start


def _snapshot(model: torch.nn.Module) -> dict[str, Any]:
    return {"parameters": {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
            "buffers": {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
            "versions": {n: int(p._version) for n, p in model.named_parameters()},
            "sted_id": id(model.sted_embed), "bbox_id": id(model.bbox_embed),
            "modes": {n: m.training for n, m in model.named_modules()}}


def _assert_unchanged(model: torch.nn.Module, snap: dict[str, Any]) -> None:
    if {n: m.training for n, m in model.named_modules()} != snap["modes"]:
        raise AssertionError("source module modes changed")
    if id(model.sted_embed) != snap["sted_id"] or id(model.bbox_embed) != snap["bbox_id"]:
        raise AssertionError("native model head was not restored")
    for name, p in model.named_parameters():
        if int(p._version) != snap["versions"][name] or not torch.equal(p.detach().cpu(), snap["parameters"][name]):
            raise AssertionError(f"cold source model parameter changed: {name}")
        if p.grad is not None:
            raise AssertionError(f"cold source model parameter received gradient: {name}")
    for name, b in model.named_buffers():
        if not torch.equal(b.detach().cpu(), snap["buffers"][name]):
            raise AssertionError(f"cold source model buffer changed: {name}")


def _decode_capture(model, memory, duration: int, caption: str, device: str):
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        if len(inputs) != 1:
            raise RuntimeError("unexpected temporal-head input signature")
        captured.append(inputs[0].detach())

    handle = model.sted_embed.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            outputs = decode_video(model, memory, duration=duration, caption=caption, device=device, use_bf16=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one temporal-head capture, observed {len(captured)}")
    hidden = captured[0]
    if hidden.ndim != 4 or tuple(hidden.shape[1:3]) != (1, duration):
        raise RuntimeError(f"unexpected head input shape {tuple(hidden.shape)}")
    if outputs["pred_sted"].shape[1] != duration:
        raise RuntimeError("head input and native temporal logits are not aligned")
    return outputs, hidden


def _native_reinsert(model, memory, adapted_temporal, adapted_spatial, duration: int, caption: str, device: str):
    original_temporal, original_spatial = model.sted_embed, model.bbox_embed
    model.sted_embed, model.bbox_embed = adapted_temporal, adapted_spatial
    try:
        with torch.no_grad():
            return _timed(device, lambda: decode_video(model, memory, duration=duration, caption=caption, device=device, use_bf16=True))
    finally:
        model.sted_embed, model.bbox_embed = original_temporal, original_spatial


def _fullspan_logits(logits: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(logits)
    result[0, 0, 0] = 20
    result[0, -1, 1] = 20
    return result


def _write_torch(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha(path)


def _lift(boxes, logits, positions, frame_ids):
    result = lift_prediction({"pred_boxes": boxes, "pred_sted": logits}, positions, frame_ids)
    return result["pred_boxes"], result["pred_sted"]


def _replay_metrics(record: dict[str, Any]) -> dict[str, float]:
    payload = record["metric_replay"]
    metrics = compute_stvg_metrics(
        torch.tensor(payload["pred_boxes"], dtype=torch.float32),
        [{"boxes": torch.tensor(x, dtype=torch.float32).reshape(-1, 4)} for x in payload["target_boxes"]],
        interval_from_logits(torch.tensor(payload["pred_sted"], dtype=torch.float64)),
        tuple(payload["gt_indices"]), frame_ids=payload["frame_ids"], gt_frame_interval=payload["gt_frame_interval"],
    )
    return {key: float(metrics[key]) for key in METRICS}


def _assert_metric_replay(record: dict[str, Any]) -> None:
    actual = _replay_metrics(record)
    for key in METRICS:
        if abs(actual[key] - float(record[key])) > 1e-7:
            raise AssertionError(f"metric replay mismatch {record['method']} {key}")


def _compare_frozen(fresh: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    a, b = fresh["metric_replay"], old["metric_replay"]
    if a["frame_ids"] != b["frame_ids"] or a["gt_indices"] != b["gt_indices"]:
        raise AssertionError("fresh/historical frame grid mismatch")
    boxes_a, boxes_b = torch.tensor(a["pred_boxes"], dtype=torch.float64), torch.tensor(b["pred_boxes"], dtype=torch.float64)
    logits_a, logits_b = torch.tensor(a["pred_sted"], dtype=torch.float64), torch.tensor(b["pred_sted"], dtype=torch.float64)
    box_diff = float((boxes_a - boxes_b).abs().max()) if boxes_a.numel() else 0.0
    logit_diff = float((logits_a - logits_b).abs().max()) if logits_a.numel() else 0.0
    exact = a["pred_boxes"] == b["pred_boxes"] and a["pred_sted"] == b["pred_sted"]
    metric_exact = all(abs(float(fresh[k]) - float(old[k])) <= 1e-7 for k in METRICS)
    if not exact or not metric_exact:
        raise AssertionError(f"fresh frozen differs from old baseline: exact={exact}, metrics={metric_exact}, box={box_diff}, logits={logit_diff}")
    return {"prediction_exact": bool(exact), "metric_exact": bool(metric_exact), "max_abs_box_diff": box_diff, "max_abs_logit_diff": logit_diff}


def _old_episode(condition: str, index: int):
    path = BASELINE / f"hc_to_hc_{condition}" / "episodes" / f"{index:06d}.json"
    return read(path), path


def _episode_predictions(model, source_head, stats, dataset, rows, capture, spec, condition, index,
                         config, lock_sha, out, snap, device, first_for_condition):
    source = spec["sources"][str(index)]
    row = rows[index]
    video_path = _resolve_video(spec, row)
    expected = spec.get("video_sha256", {}).get(str(video_path))
    if expected is None or sha(video_path) != expected:
        raise AssertionError(f"video hash mismatch/missing in replication lock: {video_path}")

    sample = dataset[index]
    clean_video, targets, video_target = load_full_video(sample)
    raw = np.asarray(capture.raw)
    if raw.ndim != 4 or len(raw) != int(clean_video.shape[1]):
        raise AssertionError(f"raw/label frame count mismatch at {condition}/{index}")
    shifted, positions = shift_pixels(raw, condition)
    old, old_path = _old_episode(condition, index)
    if (hashlib.sha256(raw).hexdigest() != old["raw_pixel_sha256"] or
            hashlib.sha256(shifted).hexdigest() != old["shifted_pixel_sha256"]):
        raise AssertionError("raw/corrupted pixels differ from the historical baseline")
    video = normalize_raw_view(shifted, resolution=224)
    if int(video.shape[1]) != len(positions):
        raise AssertionError("corrupted input and position grid disagree")
    caption = str(video_target["caption"])

    labels_path = out / condition / "labels_only" / f"{index:06d}.pt"
    label_payload = {"index": index, "source": source, "condition": condition, "targets": targets,
                     "video_target": video_target, "annotation": row, "gt_used_for_adaptation": False}
    if labels_path.exists():
        cached_labels = load(labels_path)
        if (cached_labels.get("index") != index or cached_labels.get("source") != source or
                cached_labels.get("condition") != condition or cached_labels.get("gt_used_for_adaptation") is not False):
            raise AssertionError(f"label cache identity/GT flag mismatch: {labels_path}")
        label_sha = sha(labels_path)
    else:
        label_sha = _write_torch(labels_path, label_payload)

    memory, enc_sec = _timed(device, lambda: encode_video(model, video, caption, repo=ROOT / "external/TubeDETR",
                                                            stride=2, device=device, use_bf16=True))
    (native, hidden), dec_sec = _timed(device, lambda: _decode_capture(model, memory, int(video.shape[1]), caption, device))
    native_boxes, native_logits = native["pred_boxes"].detach(), native["pred_sted"].detach()
    feature_head = AffineBoxHead(source_head)
    z, feature_sec = _timed(device, lambda: cached_features(feature_head, hidden.flatten(1, 2)))
    with torch.no_grad():
        identity_boxes = boxes_from_features(feature_head, z)
    if not torch.equal(identity_boxes, native_boxes):
        raise AssertionError("identity bbox replay is not native-exact")

    cache_path = out / condition / "cache" / f"{index:06d}.pt"
    cache_sha = _write_torch(cache_path, {
        "index": index, "source": source, "condition": condition, "caption": caption,
        "positions": positions, "raw_sha256": _raw_digest(raw), "shifted_raw_sha256": _raw_digest(shifted),
        # Keep both the descriptive names and the names used by the existing
        # cache readers.  They are the same tensors, saved once per query.
        "hidden_head_input": hidden.detach().cpu(), "head_input": hidden.detach().cpu(),
        "spatial_features": z.detach().cpu(), "features": z.detach().cpu(),
        "frozen_sampled": {"pred_boxes": native_boxes.detach().cpu(), "pred_sted": native_logits.detach().cpu()},
        "labels_used": False, "gt_used": False, "lock_sha256": lock_sha,
    })

    spatial_head = AffineBoxHead(source_head)
    spatial_cfg = AffineConfig("subspace", int(config["spatial_rank"]), float(config["spatial_lr"]),
                               int(config["spatial_steps"]), float(config["spatial_gamma"]))
    spatial_audit, spatial_sec = _timed(device, lambda: fit_variance_affine(spatial_head, z, stats, spatial_cfg))
    if spatial_audit.get("nonfinite_fallback"):
        raise RuntimeError("selected spatial fit used a nonfinite fallback")
    with torch.no_grad():
        adapted_boxes, spatial_replay_sec = _timed(device, lambda: boxes_from_features(spatial_head, z).detach())
    spatial_sec += spatial_replay_sec

    temporal_head = copy.deepcopy(model.sted_embed).eval()
    temporal_audit, temporal_sec = _timed(device, lambda: fit_fullspan_head(
        temporal_head, [{"head_input": hidden}] * int(config["temporal_steps"]),
        lr=float(config["temporal_lr"]), anchor_gamma=float(config["temporal_gamma"]), optimizer_eps=1e-4))
    if not temporal_audit["audit"]["all_steps_finite"]:
        raise RuntimeError("selected temporal fit produced a nonfinite step")
    with torch.no_grad():
        adapted_logits, temporal_replay_sec = _timed(device, lambda: replay_temporal_head(temporal_head, hidden, device).detach())
    temporal_sec += temporal_replay_sec
    full_logits = _fullspan_logits(native_logits)

    native_reinsert = {"checked": False, "prediction_exact": None, "boxes_exact": None, "seconds": 0.0,
                       "max_abs_logit_diff": None, "max_abs_box_diff": None}
    if first_for_condition:
        native_adapted, native_sec = _native_reinsert(model, memory, temporal_head, spatial_head,
                                                      int(video.shape[1]), caption, device)
        max_diff = float((native_adapted["pred_sted"].float() - adapted_logits.float()).abs().max())
        max_box_diff = float((native_adapted["pred_boxes"].float() - adapted_boxes.float()).abs().max())
        exact = bool(torch.equal(native_adapted["pred_sted"], adapted_logits))
        boxes_exact = bool(torch.equal(native_adapted["pred_boxes"], adapted_boxes))
        native_reinsert = {"checked": True, "prediction_exact": exact, "boxes_exact": boxes_exact,
                           "seconds": native_sec, "max_abs_logit_diff": max_diff,
                           "max_abs_box_diff": max_box_diff, "meaning": "joint private heads inserted into native final forward"}
        if not exact or not boxes_exact:
            raise AssertionError(f"native temporal reinsertion mismatch: {native_reinsert}")

    noop = {"checked": False, "spatial_exact": None, "temporal_exact": None}
    if first_for_condition:
        identity = AffineBoxHead(source_head)
        with torch.no_grad():
            no_boxes = boxes_from_features(identity, z)
            no_logits = replay_temporal_head(copy.deepcopy(model.sted_embed).eval(), hidden, device)
        noop = {"checked": True, "spatial_exact": bool(torch.equal(no_boxes, native_boxes)),
                "temporal_exact": bool(torch.equal(no_logits, native_logits)),
                "meaning": "identity controls, not useful adaptation evidence"}
        if not noop["spatial_exact"] or not noop["temporal_exact"]:
            raise AssertionError(f"no-op control failed: {noop}")
        noop["full_path_controls"] = {}
        for control, lr_zero, zero_steps in [("lr0", True, False), ("steps0", False, True)]:
            sh = AffineBoxHead(source_head)
            sc = AffineConfig("subspace", int(config["spatial_rank"]),
                              0.0 if lr_zero else float(config["spatial_lr"]),
                              0 if zero_steps else int(config["spatial_steps"]), float(config["spatial_gamma"]))
            sa = fit_variance_affine(sh, z, stats, sc)
            th = copy.deepcopy(model.sted_embed).eval()
            ta = fit_fullspan_head(th, [{"head_input": hidden}] * (0 if zero_steps else int(config["temporal_steps"])),
                                  lr=0.0 if lr_zero else float(config["temporal_lr"]),
                                  anchor_gamma=float(config["temporal_gamma"]), optimizer_eps=1e-4)
            with torch.no_grad():
                exact_s = torch.equal(boxes_from_features(sh, z), native_boxes)
                exact_t = torch.equal(replay_temporal_head(th, hidden, device), native_logits)
            if not exact_s or not exact_t:
                raise AssertionError(f"full-path {control} differs from frozen")
            noop["full_path_controls"][control] = {"spatial_exact": exact_s, "temporal_exact": exact_t,
                                                         "spatial_audit": sa, "temporal_audit": ta}

    outputs = {
        "frozen": (native_boxes, native_logits),
        "temporal_only_selected": (native_boxes, adapted_logits),
        "spatial_only_selected": (adapted_boxes, native_logits),
        "fullspan_scale_selected": (adapted_boxes, adapted_logits),
        "direct_fullspan_native_boxes": (native_boxes, full_logits),
        "direct_fullspan_adapted_boxes": (adapted_boxes, full_logits),
    }
    lifted = {name: _lift(boxes, logits, positions, list(video_target["frames_id"])) for name, (boxes, logits) in outputs.items()}
    _assert_unchanged(model, snap)

    old, old_path = _old_episode(condition, index)
    old_frozen = next(r for r in old["records"] if r["method"] == "frozen")
    shared = enc_sec + dec_sec
    runtimes = {
        "frozen": shared, "temporal_only_selected": shared + temporal_sec,
        "spatial_only_selected": shared + feature_sec + spatial_sec,
        "fullspan_scale_selected": shared + feature_sec + spatial_sec + temporal_sec,
        "direct_fullspan_native_boxes": shared, "direct_fullspan_adapted_boxes": shared + feature_sec + spatial_sec,
    }
    timing = {
        "frozen": {"shared_encoder_seconds": shared, "head_fit_seconds": 0.0},
        "temporal_only_selected": {"shared_encoder_seconds": shared, "head_fit_seconds": temporal_sec},
        "spatial_only_selected": {"shared_encoder_seconds": shared, "feature_cache_seconds": feature_sec, "head_fit_seconds": spatial_sec},
        "fullspan_scale_selected": {"shared_encoder_seconds": shared, "feature_cache_seconds": feature_sec, "head_fit_seconds": spatial_sec + temporal_sec},
        "direct_fullspan_native_boxes": {"shared_encoder_seconds": shared, "head_fit_seconds": 0.0, "control": True},
        "direct_fullspan_adapted_boxes": {"shared_encoder_seconds": shared, "feature_cache_seconds": feature_sec, "head_fit_seconds": spatial_sec, "control": True},
    }
    records = []
    # Deliberately after all legal prediction tensors and native/no-op audits.
    for method in NEW_METHODS:
        boxes, logits = lifted[method]
        record = prediction_record({"pred_boxes": boxes, "pred_sted": logits}, targets, video_target, row,
                                   sample_index=index, condition=condition, method=method,
                                   runtime_sec=runtimes[method], peak_vram_gb=float(torch.cuda.max_memory_allocated() / 2**30),)
        record.update(source=source, gt_used=False, labels_used_for="evaluation_only_after_all_predictions",
                      adaptation_gt_used=False, timing=timing[method])
        records.append(record)
    alignment = _compare_frozen(records[0], old_frozen)
    for record in records:
        _assert_metric_replay(record)
    historical = copy.deepcopy(old["records"])
    for record in historical:
        record["historical_source_cell"] = str(old_path.parent)
        record["historical_record_preserved"] = True
    return {
        "status": "complete", "index": index, "source": source, "condition": condition, "caption": caption,
        "selected_config": config, "selected_json_sha256": sha(TUNE / "hc_to_hc" / "selected.json"),
        "lock_sha256": lock_sha, "cache_sha256": cache_sha, "labels_sha256": label_sha,
        "records": records, "historical_records": historical, "historical_episode_sha256": sha(old_path),
        "fresh_frozen_alignment": alignment,
        "adaptation_audits": {"spatial": spatial_audit, "temporal": temporal_audit},
        "audits": {"gt_used_for_adaptation": False, "labels_used_for_selection": False,
                    "labels_used_for_metrics_only_after_prediction": True,
                    "dataset_loader_read_annotations_before_prediction": True,
                    "source_model_state_exact_after_episode": True,
                    "private_heads_reset_per_query": True, "identity_bbox_replay_exact": True,
                    "native_temporal_reinsertion": native_reinsert, "no_op_controls": noop,
                    "corruption_positions": positions},
        "timing": {"encoder_seconds": enc_sec, "frozen_decoder_seconds": dec_sec, "feature_cache_seconds": feature_sec,
                   "spatial_fit_seconds": spatial_sec, "temporal_fit_seconds": temporal_sec,
                   "native_reinsertion_seconds": native_reinsert["seconds"], "cache_not_free": True},
        "sampled_frame_count": len(positions), "evaluation_frame_count": len(video_target["frames_id"]), "gt_used": False,
    }


def run(out: Path = OUT, *, device: str = "cuda", conditions: Iterable[str] = CONDITIONS, limit: int | None = None) -> None:
    out = Path(out).resolve()
    lock = initialize(out)
    conditions = tuple(conditions)
    if not set(conditions).issubset(set(CONDITIONS)):
        raise ValueError(f"conditions must be a subset of {CONDITIONS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("run requires CUDA; prepare/analyze are CPU-only")
    add_repo_to_path(ROOT / "external/TubeDETR")
    model, _ = build_model(ROOT / "external/TubeDETR", device=device, resolution=224, stride=2)
    if sha(lock["checkpoint"]) != lock["checkpoint_sha256"]:
        raise AssertionError("checkpoint changed after prepare")
    report = load_official_checkpoint(model, lock["checkpoint"])
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"checkpoint incompatibility: {report}")
    model.eval().requires_grad_(False)
    source_head = load(SOURCE / "source_bbox_head.pt").to(torch.device(device)).eval().requires_grad_(False)
    stats = load(SOURCE / "source_statistics.pt")
    spec = lock["dataset"]
    dataset, rows, capture = make_dataset(spec)
    snapshot = _snapshot(model)
    for condition in conditions:
        completed = 0
        for raw_index in spec["indices"]:
            index = int(raw_index)
            if limit is not None and completed >= limit:
                break
            destination = out / condition / "episodes" / f"{index:06d}.json"
            if destination.exists():
                existing = read(destination)
                if existing.get("lock_sha256") != sha(out / "lock.json") or existing.get("selected_config") != lock["selected_config"]:
                    raise AssertionError(f"existing episode is not resumable: {destination}")
                _audit_episode(existing, index, spec["sources"][str(index)], condition, sha(out / "lock.json"))
                _audit_cache(out, existing)
                completed += 1
                continue
            torch.cuda.reset_peak_memory_stats()
            episode = _episode_predictions(model, source_head, stats, dataset, rows, capture, spec, condition,
                                          index, lock["selected_config"], sha(out / "lock.json"), out, snapshot, device,
                                          first_for_condition=(completed == 0))
            write(destination, episode)
            completed += 1
            print(f"[fullspan corruption] {condition} {completed}/{len(spec['indices'])}", flush=True)
        _assert_unchanged(model, snapshot)
    analyze(out)


def _summaries(rows: dict[str, list[dict[str, Any]]], reference: str) -> dict[str, Any]:
    base = rows[reference]
    result = {}
    for method, records in rows.items():
        summary = summarize(records, base)
        summary.pop("mean_runtime_sec_caveated", None)
        summary["source_bootstrap_vs_reference"] = {metric: delta_stats(records, base, metric)
                                                     for metric in ("vIoU_corrected", "sIoU", "tIoU")}
        result[method] = summary
    return result


def _audit_episode(episode: dict[str, Any], index: int, source: str, condition: str, lock_sha: str):
    if episode.get("status") != "complete" or episode.get("lock_sha256") != lock_sha:
        raise AssertionError(f"episode incomplete or unlocked: {condition}/{index}")
    if episode.get("index") != index or episode.get("source") != source or episode.get("condition") != condition:
        raise AssertionError(f"episode identity mismatch: {condition}/{index}")
    if episode.get("gt_used") is not False or episode.get("audits", {}).get("gt_used_for_adaptation") is not False:
        raise AssertionError(f"GT marked as adaptation input: {condition}/{index}")
    records = episode.get("records", [])
    if [r.get("method") for r in records] != list(NEW_METHODS):
        raise AssertionError(f"new method roster mismatch: {condition}/{index}")
    rows = defaultdict(list)
    for record in records:
        if record.get("source") != source or record.get("gt_used") is not False:
            raise AssertionError(f"new record source/GT mismatch: {condition}/{index}")
        _assert_metric_replay(record)
        rows[record["method"]].append(record)
    historical = episode.get("historical_records", [])
    if not historical:
        raise AssertionError(f"historical records missing: {condition}/{index}")
    old_rows = defaultdict(list)
    for record in historical:
        if record.get("source") != source or record.get("gt_used") is not False:
            raise AssertionError(f"historical record source/GT mismatch: {condition}/{index}")
        _assert_metric_replay(record)
        old_rows[f"historical_{record['method']}"].append({**record, "method": f"historical_{record['method']}"})
    return rows, old_rows


def _audit_cache(out: Path, episode: dict[str, Any]) -> None:
    condition, index = episode["condition"], episode["index"]
    for directory, field in [("cache", "cache_sha256"), ("labels_only", "labels_sha256")]:
        path = out / condition / directory / f"{index:06d}.pt"
        if not path.is_file() or sha(path) != episode[field]:
            raise AssertionError(f"cache hash mismatch: {path}")
    item = load(out / condition / "cache" / f"{index:06d}.pt")
    forbidden = {"targets", "annotation", "gt_boxes", "gt_interval", "labels", "video_target"}
    if forbidden & set(item) or item.get("gt_used") is not False or item.get("labels_used") is not False:
        raise AssertionError("legal feature cache contains or uses labels")
    if any(item.get(k) != episode[k] for k in ("index", "source", "condition", "lock_sha256")):
        raise AssertionError("cache identity mismatch")
    old, old_path = _old_episode(condition, index)
    if sha(old_path) != episode["historical_episode_sha256"]:
        raise AssertionError("historical episode hash mismatch")
    if len(old["records"]) != len(episode["historical_records"]):
        raise AssertionError("historical methods incomplete")
    for original, inherited in zip(old["records"], episode["historical_records"]):
        if any(inherited.get(k) != v for k, v in original.items()):
            raise AssertionError("historical baseline content changed")


def analyze(out: Path = OUT) -> dict[str, Any]:
    """Recompute metric replay and source-macro summaries without CUDA."""
    out = Path(out).resolve()
    lock_path = out / "lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"manifest missing; run prepare first: {lock_path}")
    lock = read(lock_path); lock_sha = sha(lock_path)
    for path, expected in lock["code_sha256"].items():
        if sha(Path(path)) != expected:
            raise AssertionError(f"sealed dependency changed: {path}")
    cells, total_records = {}, 0
    for condition in CONDITIONS:
        rows, old_rows = defaultdict(list), defaultdict(list)
        alignments, timings, complete = [], [], 0
        for raw_index in lock["dataset"]["indices"]:
            index = int(raw_index)
            path = out / condition / "episodes" / f"{index:06d}.json"
            if not path.exists():
                continue
            episode = read(path)
            _audit_cache(out, episode)
            new, historical = _audit_episode(episode, index, lock["dataset"]["sources"][str(index)], condition, lock_sha)
            for key, values in new.items(): rows[key].extend(values); total_records += len(values)
            for key, values in historical.items(): old_rows[key].extend(values); total_records += len(values)
            alignments.append(episode["fresh_frozen_alignment"]); timings.append(episode["timing"])
            cache = out / condition / "cache" / f"{index:06d}.pt"
            labels = out / condition / "labels_only" / f"{index:06d}.pt"
            if not cache.is_file() or not labels.is_file() or sha(cache) != episode["cache_sha256"] or sha(labels) != episode["labels_sha256"]:
                raise AssertionError(f"cache/label hash missing or mismatched: {condition}/{index}")
            item = load(cache)
            if item.get("gt_used") is not False or item.get("labels_used") is not False:
                raise AssertionError(f"feature cache marked as label-used: {condition}/{index}")
            complete += 1
        if not complete:
            cells[condition] = {"status": "not_started", "planned_queries": len(lock["dataset"]["indices"]), "completed_queries": 0}
            continue
        if any(len(rows[m]) != complete for m in NEW_METHODS):
            raise AssertionError(f"new method missing rows in {condition}")
        expected_order = [int(i) for i in lock["dataset"]["indices"] if (out / condition / "episodes" / f"{int(i):06d}.json").exists()]
        if [r["sample_index"] for r in rows["frozen"]] != expected_order:
            raise AssertionError(f"query order mismatch in {condition}")
        all_times = timings
        cells[condition] = {
            "status": "complete" if complete == len(lock["dataset"]["indices"]) else "partial",
            "planned_queries": len(lock["dataset"]["indices"]), "completed_queries": complete,
            "sources": len({r["source"] for r in rows["frozen"]}),
            "methods": _summaries(rows, "frozen"),
            "spatial_increment_vs_temporal_only": {metric: delta_stats(rows["fullspan_scale_selected"], rows["temporal_only_selected"], metric)
                                                   for metric in ("vIoU_corrected", "sIoU", "tIoU")},
            "historical_methods": _summaries(old_rows, "historical_frozen"),
            "fresh_frozen_alignment": {
                "queries": len(alignments), "prediction_exact_queries": sum(bool(a["prediction_exact"]) for a in alignments),
                "metric_exact_queries": sum(bool(a["metric_exact"]) for a in alignments),
                "max_abs_box_diff": max(a["max_abs_box_diff"] for a in alignments),
                "max_abs_logit_diff": max(a["max_abs_logit_diff"] for a in alignments),
            },
            "timing": {
                "mean_encoder_seconds": float(np.mean([x["encoder_seconds"] for x in all_times])),
                "mean_frozen_decoder_seconds": float(np.mean([x["frozen_decoder_seconds"] for x in all_times])),
                "mean_feature_cache_seconds": float(np.mean([x["feature_cache_seconds"] for x in all_times])),
                "mean_spatial_fit_seconds": float(np.mean([x["spatial_fit_seconds"] for x in all_times])),
                "mean_temporal_fit_seconds": float(np.mean([x["temporal_fit_seconds"] for x in all_times])),
                "native_reinsertion_checks": sum(bool(x["native_reinsertion_seconds"]) for x in all_times),
                "cache_not_free": True,
            },
            "no_op_controls": {"first_query_each_condition": True, "not_useful_adaptation_evidence": True},
        }
    result = {
        "status": "complete" if all(c["status"] == "complete" for c in cells.values()) else "partial",
        "experiment": str(out), "lock_sha256": lock_sha, "selected_config": lock["selected_config"],
        "selected_json_sha256": lock["selected_json_sha256"], "conditions": cells,
        "metric_records_recomputed": total_records,
        "source_macro_bootstrap": "47 source clusters per condition; paired 10,000-draw source bootstrap, no pooled cross-condition CI",
        "gt_isolation": "No GT in adaptation or condition selection; separate labels, replay metrics after predictions",
        "historical_baseline_policy": "Historical TENT/MEMO/SAR rows are copied and prefixed in summaries; no retuning",
        "clean_selected_applied_unchanged": True, "corruption_tuning": False, "not_a_global_test_claim": True,
        "analyzer_sha256": sha(Path(__file__)),
    }
    write(out / "analysis.json", result)
    return result


def prepare(out: Path = OUT) -> dict[str, Any]:
    lock = initialize(out)
    result = {"status": "prepared", "lock_sha256": sha(out / "lock.json"), "selected_config": lock["selected_config"],
              "planned_conditions": list(CONDITIONS), "planned_queries_per_condition": 47,
              "historical_baseline_aligned": True, "gpu_started": False, "corruption_tuning": False}
    write(out / "prepare.json", result)
    return result


class _NonBlockingGpuLock:
    def __init__(self, path: Path): self.path, self.handle = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close(); self.handle = None
            raise RuntimeError(f"shared GPU lock is busy: {self.path}") from exc
        self.handle.write(json.dumps({"pid": os.getpid(), "script": str(Path(__file__))}) + "\n"); self.handle.flush()
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN); self.handle.close(); self.handle = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(20260907); torch.backends.cudnn.benchmark = False
    if args.action == "prepare":
        print(json.dumps(prepare(args.out.resolve()), indent=2, ensure_ascii=False))
    elif args.action == "analyze":
        print(json.dumps(analyze(args.out.resolve()), indent=2, ensure_ascii=False))
    else:
        out = args.out.resolve()
        with _NonBlockingGpuLock(GPU_LOCK):
            run(out, device=args.device, conditions=args.conditions, limit=args.limit)


if __name__ == "__main__":
    main()
