#!/usr/bin/env python3
"""Fixed TubeDETR Tent/MEMO/SAR successor for the locked HC->Vid shifts.

This file is deliberately a small consumer of the already sealed shift
runner.  It reuses that runner's literal-video resolver, metadata-only ffmpeg
decoder, pixel corruptions, timestamp lift, label builder and metric-record
bridge.  It never imports a shifted metric row or a clean prediction as a
method result.  The only parent prediction used before the prediction barrier
is the parent *native Frozen cache*, and that cache is used solely as an exact
input/reproducibility gate.

TubeDETR has one sampled stride-2 path in this experiment.  There are no
TA-STVG offset views, no crop/high-resolution views, and no fusion.  The
external objectives receive live decoder forwards over an immutable encoder
memory.  Decoder LayerNorm affine parameters are replaced by private
parameters for the duration of each fit; this keeps the source model's
parameter versions and values untouched while still exercising the real
forward graph.  Labels are opened only after all four predictions, first-query
controls, and source reset have completed.

``prepare`` and ``analyze`` are CPU-only.  ``run`` requires CUDA and the
shared research lock.  A limited run must use a separate output directory and
can never write the formal ``complete.json``.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from scripts import evaluate_fullspan_scale_shift_corruptions_v1 as shift


PROTOCOL_PATH = ROOT / "protocols/fullspan_scale_shift_external_v1.json"
PARENT_ROOT = ROOT / "artifacts/fullspan_scale_shift_corruptions_v1"
DEFAULT_OUT = ROOT / "artifacts/fullspan_scale_shift_external_v1"
GPU_LOCK = ROOT / "artifacts/spatial_tta_research_v2/gpu.lock"
TUBE_REPO = ROOT / "external/TubeDETR"

CONDITIONS = ("blur3", "low_light3", "subsample2")
METHODS = ("frozen", "tent", "memo", "sar")
EXTERNAL_METHODS = ("tent", "memo", "sar")
EXPECTED_INDICES = tuple(shift.EXPECTED_INDICES)
METRICS = ("vIoU_corrected", "sIoU", "tIoU", "vIoU_legacy")

# These are the fixed TubeDETR baseline settings from the original external
# port.  They are intentionally not the prospective one-step/2-view cost
# control discussed in the readiness note.
FIXED_ARMS: dict[str, dict[str, Any]] = {
    "tent": {
        "lr": 0.001,
        "steps": 3,
        "optimizer_name": "adam",
        "num_views": 1,
        "momentum": None,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "reset_constant_em": 0.2,
    },
    "memo": {
        "lr": 0.005,
        "steps": 3,
        "optimizer_name": "sgd",
        "num_views": 4,
        "momentum": 0.0,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "reset_constant_em": 0.2,
    },
    "sar": {
        "lr": 0.001,
        "steps": 3,
        "optimizer_name": "sgd",
        "num_views": 1,
        "momentum": 0.9,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "reset_constant_em": 0.2,
    },
}

FORBIDDEN_CACHE_KEYS = frozenset(
    {
        "targets",
        "annotation",
        "trajectory",
        "trajectories",
        "video_target",
        "gt_boxes",
        "gt_interval",
        "gt_indices",
        "labels",
        "target_boxes",
        "tube_start_frame",
        "tube_end_frame",
        "target_id",
    }
)
GT_METADATA_KEYS = frozenset(
    {
        "target_id",
        "tube_start_frame",
        "tube_end_frame",
        "trajectory",
        "trajectories",
        "target_boxes",
        "gt_indices",
        "gt_frame_interval",
    }
)


# The sealed shift runner owns the common file/hash/atomic contracts.  Keep
# them as aliases so this successor cannot drift in lock or sidecar format.
read = shift.read
sha = shift.sha
_jsonable = shift._jsonable
atomic_json = shift.atomic_json
atomic_torch = shift.atomic_torch
load_torch = shift.load_torch


def _object_digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    # NumPy does not expose a bfloat16 dtype on all supported versions.  A
    # byte view preserves the exact tensor representation for every numeric
    # dtype, including the native TubeDETR BF16 outputs.
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or infinity")


def _exact(a: Any, b: Any, *, path: str = "value") -> None:
    """Fail-closed exact comparison for prediction/control artifacts."""

    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
            raise AssertionError(f"{path}: tensor mismatch")
        return
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)) or set(a) != set(b):
            raise AssertionError(f"{path}: mapping mismatch")
        for key in a:
            _exact(a[key], b[key], path=f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            raise AssertionError(f"{path}: sequence mismatch")
        for index, (left, right) in enumerate(zip(a, b)):
            _exact(left, right, path=f"{path}[{index}]")
        return
    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def _protocol() -> dict[str, Any]:
    protocol = read(PROTOCOL_PATH)
    if protocol.get("schema_version") != "fullspan_scale_shift_external_v1":
        raise RuntimeError("unexpected external shift protocol schema")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise RuntimeError("external method roster changed")
    condition_names = tuple(item.get("name") for item in protocol.get("conditions", ()))
    if condition_names != CONDITIONS:
        raise RuntimeError("external condition roster changed")
    target = protocol.get("target", {})
    if target.get("queries") != 192 or target.get("queries_per_condition") != 64:
        raise RuntimeError("external target query budget changed")
    if target.get("source_clusters_per_condition") != 64 or target.get("one_query_per_source") is not True:
        raise RuntimeError("external target source budget changed")
    sampling = protocol.get("sampling", {})
    expected_sampling = {
        "official_fps": 5,
        "max_frames": 200,
        "resolution": 224,
        "model_stride": 2,
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise RuntimeError(f"sampling setting changed: {key}")
    if sampling.get("native_path", "").find("no TA") < 0:
        raise RuntimeError("protocol does not disclose the single TubeDETR path")
    arms = protocol.get("arms", {})
    for method, expected in FIXED_ARMS.items():
        actual = arms.get(method, {})
        for key in ("lr", "steps", "optimizer", "views"):
            if key not in actual:
                raise RuntimeError(f"protocol arm is incomplete: {method}.{key}")
        if float(actual["lr"]) != float(expected["lr"]) or int(actual["steps"]) != int(expected["steps"]):
            raise RuntimeError(f"protocol arm changed: {method} learning settings")
        if int(actual["views"]) != int(expected["num_views"]):
            raise RuntimeError(f"protocol arm changed: {method} view count")
        expected_optimizer = {
            "tent": "Adam",
            "memo": "SGD",
            "sar": "SAM(SGD)",
        }[method]
        if str(actual["optimizer"]) != expected_optimizer:
            raise RuntimeError(f"protocol arm changed: {method} optimizer")
    boundary = protocol.get("legal_boundary", {})
    if boundary.get("gt_used_for_fit") is not False or boundary.get("gt_used_for_selection") is not False:
        raise RuntimeError("external protocol has an invalid GT fit/selection boundary")
    if boundary.get("labels_loaded_after_all_predictions") is not True:
        raise RuntimeError("external protocol lacks the prediction barrier")
    return protocol


def _assert_gt_free_metadata(value: Mapping[str, Any], *, path: str) -> None:
    for key in value:
        if str(key) in GT_METADATA_KEYS:
            raise RuntimeError(f"GT/event metadata leaked into legal metadata at {path}.{key}")
    for key, item in value.items():
        if isinstance(item, Mapping):
            _assert_gt_free_metadata(item, path=f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                if isinstance(child, Mapping):
                    _assert_gt_free_metadata(child, path=f"{path}.{key}[{index}]")


def _check_parent_gate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    """Check only parent structure/pins; never read parent metric rows."""

    lock_path = PARENT_ROOT / "lock.json"
    complete_path = PARENT_ROOT / "complete.json"
    audit_path = PARENT_ROOT / "independent_audit.json"
    for path in (lock_path, complete_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"required parent artifact is missing: {path}")
    parent_lock = read(lock_path)
    parent_lock_sha = sha(lock_path)
    if parent_lock.get("schema_version") != "fullspan_scale_shift_corruptions_v1":
        raise RuntimeError("parent shift lock schema changed")
    if parent_lock.get("status") != "prepared":
        raise RuntimeError("parent shift lock is not prepared")
    parent_complete = read(complete_path)
    if parent_complete.get("status") != "complete":
        raise RuntimeError("parent shift evaluation is not complete")
    if parent_complete.get("lock_sha256") != parent_lock_sha:
        raise RuntimeError("parent complete receipt points at a different lock")
    if parent_complete.get("conditions") != list(CONDITIONS) or parent_complete.get("queries") != 192:
        raise RuntimeError("parent shift query budget/conditions changed")
    if parent_complete.get("records") != 192 * 6:
        raise RuntimeError("parent shift record budget changed")
    parent_audit = read(audit_path)
    if (
        parent_audit.get("schema_version") != "fullspan_scale_shift_corruptions_v1_independent_audit"
        or parent_audit.get("status") != "passed"
        or parent_audit.get("passed") is not True
        or parent_audit.get("formal") is not True
        or parent_audit.get("partial_passed") is not False
        or parent_audit.get("smoke") is not False
    ):
        raise RuntimeError("parent independent shift audit is not a formal pass")
    observed = parent_audit.get("observed", {})
    if observed.get("queries") != 192 or observed.get("episodes") != 192 or observed.get("records") != 1152:
        raise RuntimeError("parent independent audit observed counts are incomplete")
    if parent_audit.get("lock_sha256") != parent_lock_sha:
        raise RuntimeError("parent independent audit lock hash mismatch")
    parent_protocol_path = Path(str(parent_lock.get("protocol_path", "")))
    if not parent_protocol_path.is_file() or parent_audit.get("protocol_sha256") != sha(parent_protocol_path):
        raise RuntimeError("parent independent audit protocol hash mismatch")
    methods = tuple(parent_lock.get("methods", ()))
    expected_parent_methods = (
        "frozen",
        "temporal_only_selected",
        "spatial_only_selected",
        "fullspan_scale_selected",
        "direct_fullspan_native_boxes",
        "direct_fullspan_adapted_boxes",
    )
    if methods != expected_parent_methods:
        raise RuntimeError("parent shift method roster changed")
    if parent_lock.get("external_baselines", {}).get("status") != "pending_safe_port":
        raise RuntimeError("parent external arms are not explicitly pending")

    # The parent lock itself is the source of target metadata.  This avoids
    # opening the annotation/trajectory JSON during prepare and keeps the
    # legal pre-prediction metadata visibly event-free.
    target = parent_lock.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("parent target section is missing")
    if tuple(int(index) for index in target.get("indices", ())) != EXPECTED_INDICES:
        raise RuntimeError("parent target roster differs from the locked Vid64 roster")
    sources = {str(key): str(value) for key, value in target.get("sources", {}).items()}
    if set(sources) != {str(index) for index in EXPECTED_INDICES} or len(set(sources.values())) != 64:
        raise RuntimeError("parent target source map is not one-per-source")
    metadata = target.get("query_metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {str(index) for index in EXPECTED_INDICES}:
        raise RuntimeError("parent target metadata roster is incomplete")
    for index in EXPECTED_INDICES:
        if not isinstance(metadata[str(index)], Mapping):
            raise RuntimeError(f"parent target metadata is malformed: {index}")
        _assert_gt_free_metadata(metadata[str(index)], path=f"target.query_metadata[{index}]")
    annotation = Path(str(target.get("annotation", "")))
    if not annotation.is_file() or sha(annotation) != target.get("annotation_sha256"):
        raise RuntimeError("parent target annotation file/hash changed")
    video_hashes = target.get("video_sha256")
    if not isinstance(video_hashes, Mapping) or len(video_hashes) != 64:
        raise RuntimeError("parent target media hash map is incomplete")
    # Verify literal path identity and media bytes, while retaining the
    # parent's spelling (do not resolve symlinks into a different lock key).
    for index in EXPECTED_INDICES:
        path = shift._resolve_literal_video(target, metadata[str(index)])
        expected_hash = video_hashes.get(str(path))
        if expected_hash is None or not path.is_file() or sha(path) != expected_hash:
            raise RuntimeError(f"parent target media pin failed: {path}")
    checkpoint = Path(str(parent_lock.get("checkpoint", "")))
    if not checkpoint.is_file() or sha(checkpoint) != parent_lock.get("checkpoint_sha256"):
        raise RuntimeError("parent HC checkpoint hash changed")
    runtime = parent_lock.get("runtime", {})
    ffmpeg_path = Path(str(runtime.get("ffmpeg_path", shift.FFMPEG_BIN)))
    ffmpeg_sha = str(runtime.get("ffmpeg_sha256", ""))
    if ffmpeg_path != shift.FFMPEG_BIN or not ffmpeg_path.is_file() or sha(ffmpeg_path) != ffmpeg_sha:
        raise RuntimeError("parent TubeDETR ffmpeg pin is incompatible with sealed decoder")

    # Check the producer's own pinned code.  This is structural lineage only;
    # no parent episode or metric record is opened here.
    for path_text, expected_hash in parent_lock.get("code_sha256", {}).items():
        path = Path(str(path_text))
        if not path.is_file() or sha(path) != expected_hash:
            raise RuntimeError(f"parent pinned code changed: {path}")
    gate = {
        "parent_lock": str(lock_path),
        "parent_lock_sha256": parent_lock_sha,
        "parent_complete": str(complete_path),
        "parent_complete_sha256": sha(complete_path),
        "parent_independent_audit": str(audit_path),
        "parent_independent_audit_sha256": sha(audit_path),
        "parent_protocol": str(parent_protocol_path),
        "parent_protocol_sha256": sha(parent_protocol_path),
        "parent_queries": 192,
        "parent_records": 1152,
        "parent_metrics_read_for_selection": False,
    }
    return parent_lock, parent_complete, gate, lock_path, audit_path


def _parent_reference_caches(parent_lock: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Hash/read label-free Frozen caches only, never parent episodes/labels."""

    references: dict[str, dict[str, dict[str, Any]]] = {}
    target_sources = parent_lock["target"]["sources"]
    for condition in CONDITIONS:
        references[condition] = {}
        for index in EXPECTED_INDICES:
            path = PARENT_ROOT / condition / "cache" / f"{index:06d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"parent Frozen cache is missing: {path}")
            item = load_torch(path)
            if not isinstance(item, Mapping):
                raise RuntimeError(f"parent cache is not a mapping: {path}")
            shift._assert_cache_without_labels(item)
            if item.get("index") != index or item.get("condition") != condition:
                raise RuntimeError(f"parent cache identity mismatch: {path}")
            if item.get("source") != target_sources[str(index)]:
                raise RuntimeError(f"parent cache source mismatch: {path}")
            if item.get("frozen", {}).keys() != {"pred_boxes", "pred_sted"}:
                raise RuntimeError(f"parent cache Frozen output schema changed: {path}")
            boxes, logits = item["frozen"]["pred_boxes"], item["frozen"]["pred_sted"]
            _finite_tensor(boxes, f"parent {condition}/{index} Frozen boxes")
            _finite_tensor(logits, f"parent {condition}/{index} Frozen logits")
            frame_ids = item.get("frame_ids")
            positions = item.get("positions")
            if not isinstance(frame_ids, list) or not isinstance(positions, list) or not frame_ids or not positions:
                raise RuntimeError(f"parent cache frame grid is malformed: {path}")
            references[condition][str(index)] = {
                "path": str(path),
                "cache_sha256": sha(path),
                "source": str(item["source"]),
                "condition": condition,
                "frame_ids": list(frame_ids),
                "positions": list(positions),
                "frame_ids_sha256": _object_digest(frame_ids),
                "positions_sha256": _object_digest(positions),
                "raw_pixel_dtype": item.get("raw_pixel_dtype"),
                "raw_pixel_shape": item.get("raw_pixel_shape"),
                "raw_pixel_sha256": item.get("raw_pixel_sha256"),
                "shifted_pixel_dtype": item.get("shifted_pixel_dtype"),
                "shifted_pixel_shape": item.get("shifted_pixel_shape"),
                "shifted_pixel_sha256": item.get("shifted_pixel_sha256"),
                "frozen_boxes_shape": list(boxes.shape),
                "frozen_logits_shape": list(logits.shape),
                "frozen_boxes_dtype": str(boxes.dtype),
                "frozen_logits_dtype": str(logits.dtype),
                "frozen_boxes_digest": _tensor_digest(boxes),
                "frozen_logits_digest": _tensor_digest(logits),
            }
    return references


def _code_inventory() -> list[Path]:
    return [
        Path(__file__),
        PROTOCOL_PATH,
        ROOT / "scripts/evaluate_fullspan_scale_shift_corruptions_v1.py",
        ROOT / "scripts/run_feasibility.py",
        ROOT / "vg_tta/external_tta_baselines.py",
        ROOT / "vg_tta/tubedetr_runtime.py",
        ROOT / "vg_tta/geometric_video_io.py",
        ROOT / "vg_tta/phase2.py",
        ROOT / "vg_tta/augmentations.py",
        ROOT / "vg_tta/baseline_expansion_data.py",
        TUBE_REPO / "main.py",
        TUBE_REPO / "models/tubedetr.py",
        TUBE_REPO / "models/transformer.py",
        TUBE_REPO / "datasets/video_transforms.py",
        TUBE_REPO / "datasets/vidstg.py",
    ]


def _build_lock() -> dict[str, Any]:
    protocol = _protocol()
    parent_lock, _parent_complete, gate, parent_lock_path, parent_audit_path = _check_parent_gate()
    parent_references = _parent_reference_caches(parent_lock)
    dependencies = _code_inventory()
    if any(not path.is_file() for path in dependencies):
        missing = [str(path) for path in dependencies if not path.is_file()]
        raise FileNotFoundError(f"external runner dependencies are missing: {missing}")
    target_parent = parent_lock["target"]
    target = {
        "kind": target_parent["kind"],
        "root": target_parent["root"],
        "annotation": target_parent["annotation"],
        "annotation_sha256": target_parent["annotation_sha256"],
        "indices": list(EXPECTED_INDICES),
        "sources": {str(index): str(target_parent["sources"][str(index)]) for index in EXPECTED_INDICES},
        "video_sha256": dict(target_parent["video_sha256"]),
        "query_metadata": {
            str(index): copy.deepcopy(target_parent["query_metadata"][str(index)])
            for index in EXPECTED_INDICES
        },
    }
    return {
        "status": "prepared",
        "schema_version": "fullspan_scale_shift_external_v1",
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha(PROTOCOL_PATH),
        "lineage": {
            "parent_shift_root": str(PARENT_ROOT),
            "parent_shift_lock": str(parent_lock_path),
            "parent_shift_lock_sha256": gate["parent_lock_sha256"],
            "parent_shift_complete": str(PARENT_ROOT / "complete.json"),
            "parent_shift_complete_sha256": gate["parent_complete_sha256"],
            "parent_shift_independent_audit": str(parent_audit_path),
            "parent_shift_independent_audit_sha256": gate["parent_independent_audit_sha256"],
            "parent_shift_metrics_copied": False,
            "parent_shift_metrics_read_for_selection": False,
            "historical_exposure": "module-held-out external-baseline evaluation; not globally untouched",
        },
        "checkpoint": parent_lock["checkpoint"],
        "checkpoint_sha256": parent_lock["checkpoint_sha256"],
        "target": target,
        "parent_frozen_cache_sha256": parent_references,
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "arms": copy.deepcopy(FIXED_ARMS),
        "parameter_scope": {
            "selector": "all transformer.decoder.* LayerNorm affine weight/bias parameters",
            "expected_layernorm_modules": 19,
            "expected_tensors": 38,
            "expected_scalar_parameters": 19 * 2 * 256,
            "private_parameter_swap": True,
            "actual_names_recorded_at_run": True,
            "non_scope_trainable": False,
        },
        "sampling": {
            "official_fps": 5,
            "max_frames": 200,
            "resolution": 224,
            "model_stride": 2,
            "one_native_sampled_path": True,
            "native_offsets": False,
            "crop_or_fusion": False,
            "timestamp_mapping": "parent shift helper; original frame IDs and retained positions",
        },
        "legal_boundary": {
            "fit_gt_used": False,
            "selection_gt_used": False,
            "labels_loaded_after_all_predictions": True,
            "labels_in_legal_cache": False,
            "metric_only_after_prediction_barrier": True,
            "no_query_drops": True,
        },
        "controls": {
            "first_query_per_condition": True,
            "names": ["lr0", "steps0"],
            "prediction_exact_to_frozen": True,
            "controls_are_not_metric_arms": True,
        },
        "statistics": {
            "unit": "one query per distinct VidSTG source per condition",
            "bootstrap_seed": 20260907,
            "bootstrap_draws": 10000,
            "configuration_fixed_not_tuned": True,
        },
        "runtime": {
            "prepare_device": "cpu",
            "run_device": "cuda:0",
            "analyze_device": "cpu",
            "global_gpu_lock": str(GPU_LOCK),
            "ffmpeg_path": str(shift.FFMPEG_BIN),
            "ffmpeg_sha256": shift.FFMPEG_SHA256,
            "raw_pixel_transform": "sealed parent shift_pixels",
            "live_decoder": "sealed tubedetr_runtime.decode_video with eval model and BF16 autocast",
            "network": False,
            "cache_not_free": True,
        },
        "code_sha256": {str(path): sha(path) for path in dependencies},
        "artifacts": {
            "output": "artifacts/fullspan_scale_shift_external_v1",
            "lock": "<output>/lock.json",
            "prepare": "<output>/prepare.json",
            "cache": "<output>/<condition>/cache/<index>.pt",
            "labels": "<output>/<condition>/labels_only/<index>.pt",
            "episodes": "<output>/<condition>/episodes/<index>.json",
            "failures": "<output>/<condition>/failures/<index>.json",
            "complete": "<output>/complete.json",
            "analysis": "<output>/analysis.json",
        },
        "protocol_snapshot": {
            "parent_gate": gate,
            "parent_roster_queries": 192,
            "parent_roster_records": 1152,
        },
    }


def initialize(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "lock.json"
    lock = _build_lock()
    if lock_path.exists():
        existing = read(lock_path)
        if existing != lock:
            raise RuntimeError(f"prepared lock differs; use a new output directory: {lock_path}")
    else:
        atomic_json(lock_path, lock)
    return lock


def verify_lock(out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    path = out / "lock.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared lock: {path}")
    lock = read(path)
    expected = _build_lock()
    if expected != lock:
        raise RuntimeError("external prepared lock no longer matches parent/data/code pins")
    for path_text, expected_hash in lock["code_sha256"].items():
        path = Path(path_text)
        if sha(path) != expected_hash:
            raise RuntimeError(f"pinned external dependency changed: {path}")
    return lock


def _episode_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "episodes" / f"{int(index):06d}.json"


def _cache_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "cache" / f"{int(index):06d}.pt"


def _labels_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "labels_only" / f"{int(index):06d}.pt"


def _failure_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "failures" / f"{int(index):06d}.json"


def _write_failure(out: Path, condition: str, index: int, stage: str, exc: BaseException, *, gt_read: bool) -> Path:
    directory = out / condition / "failures"
    path = _failure_path(out, condition, index)
    if path.exists():
        path = directory / f"{int(index):06d}.retry-{time.time_ns()}.json"
    atomic_json(
        path,
        {
            "status": "failed",
            "schema_version": "fullspan_scale_shift_external_v1_failure",
            "condition": condition,
            "index": int(index),
            "stage": str(stage),
            "gt_read_before_failure": bool(gt_read),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "lock_sha256": sha(out / "lock.json"),
        },
    )
    return path


_timed = shift._timed


def _decoder_norm_parameters(model: nn.Module) -> tuple[list[str], list[nn.Parameter]]:
    """Return exactly the TubeDETR decoder LayerNorm affine scope."""

    model.eval().requires_grad_(False)
    names: list[str] = []
    parameters: list[nn.Parameter] = []
    for name, module in model.named_modules():
        if name.startswith("transformer.decoder.") and isinstance(module, nn.LayerNorm):
            for suffix, parameter in module.named_parameters(recurse=False):
                if suffix not in {"weight", "bias"}:
                    raise RuntimeError(f"unexpected decoder LayerNorm parameter: {name}.{suffix}")
                names.append(f"{name}.{suffix}")
                parameters.append(parameter)
    if not parameters:
        raise RuntimeError("no TubeDETR decoder LayerNorm affine parameters found")
    if len(names) != 38 or sum(int(parameter.numel()) for parameter in parameters) != 9728:
        raise RuntimeError(
            "unexpected TubeDETR decoder LayerNorm scope: "
            f"{len(names)} tensors/{sum(int(parameter.numel()) for parameter in parameters)} scalars"
        )
    if any(tuple(parameter.shape) != (256,) for parameter in parameters):
        raise RuntimeError("decoder LayerNorm scope is not 256-dimensional")
    return names, parameters


def _snapshot_full(model: nn.Module) -> dict[str, Any]:
    return {
        "parameters": {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        "parameter_ids": {name: id(parameter) for name, parameter in model.named_parameters()},
        "versions": {name: int(parameter._version) for name, parameter in model.named_parameters()},
        "requires_grad": {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()},
        "grads": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        },
        "buffers": {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()},
        "modes": {name: module.training for name, module in model.named_modules()},
        "sted_id": id(model.sted_embed),
        "bbox_id": id(model.bbox_embed),
    }


def _assert_full_snapshot(model: nn.Module, snapshot: Mapping[str, Any]) -> None:
    current_parameters = dict(model.named_parameters())
    if set(current_parameters) != set(snapshot["parameters"]):
        raise AssertionError("source parameter roster changed")
    if {name: id(parameter) for name, parameter in current_parameters.items()} != snapshot["parameter_ids"]:
        raise AssertionError("source parameter identities changed")
    if {name: module.training for name, module in model.named_modules()} != snapshot["modes"]:
        raise AssertionError("source module modes changed")
    if id(model.sted_embed) != snapshot["sted_id"] or id(model.bbox_embed) != snapshot["bbox_id"]:
        raise AssertionError("native source head identity changed")
    for name, parameter in current_parameters.items():
        if int(parameter._version) != snapshot["versions"][name]:
            raise AssertionError(f"source parameter version changed: {name}")
        if not torch.equal(parameter.detach().cpu(), snapshot["parameters"][name]):
            raise AssertionError(f"source parameter value changed: {name}")
        if bool(parameter.requires_grad) != snapshot["requires_grad"][name]:
            raise AssertionError(f"source parameter requires_grad changed: {name}")
        expected_grad = snapshot["grads"][name]
        actual_grad = None if parameter.grad is None else parameter.grad.detach().cpu()
        if expected_grad is None:
            if actual_grad is not None:
                raise AssertionError(f"source parameter received a gradient: {name}")
        elif actual_grad is None or not torch.equal(actual_grad, expected_grad):
            raise AssertionError(f"source parameter gradient changed: {name}")
    current_buffers = dict(model.named_buffers())
    if set(current_buffers) != set(snapshot["buffers"]):
        raise AssertionError("source buffer roster changed")
    for name, buffer in current_buffers.items():
        if not torch.equal(buffer.detach().cpu(), snapshot["buffers"][name]):
            raise AssertionError(f"source buffer changed: {name}")


def _install_private_scope(model: nn.Module, names: Sequence[str]) -> tuple[list[nn.Parameter], dict[str, nn.Parameter]]:
    """Swap only decoder LN params; original Parameter objects remain pristine."""

    private: list[nn.Parameter] = []
    originals: dict[str, nn.Parameter] = {}
    try:
        for name in names:
            module_name, suffix = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            original = getattr(module, suffix)
            if not isinstance(original, nn.Parameter):
                raise RuntimeError(f"scope member is not a Parameter: {name}")
            if name in originals:
                raise RuntimeError(f"duplicate scope member: {name}")
            clone = nn.Parameter(original.detach().clone(), requires_grad=False)
            originals[name] = original
            setattr(module, suffix, clone)
            private.append(clone)
    except Exception:
        # A malformed scope must not leave a partially swapped source model
        # behind.  This cleanup also keeps the failure receipt truthful about
        # the source-state invariant.
        for name, original in originals.items():
            module_name, suffix = name.rsplit(".", 1)
            setattr(model.get_submodule(module_name), suffix, original)
        raise
    return private, originals


def _remove_private_scope(model: nn.Module, names: Sequence[str], private: Sequence[nn.Parameter], originals: Mapping[str, nn.Parameter]) -> None:
    for name, temporary in zip(names, private):
        module_name, suffix = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        if getattr(module, suffix) is not temporary:
            raise AssertionError(f"private scope parameter was replaced unexpectedly: {name}")
        setattr(module, suffix, originals[name])


def _validate_native_output(
    outputs: Mapping[str, Any], *, name: str, detach: bool = False
) -> dict[str, torch.Tensor]:
    required = {"pred_boxes", "pred_sted"}
    if not required.issubset(set(outputs)):
        raise RuntimeError(f"{name} native output lacks TubeDETR predictions")
    boxes = outputs["pred_boxes"]
    logits = outputs["pred_sted"]
    if not torch.is_tensor(boxes) or not torch.is_tensor(logits):
        raise TypeError(f"{name} native output contains non-tensor predictions")
    _finite_tensor(boxes, f"{name}.pred_boxes")
    _finite_tensor(logits, f"{name}.pred_sted")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise RuntimeError(f"{name} boxes are not T x 4: {tuple(boxes.shape)}")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2 or logits.shape[1] != boxes.shape[0]:
        raise RuntimeError(f"{name} endpoint logits are not 1 x T x 2: {tuple(logits.shape)}")
    if detach:
        boxes = boxes.detach()
        logits = logits.detach()
    return {"pred_boxes": boxes, "pred_sted": logits}


def _native_decode(model: nn.Module, memory: Mapping[str, Any], duration: int, caption: str, device: str) -> dict[str, torch.Tensor]:
    base = shift._lazy_runtime()
    outputs = base.decode_video(
        model,
        memory,
        duration=int(duration),
        caption=caption,
        device=device,
        use_bf16=True,
    )
    # Keep the graph for the live autograd closure.  Callers entering
    # ``torch.no_grad`` receive graph-free tensors naturally; callers that
    # need a durable CPU artifact clone/detach explicitly.
    return _validate_native_output(outputs, name="native", detach=False)


def _fixed_config(method: str, *, control: str | None = None) -> dict[str, Any]:
    if method not in EXTERNAL_METHODS:
        raise ValueError(f"no external config for {method}")
    config = copy.deepcopy(FIXED_ARMS[method])
    if control == "lr0":
        config["lr"] = 0.0
    elif control == "steps0":
        config["steps"] = 0
    elif control is not None:
        raise ValueError(f"unknown control {control}")
    return config


def _deterministic_view_seed(index: int) -> int:
    return 41001 + int(index) * 10


def _fit_one_method(
    model: nn.Module,
    names: Sequence[str],
    memories: Sequence[Mapping[str, Any]],
    method: str,
    config: Mapping[str, Any],
    *,
    duration: int,
    caption: str,
    device: str,
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], float]:
    """Fit private decoder LN parameters, then use a native forward."""

    if method not in EXTERNAL_METHODS:
        raise ValueError(f"unsupported external method: {method}")
    if len(memories) != int(config["num_views"]):
        raise RuntimeError(f"{method} memory/view count mismatch")
    from vg_tta.external_tta_baselines import run_endpoint_tta

    private, originals = _install_private_scope(model, names)
    start = time.perf_counter()
    audit: dict[str, Any] = {}
    try:
        def closure(view_index: int) -> torch.Tensor:
            if not 0 <= int(view_index) < len(memories):
                raise IndexError(f"{method} closure view index out of range: {view_index}")
            # The helper calls this closure under enable_grad for fit passes.
            outputs = _native_decode(model, memories[int(view_index)], duration, caption, device)
            return outputs["pred_sted"]

        audit = run_endpoint_tta(
            private,
            closure,
            method=method,
            lr=float(config["lr"]),
            steps=int(config["steps"]),
            num_views=int(config["num_views"]),
            rho=float(config["rho"]),
            entropy_margin_fraction=float(config["entropy_margin_fraction"]),
            reset=False,
            momentum=config["momentum"],
            reset_constant_em=float(config["reset_constant_em"]),
            optimizer_name=str(config["optimizer_name"]),
        )
        if audit.get("gt_used") is not False:
            raise RuntimeError(f"{method} helper returned an invalid GT flag")
        with torch.no_grad():
            native = _native_decode(model, memories[0], duration, caption, device)
        _finite_tensor(native["pred_boxes"], f"{method} adapted boxes")
        _finite_tensor(native["pred_sted"], f"{method} adapted logits")
        audit = copy.deepcopy(audit)
        audit.update(
            {
                "scope_names": list(names),
                "scope_tensor_count": len(names),
                "scope_scalar_count": 9728,
                "live_decoder_forward": True,
                "native_single_stride2": True,
                "native_offsets_used": False,
                "control": None,
                "gt_used_for_fit": False,
                "elapsed_seconds": float(time.perf_counter() - start),
            }
        )
        return {key: value.detach().cpu().clone() for key, value in native.items()}, audit, float(time.perf_counter() - start)
    finally:
        _remove_private_scope(model, names, private, originals)
        _assert_full_snapshot(model, snapshot)


def _make_memories(
    model: nn.Module,
    video: torch.Tensor,
    caption: str,
    index: int,
    *,
    device: str,
) -> tuple[list[Mapping[str, Any]], float, list[dict[str, Any]]]:
    base = shift._lazy_runtime()
    shift_views = []
    # Original is always the first view.  The three extra views are exactly
    # the old fixed TubeDETR MEMO port's deterministic non-geometric views.
    from vg_tta.phase2 import deterministic_views

    shift_views = deterministic_views(video, seed=_deterministic_view_seed(index), count=4)
    memories: list[Mapping[str, Any]] = []
    total = 0.0
    view_audit: list[dict[str, Any]] = []
    for view_index, view in enumerate(shift_views):
        if view.ndim != 4 or view.shape[0] != 3 or not torch.isfinite(view).all():
            raise RuntimeError(f"invalid MEMO normalized view {view_index}")
        with torch.no_grad():
            memory, elapsed = _timed(
                device,
                lambda view=view: base.encode_video(
                    model,
                    view,
                    caption,
                    repo=TUBE_REPO,
                    stride=2,
                    device=device,
                    use_bf16=True,
                ),
            )
        memories.append(memory)
        elapsed = float(elapsed)
        total += elapsed
        view_audit.append(
            {
                "view_index": int(view_index),
                "role": "original" if view_index == 0 else "deterministic_strong_view",
                "seed": _deterministic_view_seed(index) + max(view_index - 1, 0),
                "geometric": False,
                "frame_count": int(view.shape[1]),
                "encoder_seconds": elapsed,
            }
        )
    return memories, total, view_audit


def _prepare_prediction_bundle(
    model: nn.Module,
    raw: np.ndarray,
    frame_ids: list[int],
    caption: str,
    condition: str,
    index: int,
    *,
    device: str,
    first_for_condition: bool,
    snapshot: Mapping[str, Any],
    parent_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Create all legal predictions and first-query controls, label-free."""

    base = shift._lazy_runtime()
    shifted, positions = shift.shift_pixels(raw, condition)
    if shifted.dtype != np.uint8 or shifted.ndim != 4 or shifted.shape[-1] != 3:
        raise RuntimeError("parent pixel transform did not return uint8 THWC")
    video = base.normalize_raw_view(shifted, resolution=224)
    if int(video.shape[1]) != len(positions):
        raise RuntimeError("shifted video duration/retained-position mismatch")
    if list(parent_reference["frame_ids"]) != list(frame_ids):
        raise RuntimeError("parent Frozen frame grid differs from current metadata grid")
    if list(parent_reference["positions"]) != list(positions):
        raise RuntimeError("parent Frozen shift positions differ from current pixel transform")
    if shift._raw_digest(raw) != parent_reference["raw_pixel_sha256"]:
        raise RuntimeError("current raw pixel digest differs from parent Frozen cache")
    if shift._raw_digest(shifted) != parent_reference["shifted_pixel_sha256"]:
        raise RuntimeError("current shifted pixel digest differs from parent Frozen cache")

    names, _source_parameters = _decoder_norm_parameters(model)
    if any(name.startswith("sted_embed") or name.startswith("bbox_embed") for name in names):
        raise RuntimeError("native source heads entered external adaptation scope")
    duration = int(video.shape[1])
    # Encode all possible MEMO views once while the source model is frozen.
    memories, encoder_seconds, view_audit = _make_memories(
        model, video, caption, index, device=device
    )
    if len(memories) != 4:
        raise RuntimeError("MEMO view construction is not exactly four views")
    with torch.no_grad():
        frozen_native, frozen_seconds = _timed(
            device,
            lambda: _native_decode(model, memories[0], duration, caption, device),
        )
    frozen_native = {key: value.detach().cpu().clone() for key, value in frozen_native.items()}
    old_cache = load_torch(Path(parent_reference["path"]))
    old_frozen = old_cache["frozen"]
    _exact(frozen_native, old_frozen, path=f"parent Frozen {condition}/{index}")
    frozen_lifted = shift.lift_prediction(
        frozen_native["pred_boxes"], frozen_native["pred_sted"], positions, frame_ids
    )
    predictions: dict[str, dict[str, torch.Tensor]] = {
        "frozen": {key: value.detach().cpu().clone() for key, value in frozen_lifted.items()}
    }
    native_predictions: dict[str, dict[str, torch.Tensor]] = {"frozen": frozen_native}
    audits: dict[str, Any] = {
        "parameter_scope_names": names,
        "parameter_scope_tensor_count": len(names),
        "parameter_scope_scalar_count": sum(256 for _ in names),
        "source_model_state_exact_before_methods": True,
        "parent_frozen_native_exact": True,
        "parent_frozen_cache_sha256": parent_reference["cache_sha256"],
        "single_native_stride2_path": True,
        "native_offsets_used": False,
        "memo_views": view_audit,
        "methods": {},
        "controls": {},
        "gt_used_for_adaptation": False,
        "labels_used_for_selection": False,
    }
    timings = {
        "encoder_seconds": float(encoder_seconds),
        "encoder_seconds_by_view": [
            float(item["encoder_seconds"]) for item in view_audit
        ],
        # Frozen, Tent and SAR consume only the original TubeDETR view.  MEMO
        # consumes all four deterministic views.  Encoding is shared in the
        # implementation, but these per-arm attributions prevent the metric
        # runtime field from falsely charging four-view cost to every method.
        "encoder_seconds_by_method": {
            "frozen": float(view_audit[0]["encoder_seconds"]),
            "tent": float(view_audit[0]["encoder_seconds"]),
            "memo": float(encoder_seconds),
            "sar": float(view_audit[0]["encoder_seconds"]),
        },
        "frozen_decoder_seconds": float(frozen_seconds),
        "method_seconds": {},
        "control_seconds": {},
    }
    if not first_for_condition:
        # The MEMO memories remain useful for the three legal methods, while
        # controls are deliberately restricted to the first query of a cell.
        pass

    for method in EXTERNAL_METHODS:
        config = _fixed_config(method)
        method_memories = memories if method == "memo" else memories[:1]
        native, audit, elapsed = _fit_one_method(
            model,
            names,
            method_memories,
            method,
            config,
            duration=duration,
            caption=caption,
            device=device,
            snapshot=snapshot,
        )
        lifted = shift.lift_prediction(native["pred_boxes"], native["pred_sted"], positions, frame_ids)
        native_predictions[method] = native
        predictions[method] = {key: value.detach().cpu().clone() for key, value in lifted.items()}
        audits["methods"][method] = audit
        timings["method_seconds"][method] = float(elapsed)

    if first_for_condition:
        for method in EXTERNAL_METHODS:
            controls: dict[str, Any] = {}
            method_memories = memories if method == "memo" else memories[:1]
            for control_name in ("lr0", "steps0"):
                config = _fixed_config(method, control=control_name)
                native, audit, elapsed = _fit_one_method(
                    model,
                    names,
                    method_memories,
                    method,
                    config,
                    duration=duration,
                    caption=caption,
                    device=device,
                    snapshot=snapshot,
                )
                lifted = shift.lift_prediction(native["pred_boxes"], native["pred_sted"], positions, frame_ids)
                frozen_for_exact = native_predictions["frozen"]
                _exact(native, frozen_for_exact, path=f"{condition}/{index}/{method}/{control_name} native no-op")
                _exact(lifted, predictions["frozen"], path=f"{condition}/{index}/{method}/{control_name} lifted no-op")
                audit = copy.deepcopy(audit)
                audit["control"] = control_name
                audit["prediction_exact_frozen_native"] = True
                controls[control_name] = {
                    "native": native,
                    "lifted": {key: value.detach().cpu().clone() for key, value in lifted.items()},
                    "audit": audit,
                    "elapsed_seconds": float(elapsed),
                }
                timings["control_seconds"][f"{method}:{control_name}"] = float(elapsed)
            audits["controls"][method] = {
                control: {
                    "prediction_exact_frozen_native": True,
                    "prediction_exact_frozen_lifted": True,
                    "audit": payload["audit"],
                }
                for control, payload in controls.items()
            }
            # Keep controls separate from metric methods but in the label-free
            # cache so an independent auditor can inspect actual traces.
            native_predictions.setdefault("controls", {})[method] = {
                control: payload["native"] for control, payload in controls.items()
            }
            predictions.setdefault("controls", {})[method] = {
                control: payload["lifted"] for control, payload in controls.items()
            }

    _assert_full_snapshot(model, snapshot)
    cache = {
        "schema_version": "fullspan_scale_shift_external_v1_cache",
        "index": int(index),
        "source": None,
        "condition": condition,
        "caption": str(caption),
        "frame_ids": list(frame_ids),
        "positions": list(positions),
        "raw_pixel_dtype": str(raw.dtype),
        "raw_pixel_shape": list(raw.shape),
        "raw_pixel_sha256": shift._raw_digest(raw),
        "shifted_pixel_dtype": str(shifted.dtype),
        "shifted_pixel_shape": list(shifted.shape),
        "shifted_pixel_sha256": shift._raw_digest(shifted),
        "parent_reference_cache_sha256": parent_reference["cache_sha256"],
        "parent_reference_frozen_boxes_digest": parent_reference["frozen_boxes_digest"],
        "parent_reference_frozen_logits_digest": parent_reference["frozen_logits_digest"],
        "native_predictions": native_predictions,
        "lifted_predictions": predictions,
        "audits": audits,
        "timing": timings,
        "labels_used": False,
        "gt_used": False,
        "gt_used_for_adaptation": False,
        "selection_performed": False,
    }
    # The cache must not carry a source identity until the caller adds the
    # already-locked source string; no annotation/event row is needed.
    return {
        "cache": cache,
        "native": native_predictions,
        "lifted": predictions,
        "audits": audits,
        "timing": timings,
        "positions": positions,
        "raw": raw,
        "shifted": shifted,
    }


def _cache_contains_forbidden(value: Any, *, path: str = "cache") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_CACHE_KEYS:
                return f"{path}.{key_text}"
            found = _cache_contains_forbidden(child, path=f"{path}.{key_text}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _cache_contains_forbidden(child, path=f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _assert_cache_without_labels(cache: Mapping[str, Any]) -> None:
    found = _cache_contains_forbidden(cache)
    if found is not None:
        raise AssertionError(f"legal external cache contains GT field: {found}")
    if cache.get("labels_used") is not False or cache.get("gt_used") is not False:
        raise AssertionError("legal external cache is marked label-used")
    if cache.get("gt_used_for_adaptation") is not False or cache.get("selection_performed") is not False:
        raise AssertionError("legal external cache has invalid fit/selection flags")
    if tuple(cache.get("native_predictions", {}).keys())[:4] != METHODS:
        raise AssertionError("legal external cache method order changed")
    for method in METHODS:
        native = cache["native_predictions"][method]
        lifted = cache["lifted_predictions"][method]
        if set(native) != {"pred_boxes", "pred_sted"} or set(lifted) != {"pred_boxes", "pred_sted"}:
            raise AssertionError(f"prediction schema changed for {method}")
        _finite_tensor(native["pred_boxes"], f"cache.{method}.pred_boxes")
        _finite_tensor(native["pred_sted"], f"cache.{method}.pred_sted")
        _finite_tensor(lifted["pred_boxes"], f"cache.lifted.{method}.pred_boxes")
        _finite_tensor(lifted["pred_sted"], f"cache.lifted.{method}.pred_sted")


def _run_one(
    out: Path,
    lock: Mapping[str, Any],
    model: nn.Module,
    *,
    condition: str,
    index: int,
    device: str,
    first_for_condition: bool,
    snapshot: Mapping[str, Any],
    stage_state: dict[str, Any],
) -> None:
    metadata = lock["target"]["query_metadata"][str(index)]
    source = lock["target"]["sources"][str(index)]
    stage_state["stage"] = "metadata_decode"
    video_path = shift._resolve_literal_video(lock["target"], metadata)
    expected_video_sha = lock["target"]["video_sha256"].get(str(video_path))
    if expected_video_sha is None or sha(video_path) != expected_video_sha:
        raise RuntimeError(f"target media hash changed: {video_path}")
    raw, frame_ids = shift._decode_raw_metadata_only(metadata, video_path)
    parent_reference = lock["parent_frozen_cache_sha256"][condition][str(index)]
    if shift._raw_digest(raw) != parent_reference["raw_pixel_sha256"]:
        raise RuntimeError(f"raw metadata decode differs from parent cache: {condition}/{index}")
    bundle = _prepare_prediction_bundle(
        model,
        raw,
        frame_ids,
        str(metadata["caption"]),
        condition,
        index,
        device=device,
        first_for_condition=first_for_condition,
        snapshot=snapshot,
        parent_reference=parent_reference,
    )
    stage_state["stage"] = "legal_predictions_complete"
    cache = copy.deepcopy(bundle["cache"])
    cache["source"] = str(source)
    cache["lock_sha256"] = sha(out / "lock.json")
    _assert_cache_without_labels(cache)
    cache_path = _cache_path(out, condition, index)
    if cache_path.exists():
        raise RuntimeError(f"cache exists before query completion: {cache_path}")
    atomic_torch(cache_path, cache)
    cache_sha = sha(cache_path)

    # ---------------- GT barrier: no annotation access above this line. ----------------
    # Opening the annotation and constructing labels is deliberately after
    # the cache write, so a failure in metric preparation cannot be mistaken
    # for a successful legal adaptation.
    stage_state["stage"] = "labels"
    annotation_path = Path(lock["target"]["annotation"])
    if sha(annotation_path) != lock["target"]["annotation_sha256"]:
        raise RuntimeError("target annotation hash changed before post-prediction scoring")
    stage_state["gt_read"] = True
    annotation = read(annotation_path)
    rows = annotation.get("videos")
    if not isinstance(rows, list) or index >= len(rows):
        raise RuntimeError("target annotation rows are missing after prediction barrier")
    row = rows[index]
    if shift._target_metadata(row, index) != metadata:
        raise RuntimeError("post-barrier metadata identity differs from prepared lock")
    targets, video_target = shift._build_labels(annotation, row, raw, frame_ids)
    labels_payload = {
        "schema_version": "fullspan_scale_shift_external_v1_labels_only",
        "index": int(index),
        "source": str(source),
        "condition": condition,
        "targets": targets,
        "video_target": video_target,
        "annotation": row,
        "gt_used_for_adaptation": False,
        "labels_used_for": "evaluation_only_after_all_legal_predictions",
    }
    labels_path = _labels_path(out, condition, index)
    if labels_path.exists():
        raise RuntimeError(f"labels artifact exists before query completion: {labels_path}")
    atomic_torch(labels_path, labels_payload)
    labels_sha = sha(labels_path)

    peak = float(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30)
    records = []
    for method in METHODS:
        native_timing = 0.0 if method == "frozen" else float(bundle["timing"]["method_seconds"][method])
        encoder_timing = float(
            bundle["timing"]["encoder_seconds_by_method"].get(
                method, bundle["timing"]["encoder_seconds"]
            )
        )
        record = shift._metric_record(
            bundle["lifted"][method],
            targets,
            video_target,
            row,
            index=index,
            source=source,
            condition=condition,
            method=method,
            runtime=float(
                encoder_timing
                + bundle["timing"]["frozen_decoder_seconds"]
                + native_timing
            ),
            peak=peak,
        )
        record.update(
            fixed_external_config=(None if method == "frozen" else _fixed_config(method)),
            source_cluster=str(source),
            native_stride2=True,
            native_offsets_used=False,
            prediction_cache_sha256=cache_sha,
        )
        records.append(record)

    _assert_full_snapshot(model, snapshot)
    episode = {
        "status": "complete",
        "schema_version": "fullspan_scale_shift_external_v1_episode",
        "condition": condition,
        "index": int(index),
        "source": str(source),
        "caption": str(metadata["caption"]),
        "lock_sha256": sha(out / "lock.json"),
        "cache_sha256": cache_sha,
        "labels_sha256": labels_sha,
        "parent_frozen_cache_sha256": parent_reference["cache_sha256"],
        "raw_pixel_sha256": cache["raw_pixel_sha256"],
        "shifted_pixel_sha256": cache["shifted_pixel_sha256"],
        "frame_ids": list(frame_ids),
        "positions": list(bundle["positions"]),
        "records": records,
        "audits": bundle["audits"],
        "timing": bundle["timing"],
        "gt_used": False,
        "adaptation_gt_used": False,
        "selection_performed": False,
        "labels_loaded_only_after_all_predictions": True,
        "source_model_state_exact_after_episode": True,
    }
    atomic_json(_episode_path(out, condition, index), episode)


def _orphan_artifacts(out: Path, condition: str) -> list[str]:
    expected = {f"{index:06d}" for index in EXPECTED_INDICES}
    found: list[str] = []
    for directory_name, suffix in (
        ("episodes", ".json"),
        ("cache", ".pt"),
        ("labels_only", ".pt"),
        ("failures", ".json"),
    ):
        directory = out / condition / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if not path.name.endswith(suffix) or path.name[: -len(suffix)] not in expected:
                found.append(str(path))
    return found


def _verify_episode(out: Path, lock: Mapping[str, Any], episode: Mapping[str, Any]) -> None:
    condition = str(episode.get("condition"))
    index = int(episode.get("index"))
    if condition not in CONDITIONS or index not in EXPECTED_INDICES:
        raise RuntimeError("episode identity is outside the locked roster")
    lock_sha = sha(out / "lock.json")
    if episode.get("status") != "complete" or episode.get("lock_sha256") != lock_sha:
        raise RuntimeError(f"episode is incomplete/unlocked: {condition}/{index}")
    source = lock["target"]["sources"][str(index)]
    if episode.get("source") != source:
        raise RuntimeError(f"episode source mismatch: {condition}/{index}")
    if episode.get("gt_used") is not False or episode.get("adaptation_gt_used") is not False:
        raise RuntimeError(f"episode GT boundary changed: {condition}/{index}")
    if episode.get("selection_performed") is not False or episode.get("labels_loaded_only_after_all_predictions") is not True:
        raise RuntimeError(f"episode legal boundary changed: {condition}/{index}")
    records = episode.get("records", [])
    if [record.get("method") for record in records] != list(METHODS):
        raise RuntimeError(f"episode method roster changed: {condition}/{index}")
    for record in records:
        if record.get("condition") != condition or int(record.get("sample_index", -1)) != index or record.get("source") != source:
            raise RuntimeError(f"episode record identity mismatch: {condition}/{index}")
        if record.get("gt_used") is not False or record.get("adaptation_gt_used") is not False:
            raise RuntimeError(f"episode record GT boundary changed: {condition}/{index}")
        for metric in METRICS:
            value = float(record.get(metric))
            if not math.isfinite(value):
                raise RuntimeError(f"episode metric is non-finite: {condition}/{index}/{metric}")
    cache_path = _cache_path(out, condition, index)
    labels_path = _labels_path(out, condition, index)
    if not cache_path.is_file() or not labels_path.is_file():
        raise RuntimeError(f"episode sidecars are missing: {condition}/{index}")
    if sha(cache_path) != episode.get("cache_sha256") or sha(labels_path) != episode.get("labels_sha256"):
        raise RuntimeError(f"episode sidecar hash mismatch: {condition}/{index}")
    cache = load_torch(cache_path)
    _assert_cache_without_labels(cache)
    if cache.get("index") != index or cache.get("condition") != condition or cache.get("source") != source:
        raise RuntimeError(f"cache identity mismatch: {condition}/{index}")
    if cache.get("lock_sha256") != lock_sha:
        raise RuntimeError(f"cache lock mismatch: {condition}/{index}")
    if cache.get("parent_reference_cache_sha256") != lock["parent_frozen_cache_sha256"][condition][str(index)]["cache_sha256"]:
        raise RuntimeError(f"cache parent reference mismatch: {condition}/{index}")
    parent_reference = lock["parent_frozen_cache_sha256"][condition][str(index)]
    parent_cache_path = Path(parent_reference["path"])
    if sha(parent_cache_path) != parent_reference["cache_sha256"]:
        raise RuntimeError(f"parent Frozen cache changed while resuming: {condition}/{index}")
    parent_cache = load_torch(parent_cache_path)
    _exact(
        cache["native_predictions"]["frozen"],
        parent_cache["frozen"],
        path=f"resumed parent Frozen {condition}/{index}",
    )
    expected_lifted = shift.lift_prediction(
        parent_cache["frozen"]["pred_boxes"],
        parent_cache["frozen"]["pred_sted"],
        parent_reference["positions"],
        parent_reference["frame_ids"],
    )
    _exact(
        cache["lifted_predictions"]["frozen"],
        expected_lifted,
        path=f"resumed lifted Frozen {condition}/{index}",
    )
    if index == EXPECTED_INDICES[0]:
        controls = cache.get("native_predictions", {}).get("controls")
        if not isinstance(controls, Mapping) or set(controls) != set(EXTERNAL_METHODS):
            raise RuntimeError(f"first-query controls are missing: {condition}")
        for method in EXTERNAL_METHODS:
            if set(controls[method]) != {"lr0", "steps0"}:
                raise RuntimeError(f"first-query controls are incomplete: {condition}/{method}")
            lifted_controls = cache.get("lifted_predictions", {}).get("controls", {}).get(method, {})
            for control in ("lr0", "steps0"):
                _exact(
                    controls[method][control],
                    cache["native_predictions"]["frozen"],
                    path=f"resumed native no-op {condition}/{method}/{control}",
                )
                _exact(
                    lifted_controls[control],
                    cache["lifted_predictions"]["frozen"],
                    path=f"resumed lifted no-op {condition}/{method}/{control}",
                )


def analyze(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """CPU-only sidecar/hash coverage audit; parent metric rows are not read."""
    out = Path(out).resolve(); lock = verify_lock(out)
    failures: list[dict[str, Any]] = []; orphan: list[str] = []; cells: dict[str, Any] = {}
    for condition in CONDITIONS:
        orphan.extend(_orphan_artifacts(out, condition)); complete = 0
        for index in EXPECTED_INDICES:
            failure = _failure_path(out, condition, index)
            if failure.is_file(): failures.append(read(failure))
            path = _episode_path(out, condition, index)
            if path.is_file(): _verify_episode(out, lock, read(path)); complete += 1
        cells[condition] = {
            "status": "complete" if complete == 64 else ("partial" if complete else "not_started"),
            "planned_queries": 64, "completed_queries": complete,
        }
    failures.extend({"status": "orphan", "path": path} for path in orphan)
    formal = all(cells[c]["status"] == "complete" for c in CONDITIONS)
    result = {"status": "failed" if failures else ("complete" if formal else "partial"),
              "experiment": str(out), "lock_sha256": sha(out / "lock.json"), "conditions": cells,
              "failures": failures, "orphan_artifacts": orphan,
              "gt_isolation": "No GT in raw/model/fit/selection; metric replay is post-prediction.",
              "parent_shift_metrics_used_for_selection": False,
              "fixed_external_configuration_not_tuned": True,
              "module_held_out_not_globally_untouched": True, "analyzer_sha256": sha(Path(__file__))}
    atomic_json(out / "analysis.json", result); return result


def prepare(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """CPU-only parent/data/code gate; no model construction or CUDA."""

    out = Path(out).resolve()
    lock = initialize(out)
    result = {
        "status": "prepared",
        "schema_version": "fullspan_scale_shift_external_v1_prepare",
        "lock_sha256": sha(out / "lock.json"),
        "planned_conditions": list(CONDITIONS),
        "planned_queries": 192,
        "planned_records": 192 * len(METHODS),
        "parent_shift_complete": True,
        "parent_independent_audit_passed": True,
        "parent_metrics_read_for_selection": False,
        "annotation_json_opened_for_prepare": False,
        "gpu_started": False,
        "network_used": False,
        "fixed_external_configuration_not_tuned": True,
    }
    atomic_json(out / "prepare.json", result)
    return result


class _GpuLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"shared GPU lock is busy: {self.path}") from exc
        self.handle.write(json.dumps({"pid": os.getpid(), "script": str(Path(__file__))}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, _kind, _value, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _run(out: Path, *, device: str, conditions: Iterable[str], limit: int | None) -> None:
    out = Path(out).resolve()
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("external evaluation run requires CUDA; use prepare/analyze on CPU")
    if limit is not None:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if out == DEFAULT_OUT.resolve():
            raise RuntimeError("--limit requires a separate smoke output directory")
    selected_conditions = tuple(conditions)
    if not selected_conditions or not set(selected_conditions).issubset(set(CONDITIONS)):
        raise ValueError(f"conditions must be a non-empty subset of {CONDITIONS}")
    lock = verify_lock(out)
    orphans = [path for condition in CONDITIONS for path in _orphan_artifacts(out, condition)]
    if orphans:
        raise RuntimeError(f"orphaned artifacts block external run: {orphans[:8]}")
    base = shift._lazy_runtime()
    base.add_repo_to_path(TUBE_REPO)
    model, _args = base.build_model(
        TUBE_REPO,
        device=device,
        resolution=224,
        stride=2,
        video_max_len=200,
    )
    report = base.load_official_checkpoint(model, lock["checkpoint"])
    if report.get("missing_keys") or report.get("unexpected_keys"):
        raise RuntimeError(f"checkpoint incompatibility: {report}")
    model.eval().requires_grad_(False)
    names, _parameters = _decoder_norm_parameters(model)
    snapshot = _snapshot_full(model)
    runtime = {
        "status": "running",
        "schema_version": "fullspan_scale_shift_external_v1_runtime",
        "device": str(device),
        "model_eval": True,
        "checkpoint": lock["checkpoint"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "load_report": report,
        "decoder_layernorm_names": names,
        "decoder_layernorm_tensor_count": len(names),
        "decoder_layernorm_scalar_count": 9728,
        "source_head_ids": {"sted_embed": id(model.sted_embed), "bbox_embed": id(model.bbox_embed)},
        "gpu_started": True,
        "lock_sha256": sha(out / "lock.json"),
    }
    atomic_json(out / "runtime.json", runtime)
    for condition in selected_conditions:
        completed = 0
        for index in EXPECTED_INDICES:
            index = int(index)
            if limit is not None and completed >= limit:
                break
            episode_path = _episode_path(out, condition, index)
            failure_path = _failure_path(out, condition, index)
            cache_path = _cache_path(out, condition, index)
            labels_path = _labels_path(out, condition, index)
            if failure_path.exists():
                raise RuntimeError(f"refusing to resume a failed query: {failure_path}")
            if episode_path.exists():
                _verify_episode(out, lock, read(episode_path))
                completed += 1
                continue
            if cache_path.exists() or labels_path.exists():
                raise RuntimeError(f"orphaned query sidecar blocks resume: {condition}/{index}")
            stage_state = {"stage": "metadata_decode", "gt_read": False}
            try:
                _run_one(
                    out,
                    lock,
                    model,
                    condition=condition,
                    index=index,
                    device=device,
                    first_for_condition=(completed == 0),
                    snapshot=snapshot,
                    stage_state=stage_state,
                )
            except Exception as exc:
                _write_failure(
                    out,
                    condition,
                    index,
                    stage_state["stage"],
                    exc,
                    gt_read=bool(stage_state["gt_read"]),
                )
                raise
            completed += 1
            print(f"[external shift] {condition} {completed}/64", flush=True)
        _assert_full_snapshot(model, snapshot)
    formal = set(selected_conditions) == set(CONDITIONS) and limit is None
    if formal and all(
        _episode_path(out, condition, int(index)).is_file()
        for condition in CONDITIONS
        for index in EXPECTED_INDICES
    ):
        atomic_json(
            out / "complete.json",
            {
                "status": "complete",
                "schema_version": "fullspan_scale_shift_external_v1_complete",
                "conditions": list(CONDITIONS),
                "queries": 192,
                "records": 192 * len(METHODS),
                "lock_sha256": sha(out / "lock.json"),
                "gpu_started": True,
                "fixed_external_configuration_not_tuned": True,
            },
        )
    else:
        atomic_json(
            out / "partial.json",
            {
                "status": "partial",
                "schema_version": "fullspan_scale_shift_external_v1_partial",
                "conditions": list(selected_conditions),
                "limit_per_condition": limit,
                "formal_complete_written": False,
                "lock_sha256": sha(out / "lock.json"),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    torch.backends.cudnn.benchmark = False
    out = args.out.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(out), indent=2, ensure_ascii=False))
    elif args.action == "analyze":
        print(json.dumps(analyze(out), indent=2, ensure_ascii=False))
    else:
        with _GpuLock(GPU_LOCK):
            _run(out, device=args.device, conditions=args.conditions, limit=args.limit)


if __name__ == "__main__":
    main()
