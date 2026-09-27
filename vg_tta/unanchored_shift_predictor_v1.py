"""One-query, label-free TubeDETR predictor for the unanchored shift study.

This is a small consumer-side primitive, not an experiment runner.  The
caller supplies an already-frozen native TubeDETR model and its complete
source snapshot; this module does not load media, annotations, checkpoints,
or result files and never scores a prediction.  It applies the raw RGB
corruption from :mod:`vg_tta.shift_corruptions_v2`, shares one original
encoded memory across all methods, and returns native sampled predictions
plus the sealed physical-grid lift.

    ``ours`` fits a fresh private copy of ``model.sted_embed`` with the
    unanchored (``gamma=0``) Fullspan prior.  ``old_anchored`` is optional and
    uses the caller's explicit nonzero-anchor configuration.  The optional
    ``matched_anchor_gamma`` arm uses the selected ours LR/steps with only the
    requested anchor changed.  Tent/MEMO/SAR use
the sealed live-forward helper and the local fixed configuration copy below;
the optional SAR margin is only an execution declaration, never a selection
operation.  The direct Fullspan output is a separate no-parameter control and
is never used as a fallback for a failed fit.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from vg_tta.fullspan_tta import fit_fullspan_head, replay_temporal_head
from vg_tta.shift_corruptions_v2 import CorruptionResult, apply_corruption


VERSION = "unanchored_shift_predictor_v1"
METHODS = (
    "frozen",
    "ours",
    "old_anchored",
    "matched_anchor",
    "direct_fullspan",
    "tent",
    "memo",
    "sar",
)
EXTERNAL_METHODS = ("tent", "memo", "sar")

# A literal local copy is intentional: a later change to the external
# baseline module must not silently change this predictor's declared arms.
DEFAULT_BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "tent": {
        "lr": 0.001,
        "steps": 3,
        "optimizer_name": "adam",
        "num_views": 1,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "momentum": None,
        "reset_constant_em": 0.2,
    },
    "memo": {
        "lr": 0.005,
        "steps": 3,
        "optimizer_name": "sgd",
        "num_views": 4,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "momentum": 0.0,
        "reset_constant_em": 0.2,
    },
    "sar": {
        "lr": 0.001,
        "steps": 3,
        "optimizer_name": "sgd",
        "num_views": 1,
        "rho": 0.05,
        "entropy_margin_fraction": 0.4,
        "momentum": 0.9,
        "reset_constant_em": 0.2,
    },
}


class PredictorContractError(RuntimeError):
    """Raised when a caller violates the label-free predictor contract."""


def _sealed_external() -> Any:
    # Importing this module does not load data.  Delaying the import keeps the
    # predictor CPU-testable and lets callers decide when the sealed bridge is
    # available.
    from scripts import evaluate_fullspan_scale_shift_external_v1 as external

    return external


def _model_device(model: nn.Module, requested: torch.device | str | None) -> torch.device:
    parameters = list(model.parameters())
    if not parameters:
        raise PredictorContractError("native model must have parameters")
    actual = parameters[0].device
    if any(parameter.device != actual for parameter in parameters):
        raise PredictorContractError("native model parameters are on mixed devices")
    device = actual if requested is None else torch.device(requested)
    if device != actual:
        raise PredictorContractError(f"requested device {device} differs from model device {actual}")
    return device


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise PredictorContractError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} is non-finite")


def _native_prediction(value: Any, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not {"pred_boxes", "pred_sted"}.issubset(value):
        raise PredictorContractError(f"{name} lacks native pred_boxes/pred_sted")
    boxes, logits = value["pred_boxes"], value["pred_sted"]
    if not torch.is_tensor(boxes) or not torch.is_tensor(logits):
        raise PredictorContractError(f"{name} predictions must be tensors")
    _finite(boxes, f"{name}.pred_boxes")
    _finite(logits, f"{name}.pred_sted")
    if boxes.ndim != 2 or tuple(boxes.shape[-1:]) != (4,):
        raise PredictorContractError(f"{name}.pred_boxes must be [T,4], got {tuple(boxes.shape)}")
    if logits.ndim != 3 or tuple(logits.shape[:1]) != (1,) or logits.shape[-1] != 2:
        raise PredictorContractError(f"{name}.pred_sted must be [1,T,2], got {tuple(logits.shape)}")
    if int(boxes.shape[0]) != int(logits.shape[1]):
        raise PredictorContractError(f"{name} boxes/logits temporal lengths disagree")
    return {"pred_boxes": boxes, "pred_sted": logits}


def _cpu_prediction(value: Mapping[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    checked = _native_prediction(value, name)
    return {key: tensor.detach().cpu().clone() for key, tensor in checked.items()}


def _exact(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise AssertionError(f"{name} dtype/shape mismatch")
    if not torch.equal(left, right.to(device=left.device, dtype=left.dtype)):
        raise AssertionError(f"{name} is not exact")


def _head_config(value: Mapping[str, Any], *, name: str, gamma: float | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} config must be a mapping")
    if "lr" not in value or "steps" not in value:
        raise ValueError(f"{name} config requires lr and steps")
    lr = float(value["lr"])
    steps = int(value["steps"])
    if not math.isfinite(lr) or lr < 0:
        raise ValueError(f"{name}.lr must be finite and non-negative")
    if isinstance(value["steps"], bool) or steps != value["steps"] or steps < 0:
        raise ValueError(f"{name}.steps must be a non-negative integer")
    if gamma is None:
        raw_gamma = value.get("anchor_gamma", value.get("gamma", 1.0e-4))
        anchor_gamma = float(raw_gamma)
    else:
        for key in ("anchor_gamma", "gamma"):
            if key in value and float(value[key]) != float(gamma):
                raise ValueError(f"{name} violates declared anchor_gamma={gamma}")
        anchor_gamma = float(gamma)
    if not math.isfinite(anchor_gamma) or anchor_gamma < 0:
        raise ValueError(f"{name}.anchor_gamma must be finite and non-negative")
    eps = float(value.get("optimizer_eps", 1.0e-4))
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"{name}.optimizer_eps must be finite and positive")
    return {"lr": lr, "steps": steps, "anchor_gamma": anchor_gamma, "optimizer_eps": eps}


def _fit_private_head(
    source_head: nn.Module,
    head_input: torch.Tensor,
    config: Mapping[str, Any],
    *,
    name: str,
) -> tuple[nn.Module, dict[str, Any], torch.Tensor]:
    head = copy.deepcopy(source_head).to(device=head_input.device).eval().requires_grad_(False)
    records = [{"head_input": head_input}] * int(config["steps"])
    fit = fit_fullspan_head(
        head,
        records,
        lr=float(config["lr"]),
        anchor_gamma=float(config["anchor_gamma"]),
        optimizer_eps=float(config["optimizer_eps"]),
    )
    audit = fit.get("audit", {})
    if int(audit.get("optimizer_steps", -1)) != int(config["steps"]):
        raise RuntimeError(f"{name} optimizer-step audit disagrees with requested steps")
    if audit.get("all_steps_finite") is not True:
        raise FloatingPointError(f"{name} fit did not prove finite steps")
    with torch.no_grad():
        logits = replay_temporal_head(head, head_input, head_input.device)
    _finite(logits, f"{name} replay logits")
    return head, fit, logits


def _timed(external: Any, device: torch.device, function: Any) -> tuple[Any, float]:
    helper = getattr(external, "_timed", None)
    if callable(helper):
        return helper(str(device), function)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, float(time.perf_counter() - start)


def _stable_view_index(query_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _lift(external: Any, prediction: Mapping[str, torch.Tensor], positions: list[int], frame_ids: list[int]) -> dict[str, torch.Tensor]:
    lifted = external.shift.lift_prediction(
        prediction["pred_boxes"].detach().cpu(),
        prediction["pred_sted"].detach().cpu(),
        list(positions),
        list(frame_ids),
    )
    return _cpu_prediction(lifted, "lifted prediction")


def _native_reinsert(
    external: Any,
    model: nn.Module,
    memory: Mapping[str, Any],
    temporal_head: nn.Module,
    duration: int,
    caption: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    result = external.shift._native_reinsert(
        model, memory, temporal_head, model.bbox_embed, duration, caption, str(device)
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("sealed native reinsertion must return (outputs, seconds)")
    outputs, seconds = result
    return _native_prediction(outputs, "native reinsertion"), float(seconds)


def _make_original_memory(
    external: Any,
    model: nn.Module,
    video: torch.Tensor,
    caption: str,
    device: torch.device,
) -> tuple[list[Mapping[str, Any]], float, list[dict[str, Any]]]:
    """Encode only the original normalized view for non-MEMO calls."""

    runtime = external.shift._lazy_runtime()
    repo = getattr(external, "TUBE_REPO", None)
    kwargs: dict[str, Any] = {
        "stride": 2,
        "device": str(device),
        "use_bf16": True,
    }
    if repo is not None:
        kwargs["repo"] = repo
    with torch.no_grad():
        memory, elapsed = _timed(
            external,
            device,
            lambda: runtime.encode_video(model, video, caption, **kwargs),
        )
    if not isinstance(memory, Mapping):
        raise PredictorContractError("original-view encoder did not return a memory mapping")
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise PredictorContractError("normalized original view must be [3,T,H,W]")
    view_audit = [{
        "view_index": 0,
        "role": "original",
        "seed": None,
        "geometric": False,
        "frame_count": int(video.shape[1]),
        "encoder_seconds": float(elapsed),
    }]
    return [memory], float(elapsed), view_audit


def _physical_frame_ids(raw: np.ndarray, frame_ids: Sequence[int] | None) -> list[int]:
    """Validate the physical grid without consulting annotations or files."""

    count = int(raw.shape[0])
    if frame_ids is None:
        return list(range(count))
    if isinstance(frame_ids, (str, bytes)):
        raise TypeError("frame_ids must be a sequence of integer physical ids")
    try:
        values = list(frame_ids)
    except TypeError as exc:
        raise TypeError("frame_ids must be a sequence of integer physical ids") from exc
    if len(values) != count:
        raise PredictorContractError("frame_ids must have one physical id per input frame")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise PredictorContractError("frame_ids must contain integer physical ids")
        result.append(int(value))
    if result != sorted(set(result)):
        raise PredictorContractError("frame_ids must be strictly increasing")
    return result


def _clean_corruption(raw_rgb: np.ndarray, frame_ids: list[int], query_id: str, seed: int) -> CorruptionResult:
    """Build an auditable severity-zero identity without invoking a shift."""

    if raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3 or raw_rgb.dtype != np.uint8:
        raise PredictorContractError(
            "raw_rgb must have dtype uint8 and shape [T,H,W,3] for clean identity"
        )
    if any(int(size) < 1 for size in raw_rgb.shape[:3]):
        raise PredictorContractError("raw_rgb has an empty temporal or spatial dimension")
    raw = np.ascontiguousarray(raw_rgb)
    output = raw.copy()
    digest = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    metadata = {
        "schema_version": f"{VERSION}_clean_identity",
        "corruption": "clean",
        "severity": 0,
        "query_id": str(query_id),
        "seed": int(seed),
        "derived_seed": None,
        "parameters": {"identity": True},
        "realized_parameters": {},
        "input_dtype": str(raw.dtype),
        "output_dtype": str(output.dtype),
        "input_shape": list(raw.shape),
        "output_shape": list(output.shape),
        "input_sha256": digest,
        "output_sha256": hashlib.sha256(output.tobytes(order="C")).hexdigest(),
        "retained_positions": list(range(int(raw.shape[0]))),
        "retained_frame_ids": list(frame_ids),
        "input_frame_ids": list(frame_ids),
        "temporal_audit": {
            "selection": "all_frames_identity",
            "requested_retained_count": int(raw.shape[0]),
            "actual_retained_count": int(raw.shape[0]),
            "minimum_retained_frames_satisfied": True,
            "endpoint_guard_added": False,
            "minimum_guard_added": False,
            "short_clip_adjustment": False,
        },
        "physical_grid_preserved": True,
        "first_frame_preserved": True,
        "last_frame_preserved": True,
        "labels_used": False,
        "gt_used": False,
        "decoded_video": False,
        "gpu_used": False,
        "identity_exact": bool(np.array_equal(raw, output)),
    }
    if not metadata["identity_exact"]:  # pragma: no cover - defensive
        raise AssertionError("clean identity unexpectedly changed raw pixels")
    return CorruptionResult(frames=output, metadata=metadata)


def _fit_variant(
    external: Any,
    model: nn.Module,
    source_head: nn.Module,
    memory: Mapping[str, Any],
    head_input: torch.Tensor,
    frozen_native: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    name: str,
    duration: int,
    caption: str,
    device: torch.device,
    source_snapshot: Mapping[str, Any],
    check_native: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    fit_start = time.perf_counter()
    head, fit, logits = _fit_private_head(source_head, head_input, config, name=name)
    fit = copy.deepcopy(fit)
    fit.setdefault("audit", {})["elapsed_seconds"] = float(time.perf_counter() - fit_start)
    fitted_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    replay = {"pred_boxes": frozen_native["pred_boxes"], "pred_sted": logits}
    reinsertion = {"checked": False, "prediction_exact": None, "seconds": 0.0}
    native = replay
    if check_native:
        native, seconds = _native_reinsert(external, model, memory, head, duration, caption, device)
        _exact(native["pred_sted"], logits, f"{name} native reinsertion logits")
        _exact(native["pred_boxes"], frozen_native["pred_boxes"], f"{name} native reinsertion boxes")
        reinsertion = {"checked": True, "prediction_exact": True, "seconds": float(seconds)}
    external._assert_full_snapshot(model, source_snapshot)
    return _cpu_prediction(native, f"{name} native"), fit, fitted_state, reinsertion


def _check_noops(
    external: Any,
    model: nn.Module,
    source_head: nn.Module,
    memory: Mapping[str, Any],
    head_input: torch.Tensor,
    frozen_native: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    duration: int,
    caption: str,
    device: torch.device,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    controls: dict[str, Any] = {"checked": True, "scope": "ours private temp_embed"}
    for name, control_config in (("lr0", {**config, "lr": 0.0}), ("steps0", {**config, "steps": 0})):
        head, fit, logits = _fit_private_head(source_head, head_input, control_config, name=f"ours_{name}")
        _exact(logits, frozen_native["pred_sted"], f"ours {name} replay logits")
        for key, value in head.state_dict().items():
            _exact(value, source_head.state_dict()[key], f"ours {name} parameter {key}")
        native, seconds = _native_reinsert(external, model, memory, head, duration, caption, device)
        _exact(native["pred_sted"], frozen_native["pred_sted"], f"ours {name} native logits")
        _exact(native["pred_boxes"], frozen_native["pred_boxes"], f"ours {name} native boxes")
        controls[name] = {
            "fit": fit,
            "reinsert_seconds": float(seconds),
            "prediction_exact_frozen": True,
            "parameter_exact_frozen": True,
        }
        external._assert_full_snapshot(model, source_snapshot)
    return controls


def predict_episode(
    model: nn.Module,
    raw_rgb: np.ndarray,
    frame_ids: Sequence[int] | None,
    caption: str,
    query_id: str,
    *,
    corruption: str = "clean",
    severity: int = 0,
    seed: int = 20260908,
    selected_config: Mapping[str, Any],
    old_config: Mapping[str, Any] | None = None,
    check_controls: bool = False,
    device: torch.device | str = "cuda:0",
    source_snapshot: Mapping[str, Any] | None = None,
    baselines: bool = True,
    sar_entropy_margin_fraction: float | None = None,
    matched_anchor_gamma: float | None = None,
) -> dict[str, Any]:
    """Predict one corrupted, unlabeled TubeDETR episode.

    The model must already be in ``eval`` mode with every parameter frozen.
    When ``source_snapshot`` is omitted, this function takes a complete
    snapshot immediately before prediction; a supplied snapshot must be the
    matching complete snapshot from the sealed external bridge.
    ``selected_config`` contains only the selected unanchored ``lr``/``steps``
    (its anchor is forced to zero).  All returned tensors are detached CPU
    copies; labels and metric rows never enter this function.  ``baselines``
    controls the optional Tent/MEMO/SAR arms, while the Frozen/ours/direct
    arms remain available regardless of that switch.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(raw_rgb, np.ndarray):
        raise TypeError("raw_rgb must be the caller-supplied numpy RGB array")
    if not isinstance(caption, str) or not isinstance(query_id, str):
        raise TypeError("caption and query_id must be strings")
    if not isinstance(baselines, bool):
        raise TypeError("baselines must be a boolean")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PredictorContractError("predictor requires an already-frozen model")
    if any(module.training for module in model.modules()):
        raise PredictorContractError("predictor requires model.eval() before prediction")
    target_device = _model_device(model, device)
    external = _sealed_external()
    if source_snapshot is None:
        snapshot_factory = getattr(external, "_snapshot_full", None)
        if not callable(snapshot_factory):
            raise PredictorContractError("sealed external bridge lacks _snapshot_full")
        source_snapshot = snapshot_factory(model)
    if not isinstance(source_snapshot, Mapping):
        raise TypeError("source_snapshot must be a complete snapshot mapping")
    external._assert_full_snapshot(model, source_snapshot)
    own_config = _head_config(selected_config, name="selected unanchored", gamma=0.0)
    old_config = None if old_config is None else _head_config(
        old_config, name="old anchored", gamma=None
    )
    if old_config is not None and old_config["anchor_gamma"] == 0.0:
        raise ValueError("old anchored config must have a nonzero anchor_gamma")
    if sar_entropy_margin_fraction is not None:
        sar_margin = float(sar_entropy_margin_fraction)
        if not math.isfinite(sar_margin) or sar_margin < 0:
            raise ValueError("sar_entropy_margin_fraction must be finite and non-negative")
    else:
        sar_margin = None
    matched_config = None
    if matched_anchor_gamma is not None:
        matched_gamma = float(matched_anchor_gamma)
        if not math.isfinite(matched_gamma) or matched_gamma < 0:
            raise ValueError("matched_anchor_gamma must be finite and non-negative")
        matched_config = _head_config(
            {
                "lr": own_config["lr"],
                "steps": own_config["steps"],
                "anchor_gamma": matched_gamma,
                "optimizer_eps": own_config["optimizer_eps"],
            },
            name="matched anchor",
            gamma=None,
        )

    source_head = getattr(model, "sted_embed", None)
    if not isinstance(source_head, nn.Module) or not isinstance(getattr(model, "bbox_embed", None), nn.Module):
        raise PredictorContractError("native model must expose sted_embed and bbox_embed modules")
    runtime = external.shift._lazy_runtime()
    original_frame_ids = _physical_frame_ids(raw_rgb, frame_ids)
    if corruption == "clean":
        if int(severity) != 0:
            raise ValueError("clean corruption requires severity=0")
        corrupted = _clean_corruption(raw_rgb, original_frame_ids, query_id, int(seed))
    else:
        corrupted = apply_corruption(
            raw_rgb,
            corruption,
            severity,
            query_id=query_id,
            seed=int(seed),
            frame_ids=original_frame_ids,
        )
    metadata = copy.deepcopy(dict(corrupted.metadata))
    positions = list(corrupted.retained_positions)
    duration = int(corrupted.frames.shape[0])
    if len(positions) != duration or metadata["retained_frame_ids"] != [original_frame_ids[i] for i in positions]:
        raise PredictorContractError("corruption physical-frame mapping is inconsistent")

    try:
        normalized, normalization_seconds = _timed(
            external,
            target_device,
            lambda: runtime.normalize_raw_view(corrupted.frames, resolution=224),
        )
        if not torch.is_tensor(normalized) or normalized.ndim != 4 or int(normalized.shape[1]) != duration:
            raise PredictorContractError("normalized corruption view is not [3,T,H,W]")
        if not bool(torch.isfinite(normalized).all().detach().cpu()):
            raise FloatingPointError("normalized corruption view is non-finite")

        if baselines:
            memories, encoder_seconds, view_audit = external._make_memories(
                model,
                normalized,
                caption,
                _stable_view_index(query_id, int(seed)),
                device=str(target_device),
            )
            expected_views = 4
        else:
            memories, encoder_seconds, view_audit = _make_original_memory(
                external, model, normalized, caption, target_device
            )
            expected_views = 1
        if len(memories) != expected_views or len(view_audit) != expected_views:
            raise PredictorContractError(
                f"memory construction must provide exactly {expected_views} view(s)"
            )
        if any(int(item.get("frame_count", duration)) != duration for item in view_audit):
            raise PredictorContractError("MEMO views changed temporal geometry")

        (frozen_raw, head_input), frozen_seconds = _timed(
            external,
            target_device,
            lambda: runtime._decode_capture(model, memories[0], duration, caption, str(target_device)),
        )
        frozen_native = _native_prediction(frozen_raw, "frozen native")
        if not torch.is_tensor(head_input) or head_input.ndim != 4 or tuple(head_input.shape[1:3]) != (1, duration):
            raise PredictorContractError("captured native temporal head input is not [L,1,T,D]")
        if head_input.requires_grad:
            head_input = head_input.detach()
        _finite(head_input, "captured head input")
        frozen_cpu = _cpu_prediction(frozen_native, "frozen native")
        native_predictions: dict[str, dict[str, torch.Tensor]] = {"frozen": frozen_cpu}
        lifted_predictions: dict[str, dict[str, torch.Tensor]] = {
            "frozen": _lift(external, frozen_cpu, positions, original_frame_ids)
        }
        fit_audits: dict[str, Any] = {}
        fitted_states: dict[str, dict[str, torch.Tensor]] = {}
        reinsertion_audits: dict[str, Any] = {}
        variant_configs = {"ours": own_config}
        if old_config is not None:
            variant_configs["old_anchored"] = old_config
        if matched_config is not None:
            variant_configs["matched_anchor"] = matched_config
        for name, config in variant_configs.items():
            native, fit, state, reinsertion = _fit_variant(
                external, model, source_head, memories[0], head_input, frozen_native, config,
                name=name, duration=duration, caption=caption, device=target_device,
                source_snapshot=source_snapshot, check_native=check_controls,
            )
            native_predictions[name] = native
            lifted_predictions[name] = _lift(external, native, positions, original_frame_ids)
            fit_audits[name], fitted_states[name], reinsertion_audits[name] = fit, state, reinsertion

        from vg_tta.tastvg_baseline_expansion import direct_fullspan_logits

        direct = {"pred_boxes": frozen_native["pred_boxes"], "pred_sted": direct_fullspan_logits(frozen_native["pred_sted"])}
        native_predictions["direct_fullspan"] = _cpu_prediction(direct, "direct Fullspan control")
        lifted_predictions["direct_fullspan"] = _lift(external, native_predictions["direct_fullspan"], positions, original_frame_ids)

        baseline_configs = copy.deepcopy(DEFAULT_BASELINE_CONFIGS)
        if sar_margin is not None:
            baseline_configs["sar"]["entropy_margin_fraction"] = sar_margin
        baseline_audits: dict[str, Any] = {}
        baseline_seconds: dict[str, float] = {}
        if baselines:
            names, _ = external._decoder_norm_parameters(model)
            for method in EXTERNAL_METHODS:
                config = baseline_configs[method]
                method_memories = memories if method == "memo" else memories[:1]
                native, audit, elapsed = external._fit_one_method(
                    model,
                    names,
                    method_memories,
                    method,
                    config,
                    duration=duration,
                    caption=caption,
                    device=str(target_device),
                    snapshot=source_snapshot,
                )
                if not isinstance(audit, Mapping):
                    raise PredictorContractError(f"{method} baseline returned a non-mapping audit")
                for flag in ("gt_used", "gt_used_for_fit", "labels_used"):
                    if flag in audit and audit[flag] is not False:
                        raise PredictorContractError(f"{method} baseline violated GT-free flag {flag}")
                native_predictions[method] = _cpu_prediction(native, f"{method} native")
                lifted_predictions[method] = _lift(external, native_predictions[method], positions, original_frame_ids)
                baseline_audits[method] = copy.deepcopy(dict(audit))
                baseline_seconds[method] = float(elapsed)

        controls = (
            _check_noops(
                external, model, source_head, memories[0], head_input, frozen_native, own_config,
                duration=duration, caption=caption, device=target_device, source_snapshot=source_snapshot,
            )
            if check_controls else {"checked": False, "scope": "ours private temp_embed"}
        )

        external._assert_full_snapshot(model, source_snapshot)
        ours_fit = fit_audits.get("ours", {}) or {}
        old_fit = fit_audits.get("old_anchored", {}) or {}
        matched_fit = fit_audits.get("matched_anchor", {}) or {}
        ours_audit = ours_fit.get("audit", {}) if isinstance(ours_fit, Mapping) else {}
        old_audit = old_fit.get("audit", {}) if isinstance(old_fit, Mapping) else {}
        matched_audit = matched_fit.get("audit", {}) if isinstance(matched_fit, Mapping) else {}
        encoder_seconds_by_view = [float(item["encoder_seconds"]) for item in view_audit]
        if not encoder_seconds_by_view:
            raise PredictorContractError("encoder audit is empty")
        encoder_seconds_by_method = {
            method: float(sum(encoder_seconds_by_view) if method == "memo" else encoder_seconds_by_view[0])
            for method in native_predictions
        }
        return {
            "schema_version": VERSION,
            "query_id": query_id,
            "caption": caption,
            "gt_used": False,
            "labels_used": False,
            "fit_GT_used": False,
            "selection_performed": False,
            "native_predictions": native_predictions,
            "lifted_predictions": lifted_predictions,
            "fitted_head_states": fitted_states,
            "fitted_states": fitted_states,
            "head_input": head_input.detach().cpu().clone(),
            "corruption": metadata,
            "audits": {
                "source_model_state_exact": True,
                "native_boxes_fixed_for_ours": True,
                "shared_original_memory": True,
                "memo_view_count": 4 if baselines else 0,
                "memo_temporal_geometry_preserved": True,
                "selected_config": own_config,
                "old_anchored_config": old_config,
                "ours_fit": fit_audits["ours"],
                "old_anchored_fit": fit_audits.get("old_anchored"),
                "matched_anchor_config": matched_config,
                "matched_anchor_fit": fit_audits.get("matched_anchor"),
                "external_methods": baseline_audits,
                "baselines_enabled": bool(baselines),
                "controls": controls,
                "native_reinsertion": reinsertion_audits,
                "direct_fullspan": {
                    "control": True,
                    "parameter_update": False,
                    "selection_eligible": False,
                    "fallback_used": False,
                },
                "sar_margin_declaration": sar_margin,
                "baseline_config_source": (
                    "local literal copy in unanchored_shift_predictor_v1" if baselines else None
                ),
            },
            "realized_corruption": metadata,
            "frame_mapping": {
                "input_frame_ids": list(original_frame_ids),
                "retained_positions": positions,
                "retained_frame_ids": list(corrupted.retained_frame_ids),
                "lifted_to_input_grid": True,
            },
            "timing": {
                "normalization_seconds": float(normalization_seconds),
                "encoder_seconds": float(encoder_seconds),
                "frozen_decode_seconds": float(frozen_seconds),
                "ours_fit_seconds": float(ours_audit.get("elapsed_seconds", 0.0)),
                "old_anchored_fit_seconds": float(old_audit.get("elapsed_seconds", 0.0)) if old_config is not None else 0.0,
                "matched_anchor_fit_seconds": float(matched_audit.get("elapsed_seconds", 0.0)) if matched_config is not None else 0.0,
                "external_method_seconds": baseline_seconds,
                "native_reinsertion_seconds": {key: float(value["seconds"]) for key, value in reinsertion_audits.items()},
                "encoder_seconds_by_view": encoder_seconds_by_view,
                "encoder_seconds_by_method": encoder_seconds_by_method,
                "memo_encoder_shared_four_views": bool(baselines),
                "cache_not_free": True,
            },
            "view_audit": copy.deepcopy(view_audit),
        }
    finally:
        external._assert_full_snapshot(model, source_snapshot)


def predict_unanchored_shift_episode(
    model: nn.Module,
    source_snapshot: Mapping[str, Any],
    raw_rgb: np.ndarray,
    frame_ids: Sequence[int] | None,
    caption: str,
    query_id: str,
    corruption: str,
    severity: int,
    seed: int,
    selected_config: Mapping[str, Any],
    *,
    old_anchored_config: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
    check_controls: bool = True,
    sar_entropy_margin_fraction: float | None = None,
    baselines: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for the earlier positional API.

    New callers should use :func:`predict_episode`; retaining this wrapper
    avoids silently changing an already-written consumer while preserving the
    same fail-closed implementation and output schema.
    """

    return predict_episode(
        model,
        raw_rgb,
        frame_ids,
        caption,
        query_id,
        corruption=corruption,
        severity=severity,
        seed=seed,
        selected_config=selected_config,
        old_config=old_anchored_config,
        check_controls=check_controls,
        device="cuda:0" if device is None else device,
        source_snapshot=source_snapshot,
        baselines=baselines,
        sar_entropy_margin_fraction=sar_entropy_margin_fraction,
    )


__all__ = [
    "VERSION",
    "METHODS",
    "EXTERNAL_METHODS",
    "DEFAULT_BASELINE_CONFIGS",
    "PredictorContractError",
    "predict_unanchored_shift_episode",
    "predict_episode",
]
