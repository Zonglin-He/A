"""Auxiliary-loss backflow controls for the DESTA-3D v2 adapter.

The helpers in this module leave the forward values and task-loss graph alone.
They only scale the gradient from the referent and direct event-presence heads
back into their input features, or combine separately accumulated CPU gradient
buffers using the same rule.  The event hook is placed on the pooled input to
``event_presence_head``; multiplying that input gradient is equivalent to
multiplying the gradient entering the preceding linear XY pooling operation.

``B0c1`` leaves all auxiliary reader backflow at its measured weighted value,
``B1c0`` removes auxiliary backflow into readers, and ``B2`` sets one shared
reader-backflow coefficient from the completed accumulation window. Readout
head gradients always retain their original auxiliary weight in all modes.
These are gradient-combination controls; their diagnostics do not guarantee an
Adam/AdamW parameter displacement or an improvement in task metrics.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


_MODE_ALIASES = {"B0": "B0c1", "B0c1": "B0c1", "B1": "B1c0", "B1c0": "B1c0", "B2": "B2"}
_MODES = tuple(_MODE_ALIASES)


class _IdentityScaledBackward(torch.autograd.Function):
    """Identity in the forward pass with a fixed scalar on its input gradient."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * ctx.scale, None


def _checked_scale(scale: float | torch.Tensor) -> float:
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("backflow scale tensor must contain exactly one value")
        if scale.requires_grad:
            raise ValueError("backflow scale must be detached")
        value = float(scale.detach().item())
    elif isinstance(scale, numbers.Real) and not isinstance(scale, bool):
        value = float(scale)
    else:
        raise TypeError("backflow scale must be a real scalar or scalar tensor")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("backflow scale must be finite and in [0,1]")
    return value


def auxiliary_gradient_view(
    x: torch.Tensor, c: float | torch.Tensor
) -> torch.Tensor:
    """Return an exactly value-preserving view whose gradient to ``x`` is ``c``.

    The scale is treated as a detached constant.  Consequently parameters in a
    head that consumes this result receive their unchanged forward-pass
    gradients, while gradients through the result to upstream reader features
    are multiplied by ``c``.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    return _IdentityScaledBackward.apply(x, _checked_scale(c))


@contextmanager
def auxiliary_head_backflow(
    adapter: nn.Module, c: float | torch.Tensor
) -> Iterator[None]:
    """Scale only reader-side auxiliary gradients entering the two evidence heads.

    Forward pre-hooks wrap the input to ``adapter.referent_head`` and the
    already-XY-pooled input to ``adapter.event_presence_head``. They do not
    modify either residual/output-projection path. Hooks are removed even if
    the body raises. Do not nest this context on the same adapter, since nested
    scales would multiply.
    """
    coefficient = _checked_scale(c)
    for name in ("referent_head", "event_presence_head"):
        head = getattr(adapter, name, None)
        if not isinstance(head, nn.Module):
            raise TypeError(f"adapter.{name} must be an nn.Module")

    handles: list[Any] = []

    def _wrap_first_tensor(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("evidence head must receive a tensor as its first positional input")
        return (auxiliary_gradient_view(args[0], coefficient), *args[1:])

    try:
        handles.append(adapter.referent_head.register_forward_pre_hook(_wrap_first_tensor))
        handles.append(
            adapter.event_presence_head.register_forward_pre_hook(_wrap_first_tensor)
        )
        yield
    finally:
        for handle in handles:
            handle.remove()


def _parameter_layout(
    adapter: nn.Module,
) -> tuple[
    dict[str, nn.Parameter],
    dict[str, str],
    set[str],
    dict[str, tuple[str, ...]],
]:
    if not callable(getattr(adapter, "parameter_groups", None)):
        raise TypeError("adapter must expose parameter_groups()")
    parameters = dict(adapter.named_parameters())
    raw_groups = adapter.parameter_groups()
    if not isinstance(raw_groups, Mapping):
        raise TypeError("adapter.parameter_groups() must return a mapping")

    group_by_id: dict[int, str] = {}
    group_names: dict[str, tuple[str, ...]] = {}
    name_by_id = {id(parameter): name for name, parameter in parameters.items()}
    for group_name, group_parameters in raw_groups.items():
        if not isinstance(group_name, str) or not isinstance(group_parameters, Sequence):
            raise TypeError("parameter_groups() must map string names to parameter sequences")
        names: list[str] = []
        for parameter in group_parameters:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise ValueError(f"parameter group {group_name!r} contains an unregistered parameter")
            if name in group_by_id.values():
                raise ValueError(f"parameter {name!r} occurs in more than one group")
            group_by_id[id(parameter)] = group_name
            names.append(name)
        group_names[group_name] = tuple(names)
    missing = set(parameters) - set(name_by_id[id(p)] for p in parameters.values() if id(p) in group_by_id)
    if missing:
        raise ValueError(f"parameter_groups() omits parameters: {sorted(missing)}")

    head_ids: set[int] = set()
    for name in ("referent_head", "event_presence_head"):
        head = getattr(adapter, name, None)
        if not isinstance(head, nn.Module):
            raise TypeError(f"adapter.{name} must be an nn.Module")
        head_ids.update(id(parameter) for parameter in head.parameters())
    head_names = {name for name, parameter in parameters.items() if id(parameter) in head_ids}
    return parameters, {name_by_id[key]: value for key, value in group_by_id.items()}, head_names, group_names


def _validate_buffers(
    buffers: Mapping[str, torch.Tensor | None],
    parameters: Mapping[str, nn.Parameter],
    *,
    label: str,
) -> None:
    if not isinstance(buffers, Mapping):
        raise TypeError(f"{label} must be a mapping of parameter names to tensors or None")
    expected, actual = set(parameters), set(buffers)
    if expected != actual:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"{label} keys do not match adapter parameters; missing={missing}, extra={extra}")
    for name, gradient in buffers.items():
        if gradient is None:
            continue
        if not isinstance(gradient, torch.Tensor):
            raise TypeError(f"{label}[{name!r}] must be a tensor or None")
        if gradient.device.type != "cpu":
            raise ValueError(f"{label}[{name!r}] must be a CPU tensor")
        if gradient.dtype != torch.float32:
            raise ValueError(f"{label}[{name!r}] must be FP32, got {gradient.dtype}")
        if gradient.shape != parameters[name].shape:
            raise ValueError(
                f"{label}[{name!r}] shape {tuple(gradient.shape)} does not match "
                f"parameter {tuple(parameters[name].shape)}"
            )
        if not torch.isfinite(gradient).all():
            raise ValueError(f"{label}[{name!r}] contains a non-finite value")


def _norm(buffers: Mapping[str, torch.Tensor | None], names: Sequence[str]) -> float:
    squared = 0.0
    for name in names:
        value = buffers[name]
        if value is not None:
            squared += float(torch.sum(value.double() * value.double()).item())
    return math.sqrt(squared)


def _dot(
    left: Mapping[str, torch.Tensor | None],
    right: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
) -> float:
    result = 0.0
    for name in names:
        a, b = left[name], right[name]
        if a is not None and b is not None:
            result += float(torch.sum(a.double() * b.double()).item())
    return result


def _combine_norm_diagnostics(
    task: Mapping[str, torch.Tensor | None],
    auxiliary: Mapping[str, torch.Tensor | None],
    scaled_auxiliary: Mapping[str, torch.Tensor | None],
    combined: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
) -> dict[str, float | None]:
    norm_task = _norm(task, names)
    norm_auxiliary = _norm(auxiliary, names)
    norm_scaled_auxiliary = _norm(scaled_auxiliary, names)
    norm_combined = _norm(combined, names)
    dot_task_aux = _dot(task, auxiliary, names)
    dot_task_combined = _dot(task, combined, names)
    dot_aux_combined = _dot(auxiliary, combined, names)
    cosine = (
        dot_task_aux / (norm_task * norm_auxiliary)
        if norm_task > 0.0 and norm_auxiliary > 0.0
        else None
    )
    projection_coefficient = dot_task_aux / (norm_task**2) if norm_task > 0.0 else None
    if norm_auxiliary > 0.0 and norm_task > 0.0:
        orthogonal_sq = max(0.0, norm_auxiliary**2 - dot_task_aux**2 / norm_task**2)
        orthogonal_fraction = math.sqrt(orthogonal_sq) / norm_auxiliary
    elif norm_auxiliary > 0.0:
        orthogonal_fraction = 1.0
    else:
        orthogonal_fraction = None
    return {
        "gT_norm": norm_task,
        "gS_norm": norm_scaled_auxiliary,
        "sum_norm": norm_combined,
        "aux_norm": norm_auxiliary,
        "total_norm": norm_combined,
        "task_dot_weighted_aux": dot_task_aux,
        "task_aux_cosine": cosine,
        "aux_projection_coefficient_on_task": projection_coefficient,
        "aux_orthogonal_fraction_to_task": orthogonal_fraction,
        # First-order derivatives along the negative combined-gradient vector.
        # These are not optimizer-aware displacement predictions.
        "task_loss_directional_derivative_along_neg_combined": -dot_task_combined,
        "weighted_aux_loss_directional_derivative_along_neg_combined": -dot_aux_combined,
    }


def combine_window_gradient_buffers(
    adapter: nn.Module,
    task_buffers: Mapping[str, torch.Tensor | None],
    auxiliary_buffers: Mapping[str, torch.Tensor | None],
    *,
    mode: str = "B2",
    eps: float = 1e-12,
    expected_shared_stem_numel: int | None = 19_968,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    """Combine full-window FP32 CPU task and already-weighted auxiliary buffers.

    Inputs must represent the *complete* accumulation window and already be
    divided by its window size. Auxiliary buffers must already include their
    configured loss weight (for example 0.1). Missing gradients remain ``None``;
    real zero tensors remain tensors, including when a coefficient is zero.

    ``B2`` uses ``min(1, 0.25*||gT||/(||gAux||+eps))`` on the adapter's
    ``shared_stem`` group; a zero task shared-stem norm gives coefficient zero.
    Readout-head auxiliary gradients are never scaled. The coefficient affects
    every other auxiliary parameter gradient. The default shared-stem count
    checks the production ``hidden_dim=128`` adapter (19,968 parameters); pass
    a different explicit count for a deliberately smaller synthetic adapter,
    or ``None`` to disable only that size assertion.
    """
    if mode not in _MODE_ALIASES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    canonical_mode = _MODE_ALIASES[mode]
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    parameters, group_by_name, head_names, group_names = _parameter_layout(adapter)
    _validate_buffers(task_buffers, parameters, label="task_buffers")
    _validate_buffers(auxiliary_buffers, parameters, label="auxiliary_buffers")

    shared_names = group_names.get("shared_stem", ())
    shared_numel = sum(parameters[name].numel() for name in shared_names)
    if expected_shared_stem_numel is not None and shared_numel != int(expected_shared_stem_numel):
        raise ValueError(
            f"shared_stem has {shared_numel} parameters, expected "
            f"{expected_shared_stem_numel}"
        )

    if canonical_mode == "B0c1":
        coefficient = 1.0
    elif canonical_mode == "B1c0":
        coefficient = 0.0
    else:
        task_shared_norm = _norm(task_buffers, shared_names)
        aux_shared_norm = _norm(auxiliary_buffers, shared_names)
        coefficient = (
            0.0
            if task_shared_norm == 0.0
            else min(1.0, 0.25 * task_shared_norm / (aux_shared_norm + float(eps)))
        )

    scaled_auxiliary: dict[str, torch.Tensor | None] = {}
    combined: dict[str, torch.Tensor | None] = {}
    for name in parameters:
        raw_aux = auxiliary_buffers[name]
        aux_scale = 1.0 if name in head_names else coefficient
        if raw_aux is None:
            scaled = None
        else:
            scaled = raw_aux * aux_scale
        scaled_auxiliary[name] = scaled

        task = task_buffers[name]
        if task is None and scaled is None:
            combined[name] = None
        elif task is None:
            # Preserve a materialized zero tensor when a present gradient is
            # scaled to zero; do not silently reinterpret it as None.
            assert scaled is not None
            combined[name] = scaled.clone()
        elif scaled is None:
            combined[name] = task.clone()
        else:
            combined[name] = task + scaled

    all_names = tuple(parameters)
    groups = {
        group_name: _combine_norm_diagnostics(
            task_buffers,
            auxiliary_buffers,
            scaled_auxiliary,
            combined,
            names,
        )
        for group_name, names in group_names.items()
    }
    groups["total"] = _combine_norm_diagnostics(
        task_buffers,
        auxiliary_buffers,
        scaled_auxiliary,
        combined,
        all_names,
    )
    diagnostics: dict[str, Any] = {
        "mode": canonical_mode,
        "mode_requested": mode,
        "aux_reader_backflow_coefficient": coefficient,
        "aux_readout_head_coefficient": 1.0,
        "shared_stem_parameter_count": shared_numel,
        "shared_stem_task_norm": _norm(task_buffers, shared_names),
        "shared_stem_weighted_aux_norm": _norm(auxiliary_buffers, shared_names),
        "groups": groups,
        "interpretation": (
            "Gradient-buffer norms, cosine/null decomposition and ordinary first-order "
            "directional derivatives for this window. These do not guarantee Adam/AdamW "
            "parameter displacement."
        ),
    }
    return combined, diagnostics


def parameter_displacement_summary(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float | int]:
    """Summarize an observed parameter displacement, such as one Adam step.

    This reports the tensors supplied by the caller and makes no claim that a
    gradient-space diagnostic predicts the optimizer's update.
    """
    if set(before) != set(after):
        raise ValueError("before and after must have identical parameter-name keys")
    squared = 0.0
    max_abs = 0.0
    changed = 0
    numel = 0
    for name in before:
        left, right = before[name], after[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"parameter snapshots for {name!r} must be tensors")
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise ValueError(f"parameter snapshots for {name!r} have mismatched metadata")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"parameter snapshots for {name!r} contain non-finite values")
        difference = (right - left).double()
        squared += float(torch.sum(difference.square()).item())
        item_max = float(difference.abs().max().item()) if difference.numel() else 0.0
        max_abs = max(max_abs, item_max)
        changed += int(bool(torch.any(difference != 0).item()))
        numel += difference.numel()
    return {
        "l2_norm": math.sqrt(squared),
        "max_abs": max_abs,
        "changed_parameter_tensors": changed,
        "parameter_tensor_count": len(before),
        "numel": numel,
    }


__all__ = [
    "auxiliary_gradient_view",
    "auxiliary_head_backflow",
    "combine_window_gradient_buffers",
    "parameter_displacement_summary",
]
