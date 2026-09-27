"""Label-free full-span temporal-head test-time adaptation primitives.

This module implements a deliberately narrow prior-distillation objective: it
only rewards the first time position for the start channel and the last time
position for the end channel.  It never reads ground truth, annotations, or a
temporal target.  The objective is a structural full-span prior, not a claim
of event-semantic self-supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_RECORD_KEYS = frozenset({"head_input"})


def _as_device(device: torch.device | str) -> torch.device:
    try:
        value = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid temporal-head device: {device!r}") from exc
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA temporal-head replay requested but CUDA is unavailable")
    if value.type == "cuda" and value.index is None:
        value = torch.device("cuda", torch.cuda.current_device())
    return value


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _validate_head_input(head_input: torch.Tensor) -> None:
    if not torch.is_tensor(head_input):
        raise TypeError("head_input must be a tensor")
    if head_input.ndim != 4:
        raise ValueError(
            "head_input must retain the complete [L,1,T,D] shape, "
            f"got {tuple(head_input.shape)}"
        )
    if head_input.shape[1] != 1:
        raise ValueError(f"head_input batch dimension must be 1, got {head_input.shape[1]}")
    if head_input.shape[2] < 1:
        raise ValueError("head_input must contain at least one time position")
    if not head_input.is_floating_point():
        raise TypeError("head_input must be floating point")
    if not torch.isfinite(head_input).all():
        raise ValueError("head_input must be finite")


def _validate_head_output(output: Any, head_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise TypeError("temporal head must return a tensor shaped [L,1,T,2]")
    if output.ndim != 4:
        raise ValueError(f"temporal head output must be [L,1,T,2], got {tuple(output.shape)}")
    expected_prefix = tuple(head_input.shape[:3])
    if tuple(output.shape[:3]) != expected_prefix:
        raise ValueError(
            "temporal head output must preserve [L,1,T] from head_input "
            f"({expected_prefix} vs {tuple(output.shape[:3])})"
        )
    if output.shape[-1] != 2:
        raise ValueError(f"temporal head output must have two channels, got {output.shape[-1]}")
    if not torch.isfinite(output).all():
        raise ValueError("temporal head logits must be finite")
    return output


def replay_temporal_head(
    head: nn.Module,
    head_input: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Replay a native temporal head and return final-layer ``[1,T,2]`` logits.

    The complete ``[L,1,T,D]`` input is passed to the head unchanged apart
    from device/dtype placement.  CPU replay is FP32; CUDA replay uses BF16
    autocast for the head call and final-layer selection.
    """

    _validate_head_input(head_input)
    target_device = _as_device(device)
    if target_device.type == "cpu":
        module_device = _module_device(head)
        if module_device.type != "cpu":
            raise ValueError(f"CPU replay requires a CPU head, but the head is on {module_device}")
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in head.parameters()
        ):
            raise ValueError("CPU temporal-head replay requires FP32 head parameters")
        input_on_device = head_input.to(device=target_device, dtype=torch.float32)
        output = head(input_on_device)
        logits = _validate_head_output(output, input_on_device)
        result = logits[-1]
    else:
        module_device = _module_device(head)
        if module_device != target_device:
            raise ValueError(
                f"CUDA replay device {target_device} does not match head device {module_device}"
            )
        input_on_device = head_input.to(device=target_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head(input_on_device)
            logits = _validate_head_output(output, input_on_device)
            result = logits[-1]
    if tuple(result.shape) != (1, head_input.shape[2], 2):
        raise ValueError(f"final temporal logits must be [1,T,2], got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("final temporal logits must be finite")
    return result


def fullspan_prior_loss(logits: torch.Tensor) -> torch.Tensor:
    """Return the GT-free first-start/last-end temporal prior loss.

    The input is ``[1,T,2]``.  Log-softmax is taken over time independently
    for the two channels, then the first position/channel 0 and last
    position/channel 1 are averaged with a leading negative sign.  ``T=1``
    is legal and yields a differentiable zero.
    """

    if not torch.is_tensor(logits):
        raise TypeError("logits must be a tensor")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise ValueError(f"logits must be shaped [1,T,2], got {tuple(logits.shape)}")
    if logits.shape[1] < 1:
        raise ValueError("logits must contain at least one time position")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    values = logits.float()
    log_probability = F.log_softmax(values, dim=1)
    return -0.5 * (log_probability[0, 0, 0] + log_probability[0, -1, 1])


def _validate_record(record: Mapping[str, Any], index: int) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError(f"adaptation record {index} must be a mapping")
    keys = set(record.keys())
    if keys != _RECORD_KEYS:
        missing = sorted(_RECORD_KEYS - keys)
        extra = sorted(keys - _RECORD_KEYS)
        raise ValueError(
            f"adaptation record {index} must contain exactly head_input; "
            f"missing={missing}, extra={extra}"
        )
    head_input = record["head_input"]
    _validate_head_input(head_input)
    return head_input


def _anchor_loss(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    anchor: Mapping[str, torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    if gamma == 0.0:
        return named_parameters[0][1].float().sum() * 0.0
    total = named_parameters[0][1].float().sum() * 0.0
    for name, parameter in named_parameters:
        total = total + (parameter.float() - anchor[name].float()).square().sum()
    return float(gamma) * total


def _finite_gradients(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[bool, float, list[str]]:
    norm_squared = 0.0
    missing: list[str] = []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            return False, float("nan"), missing
        norm_squared += float(parameter.grad.detach().float().square().sum().item())
    return True, norm_squared**0.5, missing


def _parameter_delta(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    reference: Mapping[str, torch.Tensor],
) -> float:
    squared = 0.0
    for name, parameter in named_parameters:
        squared += float((parameter.detach().float() - reference[name].float()).square().sum().item())
    return squared**0.5


def fit_fullspan_head(
    head: nn.Module,
    records: Sequence[Mapping[str, Any]],
    lr: float,
    anchor_gamma: float = 1e-4,
    optimizer_eps: float = 1e-4,
) -> dict[str, Any]:
    """Update ``head`` in place with one prior-distillation step per record.

    The anchor is a single entry snapshot and uses a sum of squared parameter
    displacements.  The head is evaluated in ``eval`` mode throughout the
    update, then all original module modes, ``requires_grad`` flags, and
    gradients are restored.  The returned object contains only JSON-safe
    diagnostics; it does not contain the module itself.
    """

    if not isinstance(head, nn.Module):
        raise TypeError("head must be a torch.nn.Module")
    if not torch.isfinite(torch.tensor(float(lr))) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(anchor_gamma))) or anchor_gamma < 0:
        raise ValueError("anchor_gamma must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(optimizer_eps))) or optimizer_eps <= 0:
        raise ValueError("optimizer_eps must be finite and positive")
    if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of strict support mappings")
    validated = [_validate_record(record, index) for index, record in enumerate(list(records))]

    named_parameters = list(head.named_parameters())
    if not named_parameters:
        raise ValueError("temporal head must have parameters")
    if any(not torch.isfinite(parameter.detach()).all() for _, parameter in named_parameters):
        raise ValueError("temporal head parameters must be finite")
    target_device = _as_device(_module_device(head))
    if target_device.type == "cpu" and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for _, parameter in named_parameters
    ):
        raise ValueError("CPU temporal-head fitting requires FP32 head parameters")

    original_modes = {module: module.training for module in head.modules()}
    original_requires_grad = {name: parameter.requires_grad for name, parameter in named_parameters}
    anchor = {name: parameter.detach().clone() for name, parameter in named_parameters}
    for _, parameter in named_parameters:
        parameter.requires_grad_(True)
    head.eval()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_parameters],
        lr=float(lr),
        eps=float(optimizer_eps),
        weight_decay=0.0,
    )
    step_audits: list[dict[str, Any]] = []
    try:
        for step_index, head_input in enumerate(validated, start=1):
            before = {name: parameter.detach().clone() for name, parameter in named_parameters}
            optimizer.zero_grad(set_to_none=True)
            logits_before = replay_temporal_head(head, head_input, target_device)
            task_before = fullspan_prior_loss(logits_before)
            anchor_before = _anchor_loss(named_parameters, anchor, anchor_gamma)
            total_before = task_before + anchor_before
            if not torch.isfinite(total_before).all():
                raise ValueError(f"adaptation step {step_index} produced a non-finite loss before update")
            total_before.backward()
            finite_gradients, gradient_norm, missing_gradients = _finite_gradients(named_parameters)
            if not finite_gradients:
                raise ValueError(f"adaptation step {step_index} produced non-finite gradients")
            # Deliberately unconditional: lr=0 still exercises the optimizer
            # path and is audited for exact parameter preservation.
            optimizer.step()
            after = {name: parameter.detach().clone() for name, parameter in named_parameters}
            parameter_exact = all(torch.equal(before[name], after[name]) for name in before)
            if lr == 0.0 and not parameter_exact:
                raise RuntimeError("lr=0 adaptation step changed parameters")
            with torch.no_grad():
                logits_after = replay_temporal_head(head, head_input, target_device)
                task_after = fullspan_prior_loss(logits_after)
                anchor_after = _anchor_loss(named_parameters, anchor, anchor_gamma)
                total_after = task_after + anchor_after
            if not torch.isfinite(total_after).all():
                raise ValueError(f"adaptation step {step_index} produced a non-finite loss after update")
            step_audits.append(
                {
                    "step": step_index,
                    "head_input_shape": list(head_input.shape),
                    "time_count": int(head_input.shape[2]),
                    "loss_before": float(total_before.detach().item()),
                    "task_loss_before": float(task_before.detach().item()),
                    "anchor_loss_before": float(anchor_before.detach().item()),
                    "loss_after": float(total_after.detach().item()),
                    "task_loss_after": float(task_after.detach().item()),
                    "anchor_loss_after": float(anchor_after.detach().item()),
                    "gradient_norm": float(gradient_norm),
                    "finite_gradients": bool(finite_gradients),
                    "missing_gradient_names": missing_gradients,
                    "parameter_delta_l2": _parameter_delta(named_parameters, before),
                    "cumulative_parameter_delta_l2": _parameter_delta(named_parameters, anchor),
                    "parameter_finite": all(torch.isfinite(parameter).all().item() for parameter in head.parameters()),
                    "parameter_exact_after_step": parameter_exact,
                    "optimizer_step_executed": True,
                }
            )
    finally:
        for name, parameter in named_parameters:
            parameter.requires_grad_(original_requires_grad[name])
            parameter.grad = None
        for module, training in original_modes.items():
            module.training = training

    return {
        "steps": step_audits,
        "audit": {
            "record_count": len(validated),
            "effective_steps": len(step_audits),
            "backward_steps": len(step_audits),
            "optimizer_steps": len(step_audits),
            "skipped": len(validated) == 0,
            "device": str(target_device),
            "lr": float(lr),
            "anchor_gamma": float(anchor_gamma),
            "anchor_definition": "gamma * sum((parameter - initial_parameter)^2)",
            "optimizer": "AdamW(weight_decay=0)",
            "optimizer_eps": float(optimizer_eps),
            "head_input_shapes": [list(value.shape) for value in validated],
            "parameter_names": [name for name, _ in named_parameters],
            "parameter_delta_l2": _parameter_delta(named_parameters, anchor),
            "all_steps_finite": all(step["finite_gradients"] and step["parameter_finite"] for step in step_audits),
            "all_optimizer_steps_executed": all(step["optimizer_step_executed"] for step in step_audits),
            "lr_zero_parameter_exact": all(
                step["parameter_exact_after_step"] for step in step_audits
            ) if lr == 0.0 else None,
        },
    }


__all__ = ["fit_fullspan_head", "fullspan_prior_loss", "replay_temporal_head"]

