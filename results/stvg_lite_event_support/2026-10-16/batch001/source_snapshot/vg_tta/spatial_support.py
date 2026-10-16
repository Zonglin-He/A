"""Label-free support adaptation for a TubeDETR spatial bbox head.

The functions in this module intentionally operate on a complete cached bbox
head input with shape ``[L, T, D]``.  They do not know about a model, decoder,
annotation, query, or target-selection policy.  A caller may make a private
copy of the bbox head and pass only teacher boxes and non-negative support
weights here.

The CPU path is FP32.  The CUDA path uses the native BF16 autocast context and
keeps the full input shape intact before selecting the final decoder layer.
Loss computation is promoted to FP32 so that the support objective is not
silently changed by BF16 reduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .metrics import generalized_box_iou_aligned_cxcywh


_RECORD_KEYS = frozenset({"head_input", "teacher_boxes", "weights"})


def _as_device(device: torch.device | str) -> torch.device:
    try:
        value = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid spatial-head device: {device!r}") from exc
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA spatial-head replay requested but CUDA is unavailable")
    if value.type == "cuda" and value.index is None:
        # ``torch.device('cuda')`` is a valid caller spelling, but it does not
        # compare equal to ``cuda:0``.  Resolve it once so the head/device
        # contract works with the runner's ordinary ``device='cuda'`` value.
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
    if head_input.ndim != 3:
        raise ValueError(
            f"head_input must retain the complete [L,T,D] shape, got {tuple(head_input.shape)}"
        )
    if not head_input.is_floating_point():
        raise TypeError("head_input must be a floating-point tensor")
    if not torch.isfinite(head_input).all():
        raise ValueError("head_input must be finite")


def _validate_head_output(output: Any, head_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise TypeError("bbox head must return a tensor shaped [L,T,4]")
    if output.ndim != 3:
        raise ValueError(f"bbox head output must be [L,T,4], got {tuple(output.shape)}")
    if output.shape[0] != head_input.shape[0] or output.shape[1] != head_input.shape[1]:
        raise ValueError(
            "bbox head output must preserve the first two dimensions of head_input "
            f"({tuple(head_input.shape)} vs {tuple(output.shape)})"
        )
    if output.shape[-1] != 4:
        raise ValueError(f"bbox head output must have four box coordinates, got {output.shape[-1]}")
    if not torch.isfinite(output).all():
        raise ValueError("bbox head logits must be finite")
    return output


def replay_box_head(
    head: nn.Module,
    head_input: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Replay the native bbox head and return final-layer sigmoid boxes.

    ``head_input`` is deliberately not flattened, cropped, or otherwise
    reshaped.  The head must return ``[L,T,4]``; the result is exactly the
    final layer ``head(head_input).sigmoid()[-1]``.  The caller owns the head's
    device placement and parameter snapshotting.
    """

    _validate_head_input(head_input)
    target_device = _as_device(device)
    if target_device.type == "cpu":
        module_device = _module_device(head)
        if module_device.type != "cpu":
            raise ValueError(
                f"CPU replay requires a CPU head, but the head is on {module_device}"
            )
        # The locked CPU reference is FP32 rather than BF16/FP16.
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in head.parameters()
        ):
            raise ValueError("CPU spatial-head replay requires FP32 head parameters")
        input_on_device = head_input.to(device=target_device, dtype=torch.float32)
        output = head(input_on_device)
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
            boxes = logits.sigmoid()[-1]
    if target_device.type == "cpu":
        logits = _validate_head_output(output, input_on_device)
        boxes = logits.sigmoid()[-1]
    if not torch.isfinite(boxes).all():
        raise ValueError("sigmoid bbox output must be finite")
    return boxes


def _validate_box_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 2 or value.shape[-1] != 4:
        raise ValueError(f"{name} must be shaped [T,4], got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _zero_loss(pred_boxes: torch.Tensor) -> torch.Tensor:
    # Keep a real autograd edge for null/no-GT support.  ``sum`` also works for
    # an empty [0,4] tensor and returns a scalar zero with the right device.
    return pred_boxes.float().sum() * 0.0


def spatial_loss(
    pred_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the weighted final-layer spatial support objective.

    For each aligned frame the objective is ``5 * sum(abs(pred-teacher)) +
    2 * (1-aligned_GIoU)``.  Non-negative weights define the support measure;
    the denominator is their sum.  An empty or all-zero weight vector is a
    legal null/no-GT case and returns a differentiable zero.  Teacher boxes
    and weights are detached before use, so neither can receive gradients.
    """

    _validate_box_tensor(pred_boxes, "pred_boxes")
    _validate_box_tensor(teacher_boxes, "teacher_boxes")
    if pred_boxes.shape != teacher_boxes.shape:
        raise ValueError(
            "pred_boxes and teacher_boxes must be aligned [T,4] tensors "
            f"({tuple(pred_boxes.shape)} vs {tuple(teacher_boxes.shape)})"
        )
    if not torch.is_tensor(weights):
        raise TypeError("weights must be a tensor")
    if weights.ndim != 1:
        raise ValueError(f"weights must be shaped [T], got {tuple(weights.shape)}")
    if weights.numel() and not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if weights.numel() and bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    # Empty weights represent an intentionally unlabeled/null support record;
    # they are permitted even when a full teacher tensor is retained in the
    # record for shape provenance.  A non-empty vector must align exactly.
    if weights.numel() not in (0, pred_boxes.shape[0]):
        raise ValueError(
            "weights must have one value per box frame or be empty "
            f"({weights.numel()} vs {pred_boxes.shape[0]})"
        )

    pred = pred_boxes.float()
    teacher = teacher_boxes.detach().to(device=pred.device, dtype=torch.float32)
    detached_weights = weights.detach().to(device=pred.device, dtype=torch.float32)
    if detached_weights.numel() == 0:
        return _zero_loss(pred)
    denominator = detached_weights.sum()
    if float(denominator.detach().item()) == 0.0:
        return _zero_loss(pred)

    aligned_giou = generalized_box_iou_aligned_cxcywh(pred, teacher)
    per_frame = 5.0 * (pred - teacher).abs().sum(dim=-1) + 2.0 * (1.0 - aligned_giou)
    return (per_frame * detached_weights).sum() / denominator


def _validate_record(record: Mapping[str, Any], index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(record, Mapping):
        raise TypeError(f"support record {index} must be a mapping")
    keys = set(record.keys())
    if keys != _RECORD_KEYS:
        missing = sorted(_RECORD_KEYS - keys)
        extra = sorted(keys - _RECORD_KEYS)
        raise ValueError(
            f"support record {index} must contain exactly head_input, teacher_boxes, weights; "
            f"missing={missing}, extra={extra}"
        )
    head_input = record["head_input"]
    teacher_boxes = record["teacher_boxes"]
    weights = record["weights"]
    _validate_head_input(head_input)
    _validate_box_tensor(teacher_boxes, f"record[{index}].teacher_boxes")
    if not torch.is_tensor(weights):
        raise TypeError(f"record[{index}].weights must be a tensor")
    if weights.ndim != 1:
        raise ValueError(f"record[{index}].weights must be shaped [T]")
    if weights.numel() not in (0, teacher_boxes.shape[0]):
        raise ValueError(
            f"record[{index}].weights does not align with teacher_boxes "
            f"({weights.numel()} vs {teacher_boxes.shape[0]})"
        )
    if weights.numel() and (not torch.isfinite(weights).all() or bool((weights < 0).any())):
        raise ValueError(f"record[{index}].weights must be finite and non-negative")
    if head_input.shape[1] != teacher_boxes.shape[0]:
        raise ValueError(
            f"record[{index}] head_input T={head_input.shape[1]} does not align with "
            f"teacher_boxes T={teacher_boxes.shape[0]}"
        )
    return head_input, teacher_boxes, weights


def _anchor_loss(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    anchor: Mapping[str, torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    if gamma == 0.0:
        # Keep the result connected to the first parameter for a uniform
        # backward path, including models whose task branch is null-weighted.
        first = named_parameters[0][1]
        return first.float().sum() * 0.0
    total = named_parameters[0][1].float().sum() * 0.0
    for name, parameter in named_parameters:
        total = total + (parameter.float() - anchor[name].float()).square().sum()
    return float(gamma) * total


def _finite_gradients(named_parameters: Sequence[tuple[str, nn.Parameter]]) -> tuple[bool, float, list[str]]:
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


def fit_spatial_head(
    head: nn.Module,
    records: Sequence[Mapping[str, Any]],
    lr: float,
    anchor_gamma: float = 1e-4,
    optimizer_eps: float = 1e-4,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Adapt ``head`` over support records, one AdamW step per record.

    The passed head is updated in place; callers that need episodic isolation
    should deep-copy it before calling this function.  The anchor is the
    parameter snapshot at function entry and uses ``gamma * sum(delta**2)``
    (a sum, never a mean).  ``lr=0`` still executes backward and
    ``optimizer.step()`` for every record, while requiring exact parameter
    equality before and after each step.
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
    records_list = list(records)
    validated = [_validate_record(record, index) for index, record in enumerate(records_list)]

    named_parameters = list(head.named_parameters())
    if not named_parameters:
        raise ValueError("spatial head must have parameters")
    if any(not torch.isfinite(parameter.detach()).all() for _, parameter in named_parameters):
        raise ValueError("spatial head parameters must be finite")
    target_device = _as_device(device if device is not None else _module_device(head))
    if _module_device(head) != target_device:
        raise ValueError(f"head is on {_module_device(head)}, requested device is {target_device}")
    if target_device.type == "cpu" and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for _, parameter in named_parameters
    ):
        raise ValueError("CPU spatial-head fitting requires FP32 head parameters")

    # The bbox head is an inference-time module.  Temporarily force eval mode
    # so support steps cannot mutate BatchNorm/dropout behavior, then restore
    # the caller's mode and requires_grad flags after optimization.
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
        for step_index, (head_input, teacher_boxes, weights) in enumerate(validated, start=1):
            before = {name: parameter.detach().clone() for name, parameter in named_parameters}
            optimizer.zero_grad(set_to_none=True)
            pred_before = replay_box_head(head, head_input, target_device)
            task_before = spatial_loss(pred_before, teacher_boxes, weights)
            anchor_before = _anchor_loss(named_parameters, anchor, anchor_gamma)
            total_before = task_before + anchor_before
            if not torch.isfinite(total_before).all():
                raise ValueError(f"support step {step_index} produced a non-finite loss before update")
            total_before.backward()
            finite_gradients, gradient_norm, missing_gradients = _finite_gradients(named_parameters)
            if not finite_gradients:
                raise ValueError(f"support step {step_index} produced non-finite gradients")
            # This call is intentionally unconditional, including lr=0.
            optimizer.step()
            after = {name: parameter.detach().clone() for name, parameter in named_parameters}
            parameter_exact = all(torch.equal(before[name], after[name]) for name in before)
            if lr == 0.0 and not parameter_exact:
                raise RuntimeError("lr=0 support step changed parameters")
            with torch.no_grad():
                pred_after = replay_box_head(head, head_input, target_device)
                task_after = spatial_loss(pred_after, teacher_boxes, weights)
                anchor_after = _anchor_loss(named_parameters, anchor, anchor_gamma)
                total_after = task_after + anchor_after
            if not torch.isfinite(total_after).all():
                raise ValueError(f"support step {step_index} produced a non-finite loss after update")
            step_audits.append(
                {
                    "step": step_index,
                    "head_input_shape": list(head_input.shape),
                    "weight_count": int(weights.numel()),
                    "weight_sum": float(weights.detach().float().sum().item()) if weights.numel() else 0.0,
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
            module.train(training)

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
            "head_input_shapes": [list(value[0].shape) for value in validated],
            "parameter_names": [name for name, _ in named_parameters],
            "parameter_delta_l2": _parameter_delta(named_parameters, anchor),
            "all_steps_finite": all(step["finite_gradients"] and step["parameter_finite"] for step in step_audits),
            "all_optimizer_steps_executed": all(step["optimizer_step_executed"] for step in step_audits),
            "lr_zero_parameter_exact": all(
                step["parameter_exact_after_step"] for step in step_audits
            ) if lr == 0.0 else None,
        },
    }


__all__ = ["fit_spatial_head", "replay_box_head", "spatial_loss"]
