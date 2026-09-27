"""Multi-step episodic Fullspan fitting for the native TA-STVG temporal head.

This module is deliberately separate from the sealed TA-STVG baseline runner.
It adapts a copied ``temp_embed`` head using the complete native temporal-head
inputs for the two offset views.  On CUDA the replay helper uses TA-STVG's
native FP16 autocast path; CPU tests and development use FP32.

The objective is label-free and intentionally narrow::

    mean(fullspan_prior_loss(native_replay(offset_0)),
         fullspan_prior_loss(native_replay(offset_1)))
    + anchor_gamma * sum((phi - phi_0)**2)

``phi_0`` is captured once at episode entry and is never replaced by an
intermediate iterate.  A single AdamW optimizer (weight_decay=0) is used for
all requested steps.  No annotation, target, or GT argument is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn

from vg_tta.fullspan_tta import fullspan_prior_loss
from vg_tta.tastvg_baseline_expansion import replay_temporal_head_native


def _validate_nonnegative_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not torch.isfinite(torch.tensor(result)) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_optimizer_eps(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("optimizer_eps must be finite and positive") from exc
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError("optimizer_eps must be finite and positive")
    return result


def _validate_steps(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("steps must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("steps must be a non-negative integer")
    return result


def _validate_inputs(head_inputs: Sequence[Tensor]) -> list[Tensor]:
    if isinstance(head_inputs, (str, bytes, Mapping)):
        raise TypeError("head_inputs must be a sequence of exactly two tensors")
    try:
        values = list(head_inputs)
    except TypeError as exc:
        raise TypeError("head_inputs must be a sequence of exactly two tensors") from exc
    if len(values) != 2:
        raise ValueError(
            "TA-STVG Fullspan fitting requires exactly two native offset head inputs"
        )
    detached: list[Tensor] = []
    for index, value in enumerate(values):
        if not isinstance(value, Tensor):
            raise TypeError(f"head_inputs[{index}] must be a torch.Tensor")
        if value.ndim != 4 or value.shape[1] != 1 or value.shape[2] < 1:
            raise ValueError(
                f"head_inputs[{index}] must have complete [L,1,T,D] shape, "
                f"got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"head_inputs[{index}] must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"head_inputs[{index}] must be finite")
        # Detaching is intentional: the adaptation task is only allowed to
        # differentiate through the copied temporal head, never through a
        # feature/cache producer or a hidden target-bearing graph.
        detached.append(value.detach())
    return detached


def _validate_head(head: nn.Module) -> tuple[list[tuple[str, nn.Parameter]], torch.device]:
    if not isinstance(head, nn.Module):
        raise TypeError("head must be a torch.nn.Module")
    named = list(head.named_parameters())
    if not named:
        raise ValueError("temporal head must have parameters")
    if any(not parameter.is_floating_point() for _, parameter in named):
        raise TypeError("all temporal-head parameters must be floating point")
    if any(not bool(torch.isfinite(parameter.detach()).all()) for _, parameter in named):
        raise ValueError("temporal head parameters must be finite")
    device = named[0][1].device
    if any(parameter.device != device for _, parameter in named):
        raise ValueError("all temporal-head parameters must share one device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"native TA-STVG replay supports CPU/CUDA, got {device}")
    if device.type == "cpu" and any(
        parameter.dtype != torch.float32 for _, parameter in named
    ):
        raise ValueError("CPU TA-STVG temporal-head fitting requires FP32 parameters")
    return named, device


def _anchor_loss(
    named: Sequence[tuple[str, nn.Parameter]],
    anchor: Mapping[str, Tensor],
    gamma: float,
) -> Tensor:
    # Keep a differentiable zero on the same device/dtype even when gamma=0.
    result = named[0][1].float().sum() * 0.0
    if gamma == 0.0:
        return result
    for name, parameter in named:
        result = result + (parameter.float() - anchor[name].float()).square().sum()
    return float(gamma) * result


def _task_loss(head: nn.Module, inputs: Sequence[Tensor], device: torch.device) -> Tensor:
    losses = [
        fullspan_prior_loss(replay_temporal_head_native(head, value, device))
        for value in inputs
    ]
    # The mean is formed after both native offset losses, not as two
    # sequential optimizer objectives.
    return torch.stack(losses).mean()


def _finite_gradient_info(
    named: Sequence[tuple[str, nn.Parameter]],
) -> tuple[bool, float, list[str]]:
    squared = 0.0
    missing: list[str] = []
    for name, parameter in named:
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return False, float("nan"), missing
        squared += float(parameter.grad.detach().float().square().sum().item())
    return True, squared**0.5, missing


def _parameter_delta(
    named: Sequence[tuple[str, nn.Parameter]],
    reference: Mapping[str, Tensor],
) -> float:
    squared = 0.0
    for name, parameter in named:
        squared += float(
            (parameter.detach().float() - reference[name].float()).square().sum().item()
        )
    return squared**0.5


def _parameters_finite(named: Sequence[tuple[str, nn.Parameter]]) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for _, parameter in named)


def fit_tastvg_fullspan_multistep(
    head: nn.Module,
    head_inputs: Sequence[Tensor],
    *,
    lr: float,
    steps: int,
    anchor_gamma: float = 1.0e-4,
    optimizer_eps: float = 1.0e-4,
) -> dict[str, Any]:
    """Fit a copied native TA-STVG temporal head for one episode.

    Parameters
    ----------
    head:
        The copied native ``temp_embed`` module.  It is updated in place.
    head_inputs:
        Exactly two complete native offset inputs, each shaped ``[L,1,T,D]``.
        They are detached before replay.
    lr, steps, anchor_gamma, optimizer_eps:
        AdamW and fixed-initial-anchor settings.  ``weight_decay`` is always
        zero.  ``steps=0`` performs finite native replay diagnostics only;
        ``lr=0`` still executes the requested backward/optimizer path but must
        leave parameters bit-exact.

    Returns
    -------
    dict
        JSON-safe per-step and episode diagnostics.  The function never reads
        GT and has no target-bearing argument.
    """

    learning_rate = _validate_nonnegative_finite(lr, "lr")
    number_steps = _validate_steps(steps)
    gamma = _validate_nonnegative_finite(anchor_gamma, "anchor_gamma")
    eps = _validate_optimizer_eps(optimizer_eps)
    named, device = _validate_head(head)
    inputs = _validate_inputs(head_inputs)

    # Capture both mode and requires_grad state before the first mutation so
    # exceptions during replay/backward cannot leak episodic state to callers.
    original_modes = {module: module.training for module in head.modules()}
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in named
    }
    anchor = {name: parameter.detach().clone() for name, parameter in named}
    input_shapes = [list(value.shape) for value in inputs]
    step_audits: list[dict[str, Any]] = []
    optimizer: torch.optim.Optimizer | None = None

    try:
        for _, parameter in named:
            parameter.requires_grad_(True)
        head.eval()

        if number_steps == 0:
            # Exercise the same native replay and prior path, but do not build
            # an optimizer or a graph.  This keeps the no-op auditable while
            # preserving exact parameter identity.
            with torch.no_grad():
                initial_task = _task_loss(head, inputs, device)
                initial_anchor = _anchor_loss(named, anchor, gamma)
                initial_total = initial_task + initial_anchor
            if not bool(torch.isfinite(initial_total).all()):
                raise ValueError("initial Fullspan loss is non-finite")
            final_task = initial_task
            final_anchor = initial_anchor
            final_total = initial_total
            delta = _parameter_delta(named, anchor)
            if delta != 0.0:
                raise RuntimeError("steps=0 Fullspan path changed parameters")
            audit = {
                "record_count": 2,
                "effective_steps": 0,
                "steps_requested": 0,
                "backward_steps": 0,
                "optimizer_steps": 0,
                "skipped": True,
                "device": str(device),
                "lr": learning_rate,
                "anchor_gamma": gamma,
                "optimizer_eps": eps,
                "optimizer": "not constructed (steps=0)",
                "single_optimizer": True,
                "optimizer_parameter_count": 0,
                "all_parameters_in_optimizer": False,
                "weight_decay": 0.0,
                "anchor_definition": "gamma * sum((parameter - initial_parameter)^2)",
                "joint_loss": "mean(fullspan_prior_loss over the two native offsets) + anchor",
                "head_input_shapes": input_shapes,
                "parameter_names": [name for name, _ in named],
                "parameter_delta_l2": 0.0,
                "initial_loss": float(initial_total.item()),
                "final_loss": float(final_total.item()),
                "initial_task_loss": float(initial_task.item()),
                "final_task_loss": float(final_task.item()),
                "initial_anchor_loss": float(initial_anchor.item()),
                "final_anchor_loss": float(final_anchor.item()),
                "all_steps_finite": True,
                "all_optimizer_steps_executed": True,
                "features_detached": True,
                "gt_used": False,
                "native_replay_noop": True,
                "lr_zero_parameter_exact": True,
            }
            return {
                "method": "tastvg_fullspan_multistep",
                "parameter_scope": "temp_embed",
                "objective": audit["joint_loss"],
                "gt_used": False,
                "features_detached": True,
                "steps": [],
                "audit": audit,
            }

        # One optimizer is deliberately constructed outside the step loop.
        # AdamW's moments therefore persist across the complete episode.
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=learning_rate,
            eps=eps,
            weight_decay=0.0,
        )

        for step_index in range(1, number_steps + 1):
            # Keep the native head in inference mode for every episodic
            # update, even if a caller's wrapper changed a child module mode
            # between iterations.
            head.eval()
            before = {
                name: parameter.detach().clone() for name, parameter in named
            }
            optimizer.zero_grad(set_to_none=True)
            task_before = _task_loss(head, inputs, device)
            anchor_before = _anchor_loss(named, anchor, gamma)
            total_before = task_before + anchor_before
            if not bool(torch.isfinite(total_before).all()):
                raise ValueError(
                    f"Fullspan loss before optimizer step {step_index} is non-finite"
                )
            total_before.backward()
            gradients_finite, gradient_norm, missing_gradients = _finite_gradient_info(
                named
            )
            if not gradients_finite:
                raise ValueError(
                    f"Fullspan gradients at optimizer step {step_index} are non-finite"
                )
            if missing_gradients:
                raise RuntimeError(
                    "Fullspan adaptation requires a gradient for every existing "
                    f"head parameter; missing gradients at step {step_index}: "
                    f"{missing_gradients}"
                )
            optimizer.step()
            parameter_exact = all(
                torch.equal(before[name], parameter.detach())
                for name, parameter in named
            )
            if learning_rate == 0.0 and not parameter_exact:
                raise RuntimeError(
                    f"lr=0 Fullspan optimizer step {step_index} changed parameters"
                )
            with torch.no_grad():
                task_after = _task_loss(head, inputs, device)
                anchor_after = _anchor_loss(named, anchor, gamma)
                total_after = task_after + anchor_after
            if not bool(torch.isfinite(total_after).all()):
                raise ValueError(
                    f"Fullspan loss after optimizer step {step_index} is non-finite"
                )
            parameters_finite = _parameters_finite(named)
            if not parameters_finite:
                raise ValueError(
                    f"temporal-head parameters after step {step_index} are non-finite"
                )
            step_audits.append(
                {
                    "step": step_index,
                    "record_count": 2,
                    "head_input_shapes": input_shapes,
                    "loss_before": float(total_before.detach().item()),
                    "task_loss_before": float(task_before.detach().item()),
                    "anchor_loss_before": float(anchor_before.detach().item()),
                    "loss_after": float(total_after.detach().item()),
                    "task_loss_after": float(task_after.detach().item()),
                    "anchor_loss_after": float(anchor_after.detach().item()),
                    "gradient_norm": float(gradient_norm),
                    "finite_gradients": bool(gradients_finite),
                    "missing_gradient_names": missing_gradients,
                    "parameter_delta_l2": _parameter_delta(named, before),
                    "parameter_delta_from_initial_l2": _parameter_delta(named, anchor),
                    "parameter_finite": parameters_finite,
                    "parameter_exact_after_step": parameter_exact,
                    "effective_parameter_update": not parameter_exact,
                    "optimizer_step_executed": True,
                }
            )

        final_step = step_audits[-1]
        first_step = step_audits[0]
        final_delta = _parameter_delta(named, anchor)
        return {
            "method": "tastvg_fullspan_multistep",
            "parameter_scope": "temp_embed",
            "objective": "mean(fullspan_prior_loss over the two native offsets) + anchor",
            "gt_used": False,
            "features_detached": True,
            "steps": step_audits,
            "audit": {
                "record_count": 2,
                "effective_steps": sum(
                    not bool(step["parameter_exact_after_step"])
                    for step in step_audits
                ),
                "steps_requested": number_steps,
                "backward_steps": number_steps,
                "optimizer_steps": number_steps,
                "skipped": False,
                "device": str(device),
                "lr": learning_rate,
                "anchor_gamma": gamma,
                "optimizer_eps": eps,
                "optimizer": "AdamW(weight_decay=0)",
                "single_optimizer": True,
                "optimizer_parameter_count": len(named),
                "all_parameters_in_optimizer": True,
                "weight_decay": 0.0,
                "anchor_definition": "gamma * sum((parameter - initial_parameter)^2)",
                "joint_loss": "mean(fullspan_prior_loss over the two native offsets) + anchor",
                "head_input_shapes": input_shapes,
                "parameter_names": [name for name, _ in named],
                "parameter_delta_l2": final_delta,
                "initial_loss": first_step["loss_before"],
                "final_loss": final_step["loss_after"],
                "initial_task_loss": first_step["task_loss_before"],
                "final_task_loss": final_step["task_loss_after"],
                "initial_anchor_loss": first_step["anchor_loss_before"],
                "final_anchor_loss": final_step["anchor_loss_after"],
                "all_steps_finite": all(
                    bool(step["finite_gradients"] and step["parameter_finite"])
                    for step in step_audits
                ),
                "all_optimizer_steps_executed": all(
                    bool(step["optimizer_step_executed"]) for step in step_audits
                ),
                "features_detached": True,
                "gt_used": False,
                "native_replay_noop": False,
                "lr_zero_parameter_exact": (
                    all(bool(step["parameter_exact_after_step"]) for step in step_audits)
                    if learning_rate == 0.0
                    else None
                ),
            },
        }
    finally:
        # Episodic callers own the copied module, but this helper must not leak
        # mode, grad-enable, or stale-gradient state into the next query.
        for name, parameter in named:
            parameter.requires_grad_(original_requires_grad[name])
            parameter.grad = None
        for module, training in original_modes.items():
            module.training = training


__all__ = ["fit_tastvg_fullspan_multistep"]
