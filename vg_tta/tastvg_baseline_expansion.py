"""Small, explicit TA-STVG baseline-expansion primitives.

The expansion is intentionally parameter-only.  Spatial predictions stay on
the original native views and are not fused or adapted here.  The temporal
methods are:

``frozen``
    Native endpoint logits and boxes, without an optimizer.
``direct_fullspan``
    A no-parameter structural control that replaces only endpoint logits with
    a fixed first-start/last-end prior.  It is not an adaptation method.
``ours``
    A fresh copy of the native ``temp_embed`` fitted with the existing
    label-free Fullspan prior and AdamW (the caller supplies ``head_input``).
``tent`` / ``memo``
    Endpoint-entropy baselines from :mod:`vg_tta.external_tta_baselines`.
    They are valid only with a live forward closure and an explicit parameter
    scope.  A cached tensor after the temporal head cannot produce a gradient
    to decoder LayerNorm parameters, so this module never silently labels a
    detached-cache operation as TENT or MEMO.

No function in this module reads annotations or ground truth.  The runner may
use labels later, after all predictions are produced, for evaluation only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from vg_tta.fullspan_tta import fullspan_prior_loss


METHODS = ("frozen", "direct_fullspan", "ours", "tent", "memo")
EXTERNAL_METHODS = frozenset({"tent", "memo"})


def replay_temporal_head_native(
    head: nn.Module,
    head_input: Tensor,
    device: torch.device | str,
) -> Tensor:
    """Replay TA-STVG's temporal head with its native FP16 CUDA pathway.

    The generic Fullspan helper uses BF16 autocast on CUDA, which is not the
    numerical path used by the TA-STVG pipeline.  This TA-specific replay
    keeps CUDA autocast at ``float16`` and consumes the complete native
    ``[L,1,T,D]`` head input.  CPU replay is ordinary FP32.  The result is the
    final-layer endpoint tensor ``[1,T,2]``.
    """

    if not isinstance(head, nn.Module):
        raise TypeError("head must be a torch.nn.Module")
    if not isinstance(head_input, Tensor):
        raise TypeError("head_input must be a torch.Tensor")
    if head_input.ndim != 4 or head_input.shape[1] != 1 or head_input.shape[2] < 1:
        raise ValueError(
            f"head_input must have shape [L,1,T,D], got {tuple(head_input.shape)}"
        )
    if not head_input.is_floating_point() or not bool(torch.isfinite(head_input).all()):
        raise ValueError("head_input must be finite floating point")
    target_device = torch.device(device)
    parameters = list(head.parameters())
    if not parameters:
        raise ValueError("temporal head must have parameters")
    module_device = parameters[0].device
    if target_device.type == "cuda" and target_device.index is None:
        target_device = module_device
    if module_device != target_device:
        raise ValueError(
            f"native replay device {target_device} does not match head device {module_device}"
        )
    if target_device.type == "cpu":
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in parameters
        ):
            raise ValueError("CPU native temporal replay requires FP32 head parameters")
        input_on_device = head_input.to(device=target_device, dtype=torch.float32)
        output = head(input_on_device)
    elif target_device.type == "cuda":
        input_on_device = head_input.to(device=target_device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = head(input_on_device)
    else:
        raise ValueError(f"native temporal replay supports CPU/CUDA, got {target_device}")
    if not isinstance(output, Tensor) or output.ndim != 4:
        raise ValueError(
            "native temporal head must return [L,1,T,2] tensor before final-layer selection"
        )
    result = output[-1]
    expected = (1, int(head_input.shape[2]), 2)
    if tuple(result.shape) != expected:
        raise ValueError(
            f"native temporal logits must be {expected}, got {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("native temporal logits must be finite")
    return result


def _endpoint_shape(logits: Tensor) -> Tensor:
    if not isinstance(logits, Tensor):
        raise TypeError("endpoint logits must be a torch.Tensor")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise ValueError(f"endpoint logits must have shape [1,T,2], got {tuple(logits.shape)}")
    if logits.shape[1] < 1:
        raise ValueError("endpoint logits must contain at least one temporal position")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("endpoint logits must be finite floating point")
    return logits


def direct_fullspan_logits(logits: Tensor, strength: float = 10.0) -> Tensor:
    """Return the fixed Fullspan endpoint control without changing boxes.

    The output has the same shape/dtype/device as ``logits``.  It is detached
    from the input because this control has no trainable parameters.  For a
    one-frame input both endpoint channels are placed at that only position.
    """

    source = _endpoint_shape(logits)
    if not torch.isfinite(torch.tensor(float(strength))):
        raise ValueError("direct Fullspan strength must be finite")
    result = torch.zeros_like(source).detach()
    result[0, 0, 0] = float(strength)
    result[0, -1, 1] = float(strength)
    return result


def pad_view_logits(
    logits: Tensor,
    view_frame_ids: Sequence[int],
    full_frame_ids: Sequence[int],
    *,
    fill_value: float = -1.0e4,
) -> Tensor:
    """Scatter one native temporal view onto a common full-frame timeline.

    This legacy diagnostic utility is retained for explicit unit tests and
    older exploratory callers.  The v2 episodic runner deliberately does not
    use it for MEMO: MEMO receives same-frame-id original/photometric views,
    so no missing-frame sentinel or artificial offset alignment enters its
    entropy objective.
    """

    source = _endpoint_shape(logits)
    view = [int(value) for value in view_frame_ids]
    full = [int(value) for value in full_frame_ids]
    if len(view) != source.shape[1]:
        raise ValueError("view frame-id count does not match endpoint-logit length")
    if full != sorted(set(full)) or view != sorted(set(view)):
        raise ValueError("view/full frame IDs must be sorted and unique")
    positions = {frame_id: index for index, frame_id in enumerate(full)}
    if any(frame_id not in positions for frame_id in view):
        raise ValueError("view frame IDs must be a subset of the full timeline")
    if not torch.isfinite(torch.tensor(float(fill_value))):
        raise ValueError("fill_value must be finite")
    result = torch.full(
        (1, len(full), 2),
        float(fill_value),
        dtype=source.dtype,
        device=source.device,
    )
    destination = torch.as_tensor(
        [positions[frame_id] for frame_id in view], dtype=torch.long, device=source.device
    )
    result[0].index_copy_(0, destination, source[0])
    return result


def temporal_head_scope(model: nn.Module) -> tuple[nn.Module, list[str], list[nn.Parameter]]:
    """Return the exact native ``temp_embed`` module and its parameters."""

    head = getattr(model, "temp_embed", None)
    if not isinstance(head, nn.Module):
        raise ValueError("TA-STVG model has no nn.Module temp_embed temporal head")
    named = list(head.named_parameters())
    if not named:
        raise ValueError("TA-STVG temp_embed has no parameters")
    names = [f"temp_embed.{name}" for name, _ in named]
    return head, names, [parameter for _, parameter in named]


def decoder_layernorm_scope(
    model: nn.Module,
) -> tuple[list[str], list[nn.Parameter], dict[str, Any]]:
    """Select live decoder ``nn.LayerNorm`` affine parameters for TENT/MEMO.

    The scope is deliberately narrower than every normalization layer in the
    complete TA-STVG model.  It excludes ``temp_embed`` and all encoder/vision
    modules.  If a caller only has a post-head cache, it must not call this
    selector: the returned parameters require a live closure through the
    decoder for their gradients to be meaningful.
    """

    names: list[str] = []
    parameters: list[nn.Parameter] = []
    modules = dict(model.named_modules())
    for module_name, module in modules.items():
        if not module_name.startswith("ground_decoder"):
            continue
        if not isinstance(module, nn.LayerNorm):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if isinstance(parameter, nn.Parameter):
                names.append(f"{module_name}.{parameter_name}")
                parameters.append(parameter)
    if not parameters:
        raise ValueError(
            "no ground_decoder nn.LayerNorm affine parameters were found; "
            "TENT/MEMO are not evaluable from a detached post-head cache"
        )
    return names, parameters, {
        "name": "ground_decoder_layernorm_affine",
        "parameter_names": names,
        "live_forward_required": True,
        "post_head_cache_allowed": False,
        "excluded": ["temp_embed", "vision_encoder", "language_encoder", "ground_encoder"],
    }


def fit_ours_temporal_head(
    head: nn.Module,
    head_inputs: Sequence[Tensor],
    *,
    lr: float = 1.0e-2,
    anchor_gamma: float = 1.0e-4,
    optimizer_eps: float = 1.0e-4,
    steps: int = 1,
) -> dict[str, Any]:
    """Fit a native temporal head with one joint Fullspan optimizer step.

    ``head_inputs`` normally contains the two native temporal-offset inputs
    from one query.  The task losses are averaged before the single AdamW
    ``step``; they are not applied as two sequential updates.  The older
    :func:`vg_tta.fullspan_tta.fit_fullspan_head` remains available for its
    original multi-record contract, but is intentionally not used for this
    main per-query expansion protocol.
    """

    if isinstance(steps, bool) or int(steps) != steps or int(steps) not in {0, 1}:
        raise ValueError("the episodic temporal-head update supports steps=0 or steps=1")
    steps = int(steps)
    if not torch.isfinite(torch.tensor(float(lr))) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(anchor_gamma))) or anchor_gamma < 0:
        raise ValueError("anchor_gamma must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(optimizer_eps))) or optimizer_eps <= 0:
        raise ValueError("optimizer_eps must be finite and positive")
    if isinstance(head_inputs, (str, bytes)):
        raise TypeError("head_inputs must be a sequence of tensors")
    inputs = list(head_inputs)
    if not isinstance(head, nn.Module):
        raise TypeError("head must be a torch.nn.Module")
    if not inputs:
        result = {
            "steps": [],
            "audit": {
                "record_count": 0,
                "effective_steps": 0,
                "backward_steps": 0,
                "optimizer_steps": 0,
                "skipped": True,
                "parameter_delta_l2": 0.0,
                "joint_loss": "mean(fullspan_prior_loss over native offset records)",
                "gt_used": False,
            },
        }
    elif steps == 0:
        for index, value in enumerate(inputs):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 4
                or value.shape[1] != 1
                or value.shape[2] < 1
            ):
                raise ValueError(f"head_inputs[{index}] must have shape [L,1,T,D]")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"head_inputs[{index}] must be finite floating point")
        result = {
            "steps": [],
            "audit": {
                "record_count": len(inputs),
                "effective_steps": 0,
                "backward_steps": 0,
                "optimizer_steps": 0,
                "skipped": True,
                "parameter_delta_l2": 0.0,
                "lr": float(lr),
                "anchor_gamma": float(anchor_gamma),
                "optimizer": "not constructed (steps=0)",
                "native_replay_noop": True,
                "lr_zero_parameter_exact": True,
                "gt_used": False,
            },
        }
    else:
        result = _fit_joint_fullspan_head(
            head,
            inputs,
            lr=float(lr),
            anchor_gamma=float(anchor_gamma),
            optimizer_eps=float(optimizer_eps),
        )
    result["method"] = "ours"
    result["parameter_scope"] = "temp_embed"
    result["objective"] = "fullspan_prior_loss"
    result["gt_used"] = False
    result["spatial_adaptation"] = False
    result["adaptation_unit"] = "one_query"
    result["audit"]["steps_requested"] = steps
    return result


def _fit_joint_fullspan_head(
    head: nn.Module,
    head_inputs: Sequence[Tensor],
    *,
    lr: float,
    anchor_gamma: float,
    optimizer_eps: float,
) -> dict[str, Any]:
    """Internal one-step joint Fullspan update used by the v2 protocol."""

    if not torch.isfinite(torch.tensor(float(lr))) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(anchor_gamma))) or anchor_gamma < 0:
        raise ValueError("anchor_gamma must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(optimizer_eps))) or optimizer_eps <= 0:
        raise ValueError("optimizer_eps must be finite and positive")
    named = list(head.named_parameters())
    if not named:
        raise ValueError("temporal head must have parameters")
    device = next(iter(head.parameters())).device
    if device.type == "cpu" and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for _, parameter in named
    ):
        raise ValueError("CPU temporal-head fitting requires FP32 parameters")
    for index, value in enumerate(head_inputs):
        if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1 or value.shape[2] < 1:
            raise ValueError(f"head_inputs[{index}] must have shape [L,1,T,D]")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"head_inputs[{index}] must be finite floating point")

    original_modes = {module: module.training for module in head.modules()}
    original_requires_grad = {name: parameter.requires_grad for name, parameter in named}
    anchor = {name: parameter.detach().clone() for name, parameter in named}
    for _, parameter in named:
        parameter.requires_grad_(True)
    head.eval()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=float(lr),
        eps=float(optimizer_eps),
        weight_decay=0.0,
    )

    def anchor_loss() -> Tensor:
        base = named[0][1].float().sum() * 0.0
        if anchor_gamma == 0.0:
            return base
        return float(anchor_gamma) * sum(
            (parameter.float() - anchor[name].float()).square().sum()
            for name, parameter in named
        )

    def joint_task() -> Tensor:
        losses = [
            fullspan_prior_loss(replay_temporal_head_native(head, value, device))
            for value in head_inputs
        ]
        return torch.stack(losses).mean()

    def finite_gradient_info() -> tuple[bool, float, list[str]]:
        missing: list[str] = []
        squared = 0.0
        for name, parameter in named:
            if parameter.grad is None:
                missing.append(name)
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                return False, float("nan"), missing
            squared += float(parameter.grad.detach().float().square().sum().item())
        return True, squared**0.5, missing

    def delta(reference: Mapping[str, Tensor]) -> float:
        total = 0.0
        for name, parameter in named:
            total += float((parameter.detach().float() - reference[name].float()).square().sum().item())
        return total**0.5

    try:
        before = {name: parameter.detach().clone() for name, parameter in named}
        optimizer.zero_grad(set_to_none=True)
        task_before = joint_task()
        anchor_before = anchor_loss()
        total_before = task_before + anchor_before
        if not bool(torch.isfinite(total_before).all()):
            raise ValueError("joint Fullspan loss before update is non-finite")
        total_before.backward()
        finite_gradients, gradient_norm, missing_gradients = finite_gradient_info()
        if not finite_gradients:
            raise ValueError("joint Fullspan gradients are non-finite")
        optimizer.step()
        parameter_exact = all(torch.equal(before[name], parameter.detach()) for name, parameter in named)
        if lr == 0.0 and not parameter_exact:
            raise RuntimeError("lr=0 joint Fullspan update changed parameters")
        with torch.no_grad():
            task_after = joint_task()
            anchor_after = anchor_loss()
            total_after = task_after + anchor_after
        if not bool(torch.isfinite(total_after).all()):
            raise ValueError("joint Fullspan loss after update is non-finite")
        step = {
            "step": 1,
            "record_count": len(head_inputs),
            "head_input_shapes": [list(value.shape) for value in head_inputs],
            "loss_before": float(total_before.detach().item()),
            "task_loss_before": float(task_before.detach().item()),
            "anchor_loss_before": float(anchor_before.detach().item()),
            "loss_after": float(total_after.detach().item()),
            "task_loss_after": float(task_after.detach().item()),
            "anchor_loss_after": float(anchor_after.detach().item()),
            "gradient_norm": float(gradient_norm),
            "finite_gradients": bool(finite_gradients),
            "missing_gradient_names": missing_gradients,
            "parameter_delta_l2": delta(before),
            "parameter_finite": all(bool(torch.isfinite(parameter).all()) for _, parameter in named),
            "parameter_exact_after_step": parameter_exact,
            "optimizer_step_executed": True,
        }
    finally:
        for name, parameter in named:
            parameter.requires_grad_(original_requires_grad[name])
            parameter.grad = None
        for module, training in original_modes.items():
            module.training = training

    return {
        "steps": [step],
        "audit": {
            "record_count": len(head_inputs),
            "effective_steps": 1,
            "backward_steps": 1,
            "optimizer_steps": 1,
            "skipped": False,
            "device": str(device),
            "lr": float(lr),
            "anchor_gamma": float(anchor_gamma),
            "anchor_definition": "gamma * sum((parameter - initial_parameter)^2)",
            "joint_loss": "mean(fullspan_prior_loss over native offset records) + anchor",
            "optimizer": "AdamW(weight_decay=0)",
            "optimizer_eps": float(optimizer_eps),
            "head_input_shapes": [list(value.shape) for value in head_inputs],
            "parameter_names": [name for name, _ in named],
            "parameter_delta_l2": delta(anchor),
            "all_steps_finite": bool(step["finite_gradients"] and step["parameter_finite"]),
            "all_optimizer_steps_executed": True,
            "lr_zero_parameter_exact": bool(step["parameter_exact_after_step"]) if lr == 0.0 else None,
        },
    }


def run_external_endpoint_tta(
    method: str,
    parameters: Iterable[nn.Parameter],
    closure: Callable[[int], Tensor],
    *,
    lr: float,
    steps: int = 1,
    num_views: int = 1,
    reset: bool = False,
    parameter_scope: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the shared TENT/MEMO helper with explicit TA scope metadata.

    ``reset=False`` is intentional for the runner: it evaluates the query
    under the adapted live model and restores the complete model snapshot in
    the caller.  The helper itself still restores the selected parameter flags
    and reports the required caller reset.
    """

    if method not in EXTERNAL_METHODS:
        raise ValueError(f"external endpoint method must be tent or memo, got {method!r}")
    if parameter_scope is None:
        raise ValueError("TENT/MEMO require explicit parameter_scope provenance")
    if not bool(parameter_scope.get("live_forward_required", False)):
        raise ValueError("TENT/MEMO scope must require a live forward closure")
    from vg_tta.external_tta_baselines import run_endpoint_tta

    audit = run_endpoint_tta(
        list(parameters),
        closure,
        method,
        lr=float(lr),
        steps=int(steps),
        num_views=int(num_views),
        reset=bool(reset),
        **kwargs,
    )
    audit.update(
        {
            "method": method,
            "objective": "endpoint_entropy" if method == "tent" else "memo_marginal_endpoint_entropy",
            "parameter_scope": dict(parameter_scope),
            "spatial_adaptation": False,
            "spatial_loss_used": False,
            "native_boxes_updated_by_adapted_decoder": True,
            "boxes_fixed_to_frozen": False,
            "gt_used": False,
        }
    )
    return audit


def method_contracts() -> dict[str, dict[str, Any]]:
    """Return JSON-safe fixed method contracts for a lock/report."""

    return {
        "frozen": {
            "parameter_scope": "none",
            "objective": "native_forward",
            "adaptation_unit": "one_query_evaluation",
            "episodic_reset": "full_model_before_and_after_each_query",
            "spatial_prediction": "original_native_views",
            "spatial_loss_used": False,
            "boxes_change_under_adaptation": False,
            "gt_used_for_adaptation": False,
        },
        "direct_fullspan": {
            "parameter_scope": "none",
            "objective": "fixed_first_start_last_end_logit_control",
            "adaptation_unit": "one_query_evaluation",
            "episodic_reset": "full_model_before_and_after_each_query",
            "spatial_prediction": "frozen_original_native_views",
            "spatial_loss_used": False,
            "boxes_change_under_adaptation": False,
            "gt_used_for_adaptation": False,
        },
        "ours": {
            "parameter_scope": "temp_embed",
            "objective": "fullspan_prior_loss",
            "optimizer": "AdamW(weight_decay=0)",
            "lr": 0.01,
            "steps": 1,
            "anchor_gamma": 1.0e-4,
            "adaptation_unit": "one_query",
            "episodic_reset": "full_model_before_and_after_each_query",
            "adaptation_views": "native_temporal_offset_0_and_1",
            "joint_loss": "mean(fullspan_prior_loss over the two native offsets)",
            "native_head_replay": "complete [L,1,T,D] temp_embed input; CUDA FP16 autocast, CPU FP32",
            "native_replay_parity": "exact against the live native pred_sted before adaptation",
            "spatial_prediction": "original_native_views",
            "spatial_loss_used": False,
            "boxes_change_under_adaptation": False,
            "gt_used_for_adaptation": False,
        },
        "tent": {
            "parameter_scope": "ground_decoder_layernorm_affine",
            "objective": "endpoint_entropy",
            "optimizer": "external_tta_baselines default TENT",
            "adaptation_unit": "one_query",
            "episodic_reset": "full_model_before_and_after_each_query",
            "adaptation_views": "native_offset_0",
            "spatial_prediction": "original_native_views",
            "spatial_loss_used": False,
            "boxes_change_under_adaptation": True,
            "box_provenance": "live adapted decoder output; not fixed-B0",
            "gt_used_for_adaptation": False,
            "scope_caveat": "requires live forward; invalid on detached post-head cache",
        },
        "memo": {
            "parameter_scope": "ground_decoder_layernorm_affine",
            "objective": "marginal_endpoint_entropy",
            "optimizer": "external_tta_baselines default MEMO",
            "adaptation_unit": "one_query",
            "episodic_reset": "full_model_before_and_after_each_query",
            "adaptation_views": "same-frame-id original plus fixed brightness-contrast view",
            "spatial_prediction": "original_native_views",
            "spatial_loss_used": False,
            "boxes_change_under_adaptation": True,
            "box_provenance": "live adapted decoder output; not fixed-B0",
            "gt_used_for_adaptation": False,
            "scope_caveat": "requires live forward; invalid on detached post-head cache",
        },
    }


__all__ = [
    "METHODS",
    "EXTERNAL_METHODS",
    "direct_fullspan_logits",
    "pad_view_logits",
    "replay_temporal_head_native",
    "temporal_head_scope",
    "decoder_layernorm_scope",
    "fit_ours_temporal_head",
    "run_external_endpoint_tta",
    "method_contracts",
]
