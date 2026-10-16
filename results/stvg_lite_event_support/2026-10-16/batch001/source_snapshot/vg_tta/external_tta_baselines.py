"""Small, model-agnostic TTA baselines for TubeDETR-style endpoint logits.

This module deliberately does not know about TubeDETR, datasets, boxes, or
ground truth.  A caller supplies the exact parameters that are allowed to
adapt and a closure which returns endpoint logits for a requested view.  The
endpoint convention is ``[T, 2]`` (or ``[1, T, 2]``): the two columns are the
start and end distributions over the sampled temporal positions.

The implementations are intentionally explicit rather than wrappers around a
classification TTA package:

* ``tent`` minimizes the mean entropy of the two endpoint distributions for
  one view.
* ``memo`` minimizes the entropy of the marginal distribution obtained by
  averaging endpoint probabilities across views.  It is *not* the mean of
  per-view entropies.
* ``sar`` performs a reliable two-pass SAM update.  Reliability is evaluated
  at the whole-video endpoint loss because one video is the adaptation unit;
  an unreliable video/step is skipped.  The parameter snapshot is restored
  when ``reset=True`` (the episodic default).

The helper never changes ``model.train()/eval()`` and never touches buffers.
TubeDETR callers should keep the model in ``eval`` mode while passing selected
LayerNorm affine parameters (its backbone uses FrozenBatchNorm2d).  If a
caller requests ``reset=False``, the adapted parameters remain in place and
the returned audit marks that the caller must restore them before the next
episode using :func:`restore_parameter_state`.

The code is CPU-testable and intentionally has no project-specific imports.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "capture_parameter_state",
    "endpoint_entropy",
    "endpoint_probabilities",
    "memo_marginal_entropy",
    "restore_parameter_state",
    "run_endpoint_tta",
]


_METHODS = frozenset({"tent", "memo", "sar"})
_ENDPOINTS = 2


def _normalise_logits(logits: Tensor) -> Tensor:
    """Validate and return endpoint logits with shape ``[T, 2]``.

    A leading singleton batch dimension is accepted because the native
    TubeDETR bridge commonly returns ``[1, T, 2]``.  We reject every other
    shape instead of silently flattening it: accidental frame/batch mixing is
    especially dangerous in a video TTA objective.
    """

    if not isinstance(logits, Tensor):
        raise TypeError(
            "endpoint closure must return a torch.Tensor with shape [T, 2] "
            "or [1, T, 2]"
        )
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(
                f"endpoint logits batch dimension must be 1, got {tuple(logits.shape)}"
            )
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[1] != _ENDPOINTS:
        raise ValueError(
            "endpoint logits must have shape [T, 2] or [1, T, 2], "
            f"got {tuple(logits.shape)}"
        )
    if logits.shape[0] < 1:
        raise ValueError("endpoint logits must contain at least one temporal position")
    if not (logits.is_floating_point() or logits.is_complex()):
        raise TypeError("endpoint logits must be floating point")
    # Complex logits cannot be sent through softmax/entropy meaningfully.
    if logits.is_complex():
        raise TypeError("complex endpoint logits are not supported")
    logits = logits.float()
    if not bool(torch.isfinite(logits).all().detach().cpu()):
        raise ValueError("endpoint logits contain NaN or infinity")
    return logits


def endpoint_probabilities(logits: Tensor) -> Tensor:
    """Return ``[T, 2]`` endpoint probabilities from endpoint logits."""

    return torch.softmax(_normalise_logits(logits), dim=0)


def _distribution_entropy(probabilities: Tensor) -> Tensor:
    """Mean entropy of the two endpoint distributions."""

    if probabilities.ndim != 2 or probabilities.shape[1] != _ENDPOINTS:
        raise ValueError(
            "endpoint probabilities must have shape [T, 2], "
            f"got {tuple(probabilities.shape)}"
        )
    # ``tiny`` keeps log finite even for a saturated softmax while preserving
    # the normal gradient away from zero.
    tiny = torch.finfo(probabilities.dtype).tiny
    return -(probabilities * probabilities.clamp_min(tiny).log()).sum(dim=0).mean()


def endpoint_entropy(logits: Tensor) -> Tensor:
    """Return the mean start/end entropy for one endpoint-logit tensor."""

    return _distribution_entropy(endpoint_probabilities(logits))


def memo_marginal_entropy(logits_views: Sequence[Tensor]) -> Tensor:
    """Return MEMO's entropy of the *marginal* endpoint distribution.

    For each view ``v`` we compute ``p_v = softmax(logits_v)`` over temporal
    positions, then return ``H(mean_v p_v)``.  This is deliberately different
    from ``mean_v H(p_v)``; the latter is an invariance/conditional-entropy
    objective and is not the MEMO objective.
    """

    if not logits_views:
        raise ValueError("MEMO requires at least one augmentation view")
    probabilities = torch.stack(
        [endpoint_probabilities(logits) for logits in logits_views], dim=0
    )
    return _distribution_entropy(probabilities.mean(dim=0))


def _validate_parameters(parameters: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    params = list(parameters)
    if not params:
        raise ValueError("at least one trainable parameter is required")
    seen = set()
    for index, parameter in enumerate(params):
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"parameters[{index}] is not torch.nn.Parameter")
        if id(parameter) in seen:
            raise ValueError("the same parameter was supplied more than once")
        seen.add(id(parameter))
        if not parameter.is_leaf:
            raise ValueError(f"parameters[{index}] must be a leaf Parameter")
        if not parameter.is_floating_point():
            raise TypeError(f"parameters[{index}] must be floating point")
    return params


def capture_parameter_state(parameters: Iterable[nn.Parameter]) -> Tuple[Tensor, ...]:
    """Clone the current parameter values for an episodic reset."""

    return tuple(parameter.detach().clone() for parameter in _validate_parameters(parameters))


def restore_parameter_state(
    parameters: Iterable[nn.Parameter], state: Sequence[Tensor]
) -> None:
    """Restore a state returned by :func:`capture_parameter_state`."""

    params = _validate_parameters(parameters)
    if len(params) != len(state):
        raise ValueError(
            f"state has {len(state)} tensors for {len(params)} parameters"
        )
    with torch.no_grad():
        for index, (parameter, value) in enumerate(zip(params, state)):
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(
                    f"state shape mismatch at parameter {index}: "
                    f"{tuple(value.shape)} vs {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _parameter_delta(parameters: Sequence[nn.Parameter], state: Sequence[Tensor]) -> float:
    total = 0.0
    for parameter, before in zip(parameters, state):
        delta = parameter.detach().float() - before.to(
            device=parameter.device, dtype=parameter.dtype
        ).float()
        total += float(delta.pow(2).sum().detach().cpu())
    return math.sqrt(max(total, 0.0))


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            total += float(grad.pow(2).sum().detach().cpu())
    return math.sqrt(max(total, 0.0))


def _finite_scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar tensor")
    if not bool(torch.isfinite(value).detach().cpu()):
        raise FloatingPointError(f"{name} is NaN or infinity")
    return float(value.detach().cpu())


def _closure_loss(
    closure: Callable[[int], Tensor],
    method: str,
    num_views: int,
    *,
    grad_enabled: bool,
) -> Tuple[Tensor, int, List[Tensor]]:
    """Call a closure and calculate the method-specific loss."""

    # ``torch.enable_grad`` is needed for the audit evaluation path after a
    # caller has entered no_grad; ``torch.no_grad`` makes it explicit that the
    # post-update diagnostic cannot accidentally create a graph.
    context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with context:
        if method in {"tent", "sar"}:
            logits = closure(0)
            loss = endpoint_entropy(logits)
            outputs = [logits]
            calls = 1
        else:
            outputs = [closure(view_index) for view_index in range(num_views)]
            loss = memo_marginal_entropy(outputs)
            calls = num_views
    return loss, calls, outputs


def _reliability_margin(logits: Tensor, fraction: float) -> float:
    positions = _normalise_logits(logits).shape[0]
    return float(fraction) * math.log(max(positions, 2))


def _snapshot_requires_grad(parameters: Sequence[nn.Parameter]) -> Tuple[bool, ...]:
    return tuple(parameter.requires_grad for parameter in parameters)


def _restore_requires_grad(
    parameters: Sequence[nn.Parameter], state: Sequence[bool]
) -> None:
    for parameter, requires_grad in zip(parameters, state):
        parameter.requires_grad_(requires_grad)


def _validate_common(
    method: str,
    lr: float,
    steps: int,
    num_views: int,
    rho: float,
    entropy_margin_fraction: float,
) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    if not math.isfinite(float(lr)) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if isinstance(num_views, bool) or int(num_views) != num_views or num_views < 1:
        raise ValueError("num_views must be a positive integer")
    if not math.isfinite(float(rho)) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    if not math.isfinite(float(entropy_margin_fraction)) or entropy_margin_fraction < 0:
        raise ValueError("entropy_margin_fraction must be finite and non-negative")
    if method in {"tent", "sar"} and num_views != 1:
        raise ValueError(
            f"{method.upper()} is a single-view baseline; use MEMO for num_views>1"
        )


def run_endpoint_tta(
    parameters: Iterable[nn.Parameter],
    closure: Callable[[int], Tensor],
    method: str,
    lr: float,
    steps: int = 1,
    num_views: int = 1,
    rho: float = 0.05,
    entropy_margin_fraction: float = 0.4,
    *,
    reset: bool = True,
    momentum: float | None = None,
    reset_constant_em: float = 0.2,
    optimizer_name: str | None = None,
) -> dict[str, Any]:
    """Run one endpoint-distribution adaptation episode.

    Parameters
    ----------
    parameters:
        The exact parameter objects allowed to update.  No parameter discovery
        is performed.  For TubeDETR this should normally be decoder LayerNorm
        affine weights/biases selected by the caller.
    closure:
        Callable ``closure(view_index) -> Tensor`` returning ``[T, 2]`` or
        ``[1, T, 2]`` endpoint logits.  It is called with gradients during
        updates.  The helper does not switch the owning model between train and
        eval mode.
    method:
        ``"tent"``, ``"memo"``, or ``"sar"``.
    lr, steps:
        SGD learning rate and number of adaptation steps.  ``steps=0`` is a
        valid explicit no-op useful for controls.
    num_views:
        Number of views for MEMO.  TENT and SAR reject values other than one
        to prevent accidentally changing their baseline definition.
    rho:
        SAM perturbation radius for SAR.
    entropy_margin_fraction:
        SAR reliability margin is ``fraction * log(max(T, 2))`` for a video
        with ``T`` positions.  A step is updated only when both first and
        second pass losses are below this margin.
    reset:
        Restore the exact input parameter values before return (default).  If
        false, leave the adapted parameters in place and mark the audit as
        requiring caller reset.
    momentum:
        SGD momentum.  ``None`` selects 0 for MEMO and 0.9 for SAR.  It is
        ignored for the default Adam TENT optimizer.
    reset_constant_em:
        SAR's model-recovery threshold for the exponential moving average of
        the second-pass endpoint entropy.  The official SAR value is ``0.2``;
        when the EMA falls below it, the episode parameter and optimizer state
        are restored and the EMA is restarted.  Set it to ``0`` to disable
        recovery without changing the reliability gate.
    optimizer_name:
        Optional optimizer override.  ``None``/``"auto"`` uses Adam for TENT
        (the official TENT example configuration uses Adam, lr 1e-3), SGD for
        MEMO (the official MEMO CIFAR example uses SGD), and SAM with an SGD
        base optimizer for SAR.  ``"sgd"`` or ``"adam"`` may be selected for
        TENT/MEMO; SAR intentionally rejects Adam because SAM's two-pass
        contract is implemented with its SGD base optimizer.  The supplied
        ``lr`` always wins over source-repository example values.

    Returns
    -------
    dict
        JSON-serialisable scalar/list audit.  It reports method, loss and
        gradient traces, closure-call counts, reliability decisions, parameter
        deltas, reset exactness, and ``gt_used=False``.  It contains no model
        outputs or labels.
    """

    params = _validate_parameters(parameters)
    _validate_common(
        method, lr, steps, num_views, rho, entropy_margin_fraction
    )
    if not callable(closure):
        raise TypeError("closure must be callable as closure(view_index)")
    if not math.isfinite(float(reset_constant_em)) or reset_constant_em < 0:
        raise ValueError("reset_constant_em must be finite and non-negative")
    requested_optimizer = "auto" if optimizer_name is None else str(optimizer_name).lower()
    if requested_optimizer not in {"auto", "sgd", "adam", "adamw"}:
        raise ValueError("optimizer_name must be one of auto, sgd, adam, adamw")
    if requested_optimizer == "auto":
        effective_optimizer = "adam" if method == "tent" else "sgd"
    else:
        effective_optimizer = requested_optimizer
    if method == "sar" and effective_optimizer != "sgd":
        raise ValueError("SAR requires the implemented SAM(SGD) optimizer")
    if momentum is None:
        effective_momentum = 0.9 if method == "sar" else 0.0
    else:
        effective_momentum = float(momentum)
    if not math.isfinite(effective_momentum) or not 0 <= effective_momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")

    initial_state = capture_parameter_state(params)
    original_requires_grad = _snapshot_requires_grad(params)
    for parameter in params:
        parameter.requires_grad_(True)

    audit: dict[str, Any] = {
        "method": method,
        "lr": float(lr),
        "steps_requested": int(steps),
        "num_views": int(num_views),
        "rho": float(rho),
        "entropy_margin_fraction": float(entropy_margin_fraction),
        "reliability_margin_definition": "fraction * log(max(T, 2))",
        "optimizer": "SAM(SGD)" if method == "sar" else effective_optimizer.upper(),
        "optimizer_selection": requested_optimizer,
        "momentum": float(effective_momentum),
        "reset_constant_em": float(reset_constant_em),
        "sar_recovery_definition": (
            "ema_t=0.9*ema_(t-1)+0.1*second_pass_entropy; "
            "restore episode parameter+optimizer state when ema_t < reset_constant_em"
            if method == "sar"
            else None
        ),
        "episodic": bool(reset),
        "gt_used": False,
        "closure_calls": 0,
        "optimizer_steps": 0,
        "skipped_steps": 0,
        "reliable_steps": 0,
        "loss_before": [],
        "loss_after": [],
        "gradient_norms": [],
        "step_records": [],
        "ema_trace": [],
        "last_ema": None,
        "recovery_count": 0,
        "recovery_triggered": False,
        "initial_parameter_delta_l2": 0.0,
    }

    optimizer: torch.optim.Optimizer | None = None
    if int(steps) > 0:
        if method == "sar" or effective_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=float(lr),
                momentum=float(effective_momentum),
                weight_decay=0.0,
            )
        elif effective_optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=float(lr),
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
        else:  # AdamW is accepted as an explicit non-default control.
            optimizer = torch.optim.AdamW(
                params,
                lr=float(lr),
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )

    optimizer_initial_state = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    sar_ema: float | None = None

    def recover_sar_episode() -> None:
        """Mirror SAR's low-entropy model recovery for this episode."""

        nonlocal sar_ema
        restore_parameter_state(params, initial_state)
        if optimizer is not None and optimizer_initial_state is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_initial_state))
            optimizer.zero_grad(set_to_none=True)
        sar_ema = None
        audit["recovery_count"] += 1
        audit["recovery_triggered"] = True

    try:
        if method in {"tent", "memo"}:
            for step_index in range(int(steps)):
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss, calls, _ = _closure_loss(
                    closure,
                    method,
                    int(num_views),
                    grad_enabled=True,
                )
                audit["closure_calls"] += calls
                before = _finite_scalar(loss, f"{method} loss at step {step_index}")
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"{method} closure loss has no gradient; check selected parameters"
                    )
                loss.backward()
                grad_norm = _gradient_norm(params)
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"{method} gradient norm is NaN or infinity at step {step_index}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    after_loss, after_calls, _ = _closure_loss(
                        closure,
                        method,
                        int(num_views),
                        grad_enabled=False,
                    )
                audit["closure_calls"] += after_calls
                after = _finite_scalar(
                    after_loss, f"{method} post-update loss at step {step_index}"
                )
                audit["loss_before"].append(before)
                audit["loss_after"].append(after)
                audit["gradient_norms"].append(float(grad_norm))
                audit["optimizer_steps"] += 1
                audit["reliable_steps"] += 1
                audit["step_records"].append(
                    {
                        "step": step_index,
                        "updated": True,
                        "loss_before": before,
                        "loss_after": after,
                        "gradient_norm": float(grad_norm),
                    }
                )

        else:  # SAR: reliable two-pass SAM
            for step_index in range(int(steps)):
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                first_loss, first_calls, first_outputs = _closure_loss(
                    closure,
                    "sar",
                    1,
                    grad_enabled=True,
                )
                audit["closure_calls"] += first_calls
                first_value = _finite_scalar(
                    first_loss, f"SAR first-pass loss at step {step_index}"
                )
                first_margin = _reliability_margin(
                    first_outputs[0], float(entropy_margin_fraction)
                )
                first_reliable = first_value < first_margin
                record: dict[str, Any] = {
                    "step": step_index,
                    "first_loss": first_value,
                    "first_margin": float(first_margin),
                    "first_reliable": bool(first_reliable),
                    "second_loss": None,
                    "second_margin": None,
                    "second_reliable": None,
                    "updated": False,
                    "gradient_norm": 0.0,
                }
                if not first_reliable:
                    optimizer.zero_grad(set_to_none=True)
                    audit["skipped_steps"] += 1
                    audit["loss_before"].append(first_value)
                    audit["loss_after"].append(first_value)
                    audit["gradient_norms"].append(0.0)
                    audit["step_records"].append(record)
                    continue

                if not first_loss.requires_grad:
                    raise RuntimeError(
                        "SAR closure loss has no gradient; check selected parameters"
                    )
                first_loss.backward()
                grad_norm = _gradient_norm(params)
                record["gradient_norm"] = float(grad_norm)
                if not math.isfinite(grad_norm) or grad_norm == 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    audit["skipped_steps"] += 1
                    audit["loss_before"].append(first_value)
                    audit["loss_after"].append(first_value)
                    audit["gradient_norms"].append(float(grad_norm))
                    record["skip_reason"] = "nonfinite_or_zero_gradient"
                    audit["step_records"].append(record)
                    continue

                # Save the unperturbed parameters.  SAM's second gradient is
                # evaluated at the perturbed point, then applied at the saved
                # point by restoring before the base optimizer step.
                unperturbed = tuple(parameter.detach().clone() for parameter in params)

                def restore_unperturbed() -> None:
                    with torch.no_grad():
                        for parameter, value in zip(params, unperturbed):
                            parameter.copy_(value)

                scale = float(rho) / (grad_norm + 1e-12)
                with torch.no_grad():
                    for parameter in params:
                        if parameter.grad is not None:
                            parameter.add_(parameter.grad, alpha=scale)
                optimizer.zero_grad(set_to_none=True)
                try:
                    second_loss, second_calls, second_outputs = _closure_loss(
                        closure,
                        "sar",
                        1,
                        grad_enabled=True,
                    )
                    audit["closure_calls"] += second_calls
                    second_value = _finite_scalar(
                        second_loss, f"SAR second-pass loss at step {step_index}"
                    )
                    second_margin = _reliability_margin(
                        second_outputs[0], float(entropy_margin_fraction)
                    )
                except Exception:
                    # Even with reset=False, a failed second pass must not
                    # leak SAM's temporary perturbation to the caller.
                    restore_unperturbed()
                    optimizer.zero_grad(set_to_none=True)
                    raise
                second_reliable = second_value < second_margin
                if sar_ema is None:
                    sar_ema = second_value
                else:
                    sar_ema = 0.9 * sar_ema + 0.1 * second_value
                audit["ema_trace"].append(float(sar_ema))
                audit["last_ema"] = float(sar_ema)
                record["second_loss"] = second_value
                record["second_margin"] = float(second_margin)
                record["second_reliable"] = bool(second_reliable)
                if not second_reliable:
                    restore_unperturbed()
                    optimizer.zero_grad(set_to_none=True)
                    audit["skipped_steps"] += 1
                    audit["loss_before"].append(first_value)
                    audit["loss_after"].append(second_value)
                    audit["gradient_norms"].append(float(grad_norm))
                    record["skip_reason"] = "second_pass_unreliable"
                    if sar_ema < float(reset_constant_em):
                        recover_sar_episode()
                        record["recovery_triggered"] = True
                    audit["step_records"].append(record)
                    continue

                if not second_loss.requires_grad:
                    restore_unperturbed()
                    raise RuntimeError(
                        "SAR second-pass loss has no gradient; check selected parameters"
                    )
                try:
                    second_loss.backward()
                except Exception:
                    restore_unperturbed()
                    optimizer.zero_grad(set_to_none=True)
                    raise
                second_grad_norm = _gradient_norm(params)
                if not math.isfinite(second_grad_norm):
                    restore_unperturbed()
                    raise FloatingPointError(
                        f"SAR second-pass gradient norm is NaN or infinity at step {step_index}"
                    )
                restore_unperturbed()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                audit["optimizer_steps"] += 1
                audit["reliable_steps"] += 1
                audit["loss_before"].append(first_value)
                audit["loss_after"].append(second_value)
                audit["gradient_norms"].append(float(second_grad_norm))
                record["updated"] = True
                record["gradient_norm_second_pass"] = float(second_grad_norm)
                if sar_ema < float(reset_constant_em):
                    recover_sar_episode()
                    record["recovery_triggered"] = True
                audit["step_records"].append(record)

        audit["adapted_parameter_delta_l2"] = _parameter_delta(params, initial_state)
        if reset:
            restore_parameter_state(params, initial_state)
        audit["final_parameter_delta_l2"] = _parameter_delta(params, initial_state)
        audit["reset_exact"] = bool(audit["final_parameter_delta_l2"] == 0.0)
        audit["reset_required"] = not bool(reset)
        audit["status"] = "ok"
        if int(steps) == 0:
            audit["status"] = "no_op_steps_zero"
        elif float(lr) == 0.0:
            audit["status"] = "no_op_lr_zero"
        elif audit["optimizer_steps"] == 0:
            audit["status"] = "no_op_no_reliable_update"
        elif audit["recovery_triggered"]:
            audit["status"] = "ok_with_recovery"
        return audit
    finally:
        # A failed adaptation must not leave an episodic caller with perturbed
        # weights.  For reset=False, the caller explicitly owns recovery.
        if reset:
            restore_parameter_state(params, initial_state)
        _restore_requires_grad(params, original_requires_grad)
