"""Episodic test-time adaptation primitives for TubeDETR.

The module deliberately keeps adaptation small and auditable: only explicitly
selected parameter scopes can receive gradients, every episode restores the
pre-adaptation values, and both label-free consistency and GT oracle losses
share the same reporting interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .metrics import generalized_box_iou_aligned_cxcywh, interval_from_logits


UpdateScope = Literal[
    "temporal",
    "temporal_ln",
    "temporal_final_bias",
    "spatial",
    "both",
    "heads_ln_projection",
]
OptimizerName = Literal["adamw", "sgd"]
UPDATE_SCOPES: tuple[UpdateScope, ...] = (
    "temporal",
    "temporal_ln",
    "temporal_final_bias",
    "spatial",
    "both",
    "heads_ln_projection",
)


@dataclass(frozen=True)
class ConsistencyLossConfig:
    temporal_weight: float = 1.0
    box_l1_weight: float = 5.0
    box_giou_weight: float = 2.0
    epsilon: float = 1e-7


@dataclass(frozen=True)
class GroundTruthLossConfig:
    """TubeDETR's final-layer supervised loss weights."""

    temporal_weight: float = 10.0
    box_l1_weight: float = 5.0
    box_giou_weight: float = 2.0
    sigma: float = 1.0
    epsilon: float = 1e-6


def _scope_parameter_names(model: torch.nn.Module, scope: UpdateScope) -> set[str]:
    if scope not in UPDATE_SCOPES:
        raise ValueError(f"unsupported update scope {scope!r}; choose from {UPDATE_SCOPES}")

    names: set[str] = set()
    for name, _ in model.named_parameters():
        if scope in ("temporal", "temporal_ln") and name.startswith("sted_embed."):
            names.add(name)
        elif scope == "temporal_final_bias" and name == "sted_embed.layers.1.bias":
            names.add(name)
        elif scope == "spatial" and name.startswith("bbox_embed."):
            names.add(name)
        elif scope in ("both", "heads_ln_projection") and name.startswith(
            ("bbox_embed.", "sted_embed.")
        ):
            names.add(name)
        elif scope == "heads_ln_projection" and name.startswith("input_proj."):
            names.add(name)

    if scope in ("temporal_ln", "heads_ln_projection"):
        # Decoder LayerNorm affine parameters are the smallest normalization
        # scope downstream of the encoded visual features. input_proj is the
        # model's learned 1x1 visual projection.
        for module_name, module in model.named_modules():
            if not module_name.startswith("transformer.decoder"):
                continue
            if not isinstance(module, torch.nn.LayerNorm):
                continue
            for local_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{local_name}")
    return names


def configure_update_scope(
    model: torch.nn.Module,
    scope: UpdateScope = "both",
) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Freeze all parameters except the explicitly requested TTA scope."""

    selected_names = _scope_parameter_names(model, scope)
    selected: list[torch.nn.Parameter] = []
    realized_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = name in selected_names
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
            realized_names.append(name)
    if not selected:
        raise RuntimeError(f"no parameters found for update scope {scope!r}")
    return selected, realized_names


def configure_head_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Backward-compatible alias for the original two-head scope."""

    parameters, _ = configure_update_scope(model, "both")
    return parameters


def _temporal_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    time_mask: torch.Tensor | None,
) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[-1] != 2:
        raise ValueError("student and teacher temporal logits must both be BxTx2")
    # Keep the expensive model forward in BF16 but compute probability losses
    # in FP32. Direct BF16 KL can become slightly negative through rounding.
    student = student.float()
    teacher = teacher.float()
    if time_mask is not None:
        if time_mask.shape != student.shape[:2]:
            raise ValueError("time_mask must be BxT")
        student = student.masked_fill(~time_mask[..., None], -1e4)
        teacher = teacher.masked_fill(~time_mask[..., None], -1e4)
    loss = student.new_zeros(())
    for endpoint in range(2):
        teacher_probability = teacher[..., endpoint].softmax(dim=1)
        student_log_probability = student[..., endpoint].log_softmax(dim=1)
        loss = loss + F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="batchmean",
        )
    return loss


def teacher_interval_mask(
    teacher_sted: torch.Tensor,
    *,
    time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard BxT mask for the teacher's decoded target-present interval."""

    if teacher_sted.ndim != 3:
        raise ValueError("teacher temporal logits must be BxTx2")
    batch, duration, _ = teacher_sted.shape
    masks = torch.zeros((batch, duration), dtype=torch.bool, device=teacher_sted.device)
    for batch_index in range(batch):
        start, end = interval_from_logits(teacher_sted[batch_index])
        masks[batch_index, start : end + 1] = True
    if time_mask is not None:
        if time_mask.shape != masks.shape:
            raise ValueError("time_mask must be BxT")
        masks &= time_mask
    return masks


def _flatten_box_mask(box_mask: torch.Tensor, box_count: int) -> torch.Tensor:
    if box_mask.ndim not in (1, 2):
        raise ValueError("box_mask must be T or BxT")
    flat = box_mask.reshape(-1).bool()
    if flat.numel() != box_count:
        raise ValueError("box_mask does not align with flattened box predictions")
    if not flat.any():
        raise ValueError("box_mask selects no frames")
    return flat


def consistency_loss(
    student_outputs: dict[str, torch.Tensor],
    teacher_outputs: dict[str, torch.Tensor],
    config: ConsistencyLossConfig,
    *,
    time_mask: torch.Tensor | None = None,
    box_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_boxes = teacher_outputs["pred_boxes"].detach().float()
    teacher_sted = teacher_outputs["pred_sted"].detach()
    student_boxes = student_outputs["pred_boxes"].float()
    student_sted = student_outputs["pred_sted"]
    if student_boxes.shape != teacher_boxes.shape:
        raise ValueError("student and teacher boxes must be aligned")

    temporal = _temporal_kl(student_sted, teacher_sted, time_mask)
    if box_mask is None:
        selected_student = student_boxes
        selected_teacher = teacher_boxes
    else:
        flat_mask = _flatten_box_mask(box_mask, len(student_boxes))
        selected_student = student_boxes[flat_mask]
        selected_teacher = teacher_boxes[flat_mask]

    # Match TubeDETR's box-loss normalization: sum the four L1 coordinates for
    # each selected frame, then average across selected frames.
    box_l1 = F.l1_loss(selected_student, selected_teacher, reduction="none").sum(-1).mean()
    box_giou = (
        1.0 - generalized_box_iou_aligned_cxcywh(selected_student, selected_teacher)
    ).mean()
    total = (
        config.temporal_weight * temporal
        + config.box_l1_weight * box_l1
        + config.box_giou_weight * box_giou
    )
    components = {
        "loss_total": float(total.detach().item()),
        "loss_temporal": float(temporal.detach().item()),
        "loss_box_l1": float(box_l1.detach().item()),
        "loss_box_giou": float(box_giou.detach().item()),
        "box_frames": float(len(selected_student)),
    }
    return total, components


def ground_truth_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict],
    gt_indices: tuple[int, int],
    config: GroundTruthLossConfig,
    *,
    time_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Final-layer TubeDETR supervised loss used only as an adaptation oracle."""

    pred_boxes = outputs["pred_boxes"].float()
    sted = outputs["pred_sted"].float()
    if sted.ndim != 3 or sted.shape[0] != 1 or sted.shape[-1] != 2:
        raise ValueError("GT oracle currently expects temporal logits shaped 1xTx2")
    if len(targets) != len(pred_boxes):
        raise ValueError("targets and predictions must be aligned frame-by-frame")

    gt_positions = [index for index, target in enumerate(targets) if len(target["boxes"])]
    if not gt_positions:
        raise ValueError("GT oracle received an empty ground-truth tube")
    expected_positions = list(range(gt_indices[0], gt_indices[1] + 1))
    if gt_positions != expected_positions:
        raise ValueError("ground-truth boxes are not contiguous or do not match gt_indices")
    target_boxes = torch.cat(
        [targets[index]["boxes"][:1].to(pred_boxes.device).float() for index in gt_positions],
        dim=0,
    )
    selected_pred_boxes = pred_boxes[gt_positions]
    box_l1 = F.l1_loss(selected_pred_boxes, target_boxes, reduction="none").sum(-1).mean()
    box_giou = (
        1.0 - generalized_box_iou_aligned_cxcywh(selected_pred_boxes, target_boxes)
    ).mean()

    if time_mask is None:
        time_mask = torch.ones(sted.shape[:2], dtype=torch.bool, device=sted.device)
    if time_mask.shape != sted.shape[:2]:
        raise ValueError("time_mask must be BxT")
    masked_sted = sted.masked_fill(~time_mask[..., None], -1e32)
    positions = torch.arange(sted.shape[1], device=sted.device, dtype=sted.dtype)[None, :]
    temporal = sted.new_zeros(())
    for endpoint_index, target_index in enumerate(gt_indices):
        target_distribution = torch.exp(
            -((positions - float(target_index)) ** 2) / (2.0 * config.sigma**2)
        )
        target_distribution = F.normalize(
            target_distribution + config.epsilon,
            p=1,
            dim=1,
        )
        pred_probability = masked_sted[..., endpoint_index].softmax(dim=1)
        endpoint_kl = pred_probability * (
            (pred_probability + config.epsilon) / target_distribution
        ).log()
        temporal = temporal + (endpoint_kl * time_mask).mean()

    total = (
        config.temporal_weight * temporal
        + config.box_l1_weight * box_l1
        + config.box_giou_weight * box_giou
    )
    components = {
        "loss_total": float(total.detach().item()),
        "loss_temporal": float(temporal.detach().item()),
        "loss_box_l1": float(box_l1.detach().item()),
        "loss_box_giou": float(box_giou.detach().item()),
        "box_frames": float(len(selected_pred_boxes)),
    }
    return total, components


def temporal_entropy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    time_mask: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalized entropy of TubeDETR start/end distributions.

    The normalization by log(valid frame count) keeps the objective comparable
    across videos with different durations. Only the temporal distributions are
    used because TubeDETR exposes no calibrated foreground/object confidence.
    """

    logits = outputs["pred_sted"].float()
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("temporal entropy expects pred_sted shaped BxTx2")
    if time_mask is None:
        time_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if time_mask.shape != logits.shape[:2]:
        raise ValueError("time_mask must be BxT")
    if not time_mask.any(dim=1).all():
        raise ValueError("every batch element must contain at least one valid frame")
    masked_logits = logits.masked_fill(~time_mask[..., None], -1e4)
    probabilities = masked_logits.softmax(dim=1)
    log_probabilities = (probabilities + epsilon).log()
    entropy_per_endpoint = -(probabilities * log_probabilities).sum(dim=1)
    valid_counts = time_mask.sum(dim=1).to(logits.dtype).clamp(min=2)
    normalized = entropy_per_endpoint / valid_counts.log()[:, None]
    loss = normalized.mean()
    return loss, {
        "loss_total": float(loss.detach().item()),
        "loss_temporal_entropy": float(loss.detach().item()),
        "mean_valid_frames": float(time_mask.sum(dim=1).float().mean().item()),
    }


def temporal_span_entropy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    time_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalized entropy over legal ``start < end`` temporal spans.

    Unlike independent endpoint entropy, this objective matches TubeDETR's
    constrained interval decoder.  Every batch element is normalized by the
    logarithm of its own number of legal spans, so variable-duration support
    clips remain comparable.  A two-frame clip has exactly one legal span and
    therefore a defined normalized entropy of zero.
    """

    logits = outputs["pred_sted"].float()
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("temporal span entropy expects pred_sted shaped BxTx2")
    if time_mask is None:
        time_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if time_mask.shape != logits.shape[:2]:
        raise ValueError("time_mask must be BxT")
    valid_counts = time_mask.sum(dim=1)
    if (valid_counts < 2).any():
        raise ValueError("every batch element needs at least two valid frames")

    entropies: list[torch.Tensor] = []
    raw_entropies: list[torch.Tensor] = []
    candidate_counts: list[int] = []
    for batch_index in range(logits.shape[0]):
        positions = torch.nonzero(time_mask[batch_index], as_tuple=False).flatten()
        selected = logits[batch_index].index_select(0, positions)
        start_probability = selected[:, 0].softmax(dim=0)
        end_probability = selected[:, 1].softmax(dim=0)
        joint = start_probability[:, None] * end_probability[None, :]
        legal = torch.triu(torch.ones_like(joint, dtype=torch.bool), diagonal=1)
        legal_mass = joint.masked_select(legal)
        legal_mass = legal_mass / legal_mass.sum().clamp(min=epsilon)
        raw_entropy = -(
            legal_mass * legal_mass.clamp(min=epsilon).log()
        ).sum()
        candidate_count = int(legal.sum().item())
        candidate_counts.append(candidate_count)
        raw_entropies.append(raw_entropy)
        if candidate_count == 1:
            entropies.append(raw_entropy * 0.0)
        else:
            denominator = logits.new_tensor(float(candidate_count)).log()
            entropies.append(raw_entropy / denominator)
    normalized = torch.stack(entropies)
    raw = torch.stack(raw_entropies)
    loss = normalized.mean()
    return loss, {
        "loss_total": float(loss.detach().item()),
        "loss_temporal_span_entropy": float(loss.detach().item()),
        "raw_temporal_span_entropy": float(raw.mean().detach().item()),
        "mean_valid_frames": float(valid_counts.float().mean().item()),
        "mean_legal_span_count": float(sum(candidate_counts) / len(candidate_counts)),
    }


def temporal_target_kl_loss(
    outputs: dict[str, torch.Tensor],
    target_probability: torch.Tensor,
    *,
    time_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL from a detached soft start/end target to model predictions.

    ``target_probability`` is shaped ``BxTx2`` and each endpoint distribution
    is normalized over time.  This objective deliberately touches no box
    prediction, making it suitable for spatial-safe temporal-only adaptation.
    """

    logits = outputs["pred_sted"].float()
    target = target_probability.detach().float().to(logits.device)
    if logits.shape != target.shape or logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("temporal target and pred_sted must both be BxTx2")
    if time_mask is None:
        time_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if time_mask.shape != logits.shape[:2]:
        raise ValueError("time_mask must be BxT")
    masked_logits = logits.masked_fill(~time_mask[..., None], -1e4)
    target = target.masked_fill(~time_mask[..., None], 0.0)
    target_mass = target.sum(dim=1, keepdim=True)
    if not (target_mass > 0).all():
        raise ValueError("each temporal target endpoint must have positive probability mass")
    target = target / target_mass.clamp(min=epsilon)
    student_log_probability = masked_logits.log_softmax(dim=1)
    per_element = target * (
        (target + epsilon).log() - student_log_probability
    )
    loss = per_element.sum(dim=1).mean()
    return loss, {
        "loss_total": float(loss.detach().item()),
        "loss_temporal_target_kl": float(loss.detach().item()),
        "target_entropy": float(
            (-(target * (target + epsilon).log()).sum(dim=1).mean()).detach().item()
        ),
    }


class EpisodicHeadAdapter:
    """One-video adapter with a state-safe episodic boundary.

    The adapter can only update the explicitly selected parameter scope, but an
    upstream model may still mutate buffers (for example BatchNorm running
    statistics) or module training flags during a forward.  We therefore
    snapshot selected parameters, every module buffer, and all module training
    modes.  ``reset`` restores those values, clears gradients, drops optimizer
    state, and re-applies the declared update scope.  Frozen parameters are
    deliberately not duplicated: ``configure_update_scope`` guarantees that
    this adapter cannot modify them.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        scope: UpdateScope = "both",
        loss_config: ConsistencyLossConfig | None = None,
        gt_loss_config: GroundTruthLossConfig | None = None,
        minimum_loss_for_update: float | None = 1e-6,
        optimizer_name: OptimizerName = "adamw",
        optimizer_eps: float = 1e-8,
        momentum: float = 0.0,
    ) -> None:
        self.model = model
        self.scope = scope
        self.parameters, self.parameter_names = configure_update_scope(model, scope)
        if optimizer_name not in ("adamw", "sgd"):
            raise ValueError("optimizer_name must be 'adamw' or 'sgd'")
        if optimizer_eps <= 0:
            raise ValueError("optimizer_eps must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.optimizer_name = optimizer_name
        self.optimizer_eps = optimizer_eps
        self.momentum = momentum
        self.optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
        self.loss_config = loss_config or ConsistencyLossConfig()
        self.gt_loss_config = gt_loss_config or GroundTruthLossConfig()
        self.minimum_loss_for_update = minimum_loss_for_update
        named_parameters = dict(model.named_parameters())
        self._initial_state = {
            name: named_parameters[name].detach().clone() for name in self.parameter_names
        }
        self._initial_buffer_state = {
            name: value.detach().clone()
            for name, value in model.named_buffers()
        }
        self._initial_training_modes = {
            module: bool(module.training) for module in model.modules()
        }
        self._initial_parameter_norm_l2 = sum(
            float(value.float().square().sum().item()) for value in self._initial_state.values()
        ) ** 0.5
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters)

    @torch.no_grad()
    def reset(self) -> None:
        named_parameters = dict(self.model.named_parameters())
        for name, value in self._initial_state.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        named_buffers = dict(self.model.named_buffers())
        for name, value in self._initial_buffer_state.items():
            buffer = named_buffers.get(name)
            if buffer is None:
                raise RuntimeError(f"model buffer disappeared across episodic reset: {name}")
            buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
        for module, training in self._initial_training_modes.items():
            module.train(training)
        # Gradients are optimizer state from the previous episode even when
        # they do not affect the next forward.  Clear all of them, including
        # frozen parameters, before the next support/query episode.
        for parameter in self.model.parameters():
            parameter.grad = None
        # Keep the adapter usable after reset while preserving the model's
        # original train/eval modes above.
        selected_names = set(self.parameter_names)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name in selected_names)
        # No optimizer moments may cross an episodic boundary.
        self._optimizer = None

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            if self.optimizer_name == "adamw":
                self._optimizer = torch.optim.AdamW(
                    self.parameters,
                    eps=self.optimizer_eps,
                    **self.optimizer_kwargs,
                )
            else:
                self._optimizer = torch.optim.SGD(
                    self.parameters,
                    momentum=self.momentum,
                    **self.optimizer_kwargs,
                )
        return self._optimizer

    @torch.no_grad()
    def delta_from_initial(self) -> dict[str, float]:
        named_parameters = dict(self.model.named_parameters())
        squared_l2 = 0.0
        max_abs = 0.0
        for name, initial in self._initial_state.items():
            delta = named_parameters[name].detach() - initial
            squared_l2 += float(delta.float().square().sum().item())
            max_abs = max(max_abs, float(delta.float().abs().max().item()))
        delta_l2 = squared_l2**0.5
        relative = (
            delta_l2 / self._initial_parameter_norm_l2
            if self._initial_parameter_norm_l2 > 0
            else float("nan")
        )
        return {
            "parameter_delta_l2": delta_l2,
            "parameter_delta_max_abs": max_abs,
            "parameter_norm_l2": self._initial_parameter_norm_l2,
            "relative_parameter_delta_l2": relative,
        }

    @torch.no_grad()
    def _gradient_diagnostics(self) -> dict[str, float]:
        squared_l2 = 0.0
        max_abs = 0.0
        populated = 0
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            squared_l2 += float(gradient.square().sum().item())
            max_abs = max(max_abs, float(gradient.abs().max().item()))
            populated += gradient.numel()
        return {
            "gradient_norm_l2": squared_l2**0.5,
            "gradient_max_abs": max_abs,
            "gradient_element_count": float(populated),
        }

    def _apply_loss(self, loss: torch.Tensor, components: dict[str, float]) -> dict[str, float]:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite TTA loss: {components}")
        if (
            self.minimum_loss_for_update is not None
            and abs(float(loss.detach().item())) <= self.minimum_loss_for_update
        ):
            return {
                **components,
                "update_skipped_below_numerical_floor": 1.0,
                "gradient_norm_l2": 0.0,
                "gradient_max_abs": 0.0,
                "gradient_element_count": 0.0,
                **self.delta_from_initial(),
            }
        optimizer = self._get_optimizer()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_diagnostics = self._gradient_diagnostics()
        optimizer.step()
        return {
            **components,
            "update_skipped_below_numerical_floor": 0.0,
            **gradient_diagnostics,
            **self.delta_from_initial(),
        }

    def step(
        self,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        *,
        time_mask: torch.Tensor | None = None,
        box_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Backward-compatible consistency update."""

        loss, components = consistency_loss(
            student_outputs,
            teacher_outputs,
            self.loss_config,
            time_mask=time_mask,
            box_mask=box_mask,
        )
        return self._apply_loss(loss, components)

    def step_ground_truth(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Sequence[dict],
        gt_indices: tuple[int, int],
        *,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        loss, components = ground_truth_loss(
            outputs,
            targets,
            gt_indices,
            self.gt_loss_config,
            time_mask=time_mask,
        )
        return self._apply_loss(loss, components)

    def step_temporal_entropy(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        loss, components = temporal_entropy_loss(outputs, time_mask=time_mask)
        return self._apply_loss(loss, components)

    def step_temporal_span_entropy(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        loss, components = temporal_span_entropy_loss(outputs, time_mask=time_mask)
        return self._apply_loss(loss, components)

    def step_temporal_target(
        self,
        outputs: dict[str, torch.Tensor],
        target_probability: torch.Tensor,
        *,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        loss, components = temporal_target_kl_loss(
            outputs,
            target_probability,
            time_mask=time_mask,
        )
        return self._apply_loss(loss, components)
