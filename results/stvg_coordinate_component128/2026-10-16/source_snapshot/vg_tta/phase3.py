"""Temporal-first diagnostics and TrajCal primitives for Phase 3.

The functions in this module are deliberately model-agnostic.  Boxes and
decoder features are read-only evidence; every trainable objective produced
here targets temporal start/end distributions only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import aligned_box_iou_cxcywh, temporal_iou
from .tta import temporal_entropy_loss


def temporal_probabilities(pred_sted: torch.Tensor) -> torch.Tensor:
    """Return start/end probabilities normalized over time."""

    if pred_sted.ndim != 3 or pred_sted.shape[0] != 1 or pred_sted.shape[-1] != 2:
        raise ValueError("pred_sted must have shape 1xTx2")
    return pred_sted.detach().float().softmax(dim=1)


def legal_interval_joint(probabilities: torch.Tensor) -> torch.Tensor:
    """Normalized joint ``P(start,end)`` with TubeDETR's ``end > start`` rule."""

    if probabilities.ndim == 3:
        if probabilities.shape[0] != 1:
            raise ValueError("only batch size one is supported")
        probabilities = probabilities[0]
    if probabilities.ndim != 2 or probabilities.shape[-1] != 2:
        raise ValueError("probabilities must have shape Tx2 or 1xTx2")
    start = probabilities[:, 0]
    end = probabilities[:, 1]
    joint = start[:, None] * end[None, :]
    diagonal = 1 if len(start) > 1 else 0
    valid = torch.triu(torch.ones_like(joint, dtype=torch.bool), diagonal=diagonal)
    joint = joint.masked_fill(~valid, 0.0)
    return joint / joint.sum().clamp(min=1e-12)


def interval_membership_probability(joint: torch.Tensor) -> torch.Tensor:
    """Marginal probability that each frame lies in an inclusive interval."""

    if joint.ndim != 2 or joint.shape[0] != joint.shape[1]:
        raise ValueError("joint interval posterior must be square")
    duration = joint.shape[0]
    return torch.stack(
        [joint[: frame + 1, frame:].sum() for frame in range(duration)]
    )


def trajectory_identity_evidence(
    features: torch.Tensor,
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a soft prototype and cosine evidence from ``T x hidden`` features."""

    if features.ndim != 2:
        raise ValueError("features must have shape TxH")
    joint = legal_interval_joint(probabilities)
    membership = interval_membership_probability(joint)
    if len(membership) != len(features):
        raise ValueError("features and temporal probabilities must be frame-aligned")
    normalized_features = F.normalize(features.detach().float(), dim=-1)
    prototype = (
        membership[:, None] * normalized_features
    ).sum(dim=0) / membership.sum().clamp(min=1e-12)
    prototype = F.normalize(prototype, dim=0)
    evidence = normalized_features @ prototype
    return evidence, prototype, membership


def interval_inside_outside_contrast(evidence: torch.Tensor) -> torch.Tensor:
    """Inside-minus-outside mean evidence for every legal interval."""

    if evidence.ndim != 1:
        raise ValueError("trajectory evidence must be one-dimensional")
    duration = len(evidence)
    prefix = torch.cat([evidence.new_zeros(1), evidence.cumsum(dim=0)])
    total = prefix[-1]
    starts = torch.arange(duration, device=evidence.device)[:, None]
    ends = torch.arange(duration, device=evidence.device)[None, :]
    inside_count = ends - starts + 1
    safe_inside_count = inside_count.clamp(min=1)
    inside_sum = prefix[ends + 1] - prefix[starts]
    inside_mean = inside_sum / safe_inside_count
    outside_count = duration - inside_count
    outside_mean = (total - inside_sum) / outside_count.clamp(min=1)
    contrast = inside_mean - outside_mean
    contrast = torch.where(outside_count == 0, torch.zeros_like(contrast), contrast)
    diagonal = 1 if duration > 1 else 0
    valid = torch.triu(torch.ones_like(contrast, dtype=torch.bool), diagonal=diagonal)
    # Invalid cells contribute through the already-masked base score.  Keep
    # their contrast at zero so trajectory_weight=0 cannot create 0 * -inf NaNs.
    return contrast.masked_fill(~valid, 0.0)


def base_interval_log_scores(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim == 3:
        probabilities = probabilities[0]
    if probabilities.ndim != 2 or probabilities.shape[-1] != 2:
        raise ValueError("probabilities must have shape Tx2")
    start = probabilities[:, 0].clamp(min=1e-12).log()
    end = probabilities[:, 1].clamp(min=1e-12).log()
    scores = start[:, None] + end[None, :]
    diagonal = 1 if len(start) > 1 else 0
    valid = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=diagonal)
    return scores.masked_fill(~valid, -torch.inf)


def topk_intervals(
    scores: torch.Tensor,
    k: int,
) -> list[tuple[int, int, float]]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("interval scores must be square")
    finite = torch.isfinite(scores)
    count = int(finite.sum().item())
    if count == 0:
        raise ValueError("no legal intervals")
    k = min(max(int(k), 1), count)
    flat_scores, flat_indices = scores.flatten().topk(k)
    duration = scores.shape[0]
    return [
        (int(index // duration), int(index % duration), float(score))
        for score, index in zip(flat_scores.detach().cpu(), flat_indices.detach().cpu())
    ]


def trajectory_reranked_interval(
    probabilities: torch.Tensor,
    features: torch.Tensor,
    *,
    top_k: int,
    trajectory_weight: float,
) -> tuple[int, int, dict[str, float]]:
    evidence, _, membership = trajectory_identity_evidence(features, probabilities)
    base_scores = base_interval_log_scores(probabilities)
    contrast = interval_inside_outside_contrast(evidence)
    candidates = topk_intervals(base_scores, top_k)
    best = max(
        candidates,
        key=lambda item: item[2]
        + trajectory_weight * float(contrast[item[0], item[1]].item()),
    )
    return best[0], best[1], {
        "base_log_score": best[2],
        "trajectory_contrast": float(contrast[best[0], best[1]].item()),
        "trajectory_weight": float(trajectory_weight),
        "membership_mean": float(membership.mean().item()),
        "evidence_mean": float(evidence.mean().item()),
        "evidence_std": float(evidence.std(unbiased=False).item()),
    }


def trajectory_temporal_target(
    probabilities: torch.Tensor,
    features: torch.Tensor,
    *,
    trajectory_weight: float,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Construct soft start/end targets from trajectory-calibrated intervals."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    evidence, _, membership = trajectory_identity_evidence(features, probabilities)
    base_scores = base_interval_log_scores(probabilities)
    contrast = interval_inside_outside_contrast(evidence)
    scores = base_scores + float(trajectory_weight) * contrast
    if top_k is not None:
        keep = torch.zeros_like(scores, dtype=torch.bool)
        for start, end, _ in topk_intervals(base_scores, top_k):
            keep[start, end] = True
        scores = scores.masked_fill(~keep, -torch.inf)
    finite = torch.isfinite(scores)
    posterior = scores.new_zeros(scores.shape)
    posterior[finite] = torch.softmax(scores[finite] / float(temperature), dim=0)
    start_target = posterior.sum(dim=1)
    end_target = posterior.sum(dim=0)
    target = torch.stack([start_target, end_target], dim=-1)[None]
    return target.detach(), {
        "trajectory_weight": float(trajectory_weight),
        "temperature": float(temperature),
        "top_k": -1.0 if top_k is None else float(top_k),
        "interval_posterior_entropy": float(
            (-(posterior[finite] * posterior[finite].clamp(min=1e-12).log()).sum()).item()
        ),
        "trajectory_evidence_mean": float(evidence.mean().item()),
        "trajectory_evidence_std": float(evidence.std(unbiased=False).item()),
        "membership_mean": float(membership.mean().item()),
    }


class TemporalPriorBias(nn.Module):
    """Query-independent start/end position prior on normalized video time."""

    def __init__(self, bins: int = 32) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError("temporal prior needs at least two bins")
        self.bins = int(bins)
        self.bias = nn.Parameter(torch.zeros(1, 2, self.bins))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
            raise ValueError("temporal logits must have shape 1xTx2")
        interpolated = F.interpolate(
            self.bias,
            size=logits.shape[1],
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return logits + interpolated


class TemporalPriorAdapter:
    """Adapt only an external query-independent temporal position prior."""

    def __init__(
        self,
        *,
        bins: int = 32,
        learning_rate: float = 1e-4,
        optimizer_eps: float = 1e-4,
        weight_decay: float = 0.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.module = TemporalPriorBias(bins).to(device)
        self.learning_rate = float(learning_rate)
        self.optimizer_eps = float(optimizer_eps)
        self.weight_decay = float(weight_decay)
        self._initial = self.module.bias.detach().clone()
        self._optimizer: torch.optim.Optimizer | None = None

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.module.parameters(),
                lr=self.learning_rate,
                eps=self.optimizer_eps,
                weight_decay=self.weight_decay,
            )
        return self._optimizer

    @torch.no_grad()
    def reset(self) -> None:
        self.module.bias.copy_(self._initial)
        self._optimizer = None

    def adjusted_outputs(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            **outputs,
            "pred_sted": self.module(outputs["pred_sted"].detach().float()),
        }

    def step_entropy(self, outputs: dict[str, torch.Tensor]) -> dict[str, float]:
        adjusted = self.adjusted_outputs(outputs)
        mask = torch.ones_like(adjusted["pred_sted"][..., 0], dtype=torch.bool)
        loss, components = temporal_entropy_loss(adjusted, time_mask=mask)
        optimizer = self._get_optimizer()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = self.module.bias.grad.detach().float()
        gradient_norm = float(grad.square().sum().sqrt().item())
        optimizer.step()
        return {
            **components,
            "gradient_norm_l2": gradient_norm,
            **self.delta_from_initial(),
        }

    @torch.no_grad()
    def delta_from_initial(self) -> dict[str, float]:
        delta = self.module.bias.detach() - self._initial
        delta_l2 = float(delta.float().square().sum().sqrt().item())
        return {
            "parameter_delta_l2": delta_l2,
            "parameter_delta_max_abs": float(delta.float().abs().max().item()),
            # The zero-initialized prior has no meaningful relative-to-initial
            # norm. RMS gives a finite, scale-aware diagnostic instead.
            "parameter_delta_rms": float(delta_l2 / max(delta.numel(), 1) ** 0.5),
            "parameter_norm_l2": 0.0,
        }


def temporal_error_features(
    pred_indices: tuple[int, int],
    *,
    frame_ids: Sequence[int],
    gt_frame_interval: Sequence[int],
) -> dict[str, float]:
    pred = [int(frame_ids[pred_indices[0]]), int(frame_ids[pred_indices[1]]) + 1]
    gt = [int(gt_frame_interval[0]), int(gt_frame_interval[1])]
    pred_duration = max(pred[1] - pred[0], 1)
    gt_duration = max(gt[1] - gt[0], 1)
    pred_center = 0.5 * (pred[0] + pred[1])
    gt_center = 0.5 * (gt[0] + gt[1])
    return {
        "start_error_frames": float(pred[0] - gt[0]),
        "end_error_frames": float(pred[1] - gt[1]),
        "start_abs_error_frames": float(abs(pred[0] - gt[0])),
        "end_abs_error_frames": float(abs(pred[1] - gt[1])),
        "duration_ratio": float(pred_duration / gt_duration),
        "center_error_frames": float(pred_center - gt_center),
        "center_abs_error_frames": float(abs(pred_center - gt_center)),
        "top1_tIoU": temporal_iou(pred, gt),
    }


def tube_fragmentation_features(
    pred_boxes: torch.Tensor,
    targets: Sequence[dict[str, Any]],
    gt_indices: tuple[int, int],
    *,
    threshold: float = 0.3,
) -> dict[str, float]:
    positions = list(range(gt_indices[0], gt_indices[1] + 1))
    ious: list[float] = []
    for position in positions:
        boxes = targets[position].get("boxes")
        if boxes is None or len(boxes) == 0:
            continue
        iou = aligned_box_iou_cxcywh(
            pred_boxes[position : position + 1].detach().float().cpu(),
            boxes[:1].detach().float().cpu(),
        )[0]
        ious.append(float(iou.item()))
    if not ious:
        raise ValueError("ground-truth interval contains no boxes")
    hits = [value > threshold for value in ious]
    transitions = sum(left != right for left, right in zip(hits, hits[1:]))
    longest = 0
    current = 0
    for hit in hits:
        current = current + 1 if hit else 0
        longest = max(longest, current)
    midpoint = max(1, len(hits) // 2)
    first_rate = float(np.mean(hits[:midpoint]))
    second_rate = float(np.mean(hits[midpoint:])) if midpoint < len(hits) else first_rate
    seen_hit = False
    seen_gap_after_hit = False
    reacquired = False
    for hit in hits:
        if hit and seen_gap_after_hit:
            reacquired = True
        if hit:
            seen_hit = True
        elif seen_hit:
            seen_gap_after_hit = True
    return {
        "mean_gt_interval_box_iou": float(np.mean(ious)),
        "target_hit_ratio": float(np.mean(hits)),
        "hit_transition_count": float(transitions),
        "longest_contiguous_hit_ratio": float(longest / len(hits)),
        "front_hit_back_miss": float(first_rate >= 0.5 and second_rate < 0.5),
        "reacquired_after_internal_miss_proxy": float(reacquired),
        "first_half_hit_ratio": first_rate,
        "second_half_hit_ratio": second_rate,
    }


RELATION_TERMS = (
    "left of",
    "right of",
    "in front of",
    "behind",
    "next to",
    "beside",
    "near ",
    "between",
    "following",
    "followed by",
    "holding",
    "carrying",
    "wearing",
    "with the",
)
MOTION_TERMS = (
    " walk",
    " run",
    " move",
    " ride",
    " jump",
    " dance",
    " drive",
    " enter",
    " leave",
    " approach",
    " turn",
    " climb",
    " swim",
    " fly",
    " throw",
    " catch",
    " kick",
    " play",
    " talk",
    " eat",
    " drink",
    " open",
    " close",
    " pick",
    " put",
    " stand",
    " sit",
)
COMPOSITION_TERMS = (
    " while ",
    " and then ",
    " who ",
    " that ",
    " which ",
    " before ",
    " after ",
)


def classify_query(caption: str) -> str:
    """Deterministic, mutually exclusive rule taxonomy for 109-query audit."""

    text = f" {caption.lower().strip()} "
    if any(term in text for term in COMPOSITION_TERMS) or text.count(" and ") >= 2:
        return "compositional"
    if any(term in text for term in RELATION_TERMS):
        return "relation-heavy"
    if any(term in text for term in MOTION_TERMS):
        return "motion-heavy"
    return "static-heavy"


def deterministic_derangement(length: int, seed: int) -> list[int]:
    if length < 2:
        raise ValueError("derangement needs at least two elements")
    rng = np.random.default_rng(seed)
    base = np.arange(length)
    for _ in range(10_000):
        candidate = rng.permutation(length)
        if np.all(candidate != base):
            return [int(value) for value in candidate]
    raise RuntimeError("failed to construct a deterministic derangement")


def deterministic_frame_permutation(length: int, seed: int) -> list[int]:
    if length < 2:
        return list(range(length))
    return [int(value) for value in np.random.default_rng(seed).permutation(length)]
