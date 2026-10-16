"""Per-video STVG metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch


TEMPORAL_DECODER_VERSION = "cpu_fp64_raw_joint_v1"


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def aligned_box_iou_cxcywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU for aligned normalized boxes, returning one value per pair."""

    pred_xyxy = box_cxcywh_to_xyxy(pred)
    target_xyxy = box_cxcywh_to_xyxy(target)
    lt = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    rb = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    pred_area = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=0).prod(dim=-1)
    target_area = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp(min=0).prod(dim=-1)
    return inter / (pred_area + target_area - inter).clamp(min=1e-7)


def generalized_box_iou_aligned_cxcywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_xyxy = box_cxcywh_to_xyxy(pred)
    target_xyxy = box_cxcywh_to_xyxy(target)
    lt = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    rb = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    pred_area = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=0).prod(dim=-1)
    target_area = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp(min=0).prod(dim=-1)
    union = (pred_area + target_area - inter).clamp(min=1e-7)
    iou = inter / union
    enclosing_lt = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
    enclosing_rb = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enclosing = (enclosing_rb - enclosing_lt).clamp(min=0).prod(dim=-1).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


def interval_from_logits(sted_logits: torch.Tensor) -> tuple[int, int]:
    """Decode the best inclusive [start, end] interval with end >= start.

    TubeDETR's official postprocessor disallows zero-length intervals. We retain
    that behavior when T > 1 and gracefully support one-frame smoke tests.

    Decode detached CPU float64 raw joint scores. Subtracting the two
    log-softmax normalizers cannot change the mathematical argmax, but doing
    so in BF16 introduced device-dependent ties (CPU query-feature caches vs
    CUDA full forwards). Casting BEFORE arithmetic and using one device makes
    decoding identical for the same stored logits on every inference path.
    """

    if sted_logits.ndim == 3:
        if sted_logits.shape[0] != 1:
            raise ValueError("only batch size 1 is supported by per-video metrics")
        sted_logits = sted_logits[0]
    if sted_logits.ndim != 2 or sted_logits.shape[1] != 2:
        raise ValueError(f"expected Tx2 logits, got {tuple(sted_logits.shape)}")
    length = sted_logits.shape[0]
    if length == 0:
        raise ValueError("temporal logits must contain at least one frame")
    logits = sted_logits.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(logits).all():
        raise ValueError("temporal logits must be finite")
    scores = logits[:, 0, None] + logits[None, :, 1]
    diagonal = 1 if length > 1 else 0
    valid = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=diagonal)
    scores = scores.masked_fill(~valid, -torch.inf)
    flat = int(scores.argmax().item())
    return flat // length, flat % length


def temporal_iou(pred: Sequence[float], target: Sequence[float]) -> float:
    intersection = max(0.0, min(pred[1], target[1]) - max(pred[0], target[0]))
    union = max(pred[1], target[1]) - min(pred[0], target[0])
    return float(intersection / union) if union > 0 else 0.0


def compute_stvg_metrics(
    pred_boxes: torch.Tensor,
    targets: Sequence[dict],
    pred_indices: tuple[int, int],
    gt_indices: tuple[int, int],
    *,
    frame_ids: Sequence[int] | None = None,
    gt_frame_interval: Sequence[int] | None = None,
) -> dict[str, float | list[int]]:
    """Compute tIoU, spatial IoU, and vIoU for one sampled video.

    Spatial IoU is the mean box IoU over sampled ground-truth tube frames.
    vIoU sums box IoU only in the temporal intersection and divides by the
    number of sampled frames in the temporal union, matching TubeDETR's logic.
    """

    pred_start, pred_end = pred_indices
    gt_start, gt_end = gt_indices
    if frame_ids is None:
        frame_ids = list(range(len(targets)))
    if len(frame_ids) != len(targets):
        raise ValueError("frame_ids and targets must have the same length")
    pred_frame_interval = [int(frame_ids[pred_start]), int(frame_ids[pred_end]) + 1]
    if gt_frame_interval is None:
        gt_frame_interval = [int(frame_ids[gt_start]), int(frame_ids[gt_end]) + 1]
    gt_frame_interval = [int(gt_frame_interval[0]), int(gt_frame_interval[1])]

    per_gt_frame_iou: dict[int, float] = {}
    for index in range(gt_start, gt_end + 1):
        boxes = targets[index].get("boxes")
        if boxes is None or len(boxes) == 0:
            continue
        value = aligned_box_iou_cxcywh(
            pred_boxes[index : index + 1].detach().float().cpu(),
            boxes[:1].detach().float().cpu(),
        )[0]
        per_gt_frame_iou[index] = float(value.item())

    siou = float(np.mean(list(per_gt_frame_iou.values()))) if per_gt_frame_iou else 0.0
    temporal_intersection_start = max(pred_frame_interval[0], gt_frame_interval[0])
    temporal_intersection_end = min(pred_frame_interval[1], gt_frame_interval[1])
    union_start = min(pred_frame_interval[0], gt_frame_interval[0])
    # Correct TubeDETR evaluator (commit 3c32cc9): a temporal union ends at
    # max(pred_end, gt_end).  The original public evaluator used min(...),
    # which shrank the denominator and inflated vIoU.  Keep both numbers so
    # corrected results are scientifically valid while old tables remain
    # reproducible.
    union_end = max(pred_frame_interval[1], gt_frame_interval[1])
    legacy_union_end = min(pred_frame_interval[1], gt_frame_interval[1])
    union_sample_count = sum(union_start <= frame_id < union_end for frame_id in frame_ids)
    legacy_union_sample_count = sum(
        union_start <= frame_id < legacy_union_end for frame_id in frame_ids
    )
    intersection_iou_sum = sum(
        iou
        for index, iou in per_gt_frame_iou.items()
        if temporal_intersection_start <= frame_ids[index] < temporal_intersection_end
    )
    viou_corrected = intersection_iou_sum / max(union_sample_count, 1)
    viou_legacy = intersection_iou_sum / max(legacy_union_sample_count, 1)
    start_error = pred_frame_interval[0] - gt_frame_interval[0]
    end_error = pred_frame_interval[1] - gt_frame_interval[1]
    return {
        "tIoU": temporal_iou(pred_frame_interval, gt_frame_interval),
        "sIoU": siou,
        # vIoU remains an alias for the corrected primary metric so existing
        # analysis code cannot silently fall back to the legacy definition.
        "vIoU": float(viou_corrected),
        "vIoU_corrected": float(viou_corrected),
        "vIoU_legacy": float(viou_legacy),
        "vIoU_at_0.3": float(viou_corrected > 0.3),
        "vIoU_at_0.5": float(viou_corrected > 0.5),
        "vIoU_legacy_at_0.3": float(viou_legacy > 0.3),
        "vIoU_legacy_at_0.5": float(viou_legacy > 0.5),
        "start_error_frames": float(start_error),
        "end_error_frames": float(end_error),
        "start_abs_error_frames": float(abs(start_error)),
        "end_abs_error_frames": float(abs(end_error)),
        "temporal_union_sample_count": float(union_sample_count),
        "temporal_union_sample_count_legacy": float(legacy_union_sample_count),
        "predicted_interval": pred_frame_interval,
        "gt_interval": gt_frame_interval,
    }


def paired_bootstrap_ci(
    tta_values: Iterable[float],
    frozen_values: Iterable[float],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    tta = np.asarray(list(tta_values), dtype=np.float64)
    frozen = np.asarray(list(frozen_values), dtype=np.float64)
    if tta.shape != frozen.shape or tta.ndim != 1 or len(tta) == 0:
        raise ValueError("paired bootstrap requires equally sized non-empty vectors")
    differences = tta - frozen
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    means = differences[sampled_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def cluster_paired_bootstrap_ci(
    tta_values: Iterable[float],
    frozen_values: Iterable[float],
    clusters: Iterable[str],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired bootstrap that resamples source-video clusters, not clips."""

    tta = np.asarray(list(tta_values), dtype=np.float64)
    frozen = np.asarray(list(frozen_values), dtype=np.float64)
    cluster_array = np.asarray(list(clusters), dtype=object)
    if (
        tta.shape != frozen.shape
        or tta.shape != cluster_array.shape
        or tta.ndim != 1
        or len(tta) == 0
    ):
        raise ValueError("cluster bootstrap requires aligned non-empty one-dimensional inputs")
    differences = tta - frozen
    unique_clusters = np.unique(cluster_array)
    cluster_indices = {
        cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_indices = np.concatenate(
            [cluster_indices[cluster] for cluster in sampled_clusters]
        )
        means[bootstrap_index] = differences[sampled_indices].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_clusters": int(len(unique_clusters)),
    }


def cluster_macro_paired_bootstrap_ci(
    tta_values: Iterable[float],
    frozen_values: Iterable[float],
    clusters: Iterable[str],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Equal-cluster-weight paired effect and bootstrap confidence interval."""

    tta = np.asarray(list(tta_values), dtype=np.float64)
    frozen = np.asarray(list(frozen_values), dtype=np.float64)
    cluster_array = np.asarray(list(clusters), dtype=object)
    if (
        tta.shape != frozen.shape
        or tta.shape != cluster_array.shape
        or tta.ndim != 1
        or len(tta) == 0
    ):
        raise ValueError("cluster macro bootstrap requires aligned non-empty vectors")
    differences = tta - frozen
    unique_clusters = np.unique(cluster_array)
    cluster_means = np.asarray(
        [differences[cluster_array == cluster].mean() for cluster in unique_clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(cluster_means),
        size=(n_bootstrap, len(cluster_means)),
    )
    means = cluster_means[sampled_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(cluster_means.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_clusters": int(len(unique_clusters)),
    }
