"""Phase-2 helpers for temporal shifts, multi-view predictions, and split design."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .augmentations import make_strong_view


def source_cluster(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def temporally_subsample_sample(
    video: torch.Tensor,
    targets: Sequence[dict[str, Any]],
    video_target: dict[str, Any],
    *,
    factor: int = 2,
    offset: int = 0,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Keep every ``factor``-th frame and remap frame-level GT by timestamp.

    ``frames_id`` remains in original-video coordinates, so corrected vIoU is
    evaluated against the original GT time interval instead of reusing stale
    positions from the pre-shift tensor.
    """

    if factor < 2:
        raise ValueError("temporal subsampling factor must be at least 2")
    if not 0 <= offset < factor:
        raise ValueError("offset must be in [0, factor)")
    if video.ndim != 4 or len(targets) != video.shape[1]:
        raise ValueError("video and targets must be aligned CxTxHxW data")
    frame_ids = list(video_target["frames_id"])
    if len(frame_ids) != video.shape[1]:
        raise ValueError("frames_id and video duration must match")
    positions = list(range(offset, video.shape[1], factor))
    if not positions:
        raise ValueError("temporal shift removed every frame")
    shifted_targets = [targets[position] for position in positions]
    gt_positions = [
        shifted_index
        for shifted_index, target in enumerate(shifted_targets)
        if target.get("boxes") is not None and len(target["boxes"])
    ]
    if not gt_positions:
        raise ValueError("temporal shift removed every target-present frame")
    shifted_target = dict(video_target)
    shifted_target["frames_id"] = [frame_ids[position] for position in positions]
    shifted_target["inter_idx"] = [gt_positions[0], gt_positions[-1]]
    shifted_target["temporal_subsample_positions"] = positions
    return video[:, positions], shifted_targets, shifted_target


def aggregate_predictions(
    predictions: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Robustly combine aligned photometric-view TubeDETR predictions."""

    if not predictions:
        raise ValueError("at least one prediction is required")
    box_shapes = {tuple(item["pred_boxes"].shape) for item in predictions}
    sted_shapes = {tuple(item["pred_sted"].shape) for item in predictions}
    if len(box_shapes) != 1 or len(sted_shapes) != 1:
        raise ValueError("multi-view predictions must be frame-aligned")
    boxes = torch.stack([item["pred_boxes"].detach().float() for item in predictions])
    probabilities = torch.stack(
        [item["pred_sted"].detach().float().softmax(dim=1) for item in predictions]
    )
    consensus_probability = probabilities.mean(dim=0).clamp(min=1e-8)
    return {
        "pred_boxes": boxes.median(dim=0).values,
        "pred_sted": consensus_probability.log(),
    }


def subset_prediction_frames(
    prediction: dict[str, torch.Tensor], positions: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Align a full-rate teacher prediction to retained temporal positions."""

    index = torch.as_tensor(positions, dtype=torch.long, device=prediction["pred_boxes"].device)
    if index.ndim != 1 or len(index) == 0:
        raise ValueError("positions must be a non-empty one-dimensional sequence")
    return {
        "pred_boxes": prediction["pred_boxes"].index_select(0, index),
        "pred_sted": prediction["pred_sted"].index_select(1, index),
    }


def deterministic_views(
    video: torch.Tensor,
    *,
    seed: int,
    count: int = 4,
    include_identity: bool = True,
) -> list[torch.Tensor]:
    if count < 1:
        raise ValueError("view count must be positive")
    views: list[torch.Tensor] = [video] if include_identity else []
    needed = count - len(views)
    views.extend(make_strong_view(video, seed=seed + index) for index in range(needed))
    return views


def cluster_macro_mean(values: Sequence[float], clusters: Sequence[str]) -> float:
    if len(values) != len(clusters) or not values:
        raise ValueError("cluster macro mean requires aligned non-empty values")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    return float(np.mean([np.mean(group_values) for group_values in grouped.values()]))
