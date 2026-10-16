"""GT-free common-grid alignment for a prospective TA corruption study.

TA-STVG chooses its interval through per-offset native decoding and endpoint
merge. Its merged logits need not decode to that same interval. Consequently
alignment maps the ALREADY decoded indices and never performs another argmax.
Only predictions are interpolated, not ground truth or input pixels.
"""
from __future__ import annotations

import copy
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _integer_grid(values: Sequence[int], name: str) -> list[int]:
    result = list(values)
    if len(result) < 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} requires at least two integer positions")
    result = [int(v) for v in result]
    if result != sorted(set(result)) or result[0] < 0:
        raise ValueError(f"{name} must be unique, nonnegative and strictly increasing")
    return result


def align_native_prediction(
    prediction: Mapping[str, Any],
    retained_positions: Sequence[int],
    original_frame_ids: Sequence[int],
) -> dict[str, Any]:
    """Map a native merged prediction to its original clean evaluation grid.

    Spatial boxes are interpolated linearly by physical frame timestamp;
    outside the observed range the nearest predicted box is held, as in the
    existing TubeDETR corruption comparator. Temporal endpoints remain at
    observed positions, not at extrapolated times. Sentinel logits are only
    for storage/diagnostics: downstream metrics must use ``pred_indices``.
    The identity branch preserves all original tensor values and dtypes.
    """
    ids = _integer_grid(original_frame_ids, "original_frame_ids")
    positions = _integer_grid(retained_positions, "retained_positions")
    if positions[-1] >= len(ids):
        raise ValueError("retained position is outside the original grid")
    boxes, logits = prediction["pred_boxes"], prediction["pred_sted"]
    if not torch.is_tensor(boxes) or boxes.ndim != 2 or tuple(boxes.shape) != (len(positions), 4):
        raise ValueError("pred_boxes must be retained-time x 4")
    if not torch.is_tensor(logits) or tuple(logits.shape) != (len(positions), 2):
        raise ValueError("pred_sted must be merged retained-time x 2")
    if not boxes.is_floating_point() or not logits.is_floating_point():
        raise ValueError("prediction tensors must be floating point")
    if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
        raise ValueError("nonfinite predictions cannot be aligned")
    endpoints = tuple(prediction["pred_indices"])
    if len(endpoints) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in endpoints):
        raise ValueError("native pred_indices must be two integers")
    start, end = map(int, endpoints)
    if not 0 <= start < end < len(positions):
        raise ValueError("native interval must contain at least two observed positions")

    result = copy.deepcopy(dict(prediction))
    if positions == list(range(len(ids))):
        result["alignment"] = {
            "identity": True, "gt_used": False,
            "retained_positions": positions,
            "original_frame_ids": ids,
            "native_indices_not_redecoded": True,
        }
        return result

    times = np.asarray(ids, dtype=np.float64)
    observed_times = times[positions]
    original_boxes = boxes.detach().float().cpu().numpy()
    dense = np.stack([
        np.interp(times, observed_times, original_boxes[:, coordinate])
        for coordinate in range(4)
    ], axis=-1)
    # Float32 matches the native merged evaluator representation. This is a
    # prediction-only resampling, not an altered CUDA head replay claim.
    result["pred_boxes"] = torch.from_numpy(dense).float()
    aligned_logits = torch.full((len(ids), 2), -1e20, dtype=torch.float32)
    aligned_logits[positions] = logits.detach().float().cpu()
    result["pred_sted"] = aligned_logits
    mapped = (positions[start], positions[end])
    result["pred_indices"] = mapped
    frame_span = [ids[mapped[0]], ids[mapped[1]] + 1]
    # Different audited parents name this diagnostic differently. Preserve
    # whichever is already present; ensure its coordinates remain consistent.
    for key in ("pred_frame_span", "predicted_frame_span", "pred_indices_frame_span"):
        if key in result:
            result[key] = frame_span
    result["alignment"] = {
        "identity": False, "gt_used": False,
        "retained_positions": positions, "original_frame_ids": ids,
        "retained_frame_ids": [ids[position] for position in positions],
        "native_retained_indices": [start, end],
        "mapped_original_indices": list(mapped),
        "native_indices_not_redecoded": True,
        "spatial_rule": "timestamp-linear prediction interpolation; endpoint hold",
        "temporal_rule": "map native decoded indices, no new argmax or endpoint extrapolation",
        "logits_are_diagnostic_only": True,
    }
    return result
