"""Geometry-only spatial teachers for episodic view consistency.

The helpers in this module never inspect annotations or dataset metadata.  They
only transform normalized ``cxcywh`` predictions, form a conservative crop
around a predicted temporal tube, and select a per-frame teacher when at least
two geometric views agree.  A teacher is a medoid prediction, never an average
of views that are far apart.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_VIEW_ORDER = ("global_high", "crop", "original")
_VIEW_ALIASES = {"origin": "original", "global": "global_high"}
_REQUIRED_VIEWS = frozenset(_VIEW_ORDER)


def _require_box_tensor(boxes: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(boxes):
        raise TypeError(f"{name} must be a tensor")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [T,4], got {tuple(boxes.shape)}")
    if not boxes.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must be finite")
    return boxes


def _require_temporal_length(boxes: torch.Tensor) -> int:
    time_count = int(boxes.shape[0])
    if time_count < 1:
        raise ValueError("box sequences must contain at least one frame")
    return time_count


def _as_crop(crop_xyxy_norm: torch.Tensor | Sequence[float], device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(crop_xyxy_norm):
        crop = torch.as_tensor(crop_xyxy_norm, dtype=torch.float32, device=device)
    else:
        crop = crop_xyxy_norm.to(device=device, dtype=torch.float32)
    if crop.numel() != 4:
        raise ValueError(f"crop_xyxy_norm must contain four values, got shape {tuple(crop.shape)}")
    crop = crop.reshape(4)
    if not torch.isfinite(crop).all():
        raise ValueError("crop_xyxy_norm must be finite")
    x0, y0, x1, y1 = (float(value) for value in crop.tolist())
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(
            "crop_xyxy_norm must be a non-degenerate normalized rectangle "
            f"inside [0,1], got {crop.tolist()}"
        )
    return crop


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half = boxes[..., 2:].abs() * 0.5
    return torch.cat((center - half, center + half), dim=-1)


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    minimum = boxes[..., :2]
    maximum = boxes[..., 2:]
    return torch.cat(((minimum + maximum) * 0.5, maximum - minimum), dim=-1)


def map_crop_boxes(
    boxes_cxcywh: torch.Tensor,
    crop_xyxy_norm: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Map crop-relative normalized ``cxcywh`` boxes to full-image coordinates.

    ``crop_xyxy_norm`` is ``(x0,y0,x1,y1)`` in full-image normalized
    coordinates, with ``x1``/``y1`` treated as the right/bottom boundary.  The
    returned boxes keep the input shape/device/dtype and are clipped to the
    normalized full-image square.  Clipping makes malformed crop predictions
    safe as teachers without changing the crop geometry itself.
    """

    boxes = _require_box_tensor(boxes_cxcywh, "boxes_cxcywh")
    crop = _as_crop(crop_xyxy_norm, boxes.device)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop.unbind()
    crop_width = crop_x1 - crop_x0
    crop_height = crop_y1 - crop_y0

    values = boxes.float()
    centers = values[..., :2].clone()
    centers[..., 0] = crop_x0 + centers[..., 0] * crop_width
    centers[..., 1] = crop_y0 + centers[..., 1] * crop_height
    widths = values[..., 2].abs() * crop_width
    heights = values[..., 3].abs() * crop_height
    mapped = torch.cat((centers, torch.stack((widths, heights), dim=-1)), dim=-1)
    clipped_xyxy = _cxcywh_to_xyxy(mapped).clamp(0.0, 1.0)
    result = _xyxy_to_cxcywh(clipped_xyxy)
    return result.to(dtype=boxes.dtype)


def _coerce_image_hw(image_hw: Sequence[int]) -> tuple[int, int]:
    if torch.is_tensor(image_hw):
        values = image_hw.detach().cpu().reshape(-1).tolist()
    else:
        values = list(image_hw)
    if len(values) != 2:
        raise ValueError(f"image_hw must contain (height,width), got {image_hw!r}")
    height, width = (int(value) for value in values)
    if height < 1 or width < 1 or any(float(value) != int(value) for value in values):
        raise ValueError(f"image_hw must contain positive integers, got {image_hw!r}")
    return height, width


def _temporal_mask(
    temporal_indices: Any,
    time_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize an index sequence or length-T boolean mask to a bool tensor."""

    if isinstance(temporal_indices, slice):
        values = list(range(*temporal_indices.indices(time_count)))
    elif torch.is_tensor(temporal_indices):
        if temporal_indices.ndim != 1:
            raise ValueError("temporal_indices tensor must be one-dimensional")
        if temporal_indices.dtype == torch.bool:
            if temporal_indices.numel() != time_count:
                raise ValueError("boolean temporal_indices mask must have length T")
            return temporal_indices.to(device=device, dtype=torch.bool).clone()
        if not temporal_indices.is_floating_point() and not temporal_indices.is_complex():
            values = temporal_indices.detach().cpu().tolist()
        else:
            if temporal_indices.is_complex() or not torch.isfinite(temporal_indices).all():
                raise ValueError("temporal_indices must be finite real indices")
            rounded = temporal_indices.round()
            if not torch.equal(rounded, temporal_indices):
                raise ValueError("temporal_indices values must be integral")
            values = rounded.detach().cpu().tolist()
    elif isinstance(temporal_indices, (str, bytes)):
        raise TypeError("temporal_indices must be indices, a bool mask, or a slice")
    elif isinstance(temporal_indices, Sequence):
        values = list(temporal_indices)
        if values and all(isinstance(value, bool) for value in values):
            if len(values) != time_count:
                raise ValueError("boolean temporal_indices mask must have length T")
            return torch.tensor(values, dtype=torch.bool, device=device)
    elif isinstance(temporal_indices, int):
        values = [temporal_indices]
    else:
        raise TypeError("temporal_indices must be indices, a bool mask, or a slice")

    mask = torch.zeros(time_count, dtype=torch.bool, device=device)
    for value in values:
        if isinstance(value, bool):
            raise ValueError("mixed boolean/integer temporal_indices are not supported")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("temporal_indices values must be integral")
        index = int(value)
        if index < 0 or index >= time_count:
            raise ValueError(f"temporal index {index} is outside [0,{time_count})")
        mask[index] = True
    return mask


def _integer_crop_interval(center: float, extent: float, limit: int, minimum: int = 32) -> tuple[int, int]:
    if limit < 1:
        raise ValueError("image dimensions must be positive")
    target = min(float(limit), max(float(minimum), float(extent)))
    if target >= float(limit):
        return 0, limit

    start = center - target * 0.5
    end = center + target * 0.5
    start = min(max(start, 0.0), float(limit) - target)
    end = start + target
    left = max(0, min(limit, int(math.floor(start))))
    right = max(left, min(limit, int(math.ceil(end))))
    minimum_width = min(limit, minimum)
    if right - left < minimum_width:
        right = min(limit, left + minimum_width)
        left = max(0, right - minimum_width)
    return left, right


def tube_crop_rect(
    boxes_cxcywh: torch.Tensor,
    temporal_indices: Any,
    image_hw: Sequence[int],
    expansion: float = 1.5,
) -> tuple[int, int, int, int]:
    """Return one integer full-image crop around a predicted temporal tube.

    Coordinates are normalized ``cxcywh``.  The union over ``temporal_indices``
    is expanded around its center by ``expansion`` and converted to an
    ``(x0,y0,x1,y1)`` slice rectangle with exclusive right/bottom boundaries.
    The crop is clamped to the image and has at least 32 pixels per dimension
    whenever the image dimension itself permits that size.  An empty temporal
    selection conservatively returns the full image.
    """

    boxes = _require_box_tensor(boxes_cxcywh, "boxes_cxcywh")
    time_count = _require_temporal_length(boxes)
    height, width = _coerce_image_hw(image_hw)
    if not math.isfinite(float(expansion)) or expansion <= 0.0:
        raise ValueError("expansion must be finite and positive")
    mask = _temporal_mask(temporal_indices, time_count, boxes.device)
    if not bool(mask.any().item()):
        return 0, 0, width, height

    selected = boxes[mask].float()
    xyxy = _cxcywh_to_xyxy(selected).clamp(0.0, 1.0)
    minimum = xyxy[:, :2].amin(dim=0)
    maximum = xyxy[:, 2:].amax(dim=0)
    center = (minimum + maximum) * 0.5
    extent = (maximum - minimum) * float(expansion)
    pixel_center_x = float(center[0].item()) * width
    pixel_center_y = float(center[1].item()) * height
    pixel_extent_x = float(extent[0].item()) * width
    pixel_extent_y = float(extent[1].item()) * height
    x0, x1 = _integer_crop_interval(pixel_center_x, pixel_extent_x, width)
    y0, y1 = _integer_crop_interval(pixel_center_y, pixel_extent_y, height)
    return x0, y0, x1, y1


def _canonicalize_views(
    values: Mapping[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping keyed by original/global_high/crop")
    canonical: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for raw_name, value in values.items():
        name = _VIEW_ALIASES.get(raw_name, raw_name)
        if name not in _REQUIRED_VIEWS:
            unknown.append(str(raw_name))
            continue
        if name in canonical:
            raise ValueError(f"{label} contains duplicate aliases for {name!r}")
        canonical[name] = value
    missing = sorted(_REQUIRED_VIEWS - canonical.keys())
    if missing or unknown:
        raise ValueError(f"{label} must contain exactly the three required views; missing={missing}, unknown={unknown}")
    return canonical


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left_xyxy = _cxcywh_to_xyxy(left.float()).clamp(0.0, 1.0)
    right_xyxy = _cxcywh_to_xyxy(right.float()).clamp(0.0, 1.0)
    minimum = torch.maximum(left_xyxy[:2], right_xyxy[:2])
    maximum = torch.minimum(left_xyxy[2:], right_xyxy[2:])
    intersection = (maximum - minimum).clamp_min(0.0)
    intersection_area = float((intersection[0] * intersection[1]).item())
    left_size = (left_xyxy[2:] - left_xyxy[:2]).clamp_min(0.0)
    right_size = (right_xyxy[2:] - right_xyxy[:2]).clamp_min(0.0)
    left_area = float((left_size[0] * left_size[1]).item())
    right_area = float((right_size[0] * right_size[1]).item())
    union = left_area + right_area - intersection_area
    return 0.0 if union <= 0.0 else intersection_area / union


def _candidate_group(
    names: list[str],
    boxes: dict[str, torch.Tensor],
    iou_threshold: float,
) -> tuple[list[str], float, dict[str, dict[str, float]]]:
    pairwise: dict[str, dict[str, float]] = {name: {} for name in names}
    for left, right in itertools.combinations(names, 2):
        iou = _box_iou(boxes[left], boxes[right])
        pairwise[left][right] = iou
        pairwise[right][left] = iou

    best_group: list[str] = []
    best_agreement = -1.0
    best_priority: tuple[int, ...] | None = None
    priority = {name: index for index, name in enumerate(_VIEW_ORDER)}
    for size in range(len(names), 1, -1):
        for combination in itertools.combinations(names, size):
            pair_values = [pairwise[left][right] for left, right in itertools.combinations(combination, 2)]
            if not pair_values or any(value + 1e-12 < iou_threshold for value in pair_values):
                continue
            agreement = sum(pair_values) / len(pair_values)
            combination_priority = tuple(sorted(priority[name] for name in combination))
            if (
                len(combination) > len(best_group)
                or (len(combination) == len(best_group) and agreement > best_agreement + 1e-12)
                or (
                    len(combination) == len(best_group)
                    and abs(agreement - best_agreement) <= 1e-12
                    and (best_priority is None or combination_priority < best_priority)
                )
            ):
                best_group = list(combination)
                best_agreement = agreement
                best_priority = combination_priority
        if best_group:
            break
    return best_group, max(0.0, best_agreement), pairwise


def _medoid(group: list[str], pairwise: dict[str, dict[str, float]]) -> str:
    priority = {name: index for index, name in enumerate(_VIEW_ORDER)}
    scored = []
    for name in group:
        neighbors = [pairwise[name][other] for other in group if other != name]
        score = sum(neighbors) / len(neighbors) if neighbors else 0.0
        scored.append((score, -priority[name], name))
    return max(scored)[2]


def _json_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("diagnostic value became non-finite")
    return result


def build_geometric_teacher(
    box_views: dict[str, torch.Tensor],
    valid_views: dict[str, torch.Tensor],
    temporal_indices: Any,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a conservative per-frame geometry teacher from three views.

    The required view names are ``original``, ``global_high`` and ``crop``;
    ``origin`` and ``global`` are accepted as aliases.  At each frame an
    accepted group is a pairwise-IoU clique of at least two valid views.  The
    largest clique is selected, its IoU medoid becomes the teacher, and its
    mean pair IoU becomes the confidence weight after temporal masking.  When
    no clique exists, the teacher is the original-view box and the weight is
    zero.
    """

    if not math.isfinite(float(iou_threshold)) or not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0,1]")
    canonical_boxes = _canonicalize_views(box_views, "box_views")
    canonical_valid = _canonicalize_views(valid_views, "valid_views")
    original = _require_box_tensor(canonical_boxes["original"], "box_views[original]")
    time_count = _require_temporal_length(original)
    for name in _VIEW_ORDER:
        boxes = _require_box_tensor(canonical_boxes[name], f"box_views[{name}]")
        if boxes.shape != original.shape:
            raise ValueError(f"all box views must share shape [T,4]; {name} has {tuple(boxes.shape)}")
        if boxes.device != original.device:
            raise ValueError("all box views must be on the same device")
        valid = canonical_valid[name]
        if not torch.is_tensor(valid):
            raise TypeError(f"valid_views[{name}] must be a tensor")
        if valid.ndim != 1 or valid.shape[0] != time_count:
            raise ValueError(f"valid_views[{name}] must have shape [T]")
        if valid.is_floating_point() or valid.is_complex():
            if valid.is_complex() or not torch.isfinite(valid).all():
                raise ValueError(f"valid_views[{name}] must be finite")

    temporal = _temporal_mask(temporal_indices, time_count, original.device)
    valid_masks = {name: canonical_valid[name].to(device=original.device).bool() for name in _VIEW_ORDER}
    box_float = {name: canonical_boxes[name].float() for name in _VIEW_ORDER}
    teacher_frames: list[torch.Tensor] = []
    weights: list[float] = []
    frame_diagnostics: list[dict[str, Any]] = []

    for frame in range(time_count):
        names = [name for name in _VIEW_ORDER if bool(valid_masks[name][frame].item())]
        frame_boxes = {name: box_float[name][frame] for name in names}
        group, agreement, pairwise = _candidate_group(names, frame_boxes, float(iou_threshold))
        accepted = len(group) >= 2
        if accepted:
            medoid_view = _medoid(group, pairwise)
            teacher = frame_boxes[medoid_view]
            fallback = False
        else:
            medoid_view = "original"
            teacher = box_float["original"][frame]
            fallback = True
            agreement = 0.0
            group = []
        in_interval = bool(temporal[frame].item())
        weight = agreement if in_interval and accepted else 0.0
        teacher_frames.append(teacher.detach())
        weights.append(_json_float(weight))
        frame_diagnostics.append(
            {
                "frame": frame,
                "valid_views": names,
                "candidate_count": len(names),
                "accepted": accepted,
                "group_views": list(group),
                "group_size": len(group),
                "medoid_view": medoid_view,
                "agreement": _json_float(agreement),
                "temporal_in_interval": in_interval,
                "weight": _json_float(weight),
                "fallback_original": fallback,
                "pairwise_iou": {
                    left: {right: _json_float(value) for right, value in right_values.items()}
                    for left, right_values in pairwise.items()
                },
            }
        )

    # A teacher is one of the selected predictions, never an averaged or
    # round-tripped box.  Preserve its exact FP32 values so downstream audits
    # can identify the selected medoid/fallback source without quantization
    # noise.  IoU itself is safely clipped in _box_iou.
    teacher_boxes = torch.stack(teacher_frames, dim=0).to(dtype=torch.float32).detach()
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=original.device)
    if not torch.isfinite(teacher_boxes).all() or not torch.isfinite(weights_tensor).all():
        raise RuntimeError("geometric teacher produced non-finite outputs")
    diagnostics = {
        "view_order": list(_VIEW_ORDER),
        "iou_threshold": float(iou_threshold),
        "temporal_mask": [bool(value) for value in temporal.detach().cpu().tolist()],
        "frames": frame_diagnostics,
        "accepted_frame_count": sum(frame["accepted"] for frame in frame_diagnostics),
        "fallback_frame_count": sum(frame["fallback_original"] for frame in frame_diagnostics),
        "weight_sum": _json_float(float(weights_tensor.sum().item())),
        "teacher_contract": "per-frame IoU-clique medoid; fallback original with zero weight",
    }
    return teacher_boxes, weights_tensor, diagnostics


__all__ = ["build_geometric_teacher", "map_crop_boxes", "tube_crop_rect"]
