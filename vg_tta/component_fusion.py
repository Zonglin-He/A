"""Small, deterministic component fusion for spatial view ablations.

This helper intentionally contains no model call, optimizer, clipping, or
confidence heuristic.  With no validity masks it is a direct coordinate-wise
fusion of the supplied raw view tensors, preserving their dtype/device and
therefore the existing three-view median path exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


_ALLOWED_VIEWS = frozenset(("original", "global_high", "crop"))


def _validate_view_names(views: Mapping[str, torch.Tensor], names: Sequence[str]) -> list[str]:
    if not isinstance(views, Mapping):
        raise TypeError("views must be a mapping from original/global_high/crop to tensors")
    if isinstance(names, (str, bytes)):
        raise TypeError("names must be a non-empty sequence of view names")
    try:
        selected = list(names)
    except TypeError as exc:
        raise TypeError("names must be a non-empty sequence of view names") from exc
    if not selected:
        raise ValueError("names must contain at least one view")
    if any(not isinstance(name, str) for name in selected):
        raise TypeError("names must contain only strings")
    if len(set(selected)) != len(selected):
        raise ValueError("names must not contain duplicate views")
    unknown = sorted(set(selected) - _ALLOWED_VIEWS)
    missing = sorted(set(selected) - set(views))
    if unknown or missing:
        raise ValueError(f"invalid or missing view names: unknown={unknown}, missing={missing}")
    return selected


def _validate_boxes(
    views: Mapping[str, torch.Tensor], names: Sequence[str]
) -> tuple[list[str], dict[str, torch.Tensor], tuple[int, int], torch.dtype, torch.device]:
    selected = _validate_view_names(views, names)
    checked: dict[str, torch.Tensor] = {}
    shape: tuple[int, int] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for name in selected:
        value = views[name]
        if not torch.is_tensor(value):
            raise TypeError(f"views[{name!r}] must be a tensor")
        if value.ndim != 2 or value.shape[-1] != 4:
            raise ValueError(
                f"views[{name!r}] must have shape [T,4], got {tuple(value.shape)}"
            )
        if value.shape[0] < 1:
            raise ValueError("view tensors must contain at least one frame")
        if not value.is_floating_point():
            raise TypeError(f"views[{name!r}] must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"views[{name!r}] must be finite")
        if bool((value[:, 2:] <= 0).any()):
            raise ValueError(f"views[{name!r}] must have positive width and height")
        current_shape = (int(value.shape[0]), int(value.shape[1]))
        if shape is None:
            shape = current_shape
            dtype = value.dtype
            device = value.device
        elif current_shape != shape:
            raise ValueError("all selected views must have the same [T,4] shape")
        elif value.dtype != dtype:
            raise ValueError("all selected views must have the same dtype")
        elif value.device != device:
            raise ValueError("all selected views must be on the same device")
        checked[name] = value
    assert shape is not None and dtype is not None and device is not None
    return selected, checked, shape, dtype, device


def _validate_validity(
    validity: Mapping[str, torch.Tensor] | None,
    selected: Sequence[str],
    views: Mapping[str, torch.Tensor],
    time_count: int,
) -> dict[str, torch.Tensor] | None:
    if validity is None:
        return None
    if not isinstance(validity, Mapping):
        raise TypeError("validity must be a mapping from view name to [T] masks")
    if "original" not in views:
        raise ValueError("views must include original when validity masks are supplied")
    missing = sorted(set(selected) - set(validity))
    if missing:
        raise ValueError(f"validity is missing selected views: {missing}")
    checked: dict[str, torch.Tensor] = {}
    for name in selected:
        mask = validity[name]
        if not torch.is_tensor(mask):
            raise TypeError(f"validity[{name!r}] must be a tensor")
        if tuple(mask.shape) != (time_count,):
            raise ValueError(
                f"validity[{name!r}] must have shape [T], got {tuple(mask.shape)}"
            )
        if mask.is_floating_point() and not torch.isfinite(mask).all():
            raise ValueError(f"validity[{name!r}] must be finite")
        checked[name] = mask.detach().bool()
    return checked


def _reduce_rows(rows: list[torch.Tensor], reduction: str) -> torch.Tensor:
    if not rows:
        raise ValueError("cannot reduce an empty set of view rows")
    if len(rows) == 1:
        return rows[0].clone()
    stacked = torch.stack(rows, dim=0)
    if reduction == "mean":
        return stacked.mean(dim=0)
    if reduction != "median":
        raise ValueError("reduction must be 'median' or 'mean'")
    ordered = torch.sort(stacked, dim=0).values
    middle = len(rows) // 2
    if len(rows) % 2:
        return ordered[middle]
    # torch.median returns the lower middle for even counts.  The component
    # ablation defines the even median as the arithmetic mean of both middles.
    return (ordered[middle - 1] + ordered[middle]) / 2


def fuse_views(
    views: Mapping[str, torch.Tensor],
    names: list[str],
    reduction: str = "median",
    validity: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fuse normalized ``cxcywh`` spatial views without modifying inputs.

    ``names`` controls which of the named views participate and may contain a
    single view.  Without ``validity`` every selected row is reduced directly
    from the raw tensors; no coordinate clipping or additional gating occurs.
    With validity masks, each frame reduces only valid selected views and a
    frame with no valid selected view falls back to ``original``.  Width and
    height are checked positive, while coordinates are deliberately left
    untouched even when they lie outside ``[0,1]``.
    """

    if reduction not in ("median", "mean"):
        raise ValueError("reduction must be 'median' or 'mean'")
    selected, checked, (time_count, _), _dtype, _device = _validate_boxes(views, names)
    masks = _validate_validity(validity, selected, views, time_count)

    if masks is None:
        # Keep the no-mask path a direct stack/reduction over raw view values.
        # This is bit-exact with the existing odd-view torch.median path.
        output = _reduce_rows([checked[name] for name in selected], reduction)
        return output.detach()

    rows: list[torch.Tensor] = []
    original = views["original"]
    for time_index in range(time_count):
        valid_rows = [
            checked[name][time_index]
            for name in selected
            if bool(masks[name][time_index].item())
        ]
        if not valid_rows:
            rows.append(original[time_index].clone())
        else:
            rows.append(_reduce_rows(valid_rows, reduction))
    return torch.stack(rows, dim=0).detach()


__all__ = ["fuse_views"]
