"""GT-free E5 TTA math helpers for the locked small DESTA-3D adapter.

This module does not edit ``Desta3DAdapter.forward`` or the PTD backbone. The
wrapper temporarily hooks the real reader modules during one ordinary forward
and always removes those hooks. It is an engineering kernel, not a TTA result
or a locked set of hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .desta3d_v1 import Desta3DAdapter, asymmetric_event_referent_loss


BRANCHES = ("referent", "event")
MODES = ("spatial_only", "event_only", "dual_nojoint", "full")
SUGGESTED_UPDATE_GROUPS = ("input_proj", "stem", "readers")
# Numerical floor only; it is not a feature-normalization or TTA-strength knob.
VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class TTAWeights:
    """Explicit caller-supplied weights; no experimental values are locked here."""

    feature_alignment: float
    latent_consistency: float
    referent_consistency: float
    event_consistency: float
    asymmetric_joint: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} weight must be finite and nonnegative, got {value}")


def _channels_last(x: Tensor) -> Tensor:
    if x.ndim != 5:
        raise ValueError(f"reader field must be [B,C,T,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _install_capture_hooks(adapter: Desta3DAdapter) -> tuple[dict[str, Any], list[Any]]:
    """Hook the real latent/reader calls made by the adapter's existing forward."""
    captured: dict[str, Any] = {}
    handles = []

    def save(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[name] = output
        return hook

    # The pre-reader latent is after the common THW stem for shared/dual modes.
    # The early-factorized control has no THW stem, so its common projected grid
    # is the closest corresponding field.
    try:
        if adapter.architecture == "early_factorized":
            handles.append(adapter.input_proj.register_forward_hook(save("latent_channels_last")))
            handles.append(adapter.factorized_reader.register_forward_hook(save("reader_pair")))
        else:
            if len(adapter.stem) < 1:
                raise RuntimeError("expected the locked shared/dual adapter to have a THW stem")
            handles.append(adapter.stem[-1].register_forward_hook(save("latent_channels_first")))
            if adapter.architecture == "shared3d":
                handles.append(adapter.shared_reader.register_forward_hook(save("shared_reader")))
            elif adapter.architecture == "dual3d":
                handles.append(adapter.short_reader.register_forward_hook(save("referent_reader")))
                handles.append(adapter.long_reader.register_forward_hook(save("event_reader")))
            else:
                raise ValueError(f"unsupported adapter architecture {adapter.architecture!r}")
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        raise
    return captured, handles


def forward_with_features(
    adapter: Desta3DAdapter,
    visual_grid: Tensor,
    query_context: Tensor | None = None,
    *,
    frame_times: Tensor | None = None,
) -> dict[str, Tensor | bool | str | None]:
    """Run the exact adapter forward while returning its actual latent/readers.

    Inputs use the existing ``[B,T,H,W,C]`` visual-grid contract. Captured
    feature fields use the same channel-last contract; logits remain
    ``[B,T,H,W]``. No second or copied forward is used, so returned reader
    tensors are exactly the tensors consumed by the adapter's scalar heads.
    """
    captured, handles = _install_capture_hooks(adapter)
    result: dict[str, Any] | None = None
    try:
        # E5 adapts the small module, not the frozen visual encoder/prefill.
        # Detach upstream tensors while preserving gradients into adapter weights.
        frozen_visual = visual_grid.detach()
        frozen_query = None if query_context is None else query_context.detach()
        frozen_times = None if frame_times is None else frame_times.detach()
        result = adapter(frozen_visual, frozen_query, frame_times=frozen_times)
    finally:
        for handle in reversed(handles):
            handle.remove()

    if result is None:
        raise RuntimeError("adapter forward did not return")
    if adapter.architecture == "early_factorized":
        latent = captured.get("latent_channels_last")
        pair = captured.get("reader_pair")
        if not isinstance(latent, Tensor) or not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise RuntimeError("early_factorized reader hooks did not capture expected tensors")
        referent, event = (_channels_last(pair[0]), _channels_last(pair[1]))
    elif adapter.architecture == "shared3d":
        latent_cf = captured.get("latent_channels_first")
        shared = captured.get("shared_reader")
        if not isinstance(latent_cf, Tensor) or not isinstance(shared, Tensor):
            raise RuntimeError("shared3d hooks did not capture expected tensors")
        latent = _channels_last(latent_cf)
        referent = event = _channels_last(shared)
    else:
        latent_cf = captured.get("latent_channels_first")
        referent_cf, event_cf = captured.get("referent_reader"), captured.get("event_reader")
        if not all(isinstance(x, Tensor) for x in (latent_cf, referent_cf, event_cf)):
            raise RuntimeError("dual3d hooks did not capture expected tensors")
        latent, referent, event = _channels_last(latent_cf), _channels_last(referent_cf), _channels_last(event_cf)

    expected = tuple(visual_grid.shape[:4])
    if tuple(latent.shape[:4]) != expected or tuple(referent.shape[:4]) != expected or tuple(event.shape[:4]) != expected:
        raise ValueError("captured feature fields do not preserve the input physical THW grid")
    return {
        "updated_tokens": result["updated_tokens"],
        "delta": result["delta"],
        "referent_logits": result["referent_logits"],
        "event_logits": result["event_logits"],
        "frame_times": result["frame_times"],
        "latent": latent,
        "referent_features": referent,
        "event_features": event,
        "query_conditioned": bool(result["query_conditioned"]),
        "architecture": str(result["architecture"]),
    }


def _validate_grid_mask(mask: Tensor | None, shape: Sequence[int], *, name: str) -> Tensor | None:
    b, t, h, w = (int(v) for v in shape)
    if mask is None:
        return None
    if tuple(mask.shape) != (b, t, h, w):
        raise ValueError(f"{name} must exactly match [B,T,H,W]={(b, t, h, w)}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must be a boolean/binary valid-grid mask")
        mask = mask.bool()
    if not torch.any(mask):
        raise ValueError(f"{name} contains no observed THW cells")
    return mask


def validate_aligned_views(
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    *,
    grid_mask_a: Tensor | None = None,
    grid_mask_b: Tensor | None = None,
) -> Tensor | None:
    """Reject any silent interpolation, physical-grid shift, or mask mismatch."""
    latent_a, latent_b = view_a.get("latent"), view_b.get("latent")
    if not isinstance(latent_a, Tensor) or not isinstance(latent_b, Tensor) or latent_a.ndim != 5 or latent_b.ndim != 5:
        raise ValueError("both views must expose channel-last latent [B,T,H,W,C] fields")
    if tuple(latent_a.shape) != tuple(latent_b.shape):
        raise ValueError(f"same-grid views must have identical latent shapes, got {tuple(latent_a.shape)} and {tuple(latent_b.shape)}")
    grid_shape = tuple(latent_a.shape[:4])
    frame_times_a, frame_times_b = view_a.get("frame_times"), view_b.get("frame_times")
    if not isinstance(frame_times_a, Tensor) or not isinstance(frame_times_b, Tensor):
        raise ValueError("same-grid views must carry explicit physical frame_times")
    if tuple(frame_times_a.shape) != (grid_shape[0], grid_shape[1]) or tuple(frame_times_b.shape) != tuple(frame_times_a.shape):
        raise ValueError("physical frame_times must match the shared [B,T] grid")
    if not torch.isfinite(frame_times_a).all() or not torch.isfinite(frame_times_b).all():
        raise ValueError("physical frame_times must be finite")
    if not torch.equal(frame_times_a.detach().cpu(), frame_times_b.detach().cpu()):
        raise ValueError("same-grid views refer to different physical times; refusing implicit alignment")
    if grid_mask_a is None and grid_mask_b is not None or grid_mask_a is not None and grid_mask_b is None:
        raise ValueError("same-grid views must use either no mask or an identical mask on both views")
    mask_a = _validate_grid_mask(grid_mask_a, grid_shape, name="grid_mask_a")
    mask_b = _validate_grid_mask(grid_mask_b, grid_shape, name="grid_mask_b")
    if mask_a is not None and not torch.equal(mask_a.detach().cpu(), mask_b.detach().cpu()):
        raise ValueError("same-grid views have different validity masks; refusing implicit alignment")
    for key in ("referent_features", "event_features"):
        a, b = view_a.get(key), view_b.get(key)
        if (not isinstance(a, Tensor) or not isinstance(b, Tensor) or tuple(a.shape) != tuple(b.shape)
                or tuple(a.shape[:4]) != grid_shape):
            raise ValueError(f"same-grid views require identical {key} tensor shapes")
    for key in ("referent_logits", "event_logits"):
        a, b = view_a.get(key), view_b.get(key)
        if not isinstance(a, Tensor) or not isinstance(b, Tensor) or tuple(a.shape) != grid_shape:
            raise ValueError(f"{key} must match the same physical grid {grid_shape}")
        if tuple(b.shape) != grid_shape:
            raise ValueError(f"view B {key} does not match the same physical grid {grid_shape}")
    return mask_a


def _expand_mask(mask: Tensor | None, ref: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if ref.ndim == 5:  # [B,T,H,W,C]
        return mask.to(device=ref.device)[..., None].expand_as(ref)
    if ref.ndim == 4:  # [B,T,H,W]
        return mask.to(device=ref.device)
    raise ValueError(f"expected a 4D logit or 5D feature grid, got {tuple(ref.shape)}")


def masked_mean(x: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        if x.numel() == 0:
            raise ValueError("cannot average an empty tensor")
        return x.mean()
    expanded = _expand_mask(mask, x)
    assert expanded is not None
    count = expanded.sum()
    if int(count) < 1:
        raise ValueError("valid-grid mask leaves no values to average")
    return x.masked_select(expanded).mean()


def latent_consistency_loss(latent_a: Tensor, latent_b: Tensor, mask: Tensor | None = None) -> Tensor:
    if tuple(latent_a.shape) != tuple(latent_b.shape) or latent_a.ndim != 5:
        raise ValueError("latent consistency requires same-shaped channel-last [B,T,H,W,C] fields")
    return masked_mean((latent_a.float() - latent_b.float()).abs(), mask)


def bernoulli_symmetric_stopgrad_kl(logits_a: Tensor, logits_b: Tensor, mask: Tensor | None = None,
                                    *, eps: float = 1e-6) -> Tensor:
    """Bidirectional teacher/student Bernoulli KL with both views trainable.

    Each half detaches only its teacher distribution. Thus view A receives a
    student gradient from the second half and view B from the first; neither
    side is accidentally detached from the complete consistency objective.
    """
    if tuple(logits_a.shape) != tuple(logits_b.shape) or logits_a.ndim != 4:
        raise ValueError("Bernoulli consistency requires same-shaped [B,T,H,W] logits")
    p = torch.sigmoid(logits_a.float()).clamp(eps, 1 - eps)
    q = torch.sigmoid(logits_b.float()).clamp(eps, 1 - eps)

    def kl(p_teacher: Tensor, q_student: Tensor) -> Tensor:
        return p_teacher * (p_teacher.log() - q_student.log()) + (1 - p_teacher) * (
            torch.log1p(-p_teacher) - torch.log1p(-q_student)
        )

    forward = masked_mean(kl(p.detach(), q), mask)
    backward = masked_mean(kl(q.detach(), p), mask)
    return 0.5 * (forward + backward)


def _moments_per_batch(feature: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if feature.ndim != 5:
        raise ValueError("branch feature must be channel-last [B,T,H,W,C]")
    x = feature.float()
    b, t, h, w, c = x.shape
    if min(b, t, h, w, c) < 1 or not torch.isfinite(x).all():
        raise ValueError("branch feature grid must be nonempty and finite")
    if mask is None:
        m = torch.ones((b, t, h, w), device=x.device, dtype=torch.bool)
    else:
        m = _validate_grid_mask(mask, (b, t, h, w), name="feature grid mask").to(x.device)
    counts = m.sum(dim=(1, 2, 3)).to(x.dtype)
    if torch.any(counts == 0):
        raise ValueError("every batch item must contain at least one valid THW cell")
    m_f = m.to(x.dtype)[..., None]
    mean = (x * m_f).sum(dim=(1, 2, 3)) / counts[:, None]
    second = (x.square() * m_f).sum(dim=(1, 2, 3)) / counts[:, None]
    return mean, second


class SourceFeatureStatsAccumulator:
    """CPU-only equal-query/equal-parent source-train channel-moment accumulator.

    Each ``add`` is exactly one source query. Per-query THW moments are stored,
    so long videos/grid resolutions never gain more source weight. The final
    total variance is reconstructed from E[x] and E[x^2], including the
    between-query and between-parent variation.
    """

    def __init__(self, *, source_split: str) -> None:
        if source_split != "train":
            raise ValueError("source TTA moments accept source train only; validation/target splits are refused")
        self.source_split = source_split
        self._rows: dict[str, dict[str, tuple[Tensor, Tensor]]] = {branch: {} for branch in BRANCHES}
        self._parents_by_query: dict[str, str] = {}

    def add(self, *, parent_id: str, query_id: str, branch_features: Mapping[str, Tensor],
            valid_grid_mask: Tensor | None = None) -> None:
        parent_id, query_id = str(parent_id), str(query_id)
        if not parent_id or not query_id:
            raise ValueError("parent_id and query_id must be nonempty")
        if set(branch_features) != set(BRANCHES):
            raise ValueError(f"branch_features must contain exactly {BRANCHES}")
        if query_id in self._parents_by_query:
            raise ValueError(f"duplicate source query in statistics accumulator: {query_id}")
        widths = set()
        values: dict[str, tuple[Tensor, Tensor]] = {}
        for branch in BRANCHES:
            feature = branch_features[branch]
            if not isinstance(feature, Tensor) or feature.ndim != 5 or feature.shape[0] != 1:
                raise ValueError(f"{branch} feature must be one query [1,T,H,W,C]")
            mean, second = _moments_per_batch(feature.detach(), valid_grid_mask)
            values[branch] = (mean[0].detach().cpu().double(), second[0].detach().cpu().double())
            widths.add(int(feature.shape[-1]))
        if len(widths) != 1:
            raise ValueError("referent/event feature channel widths must match")
        self._parents_by_query[query_id] = parent_id
        for branch, moments in values.items():
            self._rows[branch][query_id] = moments

    @staticmethod
    def _combine(rows: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        if not rows:
            raise ValueError("cannot aggregate an empty source parent")
        return torch.stack([x[0] for x in rows]).mean(0), torch.stack([x[1] for x in rows]).mean(0)

    def finalize(self) -> dict[str, Any]:
        if not self._parents_by_query:
            raise ValueError("no source train queries were accumulated")
        parents: dict[str, dict[str, dict[str, Tensor | int]]] = {}
        parent_ids = sorted(set(self._parents_by_query.values()))
        for branch in BRANCHES:
            parent_rows: dict[str, tuple[Tensor, Tensor, int]] = {}
            for parent in parent_ids:
                qids = sorted(q for q, p in self._parents_by_query.items() if p == parent)
                moments = [self._rows[branch][q] for q in qids]
                mu, second = self._combine(moments)
                parent_rows[parent] = (mu, second, len(qids))
            mu = torch.stack([parent_rows[p][0] for p in parent_ids]).mean(0)
            second = torch.stack([parent_rows[p][1] for p in parent_ids]).mean(0)
            variance = (second - mu.square()).clamp_min(0)
            parents[branch] = {
                parent: {"mean": parent_rows[parent][0].clone(), "second_moment": parent_rows[parent][1].clone(),
                         "query_count": parent_rows[parent][2]}
                for parent in parent_ids
            }
            parents[branch]["__global__"] = {
                "mean": mu, "second_moment": second, "variance": variance, "std": variance.sqrt(),
                "parent_count": len(parent_ids), "query_count": len(self._parents_by_query),
            }
        return {"source_split": "train", "gt_read": False, "aggregation": "equal query within parent, then equal parent; total variance from E[x^2]-E[x]^2",
                "parents": parents}

    def save_json(self, path: str | Path) -> None:
        """Save CPU-readable source statistics once; never silently overwrite."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        payload = self.finalize()

        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(convert(payload), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)


def feature_alignment_loss(feature: Tensor, source_moments: Mapping[str, Any],
                           mask: Tensor | None = None) -> Tensor:
    """Raw-unit mean-channel L1 alignment of target E[x], std to source values."""
    mean, second = _moments_per_batch(feature, mask)
    current_mean = mean.mean(0)
    current_var = (second.mean(0) - current_mean.square()).clamp_min(VARIANCE_FLOOR)
    current_std = current_var.sqrt()
    source_mean = torch.as_tensor(source_moments["mean"], device=feature.device, dtype=torch.float32).detach()
    source_std = torch.as_tensor(source_moments["std"], device=feature.device, dtype=torch.float32).detach()
    if source_mean.shape != current_mean.shape or source_std.shape != current_std.shape:
        raise ValueError("source and target branch moment channel dimensions do not match")
    return (current_mean - source_mean).abs().mean() + (current_std - source_std).abs().mean()


def _component_branch(name: str, mode: str) -> bool:
    if mode == "spatial_only":
        return name == "referent"
    if mode == "event_only":
        return name == "event"
    return mode in ("dual_nojoint", "full")


def masked_asymmetric_event_referent_loss(event_logits: Tensor, referent_logits: Tensor,
                                          grid_mask: Tensor | None = None) -> Tensor:
    """Prediction-only event<=referent penalty with exact optional THW masking."""
    if tuple(event_logits.shape) != tuple(referent_logits.shape) or event_logits.ndim != 4:
        raise ValueError("joint constraint requires matching [B,T,H,W] event/referent logits")
    if grid_mask is None:
        return asymmetric_event_referent_loss(event_logits, referent_logits)
    mask = _validate_grid_mask(grid_mask, event_logits.shape, name="joint grid mask").to(event_logits.device)
    event_p = torch.sigmoid(event_logits.float()).masked_fill(~mask, -torch.inf)
    ref_p = torch.sigmoid(referent_logits.float()).masked_fill(~mask, -torch.inf)
    valid_time = mask.any(dim=(-1, -2))
    event_support = event_p.amax(dim=(-1, -2))[valid_time]
    referent_support = ref_p.amax(dim=(-1, -2))[valid_time]
    if event_support.numel() == 0:
        raise ValueError("joint constraint has no observed time slots")
    return (event_support - referent_support).clamp_min(0).square().mean()


def compute_tta_losses(
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    source_stats: Mapping[str, Any],
    *,
    mode: str,
    weights: TTAWeights,
    grid_mask_a: Tensor | None = None,
    grid_mask_b: Tensor | None = None,
) -> dict[str, Any]:
    """Compute E5 loss terms from two same-physical-grid feature forwards.

    No labels, boxes, class IDs, or missing-label masks are accepted. A valid
    grid mask can exclude only explicit padding/unobserved cells and must be
    byte-identical for both views.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if source_stats.get("source_split") != "train" or source_stats.get("gt_read") is not False:
        raise ValueError("TTA alignment requires detached source-train statistics with gt_read=false")
    mask = validate_aligned_views(view_a, view_b, grid_mask_a=grid_mask_a, grid_mask_b=grid_mask_b)
    stats_root = source_stats.get("parents", source_stats)
    stats_by_branch = {}
    for branch in BRANCHES:
        try:
            stats_by_branch[branch] = stats_root[branch]["__global__"]
        except (KeyError, TypeError):
            if "mean" in source_stats and "std" in source_stats:
                stats_by_branch[branch] = source_stats
            else:
                raise ValueError(f"source moments missing branch {branch}")

    components: dict[str, Tensor] = {}
    components["latent_consistency"] = latent_consistency_loss(view_a["latent"], view_b["latent"], mask)
    components["feature_alignment"] = components["latent_consistency"] * 0
    components["referent_consistency"] = components["latent_consistency"] * 0
    components["event_consistency"] = components["latent_consistency"] * 0
    components["asymmetric_joint"] = components["latent_consistency"] * 0
    for branch, feature_key, logits_key, consistency_key in (
        ("referent", "referent_features", "referent_logits", "referent_consistency"),
        ("event", "event_features", "event_logits", "event_consistency"),
    ):
        if not _component_branch(branch, mode):
            continue
        align_a = feature_alignment_loss(view_a[feature_key], stats_by_branch[branch], mask)
        align_b = feature_alignment_loss(view_b[feature_key], stats_by_branch[branch], mask)
        components["feature_alignment"] = components["feature_alignment"] + 0.5 * (align_a + align_b)
        components[consistency_key] = bernoulli_symmetric_stopgrad_kl(
            view_a[logits_key], view_b[logits_key], mask
        )
    if mode == "full":
        ja = masked_asymmetric_event_referent_loss(view_a["event_logits"], view_a["referent_logits"], mask)
        jb = masked_asymmetric_event_referent_loss(view_b["event_logits"], view_b["referent_logits"], mask)
        components["asymmetric_joint"] = 0.5 * (ja + jb)
    active_weights = {
        "latent_consistency": weights.latent_consistency,
        "feature_alignment": weights.feature_alignment,
        "referent_consistency": weights.referent_consistency if _component_branch("referent", mode) else 0.0,
        "event_consistency": weights.event_consistency if _component_branch("event", mode) else 0.0,
        "asymmetric_joint": weights.asymmetric_joint if mode == "full" else 0.0,
    }
    total = sum(components[name] * active_weights[name] for name in components)
    return {"mode": mode, "total": total, "components": components, "weights": active_weights,
            "valid_cells": int(mask.sum()) if mask is not None else int(view_a["latent"].shape[0] * math.prod(view_a["latent"].shape[1:4]))}


def adapter_parameter_groups(adapter: Desta3DAdapter) -> dict[str, tuple[nn.Parameter, ...]]:
    """Expose only adapter-local groups; do not mutate requires_grad flags."""
    groups: dict[str, tuple[nn.Parameter, ...]] = {"input_proj": tuple(adapter.input_proj.parameters())}
    groups["stem"] = tuple(adapter.stem.parameters()) if hasattr(adapter, "stem") else ()
    readers = []
    for attr in ("factorized_reader", "shared_reader", "short_reader", "long_reader"):
        module = getattr(adapter, attr, None)
        if module is not None:
            readers.extend(module.parameters())
    groups["readers"] = tuple(readers)
    groups["query_proj"] = tuple(adapter.query_proj.parameters())
    groups["film"] = tuple(adapter.film.parameters())
    groups["out_proj"] = tuple(adapter.out_proj.parameters())
    groups["scalar_heads"] = tuple(adapter.referent_head.parameters()) + tuple(adapter.event_head.parameters())
    return groups


def candidate_update_parameters(adapter: Desta3DAdapter,
                                group_names: Sequence[str] = SUGGESTED_UPDATE_GROUPS) -> tuple[nn.Parameter, ...]:
    """Return candidate adapter parameters for a later runner; leaves flags intact."""
    groups = adapter_parameter_groups(adapter)
    unknown = set(group_names) - set(groups)
    if unknown:
        raise ValueError(f"unknown adapter parameter groups: {sorted(unknown)}")
    params = tuple(p for name in group_names for p in groups[name] if p.requires_grad)
    if not params:
        raise ValueError("selected adapter parameter groups contain no trainable parameters")
    return params
