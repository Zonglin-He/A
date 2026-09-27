"""Small THW adapters for DESTA-3D feasibility work.

Inputs are merged visual tokens in explicit video-grid order. The module does
not change the backbone decoder, the number of physical frames, or frame times.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_HIDDEN_DIM = 256
DEFAULT_VISUAL_DIM = 2560
DEFAULT_SPATIAL_MERGE_SIZE = 2
ARCHITECTURES = ("early_factorized", "shared3d", "dual3d")


def _as_int_tuple(values: Sequence[int] | torch.Tensor, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    result = tuple(int(v) for v in values)
    if any(float(v) != int(v) for v in values):
        raise ValueError(f"{name} must contain integer dimensions")
    return result


def merged_video_grid_shape(
    video_grid_thw: Sequence[int] | torch.Tensor,
    *,
    spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
    token_count: int | None = None,
) -> tuple[int, int, int]:
    """Return the THW shape after spatial merge, validating the token count.

    ``video_grid_thw`` is the explicit processor grid supplied to the Qwen
    vision tower, before its spatial merger. Its T dimension already counts
    the temporal patch groups represented by the processor. In particular,
    this helper never divides T by the model's ``temporal_patch_size``: that
    value controls the input Conv3d patch kernel, while the processor's grid
    and the merger output count establish the actual output T.
    """
    grid = _as_int_tuple(video_grid_thw, "video_grid_thw")
    if len(grid) != 3:
        raise ValueError(f"expected one (T,H,W) video grid, got {grid}")
    t, h, w = grid
    merge = int(spatial_merge_size)
    if merge < 1 or t < 1 or h < 1 or w < 1:
        raise ValueError(f"grid and spatial merge must be positive, got {grid}, {merge}")
    if h % merge or w % merge:
        raise ValueError(
            f"patch grid H,W={h,w} cannot be spatially merged by {merge}; refusing a guessed reshape"
        )
    result = (t, h // merge, w // merge)
    expected = result[0] * result[1] * result[2]
    if token_count is not None and int(token_count) != expected:
        raise ValueError(
            f"merger emitted {token_count} video tokens, but grid {grid} and merge={merge} require {expected}"
        )
    return result


def reshape_merged_video_tokens(
    tokens: torch.Tensor,
    video_grid_thw: Sequence[int] | torch.Tensor,
    *,
    spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
) -> torch.Tensor:
    """Restore one merger output ``[N,C]`` to ``[1,T,Hm,Wm,C]``.

    The Qwen vision preprocessor groups each spatial merge neighborhood before
    the merger. After ``visual.merger`` this is already row-major on the
    merged H/W grid. Applying another patch shuffle here would corrupt
    coordinates, so this function only checks and reshapes.
    """
    if tokens.ndim != 2:
        raise ValueError(f"expected a single flat video token matrix [N,C], got {tuple(tokens.shape)}")
    t, hm, wm = merged_video_grid_shape(
        video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        token_count=tokens.shape[0],
    )
    return tokens.reshape(1, t, hm, wm, tokens.shape[-1])


def flatten_video_grid(grid: torch.Tensor) -> torch.Tensor:
    """Flatten canonical ``[B,T,H,W,C]`` video grids without reordering."""
    if grid.ndim != 5:
        raise ValueError(f"expected [B,T,H,W,C], got {tuple(grid.shape)}")
    return grid.reshape(-1, grid.shape[-1])


class R2Plus1DBlock(nn.Module):
    """Depthwise spatial then temporal convolution with pointwise mixing."""

    def __init__(self, channels: int, *, temporal_dilation: int = 1) -> None:
        super().__init__()
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.spatial_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(temporal_dilation, 0, 0),
            dilation=(temporal_dilation, 1, 1),
            groups=channels,
            bias=False,
        )
        self.temporal_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_pw(self.spatial_dw(x))
        x = x + F.silu(spatial)
        temporal = self.temporal_pw(self.temporal_dw(x))
        return x + F.silu(self.norm(temporal))


class _EarlyFactorizedReader(nn.Module):
    """2D spatial reader plus an XY-pooled long temporal reader baseline."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.spatial_pw = nn.Conv2d(channels, channels, 1)
        self.temporal_dw = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=False
        )
        self.temporal_pw = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is [B,C,T,H,W]. Spatial processing is independent per frame.
        b, c, t, h, w = x.shape
        frame_map = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        referent = frame_map + F.silu(self.spatial_pw(self.spatial_dw(frame_map)))
        referent = referent.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        pooled = x.mean(dim=(-1, -2))
        event = pooled + F.silu(self.temporal_pw(self.temporal_dw(pooled)))
        event = event[..., None, None].expand(b, c, t, h, w)
        return referent, event


class Desta3DAdapter(nn.Module):
    """Identity-initialized visual adapter and two query-conditioned readers.

    Args:
        in_channels: Width of PTD's merged visual token (2560 by default).
        query_dim: Width of the query context captured from the language-model
            prefill (2560 by default). The caller chooses/pools query token
            hidden states and passes a single vector per video.
        hidden_dim: Internal THW width (256 by default).
        architecture: ``early_factorized``, ``shared3d`` or ``dual3d``.

    ``updated_tokens`` has the same shape and width as the original merged
    visual tokens. Because ``out_proj`` starts at zero, this is bitwise the
    original token tensor at initialization. The stock Qwen/PTD language
    decoder and physical frame ordering remain caller-controlled and unchanged.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_VISUAL_DIM,
        query_dim: int = DEFAULT_VISUAL_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        architecture: str = "dual3d",
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        if min(in_channels, query_dim, hidden_dim) < 1:
            raise ValueError("all channel dimensions must be positive")
        self.in_channels = int(in_channels)
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.architecture = architecture

        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)
        if architecture in ("shared3d", "dual3d"):
            self.stem = nn.ModuleList((R2Plus1DBlock(self.hidden_dim),))
        self.query_proj = nn.Linear(self.query_dim, self.hidden_dim, bias=False)
        self.film = nn.Linear(self.query_dim, 2 * self.hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.referent_head = nn.Linear(self.hidden_dim, 1)
        self.event_head = nn.Linear(self.hidden_dim, 1)

        if architecture == "early_factorized":
            self.factorized_reader = _EarlyFactorizedReader(self.hidden_dim)
        elif architecture == "shared3d":
            self.shared_reader = R2Plus1DBlock(self.hidden_dim)
        else:
            self.short_reader = R2Plus1DBlock(self.hidden_dim, temporal_dilation=1)
            self.long_reader = R2Plus1DBlock(self.hidden_dim, temporal_dilation=2)

    def active_parameter_count(self) -> int:
        """Count real trainable parameters; the module has no padding weights."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _query_score(features: torch.Tensor, query: torch.Tensor, head: nn.Module) -> torch.Tensor:
        # Features are [B,C,T,H,W], query is [B,C].
        f = F.layer_norm(features.float().movedim(1, -1), (features.shape[1],))
        q = F.normalize(query.float(), dim=-1)
        query_score = (f * q[:, None, None, None, :]).sum(-1) * (features.shape[1] ** -0.5)
        return query_score + head(features.movedim(1, -1)).squeeze(-1).float()

    def forward(
        self,
        visual_grid: torch.Tensor,
        query_context: torch.Tensor | None = None,
        *,
        frame_times: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool | str | None]:
        if visual_grid.ndim != 5:
            raise ValueError(f"visual_grid must be canonical [B,T,H,W,C], got {tuple(visual_grid.shape)}")
        b, t, h, w, c = visual_grid.shape
        if c != self.in_channels or min(t, h, w) < 1:
            raise ValueError(
                f"visual grid must have [B,T,H,W,{self.in_channels}] with nonempty THW, got {tuple(visual_grid.shape)}"
            )
        if frame_times is not None and tuple(frame_times.shape) != (b, t):
            raise ValueError(f"frame_times must be [B,T]={b,t}; got {tuple(frame_times.shape)}")

        original = visual_grid
        x = self.input_proj(visual_grid).permute(0, 4, 1, 2, 3).contiguous()
        if self.architecture == "early_factorized":
            # This control deliberately avoids any THW-mixing stem: it is 2D
            # per-frame spatial processing plus an XY-pooled 1D temporal path.
            referent_features, event_features = self.factorized_reader(x)
        else:
            for block in self.stem:
                x = block(x)
            if self.architecture == "shared3d":
                shared = self.shared_reader(x)
                referent_features = event_features = shared
            else:
                referent_features = self.short_reader(x)
                event_features = self.long_reader(x)

        if query_context is None:
            q = x.new_zeros((b, self.hidden_dim))
            query_conditioned = False
        else:
            if query_context.ndim != 2 or query_context.shape != (b, self.query_dim):
                raise ValueError(
                    f"query_context must be [B,{self.query_dim}] matching video batch; got {tuple(query_context.shape)}"
                )
            query_context = query_context.to(dtype=x.dtype)
            q = self.query_proj(query_context)
            gamma, beta = self.film(query_context).chunk(2, dim=-1)
            query_conditioned = True

        referent_logits = self._query_score(referent_features, q, self.referent_head)
        event_logits = self._query_score(event_features, q, self.event_head)

        # Injection uses the same shared/dual evidence features, then applies
        # a query FiLM when the actual query context is available.
        injection = 0.5 * (referent_features + event_features)
        if query_conditioned:
            injection = injection * (1.0 + 0.1 * torch.tanh(gamma[:, :, None, None, None]))
            injection = injection + beta[:, :, None, None, None]
        delta = self.out_proj(injection.movedim(1, -1))
        updated = original + delta
        return {
            "updated_tokens": updated,
            "delta": delta,
            "referent_logits": referent_logits,
            "event_logits": event_logits,
            "query_conditioned": query_conditioned,
            "frame_times": frame_times,
            "architecture": self.architecture,
        }


def asymmetric_event_referent_loss(
    event_logits: torch.Tensor,
    referent_logits: torch.Tensor,
    *,
    observed_time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Positive implication penalty for ``event evidence <= referent``.

    The loss compares model evidence fields only; it does not consume box GT,
    use missing annotations as negative labels, or mark event-external regions
    as absent. ``observed_time_mask`` may exclude padded/unobserved frame slots
    only, never frames merely lacking a GT box.
    """
    if event_logits.shape != referent_logits.shape or event_logits.ndim != 4:
        raise ValueError(
            f"both logits must have matching [B,T,H,W] shapes, got {tuple(event_logits.shape)} and {tuple(referent_logits.shape)}"
        )
    event_support = torch.sigmoid(event_logits.float()).amax(dim=(-1, -2))
    referent_support = torch.sigmoid(referent_logits.float()).amax(dim=(-1, -2))
    violation = F.relu(event_support - referent_support).square()
    if observed_time_mask is not None:
        if observed_time_mask.shape != event_support.shape:
            raise ValueError(
                f"observed_time_mask must be [B,T]={tuple(event_support.shape)}, got {tuple(observed_time_mask.shape)}"
            )
        mask = observed_time_mask.to(device=violation.device, dtype=violation.dtype)
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("observed_time_mask must be binary")
        return (violation * mask).sum() / mask.sum().clamp_min(1.0)
    return violation.mean()


def capacity_matched_configs(
    *,
    in_channels: int = DEFAULT_VISUAL_DIM,
    query_dim: int = DEFAULT_VISUAL_DIM,
    target_parameters: int,
    modes: Iterable[str] = ARCHITECTURES,
    widths: Iterable[int] = range(16, 513, 8),
) -> dict[str, dict[str, int]]:
    """Find each mode's nearest real-parameter width for a requested budget.

    Counts come from instantiated trainable modules; there are no unused
    padding parameters. Returned configurations can be rebuilt directly with
    ``Desta3DAdapter(..., hidden_dim=..., architecture=...)``.
    """
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    candidates = tuple(sorted({int(w) for w in widths if int(w) > 0}))
    if not candidates:
        raise ValueError("widths must contain a positive candidate")
    result: dict[str, dict[str, int]] = {}
    for mode in modes:
        if mode not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {mode!r}")
        best: tuple[int, int, int] | None = None
        for width in candidates:
            module = Desta3DAdapter(
                in_channels=in_channels,
                query_dim=query_dim,
                hidden_dim=width,
                architecture=mode,
            )
            count = module.active_parameter_count()
            key = (abs(count - int(target_parameters)), width, count)
            if best is None or key < best:
                best = key
            del module
        assert best is not None
        _, width, count = best
        result[mode] = {"hidden_dim": width, "active_parameters": count}
    return result
