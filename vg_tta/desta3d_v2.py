"""Query-conditioned dual-reader adapter for DESTA-3D v2.

The caller supplies the already merged video grid as ``[B,T,H,W,C]``, the
language-model hidden states for the query, a mask that marks actual caption
tokens, and physical frame times.  This module does not infer caption length,
reshape PTD tokens, run the PTD decoder, or consume box annotations.

All architecture variants first apply the same lightweight THW stem and
channel-only LayerNorm. Each query is then pooled independently for the
spatial and event routes and FiLM-applied before its reader. Spatial features
use a local short 3D reader. Optional P1 event features concatenate the
appearance latent with its physical-time derivative before a small projection
and parallel temporal dilations. The two feature maps and residual
projections remain separate throughout.

``p1_enabled`` selects the optional physical-time derivative plus parallel
dilation-1/2/4 event path. It defaults to disabled for the P0 control; the
enabled path is a configurable P1 candidate, not an established requirement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_VISUAL_DIM = 2560
DEFAULT_QUERY_DIM = 2560
DEFAULT_HIDDEN_DIM = 128
ARCHITECTURES = ("early_factorized", "shared3d", "dual3d")
TRAIN_STAGES = ("A evidence", "B integration", "tta", "frozen")


class QueryAttentionPool(nn.Module):
    """Learned masked attention pool from query token states to one vector."""

    def __init__(self, query_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.pool_query = nn.Parameter(torch.empty(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.pool_query, mean=0.0, std=hidden_dim**-0.5)

    def forward(
        self, query_tokens: torch.Tensor, query_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.ndim != 3:
            raise ValueError(
                f"query_tokens must be [B,L,Q], got {tuple(query_tokens.shape)}"
            )
        if query_mask.shape != query_tokens.shape[:2]:
            raise ValueError(
                f"query_mask must be [B,L]={tuple(query_tokens.shape[:2])}, "
                f"got {tuple(query_mask.shape)}"
            )
        if query_mask.dtype != torch.bool and torch.any(
            (query_mask != 0) & (query_mask != 1)
        ):
            raise ValueError("query_mask must be binary")
        valid = query_mask.to(device=query_tokens.device, dtype=torch.bool)
        if torch.any(valid.sum(dim=1) == 0):
            raise ValueError("every query must contain at least one caption token")

        keys = self.key_proj(query_tokens)
        scores = (keys * self.pool_query.to(dtype=keys.dtype)).sum(dim=-1)
        scores = scores.float() * (keys.shape[-1] ** -0.5)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)

        values = self.value_proj(query_tokens)
        pooled = torch.einsum("bl,bld->bd", alpha.to(values.dtype), values)
        return self.output_proj(pooled), alpha


class QueryFiLM(nn.Module):
    """Small, initially nonzero query conditioning applied before a reader."""

    def __init__(self, hidden_dim: int, init_std: float = 1e-3) -> None:
        super().__init__()
        self.affine = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.normal_(self.affine.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(query).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        return x * (1.0 + gamma[:, None, None, None, :]) + beta[:, None, None, None, :]


class ChannelOnlyLayerNorm(nn.Module):
    """LayerNorm over channels independently at every THW location.

    Input and output use ``[B,C,T,H,W]``.  The normalization never reduces
    across time or space; affine parameters are just one scale and offset per
    channel.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
        return self.norm(x.movedim(1, -1)).movedim(-1, 1).contiguous()


class _SharedTHWStem(nn.Module):
    """Same lightweight local THW stem used before FiLM in every architecture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(F.silu(self.depthwise(x)))


class _FactorizedLocalReader(nn.Module):
    """Depthwise spatial then temporal convolutions forming a short 3D reader."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.spatial_pw = nn.Conv3d(channels, channels, kernel_size=1)
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.temporal_pw = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_pw(F.silu(self.spatial_dw(x)))
        x = x + spatial
        temporal = self.temporal_pw(F.silu(self.temporal_dw(x)))
        return x + temporal


class _FullLocalReader(nn.Module):
    """Depthwise short 3D convolution with pointwise channel mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(F.silu(self.depthwise(x)))


class ParallelTemporalDilations(nn.Module):
    """Parallel temporal readers over a time-derivative feature volume."""

    def __init__(self, channels: int, *, p1_enabled: bool = False) -> None:
        super().__init__()
        self.dilations = (1, 2, 4) if p1_enabled else (1,)
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        channels,
                        channels,
                        kernel_size=(3, 1, 1),
                        padding=(dilation, 0, 0),
                        dilation=(dilation, 1, 1),
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv3d(channels, channels, kernel_size=1),
                )
                for dilation in self.dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = torch.stack(
            [path(x) for path in self.paths], dim=0
        ).mean(dim=0)
        return x + F.silu(combined)


def physical_time_derivative(
    features: torch.Tensor, frame_times: torch.Tensor
) -> torch.Tensor:
    """Differentiate ``[B,T,...]`` features using each video's real times.

    Endpoints use one-sided slopes. Interior points use the three-point
    derivative for nonuniformly spaced samples. Times must be finite and
    strictly increasing; no uniform sampling interval is assumed.
    """
    if features.ndim < 2:
        raise ValueError(f"features must be [B,T,...], got {tuple(features.shape)}")
    b, t = features.shape[:2]
    if frame_times.shape != (b, t):
        raise ValueError(f"frame_times must be [B,T]={b,t}, got {tuple(frame_times.shape)}")
    times = frame_times.to(device=features.device, dtype=torch.float32)
    if not torch.isfinite(times).all():
        raise ValueError("frame_times must be finite")
    if t > 1 and torch.any(times[:, 1:] <= times[:, :-1]):
        raise ValueError("frame_times must be strictly increasing for every video")

    values = features.float()
    if t == 1:
        return torch.zeros_like(features)
    tail_dims = (1,) * (features.ndim - 2)
    intervals = (times[:, 1:] - times[:, :-1]).reshape(b, t - 1, *tail_dims)
    slopes = (values[:, 1:] - values[:, :-1]) / intervals
    if t == 2:
        derivative = torch.cat((slopes[:, :1], slopes[:, :1]), dim=1)
        return derivative.to(dtype=features.dtype)

    h_prev = (times[:, 1:-1] - times[:, :-2])
    h_next = (times[:, 2:] - times[:, 1:-1])
    h_prev_x = h_prev.reshape(b, t - 2, *tail_dims)
    h_next_x = h_next.reshape(b, t - 2, *tail_dims)
    interior = (
        -h_next_x / (h_prev_x * (h_prev_x + h_next_x)) * values[:, :-2]
        + (h_next_x - h_prev_x) / (h_prev_x * h_next_x) * values[:, 1:-1]
        + h_prev_x / (h_next_x * (h_prev_x + h_next_x)) * values[:, 2:]
    )
    derivative = torch.cat((slopes[:, :1], interior, slopes[:, -1:]), dim=1)
    return derivative.to(dtype=features.dtype)


def topk_referent_support(referent_logits: torch.Tensor, *, topk: int = 8) -> torch.Tensor:
    """Return mean top-k sigmoid referent support for each frame as ``[B,T]``."""
    if referent_logits.ndim != 4:
        raise ValueError(
            f"referent_logits must be [B,T,H,W], got {tuple(referent_logits.shape)}"
        )
    area = referent_logits.shape[-2] * referent_logits.shape[-1]
    if not 1 <= int(topk) <= area:
        raise ValueError(f"topk must be in [1,{area}], got {topk}")
    support = torch.sigmoid(referent_logits.float()).flatten(start_dim=-2)
    return torch.topk(support, k=int(topk), dim=-1).values.mean(dim=-1)


def asymmetric_event_referent_loss(
    event_logits: torch.Tensor,
    referent_logits: torch.Tensor,
    *,
    topk: int = 8,
    observed_time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize frame event support that exceeds top-k referent support.

    This is an ``a <= r`` evidence constraint.  It consumes model outputs and
    an optional observed-time mask; it does not consume boxes or missing-box
    labels.
    """
    if event_logits.ndim != 2 or tuple(event_logits.shape) != tuple(referent_logits.shape[:2]):
        raise ValueError(
            "event_logits must be [B,T] and match the first two referent dimensions"
        )
    event_support = torch.sigmoid(event_logits.float())
    referent_support = topk_referent_support(referent_logits, topk=topk)
    violation = F.relu(event_support - referent_support).square()
    if observed_time_mask is not None:
        if observed_time_mask.shape != violation.shape:
            raise ValueError(
                f"observed_time_mask must be [B,T]={tuple(violation.shape)}, "
                f"got {tuple(observed_time_mask.shape)}"
            )
        mask = observed_time_mask.to(device=violation.device, dtype=violation.dtype)
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("observed_time_mask must be binary")
        return (violation * mask).sum() / mask.sum().clamp_min(1.0)
    return violation.mean()


class Desta3DAdapterV2(nn.Module):
    """Dual spatial/event reader over a common stemmed THW latent.

    All modes use the same shared local THW stem before their separate query
    FiLM paths. ``shared3d`` then calls one local reader parameter set
    independently on both FiLM inputs. ``dual3d`` has separate local reader
    parameters. ``early_factorized`` is only an operator-factorization
    control; its name does not assert an early query-pooling stage or reproduce
    a separate pool-early E2 design. Spatial/event features, gates and output
    projections stay independent in every mode.

    The constructor starts in ``A evidence`` mode: query pools, reader and
    evidence-head parameters are trainable. Set ``B integration`` to train all
    groups or ``tta`` to expose only the small FiLM/LayerNorm/gate calibration
    set, including affine parameters from the shared-stem and both branch
    LayerNorms. The shared-stem convolutions remain frozen during TTA. For
    ``hidden_dim=128`` the TTA group has 66,818 parameters.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_VISUAL_DIM,
        query_dim: int = DEFAULT_QUERY_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        architecture: str = "dual3d",
        *,
        p1_enabled: bool = False,
        train_stage: str = "A evidence",
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
        self.p1_enabled = bool(p1_enabled)

        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)
        self.shared_stem = _SharedTHWStem(self.hidden_dim)
        self.norm_stem = ChannelOnlyLayerNorm(self.hidden_dim)
        self.query_pool_spatial = QueryAttentionPool(self.query_dim, self.hidden_dim)
        self.query_pool_event = QueryAttentionPool(self.query_dim, self.hidden_dim)
        self.film_spatial = QueryFiLM(self.hidden_dim)
        self.film_event = QueryFiLM(self.hidden_dim)

        if architecture == "shared3d":
            self.shared_reader = _FullLocalReader(self.hidden_dim)
            self.spatial_reader = None
            self.event_reader = None
        elif architecture == "early_factorized":
            self.shared_reader = None
            self.spatial_reader = _FactorizedLocalReader(self.hidden_dim)
            self.event_reader = _FactorizedLocalReader(self.hidden_dim)
        else:
            self.shared_reader = None
            self.spatial_reader = _FullLocalReader(self.hidden_dim)
            self.event_reader = _FullLocalReader(self.hidden_dim)
        self.event_temporal_reader = ParallelTemporalDilations(
            self.hidden_dim, p1_enabled=self.p1_enabled
        )
        self.event_p1_input_proj = (
            nn.Linear(2 * self.hidden_dim, self.hidden_dim) if self.p1_enabled else None
        )

        self.norm_spatial = ChannelOnlyLayerNorm(self.hidden_dim)
        self.norm_event = ChannelOnlyLayerNorm(self.hidden_dim)
        self.referent_head = nn.Linear(self.hidden_dim, 1)
        self.event_presence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.out_proj_spatial = nn.Linear(self.hidden_dim, self.in_channels)
        self.out_proj_event = nn.Linear(self.hidden_dim, self.in_channels)
        nn.init.normal_(self.out_proj_spatial.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.out_proj_event.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj_spatial.bias)
        nn.init.zeros_(self.out_proj_event.bias)
        self.gate_spatial = nn.Parameter(torch.tensor(-6.0))
        self.gate_event = nn.Parameter(torch.tensor(-6.0))

        self.set_train_stage(train_stage)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return the explicit, non-overlapping parameter groups."""
        groups: dict[str, tuple[nn.Parameter, ...]] = {
            "text_pool": tuple(
                list(self.query_pool_spatial.parameters())
                + list(self.query_pool_event.parameters())
            ),
            "input_projection": tuple(self.input_proj.parameters()),
            "shared_stem": tuple(self.shared_stem.parameters()),
            "shared_reader": tuple(
                self.shared_reader.parameters() if self.shared_reader is not None else ()
            ),
            "spatial_reader": tuple(
                self.spatial_reader.parameters() if self.spatial_reader is not None else ()
            ),
            "event_reader": tuple(
                list(self.event_reader.parameters() if self.event_reader is not None else ())
                + list(self.event_temporal_reader.parameters())
                + list(
                    self.event_p1_input_proj.parameters()
                    if self.event_p1_input_proj is not None
                    else ()
                )
            ),
            "branch_film": tuple(
                list(self.film_spatial.parameters()) + list(self.film_event.parameters())
            ),
            "norm_affine": tuple(
                list(self.norm_stem.parameters())
                + list(self.norm_spatial.parameters())
                + list(self.norm_event.parameters())
            ),
            "gates": (self.gate_spatial, self.gate_event),
            "out_projection": tuple(
                list(self.out_proj_spatial.parameters()) + list(self.out_proj_event.parameters())
            ),
            "readout_heads": tuple(
                list(self.referent_head.parameters())
                + list(self.event_presence_head.parameters())
            ),
        }
        ids = [id(parameter) for values in groups.values() for parameter in values]
        if len(ids) != len(set(ids)):
            raise RuntimeError("parameter_groups must not contain a parameter more than once")
        return groups

    def parameter_count_by_group(self) -> dict[str, int]:
        """Count actual parameters in every named group and in total."""
        groups = self.parameter_groups()
        counts = {name: sum(p.numel() for p in params) for name, params in groups.items()}
        counts["total"] = sum(counts.values())
        return counts

    def active_parameter_count(self) -> int:
        """Count currently trainable parameters; no synthetic padding is used."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def tta_parameter_count(self) -> int:
        """Count the target-time FiLM, LayerNorm affine and gate calibration set."""
        groups = self.parameter_groups()
        return sum(
            parameter.numel()
            for name in ("branch_film", "norm_affine", "gates")
            for parameter in groups[name]
        )

    def set_train_stage(self, stage: str) -> dict[str, int | str]:
        """Set trainable groups for ``A evidence``, ``B integration``, ``tta`` or ``frozen``."""
        normalized = " ".join(stage.strip().lower().replace("_", " ").replace("-", " ").split())
        aliases = {
            "a evidence": "A evidence",
            "b integration": "B integration",
            "tta": "tta",
            "frozen": "frozen",
        }
        if normalized not in aliases:
            raise ValueError(f"stage must be one of {TRAIN_STAGES}, got {stage!r}")
        canonical = aliases[normalized]
        all_groups = self.parameter_groups()
        stage_groups: dict[str, set[str]] = {
            "A evidence": {
                "text_pool",
                "input_projection",
                "shared_stem",
                "shared_reader",
                "spatial_reader",
                "event_reader",
                "branch_film",
                "norm_affine",
                "readout_heads",
            },
            "B integration": set(all_groups),
            "tta": {"branch_film", "norm_affine", "gates"},
            "frozen": set(),
        }
        enabled = stage_groups[canonical]
        for name, parameters in all_groups.items():
            for parameter in parameters:
                parameter.requires_grad_(name in enabled)
        self.train_stage = canonical
        return {
            "stage": canonical,
            "active_parameters": self.active_parameter_count(),
            **{f"{name}_parameters": sum(p.numel() for p in all_groups[name]) for name in all_groups},
        }

    @staticmethod
    def _run_local_reader(
        x: torch.Tensor,
        *,
        shared_reader: nn.Module | None,
        branch_reader: nn.Module | None,
    ) -> torch.Tensor:
        reader = shared_reader if shared_reader is not None else branch_reader
        if reader is None:
            raise RuntimeError("architecture has no local reader for this branch")
        return reader(x)

    @staticmethod
    def _gate_pair(
        gate_spatial: torch.Tensor,
        gate_event: torch.Tensor,
        override: float | Sequence[float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if override is None:
            return torch.sigmoid(gate_spatial), torch.sigmoid(gate_event)
        if isinstance(override, Sequence) and not isinstance(override, (str, bytes)):
            if len(override) != 2:
                raise ValueError("gate_override sequence must contain (spatial,event) values")
            spatial_value, event_value = float(override[0]), float(override[1])
        else:
            spatial_value = event_value = float(override)  # type: ignore[arg-type]
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (spatial_value, event_value)):
            raise ValueError("gate_override values must be finite and in [0,1]")
        return (
            gate_spatial.new_tensor(spatial_value),
            gate_event.new_tensor(event_value),
        )

    def forward(
        self,
        visual_grid: torch.Tensor,
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        frame_times: torch.Tensor,
        gate_override: float | Sequence[float] | None = None,
    ) -> dict[str, Any]:
        if visual_grid.ndim != 5:
            raise ValueError(
                f"visual_grid must be canonical [B,T,H,W,C], got {tuple(visual_grid.shape)}"
            )
        b, t, h, w, c = visual_grid.shape
        if c != self.in_channels or min(t, h, w) < 1:
            raise ValueError(
                f"visual_grid must have [B,T,H,W,{self.in_channels}] with nonempty THW, "
                f"got {tuple(visual_grid.shape)}"
            )
        if query_tokens.ndim != 3 or query_tokens.shape[0] != b or query_tokens.shape[2] != self.query_dim:
            raise ValueError(
                f"query_tokens must be [B,L,{self.query_dim}] matching video batch, "
                f"got {tuple(query_tokens.shape)}"
            )
        if frame_times.shape != (b, t):
            raise ValueError(f"frame_times must be [B,T]={b,t}, got {tuple(frame_times.shape)}")
        checked_times = frame_times.to(device=visual_grid.device, dtype=torch.float32)
        if not torch.isfinite(checked_times).all():
            raise ValueError("frame_times must be finite")
        if t > 1 and torch.any(checked_times[:, 1:] <= checked_times[:, :-1]):
            raise ValueError("frame_times must be strictly increasing for every video")

        z_spatial, alpha_spatial = self.query_pool_spatial(query_tokens, query_mask)
        z_event, alpha_event = self.query_pool_event(query_tokens, query_mask)
        latent = self.input_proj(visual_grid)
        latent_cf = latent.permute(0, 4, 1, 2, 3).contiguous()
        latent_cf = self.norm_stem(self.shared_stem(latent_cf))
        latent = latent_cf.movedim(1, -1)
        spatial_input = self.film_spatial(latent, z_spatial)
        event_input = self.film_event(latent, z_event)

        spatial_cf = spatial_input.permute(0, 4, 1, 2, 3).contiguous()
        spatial_cf = self._run_local_reader(
            spatial_cf,
            shared_reader=self.shared_reader,
            branch_reader=self.spatial_reader,
        )
        spatial_cf = self.norm_spatial(spatial_cf)
        spatial_features = F.silu(spatial_cf.movedim(1, -1))

        if self.p1_enabled:
            event_derivative = physical_time_derivative(event_input, frame_times)
            if self.event_p1_input_proj is None:
                raise RuntimeError("P1 event projection is missing while p1_enabled is true")
            event_reader_input = self.event_p1_input_proj(
                torch.cat((event_input, event_derivative), dim=-1)
            )
        else:
            # P0 control: keep a separate event pathway while disabling the
            # derivative and temporal pyramid as a coupled P1 mechanism.
            event_reader_input = event_input
            event_derivative = None
        event_cf = event_reader_input.permute(0, 4, 1, 2, 3).contiguous()
        event_cf = self._run_local_reader(
            event_cf,
            shared_reader=self.shared_reader,
            branch_reader=self.event_reader,
        )
        event_cf = self.event_temporal_reader(event_cf)
        event_cf = self.norm_event(event_cf)
        event_features = F.silu(event_cf.movedim(1, -1))

        referent_logits = self.referent_head(spatial_features).squeeze(-1)
        # Log-mean-exp pools feature values over XY while preserving a smooth
        # gradient from any strong spatial evidence. This is a direct frame
        # presence readout, not a max-occupancy proxy.
        event_pooled = torch.logsumexp(event_features.float(), dim=(2, 3)) - math.log(h * w)
        event_logits = self.event_presence_head(event_pooled.to(event_features.dtype)).squeeze(-1)

        gate_s, gate_e = self._gate_pair(
            self.gate_spatial, self.gate_event, gate_override
        )
        proposal_s = self.out_proj_spatial(spatial_features)
        proposal_e = self.out_proj_event(event_features)
        delta_s = proposal_s * gate_s.to(dtype=proposal_s.dtype)
        delta_e = proposal_e * gate_e.to(dtype=proposal_e.dtype)
        updated_s = visual_grid + delta_s
        updated_e = visual_grid + delta_e

        entropy_s = -(alpha_spatial * alpha_spatial.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_e = -(alpha_event * alpha_event.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "updated_tokens_spatial": updated_s,
            "updated_tokens_event": updated_e,
            "delta_spatial": delta_s,
            "delta_event": delta_e,
            "proposal_delta_spatial": proposal_s,
            "proposal_delta_event": proposal_e,
            "branch_features_spatial": spatial_features,
            "branch_features_event": event_features,
            "event_derivative": event_derivative,
            "z_spatial": z_spatial,
            "z_event": z_event,
            "alpha_spatial": alpha_spatial,
            "alpha_event": alpha_event,
            "alpha_entropy_spatial": entropy_s,
            "alpha_entropy_event": entropy_e,
            "referent_logits": referent_logits,
            "event_logits": event_logits,
            "gate_spatial": gate_s,
            "gate_event": gate_e,
            "architecture": self.architecture,
            "p1_enabled": self.p1_enabled,
        }
