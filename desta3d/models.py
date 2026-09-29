"""Readable direction predictor; tensor names document the full data flow.

Original research implementations remain immutable in vg_tta/. This module
preserves Local/GlobalMean parameter names, construction order and arithmetic.
The 3D adapter itself is already formatted and remains a single implementation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import AdapterConfig, DirectionConfig, ViewConfig


def build_adapter(config: AdapterConfig):
    from vg_tta.desta3d_v2 import Desta3DAdapterV2

    model = Desta3DAdapterV2(
        in_channels=config.in_channels,
        query_dim=config.query_dim,
        hidden_dim=config.hidden_dim,
        architecture=config.architecture,
        p1_enabled=config.p1_enabled,
        train_stage=config.train_stage,
    )
    # Original class tta includes two gates (66,818). The fixed v2 experiments
    # froze those gates: FiLM + LN only (66,816 at hidden128).
    if config.train_stage == "tta" and config.freeze_tta_gates:
        model.gate_event.requires_grad_(False)
        model.gate_spatial.requires_grad_(False)
    return model


class DirectionMixer(nn.Module):
    """[B,T,H,W,128] + pooled queries + evidence8 + state33 -> coefficient field.

    forward() returns coefficients, not PTD logits or a tube.
    project() separately maps them to a fixed-norm visual residual.
    """

    def __init__(self, basis: torch.Tensor, config: DirectionConfig):
        super().__init__()
        if basis.ndim != 2 or basis.shape[1] != config.output_rank:
            raise ValueError("basis columns must match output_rank")
        identity = torch.eye(config.output_rank, dtype=torch.float64, device=basis.device)
        if not torch.isfinite(basis).all() or not torch.allclose(
            basis.double().T @ basis.double(), identity, atol=2e-6, rtol=0
        ):
            raise ValueError("Expected a finite orthonormal basis")
        self.config = config
        self.radius = config.radius
        self.register_buffer("basis", basis.detach().float().clone())

        input_dim = 3 * config.feature_dim + 8 + 33
        width = config.hidden_dim
        self.input = nn.Linear(input_dim, width)
        self.local = nn.Conv3d(width, width, 3, padding=1, groups=width)
        self.mix = nn.Linear(width, width)
        self.output = nn.Linear(width, config.output_rank)
        nn.init.normal_(self.output.weight, std=config.output_init_std)
        nn.init.zeros_(self.output.bias)

    def forward(self, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        z, q_t, q_s, evidence, state = (
            cache[k].detach() for k in ("z", "qT", "qS", "evidence8", "state33")
        )
        if z.ndim != 5 or z.shape[-1] != self.config.feature_dim:
            raise ValueError("z must be [B,T,H,W,feature_dim]")
        batch, time, height, width, _ = z.shape
        grid = (batch, time, height, width)
        if q_t.shape != (batch, self.config.feature_dim) or q_s.shape != q_t.shape:
            raise ValueError("qT and qS must be [B,feature_dim]")
        if evidence.shape != (*grid, 8) or state.shape != (batch, time, 33):
            raise ValueError("evidence8/state33 do not match the THW support")

        q_t = q_t[:, None, None, None].expand(*grid, -1)
        q_s = q_s[:, None, None, None].expand(*grid, -1)
        state = state[:, :, None, None].expand(*grid, -1)
        inputs = torch.cat((z, q_t, q_s, evidence, state), dim=-1)
        h0 = F.silu(self.input(inputs))
        # Conv3d consumes [B,C,T,H,W]; all public fields use [B,T,H,W,C].
        local = F.silu(self.local(h0.movedim(-1, 1)).movedim(1, -1))
        hidden = h0 + local
        if self.config.global_mean:
            hidden = hidden + h0.mean(dim=(1, 2, 3), keepdim=True)
        return self.output(F.silu(self.mix(hidden)))

    def project(self, coefficients: torch.Tensor, stock_norm: torch.Tensor) -> torch.Tensor:
        """Map coefficient direction to radius * ||F||, separately per query."""
        field = F.linear(coefficients, self.basis.detach())
        lengths = field.flatten(1).norm(dim=1)
        norms = torch.as_tensor(stock_norm, device=field.device, dtype=field.dtype).reshape(-1)
        if norms.numel() != field.shape[0] or not torch.isfinite(norms).all() or (norms < 0).any():
            raise ValueError("stock_norm must contain one finite nonnegative norm per query")
        scale = self.radius * norms / lengths.clamp_min(1e-12)
        return field * scale.reshape(-1, *([1] * (field.ndim - 1)))


def direction_loss(prediction: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    """Equal-query 1-cosine, over each complete THW coefficient field.

    This is not a per-token cosine. Zero oracle fields have no target direction.
    """
    if prediction.shape != oracle.shape:
        raise ValueError("Prediction and oracle must have identical support")
    pred = prediction.flatten(1)
    target = oracle.detach().flatten(1)
    pred_norm = pred.norm(dim=1)
    target_norm = target.norm(dim=1)
    valid = target_norm > 0
    cosine = (pred / pred_norm[:, None].clamp_min(1e-12)
              * (target / target_norm[:, None].clamp_min(1e-12))).sum(dim=-1)
    return (1 - cosine[valid]).mean() if valid.any() else pred.sum() * 0


def make_pixel_views(frames, frame_ids, evidence, config: ViewConfig):
    """Call the audited pixel constructor with explicit editable parameters."""
    from vg_tta.desta3d_v3_policy_gate_views import build_views

    return build_views(
        frames, frame_ids, evidence,
        dim=config.temporal_dim, radius=config.spatial_blur_radius,
    )
