"""Fixed P1 spatial guidance for PE's learned global attention pool.

Geometry is NumPy-only. Load this file directly with importlib when a caller
must avoid the existing vg_tta package initializer's eager framework imports.
Torch is imported only by pool_features. No pixels, labels, model loading,
feature normalization, parameter updates, or temporal cropping occur here.
"""

import math

import numpy as np


ALPHA = 0.5
IMAGE_SIZE = 336
PATCH_GRID = 24


def fractional_mask(box, width, height, grid=PATCH_GRID):
    """Return [row, column] patch coverage after the stock PE square squash.

The input is pixel xyxy on the complete original width-by-height image. Its
edges are rescaled to 336x336, clipped to that image, and intersected with
half-open patch cells. An invalid, empty, or entirely outside box returns
zeros; pool_features explicitly falls back to the original global pool for
such a mask. Invalid image dimensions are metadata errors and raise.
"""
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid <= 0:
        raise ValueError("grid must be a positive integer")
    width, height = float(width), float(height)
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("image dimensions must be finite and positive")
    empty = np.zeros((grid, grid), dtype=np.float64)
    try:
        b = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError):
        return empty
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
        return empty
    scale = [IMAGE_SIZE / width, IMAGE_SIZE / height] * 2
    b = np.clip(b * scale, 0, IMAGE_SIZE)
    if np.any(b[2:] <= b[:2]):
        return empty
    cell = IMAGE_SIZE / grid
    lo = np.arange(grid, dtype=np.float64) * cell
    hi = lo + cell
    x = np.maximum(0, np.minimum(hi, b[2]) - np.maximum(lo, b[0]))
    y = np.maximum(0, np.minimum(hi, b[3]) - np.maximum(lo, b[1]))
    return np.clip(y[:, None] * x[None, :] / (cell * cell), 0, 1)


def pool_features(visual, tokens, masks, alpha=ALPHA):
    """Return (original_feature, softly_guided_feature, per-image_stats).

tokens are complete-frame, final-layer PE tokens after ln_post, INCLUDING
the leading CLS token: real PE-L14-336 shape [B,577,1024]. masks may be
[B,24,24] or [B,576]; a single grid also works for B=1. Small grids/widths
are permitted for CPU interface tests. No L2 normalization is applied.

Only the fixed ALPHA=.5 and the alpha0 positive-control value are accepted.
The learned probe/MHA, residual MLP and visual.proj are reused unchanged.
Alpha0, full masks, and empty/invalid-box masks return original-pool values
exactly; mixed batches retain those values without a second masked call.
"""
    import torch

    alpha = float(alpha)
    if alpha not in (0.0, ALPHA):
        raise ValueError("P1 fixes alpha=.5; alpha=0 is the parity control only")
    if tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 2:
        raise ValueError("tokens must be [B, CLS+patches, width]")
    if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError("tokens must be finite floating-point values")
    if visual.pool_type != "attn" or not visual.use_cls_token:
        raise ValueError("P1 requires the original attention pool and leading CLS")
    pool = visual.attn_pool
    heads = pool.attn.num_heads
    if pool.num_heads != heads or not pool.attn.batch_first:
        raise ValueError("original pool.num_heads must match MHA.num_heads")
    if pool.probe.shape[1] != 1 or tokens.shape[2] != pool.attn.embed_dim:
        raise ValueError("P1 requires one learned probe and the original token width")
    if visual.training or pool.training or pool.attn.training:
        raise ValueError("P1 pooling requires frozen evaluation mode")
    if visual.proj is None or visual.proj.shape[0] != tokens.shape[2]:
        raise ValueError("the original visual projection must be present")

    batch, length, _ = tokens.shape
    m = torch.as_tensor(masks, dtype=torch.float32, device=tokens.device)
    if m.ndim == 2 and batch == 1 and m.shape[0] != 1:
        m = m.reshape(1, -1)
    elif m.ndim == 3:
        m = m.flatten(1)
    if m.ndim != 2 or m.shape != (batch, length - 1):
        raise ValueError("mask must contain one row-major value per patch, excluding CLS")
    if not torch.isfinite(m).all() or torch.any((m < 0) | (m > 1)):
        raise ValueError("fractional masks must be finite and within [0,1]")

    with torch.inference_mode():
        empty = torch.all(m == 0, dim=1)
        full = torch.all(m == 1, dim=1)
        guided = (~empty) & (~full) & (alpha != 0)
        patch_prior = (1 - alpha) + alpha * m
        # These rows explicitly use the global pool, so stats report its prior.
        patch_prior = torch.where(empty[:, None], torch.ones_like(patch_prior), patch_prior)
        prior = torch.cat((torch.ones((batch, 1), device=m.device), patch_prior), dim=1)
        global_feature = visual._pool(tokens) @ visual.proj
        soft_feature = global_feature
        if torch.any(guided):
            active = tokens[guided]
            count = active.shape[0]
            q = pool.probe.repeat((count, 1, 1)).to(active.dtype)
            bias = torch.log(prior[guided]).to(active.dtype)
            bias = bias[:, None, :].repeat_interleave(heads, dim=0)
            pooled = pool.attn(q, active, active, need_weights=False, attn_mask=bias)[0]
            pooled = pooled + pool.mlp(pool.layernorm(pooled))
            soft_feature = global_feature.clone()
            soft_feature[guided] = pooled.squeeze(1) @ visual.proj
        stats = {
            "alpha": alpha,
            "pool_heads": heads,
            "patches": length - 1,
            "mask_area_fraction": m.mean(dim=1),
            "empty_fallback": empty,
            "full_mask": full,
            "guided": guided,
            "prior_min": prior.min(dim=1).values,
            "prior_max": prior.max(dim=1).values,
            "background_positive": torch.all(patch_prior > 0, dim=1),
            "cls_prior": prior[:, 0],
        }
        return global_feature, soft_feature, stats
