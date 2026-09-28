"""Fixed source diagnostic: scale an existing late mask direction, no optimization."""
import math
import torch
from vg_tta.desta3d_v3_location_probe import match_delta

# User-locked decimal ratios from two previously completed span controls.
RATIOS = {'event': 0.087687, 'spatial': 0.170316}

def large_mask_delta(stock, baseline, masked, branch):
    """Retain B1 residual; denominator is stock F, not baseline F+R_b.

    A neutral mask has no direction. It stays neutral and is explicitly marked
    inapplicable to positive-norm scaling, never replaced by an invented vector.
    """
    if branch not in RATIOS:
        raise ValueError('Unknown branch')
    if (stock.shape != baseline.shape or masked.shape != baseline.shape or
            stock.ndim != 5 or baseline.dtype != torch.float32 or masked.dtype != torch.float32):
        raise ValueError('Identical THWC support and FP32 branch endpoints required')
    if not all(torch.isfinite(t).all() for t in (stock, baseline, masked)):
        raise ValueError('Nonfinite support')
    stock_norm = float(stock.double().norm())
    if not math.isfinite(stock_norm) or stock_norm <= 0:
        raise ValueError('Nonzero finite stock visual norm required')
    requested = RATIOS[branch] * stock_norm
    applicable = bool(torch.any(masked != baseline))
    raw, scaled, log = match_delta(baseline, masked, requested if applicable else 0.)
    log.update(branch=branch, requested_ratio=RATIOS[branch], stock_norm=stock_norm,
               requested_norm=requested, applicable=applicable,
               neutral_reason=None if applicable else 'zero original mask direction',
               realized_ratio=log['realized_norm']/stock_norm,
               denominator='original captured stock visual_grid F',
               baseline='B1 F+R_b is retained; only signed mask-minus-B1 delta is scaled')
    return raw, scaled, log
