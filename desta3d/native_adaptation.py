"""DESTA: native pseudo-supervision and projected R16 latent updates.

There is no optimizer state and no trainable model here. Gradients are balanced
in the full PTD merger space BEFORE projection. Missing evidence stays missing.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int
    radius: float
    temporal_weight: float = 1.0
    spatial_weight: float = 1.0

    def __post_init__(self):
        if self.steps < 1 or self.radius < 0 or not math.isfinite(self.radius):
            raise ValueError('Invalid step count or radius')
        if any(not math.isfinite(w) or w < 0 for w in (self.temporal_weight, self.spatial_weight)):
            raise ValueError('Weights must be finite and nonnegative')


def l2(x):
    return float(x.detach().double().norm())


def temporal_target(interval, frame_ids):
    """Physical-frame endpoints to observed token indices; ties choose earlier."""
    ids = torch.as_tensor(frame_ids, dtype=torch.float64)
    if ids.ndim != 1 or not len(ids) or not bool((ids[1:] > ids[:-1]).all()):
        raise ValueError('Expected strictly increasing observed frame IDs')
    if interval is None:
        return torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool)
    x = torch.as_tensor(interval, dtype=torch.float64)
    if x.shape != (2,) or not torch.isfinite(x).all() or x[1] < x[0]:
        raise ValueError('Invalid expert interval')
    return (x[:, None] - ids).abs().argmin(1), torch.ones(2, dtype=torch.bool)


def spatial_target(boxes_by_frame, frame_ids, positions, coordinate_ids):
    """Normalized xyxy to actual PTD vocabulary IDs on fixed native anchors."""
    ids = torch.as_tensor(coordinate_ids, dtype=torch.long)
    if ids.shape != (1001,) or len(set(ids.tolist())) != 1001:
        raise ValueError('Expected the exact native coordinate token table')
    target = torch.zeros((len(positions), 4), dtype=torch.long)
    valid = torch.zeros_like(target, dtype=torch.bool)
    for j, pos in enumerate(positions):
        box = boxes_by_frame.get(str(frame_ids[pos]))
        if box is None:
            continue
        b = torch.as_tensor(box, dtype=torch.float64)
        if (b.shape != (4,) or not torch.isfinite(b).all() or bool(((b < 0) | (b > 1)).any())
                or not bool((b[2:] > b[:2]).all())):
            raise ValueError('Invalid box must be recorded as missing by provider')
        target[j] = ids[(b * 1000).round().long()]
        valid[j] = True
    return target, valid


def native_loss(logits, targets, valid):
    if logits.shape[:-1] != targets.shape or targets.shape != valid.shape:
        raise ValueError('Native support mismatch')
    if not valid.any():
        raise ValueError('Missing support is handled without a backward')
    return F.cross_entropy(logits[valid].float(), targets[valid], reduction='mean')


def project_branch(full_gradient, basis):
    """Return sufficient raw evidence for independent update reconstruction."""
    if not torch.isfinite(full_gradient).all():
        raise ValueError('Nonfinite full-space gradient')
    norm = l2(full_gradient)
    projected = (full_gradient.double() @ basis.double()).float()
    return projected, norm


def normalized_update(coeff, projected, full_norms, basis, stock_norm, config):
    """Normalize full F gradients, combine, then project and radius-bound C.

Projection commutes with the weighted sum: keeping g_F @ Q plus ||g_F||
avoids retaining enormous full-F gradients while preserving this exact order.
    """
    g = torch.zeros_like(coeff)
    for branch, weight in [('event', config.temporal_weight), ('spatial', config.spatial_weight)]:
        n = full_norms[branch]
        if not math.isfinite(n) or n < 0:
            raise ValueError('Invalid full-gradient norm')
        if n > 0 and weight > 0:
            g = g + projected[branch] * (weight / n)
    gn = l2(g)
    eta = config.radius * stock_norm / config.steps
    candidate = coeff - g * (eta / gn) if gn else coeff.clone()
    # Use the actual Gram matrix, rather than assuming perfectly orthogonal FP32 columns.
    gram = basis.double().T @ basis.double()
    flat = candidate.double().reshape(-1, basis.shape[1])
    delta_norm = float(((flat @ gram) * flat).sum().clamp_min(0).sqrt())
    cap = config.radius * stock_norm
    scale = min(1., cap / delta_norm) if delta_norm else 1.
    result = candidate * scale
    return result, {'balanced_projected_norm': gn, 'step_size': eta,
                    'pre_projection_delta_norm': delta_norm, 'cap_scale': scale,
                    'delta_norm': delta_norm * scale, 'stock_norm': stock_norm}


def grid_configs():
    return {f'K{k}_R{r:g}_T{w:g}': AdaptationConfig(k, r, w, 1.)
            for k in (1, 3, 5) for r in (.03, .07, .135) for w in (.5, 1., 2.)}
