"""Single-view episodic box-feature adaptation with precomputed source statistics.

This is an SSA-inspired bbox-only affine variant, not a reproduction of SSA.
Legal fitting accepts only frozen features/statistics. Oracle fitting is a
separate, explicitly supervised diagnostic entry point. No backbone gradients.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from .metrics import generalized_box_iou_aligned_cxcywh


@dataclass(frozen=True)
class AffineConfig:
    objective: str = 'subspace'
    rank: int = 16
    lr: float = .001
    steps: int = 3
    gamma: float = .0001
    eps: float = .0001

    def __post_init__(self):
        if self.objective not in ('subspace', 'diagonal', 'oracle'):
            raise ValueError('unknown objective')
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError('rank must be a positive integer')
        if any(not math.isfinite(x) or x < 0 for x in (self.lr, self.gamma)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('invalid optimization parameter')


class AffineBoxHead(nn.Module):
    """Identity-initialized adapter immediately before a frozen bbox last layer."""
    def __init__(self, source):
        super().__init__()
        self.base = copy.deepcopy(source).eval().requires_grad_(False)
        if not hasattr(self.base, 'layers') or len(self.base.layers) != 3:
            raise ValueError('requires the native three-layer bbox MLP')
        last = self.base.layers[-1]
        if last.out_features != 4:
            raise ValueError('bbox output must contain four coordinates')
        self.scale_delta = nn.Parameter(torch.zeros(last.in_features, device=last.weight.device))
        self.shift = nn.Parameter(torch.zeros_like(self.scale_delta))
        self.eval()

    def encode(self, x):
        with torch.no_grad():
            for layer in self.base.layers[:-1]:
                x = F.relu(layer(x))
        return x.detach()

    def calibrate(self, z):
        return z.float() * (1. + self.scale_delta) + self.shift

    def logits_from_features(self, z):
        return self.base.layers[-1](self.calibrate(z))

    def forward(self, x):
        return self.logits_from_features(self.encode(x))

    def penalty(self):
        return self.scale_delta.square().sum() + self.shift.square().sum()


def autocast_for(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()


def cached_features(head, complete_input):
    if complete_input.ndim != 3 or not torch.isfinite(complete_input).all():
        raise ValueError('expected finite complete [L,T,D] head inputs')
    with torch.no_grad(), autocast_for(head.scale_delta.device):
        return head.encode(complete_input.to(head.scale_delta.device)).detach()


def boxes_from_features(head, z):
    # Preserve complete native [L,T,D] GEMM shape for BF16 replay parity.
    with autocast_for(head.scale_delta.device):
        return head.logits_from_features(z).sigmoid()[-1]


def compute_source_statistics(features, last_weight, *, relative_floor=1e-4):
    """Equal-video moments. Inputs are one [T,D] tensor per source video.

    Only source features and frozen regression weights determine PCA/weights.
    No target data, output coordinates, or annotations enter this function.
    """
    if len(features) < 2:
        raise ValueError('need at least two source videos')
    zs = [z.detach().double().cpu() for z in features]
    d = zs[0].shape[-1]
    if any(z.ndim != 2 or len(z) < 2 or z.shape[1] != d or not torch.isfinite(z).all() for z in zs):
        raise ValueError('invalid source features')
    mean = torch.stack([z.mean(0) for z in zs]).mean(0)
    cov = sum((z - mean).T @ (z - mean) / len(z) for z in zs) / len(zs)
    values, vectors = torch.linalg.eigh(cov)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    floor = max(1e-6, float(values[0]) * relative_floor)
    sensitivity = torch.linalg.vector_norm(last_weight.detach().double().cpu() @ vectors, dim=0)
    # Vector-output extension of SSA dimension relevance; not scalar SSA verbatim.
    importance = 1. + sensitivity / sensitivity.mean().clamp_min(1e-12)
    return {'mean': mean.float(), 'vectors': vectors.float(), 'values': values.float(),
            'diagonal': cov.diag().float(), 'importance': importance.float(), 'variance_floor': floor,
            'videos': len(zs), 'frames': sum(len(z) for z in zs), 'feature_dim': d,
            'effective_rank_above_floor': int((values > floor).sum()), 'labels_used': False}


def alignment_loss(calibrated, stats, config):
    if calibrated.ndim != 2 or calibrated.shape[0] < 2:
        raise ValueError('need at least two temporal feature samples')
    centered = calibrated.float() - stats['mean'].to(calibrated.device)
    if config.objective == 'subspace':
        k = min(config.rank, stats['effective_rank_above_floor'])
        if k < 1:
            return calibrated.sum() * 0.
        projected = centered @ stats['vectors'][:, :k].to(calibrated.device)
        source_var = stats['values'][:k].to(calibrated.device)
        weights = stats['importance'][:k].to(calibrated.device)
    elif config.objective == 'diagonal':
        projected = centered
        source_var = stats['diagonal'].to(calibrated.device)
        valid = source_var > stats['variance_floor']
        projected, source_var = projected[:, valid], source_var[valid]
        if not valid.any():
            return calibrated.sum() * 0.
        weights = torch.ones_like(source_var)
    else:
        raise ValueError('alignment loss cannot consume oracle config')
    mu = projected.mean(0)
    # Add a source-only numerical floor to both variances; do not use GT masks.
    vt = projected.var(0, unbiased=False) + stats['variance_floor']
    vs = source_var + stats['variance_floor']
    symmetric_kl = .5 * ((vt + mu.square()) / vs + (vs + mu.square()) / vt - 2.)
    return (weights * symmetric_kl).sum() / weights.sum()


def oracle_loss(head, features, targets):
    boxes = boxes_from_features(head, features).float()
    valid = [i for i, t in enumerate(targets) if len(t['boxes'])]
    if not valid:
        return head.penalty() * 0.
    gt = torch.stack([targets[i]['boxes'][0] for i in valid]).to(boxes.device).float().detach()
    selected = boxes[valid]
    return 5. * (selected - gt).abs().sum(1).mean() + 2. * (1. - generalized_box_iou_aligned_cxcywh(selected, gt)).mean()


def _fit(head, features, config, objective):
    params = [head.scale_delta, head.shift]
    if head.training or any(p.requires_grad for p in head.base.parameters()):
        raise ValueError('source head must stay frozen/eval')
    if any(torch.count_nonzero(p.detach()) for p in params):
        raise ValueError('each episode must begin with a fresh identity adapter')
    features = features.detach().to(head.scale_delta.device)
    snapshot = {n: p.detach().clone() for n, p in head.base.state_dict().items()}
    optimizer = torch.optim.AdamW(params, lr=config.lr, eps=config.eps, weight_decay=0.)
    losses, gradients = [], []
    if features.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        initial = float(objective())
    failed = False
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective() + config.gamma * head.penalty()
        if not torch.isfinite(loss):
            failed = True
            break
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in params])
        if not torch.isfinite(grad).all():
            failed = True
            break
        gradients.append(float(grad.norm()))
        losses.append(float(loss.detach()))
        optimizer.step()
    if failed or not all(torch.isfinite(p).all() for p in params):
        failed = True
        with torch.no_grad():
            for p in params:
                p.zero_()
    with torch.no_grad():
        final = float(objective())
    if features.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert all(torch.equal(head.base.state_dict()[n], v) for n, v in snapshot.items())
    return {'steps': len(losses), 'requested_steps': config.steps, 'initial_loss': initial,
            'final_loss': final, 'losses': losses, 'gradient_norms': gradients,
            'parameter_delta_l2': float(head.penalty().detach().sqrt()),
            'trainable_parameters': sum(p.numel() for p in params), 'source_state_exact': True,
            'nonfinite_fallback': failed, 'update_seconds': elapsed}


def fit_affine(head, features, stats, config):
    """Legal TTA boundary: no labels, boxes, temporal GT, or evaluation records."""
    if config.objective not in ('subspace', 'diagonal') or stats.get('labels_used') is not False:
        raise ValueError('legal alignment requires label-free source statistics')
    return _fit(head, features, config, lambda: alignment_loss(head.calibrate(features[-1]), stats, config))


def fit_affine_oracle(head, features, targets, config):
    """GT local-fitting diagnostic only. Never report as a legal TTA result."""
    if config.objective != 'oracle':
        raise ValueError('oracle must be explicitly marked')
    result = _fit(head, features, config, lambda: oracle_loss(head, features, targets))
    result['gt_used'] = True
    return result
