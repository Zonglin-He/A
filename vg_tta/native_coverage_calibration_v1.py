"""Two-parameter episodic endpoint calibration; no label-bearing input.

Linear endpoint tilts do NOT preserve endpoint rankings. Unequal start/end
tilts need a native-at-the-selected-physical-length decoder to preserve the
native preference between equal-duration intervals. KL is a soft anchor,
not a guarantee of event semantics or utility.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from vg_tta.metrics import interval_from_logits


def validate(logits, frame_ids):
    z = torch.as_tensor(logits).detach().cpu().double()
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    ids = np.asarray(frame_ids, dtype=np.int64)
    if z.ndim != 2 or z.shape != (len(ids), 2) or len(ids) == 0:
        raise ValueError('Expected one nonempty T x 2 sequence')
    if not torch.isfinite(z).all() or (np.diff(ids) <= 0).any():
        raise ValueError('Require finite logits and strictly increasing frame IDs')
    u = torch.as_tensor((ids-ids[0])/max(int(ids[-1]-ids[0]), 1), dtype=torch.float64)
    return z, ids, u


class EndpointCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self, z, u):
        return z + torch.stack((-self.alpha[0]*u, self.alpha[1]*u), dim=-1)


def objective(adapted, native, strength=1.0):
    if strength <= 0:
        raise ValueError('A positive posterior anchor is required')
    logp = adapted.log_softmax(0)
    logp0 = native.log_softmax(0)
    coverage = -(logp[0, 0]+logp[-1, 1])/2
    kl = (logp0.exp()*(logp0-logp)).sum()/2
    return coverage + strength*kl, coverage, kl


def fit(logits, frame_ids, strength=1.0, max_iter=100, lr=1.0):
    """Minimize a convex, label-free objective with two fresh model parameters."""
    z, ids, u = validate(logits, frame_ids)
    model = EndpointCalibration().eval()
    calls = 0
    initial = float(objective(model(z, u), z, strength)[0].detach())
    if max_iter > 0:
        if lr == 0:
            opt = torch.optim.SGD(model.parameters(), lr=0)
            opt.zero_grad(); objective(model(z, u), z, strength)[0].backward(); opt.step()
            calls = 1
        else:
            opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter,
                                   tolerance_grad=1e-10, tolerance_change=1e-13,
                                   line_search_fn='strong_wolfe')

            def closure():
                nonlocal calls
                opt.zero_grad()
                loss = objective(model(z, u), z, strength)[0]
                loss.backward(); calls += 1
                return loss

            opt.step(closure)
    result = model(z, u)
    loss, coverage, kl = objective(result, z, strength)
    grad = torch.autograd.grad(loss, model.alpha)[0]
    # Independent moment condition of the convex exponential-tilt optimum.
    p0, p = z.softmax(0), result.softmax(0)
    expected = torch.stack((strength*(p0[:, 0]*u).sum()/(1+strength),
                           (1+strength*(p0[:, 1]*u).sum())/(1+strength)))
    if len(ids) == 1:
        expected.zero_()
    moment_residual = (p*u[:, None]).sum(0)-expected
    assert float(loss.detach()) <= initial+1e-9
    return {'logits': result.detach()[None], 'alpha': model.alpha.detach().clone(),
            'loss_before': initial, 'loss_after': float(loss.detach()),
            'coverage_loss': float(coverage.detach()), 'posterior_KL': float(kl.detach()),
            'gradient_max': float(grad.abs().max()), 'moment_residual_max': float(moment_residual.abs().max().detach()),
            'closure_calls': calls, 'parameter_count': 2, 'target_GT_used': False,
            'initialized_at_zero': True, 'eval_mode': not model.training}


def tilted(logits, frame_ids, alpha):
    z, _, u = validate(logits, frame_ids)
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.shape != (2,) or not torch.isfinite(a).all():
        raise ValueError('Expected two finite calibration parameters')
    return (z+torch.stack((-a[0]*u, a[1]*u), dim=-1))[None]


def native_at_length(logits, frame_ids, *, reference=None, fraction=None):
    """Native raw joint maximizer at exact physical length (or nearest grid).

    reference is an inclusive pair selected WITHOUT GT. It guarantees an
    achievable exact duration, including on an irregular sampling grid.
    fraction is a fixed prior, not a target-event duration.
    """
    z, ids, _ = validate(logits, frame_ids)
    if (reference is None) == (fraction is None):
        raise ValueError('Specify exactly one length source')
    t = len(ids)
    duration = ids[None, :]+1-ids[:, None]
    legal = np.triu(np.ones((t, t), bool), k=int(t > 1))
    wanted = (int(ids[reference[1]]+1-ids[reference[0]]) if reference is not None
              else float(fraction)*(ids[-1]+1-ids[0]))
    mismatch = np.where(legal, abs(duration-wanted), np.inf)
    chosen_mask = legal & (mismatch == mismatch.min())
    scores = z[:, 0, None]+z[None, :, 1]
    scores = scores.masked_fill(~torch.from_numpy(chosen_mask), -torch.inf)
    k = int(scores.argmax()); ij = (k//t, k % t)
    return {'indices': ij, 'requested_physical_length': float(wanted),
            'achieved_physical_length': int(duration[ij]), 'length_mismatch': float(mismatch[ij]),
            'eligible_same_length': int(chosen_mask.sum())}


def decode(logits, frame_ids, alpha):
    raw = interval_from_logits(tilted(logits, frame_ids, alpha))
    anchored = native_at_length(logits, frame_ids, reference=raw)
    return raw, anchored
