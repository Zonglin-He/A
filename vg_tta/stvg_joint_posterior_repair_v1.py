"""Source-only EDM-style diagonal preconditioning, preserving raw noise units.

This is a finite Native-warm-start VE approximation, not unconditional sampling.
No dataset readers, target GT, Student optimizer, or identity model lives here.
"""
import math
import torch
from torch import nn
from vg_tta.stvg_joint_posterior_v1 import JointPrior, roi, times, observation_loglik


class PreconditionedJointPrior(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.core = JointPrior(False)
        self.register_buffer('center', mean.clone().reshape(1, 33, 4))
        self.register_buffer('data_scale', scale.clone().reshape(1, 33, 4))

    def coefficients(self, sigma, batch):
        s = torch.as_tensor(sigma, device=self.center.device).expand(batch).reshape(-1, 1, 1)
        den = self.data_scale.square() + s.square()
        return self.data_scale.square()/den, s*self.data_scale/den.sqrt(), den.rsqrt()

    def forward(self, x, sigma, c):
        # Physical ROI coordinates MUST use raw x, never the preconditioned input.
        B = len(x)
        skip, out, cin = self.coefficients(sigma, B)
        u = cin*(x-self.center)
        r = roi(c['maps'], x[:, 1:].sigmoid(), times(x, c['unit']))
        r = torch.cat((r.mean(1, keepdim=True)*0, r), 1)
        noise = torch.as_tensor(sigma, device=x.device).expand(B).log()/4
        ss = torch.stack((noise, noise.sin(), noise.cos()), -1)[:, None].expand(-1, 33, -1)
        net = self.core
        v = net.state(torch.cat((u, ss), -1)) + net.roi(r) + net.position
        ctxt = torch.cat((net.context(c['context']), net.text(c['query'])), 1)
        pad = torch.cat((torch.zeros(B, 32, dtype=torch.bool, device=x.device), c['qmask']), 1)
        h, _ = net.cross(v, ctxt, ctxt, key_padding_mask=pad, need_weights=False)
        f = net.out(net.sequence(v+h, mask=net.block))
        result = self.center + skip*(x-self.center) + out*f
        # The two unused temporal-token channels are not stochastic state.
        active = torch.ones_like(result); active[:, 0, 2:] = 0
        return result*active

    def weighted_loss(self, prediction, truth, sigma, mask):
        _, out, _ = self.coefficients(sigma, len(truth))
        err = ((prediction-truth)/out.clamp_min(1e-12)).square()*mask
        per_time = err[:, 0, :2].mean(-1)
        per_space = err[:, 1:].sum((1, 2))/mask[:, 1:].sum((1, 2)).clamp_min(1)
        return ((per_time+per_space)/2).mean()


def rollout(model, c, seed, obs=None, prior=None, variance=None, levels=16):
    """Four distinct paths; clean-estimate guidance includes its true Jacobian.

    No coordinate averaging or GT-based sample selection. Caller evaluates all
    paths and the prelocked first path separately. Old event-relative likelihood
    (and its exterior blind region) is deliberately retained for this isolation.
    """
    gen = torch.Generator(device=c['native'].device).manual_seed(seed)
    sigma = torch.logspace(math.log10(2.), math.log10(.03), levels, device=c['native'].device)
    x = c['native'] + sigma[0]*torch.randn(c['native'].shape, device=c['native'].device, generator=gen)
    trace = []
    for i, s in enumerate(sigma):
        x = x.detach(); x[:, 0, 2:] = 0
        guided = obs is not None and len(obs['time']) > 0
        x.requires_grad_(guided)
        with torch.set_grad_enabled(guided):
            clean = model(x, s.expand(len(x)), c)
            if guided:
                ll = observation_loglik(clean, c, obs, prior, variance)
                grad, = torch.autograd.grad(ll.sum(), x)
            else:
                grad = torch.zeros_like(x); ll = x.new_zeros(len(x))
            sn = sigma[i+1] if i+1 < len(sigma) else s*0
            proposal = clean + (sn/s)*(x-clean)
            delta = (.02*s.square()*grad).clamp(-.2, .2)
            x = (proposal+delta).detach()
        assert torch.isfinite(x).all()
        trace.append(dict(level=i, sigma=float(s), clean_estimate_loglik=float(ll.mean().detach()), observation_gradient_norm=float(grad.norm().detach())))
    return x, trace
