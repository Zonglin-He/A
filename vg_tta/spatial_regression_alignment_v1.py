"""Bounded source-statistic diagnostic. Not part of the deployed method."""
import torch

ARMS = ('all', 'random64', 'significant64')
SEED = 20260918


def features(replay):
    """Capture the actual final bbox MLP input, without detaching its gradient."""
    captured = []
    head = replay.decoder.decoder.bbox_embed
    hook = head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        values = replay.values()
    finally:
        hook.remove()
    if len(captured) != 12:
        raise RuntimeError(f'Expected six bbox calls per original offset, got {len(captured)}')
    zz = [captured[5].reshape(-1, 256), captured[11].reshape(-1, 256)]
    z = torch.stack([zz[i % 2][i // 2] for i in range(replay.n)])
    # MLP is pointwise. Recompute with original shapes to preserve exact kernels.
    with torch.no_grad():
        bb = [head(captured[i]).sigmoid().reshape(-1, 4) for i in (5, 11)]
        b = torch.stack([bb[i % 2][i // 2] for i in range(replay.n)])
        if not torch.equal(b, values['boxes']):
            raise RuntimeError('Captured representation is not the actual final bbox readout')
    return values, z


def moments(z):
    """Uniform sampled frames; population (not unbiased) variance, stable FP64."""
    z = z.double()
    mu = z.mean(0)
    variance = (z-mu).square().mean(0)
    return mu, (variance + 1e-12).sqrt()


def sensitivity(head, z):
    """Per-frame L1 Jacobian norm over FOUR normalized cxcywh outputs."""
    with torch.enable_grad(), torch.autocast('cuda', enabled=False):
        x = z.detach().clone().float().requires_grad_(True)
        b = head(x).sigmoid()
        gs = [torch.autograd.grad(b[:, k].sum(), x, retain_graph=k < 3)[0].abs()
              for k in range(4)]
    r = torch.stack(gs).sum(0).detach().double()
    if not torch.isfinite(r).all():
        raise RuntimeError('Nonfinite source output sensitivity')
    return r


def reference(records):
    """Equal sources, uniform frames within each: mixture moments, not mean SD."""
    first = torch.stack([r['z'].double().mean(0) for r in records]).mean(0)
    second = torch.stack([r['z'].double().square().mean(0) for r in records]).mean(0)
    variance = (second-first.square()).clamp_min(0)
    r = torch.stack([v['sensitivity'].double().mean(0) for v in records]).mean(0)
    top = torch.argsort(r, descending=True, stable=True)[:64]
    gen = torch.Generator().manual_seed(SEED)
    random = torch.randperm(256, generator=gen)[:64]
    return dict(mu=first, sigma=(variance+1e-12).sqrt(), sensitivity=r,
                indices={'all': torch.arange(256), 'random64': random, 'significant64': top},
                source_count=len(records), source_equal=True, frames_uniform=True)


class AlignmentLoss:
    def __init__(self, ref, arm, device):
        self.idx = ref['indices'][arm].to(device)
        self.mu = ref['mu'][self.idx.cpu()].to(device)
        self.sigma = ref['sigma'][self.idx.cpu()].to(device)
        if arm == 'all':
            self.weights = torch.ones(256, dtype=torch.float64, device=device)
        else:
            r = ref['sensitivity'][self.idx.cpu()].to(device)
            if r.sum() <= 0:
                raise RuntimeError('Zero source sensitivity in selected subspace')
            self.weights = r / r.mean()  # mean one, sum64 for BOTH 64-dimensional arms
        self.arm = arm

    def __call__(self, z):
        mu, sigma = moments(z[:, self.idx])
        return (self.weights*((mu-self.mu).square()+(sigma-self.sigma).square())).sum()


def finite_difference(replay, lossfn, grad, capture=features, epsilon=1e-3):
    """Central difference in the normalized analytic-gradient direction; reset."""
    state = replay.state()
    direction = grad / grad.norm().clamp_min(1e-30)
    vals = []
    try:
        for sign in (-1, 1):
            i = 0
            with torch.no_grad():
                for name, p in replay.named:
                    n = p.numel()
                    p.copy_(state[name] + sign*epsilon*direction[i:i+n].reshape_as(p).to(p))
                    i += n
                _, z = capture(replay)
                vals.append(float(lossfn(z)))
    finally:
        replay.restore(state)
    numerical = (vals[1]-vals[0])/(2*epsilon)
    analytic = float(grad.norm())
    rel = abs(numerical-analytic)/max(abs(analytic), abs(numerical), 1e-12)
    return dict(epsilon=epsilon, analytic=analytic, numerical=numerical, relative_error=rel,
                passed=rel < .05, direction='source-point analytic gradient normalized')
