"""One-step source oracle in a common merger-THW coordinate system.

The query features and model parameters are held fixed. Unlike the historical
post-reader delta control, the replacement F enters BOTH the identity residual
and the frozen B1 readers. No optimizer or learnable model is created here.
"""
from contextlib import contextmanager
import math
import torch


@contextmanager
def shared_fields(adapter, values, sequence, *, allow_prefix=False):
    """Replace only the visual argument, once for each declared fresh pass."""
    calls = []
    def before(module, args):
        if len(calls) >= len(sequence):
            raise ValueError('Unexpected extra adapter call')
        branch = sequence[len(calls)]
        value = values[branch]
        if value.shape != args[0].shape or value.dtype != args[0].dtype:
            raise ValueError('Shared F support or dtype changed')
        calls.append(branch)
        return (value, *args[1:])
    handle = adapter.register_forward_pre_hook(before)
    try:
        yield calls
        if calls != list(sequence) and not (allow_prefix and calls == list(sequence)[:len(calls)] and calls):
            raise ValueError('Missing declared branch prefill')
    finally:
        handle.remove()


def norm(x):
    return float(x.detach().double().norm())


def project(x, basis):
    return (x.double() @ basis.double() @ basis.double().T).float()


def unit_to(x, radius):
    n = norm(x)
    if not math.isfinite(n) or not math.isfinite(radius) or radius < 0:
        raise ValueError('Invalid direction or radius')
    # A zero/unsupported direction remains neutral, never replaced by noise.
    return x * (radius/n) if n > 0 else torch.zeros_like(x)


def union_basis(qt, qs):
    u, s, _ = torch.linalg.svd(torch.cat([qt, qs], dim=1).double(), full_matrices=False)
    keep = s > 1e-10 * s[0]
    return u[:, keep].float(), s.cpu()


def corrections(gt, gs, qt, qs, qj, stock, ratios):
    if gt.shape != gs.shape or gt.shape != stock.shape:
        raise ValueError('Both gradients must use the same F support')
    if not torch.isfinite(gt).all() or not torch.isfinite(gs).all():
        raise ValueError('Nonfinite gradient')
    radius = norm(stock)
    dt = unit_to(-project(gt, qt), radius*ratios['event'])
    ds = unit_to(-project(gs, qs), radius*ratios['spatial'])
    budget = math.sqrt(norm(dt)**2 + norm(ds)**2)
    # Equal branch importance in local geometry, without mixing CE gradient
    # scales. Q_union gives joint at least the union of both individual spans.
    balance = unit_to(gt, 1.) + unit_to(gs, 1.)
    dj = unit_to(-project(balance, qj), budget)
    return {'T': dt, 'S': ds, 'J': dj, 'J_pass': dj/math.sqrt(2.)}


def geometry(gt, gs, deltas):
    def cos(a,b):
        n = norm(a)*norm(b)
        return float((a.double()*b.double()).sum())/n if n else None
    return {'gradient_cosine': cos(gt,gs), 'gradient_norms': {'T':norm(gt),'S':norm(gs)},
            'direction_cosine':cos(deltas['T'],deltas['S']),
            'norms':{k:norm(v) for k,v in deltas.items()},
            'descent_dot':{b:{k:float(-(g.double()*v.double()).sum()) for k,v in deltas.items()}
                           for b,g in [('T',gt),('S',gs)]}}
