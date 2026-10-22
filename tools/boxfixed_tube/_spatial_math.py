"""Fixed CPU geometry operations; no model, file or annotation access."""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
from threading import RLock
import torch
EPS = 1e-15
_CPU_THREAD_LOCK = RLock()


@contextmanager
def single_thread_cpu():
    """Use the historical CPU arithmetic thread count, then restore the host.

    Thread count affects FP32 BLAS reduction order. Re-entrant locking and
    saving the count at each entry make nested production APIs and exceptions
    safe without setting a process-global default at import time.
    """
    with _CPU_THREAD_LOCK:
        previous = torch.get_num_threads()
        try:
            if previous != 1:
                torch.set_num_threads(1)
            yield
        finally:
            if previous != 1:
                torch.set_num_threads(previous)


def scoped_single_thread(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with single_thread_cpu():
            return function(*args, **kwargs)
    return wrapped

def require(condition, message):
    if not condition: raise ValueError(message)

def cpu(value, dtype=None):
    if torch.is_tensor(value): require(value.device.type == 'cpu', 'CPU evidence required')
    return torch.as_tensor(value, dtype=dtype).detach().clone()

def occupancy(boxes, height, width):
    x0, y0 = (boxes[..., :2] - boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x1, y1 = (boxes[..., :2] + boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x = torch.arange(width + 1, dtype=torch.float64) / width
    y = torch.arange(height + 1, dtype=torch.float64) / height
    ox = (torch.minimum(x1[..., None], x[1:]) - torch.maximum(x0[..., None], x[:-1])).clamp_min(0) * width
    oy = (torch.minimum(y1[..., None], y[1:]) - torch.maximum(y0[..., None], y[:-1])).clamp_min(0) * height
    return oy[..., :, None] * ox[..., None, :], ox, oy

def soft_dice(a, b):
    return (2 * (a * b).sum(-1) / (a.sum(-1) + b.sum(-1)).clamp_min(EPS)).clamp(0, 1)

def affinity(normalized, valid, target, source):
    ti = torch.nonzero(valid[target], as_tuple=False).flatten()
    si = torch.nonzero(valid[source], as_tuple=False).flatten()
    matrix = torch.zeros(normalized.shape[1], normalized.shape[1], dtype=torch.float64)
    if len(ti) and len(si):
        cosine = normalized[target, ti] @ normalized[source, si].T
        scores, indices = torch.topk(cosine, min(20, len(si)), dim=1, sorted=False)
        weights = (scores / .1).softmax(1).double()
        weights /= weights.sum(1, keepdim=True)
        matrix[ti[:, None].expand_as(indices), si[indices]] = weights
    return matrix

def fit_axis_affine(joint, source_xy, target_xy):
    """Weighted target=s*source+delta, independently for x and y.

    Reliability checks are hard admission tests, never scale clipping. Marginal
    ESS ensures three effective source *and* target locations, rather than
    treating many copies of one cell as independent coordinate information.
    """
    joint = cpu(joint, torch.float64); x = cpu(source_xy, torch.float64); y = cpu(target_xy, torch.float64)
    require(joint.shape == (len(x), len(y)) and x.shape[1:] == y.shape[1:] == (2,), 'WLS coordinate dimensions changed')
    require(bool(torch.isfinite(joint).all()) and bool((joint >= 0).all())
            and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'Finite WLS support required')
    mass = joint.sum(); sw = joint.sum(1); tw = joint.sum(0)
    tiny = torch.finfo(torch.float64).tiny
    source_ess = mass.square() / sw.square().sum().clamp_min(tiny)
    target_ess = mass.square() / tw.square().sum().clamp_min(tiny)
    info = dict(mass=float(mass), source_ESS=float(source_ess), target_ESS=float(target_ess),
                variance=[0., 0.], scale=[0., 0.], delta=[0., 0.], valid=False, reason='insufficient_mass_or_ESS')
    if float(mass) <= EPS or float(source_ess) < 3 or float(target_ess) < 3:
        return None, info
    mx = (sw[:, None] * x).sum(0) / mass; my = (tw[:, None] * y).sum(0) / mass
    xc, yc = x - mx, y - my
    variance = (sw[:, None] * xc.square()).sum(0) / mass
    info['variance'] = variance.tolist()
    if not bool((variance > 1e-6).all()):
        info['reason'] = 'insufficient_source_axis_variance'; return None, info
    covariance = (xc * (joint @ yc)).sum(0) / mass
    scale = covariance / variance; delta = my - scale * mx
    info.update(scale=scale.tolist(), delta=delta.tolist())
    if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(delta).all()) or not bool(((scale >= .5) & (scale <= 2.)).all()):
        info['reason'] = 'nonfinite_or_out_of_range_axis_scale'; return None, info
    info.update(valid=True, reason='accepted')
    return (scale, delta), info

def transform_native_box(box, scale, delta):
    box = cpu(box, torch.float64)
    transformed = torch.cat((scale * box[:2] + delta, scale * box[2:]))
    if not bool(torch.isfinite(transformed).all()) or not bool((transformed[2:] > 0).all()): return None
    lo, hi = transformed[:2] - transformed[2:] / 2, transformed[:2] + transformed[2:] / 2
    if not bool(((hi > 0) & (lo < 1)).all()): return None
    # No image-boundary clipping: the entire positive native extent survives.
    return transformed

def extent_kernel(boxes, references):
    """Full-box RBF, centers normalized by each propagated reference's size."""
    relative = torch.cat(((boxes[..., :2] - references[..., :2]) / references[..., 2:],
                          boxes[..., 2:].log() - references[..., 2:].log()), -1)
    return torch.exp(-.5 * relative.square().sum(-1))
