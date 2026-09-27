"""Isolated, label-free R1 interventions; never imported by the locked method."""
import numpy as np
import torch

from methods.decota_v1.api import decode
from vg_tta.native_coverage_calibration_v1 import native_at_length, validate
from methods.decota_refine_uniform_v1.api import reconstruct

ALPHAS = (0., .25, .5, .75, 1.)
RHOS = (0., .1, .25, .5, None)


def local_placement(logits, ids, raw, native, rho):
    current = decode(logits, raw, ids, native_indices=native)
    if rho is None or current['unchanged_extent_preserves_native']:
        return tuple(current['indices'])
    if not np.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite nonnegative or None')
    z, tt, _ = validate(logits, ids)
    length = int(tt[raw[1]] + 1 - tt[raw[0]])
    candidates = [(i, j) for i in range(len(tt)) for j in range(i + 1, len(tt))
                  if tt[j] + 1 - tt[i] == length
                  and abs(tt[i] - tt[raw[0]]) <= rho * length + 1e-10]
    assert tuple(raw) in candidates
    return max(candidates, key=lambda ij: (float(z[ij[0], 0] + z[ij[1], 1]), -ij[0], -ij[1]))


def extent_amount(logits, ids, raw, native, eta):
    if not 0 <= eta <= 1:
        raise ValueError('eta must be in [0,1]')
    if eta == 0:
        return tuple(native)
    if eta == 1:
        return tuple(decode(logits, raw, ids, native_indices=native)['indices'])
    ln = ids[native[1]] + 1 - ids[native[0]]
    la = ids[raw[1]] + 1 - ids[raw[0]]
    if ln == la:
        return tuple(native)
    wanted = ln + eta * (la - ln)
    return tuple(native_at_length(logits, ids, fraction=wanted / (ids[-1]+1-ids[0]))['indices'])


def blend(base, absolute, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    base, absolute = base.detach().cpu().float(), absolute.detach().cpu().float()
    if base.shape != absolute.shape or not torch.isfinite(base).all() or not torch.isfinite(absolute).all():
        raise ValueError('Invalid matched box arrays')
    if alpha == 0:
        return base.clone()
    if alpha == 1:
        return absolute.clone()
    out = (1-alpha)*base + alpha*absolute
    # Preserve exact native values outside support, including FP32 rounding.
    same = (base == absolute).all(-1)
    out[same] = base[same]
    return out


def box_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo = np.maximum(a[:2]-a[2:]/2, b[:2]-b[2:]/2)
    hi = np.minimum(a[:2]+a[2:]/2, b[:2]+b[2:]/2)
    inter = np.maximum(hi-lo, 0).prod()
    return float(inter / max(a[2:].prod()+b[2:].prod()-inter, 1e-12))


def leave_one_anchor_alpha(base, pseudo, ids):
    pp = sorted(pseudo, key=lambda x: x['position'])
    if len(pp) < 3:
        return 1., dict(eligible=False, held_out=0, reason='fewer_than_three_anchors', GT_used=False)
    if len({p['position'] for p in pp}) != len(pp):
        raise ValueError('Duplicate anchors')
    losses = {a: [] for a in ALPHAS}
    for k in range(1, len(pp)-1):
        pos = pp[k]['position']
        # Held-out expert box is absent from the interpolation inputs.
        train = pp[:k] + pp[k+1:]
        assert train[0]['position'] < pos < train[-1]['position']
        pred, _ = reconstruct(base, train, ids, 'absolute')
        for a in ALPHAS:
            proposal = blend(base[pos:pos+1], pred[pos:pos+1], a)[0].numpy()
            losses[a].append(1-box_iou(proposal, pp[k]['box']))
    means = {a: float(np.mean(v)) for a, v in losses.items()}
    best = min(means.values())
    chosen = max(a for a in ALPHAS if means[a] <= best+1e-12)
    return chosen, dict(eligible=True, held_out=len(pp)-2, losses=means, GT_used=False)
