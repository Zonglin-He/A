"""Frozen, within-modality boundary contrast; no labels or model execution."""
import numpy as np
from scipy.stats import rankdata


def boundary_curve(offsets, ids, window=2):
    """Two-sided mean-embedding contrast, separately on each native offset.

    Hidden is text-conditioned, but its cosine is NOT a calibrated event score.
    Only adjacent-window changes are used. Six layers remain equally weighted.
    Midpoints and interpolation are in physical frame units, not ordinal time.
    Unknown edge regions are neutral, rather than invented negative evidence.
    """
    endpoints = np.array(ids, dtype=np.float64)
    centers = endpoints + .5
    curves, raw = [], []
    for offset in offsets:
        h = offset['hidden']
        if hasattr(h, 'numpy'):
            h = h.numpy()
        h = np.asarray(h, dtype=np.float64)[:, 0]
        h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-30)
        x = np.asarray(offset['frame_ids'], dtype=np.float64) + .5
        cuts, strengths = [], []
        for j in range(window, len(x)-window+1):
            left = h[:, j-window:j].mean(1)
            right = h[:, j:j+window].mean(1)
            # Same-side similarity minus across-side similarity = .5||muL-muR||².
            strengths.append(float(.5*np.square(left-right).sum(-1).mean()))
            cuts.append(float((x[j-1]+x[j])/2))
        values = np.asarray(strengths)
        # Bounded centered ranks: no query-specific scale or GT calibration.
        ranks = (rankdata(values, method='average')-.5)/len(values)-.5
        if np.ptp(values) < 1e-14:
            ranks[:] = 0.
        raw.append(dict(cuts=np.asarray(cuts), strengths=values, ranks=ranks))
        curves.append(np.stack([np.interp(endpoints+shift, cuts, ranks,
                                         left=0., right=0.) for shift in [0., 1.]]))
    curve = np.mean(curves, axis=0)
    assert curve.shape == (2, len(centers)) and np.isfinite(curve).all()
    return curve, raw


def groups_for(ids, positions, pairs):
    ids = np.asarray(ids)
    anchors = ids[np.asarray(positions)] + .5
    membership = ((anchors[None] >= ids[pairs[:, 0], None]) &
                  (anchors[None] < ids[pairs[:, 1], None]+1))
    patterns, inverse = np.unique(membership, axis=0, return_inverse=True)
    return patterns, inverse


def reweight(q0, pairs, scores, groups=None):
    """Preserve native total mass of every anchor-membership class if requested."""
    native = q0[pairs[:, 0], pairs[:, 1]]
    weights = native*np.exp(np.asarray(scores)-np.max(scores))
    if groups is None:
        weights /= weights.sum()
    else:
        total = np.bincount(groups, weights=native)
        den = np.bincount(groups, weights=weights)
        weights *= total[groups]/np.maximum(den[groups], 1e-300)
    out = np.zeros_like(q0)
    out[pairs[:, 0], pairs[:, 1]] = weights
    assert abs(float(out.sum())-1) < 2e-13
    return out
