"""Class-free, cross-offset target-content feedback. No model/label access."""
import numpy as np
from stvg.prototype import membership, inclusive_scores, update

PRIMARY = 'RegionKernel'
KAPPA = 16.0

def weights(box, height, width, valid):
    """Fractional grid coverage of the fixed 1.5 context; unchanged coordinates."""
    b = np.asarray(box, dtype=np.float64).copy()
    b[2:] *= 1.5
    lo, hi = b[:2] - b[2:] / 2, b[:2] + b[2:] / 2
    xx, yy = np.arange(width) / width, np.arange(height) / height
    dx = np.maximum(0, np.minimum(xx + 1 / width, hi[0]) - np.maximum(xx, lo[0])) * width
    dy = np.maximum(0, np.minimum(yy + 1 / height, hi[1]) - np.maximum(yy, lo[1])) * height
    out = dy[:, None] * dx[None, :] * np.asarray(valid, bool)
    fallback = not bool(out.sum() > 0)
    if fallback:
        out = np.asarray(valid, dtype=np.float64).copy()
    assert out.sum() > 0
    return out, fallback

def pool(features, support):
    a = np.asarray(features, dtype=np.float64)
    w = np.asarray(support, dtype=np.float64)
    assert a.shape[:2] == w.shape and np.isfinite(a).all()
    return np.einsum('hwc,hw->c', a, w) / w.sum()

def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)

def evidence(features, q0):
    """K=(1+cos)/2 after fitting-fold centering. One estimate, no fitted head.

    Positive/negative weighted kernel means are shrunk to the same fitting-fold
    unconditional kernel mean. Their ratio is a compatibility score, NOT a
    calibrated event probability. Cross-offset fits are dependent, not validation.
    """
    z = np.asarray(features, dtype=np.float64)
    assert z.ndim == 2 and len(z) == len(q0) and np.isfinite(z).all()
    w = membership(q0)
    result = np.zeros(len(z)); stats = []
    for parity in (0, 1):
        fit = np.arange(len(z)) % 2 != parity
        v = z[fit]
        center = np.where(np.ptp(v, axis=0) == 0, v[0], v.mean(0))
        a, b = unit(v - center), unit(z[~fit] - center)
        kernel = np.clip((1 + b @ a.T) / 2, 0, 1)
        mu = kernel.mean(1)
        wp, wn = w[fit], 1 - w[fit]
        plus = (kernel @ wp + KAPPA * mu) / (wp.sum() + KAPPA)
        minus = (kernel @ wn + KAPPA * mu) / (wn.sum() + KAPPA)
        zero = np.all(a == 0) or wp.sum() == 0 or wn.sum() == 0
        if not zero:
            result[~fit] = np.log(np.maximum(plus, 1e-12) / np.maximum(minus, 1e-12))
        stats.append(dict(heldout_parity=parity,positive_weight=float(wp.sum()),
            negative_weight=float(wn.sum()),zero_contrast=bool(zero),
            kernel_mean=float(kernel.mean()),score_std=float(result[~fit].std())))
    return result, stats

def candidates(features, q0, old_k16_L, ids):
    feedback = {}; stats = {}
    for name in [PRIMARY, 'FullGrid', 'Reflected']:
        feedback[name], stats[name] = evidence(features[name], q0)
    # Physical-time-only signal uses the same estimator and shrinkage, no GT.
    position = np.asarray(ids, dtype=np.float64)[:, None]
    feedback['PositionOnly'], stats['PositionOnly'] = evidence(position, q0)
    wrong = feedback[PRIMARY].copy()
    for parity in (0, 1):
        index = np.arange(parity, len(q0), 2)
        wrong[index] = np.roll(wrong[index], len(index) // 2)
    feedback['WrongTime'] = wrong
    old = np.asarray(old_k16_L, dtype=np.float64)
    factor = float(np.linalg.norm(feedback[PRIMARY]) / max(np.linalg.norm(old), 1e-300))
    feedback['K16Matched'] = old * factor
    qs = {a: update(q0, inclusive_scores(L)) for a, L in feedback.items()}
    qs['DirectSameEvidence'] = update((q0 > 0).astype(float) / (q0 > 0).sum(), inclusive_scores(feedback[PRIMARY]))
    qs['Sharpen'] = update(q0, np.log(np.maximum(q0, 1e-300)))
    constant, _ = evidence(np.ones((len(q0), 3)), q0)
    qs['ConstantNoOp'] = update(q0, inclusive_scores(constant))
    assert np.array_equal(qs['ConstantNoOp'], q0)
    return qs, dict(L=feedback, statistics=stats, k16_strength_factor=factor)
