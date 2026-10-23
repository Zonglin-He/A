"""Class-free within-video kernel feedback; no model/time-decoder dependency.

Feature dimension is arbitrary. Alternating frames are the default two folds;
folds are NOT independent evidence/validation. No learned state or optimizer.
"""
import numpy as np
from threadpoolctl import threadpool_limits
KAPPA=16.0
def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)

def evidence(features, soft_membership, folds=None):
    """K=(1+cos)/2 after fitting-fold centering. One estimate, no fitted head.

    Positive/negative weighted kernel means are shrunk to the same fitting-fold
    unconditional kernel mean. Their ratio is a compatibility score, NOT a
    calibrated event probability. Cross-offset fits are dependent, not validation.
    """
    z = np.asarray(features, dtype=np.float64)
    assert z.ndim == 2 and len(z) == len(soft_membership) and np.isfinite(z).all()
    w = np.asarray(soft_membership,dtype=np.float64)
    assert w.shape==(len(z),) and np.isfinite(w).all() and np.all((w>=-1e-12)&(w<=1+1e-12))
    folds=np.arange(len(z))%2 if folds is None else np.asarray(folds)
    assert folds.shape==(len(z),) and set(folds.tolist())=={0,1}
    result = np.zeros(len(z)); stats = []
    for parity in (0, 1):
        fit = folds != parity
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



def feedback(features,temporal,folds=None):
    if len(features)!=len(temporal.times):raise ValueError('Explicit physical-time alignment required')
    with threadpool_limits(limits=1,user_api='blas'):
        L,stats=evidence(features,temporal.soft_membership(),folds)
    return temporal.revise(L),dict(L=L,prototypes=stats,kappa=16,rounds=1)
