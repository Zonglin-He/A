"""Only common-mean shrinkage varies:4,16,64. One round/sharedtwo-directions."""
import numpy as np
from threadpoolctl import threadpool_limits
from stvg.prototype import membership,inclusive_scores,update
VALUES=(16,64)
BETAS=(.25,.5,1.)
CONFIGS={f"K{k}_B{int(b*100)}":(k,b) for k in VALUES for b in BETAS}
def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)

def evidence(features, q0, kappa):
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
        plus = (kernel @ wp + kappa * mu) / (wp.sum() + kappa)
        minus = (kernel @ wn + kappa * mu) / (wn.sum() + kappa)
        zero = np.all(a == 0) or wp.sum() == 0 or wn.sum() == 0
        if not zero:
            result[~fit] = np.log(np.maximum(plus, 1e-12) / np.maximum(minus, 1e-12))
        stats.append(dict(heldout_parity=parity,positive_weight=float(wp.sum()),
            negative_weight=float(wn.sum()),zero_contrast=bool(zero),
            kernel_mean=float(kernel.mean()),score_std=float(result[~fit].std())))
    return result, stats



def apply(features,q0,kappa,beta=1.):
 assert kappa in VALUES and beta in BETAS
 with threadpool_limits(limits=1,user_api='blas'):
  L,stats=evidence(features,q0,kappa)
 return update(q0,beta*inclusive_scores(L)),dict(L=L,prototypes=stats,kappa=kappa,beta=beta,rounds=1)
