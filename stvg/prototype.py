"""Frozen class-free RegionKernel, one round, common-mean Native2D shrinkage (frozen configuration)."""
import numpy as np
from threadpoolctl import threadpool_limits
import json
from pathlib import Path
_CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/default.json').read_text())['temporal']
KAPPA=float(_CFG['kappa']); BETA=float(_CFG.get('beta',1.))
def membership(q):
 return np.diag(np.cumsum(np.cumsum(q,axis=0)[:,::-1],axis=1)[:,::-1]).copy()

def inclusive_scores(L):
 c=np.r_[0.,np.cumsum(L)];return c[1:][None,:]-c[:-1][:,None]

def update(prior,S):
 if np.all(S[prior>0]==0):return prior.copy()
 take=prior>0;x=np.log(prior[take])+S[take];x-=x.max();out=np.zeros_like(prior);out[take]=np.exp(x);out/=out.sum();return out

def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)

def evidence(features, q0, kappa=KAPPA):
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


def one_step(features,q0,*,kappa=KAPPA,beta=BETA):
 with threadpool_limits(limits=1,user_api='blas'):
  L,stats=evidence(features,q0,kappa)
 S=beta*inclusive_scores(L);q=update(q0,S)
 return q,dict(L=L,S=S,prototypes=stats,kappa=kappa,beta=beta,rounds=1)
