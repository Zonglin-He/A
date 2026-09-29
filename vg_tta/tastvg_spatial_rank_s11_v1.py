"""Rank teacher and fixed-norm actuation on the unchanged S1 spatial actor."""
import numpy as np
import torch
from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor, geometry


def average_ranks(rewards, tol=1e-12):
    """Descending, zero-based average ranks; tie groups anchored at their maximum."""
    a=np.asarray(rewards, dtype=np.float64)
    assert a.ndim==1 and np.isfinite(a).all()
    order=np.argsort(-a, kind='stable'); ranks=np.empty(len(a),dtype=np.float64)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[i]]-a[order[j]]<=tol:j+=1
        ranks[order[i:j]]=(i+j-1)/2
        i=j
    return ranks


def reverse_kl(central,candidates,rewards,coeff):
    distances=geometry(central,candidates,*coeff);logp=(-distances).log_softmax(0)
    ranks=average_ranks(rewards.detach().cpu().numpy())
    logq=(-torch.as_tensor(ranks,device=central.device,dtype=central.dtype)).log_softmax(0)
    return (logp.exp()*(logp-logq)).sum(),logp.exp(),logq.exp(),distances


def update_scale(gradients,arm,lr,radius):
    """Global norm across all1792 parameters; zero gradient always gives no-op."""
    norm=float(torch.sqrt(sum(g.double().square().sum() for g in gradients)))
    assert np.isfinite(norm)
    return (lr if arm=='rank' else radius/norm) if norm>0 else 0.,norm
