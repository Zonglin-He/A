"""Algebraic native-hull and RegionKernel formulas; research adapters only."""
import numpy as np
from threadpoolctl import threadpool_limits
from stvg.prototype import membership,inclusive_scores,update,unit

def positive_hull(grids):
    """Independent min-start/max-end distribution in O(T^2), positive sums.

    Native two-offset grids are disjoint. Each hull endpoint thus has a unique
    owner. Equal-owner intervals multiply the other view's strict-contained mass;
    different-owner endpoints multiply row/column cumulative masses. No interval
    pruning, CDF subtraction, gain or negative-mass clipping is used.
    """
    assert len(grids)==2 and not(set(grids[0]['ids'])&set(grids[1]['ids']))
    ids=sorted(set(grids[0]['ids'])|set(grids[1]['ids']));ix={f:i for i,f in enumerate(ids)};n=len(ids)
    Q=[];contained=[];row=[];col=[]
    for g in grids:
        q=np.zeros((n,n),np.float64);local=np.array([ix[f]for f in g['ids']]);q[local[g['ij'][0]],local[g['ij'][1]]]=g['p'];Q.append(q)
        inside=q[::-1].cumsum(0)[::-1].cumsum(1);strict=np.zeros_like(q);strict[:-1,1:]=inside[1:,:-1];contained.append(strict)
        less=np.zeros_like(q);less[:,1:]=q.cumsum(1)[:,:-1];row.append(less)
        greater=np.zeros_like(q);greater[:-1]=q[::-1].cumsum(0)[::-1][1:];col.append(greater)
    out=Q[0]*contained[1]+Q[1]*contained[0]+row[0]*col[1]+row[1]*col[0]
    assert np.isfinite(out).all() and(out>=0).all() and abs(out.sum()-1)<1e-12 and not np.tril(out).any()
    return ids,out

def linear_evidence(features,q0,kappa=64):
    """K=(1+cos)/2 weighted means expressed by two vectors, not a T*T kernel.

    Same other-offset centering/L2, means, common-mean shrinkage and zero-contrast
    rule. FP64 arithmetic order differs; finite-output equivalence is audited.
    """
    z=np.asarray(features,dtype=np.float64);w=membership(q0);out=np.zeros(len(z));stats=[]
    for parity in (0,1):
        fit=np.arange(len(z))%2!=parity;v=z[fit];center=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));a,b=unit(v-center),unit(z[~fit]-center)
        wp,wn=w[fit],1-w[fit];mean=a.mean(0)
        vp=(a.T@wp+kappa*mean)/(wp.sum()+kappa);vn=(a.T@wn+kappa*mean)/(wn.sum()+kappa)
        plus=.5*(1+b@vp);minus=.5*(1+b@vn);zero=np.all(a==0)or wp.sum()==0 or wn.sum()==0
        if not zero:out[~fit]=np.log(np.maximum(plus,1e-12)/np.maximum(minus,1e-12))
        stats.append(dict(heldout_parity=parity,positive_weight=float(wp.sum()),negative_weight=float(wn.sum()),zero_contrast=bool(zero)))
    return out,stats

def linear_step(features,q0,kappa=64,beta=1):
    with threadpool_limits(limits=1,user_api='blas'):L,stats=linear_evidence(features,q0,kappa)
    q=update(q0,beta*inclusive_scores(L));return q,dict(L=L,prototypes=stats)
