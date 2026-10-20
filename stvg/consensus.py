"""One addition to frozen C: relative DINO candidate support, not a probability."""
import itertools
import numpy as np


def relative_cost(scores):
    s=np.asarray(scores,dtype=np.float64)
    assert s.ndim==1 and np.isfinite(s).all() and (s>=0).all() and (s<=1).all()
    if not len(s):return s.copy()
    s=np.maximum(s,1e-12)  # Numerical log floor; never deletes a low-score box.
    return np.log(s.max())-np.log(s)


def estimate(pools,scores,scale,cutoff,beta,diagnostics=True):
    scale=np.asarray(scale,float);assert scale.shape==(4,)and(scale>0).all()
    assert cutoff>0 and np.isfinite(cutoff)and beta>=0 and np.isfinite(beta)
    pools=[np.asarray(v,float).reshape(-1,4)for v in pools]
    costs=[relative_cost(v)for v in scores]
    assert len(pools)==len(costs)and all(len(v)==len(c)and len(v)<=3 and np.isfinite(v).all()for v,c in zip(pools,costs))
    if not pools or not any(len(v)for v in pools):
        return dict(delta=np.zeros(4),selected=[-1]*len(pools),latent_best=[-1]*len(pools),
            objective=len(pools)*cutoff,geometry_objective=len(pools)*cutoff,score_objective=0.,
            support_frames=0,rejected_frames=len(pools),assignments_evaluated=1,leave_one_delta_distances=[])
    proposals=[]
    for assign in itertools.product(*[range(-1,len(v))for v in pools]):
        active=[v[k]for v,k in zip(pools,assign)if k>=0]
        delta=np.mean(active,axis=0)if active else np.zeros(4)
        geom=[np.square((v-delta)/scale).sum(1)for v in pools]
        joint=[g+beta*cutoff*c for g,c in zip(geom,costs)]
        latent=tuple(int(g.argmin())if len(g)else -1 for g in joint)
        selected=tuple(k if k>=0 and g[k]<cutoff else -1 for k,g in zip(latent,joint))
        objective=float(sum(min(cutoff,float(g.min()))if len(g)else cutoff for g in joint))
        proposals.append((objective,selected,tuple(assign),delta,latent))
    proposals.sort(key=lambda v:(v[0],v[1],v[2]));value,selected,_,delta,latent=proposals[0]
    active=[v[k]for v,k in zip(pools,selected)if k>=0]
    assert active and np.max(abs(delta-np.mean(active,axis=0)))<1e-10
    geometry=float(sum(np.square((v[k]-delta)/scale).sum()if k>=0 else cutoff for v,k in zip(pools,selected)))
    semantic=float(sum(beta*cutoff*c[k]for c,k in zip(costs,selected)if k>=0))
    assert abs(geometry+semantic-value)<1e-8
    loo=[]
    if diagnostics:
        for i in range(len(pools)):
            sub=estimate(pools[:i]+pools[i+1:],scores[:i]+scores[i+1:],scale,cutoff,beta,False)
            loo.append(float(np.linalg.norm((sub['delta']-delta)/scale)))
    return dict(delta=delta,selected=list(selected),latent_best=list(latent),objective=value,
        geometry_objective=geometry,score_objective=semantic,support_frames=len(active),
        rejected_frames=len(pools)-len(active),assignments_evaluated=len(proposals),leave_one_delta_distances=loo)






def encode(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    assert b.ndim == 2 and b.shape[1] == 4 and np.isfinite(b).all()
    assert (b[:, 2:] > 0).all()
    return np.column_stack((b[:, :2], np.log(b[:, 2:])))


def decode(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.column_stack((z[:, :2], np.exp(z[:, 2:])))
    assert np.isfinite(out).all() and (out[:, 2:] > 0).all()
    return out



def apply(boxes, delta):
    b=np.asarray(boxes,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
    return b.copy() if np.array_equal(delta,np.zeros(4)) else decode(encode(b)+delta)
