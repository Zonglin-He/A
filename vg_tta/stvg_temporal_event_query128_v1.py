"""Student-time/target-conditioned query, frozen full-context temporal decoder."""
import numpy as np
import torch
from vg_tta.stvg_object_condition32_v1 import region_weights


def occupancy(q):
    q=np.asarray(q,dtype=np.float64);n=len(q)
    assert q.shape==(n,n)and np.all(q>=0)and abs(q.sum()-1)<1e-12
    return np.array([q[:t+1,t:].sum()for t in range(n)])


def sampled_states(q0,q1,ordinal,count=4):
    n=len(q0);ii,jj=np.triu_indices(n,1);u=np.random.default_rng(np.random.SeedSequence([41,int(ordinal)])).random(count)
    pairs=[]
    for q in [q0,q1]:
        cdf=np.cumsum(q[ii,jj]);cdf[-1]=1.
        k=np.searchsorted(cdf,u,side='right');pairs.append(np.stack([ii[k],jj[k]],-1))
    return u,pairs


def coefficients(model,kw,g,boxes):
    h,w=g['h'],g['w'];l=h*w;f=kw['encoded_feature'];H=f[-l:].permute(1,0,2);T=len(H)
    text=f[:-l].mean(1).unsqueeze(0)
    _,a=model.t_spatial_clas(H.permute(0,2,1).reshape(T,256,h,w),text[:,:1])
    assert a.shape==(T,l)and torch.isfinite(a).all()
    v=g['valid'].to(a.device);r,c,empty=region_weights(a,boxes.to(a.device),v,h,w)
    assert torch.isfinite(r).all()
    return H,dict(Global=a*v,Refer=r),dict(global_attention=(a*v).detach().cpu(),refer_attention=r.detach().cpu(),coverage=c.detach().cpu(),empty=empty)


def query(H,attention,time_weights,native):
    w=torch.as_tensor(time_weights,device=H.device,dtype=torch.float64)
    assert w.shape==(len(H),)and torch.isfinite(w).all()and(w>=0).all()
    a=attention.double()*w[:,None];den=a.sum();norm=native.double().norm()
    if den<=0 or norm<=0:
        return native,dict(noop=True,reason='zero mass or native norm')
    raw=(H.double()*a[...,None]).sum((0,1))/den
    if not torch.isfinite(raw).all()or raw.norm()<=0:
        return native,dict(noop=True,reason='zero or nonfinite query')
    q=(raw*(norm/raw.norm())).to(native)
    assert abs(float(q.double().norm()-norm))<2e-6*max(1,float(norm))
    return q,dict(noop=False,time_weights=w.detach().cpu(),query=q.detach().cpu(),native_norm=float(norm),raw_norm=float(raw.norm()),delta_norm=float((q-native).norm()))


def properties():
    p=np.zeros((6,6));p[1,4]=1;assert np.array_equal(occupancy(p),[0,1,1,1,1,0])
    rng=np.random.default_rng(41);p=np.triu(rng.random((6,6)),1);p/=p.sum();direct=np.array([sum(p[i,j]for i in range(6)for j in range(i+1,6)if i<=t<=j)for t in range(6)]);assert np.max(abs(occupancy(p)-direct))<1e-14
    H=torch.arange(3*4*5).double().reshape(3,4,5);a=torch.ones(3,4);native=torch.ones(5)
    q,_=query(H,a,[1,1,1],native);q2,_=query(H,a,[2,2,2],native);assert torch.equal(q,q2)
    z,meta=query(H,a,[0,0,0],native);assert torch.equal(z,native)and meta['noop']
    return dict(status='pass',occupancy_independent_enumeration=True,positive_time_scale_invariant=True,query_native_norm_matched=True,zero_mass_native_exact=True)
