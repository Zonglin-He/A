"""Observation-only factorial, no fitting, GT, new networks or IoU reward."""
import numpy as np


def dense_ids(ids):
    ids=list(map(int,ids));assert ids==sorted(set(ids)) and len(ids)>=4
    return sorted(set(ids+[(a+b)//2 for a,b in zip(ids[:-1],ids[1:])]))


def crop_bounds(box,width,height):
    b=np.asarray(box,dtype=np.float64);assert b.shape==(4,) and np.isfinite(b).all() and (b[2:]>0).all()
    # 2x context in both axes; integer raster bounds are part of this protocol.
    lo=np.floor((b[:2]-b[2:])*[width,height]).astype(int)
    hi=np.ceil((b[:2]+b[2:])*[width,height]).astype(int)
    lo=np.maximum(lo,0);hi=np.minimum(hi,[width,height])
    return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]


def lift_boxes(boxes,bounds,width,height):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4);x,y,r,d=bounds
    assert 0<=x<r<=width and 0<=y<d<=height
    scale=np.array([(r-x)/width,(d-y)/height])
    return np.concatenate((b[:,:2]*scale+[x/width,y/height],b[:,2:]*scale),1)


def envelope(records,logits):
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
    ids=sorted({int(x)for r in records for x in r['frame_ids']})
    raw=independent_cdf(probabilities(records,logits),ids)
    legal=np.triu(np.ones(raw.shape,bool),1)
    assert np.isfinite(raw).all() and raw.min()>-1e-12
    assert np.abs(raw[~legal]).sum()<1e-10 and abs(raw.sum()-1)<1e-10
    q=np.where(legal,np.maximum(raw,0.),0.);q/=q.sum()
    return ids,q,dict(CDF_roundoff_L1=float(np.abs(q-raw).sum()),raw_negative_cells=int((raw<0).sum()))


def embed(ids,q,newids):
    lookup={v:i for i,v in enumerate(newids)};at=np.array([lookup[i]for i in ids])
    out=np.zeros((len(newids),len(newids)));out[np.ix_(at,at)]=q
    assert abs(out.sum()-1)<1e-12
    return out


def properties():
    import torch
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    rng=np.random.default_rng(20261019);count=0;maximum=0.
    for n in [5,8,11,16]:
        ids=np.cumsum(rng.integers(1,5,n)).tolist();records=[dict(frame_ids=ids[o::2])for o in [0,1]]
        z=[torch.tensor(rng.normal(size=(1,len(r['frame_ids']),2)),dtype=torch.float32)for r in records]
        u,q,stats=envelope(records,z);v,m=mass_table(probabilities(records,z));assert u==v
        maximum=max(maximum,float(np.abs(q-m).max()));assert np.allclose(q,m,rtol=0,atol=1e-12)
        new=dense_ids(ids);g=embed(ids,q,new);b=final_mbr(new,g);s,e=np.triu_indices(len(new),1)
        k=int(np.flatnonzero((s==b['indices'][0])&(e==b['indices'][1]))[0])
        assert abs(brute_utility(new,g,[k])[0]-b['utility'])<1e-12;count+=len(s)
    b=np.array([[.5,.5,.2,.3],[.2,.7,.1,.1]]);bounds=[20,30,180,130]
    lifted=lift_boxes(b,bounds,200,160)
    back=np.concatenate(((lifted[:,:2]-[.1,30/160])/[.8,100/160],lifted[:,2:]/[.8,100/160]),1)
    assert np.allclose(back,b,atol=1e-15,rtol=0)
    assert crop_bounds([.5,.5,1.,1.],200,160)==[0,0,200,160]
    assert dense_ids([0,2,5,8])==[0,1,2,3,5,6,8]
    return dict(status='pass',CDF_vs_outer_max_error=maximum,synthetic_decisions=count,
                physical_mass_embedding=True,crop_inverse=True,no_GT=True)
