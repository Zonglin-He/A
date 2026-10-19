"""Fixed-support event tilt of the existing finite interval posterior. No labels."""
import numpy as np

def feedback(ids,edges,curve):
    ids=np.asarray(ids,float);edges=np.asarray(edges,float);n=len(ids)
    a,b=np.triu_indices(n,1);starts=ids[a];ends=ids[b]+1
    assert edges[0]==0 and starts.min()>=0 and ends.max()<=edges[-1]
    raw=np.zeros(len(a));z=np.zeros_like(raw);defined=np.zeros(len(a),bool)
    if curve is None:return np.zeros((n,n)),dict(active=False,reason='missing_feedback',mean=0.,std=0.,defined=0,full_video_increment=0.)
    x=np.asarray(curve,float);assert len(x)+1==len(edges)and np.isfinite(x).all()
    total=edges[-1];prefix=np.r_[0.,np.cumsum(np.diff(edges)*x)];inside=np.interp(ends,edges,prefix)-np.interp(starts,edges,prefix);length=ends-starts
    defined=(length>0)&(total-length>0)
    raw[defined]=inside[defined]/length[defined]-(prefix[-1]-inside[defined])/(total-length[defined])
    mean=float(raw[defined].mean())if defined.any()else 0.;std=float(raw[defined].std(ddof=0))if defined.any()else 0.
    active=bool(defined.any()and std>1e-12 and np.ptp(x)>1e-12)
    if active:z[defined]=(raw[defined]-mean)/std
    grid=np.zeros((n,n));grid[a,b]=z
    return grid,dict(active=active,reason='standardized'if active else'flat_feedback',mean=mean,std=std,defined=int(defined.sum()),undefined=int((~defined).sum()),full_video_increment=0.,standardization='unweighted population mean/std over all legal defined physical intervals, including original zero-mass decisions; not per posterior, seed, lambda or GT')

def tilt(q,z,lam):
    q=np.asarray(q,float);z=np.asarray(z,float);assert q.shape==z.shape and np.isfinite(z).all()
    assert np.isfinite(q).all()and(q>=0).all()and abs(q.sum()-1)<1e-10
    if lam==0 or not np.any(z):return q.copy()
    positive=q>0;v=np.log(q[positive])+float(lam)*z[positive];v=np.exp(v-v.max());v/=v.sum()
    out=np.zeros_like(q);out[positive]=v;assert abs(out.sum()-1)<1e-12
    return out

