"""Context evidence with crop-censored endpoints; no labels or learned gates."""
import numpy as np


def log_ratio(q,u):
    q=np.asarray(q,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
    assert q.shape==u.shape and np.isfinite(q).all() and np.isfinite(u).all()
    assert (q>=0).all() and (u>=0).all() and not np.any((q>0)&(u==0))
    out=np.zeros_like(q);take=u>0
    out[take]=np.log(np.maximum(q[take],np.finfo(np.float64).tiny))-np.log(u[take])
    return out


def censored_evidence(global_ids,window_ids,q,u):
    """Unknown endpoints carry no negative evidence. Strict interior is observable.

    q/u use exactly the same physical support and native offset hull. Marginalize
    an unavailable boundary rather than clipping it onto a crop edge.
    """
    ids=list(map(int,global_ids));w=list(map(int,window_ids));n=len(ids)
    assert len(w)>3 and all(x in ids for x in w) and w==sorted(w)
    q=np.asarray(q);u=np.asarray(u);assert q.shape==u.shape==(len(w),len(w))
    joint=log_ratio(q,u);start=log_ratio(q.sum(1),u.sum(1));end=log_ratio(q.sum(0),u.sum(0))
    wi={x:i for i,x in enumerate(w)};visible=set(w[1:-1]);score=np.zeros((n,n));counts=dict(both=0,start=0,end=0,unknown=0)
    for a in range(n):
        for b in range(a+1,n):
            x,y=ids[a],ids[b];sx=x in visible;ey=y in visible
            if sx and ey and u[wi[x],wi[y]]>0:score[a,b]=joint[wi[x],wi[y]];counts['both']+=1
            elif sx and not ey and u.sum(1)[wi[x]]>0:score[a,b]=start[wi[x]];counts['start']+=1
            elif ey and not sx and u.sum(0)[wi[y]]>0:score[a,b]=end[wi[y]];counts['end']+=1
            else:counts['unknown']+=1
    assert np.isfinite(score).all()
    return score,counts
