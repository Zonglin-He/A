"""Compose first/last subevents with a fixed monotone endpoint relation."""
import numpy as np


def ordered(q,r):
    """Condition on start1<=start2 AND end1<=end2; overlap is allowed.

    Output[s,e]=sum_{end1<=e}q[s,end1]*sum_{start2>=s}r[start2,e].
    This equals enumerating all ordered interval pairs, then outputting[start1,end2].
    Return the valid-order mass as a diagnostic, never as an acceptance threshold.
    """
    q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64);assert q.shape==r.shape
    legal=np.triu(np.ones(q.shape,bool),1)
    for x in [q,r]:assert (x>=0).all()and not x[~legal].any()and abs(x.sum()-1)<1e-10
    out=q.cumsum(1)*r[::-1].cumsum(0)[::-1];mass=float(out.sum());assert mass>0 and mass<=1+1e-12
    out/=mass;assert not out[~legal].any();return out,mass


def roles(q,r):
    out=np.outer(q.sum(1),r.sum(0));out=np.triu(out,1);assert out.sum()>0;out/=out.sum();return out
