"""Exact common-quantile coupling of physical-midpoint ordered interval atoms."""
import numpy as np

def couple(q,r,ids,reverse=False):
 n=len(ids);legal=np.triu(np.ones((n,n),bool),1);assert q.shape==r.shape==(n,n)
 def atoms(x):
  assert (x>=0).all()and not x[~legal].any()and abs(x.sum()-1)<1e-10
  i,j=np.nonzero(x);mid=np.asarray(ids)[i]+np.asarray(ids)[j]+1;order=np.lexsort((j,i,mid));i=i[order];j=j[order];p=x[i,j]/x.sum();return i,j,p
 i,j,p=atoms(q);k,l,w=atoms(r)
 if reverse:k,l,w=k[::-1],l[::-1],w[::-1]
 a=np.cumsum(p);b=np.cumsum(w);a[-1]=b[-1]=1.;cuts=np.unique(np.r_[0.,a,b]);weights=np.diff(cuts);left=np.searchsorted(a,cuts[:-1],side='right');right=np.searchsorted(b,cuts[:-1],side='right');assert (left<len(a)).all()and(right<len(b)).all()
 out=np.zeros_like(q);np.add.at(out,(np.minimum(i[left],k[right]),np.maximum(j[left],l[right])),weights)
 errq=float(abs(np.bincount(left,weights,minlength=len(p))-p).max());errr=float(abs(np.bincount(right,weights,minlength=len(w))-w).max());assert max(errq,errr)<1e-12 and abs(out.sum()-1)<1e-12
 bridge=float(weights[(j[left]<k[right])|(l[right]<i[left])].sum())
 return out,dict(first_atoms=len(p),last_atoms=len(w),segments=len(weights),marginal_error_first=errq,marginal_error_last=errr,nonoverlapping_pair_mass=bridge,reverse_rank=reverse)
