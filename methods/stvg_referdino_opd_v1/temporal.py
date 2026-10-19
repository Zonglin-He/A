"""Exact physical-interval MAP from actual full-grid FP32 endpoint logits."""
import numpy as np
from scipy.special import logsumexp

def probabilities(records,logits):
    grids=[]
    for rec,z in zip(records,logits):
        if hasattr(z,'detach'):z=z.detach().cpu().numpy()
        z=np.asarray(z);assert z.dtype==np.float32 and z.shape==(1,len(rec['frame_ids']),2)and np.isfinite(z).all();v=z[0].astype(np.float64);ij=np.asarray(np.triu_indices(len(v),1));score=v[ij[0],0]+v[ij[1],1];p=np.exp(score-logsumexp(score));assert abs(float(p.sum())-1)<1e-12;grids.append(dict(ids=list(map(int,rec['frame_ids'])),ij=ij,p=p))
    assert len(grids)==2;return grids

def mass_table(grids):
    ids=sorted({x for g in grids for x in g['ids']});idx={x:i for i,x in enumerate(ids)};n=len(ids);s=[];e=[]
    for g in grids:s.append(np.array([idx[g['ids'][i]]for i in g['ij'][0]]));e.append(np.array([idx[g['ids'][i]]for i in g['ij'][1]]))
    key=np.minimum(s[0][:,None],s[1][None,:])*n+np.maximum(e[0][:,None],e[1][None,:]);w=grids[0]['p'][:,None]*grids[1]['p'][None,:];M=np.bincount(key.ravel(),weights=w.ravel(),minlength=n*n).reshape(n,n);assert np.isfinite(M).all()and (M>=0).all()and abs(float(M.sum())-1)<1e-12;return ids,M


def expected_iou_fast(ids,q):
 """O(n^3) exact finite tIoU sum using four disjoint endpoint order regions.

All strict-index a<b decisions are included, even if their q is zero. No mass
truncation. Actual physical endpoints are start=fid, end=fid+1.
"""
 s=np.asarray(ids,dtype=np.float64);e=s+1;n=len(s);q=np.asarray(q,dtype=np.float64)
 assert q.shape==(n,n)and (q>=0).all()and abs(q.sum()-1)<1e-10 and (np.diff(s)>0).all()
 lengths=e[None,:]-s[:,None];legal=np.triu(np.ones((n,n),bool),1);assert not q[~legal].any()
 QL=q*np.maximum(lengths,0);H=np.divide(q,lengths,out=np.zeros_like(q),where=legal)
 # Rows >=a and columns <=b: contained hypothesis. Rows <a/columns >b: contains decision.
 inside=QL[::-1].cumsum(0)[::-1].cumsum(1);outer=H.cumsum(0)[:,::-1].cumsum(1)[:,::-1]
 rowQ=q.cumsum(1);rowE=(q*e[None]).cumsum(1);colQ=q.cumsum(0);colS=(q*s[:,None]).cumsum(0)
 u=np.full((n,n),-np.inf)
 for a in range(n-1):
  bs=np.arange(a+1,n);L=e[bs]-s[a];value=inside[a,bs]/L
  if a>0:
   value+=L*np.r_[outer[a-1,bs[:-1]+1],0.]
   lo=int(np.searchsorted(e,s[a],side='right'))
   bQ=rowQ[:a,bs]-(rowQ[:a,lo-1,None]if lo else 0)
   bE=rowE[:a,bs]-(rowE[:a,lo-1,None]if lo else 0)
   value+=((bE-s[a]*bQ)/(e[bs][None]-s[:a,None])).sum(0)
  # Rows a..b, columns >b. Integer physical IDs imply s[b+1]>=e[b].
  cq=colQ[bs]-(colQ[a-1][None]if a else 0);cs=colS[bs]-(colS[a-1][None]if a else 0)
  mask=np.arange(n)[None]>bs[:,None];num=e[bs,None]*cq-cs
  denom=e[None]-s[a];term=np.divide(num,denom,out=np.zeros_like(num),where=mask)
  value+=term.sum(1);u[a,bs]=value
 assert np.isfinite(u[legal]).all()
 return u

def final_mbr(ids,q):
 u=expected_iou_fast(ids,q);k=int(np.argmax(u));a,b=divmod(k,len(ids));return dict(interval=[int(ids[a]),int(ids[b])+1],indices=[a,b],utility=float(u[a,b]),utility_grid=u)

