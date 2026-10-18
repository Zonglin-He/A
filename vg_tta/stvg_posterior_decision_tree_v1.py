"""Exact final-envelope decisions and four-frame coverage; no GT/data readers."""
import numpy as np

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

def brute_utility(ids,q,decisions=None):
 s,e=np.triu_indices(len(ids),1);ids=np.asarray(ids);iv=np.stack((ids[s],ids[e]+1),1);prob=q[s,e]
 js=np.arange(len(iv))if decisions is None else np.asarray(decisions)
 out=[]
 for j in js:
  a,b=iv[j];inter=np.maximum(0,np.minimum(b,iv[:,1])-np.maximum(a,iv[:,0]));union=b-a+iv[:,1]-iv[:,0]-inter;out.append(float(np.dot(prob,inter/union)))
 return np.array(out)

def final_mbr(ids,q):
 u=expected_iou_fast(ids,q);k=int(np.argmax(u));a,b=divmod(k,len(ids));return dict(interval=[int(ids[a]),int(ids[b])+1],indices=[a,b],utility=float(u[a,b]),utility_grid=u)

def coverage_value(q,selected):
 n=len(q);a,b=np.indices((n,n));hits=sum((a<=x)&(b>=x)for x in selected)
 return float((q*np.minimum(1,hits/2)).sum())

def greedy4(ids,q):
 ids=np.asarray(ids);n=len(ids);a,b=np.triu_indices(n,1);prob=q[a,b];member=(np.arange(n)[:,None]>=a)&(np.arange(n)[:,None]<=b)
 hits=np.zeros(len(a),dtype=np.int64);selected=[];trace=[]
 for k in range(4):
  # Canonical scalar-order float64 summation, ties are exact equality, no epsilon search.
  gains=np.sum(member*(.5*prob*(hits<2))[None],axis=1);gains[selected]=-np.inf
  pool=np.flatnonzero(gains==gains.max()).tolist()
  if selected:chosen=min(pool,key=lambda j:(-min(abs(int(ids[j])-int(ids[x]))for x in selected),int(ids[j])))
  else:chosen=min(pool,key=lambda j:(abs(2*int(ids[j])-int(ids[0])-int(ids[-1])-1),int(ids[j])))
  old=coverage_value(q,selected);selected.append(chosen);hits+=member[chosen];new=coverage_value(q,selected);assert abs(new-old-gains[chosen])<1e-12
  trace.append(dict(position=int(chosen),marginal=float(gains[chosen]),model_coverage=new,tied_candidates=len(pool)))
 return sorted(selected),trace

def contract():
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 import torch
 rng=np.random.default_rng(20261018);checks=0;error=0.
 for n in [4,5,7,10,15]:
  ids=np.cumsum(rng.integers(1,8,n)).tolist();records=[dict(frame_ids=ids[o::2])for o in [0,1]];zs=[torch.tensor(rng.normal(size=(1,len(r['frame_ids']),2)),dtype=torch.float32)for r in records]
  g=probabilities(records,zs);grid,q=mass_table(g);cdf=independent_cdf(g,grid);assert abs(cdf-q).max()<1e-14
  actual=expected_iou_fast(ids,q);s,e=np.triu_indices(n,1);expected=brute_utility(ids,q);err=float(abs(actual[s,e]-expected).max());assert err<1e-12,err;error=max(error,err);checks+=len(expected)
  S,trace=greedy4(ids,q);assert len(S)==len(set(S))==4
  # Independent all-frame greedy gains using direct physical interval membership.
  selected=[]
  for k,step in enumerate(trace):
   gains=[]
   for f in range(n):gains.append(-np.inf if f in selected else coverage_value(q,selected+[f])-coverage_value(q,selected))
   assert abs(gains[step['position']]-max(gains))<1e-12;selected.append(step['position']);checks+=n
 return dict(status='pass',synthetic_direct_utility_checks=checks,max_utility_error=error,outer_product_CDF_exact=True,all_decisions_including_zero_mass=True,greedy_marginals_direct=True)
