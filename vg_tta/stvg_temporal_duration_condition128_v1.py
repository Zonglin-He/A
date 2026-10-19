"""Native node-count marginal preserving assimilation; no data/GT readers."""
import numpy as np
from scipy.special import logsumexp

def energy(curve):
 x=np.asarray(curve,dtype=np.float64);n=len(x);a,b=np.indices((n,n));s=np.r_[0.,np.cumsum(x)];return s[b+1]-s[a]

def posterior(q0,curve,conditional):
 q0=np.asarray(q0,dtype=np.float64);x=np.asarray(curve,dtype=np.float64);n=len(x);a,b=np.indices((n,n));mask=q0>0
 assert q0.shape==(n,n)and (q0>=0).all()and abs(q0.sum()-1)<1e-12 and not np.any(mask&(a>=b))and np.isfinite(x).all()
 # Exact no-op for constant evidence: within-count additive offset cancels.
 if conditional and np.all(x==x[0]):return q0.copy()
 e=energy(x);out=np.zeros_like(q0);lp=np.log(q0[mask])+e[mask]
 if not conditional:out[mask]=np.exp(lp-logsumexp(lp))
 else:
  count=b-a+1
  for k in range(2,n+1):
   h=mask&(count==k)
   if h.any():
    v=np.log(q0[h])+e[h];out[h]=q0[h].sum()*np.exp(v-logsumexp(v))
 assert np.isfinite(out).all()and np.array_equal(out>0,mask)and abs(out.sum()-1)<2e-13
 return out

def marginal(q):
 a,b=np.indices(q.shape);return np.bincount((b-a+1)[q>0],weights=q[q>0],minlength=len(q)+1)

def properties():
 rng=np.random.default_rng(20261020);err=0.;different=0
 for n in [4,7,12,65]:
  q=np.zeros((n,n));i,j=np.triu_indices(n,1);q[i,j]=rng.uniform(.01,1,len(i));q/=q.sum();x=rng.normal(size=n);t=posterior(q,x,True)
  assert np.array_equal(posterior(q,np.full(n,3.),True),q)
  for c in [-9.,4.]:err=max(err,float(abs(t-posterior(q,x+c,True)).max()))
  assert abs(marginal(t)-marginal(q)).max()<2e-13
  g=posterior(q,x,False)
  for k in range(2,n+1):
   h=(j-i+1)==k
   assert abs(t[i[h],j[h]]/t[i[h],j[h]].sum()-g[i[h],j[h]]/g[i[h],j[h]].sum()).max()<2e-13
  # Each group may carry arbitrary nuisance energy depending only on count.
  e=energy(x);weights=np.zeros_like(q)
  for k in range(2,n+1):
   h=(j-i+1)==k;v=np.log(q[i[h],j[h]])+e[i[h],j[h]]+k*k*.001
   weights[i[h],j[h]]=q[i[h],j[h]].sum()*np.exp(v-logsumexp(v))
  assert abs(weights-t).max()<2e-13
  assert abs(t[0,-1]-q[0,-1])<2e-13
  different+=int(abs(t-posterior(q,x,False)).max()>1e-6)
  ex=np.zeros_like(q)
  for k in range(2,n+1):
   h=(j-i+1)==k;v=q[i[h],j[h]]*np.exp([sum(x[u:v+1])for u,v in zip(i[h],j[h])]);ex[i[h],j[h]]=v/v.sum()*q[i[h],j[h]].sum()
  assert abs(ex-t).max()<2e-13
 assert err<2e-13 and different==4
 return dict(status='pass',native_marginal_preserved=True,flat_evidence_bitexact=True,additive_intercept_invariance_max_error=err,independent_scalar_kernel_cases=4,global_fusion_is_different=True,node_count_not_exact_physical_duration=True)
