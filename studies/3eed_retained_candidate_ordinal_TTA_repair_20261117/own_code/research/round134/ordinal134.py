"""Frozen AB ordinal evidence and same-information constrained direct readout.

No target annotation, model confidence filter, unique positive pseudo-label,
or query identity transfer between descriptions is used here.
"""
import numpy as np,hashlib
from scipy.optimize import minimize,LinearConstraint
def groupmax(prob,groups):return np.array([np.log(max(float(np.max(prob[g])),1e-300)) for g in groups])
def evidence(pA,pB,p0,groups,C,kappa):
 a,b,z=[groupmax(p,groups) for p in [pA,pB,p0]]
 gap=np.minimum(a[:,None]-a[None,:],b[:,None]-b[None,:])
 win,lose=np.where((gap>1e-10)&C[:,None]&C[None,:]&(z[:,None]<z[None,:]))
 return win.astype(np.int64),lose.astype(np.int64),np.minimum(gap[win,lose],kappa)
def wrong_pairs(win,lose,groups,C,salt):
 rng=np.random.default_rng(int(hashlib.sha256((salt+'|R134_ordinal').encode()).hexdigest()[:16],16));perm=np.arange(len(groups))
 for n in sorted(set(map(len,groups))):
  ids=np.array([i for i,g in enumerate(groups) if len(g)==n and C[i]],int);order=rng.permutation(ids)
  if len(order)>1:perm[order]=np.roll(order,1)
 return perm[win],perm[lose],perm
def project(prob,groups,win,lose,margin):
 """Least-change group-max log-score projection; all query offsets replayed.

Hard pair constraints make this a strong direct-use control. It is not claimed
to be the same optimizer as a three-step shared neural parameter update.
"""
 if len(win)==0:return np.asarray(prob).copy(),{'pairs':0,'iterations':0,'max_violation':0.,'group_shift_L2':0.}
 nodes=np.unique(np.r_[win,lose]);index={g:i for i,g in enumerate(nodes)};D=np.zeros((len(win),len(nodes)))
 for i,(w,l) in enumerate(zip(win,lose)):D[i,index[w]]=1;D[i,index[l]]=-1
 z=groupmax(prob,groups);x0=z[nodes];fun=lambda x:.5*float(np.dot(x-x0,x-x0));jac=lambda x:x-x0
 ans=minimize(fun,x0,jac=jac,constraints=[LinearConstraint(D,margin,np.inf)],method='SLSQP',options={'ftol':1e-11,'maxiter':2000})
 violation=float(max(0.,np.max(margin-D@ans.x)));assert ans.success and violation<1e-7,(ans.message,violation)
 delta=ans.x-x0;assert abs(delta.sum())<1e-6
 lp=np.log(np.maximum(np.asarray(prob,float),1e-300))
 for g,v in zip(nodes,delta):lp[groups[g]]+=v
 lp-=lp.max();p=np.exp(lp);p/=p.sum();assert np.isfinite(p).all()
 return p,{'pairs':len(win),'iterations':int(ans.nit),'max_violation':violation,'group_shift_L2':float(np.linalg.norm(delta))}
