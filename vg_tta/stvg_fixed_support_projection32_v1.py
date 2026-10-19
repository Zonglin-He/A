"""Certified convex I-projection on saved finite candidates, no new teacher calls."""
import numpy as np
from scipy.special import logsumexp

def project(d,q,valid,start,tol=1e-10,maxiter=120):
 d=np.asarray(d,dtype=np.float64);q=np.asarray(q,dtype=np.float64);v=np.asarray(valid,bool);u=np.asarray(start,dtype=np.float64);assert v.any()and np.isfinite(d[v]).all()and np.isfinite(u).all();D=d[v];Q=q[v];assert abs(Q.sum()-1)<1e-12 and(Q>0).all()
 _,s,V=np.linalg.svd(D-D[0],full_matrices=False);rank=int((s>1e-12*max(1.,s[0]if len(s)else 0.)).sum());V=V[:rank];null=u-V.T@(V@u);a=D@V.T;h=-.5*(D*D).sum(1)+D@null;target=Q@a;z=V@u
 def evaluate(z):
  logit=h+a@z;logp=logit-logsumexp(logit);p=np.exp(logp);mean=p@a;g=mean-target;center=a-mean;H=(center*p[:,None]).T@center;f=float(np.sum(Q*(np.log(Q)-logp)));return f,g,H,p
 initial=evaluate(z)[0];backtracks=0
 for it in range(maxiter+1):
  f,g,H,prob=evaluate(z)
  if not rank or np.linalg.norm(g,np.inf)<=tol:break
  assert it<maxiter,('unconverged',rank,float(np.linalg.norm(g,np.inf)))
  ev,U=np.linalg.eigh(H);direction=-(U/np.maximum(ev,1e-14))@(U.T@g);descent=float(g@direction);assert descent<0
  step=1.
  while evaluate(z+step*direction)[0]>f+1e-4*step*descent+64*np.finfo(float).eps*max(1.,abs(f)):
   step*=.5;backtracks+=1;assert step>2**-45,'line search failed'
  z=z+step*direction
 mu=null+V.T@z;final,g,H,prob=evaluate(z);assert final<=initial+1e-12;assert not rank or np.linalg.norm(g,np.inf)<=tol
 # Physical readout may extrapolate outside finite candidate hull: diagnostic only.
 lower=D.min(0);upper=D.max(0);outside=bool(((mu<lower-1e-10)|(mu>upper+1e-10)).any())
 return dict(mu=mu,rank=rank,iterations=it,backtracks=backtracks,KL_at_start=initial,KL_optimum=final,removable_KL=max(0.,initial-final),projected_gradient_inf=float(np.linalg.norm(g,np.inf))if rank else 0.,null_error=float(np.linalg.norm(mu-V.T@(V@mu)-null)),outside_candidate_axis_range=outside,minimum_Hessian_eigenvalue=float(np.linalg.eigvalsh(H).min())if rank else None)

def properties():
 d=np.array([[-1.,0.],[1.,0.]]);q=np.array([.01,.99]);r=project(d,q,[True,True],[0.,3.]);assert abs(r['mu'][0]-np.log(99)/2)<1e-8 and r['mu'][1]==3. and r['outside_candidate_axis_range']
 z=project(np.array([[2.,4.],[2.,4.]]),[.2,.8],[True,True],[3.,5.]);assert z['rank']==0 and np.array_equal(z['mu'],[3.,5.]);assert abs(z['removable_KL'])<1e-12
 return dict(status='pass',rank_from_affine_differences=True,null_component_preserved=True,rank0_keeps_start=True,two_point_analytic_solution=True,extrapolation_explicit=True,no_teacher_calls=True)
