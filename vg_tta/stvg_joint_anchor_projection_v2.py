"""Oracle diagnostic only: minimal current-knot correction, no learned module."""
import numpy as np
import torch
from scipy.optimize import minimize

def interpolation_matrix(state,z,positions):
 from vg_tta.stvg_joint_posterior_v1 import bounds
 ab=bounds(state[None])[0].detach().cpu().numpy();ids=np.asarray(z['frame_ids']);lo,hi=int(ids[0]),int(ids[-1])+1;a,b=ab;a=lo+a*(hi-lo);b=lo+b*(hi-lo);tt=np.linspace(a,max(a+1e-4,b-1),32)
 W=np.stack([np.interp(ids[positions],tt,np.eye(32)[:,j])for j in range(32)],-1)if positions else np.zeros((0,32))
 return W

def project_anchors(state,W,target):
 """Minimum L2 box-probability change on the existing 32 knots.

 Only knots touching observed physical frames can change. Solve an affine
 equality and use bounded convex SLSQP only if the closed-form solution leaves
 (1e-6,1-1e-6). No unobserved GT participates. Time is unchanged exactly.
 """
 if len(W)==0:return state.clone(),dict(error=0.,references=0,changed_knots=0,bounded_coordinates=0)
 target=np.asarray(target,dtype=np.float64);nodes=np.flatnonzero((W!=0).any(0));A=W[:,nodes];assert np.linalg.matrix_rank(A)==len(W)
 p=state[1:].sigmoid().detach().cpu().numpy().astype(np.float64);v=p[nodes].copy();bounded=0
 assert (target>1e-6).all()and(target<1-1e-6).all(),'oracle outside finite-logit representable interior'
 for j in range(4):
  raw=v[:,j];y=target[:,j];ans=raw+A.T@np.linalg.solve(A@A.T,y-A@raw)
  if np.min(ans)<1e-6 or np.max(ans)>1-1e-6:
   q=minimize(lambda a:.5*np.square(a-raw).sum(),np.clip(ans,1e-6,1-1e-6),jac=lambda a:a-raw,bounds=[(1e-6,1-1e-6)]*len(nodes),constraints=[dict(type='eq',fun=lambda a:A@a-y,jac=lambda a:A)],method='SLSQP',options=dict(ftol=1e-12,maxiter=200));assert q.success,q.message;ans=q.x;bounded+=1
  assert np.max(np.abs(A@ans-y))<2e-7;v[:,j]=ans
 out=state.clone();t=torch.as_tensor(v,device=state.device,dtype=state.dtype);out[1+torch.as_tensor(nodes,device=state.device)]=torch.logit(t)
 error=float(np.max(np.abs(W@out[1:].sigmoid().detach().cpu().numpy()-target)));assert error<2e-6
 outside=np.flatnonzero(~(W!=0).any(0));assert torch.equal(out[1+torch.as_tensor(outside,device=state.device)],state[1+torch.as_tensor(outside,device=state.device)])
 assert torch.equal(out[0],state[0])
 return out,dict(error=error,references=len(W),changed_knots=len(nodes),bounded_coordinates=bounded)

def propagate(net,c,z,W,target):
 """Deterministic propagation stress test, NOT calibrated posterior sampling.

 Native start, no new noise. Keep the old .213 to .03/16 VE algebra; anchor-
 conditioned arm projects the initial/clean/proposed states at every step.
 """
 import math
 sigma=torch.logspace(math.log10(.213),math.log10(.03),16,device=c['native'].device)
 x,initial=project_anchors(c['native'][0],W,target);trace=[]
 for i,s in enumerate(sigma):
  with torch.no_grad():clean=net(x[None],s[None],c)[0]
  clean[0]=c['native'][0,0];before=clean.clone();clean,pr=project_anchors(clean,W,target);sn=sigma[i+1]if i+1<len(sigma)else s*0
  proposal=clean+(sn/s)*(x-clean);proposal[0]=c['native'][0,0];x,pp=project_anchors(proposal,W,target)
  assert torch.isfinite(x).all()and torch.equal(x[0],c['native'][0,0])
  trace.append(dict(level=i,sigma=float(s),clean_anchor_correction_L2=float((clean-before).norm()),clean_projection=pr,proposal_projection=pp))
 return dict(final=x.cpu(),trace=trace,initial_projection=initial,initialized_with_added_noise=False)
