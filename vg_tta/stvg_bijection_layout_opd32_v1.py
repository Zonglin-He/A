"""Frozen nine-slot ROI patch bijection; no learned matcher or new model."""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from vg_tta.stvg_roi_layout_opd32_v1 import patches,build
ARMS=['E_MBR','Bijection_readout','Diagonal_readout','Bijection_fixed','Bijection_live','Bijection_shuffle','Diagonal_live','Previous_independent','Previous_joint']
FIT_ARMS=['Bijection_fixed','Bijection_live','Bijection_shuffle','Diagonal_live']

@torch.no_grad()
def feedback(context,b,eps,tau,diagonal=False,permutation=None):
 maps,valid=context['maps'],context['valid']
 if permutation is not None:maps,valid=maps[permutation],valid[permutation]
 ph,w=patches(maps,valid,b);n,k=b.shape[:2];ok=w.sum(-1)>1e-6;safe=ok.clone();safe[~ok.any(1)]=True;lp=(-.5*eps.square().sum(-1)).masked_fill(~safe,-torch.inf);prior=lp.softmax(-1)
 if context['positions']:
  t=context['templates'];tw=context['weights'];cross=torch.einsum('nkjc,aic->ankji',ph,t)*w[None,:,:,:,None]*tw[:,None,None,None,:]
  if diagonal:per=cross.diagonal(dim1=-2,dim2=-1).sum(-1)/9;assignment=None
  else:
   cc=cross.numpy().reshape(-1,9,9);score=np.empty(len(cc));assign=np.empty((len(cc),9),dtype=np.int16)
   for j,c in enumerate(cc):
    rr,col=linear_sum_assignment(c,maximize=True);assert np.array_equal(rr,np.arange(9));score[j]=c[rr,col].sum()/9;assign[j]=col
   per=torch.from_numpy(score.reshape(cross.shape[:3]));assignment=torch.from_numpy(assign.reshape(*cross.shape[:3],9));diag=cross.diagonal(dim1=-2,dim2=-1).sum(-1)/9;assert (per>=diag-1e-12).all()
  r=per.mean(0);hi=r.masked_fill(~ok,-torch.inf).amax(-1);lo=r.masked_fill(~ok,torch.inf).amin(-1);info=ok.any(1)&(hi-lo>1e-6);pt=(lp+r/tau).softmax(-1);pt[~info]=prior[~info]
 else:
  cross=torch.empty(0,n,k,9,9,dtype=torch.float64);per=torch.empty(0,n,k,dtype=torch.float64);assignment=None;r=torch.zeros(n,k,dtype=torch.float64);info=torch.zeros(n,dtype=torch.bool);pt=prior.clone()
 return dict(prob=pt,prior=prior,rewards=r,per_anchor_rewards=per,assignment=assignment,valid=ok,informative=info,visible_bin_mass=w.sum(-1),fixed_nine_slot_denominator=True)

def candidates(E,amp,eta,eps,scale):
 b=E[:,None].expand(-1,eps.shape[1],-1).clone();b[:,:,:2]+=scale*E[:,None,2:]*amp[:,None,None]*(eta[:,None]+eps);return b

def direct(E,amp,context,eps,cfg,diagonal=False):
 ee=torch.cat(tuple(eps),1);b=candidates(E,amp,torch.zeros(len(E),2,dtype=torch.float64),ee,cfg['center_scale']);f=feedback(context,b,ee,cfg['temperature'],diagonal);use=(amp>0)&f['informative'];best=f['prob'].argmax(-1);out=E.clone();out[use]=b[torch.arange(len(E))[use],best[use]];return dict(boxes=out,candidates=b,feedback=f,best=best)

def fit(E,amp,context,eps,cfg,mode,permutation):
 eta=torch.zeros(len(E),2,dtype=torch.float64);path=[dict(state=eta.clone(),boxes=E.clone())];trace=[]
 for step in range(2):
  center=torch.zeros_like(eta)if mode=='Bijection_fixed'else eta.clone();ee=eps[step];d=center[:,None]+ee;b=candidates(E,amp,center,ee,cfg['center_scale']);f=feedback(context,b,ee,cfg['temperature'],mode=='Diagonal_live',permutation if mode=='Bijection_shuffle'else None);safe=f['valid'].clone();safe[~safe.any(1)]=True;lp=(-.5*(d-eta[:,None]).square().sum(-1)).masked_fill(~safe,-torch.inf);sp=lp.softmax(-1);g=((sp-f['prob'])[...,None]*d).sum(1);g[~((amp>0)&f['informative'])]=0;new=eta-cfg['step_size']*g;out=candidates(E,amp,new,torch.zeros(len(E),1,2,dtype=torch.float64),cfg['center_scale'])[:,0]
  trace.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=g,before=eta.clone(),after=new.clone()));eta=new;path.append(dict(state=eta.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=trace,final=path[-1]['boxes'],selected_step=2,GT_read=False)

def properties():
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 eye=torch.eye(9).T.reshape(9,3,3);permuted=eye.reshape(9,9)[:,torch.arange(9).roll(1)].reshape(9,3,3);repeat=eye[:,0:1,0:1].expand(9,3,3);m=torch.stack((eye,permuted,repeat));v=torch.ones(3,3,3,dtype=torch.bool);E=torch.tensor([[.5,.5,1.,1.]]*3,dtype=torch.float64);c=build(m,v,[0],E[:1]);bb=E[:,None];eps=torch.zeros(3,1,2,dtype=torch.float64);f=feedback(c,bb,eps,.1);d=feedback(c,bb,eps,.1,True);assert abs(f['rewards'][0,0]-1)<1e-6 and abs(f['rewards'][1,0]-1)<1e-6;assert d['rewards'][1,0]<.01;assert abs(f['rewards'][2,0]-1/9)<1e-6
 cfg=dict(center_scale=.25,temperature=.1,step_size=.5);E=torch.tensor([[.5,.5,.2,.2]]*3,dtype=torch.float64);empty=build(m,v,[],[]);ee=noise('bijection-properties',3);amp=torch.tensor([0.,.5,1.],dtype=torch.float64)
 for mode in FIT_ARMS:assert torch.equal(fit(E,amp,empty,ee,cfg,mode,[0,2,1])['final'],E)
 flat=build(torch.ones_like(m),v,[0],E[:1]);assert torch.equal(fit(E,amp,flat,ee,cfg,'Bijection_live',[0,2,1])['final'],E)
 live=fit(E,amp,c,ee,cfg,'Bijection_live',[0,2,1]);fixed=fit(E,amp,c,ee,cfg,'Bijection_fixed',[0,2,1]);assert torch.equal(live['path'][1]['boxes'],fixed['path'][1]['boxes']);assert torch.equal(live['final'][amp==0],E[amp==0]);return dict(status='pass',permutation_invariance=True,duplicate_patch_cannot_explain_full_template=True,assignment_dominates_matched_diagonal=True,no_reference_exact_noop=True,constant_inside_feature_noop=True,first_fixed_live_exact=True,zero_amplitude_noop=True)
