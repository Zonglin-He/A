"""Fixed physical-time tent output state with dense frozen Layout feedback."""
import hashlib
import numpy as np
import torch
from vg_tta.stvg_roi_layout_opd32_v1 import feedback
ARMS=['E_MBR','Independent_scoped','Independent_projection','Previous_projection','Joint_direct','Joint_fixed','Joint_live','Joint_shuffle']
FIT_ARMS=['Joint_fixed','Joint_live','Joint_shuffle','Independent_scoped']

def basis(frame_ids,planned,interval):
 ids=torch.as_tensor(frame_ids,dtype=torch.float64);p=sorted(set(map(int,planned)));assert len(p)==4 and torch.all(ids[1:]>ids[:-1])
 B=torch.zeros(len(ids),3,dtype=torch.float64)
 for j,(a,b)in enumerate(zip(p[:-1],p[1:])):
  u=(ids-ids[a])/(ids[b]-ids[a]);B[:,j]=(1-(2*u-1).abs()).clamp_min(0)
 left=torch.cat((ids[:1],ids[:-1]));right=torch.cat((ids[1:],ids[-1:]));scope=(right>=interval[0])&(left<interval[1]);scope[p]=False;B*=scope[:,None]
 return B,scope

def noise(alias,active):
 seed=int.from_bytes(hashlib.sha256(('tent-layout32-v1|'+alias).encode()).digest()[:8],'little');g=np.random.default_rng(seed);out=[]
 for _ in range(2):
  a=g.standard_normal((8,3,2));z=np.concatenate((np.zeros((1,3,2)),a,-a),0);out.append(torch.from_numpy(z)*active[None,:,None])
 return torch.stack(out)

def boxes(E,B,u,scale):
 # u[K,3,2], returns[N,K,4]; no clipping or independent edge recombination.
 out=E[:,None].expand(-1,len(u),-1).clone();out[:,:,:2]+=scale*E[:,None,2:] * torch.einsum('nj,kjd->nkd',B,u)
 return out

@torch.no_grad()
def joint_feedback(E,B,context,u,eps,cfg,permutation=None):
 b=boxes(E,B,u,cfg['center_scale']);ff=feedback(context,b,torch.zeros(len(E),len(u),2,dtype=torch.float64),cfg['temperature'],permutation)
 eligible=B.abs().sum(-1)>0;active=B.abs().sum(0)>0;rr=ff['rewards'][eligible].mean(0)if eligible.any()else torch.zeros(len(u),dtype=torch.float64)
 ok=ff['valid'][eligible].any(0)if eligible.any()else torch.ones(len(u),dtype=torch.bool)
 if not ok.any():ok[:]=True
 lp=(-.5*(eps*active[None,:,None]).square().sum((1,2))).masked_fill(~ok,-torch.inf);prior=lp.softmax(-1);informative=bool(eligible.any()and context['positions']and (rr[ok].max()-rr[ok].min()>1e-6))
 pt=(lp+rr/cfg['temperature']).softmax(-1)if informative else prior.clone()
 return dict(boxes=b,reward=rr,valid=ok,prior=prior,prob=pt,informative=informative,node_rewards=ff['rewards'],node_valid=ff['valid'],eligible=eligible,active=active)

def joint_fit(E,B,context,eps,cfg,mode,permutation):
 mu=torch.zeros(3,2,dtype=torch.float64);path=[dict(state=mu.clone(),boxes=E.clone())];trace=[];active=B.abs().sum(0)>0
 for step in range(2):
  center=torch.zeros_like(mu)if mode=='Joint_fixed'else mu.clone();u=center[None]+eps[step];f=joint_feedback(E,B,context,u,eps[step],cfg,permutation if mode=='Joint_shuffle'else None);lp=(-.5*((u-mu[None])*active[None,:,None]).square().sum((1,2))).masked_fill(~f['valid'],-torch.inf);sp=lp.softmax(-1);grad=((sp-f['prob'])[:,None,None]*u).sum(0);grad[~active]=0
  if not f['informative']:grad.zero_()
  new=mu-cfg['step_size']*grad;out=boxes(E,B,new[None],cfg['center_scale'])[:,0];assert torch.isfinite(out).all()
  trace.append(dict(step=step,sampling_center=center,candidates=u,noise=eps[step],feedback=f,student_prob=sp,gradient=grad,before=mu.clone(),after=new.clone(),prior_ESS=float(1/f['prior'].square().sum()),teacher_ESS=float(1/f['prob'].square().sum()),center_prior_mass=float(f['prior'][0]),center_teacher_mass=float(f['prob'][0])))
  mu=new;path.append(dict(state=mu.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=trace,final=path[-1]['boxes'],selected_step=2,GT_read=False)

def independent_fit(E,B,context,eps,cfg):
 amp=B.sum(-1);allowed=amp>0;eta=torch.zeros(len(E),2,dtype=torch.float64);path=[dict(state=eta.clone(),boxes=E.clone())];trace=[]
 for step in range(2):
  d=eta[:,None]+eps[step];b=E[:,None].expand(-1,d.shape[1],-1).clone();b[:,:,:2]+=cfg['center_scale']*E[:,None,2:]*amp[:,None,None]*d;f=feedback(context,b,eps[step],cfg['temperature']);safe=f['valid'].clone();safe[~safe.any(1)]=True;lp=(-.5*(d-eta[:,None]).square().sum(-1)).masked_fill(~safe,-torch.inf);sp=lp.softmax(-1);g=((sp-f['prob'])[...,None]*d).sum(1);g[~(allowed&f['informative'])]=0;new=eta-cfg['step_size']*g;out=E.clone();out[:,:2]+=cfg['center_scale']*E[:,2:]*amp[:,None]*new
  trace.append(dict(step=step,sampling_center=eta.clone(),candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=g,before=eta.clone(),after=new.clone()))
  eta=new;path.append(dict(state=eta.clone(),boxes=out))
 return dict(mode='Independent_scoped',path=path,rounds=trace,final=path[-1]['boxes'],selected_step=2,GT_read=False)

def project(E,B,out,scale):
 normalized=(out[:,:2]-E[:,:2])/(scale*E[:,2:]);eligible=B.abs().sum(-1)>0;pinv=torch.linalg.pinv(B[eligible])if eligible.any()else torch.zeros(3,0,dtype=torch.float64);u=pinv@normalized[eligible];return dict(state=u,boxes=boxes(E,B,u[None],scale)[:,0],rank=int(torch.linalg.matrix_rank(B)),GT_read=False)

def properties():
 from vg_tta.stvg_roi_layout_opd32_v1 import build
 from vg_tta.stvg_discriminative_dense32_v1 import noise as independent_noise
 ids=list(range(13));planned=[0,4,8,12];B,scope=basis(ids,planned,[3,7]);assert (B[planned]==0).all()and (B[~scope]==0).all();active=B.sum(0)>0;ep=noise('properties',active);assert(ep[:,:,~active]==0).all()
 E=torch.tensor([[.5,.5,.3,.4]]*13,dtype=torch.float64);maps=torch.ones(13,12,5,6);valid=torch.ones(13,5,6,dtype=torch.bool);ctx=build(maps,valid,[0],E[:1]);cfg=dict(center_scale=.25,temperature=.1,step_size=.5)
 for m in FIT_ARMS[:3]:
  f=joint_fit(E,B,ctx,ep,cfg,m,list(range(13)));assert torch.equal(f['final'],E)
 f=independent_fit(E,B,ctx,independent_noise('properties',13),cfg);assert torch.equal(f['final'],E)
 u=torch.tensor([[1.,2.],[3.,4.],[0.,0.]],dtype=torch.float64)*active[:,None];o=boxes(E,B,u[None],.25)[:,0];p=project(E,B,o,.25);assert (p['boxes']-o).abs().max()<1e-14
 B0,_=basis(ids,planned,[30,40]);assert not B0.any();assert torch.equal(joint_fit(E,B0,ctx,noise('zero',B0.sum(0)>0),cfg,'Joint_live',list(range(13)))['final'],E)
 return dict(status='pass',physical_time_basis=True,anchors_and_scope_exact_zero=True,inactive_axes_frozen=True,zero_rank_noop=True,constant_feedback_multiround_noop=True,projection_identity=True)
