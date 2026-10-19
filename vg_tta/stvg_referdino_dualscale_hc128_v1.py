"""Target-size-normalized, two-scale mixture output-box policy.

Only mean coordinates are differentiated. Actions, targets, mixture scales and
proposal states are stopped. Positive widths/heights are encoded logarithmically;
boxes may cross the image border and are not silently clipped or rejected.
"""
import math,time
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.stvg_hc_anchor_content16_v1 import Replay,local_cost
from vg_tta.stvg_referdino_native_opd_hc128_v1 import noise_for

CONFIGS={'N005F025':(.05,.25),'N005F050':(.05,.5),'N010F050':(.1,.5)}

def chart(boxes):
 assert torch.isfinite(boxes).all() and (boxes[...,2:]>0).all()
 return torch.cat((boxes[...,:2],boxes[...,2:].log()),-1)

def unchart(z):return torch.cat((z[...,:2],z[...,2:].exp()),-1)

def diagonal(boxes):return torch.cat((boxes[...,2:],torch.ones_like(boxes[...,2:])), -1).detach()

def log_components(mu,actions,D,scales):
 assert not D.requires_grad and not actions.requires_grad
 sig=torch.tensor(scales,device=mu.device,dtype=mu.dtype)
 std=D[:,None,:]*sig[None,:,None]
 r=(actions[:,:,None,:]-mu[:,None,None,:])/std[:,None,:,:]
 return -.5*r.square().sum(-1)-std.log().sum(-1)[:,None,:]-2*math.log(2*math.pi)

def mixture_loss(mu,actions,D,weights,scales):
 assert not weights.requires_grad
 return -(weights*(torch.logsumexp(log_components(mu,actions,D,scales),-1)-math.log(2))).sum(-1).mean()

def mean_gradient(mu,actions,D,weights,scales):
 comp=log_components(mu,actions,D,scales);resp=comp.softmax(-1)
 sig=torch.tensor(scales,device=mu.device,dtype=mu.dtype)
 inv=1/(D[:,None,:]*sig[None,:,None]).square()
 return (weights[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-actions[:,:,None,:])*inv[:,None,:,:]).sum((1,2))/len(mu)

@torch.no_grad()
def feedback(proposal,D,target,alias,seed,step,scales,tau):
 raw,key=noise_for(alias,seed,step,len(proposal),torch.float32)
 # Same sixteen independent noises as old policy, eight pairs per component.
 eps=torch.cat((raw[:,:8],raw[:,16:24],raw[:,8:16],raw[:,24:]),1).to(proposal)
 sig=torch.tensor([scales[0]]*16+[scales[1]]*16,device=proposal.device,dtype=proposal.dtype)
 actions=(proposal[:,None]+D[:,None]*sig[None,:,None]*eps).detach()
 boxes=unchart(actions);cost=local_cost(boxes,target[:,None]);weights=torch.softmax(-cost/tau,-1).detach()
 assert torch.isfinite(boxes).all()and(boxes[...,2:]>0).all()and torch.isfinite(weights).all()
 xy0=boxes[...,:2]-boxes[...,2:]/2;xy1=boxes[...,:2]+boxes[...,2:]/2
 outside=((xy0<0).any(-1)|(xy1>1).any(-1));empty=(torch.minimum(xy1,torch.ones_like(xy1))<=torch.maximum(xy0,torch.zeros_like(xy0))).any(-1)
 return dict(proposal_mean=proposal.clone(),D=D.clone(),actions=actions,boxes=boxes,cost=cost,weights=weights,eps=eps,rng_seed=key,informative=bool((cost!=cost[:,:1]).any()),outside_image=int(outside.sum()),zero_image_overlap=int(empty.sum()),invalid_positive_size=0)

@torch.enable_grad()
def fit(rp,target,alias,arm,seed,cfg):
 assert arm in ['Fixed','Live'];scales=cfg['scales'];assert cfg['samples']==32
 with torch.no_grad():rp.state.zero_()
 target=torch.as_tensor(target,device=rp.state.device,dtype=torch.float64).detach()
 opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];rounds=[];initial_mu=initial_D=None;updated=0;begin=time.perf_counter()
 try:
  for k in range(cfg['steps']+1):
   rp.check();v=rp.values();bs=v['boxes'][rp.positions];mu=chart(bs);cost=local_cost(bs,target).mean()
   path.append(dict(step=k,state=rp.state.detach().cpu().clone(),native=v['native'].detach().cpu(),boxes=v['boxes'].detach().cpu(),anchor_loss=float(cost.detach())))
   if k==0:
    assert torch.equal(v['boxes'],rp.E);initial_mu=mu.detach().clone();initial_D=diagonal(bs).clone()
   if k==cfg['steps']:break
   proposal=initial_mu if arm=='Fixed'else mu.detach();D=initial_D if arm=='Fixed'else diagonal(bs)
   rr=feedback(proposal,D,target,alias,seed,k,scales,cfg['tau'])
   loss=mixture_loss(mu,rr['actions'],D,rr['weights'],scales)
   gm=torch.autograd.grad(loss,mu,retain_graph=True)[0];expected=mean_gradient(mu.detach(),rr['actions'],D,rr['weights'],scales);err=float((gm-expected).abs().max());tol=1e-9+1e-10*max(1.,float(gm.abs().max()));assert err<tol
   opt.zero_grad(set_to_none=True);before=rp.state.detach().clone();grad=None;informative=rr['informative']
   if informative:
    grad=torch.autograd.grad(loss,rp.state)[0];assert torch.isfinite(grad).all();rp.state.grad=grad.detach();opt.step();updated+=1;rp.backwards+=1
   assert torch.isfinite(rp.state).all()
   rounds.append(dict(step=k,loss=float(loss.detach()),updated=informative,rollout=detached(rr,'cpu'),current_policy_mean=mu.detach().cpu(),policy_mean_gradient=gm.detach().cpu(),analytic_error=err,analytic_tolerance=tol,gradient=detached(grad,'cpu'),state_step=(rp.state.detach()-before).cpu()))
  assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
  return dict(arm=arm,seed=seed,configuration=cfg,positions=rp.positions,initial_policy_mean=initial_mu.cpu(),initial_D=initial_D.cpu(),target=target.cpu(),path=path,rounds=rounds,actual_updates=updated,final=path[-1]['boxes'],seconds=time.perf_counter()-begin,selected_step=cfg['steps'],policy='corrected output cx,cy,logw,logh mixture; no latent candidate trajectories')
 finally:
  with torch.no_grad():rp.state.zero_()

def properties():
 mu=torch.tensor([[.5,.3,math.log(.2),math.log(.3)]],dtype=torch.float64,requires_grad=True);D=torch.tensor([[.2,.3,1.,1.]],dtype=torch.float64);target=unchart(mu.detach())
 rr=feedback(mu.detach(),D,target,'synthetic',41,0,(.05,.5),.05);loss=mixture_loss(mu,rr['actions'],D,rr['weights'],(.05,.5));g=torch.autograd.grad(loss,mu)[0];ex=mean_gradient(mu.detach(),rr['actions'],D,rr['weights'],(.05,.5));assert torch.allclose(g,ex,atol=1e-10,rtol=1e-10)
 assert torch.equal(rr['eps'][:,:8],-rr['eps'][:,8:16])and torch.equal(rr['eps'][:,16:24],-rr['eps'][:,24:])
 fd=[];h=1e-6
 for j in range(4):
  a=mu.detach().clone();a[0,j]+=h;v=mixture_loss(a,rr['actions'],D,rr['weights'],(.05,.5));a[0,j]-=2*h;u=mixture_loss(a,rr['actions'],D,rr['weights'],(.05,.5));fd.append(float((v-u)/(2*h)))
 error=max(abs(fd[j]-g[0,j].item())for j in range(4));assert error<1e-6
 u=torch.full_like(rr['weights'],1/32);m=mu.detach().clone().requires_grad_(True);ug=torch.autograd.grad(mixture_loss(m,rr['actions'],D,u,(.05,.5)),m)[0];assert ug.abs().max()<1e-10
 # Independent torch distribution density check, including both component terms.
 ds=torch.stack([torch.distributions.Independent(torch.distributions.Normal(mu.detach(),s*D),1).log_prob(rr['actions'])for s in [.05,.5]],-1);ours=log_components(mu.detach(),rr['actions'],D,(.05,.5));assert torch.allclose(ds,ours,atol=1e-11,rtol=0)
 assert torch.allclose(unchart(chart(target)),target,atol=1e-15,rtol=0)
 return dict(status='pass',mixture_density_independent_distribution=True,finite_difference_max_abs=error,analytic_gradient_max_abs=float((g-ex).abs().max()),near_far_each16_antithetic=True,uniform_feedback_gradient_max_abs=float(ug.abs().max()),same_noises_as_previous_seed_stream=True,CPU_only=True)
