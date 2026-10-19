"""Fixed native ReferDINO pseudo-boxes: Direct/fresh-Fixed/Live output OPD.

Candidates are Gaussian native-box-logit actions at eight observed frames, NOT
sampled decoder latent states or complete decoder-produced candidate trajectories.
Each actual state update does rerun the frozen decoder's complete trajectory.
"""
import hashlib,time
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.stvg_hc_anchor_content16_v1 import Replay,local_cost
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,likelihood_loss


def noise_for(alias,seed,step,anchors,dtype=torch.float32):
 key=f'refer-native-output-opd-hc128-v1|{int(seed)}|{alias}|{int(step)}'
 rng_seed=int(hashlib.sha256(key.encode()).hexdigest()[:15],16)
 g=torch.Generator(device='cpu').manual_seed(rng_seed)
 eps=torch.randn(anchors,16,4,generator=g,dtype=dtype)
 return torch.cat((eps,-eps),1),rng_seed

@torch.no_grad()
def feedback(rp,proposal,target,alias,seed,step,cfg):
 eps,key=noise_for(alias,seed,step,len(proposal),proposal.dtype);eps=eps.to(proposal)
 actions=(proposal[:,None]+cfg['sigma']*eps).detach();boxes=rp.mapped_samples(actions)
 native_actions=actions.sigmoid();costs=local_cost(boxes,target[:,None]);weights=torch.softmax(-costs/cfg['tau'],-1).detach()
 assert torch.isfinite(costs).all() and torch.isfinite(weights).all()
 return dict(proposal_mean=proposal.detach().clone(),actions=actions,boxes=boxes,cost=costs,weights=weights,rng_seed=key,eps=eps,native_action_boxes=native_actions,informative=bool((costs!=costs[:,:1]).any()))

@torch.enable_grad()
def fit(rp,target,alias,arm,seed,cfg):
 assert arm in ['Direct','Fixed','Live'];assert cfg['samples']==32
 with torch.no_grad():rp.state.zero_()
 target=torch.as_tensor(target,device='cuda',dtype=torch.float64).detach();assert target.shape==(8,4)
 opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];rounds=[];initial_mu=None;updated=0;begin=time.perf_counter()
 try:
  for k in range(cfg['steps']+1):
   rp.check();v=rp.values();direct=local_cost(v['boxes'][rp.positions],target).mean()
   path.append(dict(step=k,state=rp.state.detach().cpu().clone(),native=v['native'].detach().cpu(),boxes=v['boxes'].detach().cpu(),anchor_loss=float(direct.detach())))
   if k==0:
    assert torch.equal(v['boxes'],rp.E);initial_mu=mean_coordinates(v['native'][rp.positions]).detach().clone()
   if k==cfg['steps']:break
   rr=None;gmu=None;mu=None;err=None;tol=None
   if arm=='Direct':loss=direct;informative=True
   else:
    mu=mean_coordinates(v['native'][rp.positions]);proposal=initial_mu if arm=='Fixed'else mu.detach()
    # Fresh sample every round for BOTH controls, paired noises independent of arm.
    rr=feedback(rp,proposal,target,alias,seed,k,cfg)
    loss=likelihood_loss(mu,rr['proposal_mean'],rr['actions'],rr['weights'],cfg['sigma'])
    gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0]
    expected=((mu.detach()-rr['proposal_mean'])/cfg['sigma']**2-((rr['weights']-1/32)[:,:,None]*(rr['actions']-mu.detach()[:,None])).sum(1)/cfg['sigma']**2)/len(mu)
    err=float((gmu-expected).abs().max());tol=2e-5+8*torch.finfo(mu.dtype).eps*max(1.,float(gmu.abs().max()));assert err<tol
    informative=rr['informative']
   opt.zero_grad(set_to_none=True);before=rp.state.detach().clone();grad=None
   if informative:
    grad=torch.autograd.grad(loss,rp.state)[0];assert torch.isfinite(grad).all();rp.state.grad=grad.detach();opt.step();updated+=1;rp.backwards+=1
   assert torch.isfinite(rp.state).all()
   rounds.append(dict(step=k,loss=float(loss.detach()),updated=informative,rollout=detached(rr,'cpu'),policy_mean_gradient=detached(gmu,'cpu'),current_policy_mean=detached(mu,'cpu'),analytic_error=err,analytic_tolerance=tol,gradient=detached(grad,'cpu'),state_step=(rp.state.detach()-before).cpu()))
  assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
  return dict(arm=arm,seed=None if arm=='Direct'else seed,configuration=cfg,positions=rp.positions,initial_policy_mean=initial_mu.cpu(),target=target.cpu(),path=path,rounds=rounds,actual_updates=updated,final=path[-1]['boxes'],seconds=time.perf_counter()-begin,selected_step=cfg['steps'],sampling='native output-box logits, not complete decoder trajectories',Fixed='fresh initial-distribution candidates every round'if arm=='Fixed'else None)
 finally:
  with torch.no_grad():rp.state.zero_()

def properties():
 a,k=noise_for('synthetic',41,0,8);b,j=noise_for('synthetic',41,0,8);c,l=noise_for('synthetic',41,1,8)
 assert torch.equal(a,b)and k==j and not torch.equal(a,c)and k!=l;assert torch.equal(a[:,:16],-a[:,16:])
 mu=torch.tensor([[.1,-.1,.2,-.2]],dtype=torch.float64,requires_grad=True);proposal=torch.zeros_like(mu).detach();samples=proposal[:,None]+.025*a[:1].double();w=torch.softmax(torch.linspace(-1.,1.,32,dtype=torch.float64)[None],-1).detach();loss=likelihood_loss(mu,proposal,samples,w,.025);g=torch.autograd.grad(loss,mu)[0];analytic=((mu.detach()-proposal)/.025**2-((w-1/32)[:,:,None]*(samples-mu.detach()[:,None])).sum(1)/.025**2);assert torch.allclose(g,analytic,rtol=0,atol=1e-12)
 mu0=proposal.clone().requires_grad_(True);wu=torch.full((1,32),1/32,dtype=torch.float64);z=likelihood_loss(mu0,proposal,samples,wu,.025);assert z==0 and torch.equal(torch.autograd.grad(z,mu0)[0],torch.zeros_like(mu0))
 h=1e-6;fd=[]
 for d in range(4):
  v=mu.detach().clone();v[0,d]+=h;plus=likelihood_loss(v,proposal,samples,w,.025);v[0,d]-=2*h;minus=likelihood_loss(v,proposal,samples,w,.025);fd.append(float((plus-minus)/(2*h)))
 assert max(abs(fd[d]-g[0,d].item())for d in range(4))<1e-7
 return dict(status='pass',paired_randomness=True,new_noise_each_round=True,antithetic32=True,analytic_likelihood_gradient=True,finite_difference_max_abs=max(abs(fd[d]-g[0,d].item())for d in range(4)),uniform_target_zero_gradient=True,CPU_only=True)
