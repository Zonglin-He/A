import torch,time
from methods.decota_final_simplified_v1.tensors import detached
from .geometry import local_cost
@torch.enable_grad()
def fit(rp,target,alias,arm,seed,cfg):
 assert arm=='Direct';assert cfg['samples']==32
 with torch.no_grad():rp.state.zero_()
 target=torch.as_tensor(target,device='cuda',dtype=torch.float64).detach();assert target.shape==(8,4)
 opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];rounds=[];initial_mu=None;updated=0;begin=time.perf_counter()
 try:
  for k in range(cfg['steps']+1):
   rp.check();v=rp.values();direct=local_cost(v['boxes'][rp.positions],target).mean()
   path.append(dict(step=k,state=rp.state.detach().cpu().clone(),native=v['native'].detach().cpu(),boxes=v['boxes'].detach().cpu(),anchor_loss=float(direct.detach())))
   if k==0:
    assert torch.equal(v['boxes'],rp.E);initial_mu=v['native'][rp.positions].detach().clone()
   if k==cfg['steps']:break
   rr=None;gmu=None;mu=None;err=None;tol=None
   loss=direct;informative=True
   opt.zero_grad(set_to_none=True);before=rp.state.detach().clone();grad=None
   if informative:
    grad=torch.autograd.grad(loss,rp.state)[0];assert torch.isfinite(grad).all();rp.state.grad=grad.detach();opt.step();updated+=1;rp.backwards+=1
   assert torch.isfinite(rp.state).all()
   rounds.append(dict(step=k,loss=float(loss.detach()),updated=informative,rollout=detached(rr,'cpu'),policy_mean_gradient=detached(gmu,'cpu'),current_policy_mean=detached(mu,'cpu'),analytic_error=err,analytic_tolerance=tol,gradient=detached(grad,'cpu'),state_step=(rp.state.detach()-before).cpu()))
  assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
  return dict(arm=arm,seed=None if arm=='Direct'else seed,configuration=cfg,positions=rp.positions,initial_policy_mean=initial_mu.cpu(),target=target.cpu(),path=path,rounds=rounds,actual_updates=updated,final=path[-1]['boxes'],seconds=time.perf_counter()-begin,selected_step=cfg['steps'],sampling='native output-box logits, not complete decoder trajectories',Fixed='fresh initial-distribution candidates every round'if arm=='Fixed'else None)
 finally:
  with torch.no_grad():rp.state.zero_()
