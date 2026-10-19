"""Independent CPU saved-path audit, not a second GPU inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,math,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final'
def main(stage):
 import torch
 from methods.stvg_referdino_opd_v1.geometry import corrected,local_cost
 torch.set_num_threads(4);bar=read(D/stage/'GLOBAL_PREDICTION_BARRIER.json');counts=dict(parents=0,fits=0,updates=0,feedback=0,zero=0,actual_path=0);maximum=dict(gradient=0.,Adam=0.,mapping=0.);selection=read(D/'SELECTION_LOCK.json')if(D/'SELECTION_LOCK.json').exists()else None
 for f in bar['files']:
  assert sha(R/f['path'])==f['sha256'];z=load(R/f['path']);assert not z['GT_read'];counts['parents']+=1;E=torch.as_tensor(z['outputs']['E_MBR']);source=torch.as_tensor(z['outputs']['Source_MBR'])
  if stage=='Live':
   old=load(R/'artifacts/stvg_referdino_dualscale_hc128_v1/predictions'/(z['alias']+'.pt'))
   for seed in [41,42,43]:
    x=z['fits'][f'T10_Live_{seed}'];y=z['fits'][f'T20_Live_{seed}'];prior=old['fits'][f'N005F025_Live_{seed}']
    for key in ['actions','boxes','cost','eps','D','proposal_mean']:assert torch.equal(x['rounds'][0]['rollout'][key],y['rounds'][0]['rollout'][key])and torch.equal(x['rounds'][0]['rollout'][key],prior['rounds'][0]['rollout'][key])
    for a,b,c in zip(x['rounds'],y['rounds'],prior['rounds']):assert torch.equal(a['rollout']['eps'],b['rollout']['eps'])and torch.equal(a['rollout']['eps'],c['rollout']['eps'])
   del old
  if stage=='Fixed':
   live=load(D/'Live/predictions'/(z['alias']+'.pt'));tau=read(D/'SELECTION_LOCK.json')['tau']
   for seed in [41,42,43]:
    x=z['fits'][f'T{int(tau*100):02}_Fixed_{seed}'];y=live['fits'][f'T{int(tau*100):02}_Live_{seed}'];assert torch.equal(x['path'][1]['state'],y['path'][1]['state'])
    for key in ['actions','boxes','cost','eps','D','proposal_mean','weights']:assert torch.equal(x['rounds'][0]['rollout'][key],y['rounds'][0]['rollout'][key])
   del live
  for name,v in z['fits'].items():
   counts['fits']+=1;assert torch.equal(v['path'][0]['boxes'],E);assert torch.count_nonzero(v['path'][0]['state'])==0;counts['zero']+=1
   assert len(v['path'])==41 and len(v['rounds'])==40 and torch.equal(v['final'],v['path'][-1]['boxes'])
   x=v['path'][0]['state'].clone().requires_grad_();opt=torch.optim.Adam([x],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
   for k,rr in enumerate(v['rounds']):
    path=v['path'][k];mapped=corrected(path['native'],source,E);er=float((mapped-path['boxes']).abs().max());maximum['mapping']=max(maximum['mapping'],er);assert er<1e-12;counts['actual_path']+=1
    if rr['rollout']is not None:
     b=rr['rollout'];mu=rr['current_policy_mean'];pos=v['positions'];current=path['boxes'][pos];chart=torch.cat((current[:,:2],current[:,2:].log()),-1);assert torch.allclose(mu,chart,atol=1e-12,rtol=0)
     if v['arm']=='Live':assert torch.equal(b['proposal_mean'],mu)and torch.equal(b['D'],torch.cat((current[:,2:],torch.ones_like(current[:,2:])), -1))
     else:assert torch.equal(b['proposal_mean'],v['initial_policy_mean'])and torch.equal(b['D'],v['initial_D'])
     t=v['configuration']['tau'];C=local_cost(b['boxes'],v['target'][:,None]);assert torch.allclose(C,b['cost'],atol=1e-10,rtol=1e-12);wc=torch.softmax(-C/t,-1);assert torch.allclose(wc,b['weights'],atol=1e-11,rtol=1e-11);w=b['weights'];assert torch.equal(b['eps'][:,:8],-b['eps'][:,8:16])and torch.equal(b['eps'][:,16:24],-b['eps'][:,24:]);counts['feedback']+=1
     # Independent distribution density and autograd, no policy likelihood helper.
     m=mu.clone().requires_grad_();parts=torch.stack([torch.distributions.Independent(torch.distributions.Normal(m[:,None,:],s*b['D'][:,None,:]),1).log_prob(b['actions'])for s in [.05,.25]],-1);loss=-(w*(torch.logsumexp(parts,-1)-math.log(2))).sum(-1).mean();grad=torch.autograd.grad(loss,m)[0];err=float((grad-rr['policy_mean_gradient']).abs().max());maximum['gradient']=max(maximum['gradient'],err);assert err<1e-8+1e-10*max(1.,float(grad.abs().max()))
    if rr['updated']:
     opt.zero_grad(set_to_none=True);x.grad=rr['gradient'].clone();opt.step();counts['updates']+=1
    err=float((x.detach()-v['path'][k+1]['state']).abs().max());maximum['Adam']=max(maximum['Adam'],err);assert err<2e-6
   assert v['actual_updates']==sum(x['updated']for x in v['rounds'])
 print(stage,counts,maximum,flush=True);put(P/(stage.upper()+'_INDEPENDENT_AUDIT.json'),dict(status='pass',counts=counts,max_errors=maximum,scope='independent CPU density/target/Adam/mapping replay from saved artifacts; not a second GPU model run'))
if __name__=='__main__':main(sys.argv[1])
