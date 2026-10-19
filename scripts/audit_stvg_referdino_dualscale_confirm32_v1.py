"""Independent saved trace, mixed/old density, optimizer, metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_referdino_dualscale_confirm32_v1 import D,P,DEV,verify

def main():
 import torch
 from methods.decota_final_simplified_v1.objectives import generalized_iou
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 torch.set_num_threads(4);p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;pub=read(P/'RESULTS_PUBLIC.json');gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];start=time.perf_counter();counts=dict(parents=0,zero_states=0,Adam=0,old_density_gradients=0,mixture_density_gradients=0,fresh_batches=0,full_metrics=0,outside8_metrics=0,path_readouts=0,bootstraps=0);errors=dict(Adam=0.,density_gradient=0.,cost=0.,weights=0.,metric=0.)
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));inp=load(D/'inputs'/(r['alias']+'.pt'));target=load(D/'teacher0'/(r['alias']+'.pt'))['native']['boxes'].double();pos=r['positions8'];rec=next(t for t in pub['rows']if t['alias']==r['alias']);assert z['interval']==inp['interval']and np.array_equal(z['outputs']['E_MBR'],inp['E']);counts['parents']+=1
  for s in p['seeds']:
   f,l=z['fits'][f'Selected_Fixed_{s}'],z['fits'][f'Selected_Live_{s}'];assert torch.equal(f['path'][1]['state'],l['path'][1]['state'])
   for x,y in zip(f['rounds'],l['rounds']):assert torch.equal(x['rollout']['eps'],y['rollout']['eps'])
  for name,fit in z['fits'].items():
   assert torch.count_nonzero(fit['path'][0]['state'])==0 and np.array_equal(fit['path'][0]['boxes'].numpy(),inp['E']);assert torch.equal(fit['path'][0]['native'],inp['Source']);assert torch.equal(fit['target'],target);counts['zero_states']+=1;state=fit['path'][0]['state'].clone().requires_grad_(True);opt=torch.optim.Adam([state],lr=.01,betas=(.9,.999),eps=1e-8)
   for v in fit['path']:
    b=v['native'].double();src=inp['Source'].double();E=torch.tensor(inp['E'],dtype=torch.float64);ex=torch.cat((E[:,:2]+b[:,:2]-src[:,:2],E[:,2:]*(b[:,2:].log()-src[:,2:].log()).exp()),-1);assert float((ex-v['boxes']).abs().max())<1e-12;counts['path_readouts']+=1
   for k,step in enumerate(fit['rounds']):
    if step['updated']:
     opt.zero_grad(set_to_none=True);state.grad=step['gradient'].clone();opt.step();err=float((state-fit['path'][k+1]['state']).abs().max());errors['Adam']=max(errors['Adam'],err);assert err<3e-6;counts['Adam']+=1
    if step['rollout']is None:continue
    rr=step['rollout'];mu=step['current_policy_mean'].double();proposal=rr['proposal_mean'].double();act=rr['actions'].double();key=f"refer-native-output-opd-hc128-v1|{fit['seed']}|{r['alias']}|{k}";rng=int(hashlib.sha256(key.encode()).hexdigest()[:15],16);g=torch.Generator(device='cpu').manual_seed(rng);noise=torch.randn(8,16,4,generator=g,dtype=torch.float32);assert rr['rng_seed']==rng
    mixed='scales'in fit['configuration']
    if mixed:
     near,far=fit['configuration']['scales'];D0=rr['D'].double();noise=torch.cat((noise[:,:8],-noise[:,:8],noise[:,8:],-noise[:,8:]),1).double();assert torch.equal(noise,rr['eps']);sig=torch.tensor([near]*16+[far]*16,dtype=torch.float64);assert torch.allclose(act,proposal[:,None]+D0[:,None]*sig[None,:,None]*noise,atol=1e-12,rtol=0);boxes=torch.cat((act[...,:2],act[...,2:].exp()),-1)
     if fit['arm']=='Fixed':assert torch.equal(proposal,fit['initial_policy_mean'])and torch.equal(D0,fit['initial_D'])
     else:
      bs=fit['path'][k]['boxes'][pos];assert torch.equal(proposal,mu)and torch.equal(D0,torch.cat((bs[:,2:],torch.ones_like(bs[:,2:])), -1))
     comp=torch.stack([torch.distributions.Independent(torch.distributions.Normal(mu[:,None],s*D0[:,None]),1).log_prob(act)for s in [near,far]],-1);lp=torch.logsumexp(comp,-1)-np.log(2);assert abs(float(-(rr['weights']*lp).sum(-1).mean())-step['loss'])<1e-10;resp=comp.softmax(-1);inv=torch.stack([1/(s*D0).square()for s in [near,far]],1);analytic=(rr['weights'][:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-act[:,:,None,:])*inv[:,None,:,:]).sum((1,2))/8;tol=1e-8+1e-9*max(1.,float(analytic.abs().max()));counts['mixture_density_gradients']+=1
    else:
     noise=torch.cat((noise,-noise),1);assert torch.equal(noise,rr['eps']);assert torch.equal(rr['actions'],rr['proposal_mean'][:,None]+.025*noise);native=rr['native_action_boxes'].double();src=inp['Source'][pos].double()[:,None];E=torch.tensor(inp['E'][pos],dtype=torch.float64)[:,None];boxes=torch.cat((E[...,:2]+native[...,:2]-src[...,:2],E[...,2:]*(native[...,2:].log()-src[...,2:].log()).exp()),-1);analytic=((mu-proposal)/.025**2-((rr['weights'].double()-1/32)[:,:,None]*(act-mu[:,None])).sum(1)/.025**2)/8;tol=step['analytic_tolerance']+2e-6*max(1.,float(analytic.abs().max()));counts['old_density_gradients']+=1
     if fit['arm']=='Fixed':assert torch.equal(rr['proposal_mean'],fit['initial_policy_mean'])
     else:assert torch.equal(rr['proposal_mean'],step['current_policy_mean'])
    assert float((boxes-rr['boxes']).abs().max())<1e-12;cost=5*(boxes-target[:,None]).abs().sum(-1)+2*(1-generalized_iou(boxes,target[:,None]));w=torch.softmax(-cost/.05,-1);ec=float((cost-rr['cost']).abs().max());ew=float((w-rr['weights']).abs().max());errors['cost']=max(errors['cost'],ec);errors['weights']=max(errors['weights'],ew);assert ec<1e-11 and ew<1e-10;err=float((analytic-step['policy_mean_gradient'].double()).abs().max());errors['density_gradient']=max(errors['density_gradient'],err);assert err<tol;counts['fresh_batches']+=1
  for arm,bs in z['outputs'].items():
   q=r['query_ordinal'];m=DenseTube(bs,plans[q],gt[q],spans[q],clip=True).score(z['interval'])
   for ep in ['v','t','s']:
    e=abs(m[ep]-rec['values'][arm][ep]);errors['metric']=max(errors['metric'],e);assert e<2e-10;counts['full_metrics']+=1
   vv=[iou_gt(bs[i],gt[q][int(fid)],plans[q],'hc2')for i,fid in enumerate(z['frame_ids'])if i not in pos and int(fid)in gt[q]];actual=float(np.mean(vv))if vv else None;expect=rec['values'][arm]['unobserved_actual_s'];assert actual is None and expect is None or actual is not None and abs(actual-expect)<1e-12;counts['outside8_metrics']+=1
 for group in [pub['primary']]+list(pub['by_seed'].values()):
  for pair,endpoints in group['paired'].items():
   a,b=pair.split('_minus_')
   for ep,st in endpoints.items():
    vals=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in pub['rows']if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);ix=np.random.default_rng(20261018).integers(len(vals),size=(10000,len(vals)));assert abs(float(vals.mean())-st['mean'])<1e-12 and np.allclose(np.quantile(vals[ix].mean(1),[.025,.975]),st['CI95'],rtol=0,atol=1e-12);counts['bootstraps']+=1
 verify(True);assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=counts,max_errors=errors,CPU_seconds=time.perf_counter()-start,selection_not_changed_by_confirmation=True,saved_artifact_replay_not_second_GPU_run=True));print(counts,errors,flush=True)
if __name__=='__main__':main()
