"""Independent candidate provenance, geometry, gradient, Adam and metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_native_opd_hc128_v1 import D,P,PRE,verify

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 from methods.decota_final_simplified_v1.objectives import generalized_iou
 torch.set_num_threads(4);p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;start=time.perf_counter();pub=read(P/'RESULTS_PUBLIC.json');gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];counts=dict(targets=0,zero_states=0,paired_first_updates=0,paired_noise_rounds=0,fresh_fixed_rounds=0,live_current_centers=0,geometry_batches=0,feedback_batches=0,likelihood_gradients=0,Adam_steps=0,full_metrics=0,outside8=0,bootstraps=0);errs=dict(native_chart=0.,geometry=0.,cost=0.,weights=0.,mean_gradient=0.,Adam=0.,metric=0.)
 for r in p['records']:
  a=r['alias'];q=r['query_ordinal'];row=plans[q];z=load(D/'predictions'/(a+'.pt'));inp=load(PRE/'inputs'/(a+'.pt'));target=load(PRE/'teacher0'/(a+'.pt'))['native']['boxes'].double();rec=next(x for x in pub['rows']if x['alias']==a);pos=r['positions8'];assert z['interval']==inp['interval']and np.array_equal(z['outputs']['E_MBR'],inp['E']);counts['targets']+=1
  for seed in p['seeds']:
   f,l=z['fits'][f'Fixed_{seed}'],z['fits'][f'Live_{seed}'];assert torch.equal(f['path'][1]['state'],l['path'][1]['state']);counts['paired_first_updates']+=1
   for k,(x,y)in enumerate(zip(f['rounds'],l['rounds'])):
    assert x['rollout']['rng_seed']==y['rollout']['rng_seed']and torch.equal(x['rollout']['eps'],y['rollout']['eps']);counts['paired_noise_rounds']+=1
    if k:assert not torch.equal(x['rollout']['actions'],f['rounds'][k-1]['rollout']['actions']);counts['fresh_fixed_rounds']+=1
  for name,fit in z['fits'].items():
   assert torch.count_nonzero(fit['path'][0]['state'])==0 and np.array_equal(fit['path'][0]['boxes'].numpy(),inp['E']);assert torch.equal(fit['path'][0]['native'],inp['Source']);assert torch.equal(fit['target'],target);counts['zero_states']+=1
   state=fit['path'][0]['state'].clone().requires_grad_(True);opt=torch.optim.Adam([state],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0.)
   for k,step in enumerate(fit['rounds']):
    if step['updated']:
     opt.zero_grad(set_to_none=True);state.grad=step['gradient'].clone();opt.step();e=float((state-fit['path'][k+1]['state']).abs().max());errs['Adam']=max(errs['Adam'],e);assert e<3e-6;counts['Adam_steps']+=1
    else:assert torch.equal(state.detach(),fit['path'][k+1]['state'])
    if step['rollout']is None:continue
    rr=step['rollout'];mu=step['current_policy_mean'];proposal=rr['proposal_mean'];native=fit['path'][k]['native'][pos];chart=torch.logit(native);e=float((mu-chart).abs().max());errs['native_chart']=max(errs['native_chart'],e);assert e<2e-6+8*torch.finfo(native.dtype).eps*max(1.,float(mu.abs().max()))
    if fit['arm']=='Fixed':assert torch.equal(proposal,fit['initial_policy_mean'])
    else:assert torch.equal(proposal,mu);counts['live_current_centers']+=1
    key=f"refer-native-output-opd-hc128-v1|{fit['seed']}|{a}|{k}";rng_seed=int(hashlib.sha256(key.encode()).hexdigest()[:15],16);g=torch.Generator(device='cpu').manual_seed(rng_seed);noise=torch.randn(8,16,4,generator=g,dtype=torch.float32);noise=torch.cat((noise,-noise),1);assert rr['rng_seed']==rng_seed and torch.equal(noise,rr['eps']);assert torch.equal(proposal[:,None]+.025*noise,rr['actions'])
    exact=rr['actions'].double().sigmoid();assert float((rr['native_action_boxes'].double()-exact).abs().max())<2e-7;nb=rr['native_action_boxes'].double();source=inp['Source'][pos].double()[:,None];E=torch.tensor(inp['E'][pos],dtype=torch.float64)[:,None];mapped=torch.cat((E[...,:2]+nb[...,:2]-source[...,:2],E[...,2:]*torch.exp(nb[...,2:].log()-source[...,2:].log())),-1);e=float((mapped-rr['boxes']).abs().max());errs['geometry']=max(errs['geometry'],e);assert e<1e-12;counts['geometry_batches']+=1
    costs=5*(rr['boxes']-target[:,None]).abs().sum(-1)+2*(1-generalized_iou(rr['boxes'],target[:,None]));weights=torch.softmax(-costs/.05,-1);ec=float((costs-rr['cost']).abs().max());ew=float((weights-rr['weights']).abs().max());errs['cost']=max(errs['cost'],ec);errs['weights']=max(errs['weights'],ew);assert ec<1e-12 and ew<1e-11;counts['feedback_batches']+=1
    md=mu.double();pd=proposal.double();act=rr['actions'].double();w=rr['weights'].double();analytic=((md-pd)/.025**2-((w-1/32)[:,:,None]*(act-md[:,None])).sum(1)/.025**2)/8;e=float((analytic-step['policy_mean_gradient'].double()).abs().max());errs['mean_gradient']=max(errs['mean_gradient'],e);tol=step['analytic_tolerance']+2e-6*max(1.,float(analytic.abs().max()));assert e<tol;counts['likelihood_gradients']+=1
   assert torch.equal(fit['final'],fit['path'][-1]['boxes'])
  for arm,bs in z['outputs'].items():
   m=DenseTube(bs,row,gt[q],spans[q],clip=True).score(z['interval'])
   for ep in ['v','t','s']:
    e=abs(m[ep]-rec['values'][arm][ep]);errs['metric']=max(errs['metric'],e);assert e<2e-10;counts['full_metrics']+=1
   vv=[iou_gt(bs[i],gt[q][int(fid)],row,'hc2')for i,fid in enumerate(z['frame_ids'])if i not in pos and int(fid)in gt[q]];v=float(np.mean(vv))if vv else None;expect=rec['values'][arm]['unobserved_actual_s'];assert v is None and expect is None or v is not None and abs(v-expect)<1e-12;counts['outside8']+=1
  for arm in ['Fixed','Live']:
   for ep in pub['primary']['means']['Direct']:
    vals=[rec['values'][f'{arm}_{s}'][ep]for s in p['seeds']if rec['values'][f'{arm}_{s}'][ep]is not None];v=float(np.mean(vals))if vals else None;expect=rec['values'][arm+'_seedmean'][ep];assert v is None and expect is None or v is not None and abs(v-expect)<1e-14
 for group in [pub['primary']]+list(pub['by_seed'].values()):
  for pair,ee in group['paired'].items():
   a,b=pair.split('_minus_')
   for ep,st in ee.items():
    v=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in pub['rows']if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);idx=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));assert abs(float(v.mean())-st['mean'])<1e-12 and np.allclose(np.quantile(v[idx].mean(1),[.025,.975]),st['CI95'],rtol=0,atol=1e-12);counts['bootstraps']+=1
 verify(True);assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=counts,max_errors=errs,CPU_seconds=time.perf_counter()-start,scope='saved noise/actions/mapping/geometry feedback/likelihood gradients/Adam/actual metrics and parent bootstrap; not second full GPU replay',fresh_Fixed_verified=True,paired_seeds_verified=True,no_new_GT_candidate_selection=True));print(counts,errs,flush=True)
if __name__=='__main__':main()
