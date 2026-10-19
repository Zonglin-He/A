"""Independent actual-path audit for two separately locked finite Center experiments."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,importlib,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def main(kind):
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 mod=importlib.import_module('scripts.run_stvg_fine_roi_'+kind+'32_v1');D,P=mod.D,mod.P
 torch.set_num_threads(4);p=mod.verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxr=maxp=maxg=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));x=load(mod.FINE/'features'/(a+'.pt'));old=load(mod.CENTER/'predictions'/(a+'.pt'));E=x['E'].numpy().astype(np.float64);n=len(E);ids=np.asarray(x['frame_ids']);iv=x['interval'];allowed=(np.r_[ids[1:],ids[-1]]>=iv[0])&(np.r_[ids[0],ids[:-1]]<iv[1]);allowed[x['planned_positions']]=False
  assert np.array_equal(allowed,o['allowed'])and np.array_equal(o['outputs']['Center_live'],old['outputs']['Fine_live']);assert torch.equal(o['noise'],old['noise'])if isinstance(o['noise'],torch.Tensor)else all(torch.equal(a,b)for a,b in zip(o['noise'],old['noise']));checks['exact_original_baseline_scope_noise']+=1
  A=len(x['positions']);tt,tw=sample(x['maps'][x['positions']],x['valid'][x['positions']],np.asarray([b.numpy()for b in x['boxes']]).reshape(-1,1,4))if A else(np.empty((0,1,9,1024)),np.empty((0,1,9)));tt=tt[:,0];tw=tw[:,0]
  for mode,fit in o['fits'].items():
   cfg=dict(p['configuration'],**fit.get('actual_configuration',{}))
   for st,t in enumerate(fit['rounds']):
    b=t['candidates'].numpy();fb=t['feedback'];perm=o['permutation']if mode=='Mix_shuffle'else None;maps=x['maps'][perm]if perm is not None else x['maps'];valid=x['valid'][perm]if perm is not None else x['valid'];ph,w=sample(maps,valid,b);ok=w.sum(-1)>1e-6;assert np.array_equal(ok,fb['valid']);safe=ok.copy();safe[~ok.any(1)]=True
    if A:
     mass=w[None]*tw[:,None,None];per=(np.einsum('nkjc,ajc->ankj',ph,tt)*mass).sum(-1)/np.maximum(mass.sum(-1),1e-12);er=float(abs(per-fb['per_anchor_rewards'].numpy()).max());maxr=max(maxr,er);assert er<3e-5;checks['independent_ROI_reward_scalars']+=per.size
    ee=o['noise'][st].numpy();lp=np.where(safe,-.5*(ee*ee).sum(-1),-np.inf);pr=sm(lp);rr=fb['rewards'].numpy();info=ok.any(1)&((np.where(ok,rr,-np.inf).max(1)-np.where(ok,rr,np.inf).min(1))>1e-6)if A else np.zeros(n,bool);assert np.array_equal(info,fb['informative'])
    per=fb['per_anchor_rewards'].numpy();qa=sm(lp[None]+per/cfg['temperature'])if A else np.empty_like(per);pt=qa.mean(0)if A and kind=='anchor_mix'else sm(lp+rr/cfg['temperature']);pt[~info]=pr[~info]
    if kind=='anchor_mix':
     raw=qa.mean(0)if A else pr;assert np.max(abs(raw-fb['raw_mixture_prob'].numpy()))<1e-12
     if A:assert np.max(abs(qa-fb['per_anchor_prob'].numpy()))<1e-12;checks['per_anchor_posteriors']+=qa.size
     geometric=sm(lp+rr/cfg['temperature']);geometric[~info]=pr[~info];assert abs(geometric-fb['geometric_prob'].numpy()).max()<1e-12
    er=float(abs(pt-fb['prob'].numpy()).max());maxp=max(maxp,er);assert er<1e-12;checks['teacher_probability_scalars']+=pt.size
    eta=t['eta_before'].numpy();center=np.zeros_like(eta)if mode=='Mix_fixed'else eta;assert np.array_equal(center,t['sampling_center']);d=center[:,None]+ee;want=np.repeat(E[:,None],17,axis=1);want[:,:,:2]+=cfg['center_scale']*E[:,None,2:]*d;assert abs(want-b).max()<1e-12
    sp=sm(np.where(safe,-.5*((d-eta[:,None])**2).sum(-1),-np.inf));assert abs(sp-t['student_prob'].numpy()).max()<1e-12;g=((sp-pt)[...,None]*d).sum(1);g[~(allowed&info)]=0;er=float(abs(g-t['gradient'].numpy()).max());maxg=max(maxg,er);assert er<1e-12;new=eta-cfg['step_size']*g;assert abs(new-t['eta_after'].numpy()).max()<1e-12
    out=E.copy();out[:,:2]+=cfg['center_scale']*E[:,2:]*new;out[~allowed]=E[~allowed];assert abs(out-fit['path'][st+1]['boxes'].numpy()).max()<1e-12;checks['actual_gradient_state_output_rounds']+=1
   assert np.array_equal(fit['final'],o['outputs'][mode]);checks['final_readout']+=1
   if kind=='anchor_mix'and A==1:
    om={'Mix_fixed':'Fine_fixed','Mix_live':'Fine_live','Mix_shuffle':'Fine_shuffle'}[mode];assert torch.equal(fit['final'],old['fits'][om]['final']);checks['actual_singleton_whole_path_bitexact']+=1
  if kind=='anchor_mix':assert torch.equal(o['fits']['Mix_fixed']['path'][1]['boxes'],o['fits']['Mix_live']['path'][1]['boxes']);checks['first_fixed_live_exact']+=1
  for b in o['outputs'].values():
   assert np.array_equal(b[~allowed],E[~allowed])and np.array_equal(b[:,2:],E[:,2:]);checks['planned_scope_sizes_exactE']+=1
   if not A:assert np.array_equal(b,E);checks['no_reference_exact_noop']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==want;checks['parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_independent_reward_error=maxr,max_probability_error=maxp,max_gradient_error=maxg,no_GPU_model_or_new_features=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(kind,'audit pass',dict(checks),flush=True)
if __name__=='__main__':main(sys.argv[1])
