"""Independent NumPy ROI sampling, scoped actual updates and official evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_roi_scoped_opd32_v1 import D,P,FINE,PRE,ROI,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxr=maxp=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));x=load(FINE/'features'/(a+'.pt'));old=load(ROI/'predictions'/(a+'.pt'));ct=load(D/'context'/(a+'.pt'));co=load(PRE/'features'/(a+'.pt'));E=x['E'].numpy().astype(np.float64);n=len(E);ids=np.asarray(x['frame_ids']);iv=x['interval'];rel=(np.r_[ids[1:],ids[-1]]>=iv[0])&(np.r_[ids[0],ids[:-1]]<iv[1]);allowed=rel.copy();allowed[x['planned_positions']]=False;assert np.array_equal(allowed,o['allowed'])and np.array_equal(rel,o['relevant']);checks['prediction_only_scope']+=1
  tt,tw=(sample(x['maps'][x['positions']],x['valid'][x['positions']],np.asarray([b.numpy()for b in x['boxes']]).reshape(-1,1,4)))if x['positions']else(np.empty((0,1,9,1024)),np.empty((0,1,9)));tt=tt[:,0];tw=tw[:,0]
  if len(tt):assert abs(tt-ct['templates'].numpy()).max()<3e-5 and abs(tw-ct['weights'].numpy()).max()<3e-6;checks['independent_fine_templates']+=len(tt)
  for new,key in [('Coarse_direct','Layout_readout'),('Coarse_fixed','Fixed_dense'),('Coarse_live','On_policy_dense'),('Coarse_shuffle','Shuffled_OPD')]:
   b=o['outputs'][new];original=old['outputs'][key];assert np.array_equal(b[allowed],original[allowed])and np.array_equal(b[~allowed],E[~allowed]);times=np.arange(int(iv[0]),int(iv[1]));times=times[(times>=ids[0])&(times<=ids[-1])]
   for dim in range(4):assert np.array_equal(np.interp(times,ids,b[:,dim]),np.interp(times,ids,original[:,dim]))
   checks['coarse_scope_retained_exact_and_interval_interp_exact']+=1
  entries=[(o['direct']['candidates'],o['direct']['feedback'],None,None)]
  for mode,f in o['fits'].items():
   for st,t in enumerate(f['rounds']):entries.append((t['candidates'],t['feedback'],o['permutation']if mode=='Fine_shuffle'else None,(mode,st,t,f)))
  for b,fb,perm,entry in entries:
   maps=x['maps'][perm]if perm is not None else x['maps'];valid=x['valid'][perm]if perm is not None else x['valid'];ph,w=sample(maps,valid,b.numpy());ok=w.sum(-1)>1e-6;assert np.array_equal(ok,fb['valid']);safe=ok.copy();safe[~safe.any(1)]=True
   if len(tt):
    mass=w[None]*tw[:,None,None];per=(np.einsum('nkjc,ajc->ankj',ph,tt)*mass).sum(-1)/np.maximum(mass.sum(-1),1e-12);err=float(abs(per-fb['per_anchor_rewards'].numpy()).max());maxr=max(maxr,err);assert err<3e-5;checks['independent_fine_ROI_reward_scalars']+=per.size
   eps=(torch.cat(tuple(o['noise']),1)if entry is None else o['noise'][entry[1]]).numpy();lp=np.where(safe,-.5*(eps*eps).sum(-1),-np.inf);pr=sm(lp);rr=fb['rewards'].numpy();pt=sm(lp+rr/.1);info=ok.any(1)&((np.where(ok,rr,-np.inf).max(1)-np.where(ok,rr,np.inf).min(1))>1e-6)if len(tt)else np.zeros(n,bool);pt[~info]=pr[~info];err=float(abs(pt-fb['prob'].numpy()).max());maxp=max(maxp,err);assert err<1e-12;checks['independent_teacher_probability_scalars']+=pt.size
   if entry is None:
    best=pt.argmax(1);out=E.copy();use=allowed&info;out[use]=b.numpy()[np.arange(n)[use],best[use]];assert np.array_equal(out,o['outputs']['Fine_direct']);checks['actual_direct']+=1
   else:
    mode,st,t,f=entry;eta=t['eta_before'].numpy();center=np.zeros_like(eta)if mode=='Fine_fixed'else eta;assert np.array_equal(center,t['sampling_center']);d=center[:,None]+eps;wantb=np.repeat(E[:,None],17,axis=1);wantb[:,:,:2]+=.25*E[:,None,2:]*d;assert abs(wantb-b.numpy()).max()<1e-12;sp=sm(np.where(safe,-.5*((d-eta[:,None])**2).sum(-1),-np.inf));g=((sp-pt)[...,None]*d).sum(1);g[~(allowed&info)]=0;assert abs(g-t['gradient'].numpy()).max()<1e-12;new=eta-.5*g;assert abs(new-t['eta_after'].numpy()).max()<1e-12;out=E.copy();out[:,:2]+=.25*E[:,2:]*new;out[~allowed]=E[~allowed];assert abs(out-f['path'][st+1]['boxes'].numpy()).max()<1e-12 and not np.any(new[~allowed]);checks['actual_scoped_gradient_state_output_rounds']+=1
  assert torch.equal(o['fits']['Fine_fixed']['path'][1]['boxes'],o['fits']['Fine_live']['path'][1]['boxes']);checks['first_Fixed_Live_exact']+=1
  for b in o['outputs'].values():
   assert np.array_equal(b[~allowed],E[~allowed])and np.array_equal(b[:,2:],E[:,2:]);checks['outside_planned_and_sizes_exactE']+=1
   if not x['positions']:assert np.array_equal(b,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==want;checks['parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_independent_reward_error=maxr,max_teacher_probability_error=maxp,no_GPU_model_or_new_features=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxr,maxp,flush=True)
if __name__=='__main__':main()
