"""Independent NumPy ROI, matched candidate/step/state, metrics and bootstrap audit."""
from pathlib import Path
import os,sys,collections,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_dataset_formal64_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');z=read(P/'RESULTS_PUBLIC.json');rr={r['alias']:r for r in z['rows']};inp=read(D/'INPUT_DEPENDENCY_AUDIT.json');assert inp['status']=='pass'
 # Independently reconstruct locked identity-only roster and exclusions, no scores.
 excluded={'hc2':set(),'vidstg':set()}
 for rec in p['excluded_manifests']:
  assert sha(ROOT/rec['path'])==rec['sha256'];old=read(ROOT/rec['path'])
  for key in ['records','target','source']:
   rows=old.get(key,[])
   if isinstance(rows,list):
    for row in rows:
     if isinstance(row,dict)and row.get('dataset')in excluded and 'parent'in row:excluded[row['dataset']].add(row['parent'])
 pool=read(ROOT/'artifacts/stvg_dino_supported_source256_v1/PLAN_LOCK.json')['records']
 for ds in excluded:
  eligible=[r for r in pool if r['dataset']==ds and r['parent']not in excluded[ds]];assert len(eligible)==p['eligible_counts'][ds];eligible.sort(key=lambda r:hashlib.sha256((p['seed']+'|'+ds+'|'+r['parent']).encode()).hexdigest());actual=[r for r in p['records']if r['dataset']==ds];assert [(r['parent'],r['query_ordinal'])for r in actual]==[(r['parent'],r['query_ordinal'])for r in eligible[:32]]
 for f,v in inp['dependencies'].items():assert sha(ROOT/f)==v['sha256']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};cnt=collections.Counter();maxreward=maxprob=maxstate=0.
 for r in p['records']:
  a=r['alias'];x=load(D/'features'/(a+'.pt'));o=load(D/'predictions'/(a+'.pt'));E=x['E'].numpy().astype(float);ids=np.array(x['frame_ids']);iv=x['interval'];allow=(np.r_[ids[1:],ids[-1]]>=iv[0])&(np.r_[ids[0],ids[:-1]]<iv[1]);allow[x['planned_positions']]=False;assert np.array_equal(allow,o['allowed']);cnt['predicted_scope']+=1
  seed=int.from_bytes(hashlib.sha256(('ridge-dense32-v1|formal64-v1|'+a).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);noise=[]
  for _ in range(2):e=rng.standard_normal((len(E),8,2));noise.append(np.concatenate((np.zeros((len(E),1,2)),e,-e),1))
  assert np.array_equal(np.stack(noise),o['noise']);cnt['locked_noise_independent_reconstruction']+=1
  A=len(x['positions']);tt,tw=sample(x['maps'][x['positions']],x['valid'][x['positions']],np.asarray(x['boxes']).reshape(-1,1,4))if A else(np.empty((0,1,9,1024)),np.empty((0,1,9)));tt=tt[:,0];tw=tw[:,0]
  for mode,f in o['fits'].items():
   step=p['configuration']['step_size_by_target_dataset'][r['dataset']];fixed=mode=='Fixed'
   for st,t in enumerate(f['rounds']):
    fb=t['feedback'];b=t['candidates'].numpy();ph,w=sample(x['maps'],x['valid'],b);ok=w.sum(-1)>1e-6;assert np.array_equal(ok,fb['valid']);safe=ok.copy();safe[~ok.any(1)]=True
    if A:
     mass=w[None]*tw[:,None,None];per=(np.einsum('nkjc,ajc->ankj',ph,tt)*mass).sum(-1)/np.maximum(mass.sum(-1),1e-12);er=float(abs(per-fb['per_anchor_rewards'].numpy()).max());maxreward=max(maxreward,er);assert er<3e-5;cnt['independent_ROI_reward_scalars']+=per.size
    eps=noise[st];lp=np.where(safe,-.5*(eps*eps).sum(-1),-np.inf);prior=sm(lp);reward=fb['rewards'].numpy();q=sm(lp+reward/.1);info=ok.any(1)&((np.where(ok,reward,-np.inf).max(1)-np.where(ok,reward,np.inf).min(1))>1e-6)if A else np.zeros(len(E),bool);q[~info]=prior[~info];er=float(abs(q-fb['prob'].numpy()).max());maxprob=max(maxprob,er);assert er<1e-12
    eta=t['eta_before'].numpy();center=np.zeros_like(eta)if fixed else eta;assert np.array_equal(center,t['sampling_center']);d=center[:,None]+eps;want=np.repeat(E[:,None],17,axis=1);want[:,:,:2]+=.25*E[:,None,2:]*d;assert abs(want-b).max()<1e-12;sp=sm(np.where(safe,-.5*((d-eta[:,None])**2).sum(-1),-np.inf));g=((sp-q)[...,None]*d).sum(1);g[~(allow&info)]=0;new=eta-step*g;er=float(abs(new-t['eta_after'].numpy()).max());maxstate=max(maxstate,er);assert er<1e-12;out=E.copy();out[:,:2]+=.25*E[:,2:]*new;out[~allow]=E[~allow];assert abs(out-f['path'][st+1]['boxes'].numpy()).max()<1e-12;cnt['actual_KL_gradient_step_output_rounds']+=1
   assert np.array_equal(f['final'],o['outputs'][mode]);cnt['final_readouts']+=1
  assert torch.equal(o['fits']['Fixed']['path'][1]['boxes'],o['fits']['Live']['path'][1]['boxes']);cnt['first_Fixed_Live_exact']+=1
  for b in o['outputs'].values():
   assert np.array_equal(b[~allow],E[~allow])and np.array_equal(b[:,2:],E[:,2:]);cnt['scope_planned_size_exactE']+=1
   if not A:assert np.array_equal(b,E);cnt['no_reference_exact_noop']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rr[a],cnt)
 for ds,g in z['groups'].items():
  rows=[r for r in z['rows']if r['dataset']==ds]
  for key,ep in g['paired'].items():
   a,b=key.split('_minus_')
   for e,want in ep.items():assert paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rows])==want;cnt['parent_bootstrap']+=1
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(cnt),max_ROI_reward_error=maxreward,max_probability_error=maxprob,max_state_error=maxstate,Source_exact_assertions_at_runtime_not_independent_GPU_rerun=True,capture_dependencies_pinned_before_model_and_reverified_after=True,all_outputs_sealed_before_GT=True));print('root audit pass',dict(cnt),flush=True)
if __name__=='__main__':main()
