"""Postseal actual reward/trajectory response and finite realized candidate headroom."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_realizable_query_opd32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_realizable_query_opd32_v1 import reward
 torch.set_num_threads(4);p=verify();z=read(P/'RESULTS_PUBLIC.json');assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};mainrows={r['alias']:r for r in z['rows']};rows=[]
 for row in p['records']:
  a=row['alias'];ds=row['dataset'];q=row['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));cache={v['m'].tobytes():v for v in o['cache']};pool=o['search']['actions'];valid=[];outs={'E_MBR':o['outputs']['E_MBR']};idx=[]
  if o['active']:
   for j,m in enumerate(pool):
    rec=cache[m.tobytes()];valid.append(rec['valid'])
    if rec['valid']:outs['c'+str(j)]=rec['boxes'];idx.append(j)
  for mode,f in o['fits'].items():
   for t in range(1,3):outs[mode+'_step'+str(t)]=f['path'][t]['boxes']
  mo=dict(o,outputs=outs);mv=metrics(mo,plans[ds][q],gt[ds][0][q],gt[ds][1][q])['values'];ev=mv['E_MBR']['v'];es=mv['E_MBR']['unobserved_actual_s'];v=np.array([mv['c'+str(j)]['v']for j in idx]);rr=o['search']['scores'][1:][idx]if idx else np.empty(0);base=o['search']['base_reward'];good=v>ev+1e-12;bad=v<ev-1e-12
  if good.any()and bad.any():auc=float(((rr[good,None]>rr[None,bad])+.5*(rr[good,None]==rr[None,bad])).mean())
  else:auc=None
  diag=dict(valid_initial_candidates=len(idx),invalid_initial_candidates=sum(not v for v in valid),candidate_pool_GT_best_full_v_gain_pp=100*(max([ev]+list(v))-ev),any_beneficial_candidate=bool(good.any()),beneficial_candidate_reward_above_E=bool((rr[good]>base).any())if good.any()else None,teacher_reward_good_bad_AUC=auc,teacher_search_full_v_gain_pp=100*(mainrows[a]['values']['Latent_search']['v']-ev),initial_candidate_quality_only=True,not_global_OPD_upper_bound=True)
  fits={}
  for mode,f in o['fits'].items():
   tt=np.roll(o['targets'],1,axis=0)if mode=='Latent_shuffle'and len(o['targets'])>1 else o['targets'];path=[]
   for t,entry in enumerate(f['path']):
    v0=mv['E_MBR']if t==0 else mv[mode+'_step'+str(t)];path.append(dict(step=t,used_sparse_reward=reward(entry['boxes'],o['positions'],tt),original_binding_reward=reward(entry['boxes'],o['positions'],o['targets']),full_v_gain_pp=100*(v0['v']-ev),unobserved_actual_s_gain_pp=None if es is None or v0['unobserved_actual_s']is None else 100*(v0['unobserved_actual_s']-es),raw_u_L2=float(o['sigma']*np.linalg.norm(entry['state'])),output_L2=float(np.linalg.norm(entry['boxes']-o['outputs']['E_MBR']))))
   fits[mode]=dict(path=path,actual_used_reward_increased=[path[i+1]['used_sparse_reward']>path[i]['used_sparse_reward']+1e-12 for i in range(2)],actual_GT_v_increased=[path[i+1]['full_v_gain_pp']>path[i]['full_v_gain_pp']+1e-10 for i in range(2)],informative_steps=sum(t['weights']['informative']for t in f['rounds']))
  rows.append(dict(alias=a,dataset=ds,active=o['active'],binding_degenerate=o['binding_degenerate'],sigma=o['sigma'],initial_pool=diag,fits=fits))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,active=sum(r['active']for r in rr),binding_degenerate=sum(r['binding_degenerate']for r in rr),initial_pool={k:paired([r['initial_pool'][k]for r in rr])for k in ['candidate_pool_GT_best_full_v_gain_pp','teacher_search_full_v_gain_pp','beneficial_candidate_reward_above_E','teacher_reward_good_bad_AUC']},fits={})
  for mode in ['Latent_fixed','Latent_live','Latent_shuffle']:
   groups[ds]['fits'][mode]=dict(informative_steps=sum(r['fits'][mode]['informative_steps']for r in rr),actual_used_reward_increased_rounds=sum(sum(r['fits'][mode]['actual_used_reward_increased'])for r in rr),actual_GT_v_increased_rounds=sum(sum(r['fits'][mode]['actual_GT_v_increased'])for r in rr),actual_final_used_reward_gain=paired([r['fits'][mode]['path'][-1]['used_sparse_reward']-r['fits'][mode]['path'][0]['used_sparse_reward']for r in rr]))
 put(P/'ACTUAL_RESPONSE_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,no_GT_step_or_deployment_selection=True,all_GT_offline_after_seal=True,candidate_GT_best_only_initial32_plus_E_diagnostic=True,native_reward_AUC_probability_units_not_pp=True));print({ds:groups[ds]['initial_pool']['candidate_pool_GT_best_full_v_gain_pp']['mean']for ds in groups},flush=True)
if __name__=='__main__':main()
