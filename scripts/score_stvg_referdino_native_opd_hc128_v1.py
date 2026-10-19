"""All896 fits sealed first. Parent inference; seeds are NOT independent videos."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_native_opd_hc128_v1 import D,P,PRE,verify

def avg(v):
 v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None

def main():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['adapted_fits']==896 and bar['all_outputs_before_new_GT'];start=time.perf_counter()
 for r in bar['files']:assert sha(ROOT/r['path'])==r['sha256']
 gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];step_rows=[];cost=[];eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];seeds=p['seeds']
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));q=r['query_ordinal'];m=metrics(z,plans[q],gt[q],spans[q]);diag={}
  for name,f in z['fits'].items():
   diag[name]=dict(updates=f['actual_updates'],initial_anchor_loss=f['path'][0]['anchor_loss'],final_anchor_loss=f['path'][-1]['anchor_loss'],state_L2=float(f['path'][-1]['state'].norm()),changed_nodes=int(torch.any(f['final']!=f['path'][0]['boxes'],dim=-1).sum()),seconds=f['seconds'],sampled_candidates=0 if name=='Direct'else sum(x['rollout']['cost'].numel()for x in f['rounds']),flat_feedback_rounds=sum(not x['updated']for x in f['rounds']))
   rr=[]
   for k,x in enumerate(f['rounds']):
    v=dict(step=k,updated=x['updated'],training_loss=x['loss'],anchor_loss_before=f['path'][k]['anchor_loss'],anchor_loss_after=f['path'][k+1]['anchor_loss'],state_step_L2=float(x['state_step'].norm()),gradient_L2=None if x['gradient']is None else float(x['gradient'].norm()))
    if x['rollout']is not None:
     w=x['rollout']['weights'];v.update(teacher_KL_from_uniform=float((w*(w.clamp_min(1e-30).log()+np.log(32))).sum(-1).mean()),ESS=float((1/w.square().sum(-1)).mean()),analytic_gradient_error=x['analytic_error'])
    rr.append(v)
   step_rows.append(dict(alias=r['alias'],arm=name,seed=f['seed'],steps=rr))
  for arm in ['Fixed','Live']:m['values'][arm+'_seedmean']={ep:avg([m['values'][f'{arm}_{seed}'][ep]for seed in seeds])for ep in eps}
  # Physical time is exactly shared across real outputs. Averaging three equal
  # floats can change the final bit; check those derived means at roundoff only.
  assert max(abs(v['t']-m['values']['E_MBR']['t'])for v in m['values'].values())<1e-14
  rows.append(dict(alias=r['alias'],dataset='hc2',**m,fit_diagnostics=diag,contracts=z['contracts']));cost.append(dict(alias=r['alias'],**z['compute'],per_fit_seconds={n:f['seconds']for n,f in z['fits'].items()}))
 def summary(arms,pairs):
  return dict(parents=128,means={a:{ep:avg([r['values'][a][ep]for r in rows])for ep in eps}for a in arms},paired={a+'_minus_'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rows])for ep in eps}for a,b in pairs})
 primary=summary(['E_MBR','Direct','Fixed_seedmean','Live_seedmean'],[('Direct','E_MBR'),('Fixed_seedmean','E_MBR'),('Live_seedmean','E_MBR'),('Live_seedmean','Direct'),('Live_seedmean','Fixed_seedmean')]);by_seed={}
 for seed in seeds:
  f,l=f'Fixed_{seed}',f'Live_{seed}';by_seed[str(seed)]=summary(['E_MBR','Direct',f,l],[(f,'E_MBR'),(l,'E_MBR'),(l,'Direct'),(l,f)])
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=128,seeds=seeds,primary=primary,by_seed=by_seed,rows=rows,seed_averaged_metrics_not_averaged_boxes=True,seed_not_independent_parent=True,all896_fits_before_new_GT=True,teacher='native cached selected8',new_teacher_Source_calls=0,development_exposed=True,independent_confirmation=False,held32_and_NH16_not_run=True))
 put(P/'STEP_DIAGNOSTICS_PUBLIC.json',dict(rows=step_rows,mean_teacher_weights_are_constructed_geometry_not_native_teacher_probabilities=True))
 put(P/'COST_BREAKDOWN.json',dict(total_GPU_seconds=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json')),allocations=[read(f)for f in sorted((D/'cost').glob('*.json'))],trajectory_decoder_calls=sum(r['trajectory_decoder_calls']for r in cost),offset_decoder_calls=sum(r['offset_decoder_calls']for r in cost),actual_state_updates=sum(r['actual_updates']for r in cost),sampled_candidate_feedback=sum(x['sampled_candidates']for r in rows for x in r['fit_diagnostics'].values()),new_Source_forwards=0,new_ReferDINO_forwards=0,new_DINO_forwards=0,new_training=0,prior_source_teacher_cost_excluded_but_not_free=True,rows=cost,scoring_CPU_seconds=time.perf_counter()-start))
 print('PRIMARY',{k:{ep:(v[ep]['mean'],v[ep]['CI95'])for ep in ['v','unobserved_actual_s']}for k,v in primary['paired'].items()},flush=True)
 print('SEEDS',{s:{k:g['v']['mean']for k,g in v['paired'].items()}for s,v in by_seed.items()},flush=True)
if __name__=='__main__':main()
