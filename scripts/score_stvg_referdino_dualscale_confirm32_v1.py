"""Postseal confirmation evaluation, never used for policy selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_dualscale_confirm32_v1 import D,P,DEV,verify

def avg(xs):
 a=[float(v)for v in xs if v is not None];return float(np.mean(a))if a else None

def main():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_confirmation_GT'];assert sha(DEV/'SELECTION_LOCK.json')==p['selection_sha256']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];steps=[];cost=[];eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];start=time.perf_counter()
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));q=r['query_ordinal'];m=metrics(z,plans[q],gt[q],spans[q]);diag={}
  if p['selected_policy']=='Old':
   for s in p['seeds']:m['values'][f'Old_Live_{s}']=m['values'][f'Selected_Live_{s}']
  for stem in ['Selected_Fixed','Selected_Live','Old_Live']:m['values'][stem+'_seedmean']={ep:avg([m['values'][f'{stem}_{s}'][ep]for s in p['seeds']])for ep in eps}
  for name,f in z['fits'].items():
   diag[name]=dict(updates=f['actual_updates'],initial_anchor_loss=f['path'][0]['anchor_loss'],final_anchor_loss=f['path'][-1]['anchor_loss'],state_L2=float(f['path'][-1]['state'].norm()),changed_nodes=int(torch.any(f['final']!=f['path'][0]['boxes'],dim=-1).sum()),seconds=f['seconds'])
   vv=[]
   for k,x in enumerate(f['rounds']):
    v=dict(step=k,updated=x['updated'],loss=x['loss'],anchor_loss_before=f['path'][k]['anchor_loss'],anchor_loss_after=f['path'][k+1]['anchor_loss'],gradient_L2=None if x['gradient']is None else float(x['gradient'].norm()),state_step_L2=float(x['state_step'].norm()))
    if x['rollout']is not None:v.update(ESS=float((1/x['rollout']['weights'].square().sum(-1)).mean()),analytic_gradient_error=x['analytic_error'])
    vv.append(v)
   steps.append(dict(alias=r['alias'],arm=name,seed=f['seed'],steps=vv))
  rows.append(dict(alias=r['alias'],dataset='hc2',**m,fit_diagnostics=diag));cost.append(dict(alias=r['alias'],**z['compute'],per_fit_seconds={n:f['seconds']for n,f in z['fits'].items()}))
 def summary(suffix):
  a=f'Selected_Live_{suffix}';b=f'Selected_Fixed_{suffix}';o=f'Old_Live_{suffix}';arms=['E_MBR','Direct',a,b,o];pairs=[('Direct','E_MBR'),(a,'E_MBR'),(b,'E_MBR'),(a,'Direct'),(a,b),(a,o)]
  return dict(parents=32,means={a:{ep:avg([r['values'][a][ep]for r in rows])for ep in eps}for a in arms},paired={a+'_minus_'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rows])for ep in eps}for a,b in pairs})
 put(P/'RESULTS_PUBLIC.json',dict(status='frozen_confirmation_completed',count=32,seeds=p['seeds'],selected_policy=p['selected_policy'],selected_scales=p['selected_scales'],primary=summary('seedmean'),by_seed={str(s):summary(str(s))for s in p['seeds']},rows=rows,no_confirmation_based_selection=True,selection_lock_unchanged=True,historical_exposure=p['historical_exposure']))
 put(P/'STEP_DIAGNOSTICS_PUBLIC.json',dict(rows=steps));put(P/'COST_BREAKDOWN.json',dict(total_GPU_seconds=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json')),allocations=[read(f)for f in sorted((D/'cost').glob('*.json'))],rows=cost,trajectory_decoder_calls=sum(x['trajectory_decoder_calls']for x in cost),offset_decoder_calls=sum(x['offset_decoder_calls']for x in cost),actual_state_updates=sum(x['actual_updates']for x in cost),new_Source_query_calls=32,new_Source_offset_calls=64,new_native_ReferDINO_query_calls=32,ReferDINO_image_frames=256,new_expert_models=0,new_network_weight_training=0,scoring_CPU_seconds=time.perf_counter()-start));print('CONFIRM PRIMARY',read(P/'RESULTS_PUBLIC.json')['primary']['paired'],flush=True)
if __name__=='__main__':main()
