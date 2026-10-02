"""Paired future readout and posthoc direction diagnosis of the sealed P0."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_directional_common_v1 import *
from scripts.score_tastvg_best_quick_v1 import source_summary
ARMS=['A','E','F']
def subset(rows,group,sub):
 return [r for r in rows if (r['condition']=='clean')==(group=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
def paired(ds,arm,base='A'):
 aa=read(BASE/ds/arm/'ROWS.json');bb=read(BASE/ds/base/'ROWS.json');rows=[]
 assert len(aa)==len(bb)==384
 for a,b in zip(aa,bb):
  for k in ['parent','order','condition','arrival','expert_scheduled','source_id']:assert a[k]==b[k]
  for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:assert a['Frozen_'+k]==b['Frozen_'+k]
  rows.append({**a,**{'delta_'+k:a['Ours_'+k]-b['Ours_'+k] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']}})
 return {g:{s:source_summary(subset(rows,g,s),['delta_'+k for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]) for s in ['all','expert','nonexpert']} for g in ['clean','corruption']}
def direction_statistics(rows):
 valid=[r for r in rows if r['rewards'] is not None]
 first=[r for r in valid if r['step']==0]
 cos=[r['rkl_preference_cosine'] for r in valid if r['rkl_preference_cosine'] is not None]
 def quant(values):
  return dict(mean=float(np.mean(values)),quantiles=np.quantile(values,[0,.01,.05,.5,.95,.99,1]).tolist()) if len(values) else None
 return dict(steps=len(rows),eligible_steps=len(valid),first_eligible_steps=len(first),
  zero_direction_steps=sum(r['preferred_direction_norm']==0 for r in valid),
  no_op_reasons={reason:sum(r['no_op_reason']==reason for r in valid) for reason in ['no_rank_direction','zero_rkl_magnitude']},
  matched_steps=sum(r['magnitude_match_applicable'] for r in valid),
  actual_step=quant([r['actual_step_norm'] for r in valid]),counterfactual_rkl_step=quant([r['counterfactual_rkl_step_norm'] for r in valid]),
  cosine=quant(cos),cosine_negative=sum(v<0 for v in cos),
  probe_ratio=quant([r['executed_to_probe_radius'] for r in valid]),
  chosen_pair_GT_agrees=sum(r['critic_chosen_pair_GT_agrees'] for r in valid),
  chosen_pair_GT_neutral=sum(r['rank_contrasts'][r['top_pair']]*r['critic_chosen_pair_GT_delta']==0 for r in valid),
  chosen_pair_GT_disagrees=sum(r['rank_contrasts'][r['top_pair']]*r['critic_chosen_pair_GT_delta']<0 for r in valid),
  unique_useful_top_harm_first=sum(r['unique_useful_top_harm'] for r in first),
  unique_useful_top_severe_harm_first=sum(r['unique_useful_top_harm'] and r['delta_update']<-.05 for r in first),
  loss_down_GT_harm=sum(r['loss_decreased'] and r['delta_update']< -1e-12 for r in valid),
  GT_harm_steps=sum(r['delta_update']< -1e-12 for r in valid),
  GT_gain_steps=sum(r['delta_update']>1e-12 for r in valid),
  severe_GT_harm_steps=sum(r['delta_update']<-.05 for r in valid),
  post_fixed_time_delta=quant([r['delta_update'] for r in valid]))
def run():
 assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['cells']==2304
 for ds in DATASETS:
  p=verify(ds);contrasts={a:paired(ds,a) for a in ['E','F']};write(BASE/ds/'PAIRED.json',contrasts)
  objective={a:read(BASE/ds/a/'SUMMARY.json')['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean'] for a in ARMS}
  winner='A'
  for a in ['E','F']:
   if objective[a]>objective[winner]+1e-12:winner=a
  write(BASE/ds/'DEVELOPMENT_DECISION.json',dict(arm=winner,objectives=objective,rule='strict future nonexpert corruption source-macro mean; tie retains earlier A/E',scope='historically exposed development; no production promotion or follow-on job',time=time.time()))
  aa=read(BASE/ds/'A/ROWS.json');gross={};cases={};diagnosis={};tails={}
  for arm in ARMS:
   rr=read(BASE/ds/arm/'ROWS.json');pr=[]
   for x,y in zip(rr,aa):
    d=x['Ours_m_vIoU']-y['Ours_m_vIoU'];pr.append({**x,'delta_vs_A':d,'gross_gain_vs_A':max(0,d),'gross_loss_vs_A':max(0,-d)})
   future=subset(pr,'corruption','nonexpert');gross[arm]=source_summary(future,['delta_vs_A','gross_gain_vs_A','gross_loss_vs_A'])
   vals=np.array([r['delta_vs_A'] for r in future]);tails[arm]=dict(arrivals=len(future),harm_over5pp=int((vals<-.05).sum()),gain_over5pp=int((vals>.05).sum()),delta_quantiles=np.quantile(vals,[0,.01,.05,.5,.95,.99,1]).tolist())
   cases[arm]=dict(positive=sorted(future,key=lambda r:-r['delta_vs_A'])[:8],negative=sorted(future,key=lambda r:r['delta_vs_A'])[:8])
   steps=read(BASE/ds/arm/'SPATIAL_STEP_ROWS.json');diagnosis[arm]={g:direction_statistics([r for r in steps if (r['condition']=='clean')==(g=='clean')]) for g in ['clean','corruption']}
  for name,obj in [('GROSS_FUTURE_VS_A.json',gross),('DIRECTION_DIAGNOSIS.json',diagnosis),('NEGATIVE_TAILS.json',tails),('CASES.json',cases)]:write(BASE/ds/name,obj)
 write(BASE/'ROOT_READOUT.json',dict(status='completed_pending_root_public_audit',total_arrivals=2304,GT_after_six_stream_seal=True,bootstrap_draws=10000,bootstrap_seed=20261001,time=time.time()))
 print('P0 paired readout complete')
if __name__=='__main__':run()
