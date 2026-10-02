"""Paired future readout and posthoc direction diagnosis of the sealed P0."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_event_support_common_v1 import *
from scripts.score_tastvg_best_quick_v1 import source_summary
ARMS=['A','H']
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
 valid=[r for r in rows if r['rewards'] is not None];first=[r for r in valid if r['step']==0]
 def quant(values):
  return dict(mean=float(np.mean(values)),quantiles=np.quantile(values,[0,.01,.05,.5,.95,.99,1]).tolist()) if len(values) else None
 cos=[r['output_space_cosine'] for r in valid if r['output_space_cosine'] is not None]
 return dict(steps=len(rows),eligible_steps=len(valid),first_eligible_steps=len(first),
  no_op_reasons={reason:sum(r['no_op_reason']==reason for r in valid) for reason in ['flat_rewards','central_selected','zero_selected_gradient','zero_rkl_magnitude']},
  matched_steps=sum(r['magnitude_match_applicable'] for r in valid),
  actual_step=quant([r['actual_step_norm'] for r in valid]),counterfactual_rkl_step=quant([r['counterfactual_rkl_step_norm'] for r in valid]),
  functional_box_movement=quant([r['box_movement_mean_abs'] for r in valid]),
  output_cosine=quant(cos),cosine_positive=sum(v>0 for v in cos),cosine_negative=sum(v<0 for v in cos),cosine_undefined=len(valid)-len(cos),
  selected_distance_decreases=sum(r['selected_distance_decreased'] for r in valid),
  selected_distance_increases=sum(r['selected_loss_after']>r['selected_loss_before']+1e-12 for r in valid),
  selected_central=sum(r['selected_index']==0 for r in valid),flat_rewards=sum(r['selected_flat_noop'] for r in valid),
  unique_useful_top_harm_first=sum(r['unique_useful_top_harm'] for r in first),
  unique_useful_top_severe_harm_first=sum(r['unique_useful_top_harm'] and r['delta_update']<-.05 for r in first),
  selected_useful_harm=sum(r['selected_useful_harm'] for r in valid),
  selected_useful_harm_first=sum(r['selected_useful_harm'] for r in first),
  selected_distance_down_GT_harm=sum(r['selected_distance_decreased'] and r['delta_update']< -1e-12 for r in valid),
  loss_down_GT_harm=sum(r['loss_decreased'] and r['delta_update']< -1e-12 for r in valid),
  GT_harm_steps=sum(r['delta_update']< -1e-12 for r in valid),GT_gain_steps=sum(r['delta_update']>1e-12 for r in valid),
  severe_GT_harm_steps=sum(r['delta_update']<-.05 for r in valid),
  local_gross_gain=float(np.mean([max(0,r['delta_update']) for r in valid])) if valid else None,
  local_gross_loss=float(np.mean([max(0,-r['delta_update']) for r in valid])) if valid else None,
  post_fixed_time_delta=quant([r['delta_update'] for r in valid]))
def support_stats(rows):
 first=[x for x in rows if x['step']==0];observed=[x for x in rows if x['support_observed_this_step']]
 eligible=[x for x in rows if x['rewards'] is not None];raw=[x for x in first if x['raw_valid_reference_frames']>0]
 def mean(field,rr):
  v=[x[field] for x in rr if x.get(field) is not None]
  return float(np.mean(v)) if v else None
 return dict(total_steps=len(rows),first_arrivals=len(first),first_nonempty_reference_arrivals=len(raw),
  raw_first_reference_misses_GT=sum(x['raw_reference_GT_frames']==0 for x in raw),
  first_zero_weighted_reference_mass=sum(x.get('reference_weight_mass')==0 for x in raw),
  first_zero_weighted_GT_reference_mass=sum(x.get('reference_GT_weight_mass')==0 for x in raw),
  empty_expert_steps=sum(x['update_absent_reason']=='empty_expert' for x in rows),
  zero_weighted_mass_steps=sum(x['update_absent_reason']=='zero_weighted_reference_mass' for x in rows),
  observed_steps=len(observed),mean_support_GT_mass_fraction=mean('support_GT_mass_fraction',observed),
  mean_support_GT_frame_coverage=mean('support_GT_frame_coverage',observed),
  mean_support_frame_fraction=mean('support_frame_fraction',observed),
  mean_first_support_GT_mass_fraction=mean('support_GT_mass_fraction',first),
  mean_first_reference_weight_mass=mean('reference_weight_mass',first),
  full_D_down_GT_support_D_up=sum(bool(x.get('full_D_down_GT_support_D_up')) for x in eligible),
  full_D_down_GT_harm=sum(x['selected_full_D_decreases'] and x['delta_update']< -1e-12 for x in eligible),
  weighted_D_down_GT_harm=sum(x['selected_distance_decreased'] and x['delta_update']< -1e-12 for x in eligible))

def run():
 assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['cells']==1536
 for ds in DATASETS:
  p=verify(ds);contrasts={a:paired(ds,a) for a in ['H']};write(BASE/ds/'PAIRED.json',contrasts)
  objective={a:read(BASE/ds/a/'SUMMARY.json')['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean'] for a in ARMS}
  winner='A'
  for a in ['H']:
   if objective[a]>objective[winner]+1e-12:winner=a
  write(BASE/ds/'DEVELOPMENT_DECISION.json',dict(arm=winner,objectives=objective,rule='strict future nonexpert corruption source-macro mean; tie retains earlier A/H-lite',scope='historically exposed development; no production promotion or automatic full recollection',time=time.time()))
  aa=read(BASE/ds/'A/ROWS.json');gross={};cases={};diagnosis={};tails={}
  for arm in ARMS:
   rr=read(BASE/ds/arm/'ROWS.json');pr=[]
   for x,y in zip(rr,aa):
    d=x['Ours_m_vIoU']-y['Ours_m_vIoU'];pr.append({**x,'delta_vs_A':d,'gross_gain_vs_A':max(0,d),'gross_loss_vs_A':max(0,-d)})
   future=subset(pr,'corruption','nonexpert');gross[arm]=source_summary([{**r,'gross_gain_vs_Frozen':max(0,r['Ours_m_vIoU']-r['Frozen_m_vIoU']),'gross_loss_vs_Frozen':max(0,r['Frozen_m_vIoU']-r['Ours_m_vIoU'])} for r in future],['delta_vs_A','gross_gain_vs_A','gross_loss_vs_A','gross_gain_vs_Frozen','gross_loss_vs_Frozen'])
   vals=np.array([r['delta_vs_A'] for r in future]);tails[arm]=dict(arrivals=len(future),harm_over5pp=int((vals<-.05).sum()),gain_over5pp=int((vals>.05).sum()),delta_quantiles=np.quantile(vals,[0,.01,.05,.5,.95,.99,1]).tolist())
   cases[arm]=dict(positive=sorted(future,key=lambda r:-r['delta_vs_A'])[:8],negative=sorted(future,key=lambda r:r['delta_vs_A'])[:8])
   steps=read(BASE/ds/arm/'SPATIAL_STEP_ROWS.json');diagnosis[arm]={g:direction_statistics([r for r in steps if (r['condition']=='clean')==(g=='clean')]) for g in ['clean','corruption']}
  support_diag={a:{g:support_stats([x for x in read(BASE/ds/a/'SPATIAL_STEP_ROWS.json') if (x['condition']=='clean')==(g=='clean')]) for g in ['clean','corruption']} for a in ARMS};write(BASE/ds/'SUPPORT_DIAGNOSIS.json',support_diag)
  for name,obj in [('GROSS_FUTURE_VS_A.json',gross),('DIRECTION_DIAGNOSIS.json',diagnosis),('NEGATIVE_TAILS.json',tails),('CASES.json',cases)]:write(BASE/ds/name,obj)
 write(BASE/'ROOT_READOUT.json',dict(status='completed_pending_root_public_audit',total_arrivals=1536,GT_after_four_stream_seal=True,bootstrap_draws=10000,bootstrap_seed=20261001,time=time.time()))
 print('A/H-lite paired readout complete')
if __name__=='__main__':run()
