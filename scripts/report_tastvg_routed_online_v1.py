"""Public matched A/R summaries and GT preference/update diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_routed_common_v1 import *
import numpy as np,collections
def run():
 from scripts.score_tastvg_best_quick_v1 import source_summary
 for ds in DATASETS:
  p=read(BASE/ds/'PLAN.json');rows={a:read(BASE/ds/a/'ROWS.json') for a in ['A','R']};steps={a:read(BASE/ds/a/'SPATIAL_STEP_ROWS.json') for a in ['A','R']}
  diffs=[]
  for a,r in zip(rows['A'],rows['R']):
   assert all(a[k]==r[k] for k in ['parent','source_id','condition','order','arrival','expert_scheduled'])
   fields=['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5'];d={k:r[k] for k in ['parent','source_id','condition','order','arrival','expert_scheduled']}
   for f in fields:
    d.update({arm+'_'+f:rr['Ours_'+f] for arm,rr in [('A',a),('R',r)]});d['Frozen_'+f]=a['Frozen_'+f];assert a['Frozen_'+f]==r['Frozen_'+f]
    for aa,bb,name in [('R','A','delta_R_A'),('R','Frozen','delta_R_Frozen'),('A','Frozen','delta_A_Frozen')]:d[name+'_'+f]=d[aa+'_'+f]-d[bb+'_'+f]
   d.update(future_both=not r['expert_scheduled'] and a['has_prior_write'] and r['has_prior_write'],A_has_prior_write=a['has_prior_write'],R_has_prior_write=r['has_prior_write'])
   diffs.append(d)
  fields=[k for k in diffs[0] if any(k.startswith(z) for z in ['delta_','A_m_','R_m_','Frozen_m_'])]
  summary={};diagnosis={}
  for group in ['corruption','clean']:
   rs=[r for r in diffs if (r['condition']!='clean')==(group=='corruption')];summary[group]={}
   for sub in ['all','expert','nonexpert','future_nonexpert']:
    take=[r for r in rs if sub=='all' or (r['future_both'] if sub=='future_nonexpert' else r['expert_scheduled']==(sub=='expert'))]
    z=source_summary(take,fields);dv=np.array([r['delta_R_A_m_vIoU'] for r in take]);z.update(gross_gain_pp=float(np.maximum(dv,0).mean()*100) if len(dv) else None,gross_loss_pp=float(-np.minimum(dv,0).mean()*100) if len(dv) else None,positive_cells=int((dv>1e-12).sum()),negative_cells=int((dv< -1e-12).sum()),harm_over5pp_cells=int((dv<-.05).sum()));summary[group][sub]=z
   diagnosis[group]={}
   for arm in ['A','R']:
    ss=[s for s in steps[arm] if (s['condition']!='clean')==(group=='corruption')];first=[s for s in ss if s['step']==0];last={}
    for s in ss:last[s['condition'],s['order'],s['arrival']]=s
    sums=[r for r in rows[arm] if (r['condition']!='clean')==(group=='corruption') and r['expert_scheduled']]
    val=np.array([s['delta_update'] for s in ss]);pref=[s for s in ss if s['rewards'] is not None]
    for s in ss:
     from vg_tta.tastvg_reference_selection_v1 import pairwise
     pair=pairwise(s['rewards'],s['candidate_v']);s.update(pairwise=pair['pairwise_accuracy'],decisive_coverage=pair['decisive_coverage'],oracle_regret=max(s['candidate_v'])-(s['selected_target_GT_v'] if s['rewards'] is not None else s['candidate_v'][0]))
    take=[s for s in first if s['pairwise'] is not None];pairs=source_summary(take,['pairwise','decisive_coverage','oracle_regret'])
    diagnosis[group][arm]=dict(steps=len(ss),expert_arrivals=len(first),updated_steps=sum(s['updated'] for s in ss),empty_arrivals=sum(s['rewards'] is None for s in first),no_scored_valid=sum(s['raw_reference_GT_frames']==0 for s in first),unique_useful_top_harm=sum(s['unique_useful_top_harm'] for s in ss),selected_useful_harm=sum(s.get('selected_useful_harm',False) for s in ss),loss_down_GT_down=sum(s['loss_decreased'] and s['delta_update']< -1e-12 for s in ss),local_positive_steps=int((val>1e-12).sum()),local_negative_steps=int((val< -1e-12).sum()),local_step_gross_gain_pp=float(np.maximum(val,0).mean()*100),local_step_gross_loss_pp=float(-np.minimum(val,0).mean()*100),first_step_ranking=pairs,net_arrival_update=source_summary(sums,['delta_post_fixed_time']))
  write(PUBLIC/ds/'ONLINE_ROWS.json',diffs);write(PUBLIC/ds/'ONLINE_SUMMARY.json',summary);write(PUBLIC/ds/'PIPELINE_DIAGNOSIS.json',diagnosis)
  for arm in ['A','R']:
   for name,value in [('ROWS.json',rows[arm]),('SPATIAL_STEP_ROWS.json',steps[arm]),('AUDIT.json',read(BASE/ds/arm/'AUDIT.json'))]:write(PUBLIC/ds/arm/name,value)
  cases=sorted([r for r in diffs if r['condition']!='clean'],key=lambda x:x['delta_R_A_m_vIoU']);write(PUBLIC/ds/'ONLINE_CASES.json',dict(negative=cases[:5],positive=cases[-5:],cohort_not_selected_by_GT=True))
  write(PUBLIC/ds/'RESOURCES.json',read(BASE/ds/'RESOURCES.json'))
 write(BASE/'ONLINE_REPORT_COMPLETION.json',dict(status='scored_pending_public_audit',time=time.time()));print('ONLINE summaries exported')
if __name__=='__main__':run()
