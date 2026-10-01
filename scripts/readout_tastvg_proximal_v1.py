"""Paired development contrasts, fixed-control trigger and sealed selection."""
import sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_proximal_common_v1 import *
def paired(ds,arm,base='A'):
 from scripts.score_tastvg_best_quick_v1 import source_summary
 aa=read(BASE/ds/arm/'ROWS.json');bb=read(BASE/ds/base/'ROWS.json');rows=[]
 assert len(aa)==len(bb)==384
 for a,b in zip(aa,bb):
  for k in ['parent','order','condition','arrival','expert_scheduled','source_id']:assert a[k]==b[k]
  r={k:a[k] for k in ['source_id','order','condition','expert_scheduled']}
  for k in ['m_vIoU','m_tIoU','sIoU_dense_GT']:r['delta_'+k]=a['Ours_'+k]-b['Ours_'+k]
  rows.append(r)
 return {g:{sub:source_summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],['delta_m_vIoU','delta_m_tIoU','delta_sIoU_dense_GT']) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
def main(mode):
 for ds in DATASETS:
  p=verify(ds);contrast={a:paired(ds,a) for a in ['B','C','D']}
  if mode=='controls':
   write(BASE/ds/'PAIRED_SPATIAL.json',contrast)
   improve=contrast['B']['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean']>0
   write(BASE/ds/'CONTROL_TRIGGER.json',dict(run_fixed_mixture=bool(improve),rule='B positive paired future-corruption mean vs A; not significance',time=time.time()))
   if improve:
    req=copy.deepcopy(read(BASE/ds/'B/REQUEST.json'));req.update(arm='E_fixed',tag='E_fixed');req['method']['fixed_lambda']=read(BASE/ds/'CALIBRATION.json')['fixed_lambda'];write(BASE/ds/'E_fixed/REQUEST.json',req)
  else:
   objective={a:read(BASE/ds/a/'SUMMARY.json')['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean'] for a in ['A','B','C','D']}
   winner='A'
   for a in ['B','C','D']:
    if objective[a]>objective[winner]+1e-12:winner=a
   if (BASE/ds/'E_fixed/COMPLETION.json').exists():write(BASE/ds/'PAIRED_FIXED_CONTROL.json',dict(vs_A=paired(ds,'E_fixed'),vs_B=paired(ds,'E_fixed','B')))
   write(BASE/ds/'SPATIAL_SELECTION.json',dict(status='sealed_before_temporal_trial',arm=winner,objectives=objective,selection_scope='four-arm historically-exposed development, no promotion',fixed_control_is_mechanistic_not_selection_candidate=True,time=time.time()))
   # Native-head evidence calibration uses only stored baseline critic scores.
   spreads=[]
   for cond in p['conditions']:
    for order,seq in p['splits']['search']['orders'].items():
     for at in range(0,len(seq),4):
      x=load(BASE/ds/'A/online'/cond/order/f'{at:05}.pt');spreads.append(float(np.ptp(x['temporal']['scores'])))
   sr=float(np.median([v for v in spreads if v>0])) if any(v>0 for v in spreads) else 1.
   write(BASE/ds/'TEMPORAL_CALIBRATION.json',dict(s_ref=sr,spreads=spreads,GT_used=False,feedback='same cached UniversalVTG max confidence-times-overlap',time=time.time()))
   req=copy.deepcopy(read(BASE/ds/winner/'REQUEST.json'));req.update(arm='T',tag='T',spatial_arm=winner)
   req['head_method']=dict(head_lr=p['params']['lr'],head_teacher_temperature=p['params']['teacher_temperature'],head_s_ref=sr,head_cap_fraction=.005)
   write(BASE/ds/'T/REQUEST.json',req)
if __name__=='__main__':main(sys.argv[1])
