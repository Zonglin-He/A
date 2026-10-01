"""Self-contained anonymous scalar readback; requires only Python and NumPy."""
import sys,json,collections
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def summary(rows,fields):
 if not rows:return dict(sources=0,cells=0,metrics={})
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_id'],r['order'],r['condition']].append([r[k] for k in fields])
 so=collections.defaultdict(list)
 for (s,o,c),v in groups.items():so[s,o].append(np.mean(v,axis=0))
 sources=collections.defaultdict(list);orders=collections.defaultdict(list)
 for (s,o),v in so.items():z=np.mean(v,axis=0);sources[s].append(z);orders[o].append(z)
 matrix=np.array([np.mean(sources[s],axis=0) for s in sorted(sources)]);rng=np.random.default_rng(20261001);boots=[]
 for i in range(0,10000,100):boots.append(matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1))
 ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0);ov=np.array([np.mean(orders[o],axis=0) for o in sorted(orders)])
 return dict(sources=len(matrix),cells=len(rows),metrics={k:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),query_macro=float(np.mean([r[k] for r in rows])),order_values=ov[:,j].tolist(),order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()) if k.startswith('delta_') else None) for j,k in enumerate(fields)})
def equal(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return sum(equal(a[k],b[k]) for k in a)
 if isinstance(a,list):assert len(a)==len(b);return sum(equal(x,y) for x,y in zip(a,b))
 if a is None or isinstance(a,(str,bool)):assert a==b
 else:assert np.isclose(a,b,atol=1e-12,rtol=1e-10),(a,b)
 return 1
def run(base):
 base=Path(base);checks=0;fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_temporal_rerank','delta_post_fixed_time']
 for ds in ['vidstg','hc2']:
  folder=base/ds;cfg=read(folder/'CONFIG.json');arms=cfg['completed_arms'];scores={};data={}
  for arm in arms:
   out=folder/arm;rows=read(out/'ROWS.json');data[arm]=rows;assert len(rows)==384
   for r in rows:
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:checks+=equal(r['delta_'+k],r['Ours_'+k]-r['Frozen_'+k])
    checks+=equal(r['final_v'],r['Ours_m_vIoU'])
   ss={g:{sub:summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],fields) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
   checks+=equal(ss,read(out/'SUMMARY.json'));scores[arm]=ss['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean']
   step=read(out/'SPATIAL_STEP_ROWS.json');method=read(out/'REQUEST.json')['method']
   for s in step:
    if s['rewards'] is None:continue
    spread=np.ptp(s['rewards']);expected=1. if method['target_mode']=='rank' else method.get('fixed_lambda',float(spread)/(float(spread)+method['s_ref']))
    checks+=equal(s['strength'],expected)
    if method['arrival_radius'] is not None:assert s['actual_arrival_norm']<=method['arrival_radius']+2e-6;checks+=1
   if s.get('flat_noop'):assert s['loss_before']==0 and not s['updated'] and s['global_gradient_norm']==0;checks+=3
   norms=read(out/'ACTUAL_STEP_NORMS.json');valid=[r for r in norms if r['eligible']]
   checks+=equal(read(out/'STEP_STATISTICS.json'),dict(eligible_steps=len(valid),strength_eligible_mean=float(np.mean([r['strength'] for r in valid])) if valid else 0.,actual_step_L2_mean=float(np.mean([r['actual_step_L2'] for r in norms])),actual_step_L2_max=float(max(r['actual_step_L2'] for r in norms))))
  selection=read(folder/'SPATIAL_SELECTION.json');winner='A'
  for arm in ['B','C','D']:
   if scores[arm]>scores[winner]+1e-12:winner=arm
  assert selection['arm']==winner;checks+=equal(selection['objectives'],{a:scores[a] for a in ['A','B','C','D']})
  arows=data['A'];brows=data['B'];paired=[]
  for a,b in zip(arows,brows):
   assert (a['parent'],a['order'],a['condition'])==(b['parent'],b['order'],b['condition'])
   paired.append({**b,'delta_m_vIoU':b['Ours_m_vIoU']-a['Ours_m_vIoU']})
  gain=summary([r for r in paired if r['condition']!='clean' and not r['expert_scheduled']],['delta_m_vIoU'])['metrics']['delta_m_vIoU']['mean']
  assert ('E_fixed' in arms)==(gain>0);checks+=1
  contrasts={}
  for arm in [a for a in arms if a!='A']:
   paired_rows=[]
   for a,b in zip(data[arm],data['A']):
    assert all(a[k]==b[k] for k in ['parent','source_id','condition','order','arrival','expert_scheduled'])
    paired_rows.append({**a,**{'delta_'+k:a['Ours_'+k]-b['Ours_'+k] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT']}})
   contrasts[arm]={g:{sub:summary([r for r in paired_rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],['delta_m_vIoU','delta_m_tIoU','delta_sIoU_dense_GT']) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
  checks+=equal({a:contrasts[a] for a in ['B','C','D']},read(folder/'PAIRED_SPATIAL.json'))
  assert selection['arm']=='A'
  checks+=equal(contrasts['T'],read(folder/'PAIRED_NATIVE_TEMPORAL.json'))
  tcal=read(folder/'TEMPORAL_CALIBRATION.json');tsp=[float(np.ptp(r['temporal_scores'])) for r in data['A'] if r['expert_scheduled']]
  checks+=equal(tcal['spreads'],tsp);checks+=equal(tcal['s_ref'],np.median([x for x in tsp if x>0]) if any(x>0 for x in tsp) else 1.)
  cal=read(folder/'CALIBRATION.json');sp=[float(np.ptp(s['rewards'])) if s['rewards'] is not None else 0. for s in read(folder/'A/SPATIAL_STEP_ROWS.json') if s['step']==0]
  checks+=equal(cal['spreads'],sp);checks+=equal(cal['s_ref'],np.median([x for x in sp if x>0]));checks+=equal(cal['fixed_lambda'],np.mean([x/(x+cal['s_ref']) for x in sp]))
 print(json.dumps(dict(status='pass',scalar_checks=checks,GT_read=False,model_execution=False,scope='anonymous deltas/source-macro and paired source-bootstrap contrasts, strength/caps/step statistics, both calibrations and selection rules'),indent=2))
if __name__=='__main__':run(sys.argv[1])
