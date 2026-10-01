"""Root closure from sealed artifacts; no new inference, label access or selection."""
import csv, json, sys, time, hashlib, collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tastvg_coordinate_common_v2 import BASE, verify, read, write, sha, DEFAULT
from scripts.audit_tastvg_paper48_public_v1 import check_summary
from vg_tta.tastvg_paper48_metrics_v1 import source_summary

def run(dataset):
 p=verify(dataset);b=BASE/dataset;trials=read(b/'TRIALS.json');sel=read(b/'SELECTION.json')
 assert 0<len(trials)<=72 and all(t['state'] in ['COMPLETE','FAIL'] for t in trials)
 from scripts.prepare_tastvg_coordinate_tuning_v2 import refine
 grids=read(BASE/'GRIDS.json')['coarse']
 for stage,param in [('A','lr'),('B','teacher_temperature')]:
  tt=[t for t in trials if t['stage']==stage]
  valid=[t for t in tt if t['state']=='COMPLETE']
  best=min(valid,key=lambda t:(-t['objective'],t['number']))
  chosen=sel['stages'][stage]
  assert best['tag']==chosen['tag'] and best['params']==chosen['params'] and not chosen['confirmation_used']
  coarse_trials=[t for t in tt if '_coarse_' in t['tag']]
  assert [t['params'][param] for t in coarse_trials]==grids[param]
  cb=min([t for t in coarse_trials if t['state']=='COMPLETE'],key=lambda t:(-t['objective'],t['number']))
  refinement=read(b/f'{stage}_REFINEMENT.json')
  assert refinement['best_coarse']==cb['tag']
  assert refinement['values']==refine(grids[param],cb['params'][param])
  assert [t['params'][param] for t in tt if '_fine_' in t['tag']]==refinement['values']
  for t in tt:
   assert t['params']['rho']==.05
   assert (t['params']['teacher_temperature']==1 if stage=='A' else t['params']['lr']==sel['stages']['A']['params']['lr'])
 assert sel['params']==sel['stages']['B']['params'] and not sel['confirmation_used']
 assert sel['stages']['A']['time']<sel['stages']['B']['time']<sel['time']
 for t in trials:
  if t.get('reused_from'):
   reuse=read(b/'search'/t['tag']/'REUSE.json')
   assert reuse['same_config'] and reuse['source']==t['result_dir']
   assert reuse['score_sha256']==sha(b/t['result_dir']/'SCORE.json')
 assert read(b/'GT_EXPOSURE_confirm.json')['time']>sel['time']
 sources=[{p['rows'][i]['source'] for i in p['splits'][s]['orders']['order1']} for s in ['search','confirm']]
 assert len(sources[0])==32 and len(sources[1])==16 and not sources[0]&sources[1]
 tasks=[(b/t['result_dir'],t) for t in trials if t['state']=='COMPLETE' and not t.get('reused_from')]
 tasks += [(b/'confirmation'/n,None) for n in ['default','lr_only','selected'] if not (b/'confirmation'/n/'REUSE.json').exists() and not (b/'confirmation'/n/'NUMERICAL_FAILURE.json').exists()]
 hashes={};payloads=checks=0;state_links=updates=teachers=metric_checks=reinserts=0
 for d,t in tasks:
  req=read(d/'REQUEST.json');bar=read(d/'PREDICTION_BARRIER.json');score=read(d/'SCORE.json');audit=read(d/'AUDIT.json');rows=read(d/'ROWS.json');summ=read(d/'SUMMARY.json');sp=req['split'];cfg=req['params']
  assert score['params']==cfg and score['audit_sha256']==sha(d/'AUDIT.json') and score['prediction_barrier_sha256']==sha(d/'PREDICTION_BARRIER.json')
  assert bar['time']<score['time'] and bar['GT_read'] is False and bar['model_restored']
  assert audit['status']=='pass' and audit['max_error']==0 and audit['all_predictions_before_scoring']
  assert len(rows)==len(bar['files'])==bar['cells']==audit['state_links']==p['splits'][sp]['total']
  if t:assert t['objective']==score['objective'] and t['params']==cfg and score['time']<sel['time']
  else:assert sel['time']<bar['time'] and cfg==({'default':DEFAULT,'lr_only':sel['stages']['A']['params'],'selected':sel['params']}[d.name])
  assert len({(r['condition'],r['order'],r['arrival']) for r in rows})==len(rows)
  initial=read(d/'SUPPORT.json')['center_sha256']
  for cond in p['conditions']:
   for order,seq in p['splits'][sp]['orders'].items():
    prev=initial
    for i,parent in enumerate(seq):
     f=d/'online'/cond/order/f'{i:05}.json';receipt=read(f);rel=str(f.relative_to(d))
     assert bar['files'][rel]==sha(f) and sha(f.with_suffix('.pt'))==receipt['sha256']
     assert receipt['parent']==parent and receipt['pre_sha']==prev;prev=receipt['post_sha'];payloads+=1
     if not receipt['updated']:assert receipt['pre_sha']==receipt['post_sha']
  for r in rows:
   assert r['parent']==p['splits'][sp]['orders'][r['order']][r['arrival']]
   assert r['expert_scheduled']==(r['arrival']%4==0)
   if not r['expert_scheduled']:assert not r['updated'] and r['delta_m_tIoU']==0
   for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT']:
    assert 0<=r['Frozen_'+k]<=1+1e-12 and 0<=r['Ours_'+k]<=1+1e-12
    assert abs(r['Ours_'+k]-r['Frozen_'+k]-r['delta_'+k])<1e-12;checks+=2
  assert sum(r['updated'] for r in rows)==audit['SGD_updates']
  assert sum(r['expert_scheduled'] for r in rows)==audit['teacher_checks']
  for g in ['clean','corruption']:
   for sub in ['all','nonexpert','expert']:
    rr=[r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
    checks+=check_summary(rr,summ[g][sub])
  assert score['objective']==summ['corruption']['all']['metrics']['delta_m_vIoU']['mean']
  with (d/'SCALARS.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  buckets=collections.defaultdict(list)
  for r in rows:buckets[r['parent'],r['order']].append(r)
  excess=[]
  for (parent,order),rr in sorted(buckets.items()):
   clean=next(r['delta_m_vIoU'] for r in rr if r['condition']=='clean')
   excess.append(dict(parent=parent,order=order,delta_excess=float(np.mean([r['delta_m_vIoU'] for r in rr if r['condition']!='clean'])-clean)))
  es=source_summary(excess,['delta_excess']);checks+=check_summary(excess,es);write(d/'CORRUPTION_EXCESS.json',es)
  for name in ['REQUEST.json','PREDICTION_BARRIER.json','AUDIT.json','SCORE.json','SUMMARY.json','SCALARS.csv','CORRUPTION_EXCESS.json']:hashes[str((d/name).relative_to(b))]=sha(d/name)
  state_links+=audit['state_links'];updates+=audit['SGD_updates'];teachers+=audit['teacher_checks'];metric_checks+=audit['independent_metric_checks'];reinserts+=audit['full_reinsertions']
  print('root verified',dataset,str(d.relative_to(b)),flush=True)
 failures=[]
 for t in trials:
  if t['state']=='FAIL':
   d=b/t['result_dir'];s=read(d/'STATUS.json');assert s['status']=='numerical_failure' and t['objective'] is None
   assert not (d/'SCORE.json').exists();failures.append(dict(trial=t['number'],params=t['params'],done=s['done'],error=s['error'],retry=False));hashes[str((d/'STATUS.json').relative_to(b))]=sha(d/'STATUS.json')
 # Additional paired selected-minus-default estimate is descriptive, not reselection.
 dr=read(b/'confirmation/default/ROWS.json');selected_dir=b/'confirmation/selected'
 if (selected_dir/'REUSE.json').exists():selected_dir=b/read(selected_dir/'REUSE.json')['source']
 sr=read(selected_dir/'ROWS.json') if (selected_dir/'ROWS.json').exists() else None
 paired={}
 for group in (['clean','corruption'] if sr is not None else []):
  paired[group]={}
  for sub in ['all','nonexpert','expert']:
   rr=[]
   for a,c in zip(dr,sr):
    assert all(a[k]==c[k] for k in ['parent','order','condition','arrival','expert_scheduled'])
    assert a['Frozen_m_vIoU']==c['Frozen_m_vIoU']
    if (a['condition']=='clean')==(group=='clean') and (sub=='all' or a['expert_scheduled']==(sub=='expert')):rr.append(dict(parent=a['parent'],order=a['order'],delta_selected_minus_default=c['Ours_m_vIoU']-a['Ours_m_vIoU']))
   ss=source_summary(rr,['delta_selected_minus_default']);checks+=check_summary(rr,ss);paired[group][sub]=ss
 if sr is None:paired=dict(status='unavailable_selected_confirmation_failure',no_reselection=True)
 write(b/'CONFIRMATION_PAIRED.json',paired)
 result=dict(status='pass',dataset=dataset,scheduled=len(trials),reused=sum(bool(t.get('reused_from')) for t in trials),complete=len(trials)-len(failures),failures=failures,search_sources=32,confirmation_sources=16,source_disjoint=True,historically_exposed=True,payload_hashes_checked=payloads,state_receipt_links_checked=state_links,scalar_checks=checks,recorded_SGD_audits=updates,recorded_teacher_checks=teachers,recorded_dual_metric_checks=metric_checks,recorded_exact_full_reinsertions=reinserts,selection_before_confirmation=True,implementation_pins_verified=True,hashes=hashes,time=time.time(),scope='Root rehash of every completed prediction payload and sealed state receipt chain; independent scalar/bootstrap/harm reaggregation; verified recorded independent SGD and dual-metric audits. No new GPU, GT reads or hyperparameter selection.')
 write(b/'ROOT_READBACK.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))
if __name__=='__main__':run(sys.argv[1])
