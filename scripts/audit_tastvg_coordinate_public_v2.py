"""Recompute all public trial/confirmation aggregate statistics without private assets."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_tastvg_paper48_public_v1 import read_rows,check_summary

def run(root):
 b=Path(root);read=lambda p:json.loads(p.read_text());trials=read(b/'TRIALS.json');sel=read(b/'SELECTION.json');checks=0;cells=0
 assert 0<len(trials)<=72 and not sel['confirmation_used']
 valid=[t for t in trials if t['state']=='COMPLETE']
 for stage in ['A','B']:
  best=min([t for t in valid if t['stage']==stage],key=lambda t:(-t['objective'],t['number']))
  assert best['tag']==sel['stages'][stage]['tag'] and best['params']==sel['stages'][stage]['params']
 directories=sorted(set(b/t['result_dir'] for t in valid))+[d for d in (b/'confirmation').iterdir() if (d/'SCALARS.csv').exists()]
 for d in directories:
  rows=read_rows(d/'SCALARS.csv');s=read(d/'SUMMARY.json');a=read(d/'AUDIT.json');score=read(d/'SCORE.json');req=read(d/'REQUEST.json')
  ns,total=(32,384) if req['split']=='search' else (16,192)
  assert len(rows)==a['state_links']==total and len({r['parent'] for r in rows})==ns
  assert len({(r['parent'],r['order'],r['condition']) for r in rows})==total
  assert sum(r['updated'] for r in rows)==a['SGD_updates'] and sum(r['expert_scheduled'] for r in rows)==a['teacher_checks']
  for r in rows:
   assert r['expert_scheduled']==(r['arrival']%4==0)
   if not r['expert_scheduled']:assert not r['updated'] and r['delta_m_tIoU']==0
   for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT']:
    assert 0<=r['Frozen_'+k]<=1+1e-12 and 0<=r['Ours_'+k]<=1+1e-12
    assert abs(r['Ours_'+k]-r['Frozen_'+k]-r['delta_'+k])<1e-12;checks+=2
  for g in ['clean','corruption']:
   for sub in ['all','nonexpert','expert']:checks+=check_summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],s[g][sub])
  assert score['objective']==s['corruption']['all']['metrics']['delta_m_vIoU']['mean'];cells+=len(rows)
  if req['split']=='search':assert next(t for t in trials if t['tag']==req['tag'])['objective']==score['objective']
  else:assert req['params']=={'default':trials[0]['params'],'lr_only':sel['stages']['A']['params'],'selected':sel['params']}[d.name]
 for t in trials:
  if t['state']=='FAIL':assert t['objective'] is None and read(b/t['result_dir']/'FAILURE.json')['status']=='numerical_failure'
 default=read_rows(b/'confirmation/default/SCALARS.csv');selected_path=b/'confirmation/selected/SCALARS.csv';selected_dir=b/'confirmation/selected'
 if (selected_dir/'REUSE.json').exists():selected_dir=b/read(selected_dir/'REUSE.json')['source']
 selected_path=selected_dir/'SCALARS.csv';selected=read_rows(selected_path) if selected_path.exists() else None
 paired=read(b/'CONFIRMATION_PAIRED.json')
 if selected is None:assert paired['status']=='unavailable_selected_confirmation_failure'
 for g in (['clean','corruption'] if selected is not None else []):
  for sub in ['all','nonexpert','expert']:
   rr=[]
   for a,c in zip(default,selected):
    assert all(a[k]==c[k] for k in ['parent','order','condition','arrival','expert_scheduled','Frozen_m_vIoU'])
    if (a['condition']=='clean')==(g=='clean') and (sub=='all' or a['expert_scheduled']==(sub=='expert')):rr.append(dict(parent=a['parent'],order=a['order'],delta_selected_minus_default=c['Ours_m_vIoU']-a['Ours_m_vIoU']))
   checks+=check_summary(rr,paired[g][sub])
 return dict(status='pass',scheduled=len(trials),complete=len(valid),failures=len(trials)-len(valid),cells=cells,scalar_checks=checks,scope='All public scalar outcomes, source/order bootstrap and negative tails; not new model inference or private GT metric recomputation')
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
