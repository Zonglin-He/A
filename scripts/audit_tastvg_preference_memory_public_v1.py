"""Scalar audit for the completed, superseded KNN-memory measurements."""
import json,argparse
from pathlib import Path
import numpy as np

def audit(root):
 read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');s=read('SUMMARY.json');checks=0
 assert len(rows)==32 and len({r['parent'] for r in rows})==32
 def check(vals,z):
  nonlocal checks
  a=np.array(vals);assert len(a)==z['n'] and abs(a.mean()-z['mean'])<1e-12;rng=np.random.default_rng(20260929)
  ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]);np.testing.assert_allclose(ci,z['ci95'],atol=1e-12,rtol=0);checks+=1
 for i,r in enumerate(rows):
  assert r['position']==i+1 and r['expert']==(i%4==0)
  if i:assert r['arrival_memory_hash']==rows[i-1]['after_memory_hash']
  for a,k in r['selected'].items():assert r['arms'][a]==r['candidate_metrics'][k]
  if not r['expert']:assert r['arrival_memory_hash']==r['after_memory_hash']
 for group,z in s.items():
  rr=rows if group=='all' else [r for r in rows if r['expert']==(group=='expert')]
  for a in z['arms']:
   for m,st in z['arms'][a].items():check([r['arms'][a][m] for r in rr],st)
  for comp in z['comparisons']:
   a,b=comp.split(' - ')
   for m,st in z['comparisons'][comp].items():check([r['arms'][a][m]-r['arms'][b][m] for r in rr],st)
  assert z['changed']==sum(r['selected']['Fast + Preference Memory']!=r['selected']['Budgeted Rerank'] for r in rr)
 return dict(status='pass',arrivals=32,bootstrap_scalar_checks=checks,superseded=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
