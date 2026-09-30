"""Tier-0 directional counts from saved S0.6 scalar rows; no model or raw GT."""
import argparse,collections,hashlib,json
from pathlib import Path

def compute(rows,tol=1e-12):
 result={}
 for group in ['corruption','clean']:
  counts=collections.Counter()
  for r in rows:
   if (r['condition']=='clean')!=(group=='clean'):continue
   if r['rewards'] is None:counts['unavailable_cells']+=1;continue
   for k in range(1,9):
    e=r['rewards'][k]-r['rewards'][0];g=r['gt_sIoU'][k]-r['gt_sIoU'][0];se=int(e>tol)-int(e< -tol);sg=int(g>tol)-int(g< -tol)
    counts['comparisons']+=1;counts[{(1,1):'correct_positive',(-1,-1):'correct_negative',(1,-1):'noisy_positive',(-1,1):'noisy_negative'}.get((se,sg),'tie')]+=1
  decisive=sum(counts[k] for k in ['correct_positive','correct_negative','noisy_positive','noisy_negative']);result[group]=dict(counts)|dict(decisive=decisive,noisy_fraction=(counts['noisy_positive']+counts['noisy_negative'])/decisive)
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('rows',type=Path);ap.add_argument('--check',type=Path);a=ap.parse_args();out=compute(json.loads(a.rows.read_text()))
 if a.check:assert out==json.loads(a.check.read_text())['groups']
 print(json.dumps(dict(groups=out,input_sha256=hashlib.sha256(a.rows.read_bytes()).hexdigest(),new_model_execution=False,new_GT_read=False),indent=2))
