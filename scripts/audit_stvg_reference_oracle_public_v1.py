"""Portable public-only scalar/paired-CI audit; no media, labels, models or tensors."""
from pathlib import Path
import json,math,statistics
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'results/stvg_reference_oracle/2026-10-17/matched32'
def run():
 s=json.loads((PUB/'RESULTS.json').read_text());pairs=json.loads((PUB/'PAIRED_CONTRASTS.json').read_text())['paired'];rows=s['rows'];checks=0
 assert len(rows)==len({r['source_alias']for r in rows})==32
 assert sum(not r['empty']for r in rows)==25 and sum(r['common_anchors']for r in rows)==67
 for ds,z in s['groups'].items():
  for subset,g in z.items():
   rr=[r for r in rows if r['dataset']==ds and(subset=='all32'or not r['empty'])];assert len(rr)==g['count']
   for a,ss in g['means'].items():
    for support,mm in ss.items():
     for metric,val in mm.items():assert abs(statistics.mean(r['values'][a][support][metric]for r in rr)-val)<1e-13;checks+=1
 for p in pairs:
  rr=[r for r in rows if r['dataset']==p['dataset']and(p['subset']=='all32'or not r['empty'])];values=[]
  for r in rr:
   v={a:x[p['time']][p['metric']]for a,x in r['values'].items()}
   if p['contrast']=='(D-B)-(C-A)':d=100*((v['D']-v['B'])-(v['C']-v['A']))
   else:a,b=p['contrast'].split('-');d=100*(v[a]-v[b])
   assert abs(d-p['per_source_delta_pp'][r['source_alias']])<1e-12;values.append(d);checks+=1
  m=statistics.mean(values);half=t.ppf(.975,len(values)-1)*statistics.stdev(values)/math.sqrt(len(values))
  assert abs(m-p['mean_pp'])<1e-12 and abs((m-half)-p['CI95_t'][0])<1e-11 and abs((m+half)-p['CI95_t'][1])<1e-11;checks+=3
  for field,fn in [('positive',lambda v:v>1e-10),('negative',lambda v:v<-1e-10),('neutral',lambda v:abs(v)<=1e-10)]:assert sum(map(fn,values))==p[field];checks+=1
 for r in rows:
  for time in ['Native','projection']:assert len({r['values'][a][time]['t']for a in r['values']})==1;checks+=1
 print(json.dumps(dict(status='pass',count=32,public_scalar_checks=checks,all_means_signed_pairs_CIs_and_fixed_time_invariant=True,models_or_raw_GT_read=False)))
 return checks
if __name__=='__main__':run()
