"""Independent saved-score aggregation, no model, fitting or GT access."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,numpy as np
p=Path(__file__).resolve().parent;r=json.loads((p/'RESULTS_PUBLIC.json').read_text());rows=r['rows']
assert len(rows)==128 and len({x['alias']for x in rows})==128
checks=0;maxerr=0.
def same(x,y):
 global checks,maxerr
 e=float(np.max(abs(np.asarray(x)-np.asarray(y))));assert e<2e-12;maxerr=max(maxerr,e);checks+=np.asarray(x).size
for row in rows:
 for name in r['means']:
  for k in ['v','t','s','boundary_MAE']:
   source=[row['values'][f'Live{s}_{name}{s}'if name in ['Fixed','Live']else f'Live{s}_{name}'][k]for s in [41,42,43]]
   same(np.mean(source),row['values']['Mean_'+name][k])
 for name in r['means']:same(row['values']['Mean_'+name]['s'],row['values']['Mean_Strong']['s'])
for name,means in r['means'].items():
 for k,v in means.items():same(np.mean([x['values']['Mean_'+name][k]for x in rows]),v)
idx=np.random.default_rng(2026101909).integers(128,size=(10000,128));boot=0
for key,stat in r['paired'].items():
 a,b=key.split('_minus_')
 for k in ['v','t','s','boundary_MAE']:
  delta=np.array([(1 if k=='boundary_MAE'else 100)*(x['values'][a][k]-x['values'][b][k])for x in rows]);s=stat[k]
  same(delta.mean(),s['mean']);same(np.percentile(delta[idx].mean(1),[2.5,97.5]),s['CI95']);assert s['parents']==128;boot+=1
  if k=='v':d=delta
 t=stat['tails'];assert t['positive']==int((d>1e-10).sum())and t['negative']==int((d< -1e-10).sum())and t['flat']==int((abs(d)<=1e-10).sum())
 assert t['below_minus5']==int((d< -5).sum())and t['below_minus3']==int((d< -3).sum())
 same(np.maximum(d,0).mean(),t['positive_contribution_pp']);same(np.minimum(d,0).mean(),t['negative_contribution_pp'])
 same([np.delete(d,k).mean()for k in [int(np.argmax(d)),int(np.argmin(d))]],t['leave_one_out_mean_range_pp'])
 assert sorted([dict(alias=x['alias'],delta_pp=float(v))for x,v in zip(rows,d)],key=lambda x:x['delta_pp'])[:12]==t['worst']
fits=sum(len(x['policy_updates'])for x in rows);updates=sum(v['actual_updates']for x in rows for v in x['policy_updates'].values());assert fits==1024
v=dict(status='pass',saved_scores_only_no_new_GT_or_model=True,scalar_checks=int(checks),paired_bootstrap_checks=boot,maxerror=maxerr,all128_parents=True,within_parent_seed_aggregation_exact=True,signed_tails_and_LOO_exact=True,actual_fits=fits,actual_updates=updates,results_sha256=hashlib.sha256((p/'RESULTS_PUBLIC.json').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),epoch=time.time())
(p/'ROOT_RECOMPUTE_PUBLIC.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
