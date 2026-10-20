"""Saved-score aggregation only, no GT/model/inference; independently recompute all pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
r=json.loads((p/'RESULTS_PUBLIC.json').read_text());rows=r['rows'];assert len(rows)==128 and len(set(x['alias']for x in rows))==128
checks=0;maxerr=0.
def exact(a,b):
 global checks,maxerr
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maxerr=max(maxerr,err);assert err<2e-12;checks+=np.asarray(a).size
for row in rows:
 for arm in r['means']:
  for k in ['v','t','s','boundary_MAE']:
   exact(np.mean([row['values'][f'Live{s}_{arm}'][k]for s in [41,42,43]]),row['values'][f'Mean_{arm}'][k])
for arm,v in r['means'].items():
 for k,value in v.items():exact(np.mean([x['values']['Mean_'+arm][k]for x in rows]),value)
idx=np.random.default_rng(2026101909).integers(128,size=(10000,128));pairs=0
for name,stat in r['paired'].items():
 a,b=name.split('_minus_');d=None
 for k in ['v','t','s','boundary_MAE']:
  v=np.array([(100 if k!='boundary_MAE'else 1)*(x['values']['Mean_'+a][k]-x['values']['Mean_'+b][k])for x in rows]);s=stat[k]
  exact(v.mean(),s['mean']);exact(np.percentile(v[idx].mean(1),[2.5,97.5]),s['CI95']);assert s['parents']==128;pairs+=1
  if k=='v':d=v
 t=stat['tails'];assert t['positive']==int((d>1e-10).sum())and t['negative']==int((d< -1e-10).sum())and t['flat']==int((abs(d)<=1e-10).sum())
 assert t['below_minus5']==int((d< -5).sum())and t['below_minus3']==int((d< -3).sum())
 exact([np.delete(d,i).mean()for i in [int(np.argmax(d)),int(np.argmin(d))]],t['leave_one_out_mean_range_pp'])
 expected=sorted([{'alias':x['alias'],'delta_pp':float(v)}for x,v in zip(rows,d)],key=lambda v:v['delta_pp'])[:12];assert expected==t['worst']
 for k,s in r['expected_quality'][name].items():
  v=np.array([100*(x['expected_candidate_quality'][a][k]-x['expected_candidate_quality'][b][k])for x in rows]);exact(v.mean(),s['mean']);exact(np.percentile(v[idx].mean(1),[2.5,97.5]),s['CI95']);pairs+=1
# Fixed unobserved actual nodes are spatial quantities and must stay identical for temporal arms.
for row in rows:
 for arm in r['means']:exact(row['values']['Mean_'+arm]['s'],row['values']['Mean_HiResAuxMix']['s'])
v={'status':'pass','saved_score_only_no_new_GT':True,'scalar_checks':int(checks),'paired_bootstrap_checks':pairs,'max_error':maxerr,'all128_parents':True,'spatial_seed_within_parent_mean_exact':True,'all_tails_signed_counts_worst_and_LOO_exact':True,'source_results_sha256':hashlib.sha256((p/'RESULTS_PUBLIC.json').read_bytes()).hexdigest(),'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'epoch':time.time()}
(p/'ROOT_RECOMPUTE_PUBLIC.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
