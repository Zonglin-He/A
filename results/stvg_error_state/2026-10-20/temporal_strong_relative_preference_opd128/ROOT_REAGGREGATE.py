"""Reaggregate saved per-query scores only; never opens annotations or a model."""
from pathlib import Path
import json,time,numpy as np
P=Path(__file__).resolve().parent

def main():
 beg=time.perf_counter();res=json.loads((P/'RESULTS_PUBLIC.json').read_text());rows=res['rows'];assert len(rows)==128 and len({r['alias']for r in rows})==128
 error=0.;count=boot=0
 def close(a,b):
  nonlocal error,count
  x=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));error=max(error,x);count+=np.asarray(a).size;assert x<1e-10,x
 for row in rows:
  for mode in ['Direct','Fixed','Live']:
   for k in ['v','t','s','boundary_MAE']:
    close(np.mean([row['values'][f'Live{s}_{mode}{s}'][k]for s in [41,42,43]]),row['values']['Mean_'+mode][k])
 for mode,v in res['means'].items():
  for k,x in v.items():close(np.mean([r['values']['Mean_'+mode][k]for r in rows]),x)
 idx=np.random.default_rng(2026101909).integers(128,size=(10000,128))
 for key,values in res['paired'].items():
  a,b=key.split('_minus_')
  for k in ['v','t','s','boundary_MAE']:
   d=np.asarray([(100 if k!='boundary_MAE'else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows]);close(d.mean(),values[k]['mean']);close(np.quantile(d[idx].mean(1),[.025,.975]),values[k]['CI95']);boot+=1
  d=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);t=values['tails']
  for field,actual in [('positive',int((d>1e-10).sum())),('negative',int((d< -1e-10).sum())),('flat',int((abs(d)<=1e-10).sum())),('below_minus5',int((d< -5).sum())),('below_minus3',int((d< -3).sum()))]:assert t[field]==actual
  close(t['positive_contribution_pp'],np.maximum(d,0).mean());close(t['negative_contribution_pp'],np.minimum(d,0).mean());loo=(d.sum()-d)/127;close(t['leave_one_out_mean_range_pp'],[loo.min(),loo.max()])
  expected=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,d)],key=lambda x:x['delta_pp'])[:12];assert expected==t['worst']
 for m,diag in res['process'].items():
  for k,v in diag.items():close(np.mean([r['policy_updates'][m+str(s)][k]for r in rows for s in [41,42,43]]),v)
 out=dict(status='pass',independent_saved_score_scalars=count,parent_bootstraps=boot,maxerror=error,all_seed_parent_pairing_exact=True,
 tails_contributions_LOO_worst_exact=True,new_models_or_GT=False,CPU_seconds=time.perf_counter()-beg)
 f=P/'ROOT_SAVED_SCORE_REAGGREGATION_PUBLIC.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
