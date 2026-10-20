"""Independent arithmetic on saved per-parent scores only, no annotations/models."""
from pathlib import Path
import json,time,numpy as np
P=Path(__file__).resolve().parent

def main():
 beg=time.perf_counter();res=json.loads((P/'RESULTS_PUBLIC.json').read_text());rows=res['rows'];assert len(rows)==128 and len({r['alias']for r in rows})==128
 errors=0.;count=boot=0
 def close(a,b):
  nonlocal errors,count
  d=float(np.max(abs(np.asarray(a)-np.asarray(b))));errors=max(errors,d);count+=np.asarray(a).size;assert d<1e-10,d
 for r in rows:
  for c in ['KL0p1','KL0p3','KL1']:
   for k in ['v','t','s','boundary_MAE']:close(r['values']['Mean_'+c][k],np.mean([r['values'][f'Live{s}_{c}Live{s}'][k]for s in [41,42,43]]))
 for c,values in res['means'].items():
  for k,v in values.items():close(v,np.mean([r['values']['Mean_'+c][k]for r in rows]))
 index=np.random.default_rng(2026101909).integers(128,size=(10000,128))
 for name,values in res['paired'].items():
  a,b=name.split('_minus_')
  for k in ['v','t','s','boundary_MAE']:
   d=np.asarray([(100 if k!='boundary_MAE'else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows]);close(d.mean(),values[k]['mean']);close(np.quantile(d[index].mean(1),[.025,.975]),values[k]['CI95']);boot+=1
  d=np.asarray([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);t=values['tails']
  for key,v in [('positive',int((d>1e-10).sum())),('negative',int((d< -1e-10).sum())),('flat',int((abs(d)<=1e-10).sum())),('below_minus5',int((d< -5).sum())),('below_minus3',int((d< -3).sum()))]:assert v==t[key]
  close(np.maximum(d,0).mean(),t['positive_contribution_pp']);close(np.minimum(d,0).mean(),t['negative_contribution_pp']);loo=(d.sum()-d)/127;close([loo.min(),loo.max()],t['leave_one_out_mean_range_pp'])
  worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,d)],key=lambda x:x['delta_pp'])[:12];assert worst==t['worst']
 deltas={c:np.array([100*(r['values']['Mean_'+c]['v']-r['values']['Mean_Strong']['v'])for r in rows])for c in ['KL1','KL0p3','KL0p1']};base=deltas['KL1']
 for c,z in res['contribution_decomposition'].items():
  x=deltas[c];close(z['positive_contribution_increase_pp'],np.maximum(x,0).mean()-np.maximum(base,0).mean());close(z['negative_contribution_change_pp'],np.minimum(x,0).mean()-np.minimum(base,0).mean());close(z['net_change_pp'],np.mean(x-base))
  for key,mask in [('coefficient1_positive',base>1e-10),('coefficient1_negative',base< -1e-10),('coefficient1_flat',abs(base)<=1e-10)]:
   g=z['groups'][key];assert g['parents']==int(mask.sum());close(g['coefficient1_contribution_to_all128_pp'],base[mask].sum()/128);close(g['new_contribution_to_all128_pp'],x[mask].sum()/128)
   assert g['new_positive']==int((x[mask]>1e-10).sum())and g['new_negative']==int((x[mask]< -1e-10).sum())and g['new_flat']==int((abs(x[mask])<=1e-10).sum())
 decision=json.loads((P/'DECISION_PUBLIC.json').read_text());best=max(['KL1','KL0p3','KL0p1'],key=lambda c:res['means'][c]['v']);assert decision['development_selection_only']==best
 changed=sum(v['posterior_L1']>0 for r in rows for v in r['policy_updates'].values());assert len([v for r in rows for v in r['policy_updates'].values()])==768
 components={}
 for c in ['KL0p1','KL0p3']:
  x=deltas[c];v={}
  for name,d in [('positive_contribution_increase_pp',np.maximum(x,0)-np.maximum(base,0)),('negative_contribution_change_pp',np.minimum(x,0)-np.minimum(base,0)),('net_change_pp',x-base)]:
   v[name]=dict(mean=float(d.mean()),CI95=np.quantile(d[index].mean(1),[.025,.975]).tolist(),parents=128)
  components[c]=v
 f=P/'CONTRIBUTION_INTERVALS_PUBLIC.json';assert not f.exists();f.write_text(json.dumps(dict(status='saved_score_explanation',paired_parent_bootstrap=10000,components=components,not_online_GT_selection=True,not_independent_confirmation=True),indent=2)+'\n')
 out=dict(status='pass',saved_score_scalars=count,parent_bootstraps=boot,maxerror=errors,contribution_expansion_groups_and_tails_exact=True,
 parent_seed_pairing_exact=True,development_selection_recomputed=best,changed_new_posterior_fits=changed,total_new_fits=768,new_models_or_GT=False,CPU_seconds=time.perf_counter()-beg)
 f=P/'ROOT_SAVED_SCORE_REAGGREGATION_PUBLIC.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
