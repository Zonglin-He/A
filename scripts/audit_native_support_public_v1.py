"""Independent scalar/bootstrap verification without images, GT or checkpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def run(folder):
 folder=Path(folder);rows=json.loads((folder/'SCALAR_ROWS.json').read_text());summary=json.loads((folder/'SUMMARY.json').read_text());checks=0
 assert len(rows)==2304 and len({(r['backbone'],r['parent'],r['condition']) for r in rows})==2304
 def close(a,b):
  nonlocal checks
  assert np.allclose(a,b,atol=1e-10,rtol=1e-10),(a,b);checks+=1
 def ci(v):
  rng=np.random.default_rng(20260930);boot=[]
  for i in range(20):boot.append(np.asarray(v)[rng.integers(0,len(v),(500,len(v)))].mean(axis=1))
  return np.quantile(np.concatenate(boot),[.025,.975],axis=0)
 for model in ['tastvg','tubedetr','ptd']:
  for group in ['clean','corruption']:
   selected=[r for r in rows if r['backbone']==model and (r['condition']=='clean')==(group=='clean')]
   assert len(selected)==(128 if group=='clean' else 640);s=summary[model][group];pair=[];raw=[[],[]];bases=[[],[]];marg=[[],[]];perfect=[0,0];eligible=[0,0];pairedcells=0
   for parent in range(128):
    rr=sorted([r for r in selected if r['parent']==parent],key=lambda r:r['condition']);assert len(rr)==(1 if group=='clean' else 5)
    candidates=[np.array([r[k] for r in rr],float) for k in ['temporal','spatial']]
    masks=[];ratios=[]
    for j,a in enumerate(candidates):
     assert a.shape==(len(rr),6) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
     base=a[:,0];delta=np.max(a,axis=1)-base;mask=base<1;ratio=np.full(len(rr),np.nan);ratio[mask]=100*delta[mask]/(1-base[mask]);ratios.append(ratio);masks.append(mask);perfect[j]+=int((~mask).sum());eligible[j]+=int(mask.sum());raw[j].append(100*delta.mean());bases[j].append(base.mean())
     if mask.any():marg[j].append(ratio[mask].mean())
    mask=masks[0]&masks[1];pairedcells+=int(mask.sum())
    if mask.any():pair.append([ratios[0][mask].mean(),ratios[1][mask].mean()])
   pair=np.array(pair);values=np.c_[pair,pair[:,0]-pair[:,1]];bounds=ci(values)
   assert s['paired_sources']==len(pair) and s['paired_cells']==pairedcells and s['excluded_paired_sources']==128-len(pair)
   for j,key in enumerate(['temporal','spatial','asymmetry']):close(s['paired'][key]['mean'],values[:,j].mean());close(s['paired'][key]['ci95'],bounds[:,j])
   for j,key in enumerate(['temporal','spatial']):
    assert s[key+'_perfect_cells']==perfect[j] and s['marginal'][key]['eligible_cells']==eligible[j]
    close(s['raw_'+key+'_gain_pp'],np.mean(raw[j]));close(s['raw_'+key+'_gain_pp_with_ci']['ci95'],ci(raw[j]));close(s['raw_'+key+'_gain_pp_with_ci']['mean'],np.mean(raw[j]));close(s['baseline_'+key]['mean'],np.mean(bases[j]));close(s['baseline_'+key]['ci95'],ci(bases[j]));close(s['marginal'][key]['mean'],np.mean(marg[j]));close(s['marginal'][key]['ci95'],ci(marg[j]));close(s['fraction_sources_'+key+'_gain_gt5pp'],np.mean(np.array(raw[j])>5))
   assert s['temporal_over_spatial_supported']==bool(bounds[0,2]>0)
   close(s['coverage_mean'],np.mean([r['coverage'][0] for r in selected]));close(s['unique_temporal_mean'],np.mean([r['unique_temporal'] for r in selected]));close(s['unique_spatial_mean'],np.mean([r['unique_spatial'] for r in selected]));close(s['format_invalid_cells'],sum(not r['format_valid'] for r in selected));close(s['observed_burst_fraction_mean'],np.mean([r['observed_burst_fraction'] for r in selected]))
 result=dict(status='pass',cells=2304,sources_per_model=128,independent_scalar_checks=checks,bootstrap=10000,summary_sha256=hashlib.sha256((folder/'SUMMARY.json').read_bytes()).hexdigest(),rows_sha256=hashlib.sha256((folder/'SCALAR_ROWS.json').read_bytes()).hexdigest())
 (folder/'SCALAR_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');run(p.parse_args().folder)
