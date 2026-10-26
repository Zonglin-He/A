"""Independent scalar export integrity from anonymous public TSVs, no GT/model."""
from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,csv,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[2];P=R/'results/full_tube_controls_side_v1';D=R/'outputs/full_tube_controls_side_v1'
ARMS=['Source','EightReplacementNative','SharedCorrectionNative','ExpertInterpolationNative','NoTemporalRefinement','FullMAP']

def verify(ds):
 start=time.perf_counter();p=P/ds;report=json.loads((p/'report.json').read_text());rows=list(csv.DictReader((p/'all_metrics_signed_parent.tsv').open(),delimiter='\t'));assert len(rows)=={'hc2':1902,'vidstg':10303}[ds];checks=0;maxerr=0
 def group(rr,expected):
  nonlocal checks,maxerr
  parents=sorted({r['parent']for r in rr});index={k:i for i,k in enumerate(parents)};n=len(parents);w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);counts=np.zeros(n);keys=list(expected);sums=np.zeros((n,len(keys)*2));deltas=[]
  for r in rr:
   i=index[r['parent']];counts[i]+=1;d=[]
   for k in keys:
    a,b=k.split('-',1)
    d += [100*(float(r[a+'_'+f])-float(r[b+'_'+f]))for f in ['t','v']]
   sums[i]+=d;deltas.append(d)
  den=w@counts;boot=(w@sums)/den[:,None];means=np.asarray(deltas).mean(0);CI=np.quantile(boot,[.025,.975],axis=0)
  for i,k in enumerate(keys):
   for j,f in enumerate(['t','v']):
    v=expected[k][f];err=max(abs(means[2*i+j]-v['mean']),*abs(CI[:,2*i+j]-np.asarray(v['CI95'])));assert err<2e-11,(ds,k,f,err);maxerr=max(maxerr,err);checks+=1
  return len(parents)
 assert group(rows,report['paired'])==report['parents']
 for name,g in report['timegroups'].items():assert group([r for r in rows if r['duration_group']==name],g['paired'])>0
 for arm in ARMS:
  for f,v in report['means'][arm].items():
   key=arm+'_'+f
   if key not in rows[0]:continue
   vals=[float(r[key])for r in rows if r[key]!=''];assert(v is None and not vals)or abs(np.mean(vals)-v)<2e-12
 rec=dict(status='verified_public_scalar_means_and_all_parent_paired_tv_CIs',queries=len(rows),parents=report['parents'],paired_metric_CI_checks=checks,max_mean_or_CI_absolute_error=maxerr,full_and_short_long=True,public_full_metrics_tsv_sha256=hashlib.sha256((p/'all_metrics_signed_parent.tsv').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_model=0,new_Source=0,new_expert=0,new_GT_score=0,elapsed_wall_seconds=time.perf_counter()-start);(p/'public_integrity.json').write_text(json.dumps(rec,indent=2)+'\n');(D/ds/'public_integrity.json').write_text(json.dumps(rec,indent=2)+'\n');print(ds,rec,flush=True)
if __name__=='__main__':
 for ds in ['hc2','vidstg']:verify(ds)
