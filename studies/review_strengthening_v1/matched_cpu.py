"""Same eight observations; common selected posterior/interval, competing native paths.

No DP is applied to replacement or interpolation. This isolates space conditional
on the current method's temporal posterior; it is not independent temporal inference.
"""
from review_common import *
import argparse,traceback,gc
ARMS=['SourceMatchedTime','EightReplacement','ExpertInterpolation','Ours','FinalIndependent']
PAIRS=[('Ours',a)for a in ARMS if a!='Ours']
def derive(ds,row):
 import numpy as np
 from stvg.paper_panel import interpolation
 pe,ce=paths(ds,row['id'])
 with np.load(RUN/ds/pe['path'])as z,np.load(selected.D/ce['path'])as cur:
  ids=z['frame_ids'].copy();pos=z['positions8'].copy();assert np.array_equal(pos,cur['positions8'])and np.array_equal(ids,cur['frame_ids'])
  assert np.array_equal(z['Source'],cur['Source'])
  E=z['SharedInit'].copy();native=z['native_boxes'].copy();inter=interpolation(E,ids,pos,native)
  assert np.array_equal(inter,z['Interpolation']),'Original interpolation must remain byte-exact'
  rep=z['Source'].copy();rep[pos]=native;assert np.array_equal(rep,cur['EightReplacement'])
  iv=cur['FullMAP_interval'].copy()
  arrays=dict(SourceMatchedTime=z['Source'].copy(),EightReplacement=rep,ExpertInterpolation=inter,Ours=cur['FullMAP'].copy(),FinalIndependent=cur['FinalIndependent'].copy(),frame_ids=ids,positions8=pos)
  for arm in ARMS:arrays[arm+'_interval']=iv.copy()
  assert np.array_equal(arrays['Ours'],cur['FullMAP'])
  assert all(np.isfinite(v).all()for v in arrays.values())
 proof=dict(parent_prediction_sha256=pe['sha256'],parent_receipt_sha256=pe['receipt_sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],legacy_interpolation_exact=True,same_expert8=True,same_selected_interval=True,competitor_DP=False,GT_read=False)
 return arrays,proof

def predict(ds):
 import numpy as np,torch
 torch.set_num_threads(1);gt_guard();p=validate();who=identity();dest=D/'matched_cpu'/ds
 assert not(dest/'worker.json').exists();atomic(dest/'worker.json',who);files={};start=time.perf_counter()
 for i,row in enumerate(records(ds),1):
  assert time.perf_counter()-start<259200
  a,proof=derive(ds,row);out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json');assert not out.exists()and not rp.exists();out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=proof,worker=who,fingerprint=p['fingerprint'],GT_read=False,new_Source=0,new_expert=0);atomic(rp,rec);files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  if i%25==0:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],seconds=time.perf_counter()-start,GT_read=False))
 assert len(files)==COUNTS[ds];atomic(dest/'sealed.json',dict(queries=len(files),files=files,seconds=time.perf_counter()-start,GT_read=False,worker=who,fingerprint=p['fingerprint']))

def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();dest=D/'matched_cpu'/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=selected.labels(ds);rr=[];checks=0;maxerr=0.
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ARMS:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
  assert len({v['t']for v in vals.values()})==1,'Matched intervals must produce identical tIoU'
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
 folder=P/'matched_cpu'/ds;res=selected.suite().parent_report(rr,ARMS,PAIRS,folder);selected.suite().signed_tails(rr,PAIRS,folder)
 atomic(dest/'metrics.json',dict(private_rows=rr,report=res));atomic(dest/'score_receipt.json',dict(status='scored',queries=len(rr),checks=checks,max_error=maxerr,seal_sha256=sha(dest/'sealed.json'),whole_domain_sealed_before_GT=True,worker=identity()))

def audit(ds):
 import numpy as np,torch
 torch.set_num_threads(1);gt_guard();validate();dest=D/'matched_cpu'/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json');checks=0
 for row in records(ds):
  a,proof=derive(ds,row);e=seal['files'][row['id']];assert e['provenance']==proof and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  checks+=2
 rr=read(dest/'metrics.json')['private_rows'];res=read(P/'matched_cpu'/ds/'report.json')
 for arm,mm in res['means'].items():
  for k,v in mm.items():
   vv=[x['values'][arm][k]for x in rr if x['values'][arm][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 receipt=dict(status='verified_whole_domain',queries=len(rr),prediction_receipt_SHA_checks=checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,all_temporal_IoUs_identical=True,score_sha256=sha(dest/'score_receipt.json'),seal_sha256=sha(dest/'sealed.json'),new_Source=0,new_expert=0,worker=identity())
 atomic(dest/'reporting_receipt.json',receipt);atomic(P/'matched_cpu'/ds/'reporting_receipt.json',receipt)
 (P/'matched_cpu'/ds/'ANALYSIS.md').write_text('# Matched eight-observation spatial controls\n\nComplete full-domain predictions sealed before independent official scoring; independent array/SHA/means audit passed. All methods use the selected current temporal posterior interval; this is a spatial comparison conditional on Ours temporal refinement, not method-independent temporal evidence. EightReplacement retains Source off anchors; ExpertInterpolation reuses the original center/log-size interpolation and corrected E outside the anchor span. Neither competitor receives our DP. FinalIndependent is an internal fixed-candidate control, not an external tracker. SAM 2 is a separately acquired additional-model comparison and is NOT included in this CPU table. All signed positives/negatives, parent confidence intervals, severe tails/rescues and short/long outcomes are in report/TSVs. Research cached time is not deployment latency.\n')

def preflight():
 import numpy as np,torch
 torch.set_num_threads(1);gt_guard();validate();proof=[]
 for ds in COUNTS:
  for row in records(ds)[:2]:
   a,m=derive(ds,row);b,n=derive(ds,row);assert m==n and all(np.array_equal(a[k],b[k])for k in a);proof.append(dict(dataset=ds,all_arrays_repeat_exact=True,legacy_interpolation_exact=True,same_selected_interval=True,competitor_DP=False))
 rec=dict(status='passed',cases=proof,new_Source=0,new_expert=0,new_model=0,new_GT_scoring=0,GT_read=False)
 atomic(D/'matched_cpu_preflight_receipt.json',rec);atomic(P/'matched_cpu_preflight_receipt.json',rec);print(json.dumps(rec))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','predict','score','audit']);ap.add_argument('--dataset',choices=COUNTS);args=ap.parse_args();globals()[args.command](*([args.dataset]if args.command!='preflight'else []))
