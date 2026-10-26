from common import *
import argparse,gc,traceback

def predict(ds,preflight=False):
 import numpy as np,torch
 from core_v1 import derive
 torch.set_num_threads(1);validate();gt_guard();start=time.perf_counter();files={};rows=cohort(ds)[:2]if preflight else cohort(ds)
 for i,row in enumerate(rows,1):
  guard(start);a,proof,diag=derive(ds,row)
  if preflight:
   b,p,d=derive(ds,row);assert proof==p and diag==d and all(set(a[t])==set(b[t])and all(np.array_equal(v,b[t][k])for k,v in a[t].items())for t in a)
  else:
   key=row['id'];paths={}
   for task,arrays in a.items():
    p=D/ds/task/(key+'.npz');assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
    os.replace(p.with_suffix('.tmp'),p)
    with np.load(p)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
    paths[task]=dict(path=str(p.relative_to(D)),sha256=sha(p))
   rec=dict(outputs=paths,provenance=proof,diagnostics=diag,worker=identity(),GT_read=False);rp=D/ds/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  atomic(D/ds/('preflight_progress.json'if preflight else'predict_progress.json'),dict(completed=i,total=len(rows),worker=identity(),GT_read=False));del a;gc.collect()
 receipt=dict(status='passed_repeat_actual_ROI_pool_and_gamma_replay'if preflight else'sealed_complete_independent_retained192_cohort_before_GT',queries=len(rows),files=files,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,GT_read=False,new_model=0,new_Source=0,new_expert=0)
 atomic(D/ds/('preflight.json'if preflight else'sealed.json'),receipt)

def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();seal=read(D/ds/'sealed.json');assert seal['queries']==192 and not seal['GT_read'];truth=selected.labels(ds);start=time.perf_counter();allrows={'gamma':[],'roi':[]};checks=0;errmax=0.
 for row in cohort(ds):
  guard(start);key=row['id'];entry=seal['files'][key];assert sha(D/entry['receipt_path'])==entry['receipt_sha256'];g=truth[key]
  for task,arms in [('gamma',GAMMA),('roi',ROI)]:
   e=entry['outputs'][task];assert sha(D/e['path'])==e['sha256'];values={}
   with np.load(D/e['path'])as z:
    inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
    for arm in arms:values[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;errmax=max(errmax,err)
   allrows[task].append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 for task,rr in allrows.items():
  out=P/task/ds;arms=GAMMA if task=='gamma'else ROI;ps=pairs(task,ds);report=selected.suite().parent_report(rr,arms,ps,out);selected.suite().signed_tails(rr,ps,out);atomic(D/ds/(task+'_metrics.json'),dict(private_rows=rr,report=report))
 rec=dict(status='whole_standalone192_scored_after_full_subset_seal',queries=192,checks=checks,max_error=errmax,seal_sha256=sha(D/ds/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(D/ds/'score_receipt.json',rec)

def audit(ds):
 import numpy as np,torch
 from core_v1 import derive
 torch.set_num_threads(1);validate();gt_guard();seal=read(D/ds/'sealed.json');assert read(D/ds/'score_receipt.json')['seal_sha256']==sha(D/ds/'sealed.json');start=time.perf_counter()
 for i,row in enumerate(cohort(ds),1):
  guard(start);e=seal['files'][row['id']];assert sha(D/e['receipt_path'])==e['receipt_sha256'];a,p,d=derive(ds,row);assert p==e['provenance']and d==e['diagnostics']
  for task,arrays in a.items():
   zentry=e['outputs'][task];assert sha(D/zentry['path'])==zentry['sha256']
   with np.load(D/zentry['path'])as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
  if i%10==0:atomic(D/ds/'audit_progress.json',dict(completed=i,total=192,worker=identity()))
  del a;gc.collect()
 for task in ['gamma','roi']:
  rr=read(D/ds/(task+'_metrics.json'))['private_rows'];rep=read(P/task/ds/'report.json');assert len(rr)==192
  for a,m in rep['means'].items():
   for k,v in m.items():
    zz=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not zz)or abs(v-float(np.mean(zz)))<1e-12
 rec=dict(status='verified_complete_retained192_all_gamma_arrays_actual_ROI_repool_parent_SHA_and_means',queries=192,all_arrays_rederived_exact=True,all_means_recomputed=True,score_sha256=sha(D/ds/'score_receipt.json'),seal_sha256=sha(D/ds/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0)
 atomic(D/ds/'reporting_receipt.json',rec)
 for task in ['gamma','roi']:atomic(P/task/ds/'reporting_receipt.json',rec)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','predict','score','audit']);p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);a=p.parse_args()
 try:predict(a.dataset,True)if a.command=='preflight'else globals()[a.command](a.dataset)
 except BaseException:atomic(D/'failures'/(a.command+'_'+a.dataset+'.json'),dict(error=traceback.format_exc(),worker=identity()));raise
