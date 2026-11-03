from common_v1 import *
import numpy as np,torch
from core_v1 import derive
torch.set_num_threads(1)
def preflight():
 validate();infer_guard();out=[]
 for ds in COUNTS:
  for task in ['temporal','regions']:
   for row in rows(task,ds)[:2]:
    a,p=derive(task,ds,row);b,q=derive(task,ds,row);assert p==q and set(a)==set(b)and all(np.array_equal(v,b[k])for k,v in a.items());out.append(dict(dataset=ds,task=task,repeat_exact=True,current_interval_and_full_exact=True))
 atomic(D/'preflight_v1.json',dict(status='passed',cases=out,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0));print('preflight passed')
def predict(task,ds):
 p=validate();infer_guard();assert read(D/'preflight_v1.json')['status']=='passed';dest=D/task/ds;assert not(dest/'sealed.json').exists();start=time.perf_counter();files={}
 for i,row in enumerate(rows(task,ds),1):
  resources(start);a,proof=derive(task,ds,row);path=dest/'predictions'/(row['id']+'.npz');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
  with path.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(path.with_suffix('.tmp'),path)
  with np.load(path)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rp=dest/'receipts'/(row['id']+'.json');rec=dict(path=str(path.relative_to(D)),sha256=sha(path),proof=proof,worker=identity(),GT_read=False);atomic(rp,rec);files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  if i%25==0:atomic(dest/'progress.json',dict(stage='predict',completed=i,total=len(rows(task,ds)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start))
 atomic(dest/'sealed.json',dict(files=files,queries=len(files),GT_read=False,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=p['fingerprint']));print(task,ds,'sealed',len(files))
def score(task,ds):
 validate();dest=D/task/ds;seal=read(dest/'sealed.json');rr=rows(task,ds);assert seal['queries']==len(rr)and not seal['GT_read'];truth=selected.labels(ds)
 from full_score import metric
 arms=ARMS_A if task=='temporal'else ARMS_B;pairs=PAIRS_A if task=='temporal'else PAIRS_B;out=[];checks=0;errmax=0.;start=time.perf_counter()
 for i,row in enumerate(rr,1):
  resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for name in arms:vals[name],err=metric(z[name],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[name+'_interval'].tolist(),z['positions8']);checks+=1;errmax=max(errmax,err)
  out.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=len(rr),elapsed_wall_seconds=time.perf_counter()-start))
 # Reused controls must reproduce the already audited full-domain scalars.
 old={r['id']:r for r in read(selected.D/'components'/ds/'metrics.json')['private_rows']}
 for x in out:
  assert x['values']['Source']==old[x['id']]['values']['Source']
  if task=='temporal':assert x['values']['Ours']==old[x['id']]['values']['FullMAP']
 folder=P/task/ds;res=selected.suite().parent_report(out,arms,pairs,folder);selected.suite().signed_tails(out,pairs,folder)
 atomic(dest/'metrics.json',dict(private_rows=out));atomic(dest/'score_receipt.json',dict(status='scored',queries=len(out),official_dense_checks=checks,max_discrepancy=errmax,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),whole_seal_before_GT=True));print(task,ds,'scored',checks)
def audit(task,ds):
 validate();infer_guard();dest=D/task/ds;seal=read(dest/'sealed.json');score=read(dest/'score_receipt.json');assert score['seal_sha256']==sha(dest/'sealed.json')and score['metrics_sha256']==sha(dest/'metrics.json');start=time.perf_counter();checks=0
 for i,row in enumerate(rows(task,ds),1):
  resources(start);a,proof=derive(task,ds,row);e=seal['files'][row['id']];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  checks+=2
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=seal['queries'],elapsed_wall_seconds=time.perf_counter()-start))
 report=read(P/task/ds/'report.json');rr=read(dest/'metrics.json')['private_rows']
 for arm,mm in report['means'].items():
  for k,v in mm.items():
   vv=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 rec=dict(status='verified_complete_task_domain',queries=len(rr),all_arrays_cold_rederived_exact=True,all_input_parent_file_SHA_exact=True,all_means_recomputed=True,SHA_checks=checks,seal_sha256=sha(dest/'sealed.json'),score_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/task/ds/'report.json'),elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),new_Source=0,new_expert=0,new_model=0,new_optimizer=0)
 atomic(dest/'reporting_receipt.json',rec);atomic(P/task/ds/'reporting_receipt.json',rec);print(task,ds,'audited')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','predict','score','audit']);ap.add_argument('--task',choices=['temporal','regions']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([]if x.stage in ['prepare','preflight']else[x.task,x.dataset]))

