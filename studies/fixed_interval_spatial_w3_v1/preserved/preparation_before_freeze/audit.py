from common import *
from score import evaluate_array
import argparse
def main(ds):
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json');truth=old.selected.labels(ds);m={r['id']:r for r in read(dest/'metrics.json')['private_rows']};saved=reused_metrics(ds);assert set(m)==set(seal['files'])==set(saved);arrays=checks=0;mx=0.
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];again,proof=derive(ds,row,independent=True);e=seal['files'][key];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(k):b for k,b in g['boxes'].items()};assert str(g['parent'])==m[key]['parent']==saved[key]['parent']
  with np.load(D/e['path'])as z:
   assert set(z.files)==set(again)and all(np.array_equal(v,z[k])for k,v in again.items());arrays+=len(again)
   for arm in NEW:
    vv=evaluate_array(z[arm],z['frame_ids'],z['positions8'],row['metadata'],gt,g['interval'],z['interval'],ds,reference=True)
    for f,a in vv.items():
     b=m[key]['values'][arm][f];assert(a is None)==(b is None)
     if a is not None:err=abs(a-b);mx=max(mx,err);assert err<2e-10,(key,arm,f,err);checks+=1
  for arm,v in saved[key]['values'].items():assert v==m[key]['values'][arm]
  assert len({v['t']for v in m[key]['values'].values()})==1
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 means=read(P/ds/'report.json')['means'];nm=0
 for arm,v in means.items():
  for f,a in v.items():
   vv=[r['values'][arm][f]for r in m.values()if r['values'][arm][f]is not None];assert(a is None and not vv)or abs(a-float(np.mean(vv)))<1e-12;nm+=1
 rec=dict(status='full_domain_all_input_parent_SHA_cold_independent_numpy_DP_arrays_normalized_denseIoU_reused_metrics_means_verified',queries=len(m),parents=len({r['parent']for r in m.values()}),all_array_checks=arrays,independent_scalar_checks=checks,max_independent_metric_error=mx,means_checks=nm,all_seven_tIoU_exact=True,no_new_posterior_or_features=True,full_method_unchanged=True,seal_sha256=sha(dest/'sealed.json'),score_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/ds/'report.json'),elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0);atomic(dest/'audit_receipt.json',rec);atomic(P/ds/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
