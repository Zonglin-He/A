from common import *
from score import measure
import argparse

def main(ds):
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['metrics_sha256']==sha(dest/'metrics.json')and sc['seal_sha256']==sha(dest/'sealed.json');metrics={r['id']:r for r in read(dest/'metrics.json')['private_rows']};truth=old.selected.labels(ds);arrays=checks=0;mx=coord=0.
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];a,proof=derive(ds,row,reference=True);e=seal['files'][key];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];assert metrics[key]['parent']==str(g['parent']);gt={int(k):b for k,b in g['boxes'].items()}
  with np.load(D/e['path'])as z:
   assert set(a)==set(z.files)
   for k,v in a.items():
    err=float(np.max(abs(v-z[k])));coord=max(coord,err);assert err<2e-10,(key,k,err);arrays+=1
   for arm in ARMS:
    vv=measure(z,arm,row['metadata'],gt,g['interval'],ds,reference=True)
    for f,x in vv.items():
     y=metrics[key]['values'][arm][f];assert(x is None)==(y is None)
     if x is not None:err=abs(x-y);mx=max(mx,err);assert err<2e-10,(key,arm,f,err);checks+=1
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 rep=read(P/ds/'report.json');means=0
 for arm,vals in rep['means'].items():
  for f,v in vals.items():
   vs=[r['values'][arm][f]for r in metrics.values()if r['values'][arm][f]is not None];assert(v is None and not vs)or abs(v-float(np.mean(vs)))<1e-12;means+=1
 rec=dict(status='verified_fresh_raw8_E_independent_consensus_Full_independentDP_plainlinear_sharedTime_common_dense_support_SHA_parent_means',queries=len(metrics),parents=len({r['parent']for r in metrics.values()}),array_checks=arrays,scalar_checks=checks,max_independent_scalar_error=mx,max_array_roundoff=coord,means_checks=means,same_dense_event_support_for_all3=True,no_method_or_anchor_fix=True,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),score_sha256=sha(dest/'score_receipt.json'),report_sha256=sha(P/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0,new_model=0);atomic(dest/'audit_receipt.json',rec);atomic(P/ds/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
