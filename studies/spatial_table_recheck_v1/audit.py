from common import *
from score import evaluate_array
import argparse

def main(ds):
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json');truth=old.selected.labels(ds);metrics={r['id']:r for r in read(dest/'metrics.json')['private_rows']};report=read(P/ds/'report.json');assert set(metrics)==set(seal['files'])=={r['id']for r in rows(ds)};mx=0.;arrays=0;scalar=0;counts=dict(queries=0,observations=0,pure_raw_anchor_mismatches=0,Full_rawexpert=0,Full_E_not_expert=0,Full_neither=0,inside_Full_E_not_expert=0);maxcoord=0.
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];again,proof=inputs(ds,row);e=seal['files'][key];assert e['proof']==proof and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(f):v for f,v in g['boxes'].items()}
  with np.load(D/e['path'])as z:
   assert set(again)==set(z.files)and all(np.array_equal(v,z[k])for k,v in again.items());arrays+=len(again);maxcoord=max(maxcoord,proof['independent_log_interp_max_coordinate_difference']);pos=z['positions8'];raw=z['raw_expert_boxes'];cur=z['Full'][pos];ex=np.all(cur==raw,axis=1);ee=np.all(cur==z['E_observed'],axis=1);inside=(z['frame_ids'][pos]>=z['Full_interval'][0])&(z['frame_ids'][pos]<z['Full_interval'][1]);counts['queries']+=1;counts['observations']+=8;counts['pure_raw_anchor_mismatches']+=int(np.any(z['ComboLog'][pos]!=raw,axis=1).sum());counts['Full_rawexpert']+=int(ex.sum());counts['Full_E_not_expert']+=int((~ex&ee).sum());counts['Full_neither']+=int((~ex&~ee).sum());counts['inside_Full_E_not_expert']+=int((inside&~ex&ee).sum())
   for arm in ARMS:
    want=evaluate_array(z[arm],z['frame_ids'],z['positions8'],row['metadata'],gt,g['interval'],z[arm+'_interval'],ds,reference=True);have=metrics[key]['values'][arm]
    for k,a in want.items():
     b=have[k];assert(a is None)==(b is None)
     if a is not None:err=abs(a-b);mx=max(mx,err);assert err<2e-10,(key,arm,k,err);scalar+=1
  if i%250==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 checks=0;rows_=list(metrics.values())
 for arm,m in report['means'].items():
  for f,value in m.items():
   vv=[r['values'][arm][f]for r in rows_ if r['values'][arm][f]is not None];assert(value is None and not vv)or abs(value-float(np.mean(vv)))<1e-12;checks+=1
 assert counts['Full_neither']==counts['pure_raw_anchor_mismatches']==0;rec=dict(status='full_all7arm_coldarrays_rawteacher_contexts_independent_normalized_dense_scores_and_querycoverage_means_audited',queries=len(metrics),parents=len({r['parent']for r in rows_}),all_query_rosters_identical=True,no_missing_extra_duplicate_or_outcome_excluded_queries=True,both_same_Full_time_perquery_t_exact=True,combo_and_SourceTR_time_perquery_t_exact=True,all_array_checks=arrays,independent_scalar_checks=scalar,max_independent_reference_discrepancy=mx,means_checks=checks,independent_interpolator_max_coordinate_roundoff=maxcoord,observed_output_counts=counts,metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/ds/'report.json'),score_sha256=sha(dest/'score_receipt.json'),seal_sha256=sha(dest/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0,new_model=0,method_changed=False);atomic(dest/'audit_receipt.json',rec);atomic(P/ds/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
