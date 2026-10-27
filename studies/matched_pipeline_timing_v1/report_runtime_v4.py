"""Independent CPU validation and paired repeat-averaged latency/memory export."""
from runtime_v4_common import *
import csv
def run():
 import numpy as np
 plan=validate_runtime();sealed=read(D/'sealed.json');assert sealed['status']=='all1536_measured_complete_paths_exact';assert sha(D/'measurements.json')==sealed['measurement_sha256'];assert read(D/'GPU_exit.json')['code']==0;assert read(D/'teacher_exit.json')['code']==0
 measurements=read(D/'measurements.json');items=measurements['items'];assert len(items)==1536
 references_by_ds={ds:references(ds,rr)for ds,rr in read(D/'selection.json').items()};selection=read(D/'selection.json');tuples=set();checks=0
 for e in items:
  ds,i,rep,arm=e['dataset'],e['query'],e['repeat'],e['arm'];assert (ds,i,rep,arm)not in tuples;tuples.add((ds,i,rep,arm));key=selection[ds][i-1]['id'];ref=references_by_ds[ds][key]
  assert sha(D/e['prediction_path'])==e['prediction_sha256']
  with np.load(D/e['prediction_path'])as z:
   assert np.array_equal(z['boxes'],ref[arm])and np.array_equal(z['interval'],ref[arm+'_interval'])and np.array_equal(z['Source'],ref['Source'])and np.array_equal(z['positions8'],ref['positions8'])and np.array_equal(z['frame_ids'],ref['frame_ids'])
  assert e['RGB_sha256']==ref['RGB_sha256']and e['Source_forwards']==1 and e['ReferDINO_queries']==int(arm=='Current')and e['EdgeTAM_queries']==int(arm=='EdgeTAM');assert e['peak_pipeline_allocated_bytes']>0 and e['NVML']['sampled_process_peak_bytes']>0
  checks+=1
 assert tuples=={(ds,i,r,a)for ds in ['hc2','vidstg']for i in range(1,129)for r in range(1,4)for a in ['Current','EdgeTAM']}
 results={};allpaired=[]
 for ds in ['hc2','vidstg']:
  paired=[]
  for i in range(1,129):
   row=dict(query=i)
   for arm in ['Current','EdgeTAM']:
    sub=[e for e in items if e['dataset']==ds and e['query']==i and e['arm']==arm];assert len(sub)==3
    row[arm]=dict(seconds=float(np.mean([e['query_wall_seconds']for e in sub])),allocated_peak_bytes=max(e['peak_pipeline_allocated_bytes']for e in sub),reserved_peak_bytes=max(e['peak_pipeline_reserved_bytes']for e in sub),NVML_peak_bytes=max(e['NVML']['sampled_process_peak_bytes']for e in sub))
   paired.append(row);allpaired.append(dict(dataset=ds,**row))
  a=np.array([x['Current']['seconds']for x in paired]);b=np.array([x['EdgeTAM']['seconds']for x in paired]);rng=np.random.default_rng(20260920);idx=rng.integers(0,128,(10000,128));delta=(b-a)[idx].mean(1);ratios=b[idx].mean(1)/a[idx].mean(1)
  stats={}
  for arm,v in [('Current',a),('EdgeTAM',b)]:
   stats[arm]=dict(mean_seconds=float(v.mean()),median_query_mean_seconds=float(np.median(v)),p90_query_mean_seconds=float(np.quantile(v,.9)),max_allocated_peak_bytes=max(x[arm]['allocated_peak_bytes']for x in paired),mean_query_allocated_peak_bytes=float(np.mean([x[arm]['allocated_peak_bytes']for x in paired])),max_reserved_peak_bytes=max(x[arm]['reserved_peak_bytes']for x in paired),max_sampled_NVML_pipeline_peak_bytes=max(x[arm]['NVML_peak_bytes']for x in paired))
  results[ds]=dict(queries=128,parents=128,repeats=3,arms=stats,EdgeTAM_minus_Current_seconds=float((b-a).mean()),difference_95CI=np.quantile(delta,[.025,.975]).tolist(),EdgeTAM_over_Current_ratio_of_means=float(b.mean()/a.mean()),ratio_of_means_95CI=np.quantile(ratios,[.025,.975]).tolist(),Current_faster_queries=int((a<b).sum()),Current_slower_queries=int((a>b).sum()),same_time_queries=int((a==b).sum()),paired_repeat_averages=paired)
 P.mkdir(exist_ok=True,parents=True);report=dict(status='independently_verified_all1536_matched_complete_paths',statistics=results,measured_Source_calls=1536,measured_ReferDINO_queries=768,measured_EdgeTAM_queries=768,warmups_excluded=measurements['loads'],latency_scope=plan['timing'],memory_scope=plan['memory'],CI='128pairedparents/domain;3repeats averaged within query before10k bootstrap, pointwise95%; ratio is ratio of means',hardware=read(D/'hardware.json'),all1536_prediction_Source_RGB_interval_exact=True,new_GT_scoring=0)
 atomic(P/'paired_timing_memory.json',report)
 with(P/'all_measured_calls.tsv').open('w',newline='')as f:
  fields=['dataset','query','repeat','arm','query_wall_seconds','peak_pipeline_allocated_bytes','peak_pipeline_reserved_bytes'];w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows({k:e[k]for k in fields}for e in items)
 atomic(P/'all_query_repeat_averages.json',allpaired)
 rec=dict(status='verified_all1536_Source_output_input_reference_SHA_and_repeat_paired_report',worker=identity(),calls=checks,measurement_sha256=sha(D/'measurements.json'),report_sha256=sha(P/'paired_timing_memory.json'),all_calls_sha256=sha(P/'all_measured_calls.tsv'),code_sha256=sha(__file__),new_model=0,new_Source=0,new_GT=0);atomic(D/'reporting_receipt.json',rec);atomic(P/'reporting_receipt.json',rec);print(json.dumps({ds:{k:v for k,v in r.items()if k!='paired_repeat_averages'}for ds,r in results.items()}),flush=True)
if __name__=='__main__':run()
