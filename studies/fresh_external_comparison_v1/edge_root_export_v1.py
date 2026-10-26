"""Read independently audited fixed-cohort scalars and actual cost receipts only."""
from edge_common_v1 import *
import argparse,csv

def scalar_tree(x):
 if isinstance(x,dict):return {k:scalar_tree(v) for k,v in x.items() if k not in {'error','restored','raw_mask_checkpoints','input_binding','partial_raw_files','load','strict_official_model','files','source','config','checkpoint','path','receipt_path','support_path','id','GT_read'}}
 if isinstance(x,list):return [scalar_tree(v) for v in x if not isinstance(v,str)]
 if isinstance(x,(bool,int,float))or x is None:return x
 return None

def run(ds):
 import numpy as np
 validate();folder=D/'pilot_v1'/ds;out=P/'pilot_v1'/ds
 aud=read(folder/'reporting_receipt.json');assert aud['status']=='verified_fixed128_all_mask_support_parent_teacher_predictions_and_means'
 assert aud['metrics_sha256']==sha(folder/'metrics.json')and aud['score_sha256']==sha(folder/'score_receipt.json')and aud['seal_sha256']==sha(folder/'sealed.json')
 assert read(D/'GPU_exit.json')['code']==0
 rr=read(folder/'metrics.json')['private_rows'];rep=read(out/'report.json');seal=read(folder/'sealed.json');assert len(rr)==len(seal['files'])==rep['queries']==rep['parents']==128
 for a in ARMS:
  for k,v in rep['means'][a].items():
   values=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not values)or abs(v-float(np.mean(values)))<1e-12
 tails={}
 for a,b in PAIRS:
  dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]);assert np.array_equal(dt,np.zeros(128))
  tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()),v_improved=int((dv>0).sum()),v_harmed=int((dv<0).sum()),v_tied=int((dv==0).sum()))
 costs=[]
 for i,e in enumerate(seal['files'].values()):
  w=e.get('worker_original',e['worker']);costs.append(dict(anonymous_index=i,inherited=bool(e.get('inherited_old_receipt_sha256')),worker=w,decode_seconds=e['decode_and_provenance_seconds'],request_seconds=e['request_wall_including_checkpoints_seconds'],raw_IO_seconds=e['raw_checkpoint_IO_seconds'],conditional_query_seconds=e['conditional_query_wall_seconds'],nested_stages=e['nested_stage_seconds'],resident_GPU_bytes=e['resident_GPU_bytes'],peak_GPU_bytes=e['peak_query_GPU_bytes'],image_encoder_calls=e['actual_image_encoder_calls'],postprocess_calls=e['actual_strict_postprocess_calls']))
 atomic(out/'all_actual_query_costs_v1.json',costs)
 sums={k:sum(e[k]for e in costs)for k in ['decode_seconds','request_seconds','raw_IO_seconds','conditional_query_seconds']}
 engineering={}
 for name,p in {
  'original_model_load':OLD/'pilot_v1/model_load.json',
  'original_v3_model_load':OLD/'pilot_v1/model_load_runtime_v3.json',
  'original_v4_model_load':OLD/'pilot_v1/model_load_runtime_v4.json',
  'first_raw_exact_restore_extra_request':OLD/'first_raw_restore_v1/completion.json',
  'dtype_compatibility_extra_positive_control':OLD/'dtype_positive_control_v1/completion.json',
  'failed_BF16_FP32_worker':OLD/'pilot_v1/failure_preserved_runtime_v3.json',
  'zero_tracking_input_miss_worker':OLD/'pilot_v1/failure_preserved_runtime_v4.json',
  'current_new240_model_load':D/'pilot_v1/model_load_v1.json'
 }.items():
  assert p.exists(),p;engineering[name]=dict(receipt_sha256=sha(p),actual_scalar_receipt=scalar_tree(read(p)))
 cost=dict(logical_queries=128,inherited_queries=sum(e['inherited']for e in costs),actual_query_segments_sums_seconds=sums,conditional_query_mean_seconds=sums['conditional_query_seconds']/128,request_mean_seconds=sums['request_seconds']/128,peak_query_GPU_bytes=max(e['peak_GPU_bytes']for e in costs),whole_current_worker=read(D/'pilot_v1/state_v1.json'),model_loads_engineering_and_failure=engineering,current_baseline_preparation=dict(whole256=256,strict_cached_current_expert=250,actual_new_current_expert=6,new_Source=0,receipt_sha256=sha(N/'state_v1.json')),inherited_Source_and_current_expert_cost_nonzero=True,nested_times_must_not_be_added=True,complete_uncached_pipeline_latency_measured=False,scope='Logical256 cohort plus TWO successful engineering replays/controls and ONE failed partial attempt:259 attempts,258 completed tracking requests globally. Whole workers/model loads overlap nested query stages; inherited16 cost included once. Failed partial per-request clock missing, not zero. Current256 preparation and6 new current-expert calls are separate shared inputs, not free.')
 atomic(out/'actual_cost_lineage_v1.json',cost);atomic(out/'all_paired_severe_rescues_v1.json',tails)
 text=f'# {ds}: fixed128 EdgeTAM spatial comparison\n\nAll128queries/128parents sealed before independent official CPU labels, followed by independent full packed-mask boxes/reference/input/SHA/means audit. The four rows share each query’s actual prepared-current Ours time interval. This is a conditional spatial comparison, not independent full-tube temporal inference.\n\n|Spatial trajectory|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|\n'
 for a,m in rep['means'].items():text+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|\n"
 text+='\nParent-paired10000-bootstrap pointwise95% CIs, in percentage points:\n\n'
 for key,z in rep['paired'].items():
  q=z['v'];t=tails[key];text+=f"- {key}: v{q['mean']:+.6f}, CI{q['CI95']}; {t['v_improved']} improve/{t['v_harmed']} harm/{t['v_tied']} tie; severe t/v/union {t['severe_t']}/{t['severe_v']}/{t['severe_union_once']}, rescue union{t['rescue_union_once']}.\n"
 text+='\nShort/long strata (GT event duration<=5s/>5s), same parent CI:\n\n'
 for group,g in rep['timegroups'].items():
  for key,z in g['paired'].items():text+=f"- {group}, n{g['queries']}, {key}: v{z['v']['mean']:+.6f}, CI{z['v']['CI95']}.\n"
 text+='\nEdgeTAM exceeds the current method and interpolation in both fixed cohorts on average; all harmed cases remain, and no direction/checkpoint/configuration is selected from these scores. Same eight expert BOX prompts are used, while current correction also consumes the original expert masks/features; same observations are not equal compute or identical information interfaces. EdgeTAM uses an extra13.9M-parameter propagation model and the full native frame grid, all eight conditioning prompts before forward propagation from frame0, no reverse fusion and no current-method DP. Observed output boxes are exactly the original prompts; empty propagated masks are zero area without fallback.\n\n'
 text+=f"Conditional query mean {cost['conditional_query_mean_seconds']:.6f}s, request mean {cost['request_mean_seconds']:.6f}s, peak query GPU {cost['peak_query_GPU_bytes']}bytes. This includes cached Source/current-expert inputs and raw checkpoint/provenance work; it is not complete uncached deployment latency. All query costs, actual worker/model loads, two extra successful controls and failed partial attempt remain in cost lineage. Nested stages are not additive; missing failed-request clock is not zero. The250 inherited current-expert inputs retain acquisition cost, and6 actual new current-expert calls were shared preparation.\n\nOnly128parents per domain, not all12205queries. Resource feasibility and prediction correctness support a later separately frozen full extension; they do not authorize selecting a method from these exposed scores. CG-STVG/gamma/corruption remain user-paused, additional CG-HC cancelled; no original frozen method or paper changed.\n"
 (out/'ROOT_FINDINGS_v1.md').write_text(text)
 atomic(folder/'root_export_receipt_v1.json',dict(status='root_fixed128_all_signed_parent_CI_tails_shortlong_actual_cost_analysis_complete',code_sha256=sha(Path(__file__)),audit_sha256=sha(folder/'reporting_receipt.json'),report_sha256=sha(out/'report.json'),findings_sha256=sha(out/'ROOT_FINDINGS_v1.md'),cost_sha256=sha(out/'actual_cost_lineage_v1.json'),new_GT_scoring=0,new_model=0,new_tracking=0,full12205_complete=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);run(p.parse_args().dataset)
