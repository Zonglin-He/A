"""Summarize existing sealed scalar/engineering artifacts; no new prediction or scoring."""
import json,pathlib,hashlib,numpy as np
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={}
for tag in ['dev','confirm']:
 name=f'{tag}_native_fps_support_v1';run=R/'runs'/name;cache=R/'caches'/name
 s=json.loads((run/'SCORED.json').read_text());status=json.loads((cache/'CACHE_STATUS.json').read_text());seal=json.loads((run/'SEALED.json').read_text())
 assert all(sha(run/n)==h for n,h in seal.items())
 select=lambda obj,step,arm:next(x for x in s['averages'] if x['objective']==obj and x['step']==step and x['arm']==arm)
 metrics={k:select(o,st,a) for k,o,st,a in [('source','hm_support',0,'source'),('mean','hm_support',0,'ensemble'),('median','hm_support',0,'median'),('q75','hm_support',0,'q75'),('full','hm_support',3,'adapted'),('hm_only','hm_only',3,'adapted')]}
 calls=[];operators=[];bridges=[]
 for p in cache.glob('*.json'):
  if p.name.startswith('CACHE'):continue
  d=json.loads(p.read_text());calls+=d['view_calls'];operators.append(d['operator_diagnostics']);bridges.append(d['normalization_bridge'])
 costs={ 'source_primary_calls':len(calls),'primary_forward_sum_seconds':sum(d['seconds'] for d in calls),'mean_forward_seconds':np.mean([d['seconds'] for d in calls]),'max_forward_peak_cuda_allocated_bytes':max(d['peak_cuda_allocated_bytes'] for d in calls),'legacy_bridge_calls':status['extra_legacy_bridge_source_calls'],'repeat_native_calls':status['extra_native_repeat_source_calls'],'scope':'Synchronized model forward only. Excludes loading, images/text preparation, offline MLLM CoT and cached decoder diagnostics; not end-to-end deployment latency.'}
 for obj in ['hm_only','hm_support']:
  di=[json.loads(p.read_text()) for p in run.glob(f'*_{obj}_k3.json')]; costs[obj]={'mean_cached_three_step_seconds':float(np.mean([d['seconds'] for d in di])),'sum_cached_three_step_seconds':sum(d['seconds'] for d in di),'max_cached_peak_cuda_allocated_bytes':max(d['peak_cuda_allocated_bytes'] for d in di),'all_nonzero_gradient':all(all(h['gradient_l2']>0 for h in d['history']) for d in di),'all_parameter_delta_nonzero':all(d['parameter_l2']>0 for d in di),'max_abs_logit_delta':max(d['maximum_absolute_logit_residual'] for d in di),'all_native_gpu_views_bitexact':all(d['all_five_native_cached_view_probabilities_bitexact'] for d in di)}
 summary[tag]={'metrics':metrics,'costs':costs,'sealed_prediction_files':len(seal),'scored_sha256':sha(run/'SCORED.json'),'seal_sha256':sha(run/'SEALED.json'),'all_metric_valid_n':{m:metrics['source']['valid_counts'][m] for m in ['auc','aiou','sim','mae']}}
 if tag=='dev':summary[tag]['preprocessing_and_operator_bridges']={'native64_minus_legacy32_aggregate_delta':{m:metrics['source'][m]-select('hm_support',0,'legacy32_source')[m] for m in ['auc','aiou','sim','mae']},'max_native64_legacy32_point_probability_diff':max(d['source_output_max_abs_diff'] for d in bridges),'all_same_seed_fresh_source_repeats_bitexact':all(d['same_seed_fresh_source_repeat_bitexact'] for d in operators),'index_only_mean_abs_diff_mean':float(np.mean([d['index_only_mean_abs_diff'] for d in operators])),'paired_projection_restore_max_abs_diff':max(d['paired_projection_restore_max_abs_diff'] for d in operators),'interpretation':'Preprocessing bridge is source correction, not TTA. Fixed encoder diagnostic isolates decoder projection/index sensitivity, not deployment permutation invariance.'}
summary['claim']={'local_support_mechanism_adopted':False,'reason':'Full improves threshold-averaged IoU vs prelocked q75 but worsens AUC/SIM/MAE and loses aIoU to HM-only; these results do not establish improved semantic localization.','confirm_labels_exposed':True,'posthoc_diagnostics_new_validation':False,'future_experiments_run':False,'independent_parent_metric_audit':'await_parent_receipt'}
summary['preservation']={'original_execution_lock_sha256':sha(R/'EXECUTION_LOCK.json'),'original_mechanism_lock_sha256':sha(R/'MECHANISM_LOCK.json'),'original_cohort_lock_sha256':sha(R/'cohorts/COHORT_LOCK.json'),'old_public_export_manifest_sha256':sha(R.parent/'public_export/manifest.json')}
assert summary['preservation']['old_public_export_manifest_sha256']=='c4945fe694808d5588a4ccbe7326e99c5ceb503bc737ca07b3a1d5273604bf4e'
(R/'RESULT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
(R/'STATUS.json').write_text(json.dumps({'run':'scored','measurement':'valid_for_locked_dev16_confirm32; independent_parent_audit_pending','claim':'local_support_not_adopted','cohort':'dev16/confirm32_locked_and_scored','GPU_owner_stage':'released_after_confirmation','primary_real_source_forward_calls':240,'additional_dev_bridge_and_repeat_calls':32,'engineering_real_source_calls':7,'diagnostic_decoder_replays_counted_separately':True,'confirm_labels_exposed':True,'new_model_calls_after_confirm_scoring':0,'locked_source_count':11,'execution_lock_sha256':sha(R/'EXECUTION_LOCK.json')},indent=2))
print(json.dumps({k:v['costs'] for k,v in summary.items() if k in ['dev','confirm']},indent=2))
