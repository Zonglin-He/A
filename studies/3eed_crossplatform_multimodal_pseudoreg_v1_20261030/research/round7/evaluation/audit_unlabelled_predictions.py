"""Independent label-free R7 sealed prediction audit; no scoring-index/GT API.
Never opens metapath/imagepath/pointpath from the inference manifest. No model/GPU.
"""
from pathlib import Path
import argparse,collections,json,math,time,traceback
import numpy as np
import score_crossdomain as S
ATOL=2e-6;RTOL=2e-6
KINDS=('direct_G','param1','param3','outputspace1','outputspace3')
NAMES=('source',)+tuple(m+'__'+k for m in S.MODES for k in KINDS)
PARAMETERS=tuple('prediction_heads.5.'+head+'.net.8.'+suffix for head in ('center_residual_head','size_pred_head') for suffix in ('weight','bias'))

def close(a,b):return bool(np.allclose(a,b,atol=ATOL,rtol=RTOL))
def audit_sample(a,trace,plan):
 assert tuple(a['names'])==NAMES;source=a['source_boxes_raw6'][a['q']];assert a['q']==trace['query_index'];assert close(a['source_boxes_eval6'],S.evaluation_box(a['source_boxes_raw6']))
 if 'point_sample_sha256' in a:assert a['point_sample_sha256'].tobytes().hex()==trace['point_sample_sha256']
 if 'image_file_sha256' in a:assert a['image_file_sha256'].tobytes().hex()==trace['image_file_sha256']
 center=trace['feature_norm_squared_plus_bias']['center'];size=trace['feature_norm_squared_plus_bias']['size'];lr=trace['lr'];assert math.isfinite(center) and math.isfinite(size) and min(center,size)>=1 and math.isclose(lr,plan['alpha']*plan['beta_metres']/max(center,size),rel_tol=1e-12)
 stats={};maxeq=0.;max_interval_excess=0.
 for m in S.MODES:
  mi=a['mode_names_list'].index(m);target=a['pseudo_target6'][mi];w=a['pseudo_reliability6'][mi];assert np.isfinite(target).all() and not w[3:].any() and np.array_equal(target[3:],source[3:]);active=bool(w.any());delta_nonzero=bool(np.any((target-source)[w>0]!=0));mt=trace['modes'][m];steps=mt['steps'];mode={'accepted_any_axis':active,'nonzero_pseudo_delta':delta_nonzero,'identity_w0':not active,'reason':mt['reason'],'executed_parameter_steps_total_one_plus_three':0,'nonzero_gradient_steps_total':0,'nonzero_parameter_update_runs':0,'max_effective_selected_center_quadratic_factor':0.}
  direct=a['arm_boxes_raw6'][a['names'].index(m+'__direct_G')];assert close(direct,np.where(w>0,target,source))
  for k in (1,3):
   parameter=a['arm_boxes_raw6'][a['names'].index(m+'__param'+str(k))];output=a['arm_boxes_raw6'][a['names'].index(m+'__outputspace'+str(k))];err=float(np.max(np.abs(parameter-output)));maxeq=max(maxeq,err);assert close(parameter,output),'Parameter/output-space numeric bridge failed'
   # Bound is a contract of normalized-alpha input replay, not an unconditional G-cap property.
   lo=np.minimum(source,target);hi=np.maximum(source,target);eps=ATOL+RTOL*np.maximum(np.abs(lo),np.abs(hi));excess=np.maximum(lo-parameter,parameter-hi);max_interval_excess=max(max_interval_excess,float(np.max(excess)));assert np.all(excess<=eps),'Selected update outside source-to-pseudo interval'
   assert np.array_equal(parameter[3:],source[3:]) and np.array_equal(output[3:],source[3:]),'Size update despite w_size0'
   record=steps.get(str(k));ts=record['trace'] if record else [];assert len(ts)==(k if active else 0);mode['executed_parameter_steps_total_one_plus_three']+=len(ts)
   if ts:
    assert close(np.asarray(ts[0]['selected_box_before']),source) and close(np.asarray(ts[-1]['selected_box_after']),parameter)
    fn=record['feature_norms'];eff=lr*fn['center_self_kernel_with_bias']/plan['beta_metres']/float(w.sum());mode['max_effective_selected_center_quadratic_factor']=max(mode['max_effective_selected_center_quadratic_factor'],eff);assert eff<=plan['alpha']*(1+1e-5)
    for j,t in enumerate(ts):
     assert t['step']==j+1 and math.isfinite(t['loss']) and math.isfinite(t['parameter_gradient_norm']) and t['loss']>=0 and t['parameter_gradient_norm']>=0
     assert t['clip_enabled'] is False and t['clip_factor']==1.;g=np.asarray(t['weighted_SmoothL1_box_gradient6']);assert g.shape==(6,) and np.isfinite(g).all() and not g[3:].any();assert set(t['parameter_delta_max_by_name'])==set(PARAMETERS);assert all(math.isfinite(x) and x>=0 for x in t['parameter_delta_max_by_name'].values());assert all(v==0 for name,v in t['parameter_delta_max_by_name'].items() if '.size_pred_head.' in name)
     if t['parameter_gradient_norm']>0:assert t['nonzero_gradient_parameter_tensors']>0;mode['nonzero_gradient_steps_total']+=1
     assert math.isfinite(t['output_space_replay_max_abs_error'])
    mode['nonzero_parameter_update_runs']+=int(any(v>0 for v in ts[-1]['parameter_delta_max_by_name'].values()))
    if k==3 and '1' in steps and steps['1']['trace']:assert close(np.asarray(ts[0]['selected_box_after']),np.asarray(steps['1']['trace'][0]['selected_box_after'])),'1step and3step did not reset to common source'
   if not active:assert np.array_equal(parameter,source) and np.array_equal(output,source) and np.array_equal(direct,source)
  stats[m]=mode
 return {'query':a['q'],'native_first_differs_from_fixed_argmax_under_tie':int(a['source_native_top10'][0])!=a['q'],'selected_raw_nonpositive_size':bool((source[3:]<=0).any()),'source_pool_raw_clamped_candidate_count':int(np.any(a['source_boxes_raw6'][:,3:]<1e-6,axis=1).sum()),'modes':stats,'max_parameter_vs_outputspace_error':maxeq,'max_source_pseudo_interval_excess':max_interval_excess}

def main(a):
 start=time.perf_counter();sealpath=a.predictions/'SEALED.json';assert S.sha(sealpath)==a.expected_seal_sha256;seal=S.read(sealpath);plan=S.read(a.predictions/'registered_plan.json');assert seal['state']=='sealed_crossdomain_R7v1_no_target_labels_or_scoring';assert seal['all_16arms_sealed_before_CPUscoring'] and seal['model_restored_source_state'] and seal['all_BN_eval_dropout_off'] and seal['no_teacher_EMA_or_source_retraining'];assert seal['source_checkpoint_sha256']==plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';assert seal['plan_sha256']==S.sha(a.predictions/'registered_plan.json');assert S.sha(a.manifest)==seal['manifest_sha256'];assert plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3] and plan['missing_distortion_policy']=='assume_zero_declared'
 for name,digest in seal['files'].items():assert S.sha(S.bounded_path(a.predictions,name))==digest
 for name,digest in seal['program_sha256'].items():assert S.sha(S.bounded_path(a.program_root,name))==digest
 rows=S.read(a.manifest);traces=S.read(a.predictions/'trace.json');lookup={t['id']:t for t in traces};assert len(rows)==len(traces)==seal['n_samples'];assert {r['id'] for r in rows}==set(lookup);assert {r['id']+'.npz' for r in rows}=={p.name for p in a.predictions.glob('*.npz')};repeat=S.read(a.predictions/'repeat_restore_control.json');assert all(repeat[k] is True for k in ['sample0_source_repeat_bitexact','sample0_real_final_Conv_bridge_bitexact','sample0_post_alladapt_full_model_fingerprint_restored']);assert repeat['source_model_state_fingerprint']==seal['source_model_state_fingerprint']==plan['source_model_state_fingerprint']
 records=[]
 for r in rows:
  z=S.load_prediction(a.predictions/(r['id']+'.npz'));t=lookup[r['id']];assert t['platform']==r['platform'];record=audit_sample(z,t,plan);record.update(id=r['id'],platform=r['platform']);records.append(record)
 summary={}
 for p in ('drone','quad'):
  rr=[r for r in records if r['platform']==p];summary[p]={'n':len(rr),'native_tie_top1_different_from_fixedq':sum(r['native_first_differs_from_fixed_argmax_under_tie'] for r in rr),'nonpositive_selected_source_size':sum(r['selected_raw_nonpositive_size'] for r in rr),'modes':{}}
  for mode in S.MODES:
   m=[r['modes'][mode] for r in rr];summary[p]['modes'][mode]={'accepted_any_axis':sum(x['accepted_any_axis'] for x in m),'nonzero_pseudo_delta':sum(x['nonzero_pseudo_delta'] for x in m),'w0_identity_skip':sum(x['identity_w0'] for x in m),'executed_parameter_steps_1plus3':sum(x['executed_parameter_steps_total_one_plus_three'] for x in m),'nonzero_gradient_steps':sum(x['nonzero_gradient_steps_total'] for x in m),'nonzero_parameter_update_runs':sum(x['nonzero_parameter_update_runs'] for x in m),'reason_histogram':dict(collections.Counter(x['reason'] for x in m))}
 result={'state':'passed_unlabelled_sealed_prediction_contracts','prediction_seal_sha256':a.expected_seal_sha256,'auditor_sha256':S.sha(Path(__file__)),'scorer_loader_sha256':S.sha(Path(S.__file__)),'phase':seal['phase'],'n':len(records),'summary':summary,'max_parameter_outputspace_abs_error':max(r['max_parameter_vs_outputspace_error'] for r in records),'max_source_pseudo_interval_excess':max(r['max_source_pseudo_interval_excess'] for r in records),'numeric_comparison_atol':ATOL,'numeric_comparison_rtol':RTOL,'fixed_alpha':.25,'all_input_code_prediction_hashes_verified':True,'real_source_repeat_Conv_bridge_and_state_restore_receipt_verified':True,'rows_PRIVATE':records,'no_scoring_index_or_target_metadata_GT_opened':True,'no_GPU_model_or_task_IoU_scoring':True,'seconds':time.perf_counter()-start};assert not a.output.exists();S.dump(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows_PRIVATE','summary')}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ('predictions','manifest','program-root','output'):ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--expected-seal-sha256',required=True);args=ap.parse_args()
 try:main(args)
 except BaseException as e:
  S.dump(args.output.with_name(args.output.stem+'_FAILED_'+str(time.time_ns())+'.json'),{'state':'unlabelled_audit_failed_preserved','exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'auditor_sha256':S.sha(Path(__file__))});raise
