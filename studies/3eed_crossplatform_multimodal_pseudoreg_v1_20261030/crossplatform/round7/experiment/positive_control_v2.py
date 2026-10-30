"""Real frozen-head engineering positive control, not G/TTA/task evidence.
Parent GPU authorization required; run only after full prediction seal, under the
shared physical GPU flock. Uses no RGB, metadata camera, target annotations or GT.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
from runner_helpers import sha,read,validate_manifest,sample_points,array_sha
from run_crossdomain import load_model,token_mask_for,fingerprint,R7

PURPOSE='R7_real_head_engineering_positive_control_v2_crossprocess_drift_recorded'
DELTA6=[.05,0.,0.,0.,0.,0.]
WEIGHT6=[1.,0.,0.,0.,0.,0.]
ATOL_METRES=5e-5

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['GPU_execution_authorized'] is True and plan['purpose']==PURPOSE
    assert plan['cross_process_source_pool_policy']=='record_no_numeric_gate_same_q_data'
    assert plan['in_process_source_repeat_policy']=='bitexact'
    assert sha(Path(__file__).with_name('POSITIVE_CONTROL_V1_FAILURE.json'))==plan['old_failed_attempt_receipt_sha256']
    assert plan['target_delta6_metres']==DELTA6 and plan['reliability6']==WEIGHT6
    assert plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3]
    assert plan['equivalence_atol_metres']==ATOL_METRES and plan['selected_smoke_manifest_index']==0
    assert plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    for rel,digest in plan['program_sha256'].items():
        p=R7/rel;assert p.resolve().is_relative_to(R7.resolve()) and sha(p)==digest
    manifest=Path(plan['smoke_manifest_path']);assert sha(manifest)==plan['smoke_manifest_sha256']
    smoke_plan=Path(plan['smoke_plan_path']);assert sha(smoke_plan)==plan['smoke_plan_sha256']
    old=read(smoke_plan);assert old['manifest_sha256']==sha(manifest)
    for k in ('r6_factory_sha256','upstream_code_sha256','roberta_sha256','source_checkpoint_sha256','source_training_plan_sha256','source_model_state_fingerprint','seed','cudnn_benchmark','cudnn_deterministic','sentinel','noun_fallback'):
        assert plan[k]==old[k],('Positive control must use same locked source/input policy',k)
    smoke_seal=Path(plan['smoke_prediction_seal_path']);assert sha(smoke_seal)==plan['smoke_prediction_seal_sha256']
    assert read(smoke_seal)['manifest_sha256']==sha(manifest)
    # Scheduling guard: no positive-control GPU start before the all-target-data
    # prediction seal exists. Root still owns flock and scheduling authorization.
    full_seal=Path(plan['completed_full_prediction_seal_path']);assert full_seal.is_file() and sha(full_seal)==plan['completed_full_prediction_seal_sha256']
    full=read(full_seal);assert full['phase']=='full' and full['n_samples']==5912 and full['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
    row=validate_manifest(read(manifest))[0]
    assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    import torch
    sys.path.insert(0,str(R7))
    from regression_adapter import capture_source,adapt,PseudoTarget,StepConfig,parameter_sha,PARAMETER_NAMES
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=plan['cudnn_benchmark'];torch.backends.cudnn.deterministic=plan['cudnn_deterministic']
    started=time.perf_counter();model,entries=load_model(plan);initial_fp=fingerprint(model);head_fp=parameter_sha(model)
    buffers={n:b.clone() for n,b in model.named_buffers()}
    pc,ix,nraw=sample_points(row);text=row['caption']+plan['sentinel'];mask,text_info,unsupported=token_mask_for(model,text,plan['noun_fallback'])
    assert not unsupported and not text_info['unsupported_noun'],'This control requires actual smoke first-noun identity'
    points=torch.from_numpy(pc)[None].cuda();torch.cuda.reset_peak_memory_stats()
    cache=capture_source(model,{'point_clouds':points,'text':[text]},mask)
    q=cache.query_index;source=cache.boxes_raw6.detach().clone();assert (source[0,q,3:]>0).all()
    first_cache=smoke_seal.parent/(row['id']+'.npz')
    assert sha(first_cache)==read(smoke_seal)['files'][first_cache.name]
    tracefile=smoke_seal.parent/'trace.json';assert sha(tracefile)==read(smoke_seal)['files']['trace.json']
    old_row=next(r for r in read(tracefile) if r['id']==row['id'])
    def differences(reference,current):
        reference=np.asarray(reference,np.float64);current=np.asarray(current,np.float64)
        diff=np.abs(current-reference);relative=diff/np.maximum(np.abs(reference),1e-12)
        return {'shape':list(reference.shape),'bitexact':bool(np.array_equal(reference,current)),'max_absolute_difference':float(diff.max()),'mean_absolute_difference':float(diff.mean()),'p99_absolute_difference':float(np.quantile(diff,.99)),'max_relative_difference_reference_floor_1e_minus12':float(relative.max()),'different_value_count':int(np.count_nonzero(diff))}
    with np.load(first_cache,allow_pickle=False) as saved:
        old_q=int(saved['fixed_query_index']);old_top=saved['source_native_top10'].astype(np.int64)
        current_top=torch.argsort(cache.text_scores[0],descending=True)[:10].cpu().numpy()
        point_equal=saved['point_sample_sha256'].tobytes().hex()==array_sha(pc)==old_row['point_sample_sha256']
        indices_equal=array_sha(ix)==old_row['sample_indices_sha256']
        drift={'state':'cross_process_source_drift_measured_no_numeric_tolerance_gate','source_pool':differences(saved['source_boxes_raw6'],source[0].cpu().numpy()),'source_text_scores':differences(saved['source_text_scores256'],cache.text_scores[0].cpu().numpy()),'same_fixed_source_q':old_q==q,'smoke_fixed_q':old_q,'current_fixed_q':q,'same_native_top10_order':bool(np.array_equal(old_top,current_top)),'same_native_top10_set':bool(set(old_top)==set(current_top)),'native_top10_changed_positions':int(np.count_nonzero(old_top!=current_top)),'native_top1_changed':bool(old_top[0]!=current_top[0]),'point_sample_SHA_strict_equal':point_equal,'raw_indices_SHA_strict_equal':indices_equal,'raw_point_file_SHA_strict_equal':sha(row['pointpath'])==old_row['point_file_sha256'],'runtime_flags':{'torch_version':torch.__version__,'cuda_runtime_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()},'smoke_registered_flags':{k:old[k] for k in ('cudnn_benchmark','cudnn_deterministic')},'smoke_actual_TF32_flags_not_saved_in_original_trace':True,'full_task_caches_unmodified':True,'comparison_scope':'Cross-process numerical description only; no output tolerance was chosen/relaxed, no task delta is computed. Positive supervision below uses the fresh current source cache.'}
    (a.output/'CROSS_PROCESS_DRIFT.json').write_text(json.dumps(drift,indent=2,allow_nan=False)+'\n')
    assert point_equal and indices_equal and drift['raw_point_file_SHA_strict_equal'],'Physical point input must be identical'
    assert old_q==q,'Fixed puretext source identity changed; fail before artificial backward'

    delta=torch.tensor(DELTA6,device=source.device,dtype=source.dtype)
    target=source[0,q]+delta;w=torch.tensor(WEIGHT6,device=source.device,dtype=source.dtype)
    center_kernel=float(cache.center_features[0,:,q].square().sum())+1.
    size_kernel=float(cache.size_features[0,:,q].square().sum())+1.
    lr=plan['alpha']*plan['beta_metres']/max(center_kernel,size_kernel)
    factor=lr*center_kernel/plan['beta_metres'];assert 0<factor<=.25+1e-12
    results={}
    for steps in plan['steps']:
        assert parameter_sha(model)==head_fp
        result=adapt(model,cache,PseudoTarget(q,target,w),StepConfig(steps,lr,plan['beta_metres']))
        assert len(result['trace'])==steps
        assert all(t['parameter_gradient_norm']>0 and t['nonzero_gradient_parameter_tensors']>=2 for t in result['trace'])
        center_changes={k:max(t['parameter_delta_max_by_name'][k] for t in result['trace']) for k in PARAMETER_NAMES if '.center_residual_head.' in k}
        size_changes={k:max(t['parameter_delta_max_by_name'][k] for t in result['trace']) for k in PARAMETER_NAMES if '.size_pred_head.' in k}
        assert all(x>0 for x in center_changes.values()) and all(x==0 for x in size_changes.values())
        update=result['parameter_SGD'];control=result['exact_output_space_SGD']
        error=float((update-control).abs().max());assert error<=ATOL_METRES
        observed_delta=(update[0,q]-source[0,q]).cpu().numpy()
        assert observed_delta[0]>0 and observed_delta[0]<.05
        assert np.array_equal(observed_delta[1:],np.zeros(5,np.float32))
        ideal_delta=.05*(1-(1-factor)**steps)
        assert abs(float(observed_delta[0])-ideal_delta)<=ATOL_METRES
        assert parameter_sha(model)==head_fp and fingerprint(model)==initial_fp
        assert all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        # Keep local scalar/gradient diagnostics but do not write actual source
        # coordinates or candidate arrays as a positive-control task result.
        scalar_trace=[{k:v for k,v in t.items() if k not in ('selected_box_before','selected_box_after')} for t in result['trace']]
        results[str(steps)]={'steps':steps,'true_nonzero_backward_and_parameter_change':True,'observed_selected_delta6_metres':observed_delta.tolist(),'ideal_real_arithmetic_selected_x_delta_metres':ideal_delta,'parameter_vs_output_space_allquery_max_abs_error_metres':error,'center_parameter_delta_max_by_name':center_changes,'size_parameter_delta_max_by_name':size_changes,'trace_without_actual_box_coordinates':scalar_trace,'feature_norms':result['feature_norms'],'spillover':result['spillover'],'post_episode_full_model_buffers_grad_and_mode_restored':True}
    repeat=capture_source(model,{'point_clouds':points,'text':[text]},mask)
    assert torch.equal(source,repeat.boxes_raw6) and torch.equal(cache.text_scores,repeat.text_scores)
    assert fingerprint(model)==initial_fp and all(not m.training for m in model.modules())
    torch.cuda.synchronize()
    receipt={'state':'engineering_positive_control_pass_no_G_feedback_no_GT_no_taskscore','purpose':PURPOSE,'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'strict_state_entries':entries,'plan_sha256':sha(a.plan),'smoke_manifest_sha256':sha(manifest),'smoke_prediction_seal_sha256':sha(smoke_seal),'completed_full_prediction_seal_sha256':sha(full_seal),'smoke_sample_index':0,'sample_id':row['id'],'fixed_source_query_index':q,'sourcepoint_sample_sha256':array_sha(pc),'sourcepoint_indices_sha256':array_sha(ix),'rawpoint_count':nraw,'same_source_weights_and_smoke_first_sample_q_points_indices_verified':True,'cross_process_pool_and_score_drift':drift,'artificial_target_uses_fresh_current_source_cache_only':True,'puretext_source_repeat_bitexact':True,'real_Conv_source_bridge_bitexact':True,'source_model_fingerprint_restored':initial_fp,'artificial_target_delta6_metres':DELTA6,'artificial_reliability6':WEIGHT6,'alpha':.25,'beta_metres':.1,'lr':lr,'selected_x_effective_shrink_factor':factor,'no_clip_no_weight_decay_no_momentum':True,'predeclared_equivalence_atol_metres':ATOL_METRES,'results':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started,'no_RGB_camera_G_model_or_targetannotation_GT_or_taskmetric_used':True,'interpretation':'Only proves current-episode real source final-layer backward/update and restore under artificial supervision. Cross-process source numerical drift is separately retained; no full-task cache is replaced or compared as a method delta. Provides no evidence that G is informative or TTA improves target grounding.'}
    dest=a.output/'ENGINEERING_POSITIVE_CONTROL.json';dest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    (a.output/'SEALED.json').write_text(json.dumps({'purpose':PURPOSE,'state':receipt['state'],'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'files':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}},indent=2)+'\n')
    print('REAL_HEAD_POSITIVE_CONTROL_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
