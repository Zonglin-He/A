"""Two-mode real-head precision diagnostic; engineering only, no G/GT/taskscore.
Every raw observation is persisted before tolerance decisions. Old failure lineage
is retained. Original-precision failure cannot be hidden by strict-FP32 success.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time,traceback
import numpy as np
from runner_helpers import sha,read,validate_manifest,sample_points,array_sha
from run_crossdomain import load_model,token_mask_for,fingerprint,R7
PURPOSE='R7_real_head_positive_control_v3_precision_diagnostic'
DELTA6=[.05,0.,0.,0.,0.,0.];WEIGHT6=[1.,0.,0.,0.,0.,0.];ATOL_METRES=5e-5
PRECISION_MODES={'original_plan':{'cudnn_allow_tf32':True,'matmul_allow_tf32':False,'cudnn_benchmark':True,'cudnn_deterministic':True},'strict_FP32':{'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}}


def differences(reference,current):
    a=np.asarray(reference,np.float64);b=np.asarray(current,np.float64);d=np.abs(a-b)
    return {'bitexact':bool(np.array_equal(a,b)),'max_absolute_difference':float(d.max()),'mean_absolute_difference':float(d.mean()),'p99_absolute_difference':float(np.quantile(d,.99)),'max_relative_difference_reference_floor1e_minus12':float((d/np.maximum(np.abs(a),1e-12)).max()),'different_value_count':int(np.count_nonzero(d))}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['purpose']==PURPOSE and plan['GPU_execution_authorized'] is True
    assert plan['precision_modes']==PRECISION_MODES and plan['mode_order']==['original_plan','strict_FP32']
    assert plan['target_delta6_metres']==DELTA6 and plan['reliability6']==WEIGHT6 and plan['steps']==[1,3]
    assert plan['alpha']==.25 and plan['beta_metres']==.1 and plan['equivalence_atol_metres']==ATOL_METRES
    assert plan['selected_smoke_manifest_index']==0 and plan['mode_specific_source_query_allowed'] is True
    for name,expected in plan['old_failure_receipt_sha256'].items():assert sha(R7/'experiment'/name)==expected
    assert set(plan['old_failure_receipt_sha256'])=={'POSITIVE_CONTROL_V1_FAILURE.json','POSITIVE_CONTROL_V2_FAILURE.json'}
    for rel,digest in plan['program_sha256'].items():
        p=R7/rel;assert p.resolve().is_relative_to(R7.resolve()) and sha(p)==digest
    assert plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    manifest=Path(plan['smoke_manifest_path']);assert sha(manifest)==plan['smoke_manifest_sha256']
    smoke_plan=Path(plan['smoke_plan_path']);assert sha(smoke_plan)==plan['smoke_plan_sha256'];old=read(smoke_plan)
    for k in ('r6_factory_sha256','upstream_code_sha256','roberta_sha256','source_checkpoint_sha256','source_training_plan_sha256','source_model_state_fingerprint','seed','sentinel','noun_fallback'):
        assert plan[k]==old[k]
    smoke_seal=Path(plan['smoke_prediction_seal_path']);assert sha(smoke_seal)==plan['smoke_prediction_seal_sha256'];sseal=read(smoke_seal)
    full_seal=Path(plan['completed_full_prediction_seal_path']);assert sha(full_seal)==plan['completed_full_prediction_seal_sha256'];full=read(full_seal)
    assert full['phase']=='full' and full['n_samples']==5912 and full['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
    assert sseal['manifest_sha256']==sha(manifest);row=validate_manifest(read(manifest))[0]
    old_npz=smoke_seal.parent/(row['id']+'.npz');assert sha(old_npz)==sseal['files'][old_npz.name]
    with np.load(old_npz,allow_pickle=False) as z:
        old_boxes=z['source_boxes_raw6'].copy();old_scores=z['source_text_scores256'].copy();old_q=int(z['fixed_query_index']);old_top=z['source_native_top10'].copy();old_point_sha=z['point_sample_sha256'].tobytes().hex()
    old_trace=smoke_seal.parent/'trace.json';assert sha(old_trace)==sseal['files']['trace.json'];old_row=next(x for x in read(old_trace) if x['id']==row['id'])
    pc,ix,nraw=sample_points(row)
    assert array_sha(pc)==old_point_sha==old_row['point_sample_sha256']
    assert array_sha(ix)==old_row['sample_indices_sha256'] and sha(row['pointpath'])==old_row['point_file_sha256']
    assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    import torch
    sys.path.insert(0,str(R7));from regression_adapter import capture_source,adapt,PseudoTarget,StepConfig,parameter_sha,PARAMETER_NAMES
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    started=time.perf_counter();model,entries=load_model(plan);initial_fp=fingerprint(model)
    # Once-loaded source state kept on CPU. Every mode restores identical values;
    # no second checkpoint read/model constructor and no target-label selection.
    initial_state={n:x.detach().cpu().clone() for n,x in model.state_dict().items()}
    points=torch.from_numpy(pc)[None].cuda();text=row['caption']+plan['sentinel'];mask,text_info,unsupported=token_mask_for(model,text,plan['noun_fallback']);assert not unsupported and not text_info['unsupported_noun']
    def flags():
        return {'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_enabled':torch.backends.cudnn.enabled,'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()}
    saved_flags=flags();results={}
    def restore_source():
        model.load_state_dict(initial_state,strict=True);model.requires_grad_(False).eval()
        for p in model.parameters():p.grad=None
        assert fingerprint(model)==initial_fp
    def write(name,value):
        (a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    try:
        for mode in plan['mode_order']:
            restore_source();setting=PRECISION_MODES[mode]
            torch.backends.cudnn.allow_tf32=setting['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=setting['matmul_allow_tf32'];torch.backends.cudnn.benchmark=setting['cudnn_benchmark'];torch.backends.cudnn.deterministic=setting['cudnn_deterministic']
            actual=flags();assert all(actual[k]==v for k,v in setting.items())
            observed={'precision_mode':mode,'flags':actual,'stage':'fresh_source_capture','results':{},'mode_pass':False,'current_mode_Conv_bridge_bitexact':False,'current_mode_source_repeat_bitexact':False,'source_model_fingerprint_before':fingerprint(model)}
            write(mode+'_RAW_OBSERVATIONS.json',observed)
            try:
                torch.cuda.reset_peak_memory_stats();cache=capture_source(model,{'point_clouds':points,'text':[text]},mask);observed['current_mode_Conv_bridge_bitexact']=True
                source=cache.boxes_raw6.detach().clone();q=cache.query_index;assert (source[0,q,3:]>0).all()
                top=torch.argsort(cache.text_scores[0],descending=True)[:10].cpu().numpy()
                observed.update(stage='source_drift_measured',current_mode_query=q,smoke_query=old_q,query_equal_smoke=old_q==q,source_pool_crossprocess=differences(old_boxes,source[0].cpu().numpy()),source_scores_crossprocess=differences(old_scores,cache.text_scores[0].cpu().numpy()),nativeTop10_equal_smoke=bool(np.array_equal(old_top,top)),nativeTop10_changed_positions=int(np.count_nonzero(old_top!=top)),same_source_weights_and_P_indices_strict=True)
                write(mode+'_RAW_OBSERVATIONS.json',observed)
                target=source[0,q]+torch.tensor(DELTA6,device=source.device);w=torch.tensor(WEIGHT6,device=source.device)
                ck=float(cache.center_features[0,:,q].square().sum())+1.;sk=float(cache.size_features[0,:,q].square().sum())+1.;lr=.25*.1/max(ck,sk);factor=lr*ck/.1
                observed.update(actual_lr=lr,center_kernel=ck,size_kernel=sk,effective_x_shrink_factor=factor,artificial_target_from_current_mode_source=True)
                for steps in (1,3):
                    before_versions={n:p._version for n,p in model.named_parameters() if n not in PARAMETER_NAMES}
                    out=adapt(model,cache,PseudoTarget(q,target,w),StepConfig(steps,lr,.1))
                    after=dict(model.named_parameters());changed=[n for n,v in before_versions.items() if after[n]._version!=v]
                    nonselected_gradients=sum(p.grad is not None for n,p in after.items() if n not in PARAMETER_NAMES)
                    raw_steps=[]
                    for t in out['trace']:
                        item={k:v for k,v in t.items() if k not in ('selected_box_before','selected_box_after')}
                        item['observed_selected_delta6_metres']=(np.asarray(t['selected_box_after'])-source[0,q].cpu().numpy()).tolist()
                        item['nonselected_parameter_version_changes_across_entire_call']=len(changed)
                        item['nonselected_parameter_max_abs_change_if_versions_unchanged']=0. if not changed else None
                        item['nonselected_version_evidence']='Monotonic parameter version counters unchanged across entire call imply no in-place parameter update in any constituent step.'
                        raw_steps.append(item)
                    delta=(out['parameter_SGD'][0,q]-source[0,q]).cpu().numpy();err=float((out['parameter_SGD']-out['exact_output_space_SGD']).abs().max());restored=fingerprint(model)==initial_fp
                    data={'steps':steps,'raw_step_observations':raw_steps,'observed_selected_delta6_metres':delta.tolist(),'final_allquery_equivalence_max_abs_error_metres':err,'all_step_max_equivalence_error_metres':max(t['output_space_replay_max_abs_error'] for t in out['trace']),'nonselected_parameter_version_changed_count':len(changed),'nonselected_gradients_after_restore_count':nonselected_gradients,'post_episode_full_model_fingerprint_restored':restored,'feature_norms':out['feature_norms'],'spillover':out['spillover'],'expected_real_arithmetic_x_delta_metres':.05*(1-(1-factor)**steps),'judgement':'not_yet_applied'}
                    observed['results'][str(steps)]=data;observed['stage']='raw_update_observations_serialized_before_judgement';write(mode+'_RAW_OBSERVATIONS.json',observed)
                    # The original fixed 5e-5 threshold is applied only after all
                    # actual error/gradient/parameter observations reach disk.
                    checks={'all_equivalence_errors_le_registered5e_minus5':max(err,data['all_step_max_equivalence_error_metres'])<=ATOL_METRES,'true_nonzero_gradient_and_center_parameter_change':all(t['parameter_gradient_norm']>0 and t['nonzero_gradient_parameter_tensors']>=2 for t in raw_steps) and all(out['trace'][-1]['parameter_delta_max_by_name'][n]>0 for n in PARAMETER_NAMES if '.center_residual_head.' in n),'size_parameters_unchanged':all(out['trace'][-1]['parameter_delta_max_by_name'][n]==0 for n in PARAMETER_NAMES if '.size_pred_head.' in n),'only_selected_x_output_changed':bool(0<delta[0]<.05 and np.array_equal(delta[1:],np.zeros(5,np.float32))),'closed_form_x_delta_within_registered_atol':abs(float(delta[0])-data['expected_real_arithmetic_x_delta_metres'])<=ATOL_METRES,'nonselected_unchanged':not changed and nonselected_gradients==0,'full_model_restored':restored}
                    data.update(judgement='pass' if all(checks.values()) else 'fail_preserved',checks=checks);write(mode+'_RAW_OBSERVATIONS.json',observed)
                repeat=capture_source(model,{'point_clouds':points,'text':[text]},mask)
                observed['current_mode_source_repeat_bitexact']=bool(torch.equal(source,repeat.boxes_raw6) and torch.equal(cache.text_scores,repeat.text_scores))
                observed['mode_pass']=observed['current_mode_Conv_bridge_bitexact'] and observed['current_mode_source_repeat_bitexact'] and all(r['judgement']=='pass' for r in observed['results'].values()) and fingerprint(model)==initial_fp
                observed['stage']='mode_completed_with_explicit_pass_or_fail';observed['peak_allocated_bytes']=torch.cuda.max_memory_allocated();write(mode+'_RAW_OBSERVATIONS.json',observed)
            except Exception as e:
                observed.update(stage='mode_exception_preserved_continue_next_mode',mode_pass=False,exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc());write(mode+'_RAW_OBSERVATIONS.json',observed)
            finally:
                restore_source()
            results[mode]=observed
    finally:
        restore_source();torch.set_float32_matmul_precision(saved_flags['float32_matmul_precision']);torch.backends.cuda.matmul.allow_tf32=saved_flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=saved_flags['cudnn_allow_tf32'];torch.backends.cudnn.benchmark=saved_flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=saved_flags['cudnn_deterministic']
    original=results['original_plan']['mode_pass'];strict=results['strict_FP32']['mode_pass']
    state=('strict_FP32_only_engineering_pass_original_mode_failed_preserved' if strict and not original else 'both_precision_modes_engineering_pass' if strict and original else 'precision_diagnostic_not_passed_strict_FP32')
    receipt={'state':state,'purpose':PURPOSE,'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'source_model_state_fingerprint':initial_fp,'strict_state_entries':entries,'plan_sha256':sha(a.plan),'old_failure_receipt_sha256':plan['old_failure_receipt_sha256'],'sourcepoint_sample_sha256':array_sha(pc),'sourcepoint_indices_sha256':array_sha(ix),'rawpoint_count':nraw,'artificial_target_delta6_metres':DELTA6,'artificial_reliability6':WEIGHT6,'alpha':.25,'beta_metres':.1,'predeclared_equivalence_atol_metres':ATOL_METRES,'original_mode_passed':original,'strict_FP32_mode_passed':strict,'mode_results':results,'one_model_load_and_same_source_snapshot_each_mode':True,'mode_source_queries_may_differ_reported_not_GT_selected':True,'no_RGB_G_feedback_target_GT_or_taskscore':True,'main_full_task_caches_metrics_and_precision_not_changed_or_retested':True,'original_mode_failure_not_overridden_by_strict_mode':True,'elapsed_seconds':time.perf_counter()-started,'interpretation':'Strict-FP32 success, if observed, supports the head update/equivalence interface only under strict FP32. It does not validate original-main precision, explain cross-process pool drift by itself, or show informative G/TTA gains.'}
    write('ENGINEERING_POSITIVE_CONTROL.json',receipt);write('SEALED.json',{'state':state,'purpose':PURPOSE,'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'original_mode_passed':original,'strict_FP32_mode_passed':strict,'files':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}})
    print('PRECISION_DIAGNOSTIC_SEALED',state,sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
