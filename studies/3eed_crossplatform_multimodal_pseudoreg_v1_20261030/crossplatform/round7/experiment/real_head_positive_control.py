"""Real frozen-head engineering positive control, not G/TTA/task evidence.
Parent GPU authorization required; run only after full prediction seal, under the
shared physical GPU flock. Uses no RGB, metadata camera, target annotations or GT.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
from runner_helpers import sha,read,validate_manifest,sample_points,array_sha
from run_crossdomain import load_model,token_mask_for,fingerprint,R7

PURPOSE='R7_real_head_engineering_positive_control_no_task_metrics'
DELTA6=[.05,0.,0.,0.,0.,0.]
WEIGHT6=[1.,0.,0.,0.,0.,0.]
ATOL_METRES=5e-5

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['GPU_execution_authorized'] is True and plan['purpose']==PURPOSE
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
    with np.load(first_cache,allow_pickle=False) as saved:
        assert int(saved['fixed_query_index'])==q
        assert np.array_equal(saved['source_boxes_raw6'],source[0].cpu().numpy())
        assert np.array_equal(saved['source_text_scores256'],cache.text_scores[0].cpu().numpy())
        assert saved['point_sample_sha256'].tobytes().hex()==array_sha(pc)

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
    receipt={'state':'engineering_positive_control_pass_no_G_feedback_no_GT_no_taskscore','purpose':PURPOSE,'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'strict_state_entries':entries,'plan_sha256':sha(a.plan),'smoke_manifest_sha256':sha(manifest),'smoke_prediction_seal_sha256':sha(smoke_seal),'completed_full_prediction_seal_sha256':sha(full_seal),'smoke_sample_index':0,'sample_id':row['id'],'fixed_source_query_index':q,'sourcepoint_sample_sha256':array_sha(pc),'sourcepoint_indices_sha256':array_sha(ix),'rawpoint_count':nraw,'same_smoke_first_sample_source_pool_scores_query_and_pointSHA_verified':True,'puretext_source_repeat_bitexact':True,'real_Conv_source_bridge_bitexact':True,'source_model_fingerprint_restored':initial_fp,'artificial_target_delta6_metres':DELTA6,'artificial_reliability6':WEIGHT6,'alpha':.25,'beta_metres':.1,'lr':lr,'selected_x_effective_shrink_factor':factor,'no_clip_no_weight_decay_no_momentum':True,'predeclared_equivalence_atol_metres':ATOL_METRES,'results':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started,'no_RGB_camera_G_model_or_targetannotation_GT_or_taskmetric_used':True,'interpretation':'Only proves real source final-layer backward/update and restore interface under artificial supervision; provides no evidence that G is informative or TTA improves target grounding.'}
    dest=a.output/'ENGINEERING_POSITIVE_CONTROL.json';dest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    (a.output/'SEALED.json').write_text(json.dumps({'purpose':PURPOSE,'state':receipt['state'],'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'files':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}},indent=2)+'\n')
    print('REAL_HEAD_POSITIVE_CONTROL_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
