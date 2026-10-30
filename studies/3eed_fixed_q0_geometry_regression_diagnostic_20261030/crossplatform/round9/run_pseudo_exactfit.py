"""Post-normal-score exploratory optimizer bridge, not unique TTA evidence.
Only reads sealed normal caches + normal directG targets. No GT/scored/RGB/P/G API.
Three arms: same source q0, RGB pseudo-exactfit, PC pseudo-exactfit. Root owns GPU.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
from run_diagnostic import load_model,fingerprint,STRICT_FLAGS,R7,ROOT
from runner_helpers import sha,read,eval_boxes
PURPOSE='R9_post_normal_score_exploratory_pseudo_exactfit_optimizer_bridge'
MODES=('rgb_fulltext','pc_only')
ARM_NAMES=('source','rgb_fulltext__pseudo_exactfit','pc_only__pseudo_exactfit')
FORBIDDEN_LABEL_OR_SCORE_KEYS={'target_index_path','target_index_sha256','GT_index_read_release_path','GT_index_read_release_sha256','scored_path','scored_seal_path','results_private_path'}


def validate_direct_targets(source_q,targets):
    s=np.asarray(source_q,np.float32);t=np.asarray(targets,np.float32)
    assert s.shape==(6,) and t.shape==(2,6) and np.isfinite(s).all() and np.isfinite(t).all()
    assert np.array_equal(t[:,3:],np.tile(s[3:],(2,1))),'Pseudo bridge cannot change source size prior'
    return bool(np.any(s[3:]<=0))


def verify_source_stage(plan):
    assert not FORBIDDEN_LABEL_OR_SCORE_KEYS.intersection(plan),'Bridge has no GT/scored input fields'
    path=Path(plan['normal_prediction_seal_path']);assert sha(path)==plan['normal_prediction_seal_sha256'];seal=read(path)
    assert seal['state']=='sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT' and seal['phase']=='normal' and seal['n_samples']==160 and seal['n_arms']==11
    assert seal['source_checkpoint_sha256']==plan['source_checkpoint_sha256'] and seal['source_model_state_fingerprint']==plan['source_model_state_fingerprint'] and seal['strict_FP32_flags']==STRICT_FLAGS
    assert len(seal['source_cache_files'])==160
    for rel,digest in seal['files'].items():
        p=path.parent/rel;assert p.resolve().is_relative_to(path.parent.resolve()) and sha(p)==digest
    predictions={k:v for k,v in seal['files'].items() if '/' not in k and k.endswith('.npz')};assert len(predictions)==160
    assert plan['source_cache_files']==seal['source_cache_files'] and plan['normal_prediction_files']==predictions
    release=Path(plan['bridge_read_release_path']);assert sha(release)==plan['bridge_read_release_sha256'];r=read(release)
    assert r['unlabelled_pseudo_exactfit_execution_authorized'] is True
    assert r['normal_prediction_seal_sha256']==sha(path) and r['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
    assert r['source_cache_files']==seal['source_cache_files'] and r['normal_prediction_files']==predictions and r['program_sha256']==plan['program_sha256']
    assert r['post_normal_score_exploratory_optimizer_bridge'] is True and r['no_GT_scored_media_or_new_G_inputs'] is True
    return path,seal,predictions


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['purpose']==PURPOSE and plan['GPU_execution_authorized'] is True and plan['expected_samples']==160
    assert plan['post_normal_score_exploratory_optimizer_bridge'] is True and plan['strict_FP32_flags']==STRICT_FLAGS
    assert plan['mask_C']==[1,1,1,0,0,0] and plan['exactfit_coordinate_atol_metres']==5e-5
    assert plan['head_step_rule']=='one_real_half_sum_L2_SGD_center_lr_inverse_feature_kernel_no_search'
    assert plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    assert sha(ROOT/'research/round9/experiment/AUTHORIZATION.json')==plan['authorization_sha256']
    for rel,digest in plan['program_sha256'].items():
        p=ROOT/rel;assert p.resolve().is_relative_to(ROOT.resolve()) and sha(p)==digest
    normal_path,seal,predfiles=verify_source_stage(plan)
    assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    import torch
    sys.path.insert(0,str(R7));from regression_adapter import SourceCache,raw_boxes,parameter_sha
    from head_exact_fit import exact_fit_same_head
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    started=time.perf_counter();model,entries=load_model(plan);basefp=fingerprint(model);files={};traces=[];all_pass=True
    for i,cache_rel in enumerate(sorted(seal['source_cache_files'])):
        ident=Path(cache_rel).stem;cache_file=normal_path.parent/cache_rel;pred_file=normal_path.parent/(ident+'.npz')
        assert sha(cache_file)==seal['source_cache_files'][cache_rel] and sha(pred_file)==predfiles[pred_file.name]
        with np.load(cache_file,allow_pickle=False) as z:
            assert str(z['sample_id'])==ident and str(z['source_checkpoint_sha256'])==plan['source_checkpoint_sha256']
            params={k:torch.from_numpy(z[k].copy()).cuda() for k in ('base_xyz','center_features','size_features','boxes_raw6','text_scores')}
            cache=SourceCache(**params,query_index=int(z['query_index']),selected_parameter_sha256=str(z['selected_parameter_sha256']));ph=z['point_sample_sha256'].copy();ih=z['raw_indices_sha256'].copy()
        q=cache.query_index;source=cache.boxes_raw6[0].cpu().numpy()
        with np.load(pred_file,allow_pickle=False) as z:
            assert int(z['q_source'])==q and np.all(z['arm_query_index']==q)
            assert np.array_equal(z['source_boxes_raw6'],source) and np.array_equal(z['source_text_scores256'],cache.text_scores[0].cpu().numpy())
            assert np.array_equal(ph,z['point_sample_sha256']) and np.array_equal(ih,z['raw_indices_sha256'])
            names=z['arm_names'].tolist();targets=np.stack([z['arm_boxes_raw6'][names.index(m+'__direct_G')] for m in MODES]).astype(np.float32)
        invalid=validate_direct_targets(source[q],targets)
        assert parameter_sha(model)==cache.selected_parameter_sha256
        with torch.no_grad():replayed=raw_boxes(model,cache)
        bridge_error=float((replayed-cache.boxes_raw6).abs().max());bridge_exact=bool(torch.equal(replayed,cache.boxes_raw6))
        with (a.output/'cache_replay.jsonl').open('a') as f:f.write(json.dumps({'id':ident,'max_abs_error':bridge_error,'bitexact':bridge_exact})+'\n')
        assert bridge_exact,'Keep actual cache replay evidence, never silently relax'
        selected=[source[q].copy()];statuses=['same_normal_source'];mode_trace={};residuals=[];passes=[]
        for mi,mode in enumerate(MODES):
            if invalid:
                chosen=source[q].copy();status='nonpositive_source_size_fallback';details={'real_backward_and_optimizer_step_executed':False,'skip_reason':status};residual=chosen-targets[mi];passed=True
            else:
                result=exact_fit_same_head(model,cache,targets[mi],[1,1,1,0,0,0]);chosen=result['boxes_raw6'][0,q].cpu().numpy();residual=chosen-targets[mi]
                passed=result['supervised_max_abs_coordinate_residual_m']<=5e-5 and result['inactive_coordinates_bitexact_source']
                details={k:v for k,v in result.items() if k!='boxes_raw6'};status='pseudo_exactfit_coordinate_pass' if passed else 'pseudo_exactfit_coordinate_failure_preserved'
            assert np.array_equal(chosen[3:],source[q,3:])
            selected.append(chosen);statuses.append(status);residuals.append(residual);passes.append(bool(passed));all_pass&=passed
            mode_trace[mode]={'target_origin':'exact same normal sealed direct_G raw6; not GT or new G','normal_directG_target_minus_source6':(targets[mi]-source[q]).tolist(),'pseudo_equals_source':bool(np.array_equal(targets[mi],source[q])),'target_center_delta_L2_m':float(np.linalg.norm(targets[mi,:3]-source[q,:3])),'center_residual_L2_m':float(np.linalg.norm(residual[:3])),'center_relative_residual_if_target_nonzero':float(np.linalg.norm(residual[:3])/np.linalg.norm(targets[mi,:3]-source[q,:3])) if np.linalg.norm(targets[mi,:3]-source[q,:3])>0 else None,'observed_selected_delta6':(chosen-source[q]).tolist(),'coordinate_residual_to_normal_directG6':residual.tolist(),'coordinate_check_applicable':not invalid,'coordinate_atol5e_minus5_pass':bool(passed),'exactfit':details}
            assert fingerprint(model)==basefp
        raw=np.asarray(selected,np.float32);payload={'sample_id':np.array(ident),'source_boxes_raw6':source,'source_boxes_eval6':eval_boxes(source),'source_text_scores256':cache.text_scores[0].cpu().numpy(),'q_source':np.array(q,np.int64),'modes':np.array(MODES),'arm_names':np.array(ARM_NAMES),'arm_modes':np.array(('source',)+MODES),'arm_query_index':np.full(3,q,np.int64),'arm_status':np.array(statuses),'arm_boxes_raw6':raw,'arm_boxes_eval6':eval_boxes(raw),'normal_directG_targets_raw6':targets,'coordinate_residual_to_normal_directG6':np.asarray(residuals,np.float32),'exactfit_coordinate_pass_by_mode':np.asarray(passes,bool),'coordinate_check_applicable':np.array(not invalid),'mask_C':np.array([1,1,1,0,0,0],np.float32),'point_sample_sha256':ph,'raw_indices_sha256':ih,'normal_source_cache_sha256':np.frombuffer(bytes.fromhex(seal['source_cache_files'][cache_rel]),np.uint8),'normal_prediction_file_sha256':np.frombuffer(bytes.fromhex(predfiles[pred_file.name]),np.uint8)}
        name=ident+'.npz';np.savez_compressed(a.output/name,**payload);files[name]=sha(a.output/name)
        tr={'id':ident,'q_source':q,'normal_source_cache':cache_rel,'normal_source_cache_sha256':seal['source_cache_files'][cache_rel],'normal_prediction_file_sha256':predfiles[pred_file.name],'normal_Conv_replay_bitexact':True,'invalid_source_size_fallback':invalid,'modes':mode_trace,'full_source_model_restored':True,'post_normal_score_exploratory':True};traces.append(tr)
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(tr,allow_nan=False)+'\n')
        if i%10==0 or i+1==160:print(json.dumps({'phase':'unlabelled_pseudo_exactfit_optimizer_bridge','completed':i+1,'n':160,'elapsed_seconds':time.perf_counter()-started,'all_coordinate_checks_pass_sofar':all_pass}),flush=True)
    assert fingerprint(model)==basefp and all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules())
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    for n in ('registered_plan.json','trace.json','trace.jsonl','cache_replay.jsonl'):files[n]=sha(a.output/n)
    out={'state':'sealed_post_normal_score_unlabelled_pseudo_exactfit_optimizer_bridge','purpose':PURPOSE,'n_samples':160,'n_prediction_files':160,'n_arms':3,'normal_prediction_seal_sha256':sha(normal_path),'normal_source_cache_files':seal['source_cache_files'],'normal_prediction_files':predfiles,'plan_sha256':sha(a.plan),'bridge_read_release_sha256':plan['bridge_read_release_sha256'],'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'source_model_state_fingerprint':basefp,'strict_state_entries':entries,'strict_FP32_flags':STRICT_FLAGS,'program_sha256':plan['program_sha256'],'program_sha256_base':'workspace_root','post_normal_score_exploratory_optimizer_bridge':True,'same_normal_directG_targets_no_new_G':True,'all_query_indices_same_normal_q0':True,'all_applicable_coordinate_checks_pass':bool(all_pass),'no_GT_scored_files_RGB_PC_CLIP_G_or_backbone_forward_consumed':True,'not_unique_TTA_benefit_or_heldout_claim':True,'source_state_restored':True,'elapsed_seconds':time.perf_counter()-started,'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(out,indent=2)+'\n');print('PSEUDO_EXACTFIT_BRIDGE_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
