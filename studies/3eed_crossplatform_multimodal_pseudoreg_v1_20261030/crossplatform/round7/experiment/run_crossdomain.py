"""Waymo-only source100 -> Drone/Quad, R7v1. B1; no target label API.
Parent registers manifest/program/source/optimization hashes and grants GPU.
All selected arms are sealed before an independently authorized CPU scorer.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time,random
import numpy as np
from runner_helpers import (sha,read,validate_manifest,sample_points,array_sha,npz_payload,to_json,MODES,KINDS)
HERE=Path(__file__).resolve().parent;R7=HERE.parent;ROOT=R7.parents[1];R6=R7.parent/'round6'
sys.path.insert(0,str(R7))


def fingerprint(model):
    import hashlib
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        x=v.detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def load_model(plan):
    import torch
    # Do not call R6 configure/dataset: they own historical runtime/data paths.
    assert sha(R6/'factory.py')==plan['r6_factory_sha256']
    runtime=HERE/'runtime';(runtime/'data').mkdir(parents=True,exist_ok=True)
    assets=(R7.parent/'upstream/data/roberta_base').resolve();link=runtime/'data/roberta_base'
    if not link.exists():link.symlink_to(assets,target_is_directory=True)
    assert link.resolve()==assets;os.chdir(runtime)
    spec=importlib.util.spec_from_file_location('round6_source_factory',R6/'factory.py');factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    for rel,digest in plan['upstream_code_sha256'].items():assert sha(factory.U/rel)==digest
    for rel,digest in plan['roberta_sha256'].items():assert sha(assets/rel)==digest
    model=factory.model()
    cp=Path(plan['source_checkpoint_path']);assert sha(cp)==plan['source_checkpoint_sha256']
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    assert payload['epoch']==100 and payload['trained_platforms']==['waymo']
    assert payload['source_provenance']['joint_task_checkpoint_initialization'] is False
    assert payload['source_provenance']['plan_sha256']==plan['source_training_plan_sha256']
    state=payload['model'];assert all(torch.isfinite(x).all() for x in state.values() if x.is_floating_point())
    model.load_state_dict(state,strict=True);entries=len(state);del state,payload
    model.requires_grad_(False).cuda().eval()
    assert fingerprint(model)==plan['source_model_state_fingerprint']
    return model,entries


def token_mask_for(model,text,policy):
    import torch
    from regression_adapter import first_noun_mask
    enc=model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True)
    off=enc['offset_mapping'][0];assert len(off)<=256
    try:return first_noun_mask(text,off),{'readout':'first_lexicon_noun','unsupported_noun':False,'fallback':False,'tokens':len(off)},False
    except ValueError:
        assert policy in ('all_content_tokens','fail_closed_source_only')
        mask=off[:,1]>off[:,0];assert mask.any()
        return mask,{'readout':'all_content_tokens','unsupported_noun':True,'fallback':True,'registered_policy':policy,'tokens':len(off)},policy=='fail_closed_source_only'


def selected_fallback(source,q,reason):
    return [source[q].copy() for _ in KINDS],[reason for _ in KINDS]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('smoke','full'),required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['GPU_execution_authorized'] is True and plan['phase']==a.phase and plan['batch_size']==1
    assert plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    assert plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3]
    assert plan['missing_distortion_policy']=='assume_zero_declared'
    assert plan['noun_fallback'] in ('all_content_tokens','fail_closed_source_only')
    assert sha(a.manifest)==plan['manifest_sha256']
    for rel,digest in plan['program_sha256'].items():
        p=R7/rel;assert p.resolve().is_relative_to(R7.resolve());assert sha(p)==digest
    rows=validate_manifest(read(a.manifest));assert len(rows)==plan['expected_samples']
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    (a.output/'manifest_input.json').write_bytes(a.manifest.read_bytes());started=time.perf_counter()
    import torch
    from PIL import Image
    from regression_adapter import capture_source,adapt,PseudoTarget,StepConfig,parameter_sha
    from camera_projection import read_camera_allowlist,CameraSpec,project_model_points
    from multimodal_pseudo import generate_pseudo,DEFAULT_CONFIG
    import dataclasses,hashlib
    assert hashlib.sha256(json.dumps(dataclasses.asdict(DEFAULT_CONFIG),sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['G_config_sha256']
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=plan['cudnn_benchmark'];torch.backends.cudnn.deterministic=plan['cudnn_deterministic']
    model,entries=load_model(plan);source_fp=fingerprint(model);selected_fp=parameter_sha(model);baseline_buffers={n:b.clone() for n,b in model.named_buffers()};trace=[];files={};repeat_receipt=None;file_hash_cache={};camera_cache={}
    def file_sha(p):
        p=str(p)
        if p not in file_hash_cache:file_hash_cache[p]=sha(p)
        return file_hash_cache[p]
    for index,row in enumerate(rows):
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();pc,ix,nraw=sample_points(row)
        ph=array_sha(pc);ih=array_sha(ix)
        with Image.open(row['imagepath']) as img:rgb=np.asarray(img.convert('RGB'),dtype=np.uint8).copy()
        # Physical metadata container is opened, but only the camera allowlist is
        # retained; no caption/class/GT fields are requested or used here.
        if row['metapath'] not in camera_cache:
            try:camera_cache[row['metapath']]=('ok',read_camera_allowlist(row['metapath']))
            except ValueError as e:camera_cache[row['metapath']]=('unsupported',str(e))
        actual_text=row['caption']+plan['sentinel'];mask,text_info,force_fallback=token_mask_for(model,actual_text,plan['noun_fallback'])
        points=torch.from_numpy(pc)[None].cuda()
        torch.cuda.synchronize();forward_start=time.perf_counter()
        cache=capture_source(model,{'point_clouds':points,'text':[actual_text]},mask)
        torch.cuda.synchronize();forward_seconds=time.perf_counter()-forward_start
        source=cache.boxes_raw6[0].cpu().numpy();scores=cache.text_scores[0].cpu().numpy();q=cache.query_index
        assert source.shape==(256,6) and np.isfinite(source).all() and np.isfinite(scores).all()
        native_top10=torch.argsort(cache.text_scores[0],descending=True)[:10].cpu().numpy()
        hc=cache.center_features[0,:,q];hs=cache.size_features[0,:,q]
        center_norm=float(hc.square().sum())+1.;size_norm=float(hs.square().sum())+1.
        lr=plan['alpha']*plan['beta_metres']/max(center_norm,size_norm)
        assert np.isfinite(lr) and lr>0
        fixed_reason='unsupported_noun_source_only' if force_fallback else ('nonpositive_source_size' if (source[q,3:]<=0).any() else None)
        projected=None;camdiagnostic={};camera_state,camera_value=camera_cache[row['metapath']]
        if camera_state!='ok':fixed_reason=fixed_reason or 'camera_missing';camdiagnostic={'reason':camera_value}
        else:
            try:
                spec=CameraSpec.from_camera_fields(row['platform'],camera_value,waymo_view_suffix=0,missing_distortion_policy=plan['missing_distortion_policy'])
                projected=project_model_points(pc,spec,(rgb.shape[1],rgb.shape[0]));camdiagnostic=projected['diagnostics']
            except ValueError as e:fixed_reason=fixed_reason or 'camera_contract_rejected';camdiagnostic={'reason':str(e)}
        selected=[source[q].copy()];statuses=['source'];pseudo_targets=[];pseudo_weights=[];mode_trace={}
        for mode in MODES:
            mode_tick=time.perf_counter()
            if fixed_reason:
                pseudo={'target6':source[q].copy(),'reliability6':np.zeros(6),'diagnostics':{'reason':fixed_reason}}
                arms,states=selected_fallback(source,q,fixed_reason);mode_trace[mode]={'reason':fixed_reason,'steps':{},'seconds':time.perf_counter()-mode_tick}
            else:
                pseudo=generate_pseudo(points=pc,rgb=rgb,text=actual_text,source_boxes=source,text_scores=scores,query_index=q,uv=projected['uv'],depth=projected['depth'],mode=mode)
                target=PseudoTarget(q,torch.as_tensor(pseudo['target6']),torch.as_tensor(pseudo['reliability6']))
                one=adapt(model,cache,target,StepConfig(1,lr,plan['beta_metres']))
                three=adapt(model,cache,target,StepConfig(3,lr,plan['beta_metres']))
                arms=[one['direct_G'][0,q].cpu().numpy(),one['parameter_SGD'][0,q].cpu().numpy(),three['parameter_SGD'][0,q].cpu().numpy(),one['exact_output_space_SGD'][0,q].cpu().numpy(),three['exact_output_space_SGD'][0,q].cpu().numpy()]
                active=bool(np.any(pseudo['reliability6']))
                states=['direct_G' if active else 'zero_weight_source']+['updated' if active else 'zero_weight_source']*4
                mode_trace[mode]={'reason':pseudo['diagnostics']['reason'],'steps':{str(k):{'trace':v['trace'],'feature_norms':v['feature_norms'],'spillover':v['spillover']} for k,v in [(1,one),(3,three)]},'seconds':time.perf_counter()-mode_tick}
            selected.extend(arms);statuses.extend(states);pseudo_targets.append(pseudo['target6']);pseudo_weights.append(pseudo['reliability6']);mode_trace[mode]['pseudo_diagnostics']=pseudo['diagnostics']
        assert parameter_sha(model)==selected_fp
        if index==0:
            torch.cuda.synchronize();repeat=capture_source(model,{'point_clouds':points,'text':[actual_text]},mask);torch.cuda.synchronize()
            assert torch.equal(cache.boxes_raw6,repeat.boxes_raw6) and torch.equal(cache.text_scores,repeat.text_scores)
            assert source_fp==fingerprint(model)
            assert all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in baseline_buffers.items())
            repeat_receipt={'sample0_source_repeat_bitexact':True,'sample0_real_final_Conv_bridge_bitexact':True,'sample0_post_alladapt_full_model_fingerprint_restored':True,'source_model_state_fingerprint':source_fp,'extra_source_forward_count':1}
        payload=npz_payload(sample_id=row['id'],source_boxes=source,scores=scores,native_top10=native_top10,q=q,selected_arms=selected,statuses=statuses,pseudo_targets=pseudo_targets,pseudo_weights=pseudo_weights,point_sha=ph,image_sha=file_sha(row['imagepath']))
        predname=row['id']+'.npz';np.savez_compressed(a.output/predname,**payload);files[predname]=sha(a.output/predname)
        torch.cuda.synchronize()
        entry={'id':row['id'],'platform':row['platform'],'scene':row['scene'],'query_index':q,'nativeTop1_differs_fixedq':int(native_top10[0])!=q,'text_info':text_info,'source_model_forward_and_Conv_bridge_seconds':forward_seconds,'sample_end_to_end_seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'raw_point_count':nraw,'unique_sample_indices':len(np.unique(ix)),'point_sample_sha256':ph,'sample_indices_sha256':ih,'point_file_sha256':file_sha(row['pointpath']),'image_file_sha256':file_sha(row['imagepath']),'camera_metadata_file_sha256':file_sha(row['metapath']),'camera':camdiagnostic,'feature_norm_squared_plus_bias':{'center':center_norm,'size':size_norm},'lr':lr,'lr_rule':'alpha*beta/max(center_norm2+1,size_norm2+1)','no_SGD_displacement_cap_claim':True,'selected_raw_nonpositive_size':bool((source[q,3:]<=0).any()),'modes':mode_trace}
        trace.append(to_json(entry))
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(trace[-1],allow_nan=False)+'\n')
        if index%10==0 or index+1==len(rows):print(json.dumps({'phase':a.phase,'completed':index+1,'expected':len(rows),'elapsed_seconds':time.perf_counter()-started,'platform':row['platform']}),flush=True)
    assert source_fp==fingerprint(model) and parameter_sha(model)==selected_fp
    assert all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in baseline_buffers.items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules())
    (a.output/'trace.json').write_text(json.dumps(trace,indent=2,allow_nan=False)+'\n');(a.output/'repeat_restore_control.json').write_text(json.dumps(repeat_receipt,indent=2)+'\n')
    files.update({n:sha(a.output/n) for n in ('trace.json','trace.jsonl','repeat_restore_control.json','registered_plan.json','manifest_input.json')})
    seal={'state':'sealed_crossdomain_R7v1_no_target_labels_or_scoring','phase':a.phase,'n_samples':len(rows),'n_prediction_files':len(rows),'manifest_sha256':sha(a.manifest),'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'source_model_state_fingerprint':source_fp,'strict_model_state_entries':entries,'program_sha256':plan['program_sha256'],'R6_factory_sha256':plan['r6_factory_sha256'],'upstream_code_sha256':plan['upstream_code_sha256'],'all_target_model_inputs_XYZ_and_puretext_no_class_positive_map':True,'metadata_camera_allowlist_only':True,'all_16arms_sealed_before_CPUscoring':True,'model_restored_source_state':True,'all_BN_eval_dropout_off':True,'no_teacher_EMA_or_source_retraining':True,'total_elapsed_through_all_files_seconds':time.perf_counter()-started,'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print('R7_CROSSDOMAIN_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
