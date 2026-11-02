"""R9 fixed-q0 geometry / isolated GT capability diagnostics.
Parent registers code/inputs/lease. No target-GT API. Sequential CUDA use only:
grounder source then frozen CLIP then tiny heads; no simultaneous streams/workers.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,random,sys,time
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1];R7=D.parent/'round7';R6=D.parent/'round6';R8=D.parent/'round8'
sys.path[:0]=[str(D),str(R8),str(R7),str(R7/'experiment')]
from runner_helpers import sha,read,validate_manifest,sample_points,array_sha,to_json,eval_boxes
MODES=('rgb_fulltext','pc_only')
KINDS=('direct_G','param1','param3','outputspace1','outputspace3')
ARM_NAMES=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)
STRICT_FLAGS={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}

def fingerprint(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        x=v.detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def load_model(plan):
    import torch
    assert sha(R6/'factory.py')==plan['r6_factory_sha256']
    runtime=D/'runtime';(runtime/'data').mkdir(parents=True,exist_ok=True);assets=(D.parent/'upstream/data/roberta_base').resolve();link=runtime/'data/roberta_base'
    if not link.exists():link.symlink_to(assets,target_is_directory=True)
    assert link.resolve()==assets;os.chdir(runtime)
    spec=importlib.util.spec_from_file_location('R8_original_source_factory',R6/'factory.py');factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    for rel,digest in plan['upstream_code_sha256'].items():assert sha(factory.U/rel)==digest
    for rel,digest in plan['roberta_sha256'].items():assert sha(assets/rel)==digest
    cp=Path(plan['source_checkpoint_path']);assert sha(cp)==plan['source_checkpoint_sha256']
    model=factory.model();state=torch.load(cp,map_location='cpu',weights_only=True)
    assert state['epoch']==100 and state['trained_platforms']==['waymo'] and not state['source_provenance']['joint_task_checkpoint_initialization']
    assert state['source_provenance']['plan_sha256']==plan['source_training_plan_sha256']
    assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point());model.load_state_dict(state['model'],strict=True);entries=len(state['model']);del state
    model.requires_grad_(False).cuda().eval();assert fingerprint(model)==plan['source_model_state_fingerprint'];return model,entries

def mask_for(model,text,policy):
    from regression_adapter import first_noun_mask
    tok=model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True);off=tok['offset_mapping'][0];assert len(off)<=256
    try:return first_noun_mask(text,off),{'readout':'first_lexicon_noun','fallback':False,'tokens':len(off)}
    except ValueError:
        assert policy=='all_content_tokens';mask=off[:,1]>off[:,0];assert mask.any();return mask,{'readout':'all_content_tokens','fallback':True,'tokens':len(off)}

def empty_G(source,q,mode,reason):
    return {'original_source_q':q,'associated_q':q,'association_only_box':source[q].copy(),'pseudo6':source[q].copy(),'raw_consensus_center':source[q,:3].copy(),'r_global':0.,'w6':np.zeros(6),'mode':mode,'yaw':0.,'diagnostics':{'reason':reason,'nonzero_pseudo_supervision':False}}

def serialize_npz(*,sample_id,source,scores,top,q0,G_results,selected_arms,statuses,prepared,cosines,point_sha,index_sha,image_sha):
    assert source.shape==(256,6) and scores.shape==(256,) and len(G_results)==2
    raw=np.asarray(selected_arms,np.float32);assert raw.shape==(11,6) and np.array_equal(raw[0],source[q0])
    assert all(int(g['associated_q'])==q0 and int(g['original_source_q'])==q0 for g in G_results)
    actual=np.array([g['pseudo6'] for g in G_results],np.float32);w=np.array([g['w6'] for g in G_results],np.float32)
    assert not w[:,3:].any() and np.array_equal(actual[:,3:],np.tile(source[q0,3:],(2,1)))
    rois=[] if prepared is None else prepared['roi_groups']
    return {'sample_id':np.array(sample_id),'source_boxes_raw6':source,'source_boxes_eval6':eval_boxes(source),'source_text_scores256':scores,'source_native_top10':top,'q_source':np.array(q0,np.int64),'modes':np.array(MODES),'forced_q_source':np.array([q0,q0],np.int64),'raw_consensus_center':np.array([g['raw_consensus_center'] for g in G_results],np.float64),'r_global':np.array([g['r_global'] for g in G_results],np.float64),'pseudo_target6_float64':np.array([g['pseudo6'] for g in G_results],np.float64),'pseudo_target6':actual,'pseudo_reliability6':w,'arm_names':np.array(ARM_NAMES),'arm_modes':np.array(('source',)+tuple(m for m in MODES for k in KINDS)),'arm_query_index':np.full(11,q0,np.int64),'arm_status':np.array(statuses),'arm_boxes_raw6':raw,'arm_boxes_eval6':eval_boxes(raw),'dedup_membership':np.full(256,-1,np.int64) if prepared is None else prepared['dedup_membership'],'semantic_cosines_tight_context':np.asarray(cosines,np.float32),'roi_group_ids':np.array([g['group_id'] for g in rois],np.int64),'roi_xyxy':np.array([[g['tight_xyxy'],g['context_xyxy']] for g in rois],np.int64).reshape(-1,2,4),'point_sample_sha256':np.frombuffer(bytes.fromhex(point_sha),np.uint8),'raw_indices_sha256':np.frombuffer(bytes.fromhex(index_sha),np.uint8),'image_file_sha256':np.frombuffer(bytes.fromhex(image_sha),np.uint8)}


def save_source_cache(path,cache,*,sample_id,point_sha,index_sha,model_text,manifest_row,checkpoint_sha):
    arrays={name:getattr(cache,name).detach().cpu().numpy() for name in ('base_xyz','center_features','size_features','boxes_raw6','text_scores')}
    arrays.update(sample_id=np.array(sample_id),query_index=np.array(cache.query_index,np.int64),selected_parameter_sha256=np.array(cache.selected_parameter_sha256),point_sample_sha256=np.frombuffer(bytes.fromhex(point_sha),np.uint8),raw_indices_sha256=np.frombuffer(bytes.fromhex(index_sha),np.uint8),actual_model_text_sha256=np.frombuffer(hashlib.sha256(model_text.encode()).digest(),np.uint8),manifest_row_sha256=np.frombuffer(hashlib.sha256(json.dumps(manifest_row,sort_keys=True,separators=(',',':')).encode()).digest(),np.uint8),source_checkpoint_sha256=np.array(checkpoint_sha))
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)


def normal(a,plan):
    assert a.manifest is not None
    assert plan['strict_FP32_flags']==STRICT_FLAGS and plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3]
    assert plan['r_global_policy']=='shrink_shared_pseudo_target_no_extra_LR_multiplier' and plan['selection_policy']=='fixed_original_q0_group_no_reselect'
    assert plan['CLIP_device']=='cuda' and plan['CLIP_batch_size']==32 and plan['sequential_single_GPU_only'] is True
    assert plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    assert sha(a.manifest)==plan['manifest_sha256'];rows=validate_manifest(read(a.manifest));assert len(rows)==plan['expected_samples']
    for rel,digest in plan['program_sha256'].items():
        file=ROOT/rel;assert file.resolve().is_relative_to(ROOT.resolve()) and sha(file)==digest
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes());(a.output/'manifest_input.json').write_bytes(a.manifest.read_bytes());start=time.perf_counter()
    import torch
    from PIL import Image
    from regression_adapter import capture_source,adapt,PseudoTarget,StepConfig,parameter_sha
    from camera_projection import read_camera_allowlist,CameraSpec,project_model_points
    from fixed_q0_geometry import prepare_candidates,fixed_q0_pseudo,DEFAULT_CONFIG
    from semantic_encoder import FrozenSemanticEncoder,crop_rgb
    assert hashlib.sha256(json.dumps(dataclasses.asdict(DEFAULT_CONFIG),sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['G_config_sha256']
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
    model,entries=load_model(plan);basefp=fingerprint(model);headfp=parameter_sha(model)
    encoder=FrozenSemanticEncoder(asset_dir=Path(plan['CLIP_asset_dir']),device='cuda',batch_size=32)
    assert sha(Path(plan['CLIP_asset_dir'])/'SAFE_CONVERSION_RECEIPT.json')==plan['CLIP_conversion_receipt_sha256']
    clipversions={n:p._version for n,p in encoder.model.named_parameters()};clipbuffers={n:b.clone() for n,b in encoder.model.named_buffers()};clipfp=fingerprint(encoder.model)
    files={};source_cache_files={};traces=[];repeat_receipt=None;camera_cache={};hashcache={}
    def fhash(p):
        p=str(p)
        if p not in hashcache:hashcache[p]=sha(p)
        return hashcache[p]
    for index,row in enumerate(rows):
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();pc,ix,nraw=sample_points(row);ph=array_sha(pc);ih=array_sha(ix)
        with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB'),np.uint8).copy()
        text=row['caption']+plan['sentinel'];mask,text_info=mask_for(model,text,plan['noun_fallback']);points=torch.from_numpy(pc)[None].cuda()
        torch.cuda.synchronize();t=time.perf_counter();cache=capture_source(model,{'point_clouds':points,'text':[text]},mask);torch.cuda.synchronize();source_seconds=time.perf_counter()-t
        source=cache.boxes_raw6[0].cpu().numpy();scores=cache.text_scores[0].cpu().numpy();q0=cache.query_index;top=torch.argsort(cache.text_scores[0],descending=True)[:10].cpu().numpy()
        assert source.shape==(256,6) and np.isfinite(source).all() and np.isfinite(scores).all()
        cache_rel='source_cache/'+row['id']+'.npz'
        save_source_cache(a.output/cache_rel,cache,sample_id=row['id'],point_sha=ph,index_sha=ih,model_text=text,manifest_row=row,checkpoint_sha=plan['source_checkpoint_sha256']);source_cache_files[cache_rel]=sha(a.output/cache_rel);files[cache_rel]=source_cache_files[cache_rel]
        prepared=None;camera_diagnostic={};camera_reason=None
        if row['metapath'] not in camera_cache:
            try:camera_cache[row['metapath']]=read_camera_allowlist(row['metapath'])
            except ValueError:camera_cache[row['metapath']]=None
        try:
            if camera_cache[row['metapath']] is None:raise ValueError('missing camera fields')
            camera=CameraSpec.from_camera_fields(row['platform'],camera_cache[row['metapath']],waymo_view_suffix=0,missing_distortion_policy=plan['missing_distortion_policy'])
            projection=project_model_points(pc,camera,(rgb.shape[1],rgb.shape[0]));camera_diagnostic=projection['diagnostics']
            prepared=prepare_candidates(points=pc,uv=projection['uv'],depth=projection['depth'],image_size_wh=(rgb.shape[1],rgb.shape[0]),source_boxes=source,text_scores=scores,source_q=q0)
        except ValueError as e:camera_reason=str(e)
        rois=[] if prepared is None else prepared['roi_groups'];cropbounds=[];roles=[]
        for g in rois:
            for role in ('tight','context'):cropbounds.append(g[role+'_xyxy']);roles.append(str(g['group_id'])+'__'+role)
        if rois:
            crops,external_crop_receipt=crop_rgb(rgb,cropbounds)
            # Literal fullcaption only: grounder's sentinel is NOT CLIP text.
            torch.cuda.synchronize();semantic=encoder.match(crops,[row['caption']],crop_roles=roles);torch.cuda.synchronize();cosines=semantic['cosines'][:,0].reshape(len(rois),2)
        else:
            _,text_receipt=encoder.tokenize_texts([row['caption']]);semantic={'cosines':np.empty((0,1),np.float32),'text_receipt':text_receipt,'crop_receipt':{'crop_count':0},'resource':encoder.resource,'seconds':0.};external_crop_receipt=[];cosines=np.empty((0,2),np.float32)
        G_results=[];arms=[source[q0].copy()];states=['source_q0'];mode_trace={}
        for mode in MODES:
            g=empty_G(source,q0,mode,'camera_contract_rejected: '+str(camera_reason)) if prepared is None else fixed_q0_pseudo(prepared,cosines,mode=mode)
            q1=q0;assert int(g['associated_q'])==q0;association=source[q0].copy();target32=np.asarray(g['pseudo6'],np.float32);weights32=np.asarray(g['w6'],np.float32)
            # Copy, do not mutate or retag original cache.query_index/q0.
            associated_cache=cache
            hc=cache.center_features[0,:,q1];hs=cache.size_features[0,:,q1];cn=float(hc.square().sum())+1.;sn=float(hs.square().sum())+1.;lr=.25*.1/max(cn,sn)
            if (association[3:]<=0).any():
                selected=[association.copy() for _ in KINDS];status=['invalid_source_size_q0_fallback']*5;steps={}
            else:
                pseudo=PseudoTarget(q1,torch.as_tensor(target32),torch.as_tensor(weights32))
                one=adapt(model,associated_cache,pseudo,StepConfig(1,lr,.1));three=adapt(model,associated_cache,pseudo,StepConfig(3,lr,.1))
                selected=[one['direct_G'][0,q1].cpu().numpy(),one['parameter_SGD'][0,q1].cpu().numpy(),three['parameter_SGD'][0,q1].cpu().numpy(),one['exact_output_space_SGD'][0,q1].cpu().numpy(),three['exact_output_space_SGD'][0,q1].cpu().numpy()]
                effective=bool(np.any((weights32[:3]>0)&(target32[:3]!=association[:3])))
                status=['direct_G']+['effective_pseudo_target_requested' if effective else 'zero_effective_pseudo_delta']*4
                steps={str(k):{'trace':r['trace'],'feature_norms':r['feature_norms'],'spillover':r['spillover'],'selected_final_delta_from_association':(r['parameter_SGD'][0,q1].cpu().numpy()-association).tolist()} for k,r in [(1,one),(3,three)]}
            arms.extend(selected);states.extend(status);G_results.append(g)
            mode_trace[mode]={'G_diagnostics':g['diagnostics'],'original_q0':q0,'associated_q1':q1,'actual_lr':lr,'lr_has_no_global_trust_multiplier':True,'feature_norm2_plus1':{'center':cn,'size':sn},'raw_consensus_minus_anchor_m':(g['raw_consensus_center']-association[:3]).tolist(),'r_global_heuristic_strength_not_probability':g['r_global'],'target64_minus_anchor_m':(g['pseudo6'][:3]-association[:3]).tolist(),'actual_target32_minus_anchor_m':(target32[:3]-association[:3]).tolist(),'target64_to32_max_abs_quantization':float(np.abs(g['pseudo6']-target32).max()),'steps':steps}
        assert cache.query_index==q0 and parameter_sha(model)==headfp
        if index==0:
            repeated=capture_source(model,{'point_clouds':points,'text':[text]},mask)
            assert torch.equal(cache.boxes_raw6,repeated.boxes_raw6) and torch.equal(cache.text_scores,repeated.text_scores)
            assert fingerprint(model)==basefp and fingerprint(encoder.model)==clipfp
            repeat_receipt={'source_repeat_bitexact':True,'source_final_Conv_bridge_bitexact':True,'post_all_modes_grounder_fullstate_restored':True,'frozen_CLIP_fullstate_unchanged':True,'strict_FP32_flags':STRICT_FLAGS,'extra_source_forward_count':1}
        z=serialize_npz(sample_id=row['id'],source=source,scores=scores,top=top,q0=q0,G_results=G_results,selected_arms=arms,statuses=states,prepared=prepared,cosines=cosines,point_sha=ph,index_sha=ih,image_sha=fhash(row['imagepath']))
        name=row['id']+'.npz';np.savez_compressed(a.output/name,**z);files[name]=sha(a.output/name);torch.cuda.synchronize()
        groups=[]
        if prepared is not None:
            import itertools
            for group in prepared['groups']:
                summary={k:v for k,v in group.items() if k!='support_ids'}
                pairs=[prepared['pair_evidence'][tuple(sorted(pair))] for pair in itertools.combinations(group['members'],2)]
                summary['member_visible_support_counts']=[len(prepared['supports'][q]) for q in group['members']]
                summary['complete_link_pair_minima']={key:min(x[key] for x in pairs) for key in ('shared_points','jaccard','support_overlap_coefficient','box_AABB_IoU')} if pairs else None
                groups.append(summary)
        entry={'source_cache_file':cache_rel,'source_cache_file_sha256':source_cache_files[cache_rel],'id':row['id'],'platform':row['platform'],'scene':row['scene'],'q_source0':q0,'text_readout':text_info,'literal_fullcaption_for_CLIP_no_sentinel':True,'point_sample_sha256':ph,'raw_indices_sha256':ih,'raw_point_file_sha256':fhash(row['pointpath']),'image_file_sha256':fhash(row['imagepath']),'camera_metadata_file_sha256':fhash(row['metapath']),'raw_point_count':nraw,'unique_sample_indices':len(np.unique(ix)),'camera':camera_diagnostic,'camera_skip_reason':camera_reason,'candidate_pool':[] if prepared is None else prepared['candidate_pool'],'groups':groups,'roi_group_ids':[g['group_id'] for g in rois],'external_crop_receipt':external_crop_receipt,'CLIP':{k:v for k,v in semantic.items() if k!='cosines'},'G_modes':mode_trace,'source_forward_snapshot_Convbridge_seconds':source_seconds,'sample_end_to_end_seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'single_GPU_serial_source_then_CLIP_then_heads':True}
        traces.append(to_json(entry))
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(traces[-1],allow_nan=False)+'\n')
        if index%10==0 or index+1==len(rows):print(json.dumps({'phase':a.phase,'completed':index+1,'n':len(rows),'elapsed_seconds':time.perf_counter()-start,'roi_count':len(rois),'eligible_modes':sum(bool(np.any(g['w6'])) for g in G_results),'real_parameter_change_modes':sum(any(any(v>0 for v in t['parameter_delta_max_by_name'].values()) for t in m['steps'].get('3',{}).get('trace',[])) for m in mode_trace.values())}),flush=True)
    assert fingerprint(model)==basefp and fingerprint(encoder.model)==clipfp
    assert clipversions=={n:p._version for n,p in encoder.model.named_parameters()} and all(torch.equal(b,dict(encoder.model.named_buffers())[n]) for n,b in clipbuffers.items());encoder._assert_frozen()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules())
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n');(a.output/'repeat_restore_control.json').write_text(json.dumps(repeat_receipt,indent=2)+'\n')
    files.update({n:sha(a.output/n) for n in ('trace.json','trace.jsonl','registered_plan.json','manifest_input.json','repeat_restore_control.json')})
    seal={'state':'sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT','phase':a.phase,'n_samples':len(rows),'n_prediction_files':len(rows),'n_arms':11,'manifest_sha256':sha(a.manifest),'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'source_model_state_fingerprint':basefp,'source_strict_state_entries':entries,'CLIP_resource':encoder.resource,'CLIP_state_fingerprint':clipfp,'strict_FP32_flags':STRICT_FLAGS,'program_sha256':plan['program_sha256'],'program_sha256_base':'workspace_root','sourceQ0_not_retagged_as_associationQ1':True,'all_11arms_and_sourcecache_sealed_before_GT':True,'source_cache_count':len(source_cache_files),'source_cache_files':source_cache_files,'all_normal_arm_queries_fixed_q0':True,'grounder_restored_and_CLIP_frozen':True,'no_backbone_or_source_retraining_or_EMA':True,'elapsed_seconds':time.perf_counter()-start,'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print('R9_NORMAL_SEALED',sha(a.output/'SEALED.json'),flush=True)


GT_GROUPS={'C':[1,1,1,0,0,0],'S':[0,0,0,1,1,1],'CS':[1,1,1,1,1,1]}
GT_KINDS=('direct_target','param1','param3','exactfit')
GT_ARM_NAMES=('source',)+tuple('GT_'+g+'__'+k for g in GT_GROUPS for k in GT_KINDS)

def checked_normal_before_GT(plan):
    seal_path=Path(plan['normal_prediction_seal_path']);assert sha(seal_path)==plan['normal_prediction_seal_sha256'];s=read(seal_path)
    assert s['state']=='sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT' and s['phase']=='normal'
    assert s['n_samples']==160 and s['n_arms']==11 and s['source_cache_count']==160 and s['all_normal_arm_queries_fixed_q0']
    assert s['source_checkpoint_sha256']==plan['source_checkpoint_sha256'] and s['source_model_state_fingerprint']==plan['source_model_state_fingerprint'] and s['strict_FP32_flags']==STRICT_FLAGS
    for rel,digest in s['files'].items():
        p=seal_path.parent/rel;assert p.resolve().is_relative_to(seal_path.parent.resolve()) and sha(p)==digest
    assert s['source_cache_files']=={k:v for k,v in s['files'].items() if k.startswith('source_cache/')}
    release=Path(plan['GT_index_read_release_path']);assert sha(release)==plan['GT_index_read_release_sha256'];r=read(release)
    assert r['GT_index_read_authorized'] is True and r['normal_prediction_seal_sha256']==sha(seal_path)
    assert r['target_index_sha256']==plan['target_index_sha256'] and r['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
    assert r['program_sha256']==plan['program_sha256'] and r['source_cache_count']==160
    index=Path(plan['target_index_path']);assert sha(index)==plan['target_index_sha256']
    # First authorized numerical GT read occurs only after complete normal/cache
    # seal and explicit source/index/current-program release above are verified.
    rows=read(index);assert isinstance(rows,list) and len(rows)==160
    for row in rows:
        assert set(row)=={'id','target6_modelcoords'}
        target=np.asarray(row['target6_modelcoords'],np.float64)
        if target.shape!=(6,) or not np.isfinite(target).all() or np.any(target[3:]<=0):raise ValueError('Invalid registered GT target; fail explicitly, never drop a sample')
    assert len({r['id'] for r in rows})==160
    assert {r['id'] for r in rows}=={Path(name).stem for name in s['source_cache_files']}
    return s,seal_path,rows


def gt_fit(a,plan):
    assert a.manifest is None,'GT_fit consumes only sealed source cache and separate target index, never P/RGB/normalmanifest input'
    assert plan['strict_FP32_flags']==STRICT_FLAGS and plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3]
    assert plan['exactfit_coordinate_atol_metres']==5e-5 and plan['GT_groups']==GT_GROUPS
    seal,seal_path,targets=checked_normal_before_GT(plan)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    (a.output/'GT_INDEX_READ_TRANSACTION.json').write_text(json.dumps({'normal_prediction_seal_sha256':sha(seal_path),'source_cache_files':seal['source_cache_files'],'target_index_sha256':plan['target_index_sha256'],'GT_index_release_sha256':plan['GT_index_read_release_sha256'],'n':160,'GTyaw_not_loaded':True,'no_GT_in_normal':True},indent=2)+'\n')
    import torch
    from regression_adapter import SourceCache,adapt,PseudoTarget,StepConfig,raw_boxes,parameter_sha
    from head_exact_fit import exact_fit_same_head
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    started=time.perf_counter();model,entries=load_model(plan);basefp=fingerprint(model);files={};traces=[];all_exactfit_pass=True
    for i,row in enumerate(targets):
        cache_rel='source_cache/'+row['id']+'.npz';cache_path=seal_path.parent/cache_rel;assert sha(cache_path)==seal['source_cache_files'][cache_rel]
        with np.load(cache_path,allow_pickle=False) as z:
            assert str(z['sample_id'])==row['id'] and str(z['source_checkpoint_sha256'])==plan['source_checkpoint_sha256']
            kwargs={k:torch.from_numpy(z[k].copy()).cuda() for k in ('base_xyz','center_features','size_features','boxes_raw6','text_scores')}
            cache=SourceCache(**kwargs,query_index=int(z['query_index']),selected_parameter_sha256=str(z['selected_parameter_sha256']))
            ph=z['point_sample_sha256'].copy();ih=z['raw_indices_sha256'].copy();textsha=z['actual_model_text_sha256'].copy();rowsha=z['manifest_row_sha256'].copy()
        assert parameter_sha(model)==cache.selected_parameter_sha256
        with torch.no_grad():replayed=raw_boxes(model,cache)
        bridge=float((replayed-cache.boxes_raw6).abs().max());bitexact=bool(torch.equal(replayed,cache.boxes_raw6))
        with (a.output/'cache_replay.jsonl').open('a') as f:f.write(json.dumps({'id':row['id'],'max_abs_error':bridge,'bitexact':bitexact})+'\n')
        assert bitexact,'Normal cached finalConv replay must be exact; retain diagnostic before failing'
        q=cache.query_index;source=cache.boxes_raw6[0].cpu().numpy();GT=np.asarray(row['target6_modelcoords'],np.float32)
        hc=cache.center_features[0,:,q];hs=cache.size_features[0,:,q];cn=float(hc.square().sum())+1.;sn=float(hs.square().sum())+1.;lr=.25*.1/max(cn,sn)
        arms=[source[q].copy()];masks=[np.zeros(6,np.float32)];expected_targets=[source[q].copy()];states=['source_same_normal_cache'];tracegroups={}
        for group,mask_values in GT_GROUPS.items():
            mask=np.asarray(mask_values,np.float32);pseudo=PseudoTarget(q,torch.as_tensor(GT),torch.as_tensor(mask))
            one=adapt(model,cache,pseudo,StepConfig(1,lr,.1));three=adapt(model,cache,pseudo,StepConfig(3,lr,.1));exact=exact_fit_same_head(model,cache,GT,mask)
            constructed=np.where(mask>0,GT,source[q]).astype(np.float32)
            selected=[one['direct_G'][0,q].cpu().numpy(),one['parameter_SGD'][0,q].cpu().numpy(),three['parameter_SGD'][0,q].cpu().numpy(),exact['boxes_raw6'][0,q].cpu().numpy()]
            assert np.array_equal(selected[0],constructed)
            for value in selected:assert np.array_equal(value[mask==0],source[q][mask==0]),'Inactive coordinates must remain raw source, never clamp'
            arms.extend(selected);masks.extend([mask]*4);expected_targets.extend([constructed]*4)
            passed=exact['supervised_max_abs_coordinate_residual_m']<=5e-5 and exact['inactive_coordinates_bitexact_source'];all_exactfit_pass&=passed
            states.extend(['direct_GT_control','SGD_GT_control','SGD_GT_control','exactfit_coordinate_pass' if passed else 'exactfit_coordinate_failure_preserved'])
            tracegroups[group]={'coordinate_mask':mask.tolist(),'actual_SGD_lr':lr,'SGD1':{k:one[k] for k in ('trace','feature_norms','spillover')},'SGD3':{k:three[k] for k in ('trace','feature_norms','spillover')},'exactfit':{k:v for k,v in exact.items() if k!='boxes_raw6'},'exactfit_atol5e_minus5_pass':passed}
        raw=np.asarray(arms,np.float32);expected=np.asarray(expected_targets,np.float32)
        z={'sample_id':np.array(row['id']),'source_boxes_raw6':source,'source_boxes_eval6':eval_boxes(source),'source_text_scores256':cache.text_scores[0].cpu().numpy(),'q_source':np.array(q,np.int64),'arm_names':np.array(GT_ARM_NAMES),'arm_modes':np.array(('source',)+tuple('GT_'+g for g in GT_GROUPS for k in GT_KINDS)),'arm_query_index':np.full(13,q,np.int64),'arm_status':np.array(states),'arm_boxes_raw6':raw,'arm_boxes_eval6':eval_boxes(raw),'target6_modelcoords':GT,'GT_group_names':np.array(list(GT_GROUPS)),'coordinate_masks6':np.array(list(GT_GROUPS.values()),np.float32),'arm_coordinate_masks6':np.asarray(masks,np.float32),'constructed_target6_by_arm':expected,'coordinate_residual6_by_arm':raw-expected,'point_sample_sha256':ph,'raw_indices_sha256':ih,'actual_model_text_sha256':textsha,'manifest_row_sha256':rowsha,'normal_source_cache_sha256':np.frombuffer(bytes.fromhex(seal['source_cache_files'][cache_rel]),np.uint8)}
        name=row['id']+'.npz';np.savez_compressed(a.output/name,**z);files[name]=sha(a.output/name)
        assert fingerprint(model)==basefp
        trace={'id':row['id'],'q_source':q,'normal_source_cache':cache_rel,'normal_source_cache_sha256':seal['source_cache_files'][cache_rel],'normal_Conv_replay_bitexact':True,'same_normal_source_cache_no_fresh_backbone_or_slot':True,'groups':tracegroups,'full_source_model_restored_after_allGTcontrols':True,'GTyaw_not_consumed':True}
        traces.append(trace)
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
        if i%10==0 or i+1==len(targets):print(json.dumps({'phase':'gt_fit','completed':i+1,'n':len(targets),'elapsed_seconds':time.perf_counter()-started,'all_exactfit_coordinate_checks_pass_sofar':all_exactfit_pass}),flush=True)
    assert fingerprint(model)==basefp and all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules())
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    for n in ('trace.json','trace.jsonl','cache_replay.jsonl','registered_plan.json','GT_INDEX_READ_TRANSACTION.json'):files[n]=sha(a.output/n)
    outseal={'state':'sealed_R9_GTcontrolled_samecache_coordinate_capacity_diagnostic','phase':'gt_fit','n_samples':len(targets),'n_prediction_files':len(targets),'n_arms':13,'normal_prediction_seal_sha256':sha(seal_path),'normal_source_cache_files':seal['source_cache_files'],'target_index_sha256':plan['target_index_sha256'],'GT_index_release_sha256':plan['GT_index_read_release_sha256'],'plan_sha256':sha(a.plan),'source_checkpoint_sha256':plan['source_checkpoint_sha256'],'source_model_state_fingerprint':basefp,'strict_state_entries':entries,'strict_FP32_flags':STRICT_FLAGS,'program_sha256':plan['program_sha256'],'program_sha256_base':'workspace_root','all_exactfit_coordinate_checks_pass':bool(all_exactfit_pass),'all_query_indices_same_normal_q0':True,'no_GT_yaw_RGB_PC_G_or_backbone_forward_in_GTphase':True,'GTcontrolled_not_unlabelled_TTA_or_taskIoU_upperbound':True,'source_model_parameters_buffers_grads_modes_restored':True,'elapsed_seconds':time.perf_counter()-started,'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(outseal,indent=2)+'\n');print('R9_GTCONTROL_SEALED',sha(a.output/'SEALED.json'),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',choices=('normal','gt_fit'),required=True);ap.add_argument('--manifest',type=Path);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan);assert plan['phase']==a.phase and plan['GPU_execution_authorized'] is True and plan['batch_size']==1
    assert sha(ROOT/'research/round9/experiment/AUTHORIZATION.json')==plan['authorization_sha256']
    assert plan['expected_samples']==160 and plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    for rel,digest in plan['program_sha256'].items():
        p=ROOT/rel;assert p.resolve().is_relative_to(ROOT.resolve()) and sha(p)==digest
    assert not a.output.exists()
    if a.phase=='normal':normal(a,plan)
    else:gt_fit(a,plan)
if __name__=='__main__':main()
