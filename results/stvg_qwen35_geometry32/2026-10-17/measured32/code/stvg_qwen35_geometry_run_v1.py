"""Authorized serial Qwen geometry32 execution. No inference occurs on import."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen35_geometry32_v1 import BASE,CHECKPOINT,PREVIOUS,COMPLEMENT,read,sha,write,canonical_sha,execution_gate,download_status
from scripts.stvg_qwen35_geometry_contract_v1 import parse_final,original_pixel_box

def verify_run():
    r=read(BASE/'EXECUTION_RUNTIME001.json')
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    active=BASE/'ACTIVE_EXECUTION.json'
    if active.exists():assert time.time()<read(active)['deadline_epoch'],'scientific wall budget expired'
    return r

def model_snapshot():
    verify_run();execution_gate('interface_qualification');assert download_status()['snapshot_for_model_ready']
    expected=read(CHECKPOINT/'.download-control/expected_side001.json')
    # Downloader already independently streams every LFS SHA. Bind all file sizes and small metadata here.
    rows=[]
    for f in expected['files']:
        p=CHECKPOINT/f['name'];assert p.stat().st_size==f['bytes']
        rows.append(dict(name=f['name'],bytes=f['bytes'],sha256=f['sha256'] or sha(p),hash_source='downloader_verified_LFS' if f['sha256'] else 'local_metadata_readback'))
    write(BASE/'MODEL_SNAPSHOT_LOCK.json',dict(status='locked',model=expected['repo'],revision=expected['revision'],files=rows,
        chat_template_sha256=sha(CHECKPOINT/'chat_template.jinja'),environment_sha256=sha(BASE/'ENVIRONMENT_LOCK.json'),GT_read=False))

def native_video_adapter(processor,text,image,video,context):
    """Native Qwen3-VL processor inherited by Qwen3.5, explicit four sampled frames."""
    import numpy as np,torch
    from transformers.video_utils import VideoMetadata
    stamps=list(map(float,context['actual_timestamps_seconds']))
    assert len(stamps)==4 and all(b>a for a,b in zip(stamps,stamps[1:])),stamps
    # This input has already been sampled by the historical context builder. Preserve its
    # disclosed uniform clock approximation rather than inventing physical source-frame spacing.
    fps=float(context['fps']);assert fps>0 and context['encoder_uniform_spacing_approximation'] is True
    relative=[i/fps for i in range(4)]
    metadata=VideoMetadata(total_num_frames=4,fps=fps,frames_indices=[0,1,2,3],duration=4/fps)
    opts=dict(do_sample_frames=False,video_metadata=[metadata],return_metadata=True)
    batch=processor(text=[text],images=[image],videos=[torch.from_numpy(video.copy())],return_tensors='pt',**opts)
    # Independent image-only processor must produce precisely the anchor block in combined input.
    solo=processor.image_processor(images=[image],return_tensors='pt')
    assert torch.equal(batch['pixel_values'],solo['pixel_values']) and torch.equal(batch['image_grid_thw'],solo['image_grid_thw'])
    grid=batch['image_grid_thw'][0].tolist();patch=processor.image_processor.patch_size;merge=processor.image_processor.merge_size
    assert grid[0]==1 and patch==16 and merge==2
    assert not getattr(processor.image_processor,'do_center_crop',False) and not getattr(processor.image_processor,'do_pad',False)
    H,W=int(grid[1])*patch,int(grid[2])*patch
    ids=batch['input_ids'][0];assert int((ids==248056).sum())==int(np.prod(grid))//merge**2
    vgrid=batch['video_grid_thw'][0].tolist();assert vgrid[0]==2
    assert int((ids==248057).sum())==int(np.prod(vgrid))//merge**2
    assert int((ids==248057).nonzero().max())<int((ids==248056).nonzero().min()),'context must precede the distinct anchor image'
    decoded=processor.tokenizer.decode(ids.tolist(),skip_special_tokens=False)
    # Processor stores paired-frame timestamps in the actual prompt; check their native formatting.
    expected_t=[sum(relative[i:i+2])/2 for i in [0,2]]
    for t in expected_t:assert f'<{t:.1f} seconds>' in decoded,'native video timestamp missing'
    audit=dict(anchor_image_identity_verified=True,native_video_timestamps_verified=True,resize_crop_pad_transform_verified=True,
        original_width=image.width,original_height=image.height,processed_width=W,processed_height=H,
        no_crop_or_pad=True,image_grid_thw=grid,video_grid_thw=vgrid,patch_size=patch,merge_size=merge,
        anchor_image_only_tensor_exact=True,source_timestamps_seconds=stamps,relative_timestamps_seconds=relative,
        historical_uniform_clock_approximation_retained=True,coordinate_protocol='xyxy_normalized_0_1000',
        input_tokens=int(ids.numel()),image_tokens=int((ids==248056).sum()),video_tokens=int((ids==248057).sum()),
        input_tensor_sha256={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in batch.items() if torch.is_tensor(v)},GT_read=False)
    # Processor-only metadata is not a model forward kwarg.
    batch.pop('video_metadata',None)
    return batch,audit

def request(model,processor,job,qualification=False):
    import torch
    from scripts.stvg_qwen35_geometry_adapters_v1 import prepare_native_batch,claim_formal_request
    verify_run()
    if qualification:
        execution_gate('interface_qualification');assert job['job_id'] in read(BASE/'QUALIFICATION_PLAN.json')['fixed_parent_probe_jobs']
        p=BASE/'formal_claims'/(job['job_id']+'.json');p.parent.mkdir(exist_ok=True)
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(dict(status='claimed_once',job_id=job['job_id'],input_lock_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),qualification_probe=True,GT_read=False),f)
    else:claim_formal_request(job['job_id'])
    batch,audit=prepare_native_batch(processor,job,verified_video_adapter=native_video_adapter)
    n=int(batch['input_ids'].shape[1]);batch=batch.to(model.device);tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():out=model.generate(**batch,max_new_tokens=256,do_sample=False,use_cache=True)
    torch.cuda.synchronize();tokens=out[0,n:].cpu().tolist();raw=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    truncated=len(tokens)>=256 and (not tokens or tokens[-1] not in (eos or []))
    parsed=parse_final(raw,job['job_id'],truncated=truncated)
    d=dict(status='complete',job_id=job['job_id'],case_id=job['case_id'],dataset=job['dataset'],slot=job['slot'],
        input_job_sha256=canonical_sha(job),model_snapshot_sha256=sha(BASE/'MODEL_SNAPSHOT_LOCK.json'),prompt_lock_sha256=sha(BASE/'PROMPT_LOCK.json'),
        actual_model_call=True,raw_final_output=raw,output_tokens=len(tokens),truncated=truncated,parsed=parsed,input_audit=audit,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-tick,GT_read=False)
    write(BASE/'formal_responses'/(job['job_id']+'.json'),d)
    del batch,out
    return d

def teacher():
    import torch
    from scripts.stvg_qwen35_geometry_adapters_v1 import load_future_model
    verify_run();model_snapshot();model,processor=load_future_model();params={str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    assert 'torch.bfloat16' in params and params<={'torch.bfloat16','torch.float32'}
    jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];byid={j['job_id']:j for j in jobs};probes=read(BASE/'QUALIFICATION_PLAN.json')['fixed_parent_probe_jobs'];proof=[]
    for job_id in probes:
        d=request(model,processor,byid[job_id],qualification=True);proof.append(d)
        assert d['parsed']['status'] in ['valid','abstain','no_target'],'interface probe returned non-protocol/truncated output; no retry'
    # Deterministic non-square coordinate conversion checks use exact processed geometry,
    # do not ask the model to recognize a colored square or use GT.
    for d in proof:
        a=d['input_audit'];b=[.35,.5,.5,.6];px=original_pixel_box(b,a['original_width'],a['original_height'])
        assert abs(px[0]-.1*a['original_width'])<1e-7 and abs(px[3]-.8*a['original_height'])<1e-7
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.UNIT='stvg-qwen35-geometry32-001.service';sample=resource.sample()
    assert sample['gpu_free']>=4*2**30 and sample['available']>=6*2**30 and sample['disk_free']>=30*2**30
    assert sample['current']<23*2**30 and sample['anon']<21*2**30
    write(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=4,input_lock_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),
        model_snapshot_sha256=sha(BASE/'MODEL_SNAPSHOT_LOCK.json'),chat_template_sha256=sha(CHECKPOINT/'chat_template.jinja'),prompt_lock_sha256=sha(BASE/'PROMPT_LOCK.json'),
        same_native_context_and_anchor_pixels=True,anchor_image_identity_verified=True,resize_crop_pad_transform_verified=True,
        normalized_coordinate_mapping_verified=True,complete_final_answer_verified=True,enable_thinking_false_via_native_template=True,
        actual_BF16_model_verified=True,parameter_dtypes=sorted(params),resource_guard_pass=True,no_GT_used=True,
        native_processor_class=type(processor).__name__,model_class=type(model).__name__,real_parent_requests=4,extra_model_coordinate_probe_requests=0,
        resource_sample=sample,live_resource_limits_sha256=sha(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json'),
        probe_jobs=probes,probe_files={j:sha(BASE/'formal_responses'/(j+'.json')) for j in probes},GT_read=False))
    for j in jobs:
        if j['job_id'] not in probes:request(model,processor,j)
        done=len(list((BASE/'formal_responses').glob('*.json')))
        write(BASE/'TEACHER_STATUS.json',dict(status='running',done=done,total=128,count=32,GT_read=False),mutable=True)
    write(BASE/'TEACHER_COMPLETION.json',dict(status='complete',count=32,formal_requests=128,actual_new_generations=128,GT_read=False))

def score():
    import numpy as np,torch
    from scripts.stvg_qwen35_geometry_adapters_v1 import require_prediction_seals_before_score
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    verify_run();new=require_prediction_seals_before_score();assert read(BASE/'D2_ROOT_READBACK.json')['status']=='pass'
    bars={'R1_DIRECT':read(PREVIOUS/'arms/DIRECT/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
          'R1_HEADS':read(PREVIOUS/'arms/HEADS/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
          'D0':read(PREVIOUS/'arms/DINO/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
          'D1':read(COMPLEMENT/'arms/D1/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),
          'D2':read(COMPLEMENT/'D2_GLOBAL_PREDICTION_BARRIER.json'), 'QWEN_OPD':new['OPD'],'QWEN_D2':new['D2']}
    assert all(b['status']=='complete' and b['count']==32 for b in bars.values())
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,new_arms=['QWEN_OPD','QWEN_D2'],all_prediction_files_sealed=True,GT_read=False))
    maps={a:{r['case_id']:r for r in b['files']} for a,b in bars.items()};rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr}
    targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();bm={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']}
    jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];refs={r['case_id']:{a['position']:a for a in r['anchors']} for r in read(BASE/'QWEN_REFERENCE_LOCK.json')['records']};rows=[];quality=[];usage=[]
    def iou(a,b):
        inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return float(inter/max(1e-12,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter))
    oldrows={(x['case_id'],x['arm']):x['metrics'] for x in read(COMPLEMENT/'POSTSEAL_RESULTS.json')['rows']}
    for x in read(PREVIOUS/'POSTSEAL_RESULTS.json')['rows']:
        if x['arm'] in ['DIRECT','HEADS']:oldrows[x['case_id'],'R1_'+x['arm']]=x['metrics']
    for r in rr:
        cid=r['probe_alias'];case=cases[cid];f=bm[cid];assert sha(ROOT/f['path'])==f['sha256'];old=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
        preds={'Frozen':old['arms']['F'],'T2':old['arms']['T'],'Current_WF':old['arms']['TS_M2']};support={j['position'] for j in jobs if j['case_id']==cid}
        for a,m in maps.items():
            f=m[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds[a]=d['arm'] if a in ['D2','QWEN_D2'] else d['arms']['TS_M2']
            if a=='QWEN_OPD':
                fit=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=r['dataset'],actual_fit=fit is not None,reference_count=len(refs[cid]),accepted_steps=fit['optimizer_steps'] if fit else 0,proposed_steps=fit['proposed_optimizer_steps'] if fit else 0))
        for a,p in preds.items():
            v,_=metric._score_student_arm(p,case,targets[cid],support,off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5))
            if (cid,a) in oldrows:assert v==oldrows[cid,a],'matched old metric drift'
            rows.append(dict(case_id=cid,dataset=r['dataset'],arm=a,metrics=v));assert a=='Frozen' or p['interval']==preds['T2']['interval']
        gt=metric._truth_map(targets[cid]);scale=np.array([case['input']['width'],case['input']['height']]*2,float)
        for j in [j for j in jobs if j['case_id']==cid]:
            a=refs[cid].get(j['position']);known=j['physical_frame_id'] in gt;value=None
            if a is not None and known:
                x,y,w,h=a['box_cxcywh'];value=iou(np.array([x-w/2,y-h/2,x+w/2,y+h/2])*scale,np.asarray(gt[j['physical_frame_id']]))
            quality.append(dict(case_id=cid,dataset=r['dataset'],job_id=j['job_id'],GT_known=known,reference_available=a is not None,
                conditional_IoU=value,success_at_0p5=None if not known else bool(value is not None and value>=.5)))
    summary,paired=subset_aggregate(rows)
    write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=usage,reference_quality=quality,
        GT_bindings=bindings,GT_read=True,development_not_confirmation=True,all_no_fit_and_extreme_sources_retained=True,
        matched_old_reference_quality=dict(R1_and_DINO=read(PREVIOUS/'POSTSEAL_RESULTS.json')['reference_quality'],D1=read(COMPLEMENT/'POSTSEAL_RESULTS.json')['reference_quality']),
        no_automatic_significance_or_method_promotion_gate=True))
    write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))

def main(stage):
    from scripts import stvg_qwen35_geometry_adapters_v1 as a
    if stage=='teacher':return teacher()
    if stage=='reference_root':return a.seal_references()
    if stage=='student_prepare':return a.prepare_student()
    if stage in ['student','student_root']:return a.student('worker' if stage=='student' else 'root')
    if stage=='propagate':return a.propagate()
    if stage=='score':return score()
    raise ValueError(stage)
if __name__=='__main__':main(sys.argv[1])
