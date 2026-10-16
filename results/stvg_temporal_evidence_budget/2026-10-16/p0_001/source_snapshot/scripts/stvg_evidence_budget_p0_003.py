"""Bounded, frozen-expert visual-budget experiment. Private inputs stay private.

All arms are prelocked before model execution. No annotation/score reads in
prepare, teacher, fit or readback. One hot teacher serves all four fixed arms.
"""
from pathlib import Path
import sys, os, json, time, hashlib, math, traceback, subprocess, threading
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
OLD_BASE=ROOT/'artifacts/stvg_temporal_evidence_budget_v1'
BASE=OLD_BASE/'revision002'
SEARCH=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1'
ARMS={'A':(8,448),'B':(16,320),'C':(32,224),'D':(8,448)}
UNIT='stvg-visual-evidence-p0-003.service'

def verify():
    r=read(BASE/'RUNTIME.json');assert r['status']=='locked' and r['GT_read'] is False
    for rel,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/rel)==h,rel
    recovery=read(BASE/'RECOVERY_RUNTIME003.json')
    for rel,h in recovery['pins'].items():assert sha(ROOT/rel)==h,rel
    return r

def select_roster():
    rows=read(SEARCH/'ROSTER_LOCK.json')['records']
    order=lambda r:(r['parent_sha256'],r['query_key_sha256'])
    selected=sorted([r for r in rows if r['dataset']=='hc2'],key=order)[:8]
    for question in (False,True):
        pool=[r for r in rows if r['dataset']=='vidstg' and ('interrog' in r['query_type'].lower() or 'question' in r['query_type'].lower())==question]
        assert len(pool)>=4
        selected+=sorted(pool,key=order)[:4]
    assert len({r['parent_sha256']for r in selected})==16
    confirm=read(ROOT/'artifacts/stvg_lite_event_support_v1/confirmation/ROSTER_LOCK.json')['records']
    assert not {r['parent_sha256']for r in selected}&{r['parent_sha256']for r in confirm}
    return selected

def sample_ids(case,candidate,available,n,feature=False):
    import numpy as np
    start=max(min(available),int(candidate['interval_frames'][0]));end=min(max(available)+1,int(candidate['interval_frames'][1]))
    # Actual unique support is preserved, without duplicating physical frames.
    actual=min(n,end-start)
    if actual<2:return [],dict(status='unknown_fewer_than_two_physical_frames')
    # Native video temporal patches pair frames. Avoid implicit processor padding.
    actual-=actual%2
    maxstride=max(1,(end-start-1)//(actual-1))
    default=[start+k*maxstride for k in range(actual)]
    if not feature:return default,dict(status='uniform',requested=n,actual=actual,stride=maxstride)
    hidden=[];ids=[]
    for h,rec in zip(case['temporal_hidden'],case['records']):
        hidden.extend(h.detach().double().cpu().numpy());ids.extend(rec['frame_ids'])
    at=np.argsort(ids);ids=np.asarray(ids)[at];hs=np.asarray(hidden)[at]
    assert len(np.unique(ids))==len(ids)
    norm=np.linalg.norm(hs,axis=1);norm=norm/(norm.max()+1e-12)
    unit=hs/(np.linalg.norm(hs,axis=1,keepdims=True)+1e-12)
    changes=np.r_[0,np.linalg.norm(np.diff(unit,axis=0),axis=1)];changes=changes/(changes.max()+1e-12)
    options=[]
    for stride in sorted({maxstride,max(1,maxstride-1),max(1,int(.8*maxstride))}):
        room=end-start-1-(actual-1)*stride
        for off in sorted({0,room//2,room}):
            f=[start+off+k*stride for k in range(actual)]
            coverage=(f[-1]-f[0])/max(1,end-start-1)
            if coverage<.75:continue
            score=.5*coverage+.3*float(np.interp(f,ids,changes).mean())+.2*float(np.interp(f,ids,norm).mean())
            options.append((score,f,stride,coverage))
    if not options:return default,dict(status='feature_uniform_family_short_support_fallback',requested=n,actual=actual,stride=maxstride)
    score,f,stride,coverage=max(options,key=lambda v:(v[0],-v[1][0],v[2]))
    return f,dict(status='feature_guided_uniform_family',requested=n,actual=actual,stride=stride,coverage=coverage,selection_score=score,
        feature_kind='existing_query_conditioned_frozen_temporal_hidden_norm_and_change_not_CLIP_similarity',
        weights=[.5,.3,.2],candidate_families=len(options),nonuniform_clock_not_approximated=True)

def render(case,candidate,request,fids,side):
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    fmap={int(r['physical_frame_id']):int(r['local_frame_index'])for r in request['frame_map']}
    target={fmap[f]:f for f in fids};cap=cv2.VideoCapture(request['video_path']);images={};pixels={};idx=0
    assert sha(Path(request['video_path']))==request['canonical_video_sha256']
    font=ImageFont.truetype('DejaVuSans.ttf',18)
    try:
        while target:
            ok,bgr=cap.read()
            if not ok:raise RuntimeError('video ended before selected physical frame')
            if idx in target:
                fid=target.pop(idx);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                ph=hashlib.sha256(rgb.tobytes()).hexdigest();prior=case['source_RGB'].get(fid,case['source_RGB'].get(str(fid)))
                if prior:assert ph==prior['RGB_sha256']
                pixels[str(fid)]=ph;im=Image.fromarray(rgb);w,h=im.size;scale=min(side/w,side/h);rw,rh=round(w*scale),round(h*scale);ox,oy=(side-rw)//2,(side-rh)//2
                canvas=Image.new('RGB',(side,side),(18,20,24));canvas.paste(im.resize((rw,rh)),(ox,oy))
                b=[float(np.interp(fid,case['frame_ids'],case['native_boxes_xyxy_normalized'][:,k]))for k in range(4)]
                rect=(ox+b[0]*rw,oy+b[1]*rh,ox+b[2]*rw,oy+b[3]*rh);d=ImageDraw.Draw(canvas);d.rectangle(rect,outline='red',width=2)
                d.text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),'1',anchor='mm',font=font,fill='red',stroke_width=1,stroke_fill='white');images[fid]=np.asarray(canvas)
            idx+=1
    finally:cap.release()
    frames=np.stack([images[f]for f in fids]);return frames.transpose(0,3,1,2),pixels,hashlib.sha256(frames.tobytes()).hexdigest()

def prepare():
    import numpy as np
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import run_stvg_table1_native_instance_adapter_v1 as canonical
    verify();sys.addaudithook(guard);rows=read(BASE/'ROSTER_LOCK.json')['records'];files=[];jobs=[]
    for rec in rows:
        cid=rec['probe_alias'];case,cp=load_case(rec);_,plan=source._validate_roster_record(rec)
        old_request=OLD_BASE/'requests'/(cid+'.json')
        request=read(old_request)if old_request.exists()else canonical.canonicalize(case,plan['input']['caption'],BASE/'canonical_cache')
        assert request['query_text']==plan['input']['caption'] and request['source_video_sha256']==case['input']['video_sha256']
        assert sha(Path(request['video_path']))==request['canonical_video_sha256']
        rp=BASE/'requests'/(cid+'.json');write(rp,request);available=[int(x['physical_frame_id'])for x in request['frame_map']]
        for arm,(n,side)in ARMS.items():
            for cand in case['temporal_candidates']:
                key=cand['canonical_id'];fids,selection=sample_ids(case,cand,available,n,arm=='D')
                old_meta=OLD_BASE/'inputs'/arm/cid/(key+'.json')
                if old_meta.exists():
                    old=read(old_meta)
                    assert old['source_case_sha256']==case['source_case_sha256'] and old['source_file_sha256']==sha(cp)
                    assert old['selection']==selection and old.get('sampled_physical_frame_ids',[])==fids
                    assert old['request_sha256']==sha(rp)
                    if old['status']=='ready':
                        assert sha(ROOT/old['input_path'])==old['input_sha256']
                        files.append(dict(path=old['input_path'],sha256=old['input_sha256']))
                    jobs.append(dict(case_id=cid,arm=arm,candidate_id=key,path=str(old_meta.relative_to(ROOT)),sha256=sha(old_meta)))
                    continue
                meta=dict(case_id=cid,dataset=rec['dataset'],candidate_id=key,interval_frames=cand['interval_frames'],arm=arm,requested_frames=n,requested_side=side,selection=selection,
                    source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),request_sha256=sha(rp),GT_read=False)
                if fids:
                    video,pixels,marked=render(case,cand,request,fids,side)
                    p=BASE/'inputs'/arm/cid/(key+'.npy');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,video,allow_pickle=False)
                    meta.update(status='ready',input_path=str(p.relative_to(ROOT)),input_sha256=sha(p),shape=list(video.shape),sampled_physical_frame_ids=fids,
                        actual_unique_frames=len(set(fids)),clip_fps=case['input']['fps']/(fids[1]-fids[0]),
                        physical_timestamps_s=[(f-case['video_bounds'][0])/case['input']['fps']for f in fids],pixel_sha256_by_frame=pixels,marked_RGB_sha256=marked)
                    files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
                else:meta.update(status='unknown_physical_input',actual_unique_frames=0)
                mp=BASE/'inputs'/arm/cid/(key+'.json');write(mp,meta);jobs.append(dict(case_id=cid,arm=arm,candidate_id=key,path=str(mp.relative_to(ROOT)),sha256=sha(mp)))
        write(BASE/'CANDIDATE_STATUS.json',dict(status='running',done=len({j['case_id']for j in jobs}),total=16,GT_read=False),mutable=True)
    write(BASE/'INPUT_BARRIER.json',dict(status='complete',count=16,jobs=jobs,files=files,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME.json')))
    write(BASE/'CANDIDATE_STATUS.json',dict(status='complete',count=16,done=16,total=16,GT_read=False),mutable=True)

def input_readback():
    import numpy as np
    verify();bar=read(BASE/'INPUT_BARRIER.json');assert bar['status']=='complete'and bar['count']==16
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    for f in bar['jobs']:
        assert sha(ROOT/f['path'])==f['sha256'];m=read(ROOT/f['path'])
        if m['status']!='ready':continue
        a=np.load(ROOT/m['input_path'],allow_pickle=False);ids=m['sampled_physical_frame_ids']
        assert list(a.shape)==m['shape'] and a.dtype==np.uint8 and len(ids)==len(set(ids))==a.shape[0]
        assert len(set(np.diff(ids)))==1 and all(int(m['interval_frames'][0])<=x<int(m['interval_frames'][1])for x in ids)
        assert hashlib.sha256(a.transpose(0,2,3,1).tobytes()).hexdigest()==m['marked_RGB_sha256']
    write(BASE/'ROOT_INPUT_READBACK.json',dict(status='pass',count=16,job_count=len(bar['jobs']),GT_read=False,CPU_only=True,input_barrier_sha256=sha(BASE/'INPUT_BARRIER.json')))

def teacher():
    import numpy as np,torch,gc,resource
    from transformers import AutoProcessor
    from scripts import run_stvg_native_instance_teacher_storage002 as loader
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    verify();sys.addaudithook(guard);assert read(BASE/'ROOT_INPUT_READBACK.json')['status']=='pass';torch.set_num_threads(2)
    loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
    _,model,torch,device,mapping,load_seconds,peak=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,BASE/'offload')
    model.generation_config.use_cache=True
    from scripts.stvg_qwen_uniform_clock002 import apply
    clock_proof=apply(model)
    write(BASE/'CLOCK_CORRECTION_READBACK.json',dict(status='pass',GT_read=False,proof=clock_proof))
    processor=AutoProcessor.from_pretrained(ROOT/loader.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False)
    ids=adapter._abc_token_ids(processor);files=[];qual=[];bar=read(BASE/'INPUT_BARRIER.json');started=time.perf_counter()
    for arm in ARMS:
        armfiles=[]
        for job in [j for j in bar['jobs']if j['arm']==arm]:
            m=read(ROOT/job['path']);key=m['case_id']+'_'+m['candidate_id'];dst=BASE/'teacher'/arm/(key+'.json')
            if m['status']!='ready':
                out=dict(status='unknown_physical_input',metadata=m,reward=None,GT_read=False)
            else:
                video=np.load(ROOT/m['input_path'],allow_pickle=False);tick=time.perf_counter()
                # Keep native processor/backend unchanged. Its actual integer
                # mRoPE time quantization is measured, not silently corrected.
                adapter.configure_original_processor(processor)
                for c in (processor.image_processor,processor.video_processor):c.max_pixels=m['requested_side']**2;c.size=dict(shortest_edge=3136,longest_edge=m['requested_side']**2)
                batch,ph=adapter._processor_batch(processor,adapter.caption_prompt(),video=video,fps=m['clip_fps'],device=device)
                grid=batch['video_grid_thw'].detach().cpu().tolist();sec=list(map(float,batch['second_per_grid_ts']))
                rope,delta=model.model.get_rope_index(input_ids=batch['input_ids'],video_grid_thw=batch['video_grid_thw'],second_per_grid_ts=batch['second_per_grid_ts'],attention_mask=batch.get('attention_mask'))
                mask=batch['input_ids'][0]==model.config.video_token_id;actual_t=rope[0,0,mask].detach().cpu().unique_consecutive().tolist()
                native_time_quantized=len(actual_t)<grid[0][0]
                q=dict(arm=arm,case_id=m['case_id'],candidate_id=m['candidate_id'],video_grid_thw=grid,second_per_grid_ts=sec,actual_video_temporal_positions=actual_t,
                    temporal_positions_collapsed=native_time_quantized,uniform_physical_frames=True,physical_timestamps_s=m['physical_timestamps_s'],actual_processed_visual_tokens=int(mask.sum()),input_tokens=int(batch['input_ids'].shape[1]),GT_read=False)
                qual.append(q);tokens=q['input_tokens'];torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():gen=model.generate(**batch,max_new_tokens=160,do_sample=False,use_cache=True)
                torch.cuda.synchronize();caption=processor.tokenizer.decode(gen[0,tokens:],skip_special_tokens=True).strip();outtokens=int(gen.shape[1]-tokens);cap_seconds=time.perf_counter()-tick
                cappeak=torch.cuda.max_memory_allocated();del batch,gen,rope,delta,video
                req=read(BASE/'requests'/(m['case_id']+'.json'));align,aph=adapter._processor_batch(processor,adapter.alignment_prompt(req['query_text'],caption),device=device);tick2=time.perf_counter()
                with torch.inference_mode():o=model(**align,use_cache=False);logits=o.logits[0,-1,ids].float();prob=torch.softmax(logits,0)
                torch.cuda.synchronize();out=dict(status='actual_frozen_teacher',candidate_id=m['candidate_id'],raw_caption=caption,caption_status='actual_plain_description'if caption else'empty_description',output_truncated_at_cap=outtokens>=160,
                    probabilities=prob.cpu().tolist(),label_logits=logits.cpu().tolist(),reward=float(prob[0]-prob[1])if caption else None,caption_prompt_sha256=ph,alignment_prompt_sha256=aph,
                    caption_input_tokens=tokens,caption_output_tokens=outtokens,alignment_input_tokens=int(align['input_ids'].shape[1]),caption_seconds=cap_seconds,alignment_seconds=time.perf_counter()-tick2,
                    GPU_peak_bytes=cappeak,CPU_process_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,qualification=q,metadata=m,GT_read=False,query_is_own_case=True)
                del align,o,logits,prob;gc.collect();torch.cuda.empty_cache()
            write(dst,out);item=dict(path=str(dst.relative_to(ROOT)),sha256=sha(dst));files.append(item);armfiles.append(item)
            write(BASE/'GPU_STATUS.json',dict(status='running',stage='teacher_'+arm,done=len(files),total=len(bar['jobs']),GT_read=False,time=time.time()),mutable=True)
            print('VISUAL_BUDGET',arm,len(files),len(bar['jobs']),flush=True)
        write(BASE/arm/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=16,response_count=len(armfiles),files=armfiles,GT_read=False))
    write(BASE/'TIME_ENCODING_QUALIFICATION.json',dict(status='complete',count=len(qual),records=qual,GT_read=False,
        native_backend_preserved_except_single_fractional_time_dtype_fix=True,D_restricted_to_uniform_sampling_families=True,nonuniform_timestamps_not_faked=True))
    write(BASE/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=16,response_count=len(files),files=files,model_loads=1,model_load_seconds=load_seconds,wall_seconds=time.perf_counter()-started,GT_read=False))

def fit_arm(arm,readback=False):
    import torch
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    from scripts.stvg_evidence_temporal_fit_v1 import fit
    from scripts.run_stvg_component_temporal005 import _exact_equal
    verify();sys.addaudithook(guard);torch.set_num_threads(2);rows=read(BASE/'ROSTER_LOCK.json')['records'];bar=read(BASE/arm/'TEACHER_GLOBAL_BARRIER.json')
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    files=[];fits=0
    for rec in rows:
        cid=rec['probe_alias'];case,cp=load_case(rec);responses=[];rewards=[]
        for cand in case['temporal_candidates']:
            p=BASE/'teacher'/arm/(cid+'_'+cand['canonical_id']+'.json');r=read(p);responses.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
            rewards.append(r['reward'])
        # Missing/duplicate original candidate pools remain Native fallback;
        # finer evidence does not silently add a missing temporal hypothesis.
        eligible=len(case['temporal_candidates'])==4 and all(x is not None for x in rewards)
        fitted=fit(case,adapter._hidden_from_source_case(case),case['temporal_candidates'],rewards,'offset_trust')if eligible else None
        interval=fitted['final_interval']if fitted else case['Native_interval'];fits+=int(fitted is not None)
        value=dict(case_id=cid,dataset=rec['dataset'],arm=arm,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),responses=responses,fit=fitted,
            interval=interval,GT_read=False,source_geometry_unchanged=True)
        path=BASE/arm/'student'/(cid+'.pt')
        if readback:
            assert _exact_equal(torch.load(path,map_location='cpu',weights_only=False),value)
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
        else:files.append(save_pt(path,value))
    if readback:
        pred=read(BASE/arm/'STUDENT_GLOBAL_BARRIER.json');assert pred['files']==files
        write(BASE/arm/'ROOT_READBACK.json',dict(status='pass',count=16,fit_exact_count=fits,GT_read=False,CPU_only=True,student_barrier_sha256=sha(BASE/arm/'STUDENT_GLOBAL_BARRIER.json')))
    else:write(BASE/arm/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=16,fit_count=fits,files=files,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME.json')))

def contracts():
    import numpy as np,torch
    c=dict(temporal_hidden=[torch.eye(8,256),torch.eye(8,256)],records=[dict(frame_ids=list(range(0,16,2))),dict(frame_ids=list(range(1,16,2)))])
    for n in (8,16,32):
        for length in (2,7,8,15,32,69):
            for feature in (False,True):
                ids,m=sample_ids(c,dict(interval_frames=[0,length]),list(range(length)),n,feature)
                assert len(ids)%2==0 and len(ids)==len(set(ids)) and max(ids)<length and len(set(np.diff(ids)))==1
    return dict(status='pass',count=36,CPU_only=True,GT_read=False,scope='uniform unique physical sampling, short clips and feature-uniform family')

def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='p0';g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
    assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    def call(stage,*args,gpuenv=False):
        write(BASE/'STATUS003.json',dict(status='running',stage=stage,GT_read=False,time=time.time()),mutable=True)
        p=BASE/'execution003'/stage;p.mkdir(parents=True,exist_ok=False)
        interp=ROOT/('.venv/stvg_native_distill_v1/bin/python'if gpuenv else'.conda/tubedetr/bin/python')
        with(p/'worker.log').open('xb')as f:
            child=subprocess.Popen([str(interp),'-B',str(Path(__file__)),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=f,stderr=subprocess.STDOUT)
            write(p/'LAUNCH.json',dict(status='running',worker_pid=child.pid,GT_read=False,time=time.time()));code=child.wait()
        write(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,GT_read=False,time=time.time()));assert code==0,stage
    try:
        call('input_readback','input_readback');call('teacher','teacher',gpuenv=True)
        for arm in ARMS:call('fit_'+arm,'fit',arm);call('readback_'+arm,'readback',arm)
        write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=16,arm_count=4,GT_read=False,
            bindings={str((BASE/a/n).relative_to(ROOT)):sha(BASE/a/n)for a in ARMS for n in ('ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json')}))
        write(BASE/'STATUS003.json',dict(status='pending_postseal_scoring_and_publication',stage='root_handoff',count=16,GT_read=False,time=time.time()),mutable=True)
    finally:stop.set()

if __name__=='__main__':
    try:
        command=sys.argv[1]
        if command=='contracts':print(json.dumps(contracts()))
        elif command=='fit':fit_arm(sys.argv[2])
        elif command=='readback':fit_arm(sys.argv[2],True)
        else:{'prepare':prepare,'input_readback':input_readback,'teacher':teacher,'supervise':supervise}[command]()
    except BaseException as e:
        write(BASE/'FAILURE003.json',dict(status='failed',stage=sys.argv[1],error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
