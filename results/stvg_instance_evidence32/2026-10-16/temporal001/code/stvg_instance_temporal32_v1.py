"""Matched complete Temporal-Lite clock correction on the locked development32.

Original sampling, rendering, prompts, 160-token cap and 512-param objective.
All feedback is private; exact corrected P0 inputs may be reused explicitly.
"""
from pathlib import Path
import copy,gc,hashlib,json,os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,OLD,LITE,SEARCH,read,write,put,sha,verify,guard
OUT=BASE/'temporal';UNIT='stvg-instance-evidence32-temporal-001.service'

def context(rec):
    import torch
    p=ROOT/rec['cached_temporal_path']if rec.get('cached_temporal_path')else SEARCH/'confirmation/temporal'/f"{rec['probe_alias']}.pt"
    d=torch.load(p,map_location='cpu',weights_only=False)
    return 2 if d.get('source_round1_sha256')is not None else {'libopenblas':20,'libscipy_openblas':20}

def prepare():
    import torch,numpy as np
    from threadpoolctl import threadpool_limits
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import run_stvg_table1_teacher_v1 as render
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    from scripts.stvg_temporal_lite_fit_v1 import fit
    from scripts.run_stvg_component_temporal005 import _exact_equal,_candidate_rewards
    verify();sys.addaudithook(guard);torch.set_num_threads(2);files=[];jobs=[];cases=[];baseline_exact=0
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];case,cp=load_case(rec);req=read(LITE/'reclocked_inputs/requests'/f'{cid}.json');_,plan=source._validate_roster_record(rec);assert req['query_text']==plan['input']['caption']
        put(OUT/'requests'/f'{cid}.json',req)
        legacy=LITE/'temporal/revision003/student'/f'{cid}.pt';d=torch.load(legacy,map_location='cpu',weights_only=False);assert not d['GT_read']and d['source_case_sha256']==case['source_case_sha256']
        eligible,reason=render.temporal_clip_eligibility(case,req);profile=context(rec)
        if eligible:
            oldfit=d['fits']['temporal_lite'];oldrewards,_=_candidate_rewards(case,oldfit,None)
            with threadpool_limits(limits=profile,user_api='blas'):
                repeat=fit(case,adapter._hidden_from_source_case(case),case['temporal_candidates'],oldrewards,'temporal_lite')
            assert _exact_equal(repeat,oldfit),cid;baseline_exact+=1
            rgb=render.rgb_for_candidates(case,req,case['temporal_candidates']);clips=adapter.prepare_candidate_clips(case,case['temporal_candidates'],decoded_rgb_by_physical_id=rgb,frame_map=req['frame_map'])
            for cand in case['temporal_candidates']:
                key=cand['canonical_id'];clip=clips[key];p=OUT/'inputs'/cid/(key+'.npy');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,clip['video'],allow_pickle=False)
                meta=dict(case_id=cid,dataset=rec['dataset'],candidate=copy.deepcopy(cand),metadata=clip['metadata'],input_path=str(p.relative_to(ROOT)),input_sha256=sha(p),source_file_sha256=sha(cp),source_case_sha256=case['source_case_sha256'],GT_read=False)
                mp=OUT/'inputs'/cid/(key+'.json');put(mp,meta);jobs.append(dict(case_id=cid,path=str(mp.relative_to(ROOT)),sha256=sha(mp)));files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
            del rgb,clips
        cases.append(dict(case_id=cid,dataset=rec['dataset'],eligible=eligible,fallback_reason=reason,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),legacy_interval=d['arms']['TS_temporal_lite']['interval'],legacy_file_sha256=sha(legacy),cpu_blas_profile=profile,GT_read=False))
        print('TEMP_INPUT',len(cases),32,cid,eligible,flush=True);del case,d;gc.collect()
    put(OUT/'INPUT_BARRIER.json',dict(status='complete',count=32,records=cases,jobs=jobs,files=files,legacy_Lite_fit_exact=baseline_exact,GT_read=False))
    pins=copy.deepcopy(read(BASE/'RUNTIME.json')['pins'])
    for p in [Path(__file__),OUT/'INPUT_BARRIER.json',ROOT/'scripts/run_stvg_native_instance_teacher_storage002.py',ROOT/'scripts/stvg_temporal_lite_fit_v1.py',ROOT/'scripts/stvg_table1_temporal_adapter_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    put(OUT/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False,scope='same 8x448/FPS/Prompt/target; corrected Temporal time only; exact inherited CPU lineage'))

def verify_temporal():
    verify()
    for rel,h in read(OUT/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel

def p0_reuse(meta,query,processor):
    import numpy as np
    from scripts import stvg_table1_temporal_adapter_v1 as a
    p=ROOT/'artifacts/stvg_temporal_evidence_budget_v1/revision002/teacher/A'/f"{meta['case_id']}_{meta['candidate']['canonical_id']}.json"
    if not p.exists():return None
    d=read(p);m=d['metadata'];n=meta['metadata'];key=meta['candidate']['canonical_id']
    if not(m['source_case_sha256']==meta['source_case_sha256']and m['candidate_id']==key and m.get('sampled_physical_frame_ids')==n['sampled_physical_frame_ids']and m.get('marked_RGB_sha256')==n['marked_RGB_sha256']and m.get('clip_fps')==n['clip_fps']and m.get('requested_side')==448):return None
    req=read(ROOT/'artifacts/stvg_temporal_evidence_budget_v1/revision002/requests'/f"{meta['case_id']}.json")
    if req['query_text']!=query:return None
    # Hash the complete token-template strings and check the original output cap.
    _,ch=a._processor_batch(processor,a.caption_prompt(),video=np.load(ROOT/meta['input_path'],allow_pickle=False),fps=n['clip_fps'])
    _,ah=a._processor_batch(processor,a.alignment_prompt(query,d['raw_caption']))
    if ch!=d['caption_prompt_sha256']or ah!=d['alignment_prompt_sha256']:return None
    assert d['qualification']['arm']=='A'and d['GT_read']is False
    return dict(d,reused_corrected_P0_response=True,reused_source_sha256=sha(p),reuse_basis='exact 8x448 physical sampling, marked RGB, FPS, source, own Query, complete prompt templates, corrected clock and 160-token cap')

def teacher():
    import torch,numpy as np
    from transformers import AutoProcessor
    from scripts import run_stvg_native_instance_teacher_storage002 as loader
    from scripts import stvg_table1_temporal_adapter_v1 as a
    from scripts.stvg_qwen_uniform_clock002 import apply
    from scripts.stvg_reference_clock_control_v1 import qualify
    verify_temporal();sys.addaudithook(guard);torch.set_num_threads(2);loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
    _,model,_,device,_,load_seconds,_=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,OUT/'offload');model.generation_config.use_cache=True;proof=apply(model)
    processor=AutoProcessor.from_pretrained(ROOT/loader.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False);a.configure_original_processor(processor)
    files=[];fresh=reuse=0;start=time.perf_counter()
    for j in read(OUT/'INPUT_BARRIER.json')['jobs']:
        assert sha(ROOT/j['path'])==j['sha256'];m=read(ROOT/j['path']);assert sha(ROOT/m['input_path'])==m['input_sha256'];cid=m['case_id'];query=read(OUT/'requests'/f'{cid}.json')['query_text'];cached=p0_reuse(m,query,processor)
        if cached is not None:response=cached;reuse+=1
        else:
            video=np.load(ROOT/m['input_path'],allow_pickle=False);clip=dict(video=video,fps=m['metadata']['clip_fps'],metadata=m['metadata'])
            batch,_=a._processor_batch(processor,a.caption_prompt(),video=video,fps=clip['fps'],device=device);q=qualify(model,batch,dict(job_id=cid+'_'+m['candidate']['canonical_id']));del batch
            response=a.run_hot_candidate_teacher(model,processor,query,m['candidate'],clip,device=device);response['actual_time_encoding_readback']=q;fresh+=1
        response.update(case_id=cid,source_case_sha256=m['source_case_sha256'],corrected_clock_proof=proof,new_teacher_pair=cached is None,source_input_sha256=j['sha256'])
        p=OUT/'teacher'/cid/(m['candidate']['canonical_id']+'.json');put(p,response);files.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p)))
        write(OUT/'STATUS.json',dict(status='running',stage='Temporal_teacher',done=len(files),total=len(read(OUT/'INPUT_BARRIER.json')['jobs']),GT_read=False),mutable=True);torch.cuda.empty_cache()
    put(OUT/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,response_count=len(files),files=files,new_caption_alignment_pairs=fresh,exact_P0_reused_pairs=reuse,model_load_seconds=load_seconds,wall_seconds=time.perf_counter()-start,proof=proof,GT_read=False))

def fit_cases(readback=False):
    import torch
    from threadpoolctl import threadpool_limits
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts import stvg_table1_temporal_adapter_v1 as a
    from scripts.stvg_temporal_lite_fit_v1 import fit
    from scripts.run_stvg_component_temporal005 import _exact_equal
    verify_temporal();sys.addaudithook(guard);torch.set_num_threads(2);bar=read(OUT/'TEACHER_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==32
    inputs={r['case_id']:r for r in read(OUT/'INPUT_BARRIER.json')['records']};responses={f['path']:f['sha256']for f in bar['files']};files=[];times=[];nfit=0
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];case,cp=load_case(rec);inp=inputs[cid];fitdoc=None;reward=[]
        if inp['eligible']:
            for c in case['temporal_candidates']:
                p=OUT/'teacher'/cid/(c['canonical_id']+'.json');assert sha(p)==responses[str(p.relative_to(ROOT))];reward.append(a._alignment_reward(c['canonical_id'],read(p)))
            with threadpool_limits(limits=inp['cpu_blas_profile'],user_api='blas'):fitdoc=fit(case,a._hidden_from_source_case(case),case['temporal_candidates'],reward,'temporal_lite')
            interval=list(fitdoc['final_interval']);nfit+=1
        else:interval=list(case['Native_interval'])
        payload=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),fit=fitdoc,rewards=reward,selected_interval=interval,legacy_interval=inp['legacy_interval'],no_fit_reason=inp['fallback_reason']if not inp['eligible']else None,GT_read=False)
        p=OUT/'student'/f'{cid}.pt'
        if readback:assert _exact_equal(torch.load(p,map_location='cpu',weights_only=False),payload);item=dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
        else:item=save_pt(p,payload);item['case_id']=cid
        files.append(item);times.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],selected_interval=interval,GT_read=False));del case;gc.collect()
    if readback:
        assert read(OUT/'STUDENT_GLOBAL_BARRIER.json')['files']==files;put(OUT/'ROOT_READBACK.json',dict(status='pass',count=32,fit_exact_count=nfit,no_fit_count=32-nfit,GT_read=False,barrier_sha256=sha(OUT/'STUDENT_GLOBAL_BARRIER.json')))
        put(OUT/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,root_sha256=sha(OUT/'ROOT_READBACK.json'),GT_read=False))
    else:
        put(OUT/'TEMPORAL_LOCK.json',dict(status='locked',count=32,records=times,GT_read=False));put(OUT/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,fit_count=nfit,no_fit_count=32-nfit,GT_read=False))

def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify_temporal();resource.BASE=OUT;resource.UNIT=UNIT;resource.STAGE='temporal';g=1024**3
    v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for mode in ('teacher','fit','readback'):
            p=OUT/'execution'/mode;p.mkdir(parents=True,exist_ok=False)
            interp=ROOT/('.venv/stvg_native_distill_v1/bin/python'if mode=='teacher'else'.conda/tubedetr/bin/python')
            with(p/'worker.log').open('xb')as log:
                t=time.perf_counter();c=subprocess.Popen([str(interp),'-B',str(Path(__file__)),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False,time=time.time()));code=c.wait()
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=False,time=time.time()));assert code==0,mode
    finally:stop.set()
if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode=='fit':fit_cases()
        elif mode=='readback':fit_cases(True)
        else:{'prepare':prepare,'teacher':teacher,'supervise':supervise}[mode]()
    except BaseException as e:
        put(OUT/('FAILURE_'+sys.argv[1]+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
