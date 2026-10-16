"""Bounded matched global-context study after six fixed256 core groups."""
from pathlib import Path
import copy,gc,hashlib,inspect,json,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_global32_context_v1 import BASE,CORE,read,put,sha
from scripts.stvg_native_instance_common_v1 import write,guard
UNIT='stvg-global32-context-001.service'
def verify():
    from scripts.prepare_stvg_global32_context_v1 import verify as prep_verify
    prep_verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    from scripts.continue_stvg_spatial_controls256_native003 import verify as core_verify
    core_verify()
def lock():
    files=[Path(__file__),ROOT/'scripts/prepare_stvg_global32_context_v1.py',ROOT/'scripts/stvg_global32_native_v1.py',ROOT/'scripts/run_stvg_global32_student_v1.py',ROOT/'scripts/score_stvg_global32_context_v1.py',BASE/'PREPARATION_RUNTIME.json',CORE/'NATIVE_DISPATCH_RUNTIME003.json',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/run_stvg_spatial_controls256_student_v1.py',ROOT/'scripts/score_stvg_spatial_controls256_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',ROOT/'scripts/run_stvg_referent_abc002.py']
    put(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=32,GT_read=False,pins={str(p.relative_to(ROOT)):sha(p) for p in files},waits_for_six_core256_measurements=True,does_not_read_core_scores_to_choose_protocol=True))
def load():
    from scripts import run_stvg_referent_abc002 as loader
    loader.BASE=BASE;return loader.load_teacher()
def core_cache(j):
    import numpy as np
    p=CORE/'native_cache'/j['job_id'];s=read(p/'POLICIES.json')
    assert sha(p/'native.npz')==s['native_sha256'] and s['job_sha256']==j['original_job_sha256']
    with np.load(p/'native.npz',allow_pickle=False) as a:
        f=a['features'][512:].copy();q=a['query_embedding'].copy()
    assert f.shape[0]==768 and len(s['records'][1]['indices']['A'])==96
    return f,q,s
def harray(a):return hashlib.sha256(a.tobytes()).hexdigest()
def global_cache(model,batch,j,q,dims,proof):
    import numpy as np,torch
    from scripts.stvg_global32_native_v1 import global_geometry,select_A_only
    b=BASE/'global_features'/j['case_id'];tick=time.perf_counter();fresh=False
    pixels=batch['pixel_values_videos'][:16*32*32];grid=batch['video_grid_thw'][:1]
    binding=dict(input_sha256=j['videos'][0]['sha256'],tensor_sha256=j['videos'][0]['tensor_sha256'],fps=j['videos'][0]['fps'],grid=grid.cpu().tolist(),pixel_values_sha256=harray(pixels.float().cpu().numpy()),query_embedding_sha256=harray(q),clock_proof=proof,checkpoint_config_sha256=sha(ROOT/'checkpoints/STVG-R1-7B/config.json'))
    if not(b/'GLOBAL_FEATURE_LOCK.json').exists():
        b.mkdir(parents=True,exist_ok=False)
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():f=model.model.get_video_features(pixels,grid)
        torch.cuda.synchronize();encoding=time.perf_counter()-t;z=f.float().cpu().numpy();assert z.shape[0]==4096
        v,ma,gr=global_geometry(dims['width'],dims['height']);policy=select_A_only(z.reshape(16,256,-1),q,v,ma,gr);assert policy['total_tokens_kept']==512
        np.savez_compressed(b/'global.npz',features=z)
        put(b/'GLOBAL_FEATURE_LOCK.json',dict(status='locked',binding=binding,feature_sha256=sha(b/'global.npz'),feature_tensor_sha256=harray(z),indices=policy['indices'].tolist(),quotas=policy['quotas'],role_quotas=policy['role_quotas'],encoder_seconds=encoding,selection_and_store_seconds=time.perf_counter()-tick-encoding,GT_read=False));fresh=True
    else:
        d=read(b/'GLOBAL_FEATURE_LOCK.json');assert d['binding']==binding and sha(b/'global.npz')==d['feature_sha256']
        with np.load(b/'global.npz',allow_pickle=False) as a:z=a['features'].copy()
        assert harray(z)==d['feature_tensor_sha256'];f=torch.from_numpy(z).to(device=batch['input_ids'].device,dtype=model.dtype)
    d=read(b/'GLOBAL_FEATURE_LOCK.json');return f,d,fresh,time.perf_counter()-tick
def timed_greedy(m,e,pos,keep,ids,cap=96,check_recompute=False):
    import torch
    from scripts.stvg_global32_native_v1 import greedy
    calls=[];forward=m.forward
    def measured(*a,**kw):
        torch.cuda.synchronize();t=time.perf_counter();out=forward(*a,**kw);torch.cuda.synchronize();calls.append(time.perf_counter()-t);return out
    m.forward=measured
    try:tokens,audit=greedy(m,e,pos,keep,ids,cap,check_recompute=check_recompute)
    finally:m.forward=forward
    audit.update(prefill_seconds=calls[0],generation_forward_seconds=sum(calls[1:]),forward_call_count=len(calls),qualification_includes_recompute=check_recompute)
    return tokens,audit
def qualification():
    import torch,numpy as np
    from scripts.stvg_global32_native_v1 import processor_batch,pack,sequence_keep
    verify();sys.addaudithook(guard);m,p,device,proof,secs=load();rows=[];jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];dims=read(CORE/'SOURCE_GEOMETRY.json')['records']
    for ds in ('hc2','vidstg'):
        candidates=[j for j in jobs if j['dataset']==ds]
        if not candidates:
            rows.append(dict(dataset=ds,status='unavailable_no_eligible_job',GT_read=False));continue
        j=candidates[0];batch,ps=processor_batch(p,j,ROOT,device);local,q,cp=core_cache(j)
        assert cp['model_clock_proof']==proof
        with torch.inference_mode():
            f,lock,fresh,_=global_cache(m,batch,j,q,dims[j['case_id']],proof)
            assert fresh
            torch.cuda.synchronize();t=time.perf_counter();e,pos,delta,joint,runs=pack(m,batch);torch.cuda.synchronize();jointsecs=time.perf_counter()-t
            assert torch.equal(f,joint[:4096]) and torch.equal(torch.from_numpy(local).to(device=device,dtype=m.dtype),joint[4096:])
            m.model.rope_deltas=None;official=m.generate(**batch,max_new_tokens=8,do_sample=False,use_cache=True)[0,batch['input_ids'].shape[1]:].cpu().tolist()
            fullkeep=sequence_keep(batch,m,runs,[range(4096),range(768)]);ours,fullaudit=timed_greedy(m,e,pos,fullkeep,batch['input_ids'],8);assert official==ours
            ix=[lock['indices'],cp['records'][1]['indices']['A']];keep=sequence_keep(batch,m,runs,ix);assert len(ix[0])==512 and len(ix[1])==96
            _,trace=timed_greedy(m,e,pos,keep,batch['input_ids'],8,check_recompute=True);assert trace['cached_uncached_next_token_checks']>=2
        rows.append(dict(dataset=ds,status='pass',case_id=j['case_id'],job_id=j['job_id'],global_isolated_joint_exact=True,core_local_joint_exact=True,native_first8_exact=True,compressed_cached_recompute_exact=True,raw_visual_tokens=4864,kept_visual_tokens=608,global_cache_new_encoder_calls=1,extra_joint_qualification_encoder_calls=1,original_generate_extra_encoder_calls=1,joint_encoding_pack_seconds=jointsecs,pruned_trace=trace,global_feature_lock_sha256=sha(BASE/'global_features'/j['case_id']/'GLOBAL_FEATURE_LOCK.json'),GT_read=False))
        del batch,e,pos,joint,f;gc.collect();torch.cuda.empty_cache()
    assert all(r['status'] in ('pass','unavailable_no_eligible_job') for r in rows)
    put(BASE/'NATIVE_QUALIFICATION.json',dict(status='pass',count=32,qualified_domains=sum(r['status']=='pass' for r in rows),records=rows,model_load_seconds=secs,clock_proof=proof,GT_read=False))
def teacher():
    import torch,numpy as np
    from scripts.stvg_global32_native_v1 import processor_batch,pack,sequence_keep
    from scripts.run_stvg_spatial_who_order_v1 import parse
    verify();assert read(BASE/'NATIVE_QUALIFICATION.json')['status']=='pass';sys.addaudithook(guard)
    m,p,device,proof,secs=load();dims=read(CORE/'SOURCE_GEOMETRY.json')['records'];files=[];start=time.perf_counter();newenc=0;jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records']
    for j in jobs:
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();batch,ps=processor_batch(p,j,ROOT,device);processing=time.perf_counter()-tick;local,q,cp=core_cache(j);assert cp['model_clock_proof']==proof
        f,lock,fresh,cache_seconds=global_cache(m,batch,j,q,dims[j['case_id']],proof);newenc+=int(fresh)
        with torch.inference_mode():
            features=torch.cat((f,torch.from_numpy(local).to(device=device,dtype=m.dtype)),0);e,pos,delta,features,runs=pack(m,batch,features)
            ix=[lock['indices'],cp['records'][1]['indices']['A']];assert len(ix[0])==512 and len(ix[1])==96
            keep=sequence_keep(batch,m,runs,ix);tokens,audit=timed_greedy(m,e,pos,keep,batch['input_ids'],96)
        raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();dest=BASE/'teacher/G32'/(j['job_id']+'.json')
        put(dest,dict(job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],arm='G32',parsed=parse(raw,j),raw_output=raw,job_binding_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),prompt_sha256=ps,global_feature_lock_sha256=sha(BASE/'global_features'/j['case_id']/'GLOBAL_FEATURE_LOCK.json'),core_local_policy_sha256=sha(CORE/'native_cache'/j['job_id']/'POLICIES.json'),visual_indices_sha256=hashlib.sha256(json.dumps(ix).encode()).hexdigest(),visual_token_count=608,output_count=len(tokens),truncated=len(tokens)>=96,sequence_readback=audit,processor_seconds=processing,global_cache_seconds=cache_seconds,new_global_encoder_call=fresh,new_local_encoder_calls=0,wall_seconds=time.perf_counter()-tick,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),GT_read=False))
        files.append(dict(case_id=j['case_id'],path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size));write(BASE/'TEACHER_STATUS.json',dict(status='running',stage='G32_WHO',done=len(files),total=len(jobs),count=32,GT_read=False),mutable=True)
        del batch,e,pos,f,features,local;gc.collect();torch.cuda.empty_cache()
    put(BASE/'teacher/G32/TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,response_count=len(files),files=files,new_global_encoder_calls=newenc,new_local_encoder_calls=0,new_generation_calls=len(files),model_load_seconds=secs,worker_loop_seconds=time.perf_counter()-start,GT_read=False))
def teacher_root():
    import numpy as np
    from scripts.run_stvg_spatial_who_order_v1 import parse
    from scripts.stvg_global32_native_v1 import select_A_only,global_geometry
    verify();b=BASE/'teacher/G32';bar=read(b/'TEACHER_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==32 and not bar['GT_read']
    jobs={j['job_id']:j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']};assert len(bar['files'])==len(jobs);dims=read(CORE/'SOURCE_GEOMETRY.json')['records'];checked=set();rows=[]
    for entry in bar['files']:
        assert sha(ROOT/entry['path'])==entry['sha256'];d=read(ROOT/entry['path']);j=jobs[d['job_id']];assert d['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest() and d['parsed']==parse(d['raw_output'],j)
        local,q,cp=core_cache(j);p=BASE/'global_features'/j['case_id'];g=read(p/'GLOBAL_FEATURE_LOCK.json');assert sha(p/'GLOBAL_FEATURE_LOCK.json')==d['global_feature_lock_sha256'] and sha(p/'global.npz')==g['feature_sha256'];ix=[g['indices'],cp['records'][1]['indices']['A']]
        assert d['visual_token_count']==608 and hashlib.sha256(json.dumps(ix).encode()).hexdigest()==d['visual_indices_sha256'] and not d['new_local_encoder_calls']
        trace=d['sequence_readback'];assert trace['all_nonvisual_tokens_preserved'] and trace['original_mRoPE_next']-trace['pruned_length']==trace['compressed_cache_rope_delta'] and trace['cache_steps']==d['output_count']
        if j['case_id'] not in checked:
            with np.load(p/'global.npz',allow_pickle=False) as a:f=a['features'].copy()
            assert harray(f)==g['feature_tensor_sha256'];v,ma,gr=global_geometry(**dims[j['case_id']]);policy=select_A_only(f.reshape(16,256,-1),q,v,ma,gr)
            assert policy['indices'].tolist()==g['indices'] and policy['quotas']==g['quotas'];checked.add(j['case_id'])
        rows.append(dict(case_id=j['case_id'],dataset=j['dataset'],slot=j['center_slot'],status=d['parsed']['status'],selected_candidate_id=d['parsed'].get('selected_candidate_id'),GT_read=False))
    allenc=bar['new_global_encoder_calls']+sum(r.get('global_cache_new_encoder_calls',0) for r in read(BASE/'NATIVE_QUALIFICATION.json')['records'])
    assert allenc==len(checked)
    put(b/'ROOT_READBACK.json',dict(status='pass',count=32,response_count=len(rows),records=rows,global_encodes_including_qualification_caches=allenc,unique_global_inputs=len(checked),same_local_features_indices_and_binding=True,barrier_sha256=sha(b/'TEACHER_GLOBAL_BARRIER.json'),GT_read=False))
def wait_core(pid):
    if not(CORE/'CORE6_MEASUREMENT_HANDOFF003.json').exists():
        fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    assert read(CORE/'CORE6_MEASUREMENT_HANDOFF003.json')['group_count']==6
    for arm in ('Full_E1','A','B','C','A_BT','B_BT'):
        b=CORE/'trials'/arm
        for n,status,gt in (('ROOT_READBACK.json','pass',False),('STUDENT_GLOBAL_BARRIER.json','complete',False),('GLOBAL_PREDICTION_BARRIER.json','complete',False),('CPU_COMPLETION002.json','complete',True)):
            d=read(b/n);assert d['status']==status and d['count']==256 and d['GT_read']==gt
def supervise(pid):
    wait_core(pid);verify()
    from scripts import launch_stvg_instance_identity_v1 as r
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='global32_context';g=1024**3;v=r.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    stages=['jobs','qualification','teacher','teacher_root','student_prepare','student_worker','student_root','score']
    try:
        for i,stage in enumerate(stages):
            write(BASE/'STATUS.json',dict(status='running',stage=stage,done=i,total=len(stages),count=32,GT_read=stage=='score'),mutable=True)
            out=BASE/'execution'/stage;out.mkdir(parents=True,exist_ok=False)
            script=ROOT/'scripts/prepare_stvg_global32_context_v1.py' if stage=='jobs' else ROOT/'scripts/run_stvg_global32_student_v1.py' if stage.startswith('student_') else ROOT/'scripts/score_stvg_global32_context_v1.py' if stage=='score' else Path(__file__)
            mode=stage.split('_',1)[1] if stage.startswith('student_') else stage
            inter='.venv/stvg_native_distill_v1/bin/python' if stage in ('qualification','teacher') else '.conda/tubedetr/bin/python'
            with (out/'worker.log').open('xb') as log:
                tick=time.perf_counter();p=subprocess.Popen([str(ROOT/inter),'-B',str(script),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        put(BASE/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_semantic_view_public_archive',count=32,GT_read=True))
        write(BASE/'STATUS.json',dict(status='root_handoff',stage='measurement_complete',count=32,GT_read=True),mutable=True)
    except Exception as e:put(BASE/'FAILURE.json',dict(status='failed_closed',stage=stage,error_type=type(e).__name__,error=str(e),GT_read=stage=='score'));raise
    finally:stop.set()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='lock':lock()
    elif mode=='after_core':supervise(int(sys.argv[2]))
    else:{'qualification':qualification,'teacher':teacher,'teacher_root':teacher_root}[mode]()
