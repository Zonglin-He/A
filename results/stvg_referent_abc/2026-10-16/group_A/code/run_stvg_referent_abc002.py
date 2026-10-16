"""User-authorized two-domain A/B/C, replacing the prior resource stop rule.

Frozen Temporal/candidate clocks and OPD; native token intervention only in
Spatial WHO. Run each entire Teacher/Student group before independent scoring.
"""
from pathlib import Path
import copy,gc,hashlib,inspect,json,math,os,subprocess,sys,threading,time,traceback,types,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import PRIOR,read,put,sha,guard
BASE=ROOT/'artifacts/stvg_referent_abc_v1/revision002';P0=ROOT/'artifacts/stvg_referent_evidence_p0_v1'
UNIT='stvg-referent-abc-001.service'
def setup():
    from scripts.stvg_referent_abc_tokens002 import contracts
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    BASE.mkdir(exist_ok=False);put(BASE/'CPU_CONTRACTS.json',contracts());roster=read(P0/'ROSTER_LOCK.json');assert roster['count']==16
    put(BASE/'ROSTER_LOCK.json',roster);rr=roster['records'];ids={r['probe_alias']for r in rr};jobs=[j for j in read(PRIOR/'spatial/TEACHER_JOB_LOCK.json')['records']if j['evidence_group']=='E1'and j['case_id']in ids];assert len(jobs)==64
    cc={};sources=[];temporal={r['case_id']:r for r in read(PRIOR/'temporal/TEMPORAL_LOCK.json')['records']};caps=[]
    for rec in rr:
        cid=rec['probe_alias'];case,p=load_case(rec);sources.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,source_case_sha256=case['source_case_sha256']));cc[cid]=dict(width=case['input']['width'],height=case['input']['height']);del case;gc.collect()
        target=temporal[cid]['selected_interval'];available=[]
        for j in read(PRIOR/'temporal/INPUT_BARRIER.json')['jobs']:
            m=read(ROOT/j['path'])
            if m['case_id']!=cid:continue
            a,b=m['candidate']['interval_frames'];inter=max(0,min(b,target[1])-max(a,target[0]));union=max(b,target[1])-min(a,target[0]);available.append((inter/union,m['candidate']['canonical_id']))
        _,key=sorted(available,key=lambda x:(-x[0],x[1]))[0];p=PRIOR/'temporal/teacher'/cid/(key+'.json');caption=read(p)['raw_caption'];assert caption.strip();caps.append(dict(case_id=cid,candidate_id=key,path=str(p.relative_to(ROOT)),sha256=sha(p),selection='largest interval IoU with fixed Temporal interval; tie candidate ID, no GT',GT_read=False))
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=16,files=sources,GT_read=False));put(BASE/'TEMPORAL_LOCK.json',dict(status='locked',count=16,records=[temporal[r['probe_alias']]for r in rr],GT_read=False));put(BASE/'CAPTION_LOCK.json',dict(status='locked',count=16,records=caps,GT_read=False));put(BASE/'SOURCE_GEOMETRY.json',dict(status='locked',records=cc,GT_read=False))
    put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=64,records=jobs,GT_read=False))
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=16,domain_counts={'hc2':8,'vidstg':8},all_exposed_development=True,user_override='continue both HC and Vid A/B/C after valid-area correction; no simultaneous-domain-improvement start gate',A='SemVID-style paper-equation adaptation, released .7 frame mix/.8 object ratio/.5 MMR/.5 motion-query; common valid-area and ID safeguards, not exact upstream published baseline',B='dual-text consensus/disagreement, explicit candidate coverage and state-transition score; no special connection selection',C='exact B ID/context/object indices and per-frame role quotas; only transition slots mix reciprocal same-candidate/proximity connection with state score',token_budget='ceil(.125*original visual tokens) per input video, A-derived common frame quotas floored for complete numeric ID glyphs plus two evidence slots; all nonvisual tokens preserved',padding='geometric valid image support after actual crop thumbnail placement; temporal pair intersection; numeric ID glyphs are protected content, not discarded as padding',geometry_feedback='same E1 candidates and original source/visibility admission; Teacher selects WHO only',temporal='fixed corrected Temporal-Lite512 baseline and same four observation positions; not retimed per arm',Student='same episodic1792 M2 Lite-fullstep/full-grid, original rejection/stop semantics; no OPD target or hyperparameter changes',caption='same sealed Source-marked caption per query selected by interval overlap, all arms pay identical acquisition information; no fresh caption generation',sampling='same original context4/local5 physical frames and FPS; native temporal merge and corrected mRoPE, no fake time',Teacher='frozen STVG-R1-7B; original prompt/ID0..N/96 generation cap and deterministic greedy; native bridge must match original unpruned generation',primary_readout='real full-model Student vIoU and paired gains versus Frozen/Temporal/full-unpruned E1 and A/B/C; five metrics and negative/unknown tails; no benefit threshold gates',identity_role='fixed first four SHA sources/domain, slots0 and3: sixteen private panels annotated before arm outputs are read; correct instance/partial/background/wrong participant and role-supported/contradicted/unknown separately, not inferred from IoU',no_source_regeneration=True,no_parameter_search=True,new_models=False,bootstrap=False,CURRENT_change=False,official_fullset=False,GT_read=False))
    files=[Path(__file__),ROOT/'scripts/stvg_referent_abc_tokens002.py',ROOT/'scripts/stvg_referent_abc_native002.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',BASE/'DESIGN_LOCK.json',BASE/'TEACHER_JOB_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'CAPTION_LOCK.json',BASE/'SOURCE_GEOMETRY.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'TEMPORAL_LOCK.json']
    put(BASE/'RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in files},GT_read=False));print('ABC_SETUP',16,64)
def verify():
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
def load_teacher():
    import torch
    from transformers import AutoProcessor
    from scripts import run_stvg_native_instance_teacher_storage002 as loader
    from scripts.stvg_table1_temporal_adapter_v1 import configure_original_processor
    from scripts.stvg_qwen_uniform_clock002 import apply
    loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'};_,m,_,device,_,seconds,_=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,BASE/'offload');proof=apply(m);m.generation_config.use_cache=True;p=AutoProcessor.from_pretrained(ROOT/loader.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False);configure_original_processor(p)
    assert m.generation_config.repetition_penalty==1.05
    m.generation_config.do_sample=False
    return m,p,device,proof,seconds
def prepare_job(m,p,device,job,frozen_features=None):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import processor_batch,pack,geometry
    from scripts.stvg_referent_abc_tokens002 import select
    batch,promptsha=processor_batch(p,job,ROOT,device);cid=job['case_id'];dims=read(BASE/'SOURCE_GEOMETRY.json')['records'][cid];valid,mandatory,groups=geometry(job,dims['width'],dims['height'])
    cached=torch.from_numpy(frozen_features).to(device=device,dtype=m.dtype)if frozen_features is not None else None
    with torch.inference_mode():emb,pos,delta,features,runs=pack(m,batch,cached)
    query=read(PRIOR/'temporal/requests'/f'{cid}.json')['query_text'];cap=next(r for r in read(BASE/'CAPTION_LOCK.json')['records']if r['case_id']==cid);assert sha(ROOT/cap['path'])==cap['sha256'];caption=read(ROOT/cap['path'])['raw_caption']
    def te(text):
        ids=p.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
        with torch.inference_mode():return m.get_input_embeddings()(ids)[0].float().cpu().numpy()
    q,c=te(query),te(caption);z=features.float().cpu().numpy();policies=[]
    for i,(a,b)in enumerate(((0,512),(512,1280))):policies.append(select(z[a:b].reshape(len(valid[i]),256,-1),q,c,valid[i],mandatory[i],groups[i]))
    return batch,emb,pos,features,runs,policies,promptsha,valid,mandatory,groups,q,c
def qualification():
    import torch
    from scripts.stvg_referent_abc_native002 import sequence_keep,greedy
    verify();sys.addaudithook(guard);m,p,device,proof,load_seconds=load_teacher();jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];checks=[]
    for ds in ('hc2','vidstg'):
        j=next(j for j in jobs if j['dataset']==ds and (ds!='hc2'or j['case_id']=='HC03'and j['center_slot']==0));batch,e,pos,features,runs,policies,ps,v,ma,gr,q,c=prepare_job(m,p,device,j)
        with torch.inference_mode():
            m.model.rope_deltas=None;official=m.generate(**batch,max_new_tokens=8,do_sample=False,use_cache=True)[0,batch['input_ids'].shape[1]:].cpu().tolist();keep=torch.arange(e.shape[1],device=device);ours,audit=greedy(m,e,pos,keep,batch['input_ids'],cap=8)
            assert official==ours,(ds,official,ours)
            arm=[]
            for kind in ('A','B','C'):
                ix=sequence_keep(batch,m,runs,[r['indices'][kind]for r in policies]);out,trace=greedy(m,e,pos,ix,batch['input_ids'],cap=8,check_recompute=True);assert trace['cached_uncached_next_token_checks']>=2;arm.append(dict(arm=kind,trace=trace,output_count=len(out),GT_read=False))
        checks.append(dict(dataset=ds,job_id=j['job_id'],no_prune_native_greedy_exact=True,full_prefix=ours,arms=arm,per_video_quotas=[r['quotas']for r in policies],GT_read=False));del batch,e,pos,features,runs,policies;gc.collect();torch.cuda.empty_cache()
    put(BASE/'NATIVE_QUALIFICATION.json',dict(status='pass',count=2,records=checks,clock_proof=proof,model_load_seconds=load_seconds,GT_read=False,scope='actual native no-prune greedy tokens exact plus each pruned A/B/C two cached-versus-uncached next-token checks and original mRoPE/nonvisual preservation'))
    print('ABC_NATIVE_QUALIFICATION_PASS')
def teacher(arm):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import sequence_keep,greedy,pack,processor_batch
    assert arm in ('A','B','C');verify();assert read(BASE/'NATIVE_QUALIFICATION.json')['status']=='pass';sys.addaudithook(guard);m,p,device,proof,load_seconds=load_teacher();files=[];start=time.perf_counter()
    for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
        cache=BASE/'native_cache'/j['job_id'];sig=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest();tick=time.perf_counter()
        reused_response=None;new_vision=False
        if arm=='A':
            legacy=BASE.parent/'native_cache'/j['job_id'];lp=BASE.parent/'teacher/A'/(j['job_id']+'.json')
            if (legacy/'POLICIES.json').exists()and lp.exists():
                previous=read(legacy/'POLICIES.json');assert previous['job_sha256']==sig and previous['model_clock_proof']==proof and sha(legacy/'native.npz')==previous['native_sha256']
                with np.load(legacy/'native.npz',allow_pickle=False)as old:
                    batch,e,pos,features,runs,policies,ps,valid,mandatory,groups,q,c=prepare_job(m,p,device,j,old['features']);assert np.array_equal(q,old['query_embedding'])and np.array_equal(c,old['caption_embedding'])
                assert ps==previous['prompt_sha256']
                same=all(r['indices']['A'].tolist()==prev['indices']['A']for r,prev in zip(policies,previous['records']))
                if same:reused_response=read(lp);assert reused_response['job_binding_sha256']==sig
            else:
                batch,e,pos,features,runs,policies,ps,valid,mandatory,groups,q,c=prepare_job(m,p,device,j);new_vision=True
            cache.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(cache/'native.npz',features=features.float().cpu().numpy(),query_embedding=q,caption_embedding=c,**{f'valid{i}':x for i,x in enumerate(valid)},**{f'mandatory{i}':x for i,x in enumerate(mandatory)},**{f'groups{i}':x for i,x in enumerate(groups)})
            serial=[]
            for r in policies:serial.append({k:({a:v.tolist()for a,v in z.items()}if k=='indices'else z)for k,z in r.items()})
            put(cache/'POLICIES.json',dict(status='locked',records=serial,job_sha256=sig,prompt_sha256=ps,native_sha256=sha(cache/'native.npz'),model_clock_proof=proof,GT_read=False))
        else:
            saved=read(cache/'POLICIES.json');assert saved['job_sha256']==sig and saved['model_clock_proof']==proof and sha(cache/'native.npz')==saved['native_sha256'];batch,ps=processor_batch(p,j,ROOT,device);assert ps==saved['prompt_sha256'];a=np.load(cache/'native.npz',allow_pickle=False)
            with torch.inference_mode():e,pos,_,features,runs=pack(m,batch,torch.from_numpy(a['features']).to(device=device,dtype=m.dtype))
            policies=saved['records'];a.close()
        keep=sequence_keep(batch,m,runs,[r['indices'][arm]for r in policies])
        if reused_response is None:
            with torch.inference_mode():tokens,audit=greedy(m,e,pos,keep,batch['input_ids'],96)
            raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();output_count=len(tokens)
        else:
            audit=reused_response['sequence_readback'];assert audit['pruned_length']==len(keep);raw=reused_response['raw_output'];output_count=reused_response['output_count']
        # Same original strict parser, isolated from its command-line BASE.
        clean=raw.strip();parsed=dict(status='invalid',selected_candidate_id=None)
        try:
            import re
            clean=re.sub(r'^```(?:json)?\s*','',clean);clean=re.sub(r'\s*```$','',clean);answer=json.loads(clean);id_=answer['id'];assert isinstance(id_,int)and not isinstance(id_,bool)and id_ in j['allowed_output_ids'];parsed=dict(status='valid',visible_id=id_,selected_candidate_id=j['visible_id_to_candidate'].get(str(id_)),reason=answer.get('reason'))
        except Exception:pass
        result=dict(job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],arm=arm,job_binding_sha256=sig,new_teacher_call=reused_response is None,new_vision_call=new_vision,reused_response_sha256=sha(lp)if reused_response is not None else None,raw_output=raw,parsed=parsed,time_encoding_proof=proof,prompt_sha256=ps,native_policy_sha256=sha(cache/'POLICIES.json'),output_count=output_count,truncated=output_count>=96,sequence_readback=audit,wall_seconds=time.perf_counter()-tick,GT_read=False)
        dest=BASE/'teacher'/arm/(j['job_id']+'.json');put(dest,result);files.append(dict(case_id=j['case_id'],path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size))
        from scripts.stvg_native_instance_common_v1 import write
        write(BASE/'STATUS.json',dict(status='running',stage='Teacher_'+arm,done=len(files),total=64,GT_read=False),mutable=True)
        del batch,e,pos,features,runs,policies;gc.collect();torch.cuda.empty_cache()
    outputs=[read(ROOT/f['path'])for f in files]
    put(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=16,response_count=64,files=files,GT_read=False,model_load_seconds=load_seconds,worker_loop_seconds=time.perf_counter()-start,new_vision_calls=sum(x['new_vision_call']for x in outputs),new_greedy_generation_calls=sum(x['new_teacher_call']for x in outputs),exact_response_reuses=sum(not x['new_teacher_call']for x in outputs),clock_proof=proof))
def run_stage(stage,arm=None):
    from scripts import launch_stvg_instance_identity_v1 as resource
    global UNIT
    UNIT='stvg-referent-abc-'+(arm.lower()if arm else'qualification')+'-002.service';resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE=stage;g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        modes=['qualification']if stage=='qualification'else['teacher','teacher_root','student_prepare','student_worker','student_root']
        for mode in modes:
            out=BASE/'execution'/(arm or'qualification')/mode;out.mkdir(parents=True,exist_ok=False);interp='.venv/stvg_native_distill_v1/bin/python'if mode in ('qualification','teacher')else'.conda/tubedetr/bin/python'
            args=[str(ROOT/interp),'-B',str(Path(__file__)),mode]+([arm]if arm else[])
            with(out/'worker.log').open('xb')as log:
                t=time.perf_counter();p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,(mode,arm)
    finally:stop.set()
if __name__=='__main__':
    try:
        mode=sys.argv[1];arm=sys.argv[2]if len(sys.argv)>2 else None
        if mode=='setup':setup()
        elif mode=='supervise':run_stage('qualification'if arm is None else'group',arm)
        elif mode=='qualification':qualification()
        elif mode=='teacher':teacher(arm)
        else:
            from scripts.stvg_referent_abc_student002 import run
            run(mode,arm)
    except BaseException as e:
        if BASE.exists():
            p=BASE/'failures'/((sys.argv[1]if len(sys.argv)>1 else'unknown')+'_'+(sys.argv[2]if len(sys.argv)>2 else'none')+'.json')
            if not p.exists():put(p,dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False))
        raise
