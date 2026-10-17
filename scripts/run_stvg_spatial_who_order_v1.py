"""One ordered WHO prompt crossed with unchanged Full/A visual sequences."""
from pathlib import Path
import ast,copy,gc,hashlib,inspect,json,os,re,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE as ABC,PRIOR,read,put,sha,guard
BASE=ROOT/'artifacts/stvg_spatial_who_order_v1';UNIT='stvg-spatial-who-order-001.service';ARMS=('FULL_ORDER','A_ORDER')
OLD_CRITERIA='Select the ID whose outlined region best corresponds to the complete referent in the query, including its action/role. Do not select a body part or a nearby different object. If none is reliable, return 0. '
ORDERED=('Select one candidate using the following priority order. '
 '1. Referent identity: determine which entity the query refers to, rather than choosing the most active participant. '
 '2. Relation and role: keep candidates for that entity only when its observable actor, affected-object or recipient role and relations agree with the query. Do not interchange participants. '
 '3. Visible extent: among candidates compatible with that identity and role, prefer the region covering the complete visible target entity rather than a part or a region mixing other entities and background. Greater area alone is not evidence of completeness. '
 '4. Abstention: if no candidate reliably satisfies identity, role and visible target extent, return 0. '
 'Apply these checks internally; do not output explanations. ')
def ordered(prompt):
    assert prompt.count(OLD_CRITERIA)==1
    out=prompt.replace(OLD_CRITERIA,ORDERED);assert out.split('Allowed IDs: ')[1]==prompt.split('Allowed IDs: ')[1]
    assert out.split(ORDERED)[0]==prompt.split(OLD_CRITERIA)[0]
    return out
def setup():
    assert read(ABC/'FINAL_COMPLETION.json')['status']=='complete';BASE.mkdir(exist_ok=False)
    for n in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json','SEMANTIC_ANNOTATION_LOCK.json'):(BASE/n).write_bytes((ABC/n).read_bytes())
    jobs=copy.deepcopy(read(ABC/'TEACHER_JOB_LOCK.json'));pins={};contracts=[]
    for j in jobs['records']:
        old=copy.deepcopy(j);assert sha(ABC/'native_cache'/j['job_id']/'native.npz')==read(ABC/'native_cache'/j['job_id']/'POLICIES.json')['native_sha256']
        j['prompt']=ordered(j['prompt']);j['prompt_sha256']=hashlib.sha256(j['prompt'].encode()).hexdigest()
        assert {k:v for k,v in j.items()if k not in ('prompt','prompt_sha256')}=={k:v for k,v in old.items()if k not in ('prompt','prompt_sha256')};contracts.append(j['prompt']!=old['prompt'])
        for p in [ABC/'native_cache'/j['job_id']/'POLICIES.json',ABC/'native_cache'/j['job_id']/'native.npz']:pins[str(p.relative_to(ROOT))]=sha(p)
    assert len(contracts)==64 and all(contracts);put(BASE/'TEACHER_JOB_LOCK.json',jobs)
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=16,arms=list(ARMS),old_controls=['Full_E1','A'],ordered_prompt_template=ORDERED,original_criteria=OLD_CRITERIA,all_exposed_development=True,domain_counts={'hc2':8,'vidstg':8},new_prompt_versions=1,new_visual_policy=False,frozen_Temporal=True,original_OPD_fullstep=True,no_backtracking_or_extent_change=True,no_GT_decisions=True,bootstrap=False,CURRENT_change=False,GT_read=False))
    put(BASE/'CPU_CONTRACTS.json',dict(status='pass',count=64,exact_unmodified_nonprompt_job_fields=True,one_criteria_block_only=True,source_clock_ID_query_tail_unchanged=True,GT_read=False))
    files=[Path(__file__),ROOT/'protocols/stvg_spatial_who_order_v1.md',ROOT/'scripts/stvg_referent_abc_native002.py',ROOT/'scripts/stvg_referent_abc_student002.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',ABC/'RUNTIME.json',ABC/'FINAL_COMPLETION.json',ABC/'TEACHER_JOB_LOCK.json',ABC/'teacher/A/ROOT_READBACK.json']+[BASE/n for n in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json','SEMANTIC_ANNOTATION_LOCK.json','DESIGN_LOCK.json','TEACHER_JOB_LOCK.json')]
    pins.update({str(p.relative_to(ROOT)):sha(p)for p in files});put(BASE/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False));print('WHO_ORDER_LOCKED',16,64)
def verify():
    for p,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def load():
    from scripts import run_stvg_referent_abc002 as old
    old.BASE=BASE;return old.load_teacher()
def inputs(m,p,device,j):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import processor_batch,pack
    c=ABC/'native_cache'/j['job_id'];s=read(c/'POLICIES.json');orig=next(x for x in read(ABC/'TEACHER_JOB_LOCK.json')['records']if x['job_id']==j['job_id']);assert j['prompt']==ordered(orig['prompt'])
    assert s['job_sha256']==hashlib.sha256(json.dumps(orig,sort_keys=True).encode()).hexdigest()and sha(c/'native.npz')==s['native_sha256']
    b,ps=processor_batch(p,j,ROOT,device)
    with np.load(c/'native.npz',allow_pickle=False)as a,torch.inference_mode():e,pos,delta,f,runs=pack(m,b,torch.from_numpy(a['features']).to(device=device,dtype=m.dtype))
    return b,ps,e,pos,f,runs,s
def qualification():
    import torch
    from scripts.stvg_referent_abc_native002 import sequence_keep,greedy
    verify();sys.addaudithook(guard);m,p,device,proof,secs=load();rows=[]
    for ds in ('hc2','vidstg'):
        j=next(j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']if j['dataset']==ds);b,ps,e,pos,f,runs,s=inputs(m,p,device,j);assert s['model_clock_proof']==proof
        with torch.inference_mode():
            m.model.rope_deltas=None;original=m.generate(**b,max_new_tokens=8,do_sample=False,use_cache=True)[0,b['input_ids'].shape[1]:].cpu().tolist()
            allix=sequence_keep(b,m,runs,[list(range(512)),list(range(768))]);ours,tr=greedy(m,e,pos,allix,b['input_ids'],8);assert original==ours
            aix=sequence_keep(b,m,runs,[x['indices']['A']for x in s['records']]);_,at=greedy(m,e,pos,aix,b['input_ids'],8,check_recompute=True);assert at['cached_uncached_next_token_checks']>=2
        rows.append(dict(dataset=ds,no_prune_native_greedy_exact=True,A_cached_uncached_checks=at['cached_uncached_next_token_checks'],GT_read=False));del b,e,pos,f,runs;gc.collect();torch.cuda.empty_cache()
    put(BASE/'NATIVE_QUALIFICATION.json',dict(status='pass',count=2,records=rows,new_visual_encodings=2,clock_proof=proof,model_load_seconds=secs,GT_read=False));print('WHO_QUALIFICATION_PASS')
def parse(raw,j):
    parsed=dict(status='invalid',selected_candidate_id=None)
    try:
        s=re.sub(r'^```(?:json)?\s*','',raw.strip());s=re.sub(r'\s*```$','',s);a=json.loads(s);v=a['id'];assert isinstance(v,int)and not isinstance(v,bool)and v in j['allowed_output_ids'];parsed=dict(status='valid',visible_id=v,selected_candidate_id=j['visible_id_to_candidate'].get(str(v)),reason=a.get('reason'))
    except Exception:pass
    return parsed
def teacher(arm):
    import torch
    from scripts.stvg_referent_abc_native002 import sequence_keep,greedy
    from scripts.stvg_native_instance_common_v1 import write
    verify();assert arm in ARMS and read(BASE/'NATIVE_QUALIFICATION.json')['status']=='pass';sys.addaudithook(guard);m,p,device,proof,secs=load();files=[];tick=time.perf_counter()
    for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
        start=time.perf_counter();b,ps,e,pos,f,runs,s=inputs(m,p,device,j);assert proof==s['model_clock_proof'];indices=[list(range(512)),list(range(768))]if arm=='FULL_ORDER'else[x['indices']['A']for x in s['records']];keep=sequence_keep(b,m,runs,indices)
        with torch.inference_mode():tokens,audit=greedy(m,e,pos,keep,b['input_ids'],96)
        raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();dest=BASE/'teacher'/arm/(j['job_id']+'.json');put(dest,dict(job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],arm=arm,job_binding_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),raw_output=raw,parsed=parse(raw,j),new_teacher_call=True,new_vision_call=False,time_encoding_proof=proof,prompt_sha256=ps,native_policy_sha256=sha(ABC/'native_cache'/j['job_id']/'POLICIES.json'),visual_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),visual_token_count=sum(map(len,indices)),sequence_readback=audit,output_count=len(tokens),truncated=len(tokens)>=96,wall_seconds=time.perf_counter()-start,GT_read=False));files.append(dict(case_id=j['case_id'],path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size));write(BASE/'STATUS.json',dict(status='running',stage='teacher',arm=arm,done=len(files),total=64,GT_read=False),mutable=True);del b,e,pos,f,runs;gc.collect();torch.cuda.empty_cache()
    put(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=16,response_count=64,files=files,new_vision_calls=0,new_greedy_generation_calls=64,exact_response_reuses=0,model_load_seconds=secs,worker_loop_seconds=time.perf_counter()-tick,GT_read=False))
def teacher_root(arm):
    verify();bar=read(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['response_count']==64;rr=[];jobs={j['job_id']:j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']}
    for x in bar['files']:
        assert sha(ROOT/x['path'])==x['sha256'];d=read(ROOT/x['path']);j=jobs[d['job_id']];s=read(ABC/'native_cache'/j['job_id']/'POLICIES.json');ix=[list(range(512)),list(range(768))]if arm=='FULL_ORDER'else[r['indices']['A']for r in s['records']]
        assert d['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()and d['native_policy_sha256']==sha(ABC/'native_cache'/j['job_id']/'POLICIES.json');assert d['visual_indices_sha256']==hashlib.sha256(json.dumps(ix).encode()).hexdigest()and d['visual_token_count']==(1280 if arm=='FULL_ORDER'else 160);assert d['sequence_readback']['all_nonvisual_tokens_preserved']and d['parsed']==parse(d['raw_output'],j)and not d['GT_read'];rr.append(dict(case_id=j['case_id'],slot=j['center_slot'],selected_candidate_id=d['parsed'].get('selected_candidate_id'),GT_read=False))
    put(BASE/'teacher'/arm/'ROOT_READBACK.json',dict(status='pass',count=16,response_count=64,records=rr,barrier_sha256=sha(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json'),exact_original_visual_indices_and_cached_features=True,GT_read=False));print('WHO_TEACHER_ROOT_PASS',arm)
def student(mode,arm):
    from scripts import stvg_referent_abc_student002 as old
    old.BASE=BASE;old.verify=verify
    if mode=='student_prepare':old.prepare(arm)
    else:old.student('worker'if mode=='student_worker'else'root',arm)
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.stvg_native_instance_common_v1 import write
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='ordered_WHO';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        stages=[('qualification',None)]+[(s,a)for a in ARMS for s in ('teacher','teacher_root','student_prepare','student_worker','student_root')]
        for stage,arm in stages:
            out=BASE/'execution'/(arm or'qualification')/stage;out.mkdir(parents=True,exist_ok=False);interp='.venv/stvg_native_distill_v1/bin/python'if stage in ('qualification','teacher')else'.conda/tubedetr/bin/python';args=[str(ROOT/interp),'-B',str(Path(__file__)),stage]+([arm]if arm else[]);write(BASE/'STATUS.json',dict(status='running',stage=stage,arm=arm,GT_read=False),mutable=True)
            with(out/'worker.log').open('xb')as log:
                start=time.perf_counter();p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-start,GT_read=False));assert rc==0,(stage,arm)
        write(BASE/'STATUS.json',dict(status='measurements_complete_pending_independent_score_public_archive',stage='root_handoff',count=16,GT_read=False),mutable=True)
    finally:stop.set()
if __name__=='__main__':
    try:
        mode=sys.argv[1];a=sys.argv[2]if len(sys.argv)>2 else None
        if mode=='setup':setup()
        elif mode=='supervise':supervise()
        elif mode=='qualification':qualification()
        elif mode=='teacher':teacher(a)
        elif mode=='teacher_root':teacher_root(a)
        else:student(mode,a)
    except BaseException as e:
        if BASE.exists():
            p=BASE/'failures'/((sys.argv[1]if len(sys.argv)>1 else'unknown')+'_'+(sys.argv[2]if len(sys.argv)>2 else'none')+'.json')
            if not p.exists():put(p,dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False))
        raise
