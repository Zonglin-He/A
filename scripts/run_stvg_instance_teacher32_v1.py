"""One hot corrected-clock R1 for matched E0/E1; one request per window."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,OLD,read,put,sha
SP=BASE/'spatial';UNIT='stvg-instance-evidence32-spatial-teacher-001.service'
def signature(job):
    videos=[{k:v for k,v in video.items()if k!='path'}for video in job['videos']]
    return hashlib.sha256(json.dumps(dict(prompt=job['prompt'],videos=videos,candidate_bindings=job['candidate_bindings'],source_case_sha256=job['source_case_sha256'],physical_frame_ids=job['physical_frame_ids'],context_center_physical_frame_ids=job['context_center_physical_frame_ids']),sort_keys=True).encode()).hexdigest()
def prepare():
    assert read(SP/'ROOT_INPUT_READBACK.json')['status']=='pass';known={};old=OLD/'clock_control';ob=read(old/'TEACHER_GLOBAL_BARRIER.json');bound={f['path']:f['sha256']for f in ob['files']}
    for j in read(old/'TEACHER_JOB_LOCK.json')['records']:
        p=old/'teacher'/f"{j['job_id']}.json";assert sha(p)==bound[str(p.relative_to(ROOT))];r=read(p);assert r['new_teacher_call']and r['time_encoding_proof']
        known[signature(j)]=(r,p)
    reused=[]
    for j in read(SP/'TEACHER_JOB_LOCK.json')['records']:
        s=signature(j)
        if s not in known:continue
        r,p=known[s];d=copy.deepcopy(r);d.update(job_binding_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),job_id=j['job_id'],new_teacher_call=False,reused_corrected_teacher_sha256=sha(p),exact_input_signature=s,GT_read=False)
        dest=SP/'teacher'/f"{j['job_id']}.json";put(dest,d);reused.append(dict(job_id=j['job_id'],source_sha256=sha(p),output_sha256=sha(dest)))
    put(SP/'CORRECTED_TEACHER_REUSE_LOCK.json',dict(status='locked',records=reused,count=len(reused),GT_read=False,scope='exact marked NPY/video FPS, prompts, candidate bindings, physical clocks and frozen corrected model protocol'))
    paths=[Path(__file__),ROOT/'scripts/stvg_reference_clock_control_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',SP/'ROOT_INPUT_READBACK.json',SP/'TEACHER_RUNTIME.json',SP/'CORRECTED_TEACHER_REUSE_LOCK.json']
    put(SP/'TEACHER_EXECUTION_RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
    print(json.dumps({'status':'locked','corrected_exact_reuses':len(reused)}))
def verify():
    for rel,h in read(SP/'TEACHER_EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
def worker():
    verify();from scripts import stvg_reference_clock_control_v1 as control
    control.BASE=SP;control.worker()
def root():
    verify();bar=read(SP/'TEACHER_GLOBAL_BARRIER.json');jobs=read(SP/'TEACHER_JOB_LOCK.json')['records'];assert bar['status']=='complete'and bar['count']==len(jobs)
    files={f['path']:f['sha256']for f in bar['files']};records=[]
    for j in jobs:
        p=SP/'teacher'/f"{j['job_id']}.json";assert sha(p)==files[str(p.relative_to(ROOT))];d=read(p);assert d['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()and not d['GT_read']
        assert d['time_encoding_proof']==read(BASE/'temporal/TEACHER_GLOBAL_BARRIER.json')['proof']
        records.append(dict(case_id=j['case_id'],group=j['evidence_group'],slot=j['center_slot'],selected_candidate_id=d['parsed']['selected_candidate_id'],parse_status=d['parsed']['status'],new_teacher_call=d['new_teacher_call'],GT_read=False))
    put(SP/'ROOT_TEACHER_READBACK.json',dict(status='pass',count=32,response_count=len(records),records=records,new_teacher_calls=sum(r['new_teacher_call']for r in records),GT_read=False,barrier_sha256=sha(SP/'TEACHER_GLOBAL_BARRIER.json')))
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=SP;resource.UNIT=UNIT;resource.STAGE='teacher';g=1024**3
    v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for mode in ('worker','root'):
            p=SP/'execution'/mode;p.mkdir(parents=True,exist_ok=False);interp=ROOT/('.venv/stvg_native_distill_v1/bin/python'if mode=='worker'else'.conda/tubedetr/bin/python')
            with(p/'worker.log').open('xb')as log:
                t=time.perf_counter();c=subprocess.Popen([str(interp),'-B',str(Path(__file__)),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False,time=time.time()));code=c.wait()
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=False,time=time.time()));assert code==0,mode
    finally:stop.set()
if __name__=='__main__':{'prepare':prepare,'worker':worker,'root':root,'supervise':supervise}[sys.argv[1]]()
