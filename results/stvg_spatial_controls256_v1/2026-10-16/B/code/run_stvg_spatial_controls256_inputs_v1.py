"""Prospective original E1 inputs for the fixed256 sources; no new proposals."""
from pathlib import Path
import ast,gc,inspect,json,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha,guard
SP=BASE/'spatial';UNIT='stvg-spatial-controls256-inputs-001.service'
def verify():
    for p,h in read(BASE/'INPUT_DISPATCH_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert read(BASE/'temporal/ROOT_READBACK.json')['status']=='pass'and read(BASE/'temporal/GLOBAL_PREDICTION_BARRIER.json')['count']==256
def prepare():
    from scripts import stvg_instance_spatial32_v1 as old
    verify();s=inspect.getsource(old.prepare);start=s.index('oldpath=LITE/');end=s.index('pool=builder.build_candidate_pool',start);s=s[:start]+'cache={};mcache={}\n        '+s[end:];assert s.count("[('E0',windows),('E1',updated)]")==1;s=s.replace("[('E0',windows),('E1',updated)]","[('E1',updated)]");tree=ast.parse(s)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n)if type(n.value)is int and n.value==32 else n
    s=ast.unparse(ast.fix_missing_locations(Count().visit(tree)));ns=dict(old.__dict__);ns.update(BASE=BASE,SP=SP,verify=verify);exec(compile(s,str(Path(__file__)),'exec'),ns);ns['prepare']()
def input_root():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    verify();b=read(SP/'CANDIDATE_INPUT_BARRIER.json');assert b['status']=='complete'and b['count']==256 and not b['GT_read'];jobs=read(SP/'TEACHER_JOB_LOCK.json')['records'];assert all(j['evidence_group']=='E1'for j in jobs)
    for f in b['files']:assert sha(ROOT/f['path'])==f['sha256']
    covered=set()
    for j in jobs:
        covered.add(j['case_id']);assert hashlib_sha(j['prompt'])==j['prompt_sha256']and j['allowed_output_ids']==list(range(j['candidate_count']+1));assert len(j['videos'])==2
        for v in j['videos']:
            z=np.load(ROOT/v['path'],allow_pickle=False);assert sha(ROOT/v['path'])==v['sha256']and __import__('hashlib').sha256(z.tobytes()).hexdigest()==v['tensor_sha256']
        pool=read(SP/'E1/proposals'/(j['case_id']+'.json'));w=next(w for w in pool['windows']if w['slot']==j['center_slot']);assert [x['candidate_id']for x in w['tracks']]==[x['candidate_id']for x in j['candidate_bindings']];assert j['center_physical_frame_id']==w['center_physical_frame_id']and len(w['physical_frame_ids'])==5;assert w['tracks'][0]['candidate_id']==1
    put(SP/'ROOT_INPUT_READBACK.json',dict(status='pass',count=256,job_count=len(jobs),queries_with_jobs=len(covered),queries_without_jobs=256-len(covered),all_job_NPY_prompt_mask_proposal_SHAs=True,original_fresh_E1_algorithm_only=True,no_E0_unused_rendering=True,GT_read=False))
    # Common metadata for native evidence policies; fixed existing caption only.
    rr=read(BASE/'ROSTER_LOCK.json')['records'];tt={x['case_id']:x for x in read(BASE/'temporal/TEMPORAL_LOCK.json')['records']};inputs={x['case_id']:x for x in read(BASE/'temporal/INPUT_BARRIER.json')['records']};jj=read(BASE/'temporal/INPUT_BARRIER.json')['jobs'];dims={};caps=[]
    for r in rr:
        cid=r['probe_alias'];case,p=load_case(r);dims[cid]=dict(width=case['input']['width'],height=case['input']['height']);candidates=[];target=tt[cid]['selected_interval']
        for j in jj:
            if j['case_id']!=cid:continue
            m=read(ROOT/j['path']);a,b=m['candidate']['interval_frames'];inter=max(0,min(b,target[1])-max(a,target[0]));union=max(b,target[1])-min(a,target[0]);candidates.append((inter/union,m['candidate']['canonical_id']))
        key=sorted(candidates,key=lambda x:(-x[0],x[1]))[0][1]if candidates else None;p=BASE/'temporal/teacher'/cid/(key+'.json')if key is not None else None;available=p is not None and p.exists()and bool(read(p)['raw_caption'].strip());caps.append(dict(case_id=cid,candidate_id=key,path=str(p.relative_to(ROOT))if available else None,sha256=sha(p)if available else None,available=available,GT_read=False));del case;gc.collect()
    put(BASE/'CAPTION_LOCK.json',dict(status='locked',count=256,records=caps,missing_caption_policy='A/Full remain qualified with A-only scoring independent of caption; B/C and their derived reference controls have no spatial evidence when no caption exists; no fabricated caption or extra generation',GT_read=False));put(BASE/'SOURCE_GEOMETRY.json',dict(status='locked',records=dims,GT_read=False));put(BASE/'TEACHER_JOB_LOCK.json',read(SP/'TEACHER_JOB_LOCK.json'));put(BASE/'TEMPORAL_LOCK.json',read(BASE/'temporal/TEMPORAL_LOCK.json'));put(BASE/'SPATIAL_INPUT_HANDOFF.json',dict(status='complete_pending_native_evidence',count=256,job_count=len(jobs),GT_read=False));print('SPATIAL256_INPUT_ROOT_PASS',len(jobs))
def hashlib_sha(s):return __import__('hashlib').sha256(s.encode()).hexdigest()
def supervise(pid):
    fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd);assert read(BASE/'TEMPORAL_HANDOFF.json')['status']=='complete_pending_spatial_inputs';verify()
    from scripts import launch_stvg_instance_identity_v1 as r
    from scripts.stvg_native_instance_common_v1 import write
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='spatial_inputs256';g=1024**3;v=r.sample();assert v['available']>=8*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ('prepare','root'):
            out=BASE/'execution/spatial_inputs256'/stage;out.mkdir(parents=True,exist_ok=False);write(BASE/'STATUS.json',dict(status='running',stage='spatial_inputs_'+stage,count=256,GT_read=False),mutable=True)
            with(out/'worker.log').open('xb')as log:
                t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),stage],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,stage
    finally:stop.set()
if __name__=='__main__':
    m=sys.argv[1]
    if m=='after_temporal':supervise(int(sys.argv[2]))
    elif m=='prepare':prepare()
    else:input_root()
