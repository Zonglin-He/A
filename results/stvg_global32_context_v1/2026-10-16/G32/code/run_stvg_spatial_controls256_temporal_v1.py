"""One corrected frozen Temporal baseline for every fixed256 spatial arm."""
from pathlib import Path
import ast,gc,inspect,json,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha,guard
OUT=BASE/'temporal';UNIT='stvg-spatial-controls256-temporal-001.service'
def verify():
    for p,h in read(BASE/'TEMPORAL_DISPATCH_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def verify_temporal():
    verify()
    for p,h in read(OUT/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def prepare():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import run_stvg_table1_source_v1 as source,run_stvg_table1_teacher_v1 as render,stvg_table1_temporal_adapter_v1 as adapter,prepare_stvg_lite_reclocked_inputs002 as reqhelper
    verify();sys.addaudithook(guard);bar=read(BASE/'SOURCE_CAPTURE_BARRIER.json');assert bar['status']=='complete'and bar['count']==256 and not bar['GT_read'];files=[];jobs=[];cases=[];reqhelper.BASE=BASE;reqhelper.CURRENT=dict(cohort='development')
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];case,cp=load_case(rec);assert case['case_id']==cid;_,plan=source._validate_roster_record(rec);req,raw,hs,origin=reqhelper._resolve_request(rec,case,plan['input']['caption']);dest=OUT/'requests'/f'{cid}.json';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();dest.write_bytes(raw);assert sha(dest)==hs
        eligible,reason=render.temporal_clip_eligibility(case,req)
        if eligible:
            rgb=render.rgb_for_candidates(case,req,case['temporal_candidates']);clips=adapter.prepare_candidate_clips(case,case['temporal_candidates'],decoded_rgb_by_physical_id=rgb,frame_map=req['frame_map'])
            for c in case['temporal_candidates']:
                key=c['canonical_id'];clip=clips[key];p=OUT/'inputs'/cid/(key+'.npy');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,clip['video'],allow_pickle=False);meta=dict(case_id=cid,dataset=rec['dataset'],candidate=c,metadata=clip['metadata'],input_path=str(p.relative_to(ROOT)),input_sha256=sha(p),source_file_sha256=sha(cp),source_case_sha256=case['source_case_sha256'],GT_read=False);mp=p.with_suffix('.json');put(mp,meta);jobs.append(dict(case_id=cid,path=str(mp.relative_to(ROOT)),sha256=sha(mp)));files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
            del rgb,clips
        cases.append(dict(case_id=cid,dataset=rec['dataset'],eligible=eligible,fallback_reason=reason,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),legacy_interval=list(case['Native_interval']),cpu_blas_profile=2,request_origin=origin,GT_read=False));print('TEMPORAL_INPUT_PROGRESS',len(cases),256,eligible,flush=True);del case;gc.collect()
    put(OUT/'INPUT_BARRIER.json',dict(status='complete',count=256,records=cases,jobs=jobs,files=files,legacy_Lite_fit_exact=0,baseline='new fixed corrected Temporal on selected256, not an old adapted-state reuse',GT_read=False));pins=dict(read(BASE/'TEMPORAL_DISPATCH_RUNTIME.json')['pins']);pins[str((OUT/'INPUT_BARRIER.json').relative_to(ROOT))]=sha(OUT/'INPUT_BARRIER.json');put(OUT/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False));print('TEMPORAL256_INPUTS_SEALED')
def invoke(mode):
    from scripts import stvg_instance_temporal32_v1 as old
    fn=old.teacher if mode=='teacher'else old.fit_cases;s=inspect.getsource(fn);tree=ast.parse(s)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n)if type(n.value)is int and n.value==32 else n
    s=ast.unparse(ast.fix_missing_locations(Count().visit(tree)));ns=dict(old.__dict__);ns.update(BASE=BASE,OUT=OUT,verify_temporal=verify_temporal,p0_reuse=lambda *a:None);exec(compile(s,str(Path(__file__)),'exec'),ns);ns[fn.__name__](True)if mode=='readback'else ns[fn.__name__]()
def supervise(pid):
    verify();fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd);bar=read(BASE/'SOURCE_CAPTURE_BARRIER.json');assert bar['status']=='complete'and bar['count']==256 and read(BASE/'SOURCE_RECOVERY_COMPLETION002.json')['status']=='complete'
    from scripts import launch_stvg_instance_identity_v1 as r
    from scripts.stvg_native_instance_common_v1 import write
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='temporal256';g=1024**3;v=r.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ('prepare','teacher','fit','readback'):
            out=BASE/'execution/temporal256'/stage;out.mkdir(parents=True,exist_ok=False);interp='.venv/stvg_native_distill_v1/bin/python'if stage=='teacher'else'.conda/tubedetr/bin/python';write(BASE/'STATUS.json',dict(status='running',stage='temporal_'+stage,count=256,GT_read=False),mutable=True)
            with(out/'worker.log').open('xb')as log:
                t=time.perf_counter();p=subprocess.Popen([str(ROOT/interp),'-B',str(Path(__file__)),stage],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,stage
        put(BASE/'TEMPORAL_HANDOFF.json',dict(status='complete_pending_spatial_inputs',count=256,GT_read=False))
    finally:stop.set()
if __name__=='__main__':
    m=sys.argv[1]
    if m=='after_source':supervise(int(sys.argv[2]))
    elif m=='prepare':prepare()
    else:invoke(m)
