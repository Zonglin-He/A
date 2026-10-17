"""One serial R1 preference32 -> matched Student32/root -> D2 -> postseal scoring chain."""
from pathlib import Path
import copy,inspect,os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_student32_v1 import BASE,PREF,OLD,read,write,sha,verify
UNIT='stvg-semantic-preference-student32-001.service'
def release():
    from scripts import continue_stvg_semantic_preference_r1_002 as old
    ns=dict(old.__dict__);ns.update(BASE=BASE,QWEN=OLD)
    exec(compile(inspect.getsource(old.release),__file__,'exec'),ns);return ns['release']()
def lock_trials():
    r=read(BASE/'EXECUTION_RUNTIME.json')
    for arm in ['GEOM','PREF']:
        a=BASE/'arms'/arm;t=a/'trials/G32';pins=dict(r['pins'])
        paths=[BASE/'SUPPORT_LOCK.json',PREF/'R1_ROOT_READBACK.json',PREF/'teachers/R1/QUALIFICATION_COMPLETION.json',a/'ROSTER_LOCK.json',a/'SOURCE_CAPTURE_BARRIER.json',a/'SCOPE_CONFIG.json',a/'teacher/G32/ROOT_READBACK.json',t/'REFERENCE_LOCK.json',t/'TEMPORAL_LOCK.json',t/'DESIGN_LOCK.json']
        pins.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
        write(t/'RUNTIME.json',dict(status='locked',trial_id='G32',pins=pins,protected={},GT_read=False))
    write(BASE/'STUDENT_INPUT_COMPLETION.json',dict(status='pass',count=32,same_refs_and_admission_for_GEOM_PREF_D2=True,GT_read=False))
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='semantic_student32'
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines())
    assert [int(props[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*2**30,24*2**30,2**30]
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        v=resource.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
        write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False))
        tick=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+7200,GT_read=False));write(PREF/'R1_ACTIVE_EXECUTION002.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+7200,GT_read=False))
        stages=[('teacher32','stvg_semantic_preference_r1_collect32_v1.py',['collect']),('setup','stvg_semantic_preference_student32_v1.py',['setup']),('lock',None,[]),('GEOM','stvg_semantic_preference_student32_v1.py',['worker','GEOM']),('GEOM_root','stvg_semantic_preference_student32_v1.py',['root','GEOM']),('PREF','stvg_semantic_preference_student32_v1.py',['worker','PREF']),('PREF_root','stvg_semantic_preference_student32_v1.py',['root','PREF']),('propagate','stvg_semantic_preference_student32_v1.py',['propagate']),('score','stvg_semantic_preference_student32_v1.py',['score'])]
        for stage,script,args in stages:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),mutable=True)
            if stage=='lock':lock_trials();continue
            folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if stage=='teacher32' else '.conda/tubedetr/bin/python')
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                t=time.perf_counter();child=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,stage=stage,GT_read=stage=='score'))
                try:code=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=stage=='score'));assert code==0,stage+' failed; preserve and stop, no auto-retry'
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',stage='root_publication',count=32,GT_read=False),mutable=True)
        write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set();release()
if __name__=='__main__':release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()
