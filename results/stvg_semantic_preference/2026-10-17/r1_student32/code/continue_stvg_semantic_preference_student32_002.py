"""Resume after a root-only audit failure; reuse all Teacher and GEOM predictions exactly."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_student32_v1 import BASE,read,write,sha,verify
from scripts.continue_stvg_semantic_preference_student32_v1 import release as release_original
def release():
    release_original()
    from scripts.stvg_semantic_preference_student32_v1 import OLD
    r=read(BASE/'QWEN_PRIORITY_HANDOFF002.json')
    status=read(OLD/'STATUS.json')
    if status.get('status','').startswith('waiting_priority_R1') and read(BASE/'QWEN_RESUME_RECEIPT002.json')['status']=='resumed_verified_CPU_controller':write(OLD/'STATUS.json',r['Qwen_original_status'],mutable=True)
UNIT='stvg-semantic-preference-student32-002.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify()
    for p,h in read(BASE/'RECOVERY_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='semantic_student32_recovery002'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        stages=[('GEOM_root002','root_stvg_semantic_preference_student32_002.py',['GEOM']),('PREF','stvg_semantic_preference_student32_v1.py',['worker','PREF']),('PREF_root002','root_stvg_semantic_preference_student32_002.py',['PREF']),('propagate','stvg_semantic_preference_student32_v1.py',['propagate']),('score','stvg_semantic_preference_student32_v1.py',['score'])]
        for stage,script,args in stages:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),mutable=True)
            folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                tick=time.perf_counter();child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=stage=='score'))
                try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage+' failed; outputs preserved'
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',stage='root_publication',count=32,GT_read=False),mutable=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE002.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set();release()
if __name__=='__main__':release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()
