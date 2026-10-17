"""Continue after immutable speed-profile namespace collision; no repeated GEOM fits."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_4anchor32_v2 import BASE,read,write,sha,verify
from scripts import continue_stvg_semantic_preference_4anchor32_v1 as first
UNIT='stvg-semantic-preference-4anchor32-002.service'
def release():return first.release()
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();assert read(BASE/'FAILURE.json')['status']=='technical_failure'
    assert read(BASE/'arms/GEOM/trials/G32/STUDENT_GLOBAL_BARRIER.json')['count']==32
    assert read(BASE/'execution/GEOM/WORKER_EXIT.json')['exit_code']==1
    first.UNIT=UNIT;resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='four_anchor32_recovery002'
    p=first.props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*2**30,24*2**30,2**30]
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
        # First hold/release receipt remains immutable evidence; acquire a new
        # identity-checked hold under this recovery-specific directory.
        hbase=BASE/'recovery/profile_namespace002';hbase.mkdir(exist_ok=True,parents=True);first.BASE=hbase;first.hold();write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS002.json',dict(status='pass',properties=p,admission=v,GT_read=False))
        for stage,command,arm in [('GEOM_root002','root','GEOM'),('PREF','worker','PREF'),('PREF_root002','root','PREF'),('propagate','propagate',None),('score','score',None)]:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),mutable=True);folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as f:
                t=time.perf_counter();child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/stvg_semantic_preference_4anchor32_v2.py'),command]+([arm] if arm else []),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=stage=='score'))
                try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=stage=='score'));assert rc==0,stage+' failed; preserve without repeated fits'
        write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',stage='root_publication',count=32,GT_read=False),mutable=True)
    except BaseException as e:
        write(BASE/'FAILURE002.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set();first.BASE=BASE/'recovery/profile_namespace002';first.release()
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='release':first.BASE=BASE/'recovery/profile_namespace002';first.release()
    else:main()
