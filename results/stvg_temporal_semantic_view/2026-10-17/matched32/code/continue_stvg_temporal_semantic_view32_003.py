"""Explicit remaining-stage dispatcher; retains both CPU-only launcher failures."""
from pathlib import Path
import os,sys,subprocess,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_semantic_view32_v1 import BASE,read,write,sha
from scripts.stvg_temporal_semantic_grounder32_v2 import verify as prior_verify
UNIT='stvg-temporal-semantic-view32-003.service'
def verify():
    prior_verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME003.json')['pins'].items():assert sha(ROOT/p)==h,p
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();assert not (BASE/'targets').exists();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='temporal_semantic_view32_003';g=2**30
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS003.json',dict(status='pass',unit=UNIT,admission=v,original_deadline_unchanged=True,GT_read=False));threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for command,script in [('targets','stvg_temporal_semantic_grounder32_v2.py'),('fit','stvg_temporal_semantic_view32_v1.py'),('readout','stvg_temporal_semantic_grounder32_v2.py')]:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=command,count=32,GT_read=False),mutable=True);folder=BASE/'execution'/('resume003_'+command);folder.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                t=time.perf_counter();child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=False))
                try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,command
        write(BASE/'PRIMARY_COMPLETION.json',dict(status='complete',count=32,GT_read=False));write(BASE/'STATUS.json',dict(status='primary_complete_pending_full',count=32,GT_read=False),mutable=True)
    except BaseException as e:write(BASE/'FAILURE003.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
