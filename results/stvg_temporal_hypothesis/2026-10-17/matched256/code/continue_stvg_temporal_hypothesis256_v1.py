"""One finite original256 B-only execution; no new Spatial stage."""
from pathlib import Path
import os,sys,subprocess,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis256_v1 import BASE,read,write,verify
UNIT='stvg-temporal-hypothesis256-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.continue_stvg_semantic_preference_4anchor32_v1 import props
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='hypothesis256';g=2**30;p=props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,24*g,g]
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=p,admission=v,GT_read=False));now=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=now,deadline_epoch=now+14400,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    def stage(label,teacher=False,gt=False):
        verify();write(BASE/'STATUS.json',dict(status='running',stage=label,count=256,GT_read=gt),mutable=True);folder=BASE/'execution'/label;folder.mkdir(parents=True,exist_ok=False);py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if teacher else '.conda/tubedetr/bin/python');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1');cmd=[str(py),'-B',str(ROOT/'scripts'/('score_stvg_temporal_hypothesis256_v1.py' if gt else 'stvg_temporal_hypothesis256_v1.py'))]+([] if gt else [label])
        with (folder/'worker.log').open('xb') as log:
            tick=time.perf_counter();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=gt))
            try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
        write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=gt));assert rc==0,label
    try:
        for label in ['prepare','teacher','teacher_root','fit','readout']:stage(label,teacher=label=='teacher')
        stage('score',gt=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=256,GT_read=False));write(BASE/'COMPLETION.json',dict(status='complete',count=256,GT_read=True));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=256,GT_read=False),mutable=True)
    except BaseException as e:write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=256,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
