"""Finite serial Qwen temporal Teacher, original marginal fitting, real readout."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen_b3_temporal32_v1 import BASE,Q,read,write,verify
UNIT='stvg-qwen-b3-temporal32-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as r
    verify();assert read(Q/'CPU_COMPLETION.json')['status']=='complete'
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='qwen_temporal32'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active']and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    r.trigger=trigger;v=r.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
    write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+4*3600,GT_read=False));write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ['teacher','fit','readout','score','report']:
            verify();folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);script='report_stvg_qwen_b3_temporal32_v1.py'if stage=='report'else'stvg_qwen_b3_temporal32_v1.py';args=[]if stage=='report'else[stage];py=ROOT/('.venv/stvg_qwen35_geometry_v1/bin/python'if stage=='teacher'else'.conda/tubedetr/bin/python')
            with (folder/'worker.log').open('xb')as log:
                tick=time.perf_counter();p=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_publication',count=32,GT_read=False),mutable=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=repr(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
