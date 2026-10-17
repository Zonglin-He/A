"""Finite serial head test; subsequent authorized Qwen runs dispatched separately."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_head_adapter32_v1 import BASE,read,write,sha,verify
UNIT='stvg-native-head-adapter32-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='head32'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active']and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MemoryHigh,MemoryMax,MemorySwapMax,MainPID'],text=True).splitlines());assert int(props['MemoryHigh'])==16*2**30 and int(props['MemoryMax'])==24*2**30 and int(props['MemorySwapMax'])==2**30
    v=resource.sample();assert v['available']>=8*2**30 and v['gpu_free']>=22*2**30 and v['disk_free']>=30*2**30
    write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False));write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+4*3600,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ['worker','score','report']:
            verify();folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);script='report_stvg_native_head_adapter32_v1.py'if stage=='report'else'stvg_native_head_adapter32_v1.py';args=[]if stage=='report'else[stage]
            with (folder/'worker.log').open('xb')as log:
                tick=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_publication',count=32,GT_read=False),mutable=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=repr(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
