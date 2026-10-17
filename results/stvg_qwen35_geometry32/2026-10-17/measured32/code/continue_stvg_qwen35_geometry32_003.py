"""Continue remaining unique Qwen requests after fixed native-head test."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen35_geometry_run_v3 import BASE,UNIT,verify_run,read,write,sha

def main():
    from scripts import launch_stvg_instance_identity_v1 as r
    verify_run();assert read(ROOT/'artifacts/stvg_native_head_adapter32_v1/CPU_COMPLETION.json')['status']=='complete'
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='qwen003'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active']and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    r.trigger=trigger
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MemoryHigh,MemoryMax,MemorySwapMax,MainPID'],text=True).splitlines());assert int(props['MemoryHigh'])==16*2**30 and int(props['MemoryMax'])==24*2**30 and int(props['MemorySwapMax'])==2**30
    v=r.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
    write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS003.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ['teacher','reference_root','student_prepare','student','student_root','propagate','score','report','render']:
            verify_run();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,revision='003',GT_read=stage=='score'),mutable=True);folder=BASE/'execution003'/stage;folder.mkdir(parents=True,exist_ok=False)
            py=ROOT/('.venv/stvg_qwen35_geometry_v1/bin/python'if stage=='teacher'else'.conda/tubedetr/bin/python');script='report_stvg_qwen35_geometry32_v3.py'if stage in ['report','render']else'stvg_qwen35_geometry_run_v3.py'
            with (folder/'worker.log').open('xb')as log:
                tick=time.perf_counter();p=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),stage],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        write(BASE/'GROUP_PUBLICATION_HANDOFF003.json',dict(status='ready',count=32,GT_read=False));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_publication',count=32,GT_read=False),mutable=True)
    except BaseException as e:
        write(BASE/'FAILURE003.json',dict(status='technical_failure',error=repr(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',stage='003',count=32,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
