"""One bounded, serial, download-dependent Qwen32 controller; no automatic scientific retries."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen35_geometry32_v1 import BASE,read,sha,write,download_status
from scripts.stvg_qwen35_geometry_run_v1 import verify_run
UNIT='stvg-qwen35-geometry32-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify_run();runtime=read(BASE/'EXECUTION_RUNTIME001.json');deadline=runtime['download_wait_deadline_epoch']
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='qwen35_geometry32'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState'],text=True).splitlines() if '=' in x)
    assert int(props['MemoryHigh'])==16*2**30 and int(props['MemoryMax'])==24*2**30 and int(props['MemorySwapMax'])==2**30
    write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=props,GT_read=False))
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        write(BASE/'STATUS.json',dict(status='waiting_download',stage='verified_checkpoint_dependency',count=32,actual_model_calls=0,GT_read=False),mutable=True)
        while not download_status()['snapshot_for_model_ready']:
            verify_run();assert time.time()<deadline,'bounded download wait expired'
            state=subprocess.check_output(['systemctl','--user','show','qwen35-9b-download-side-001.service','-p','ActiveState','--value'],text=True).strip()
            if state=='failed':raise RuntimeError('existing downloader failed; no restart or replacement download')
            time.sleep(20)
        while True:
            assert time.time()<deadline,'resource admission wait expired';v=resource.sample()
            if v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30:break
            write(BASE/'STATUS.json',dict(status='waiting_resources',stage='exclusive_model_admission',count=32,GT_read=False),mutable=True);time.sleep(20)
        write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+8*3600,scientific_wall_hours=8,download_wait_not_scientific_compute=True,GT_read=False))
        for stage in ['teacher','reference_root','student_prepare','student','student_root','propagate','score','report','render']:
            verify_run();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),mutable=True)
            folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False)
            py=ROOT/('.venv/stvg_qwen35_geometry_v1/bin/python' if stage=='teacher' else '.conda/tubedetr/bin/python')
            script='stvg_qwen35_geometry_run_v1.py';args=[stage]
            if stage in ['report','render']:script='report_stvg_qwen35_geometry32_v1.py';args=[stage]
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                tick=time.perf_counter();p=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(folder/'LAUNCH.json',dict(status='running',pid=p.pid,stage=stage,GT_read=stage=='score'));rc=p.wait()
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'))
            assert rc==0,stage+' failed; outputs preserved, no automatic retry'
        write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,independent_actual_pixel_and_remote_review_required=True,GT_read=False))
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_publication',stage='root_publication',count=32,GT_read=False),mutable=True)
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False))
        write(BASE/'STATUS.json',dict(status='failed',count=32,error_type=type(e).__name__,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
