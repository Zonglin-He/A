"""One serial8h R1 interface correction, unchanged Student, no parameter search."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_direct_reliable32_v1 import BASE,verify,put,read
UNIT='stvg-r1-direct-reliable32-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();assert read(ROOT/'artifacts/stvg_qwen35_geometry32_v1/STATUS.json')['status']=='paused_by_user'
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='r1_direct_reliable32';v=resource.sample();g=2**30
    assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID'],text=True).splitlines() if '=' in x)
    assert int(props['MemoryHigh'])==16*g and int(props['MemoryMax'])==20*g and int(props['MemorySwapMax'])==g
    put(BASE/'ACTUAL_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=props,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ['teacher','teacher_root','prepare_student','worker','root','propagate','score','report','render']:
            verify();put(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),replace=True)
            f=BASE/'execution'/stage;f.mkdir(parents=True,exist_ok=False);py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if stage=='teacher' else '.conda/tubedetr/bin/python')
            script='report_stvg_r1_direct_reliable32_v1.py' if stage in ['report','render'] else 'stvg_r1_direct_reliable32_v1.py'
            with (f/'worker.log').open('xb') as log:
                tick=time.perf_counter();p=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
                put(f/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
            put(f/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False));put(BASE/'STATUS.json',dict(status='measurement_complete_pending_publication',count=32,GT_read=False),replace=True)
    except BaseException as e:put(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
    finally:stop.set()
if __name__=='__main__':main()
