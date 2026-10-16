"""One finite, guarded dispatcher; signals only its own explicitly named unit."""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,verify
from scripts import launch_stvg_instance_identity_v1 as resource

UNIT='stvg-lite-event128-001.service'


def child(stage,script,*args,gpu=False,teacher=False):
    if gpu:
        v=resource.sample();g=1024**3
        assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    resource.STAGE=stage
    out=BASE/'execution'/stage;out.mkdir(parents=True,exist_ok=False)
    interpreter=ROOT/('.venv/stvg_native_distill_v1/bin/python' if teacher else '.conda/tubedetr/bin/python')
    with (out/'worker.log').open('xb') as log:
        p=subprocess.Popen([str(interpreter),'-B','scripts/'+script,*args],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json',dict(status='running',stage=stage,worker_pid=p.pid,controller_pid=os.getpid(),GT_read=False,time=time.time()))
        code=p.wait()
    write(out/'WORKER_EXIT.json',dict(status='exited',stage=stage,exit_code=code,GT_read=False,time=time.time()))
    if code:raise RuntimeError('stage failed: '+stage+' / '+str(code))


def main():
    verify();resource.BASE=BASE/'execution';resource.UNIT=UNIT;resource.STAGE='admission'
    g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
    assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,20*g,g]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for i,trial in enumerate(['L000','L001','L002']):
            write(BASE/'STATUS.json',dict(status='running',stage='spatial_combination_'+trial,done=i,total=3,GT_read=False,time=time.time()),mutable=True)
            child(trial+'_student','run_stvg_lite_event_student_v1.py',trial,gpu=True)
            child(trial+'_root','review_stvg_lite_event_v1.py',trial)
            child(trial+'_score','measure_stvg_lite_event_v1.py',trial)
        if '--spatial-only' in sys.argv:
            write(BASE/'STATUS.json',dict(status='phase_complete_pending_temporal_event_LR_reclocked_confirmation',stage='spatial_combination_root_handoff',done=1,total=5,GT_read=False,time=time.time()),mutable=True)
            return
        child('temporal_combination','run_stvg_temporal_lite128_v1.py')
        child('temporal_root','readback_stvg_temporal_lite128_v1.py')
        child('temporal_score','score_stvg_temporal_lite128_v1.py')
        child('structural_selection','select_stvg_lite_event_v1.py')
        write(BASE/'STATUS.json',dict(status='phase_complete_pending_event_LR_reclocked_confirmation',stage='structural_root_handoff',done=2,total=5,GT_read=False,time=time.time()),mutable=True)
    finally:stop.set()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='failed',stage='finite_dispatch',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True)
        write(BASE/'STATUS.json',dict(status='failed',stage='root_engineering_review',GT_read=False,time=time.time()),mutable=True)
        raise
