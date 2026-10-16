"""One finite engineering continuation; reuses all completed L001 GPU work."""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,make_trial
from scripts.stvg_lite_recovery004_common import verify_recovery004 as verify_recovery
from scripts import launch_stvg_instance_identity_v1 as resource
UNIT='stvg-lite-recovery004.service'
EXEC=BASE/'execution_recovery004'
PY=ROOT/'.conda/tubedetr/bin/python'


def stage(name,script,*args,gpu=False):
    verify_recovery()
    if gpu:
        v=resource.sample();g=1024**3
        assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    resource.STAGE=name;out=EXEC/name;out.mkdir(parents=True,exist_ok=False)
    write(BASE/'RECOVERY_STATUS004.json',dict(status='running',stage=name,GT_read=False,time=time.time()),mutable=True)
    with(out/'worker.log').open('xb')as log:
        p=subprocess.Popen([str(PY),'-B','scripts/'+script,*args],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json',dict(status='running',stage=name,worker_pid=p.pid,controller_pid=os.getpid(),GT_read=False,time=time.time()))
        code=p.wait()
    write(out/'WORKER_EXIT.json',dict(status='exited',stage=name,exit_code=code,GT_read=False,time=time.time()))
    if code:raise RuntimeError(name+' failed '+str(code))


def unit(name,script,*args):
    subprocess.run(['systemd-run','--user','--wait','--collect','--unit='+name,'--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),'--setenv=PYTHONPATH='+str(ROOT),'--setenv=PYTHONUNBUFFERED=1','--setenv=OMP_NUM_THREADS=2','--setenv=OPENBLAS_NUM_THREADS=2',str(PY),'-B','scripts/'+script,*args],cwd=ROOT,check=True)


def main():
    verify_recovery();EXEC.mkdir(exist_ok=False)
    resource.BASE=EXEC;resource.UNIT=UNIT;resource.STAGE='reader002'
    g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
    assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for tid in ('L000','L001','L002'):
            assert read(BASE/'trials'/tid/'ROOT_READBACK.json')['status']=='pass'
            d=read(BASE/'trials'/tid/'CPU_COMPLETION.json');assert d['status']=='complete' and d['count']==128
        write(BASE/'STATUS.json',dict(status='running',stage='structural_schema004',done=1,total=5,GT_read=False,time=time.time()),mutable=True)
        for n,st in [('ROOT_READBACK.json','pass'),('CPU_COMPLETION.json','complete')]:
            d=read(BASE/'temporal/revision003'/n);assert d['status']==st and d['count']==128
        stage('structure','select_stvg_lite_event004.py')
        write(BASE/'RECOVERY_STATUS004.json',dict(status='running',stage='event_support_and_five_LR',GT_read=False,time=time.time()),mutable=True)
        unit('stvg-lite-event-final-search-001','continue_stvg_lite_event_search004.py','--dispatch')
        assert read(BASE/'FINAL_RECIPE_LOCK.json')['status']=='locked'
        write(BASE/'RECOVERY_STATUS004.json',dict(status='running',stage='reclocked_complete_and_confirmation',GT_read=False,time=time.time()),mutable=True)
        unit('stvg-lite-complete-001','continue_stvg_lite_complete004.py')
        assert read(BASE/'MEASUREMENT_COMPLETION.json')['status']=='complete_pending_independent_study_view_public_archive'
        write(BASE/'RECOVERY_STATUS004.json',dict(status='complete_pending_independent_study_view_public_archive',stage='root_handoff',GT_read=True,time=time.time()),mutable=True)
    finally:stop.set()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'RECOVERY_FAILURE004.json',dict(status='failed',stage='required_root_review',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True)
        write(BASE/'RECOVERY_STATUS004.json',dict(status='failed',stage='required_root_review',GT_read=False,time=time.time()),mutable=True)
        raise
