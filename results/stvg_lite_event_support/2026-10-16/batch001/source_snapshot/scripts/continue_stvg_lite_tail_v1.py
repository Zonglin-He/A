"""One pidfd-bound finite continuation. No recurring scheduler or retry loop."""
from pathlib import Path
import ctypes
import os
import select
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,verify
PY=ROOT/'.conda/tubedetr/bin/python'
EXEC=BASE/'execution_tail001'


def verify_tail():
    verify();r=read(BASE/'FINITE_TAIL_RUNTIME.json');assert r['status']=='locked' and not r['GT_read']
    for rel,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/rel)==h,rel
    return r


def call(stage,script,*args):
    b=EXEC/stage;b.mkdir(parents=True,exist_ok=False)
    with(b/'worker.log').open('xb')as log:
        p=subprocess.Popen([str(PY),'-B','scripts/'+script,*args],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        write(b/'LAUNCH.json',dict(status='running',stage=stage,worker_pid=p.pid,GT_read=False,time=time.time()))
        code=p.wait()
    write(b/'WORKER_EXIT.json',dict(status='exited',exit_code=code,GT_read=False,time=time.time()))
    if code:raise RuntimeError(stage+' failed '+str(code))


def wait_first(rt):
    parent=rt['parent'];unit=parent['unit'];pid=parent['pid']
    properties=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState,MainPID,Result'],text=True).splitlines()if'='in x)
    if properties['ActiveState']=='active':
        assert int(properties['MainPID'])==pid
        stat=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split();assert int(stat[19])==parent['start_ticks']
        assert sha(Path('/proc')/str(pid)/'cmdline')==parent['cmdline_sha256']
        libc=ctypes.CDLL(None,use_errno=True);libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint];libc.pidfd_open.restype=ctypes.c_int
        fd=libc.pidfd_open(pid,0)
        if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
        try:
            write(BASE/'TAIL_STATUS.json',dict(status='waiting',stage='three_spatial_trials_and_source_bound_exit',pid=pid,GT_read=False,time=time.time()),mutable=True)
            select.select([fd],[],[])
        finally:os.close(fd)
    assert read(BASE/'execution/L002_score/WORKER_EXIT.json')['exit_code']==0
    for tid in ('L000','L001','L002'):
        for name,status in [('ROOT_READBACK.json','pass'),('CPU_COMPLETION.json','complete')]:
            d=read(BASE/'trials'/tid/name);assert d['status']==status and d['count']==128


def gpu_stage(unit,script,*args):
    subprocess.run(['systemd-run','--user','--wait','--collect','--unit='+unit,'--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),'--setenv=PYTHONPATH='+str(ROOT),'--setenv=PYTHONUNBUFFERED=1','--setenv=OMP_NUM_THREADS=2','--setenv=OPENBLAS_NUM_THREADS=2',str(PY),'-B','scripts/'+script,*args],cwd=ROOT,check=True)


def main():
    rt=verify_tail();EXEC.mkdir(exist_ok=False);wait_first(rt)
    write(BASE/'TAIL_STATUS.json',dict(status='running',stage='temporal_combination',GT_read=False,time=time.time()),mutable=True)
    call('temporal','run_stvg_temporal_lite128_v1.py','--run')
    call('temporal_root','readback_stvg_temporal_lite128_v1.py')
    call('temporal_score','score_stvg_temporal_lite128_v1.py','--score')
    call('structural_selection','select_stvg_lite_event_v1.py')
    write(BASE/'TAIL_STATUS.json',dict(status='running',stage='one_event_support_and_five_LR',GT_read=False,time=time.time()),mutable=True)
    gpu_stage('stvg-lite-event-final-search-001','continue_stvg_lite_event_search_v1.py','--dispatch')
    d=read(BASE/'FINAL_RECIPE_LOCK.json');assert d['status']=='locked' and not d['suite_final']
    write(BASE/'TAIL_STATUS.json',dict(status='running',stage='reclocked_full_loop_and_disjoint_confirmation',GT_read=False,time=time.time()),mutable=True)
    gpu_stage('stvg-lite-complete-001','continue_stvg_lite_complete_v1.py')
    assert read(BASE/'MEASUREMENT_COMPLETION.json')['status']=='complete_pending_independent_study_view_public_archive'
    write(BASE/'TAIL_STATUS.json',dict(status='complete_pending_independent_study_view_public_archive',stage='root_handoff',GT_read=True,time=time.time()),mutable=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'TAIL_FAILURE.json',dict(status='failed',stage='required_root_review',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True)
        write(BASE/'TAIL_STATUS.json',dict(status='failed',stage='required_root_review',GT_read=False,time=time.time()),mutable=True)
        raise
