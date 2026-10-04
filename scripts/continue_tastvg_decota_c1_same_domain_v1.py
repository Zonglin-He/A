"""One finite serial GPU/CPU chain. No recurring polling or historical queues."""
import sys, os, time, traceback, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_c1_common_v1 import *


def call(stage, ds=None):
    cmd=['bash','scripts/with_local_cuda.sh',str(ROOT/'.conda/tubedetr/bin/python'),'-B',
        'scripts/run_tastvg_decota_c1_same_domain_v1.py',stage]
    if ds:cmd.append(ds)
    log=BASE/(f'{ds}_{stage}.log' if ds else f'{stage}.log')
    assert not log.exists(),log
    with log.open('w') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,
            controller_pid=os.getpid(),worker_pid=child.pid,log=str(log),GT_read=False,time=time.time()))
        assert child.wait()==0,('stage failed',stage,ds,child.returncode)


def run():
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    handle=(BASE/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    write(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv,finite=True,duplicate_launch_forbidden=True))
    try:
        for ds in DATASETS:
            call('evidence',ds);call('online',ds)
        call('seal')
        cpu=read(BASE/'CPU_RUNTIME_LOCK.json')
        for f,h in cpu['pins'].items():assert sha(ROOT/f)==h,f
        with (BASE/'CPU.log').open('w') as f:
            cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/score_audit_tastvg_decota_c1_same_domain_v1.py','run']
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            status(BASE/'STATUS.json',dict(status='CPU_scoring_audit',controller_pid=os.getpid(),worker_pid=child.pid,time=time.time()))
            assert child.wait()==0,('CPU failed',child.returncode)
        subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/report_tastvg_decota_c1_same_domain_v1.py'],cwd=ROOT,check=True)
        status(BASE/'STATUS.json',dict(status='completed_pending_root_visual_publication',arrivals=1152,time=time.time()))
    except BaseException as e:
        status(BASE/'CONTROLLER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e),time=time.time()))
        raise
    finally:handle.close()


if __name__=='__main__':run()
