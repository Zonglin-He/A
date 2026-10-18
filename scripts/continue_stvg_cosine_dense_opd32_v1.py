"""Finite CPU-only all32 fits -> new scoring -> audit -> report."""
from pathlib import Path
import sys,os,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_cosine_dense_opd32_v1'

def main():
    f=(D/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);put(D/'EXECUTION_RUNTIME001.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),scope='CPU cached cosine five-arm32 two-round actual fits'))
    for stage in ['measure','score','audit','report']:
        remaining=read(D/'PLAN_LOCK.json')['deadline_epoch']-time.time();assert remaining>0
        script='scripts/run_stvg_cosine_dense_opd32_v1.py'if stage in ['measure','score']else'scripts/audit_stvg_cosine_dense_opd32_v1.py';cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',script,stage]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        with (D/(stage+'.log')).open('ab')as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);put(D/'WORKER_RUNTIME.json',dict(stage=stage,pid=p.pid,start_epoch=time.time(),command=cmd),True);rc=p.wait(timeout=remaining)
        assert rc==0,(stage,rc)
    put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=32,all_three_update_arms_actually_run=True,publication_archive_pending=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),all_partials_preserved=True));raise
