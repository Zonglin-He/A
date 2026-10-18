"""One finite stage1 pipeline, no retry, queue restoration or stage2 auto-launch."""
from pathlib import Path
import os,sys,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_e_tube_critic64_v1'


def main():
    lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(D/'EXECUTION_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time()))
    deadline=read(D/'PLAN_LOCK.json')['deadline_epoch'];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for stage,cmd in [('measure',['scripts/run_stvg_e_tube_critic64_v1.py','measure']),('score',['scripts/score_stvg_e_tube_critic64_v1.py'])]:
        with(D/(stage+'.log')).open('ab')as log:
            p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            put(D/('WORKER_'+stage+'.json'),dict(pid=p.pid,start_epoch=time.time()))
            try:rc=p.wait(timeout=max(.1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                p.terminate();p.wait(timeout=15);raise RuntimeError('stage1 wall cap reached')
        assert rc==0,(stage,rc)
    assert time.time()<deadline
    put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=64,decision=read(D/'STAGE1_DECISION.json'),root_audit_public_archive_pending=True))


if __name__=='__main__':
    try:main()
    except BaseException as e:put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),partials_preserved=True));raise
