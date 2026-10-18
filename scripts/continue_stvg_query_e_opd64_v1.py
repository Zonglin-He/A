"""One bounded measurement subprocess; CPU score remains gated by full seal."""
from pathlib import Path
import os,sys,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_query_e_opd64_v1'


def main():
    lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(D/'EXECUTION_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),scope='Source-reset frozen decoder query-only E64 four updater arms'))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with(D/'measure.log').open('ab')as log:
        p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_query_e_opd64_v1.py','measure'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        put(D/'WORKER_RUNTIME.json',dict(status='running',stage='measure',pid=p.pid,start_epoch=time.time()))
        rc=p.wait(timeout=max(1,read(D/'PLAN_LOCK.json')['deadline_epoch']-time.time()))
    assert rc==0,('measurement failed',rc);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==64
    put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=64,root_CPU_score_and_independent_publication_pending=True))


if __name__=='__main__':
    try:main()
    except BaseException as e:put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),partials_preserved=True));raise
