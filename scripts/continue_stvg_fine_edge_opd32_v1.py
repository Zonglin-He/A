"""One bounded CPU actual OPD chain only; no old queue or model training."""
from pathlib import Path
import os,sys,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_fine_edge_opd32_v1'

def main():
    f=(D/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(D/'EXECUTION_RUNTIME001.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),scope='Ordered ROI layout feedback actual OPD32'))
    stages=[('measure',None,'GLOBAL_PREDICTION_BARRIER.json'),
        ('score',None,'CPU_SCORE_COMPLETION.json')]
    for stage,ds,bar in stages:
        assert not (D/bar).exists(),('preserve completed stage; no automatic recovery',stage)
        remaining=read(D/'PLAN_LOCK.json')['deadline_epoch']-time.time();assert remaining>0
        script='scripts/run_stvg_fine_edge_opd32_v1.py'
        cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',script]
        cmd.append(stage)
        if ds:cmd+=['--dataset',ds]
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if stage in ['E','score','audit','report']:env['CUDA_VISIBLE_DEVICES']=''
        with (D/(stage+'_'+str(ds)+'.log')).open('ab')as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            put(D/'WORKER_RUNTIME.json',dict(status='active',stage=stage,dataset=ds,pid=p.pid,command=cmd,start_epoch=time.time()),True)
            rc=p.wait(timeout=remaining)
        assert rc==0,(stage,ds,rc);assert (D/bar).exists(),bar
    put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=32,scientific_finite_chain_only=True,publication_archive_pending=True))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
