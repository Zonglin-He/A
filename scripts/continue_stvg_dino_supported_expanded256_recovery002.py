"""Single bounded serial recovery; never starts historical controllers."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import put,read
D=ROOT/'artifacts/stvg_dino_supported_expanded256_v1'


def main():
    lock=(D/'controller002.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stages=[('qualify','hc2','QUALIFICATION_hc2.json'),('qualify','vidstg','QUALIFICATION_vidstg.json'),
        ('replay','hc2','BEFORE_BARRIER_hc2.json'),('replay','vidstg','BEFORE_BARRIER_vidstg.json'),
        ('predict',None,'GLOBAL_PREDICTION_BARRIER.json'),('score',None,'CPU_COMPLETION.json')]
    put(D/'EXECUTION_RUNTIME002.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),
        argv=sys.argv,authorization='RECOVERY_AUTHORIZATION002.json',scope='original order1 recovery then frozen Before/A/E expanded256'))
    for stage,ds,bar in stages:
        if (D/bar).exists():continue
        remain=read(D/'RECOVERY_AUTHORIZATION002.json')['deadline_epoch']-time.time()
        assert remain>0,'Original48h deadline reached; partial state preserved'
        cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_dino_supported_expanded256_recovery002.py',stage]
        if ds:cmd+=['--dataset',ds]
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if stage in ['score','predict']:env['CUDA_VISIBLE_DEVICES']=''
        with (D/(stage+'_'+str(ds)+'002.log')).open('ab')as out:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            put(D/'WORKER_RUNTIME002.json',dict(status='running',stage=stage,dataset=ds,pid=p.pid,command=cmd,start_epoch=time.time()),True)
            rc=p.wait(timeout=remain)
        if rc:raise RuntimeError(f'{stage} {ds} exit {rc}; original logs and predictions preserved')
        assert (D/bar).exists(),f'Missing required stage barrier {bar}'
    put(D/'CONTROLLER_COMPLETION002.json',dict(status='success',count=256,root_publication_pending=True))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        put(D/('CONTROLLER_FAILURE002_'+str(time.time_ns())+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),all_partials_preserved=True))
        raise
