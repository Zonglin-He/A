"""Serial no-update Source256 capture and all-seal scoring, bounded once."""
from pathlib import Path
import fcntl,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import put,read
D=ROOT/'artifacts/stvg_dino_supported_source256_v1'


def main():
    f=(D/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(D/'EXECUTION_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),scope='frozen Source/A/E256; zero Student updates'))
    stages=[('capture','hc2','SOURCE_BARRIER_hc2.json'),('capture','vidstg','SOURCE_BARRIER_vidstg.json'),('predict',None,'GLOBAL_PREDICTION_BARRIER.json'),('score',None,'CPU_COMPLETION.json'),('finalize',None,'ROOT_READBACK_COMPLETION.json')]
    for stage,ds,bar in stages:
        if (D/bar).exists():continue
        remain=read(D/'PLAN_LOCK.json')['deadline_epoch']-time.time();assert remain>0,'Original8h deadline'
        cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B']
        cmd+=['scripts/finalize_stvg_dino_supported_source256_v1.py']if stage=='finalize'else['scripts/run_stvg_dino_supported_source256_v1.py',stage]
        if ds:cmd+=['--dataset',ds]
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if stage in ['predict','score','finalize']:env['CUDA_VISIBLE_DEVICES']=''
        with (D/(stage+'_'+str(ds)+'.log')).open('ab')as out:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            put(D/'WORKER_RUNTIME.json',dict(status='running',stage=stage,dataset=ds,pid=p.pid,command=cmd,start_epoch=time.time()),True)
            rc=p.wait(timeout=remain)
        if rc:raise RuntimeError(f'{stage}/{ds} failed exit{rc}; all records preserved')
        assert (D/bar).exists()
    put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=256,root_pixels_privacy_publication_archive_pending=True))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
