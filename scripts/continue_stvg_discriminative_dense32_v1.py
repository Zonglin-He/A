"""Finite serial capture/feedback/gate; actual updates only on locked pass."""
from pathlib import Path
import os,sys,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_discriminative_dense32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/discriminative_dense32'

def run(stage,audit=False):
    script='scripts/audit_stvg_discriminative_dense32_v1.py'if audit else'scripts/run_stvg_discriminative_dense32_v1.py';remaining=read(D/'PLAN_LOCK.json')['deadline_epoch']-time.time();assert remaining>0
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if stage!='capture_features':env['CUDA_VISIBLE_DEVICES']=''
    cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',script,stage]
    with (D/(stage+'.log')).open('ab')as log:
        worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);put(D/'WORKER_RUNTIME.json',dict(stage=stage,pid=worker.pid,start_epoch=time.time(),command=cmd),True);rc=worker.wait(timeout=remaining)
    assert rc==0,(stage,rc)

def main():
    f=(D/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);put(D/'EXECUTION_RUNTIME001.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),scope='ridge-feedback32 then conditional two rounds'))
    for stage in ['capture_features','feedback','score_phase1']:run(stage)
    if read(P/'GATE.json')['passed']:
        for stage in ['adapt','score_phase2']:run(stage)
    else:put(D/'PHASE2_NOT_STARTED.json',dict(status='not_started_by_prelocked_stop_rule',reason='new teacher did not pass both-domain ranking controls',no_optimizer_search=True))
    run('audit',True);run('report',True);put(D/'CONTROLLER_COMPLETION.json',dict(status='success',count=32,phase2_ran=read(P/'GATE.json')['passed'],publication_archive_pending=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
