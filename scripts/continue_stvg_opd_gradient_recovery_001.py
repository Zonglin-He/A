"""Finite engineering recovery: complete eight missing fits then original CPU."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PYTHON,verify,read,write,status,sha
RECOVERY=BASE/'recovery/gradient_audit_001'

def run():
    verify();r=read(RECOVERY/'REVISION_RUNTIME.json')
    for f,h in r['pins'].items():assert sha(ROOT/f)==h
    assert read(RECOVERY/'CPU_QUALIFICATION.json')['status']=='pass'
    lock=(BASE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for job,args,cpu in [('GPU',['P0_hc2'],False),('P0_score',['P0_hc2'],True),('P0_finalize',[],True)]:
        if job=='GPU' and (BASE/'stages/P0_hc2/PREDICTION_BARRIER.json').exists():continue
        if job=='P0_score':
            import scripts.continue_stvg_opd_paper_hc2_revision_v2 as original
            if not (BASE/'P0_PREDICTION_BARRIER.json').exists():original.global_barrier('P0')
            if (BASE/'stages/P0_hc2/CPU_COMPLETION.json').exists():continue
        if job=='P0_finalize' and (BASE/'P0_CPU_COMPLETION.json').exists():continue
        cmd=[str(PYTHON),'-B','scripts/run_stvg_opd_gradient_recovery_001.py',job,*args]
        log=RECOVERY/(job+'.log')
        with log.open('a') as out:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if cpu else None)
            status(BASE/'STAGE.json',dict(status='running_revision001',controller_pid=os.getpid(),worker_pid=p.pid,
                job=job,args=args,command=cmd,log=str(log.relative_to(ROOT)),CPU=cpu,
                engineering_revision_sha256=sha(RECOVERY/'REVISION_RUNTIME.json'),time=time.time()))
            assert p.wait()==0,'Preserve further failure without changing scientific configuration'
    status(BASE/'STATUS.json',dict(status='revised_P0_CPU_complete_pending_actual_root_attribution_visual_publication_and_P1_decision',
        precision_audit_revision001=True,HC2_changed=True,VidSTG_predictions_reused=True,paper_suite_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        d=RECOVERY/'controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='new_recovery_failure_preserved',evidence=str(d),time=time.time()));raise
