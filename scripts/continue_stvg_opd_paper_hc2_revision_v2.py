"""Finite continuation after actual HC2 tuning closure; hands back for root reviews."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def child(job,*args,cpu=False):
    verify();command=[str(PYTHON),'-B','scripts/run_stvg_opd_paper_hc2_revision_v2.py',job,*args]
    log=BASE/'logs'/('_'.join([job,*args])+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as out:
        p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if cpu else None)
        status(BASE/'STAGE.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,
            job=job,args=list(args),command=command,log=str(log.relative_to(ROOT)),CPU=cpu,time=time.time()))
        assert p.wait()==0,'Preserve failed stage and original lock; root must diagnose before a repair.'

def global_barrier(phase):
    barriers={f'stages/{phase}_{ds}/PREDICTION_BARRIER.json':sha(BASE/f'stages/{phase}_{ds}/PREDICTION_BARRIER.json') for ds in ['hc2','vidstg']}
    write(BASE/(phase+'_PREDICTION_BARRIER.json'),dict(status='sealed',barriers=barriers,
        all_deployment_arms_both_directions=True,all_deployment_OPD_directions=True,
        adapted_arrivals=1536 if phase=='P0' else 41355,GT_read=False,time=time.time()))
    import scripts.audit_stvg_opd_revision_aliases_v2 as audit;audit.run(phase)

def run(phase='P0'):
    prepare();lease=(BASE/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d=read(BASE/'DESIGN_LOCK.json')
    if phase=='P0':
        if not (BASE/'QUALIFICATION.json').exists():
            if d['HC2_changed']:
                child('GPU','P0_hc2','qualification');hc=read(BASE/'qualification/P0_hc2/QUALIFICATION.json')
            else:hc=read(OLD/'qualification/P0_hc2/QUALIFICATION.json')
            vid=read(OLD/'qualification/P0_vidstg/QUALIFICATION.json');assert hc['status']==vid['status']=='pass'
            write(BASE/'QUALIFICATION.json',dict(status='pass',datasets=[hc,vid],
                actual_new_GPU_fits=6 if d['HC2_changed'] else 0,logical_qualification_fits=12,
                unchanged_VidSTG_qualification_reused=True,GT_read=False,time=time.time()))
        if not (BASE/'stages/P0_vidstg/PREDICTION_BARRIER.json').exists():reuse_stage('P0_vidstg')
        if not (BASE/'stages/P0_hc2/PREDICTION_BARRIER.json').exists():
            child('GPU','P0_hc2') if d['HC2_changed'] else reuse_stage('P0_hc2')
        if not (BASE/'P0_PREDICTION_BARRIER.json').exists():global_barrier('P0')
        for ds in ['hc2','vidstg']:
            if not (BASE/f'stages/P0_{ds}/CPU_COMPLETION.json').exists():child('P0_score','P0_'+ds,cpu=True)
        if not (BASE/'P0_CPU_COMPLETION.json').exists():child('P0_finalize',cpu=True)
        status(BASE/'STATUS.json',dict(status='revised_P0_CPU_complete_pending_actual_root_attribution_visual_publication_and_P1_decision',
            HC2_changed=d['HC2_changed'],VidSTG_predictions_reused=True,original_paper_continuation_authorized=True,
            paper_suite_complete=False,time=time.time()))
    elif phase=='P1':
        assert read(BASE/'P0_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
        assert read(BASE/'P1_LAUNCH_AUTHORIZATION.json')['root_decision_after_actual_revised_P0_readback']
        if not (BASE/'P1_EXACT_PREFIX_REUSE.json').exists():reuse_unchanged_P1_prefix()
        for ds in ['hc2','vidstg']:child('GPU','P1_'+ds)
        if not (BASE/'P1_PREDICTION_BARRIER.json').exists():global_barrier('P1')
        write(BASE/'P1_GPU_COMPLETION.json',dict(status='all_revised_P1_OPD_predictions_sealed',time=time.time()))
        for ds in ['hc2','vidstg']:child('P1_score','P1_'+ds,cpu=True)
        child('P1_assemble',cpu=True);child('P1_finalize',cpu=True)
        status(BASE/'STATUS.json',dict(status='P1_revised_table_CPU_complete_pending_actual_root_visual_publication_then_P2_P6',paper_suite_complete=False,time=time.time()))
    else:raise ValueError(phase)

if __name__=='__main__':
    try:run(sys.argv[1] if len(sys.argv)>1 else 'P0')
    except BaseException:
        d=BASE/'controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved',evidence=str(d.relative_to(ROOT)),time=time.time()));raise
