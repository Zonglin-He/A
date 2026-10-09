"""Finite original P2 continuation after actual-action reward qualification."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_matched_gradient006 import REC,verify,ALLOCATOR
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PYTHON,read,write,status,sha


def run():
    verify()
    from scripts.stvg_opd_paper_later_common_v1 import phases,authorize_after_root
    assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    assert read(REC/'ROOT_ORIGINAL_PREFIX_READBACK.json')['status']=='pass'
    assert read(BASE/'P2_QUALIFICATION.json')['status']=='pass'
    with (BASE/'later_controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);authorize_after_root('P2','P1')
        def child(job,name,cpu=False):
            verify();cmd=[str(PYTHON),'-B','scripts/run_stvg_opd_p2_matched_gradient006.py',job,name]
            log=REC/'logs'/(job+'_'+name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR}
            if cpu:env['CUDA_VISIBLE_DEVICES']=''
            with log.open('a') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
                state=dict(status='running',phase='P2',controller_pid=os.getpid(),worker_pid=p.pid,
                    command=cmd,log=str(log.relative_to(ROOT)),qualification=False,CPU=cpu,
                    matched_gradient_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time())
                status(BASE/'STAGE.json',state);status(BASE/('LATER_CPU_STAGE.json' if cpu else 'COMPONENT_STAGE.json'),state)
                assert p.wait()==0,'Preserve any new real failure without changing original science'
        if not (BASE/'P2_PREDICTION_BARRIER.json').exists():
            barriers={}
            for name in phases()['P2']:
                barrier=BASE/'stages'/name/'PREDICTION_BARRIER.json'
                if not barrier.exists():child('GPU',name)
                assert read(barrier)['status']=='sealed';barriers[str(barrier.relative_to(BASE))]=sha(barrier)
            write(BASE/'P2_PREDICTION_BARRIER.json',dict(status='sealed',phase='P2',barriers=barriers,
                all_deployment_arms_and_directions=True,GT_read=False,
                runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
                engineering_runtime_sha256=sha(BASE/'P2_engineering_revision002/REVISION_RUNTIME.json'),
                input_schema_runtime_sha256=sha(BASE/'recovery/P2_input_schema_003/REVISION_RUNTIME.json'),
                inline_precision_runtime_sha256=sha(BASE/'recovery/P2_inline_gradient_precision_004/REVISION_RUNTIME.json'),
                actual_action_runtime_sha256=sha(BASE/'recovery/P2_actual_action_reward_precision_005/REVISION_RUNTIME.json'),
                matched_gradient_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
            write(BASE/'P2_GPU_COMPLETION.json',dict(status='all_phase_predictions_sealed',phase='P2',
                barrier_sha256=sha(BASE/'P2_PREDICTION_BARRIER.json'),root_CPU_and_visual_publication_required=True,time=time.time()))
        child('score','P2',cpu=True);child('finalize','P2',cpu=True)
        status(BASE/'STATUS.json',dict(status='P2_CPU_complete_pending_actual_root_visual_publication_then_P3',paper_suite_complete=False,time=time.time()))
        write(REC/'COMPLETION.json',dict(status='P2_CPU_complete_pending_actual_root',paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'controller_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='P2_matched_gradient006_failure_preserved_pending_actual_root',
            evidence=str(p.relative_to(ROOT)),paper_suite_complete=False,time=time.time()));raise
