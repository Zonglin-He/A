"""Finite original P5 qualification/root/serial deployment/CPU handoff."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p5_budget_precision001 import verify,REC,ALLOCATOR
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PYTHON,read,write,status,sha


def run():
    verify()
    from scripts.stvg_opd_paper_later_common_v1 import phases,authorize_after_root
    with (BASE/'later_controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);authorize_after_root('P5','P4')
        def child(job,name,cpu=False):
            verify();cmd=[str(PYTHON),'-B','scripts/run_stvg_opd_p5_budget_precision001.py',job,name]
            log=REC/'logs'/(job+'_'+name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR}
            if cpu:env['CUDA_VISIBLE_DEVICES']=''
            with log.open('a') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
                record=dict(status='running',phase='P5',controller_pid=os.getpid(),worker_pid=p.pid,
                    command=cmd,log=str(log.relative_to(ROOT)),qualification=job=='qualification',CPU=cpu,
                    scope_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),GT_read=False if job not in ['score','finalize'] else 'post_global_seal',time=time.time())
                status(BASE/'STAGE.json',record);status(BASE/('LATER_CPU_STAGE.json' if cpu else 'COMPONENT_STAGE.json'),record)
                status(BASE/'STATUS.json',dict(record,paper_suite_complete=False))
                assert p.wait()==0,'Preserve new actual failure; do not change original science'
        if not (BASE/'P5_QUALIFICATION.json').exists():
            for name in phases()['P5']:
                path=REC/'qualification'/name/'BITWISE_CONTROLS.json'
                if not path.exists():child('qualification',name)
                assert read(path)['status']=='pass'
            child('root_qualification','P5',True)
        assert read(BASE/'P5_QUALIFICATION.json')['status']=='pass'
        assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
        if not (BASE/'P5_PREDICTION_BARRIER.json').exists():
            barriers={}
            for name in phases()['P5']:
                p=BASE/'stages'/name/'PREDICTION_BARRIER.json'
                if not p.exists():child('GPU',name)
                assert read(p)['status']=='sealed';barriers[str(p.relative_to(BASE))]=sha(p)
            write(BASE/'P5_PREDICTION_BARRIER.json',dict(status='sealed',phase='P5',barriers=barriers,
                all_deployment_arms_and_directions=True,GT_read=False,
                original_component_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
                parameter_scope_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
            write(BASE/'P5_GPU_COMPLETION.json',dict(status='all_phase_predictions_sealed',phase='P5',
                barrier_sha256=sha(BASE/'P5_PREDICTION_BARRIER.json'),root_CPU_and_view_publication_required=True,time=time.time()))
        child('score','P5',True);child('finalize','P5',True)
        status(BASE/'STATUS.json',dict(status='P5_CPU_complete_pending_actual_root_visual_publication_then_P6',paper_suite_complete=False,time=time.time()))
        write(REC/'COMPLETION.json',dict(status='P5_CPU_complete_pending_actual_root',paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'controller_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='P5_scope001_failure_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),paper_suite_complete=False,time=time.time()));raise
