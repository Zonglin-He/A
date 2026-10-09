"""Finite original P2: real controls and root qualification precede deployment.

The scientific stage definitions and original runtime stay immutable; all
deployment arms/directions must seal before the original CPU scoring path.
"""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_engineering002 import REC,verify_revision,ALLOCATOR
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PYTHON,read,write,status,sha


def run():
    verify_revision()
    from scripts.stvg_opd_paper_later_common_v1 import phases,authorize_after_root
    with (BASE/'later_controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);authorize_after_root('P2','P1')
        def child(job,name,qualification=False,cpu=False):
            verify_revision()
            command=[str(PYTHON),'-B','scripts/run_stvg_opd_p2_engineering002.py',job,name]
            if qualification:command.append('qualification')
            log=REC/'logs'/(job+'_'+name+('_qualification' if qualification else '')+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR}
            if cpu:env['CUDA_VISIBLE_DEVICES']=''
            with log.open('a') as out:
                p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
                state=dict(status='running',phase='P2',controller_pid=os.getpid(),worker_pid=p.pid,
                    command=command,log=str(log.relative_to(ROOT)),qualification=qualification,CPU=cpu,
                    engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time())
                status(BASE/('LATER_CPU_STAGE.json' if cpu else 'COMPONENT_STAGE.json'),state)
                status(BASE/'STAGE.json',state)
                assert p.wait()==0,'Preserve actual failure; no automatic science change or duplicate GPU'
        names=phases()['P2']
        if not (BASE/'P2_QUALIFICATION.json').exists():
            records=[]
            for name in names:
                root=REC/'qualification'/name/'ROOT_READBACK.json'
                if not root.exists():
                    if not (BASE/'component_qualification'/name/'QUALIFICATION.json').exists():child('GPU',name,True)
                    child('root_qualification',name,cpu=True)
                r=read(root);assert r['status']=='pass'
                controls=read(root.parent/'BITWISE_CONTROLS.json')
                records.append(dict(stage=name,actual_fit_arrivals=r['actual_fits'],actual_GPU_fits=controls['actual_GPU_fits'],
                    root_readback_sha256=sha(root),qualification_sha256=sha(BASE/'component_qualification'/name/'QUALIFICATION.json'),
                    bitwise_controls_sha256=sha(root.parent/'BITWISE_CONTROLS.json'),GT_read=False))
            write(BASE/'P2_QUALIFICATION.json',dict(status='pass',scope='all original P2 arms/conditions actual old/new GPU controls and root readback',
                actual_GPU_fits=sum(r['actual_GPU_fits'] for r in records),stages=records,
                includes_all_physical_conditions=True,synthetic_only=False,GT_read=False,
                engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
        if not (BASE/'P2_PREDICTION_BARRIER.json').exists():
            barriers={}
            for name in names:
                child('GPU',name);p=BASE/'stages'/name/'PREDICTION_BARRIER.json'
                assert read(p)['status']=='sealed';barriers[str(p.relative_to(BASE))]=sha(p)
            write(BASE/'P2_PREDICTION_BARRIER.json',dict(status='sealed',phase='P2',barriers=barriers,
                all_deployment_arms_and_directions=True,GT_read=False,
                runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
                engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
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
        status(BASE/'STATUS.json',dict(status='P2_failure_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),paper_suite_complete=False,time=time.time()));raise
