"""One genuinely authorized phase after its preceding actual root/public closure."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def run(phase):
    activate()
    import scripts.stvg_opd_paper_later_common_v1 as c
    c.verify_later();assert phase in c.phases()
    previous={'P2':'P1','P3':'P2','P4':'P3','P5':'P4'}[phase]
    with (BASE/'later_controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);c.authorize_after_root(phase,previous)
        def child(job,name,qual=False,cpu=False):
            c.verify_later();command=[str(PYTHON),'-B','scripts/run_stvg_opd_revision_later_v2.py',job,name]
            if qual:command.append('qualification')
            log=BASE/'logs'/('_'.join([phase,job,name])+('_qualification' if qual else '')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as out:
                p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if cpu else None)
                status(BASE/('LATER_CPU_STAGE.json' if cpu else 'COMPONENT_STAGE.json'),dict(status='running',phase=phase,
                    controller_pid=os.getpid(),worker_pid=p.pid,command=command,log=str(log.relative_to(ROOT)),qualification=qual,CPU=cpu,time=time.time()))
                assert p.wait()==0,'Preserve failed stage; root engineering diagnosis required, no duplicate GPU.'
        names=c.phases()[phase]
        if not (BASE/(phase+'_QUALIFICATION.json')).exists():
            records=[]
            for name in names:
                child('GPU',name,True);p=BASE/'component_qualification'/name/'QUALIFICATION.json';r=read(p);assert r['status']=='pass'
                records.append(dict(stage=name,qualification_sha256=sha(p),actual_fits=r['actual_fits'],
                    actual_updates=sum(x['actual_backward_rounds'] for x in r['records']),GT_read=False))
            write(BASE/(phase+'_QUALIFICATION.json'),dict(status='pass',actual_GPU_fits=sum(x['actual_fits'] for x in records),
                stages=records,includes_all_physical_conditions=True,synthetic_only=False,GT_read=False,time=time.time()))
        if not (BASE/(phase+'_PREDICTION_BARRIER.json')).exists():
            barriers={}
            for name in names:
                child('GPU',name);p=BASE/'stages'/name/'PREDICTION_BARRIER.json';assert read(p)['status']=='sealed';barriers[str(p.relative_to(BASE))]=sha(p)
            write(BASE/(phase+'_PREDICTION_BARRIER.json'),dict(status='sealed',phase=phase,barriers=barriers,
                all_deployment_arms_and_directions=True,GT_read=False,runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),time=time.time()))
            write(BASE/(phase+'_GPU_COMPLETION.json'),dict(status='all_phase_predictions_sealed',phase=phase,
                barrier_sha256=sha(BASE/(phase+'_PREDICTION_BARRIER.json')),root_CPU_and_visual_publication_required=True,time=time.time()))
        child('score',phase,cpu=True);child('finalize',phase,cpu=True)
        status(BASE/'STATUS.json',dict(status=phase+'_CPU_complete_pending_actual_root_visual_publication_then_next_phase',paper_suite_complete=False,time=time.time()))

if __name__=='__main__':
    try:run(sys.argv[1])
    except BaseException:
        d=BASE/'later_controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'COMPONENT_STAGE.json',dict(status='failed_preserved',evidence=str(d.relative_to(ROOT)),time=time.time()));raise
