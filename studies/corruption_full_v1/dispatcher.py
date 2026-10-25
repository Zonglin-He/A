"""Wait for the existing full-TTA owner; no stdlib queue module shadowing."""
from common import *
import subprocess, traceback, fcntl

def run_stage(ds, name, stage):
    root=unit(ds,name);root.mkdir(parents=True,exist_ok=True)
    assert not(root/(stage+'_child.json')).exists(),'Preserve prior stage attempt; no blind restart'
    resources()
    args=[sys.executable,'-B',str(S/'run.py'),stage,'--dataset',ds,'--condition',name]
    env={**os.environ}
    if stage!='predict':
        args=['taskset','-c','14,15','nice','-n','19','ionice','-c','3',*args]
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/(stage+'.log')).open('ab')as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        w=identity(child.pid);atomic(root/(stage+'_child.json'),w)
        while child.poll()is None:
            atomic(D/'latest_monitor.json',dict(status=stage,dataset=ds,condition=name,controller=identity(),
                child=w,progress=read(root/'progress.json')if(root/'progress.json').exists()else None,
                disk_free=shutil.disk_usage(R).free,time=time.time()))
            time.sleep(5)
        atomic(root/(stage+'_exit.json'),dict(code=child.returncode,worker=w,time=time.time()))
        assert child.returncode==0,f'Preserve {ds}/{name}/{stage} outputs; no automatic retry'

def queue():
    p=validate();assert read(D/'preflight_receipt.json')['status']=='passed'
    assert not(D/'controller.json').exists()
    leasefile=(D/'queue.lock').open('a');fcntl.flock(leasefile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    who=identity();atomic(D/'controller.json',who)
    parent=read(R/'outputs/tta_full_v1/controller.json')
    assert parent==dict(pid=1924421,startticks=245882503)
    atomic(D/'launch.json',dict(controller=who,parent_controller=parent,fingerprint=p['fingerprint'],
                              GT_read=False,new_Source=0,new_expert=0,time=time.time()))
    try:
        while not exited(parent):
            state=R/'outputs/tta_full_v1/state.json'
            if state.exists():assert read(state).get('status')!='failed_preserved','Parent failed; preserve queue, do not bypass'
            atomic(D/'state.json',dict(status='waiting_existing_fullTTA_and_structural_owner_without_GPU_lock',
                                      controller=who,parent=parent,new_Source=0,new_expert=0,completed_units=0,time=time.time()))
            atomic(D/'latest_monitor.json',read(D/'state.json'));time.sleep(30)
        state=read(R/'outputs/tta_full_v1/state.json')
        assert state['status']=='all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication'
        parents={}
        for ds in COUNTS:
            for path in [R/f'outputs/tta_full_v1/{ds}_reporting_receipt.json',
                         *[ns.D/f'{task}_{ds}_reporting_receipt.json'for task in ['components','CG','budget','corruption','spatial']]]:
                parents[str(path)]=sha(path)
        atomic(D/'parent_gate_receipt.json',dict(all_previous_required_audits=parents,parent_exited=True))
        completed=[]
        for c in conditions():
            for ds in COUNTS:
                for stage in ['predict','score','audit','cleanup']:
                    run_stage(ds,c['id'],stage)
                completed.append(dict(dataset=ds,condition=c['id'],audit_SHA=sha(unit(ds,c['id'])/'reporting_receipt.json')))
                atomic(D/'completed_units.json',completed)
        assert len(completed)==2*len(conditions())
        atomic(D/'state.json',dict(status='all_full_corruption_units_scored_audited_pending_root_analysis_figures_publication',
                                   controller=who,completed_units=completed,time=time.time()))
    except BaseException:
        atomic(D/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),controller=who));raise
    finally:
        leasefile.close()

if __name__=='__main__':
    queue()
