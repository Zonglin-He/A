"""One CPU dependency waiter, followed by only the missing finite LN-only streams."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_mechanism_LN_order_common_v1 import *
CLAIM=ROOT/'artifacts/stvg_opd_claim_strengthening_v1'
SUP=ROOT/'artifacts/stvg_opd_supplement_v1'

def active(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return False
    return p.read_text().split(') ')[1].split()[0] not in ['Z','X']

def run():
    verify();BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dep=CLAIM/'CPU_COMPLETION.json'
        while not dep.exists():
            p=CLAIM/'STATUS.json'
            if p.exists():assert 'fail' not in read(p)['status'],'Original dependency failure preserved; root must handle it'
            status(BASE/'STATUS.json',dict(status='CPU_waiting_after_existing_baselines_cost_expanded_state_and_latency',controller_pid=os.getpid(),CPU_only=True,
               new_formal_fits=0,new_GPU_fits=0,GT_read=False,experiment_complete=False,time=time.time()))
            time.sleep(15)
        assert read(dep)['status']=='pending_actual_root_PDF_pixel_view_publication_archive'
        assert read(CLAIM/'expanded_state/GLOBAL_PREDICTION_BARRIER.json')['new_formal_fits']==5430
        assert read(CLAIM/'deployment_latency/CPU_COMPLETION.json')['status']
        # CPU handoff is valid, but old GPU/cost/controller processes must have exited.
        for launch in [CLAIM/'LAUNCH.json',SUP/'actual_root/postseal_continuation006/LAUNCH.json']:
            if launch.exists():
                r=read(launch);pid=r.get('pid',r.get('controller_pid'))
                if pid:
                    while active(pid):time.sleep(5)
        def child(script,args,CPU=False):
            verify();cmd=[str(PYTHON),'-B',script]+args;log=BASE/'logs'/((('_'.join(args)) or Path(script).stem)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            if CPU:env['CUDA_VISIBLE_DEVICES']=''
            else:env.pop('CUDA_VISIBLE_DEVICES',None)
            with log.open('a') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
                status(BASE/'STATUS.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,command=cmd,CPU_only=CPU,
                   stage=args[-1] if args else None,job=args[0] if args else Path(script).stem,experiment_complete=False,time=time.time()))
                assert p.wait()==0,'Actual child failure preserved; root repair required'
        if not (BASE/'ROOT_QUALIFICATION_READBACK.json').exists():
            for name in stages():
                if not (BASE/'qualification'/name/'PREDICTION_BARRIER.json').exists():child('scripts/run_stvg_opd_mechanism_LN_order_v1.py',['qualification',name])
            child('scripts/run_stvg_opd_mechanism_LN_order_v1.py',['root_qualification'],True)
        assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
        if not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
            total=0;barriers={}
            for name,s in stages().items():
                p=BASE/'stages'/name/'PREDICTION_BARRIER.json'
                if not p.exists():child('scripts/run_stvg_opd_mechanism_LN_order_v1.py',['GPU',name])
                b=read(p);assert b['status']=='sealed' and not b['GT_read'];assert b['new_fit_arrivals']==384;total+=384;barriers[str(p.relative_to(ROOT))]=sha(p)
            assert total==768
            write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',all_missing_LN_orders_and_both_targets=True,new_formal_fits=768,barriers=barriers,GT_read=False,
               runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
        if not (BASE/'CPU_SCORING_COMPLETION.json').exists():child('scripts/score_stvg_opd_mechanism_LN_order_v1.py',[],True)
        if not (BASE/'CPU_COMPLETION.json').exists():child('scripts/finalize_stvg_opd_mechanism_LN_order_v1.py',[],True)
        status(BASE/'STATUS.json',dict(status='pending_actual_root_independent_statistics_PDF_view_publication_archive',controller_pid=os.getpid(),experiment_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='dependency_or_actual_child_failure_preserved_pending_root',CPU_only=True,evidence=str(p.relative_to(ROOT)),experiment_complete=False,time=time.time()));raise
