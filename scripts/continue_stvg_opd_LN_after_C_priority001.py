"""Finite original 24 qualifications/768 missing fits/global-before-GT/CPU handoff."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.run_stvg_opd_LN_after_C_priority001 import BASE,SCIENCE,verify,stopped
def run():
    verify();stopped()
    from scripts.stvg_opd_mechanism_LN_order_common_v1 import stages,ALLOCATOR
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def child(script,args,CPU=False):
            verify();stopped();cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',script]+args
            log=BASE/'logs'/((('_'.join(args)) or Path(script).stem)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            if CPU:env['CUDA_VISIBLE_DEVICES']=''
            else:env.pop('CUDA_VISIBLE_DEVICES',None)
            with log.open('a') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
                v=dict(status='running',stage=args[-1] if args else Path(script).stem,job=args[0] if args else None,
                    controller_pid=os.getpid(),worker_pid=p.pid,CPU_only=CPU,GT_read=bool(CPU and ('score_' in script or 'finalize_' in script)),
                    original_science_unchanged=True,experiment_complete=False,command=cmd,time=time.time())
                status(BASE/'STATUS.json',v);status(SCIENCE/'STATUS.json',dict(v,latest_serial_revision='after_C_priority001'))
                assert p.wait()==0,'Actual child failure preserved: '+str(cmd)
        if not (SCIENCE/'ROOT_QUALIFICATION_READBACK.json').exists():
            for name in stages():
                if not (SCIENCE/'qualification'/name/'PREDICTION_BARRIER.json').exists():child('scripts/run_stvg_opd_LN_after_C_priority001.py',['qualification',name])
            child('scripts/run_stvg_opd_mechanism_LN_order_v1.py',['root_qualification'],True)
        q=read(SCIENCE/'ROOT_QUALIFICATION_READBACK.json');assert q['status']=='pass'
        write(BASE/'ROOT_PRIORITY_QUALIFICATION_READBACK.json',dict(status='pass',scope='original complete 12 arrivals/24 actual repeated GPU LN-only fits/native-source/VJP/math/state/full repeat independent root; not768 formal completion or future OOM safety',
            qualification_arrivals=12,actual_GPU_fits=24,original_root_sha256=sha(SCIENCE/'ROOT_QUALIFICATION_READBACK.json'),CPU_only=True,GT_read=False,time=time.time()))
        if not (SCIENCE/'GLOBAL_PREDICTION_BARRIER.json').exists():
            total=0;barriers={}
            for name,s in stages().items():
                p=SCIENCE/'stages'/name/'PREDICTION_BARRIER.json'
                if not p.exists():child('scripts/run_stvg_opd_LN_after_C_priority001.py',['GPU',name])
                b=read(p);assert b['status']=='sealed' and not b['GT_read'] and b['new_fit_arrivals']==s['expected_new_fits']==384
                total+=b['new_fit_arrivals'];barriers[str(p.relative_to(ROOT))]=sha(p)
            assert total==768
            write(SCIENCE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',all_missing_LN_orders_and_both_targets=True,new_formal_fits=768,barriers=barriers,
                runtime_sha256=sha(SCIENCE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
        if not (SCIENCE/'CPU_SCORING_COMPLETION.json').exists():child('scripts/score_stvg_opd_mechanism_LN_order_v1.py',[],True)
        if not (SCIENCE/'CPU_COMPLETION.json').exists():child('scripts/finalize_stvg_opd_mechanism_LN_order_v1.py',[],True)
        v=dict(status='pending_actual_LN_complete_independent_root_PDF_pixel_view_publication_archive_then_cost',scope='original768 LN-only new fits/2304 old controls exact reuse/three orders/GT after all new global seal/old CPU; actual root figures/public/archive remain',
            new_formal_fits=768,reused_formal_fits=2304,CPU_only=True,GT_read=True,experiment_complete=False,controller_pid=os.getpid(),time=time.time())
        write(BASE/'CPU_COMPLETION.json',v);status(BASE/'STATUS.json',v);status(SCIENCE/'STATUS.json',dict(v,latest_serial_revision='after_C_priority001'))
if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=True,GT_read=False,time=time.time()));raise
