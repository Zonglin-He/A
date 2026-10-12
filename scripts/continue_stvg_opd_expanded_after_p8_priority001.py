"""Finite next C expanded-state experiment, bypassing only retired scheduling dependencies."""
import os,sys,subprocess,time,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.run_stvg_opd_expanded_after_p8_priority001 import BASE,EXP,verify,stopped
CLAIM=ROOT/'artifacts/stvg_opd_claim_strengthening_v1'
def run():
    verify();stopped()
    from scripts.stvg_opd_expanded_state_common_v1 import stages,ALLOCATOR
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
                z=dict(status='running',stage=args[-1] if args else Path(script).stem,job=args[0] if args else None,controller_pid=os.getpid(),worker_pid=p.pid,CPU_only=CPU,GT_read=bool(CPU and ('score_' in script or 'finalize_' in script)),command=cmd,log=str(log.relative_to(ROOT)),science_changed=False,time=time.time())
                status(BASE/'STATUS.json',z);status(CLAIM/'STATUS.json',dict(z,latest_priority_revision='after_P8_priority001'))
                assert p.wait()==0,'Actual child failure preserved: '+str(cmd)
        if not (EXP/'ROOT_QUALIFICATION_READBACK.json').exists():
            for name in stages():
                if not (EXP/'qualification'/name/'PREDICTION_BARRIER.json').exists():child('scripts/run_stvg_opd_expanded_after_p8_priority001.py',['qualification',name])
            child('scripts/run_stvg_opd_expanded_state_v1.py',['root_qualification'],True)
        assert read(EXP/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
        write(BASE/'ROOT_PRIORITY_QUALIFICATION_READBACK.json',dict(status='pass',scope='actual whole C expanded-state qualification/native-source/48 full repeat GPU fits and independent original math-state-repeat root; does not establish all5430 formal completion or future OOM safety',qualification_arrivals=24,complete_GPU_fits=48,original_root_sha256=sha(EXP/'ROOT_QUALIFICATION_READBACK.json'),original_scientific_runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),CPU_only=True,GT_read=False,time=time.time()))
        if not (EXP/'GLOBAL_PREDICTION_BARRIER.json').exists():
            total=0;bindings={}
            for name,s in stages().items():
                p=EXP/'stages'/name/'PREDICTION_BARRIER.json'
                if not p.exists():child('scripts/run_stvg_opd_expanded_after_p8_priority001.py',['GPU',name])
                b=read(p);assert b['status']=='sealed' and not b['GT_read'] and b['new_fit_arrivals']==s['expected_new_fits']
                total+=b['new_fit_arrivals'];bindings[str(p.relative_to(ROOT))]=sha(p)
            assert total==5430
            write(EXP/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',new_formal_fits=5430,all_three_arms_conditions_orders_and_both_targets=True,barriers=bindings,runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
        if not (EXP/'CPU_SCORING_COMPLETION.json').exists():child('scripts/score_stvg_opd_expanded_state_v1.py',[],True)
        if not (EXP/'CPU_COMPLETION.json').exists():child('scripts/finalize_stvg_opd_expanded_state_v1.py',[],True)
        assert read(EXP/'ROOT_COMPLETE_READBACK.json')['status']=='pass'
        write(BASE/'EXPANDED_CPU_COMPLETION.json',dict(status='pending_actual_expanded_root_PDF_view_publication_archive_and_next_serial_timing_LN_cost',scope='complete next C expanded-state GPU/GT-after-global/CPU root, not whole C or original S_N task',new_formal_fits=5430,root_sha256=sha(EXP/'ROOT_COMPLETE_READBACK.json'),CPU_only=True,GT_read=True,experiment_complete=False,time=time.time()))
        status(BASE/'STATUS.json',dict(status='pending_actual_expanded_root_PDF_view_publication_archive_and_next_serial_timing_LN_cost',controller_pid=os.getpid(),CPU_only=True,GT_read=True,time=time.time()))
        status(CLAIM/'STATUS.json',dict(status='expanded_complete_pending_actual_root_and_remaining_deployment_latency',latest_priority_revision='after_P8_priority001',CPU_only=True,experiment_complete=False,time=time.time()))
if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',evidence=str(p.relative_to(ROOT)),CPU_only=True,GT_read=False,time=time.time()));raise
