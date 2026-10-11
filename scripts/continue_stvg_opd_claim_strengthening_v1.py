"""Single finite serial continuation of the newly authorized three items."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.stvg_opd_expanded_state_common_v1 import BASE as EXP,verify as verify_exp,stages,ALLOCATOR,PYTHON
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1'
SUP=ROOT/'artifacts/stvg_opd_supplement_v1'
LAT=BASE/'deployment_latency'


def verify():
    verify_exp()
    from scripts.run_stvg_opd_deployment_latency_v1 import verify as verify_latency
    verify_latency()
    r=read(BASE/'CONTROLLER_RUNTIME.json')
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    return r


def run():
    verify()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dep=SUP/'actual_root/postseal_continuation006/COMPLETION.json'
        while not dep.exists():
            for namespace in ['baselines','cost','actual_root/postseal_continuation006']:
                p=SUP/namespace/'STATUS.json'
                if p.exists():assert 'fail' not in read(p)['status'],f'Preserved dependency failure: {namespace}'
            status(BASE/'STATUS.json',dict(status='CPU_waiting_for_existing_baseline_audited_cost_and_root_completion',
                controller_pid=os.getpid(),CPU_only=True,new_model_calls=0,GT_read=False,
                experiment_complete=False,time=time.time()))
            time.sleep(15)
        assert read(dep)['status']=='pending_actual_root_PDF_pixel_view_and_publication'
        assert read(SUP/'baselines/FORMAL_GLOBAL_BARRIER.json')['logical_arrivals']==82710
        assert read(SUP/'actual_root/postseal_continuation006/COMPLETE_baselines.json')['status']=='actual_CPU_root_complete_pending_visual_publication'
        assert read(SUP/'actual_root/postseal_continuation006/COMPLETE_cost.json')['status']=='actual_CPU_root_complete_pending_visual_publication'
        def child(script,args,CPU=False,label=None):
            verify();cmd=[str(PYTHON),'-B',script]+args
            log=BASE/'logs'/((label or '_'.join(args) or Path(script).stem)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            if CPU:env['CUDA_VISIBLE_DEVICES']=''
            else:env.pop('CUDA_VISIBLE_DEVICES',None)
            with log.open('a') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
                status(BASE/'STATUS.json',dict(status='running',job=label or args,controller_pid=os.getpid(),
                    worker_pid=p.pid,CPU_only=CPU,command=cmd,log=str(log.relative_to(ROOT)),
                    experiment_complete=False,time=time.time()))
                assert p.wait()==0,f'Actual {label or args} failure preserved'
        if not (EXP/'ROOT_QUALIFICATION_READBACK.json').exists():
            for name in stages():child('scripts/run_stvg_opd_expanded_state_v1.py',['qualification',name],label='expanded_qualification_'+name)
            child('scripts/run_stvg_opd_expanded_state_v1.py',['root_qualification'],CPU=True,label='expanded_root_qualification')
        assert read(EXP/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
        if not (EXP/'GLOBAL_PREDICTION_BARRIER.json').exists():
            barriers={};total=0
            for name,s in stages().items():
                p=EXP/'stages'/name/'PREDICTION_BARRIER.json'
                if not p.exists():child('scripts/run_stvg_opd_expanded_state_v1.py',['GPU',name],label='expanded_formal_'+name)
                b=read(p);assert b['status']=='sealed' and not b['GT_read']
                assert b['new_fit_arrivals']==s['expected_new_fits'];total+=b['new_fit_arrivals']
                barriers[str(p.relative_to(ROOT))]=sha(p)
            assert total==5430
            write(EXP/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',new_formal_fits=total,
                all_three_arms_conditions_orders_and_both_targets=True,barriers=barriers,
                runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
        if not (EXP/'CPU_SCORING_COMPLETION.json').exists():child('scripts/score_stvg_opd_expanded_state_v1.py',[],CPU=True,label='expanded_all_CPU_math_state_dense')
        if not (EXP/'CPU_COMPLETION.json').exists():child('scripts/finalize_stvg_opd_expanded_state_v1.py',[],CPU=True,label='expanded_direct_root_statistics')
        assert read(EXP/'ROOT_COMPLETE_READBACK.json')['status']=='pass'
        if not (LAT/'ROOT_QUALIFICATION_READBACK.json').exists():
            for ds in ['hc2','vidstg']:child('scripts/run_stvg_opd_deployment_latency_v1.py',['qualification',ds],label='deployment_qualification_'+ds)
            child('scripts/run_stvg_opd_deployment_latency_v1.py',['root_qualification'],CPU=True,label='deployment_root_qualification')
        for ds in ['hc2','vidstg']:
            for method,repeat in [('Frozen',0),('Full',0),('Full',1),('Frozen',1)]:
                child('scripts/run_stvg_opd_deployment_latency_v1.py',['timing',ds,method,str(repeat)],label=f'deployment_timing_{ds}_{method}_{repeat}')
        if not (LAT/'CPU_COMPLETION.json').exists():child('scripts/finalize_stvg_opd_deployment_latency_v1.py',[],CPU=True,label='deployment_paired_root_statistics')
        write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_root_PDF_pixel_view_publication_archive',
            expanded_state_formal_fits=5430,paired_latency_unique_queries=32,
            new_model_calls='actual serial GPU stages recorded separately',
            experiment_complete=False,public_remote_verified=False,time=time.time()))
        status(BASE/'STATUS.json',dict(status='pending_actual_root_PDF_pixel_view_publication_archive',
            experiment_complete=False,controller_pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',
            evidence=str(p.relative_to(ROOT)),experiment_complete=False,time=time.time()))
        raise
