"""Finite fixed P1 after actual P0 root review, including a disclosed negative gate."""
import os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def run():
    verify();auth=read(BASE/'P1_LAUNCH_AUTHORIZATION.json')
    assert auth['status']=='root_decision_after_complete_P0_readback_fixed_parameters'
    assert auth['failure_attribution_complete'] and auth['configuration_or_roster_retuned']==False
    assert sha(BASE/'P0_ROOT_CLOSING_RECEIPT.json')==auth['P0_root_receipt_sha256']
    assert sha(BASE/'P1_RUNTIME_LOCK_revision001.json')==auth['P1_runtime_sha256']
    runtime=read(BASE/'P1_RUNTIME_LOCK_revision001.json')
    for f,h in runtime['pins'].items():assert sha(ROOT/f)==h,f
    for ds in DATASETS:
        stage='P1_'+ds;command=[str(PYTHON),'-B','scripts/run_stvg_opd_paper_v1.py',stage]
        logfile=BASE/'logs'/f'{stage}.log'
        with logfile.open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(BASE/'STAGE.json',dict(status='running',stage=stage,controller_pid=os.getpid(),worker_pid=p.pid,
                command=command,log=str(logfile.relative_to(ROOT)),qualification=False,GT_read=False,time=time.time()))
            assert p.wait()==0,f'Full query worker failed: {stage}; preserve prefix and pin engineering repair.'
    write(BASE/'P1_PREDICTION_BARRIER.json',dict(status='sealed',all_deployment_OPD_directions=True,
        barriers={f'stages/P1_{ds}/PREDICTION_BARRIER.json':sha(BASE/f'stages/P1_{ds}/PREDICTION_BARRIER.json') for ds in DATASETS},
        adapted_arrivals=41355,other_baselines_reused=True,GT_read=False,time=time.time()))
    write(BASE/'P1_GPU_COMPLETION.json',dict(status='pending_root_full_table_postseal_audit',
        prediction_complete=True,full_paper_complete=False,controller_pid=os.getpid(),time=time.time()))
    status(BASE/'STATUS.json',dict(status='P1_all_OPD_predictions_sealed_pending_root_full_table_audit',
        root_must_continue_P1_CPU_then_P2_P6=True,paper_suite_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'P1_controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved',evidence=str(d),time=time.time()))
        raise
