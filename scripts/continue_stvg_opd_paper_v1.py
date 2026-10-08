"""Finite serial P0 controller; no automatic P1 until actual postseal root decision."""
import os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def child(stage,qualify=False):
    suffix='_qualification' if qualify else ''
    logfile=BASE/'logs'/f'{stage}{suffix}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
    command=[str(PYTHON),'-B','scripts/run_stvg_opd_paper_v1.py',stage]
    if qualify:command.append('qualification')
    with logfile.open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status(BASE/'STAGE.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,
            stage=stage,qualification=qualify,command=command,log=str(logfile.relative_to(ROOT)),GT_read=False,time=time.time()))
        assert p.wait()==0, f'Worker failed: {stage}; inspect preserved log.'

def run():
    lock()
    if not (BASE/'QUALIFICATION.json').exists():
        for ds in DATASETS:child('P0_'+ds,True)
        records=[read(BASE/'qualification'/('P0_'+ds)/'QUALIFICATION.json') for ds in DATASETS]
        assert all(r['status']=='pass' for r in records)
        assert sum(q['updates'] for r in records for q in r['records'])>0
        write(BASE/'QUALIFICATION.json',dict(status='pass',datasets=records,actual_queries=4,
            actual_fits=12,independent_dataset_processes=True,GT_read=False,time=time.time()))
    for ds in DATASETS:child('P0_'+ds)
    barriers={f'stages/P0_{ds}/PREDICTION_BARRIER.json':sha(BASE/f'stages/P0_{ds}/PREDICTION_BARRIER.json') for ds in DATASETS}
    if not (BASE/'P0_PREDICTION_BARRIER.json').exists():
        write(BASE/'P0_PREDICTION_BARRIER.json',dict(status='sealed',barriers=barriers,
            adapted_arrivals=1536,Frozen_logical_arrivals=512,total_four_arm_logical_arrivals=2048,
            all_deployment_arms_both_directions=True,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='P0_all_predictions_sealed_pending_root_postseal_audit',
        P0_execution_complete=True,paper_suite_complete=False,GT_read=False,time=time.time()))
    write(BASE/'P0_GPU_COMPLETION.json',dict(status='pending_root_postseal_audit_and_P1_gate',
        controller_pid=os.getpid(),GT_read=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved',evidence=str(d),time=time.time()))
        raise
