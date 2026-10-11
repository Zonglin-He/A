"""Finite CPU continuation after the single unchanged GPU controller.

This is part of one authorized experiment, not a recurring monitor. It never
starts a GPU job, reads preseal labels, changes a fit or publishes remotely.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE, PYTHON, read, write, sha, verify


def alive(pid):
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False


def run():
    verify(); target=BASE/'CPU_actual_root_continuation002'
    target.mkdir(parents=True,exist_ok=True)
    with (target/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pins={p:sha(ROOT/p) for p in ['scripts/continue_stvg_opd_design_ablation_CPU_root002.py',
            'scripts/finalize_stvg_opd_design_ablation_cpu002.py','scripts/review_stvg_opd_design_ablation_v1.py',
            'scripts/audit_stvg_opd_design_ablation_public_v1.py']}
        rp=target/'RUNTIME.json'
        if rp.exists(): assert read(rp)['pins']==pins
        else: write(rp,dict(status='locked',pins=pins,CPU_only=True,
            scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
            scope='finite corrected CPU table/root readback after original global deployment seal; actual view/publication still require root',time=time.time()))
        original_pid=read(BASE/'LAUNCH.json')['controller_pid']
        write(target/'LAUNCH.json',dict(status='actually_launched',pid=os.getpid(),original_GPU_controller_pid=original_pid,
            CPU_only=True,GT_read=False,runtime_sha256=sha(rp),time=time.time()))
        while not (BASE/'CPU_SCORING_COMPLETION.json').exists():
            if not alive(original_pid):
                write(target/'STATUS.json',dict(status='dependency_exited_before_postseal_CPU_completion_pending_actual_root',
                    CPU_only=True,GT_read=False,original_GPU_controller_pid=original_pid,root_needed=True,time=time.time()))
                return
            write(target/'STATUS.json',dict(status='waiting_for_original_global_seal_and_complete_CPU_scoring',
                CPU_only=True,GT_read=False,pid=os.getpid(),time=time.time()))
            time.sleep(15)
        assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['new_formal_fits']==5376
        # Let the original immutable controller finish its old table projection.
        # Its known assertion is preserved; only the new CPU namespace is used below.
        while alive(original_pid): time.sleep(2)
        verify()
        for rel,digest in pins.items(): assert sha(ROOT/rel)==digest
        def child(script,logname):
            log=target/logname
            with log.open('a') as out:
                proc=subprocess.Popen([str(PYTHON),'-B',script],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
                write(target/'STATUS.json',dict(status='actual_CPU_root_running',script=script,worker_pid=proc.pid,
                    pid=os.getpid(),CPU_only=True,GT_read='post_global_seal',log=str(log.relative_to(ROOT)),time=time.time()))
                assert proc.wait()==0,'CPU continuation failure preserved: '+script
        if not (BASE/'CPU_COMPLETION.json').exists():
            child('scripts/finalize_stvg_opd_design_ablation_cpu002.py','corrected_table_and_figure.log')
        if not (BASE/'actual_root/ROOT_ALL_FITS_READBACK.json').exists():
            child('scripts/review_stvg_opd_design_ablation_v1.py','all_formal_math_state_readback.log')
        write(target/'COMPLETION.json',dict(status='pending_actual_root_figure_view_and_anonymous_publication',
            CPU_only=True,CPU_completion_sha256=sha(BASE/'CPU_COMPLETION.json'),
            actual_root_all_fits_sha256=sha(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json'),
            original_GPU_runtime_unchanged=True,new_model_calls=0,experiment_complete=False,time=time.time()))
        write(target/'STATUS.json',read(target/'COMPLETION.json'))
        write(BASE/'STATUS.json',dict(status='pending_actual_root_visual_and_publication',new_formal_fits=5376,
            CPU_root_readback_complete=True,experiment_complete=False,original_P1_P6_suite_unchanged=True,time=time.time()))


if __name__=='__main__':
    try: run()
    except BaseException:
        p=BASE/'CPU_actual_root_continuation002/recovery'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc())
        write(BASE/'CPU_actual_root_continuation002/STATUS.json',dict(status='actual_CPU_root_failure_preserved_pending_root',
            evidence=str(p.relative_to(ROOT)),CPU_only=True,experiment_complete=False,time=time.time()))
        raise
