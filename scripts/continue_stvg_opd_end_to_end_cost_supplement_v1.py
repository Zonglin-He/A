"""One finite CPU dependency waiter and serial timing worker, no scoring."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_end_to_end_cost_supplement_v1 import BASE,PUB,PYTHON,prepare,verify,read,write,status,sha

def run():
    verify();dependency=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (dependency/'FORMAL_GLOBAL_BARRIER.json').exists():
            state=read(dependency/'STATUS.json');assert 'fail' not in state['status'],'Preserved baseline dependency failure needs root'
            status(BASE/'STATUS.json',dict(status='CPU_waiting_for_actual_selected_baseline_formal_GPU_seal',controller_pid=os.getpid(),GT_read=False,time=time.time()));time.sleep(15)
        for ds in ['hc2','vidstg']:
            if (BASE/ds/'COMPLETION.json').exists():continue
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'};log=BASE/'logs'/f'{ds}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                p=subprocess.Popen([str(PYTHON),'-B','scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py',ds],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
                status(BASE/'STATUS.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,dataset=ds,GT_read=False,time=time.time()));assert p.wait()==0,'Preserved actual timing failure needs root'
        status(BASE/'STATUS.json',dict(status='pending_actual_root_statistics_figures_publication',query_fits=32,experiment_complete=False,GT_read=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'controller_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
