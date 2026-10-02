"""Finite serial stage, returning to root for measurement and scientific readback."""
import os,sys,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_views_common_v1 import *
def run(stage,split):
    verify();h=(BASE/'PROCESS.lock').open('w');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(BASE/'LAUNCH.json',dict(pid=os.getpid(),stage=stage,split=split,time=time.time()))
    if stage=='round2':assert (BASE/'ROUND1_SELECTION.json').exists()
    if split=='confirm':assert (BASE/'FINAL_SELECTION.json').exists()
    jobs=[]
    if stage=='round2':jobs.append(['scripts/run_tastvg_correction_experts_v1.py','temporal',split])
    jobs.append(['scripts/run_tastvg_correction_experts_v1.py',stage,split])
    jobs += [['scripts/run_tastvg_current_correction_v1.py',d,split,stage] for d in DATASETS]
    for job in jobs:
        status(BASE/'STATUS.json',dict(status='running',stage=stage,split=split,job=job,pid=os.getpid(),GT_read=False,time=time.time()))
        with (BASE/f'{stage}_{split}_WORKER.log').open('a') as log:
            interpreter='.venv-exost/bin/python' if job[0].endswith('correction_experts_v1.py') and job[1]=='temporal' else '.conda/tubedetr/bin/python'
            env=os.environ.copy()
            if job[0].endswith('correction_experts_v1.py') and job[1]!='temporal':
                env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
            subprocess.run([str(ROOT/interpreter),'-B',*job],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(BASE/f'{stage}_{split}_GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',stage=stage,split=split,
        cells=sum(read(BASE/d/stage/split/'PREDICTION_BARRIER.json')['cells'] for d in DATASETS),GT_read=False,
        datasets={d:sha(BASE/d/stage/split/'PREDICTION_BARRIER.json') for d in DATASETS},time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_root_audit',stage=stage,split=split,GT_read=False,time=time.time()))
if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException as e:
        status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
