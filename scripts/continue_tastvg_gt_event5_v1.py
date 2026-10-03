"""Finite serial GT-event expert and temporary update queue, after CPU Experiment1."""
import os,sys,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_oracle_event5_common_v1 import *
def run():
    verified();assert read(BASE/'EXPERIMENT1_ROOT_AUDIT.json')['status']=='pass'
    h=(BASE/'PROCESS.lock').open('w');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time()))
    jobs=[['scripts/run_tastvg_gt_event5_expert_v1.py']]+[['scripts/run_tastvg_gt_event5_update_v1.py',d] for d in DATASETS]
    for job in jobs:
        status(BASE/'STATUS.json',dict(status='running',job=job,pid=os.getpid(),GT_assisted_observation=True,time=time.time()))
        env=os.environ.copy()
        if len(job)==1:env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
        with (BASE/'WORKER.log').open('a') as log:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',*job],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(BASE/'GLOBAL_INTERVENTION_BARRIER.json',dict(status='sealed',donors=288,GT_assisted_observation=True,
        raw_GT_read=False,datasets={d:sha(BASE/d/'updates/PREDICTION_BARRIER.json') for d in DATASETS},time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_CPU_GT_score',time=time.time()))
if __name__=='__main__':
    try:run()
    except BaseException as e:
        status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
