"""One serial, finite queue and global barrier; never reads GT."""
import os, sys, subprocess, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_correction_scope_common_v1 import *

def run():
    verify();write(BASE/'LAUNCH.json',dict(controller_pid=os.getpid(),time=time.time(),GT_read=False))
    for stage in ('prepare','episodic','matrix'):
        for ds in DATASETS:
            status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,controller_pid=os.getpid(),GT_read=False,time=time.time()))
            f=BASE/'logs'/f'{stage}_{ds}.log';f.parent.mkdir(parents=True,exist_ok=True)
            with f.open('a') as log:
                subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_correction_scope_v1.py',ds,stage],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify();cohort=read(BASE/'COHORT.json');files={}
    for f in BASE.rglob('*.pt'):
        checked(f)
        for g in (f,f.with_suffix('.json')):files[str(g.relative_to(BASE))]=sha(g)
    for ds in DATASETS:
        for name in ['TARGET_LOCK.json']+[s+x+'.json' for s in ('prepare','episodic','matrix') for x in ('_RESOURCES','_SMOKE')]:
            f=BASE/ds/name;files[str(f.relative_to(BASE))]=sha(f)
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,GT_read=False,
        logical_arrivals=1152,donor_writes=288,time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_pending_root_scoring_audit_publication',GT_read=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException as e:
        status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
