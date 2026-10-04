"""Finite serial GPU batch; no scoring/GT in this controller."""
import sys,os,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_negative_evidence_common_v1 import *
def run():
    verify();write(BASE/'LAUNCH.json',dict(controller_pid=os.getpid(),time=time.time(),GT_read=False))
    for stage in ['local','reset_u']:
        for ds in DATASETS:
            if read(BASE/ds/(stage+'_STATUS.json'))['status']=='completed' if (BASE/ds/(stage+'_STATUS.json')).exists() else False:continue
            f=BASE/'logs'/f'{stage}_{ds}.log';f.parent.mkdir(parents=True,exist_ok=True)
            status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,controller_pid=os.getpid(),GT_read=False,time=time.time()))
            with f.open('a') as log:
                subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_negative_evidence_v1.py',ds,stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify();cells=read(BASE/'COHORT.json')['cells'];expected=[]
    for c in cells:
        expected.append(reset_payload_path(c))
        if c['scheduled']:expected.append(local_payload_path(c))
    files={}
    for stem in expected:
        checked(stem)
        for ext in ['.pt','.json']:
            f=Path(str(stem)+ext);files[str(f.relative_to(BASE))]=sha(f)
    for ds in DATASETS:
        for stage in ['local','reset_u']:
            f=BASE/ds/(stage+'_RESOURCES.json');files[str(f.relative_to(BASE))]=sha(f)
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',local_experts=288,reset_u_arrivals=1152,files=files,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_pending_root_scoring_audit_publication',GT_read=False,time=time.time()))
if __name__=='__main__':
    try:run()
    except BaseException as e:
        status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
