"""One finite CPU continuation; waits for actual global seal, never early GT."""
import os,sys,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1';REC=BASE/'postseal_CPU_continuation001'

def run():
    r=read(REC/'RUNTIME.json')
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
        state=read(BASE/'CONTROLLER_STATUS.json')
        if 'failed' in state['status']:raise RuntimeError('Actual O failure requires bounded root handling; no early GT')
        status(REC/'STATUS.json',dict(status='CPU_waiting_for_actual_all_arm_global',CPU_only=True,pid=os.getpid(),GT_read=False,time=time.time()))
        time.sleep(15)
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    if not (BASE/'COMMON_STATE_CPU_COMPLETION.json').exists():
        with (REC/'CPU.log').open('a') as f:
            p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/score_stvg_opd_common_state_geometry_v1.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            status(REC/'STATUS.json',dict(status='running_actual_postseal_CPU_root',worker_pid=p.pid,CPU_only=True,GT_read=True,time=time.time()))
            assert p.wait()==0,'Actual postseal CPU failure preserved'
    status(REC/'STATUS.json',dict(status='pending_actual_independent_statistics_PDF_view_publication_archive',CPU_only=True,GT_read=True,experiment_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(REC/'STATUS.json',dict(status='failed_preserved_pending_actual_root',CPU_only=True,evidence=str(p.relative_to(ROOT)),time=time.time()));raise
