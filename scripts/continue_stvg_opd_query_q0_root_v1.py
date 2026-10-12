"""Finite CPU independent-root handoff, no early GT or automatic completion."""
import os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,read,write,status,sha
REC=BASE/'postseal_actual_root001'


def run():
    for p,h in read(REC/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    while not (BASE/'CPU_COMPLETION.json').exists():
        s=read(BASE/'STATUS.json');assert 'failed' not in s['status'],'Actual Q0 dependency failure requires root'
        status(REC/'STATUS.json',dict(status='CPU_waiting_for_all_Q0_global_and_original_scoring',pid=os.getpid(),GT_read=False,CPU_only=True,time=time.time()))
        time.sleep(15)
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['cells']==64
    with (REC/'CPU.log').open('a') as log:
        p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/review_stvg_opd_query_q0_v1.py'],cwd=ROOT,
              stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        status(REC/'STATUS.json',dict(status='actual_Q0_independent_statistics_root',worker_pid=p.pid,CPU_only=True,GT_read=True,time=time.time()))
        assert p.wait()==0,'Actual Q0 independent root failed; preserve'
    assert read(BASE/'ROOT_STATISTICS_READBACK.json')['status']=='pass'
    status(REC/'STATUS.json',dict(status='pending_actual_two_PDF_export_pixel_view_publication_archive',CPU_only=True,GT_read=True,task_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(REC/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=True,time=time.time()));raise
