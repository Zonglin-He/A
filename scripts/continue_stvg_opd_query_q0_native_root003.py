"""Independent root handoff in the corrected Native-support namespace."""
import os,subprocess,time,traceback,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE,ROOT,context
context()
from scripts.decota_matrix_common_v1 import read,status,sha
REC=BASE/'postseal_actual_root001'


def run():
    for p,h in read(REC/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    while not (BASE/'CPU_COMPLETION.json').exists():
        assert 'failed' not in read(BASE/'STATUS.json')['status']
        status(REC/'STATUS.json',dict(status='CPU_waiting_for_all_Native_Q0_global_and_scoring',pid=os.getpid(),GT_read=False,CPU_only=True,time=time.time()));time.sleep(15)
    command=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','-c','from scripts.stvg_opd_query_q0_native_scope003 import context; context(); from scripts.review_stvg_opd_query_q0_v1 import run; run()']
    with (REC/'CPU.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        status(REC/'STATUS.json',dict(status='running_actual_Native_Q0_independent_root',worker_pid=p.pid,CPU_only=True,GT_read=True,time=time.time()))
        assert p.wait()==0,'Actual Native Q0 root failed; preserve'
    assert read(BASE/'ROOT_STATISTICS_READBACK.json')['status']=='pass'
    status(REC/'STATUS.json',dict(status='pending_actual_two_PDF_export_pixel_view_publication_archive',CPU_only=True,GT_read=True,task_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(REC/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p),CPU_only=True,time=time.time()));raise
