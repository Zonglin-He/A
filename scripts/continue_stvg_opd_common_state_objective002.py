"""Finite replacement for an idle preGT CPU-only root preparation helper."""
import os,sys,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.review_stvg_opd_common_state_postseal_v1 import BASE,REC,require_seal
REV=REC/'preGT_objective_dispatch002'


def run():
    for p,h in read(REV/'REVISION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    while not (BASE/'COMMON_STATE_CPU_COMPLETION.json').exists():
        s=read(BASE/'CONTROLLER_STATUS.json');assert 'failed' not in s['status']
        status(REC/'STATUS.json',dict(status='CPU_waiting_for_actual_global_and_original_CPU_completion_objective002',
              pid=os.getpid(),CPU_only=True,GT_read=False,task_complete=False,time=time.time()))
        time.sleep(15)
    require_seal()
    with (REV/'CPU.log').open('a') as log:
        p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/review_stvg_opd_common_state_objective002.py'],cwd=ROOT,
              stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        status(REC/'STATUS.json',dict(status='running_actual_independent_root_objective002',worker_pid=p.pid,CPU_only=True,GT_read=True,task_complete=False,time=time.time()))
        assert p.wait()==0,'Actual independent root failure preserved'
    assert read(REC/'ROOT_READBACK.json')['status']=='pass'
    status(REC/'STATUS.json',dict(status='pending_actual_two_PDF_export_pixel_view_publication_archive',CPU_only=True,GT_read=True,task_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REV/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(REC/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=True,time=time.time()));raise
