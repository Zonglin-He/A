"""Finite serial P7-A and conditional P7-B controller, no self scheduling."""
import os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *

def child(folder,script,job,CPU=False):
    environment={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    if CPU:environment['CUDA_VISIBLE_DEVICES']=''
    else:environment.pop('CUDA_VISIBLE_DEVICES',None)
    with (folder/(job+'.log')).open('a') as log:
        p=subprocess.Popen([str(PYTHON),'-B',script,job],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        status(BASE/'STATUS.json',dict(status='running',stage=folder.name+'/'+job,controller_pid=os.getpid(),worker_pid=p.pid,
            CPU_only=CPU,GT_read=job in ['score','root'],time=time.time()))
        result=p.wait();assert result==0,f'Actual {folder.name}/{job} failed and preserved'

def run():
    verify();pause(A)
    try:
        child(A,'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','qualification')
        child(A,'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','qualification_root',True)
        child(A,'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','formal')
    finally:
        # The child has actually exited before original optimizer/PIDs resume.
        if (A/'PRIORITY_PAUSE_RECEIPT.json').exists():resume(A)
    child(A,'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','assemble',True)
    child(A,'scripts/score_stvg_opd_temporal_capture_p7a_v1.py','score',True)
    child(A,'scripts/score_stvg_opd_temporal_capture_p7a_v1.py','root',True)
    decision=read(A/'P7_B_BRANCH_DECISION.json');assert decision['status']=='decided_after_complete_A_root'
    status(BASE/'STATUS.json',dict(status='P7_A_pending_actual_PDF_view_publication_archive',P7_B_eligible=decision['P7_B_eligible'],
        CPU_only=True,GT_read=True,task_complete=False,time=time.time()))
    # The root finishes the bounded actual A report/view/public/archive, then
    # activates the already authorized single B branch with its own runtime.

if __name__=='__main__':
    try:run()
    except BaseException:
        f=BASE/'controller_failures'/str(time.time_ns());f.mkdir(parents=True);(f/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(f.relative_to(ROOT)),time=time.time()));raise
