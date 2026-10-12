"""Finite CPU signal seal, offline labels and independent P9 root."""
import os,sys,subprocess,time,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,read,write,status,sha,verify
def run():
    verify()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job,receipt in [('seal','GLOBAL_READOUT_BARRIER.json'),('score','CPU_COMPLETION.json'),('root','ROOT_READBACK.json')]:
            if (BASE/receipt).exists():continue
            log=BASE/'logs'/(job+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
            with log.open('a') as f:
                p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/stvg_opd_evidence_actionability_audit_v1.py',job],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                status(BASE/'CONTROLLER_STATUS.json',dict(status='running',stage=job,controller_pid=os.getpid(),worker_pid=p.pid,CPU_only=True,GT_read=job!='seal',time=time.time()))
                assert p.wait()==0,'Actual CPU child failure: '+job
        assert read(BASE/'ROOT_READBACK.json')['status']=='pass';verify()
        status(BASE/'CONTROLLER_STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',CPU_only=True,GT_read=True,controller_pid=os.getpid(),time=time.time()))
if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'CONTROLLER_STATUS.json',dict(status='failed_preserved_pending_actual_root',scope='finite CPU cached diagnostic failure, not active C GPU experiment failure',CPU_only=True,GT_read=False,time=time.time()));raise
