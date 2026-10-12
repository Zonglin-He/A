"""One finite CPU-only P9-P0 seal/analysis/independent-root controller."""
import os,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_conditional_actionability_p0_v1 import BASE,write,status,sha,verify
def run():
    verify();write(BASE/'CPU_LAUNCH.json',dict(status='launched_one_finite_CPU_controller',pid=os.getpid(),controller_sha256=sha(Path(__file__)),CPU_only=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_GT_scoring_calls=0,C_GPU_interrupted=False,time=time.time()))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    for job in ['seal','analyze','root']:
        with (BASE/(job+'.log')).open('w') as f:
            p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/stvg_opd_state_conditional_actionability_p0_v1.py',job],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            status(BASE/'CONTROLLER_STATUS.json',dict(status='running',stage=job,pid=os.getpid(),worker_pid=p.pid,CPU_only=True,GT_read=job!='seal',time=time.time()));code=p.wait()
        if code:
            status(BASE/'CONTROLLER_STATUS.json',dict(status='failed',stage=job,pid=os.getpid(),returncode=code,CPU_only=True,time=time.time()));raise RuntimeError('P9-P0 '+job+' failed; complete log preserved')
    status(BASE/'CONTROLLER_STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',pid=os.getpid(),all_CPU_jobs_exit_zero=True,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
