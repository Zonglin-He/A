from common import *
import subprocess,fcntl,traceback

def run():
 validate();lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);start=time.perf_counter()
 try:
  for ds in ['hc2','vidstg']:
   assert read(D/ds/'preflight.json')['status']=='passed_repeat_actual_ROI_pool_and_gamma_replay'
   for command in ['predict','score','audit']:
    stage=ds+'_'+command;atomic(D/'state.json',dict(status=stage,controller=who))
    with(D/(stage+'.log')).open('xb')as log:
     c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'run_v1.py'),command,'--dataset',ds],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(D/(stage+'_child.json'),identity(c.pid));code=c.wait()
    atomic(D/(stage+'_exit.json'),dict(code=code));assert code==0,stage
  atomic(D/'state.json',dict(status='both_standalone192_gamma_and_ROI_controls_scored_audited_pending_root_analysis_image_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),worker=who));raise
if __name__=='__main__':run()
