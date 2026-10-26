from mask_common import *
import subprocess,fcntl,traceback

def run():
 p=validate();assert read(D/'preflight_v1.json')['status']=='passed_four_actual_Source_rectangle_mask_RGB_event_box_baseline_contracts';lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v1.json').exists();who=identity();atomic(D/'controller_v1.json',who);start=time.perf_counter();release()
 try:
  for stage,entry,args,device in [('GPU','worker_v1.py',[],'0')]+[(ds+'_'+action,'report_v1.py',[action,'--dataset',ds],'')for ds in ['hc2','vidstg']for action in ['score','audit']]:
   resources(start);atomic(D/'state_v1.json',dict(status=stage,controller=who))
   with(D/(stage+'.log')).open('xb')as log:
    c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(HERE/entry),*args],env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(D/(stage+'_child.json'),identity(c.pid));code=c.wait()
   atomic(D/(stage+'_exit.json'),dict(code=code));assert code==0,stage+' preserve paid raw; no automatic retry'
  atomic(D/'state_v1.json',dict(status='both128_Source_rectangular_mask_EdgeTAM_scored_audited_pending_root_cost_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start))
 except BaseException:atomic(D/'failure_v1.json',dict(error=traceback.format_exc(),worker=who));raise
if __name__=='__main__':run()
