"""Single CPU measurement stage, independent audit/export, no old queue use."""
from run import *
import traceback
def main():
 plan=check();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity());start=time.perf_counter()
 try:
  for script,stage in [('run.py','preflight'),('run.py','measure'),('report.py','audit'),('report.py','export')]:
   old.parent.resources(start);cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),stage];tick=time.perf_counter()
   with(D/(stage+'.log')).open('wb')as log:
    p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});w=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(D/'state.json',dict(status='running_'+stage,controller=identity(),child=w,old_work_resumed=False));code=p.wait()
   atomic(D/(stage+'_exit.json'),dict(stage=stage,worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-tick));assert code==0,(stage,code)
  atomic(D/'state.json',dict(status='measured_scored_reference_reused_independently_audited_exported_PENDING_ROOT_publication',controller=identity(),measured_calls=2304,new_Source=0,new_expert=0,new_GPU=0,other_new_experiments_paused=True))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),controller=identity(),all_attempts_and_existing_artifacts_preserved=True));raise
if __name__=='__main__':main()
