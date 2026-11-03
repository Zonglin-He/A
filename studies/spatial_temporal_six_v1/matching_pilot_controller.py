"""Only source-accepted cache128 pilot; no GPU/full acquisition release."""
from matching_pilot import *
import subprocess,traceback
def queue():
 plan=check_pilot();assert not(M/'controller.json').exists();atomic(M/'controller.json',identity());stages=[('preflight',None)]+[(s,ds)for ds in COUNTS for s in ['predict','score','audit']];start=time.perf_counter()
 try:
  for stage,ds in stages:
   parent.resources(start);label=(ds or'both')+'_'+stage;cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'matching_pilot.py'),stage]+(['--dataset',ds]if ds else[]);log=M/(label+'.log');tick=time.perf_counter()
   with log.open('wb')as out:
    child=subprocess.Popen(cmd,cwd=R,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});who=dict(pid=child.pid,startticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]));atomic(M/'controller_state.json',dict(status='running_'+label,controller=identity(),child=who,new_GPU=False,full_GPU_release=False));code=child.wait()
   atomic(M/(label+'_exit.json'),dict(stage=stage,dataset=ds,worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-tick));assert code==0,(label,code)
  atomic(M/'controller_state.json',dict(status='both_matching_pilots_sealed_scored_audited_PENDING_ROOT_analysis',controller=identity(),new_GPU=False,full_GPU_release=False,whole_six_complete=False))
 except BaseException:atomic(M/'failure.json',dict(error=traceback.format_exc(),controller=identity(),new_GPU=False,all_inputs_preserved=True));raise
if __name__=='__main__':queue()
