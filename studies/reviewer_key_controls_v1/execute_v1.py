"""One owned sequential CPU child; independent prediction, official score, audit."""
from pathlib import Path
import os,json,subprocess,time,traceback
R=Path(__file__).resolve().parents[2];D=R/'outputs/reviewer_key_controls_v1'
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def who(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
 assert not(D/'execution_launch_v1.json').exists();me=who(os.getpid());put(D/'execution_launch_v1.json',dict(worker=me,entry=__file__))
 try:
  for task in ['temporal','regions']:
   for ds in ['hc2','vidstg']:
    for stage in ['predict','score','audit']:
     dest=D/task/ds;assert not(dest/f'{stage}_exit_v1.json').exists();cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('run_v1.py')),stage,'--task',task,'--dataset',ds];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
     with(dest/f'{stage}.log').open('w')as f:
      child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ident=who(child.pid);put(D/'execution_state_v1.json',dict(status='running',task=task,dataset=ds,stage=stage,worker=me,child=ident));code=child.wait()
     put(dest/f'{stage}_exit_v1.json',dict(worker=ident,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(task,ds,stage,code)
  put(D/'execution_state_v1.json',dict(status='all_cached_prediction_controls_scored_audited_pending_root_analysis_publication',worker=me))
 except BaseException:put(D/'execution_failure_v1.json',dict(worker=me,error=traceback.format_exc()));raise
