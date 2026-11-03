"""CPU-only successor after user cancellation of remaining full acquisition."""
from pathlib import Path
import os,json,time,subprocess,traceback
R=Path(__file__).resolve().parents[2];U=R/'outputs/reviewer_key_controls_v1/subset128_v1'
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def who(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
 assert not(U/'launch.json').exists();me=who(os.getpid());put(U/'launch.json',dict(worker=me,entry=__file__))
 try:
  for ds in ['hc2','vidstg']:
   for stage in ['predict','score','audit']:
    folder=U/ds;folder.mkdir(parents=True,exist_ok=True);assert not(folder/f'{stage}_exit.json').exists();start=time.perf_counter()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('subset_v1.py')),stage,'--dataset',ds]
    with(folder/f'{stage}.log').open('w')as f:
     child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ident=who(child.pid);put(U/'state.json',dict(status='running_CPU_only',dataset=ds,stage=stage,worker=me,child=ident));code=child.wait()
    put(folder/f'{stage}_exit.json',dict(worker=ident,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
  put(U/'state.json',dict(status='both_fixed128_domains_scored_audited_pending_root_analysis_publication',worker=me))
 except BaseException:put(U/'failure.json',dict(error=traceback.format_exc(),worker=me));raise
