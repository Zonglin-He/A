from pathlib import Path
import os,json,subprocess,time,traceback
R=Path(__file__).resolve().parents[2];D=R/'outputs/pure_expert_interpolation_v1'
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def who(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
 assert not(D/'launch.json').exists();me=who(os.getpid());put(D/'launch.json',dict(worker=me,entry=__file__))
 try:
  for ds in ['hc2','vidstg']:
   for stage in ['predict','score','audit']:
    dest=D/ds;dest.mkdir(parents=True,exist_ok=True);t=time.perf_counter();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('run.py')),stage,'--dataset',ds]
    with(dest/f'{stage}.log').open('w')as f:
     ch=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);w=who(ch.pid);put(D/'state.json',dict(status='running_CPU_only',dataset=ds,stage=stage,worker=me,child=w));code=ch.wait()
    put(dest/f'{stage}_exit.json',dict(worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-t));assert code==0,(ds,stage,code)
  put(D/'state.json',dict(status='both_pure128_domains_scored_audited_pending_root_publication',worker=me))
 except BaseException:put(D/'failure.json',dict(error=traceback.format_exc(),worker=me));raise
