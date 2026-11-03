from pathlib import Path
import os,json,time,subprocess,traceback
R=Path(__file__).resolve().parents[2];D=R/'outputs/spatial_temporal_six_v1'
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def who(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
 assert not(D/'cpu_launch.json').exists();me=who(os.getpid());put(D/'cpu_launch.json',dict(worker=me,entry=__file__))
 try:
  for ds in ['hc2','vidstg']:
   for stage in ['predict','score','audit','diagnostics','diagnostic_audit']:
    folder=D/'execution'/ds;folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('cpu.py')),stage,'--dataset',ds]
    with(folder/(stage+'.log')).open('w')as f:
     child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);w=who(child.pid);put(D/'cpu_state.json',dict(status='running',dataset=ds,stage=stage,worker=me,child=w));code=child.wait()
    put(folder/(stage+'_exit.json'),dict(worker=w,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
  put(D/'cpu_state.json',dict(status='full_pure_and_both_offline_diagnostics_complete_pending_root_analysis',worker=me))
 except BaseException:put(D/'cpu_failure.json',dict(worker=me,error=traceback.format_exc()));raise
