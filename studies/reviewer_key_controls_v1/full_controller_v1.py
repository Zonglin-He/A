"""New gated owner only; no old queue resumption and no foreign process signals."""
from pathlib import Path
import os,json,time,subprocess,traceback
R=Path(__file__).resolve().parents[2];D=R/'outputs/reviewer_key_controls_v1';F=D/'full'
def put(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,path)
def read(path):return json.loads(path.read_text())
def ident(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
 assert not(F/'launch_v1.json').exists();me=ident(os.getpid());put(F/'launch_v1.json',dict(worker=me,entry=__file__));start=time.perf_counter()
 try:
  while not(D/'execution_state_v1.json').exists()or read(D/'execution_state_v1.json')['status']!='all_cached_prediction_controls_scored_audited_pending_root_analysis_publication':
   assert time.perf_counter()-start<259200
   if(D/'execution_failure_v1.json').exists():raise RuntimeError('CPU prerequisite failed; no GPU launch')
   put(F/'state_v1.json',dict(status='waiting_all_CPU_prediction_control_audits',worker=me));time.sleep(10)
  for ds in ['hc2','vidstg']:
   for stage in ['predict','score','audit']:
    assert not(F/ds/f'{stage}_exit_v1.json').exists();env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if stage!='predict':env['CUDA_VISIBLE_DEVICES']=''
    cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('full_gpu_v1.py')),stage,'--dataset',ds];path=F/ds/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    with path.open('w')as f:
     child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);w=ident(child.pid);put(F/'state_v1.json',dict(status='running',stage=stage,dataset=ds,worker=me,child=w));code=child.wait()
    put(F/ds/f'{stage}_exit_v1.json',dict(worker=w,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-t));assert code==0,(ds,stage,code)
  put(F/'state_v1.json',dict(status='both_full_domains_scored_audited_pending_root_analysis_publication',worker=me))
 except BaseException:put(F/'failure_v1.json',dict(worker=me,error=traceback.format_exc()));raise
