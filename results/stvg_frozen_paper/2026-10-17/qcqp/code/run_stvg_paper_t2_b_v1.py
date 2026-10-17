"""Explicit paper entry: never dispatches historical CURRENT or old full queues."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_paper_t2_b_v1 import BASE,read,put,sha,verify
UNIT='stvg-paper-t2-b-001.service'
STAGES=('prepare','input_root','features','jobs','qualification','teacher','teacher_root','student_prepare','student_worker','student_root','score')
def main():
 from scripts import launch_stvg_instance_identity_v1 as resource
 verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='paper_t2_b256';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines() if '='in x);assert [int(props[k]) for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
 put(BASE/'ACTUAL_RESOURCE_READBACK.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 try:
  for stage in STAGES:
   verify();put(BASE/'STATUS.json',dict(status='running',stage=stage,count=256,GT_read=stage=='score'),replace=True);folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);script='score_stvg_paper_t2_b_v1.py' if stage=='score' else 'stvg_paper_t2_b_v1.py';inter='.venv/stvg_native_distill_v1/bin/python' if stage in ('features','qualification','teacher') else '.conda/tubedetr/bin/python';args=[] if stage=='score' else [stage]
   with (folder/'worker.log').open('xb') as log:
    tick=time.perf_counter();p=subprocess.Popen([str(ROOT/inter),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',unit=UNIT,pid=p.pid,GT_read=stage=='score'));rc=p.wait()
   put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,rc)
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=256,independent_science_pixels_privacy_remote_archive_required=True,GT_read=False));put(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=256,GT_read=False),replace=True)
 except BaseException as e:put(BASE/'FAILURE.json',dict(status='technical_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':
 if len(sys.argv)>1:
  from scripts import stvg_paper_t2_b_v1 as pipeline
  getattr(pipeline,sys.argv[1])() if sys.argv[1] in ('setup','prepare','input_root','features') else pipeline.arm_stage(sys.argv[1])
 else:main()
