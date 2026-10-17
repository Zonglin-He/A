"""Single bounded R1 geometry stage. No concurrent models or automatic historical queues."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,put,verify
UNIT='stvg-r1-geometry32-002.service'
def main():
 from scripts import launch_stvg_instance_identity_v1 as resource
 verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='r1_geometry';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start();folder=BASE/'execution/r1002';folder.mkdir(parents=True,exist_ok=False)
 try:
  with (folder/'worker.log').open('xb') as log:
   t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(ROOT/'scripts/stvg_r1_geometry32_v2.py'),'r1'],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
  put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0;put(BASE/'R1_HANDOFF_COMPLETION.json',dict(status='complete_pending_DINO_and_Student',count=32,GT_read=False))
 except BaseException as e:put(BASE/'FAILURE002.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':main()
