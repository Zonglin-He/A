"""Bounded serial geometry/config recovery; no old queue resumption."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_geometry_config32_v1 import BASE,put,verify
UNIT='stvg-geometry-config32-001.service'
def main():
 from scripts import launch_stvg_instance_identity_v1 as resource
 verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='geometry_config';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 put(BASE/'ACTUAL_RESOURCE_READBACK.json',dict(status='pass',unit=UNIT,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 try:
  for stage in ['worker','root','score']:
   verify();put(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),replace=True);folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False)
   with (folder/'worker.log').open('xb') as log:
    t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/stvg_geometry_config32_v1.py'),stage],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
   put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=stage=='score'));assert rc==0,(stage,rc)
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False));put(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=32,GT_read=False),replace=True)
 except BaseException as e:put(BASE/'FAILURE.json',dict(status='technical_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':main()
