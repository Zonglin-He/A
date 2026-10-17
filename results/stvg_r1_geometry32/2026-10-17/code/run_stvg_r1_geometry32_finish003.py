"""Continue only unfinished DINO work, then independent root and complete-group score."""
from pathlib import Path
import os,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,put,verify
UNIT='stvg-r1-geometry32-finish-003.service'
def main():
 from scripts import launch_stvg_instance_identity_v1 as resource
 verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='r1geom_finish003';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 try:
  for stage in ['worker','root','score']:
   verify();folder=BASE/'execution_measure/recovery003'/stage;folder.mkdir(parents=True,exist_ok=False);script='resume_stvg_r1_geometry32_dino_v1.py' if stage!='score' else 'stvg_r1_geometry32_measure_v1.py';put(BASE/'STATUS.json',dict(status='running',stage=stage,arm='DINO' if stage!='score' else None,count=32,GT_read=stage=='score'),replace=True)
   with (folder/'worker.log').open('xb') as log:
    tick=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),stage],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
   put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
 except BaseException as e:put(BASE/'MEASUREMENT_FAILURE003.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':main()
