"""Serial post-R1 measurement, waits original process and never overlaps Teacher/Student."""
from pathlib import Path
import os,select,subprocess,sys,time,threading,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,put,read,verify
UNIT='stvg-r1-geometry32-measure-002.service'
def main():
 props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show','stvg-r1-geometry32-002.service','-p','MainPID'],text=True).splitlines());pid=int(props['MainPID'])
 if pid:
  cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes();assert b'run_stvg_r1_geometry32_v2.py' in cmd
  fd=os.pidfd_open(pid);p=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)
 verify();assert read(BASE/'R1_GLOBAL_BARRIER.json')['status']=='complete'
 from scripts import launch_stvg_instance_identity_v1 as resource
 resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='r1geom_measure';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 try:
  stages=[('r1root',None),('dino',None),('prepare',None)]+[(s,a) for a in ['DIRECT','HEADS','DINO'] for s in ['worker','root']]+[('score',None)]
  for stage,arm in stages:
   verify();put(BASE/'STATUS.json',dict(status='running',stage=stage,arm=arm,count=32,GT_read=stage=='score'),replace=True);folder=BASE/'execution_measure'/(arm or stage)/stage;folder.mkdir(parents=True,exist_ok=False)
   with (folder/'worker.log').open('xb') as log:
    tick=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/stvg_r1_geometry32_measure_v1.py'),stage]+([arm] if arm else []),cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
   put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,arm)
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False))
 except BaseException as e:put(BASE/'MEASUREMENT_FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':main()
