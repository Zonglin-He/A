"""One bounded, serial T2 run; OPD remains paused by direct human instruction."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_t2_v1 import BASE,verify,put
UNIT='stvg-temporal-t2-vs-t4-001.service'
def main():
 from scripts import launch_stvg_instance_identity_v1 as resource
 verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='t2';g=2**30;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines() if '='in x);assert [int(props[k]) for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
 put(BASE/'ACTUAL_RESOURCE_READBACK.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 try:
  for mode in ('prepare','teacher','fit','readback','score'):
   verify();put(BASE/'STATUS.json',dict(status='running',stage=mode,count=256,model_started=mode not in ('prepare',),GT_read=mode=='score'),replace=True);folder=BASE/'execution'/mode;folder.mkdir(parents=True,exist_ok=False);interp=ROOT/('.venv/stvg_native_distill_v1/bin/python' if mode=='teacher' else '.conda/tubedetr/bin/python');script=ROOT/('scripts/score_stvg_temporal_t2_v1.py' if mode=='score' else 'scripts/stvg_temporal_t2_v1.py');command=[str(interp),'-B',str(script)]+([] if mode=='score' else [mode])
   with (folder/'worker.log').open('xb') as log:
    tick=time.perf_counter();child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(folder/'LAUNCH.json',dict(status='running',pid=child.pid,controller_pid=os.getpid(),unit=UNIT,GT_read=mode=='score'));rc=child.wait()
   put(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=mode=='score'));assert rc==0,(mode,rc)
  put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=256,root_independent_science_privacy_actual_pixel_remote_and_archive_required=True,GT_read=False));put(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=256,GT_read=False),replace=True)
 except BaseException as e:
  put(BASE/'FAILURE.json',dict(status='technical_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
 finally:stop.set()
if __name__=='__main__':main()
