"""Fail-closed resource supervisor for this scope's one spatial worker only."""
import json,os,sys,time,threading,subprocess,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import write,sha
BASE=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1'
UNIT='stvg-motion-guided-teacher-001.service';GiB=1024**3

def sample():
 line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 assert line[3:].endswith('/'+UNIT),'supervisor must be in its own named unit'
 cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
 stats={k:int(v) for k,v in (x.split() for x in (cg/'memory.stat').read_text().splitlines())}
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0])*1024**2
 own={int(x) for p in cg.rglob('cgroup.procs') for x in p.read_text().split()}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=3)
 active=bool(own & {int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()})
 return dict(current=int((cg/'memory.current').read_text()),anon=stats.get('anon',0)+stats.get('shmem',0),available=available,gpu_free=free,own_gpu_active=active,disk_free=shutil.disk_usage(ROOT).free)

def trigger(v):
 if v['current']>=18*GiB:return 'own_RAM_18GiB'
 if v['anon']>=14*GiB:return 'own_nonreclaimable_RAM_14GiB'
 if v['available']<6*GiB:return 'host_RAM_headroom_6GiB'
 if v['own_gpu_active'] and v['gpu_free']<4*GiB:return 'GPU_headroom_4GiB'
 if v['disk_free']<30*GiB:return 'disk_floor_30GiB'
 return None

def monitor(stop):
 errors=0
 while not stop.wait(.5):
  try:
   values=sample();errors=0;why=trigger(values)
   if not why:continue
  except Exception as e:
   errors+=1;values={'error_type':type(e).__name__};why='resource_sampling_failed_three_consecutive_times'
   write(BASE/'RESOURCE_SAMPLING_ERROR.json',dict(status='engineering_sampling_error',count=errors,GT_read=False,time=time.time()),mutable=True)
   if errors<3:continue
  write(BASE/'RESOURCE_STOP_RECEIPT.json',dict(status='own_resource_guard_stop',reason=why,sample=values,unit=UNIT,other_processes_signalled=False,GT_read=False,time=time.time()),mutable=True)
  subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',UNIT],timeout=5,check=True)
  return

def main():
 v=sample();assert v['available']>=8*GiB and v['gpu_free']>=22*GiB and v['disk_free']>=30*GiB,'admission headroom'
 effective=subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState'],text=True)
 props=dict(x.split('=',1) for x in effective.splitlines() if '=' in x)
 assert int(props['MemoryHigh'])==16*GiB and int(props['MemoryMax'])==20*GiB and int(props['MemorySwapMax'])==GiB
 write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=monitor,args=(stop,),daemon=True).start()
 log=BASE/'logs/teacher_worker001.log';log.parent.mkdir(exist_ok=True)
 with log.open('xb') as f:
  child=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(ROOT/'scripts/run_stvg_motion_guided_teacher_v1.py')],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
  write(BASE/'TEACHER_LAUNCH.json',dict(status='actual_launched',controller_pid=os.getpid(),worker_pid=child.pid,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,time=time.time()))
  code=child.wait()
 stop.set();write(BASE/'TEACHER_WORKER_EXIT.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()))
 return code

if __name__=='__main__':raise SystemExit(main())
