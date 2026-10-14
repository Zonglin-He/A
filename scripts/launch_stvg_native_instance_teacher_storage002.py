"""Fail-closed resource supervisor for this scope's one spatial worker only."""
import json,os,sys,time,threading,subprocess,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import write,sha,read,verify
BASE=ROOT/'artifacts/stvg_native_instance_spatial_opd_v1'
STAGE=sys.argv[1];assert STAGE=='teacher'
UNIT='stvg-native-instance-'+STAGE+'-002.service';GiB=1024**3

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
   write(BASE/(STAGE.upper()+'_RESOURCE_SAMPLING_ERROR.json'),dict(status='engineering_sampling_error',count=errors,GT_read=False,time=time.time()),mutable=True)
   if errors<3:continue
  write(BASE/(STAGE.upper()+'_RESOURCE_STOP_RECEIPT.json'),dict(status='own_resource_guard_stop',reason=why,sample=values,unit=UNIT,other_processes_signalled=False,GT_read=False,time=time.time()),mutable=True)
  subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',UNIT],timeout=5,check=True)
  return

def main():
 verify(STAGE);v=sample();assert v['available']>=8*GiB and v['gpu_free']>=22*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState'],text=True).splitlines()if '='in x)
 assert int(props['MemoryHigh'])==16*GiB and int(props['MemoryMax'])==20*GiB and int(props['MemorySwapMax'])==GiB
 write(BASE/(STAGE.upper()+'_ACTUAL_LIVE_RESOURCE_LIMITS.json'),dict(status='pass',unit=UNIT,properties=props,admission=v,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=monitor,args=(stop,),daemon=True).start();accepted=[]
 interpreter=ROOT/('.conda/tubedetr/bin/python'if STAGE=='student'else'.venv/stvg_native_distill_v1/bin/python')
 rows=read(BASE/'ROSTER_LOCK.json')['records']
 for r in rows:
  case=r['case_id'];log=BASE/'logs'/f'{STAGE}_{case}.log';log.parent.mkdir(exist_ok=True)
  with log.open('xb')as f:
   child=subprocess.Popen([str(interpreter),'-B',str(ROOT/'scripts/run_stvg_native_instance_teacher_storage002.py'),'--case',case],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
   write(BASE/(STAGE.upper()+'_LAUNCH.json'),dict(status='running',controller_pid=os.getpid(),worker_pid=child.pid,case_id=case,done=len(accepted),total=8,unit=UNIT,GT_read=False,time=time.time()),mutable=True)
   code=child.wait()
  write(BASE/'worker_exits'/f'{STAGE}_{case}.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()))
  if code:
   stop.set();write(BASE/'STATUS.json',dict(status='failed',stage=STAGE,case_id=case,done=len(accepted),total=8,exit_code=code,GT_read=False,time=time.time()),mutable=True);return code
  out=BASE/'TRACKING'/case/'ACCEPTED.json'if STAGE=='tracking'else BASE/STAGE/(case+'.receipt.json')
  assert out.exists();accepted.append(dict(case_id=case,path=str(out.relative_to(ROOT)),sha256=sha(out)))
  write(BASE/'STATUS.json',dict(status='running',stage=STAGE,done=len(accepted),total=8,GT_read=False,time=time.time()),mutable=True)
 stop.set();write(BASE/(STAGE.upper()+'_GLOBAL_BARRIER.json'),dict(status='complete',count=len(accepted),files=accepted,runtime_sha256=sha(BASE/(STAGE.upper()+'_RUNTIME.json')),GT_read=False,time=time.time()))
 write(BASE/'STATUS.json',dict(status='phase_complete_pending_root',stage=STAGE,done=len(accepted),total=8,GT_read=False,time=time.time()),mutable=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
