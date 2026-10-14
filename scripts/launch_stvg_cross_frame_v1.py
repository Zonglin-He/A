"""One own cgroup, one hot Student worker, bounded fail-closed resources."""
import os,sys,time,threading,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,read,write,sha,verify
from scripts import launch_stvg_instance_identity_v1 as resource
UNIT='stvg-cross-frame-student-001.service'
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='student'
GiB=1024**3

def main():
 verify();v=resource.sample()
 if v['available']<8*GiB or v['gpu_free']<22*GiB or v['disk_free']<30*GiB:raise RuntimeError('admission resource headroom insufficient')
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState'],text=True).splitlines()if '='in x)
 if [int(props[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]!=[16*GiB,20*GiB,GiB]:raise RuntimeError('own cgroup limits differ')
 attempt=BASE/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
 write(attempt/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=props,admission=v,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with (attempt/'student_hot.log').open('xb') as log:
  worker=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/run_stvg_cross_frame_student_v1.py')],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  launch=dict(status='running',unit=UNIT,controller_pid=os.getpid(),worker_pid=worker.pid,attempt_path=str(attempt.relative_to(ROOT)),GT_read=False,time=time.time())
  write(attempt/'LAUNCH.json',launch);write(BASE/'LAUNCH.json',launch,mutable=True)
  code=worker.wait()
 stop.set();write(attempt/'WORKER_EXIT.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()))
 write(BASE/'STATUS.json',dict(status='student_complete_pending_root' if code==0 else 'failed',stage='cross_frame_student',exit_code=code,GT_read=False,time=time.time()),mutable=True)
 return code
if __name__=='__main__':raise SystemExit(main())
