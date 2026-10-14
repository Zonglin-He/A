"""One KV-equivalence worker under the existing own-unit resource policy."""
import os,sys,time,threading,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_native_hot_teacher_v1/task_equivalence003';UNIT='stvg-native-hot-task-003.service';GiB=1024**3
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='teacher'
def main():
 v=resource.sample();assert v['available']>=8*GiB and v['gpu_free']>=26*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 resource.write(BASE/'RESOURCE_ADMISSION.json',dict(status='pass',sample=v,properties=props,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with (BASE/'worker.log').open('xb')as log:
  child=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(ROOT/'scripts/benchmark_stvg_native_kv_task003.py')],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  resource.write(BASE/'LAUNCH.json',dict(status='running',pid=child.pid,controller_pid=os.getpid(),GT_read=False,time=time.time()));code=child.wait()
 stop.set();resource.write(BASE/'WORKER_EXIT.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()));return code
if __name__=='__main__':raise SystemExit(main())
