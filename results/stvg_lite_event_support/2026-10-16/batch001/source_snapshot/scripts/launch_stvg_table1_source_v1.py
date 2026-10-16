"""Fixed 64-source capture under the existing own-unit resource policy."""
import os,sys,time,threading,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_final_table1_v1/source_execution001';UNIT='stvg-table1-source-001.service';GiB=1024**3
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='source'
def main():
 v=resource.sample();assert v['available']>=8*GiB and v['gpu_free']>=22*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 resource.write(BASE/'RESOURCE_ADMISSION.json',dict(status='pass',sample=v,properties=props,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with (BASE/'worker.log').open('xb')as log:
  child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/run_stvg_table1_source_v1.py'),'--all-preflight'],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  resource.write(BASE/'LAUNCH.json',dict(status='running',pid=child.pid,controller_pid=os.getpid(),GT_read=False,time=time.time()));code=child.wait()
 stop.set();resource.write(BASE/'WORKER_EXIT.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()));return code
if __name__=='__main__':raise SystemExit(main())
