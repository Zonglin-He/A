"""One own-cgroup child with the existing bounded RAM/GPU/disk guard."""
import sys,os,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as resource
from scripts.stvg_evidence_common_v1 import BASE,read,write,sha

def main():
 phase=sys.argv[1];label=sys.argv[2];script=sys.argv[3];args=sys.argv[4:];unit='stvg-evidence-'+label+'-001.service';out=BASE/phase/'execution'/label;out.mkdir(parents=True,exist_ok=True)
 resource.BASE=out;resource.UNIT=unit;resource.STAGE=label
 v=resource.sample();GiB=1024**3;assert v['available']>=8*GiB and v['gpu_free']>=26*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if '='in x)
 assert[int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 write(out/'ACTUAL_LIMITS.json',dict(status='pass',properties=props,admission=v,GT_read=False,time=time.time()))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 py=ROOT/('.venv/stvg_native_distill_v1/bin/python'if label.endswith('teacher')or label.endswith('temporal_teacher')else'.conda/tubedetr/bin/python')
 with(out/'worker.log').open('xb')as log:
  phase_runtime=BASE/phase/'RUNTIME005.json'
  command=([str(py),'-B',str(ROOT/'scripts/stvg_evidence_stage003.py'),str(phase_runtime.relative_to(ROOT)),script,*args]if phase_runtime.exists()else[str(py),'-B',str(ROOT/script),*args])
  p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT);write(out/'LAUNCH.json',dict(status='running',unit=unit,controller_pid=os.getpid(),worker_pid=p.pid,GT_read=False,time=time.time()));code=p.wait()
 stop.set();write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time()));return code
if __name__=='__main__':raise SystemExit(main())
