"""Single own-unit GPU stage with bounded host/GPU resource guard."""
import os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import launch_stvg_instance_identity_v1 as resource

def main():
 stage=sys.argv[1];assert stage in ['teacher','student']
 base=ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap'
 runtime=read(base/('RUNTIME.json' if stage=='teacher' else 'STUDENT_RUNTIME.json'));assert runtime['status']=='locked' and not runtime['GT_read']
 for p,h in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/p)==h,p
 unit='stvg-reference-swap-'+stage+'-006.service';out=base/('execution_'+stage+'006');out.mkdir(parents=True,exist_ok=False)
 resource.BASE,resource.UNIT,resource.STAGE=out,unit,stage;GiB=1024**3;s=resource.sample()
 assert s['available']>=8*GiB and s['gpu_free']>=26*GiB and s['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 write(out/'ACTUAL_LIMITS.json',dict(status='pass',admission=s,properties=props,GT_read=False))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 script='run_stvg_reference_swap_teacher002.py' if stage=='teacher' else 'run_stvg_reference_swap_student004.py'
 cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/'+script]+(['--run'] if stage=='teacher' else [])
 with(out/'worker.log').open('xb')as log:
  p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  write(out/'LAUNCH.json',dict(status='running',worker_pid=p.pid,controller_pid=os.getpid(),time=time.time(),GT_read=False));code=p.wait()
 stop.set();write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=False));return code
if __name__=='__main__':raise SystemExit(main())
