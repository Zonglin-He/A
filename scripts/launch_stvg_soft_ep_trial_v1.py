"""Single own-unit stage; bounded host RAM and GPU admission/guard."""
import os,sys,time,subprocess,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'
def main():
 label=sys.argv[1];assert label in ['mu025','mu4','response1'];base=BASE/label;stage=sys.argv[2];assert stage in ['qualification','development'];r=read(base/'RUNTIME.json')
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 unit='stvg-soft-ep-'+label+'-'+stage+'-001.service';out=base/('execution_'+stage+'001');out.mkdir(exist_ok=False)
 resource.BASE,resource.UNIT,resource.STAGE=out,unit,stage;g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert[int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,20*g,g]
 write(out/'ACTUAL_LIMITS.json',dict(status='pass',sample=v,properties=props,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with(out/'worker.log').open('xb')as f:
  p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_soft_ep_trial_v1.py',label,stage],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
  write(out/'LAUNCH.json',dict(status='running',worker_pid=p.pid,time=time.time(),GT_read=False));code=p.wait()
 stop.set();write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=False));return code
if __name__=='__main__':raise SystemExit(main())
