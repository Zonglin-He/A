"""Guard a single 128-source parameter trial; never touch another scope's worker."""
from pathlib import Path
import sys,os,time,threading,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_component_ablation_common_v1 import *
from scripts import launch_stvg_instance_identity_v1 as resource
def main():
 verify_final();verify_preparation()
 trial=sys.argv[1];assert trial.isalnum();b=BASE/'trials'/trial;r=read(b/'RUNTIME.json')
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 unit='stvg-component128-'+trial.lower()+'-001.service';out=b/'execution001';out.mkdir(exist_ok=False);resource.BASE,resource.UNIT,resource.STAGE=out,unit,'student';g=1024**3
 v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert[int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,20*g,g]
 write(out/'ACTUAL_LIMITS.json',dict(status='pass',sample=v,properties=props,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with(out/'worker.log').open('xb')as f:
  p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_component_trial_v1.py',trial],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
  write(out/'LAUNCH.json',dict(status='running',worker_pid=p.pid,time=time.time(),GT_read=False));code=p.wait()
 stop.set();write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=False));return code
if __name__=='__main__':raise SystemExit(main())
