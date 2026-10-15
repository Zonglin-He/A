"""Guard one bounded P1/P2 frozen teacher worker under its own unit."""
from pathlib import Path
import sys,os,subprocess,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_temporal_role_alignment_v1/revision002'
def main():
 phase=sys.argv[1];assert phase in ['P1','P2'];rt=read(BASE/'RUNTIME.json');assert rt['status']=='locked'and rt['GT_read']is False
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 out=BASE/phase/'execution001';out.mkdir(parents=True,exist_ok=False);unit='stvg-temporal-role-'+phase.lower()+'-001.service';resource.BASE,resource.UNIT,resource.STAGE=out,unit,'teacher';g=1024**3
 v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert[int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,20*g,g]
 write(out/'ACTUAL_LIMITS.json',dict(status='pass',sample=v,properties=props,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 with(out/'teacher.log').open('xb')as f:
  p=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B','scripts/run_stvg_temporal_role_alignment002.py','teacher',phase],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
  write(out/'LAUNCH.json',dict(status='running',worker_pid=p.pid,time=time.time(),GT_read=False));code=p.wait()
 stop.set();write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=False));return code
if __name__=='__main__':raise SystemExit(main())
