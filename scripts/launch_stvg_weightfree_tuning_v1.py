"""Finite stage, single own GPU cgroup and existing resource protection."""
import os,sys,time,threading,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_weightfree_tuning_v1';STAGE=sys.argv[1];assert STAGE in ['teacher','student']
UNIT='stvg-weightfree-tuning-'+STAGE+'-001.service';GiB=1024**3
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE=STAGE

def main():
 rt=resource.read(BASE/(STAGE.upper()+'_RUNTIME.json'))
 for p,h in rt['pins'].items():assert resource.sha(ROOT/p)==h,p
 v=resource.sample()
 if v['available']<8*GiB or v['gpu_free']<26*GiB or v['disk_free']<30*GiB:raise RuntimeError('admission resource margin insufficient')
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 resource.write(BASE/(STAGE.upper()+'_ACTUAL_LIMITS.json'),dict(status='pass',properties=props,admission=v,GT_read=False))
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if STAGE=='teacher' else '.conda/tubedetr/bin/python')
 with (BASE/(STAGE+'.log')).open('xb')as log:
  child=subprocess.Popen([str(py),'-B',str(ROOT/('scripts/run_stvg_weightfree_tuning_'+STAGE+'_v1.py'))],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  resource.write(BASE/(STAGE.upper()+'_LAUNCH.json'),dict(status='running',unit=UNIT,controller_pid=os.getpid(),worker_pid=child.pid,GT_read=False,time=time.time()));code=child.wait()
 stop.set();resource.write(BASE/(STAGE.upper()+'_WORKER_EXIT.json'),dict(status='exited',exit_code=code,GT_read=False,time=time.time()));return code
if __name__=='__main__':raise SystemExit(main())
