"""Single bounded confirmation phase with the existing own-unit resource guard."""
import sys,os,subprocess,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as r
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1/confirmation';STAGE=sys.argv[1]
assert STAGE in ['temporal','candidates','teacher','student']
UNIT='stvg-coordinate-confirm-' +STAGE+'-001.service';GiB=1024**3
r.BASE=BASE;r.UNIT=UNIT;r.STAGE=STAGE
def main():
 rt=r.read(BASE/'RUNTIME.json')
 for p,h in rt['pins'].items():assert r.sha(ROOT/p)==h,p
 v=r.sample();assert v['available']>=8*GiB and v['gpu_free']>=26*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 r.write(BASE/(STAGE.upper()+'_ACTUAL_LIMITS.json'),dict(status='pass',properties=props,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
 py=ROOT/('.venv/stvg_native_distill_v1/bin/python'if STAGE in ['temporal','teacher']else'.conda/tubedetr/bin/python')
 script='scripts/run_stvg_coordinate_confirmation_v1.py';args=[STAGE]
 if STAGE=='teacher':script='scripts/run_stvg_coordinate_confirmation_teacher_v1.py';args=[]
 if STAGE=='student':script='scripts/run_stvg_coordinate_confirmation_student_v1.py';args=[]
 with (BASE/(STAGE+'.log')).open('xb')as log:
  p=subprocess.Popen([str(py),'-B',str(ROOT/script),*args],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  r.write(BASE/(STAGE.upper()+'_LAUNCH.json'),dict(status='running',unit=UNIT,controller_pid=os.getpid(),worker_pid=p.pid,GT_read=False,time=time.time()));code=p.wait()
 stop.set();r.write(BASE/(STAGE.upper()+'_WORKER_EXIT.json'),dict(status='exited',exit_code=code,GT_read=False,time=time.time()));return code
if __name__=='__main__':raise SystemExit(main())
