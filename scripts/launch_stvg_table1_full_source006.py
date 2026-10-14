"""Finite conditional handoff: successful 64-query preflight -> official Source."""
import os,sys,time,json,hashlib,subprocess,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import launch_stvg_instance_identity_v1 as resource
B=ROOT/'artifacts/stvg_final_table1_v1';BASE=B/'FULL/source_execution006';UNIT='stvg-table1-full-source-006.service';GiB=1024**3
resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='full_source'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rt=read(B/'FULL/SOURCE_EXECUTION_LOCK006.json')
 for p,h in rt['pins'].items():assert sha(ROOT/p)==h,p
 resource.write(BASE/'STATUS.json',dict(status='waiting_actual_preflight_exit_and_root_pass',GT_read=False,time=time.time()),mutable=True)
 while True:
  out=subprocess.check_output(['systemctl','--user','show','stvg-table1-preflight-005.service','-p','ActiveState','-p','Result'],text=True)
  props=dict(x.split('=',1)for x in out.splitlines()if '='in x)
  if props['ActiveState']not in ('active','activating','deactivating'):break
  time.sleep(15)
 assert props['Result']=='success',props
 assert read(B/'preflight_execution005/STATUS.json')['status']=='complete_engineering_preflight_pending_full_execution'
 root=read(B/'ROOT_PREFLIGHT64_READBACK.json');assert root['status']=='pass'and root['count']==64 and root['GT_read']is False
 # Independent alias-stable qualification repetition on the same query.
 import torch,numpy as np
 from scripts.stvg_native_instance_common_v1 import guard
 sys.addaudithook(guard)
 a=torch.load(B/'student_cases/HC01.pt',map_location='cpu',weights_only=False);z=torch.load(B/'student_preflight_cases/HC01.pt',map_location='cpu',weights_only=False)
 for name in a['arms']:
  assert np.array_equal(a['arms'][name]['boxes'],z['arms'][name]['boxes'])
  assert a['arms'][name]['interval']==z['arms'][name]['interval']
 for k,v in a['fits']['SpatialA']['final_state'].items():assert torch.equal(v,z['fits']['SpatialA']['final_state'][k])
 resource.write(BASE/'ROOT_REPEAT_HANDOFF.json',dict(status='pass',count=1,actual_complete_repeat=True,all_five_arm_boxes_and_intervals_exact=True,all_1792_final_state_exact=True,preflight_root_sha256=sha(B/'ROOT_PREFLIGHT64_READBACK.json'),GT_read=False,time=time.time()))
 del a,z
 v=resource.sample();assert v['available']>=8*GiB and v['gpu_free']>=22*GiB and v['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if '='in x)
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 resource.write(BASE/'STATUS.json',dict(status='running',stage='official_13785_source_capture',GT_read=False,time=time.time()),mutable=True)
 with (BASE/'worker.log').open('xb')as log:
  child=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B','scripts/run_stvg_table1_full_source_v1.py','--all'],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
  resource.write(BASE/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=False,time=time.time()));code=child.wait()
 stop.set();resource.write(BASE/'WORKER_EXIT.json',dict(status='actual_worker_exited',exit_code=code,GT_read=False,time=time.time()))
 resource.write(BASE/'STATUS.json',dict(status='source_complete_pending_expert_student'if code==0 else'failed',exit_code=code,GT_read=False,time=time.time()),mutable=True);return code
if __name__=='__main__':raise SystemExit(main())
