"""Finite dependent tail: coordinate completion, temporal coordinates, locked 32-source confirmation."""
import sys,os,time,json,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1';H=BASE/'confirmation';PY=ROOT/'.conda/tubedetr/bin/python'

def call(label,args):
 with (BASE/(label+'.log')).open('xb')as log:
  child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
  write(BASE/(label.upper()+'_LAUNCH.json'),dict(status='running',pid=child.pid,time=time.time()))
  code=child.wait();write(BASE/(label.upper()+'_WORKER_EXIT.json'),dict(status='exited',exit_code=code,time=time.time()))
  if code:raise RuntimeError(f'{label} exited {code}')

def main():
 write(BASE/'TAIL_STATUS.json',dict(status='waiting',stage='spatial coordinate dependency',GT_read=False),mutable=True)
 # This is a finite experiment dependency, not a recurring monitor or a second GPU worker.
 while True:
  state=subprocess.check_output(['systemctl','--user','show','stvg-weightfree-coordinate-002.service','--property=ActiveState','--value'],text=True).strip()
  if state not in ['active','activating','deactivating']:break
  time.sleep(10)
 assert state=='inactive',state
 assert read(BASE/'SPATIAL_CPU_COMPLETION.json')['status']=='complete'
 call('temporal_coordinates',[str(PY),'-B',str(ROOT/'scripts/run_stvg_coordinate_temporal_v1.py'),'controller'])
 assert read(BASE/'TEMPORAL_SELECTION_LOCK.json')['status']=='locked'
 # Lock the complete recipe before reading any confirmation teacher or scoring outputs.
 write(BASE/'JOINT_METHOD_LOCK.json',dict(status='locked',spatial=read(BASE/'SPATIAL_SELECTION_LOCK.json')['configs'],temporal=read(BASE/'TEMPORAL_SELECTION_LOCK.json')['configs'],observation_clock='original QA configuration for proposal windows; tuned temporal config only changes final interval',confirmation_used_for_selection=False,GT_read=False,CURRENT_promoted=False,time=time.time()))
 old=ROOT/'artifacts/stvg_weightfree_tuning_v1/confirmation003'
 for name in ['RUNTIME.json','TEACHER_RUNTIME.json','STUDENT_RUNTIME.json']:
  rt=read(old/name)
  for p in list((ROOT/'scripts').glob('*coordinate*.py'))+[BASE/'JOINT_METHOD_LOCK.json',BASE/'SPATIAL_SELECTION_LOCK.json',BASE/'TEMPORAL_SELECTION_LOCK.json',BASE/'CONFIRMATION_ROSTER_LOCK.json']:
   rt['pins'][str(p.relative_to(ROOT))]=sha(p)
  rt['GT_read']=False;rt['confirmation_GT_read']=False
  write(H/name,rt)
 for stage in ['temporal','candidates','teacher','references','student','root','score']:
  write(BASE/'TAIL_STATUS.json',dict(status='running',stage='confirmation '+stage,GT_read=stage=='score'),mutable=True)
  if stage in ['temporal','candidates','teacher','student']:
   unit='stvg-coordinate-confirm-'+stage+'-001'
   args=['systemd-run','--user','--wait','--pipe','--unit='+unit,'--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--property=OOMPolicy=stop','--working-directory='+str(ROOT),'/usr/bin/python3','-B',str(ROOT/'scripts/launch_stvg_coordinate_confirmation_v1.py'),stage]
  else:
   script={'references':'run_stvg_coordinate_confirmation_v1.py','root':'review_stvg_coordinate_confirmation_v1.py','score':'score_stvg_coordinate_confirmation_v1.py'}[stage]
   args=[str(PY),'-B',str(ROOT/'scripts'/script)]+(['references']if stage=='references'else[])
  call('confirmation_'+stage,args)
 write(BASE/'MEASUREMENT_HANDOFF.json',dict(status='pending_actual_independent_final_readback_report_view_public_archive',development_sources=32,confirmation_sources=32,scope='finite sequential per-domain parameter search and frozen confirmation',GT_read=True,time=time.time(),not_FINAL=True))
 write(BASE/'TAIL_STATUS.json',dict(status='measurement_complete_pending_root_closure',stage='root handoff',GT_read=True),mutable=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'TAIL_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
