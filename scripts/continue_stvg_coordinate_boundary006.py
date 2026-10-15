"""Finite engineering resumption after Source ROI boundary error; no retuning."""
import sys,os,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1';H=B/'confirmation';D=H/'boundary_adapter005/execution006';PY=ROOT/'.conda/tubedetr/bin/python'
def call(stage,args):
 with (D/(stage+'.log')).open('xb')as f:
  p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
  write(D/(stage.upper()+'_LAUNCH.json'),dict(status='running',pid=p.pid,time=time.time()));code=p.wait();write(D/(stage.upper()+'_EXIT.json'),dict(status='exited',exit_code=code,time=time.time()))
  assert code==0,(stage,code)
def main():
 for p,h in read(D/'REVISION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
 for phase in ['candidates','boundary_readback','teacher','references','student','root','score']:
  write(B/'TAIL_STATUS.json',dict(status='running',stage='confirmation '+phase,revision='boundary_adapter005',GT_read=phase=='score'),mutable=True)
  if phase in ['candidates','teacher','student']:
   suffix='006'if phase=='candidates'else'001';script='launch_stvg_coordinate_confirmation_boundary006.py'if phase=='candidates'else'launch_stvg_coordinate_confirmation_v1.py'
   args=['systemd-run','--user','--wait','--pipe','--unit=stvg-coordinate-confirm-'+phase+'-'+suffix,'--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--property=OOMPolicy=stop','--working-directory='+str(ROOT),'/usr/bin/python3','-B',str(ROOT/'scripts'/script),phase]
  else:
   script={'boundary_readback':'review_stvg_coordinate_boundary005.py','references':'run_stvg_coordinate_confirmation_v1.py','root':'review_stvg_coordinate_confirmation_boundary006.py','score':'score_stvg_coordinate_confirmation_v1.py'}[phase];args=[str(PY),'-B',str(ROOT/'scripts'/script)]+(['references']if phase=='references'else[])
  call('resume_'+phase,args)
 write(B/'MEASUREMENT_HANDOFF.json',dict(status='pending_actual_independent_final_readback_report_view_public_archive',development_sources=32,confirmation_sources=32,scope='finite sequential per-domain parameter search and frozen confirmation',GT_read=True,time=time.time(),not_FINAL=True))
 write(B/'TAIL_STATUS.json',dict(status='measurement_complete_pending_root_closure',stage='root handoff',GT_read=True),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(D/'FAILURE_CAPTURE.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()));write(B/'TAIL_STATUS.json',dict(status='failed',revision='boundary_adapter005',GT_read=False),mutable=True);raise
