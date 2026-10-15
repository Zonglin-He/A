"""One finite question-role prompt ablation, separate from spatial selection."""
from pathlib import Path
import sys,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_temporal_role_alignment_v1/revision002';PY=ROOT/'.conda/tubedetr/bin/python'
def stage(phase):
 write(BASE/'STATUS.json',dict(status='running',stage=phase+'_teacher',GT_read=False,time=time.time()),mutable=True)
 subprocess.run(['systemd-run','--user','--wait','--collect','--unit=stvg-temporal-role-'+phase.lower()+'-001','--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),str(PY),'-B','scripts/launch_stvg_temporal_role_alignment002.py',phase],cwd=ROOT,check=True)
 assert read(BASE/phase/'execution001/WORKER_EXIT.json')['exit_code']==0
 subprocess.run([str(PY),'-B','scripts/run_stvg_temporal_role_alignment002.py','fit',phase],cwd=ROOT,check=True)
 subprocess.run([str(PY),'-B','scripts/review_stvg_temporal_role_alignment002.py',phase],cwd=ROOT,check=True)
 subprocess.run([str(PY),'-B','scripts/measure_stvg_temporal_role_alignment002.py',phase],cwd=ROOT,check=True)
 assert read(BASE/phase/'CPU_COMPLETION.json')['status']=='complete'
def main():
 rt=read(BASE/'RUNTIME.json');assert rt['status']=='locked'and rt['GT_read']is False
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 assert not(BASE/'CPU_COMPLETION.json').exists()
 stage('P1');p1=read(BASE/'P1/CPU_COMPLETION.json');assert isinstance(p1['p2_eligible'],bool)
 phases=['P1']
 p2_pending=False
 if p1['p2_eligible']:stage('P2');phases.append('P2')
 write(BASE/'CPU_COMPLETION.json',dict(status='complete',scope='finite P1 and conditionally P2 developmental comparison; not method/public/archive FINAL',phases=phases,P2_executed='P2'in phases,P2_pending=p2_pending,complete_current_prompt_authorization=not p2_pending,no_bootstrap=True,full_evaluation_started=False,CURRENT_promoted=False,GT_read=True,time=time.time()))
 write(BASE/'STATUS.json',dict(status='pending_actual_independent_root_view_public_archive',stage='root_handoff',GT_read=True,time=time.time()),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  exposed=any(BASE.glob('P*/GT_EXPOSURE.json'));write(BASE/'FAILURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=exposed,time=time.time()),mutable=True);raise
