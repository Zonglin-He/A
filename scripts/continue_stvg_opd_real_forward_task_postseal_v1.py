"""One bounded postseal handoff, not a recurring monitor or a research expansion."""
import os,subprocess,time,sys
from scripts.stvg_opd_real_forward_task_common_v1 import *

def run():
 pid=read(BASE/'FORMAL_LAUNCH.json')['pid'];deadline=time.time()+2400
 while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
  assert time.time()<deadline,'Bounded formal handoff deadline reached'
  stat=Path('/proc')/str(pid)/'stat';assert stat.exists(),'Finite formal worker exited without the complete seal'
  assert stat.read_text().split(') ',1)[1][0]!='Z','Finite formal worker failed before the complete seal'
  time.sleep(4)
 deadline=time.time()+120
 while not (BASE/'formal_worker_exit.json').exists():
  assert time.time()<deadline;time.sleep(1)
 verify()
 for module in ['scripts.score_stvg_opd_real_forward_task_v1','scripts.audit_stvg_opd_real_forward_task_v1','scripts.review_stvg_opd_real_forward_policy_independent_v1','scripts.report_stvg_opd_real_forward_task_v1']:
  print('POSTSEAL_STAGE',module,flush=True)
  result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','-m',module],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
  assert result.returncode==0,module
 status(BASE/'STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',done=256,total=256,CPU_only=True,GT_read=True,monitor_recreated=False,time=time.time()))
 write(BASE/'POSTSEAL_HANDOFF_EXIT.json',dict(status='pass',scope='one finite all256-global then CPU score/independent full roots/report; actual view/publication/archive still required',CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
