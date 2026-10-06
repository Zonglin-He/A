"""One finite serial queue; all method outputs sealed before each CPU audit."""
import sys,os,time,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_spatial_opd_common_v1 import *

def run():
 BASE.mkdir(parents=True,exist_ok=True);gate=(BASE/'CONTROLLER.lock').open('a');fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock();assert read(PAPER/'user_opd_pause_20261006/EXACT_RESUME_RECEIPT.json')['status']=='pass'
 def step(script,action):
  verify();budget();log=BASE/(Path(script).stem+'_'+action+'.log')
  with log.open('ab') as out:
   p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/script),action],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
   status(BASE/'STATUS.json',dict(status='running',pid=os.getpid(),controller_pid=os.getpid(),worker_pid=p.pid,stage=action,script=script,log=str(log),GT_read=False,time=time.time()))
   assert p.wait()==0,'Finite OPD stage failed; preserve artifacts and root review required: '+action
 if not (BASE/'QUALIFICATION.json').exists():
  for ds in ['hc2','vidstg']:
   if not (BASE/'qualification'/('dev_'+ds)/'QUALIFICATION.json').exists():
    step('scripts/run_decota_spatial_opd_v1.py','qual_dev_'+ds)
  step('scripts/run_decota_spatial_opd_v1.py','qualification')
 archive('四真实query两offset接口/末轮执行/独立NumPy反馈梯度和Adam资格通过，正式三臂开发→更大来源互斥确认→同域5%复核有限队列实际接续')
 for phase in ['dev','confirm','mechanism']:
  for ds in ['hc2','vidstg']:
   name=phase+'_'+ds
   if not (BASE/'stages'/name/'PREDICTION_BARRIER.json').exists():step('scripts/run_decota_spatial_opd_v1.py',name)
  # Both directions of this phase must seal before any phase GT scoring.
  for ds in ['hc2','vidstg']:
   name=phase+'_'+ds
   if not (BASE/'stages'/name/'CPU_COMPLETION.json').exists():step('scripts/score_decota_spatial_opd_v1.py',name)
 write(BASE/'GPU_CPU_COMPLETION.json',dict(status='all_2880_predictions_and_postseal_CPU_audits_complete_pending_root_publication',adapted_arrivals=2880,
  qualification_queries=4,outputs_separate_from_old_paper=True,old_paper_resume_required=True,time=time.time()))
 status(BASE/'STATUS.json',dict(status='completed_pending_root_visual_publication_and_saved_paper_resume',GT_read=True,time=time.time()))
 archive('全部2880新策略/控制预测与阶段封存后CPU数学/状态/dense/源bootstrap实际完成，根须图表目检/匿名代码结果GitHub独立分支远端核验后接续已保存旧paper队列；未自动晋升')

if __name__=='__main__':
 try:run()
 except BaseException:
  fd=BASE/'controller_failure'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed',pid=os.getpid(),failure=str(fd),time=time.time()));raise
