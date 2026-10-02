"""Finite CPU closeout after the complete P0 prediction barrier; no GPU jobs."""
import os,sys,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_event_support_common_v1 import *
def call(args):
 subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',*args],cwd=ROOT,check=True)
def run():
 try:
  status(BASE/'ROOT_FINISH_STATUS.json',dict(status='waiting_for_global_prediction_barrier',pid=os.getpid(),GT_read=False,time=time.time()))
  while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
   s=read(BASE/'STATUS.json')
   if s['status']=='failed':raise RuntimeError('GPU controller failed; no scoring of partial predictions')
   time.sleep(2)
  verify();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['cells']==1536
  for ds in DATASETS:
   call(['scripts/audit_tastvg_event_support_predictions_v1.py',ds])
   for arm in ['A','H']:
    status(BASE/'ROOT_FINISH_STATUS.json',dict(status='CPU_score_audit',dataset=ds,arm=arm,pid=os.getpid(),time=time.time()))
    call(['scripts/score_tastvg_event_support_v1.py',ds,arm])
  call(['scripts/readout_tastvg_event_support_v1.py'])
  call(['scripts/export_tastvg_event_support_v1.py'])
  dst=Path('/home/wwww/visual-grounding-public-A')/'results/tastvg_event_support/2026-10-02'
  r=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/audit_tastvg_event_support_public_v1.py',str(dst)],cwd=ROOT,capture_output=True,text=True,check=True)
  audit=__import__('json').loads(r.stdout);write(BASE/'ROOT_PUBLIC_SCALAR_AUDIT.json',audit);write(dst/'PUBLIC_SCALAR_AUDIT.json',audit)
  call(['scripts/draw_tastvg_event_support_v1.py',str(dst)])
  status(BASE/'ROOT_FINISH_STATUS.json',dict(status='completed_pending_root_review_remote_publication',pid=os.getpid(),time=time.time()))
 except BaseException as e:
  status(BASE/'ROOT_FINISH_STATUS.json',dict(status='failed',pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':run()
