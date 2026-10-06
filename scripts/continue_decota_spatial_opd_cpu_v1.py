"""One finite CPU continuation after the existing OPD prediction controller.

This creates a review draft and figures; root still has to view them, publish
verified anonymous contents, and authorize the already requested paper resume
by an actual closing receipt. It never starts another GPU or changes the run.
"""
import os,sys,time,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_spatial_opd_common_v1 import *

def active(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split(') ')[1].split()[0]!='Z'

def run():
 g=(BASE/'ROOT_CPU_CONTROLLER.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=read(BASE/'LAUNCH.json');pid=launch['pid'];pins=read(BASE/'ROOT_CONTINUATION_RUNTIME_v2.json')
 for p,h in pins['pins'].items():assert sha(ROOT/p)==h,p
 write(BASE/'CPU_STAGE.json',dict(status='waiting_for_existing_OPD_controller',pid=os.getpid(),predecessor_pid=pid,GT_read=False,time=time.time()))
 while active(pid):
  command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
  assert b'continue_decota_spatial_opd_v1.py' in command,'Predecessor PID reused'
  time.sleep(10)
 assert read(BASE/'STATUS.json')['status']=='completed_pending_root_visual_publication_and_saved_paper_resume','No valid OPD all-stage handoff'
 assert read(BASE/'GPU_CPU_COMPLETION.json')['adapted_arrivals']==2880
 verify()
 log=BASE/'finalize_root.log'
 with log.open('ab') as out:
  p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/finalize_decota_spatial_opd_v1.py'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  write(BASE/'CPU_STAGE.json',dict(status='root_aggregate_audit_report_render_running',pid=os.getpid(),worker_pid=p.pid,log=str(log),GT_read=True,time=time.time()))
  assert p.wait()==0,'Root postseal audit failed; retain originals, no GPU rerun'
 assert read(BASE/'ROOT_AUDIT_REPORT_COMPLETION.json')['adapted_arrivals']==2880
 write(BASE/'CPU_STAGE.json',dict(status='completed_pending_actual_visual_publication_and_saved_paper_resume',pid=os.getpid(),GT_read=True,time=time.time()))

if __name__=='__main__':
 try:run()
 except BaseException:
  f=BASE/'root_CPU_failures'/str(time.time_ns());f.mkdir(parents=True,exist_ok=True);(f/'traceback.txt').write_text(traceback.format_exc())
  write(BASE/'CPU_STAGE.json',dict(status='failed',pid=os.getpid(),failure=str(f),GT_read=False,time=time.time()));raise
