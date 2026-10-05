"""One finite CPU handoff; wait for global seal, then score and diagnose."""
import os,sys,time,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_fixed_full_common_v1 import *
def run():
 f=(BASE/'CPU_HANDOFF.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
  s=read(BASE/'STATUS.json')
  if s['status']=='failed':raise RuntimeError('GPU controller failed; root recovery required, no partial scoring')
  pid=s.get('controller_pid')
  assert pid and Path(f'/proc/{pid}').exists(),'GPU controller disappeared without global seal; no partial scoring'
  status(BASE/'CPU_STAGE.json',dict(status='waiting_for_global_prediction_barrier',pid=os.getpid(),GT_read=False,time=time.time()))
  time.sleep(30)
 verify();pins=read(BASE/'CPU_RUNTIME_LOCK.json')['pins']
 for rel,h in pins.items():assert sha(ROOT/rel)==h,rel
 log=BASE/'score_cpu.log'
 with log.open('ab') as out:
  p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/score_decota_fixed_full_v1.py'],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
  status(BASE/'CPU_STAGE.json',dict(status='running_CPU_score_diagnosis',pid=os.getpid(),score_pid=p.pid,log=str(log),time=time.time()))
  assert p.wait()==0,'CPU scoring failed; preserve inputs and ask root engineering recovery'
 assert read(BASE/'CPU_COMPLETION.json')['status']=='completed_pending_root_visual_publication'
 status(BASE/'CPU_STAGE.json',dict(status='completed_pending_root_visual_publication',time=time.time(),pid=os.getpid()))
if __name__=='__main__':
 try:run()
 except BaseException:
  fd=BASE/'CPU_failure'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'CPU_FAILURE.json',dict(status='failed',failure_path=str(fd),pid=os.getpid(),time=time.time()));raise
