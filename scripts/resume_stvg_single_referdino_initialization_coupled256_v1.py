"""Resume the already-locked supplement after the owned primary GPU worker releases."""
import sys,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,put,read
PY=str(ROOT/'.conda/tubedetr/bin/python');beg=time.time();active=None

def run(script,args):
 global active
 print('COUPLING_RECOVERY',script,*args,flush=True)
 log=D/'initialization_coupled'/('resume_'+('_'.join(args))+'.log')
 with log.open('a')as f:
  active=subprocess.Popen([PY,'-B','scripts/'+script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);rc=active.wait()
 assert rc==0,(script,args,rc)
try:
 assert (D/'initialization_coupled/PLAN_LOCK.json').exists()
 while not(D/'vidstg/FINE_FIT_BARRIER.json').exists():
  assert time.time()-beg<4000
  assert not list(D.glob('CONTROLLER_FAILURE_*.json'))
  time.sleep(3)
 # The worker writes its barrier just before its finally block persists cost and closes lease.
 while True:
  with (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:
    assert time.time()-beg<4100;time.sleep(2);continue
  break
 for phase in ['coarse','fine']:
  run('run_stvg_single_referdino_initialization_coupled256_v1.py',['hc2',phase])
  run('score_stvg_single_referdino_initialization_coupled256_v1.py',['seal','hc2',phase])
  run('audit_stvg_single_referdino_initialization_coupled256_v1.py',['pre-gt','hc2',phase])
  run('score_stvg_single_referdino_initialization_coupled256_v1.py',['score','hc2',phase])
  run('audit_stvg_single_referdino_initialization_coupled256_v1.py',['scored','hc2',phase])
  if phase=='coarse':run('score_stvg_single_referdino_initialization_coupled256_v1.py',['fine-plan','hc2'])
 put(D/'INITIALIZATION_COUPLING_COMPLETION.json',dict(status='completed_both_phases_and_independent_audits',wall=time.time()-beg,prior_nonblocking_lease_collision_preserved=True))
except BaseException as e:
 put(D/('COUPLING_RESUME_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e),owned_active_pid=active.pid if active and active.poll()is None else None));raise
