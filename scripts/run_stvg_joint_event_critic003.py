"""Only correct discrete time membership; retain source data and all budgets."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_joint_event_critic_v1 as run
def verify():
 lock=run.read(run.BASE/'RUNTIME_LOCK003.json');p=run.read(run.BASE/'PLAN_LOCK.json')
 assert run.sha(run.BASE/'PLAN_LOCK.json')==lock['plan_sha256']
 assert time.time()<p['deadline_epoch']
 for rel,h in lock['pins'].items():assert run.sha(ROOT/rel)==h,rel
 from main_method.runtime import resource_guard
 resource_guard();return p
if __name__=='__main__':
 run.verify=verify
 try:getattr(run,sys.argv[1])()
 except BaseException as e:
  run.put(run.BASE/('FAILURE003_'+sys.argv[1]+'_'+str(time.time_ns())+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),partials_preserved=True));raise
