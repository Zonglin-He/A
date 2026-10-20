"""Explicit budget-only reopening of the immutable fixed-B8 local-motion probe."""
from pathlib import Path
import sys,time,signal,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts import run_stvg_temporal_local_motion_fixedbatch128_v1 as base
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
D=R/'artifacts/stvg_temporal_local_motion_completion128_v1'
P=R/'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128'
OLD,EARLY,PRE,FIN,DEV,STAGE,ARMS=base.OLD,base.EARLY,base.PRE,base.FIN,base.DEV,base.STAGE,base.ARMS

def prepare():
    old=read(base.D/'PLAN_LOCK.json')
    assert all(sha(R/f)==h for f,h in old['pins'].items())
    old_seal=read(base.D/'PILOT_CLOSURE_SEAL.json')
    assert old_seal['count']==2 and not (base.D/'GLOBAL_PREDICTION_BARRIER.json').exists()
    own=['scripts/run_stvg_temporal_local_motion_completion128_v1.py',
         'scripts/score_stvg_temporal_local_motion_completion128_v1.py',
         'scripts/publish_stvg_temporal_local_motion_completion128_v1.py',
         'protocols/stvg_temporal_local_motion_completion128_v1.md',
         str((base.D/'PLAN_LOCK.json').relative_to(R)),
         str((base.D/'PILOT_CLOSURE_SEAL.json').relative_to(R)),
         str((base.D/'COST.json').relative_to(R))]
    p=dict(old)
    p.update(GPU_seconds_cap=1800,created_epoch=time.time(),deadline_epoch=time.time()+14400,
             execution_revision='budget-only continuation; all128 rerun together with unchanged exact controls',
             user_authorized_finite_completion=True,old_round_GPU_seconds=read(base.D/'COST.json')['GPU_seconds'],
             old_plan_sha256=sha(base.D/'PLAN_LOCK.json'),
             old_budget_stop_preserved=True,no_automatic_next_experiment=True,
             prior_pilot_predictions={a:read(base.D/'predictions'/f'{a}.receipt.json')for a in ['RC001','RC002']})
    p['pins']=dict(old['pins']);p['pins'].update({f:sha(R/f)for f in own})
    for receipt in p['prior_pilot_predictions'].values():
        assert sha(R/receipt['path'])==receipt['sha256']
        p['pins'][receipt['path']]=receipt['sha256']
    put(D/'PLAN_LOCK.json',p)
    public={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch','prior_pilot_predictions']}
    put(P/'PROTOCOL_PUBLIC.json',public)
    print('locked budget-only full128 completion; 1800 GPU seconds',flush=True)

def measure():
    base.D,base.P=D,P
    base.measure()

if __name__=='__main__':
    def terminated(signum,frame):raise TimeoutError('owned1800s completion RuntimeMax SIGTERM')
    signal.signal(signal.SIGTERM,terminated)
    D.mkdir(parents=True,exist_ok=True)
    lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:globals()[sys.argv[1]]()
    except BaseException as e:
        put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),old_budget_stop_preserved=True))
        raise
    finally:lock.close()
