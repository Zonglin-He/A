"""Serial execution of the already locked two-seed fit and confirmation only.

Resume only safe completed-window pauses. Any failure exits for root repair.
Every wrapper overhead is measured separately without double-counting worker.
"""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_joint_learnability import D,OUT
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,total_prior
PY=ROOT/'.venv-ptd-audit/bin/python'

def active(data):
    p=D/'ACTIVE.tmp.json';p.write_text(json.dumps(data,indent=2)+'\n');os.replace(p,D/'ACTIVE.json')

def execute(name,cmd):
    log=D/(name+'.log');assert not log.exists();start=time.monotonic()
    with log.open('xb') as f:
        p=subprocess.Popen([str(PY),'-B',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        active(dict(controller_pid=os.getpid(),pid=p.pid,name=name,cmd=cmd,log=str(log),time=time.time(),status='running'))
        rc=p.wait()
    elapsed=time.monotonic()-start;worker=read(OUT/name/'RECEIPT.json') if (OUT/name/'RECEIPT.json').exists() else None
    seconds=max(0.,elapsed-(worker['seconds'] if worker else 0.));prior=total_prior()
    w=OUT/(name+'_wrapper');write(w/'RECEIPT.json',dict(status='completed' if rc==0 else 'failed',seconds=seconds,
        child_wall_seconds=elapsed,worker_seconds=worker['seconds'] if worker else 0.,prior_seconds=prior,cumulative_seconds=prior+seconds,
        cap=None,includes='actual child imports/hash checks/process teardown outside worker timer; conservative allocation accounting'))
    if rc!=0:
        active(dict(status='failed',name=name,exit_code=rc,log=str(log),time=time.time(),controller_pid=os.getpid()))
        raise RuntimeError(f'{name} failed with exit {rc}; no restart queued')
    return read(OUT/name/'COMPLETE.json')

def main():
    check_pins(read(D/'LOCK.json')['pins']);assert read(OUT/'joint_learnability_probe001/ROOT_READBACK.json')['status']=='independently_audited'
    write(D/'CONTROLLER_STARTED.json',dict(time=time.time(),pid=os.getpid(),code_sha=sha(Path(__file__)),
        scope='only registered two one-epoch mixer seeds then fixed validation; no retries/scientific changes'))
    for seed in read(D/'CONFIG.json')['seeds']:
        for k in range(1,25):
            if (D/f'seed{seed}'/'COMPLETE.json').exists():break
            name=f'joint_learnability_fit{seed}_{k:03}'
            done=execute(name,['scripts/desta3d_v3_joint_learnability.py','fit','--name',name,'--seed',str(seed)])
            assert done['status'] in ('safe_pause','complete')
        else:raise RuntimeError('24 allocation engineering review boundary; progress retained')
    for k in range(1,9):
        if (D/'evaluation/COMPLETE.json').exists():break
        name=f'joint_learnability_eval_{k:03}'
        done=execute(name,['scripts/desta3d_v3_joint_learnability_eval.py','--name',name])
        assert done['status'] in ('safe_pause','complete')
    else:raise RuntimeError('8 inference allocation review boundary')
    active(dict(status='all_training_and_predictions_complete_awaiting_root_score',time=time.time(),controller_pid=os.getpid()))
    write(D/'CONTROLLER_COMPLETE.json',dict(time=time.time(),scope='fixed two seeds and inference; scores/root audit pending'))

if __name__=='__main__':main()
