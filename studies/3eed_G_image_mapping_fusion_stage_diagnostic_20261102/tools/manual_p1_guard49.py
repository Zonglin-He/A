"""Record only the newly human-authorized manual P1 CPU/network work.

The expired overnight guard and its deadline remain unchanged. No GPU job,
training, final-cohort access or new TTA is authorized by this wrapper.
Actual input allowlists and stage grants are checked by the locked scripts.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'research/round49'
def main():
    p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','network'],required=True);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    state=json.loads((TASK/'STATE.json').read_text());scope=json.loads((TASK/'experiment/HUMAN_SCOPE_AUTHORIZATION.json').read_text())
    assert state.get('execution_open') is True and scope['execution_mode']=='new explicit manual task, separate from ended overnight automation'
    assert scope['night_deadline_not_extended'] and scope['no_FINAL_access'] and scope['no_parameter_or_box_update']
    assert len(cmd)>=2 and 0<a.timeout_seconds<=3600
    assert Path(cmd[0]).resolve()==(ROOT/'crossplatform/env/bin/python').resolve()
    script=Path(cmd[1]).resolve();allowed=[ROOT/'research/round49',ROOT/'crossplatform/round49']
    assert any(script.is_relative_to(q) for q in allowed) or script==ROOT/'tools/verify_round15_remote.py'
    assert not any('FINAL384' in x or 'round46/final384' in x for x in cmd)
    lock=(TASK/(a.resource.upper()+'.lock')).open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This manual P1 resource is already owned')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    start=datetime.datetime.now(datetime.timezone.utc);child=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True)
    rec={'resource':a.resource,'pid':child.pid,'command':cmd,'started_utc':start.isoformat(),'manual_human_P1':True,'old_night_deadline_not_extended':True,'GPU_disabled':True};active=TASK/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(rec,indent=2)+'\n')
    try:code=child.wait(timeout=a.timeout_seconds);rec.update(returncode=code,timed_out=False)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        code=124;rec.update(returncode=code,timed_out=True)
    finally:
        rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out=TASK/'job_receipts';out.mkdir(exist_ok=True);(out/(start.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(rec,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    raise SystemExit(code)
if __name__=='__main__':main()
