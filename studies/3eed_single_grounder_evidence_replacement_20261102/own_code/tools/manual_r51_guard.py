"""Resource guard for the newly human-authorized R51 manual experiment.

No overnight authorization, deadline extension, or automation is created.
Individual experiment scripts enforce their own frozen plan/input grants.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];H=ROOT/'research/round51'
def main():
 p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu','network'],required=True);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 state=json.loads((H/'STATE.json').read_text());assert state['execution_open'] and state['new_manual_human_authorization'] and state['automations_remain_off']
 assert len(cmd)>=2 and Path(cmd[0]).resolve() in {(ROOT/'crossplatform/env/bin/python').resolve(),Path('LOCAL_HOME/mllm/.venv/bin/python').resolve()} and 0<a.timeout_seconds<=7200
 script=Path(cmd[1]).resolve();assert script.is_relative_to(H) or script.is_relative_to(ROOT/'crossplatform/round51') or script==ROOT/'tools/verify_round15_remote.py'
 assert not any('NIGHT_' in q or 'round49/' in q or 'FINAL384' in q or 'round46/final' in q for q in cmd)
 if a.resource=='gpu':assert state.get('GPU_stage_authorized') is True
 lockpath=ROOT/'research/GPU_RESOURCE.lock' if a.resource=='gpu' else H/(a.resource.upper()+'.lock');lock=lockpath.open('a+')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Resource already in use; no duplicate launch')
 env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['CUDA_VISIBLE_DEVICES']='0' if a.resource=='gpu' else ''
 start=datetime.datetime.now(datetime.timezone.utc);child=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True);record={'resource':a.resource,'pid':child.pid,'command':cmd,'started_utc':start.isoformat(),'new_manual_human_R51':True,'old_night_not_resumed':True};active=H/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(record,indent=2)+'\n')
 try:code=child.wait(timeout=a.timeout_seconds);record.update(returncode=code,timed_out=False)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=124;record.update(returncode=code,timed_out=True)
 finally:
  record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out=H/'job_receipts';out.mkdir(exist_ok=True);(out/(start.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(record,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
 raise SystemExit(code)
if __name__=='__main__':main()
