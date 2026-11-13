"""Manual R92 bounded latent experiment, exclusive GPU, no old night timers."""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];H=ROOT/'research/round92'
def main():
 p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu','network'],required=True);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 state=json.loads((H/'STATE.json').read_text());assert state['execution_open'] and state['manual_authorization'] and state['automations_remain_off']
 assert len(cmd)>=2 and Path(cmd[0]).resolve()==(ROOT/'crossplatform/env/bin/python').resolve() and 0<a.timeout_seconds<=3600
 f=Path(cmd[1]).resolve();assert f.is_relative_to(H) or f==ROOT/'tools/verify_round15_remote.py'
 assert not any('NIGHT_' in x or 'FINAL384' in x or 'round46/final' in x for x in cmd)
 if a.resource=='gpu':assert state['GPU_stage_authorized']
 lock=(ROOT/'research/GPU_RESOURCE.lock' if a.resource=='gpu' else H/(a.resource.upper()+'.lock')).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if a.resource=='gpu' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 start=datetime.datetime.now(datetime.timezone.utc);child=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True);r={'pid':child.pid,'resource':a.resource,'command':cmd,'started_utc':start.isoformat()};active=H/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(r,indent=2)+'\n')
 try:code=child.wait(timeout=a.timeout_seconds)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=124
 finally:
  r.update(returncode=child.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());folder=H/'job_receipts';folder.mkdir(exist_ok=True);(folder/(start.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(r,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 raise SystemExit(code)
if __name__=='__main__':main()
