"""One newly authorized R128 experiment; old deadline-completed work remains closed."""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];N=R/'research/round128'
def main():
 p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu'],required=True);p.add_argument('--timeout-seconds',type=int,default=21600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 s=json.loads((N/'STATE.json').read_text());assert s['execution_open'] and s['authorization_scope']=='R128_eight_baselines_six_direction_suite'
 assert len(cmd)>=2 and Path(cmd[0]).resolve()==(R/'crossplatform/env/bin/python').resolve();assert Path(cmd[1]).resolve().is_relative_to(N)
 assert not any('FINAL' in x or 'NIGHT_' in x for x in cmd);assert 0<a.timeout_seconds<=21600
 lock=(R/'research/GPU_RESOURCE.lock' if a.resource=='gpu' else N/'CPU.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if a.resource=='gpu' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 child=subprocess.Popen(cmd,cwd=R,env=env,start_new_session=True);now=datetime.datetime.now(datetime.timezone.utc);receipt={'resource':a.resource,'pid':child.pid,'command':cmd,'started_utc':now.isoformat(),'max_seconds':a.timeout_seconds};active=N/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(receipt,indent=2)+'\n')
 try:code=child.wait(timeout=a.timeout_seconds)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=124
 finally:
  receipt.update(returncode=child.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());d=N/'job_receipts';d.mkdir(exist_ok=True);(d/(now.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 raise SystemExit(code)
if __name__=='__main__':main()
