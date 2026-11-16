"""Bound this authorized research session by exclusive resources and hard deadline."""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];N=R/'research/EXPLORATION_20261115'
def main():
 p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu','network'],required=True);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command;s=json.loads((N/'STATE.json').read_text());now=datetime.datetime.now(datetime.timezone.utc);end=datetime.datetime.fromisoformat(s['deadline_utc']);remaining=(end-now).total_seconds();assert s['execution_open'] and remaining>0,'Research deadline passed; no new experiment'
 assert len(cmd)>=2 and Path(cmd[0]).resolve()==(R/'crossplatform/env/bin/python').resolve();f=Path(cmd[1]).resolve();assert f.is_relative_to(R/'research/round102') or f.is_relative_to(N) or f==R/'tools/verify_round15_remote.py'
 assert not any('FINAL' in x or 'NIGHT_20261101' in x for x in cmd);assert 0<a.timeout_seconds<=43200
 lock=(R/'research/GPU_RESOURCE.lock' if a.resource=='gpu' else N/(a.resource.upper()+'.lock')).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if a.resource=='gpu' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');child=subprocess.Popen(cmd,cwd=R,env=env,start_new_session=True);receipt={'resource':a.resource,'pid':child.pid,'command':cmd,'started_utc':now.isoformat(),'deadline_utc':s['deadline_utc']};active=N/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(receipt,indent=2)+'\n')
 try:code=child.wait(timeout=min(a.timeout_seconds,remaining))
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=124
 finally:
  receipt.update(returncode=child.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());d=N/'job_receipts';d.mkdir(exist_ok=True);(d/(now.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 raise SystemExit(code)
if __name__=='__main__':main()
