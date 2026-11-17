"""Bounded R148 jobs; exclusive GPU and hard overnight deadline."""
import argparse,datetime,fcntl,json,os,signal,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'research/round148'
def main():
 p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu'],required=True);p.add_argument('--timeout-seconds',type=int,default=7200);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 state=json.loads((H/'STATE.json').read_text());night=json.loads((R/'research/NIGHT_20261117/STATE.json').read_text());now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime.fromisoformat(night['deadline_utc']);assert state['execution_open'] and night['status']=='active' and now<deadline
 assert 0<a.timeout_seconds<=10800 and len(cmd)>=2 and Path(cmd[0]).resolve()==(R/'crossplatform/env/bin/python').resolve() and Path(cmd[1]).resolve().is_relative_to(H)
 assert not any('FINAL' in x for x in cmd);limit=min(a.timeout_seconds,max(1,int((deadline-now).total_seconds())))
 lock=(R/'research/GPU_RESOURCE.lock' if a.resource=='gpu' else H/'CPU.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if a.resource=='gpu' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 child=subprocess.Popen(cmd,cwd=R,env=env,start_new_session=True);record={'resource':a.resource,'command':cmd,'pid':child.pid,'started_utc':now.isoformat(),'deadline_utc':deadline.isoformat(),'max_seconds':limit};active=H/('RUNNING_'+a.resource.upper()+'.json');active.write_text(json.dumps(record,indent=2)+'\n');code=None
 try:code=child.wait(timeout=limit)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=124
 finally:
  record.update(returncode=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());d=H/'job_receipts';d.mkdir(exist_ok=True);(d/(now.strftime('%Y%m%dT%H%M%S')+'_'+str(child.pid)+'.json')).write_text(json.dumps(record,indent=2)+'\n');active.unlink(missing_ok=True);fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 raise SystemExit(code)
if __name__=='__main__':main()
