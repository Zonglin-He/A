"""Finite CPU-only cached diagnosis; does not reopen the R135 model experiment."""
from common import *
import argparse,fcntl,os,subprocess,signal
ap=argparse.ArgumentParser();ap.add_argument('script');a=ap.parse_args();assert rd(H/'STATE.json')['execution_open'];p=(H/a.script).resolve();assert p.parent==H and p.suffix=='.py' and 'FINAL' not in str(p)
lock=(H/'CPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1');start=utc();child=subprocess.Popen([str(ROOT/'crossplatform/env/bin/python'),str(p)],cwd=ROOT,env=env,start_new_session=True);dump(H/'RUNNING.json',{'pid':child.pid,'script':a.script,'started_utc':start})
try:code=child.wait(timeout=3600)
except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait();code=124
finally:
 dump(H/'job_receipts'/(start.replace(':','_')+'_'+a.script+'.json'),{'script':a.script,'started_utc':start,'finished_utc':utc(),'returncode':child.returncode,'resource':'cpu','max_seconds':3600});(H/'RUNNING.json').unlink(missing_ok=True)
raise SystemExit(code)
