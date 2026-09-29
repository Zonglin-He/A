"""Serial measured subprocess wrapper; failures always return to the root agent."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta_native_common import D, OUT, read, write, total_prior


def launch(kind, name):
    interpreter = ROOT / ('.venv-exost/bin/python' if kind=='temporal' else '.venv-ptd-audit/bin/python')
    if kind in ('temporal','spatial'):
        cmd = [str(interpreter), '-B', 'scripts/desta_native_experts.py', kind, '--run', name]
    else:
        cmd = [str(interpreter), '-B', 'scripts/desta_native_run_v2.py', kind, '--run', name]
    cmd = ['bash', 'scripts/with_local_cuda.sh', *cmd]
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'OMP_NUM_THREADS':'4',
           'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'HF_HUB_OFFLINE':'1'}
    start = time.monotonic()
    with (D/(name+'.log')).open('xb') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (D/'ACTIVE.json').write_text(json.dumps(dict(kind=kind, name=name, pid=child.pid,
            wrapper_pid=os.getpid(), time=time.time(), cmd=cmd, status='running'), indent=2)+'\n')
        code = child.wait()
    elapsed = time.monotonic()-start
    run = OUT/('desta_native_'+name)
    rec = read(run/'RECEIPT.json') if (run/'RECEIPT.json').exists() else {'seconds':0.}
    overhead = max(0., elapsed-rec['seconds']); prior = total_prior()
    write(OUT/('desta_native_'+name+'_wrapper')/'RECEIPT.json', dict(seconds=overhead,
        worker_seconds=rec['seconds'], child_wall_seconds=elapsed, code=code,
        prior_seconds=prior, cumulative_seconds=prior+overhead, cap=None,
        status='completed' if code==0 else 'failed'))
    write(run/'EXIT.json', {'code':code,'time':time.time()})
    (D/'ACTIVE.json').write_text(json.dumps(dict(kind=kind,name=name,status='completed' if code==0 else 'failed',code=code))+'\n')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('kind'); p.add_argument('--run', required=True)
    a=p.parse_args(); sys.exit(launch(a.kind,a.run))
