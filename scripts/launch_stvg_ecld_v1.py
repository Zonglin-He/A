"""Own-unit resource guard; never restarts the paused old full experiment."""
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, read, write, sha, verify_runtime
from scripts import launch_stvg_instance_identity_v1 as resource


def main():
    stage = sys.argv[1]
    assert stage in ['teacher', 'student']
    verify_runtime()
    unit = 'stvg-ecld-' + stage + '-001.service'
    out = BASE / 'execution' / stage
    out.mkdir(parents=True, exist_ok=False)
    resource.BASE, resource.UNIT, resource.STAGE = out, unit, stage
    GiB = 1024**3
    sample = resource.sample()
    assert sample['available'] >= 8*GiB and sample['gpu_free'] >= 26*GiB and sample['disk_free'] >= 30*GiB
    props = dict(x.split('=', 1) for x in subprocess.check_output(['systemctl', '--user', 'show', unit,
        '--property=MemoryHigh,MemoryMax,MemorySwapMax'], text=True).splitlines() if '=' in x)
    assert [int(props[k]) for k in ['MemoryHigh', 'MemoryMax', 'MemorySwapMax']] == [16*GiB, 20*GiB, GiB]
    write(out/'ACTUAL_LIMITS.json', dict(status='pass', admission=sample, properties=props, GT_read=False))
    stop = threading.Event()
    threading.Thread(target=resource.monitor, args=(stop,), daemon=True).start()
    interpreter = ROOT / ('.venv/stvg_native_distill_v1/bin/python' if stage=='teacher'
                          else '.conda/tubedetr/bin/python')
    script = ROOT / ('scripts/run_stvg_ecld_teacher_v1.py' if stage=='teacher'
                     else 'scripts/run_stvg_ecld_student_v1.py')
    with (out/'worker.log').open('xb') as log:
        command = [str(interpreter), '-B', str(script)] + (['--run'] if stage=='teacher' else [])
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ,
            PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
            stdout=log, stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json', dict(status='running', unit=unit, controller_pid=os.getpid(),
              worker_pid=process.pid, command=command, GT_read=False, time=time.time()))
        code = process.wait()
    stop.set()
    write(out/'WORKER_EXIT.json', dict(status='exited', exit_code=code, GT_read=False, time=time.time()))
    if code:
        write(BASE/'STATUS.json', dict(status='engineering_failure', stage=stage,
              exit_code=code, GT_read=False, time=time.time()), mutable=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
