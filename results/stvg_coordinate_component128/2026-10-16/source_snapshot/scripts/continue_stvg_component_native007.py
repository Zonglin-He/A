"""Finite guarded Teacher -> native reference -> Student -> root -> score."""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_component_native007_common import BASE, COMPONENT, read, write, sha, verify, build_reference_locks
from scripts import launch_stvg_instance_identity_v1 as resource

UNIT = 'stvg-component128-native-run007.service'
EXEC = BASE / 'execution007'


def child(stage, script, *args):
    resource.STAGE = stage
    interpreter = ROOT / ('.venv/stvg_native_distill_v1/bin/python' if stage == 'teacher' else '.conda/tubedetr/bin/python')
    with (EXEC / (stage + '_worker.log')).open('xb') as log:
        p = subprocess.Popen([str(interpreter), '-B', 'scripts/' + script, *args], cwd=ROOT,
              env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
              stdout=log, stderr=subprocess.STDOUT)
        write(EXEC/(stage.upper()+'_LAUNCH.json'), dict(status='running', worker_pid=p.pid,
              controller_pid=os.getpid(), stage=stage, GT_read=False, time=time.time()))
        code = p.wait()
    write(EXEC/(stage.upper()+'_WORKER_EXIT.json'), dict(status='exited', stage=stage, exit_code=code,
          GT_read=False, time=time.time()))
    assert code == 0, (stage, code)


def main():
    verify()
    EXEC.mkdir(parents=True, exist_ok=False)
    resource.BASE, resource.UNIT = EXEC, UNIT
    g = 1024**3
    v = resource.sample()
    assert v['available'] >= 8*g and v['gpu_free'] >= 26*g and v['disk_free'] >= 30*g
    props = dict(x.split('=', 1) for x in subprocess.check_output(
        ['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'], text=True).splitlines() if '=' in x)
    assert [int(props[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']] == [16*g,20*g,g]
    write(EXEC/'ACTUAL_LIMITS.json', dict(status='pass', admission=v, properties=props, GT_read=False, time=time.time()))
    stop = threading.Event()
    threading.Thread(target=resource.monitor, args=(stop,), daemon=True).start()
    try:
        write(COMPONENT/'STATUS.json', dict(status='running', stage='native_teacher', done=0, total=512, GT_read=False, time=time.time()), mutable=True)
        child('teacher', 'run_stvg_component_native_teacher007.py')
        # The Teacher child has exited; its CUDA context is gone before Student.
        build_reference_locks()
        write(COMPONENT/'STATUS.json', dict(status='running', stage='native_student', done=0, total=128, GT_read=False, time=time.time()), mutable=True)
        child('student', 'run_stvg_component_native_student007.py')
        child('readback', 'review_stvg_component_native007.py', 'N000')
        child('score', 'measure_stvg_component_native007.py', '--trial', 'N000', '--score')
        write(BASE/'CPU_COMPLETION007.json', dict(status='complete_pending_study_root_view_public_archive', count=128, GT_read=True, time=time.time()))
        write(COMPONENT/'STATUS.json', dict(status='complete_pending_independent_root_view_public_archive', stage='root_handoff', done=128, total=128, GT_read=True, time=time.time()), mutable=True)
    finally:
        stop.set()


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        write(BASE/'FAILURE007.json', dict(status='failed', stage='required_root_review', error_type=type(e).__name__,
              error=str(e), traceback=traceback.format_exc(), GT_read=False, time=time.time()), mutable=True)
        write(COMPONENT/'STATUS.json', dict(status='failed', stage='required_root_review', GT_read=False, time=time.time()), mutable=True)
        raise
