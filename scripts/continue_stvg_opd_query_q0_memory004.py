"""One saved-boundary pause for fixed Q0 frozen-memory repair and continuation."""
import inspect
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE, context
context()
import scripts.continue_stvg_opd_query_q0_v1 as original
from scripts.decota_matrix_common_v1 import read, write, status, sha

REC = BASE / 'frozen_memory_revision004'


def run():
    original.verify()
    for path, digest in read(REC / 'REVISION_RUNTIME.json')['pins'].items():
        assert sha(ROOT / path) == digest, path
    assert read(original.O / 'GLOBAL_PREDICTION_BARRIER.json')['status'] == 'sealed'
    pause_source = inspect.getsource(original.pause_priority).replace('BASE/', 'REC/')
    resume_source = inspect.getsource(original.resume_priority).replace('BASE/', 'REC/')
    namespace = dict(original.__dict__, REC=REC)
    exec(compile(pause_source + '\n' + resume_source, __file__, 'exec'), namespace)
    namespace['pause_priority']()

    def child(job, CPU=False):
        environment = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        if CPU:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        else:
            environment.pop('CUDA_VISIBLE_DEVICES', None)
        command = [str(original.PYTHON), '-B', 'scripts/run_stvg_opd_query_q0_memory004.py', job]
        with (REC / (job + '.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, env=environment)
            status(BASE / 'STATUS.json', dict(status='running_correctNative_Q0_memory004_' + job,
                   stage=job, controller_pid=os.getpid(), worker_pid=process.pid,
                   GT_read=False, CPU_only=CPU, time=time.time()))
            assert process.wait() == 0, 'Actual correct-Native memory004 ' + job + ' failed; preserve'

    try:
        child('qualification')
        child('qualification_root', True)
        child('formal')
    finally:
        namespace['resume_priority']()
    assert read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')['status'] == 'sealed'
    # These original roots keep the correct Native support and original GT gate.
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    with (REC / 'deployment_root.log').open('a') as log:
        process = subprocess.Popen(
            [str(original.PYTHON), '-B', 'scripts/run_stvg_opd_query_q0_native003.py', 'deployment_root'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
        status(BASE / 'STATUS.json', dict(status='running_correctNative_Q0_deployment_root',
               worker_pid=process.pid, CPU_only=True, GT_read=False, time=time.time()))
        assert process.wait() == 0
    with (REC / 'score.log').open('a') as log:
        process = subprocess.Popen(
            [str(original.PYTHON), '-B', '-c',
             'from scripts.stvg_opd_query_q0_native_scope003 import context; context(); '
             'from scripts.score_stvg_opd_query_q0_v1 import run; run()'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
        status(BASE / 'STATUS.json', dict(status='running_correctNative_Q0_postseal_score',
               worker_pid=process.pid, CPU_only=True, GT_read=True, time=time.time()))
        assert process.wait() == 0
    status(BASE / 'STATUS.json', dict(
        status='pending_actual_independent_statistics_PDF_view_publication_archive',
        CPU_only=True, GT_read=True, task_complete=False, time=time.time()))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        path = REC / 'failures' / str(time.time_ns())
        path.mkdir(parents=True, exist_ok=True)
        (path / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'STATUS.json', dict(status='failed_preserved_pending_actual_root',
               revision='memory004', evidence=str(path.relative_to(ROOT)), time=time.time()))
        raise
