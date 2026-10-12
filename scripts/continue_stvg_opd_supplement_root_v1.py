"""One finite CPU continuation for actual postseal baseline and cost root work.

It never starts a GPU worker, chooses a configuration, scores partial arms,
changes a prediction, or marks visual/publication work complete.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_opd_supplement_v1'
DEST = BASE / 'actual_root/postseal_continuation006'
PYTHON = ROOT / '.conda/tubedetr/bin/python'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def status(value):
    temp = DEST / f'STATUS.{os.getpid()}.tmp'
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(DEST / 'STATUS.json')


def verify():
    runtime = read(DEST / 'REVISION_RUNTIME.json')
    for rel, digest in runtime['pins'].items():
        assert sha(ROOT / rel) == digest, rel
    baseline_pin = ROOT / runtime['baseline_runtime_path']
    assert sha(baseline_pin) == runtime['baseline_runtime_sha256']
    return runtime


def run():
    verify()
    with (DEST / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in ['baselines', 'cost']:
            completed = DEST / f'COMPLETE_{job}.json'
            if completed.exists():
                assert read(completed)['status'] == 'actual_CPU_root_complete_pending_visual_publication'
                assert (BASE / job / 'ROOT_FIGURE_COMPLETION.json').exists()
                continue
            dependencies = ([BASE / 'baselines/FORMAL_CPU_COMPLETION.json']
                            if job == 'baselines' else
                            [BASE / 'cost/hc2/COMPLETION.json', BASE / 'cost/vidstg/COMPLETION.json'])
            while not all(path.exists() for path in dependencies):
                for namespace in (['baselines'] if job == 'baselines' else ['baselines', 'cost']):
                    state = read(BASE / namespace / 'STATUS.json')
                    assert 'fail' not in state['status'], f'{namespace}: preserved actual dependency failure'
                status(dict(status='CPU_waiting_for_complete_postseal_dependency', job=job,
                            controller_pid=os.getpid(), CPU_only=True, new_model_calls=0,
                            GT_read=False, experiment_complete=False, time=time.time()))
                time.sleep(15)
            verify()
            if job == 'baselines':
                assert read(dependencies[0])['counts']['arrivals'] == 82710
                assert read(BASE / 'baselines/FORMAL_GLOBAL_BARRIER.json')['logical_arrivals'] == 82710
            else:
                assert all(read(path)['fits'] == 16 for path in dependencies)
            log = DEST / f'{job}.log'
            with log.open('a') as output:
                worker = subprocess.Popen(
                    [str(PYTHON), '-B', 'scripts/review_stvg_opd_supplement_v1.py', job],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                status(dict(status='actual_CPU_root_running', job=job,
                            controller_pid=os.getpid(), worker_pid=worker.pid, CPU_only=True,
                            new_model_calls=0, experiment_complete=False, time=time.time()))
                assert worker.wait() == 0, f'{job}: preserved actual CPU root failure'
            figure = BASE / job / 'ROOT_FIGURE_COMPLETION.json'
            assert read(figure)['status'] == 'pending_actual_PDF_pixel_view'
            immutable(completed, dict(
                status='actual_CPU_root_complete_pending_visual_publication', job=job,
                dependency_sha256={str(path.relative_to(ROOT)): sha(path) for path in dependencies},
                root_figure_completion_sha256=sha(figure), new_model_calls=0,
                CPU_only=True, experiment_complete=False, time=time.time()))
        immutable(DEST / 'COMPLETION.json', dict(
            status='pending_actual_root_PDF_pixel_view_and_publication', finite_CPU_jobs=2,
            new_model_calls=0, CPU_only=True, experiment_complete=False, time=time.time()))
        status(dict(status='pending_actual_root_PDF_pixel_view_and_publication',
                    finite_CPU_jobs=2, CPU_only=True, experiment_complete=False, time=time.time()))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        evidence = DEST / 'failures' / str(time.time_ns())
        evidence.mkdir(parents=True)
        (evidence / 'traceback.txt').write_text(traceback.format_exc())
        status(dict(status='failed_preserved_pending_actual_root',
                    evidence=str(evidence.relative_to(ROOT)), CPU_only=True,
                    experiment_complete=False, time=time.time()))
        raise
