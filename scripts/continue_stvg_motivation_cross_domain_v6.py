"""Finite serial native diagnostic; CPU-only waiter until the active P1 GPU seals."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_motivation_cross_domain_common_v6 import *


def wait_for_priority():
    while True:
        verify()
        release = priority_release()
        if release is not None:
            write(BASE/'P1_PRIORITY_RELEASE.json', {**release, 'time': time.time()})
            return
        s = read(P1/'STATUS.json')
        if 'failure' in s.get('status', '') or s.get('status', '').startswith('failed'):
            raise RuntimeError('Active P1 reports failure; preserve queue and hand back for root diagnosis')
        status(BASE/'STATUS.json', dict(status='waiting_for_current_P1_GPU_release',
            controller_pid=os.getpid(), formal_predictions=0, qualification_cells_passed=0,
            GPU_started=False, GT_read=False, cells_total=512, whole_figure_complete=False,
            P1_status=s.get('status'), time=time.time()))
        time.sleep(60)


def acquire_gpu():
    while True:
        verify()
        p = ROOT/'artifacts/spatial_tta_research_v2/gpu.lock'
        f = p.open('a')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            f.close()
            status(BASE/'GPU_WAIT.json', dict(status='waiting_for_exclusive_GPU_lease',
                controller_pid=os.getpid(), no_GPU_started_in_this_wait=True, time=time.time()))
            time.sleep(60)


def child(phase, direction=None, model=None):
    verify()
    cpu = phase == 'score'
    command = [str(PYTHON), '-B', 'scripts/score_stvg_motivation_cross_domain_v6.py'] if cpu else [
        str(PYTHON), '-B', 'scripts/run_stvg_motivation_cross_domain_v6.py', phase, direction, model]
    log = BASE/'logs'/('_'.join(x for x in [phase, direction, model] if x)+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu = None if cpu else acquire_gpu()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if cpu else '0',
        'OMP_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    if gpu is not None:
        env['STVG_FIGURE_GPU_LEASE_FD'] = str(gpu.fileno())
    try:
        with log.open('a') as output:
            worker = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                env=env, pass_fds=() if gpu is None else (gpu.fileno(),))
            status(BASE/'STAGE.json', dict(status='running', controller_pid=os.getpid(), worker_pid=worker.pid,
                phase=phase, direction=direction, model=model, command=command,
                GPU=not cpu, GT_read=cpu, log=str(log.relative_to(ROOT)), time=time.time()))
            assert worker.wait() == 0, 'Preserve failed cell; no auto-skipping or retry with another protocol'
    finally:
        if gpu is not None:
            gpu.close()


def run():
    verify()
    control = (BASE/'controller.lock').open('a')
    fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        wait_for_priority()
        files = {}
        for direction in DIRECTIONS:
            for model in MODELS:
                child('qualification', direction, model)
                rel = f'{direction}/{model}/QUALIFICATION.json'
                q = read(BASE/rel)
                assert q['status'] == 'pass' and q['actual_GPU_qualification_cells'] == 2 and not q['GT_read']
                files[rel] = sha(BASE/rel)
        write(BASE/'QUALIFICATION_BARRIER.json', dict(status='pass', qualification_cells=8,
            accepted_formal_predictions=0, files=files, GT_read=False, time=time.time()))
        for direction in DIRECTIONS:
            for model in MODELS:
                child('formal', direction, model)
        global_seal()
        child('score')
        write(BASE/'CONTROLLER_COMPLETION.json', dict(status='finite_CPU_handoff_pending_actual_root',
            all_512_predictions_sealed_and_scored=True, whole_figure_complete=False,
            whole_paper_complete=False, time=time.time()))
    except BaseException:
        folder = BASE/'failures'/str(time.time_ns())
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json', dict(status='failed_preserved_pending_root',
            evidence=str(folder.relative_to(ROOT)), whole_figure_complete=False,
            no_protocol_bypass=True, time=time.time()))
        raise
    finally:
        control.close()


if __name__ == '__main__':
    run()
