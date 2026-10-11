"""One finite serial controller; all deployment predictions precede new GT."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import *


def run():
    verify()
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def child(job, name=None, CPU=False):
            script = 'scripts/run_stvg_opd_design_ablation_v1.py'
            if job == 'score': script = 'scripts/score_stvg_opd_design_ablation_v1.py'
            if job == 'finalize': script = 'scripts/finalize_stvg_opd_design_ablation_v1.py'
            cmd = [str(PYTHON), '-B', script]
            if job not in ['score', 'finalize']: cmd += [job] + ([name] if name else [])
            log = BASE / 'logs' / (job + ('_' + name if name else '') + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': ALLOCATOR, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            if CPU: env['CUDA_VISIBLE_DEVICES'] = ''
            with log.open('a') as stream:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
                status(BASE / 'STATUS.json', dict(status='running', job=job, stage=name, CPU_only=CPU,
                    controller_pid=os.getpid(), worker_pid=p.pid, command=cmd, log=str(log.relative_to(ROOT)),
                    GT_read=False if job not in ['score', 'finalize'] else 'post_global_seal', time=time.time()))
                code = p.wait()
                assert code == 0, f'Actual {job} failure preserved, exit={code}'
        if not (BASE / 'ROOT_QUALIFICATION_READBACK.json').exists():
            for name in stages():
                if not (BASE / 'qualification' / name / 'PREDICTION_BARRIER.json').exists(): child('qualification', name)
            child('root_qualification', CPU=True)
        assert read(BASE / 'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
        if not (BASE / 'GLOBAL_PREDICTION_BARRIER.json').exists():
            barriers = {}; total = 0
            for name, stage in stages().items():
                p = BASE / 'stages' / name / 'PREDICTION_BARRIER.json'
                if not p.exists(): child('GPU', name)
                b = read(p); assert b['status'] == 'sealed' and not b['GT_read']
                assert b['new_fit_arrivals'] == stage['expected_new_fits']
                total += b['new_fit_arrivals']; barriers[str(p.relative_to(ROOT))] = sha(p)
            assert total == 5376
            write(BASE / 'GLOBAL_PREDICTION_BARRIER.json', dict(status='sealed', all_three_arms_conditions_orders_and_both_targets=True,
                new_formal_fits=total, barriers=barriers, GT_read=False, runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))
        if not (BASE / 'CPU_SCORING_COMPLETION.json').exists(): child('score', CPU=True)
        if not (BASE / 'CPU_COMPLETION.json').exists(): child('finalize', CPU=True)
        status(BASE / 'STATUS.json', dict(status='CPU_complete_pending_actual_root_visual_and_publication',
            controller_pid=os.getpid(), new_formal_fits=5376, experiment_complete=False,
            original_P1_P6_suite_unchanged=True, time=time.time()))


if __name__ == '__main__':
    try: run()
    except BaseException:
        d = BASE / 'controller_failures' / str(time.time_ns()); d.mkdir(parents=True, exist_ok=True)
        (d / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'STATUS.json', dict(status='controller_failure_preserved_pending_actual_root',
            evidence=str(d.relative_to(ROOT)), experiment_complete=False, time=time.time()))
        raise
