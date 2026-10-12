"""Finite original32 actual audited paths; no scoring or cancelled S resumption."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status
from scripts.run_stvg_opd_cost_after_N_priority001 import BASE, SCIENCE, verify


def run():
    verify()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for ds in ['hc2', 'vidstg']:
            complete = SCIENCE/ds/'COMPLETION.json'
            if complete.exists():
                assert read(complete)['fits'] == 16
                continue
            env = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            env.pop('CUDA_VISIBLE_DEVICES', None)
            log = BASE/'logs'/f'{ds}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as f:
                p = subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'), '-B', 'scripts/run_stvg_opd_cost_after_N_priority001.py', ds], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
                v = dict(status='running_original_audited_timing', dataset=ds, controller_pid=os.getpid(), worker_pid=p.pid,
                         total=16, CPU_only=False, GT_read=False, original_timing_source_unchanged=True, time=time.time())
                status(BASE/'STATUS.json', v)
                status(SCIENCE/'STATUS.json', dict(v, latest_serial_revision='after_N_priority001'))
                assert p.wait() == 0, 'Preserved original cost worker failure'
            assert read(complete)['fits'] == 16
        v = dict(status='pending_actual_original32_audited_cost_independent_root_PDF_view_public_archive',
                 query_fits=32, CPU_only=True, GT_read=False, experiment_complete=False, time=time.time())
        write(BASE/'CPU_COMPLETION.json', v)
        status(BASE/'STATUS.json', v)
        status(SCIENCE/'STATUS.json', dict(v, latest_serial_revision='after_N_priority001'))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        p = BASE/'controller_failures'/str(time.time_ns())
        p.mkdir(parents=True, exist_ok=False)
        (p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json', dict(status='failed_preserved_pending_actual_root', evidence=str(p.relative_to(ROOT)), CPU_only=True, GT_read=False, time=time.time()))
        raise
