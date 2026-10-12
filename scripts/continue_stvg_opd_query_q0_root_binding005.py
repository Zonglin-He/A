"""Finite all64 postseal root continuation with the receipt-binding repair."""
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE
from scripts.decota_matrix_common_v1 import read, write, status, sha

REC = BASE / 'postseal_actual_root001/receipt_binding005'


def run():
    runtime = read(REC / 'REVISION_RUNTIME.json')
    for path, digest in runtime['pins'].items():
        assert sha(ROOT / path) == digest, path
    assert read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')['cells'] == 64
    assert read(BASE / 'CPU_COMPLETION.json')['cells'] == 64
    command = [str(ROOT / '.conda/tubedetr/bin/python'), '-B',
               str(ROOT / 'scripts/review_stvg_opd_query_q0_root_binding005.py')]
    with (REC / 'CPU.log').open('ab', buffering=0) as log:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT,
                                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
        status(REC / 'STATUS.json', dict(status='running_actual_all64_independent_CPU_root',
               worker_pid=worker.pid, GT_read=True, CPU_only=True, time=time.time()))
        assert worker.wait() == 0, 'Preserve actual Q0 root failure'
    assert read(BASE / 'ROOT_STATISTICS_READBACK.json')['status'] == 'pass'
    result = dict(status='pending_actual_two_PDF_export_pixel_view_publication_archive',
                  GT_read=True, CPU_only=True, task_complete=False, time=time.time())
    status(REC / 'STATUS.json', result)
    status(BASE / 'postseal_actual_root001/STATUS.json', result)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        path = REC / 'failures' / str(time.time_ns())
        path.mkdir(parents=True, exist_ok=False)
        (path / 'traceback.txt').write_text(traceback.format_exc())
        status(REC / 'STATUS.json', dict(status='failed_preserved_pending_actual_root',
               evidence=str(path), GT_read=True, CPU_only=True, time=time.time()))
        raise
