"""Finite CPU-only source-bound root continuation after preserved assertion."""
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, status, sha
from scripts.review_stvg_opd_common_state_postseal_v1 import REC, require_seal

REV = REC / 'source_origin_stream003'


def run():
    for path, digest in read(REV / 'REVISION_RUNTIME.json')['pins'].items():
        assert sha(ROOT / path) == digest, path
    require_seal()
    with (REV / 'CPU.log').open('a') as log:
        process = subprocess.Popen(
            [str(ROOT / '.conda/tubedetr/bin/python'), '-B',
             'scripts/review_stvg_opd_common_state_source003.py'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
        status(REC / 'STATUS.json', dict(status='running_actual_source_bound_root003',
               worker_pid=process.pid, CPU_only=True, GT_read=True,
               task_complete=False, time=time.time()))
        assert process.wait() == 0, 'Actual source-bound root failed; preserve'
    assert read(REC / 'ROOT_READBACK.json')['status'] == 'pass'
    status(REC / 'STATUS.json', dict(
        status='pending_actual_two_PDF_export_pixel_view_publication_archive',
        CPU_only=True, GT_read=True, task_complete=False, time=time.time()))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        path = REV / 'failures' / str(time.time_ns())
        path.mkdir(parents=True, exist_ok=True)
        (path / 'traceback.txt').write_text(traceback.format_exc())
        status(REC / 'STATUS.json', dict(status='failed_preserved_pending_actual_root',
               evidence=str(path.relative_to(ROOT)), CPU_only=True,
               time=time.time()))
        raise
