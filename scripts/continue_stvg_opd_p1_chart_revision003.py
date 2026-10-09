"""Finite missing-only VidSTG continuation; retains original global GT barrier."""
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, PYTHON, read, write, status, sha
from scripts.run_stvg_opd_p1_chart_revision003 import REC, verify_revision


def run():
    verify_revision()
    assert read(REC / 'GPU_QUALIFICATION.json')['status'] == 'pass'
    assert read(BASE / 'P0_ROOT_CLOSING_RECEIPT.json')['status'] == 'complete'
    import scripts.continue_stvg_opd_paper_hc2_revision_v2 as original
    def child(job, *args, cpu=False):
        verify_revision()
        if job == 'GPU' and args == ('P1_hc2',):
            b = BASE / 'stages/P1_hc2/PREDICTION_BARRIER.json'
            assert sha(b) == read(REC / 'REVISION_RUNTIME.json')['HC2_sealed_barrier_sha256']
            assert read(b)['adapted_arrivals'] == 10446
            return
        cmd = [str(PYTHON), '-B', 'scripts/run_stvg_opd_p1_chart_revision003.py', job, *args]
        log = REC / ('continuation_' + '_'.join([job, *args]) + '.log')
        with log.open('a') as out:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''} if cpu else None)
            status(BASE / 'STAGE.json', dict(status='running_authorized_chart_revision003',
                controller_pid=os.getpid(), worker_pid=p.pid, job=job, args=list(args),
                command=cmd, log=str(log.relative_to(ROOT)), CPU=cpu,
                engineering_revision_sha256=sha(REC / 'REVISION_RUNTIME.json'),
                human_authorization_sha256=sha(REC / 'AUTHORIZATION_RECEIPT.json'), time=time.time()))
            assert p.wait() == 0, 'Preserve actual new failure; do not bypass scientific or GT barriers'
    original.child = child
    original.run('P1')
    write(REC / 'COMPLETION.json', dict(status='fixed_P1_CPU_complete_pending_actual_root',
        original_phase_and_GT_barriers_retained=True, HC2_GPU_not_rerun=True,
        paper_suite_complete=False, time=time.time()))


if __name__ == '__main__':
    try: run()
    except BaseException:
        dest = REC / 'controller_failures' / str(time.time_ns())
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'STATUS.json', dict(status='new_chart_revision003_failure_preserved',
            evidence=str(dest.relative_to(ROOT)), paper_suite_complete=False, GT_read=False, time=time.time()))
        raise
