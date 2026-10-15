"""Finite P0/P1 chain only: teacher, references, Student, seal/readback, score."""
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, read, write, sha, verify_runtime


def cpu(name, *args):
    command = [str(ROOT/'.conda/tubedetr/bin/python'), '-B', 'scripts/'+name, *args]
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'), check=True)


def gpu(stage):
    unit = 'stvg-ecld-'+stage+'-001'
    command = ['systemd-run', '--user', '--wait', '--collect', '--unit='+unit,
        '--property=MemoryHigh=16G', '--property=MemoryMax=20G', '--property=MemorySwapMax=1G',
        '--working-directory='+str(ROOT), str(ROOT/'.conda/tubedetr/bin/python'),
        '-B', 'scripts/launch_stvg_ecld_v1.py', stage]
    subprocess.run(command, cwd=ROOT, check=True)
    exit_record = read(BASE/'execution'/stage/'WORKER_EXIT.json')
    assert exit_record['exit_code'] == 0
    assert read(BASE/(stage.upper()+'_GLOBAL_BARRIER.json'))['status'] == 'complete'


def main():
    verify_runtime()
    assert read(BASE/'p0/ROOT_SUPPORT_READBACK004.json')['status'] == 'pass'
    assert read(BASE/'ROOT_TEACHER_INPUT_READBACK.json')['status'] == 'pass'
    start = time.time()
    try:
        for stage in ['teacher', 'references', 'student', 'root', 'score']:
            write(BASE/'SEQUENCE_STATUS.json', dict(status='running', stage=stage,
                GT_read=stage=='score', controller_pid=os.getpid(), time=time.time()), mutable=True)
            if stage in ['teacher', 'student']:
                gpu(stage)
            elif stage == 'references':
                cpu('prepare_stvg_ecld_references_v1.py')
            else:
                cpu('measure_stvg_ecld_v1.py', stage)
        write(BASE/'ACTUAL_MEASUREMENT_HANDOFF.json', dict(status='complete_pending_root_review_publication',
            case_count=32, GT_read=True, bootstrap_executed=False, full_started=False,
            teacher_barrier_sha256=sha(BASE/'TEACHER_GLOBAL_BARRIER.json'),
            student_barrier_sha256=sha(BASE/'STUDENT_GLOBAL_BARRIER.json'),
            root_readback_sha256=sha(BASE/'ROOT_STUDENT_READBACK.json'),
            summary_sha256=sha(BASE/'POSTSEAL_SUMMARY.json'), elapsed_seconds=time.time()-start))
        write(BASE/'SEQUENCE_STATUS.json', dict(status='measurement_handoff_pending_root_review',
            GT_read=True, full_started=False, time=time.time()), mutable=True)
    except BaseException as error:
        write(BASE/'SEQUENCE_FAILURE.json', dict(status='preserved_engineering_failure',
            stage=stage, error_type=type(error).__name__, error=str(error),
            GT_read=(BASE/'GT_EXPOSURE.json').exists(), old_full_restarted=False, time=time.time()))
        write(BASE/'SEQUENCE_STATUS.json', dict(status='failed_preserved', stage=stage,
            GT_read=(BASE/'GT_EXPOSURE.json').exists(), time=time.time()), mutable=True)
        raise


if __name__ == '__main__':
    main()
