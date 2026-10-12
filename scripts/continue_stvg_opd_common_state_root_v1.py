"""One finite CPU root handoff after the existing postseal scorer completes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, subprocess, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha
from scripts.review_stvg_opd_common_state_postseal_v1 import BASE, REC, contracts, require_seal


def prepare():
    REC.mkdir(parents=True, exist_ok=True)
    files = ['scripts/review_stvg_opd_common_state_postseal_v1.py',
             'scripts/report_stvg_opd_common_state_postseal_v1.py',
             'scripts/continue_stvg_opd_common_state_root_v1.py',
             'scripts/score_stvg_opd_common_state_geometry_v1.py',
             'scripts/stvg_opd_legacy_input_bridge003.py',
             'vg_tta/stvg_opd_common_state_geometry_arithmetic004.py',
             'artifacts/stvg_opd_optimization_geometry_v1/COMMON_STATE_COHORT.json',
             'artifacts/stvg_opd_optimization_geometry_v1/GPU_RUNTIME.json',
             'artifacts/stvg_opd_optimization_geometry_v1/ROOT_QUALIFICATION_READBACK.json']
    write(REC/'CPU_CONTRACTS.json', contracts())
    write(REC/'RUNTIME.json', dict(status='prepared_before_actual_global', scope='separate CPU independent root continuation; original running pins unchanged',
          pins={p:sha(ROOT/p) for p in files}, CPU_only=True, GT_read=False, task_complete=False, time=time.time()))


def run():
    runtime = read(REC/'RUNTIME.json')
    for p, h in runtime['pins'].items():
        assert sha(ROOT/p) == h, p
    while not (BASE/'COMMON_STATE_CPU_COMPLETION.json').exists():
        state = read(BASE/'CONTROLLER_STATUS.json')
        dependent = read(BASE/'postseal_CPU_continuation001/STATUS.json')
        assert 'failed' not in state['status'] and 'failed' not in dependent['status'], 'Actual dependency failure needs root handling'
        status(REC/'STATUS.json', dict(status='waiting_for_actual_global_and_original_CPU_completion', CPU_only=True,
              GT_read=False, pid=os.getpid(), task_complete=False, time=time.time()))
        time.sleep(15)
    require_seal()
    if not (REC/'ROOT_READBACK.json').exists():
        with (REC/'CPU.log').open('a') as log:
            p = subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'), '-B', 'scripts/review_stvg_opd_common_state_postseal_v1.py'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'})
            status(REC/'STATUS.json', dict(status='running_actual_independent_root', worker_pid=p.pid, CPU_only=True, GT_read=True, task_complete=False, time=time.time()))
            assert p.wait() == 0, 'Independent root failed; evidence preserved'
    assert read(REC/'ROOT_READBACK.json')['status'] == 'pass'
    status(REC/'STATUS.json', dict(status='pending_actual_two_PDF_export_pixel_view_publication_archive',
          CPU_only=True, GT_read=True, task_complete=False, time=time.time()))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'prepare':
        prepare()
    else:
        try:
            run()
        except BaseException:
            p = REC/'failures'/str(time.time_ns()); p.mkdir(parents=True)
            (p/'traceback.txt').write_text(traceback.format_exc())
            status(REC/'STATUS.json', dict(status='failed_preserved_pending_actual_root', evidence=str(p.relative_to(ROOT)), CPU_only=True, time=time.time()))
            raise
