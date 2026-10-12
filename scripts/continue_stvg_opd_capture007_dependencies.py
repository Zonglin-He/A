"""Finite original CPU/Q0 dependency resumption after preserved O memory repair."""
import importlib, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, sha, status
REC = ROOT / 'artifacts/stvg_opd_optimization_geometry_v1/frozen_capture_CPU007/CPU_dependency_restart003'
MODULES = {
    'postseal': 'scripts.continue_stvg_opd_geometry_postseal_CPU_v1',
    'independent': 'scripts.continue_stvg_opd_common_state_objective002',
    'Q0': 'scripts.continue_stvg_opd_query_q0_native003',
    'Q0_root': 'scripts.continue_stvg_opd_query_q0_native_root003',
}


def run(job):
    runtime = read(REC / 'REVISION_RUNTIME.json')
    for p, h in runtime['pins'].items():
        assert sha(ROOT / p) == h, p
    assert read(ROOT / 'artifacts/stvg_opd_optimization_geometry_v1/CONTROLLER_STATUS.json')['status'] == 'running'
    status(REC / (job + '_STATUS.json'), dict(status='running_original_finite_dependency',
           job=job, pid=os.getpid(), CPU_only=True, GT_read=False, time=time.time()))
    # Each original helper keeps its science pins, barriers, and readback logic.
    importlib.import_module(MODULES[job]).run()
    status(REC / (job + '_STATUS.json'), dict(status='original_finite_dependency_returned',
           job=job, pid=os.getpid(), CPU_only=True, task_complete=False, time=time.time()))


if __name__ == '__main__':
    job = sys.argv[1]
    try:
        run(job)
    except BaseException:
        p = REC / 'failures' / job / str(time.time_ns())
        p.mkdir(parents=True, exist_ok=True)
        (p / 'traceback.txt').write_text(traceback.format_exc())
        status(REC / (job + '_STATUS.json'), dict(status='failed_preserved_pending_actual_root',
               job=job, evidence=str(p.relative_to(ROOT)), CPU_only=True, time=time.time()))
        raise
