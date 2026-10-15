"""One frozen, finite fresh32 execution; never selects or tunes on its results."""
import sys, subprocess, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_evidence_common_v1 import BASE, read, write, sha
PY = ROOT / '.conda/tubedetr/bin/python'
B = BASE / 'confirmation'
RT = B / 'RUNTIME006.json'

def cpu(script, *args, opencv=False):
    entry = 'scripts/stvg_evidence_opencv004.py' if opencv else 'scripts/stvg_evidence_stage003.py'
    subprocess.run([str(PY), '-B', str(ROOT / entry), str(RT.relative_to(ROOT)), script, *args], cwd=ROOT, check=True)

def gpu(label, script, *args):
    subprocess.run(['systemd-run', '--user', '--wait', '--collect', '--unit=stvg-evidence-' + label + '-001',
                    '--property=MemoryHigh=16G', '--property=MemoryMax=20G', '--property=MemorySwapMax=1G',
                    '--working-directory=' + str(ROOT), str(PY), '-B',
                    str(ROOT / 'scripts/launch_stvg_evidence006.py'), 'confirmation', label, script, *args], check=True)

def main():
    rt = read(RT)
    assert rt['status'] == 'locked' and not rt['GT_read']
    for p, h in {**rt['pins'], **rt.get('protected', {})}.items():
        assert sha(ROOT / p) == h, p
    frozen_sha = sha(BASE / 'FROZEN_METHOD_LOCK.json')
    lock = read(BASE / 'FROZEN_METHOD_LOCK.json')
    assert lock['search_finished'] and not lock['confirmation_used_for_selection']
    assert read(B / 'SOURCE_CAPTURE_BARRIER.json')['count'] == 32
    stages = ['temporal_teacher', 'temporal_fit', 'inputs', 'input_root', 'spatial_teacher',
              'references', 'student', 'temporal_root', 'spatial_root', 'postseal_metrics']
    for stage in stages:
        assert sha(BASE / 'FROZEN_METHOD_LOCK.json') == frozen_sha
        write(B / 'CONTROLLER_STATUS012.json', dict(status='running', stage=stage,
              GT_read=stage == 'postseal_metrics', time=time.time()), mutable=True)
        if stage == 'temporal_teacher':
            gpu('confirmation012-temporal_teacher', 'scripts/run_stvg_evidence_confirmation_temporal012.py')
        elif stage == 'temporal_fit':
            cpu('scripts/fit_stvg_evidence_confirmation012.py')
        elif stage == 'inputs':
            cpu('scripts/prepare_stvg_evidence_confirmation012.py', 'confirmation', opencv=True)
        elif stage == 'input_root':
            output = B / 'ROOT_INPUT_REVIEW012.json'
            cpu('scripts/review_stvg_evidence_inputs012.py', '--phase', 'confirmation', '--output', str(output))
            r = read(output)
            assert r['status'] == 'pass' and r['scope']['GT_read'] is False
            assert r['scope']['model_called'] is False and r['scope']['GPU_used'] is False
            assert r['case_count'] == 32
            write(B / 'ROOT_INPUT_READBACK.json', dict(r, CPU_only=True, GT_read=False, case_count=32,
                  independent_reader_receipt_sha256=sha(output)))
        elif stage == 'spatial_teacher':
            gpu('confirmation012-teacher', 'scripts/run_stvg_evidence_teacher_v1.py', 'confirmation')
        elif stage == 'references':
            cpu('scripts/stvg_evidence_spatial_phase009.py', 'references', 'confirmation')
        elif stage == 'student':
            gpu('confirmation012-student', 'scripts/run_stvg_evidence_student006.py', 'confirmation')
        elif stage == 'temporal_root':
            cpu('scripts/review_stvg_evidence_temporal002.py', 'confirmation')
        elif stage == 'spatial_root':
            cpu('scripts/stvg_evidence_spatial_phase009.py', 'root', 'confirmation')
        elif stage == 'postseal_metrics':
            assert read(B / 'GLOBAL_PREDICTION_BARRIER.json')['count'] == 32
            cpu('scripts/score_stvg_evidence_v1.py', 'confirmation')
    assert sha(BASE / 'FROZEN_METHOD_LOCK.json') == frozen_sha
    write(B / 'CONFIRMATION_COMPLETION.json', dict(status='complete', no_selection=True, GT_read=True,
          frozen_method_lock_sha256=frozen_sha,
          prediction_barrier_sha256=sha(B / 'GLOBAL_PREDICTION_BARRIER.json'),
          postseal_rows_sha256=sha(B / 'POSTSEAL_ROWS.json'),
          postseal_summary_sha256=sha(B / 'POSTSEAL_SUMMARY.json'), time=time.time()))
    write(B / 'CONTROLLER_STATUS012.json', dict(status='complete', stage='root_handoff', GT_read=True,
          no_selection=True, time=time.time()), mutable=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(B / 'CONTROLLER_FAILURE012.json', dict(status='failed', error_type=type(exc).__name__,
              error=str(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
