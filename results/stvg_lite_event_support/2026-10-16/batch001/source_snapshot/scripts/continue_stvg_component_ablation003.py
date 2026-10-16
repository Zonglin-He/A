"""Finite CPU continuation after a captured temporal runtime binding omission.

All sixteen sealed spatial trials are reused. No old unit is restarted and no
source, scientific formula, prediction or original runtime is overwritten.
"""
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_component_ablation_common_v1 import BASE, SPATIAL, read, sha, write, verify_final

PY = ROOT / '.conda/tubedetr/bin/python'
RECOVERY = BASE / 'recovery/spatial_empty_reference003'


def run(script, *args):
    subprocess.run([str(PY), '-B', 'scripts/' + script, *args], cwd=ROOT, check=True)


def main():
    verify_final()
    runtime = read(BASE / 'TEMPORAL_COMPONENT_RUNTIME003.json')
    for rel, expected in {**runtime['pins'], **runtime['protected']}.items():
        assert sha(ROOT / rel) == expected, rel
    for trial, _, _ in SPATIAL:
        p = BASE / 'trials' / trial
        root = read(p / 'ROOT_READBACK.json')
        cpu = read(p / 'CPU_COMPLETION.json')
        assert root['status'] == 'pass' and root['count'] == 128 and not root['GT_read']
        assert cpu['status'] == 'complete' and cpu['count'] == 128
    write(BASE / 'STATUS.json', dict(status='running', stage='temporal_component',
          done=16, total=16, GT_read=False, time=time.time()), mutable=True)
    run('run_stvg_component_temporal003.py', '--run')
    run('measure_stvg_component_trial003.py', '--temporal', '--score')
    run('prepare_stvg_component_native_clock_v1.py', '--prepare')
    write(BASE / 'ACTUAL_COMPONENT_HANDOFF003.json', dict(
          status='pending_actual_native_clock_teacher_student_root_then_complete_suite_public_archive',
          spatial_count=16, GT_read=False, time=time.time()))
    write(RECOVERY / 'RESOLUTION.json', dict(status='pass', scope='temporal CPU continuation and native input handoff',
          GT_read=False, time=time.time()))
    write(BASE / 'STATUS.json', dict(status='pending_actual_native_clock_teacher_student_root',
          stage='root_handoff', done=16, total=16, GT_read=False, time=time.time()), mutable=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        write(RECOVERY / 'FAILURE.json', dict(status='failed', stage='required_root_review',
              error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc(),
              GT_read=False, time=time.time()), mutable=True)
        write(BASE / 'STATUS.json', dict(status='failed', stage='required_root_review',
              GT_read=False, time=time.time()), mutable=True)
        raise
