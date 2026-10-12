"""Defer the original bounded timing until heavy baseline CPU review has exited."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_opd_supplement_v1'
DEST = BASE / 'cost/CPU_profile_isolation002'


def read(path):
    return json.loads(path.read_text())


def status(value):
    temp = DEST / f'STATUS.{os.getpid()}.tmp'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(DEST / 'STATUS.json')


def run():
    runtime = read(DEST / 'REVISION_RUNTIME.json')
    done = BASE / 'actual_root/postseal_continuation006/COMPLETE_baselines.json'
    while not done.exists():
        for path in [BASE / 'baselines/STATUS.json', BASE / 'actual_root/postseal_continuation006/STATUS.json']:
            assert 'fail' not in read(path)['status'], 'Preserved actual dependency failure needs root'
        status(dict(status='CPU_waiting_for_heavy_baseline_scoring_and_root_exit',
                    controller_pid=os.getpid(), new_model_calls=0, GT_read=False, time=time.time()))
        time.sleep(15)
    assert read(done)['status'] == 'actual_CPU_root_complete_pending_visual_publication'
    assert read(BASE / 'baselines/ROOT_ALL_SELECTED_FITS_READBACK.json')['selected_logical_arrivals'] == 82710
    for rel, digest in runtime['pins'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    with (DEST / 'original_timing_controller.log').open('a') as output:
        worker = subprocess.Popen(
            [str(ROOT / '.conda/tubedetr/bin/python'), '-B',
             'scripts/continue_stvg_opd_end_to_end_cost_supplement_v1.py'],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=env)
        status(dict(status='original_bounded_timing_controller_running',
                    controller_pid=os.getpid(), timing_controller_pid=worker.pid,
                    heavy_baseline_CPU_readback_exited=True, GT_read=False,
                    sample_or_model_or_timing_code_changed=False, time=time.time()))
        assert worker.wait() == 0, 'Preserved original timing failure needs root'
    assert all(read(BASE / 'cost' / ds / 'COMPLETION.json')['fits'] == 16 for ds in ['hc2', 'vidstg'])
    status(dict(status='original_32_timing_fits_complete_pending_actual_cost_root',
                query_fits=32, experiment_complete=False, GT_read=False, time=time.time()))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        evidence = DEST / 'failures' / str(time.time_ns())
        evidence.mkdir(parents=True)
        (evidence / 'traceback.txt').write_text(traceback.format_exc())
        status(dict(status='failed_preserved_pending_actual_root',
                    evidence=str(evidence.relative_to(ROOT)), experiment_complete=False,
                    GT_read=False, time=time.time()))
        raise
