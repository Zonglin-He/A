"""One explicitly invoked, serial DESTA source stage with full-wall accounting."""
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/desta3d_v1'
OUT = ART / 'source_fit'
PHASES = {
    'capture-queries': ('exact_clean_query_capture', 'query_capture_stage_seconds'),
    'fit': ('source_three_arm_fit_and_clean_validation', 'fit_plus_validation_stage_seconds'),
    'eval-corrupt': ('source_validation_corruptions', 'corruption_readout_stage_seconds'),
}


def read(path):
    return json.loads(Path(path).read_text())


def receipts():
    records = []
    for path in ART.rglob('*.json'):
        if path.parent.name != 'receipts':
            continue
        row = read(path)
        seconds = float(row['seconds'])
        assert math.isfinite(seconds) and seconds >= 0, path
        records.append(row)
    return records


def main(action):
    config_path = OUT / 'TRAIN_CONFIG.json'
    lock_path = OUT / 'TRAIN_LOCK.json'
    config, lock = read(config_path), read(lock_path)
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == lock['config_sha256']
    stage, cap_key = PHASES[action]
    global_cap = float(config['resource_caps']['cumulative_gpu_seconds'])
    assert global_cap == 28800
    stage_cap = float(config['resource_caps'][cap_key])
    prior = receipts()
    total_before = sum(r['seconds'] for r in prior)
    stage_before = sum(r['seconds'] for r in prior if r.get('stage') == stage or r.get('supervised_stage') == stage)
    allowed = min(global_cap-total_before, stage_cap-stage_before)
    assert allowed > 30, 'Insufficient remaining registered stage/global budget'
    invocation = str(time.time_ns())
    logs = OUT / 'supervisor_logs'
    logs.mkdir(exist_ok=True, parents=True)
    log_path = logs / (action+'_'+invocation+'.log')
    command = [str(ROOT / '.venv-ptd-audit/bin/python'), '-B', str(ROOT / 'scripts/desta3d_source_fit_v1.py'), action]
    started = time.monotonic()
    status = None
    print(json.dumps(dict(action=action, log=str(log_path), remaining_seconds=allowed,
                          global_accounted_seconds_before=total_before)), flush=True)
    try:
        with log_path.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=allowed-15)
        status = result.returncode
    except subprocess.TimeoutExpired:
        status = 'outer_budget_timeout'
    finally:
        elapsed = time.monotonic()-started
        current = sum(r['seconds'] for r in receipts())
        child_charge = current-total_before
        overhead = max(0., elapsed-child_charge)
        receipt = dict(stage='source_supervisor_full_wall_overhead', supervised_stage=stage,
                       seconds=overhead, child_accounted_seconds=child_charge, worker_wall_seconds=elapsed,
                       action=action, status=status, log=str(log_path), global_before=total_before,
                       stage_before=stage_before, phase_cap=stage_cap, allowed_seconds=allowed,
                       supervisor_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       train_lock_sha=hashlib.sha256(lock_path.read_bytes()).hexdigest())
        out = OUT / 'receipts' / ('supervisor_'+invocation+'.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        assert not out.exists()
        out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(action=action, status=status, wall_seconds=elapsed,
                          cumulative_seconds=sum(r['seconds'] for r in receipts()))), flush=True)
    if status != 0:
        raise SystemExit(f'{action} stopped ({status}); preserve log and diagnose before explicit retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=PHASES)
    main(parser.parse_args().action)
