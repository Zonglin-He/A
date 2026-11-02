"""One-shot continuation of the already frozen R51 full evidence experiment.

Wait only for the two existing full input jobs, then invoke two pinned CPU
workers through the resource guard. No scheduler, GT, metric, selection,
retry, external model, or parameter adaptation is implemented here.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, time

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'research/round51/experiment'
PY = ROOT / 'crossplatform/env/bin/python'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, v):
    p = Path(p)
    assert not p.exists(), 'Never replace an existing lock or receipt'
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    p = read(a.plan)
    assert p['state'] == 'ROOT_AUTHORIZED_ONESHOT_ALREADY_FROZEN_FULL_EVIDENCE_NO_GT'
    assert p['program_sha256'] == sha(__file__)
    freeze = ROOT / p['method_freeze']
    assert sha(freeze) == p['method_freeze_sha256']
    f = read(freeze)
    assert f['selected_method'] == 'grouped_replace_all_c05' and f['selected_factor'] == .5
    assert f['TTA_will_execute'] is False and f['no_new_full_calibration'] is True
    for rel, h in p['programs_sha256'].items():
        assert sha(ROOT / rel) == h, rel
    prep = 'crossplatform/round51/prepared5912_attempt001'
    refs = 'crossplatform/round51/reference5912_headspan_attempt001'
    source = 'crossplatform/round8/full5912_attempt001'
    manifest = 'research/round8/experiment/full5912_gpu_inputs.json'
    start = time.perf_counter()
    while not all((ROOT / x / 'SEALED.json').is_file() for x in (prep, refs)):
        for x in (prep, refs):
            assert not list((ROOT / x).glob('FAILED*.json')), 'Existing prerequisite failed; do not retry'
        assert time.perf_counter() - start < p['input_wait_timeout_seconds']
        time.sleep(5)
    # A worker seals its output before its guard releases the shared CPU lock.
    # Wait for both the active marker and a successful lock probe; do not retry
    # an experiment payload or race an existing worker's finalization.
    import fcntl
    while True:
        if (ROOT/'research/round51/RUNNING_CPU.json').exists():
            time.sleep(.2)
            continue
        with (ROOT/'research/round51/CPU.lock').open('a+') as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(.2)
                continue
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            break
    stages = [
        ('GROUP_MEDOID_FULL5912', 'crossplatform/round51/grouped_full51.py', {
            'state': 'ROOT_LOCKED_R51_FULL_GROUP_MEDOID_G', 'n': 5912,
            'manifest': manifest, 'canonical_source_root': source,
            'prepared_root': prep, 'reference_root': refs,
            'output_root': 'crossplatform/round51/grouped5912_attempt001',
            'appearance_factor': .5, 'single_registered_G_carrier_mechanism_repair': True,
            'method_selected_before_full_run': True, 'selected_method': 'grouped_replace_all_c05',
            'no_new_Qwen_models_GT_or_TTA': True, 'max_wall_seconds': 6500,
            'programs_sha256': {k:v for k,v in p['programs_sha256'].items()
                               if k != 'crossplatform/round51/noassociation_full51.py'},
            'inputs_sha256': {x:sha(ROOT/x) for x in [manifest, prep+'/SEALED.json',
                                                   refs+'/SEALED.json', source+'/SEALED.json']}
        }, 'ROOT_AUTHORIZED_R51_FULL_GROUP_MEDOID_G_NO_GT', 7200),
        ('NO_ASSOCIATION_FULL5912', 'crossplatform/round51/noassociation_full51.py', {
            'state': 'ROOT_LOCKED_R51_NO_ASSOCIATION_RAW_SAFE_ABLATION', 'n': 5912,
            'manifest': manifest, 'appearance_factor_values': [.5],
            'retain_RGB_and_remove_instance_binding_and_duplicate_suppression': True,
            'prepared_root': prep, 'reference_root': refs,
            'output_root': 'crossplatform/round51/noassociation5912_attempt001',
            'method_selected_before_full_run': True, 'selected_method': 'grouped_replace_all_c05',
            'programs_sha256': {k:v for k,v in p['programs_sha256'].items()
                               if k != 'crossplatform/round51/grouped_full51.py'},
            'inputs_sha256': {x:sha(ROOT/x) for x in [manifest, prep+'/SEALED.json', refs+'/SEALED.json']}
        }, 'ROOT_AUTHORIZED_R51_NO_ASSOCIATION_RAW_SAFE_ABLATION_NO_GT', 7200)
    ]
    for label, script, plan, state, timeout in stages:
        pp = D / (label+'_PLAN_LOCK.json')
        aa = D / (label+'_AUTHORIZATION.json')
        dump(pp, plan)
        dump(aa, {'state':state, 'program_sha256':sha(ROOT/script), 'plan_sha256':sha(pp)})
        print('START_FIXED_STAGE', label, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
        code = subprocess.call([str(PY), 'tools/manual_r51_guard.py', '--resource', 'cpu',
                                '--timeout-seconds', str(timeout), '--', str(PY), script,
                                '--plan', str(pp.relative_to(ROOT)), '--authorization', str(aa.relative_to(ROOT))], cwd=ROOT)
        assert code == 0, (label, code, 'No automatic retry or method change')
        assert (ROOT/plan['output_root']/'SEALED.json').is_file()
        print('SEALED_FIXED_STAGE', label, flush=True)
    dump(ROOT/'research/round51/FULL_EVIDENCE_ONESHOT_RECEIPT.json', {
        'state':'COMPLETE_TWO_ALREADY_FROZEN_FULL_EVIDENCE_WORKERS_NO_GT',
        'method_freeze_sha256':sha(freeze), 'plan_sha256':sha(a.plan),
        'seconds_including_existing_inputs_wait':time.perf_counter()-start,
        'no_scheduler_TTA_scoring_or_selection':True})

if __name__ == '__main__':
    main()
