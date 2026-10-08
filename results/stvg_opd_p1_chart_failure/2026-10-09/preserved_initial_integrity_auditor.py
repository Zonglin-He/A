"""Opaque readback of one sealed direction; never authorize partial GT scoring."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def coverage(barrier, stage, config, runtime):
    assert barrier['status'] == 'sealed' and not barrier['qualification']
    assert barrier['orders'] == stage['orders'] and barrier['config'] == config
    assert barrier['source_checkpoint_unchanged'] and not barrier['GT_read']
    assert barrier['runtime_lock_sha256'] == runtime
    n = len(next(iter(stage['orders'].values())))
    target = list(range(n))
    assert all(sorted(seq) == target for seq in stage['orders'].values())
    assert len(stage['orders']) == 3 and stage['conditions'] == ['clean']
    assert stage['arms'] == ['on_policy']
    name = barrier['stage']
    expected = {f'stages/{name}/clean/{order}/on_policy/{at:05}.pt'
                for order, seq in stage['orders'].items() for at in range(len(seq))}
    expected_inputs = {f'inputs/{stage["source"]}_to_{stage["dataset"]}/clean/{q:05}.pt'
                       for q in target}
    assert set(barrier['files']) == expected and set(barrier['inputs']) == expected_inputs
    assert barrier['adapted_arrivals'] == len(expected) == stage['adapted_arrivals']
    assert barrier['Frozen_logical_arrivals'] == len(expected)
    return expected, expected_inputs


def receipt_check(receipt, size, digest, runtime, seal_time, prediction):
    assert receipt['sha256'] == digest and receipt['runtime_lock_sha256'] == runtime
    assert not receipt['GT_read'] and receipt['time'] <= seal_time
    if prediction:
        assert receipt['bytes'] == size
        assert set(receipt) == {'sha256', 'bytes', 'GT_read', 'runtime_lock_sha256', 'time'}
    else:
        assert set(receipt) == {'sha256', 'GT_read', 'runtime_lock_sha256', 'time'}


def run(dataset):
    assert dataset in ('hc2', 'vidstg')
    name = 'P1_' + dataset
    path = BASE / 'stages' / name / 'PREDICTION_BARRIER.json'
    barrier = read(path)
    design = read(BASE / 'DESIGN_LOCK.json')
    lock = read(BASE / 'RUNTIME_LOCK.json')
    runtime = sha(BASE / 'RUNTIME_LOCK.json')
    assert sha(BASE / 'DESIGN_LOCK.json') == lock['design_sha256']
    for rel, digest in lock['pins'].items():
        assert sha(ROOT / rel) == digest, rel
    assert sha(ROOT / 'methods/decota_spatial_opd_v1/configs.json') == lock['selected_config_file_sha256']
    expected, inputs = coverage(barrier, design['stages'][name], design['datasets'][dataset]['config'], runtime)
    assert len(inputs) == (3482 if dataset == 'hc2' else 10303)
    actual = {str(p.relative_to(BASE)) for p in (BASE / 'stages' / name / 'clean').glob('*/*/*.pt')}
    assert actual == expected
    groups = {}
    latest_receipt = 0.0
    for key in ['files', 'inputs']:
        manifest = hashlib.sha256()
        nbytes = 0
        for rel, digest in sorted(barrier[key].items()):
            p = BASE / rel
            assert not p.is_symlink() and p.is_file()
            size = p.stat().st_size
            assert sha(p) == digest, rel
            receipt_path = p.with_suffix('.json')
            assert not receipt_path.is_symlink()
            rc = read(receipt_path)
            receipt_check(rc, size, digest, runtime, barrier['time'], key == 'files')
            latest_receipt = max(latest_receipt, rc['time'])
            manifest.update(json.dumps([rel, digest, size, sha(receipt_path)], separators=(',', ':')).encode())
            nbytes += size
        groups[key] = dict(opaque_files=len(barrier[key]), bytes=nbytes,
                           ordered_path_payload_receipt_digest=manifest.hexdigest())
    assert sha(path) == sha(BASE / 'stages' / name / 'PREDICTION_BARRIER.json')
    result = dict(status='pass', scope='single_sealed_direction_opaque_integrity_only', stage=name,
                  dataset=dataset, official_queries=len(inputs), adapted_arrivals=len(expected),
                  Frozen_logical_arrivals=barrier['Frozen_logical_arrivals'],
                  order_arrivals={k: len(v) for k, v in barrier['orders'].items()},
                  opaque_files=sum(g['opaque_files'] for g in groups.values()),
                  bytes=sum(g['bytes'] for g in groups.values()), groups=groups,
                  complete_exact_roster_coverage=True, payload_and_receipt_SHA256_verified=True,
                  receipt_runtime_and_before_seal_time_verified=True,
                  latest_receipt_time=latest_receipt, single_direction_seal_time=barrier['time'],
                  source_checkpoint_unchanged_as_sealed_metadata=True,
                  actual_parameter_or_Jacobian_validation_not_claimed=True,
                  prediction_array_GT_weights_gradients_or_scores_read=False,
                  GT_scoring_authorized_by_this_check=False, all_P1_or_paper_complete=False,
                  original_runtime_sha256=runtime, single_direction_barrier_sha256=sha(path),
                  auditor_sha256=sha(Path(__file__)), time=time.time())
    out = BASE / (name.upper() + '_ROOT_BYTE_READBACK.json')
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'stage', 'adapted_arrivals', 'opaque_files', 'bytes', 'all_P1_or_paper_complete']}))


if __name__ == '__main__':
    run(sys.argv[1])
