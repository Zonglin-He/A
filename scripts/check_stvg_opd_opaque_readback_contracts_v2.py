"""Synthetic contracts for opaque byte and receipt integrity, with no model data."""
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/audit_stvg_opd_single_direction_bytes_v2.py'
spec = importlib.util.spec_from_file_location('opaque_readback', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    runtime = 'a' * 64
    config = {'lr': 0.01}
    stage = dict(source='vidstg', dataset='hc2', conditions=['clean'],
                 arms=['on_policy'], adapted_arrivals=9,
                 orders={f'order{i}': [0, 1, 2] for i in range(1, 4)})
    files = {f'stages/P1_hc2/clean/{order}/on_policy/{at:05}.pt': 'b' * 64
             for order in stage['orders'] for at in range(3)}
    inputs = {f'inputs/vidstg_to_hc2/clean/{q:05}.pt': 'c' * 64 for q in range(3)}
    barrier = dict(status='sealed', qualification=False, stage='P1_hc2',
                   orders=stage['orders'], config=config, source_checkpoint_unchanged=True,
                   GT_read=False, runtime_lock_sha256=runtime, adapted_arrivals=9,
                   Frozen_logical_arrivals=9, files=files, inputs=inputs)
    expected, expected_inputs = audit.coverage(barrier, stage, config, runtime)
    assert expected == set(files) and expected_inputs == set(inputs)
    data = b'opaque bytes, never deserialize\x00\xff'
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'synthetic.pt'
        path.write_bytes(data)
        assert audit.sha(path) == digest
    receipt = dict(sha256=digest, bytes=len(data), GT_read=False,
                   runtime_lock_sha256=runtime, time=1.0)
    audit.receipt_check(receipt, len(data), digest, runtime, 2.0, True)
    input_receipt = {k: v for k, v in receipt.items() if k != 'bytes'}
    audit.receipt_check(input_receipt, len(data), digest, runtime, 2.0, False)
    rejected = []

    def reject(name, call):
        try:
            call()
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Corrupt synthetic metadata accepted: ' + name)

    def bad_barrier(name, change):
        b, s = copy.deepcopy((barrier, stage))
        change(b, s)
        reject(name, lambda: audit.coverage(b, s, config, runtime))

    bad_barrier('not_sealed', lambda b, s: b.update(status='running'))
    bad_barrier('qualification_is_not_formal', lambda b, s: b.update(qualification=True))
    bad_barrier('GT_read_true', lambda b, s: b.update(GT_read=True))
    bad_barrier('GT_read_numeric_zero', lambda b, s: b.update(GT_read=0))
    bad_barrier('source_changed', lambda b, s: b.update(source_checkpoint_unchanged=False))
    bad_barrier('wrong_runtime', lambda b, s: b.update(runtime_lock_sha256='d' * 64))
    bad_barrier('wrong_config', lambda b, s: b.update(config={'lr': 1.0}))
    bad_barrier('missing_prediction', lambda b, s: b['files'].pop(next(iter(b['files']))))
    bad_barrier('extra_prediction', lambda b, s: b['files'].update({'unexpected.pt': 'e' * 64}))
    bad_barrier('missing_input', lambda b, s: b['inputs'].pop(next(iter(b['inputs']))))
    bad_barrier('duplicate_query', lambda b, s: s['orders']['order1'].__setitem__(1, 0))
    bad_barrier('missing_order', lambda b, s: s['orders'].pop('order3'))
    bad_barrier('wrong_condition', lambda b, s: s.update(conditions=['blur']))
    bad_barrier('wrong_arm', lambda b, s: s.update(arms=['shuffled']))
    bad_barrier('wrong_adapted_count', lambda b, s: b.update(adapted_arrivals=8))
    bad_barrier('wrong_Frozen_count', lambda b, s: b.update(Frozen_logical_arrivals=8))
    bad_barrier('wrong_design_count', lambda b, s: s.update(adapted_arrivals=8))
    for name, change in [
        ('wrong_SHA', {'sha256': 'f' * 64}),
        ('wrong_receipt_runtime', {'runtime_lock_sha256': 'f' * 64}),
        ('receipt_GT_true', {'GT_read': True}),
        ('receipt_GT_numeric_zero', {'GT_read': 0}),
        ('receipt_after_seal', {'time': 3.0}),
        ('wrong_byte_count', {'bytes': len(data) + 1}),
        ('unexpected_receipt_field', {'extra': 'not allowed'}),
    ]:
        candidate = dict(receipt, **change)
        reject(name, lambda c=candidate: audit.receipt_check(c, len(data), digest, runtime, 2.0, True))
    reject('input_receipt_prediction_bytes', lambda: audit.receipt_check(
        receipt, len(data), digest, runtime, 2.0, False))
    result = dict(status='pass', valid_contracts=4, rejected_invalid_contracts=len(rejected),
                  rejected_cases=rejected, synthetic_only=True, model_execution=False,
                  GT_or_real_prediction_array_read=False,
                  auditor_sha256=audit.sha(SCRIPT), checker_sha256=audit.sha(Path(__file__)))
    output = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2/P1_OPAQUE_READBACK_CONTRACTS.json'
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
