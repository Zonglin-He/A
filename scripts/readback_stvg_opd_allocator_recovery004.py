"""CPU root readback of allocator-only qualification and actual resumption."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_allocator_recovery004 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
from scripts.decota_matrix_common_v1 import load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
from scripts.stvg_opd_paper_common_v1 import committed


def run():
    verify()
    qualification = read(REC / 'GPU_QUALIFICATION.json')
    assert qualification['status'] == 'pass' and qualification['accepted_predictions'] == 0
    assert read(REC / 'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    count = size = 0
    for rel, expected in capture['prefix_files'].items():
        p = ROOT / rel
        rc = p.with_suffix('.json')
        assert sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes']
        assert sha(rc) == expected['receipt_sha256']
        r = read(rc)
        assert r['sha256'] == expected['sha256'] and not r['GT_read']
        count += 1
        size += expected['bytes']
    assert count == capture['done'] == 12488 and size == capture['prefix_bytes']
    qualified = REC / 'QUALIFIED_FIRST_MISSING_FIT.pt'
    assert sha(qualified) == qualification['qualified_fit_sha256']
    expected = load(qualified)
    first = BASE / 'stages/P1_vidstg/clean/order2/on_policy/02185.pt'
    r = read(first.with_suffix('.json'))
    digest = sha(first)
    assert r['sha256'] == digest and r['bytes'] == first.stat().st_size and not r['GT_read']
    z = load(first)
    exact = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    equal(z['fit'], expected['fit'], exact)
    assert exact == qualification['repeated_full_fit_bitwise']
    assert (z['order'], z['arrival'], z['query_ordinal']) == ('order2', 2185, 2685)
    assert not z['GT_read'] and not z['fit']['GT_used']
    assert z['previous_payload_sha256'] == expected['previous_payload_sha256']
    assert z['previous_payload_sha256'] == sha(first.with_name('02184.pt'))
    assert z['input']['sha256'] == expected['input']['sha256'] == sha(BASE / z['input']['path'])
    assert z['runtime_lock_sha256'] == sha(BASE / 'RUNTIME_LOCK.json')
    assert z['config'] == expected['fit']['config']
    assert z['math_audit'] == expected['independent_math']
    writeback = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    equal(z['committed'], committed(z['fit']['initial'], z['fit']['state'], z['config']['writeback']), writeback)
    assert z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
    formal = read(REC / 'FIRST_FORMAL_FIT_BITWISE.json')
    assert formal['status'] == 'pass' and formal['time'] <= r['time']
    assert not (BASE / 'P1_PREDICTION_BARRIER.json').exists()
    hc = read(BASE / 'stages/P1_hc2/PREDICTION_BARRIER.json')
    assert hc['status'] == 'sealed' and hc['adapted_arrivals'] == 10446
    progress = read(BASE / 'stages/P1_vidstg/STATUS.json')
    assert progress['done'] > 12488 and not progress['GT_read']
    stage = read(BASE / 'STAGE.json')
    launch = read(REC / 'RELAUNCH.json')
    for pid in (launch['controller_pid'], stage['worker_pid']):
        assert Path('/proc', str(pid), 'cmdline').exists()
    result = dict(status='actual_missing_only_continuation_verified',
        scope='allocator-only engineering repair qualification and resumption',
        old_prefix_files_opaque_verified=count, old_prefix_bytes_unchanged=size,
        old_prefix_capture_sha256=sha(REC / 'CAPTURE_RECEIPT.json'),
        first_accepted_order='order2', first_accepted_arrival=2185, first_accepted_query_ordinal=2685,
        accepted_payload_sha256=digest, accepted_bytes=first.stat().st_size,
        original_prefix_previous_payload_sha256=z['previous_payload_sha256'],
        accepted_fit_equals_actual_GPU_qualification_bitwise=exact,
        actual_LN_writeback_comparisons=writeback,
        mathematical_audit_dictionary_exact=True, independent_LN_writeback_and_reset_readback=True,
        original_runtime_sha256=z['runtime_lock_sha256'],
        engineering_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        readback_code_sha256=sha(Path(__file__)),
        progress_snapshot={k: progress[k] for k in ('status', 'done', 'total', 'order', 'GT_read', 'time')},
        controller_pid=launch['controller_pid'], worker_pid=stage['worker_pid'],
        HC2_sealed_arrivals=10446, HC2_not_rerun=True,
        P1_global_seal_present=False, P1_GT_read=False, P1_or_paper_complete=False,
        equality_with_dead_failed_capture_not_claimed=True, future_memory_safety_not_claimed=True,
        time=time.time())
    write(REC / 'ROOT_RESUME_READBACK.json', result)
    print(json.dumps(dict(status=result['status'], old_prefix_files=count,
        old_prefix_bytes=size, progress=progress['done'], total=progress['total'])))


if __name__ == '__main__':
    run()
