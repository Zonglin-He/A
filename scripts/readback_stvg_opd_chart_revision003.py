"""Root CPU readback of actual qualified fit and first accepted missing arrival."""
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_chart_revision003 import REC, OLD_FAILURE, verify_revision, equal, counts
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
from scripts.stvg_opd_paper_common_v1 import committed
from scripts.decota_matrix_common_v1 import load


def qualification():
    verify_revision()
    receipt = read(REC / 'GPU_QUALIFICATION.json')
    assert receipt['status'] == 'pass' and receipt['predictions_accepted'] == 0
    assert sha(REC / 'QUALIFIED_BOUNDARY_FIT.pt') == receipt['qualified_fit_sha256']
    saved = load(REC / 'QUALIFIED_BOUNDARY_FIT.pt')
    old = load(OLD_FAILURE / 'REPRODUCED_CHART_FAILURE.pt')
    fit = saved['fit']; exact = counts()
    equal(fit['path'][:9], old['path'], exact)
    equal(fit['rounds'][:8], old['rounds'], exact)
    equal(fit['path'][9]['state'], old['failed_state'], exact)
    equal(fit['initial'], old['initial'], exact)
    from vg_tta.decota_spatial_opd_chart_revision003 import audit
    begin = time.perf_counter(); checked = audit(fit, old['expert'])
    seconds = time.perf_counter() - begin
    assert checked == receipt['independent_math']
    assert all(r['status'] == 'pass' for r in receipt['ordinary_controls'])
    write(REC / 'ROOT_QUALIFICATION_READBACK.json', dict(status='pass',
        qualification_receipt_sha256=sha(REC / 'GPU_QUALIFICATION.json'),
        full_qualified_fit_sha256=sha(REC / 'QUALIFIED_BOUNDARY_FIT.pt'),
        independent_CPU_math=checked, original_partial_fit_exact=exact,
        CPU_independent_audit_seconds=seconds, actual_GPU_fits=receipt['actual_fits'],
        original_failed_fit_not_serialized=True, equality_with_dead_memory_not_claimed=True,
        GT_read=False, P1_complete=False, time=time.time()))
    print(json.dumps(dict(status='root_actual_qualification_readback_pass', CPU_seconds=seconds)))


def resume():
    verify_revision()
    assert read(REC / 'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    expected = load(REC / 'QUALIFIED_BOUNDARY_FIT.pt')
    p = BASE / 'stages/P1_vidstg/clean/order1/on_policy/01882.pt'
    r = read(p.with_suffix('.json')); digest = sha(p)
    assert r['sha256'] == digest and r['bytes'] == p.stat().st_size and not r['GT_read']
    z = load(p); exact = counts(); equal(z['fit'], expected['fit'], exact)
    assert z['arrival'] == 1882 and z['query_ordinal'] == 10220 and not z['GT_read']
    assert z['previous_payload_sha256'] == expected['previous_payload_sha256']
    assert z['previous_payload_sha256'] == sha(p.with_name('01881.pt'))
    assert z['input']['sha256'] == expected['input']['sha256']
    assert z['input']['sha256'] == sha(BASE / z['input']['path'])
    assert z['runtime_lock_sha256'] == sha(BASE / 'RUNTIME_LOCK.json')
    assert z['config'] == expected['fit']['config']
    equal(z['committed'], committed(z['fit']['initial'], z['fit']['state'], z['config']['writeback']), exact)
    assert z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
    formal = read(REC / 'FIRST_FORMAL_FIT_BITWISE.json')
    assert formal['status'] == 'pass' and formal['time'] <= r['time']
    assert not (BASE / 'P1_PREDICTION_BARRIER.json').exists()
    progress = read(BASE / 'stages/P1_vidstg/STATUS.json')
    launch = read(REC / 'RELAUNCH.json'); stage = read(BASE / 'STAGE.json')
    for pid in [launch['controller_pid'], stage['worker_pid']]:
        assert Path('/proc', str(pid), 'cmdline').exists()
    write(REC / 'ROOT_RESUME_READBACK.json', dict(status='actual_missing_only_continuation_verified',
        first_accepted_arrival=1882, first_accepted_query_ordinal=10220,
        accepted_payload_sha256=digest, accepted_bytes=p.stat().st_size,
        original_prefix_previous_payload_sha256=z['previous_payload_sha256'],
        accepted_fit_equals_actual_GPU_qualification_bitwise=exact,
        independent_LN_writeback_and_reset_readback=True,
        original_runtime_sha256=z['runtime_lock_sha256'],
        authorized_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        progress_snapshot={k: progress[k] for k in ('status', 'done', 'total', 'order', 'GT_read', 'time')},
        controller_pid=launch['controller_pid'], worker_pid=stage['worker_pid'],
        HC2_sealed_arrivals=10446, HC2_not_rerun=True,
        P1_global_seal_present=False, P1_GT_read=False, P1_or_paper_complete=False,
        time=time.time()))
    print(json.dumps(dict(status='root_actual_resume_readback_pass', progress=progress['done'], total=progress['total'])))


if __name__ == '__main__':
    qualification() if sys.argv[1] == 'qualification' else resume()
