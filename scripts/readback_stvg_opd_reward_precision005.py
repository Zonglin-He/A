"""Actual no-GT CPU readback of qualification and first accepted continuation."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_reward_precision005 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
from scripts.decota_matrix_common_v1 import load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
from scripts.stvg_opd_paper_common_v1 import committed
from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit


def counter(): return dict(tensors=0, tensor_coordinates=0, scalar_values=0)


def qualification():
    verify(); q = read(REC/'GPU_QUALIFICATION.json')
    assert q['status'] == 'pass' and q['actual_complete_fits'] == 6 and q['accepted_predictions'] == 0
    assert not q['GT_read'] and q['input']['new_DINO_calls'] == 0
    assert sha(REC/'QUALIFIED_FIRST_MISSING_FIT.pt') == q['qualified_fit_sha256']
    failed = load(REC/'FAILED_FIT.pt'); actual = load(REC/'QUALIFIED_FIRST_MISSING_FIT.pt')
    comparison = counter(); equal(actual['fit'], failed['fit'], comparison)
    assert comparison == q['complete_fit_equals_actual_failed_process_serialization']
    begin = time.perf_counter(); checked = audit(actual['fit'], actual['expert']); seconds = time.perf_counter()-begin
    assert checked == q['independent_math'] == actual['independent_math']
    assert sha(BASE/actual['input']['path']) == actual['input']['sha256']
    assert sha(BASE/'stages/P1_vidstg/clean/order2/on_policy/05766.pt') == actual['previous_payload_sha256']
    # Historical math dictionary dispatch remains byte/value exact, with no GT.
    from vg_tta.decota_spatial_opd_precision_audit_revision001 import audit as a1, original_audit
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as a2
    from vg_tta.decota_spatial_opd_chart_revision003 import audit as a3, REVISION as rev3
    mapping = {None: original_audit, 'precision_audit_revision001': a1, 'precision_audit_revision002': a2, rev3: a3}
    from scripts.run_decota_paper_main_v1 import unpack_expert
    records = []
    for rel in ('P1_hc2/clean/order1/on_policy/00000.pt', 'P1_hc2/clean/order2/on_policy/00148.pt',
                'P1_vidstg/clean/order1/on_policy/00000.pt', 'P1_vidstg/clean/order1/on_policy/01882.pt',
                'P1_vidstg/clean/order2/on_policy/02185.pt'):
        p = BASE/'stages'/rel; z = load(p); inp = load(BASE/z['input']['path']); saved = z['math_audit']
        assert sha(p) == read(p.with_suffix('.json'))['sha256'] and not z['GT_read']
        revision = saved.get('revision'); assert mapping[revision](z['fit'], unpack_expert(inp['expert'])) == saved
        records.append(dict(status='pass', saved_revision=revision, exact_math_dictionary=True, payload_sha256=sha(p)))
    result = dict(status='pass', scope='actual GPU qualification and independent CPU audit-only root readback',
        actual_GPU_fits=6, accepted_predictions=0, new_DINO_calls=0, actual_failed_process_full_fit_bitwise=comparison,
        saved_math_dictionary_exact=True, historical_saved_math_dictionaries=records, historical_dictionary_count=5,
        qualified_fit_sha256=q['qualified_fit_sha256'], independent_CPU_seconds=seconds,
        reward_original_cross_precision_failure_preserved=True,
        max_original_GPU32_float64_reward_error=checked['max_detached_reward_error'],
        max_independent_CPU32_GPU32_reward_error=max(x['max_independent_CPU32_vs_GPU32_error'] for x in checked['reward_precision_checks']),
        max_interval_bound_fraction=max(x['max_cross_precision_bound_fraction'] for x in checked['reward_precision_checks']),
        full_10round_Gaussian_softmax_gradient_Adam_state_chart_checked=True,
        independent_full_decoder_Jacobian=False, CUDA_transcendental_kernel_proof=False,
        GT_read=False, P1_or_paper_complete=False, time=time.time())
    write(REC/'ROOT_QUALIFICATION_READBACK.json', result)
    print(json.dumps(dict(status='pass', GPU_fits=6, original_failed_fit_bitwise=comparison, historical_exact_dictionaries=5)))


def resume():
    verify(); assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    cap = read(REC/'CAPTURE_RECEIPT.json'); size = count = 0
    for rel, expected in cap['prefix_files'].items():
        p = ROOT/rel; rc = read(p.with_suffix('.json'))
        assert sha(p) == expected['sha256'] == rc['sha256']
        assert p.stat().st_size == expected['bytes'] == rc['bytes']
        assert sha(p.with_suffix('.json')) == expected['receipt_sha256'] and not rc['GT_read']
        size += expected['bytes']; count += 1
    assert count == cap['done'] == 16070 and size == cap['prefix_bytes']
    p = BASE/'stages/P1_vidstg/clean/order2/on_policy/05767.pt'; z = load(p)
    rc = read(p.with_suffix('.json')); expected = load(REC/'QUALIFIED_FIRST_MISSING_FIT.pt')
    assert sha(p) == rc['sha256'] and not rc['GT_read']
    comparison = counter(); equal(z['fit'], expected['fit'], comparison)
    assert z['math_audit'] == expected['independent_math'] == audit(z['fit'], expected['expert'])
    assert (z['order'], z['arrival'], z['query_ordinal']) == ('order2', 5767, 5250)
    assert z['previous_payload_sha256'] == expected['previous_payload_sha256'] == sha(p.with_name('05766.pt'))
    assert z['input']['sha256'] == expected['input']['sha256'] == sha(BASE/z['input']['path'])
    assert z['config'] == expected['fit']['config'] and z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
    writeback = counter(); equal(z['committed'], committed(z['fit']['initial'], z['fit']['state'], z['config']['writeback']), writeback)
    import numpy as np
    for name, tensor in z['committed'].items():
        initial = z['fit']['initial'][name].numpy(); final = z['fit']['state'][name].numpy()
        if name != 'spatial.query_residual':
            independent = initial + np.float32(z['config']['writeback'])*(final-initial)
            assert np.array_equal(independent, tensor.numpy())
        else: assert np.array_equal(np.zeros_like(initial), tensor.numpy())
    f = read(REC/'FIRST_FORMAL_FIT_BITWISE.json'); assert f['status'] == 'pass' and f['time'] <= rc['time']
    assert not (BASE/'P1_PREDICTION_BARRIER.json').exists()
    assert sha(BASE/'stages/P1_hc2/PREDICTION_BARRIER.json') == cap['HC2_barrier_sha256']
    progress = read(BASE/'stages/P1_vidstg/STATUS.json'); assert progress['done'] > 16070 and not progress['GT_read']
    launch = read(REC/'RELAUNCH.json'); stage = read(BASE/'STAGE.json')
    for pid in (launch['controller_pid'], stage['worker_pid']): assert Path('/proc', str(pid), 'cmdline').exists()
    result = dict(status='actual_missing_only_continuation_verified', scope='detached-reward auditor precision repair/qualification/resumption only',
        old_prefix_files_opaque_verified=count, old_prefix_bytes_unchanged=size,
        first_accepted_order='order2', first_accepted_arrival=5767, first_accepted_query_ordinal=5250,
        accepted_payload_sha256=sha(p), accepted_bytes=p.stat().st_size,
        accepted_fit_equals_actual_GPU_qualification_and_original_failed_fit_bitwise=comparison,
        mathematical_audit_dictionary_exact=True, actual_LN_writeback_comparisons=writeback,
        independent_NumPy_LN_writeback_and_resets=True, original_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'), readback_code_sha256=sha(Path(__file__)),
        progress_snapshot={k: progress[k] for k in ('status', 'done', 'total', 'order', 'GT_read', 'time')},
        controller_pid=launch['controller_pid'], worker_pid=stage['worker_pid'], HC2_sealed_arrivals=10446,
        old_prefix_and_sealed_HC2_not_rewritten=True, P1_global_seal_present=False, P1_GT_read=False,
        P1_or_paper_complete=False, original_failed_fit_serialized_before_exit=True, time=time.time())
    write(REC/'ROOT_RESUME_READBACK.json', result)
    print(json.dumps(dict(status=result['status'], prefix=count, bytes=size, progress=progress['done'])))


if __name__ == '__main__':
    {'qualification': qualification, 'resume': resume}[sys.argv[1]]()
