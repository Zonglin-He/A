"""Qualify authorized chart extension, then continue original missing P1 suffix."""
import collections
import gc
import os
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, status, sha, activate
from scripts.decota_matrix_common_v1 import save, load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import verify_revision as verify002, equal, caller_metadata
REC = BASE / 'recovery/P1_box_chart_003/authorized_revision003'
OLD_FAILURE = REC.parent


class QualificationComplete(Exception): pass


def verify_revision():
    verify002()
    auth = read(REC / 'AUTHORIZATION_RECEIPT.json')
    assert auth['status'] == 'authorized_by_human' and auth['changes_original_exact_endpoint_failure_behavior']
    rt = read(REC / 'REVISION_RUNTIME.json')
    for f, h in rt['pins'].items(): assert sha(ROOT / f) == h, f
    assert sha(REC / 'AUTHORIZATION_RECEIPT.json') == rt['authorization_receipt_sha256']
    assert sha(BASE / 'RUNTIME_LOCK.json') == rt['original_runtime_sha256']
    assert sha(OLD_FAILURE / 'REPRODUCED_CHART_FAILURE.pt') == rt['original_reproduced_failure_sha256']
    dynamic = {'STATUS.json', 'STAGE.json'}
    for f, item in read(OLD_FAILURE / 'CAPTURE_RECEIPT.json')['original_code'].items():
        original = OLD_FAILURE / 'originals' / f
        assert sha(original) == item['sha256'], str(original)
        if Path(f).name not in dynamic: assert sha(ROOT / f) == item['sha256'], f
    import torch
    assert torch.version.git_version == rt['torch_git_version']
    return rt


def prefix_readback():
    cap = read(OLD_FAILURE / 'CAPTURE_RECEIPT.json')
    total = 0
    for f, item in cap['prefix_files'].items():
        path = ROOT / f
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        assert sha(path.with_suffix('.json')) == item['receipt_sha256']
        total += item['bytes']
    assert total == cap['prefix_bytes'] and cap['done'] == 1882
    return dict(arrivals=1882, bytes=total, original_prefix_bytes_unchanged=True,
        full_prefix_capture_sha256=sha(OLD_FAILURE / 'CAPTURE_RECEIPT.json'))


def counts(): return dict(tensors=0, tensor_coordinates=0, scalar_values=0)


def ordinary_controls(frame, installation):
    """Predeclared actual controls, same source model, cached experts, saved initial."""
    import torch
    from scripts.run_decota_paper_main_v1 import read_row
    import scripts.run_decota_spatial_opd_v1 as capture
    from methods.decota_final_simplified_v1.tensors import state_hash
    loc = frame.f_locals
    model = loc['model']; original_base = loc['base']
    original_base.restore(original_base.initial)
    records = []
    for at in read(REC / 'REVISION_RUNTIME.json')['ordinary_control_arrivals']:
        oldpath = BASE / 'stages/P1_vidstg/clean/order1/on_policy' / f'{at:05}.pt'
        old = load(oldpath); row = read_row('vidstg', old['query_ordinal'])
        base, native, expert, rc = capture.capture(model, loc['expert'], loc['stage'], row, 'clean', collections.OrderedDict())
        assert rc['sha256'] == old['input']['sha256'] and rc['new_DINO_calls'] == 0
        torch.cuda.synchronize(); begin = time.perf_counter()
        with installation.original() as original_fit:
            control = original_fit(base, old['fit']['initial'], expert, row['frame_ids'], row['key'],
                                   'on_policy', config=old['config'])
        torch.cuda.synchronize(); original_seconds = time.perf_counter() - begin
        old_counts = counts(); equal(control, old['fit'], old_counts)
        torch.cuda.synchronize(); begin = time.perf_counter()
        revised = installation.fit(base, old['fit']['initial'], expert, row['frame_ids'], row['key'],
                                   'on_policy', config=old['config'])
        torch.cuda.synchronize(); revised_seconds = time.perf_counter() - begin
        parity_counts = counts()
        equal({k: v for k, v in revised.items() if k != 'numerical_chart'}, control, parity_counts)
        from vg_tta.decota_spatial_opd_chart_revision003 import audit
        checked = audit(revised, expert)
        assert checked['authorized_chart_extension']['endpoint_coordinates'] == 0
        records.append(dict(status='pass', arrival=at, query_ordinal=old['query_ordinal'],
            saved_fit_replay_bitwise=old_counts, revised_original_fields_bitwise=parity_counts,
            original_fit_GPU_seconds=original_seconds, revised_fit_GPU_seconds=revised_seconds,
            independent_math=checked, old_payload_sha256=sha(oldpath), new_DINO_calls=0,
            GT_read=False, source_checkpoint_restored=True))
        base.restore(base.initial); del base, control, revised, native, expert
        gc.collect(); torch.cuda.empty_cache()
    assert state_hash(model.state_dict()) == loc['modelhash']
    return records


def run(job, *args):
    verify_revision(); activate()
    from scripts.run_stvg_opd_gradient_recovery_001 import activate_revision
    prior001 = activate_revision()
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as prior002, original_audit
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation, audit as current_audit, REVISION
    import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
    import torch
    assert read(REC / 'CPU_CONTRACTS.json')['status'] == 'pass'
    requal = job == 'GPU_requal'
    if job in ('GPU', 'GPU_requal'):
        assert args == ('P1_vidstg',), 'Sealed HC2 is not a new execution'
        assert not (BASE / 'P1_PREDICTION_BARRIER.json').exists()
        if requal:
            assert not (REC / 'GPU_QUALIFICATION.json').exists()
            prefix = prefix_readback()
            expected = load(OLD_FAILURE / 'REPRODUCED_CHART_FAILURE.pt')
        else:
            assert read(REC / 'GPU_QUALIFICATION.json')['status'] == 'pass'
            expected = load(REC / 'QUALIFIED_BOUNDARY_FIT.pt')
        installation = Installation(qualification_vjp=requal)

        def gpu_audit(fit, expert):
            frame = sys._getframe(1); meta = caller_metadata(frame)
            try:
                checked = current_audit(fit, expert)
                if meta['order'] == 'order1' and meta['arrival'] == 1882:
                    assert meta['query_ordinal'] == 10220
                    assert meta['previous_payload_sha256'] == expected['previous_payload_sha256']
                    assert meta['input']['sha256'] == expected['input']['sha256']
                    assert meta['input']['new_DINO_calls'] == 0
                    if requal:
                        parity = counts()
                        equal(fit['path'][:9], expected['path'], parity)
                        equal(fit['rounds'][:8], expected['rounds'], parity)
                        equal(fit['path'][9]['state'], expected['failed_state'], parity)
                        equal(fit['initial'], expected['initial'], parity)
                        vjp_checks = installation.last_vjp_checks
                        assert len(vjp_checks) == 10 and fit['selected_step'] == 10
                        endpoints = [r for r in fit['numerical_chart']['trace'] if r['boundary'].any()]
                        assert endpoints and (endpoints[0]['round'], endpoints[0]['phase']) == (8, 'after')
                        equal(endpoints[0]['raw_logits'], expected['native_pre_sigmoid_logits'][fit['positions']], parity)
                        equal(endpoints[0]['boxes'], expected['chart_boxes'], parity)
                        installation.qualification_vjp = False
                        loc = frame.f_locals
                        torch.cuda.synchronize(); begin = time.perf_counter()
                        repeated = installation.fit(loc['base'], loc['initial'], expert, loc['row']['frame_ids'],
                            loc['row']['key'], loc['arm'], config=loc['cfg'])
                        torch.cuda.synchronize(); repeat_seconds = time.perf_counter() - begin
                        repeated_counts = counts(); equal(fit, repeated, repeated_counts)
                        assert current_audit(repeated, expert) == checked
                        controls = ordinary_controls(frame, installation)
                        save(REC / 'QUALIFIED_BOUNDARY_FIT.pt', dict(fit=fit, **meta))
                        write(REC / 'GPU_QUALIFICATION.json', dict(status='pass', **meta,
                            actual_fits=6, boundary_complete_fits=2, ordinary_control_fits=4,
                            qualified_final_round=10, complete_rounds=10,
                            original_completed_rounds_and_ninth_update_bitwise=parity,
                            repeated_complete_fit_bitwise=repeated_counts,
                            real_native_head_VJP_checks=vjp_checks,
                            qualified_fit_sha256=sha(REC / 'QUALIFIED_BOUNDARY_FIT.pt'),
                            original_reproduction_sha256=sha(OLD_FAILURE / 'REPRODUCED_CHART_FAILURE.pt'),
                            original_failed_fit_not_serialized=True,
                            equality_with_dead_memory_not_claimed=True,
                            first_fit_GPU_seconds=loc['seconds'], repeated_fit_GPU_seconds=repeat_seconds,
                            ordinary_controls=controls, independent_math=checked, prefix_readback=prefix,
                            predictions_accepted=0, HC2_not_rerun=True,
                            runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'), time=time.time()))
                        raise QualificationComplete()
                    parity = counts(); equal(fit, expected['fit'], parity)
                    assert checked == read(REC / 'GPU_QUALIFICATION.json')['independent_math']
                    if not (REC / 'FIRST_FORMAL_FIT_BITWISE.json').exists():
                        write(REC / 'FIRST_FORMAL_FIT_BITWISE.json', dict(status='pass', **meta,
                            **parity, full_qualified_fit_bitwise=True,
                            qualified_fit_sha256=sha(REC / 'QUALIFIED_BOUNDARY_FIT.pt'),
                            runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
                            fit_GPU_seconds=frame.f_locals['seconds'], GT_scoring=False,
                            next_operation='write only first missing formal prediction', time=time.time()))
                assert not requal, 'Stop qualification at exact first missing arrival'
                return checked
            except QualificationComplete: raise
            except BaseException:
                dest = REC / 'new_failures' / str(time.time_ns())
                dest.mkdir(parents=True, exist_ok=True)
                from methods.decota_final_simplified_v1.tensors import detached
                save(dest / 'FAILED_FIT.pt', dict(fit=fit, expert=detached(expert, 'cpu'), **meta))
                (dest / 'traceback.txt').write_text(traceback.format_exc())
                write(dest / 'FAILURE_RECEIPT.json', dict(status='actual_new_failure_preserved', **meta,
                    failed_fit_sha256=sha(dest / 'FAILED_FIT.pt'), time=time.time()))
                raise
        target.audit = gpu_audit
    elif job == 'P1_score':
        assert read(BASE / 'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
        def receipt_audit(fit, expert):
            frame = sys._getframe(1)
            assert frame.f_code.co_filename.endswith('score_stvg_opd_p1_v1.py')
            revision = frame.f_locals['z']['math_audit'].get('revision')
            if revision is None: return original_audit(fit, expert)
            if revision == 'precision_audit_revision001': return prior001(fit, expert)
            if revision == 'precision_audit_revision002': return prior002(fit, expert)
            assert revision == REVISION, revision
            return current_audit(fit, expert)
        target.audit = receipt_audit
    else: target.audit = current_audit
    from scripts.run_stvg_opd_paper_hc2_revision_v2 import run as dispatch
    try: dispatch('GPU' if requal else job, *args)
    except QualificationComplete:
        print('ACTUAL_NATIVE_LOGIT_CHART_REVISION003_GPU_QUALIFICATION_PASS_NO_PREDICTION_ACCEPTED', flush=True)


if __name__ == '__main__':
    try: run(*sys.argv[1:])
    except BaseException:
        dest = REC / 'worker_failures' / str(time.time_ns())
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'traceback.txt').write_text(traceback.format_exc())
        raise
