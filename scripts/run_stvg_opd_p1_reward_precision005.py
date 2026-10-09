"""Pinned reward-audit-only extension; qualify a saved complete fit, then resume."""
import gc, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, status, sha, activate
from scripts.run_stvg_opd_p1_allocator_recovery004 import verify as verify004, ALLOCATOR, formal_frame
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal, caller_metadata
from scripts.decota_matrix_common_v1 import load, save
REC = BASE/'recovery/P1_reward_precision_005'


def verify():
    verify004(); r = read(REC/'REVISION_RUNTIME.json')
    for f, h in r['pins'].items(): assert sha(ROOT/f) == h, f
    assert sha(REC/'CAPTURE_RECEIPT.json') == r['capture_sha256']
    assert sha(REC/'FAILED_FIT.pt') == r['failed_fit_sha256']
    assert sha(BASE/'RUNTIME_LOCK.json') == r['original_runtime_sha256']
    cap = read(REC/'CAPTURE_RECEIPT.json')
    for f, h in cap['originals'].items():
        assert sha(REC/'originals'/f) == h['sha256']
        if Path(f).name not in ('STATUS.json', 'STAGE.json', 'continuation_GPU_P1_vidstg.log'):
            assert sha(ROOT/f) == h['sha256'], f
    assert read(REC/'CPU_CONTRACTS.json')['status'] == 'pass'
    return r


def receipt_audit(fit, expert):
    frame = sys._getframe(1)
    assert frame.f_code.co_filename.endswith('score_stvg_opd_p1_v1.py')
    revision = frame.f_locals['z']['math_audit'].get('revision')
    from vg_tta.decota_spatial_opd_precision_audit_revision001 import audit as a1, original_audit
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as a2
    from vg_tta.decota_spatial_opd_chart_revision003 import audit as a3, REVISION as rev3
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit as a5, REVISION as rev5
    mapping = {None: original_audit, 'precision_audit_revision001': a1,
        'precision_audit_revision002': a2, rev3: a3, rev5: a5}
    assert revision in mapping, revision
    return mapping[revision](fit, expert)


def run(job, *args):
    verify(); activate()
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit as audit005
    if job == 'P1_score':
        assert read(BASE/'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
        from scripts.run_stvg_opd_gradient_recovery_001 import activate_revision
        activate_revision()
        import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
        target.audit = receipt_audit
        from scripts.run_stvg_opd_paper_hc2_revision_v2 import run as dispatch
        return dispatch(job, *args)
    import scripts.run_stvg_opd_p1_chart_revision003 as old
    if job not in ('GPU', 'GPU_requal'): return old.run(job, *args)
    assert args == ('P1_vidstg',) and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == ALLOCATOR
    assert not (BASE/'P1_PREDICTION_BARRIER.json').exists()
    qualification = job == 'GPU_requal'
    if qualification: assert not (REC/'GPU_QUALIFICATION.json').exists()
    else:
        assert read(REC/'GPU_QUALIFICATION.json')['status'] == 'pass'
        assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    import vg_tta.decota_spatial_opd_chart_revision003 as chart
    prior = chart.audit; active = False; seen = False
    original_failed = load(REC/'FAILED_FIT.pt')
    cap = read(REC/'CAPTURE_RECEIPT.json')

    def wrapper(fit, expert):
        nonlocal active, seen
        checked = audit005(fit, expert)
        frame = formal_frame()
        if frame is None or active: return checked
        meta = caller_metadata(frame)
        wanted = (cap['first_missing_order'], cap['first_missing_arrival'], cap['first_missing_query_ordinal'])
        if (meta['order'], meta['arrival'], meta['query_ordinal']) != wanted:
            assert not qualification, 'Stop qualification at exact first missing arrival'
            return checked
        seen = True; active = True
        try:
            assert meta['input']['new_DINO_calls'] == 0
            assert meta['input']['sha256'] == original_failed['input']['sha256']
            assert meta['previous_payload_sha256'] == original_failed['previous_payload_sha256']
            actual_failed = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
            equal(fit, original_failed['fit'], actual_failed)
            equal(detached(expert, 'cpu'), original_failed['expert'], dict(tensors=0, tensor_coordinates=0, scalar_values=0))
            assert checked['revision'] == 'detached_reward_precision_revision005'
            if qualification:
                loc = frame.f_locals
                torch.cuda.synchronize(); start = time.perf_counter()
                repeated = loc['fit'](loc['base'], loc['initial'], expert, loc['row']['frame_ids'],
                    loc['row']['key'], loc['arm'], config=loc['cfg'])
                torch.cuda.synchronize(); seconds = time.perf_counter()-start
                comparison = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                equal(fit, repeated, comparison); assert audit005(repeated, expert) == checked
                installation = sys._getframe(1).f_locals['installation']
                controls = old.ordinary_controls(frame, installation)
                save(REC/'QUALIFIED_FIRST_MISSING_FIT.pt', dict(fit=fit, expert=detached(expert, 'cpu'), independent_math=checked, **meta))
                write(REC/'GPU_QUALIFICATION.json', dict(status='pass',
                    scope='detached reward audit precision only; unchanged actual GPU fitting', **meta,
                    actual_complete_fits=6, saved_failure_repeat_fits=2, ordinary_control_fits=4,
                    complete_fit_equals_actual_failed_process_serialization=actual_failed,
                    repeated_full_fit_bitwise=comparison, ordinary_controls=controls,
                    independent_math=checked, original_actual_failed_fit_sha256=sha(REC/'FAILED_FIT.pt'),
                    qualified_fit_sha256=sha(REC/'QUALIFIED_FIRST_MISSING_FIT.pt'),
                    first_fit_GPU_seconds=loc['seconds'], repeated_fit_GPU_seconds=seconds,
                    original_failure_serialized_before_exit=True, accepted_predictions=0,
                    fitter_loss_reward_sampling_gradient_Adam_LN_unchanged=True,
                    allocator=ALLOCATOR, runtime_sha256=sha(REC/'REVISION_RUNTIME.json'), time=time.time()))
                raise old.QualificationComplete()
            expected = load(REC/'QUALIFIED_FIRST_MISSING_FIT.pt')
            comparison = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
            equal(fit, expected['fit'], comparison); assert checked == expected['independent_math']
            if not (REC/'FIRST_FORMAL_FIT_BITWISE.json').exists():
                write(REC/'FIRST_FORMAL_FIT_BITWISE.json', dict(status='pass', **meta,
                    qualified_full_fit_bitwise=comparison, original_actual_failed_fit_bitwise=actual_failed,
                    qualified_fit_sha256=sha(REC/'QUALIFIED_FIRST_MISSING_FIT.pt'),
                    runtime_sha256=sha(REC/'REVISION_RUNTIME.json'), GT_scoring=False,
                    next_operation='write only first missing formal prediction', time=time.time()))
            return checked
        finally: active = False

    chart.audit = wrapper
    try:
        old.run('GPU', *args)
        assert seen
        if qualification: print('ACTUAL_REWARD_AUDIT_REVISION005_6_GPU_FITS_PASS_ZERO_ACCEPTED', flush=True)
    finally:
        chart.audit = prior; gc.collect()


if __name__ == '__main__':
    try: run(*sys.argv[1:])
    except BaseException:
        folder = REC/'worker_failures'/str(time.time_ns()); folder.mkdir(parents=True, exist_ok=True)
        (folder/'traceback.txt').write_text(traceback.format_exc())
        raise
