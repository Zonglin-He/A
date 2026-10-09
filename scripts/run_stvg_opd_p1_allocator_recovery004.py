"""Allocator-only P1 recovery. Original chart/fit/audit code remains immutable."""
import gc
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_chart_revision003 import verify_revision as verify003
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, status, sha
from scripts.decota_matrix_common_v1 import save, load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal, caller_metadata
REC = BASE/'recovery/P1_cuda_memory_004'
ALLOCATOR = 'expandable_segments:True'


def verify():
    verify003()
    r = read(REC/'REVISION_RUNTIME.json')
    for f, h in r['pins'].items():
        assert sha(ROOT/f) == h, f
    cap = read(REC/'CAPTURE_RECEIPT.json')
    assert sha(REC/'CAPTURE_RECEIPT.json') == r['capture_sha256']
    for rel, value in cap['originals'].items():
        assert sha(REC/'originals'/rel) == value['sha256']
        if Path(rel).name not in ['STATUS.json', 'STAGE.json', 'continuation_GPU_P1_vidstg.log']:
            assert sha(ROOT/rel) == value['sha256'], rel
    assert sha(BASE/'RUNTIME_LOCK.json') == r['original_runtime_sha256']
    return r


def formal_frame():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.endswith('/scripts/run_stvg_opd_paper_v1.py') and 'result' in frame.f_locals:
            return frame
        frame = frame.f_back
    return None


def run(job, *args):
    r = verify()
    if job not in ('GPU', 'GPU_requal'):
        from scripts.run_stvg_opd_p1_chart_revision003 import run as old
        return old(job, *args)
    assert args == ('P1_vidstg',)
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == ALLOCATOR
    assert not (BASE/'P1_PREDICTION_BARRIER.json').exists()
    qualification = job == 'GPU_requal'
    if qualification:
        assert not (REC/'GPU_QUALIFICATION.json').exists()
    else:
        assert read(REC/'GPU_QUALIFICATION.json')['status'] == 'pass'
    import torch
    import vg_tta.decota_spatial_opd_chart_revision003 as chart
    import scripts.run_stvg_opd_p1_chart_revision003 as old
    original_audit = chart.audit
    active = False
    seen = False
    cap = read(REC/'CAPTURE_RECEIPT.json')

    def wrapper(fit, expert):
        nonlocal active, seen
        checked = original_audit(fit, expert)
        frame = formal_frame()
        if frame is None or active:
            return checked
        meta = caller_metadata(frame)
        wanted = (cap['first_missing_order'], cap['first_missing_arrival'], cap['first_missing_query_ordinal'])
        if (meta['order'], meta['arrival'], meta['query_ordinal']) != wanted:
            assert not qualification, 'Qualification stops at the first missing arrival'
            return checked
        active = True; seen = True
        try:
            assert meta['input']['new_DINO_calls'] == 0
            loc = frame.f_locals
            if qualification:
                # Two actual repeats under the same allocator establish
                # repeatability. The failed native capture had no serialized
                # output and is never claimed to be bitwise compared.
                torch.cuda.synchronize(); begin = time.perf_counter()
                repeated = loc['fit'](loc['base'], loc['initial'], expert,
                    loc['row']['frame_ids'], loc['row']['key'], loc['arm'], config=loc['cfg'])
                torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                comparison = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                equal(fit, repeated, comparison)
                assert original_audit(repeated, expert) == checked
                installed = sys._getframe(1).f_locals['installation']
                assert installed.core.fit == installed.fit
                controls = old.ordinary_controls(frame, installed)
                save(REC/'QUALIFIED_FIRST_MISSING_FIT.pt', dict(fit=fit, independent_math=checked, **meta))
                write(REC/'GPU_QUALIFICATION.json', dict(status='pass', scope='allocator-only repair actual qualification',
                    **meta, allocator=ALLOCATOR, failed_query_full_native_capture_completed=True,
                    actual_complete_fits=6, first_missing_fits=2, ordinary_control_fits=4,
                    repeated_full_fit_bitwise=comparison, ordinary_controls=controls,
                    independent_math=checked, repeated_fit_GPU_seconds=seconds,
                    first_fit_GPU_seconds=loc['seconds'], full_input_frame_count=len(loc['row']['frame_ids']),
                    input_frame_ids_sha256=__import__('hashlib').sha256(str(loc['row']['frame_ids']).encode()).hexdigest(),
                    CUDA_peak_allocated=torch.cuda.max_memory_allocated(), CUDA_peak_reserved=torch.cuda.max_memory_reserved(),
                    qualified_fit_sha256=sha(REC/'QUALIFIED_FIRST_MISSING_FIT.pt'),
                    failed_capture_not_serialized=True, equality_with_dead_memory_not_claimed=True,
                    accepted_predictions=0, original_fit_sampling_loss_gradient_Adam_state_unchanged=True,
                    runtime_sha256=sha(REC/'REVISION_RUNTIME.json'), time=time.time()))
                raise old.QualificationComplete()
            expected = load(REC/'QUALIFIED_FIRST_MISSING_FIT.pt')
            assert meta['input']['sha256'] == expected['input']['sha256']
            assert meta['previous_payload_sha256'] == expected['previous_payload_sha256']
            comparison = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
            equal(fit, expected['fit'], comparison)
            assert checked == expected['independent_math']
            if not (REC/'FIRST_FORMAL_FIT_BITWISE.json').exists():
                write(REC/'FIRST_FORMAL_FIT_BITWISE.json', dict(status='pass', **meta,
                    qualified_full_fit_bitwise=comparison, allocator=ALLOCATOR,
                    qualified_fit_sha256=sha(REC/'QUALIFIED_FIRST_MISSING_FIT.pt'),
                    saved_prefix_not_rewritten=True, GT_scoring=False,
                    next_operation='write only first missing formal prediction', time=time.time()))
            return checked
        finally:
            active = False
    chart.audit = wrapper
    try:
        old.run('GPU', *args)
        assert seen, 'Must actually reach the original missing arrival'
    finally:
        chart.audit = original_audit
        gc.collect()


if __name__ == '__main__':
    try:
        run(*sys.argv[1:])
    except BaseException:
        p = REC/'worker_failures'/str(time.time_ns())
        p.mkdir(parents=True, exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc())
        status(REC/'STATUS.json', dict(status='new_failure_preserved', evidence=str(p.relative_to(ROOT)), GT_read=False, time=time.time()))
        raise
