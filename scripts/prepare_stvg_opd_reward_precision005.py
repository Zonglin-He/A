"""Capture the new saved IoU audit failure and the complete opaque P1 prefix."""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_allocator_recovery004 import verify as verify004
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
REC = BASE/'recovery/P1_reward_precision_005'
FAILED = BASE/'recovery/P1_box_chart_003/authorized_revision003/new_failures/1791526309608256157'


def run():
    verify004(); REC.mkdir(parents=True, exist_ok=True)
    assert not (REC/'CAPTURE_RECEIPT.json').exists()
    meta = read(FAILED/'FAILURE_RECEIPT.json')
    assert meta['query_ordinal'] == 5250 and meta['arrival'] == 5767 and meta['order'] == 'order2'
    assert sha(FAILED/'FAILED_FIT.pt') == meta['failed_fit_sha256']
    for pid in (2768904, 2768962, 2765898): assert not Path('/proc', str(pid)).exists()
    progress = read(BASE/'stages/P1_vidstg/STATUS.json'); assert progress['done'] == 16070 and not progress['GT_read']
    design = read(BASE/'DESIGN_LOCK.json')['stages']['P1_vidstg']
    assert design['orders']['order2'][5767] == 5250
    files = {}; total = 0
    for order in ('order1', 'order2', 'order3'):
        paths = sorted((BASE/f'stages/P1_vidstg/clean/{order}/on_policy').glob('*.pt'))
        assert [int(p.stem) for p in paths] == list(range(len(paths)))
        for p in paths:
            rc = p.with_suffix('.json'); receipt = read(rc); digest = sha(p)
            assert receipt['sha256'] == digest and receipt['bytes'] == p.stat().st_size and not receipt['GT_read']
            files[str(p.relative_to(ROOT))] = dict(sha256=digest, bytes=p.stat().st_size, receipt_sha256=sha(rc))
            total += p.stat().st_size
    assert len(files) == 16070
    originals = {}
    rt004 = read(BASE/'recovery/P1_cuda_memory_004/REVISION_RUNTIME.json')
    selected = list(rt004['pins']) + ['vg_tta/decota_spatial_opd_precision_audit_revision002.py',
        'vg_tta/decota_spatial_opd_tunable_audit_v1.py', 'vg_tta/tastvg_decota_critic_p0_v1.py',
        'methods/decota_final_simplified_v1/objectives.py', str((BASE/'STATUS.json').relative_to(ROOT)),
        str((BASE/'STAGE.json').relative_to(ROOT)), str((BASE/'stages/P1_vidstg/STATUS.json').relative_to(ROOT)),
        str((BASE/'recovery/P1_cuda_memory_004/continuation_GPU_P1_vidstg.log').relative_to(ROOT)),
        str((BASE/'recovery/P1_cuda_memory_004/REVISION_RUNTIME.json').relative_to(ROOT)),
        'artifacts/stvg_motivation_cross_domain_v6/STATUS.json',
        'artifacts/stvg_motivation_cross_domain_v6/failures/1791526352329619770/traceback.txt']
    for rel in list(dict.fromkeys(selected)):
        p = ROOT/rel; d = REC/'originals'/rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, d)
        assert sha(p) == sha(d); originals[rel] = dict(sha256=sha(p), bytes=p.stat().st_size)
    for name in ('FAILED_FIT.pt', 'FAILURE_RECEIPT.json', 'traceback.txt'): shutil.copy2(FAILED/name, REC/name)
    write(REC/'CAPTURE_RECEIPT.json', dict(status='complete_failed_fit_and_opaque_prefix_preserved',
        scope='no-GT detached reward precision diagnosis and engineering-only repair',
        done=16070, total=30909, prefix_files=files, prefix_bytes=total, originals=originals,
        first_missing_order='order2', first_missing_arrival=5767, first_missing_query_ordinal=5250,
        actual_failed_fit_serialized_before_exit=True, failed_fit_sha256=sha(REC/'FAILED_FIT.pt'),
        original_failure_receipt_sha256=sha(REC/'FAILURE_RECEIPT.json'),
        original_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'), allocator_runtime_sha256=sha(BASE/'recovery/P1_cuda_memory_004/REVISION_RUNTIME.json'),
        HC2_barrier_sha256=sha(BASE/'stages/P1_hc2/PREDICTION_BARRIER.json'), HC2_sealed=10446,
        figure_queue_failure_is_P1_dependency_only=True, no_figure_GPU_or_GT_started=True,
        P1_global_seal_present=False, GT_read=False, predictions_or_science_changed=False, time=time.time()))
    print(json.dumps(dict(status='capture_complete', prefix=len(files), bytes=total, failed_fit_saved=True)))


if __name__ == '__main__': run()
