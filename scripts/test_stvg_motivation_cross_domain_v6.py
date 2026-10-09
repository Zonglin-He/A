"""CPU contracts for real metric distinctions, pairing and source/GT barriers."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import stvg_motivation_cross_domain_common_v6 as c
from vg_tta.stvg_motivation_quadrants_v6 import quadrant, independent_components, summarize_models


def run():
    checks = []
    gt = {i: [0, 0, 10, 10] for i in range(10)}
    boxes = [[0, 0, 1, 1], [0, 0, 1, 1]]
    # Correct spatial identity survives a wrong predicted event: the axes must
    # not substitute vIoU for GT-fixed-frame sIoU.
    z = independent_components(boxes, [0, 9], [20, 30], gt, [0, 10], 10, 10)
    assert z['tIoU'] == 0 and z['sIoU'] == 1
    assert quadrant([z['tIoU']], [z['sIoU']]).tolist() == [2]; checks.append('spatial_correct_wrong_moment')
    z = independent_components([[2, 2, 3, 3], [2, 2, 3, 3]], [0, 9], [0, 10], gt, [0, 10], 10, 10)
    assert z['tIoU'] == 1 and z['sIoU'] == 0; checks.append('temporal_correct_wrong_referent')
    z = independent_components(boxes, [2, 7], [0, 10], gt, [0, 10], 10, 10)
    assert z['sIoU'] == .6 and z['uncovered_GT_frames'] == 4; checks.append('missing_support_full_denominator')
    z = independent_components([[0, 0, 0, 0], [0, 0, 0, 0]], [0, 9], [0, 10], gt, [0, 10], 10, 10)
    assert z['sIoU'] == 0 and z['invalid_GT_box_frames'] == 10; checks.append('degenerate_geometry_zero_without_drop')
    assert quadrant([.5, .50001, 1., 0], [1., .5, 1., 0]).tolist() == [2, 1, 0, 3]; checks.append('strict_threshold_and_all_four_categories')
    z = independent_components(boxes, [0, 9], None, gt, [0, 10], 10, 10, temporal_valid=False)
    assert z['tIoU'] == 0 and z['sIoU'] == 1; checks.append('invalid_temporal_stays_separate')
    z = independent_components([], [], [0, 10], gt, [0, 10], 10, 10, spatial_valid=False)
    assert z['sIoU'] == 0 and z['tIoU'] == 1; checks.append('missing_spatial_not_removed')
    z = independent_components(boxes, [0, 9], [10, 20], gt, [0, 10], 10, 10)
    assert z['tIoU'] == 0; checks.append('half_open_boundary_no_overlap')
    values = {'A': ([0, 1, 1, 0], [1, 0, 1, 0]), 'B': ([0, 1, 1, 0], [1, 0, 1, 0])}
    r = summarize_models(values)
    assert r['models']['A']['counts'] == [1, 1, 1, 1]
    assert np.array(r['paired_between_backbones']['A minus B']['ci95_paired_parent_bootstrap']).max() == 0
    assert r == summarize_models(values); checks.append('10000_bootstrap_exact_pairing_repeatability')
    rejected = 0
    for args in [([np.nan], [0]), ([0], [1.2]), ([], [])]:
        try: quadrant(*args)
        except AssertionError: rejected += 1
    assert rejected == 3; checks.append('nonfinite_out_of_range_and_empty_rejected')
    for path in ['/annos/train.json', '/annotations/test.json', '/data/val_v2.json',
            '/x/labels_diagnostic_only.json', str(c.P1/'predictions/00000.npz')]:
        try: c.guard('open', (path,))
        except PermissionError: rejected += 1
        else: raise AssertionError(path)
    checks.append('prediction_process_annotation_and_OPD_payload_denial')
    # Source binding and qualification rejection use the actual seal checker
    # on synthetic files, with no model/video/GT access.
    actual_base = c.BASE
    with tempfile.TemporaryDirectory(prefix='cross_native_contract_') as folder:
        c.BASE = Path(folder)
        try:
            c.write(c.BASE/'RUNTIME_LOCK.json', {'synthetic': True})
            rh = c.sha(c.BASE/'RUNTIME_LOCK.json')
            d, m = c.DIRECTIONS[0], c.MODELS[0]
            row = dict(ordinal=0, frame_ids=[0, 9], input={'fixture': 'sanitized'}, key='k0', source='p0')
            cohort = [{**row, 'ordinal': i, 'key': f'k{i}', 'source': f'p{i}'} for i in range(128)]
            c.write(c.BASE/d/'ROSTER.json', dict(rows=cohort))
            p = c.BASE/d/m/'predictions/00000.json'
            z = dict(model=m, direction=d, ordinal=0, frame_ids=[0, 9], box_frame_ids=[0, 9],
                input_binding_sha256=c.digest(row['input']), parameter_updates=0, GT_read=False, native_parity=True,
                source_dataset='vidstg', target_dataset='hc2', checkpoint_sha256=c.CHECKPOINTS[m]['vidstg'][1],
                runtime_lock_sha256=rh, qualification=False, native_final_layer_only=True,
                key_sha256=c.digest('k0'), parent_sha256=c.digest('p0'), pixel_sha256='fixturepixels')
            def save(value):
                c.status(p, value)
                c.status(p.with_suffix('.receipt.json'), dict(sha256=c.sha(p), bytes=p.stat().st_size,
                    runtime_lock_sha256=rh, GT_read=False, qualification=False))
            c.write(c.BASE/d/'common_pixels/00000.json', dict(pixel_sha256='fixturepixels', frame_ids=[0, 9]))
            for r in cohort[1:]:
                other = c.BASE/d/m/'predictions'/f"{r['ordinal']:05}.json"
                zz = {**z, 'ordinal': r['ordinal'], 'key_sha256': c.digest(r['key']),
                    'parent_sha256': c.digest(r['source'])}
                c.write(other, zz)
                c.write(other.with_suffix('.receipt.json'), dict(sha256=c.sha(other), bytes=other.stat().st_size,
                    runtime_lock_sha256=rh, GT_read=False, qualification=False))
                c.write(c.BASE/d/'common_pixels'/f"{r['ordinal']:05}.json", dict(pixel_sha256='fixturepixels', frame_ids=[0, 9]))
            save(z)
            assert len(c.seal_cell(m, d)) == 128
            checks.append('complete_valid_cross_source_formal_seal')
            for field, bad in [('source_dataset', 'hc2'), ('target_dataset', 'vidstg'), ('qualification', True),
                    ('parameter_updates', 1), ('GT_read', True), ('native_final_layer_only', False),
                    ('pixel_sha256', 'different_pixels'), ('checkpoint_sha256', 'wrong_checkpoint')]:
                save({**z, field: bad})
                try: c.seal_cell(m, d)
                except AssertionError: rejected += 1
                else: raise AssertionError(field)
            checks.append('wrong_source_target_qualification_update_GT_layer_pixel_checkpoint_rejected')
            save(z)
            c.status(c.BASE/d/'ROSTER.json', dict(rows=[row]))
            try: c.seal_cell(m, d)
            except AssertionError: rejected += 1
            else: raise AssertionError('N=1 must not pass the fixed128-cell seal')
            checks.append('incomplete_formal_roster_rejected')
        finally:
            c.BASE = actual_base
    result = dict(status='pass', CPU_contracts=len(checks), rejected_invalid_cases=rejected,
        contracts=checks, GPU_qualification=False, model_forwards=0, videos_opened=0,
        GT_read=False, time=time.time())
    if len(sys.argv) > 1 and sys.argv[1] == 'record':
        c.verify()
        result['runtime_lock_sha256'] = c.sha(c.BASE/'RUNTIME_LOCK.json')
        c.write(c.BASE/'CPU_CONTRACTS.json', result)
    print(result)


if __name__ == '__main__':
    run()
