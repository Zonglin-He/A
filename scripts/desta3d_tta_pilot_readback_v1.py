"""CPU, GT-free readback of the sealed four-query E5 engineering pilot."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/desta3d_v1/tta_run'
RUN = 'dualE1_4source_noise_1step_v1_rev009'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prediction(p, n):
    boxes = p['boxes_cxcywh']
    mask = p['geometry_valid']
    positions = p['positions']
    assert isinstance(boxes, torch.Tensor) and boxes.ndim == 2 and boxes.shape[1] == 4
    assert tuple(mask.shape) == (len(positions),) == (len(boxes),)
    assert mask.dtype == torch.bool and torch.isfinite(boxes).all()
    assert len(set(positions)) == len(positions)
    assert all(isinstance(x, int) and 0 <= x < n for x in positions)
    interval = p['interval']
    interval_valid = interval is not None and len(interval) == 2 and 0 <= interval[0] <= interval[1] < n
    if p['format_ok']:
        assert interval_valid
    assert p['merger_hook_calls'] == 1 and p['GT_used'] is False
    return {'format_ok': p['format_ok'], 'interval': interval,
            'interval_valid': interval_valid, 'sparse_box_count': len(boxes),
            'geometry_valid_count': int(mask.sum()), 'box_shape': list(boxes.shape),
            'all_box_values_finite': True}


def main():
    torch.set_num_threads(2)
    reg = read(ART / 'pilot_v1/PILOT_REV009_REGISTRATION.json')
    inputs = read(ART / 'pilot_v1/INPUTS.json')
    expected = {r['key']: r for r in inputs}
    assert len(expected) == 4 and len({r['source'] for r in inputs}) == 4
    for p, h in reg['pins'].items():
        assert sha(p) == h, p
    run_dir = ART / 'runs' / RUN
    context = read(run_dir / 'RUN_CONTEXT.json')
    seal = read(run_dir / 'PREDICTION_SEAL.json')
    assert seal['status'] == 'complete_gt_free_e5_tta_predictions'
    assert seal['run_context_sha256'] == context['context_sha256']
    assert seal['count'] == len(seal['files']) == 4
    assert {r['key'] for r in seal['files']} == set(expected)
    assert context['config']['steps'] == 1 and context['config']['mode'] == 'dual_nojoint'
    assert context['condition'] == 'noise_medium' and context['split'] == 'validation'
    assert seal['target_GT_read'] is False
    # All hashes are checked before any torch record is loaded. No labels loaded.
    for f in seal['files']:
        assert sha(f['path']) == f['prediction_sha256']
        side = read(str(f['path']) + '.json')
        assert side['prediction_sha256'] == f['prediction_sha256']
        assert side['context_sha256'] == context['context_sha256']
    rows = []
    for f in seal['files']:
        d = torch.load(f['path'], map_location='cpu', weights_only=False)
        e = expected[d['key']]
        assert d['source'] == e['source'] and d['frame_ids'] == e['input']['frame_ids']
        assert d['fps'] == e['input']['fps']
        assert d['video_grid_thw'][0][0] == len(d['frame_ids'])
        assert d['GT_used'] is False and d['target_GT_read'] is False
        initial = reg['source_checkpoint']['adapter_sha256']
        assert d['source_fit_adapter_sha256'] == d['adapter_sha256_before_tta'] == initial
        assert d['adapter_sha256_after_tta'] != initial
        assert d['source_stats_sha256'] == reg['source_statistics_sha256']
        assert d['source_stats_parent_count'] == 95
        assert d['update_groups'] == ['input_proj', 'stem', 'readers']
        assert len(d['updates']) == 1
        update = d['updates'][0]
        assert update['step'] == 1 and update['frozen_heads_no_grad'] is True
        assert update['adapter_state_sha256_after_update'] == d['adapter_sha256_after_tta']
        assert math.isfinite(update['total_loss_before_update'])
        assert all(math.isfinite(x) for x in update['components_before_update'].values())
        assert all(math.isfinite(x) and x > 0 for x in update['gradient_l2_by_group'].values())
        assert update['active_weights']['asymmetric_joint'] == 0
        base, mild = d['views']['base'], d['views']['mild']
        assert base['pixel_sha256'] != mild['pixel_sha256']
        assert base['preprocess']['grid'] == mild['preprocess']['grid'] == d['video_grid_thw']
        b = d['baseline_source_fit_frozen_no_tta']
        a = d['adapted_fixed_terminal']
        rows.append({'key': d['key'], 'source': d['source'], 'sampled_frames': len(d['frame_ids']),
                     'baseline': prediction(b, len(d['frame_ids'])),
                     'adapted': prediction(a, len(d['frame_ids'])),
                     'gradients': update['gradient_l2_by_group'],
                     'loss_components': update['components_before_update'],
                     'visual_injection_changed': b['updated_tokens_sha256'] != a['updated_tokens_sha256'],
                     'interval_changed': b['interval'] != a['interval'],
                     'output_boxes_changed': b['positions'] != a['positions'] or not torch.equal(b['boxes_cxcywh'], a['boxes_cxcywh'])})
    receipts = [read(p) for p in (ART / 'receipts').glob('*.json') if read(p).get('run_id') == RUN]
    assert len(receipts) == 1 and receipts[0]['status'] == 'completed'
    receipt = receipts[0]
    result = {'time_utc': datetime.now(timezone.utc).isoformat(), 'status': 'engineering_interface_passed',
              'script_sha256': sha(__file__), 'seal_sha256': sha(run_dir / 'PREDICTION_SEAL.json'),
              'queries': 4, 'parents': 4, 'updates': 4, 'PTD_prediction_pairs': 4,
              'rows': rows, 'GT_read': False, 'GPU_started_by_audit': False,
              'actual_retry_worker_seconds': receipt['seconds'],
              'prior_includes_failed_rev008': receipt['prior_seconds'],
              'cumulative_actual_GPU_seconds': receipt['prior_seconds'] + receipt['seconds'],
              'scope': 'Real GPU fixed-step gradients/reset/injection/full-decode sealed results, CPU structure/hash audit. No GT scores or efficacy gate.',
              'freeze_evidence': 'Runner asserts all backbone requires_grad false, frozen local heads have no gradients, exact source/terminal adapter digests preserved during respective replays; no independent post-run 4B tensor rehash.',
              'format_failures_retained': True}
    out = ART / 'pilot_v1/ROOT_PILOT_REV009_READBACK.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
