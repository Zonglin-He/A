"""GT-free structural readback and all-condition seal for fixed E5 target64."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_tta_pilot_readback_v1 import prediction, read, sha

ART = ROOT / 'artifacts/desta3d_v1/tta_run'
TARGET = ART / 'target64_v1'
OLD = ROOT / 'artifacts/ptd_corruption_coupling_v1'


def write_once(p, data):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def audit_condition(condition):
    reg = read(TARGET / 'REGISTRATION.json')
    assert condition in reg['conditions']
    for p, h in reg['pins'].items():
        assert sha(p) == h, p
    inputs = read(TARGET / 'INPUTS.json')
    expected = {r['key']: r for r in inputs}
    old_inputs = {r['key']: r for r in read(OLD / 'INPUTS.json')}
    assert len(expected) == len({r['source'] for r in inputs}) == 64
    run_id = reg['run_ids'][condition]
    run_dir = ART / 'runs' / run_id
    context = read(run_dir / 'RUN_CONTEXT.json')
    seal_path = run_dir / 'PREDICTION_SEAL.json'
    seal = read(seal_path)
    assert seal['status'] == 'complete_gt_free_e5_tta_predictions'
    assert seal['run_context_sha256'] == context['context_sha256']
    assert seal['count'] == len(seal['files']) == 64
    assert {f['key'] for f in seal['files']} == set(expected)
    assert seal['target_GT_read'] is False
    assert context['condition'] == condition and context['split'] == 'development'
    assert context['config'] == reg['config']
    # Validate every new prediction and historical Frozen record before torch reads.
    barrier = read(OLD / 'CAPTURE_DEVELOPMENT_BARRIER.json')
    frozen_paths = {}
    for f in seal['files']:
        assert sha(f['path']) == f['prediction_sha256']
        side = read(str(f['path']) + '.json')
        assert side['prediction_sha256'] == f['prediction_sha256']
        assert side['context_sha256'] == context['context_sha256']
        p = OLD / 'capture' / condition / (hashlib.sha256(f['key'].encode()).hexdigest() + '_F.pt')
        assert sha(p) == barrier['files'][str(p)], p
        frozen_paths[f['key']] = p
    rows = []
    initial = 'e9b46983658b75b605cee7687575cf9d21f8dbf21b8b0029da00a7c0e30bcf33'
    stats_path = ART / 'source_stats/dual3d_618q_88d458dd9173.json'
    for f in seal['files']:
        d = torch.load(f['path'], map_location='cpu', weights_only=False)
        e = expected[d['key']]
        o = old_inputs[d['key']]
        z = torch.load(frozen_paths[d['key']], map_location='cpu', weights_only=False)['z']
        assert set(o['input']) - set(e['input']) == {'duration'}
        assert all(value == o['input'][key] for key, value in e['input'].items())
        assert d['source'] == e['source'] and d['cohort'] == e['cohort']
        assert d['condition'] == condition and d['split'] == 'development'
        assert d['frame_ids'] == e['input']['frame_ids'] == z['frame_ids']
        assert d['fps'] == e['input']['fps']
        assert d['video_grid_thw'][0][0] == len(d['frame_ids']) == 32
        assert d['GT_used'] is False and d['target_GT_read'] is False
        assert d['source_fit_adapter_sha256'] == d['adapter_sha256_before_tta'] == initial
        assert d['adapter_sha256_after_tta'] != initial
        assert d['source_stats_sha256'] == sha(stats_path)
        assert d['source_stats_parent_count'] == 95
        assert d['update_groups'] == ['input_proj', 'stem', 'readers']
        assert len(d['updates']) == 1
        u = d['updates'][0]
        assert u['step'] == 1 and u['frozen_heads_no_grad'] is True
        assert u['adapter_state_sha256_after_update'] == d['adapter_sha256_after_tta']
        assert math.isfinite(u['total_loss_before_update'])
        assert all(math.isfinite(x) for x in u['components_before_update'].values())
        assert all(math.isfinite(x) and x > 0 for x in u['gradient_l2_by_group'].values())
        assert u['active_weights']['asymmetric_joint'] == 0
        base, mild = d['views']['base'], d['views']['mild']
        assert base['pixel_sha256'] != mild['pixel_sha256']
        assert base['preprocess']['grid'] == mild['preprocess']['grid'] == d['video_grid_thw']
        assert base['preprocess'] == z['preprocess'], (d['key'], condition, 'native input mismatch')
        assert base['pixel_sha256'] == z['preprocess']['pixel_sha']
        b, a = d['baseline_source_fit_frozen_no_tta'], d['adapted_fixed_terminal']
        rows.append({'key': d['key'], 'source': d['source'], 'domain': o['domain'],
                     'condition': condition, 'baseline': prediction(b, 32), 'adapted': prediction(a, 32),
                     'native_Frozen_input_exact': True, 'gradients': u['gradient_l2_by_group'],
                     'loss_components': u['components_before_update'],
                     'visual_injection_changed': b['updated_tokens_sha256'] != a['updated_tokens_sha256'],
                     'interval_changed': b['interval'] != a['interval'],
                     'output_boxes_changed': b['positions'] != a['positions'] or not torch.equal(b['boxes_cxcywh'], a['boxes_cxcywh'])})
    receipts = [read(p) for p in (ART / 'receipts').glob('*.json') if read(p).get('run_id') == run_id]
    assert len(receipts) == 1 and receipts[0]['status'] == 'completed'
    receipt = receipts[0]
    result = {'time': time.time(), 'status': 'condition_structural_readback_passed',
              'condition': condition, 'script_sha256': sha(__file__),
              'seal_path': str(seal_path), 'seal_sha256': sha(seal_path),
              'queries': 64, 'parents': 64, 'updates': 64, 'prediction_pairs': 64,
              'baseline_format_ok': sum(r['baseline']['format_ok'] for r in rows),
              'adapted_format_ok': sum(r['adapted']['format_ok'] for r in rows),
              'interval_changes': sum(r['interval_changed'] for r in rows),
              'box_changes': sum(r['output_boxes_changed'] for r in rows),
              'actual_worker_seconds': receipt['seconds'],
              'cumulative_actual_GPU_seconds': receipt['prior_seconds'] + receipt['seconds'],
              'GT_read': False, 'GPU_started_by_audit': False,
              'freeze_scope': 'runtime frozen assertions and adapter digests; no independent post-run all-4B tensor rehash',
              'rows': rows}
    write_once(TARGET / ('STRUCTURE_' + condition + '.json'), result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


def seal_all():
    reg = read(TARGET / 'REGISTRATION.json')
    files = {}
    for c in reg['conditions']:
        p = TARGET / ('STRUCTURE_' + c + '.json')
        s = read(p)
        assert s['status'] == 'condition_structural_readback_passed' and s['GT_read'] is False
        assert sha(s['seal_path']) == s['seal_sha256']
        files[str(p)] = sha(p)
        files[s['seal_path']] = s['seal_sha256']
        for f in read(s['seal_path'])['files']:
            assert sha(f['path']) == f['prediction_sha256']
            files[f['path']] = f['prediction_sha256']
    write_once(TARGET / 'ALL_192_PAIRS_BARRIER.json', {
        'status':'all_192_pairs_structurally_audited_before_target_GT_read',
        'time':time.time(), 'pairs':192, 'individual_predictions':384,
        'conditions':reg['conditions'], 'files':files, 'GT_read':False,
        'registration_sha256':sha(TARGET / 'REGISTRATION.json'),
        'audit_script_sha256':sha(__file__)})
    print('ALL_192_PAIRS_BARRIER written, no GT read')


if __name__ == '__main__':
    torch.set_num_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['condition', 'seal-all'])
    p.add_argument('--condition')
    args = p.parse_args()
    audit_condition(args.condition) if args.action == 'condition' else seal_all()
