"""Offline, seal-gated scalar/P3 crosschecked scoring for the fixed 8-parent screen."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_tta8_temporal_anchor import SOURCE, ARMS, CONDITIONS, read, sha, NEW_ARM, OUT
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.score_desta3d_tta_target_v1 import metric_boxes, recovered_tokens, stats, crosscheck, synthetic
from scripts.score_ptd_corruption_coupling_v1 import metric as original_metric
from scripts.score_ptd_spatial_adapter_ab_v1 import truth
from vg_tta.desta3d_v2_prediction_contract import validate_prediction

SCORE = OUT / 'independent_readback_v1'
LABEL = ROOT / 'artifacts/ptd_joint_box_opd_v1/LABELS_SCORER_ONLY.json'
LABEL_SHA = 'f9c766be993784b2b5f987cc531477e83ba73a994a249eabc21b6931c0dc4b09'


def preflight():
    assert synthetic()['status'] == 'passed'
    paths = [Path(__file__), ROOT / 'scripts/score_desta3d_tta_target_v1.py',
        ROOT / 'scripts/score_ptd_corruption_coupling_v1.py', ROOT / 'scripts/score_ptd_spatial_adapter_ab_v1.py',
        ROOT / 'vg_tta/ptd_spatial_adapter_ab_v1.py']
    save_once(OUT / 'SCORER_PREFLIGHT.json', {'status': 'passed', 'pins': {str(p): sha(p) for p in paths},
        'controls': synthetic(), 'target_GT_read': False, 'predictions_read': False,
        'metric': 'independent scalar cxcywh geometry vs original vectorized P3/tokens; sampled union and fixed GT-valid spatial support'})


def run():
    torch.set_num_threads(2)
    assert not SCORE.exists()
    complete = read(OUT / 'COMPLETE.json'); assert complete['status'] == 'completed_predictions_unscored'
    assert complete['predictions'] == 192 and complete['seal_sha'] == sha(OUT / 'ALL_PREDICTIONS_SEAL.json')
    seal = read(OUT / 'ALL_PREDICTIONS_SEAL.json')
    assert seal['predictions'] == 192 and seal['episodes'] == 24 and not seal['GT_read']
    pins = {**read(OUT / 'LOCK.json')['pins'], **seal['pins'], **read(OUT / 'SCORER_PREFLIGHT.json')['pins']}
    for p, h in pins.items(): assert sha(Path(p)) == h, p
    assert len(seal['pins']) == 24 * 16
    cfg = read(OUT / 'CONFIG.json'); rows = read(OUT / 'INPUTS.json')
    original = {r['key']: r for r in read(ROOT / 'artifacts/ptd_corruption_coupling_v1/INPUTS.json')}
    payloads = {}; structures = []
    for condition in CONDITIONS:
        for i, row in enumerate(rows):
            e = OUT / 'episodes' / condition / f'{i:02}'
            identity = read(e / 'INPUT_IDENTITY.json')
            assert identity['key'] == row['key'] and identity['source'] == row['source']
            assert identity['condition'] == condition and identity['frame_ids'] == row['input']['frame_ids']
            assert identity['cohort'] == row['cohort'] and identity['source_moments_sha'] == sha(SOURCE / 'MOMENTS.json')
            assert identity['sourcefit_adapter_sha'] == cfg['checkpoint']['adapter_sha256']
            assert read(e / 'EPISODE_COMPLETE.json')['reset_exact']
            replay = read(e / 'REPLAY_INPUT.json')
            assert all(identity[k] == v for k, v in replay.items())
            assert read(e / 'EPISODE_COMPLETE.json')['baseline_replayed_exact']
            old = original[row['key']]
            assert all(old['input'][k] == v for k, v in row['input'].items())
            arm_data = {}
            for arm in ARMS:
                p = torch.load(e / (arm + '.pt'), map_location='cpu', weights_only=False)
                assert p['key'] == row['key'] and p['source'] == row['source'] and not p['GT_read']
                assert p['frame_ids'] == row['input']['frame_ids'] and p['preprocess'] == identity['preprocess']
                assert p.get('video_sha256', p.get('video_sha')) == row['input']['video_sha256']
                validate_prediction(p, len(p['frame_ids']))
                temporal = p['time_distribution']['endpoint_logits']
                assert temporal is None or (temporal.shape == (2, 32) and torch.isfinite(temporal).all())
                if arm == 'sourcefit_noTTA': assert p['adapter_sha'] == cfg['checkpoint']['adapter_sha256']
                if arm in ARMS[2:]:
                    u = p['update']; assert u['steps'] == 3 and u['gates_unchanged']
                    assert u['parameter_count'] == cfg['updated_parameters'][u['interface']]
                    assert p['sourcefit_adapter_sha'] == cfg['checkpoint']['adapter_sha256']
                    assert u['alignment'] == (.01 if 'alignment' in arm else 0.)
                    for j, h in enumerate(u['history'], 1):
                        assert h['step'] == j and h['actual_Adam_steps'] == [j]
                        for name, group in h['gradient_groups'].items():
                            if name not in u['updated_groups']:
                                assert group['zero_tensors'] == group['nonzero_tensors'] == 0
                if arm == NEW_ARM:
                    assert p['update']['output_coefficient'] == cfg['output_coefficient']
                    assert set(p['update']['output_support']) <= {'event'}
                    assert cfg['coordinate_output_coefficient'] == 0.
                    for step in range(1, 4):
                        raw = torch.load(e / f'ANCHOR_STEP_{step}.pt', map_location='cpu', weights_only=False)
                        assert raw['history'] == p['update']['history'][step-1]
                        vectors = raw['raw_gradients']
                        assert 'spatial_weighted' not in vectors
                        assert set(raw['cache_audits']) <= {'event'}
                        assert all(v.shape == (66816,) and torch.isfinite(v).all() for v in vectors.values())
                        total = vectors['pre_gate'].clone()
                        for name in ['event_weighted', 'spatial_weighted']:
                            if name in vectors: total += vectors[name]
                        assert torch.allclose(total, vectors['total_before_clip'], atol=1e-8, rtol=1e-5)
                        for name, v in vectors.items():
                            assert abs(float(v.double().norm()) - raw['history']['gradient_norms'][name]) < 1e-10
                        for cache in raw['cache_audits'].values():
                            assert cache['attention_or_KV_checkpointed'] is False
                            assert cache['injection']['calls'] == 1
                            assert all(x['values_equal'] for x in cache['bias_layout'])
                arm_data[arm] = p
            payloads[(row['key'], condition)] = arm_data
            structures.append({'key': row['key'], 'source': row['source'], 'condition': condition,
                               'format_ok': {a: p['format_ok'] for a, p in arm_data.items()}})
    save_once(SCORE / 'PRE_GT_AUDIT.json', {'status': 'passed', 'time': time.time(),
        'seal_sha': sha(OUT / 'ALL_PREDICTIONS_SEAL.json'), 'predictions': 192, 'queries': 8, 'parents': 8,
        'GT_read': False, 'scorer_sha': sha(Path(__file__)), 'structures': structures,
        'checks': 'all seals; same actual observed pixel/preprocess in Frozen and all B1 paths; registered identities; 3 actual steps; excluded-group gradients; exact-reset runtime assertions'})
    assert sha(LABEL) == LABEL_SHA
    save_once(SCORE / 'OFFLINE_GT_READ_STARTED.json', {'time': time.time(), 'label_sha': LABEL_SHA,
        'barrier_sha': sha(SCORE / 'PRE_GT_AUDIT.json'), 'development_exposed': True,
        'used_for_online_updates_or_state_choice': False})
    labels = read(LABEL)
    records = []; max_error = 0.; diagnostics = []
    for condition in CONDITIONS:
        for row in rows:
            old = original[row['key']]; gt = truth(old, labels)
            predictions = payloads[(row['key'], condition)]
            metrics = {}
            for arm, p in predictions.items():
                m = metric_boxes(p, old, gt)
                check = original_metric(p, recovered_tokens(p), old, gt)
                max_error = max(max_error, crosscheck(m, check))
                metrics[arm] = m
            good = gt['valid'] & (np.asarray(metrics['Frozen']['ious']) >= .5)
            records.append({'key': row['key'], 'source': row['source'], 'domain': old['domain'], 'condition': condition,
                'arms': metrics, 'native_good_frames': int(good.sum()),
                'native_good_retained': {a: int((good & (np.asarray(m['ious']) >= .5)).sum()) for a, m in metrics.items()}})
            base = predictions['sourcefit_noTTA']
            for arm in ARMS[2:]:
                p = predictions[arm]; b = base['time_distribution']['endpoint_logits']; a = p['time_distribution']['endpoint_logits']
                time_kl = None
                if b is not None and a is not None:
                    bp, ap = b.double().log_softmax(-1), a.double().log_softmax(-1)
                    time_kl = float((bp.exp() * (bp-ap)).sum(-1).mean())
                fixed = p['fixed_prefix_control']; kl = None; coord_error = None
                if fixed['defined']:
                    bp = base['readout']['coordinate_logits'].double().log_softmax(-1)
                    ap = fixed['coordinate_logits'].double().log_softmax(-1)
                    kl = float((bp.exp() * (bp-ap)).sum(-1).mean())
                    coord_error = abs(kl-fixed['KL_sourcefit_to_updated'])
                    assert coord_error < 2e-6
                diagnostics.append({'key': row['key'], 'condition': condition, 'arm': arm,
                    'event_reference_unchanged': base['readout']['spatial_reference_token_ids'] == p['readout']['spatial_reference_token_ids'],
                    'interval_changed': p['interval'] != base['interval'], 'time_KL_free_reference': time_kl,
                    'fixed_prefix_coordinate_KL': kl, 'coordinate_KL_FP64_crosscheck_error': coord_error,
                    'loss_before': p['update']['history'][0]['loss_before'], 'loss_after': p['update']['loss_after'],
                    'anchor_after': p['update']['terms_after']['parameter_anchor'],
                    'event_injection': p['readout']['event_injection'], 'spatial_injection': p['readout']['spatial_injection']})
    by = {(r['key'], r['condition']): r for r in records}; summary = {}
    contrasts = [(a, b) for a in ARMS[2:] for b in ['sourcefit_noTTA', 'Frozen']]
    contrasts += [(NEW_ARM, 'calibration_alignment'), (NEW_ARM, 'calibration_alignment_output_anchor'), ('sourcefit_noTTA', 'Frozen'), ('calibration_view', 'convolution_view'),
                  ('calibration_alignment', 'convolution_alignment'),
                  ('calibration_alignment', 'calibration_view'), ('convolution_alignment', 'convolution_view')]
    for domain in ['all', 'HC', 'Vid']:
        keys = sorted({r['key'] for r in records if domain == 'all' or r['domain'] == domain})
        summary[domain] = {}
        for condition in CONDITIONS + ['corruption']:
            cc = ['noise_medium', 'defocus_extreme'] if condition == 'corruption' else [condition]
            table = [{a: {m: float(np.mean([by[(k, c)]['arms'][a][m] for c in cc]))
                          if all(by[(k, c)]['arms'][a][m] is not None for c in cc) else None
                          for m in ['v', 's', 't']} for a in ARMS} for k in keys]
            relevant = [by[(k, c)] for k in keys for c in cc]
            def stat(values):
                valid = [v for v in values if v is not None]
                return stats(valid) if valid else None
            summary[domain][condition] = {'parents': len(keys),
                'absolute': {a: {m: stat([x[a][m] for x in table]) for m in ['v', 's', 't']} for a in ARMS},
                'contrasts': {a+'-'+b: {m: stat([x[a][m]-x[b][m] for x in table if x[a][m] is not None and x[b][m] is not None])
                                      for m in ['v', 's', 't']} for a, b in contrasts},
                'format_ok': {a: sum(r['arms'][a]['format_ok'] for r in relevant) for a in ARMS},
                'native_good_frames': sum(r['native_good_frames'] for r in relevant),
                'native_good_retained': {a: sum(r['native_good_retained'][a] for r in relevant) for a in ARMS}}
    save_once(SCORE / 'QUERY_METRICS.json', records)
    save_once(SCORE / 'OUTPUT_DIAGNOSTICS.json', diagnostics)
    report = {'status': 'completed_offline_development_screen', 'time': time.time(), 'groups': summary,
        'scalar_vs_original_P3_max_error': max_error, 'metric_comparisons': 192*3,
        'primary': 'one branch removal: time-only vs both-output anchor; no-output, B1 and Frozen retained',
        'parents': 8, 'historically_exposed': True, 'optimization_seeds': 1, 'source_only_fixed_output_coefficient': cfg['output_coefficient'], 'comparison_note': 'single added factor on previously exposed target parents; descriptive CIs',
        'target_GT_read_only_after_all_predictions_sealed': True, 'production_promoted': False}
    save_once(SCORE / 'REPORT.json', report)
    lines = ['# DESTA-3D v2 fixed 8-parent TTA development screen', '',
             'Eight historically exposed parents (HC4/Vid4), fixed B1; 192 predictions sealed before offline GT. Not fresh confirmation.', '',
             '|Arm|Corruption vIoU %|TTA-sourcefit pp [95% CI]|TTA-Frozen pp|', '|---|---:|---:|---:|']
    group = summary['all']['corruption']
    for arm in ARMS:
        value = group['absolute'][arm]['v']['mean']*100
        if arm in ARMS[2:]:
            dv = group['contrasts'][arm+'-sourcefit_noTTA']['v']; df = group['contrasts'][arm+'-Frozen']['v']
            lines.append(f"|{arm}|{value:.6f}|{dv['mean']*100:+.6f} [{dv['ci95'][0]*100:+.6f}, {dv['ci95'][1]*100:+.6f}]|{df['mean']*100:+.6f}|")
        else: lines.append(f'|{arm}|{value:.6f}|—|—|')
    lines += ['', '|Condition|Time-only anchor minus both-output anchor vIoU pp [95% CI]|', '|---|---:|']
    for condition in CONDITIONS + ['corruption']:
        d = summary['all'][condition]['contrasts'][NEW_ARM+'-calibration_alignment_output_anchor']['v']
        lines.append(f"|{condition}|{d['mean']*100:+.6f} [{d['ci95'][0]*100:+.6f}, {d['ci95'][1]*100:+.6f}]|")
    lines += ['', 'The recorded loss_after is the unchanged pre-gate objective only; output KL is separately diagnosed, so it is not a full-objective after/before comparison.', '', f'Scalar versus original P3 metric max error: {max_error:.9g}. Full clean/domain/retention/tail readout is in REPORT.json; all query metrics and raw-derived distribution diagnostics are retained.',
              '', 'No automatic 64-parent launch or production promotion. Loss changes, cast injection and endpoint/logit drift are diagnostics, not task gains.']
    (SCORE / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    save_once(SCORE / 'COMPLETE.json', {'status': 'completed_and_two_metric_implementations_crosschecked',
        'report_sha': sha(SCORE / 'REPORT.json'), 'pre_GT_audit_sha': sha(SCORE / 'PRE_GT_AUDIT.json')})
    print({a: group['contrasts'][a+'-sourcefit_noTTA']['v'] for a in ARMS[2:]})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['preflight', 'score'])
    {'preflight': preflight, 'score': run}[p.parse_args().action]()
