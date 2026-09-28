"""Isolated full-B1 reference readback with physical-input and replay barriers.

The accepted scalar reference scorer is unchanged. A second tensor calculation
crosschecks all query metrics and parent CIs after the scalar readback finishes.
No model inference, optimization, target labels, or state selection.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_desta3d_v2_reference_audit as scalar
from scripts import score_desta3d_v2_aux_recovery as aux
from scripts.crosscheck_desta3d_v2_aux_recovery import pairwise_auc

BASE = ROOT / 'artifacts/desta3d_v2/aux_backflow_v1/final_B1_reference_v1'
RUN = BASE / 'B1_final001'
ARMS = scalar.ARMS


def preflight():
    complete = scalar.read_json(RUN / 'COMPLETE.json')
    assert complete['status'] == 'completed' and complete['interface_pass']
    seal = scalar.read_json(RUN / 'PREDICTIONS_SEAL.json')
    assert not seal['source_val_GT_read'] and not seal['target_GT_read']
    expected = {str(RUN / 'predictions' / arm / f'{i:03}.pt') for arm in ARMS for i in range(198)}
    expected |= {str(RUN / 'diagnostics' / f'{i:03}.pt') for i in range(198)}
    assert set(seal['pins']) == expected and seal['queries'] == 198
    for path, digest in seal['pins'].items():
        assert scalar.sha256_file(Path(path)) == digest, path
    for path, digest in scalar.read_json(RUN / 'LOCK.json')['pins'].items():
        assert scalar.sha256_file(Path(path)) == digest, path
    aux.sealed_files(aux.BASE)
    cfg = scalar.read_json(RUN / 'CONFIG.json')
    rows = scalar.read_json(RUN / 'INPUTS.json')
    assert len(rows) == 198 and len({r['source'] for r in rows}) == 31
    ck = torch.load(cfg['checkpoint']['checkpoint'], map_location='cpu', weights_only=False)
    assert aux.state_sha(ck['adapter']) == cfg['checkpoint']['adapter_sha256']
    original = torch.load(aux.BASE / 'FINAL_B.pt', map_location='cpu', weights_only=False)['B1']
    assert aux.state_sha(original) == aux.state_sha(ck['adapter'])
    barrier = scalar.read_json(aux.SOURCE / 'SOURCE_FULL_FEATURE_BARRIER.json')
    physical = []; replay = []; reference = []
    for i, row in enumerate(rows):
        pair = [torch.load(RUN / 'predictions' / a / f'{i:03}.pt', map_location='cpu', weights_only=False) for a in ARMS]
        pres = [aux.check_prediction(p, row, cfg['checkpoint']['adapter_sha256']) for p in pair]
        assert pres[0] == pres[1]
        old = torch.load(aux.BASE / 'predictions/B1' / f'{i:03}.pt', map_location='cpu', weights_only=False)
        assert aux.check_prediction(old, row, cfg['checkpoint']['adapter_sha256']) == pres[0]
        fields = ['key', 'source', 'frame_ids', 'positions', 'interval', 'format_ok', 'event_completion', 'spatial_completion', 'video_sha256']
        equals = {k: old[k] == pair[0][k] for k in fields}
        equals.update({k: torch.equal(old[k], pair[0][k]) for k in ['boxes_cxcywh', 'geometry_valid', 'event_logits']})
        assert all(equals.values()), (row['key'], equals)
        replay.append({'key': row['key'], 'all_fields_equal': True})
        cache_path = aux.SOURCE / 'source_features/clean' / (hashlib.sha256(row['key'].encode()).hexdigest() + '.pt')
        assert scalar.sha256_file(cache_path) == barrier['files'][str(cache_path)]
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        aux.frozen_prediction(cache, row, pres[0])
        physical.append({'key': row['key'], 'pixel_sha': pres[0]['pixel_sha'], 'source': str(row['source'])})
        diag = torch.load(RUN / 'diagnostics' / f'{i:03}.pt', map_location='cpu', weights_only=False)
        refs = [diag[a]['spatial_reference_token_ids'] for a in ARMS]
        reference.append({'key': row['key'], 'valid_reference_pair': bool(refs[0]) and bool(refs[1]), 'equal': refs[0] == refs[1]})
    report = {'status': 'passed', 'time': time.time(), 'new_sealed_files': 594, 'queries': 198, 'parents': 31,
        'seal_sha': scalar.sha256_file(RUN / 'PREDICTIONS_SEAL.json'), 'physical_inputs': physical,
        'original_B1_replay': replay, 'reference_rows': reference,
        'valid_reference_pairs': sum(r['valid_reference_pair'] for r in reference),
        'different_valid_reference_pairs': sum(r['valid_reference_pair'] and not r['equal'] for r in reference),
        'source_labels_opened_by_this_preflight': False, 'source_validation_previously_exposed': True, 'target_GT_read': False}
    aux.save_once(BASE / 'PRE_GT_PHYSICAL_REPLAY_AUDIT.json', report)
    return rows, report


def crosscheck(rows, pre):
    from scripts.desta3d_source_fit_v1 import _tube_metric, _frozen_prediction
    path = RUN / 'independent_cpu_readback/INDEPENDENT_SCORE.json'
    report = scalar.read_json(path)
    assert report['provenance']['status'] == 'passed'
    label_path = aux.SOURCE / 'SOURCE_LABELS_TRAINING_ONLY.json'
    assert scalar.sha256_file(label_path) == report['provenance']['source_labels_sha256']
    labels = scalar.read_json(label_path)
    barrier = scalar.read_json(aux.SOURCE / 'SOURCE_FULL_FEATURE_BARRIER.json')
    errors = []; auc_errors = []; parent_maps = {}
    for arm in (*ARMS, 'Frozen'):
        data = report['frozen_reference'] if arm == 'Frozen' else report['arms'][arm]
        lookup = {r['key']: r for r in data['query_rows']}
        parents = {}; events = {}
        for i, row in enumerate(rows):
            pred = (_frozen_prediction(row, barrier) if arm == 'Frozen' else
                    torch.load(RUN / 'predictions' / arm / f'{i:03}.pt', map_location='cpu', weights_only=False))
            values = _tube_metric(pred, labels[row['key']])
            for m in scalar.METRICS:
                err = abs(values[m] - lookup[row['key']]['metrics'][m]); errors.append(err)
                assert err < 2e-6, (arm, row['key'], m, err)
            parent = str(row['source'])
            parents.setdefault(parent, []).append([values[m] for m in scalar.METRICS])
            if arm != 'Frozen':
                e = events.setdefault(parent, [[], []]); e[0].extend(labels[row['key']]['event_active'])
                e[1].extend(pred['event_logits'].float().reshape(-1).tolist())
        parent_maps[arm] = {p: np.mean(v, axis=0) for p, v in parents.items()}
        if events:
            lookup_auc = {r['source']: r['event_frame_AUROC'] for r in data['event_endpoint']['source_points']}
            for p, (y, x) in events.items():
                actual = pairwise_auc(y, x); expected = lookup_auc[p]
                assert (actual is None) == (expected is None)
                if actual is not None:
                    auc_errors.append(abs(actual - expected)); assert abs(actual - expected) < 1e-12
    ci_error = 0.
    pairs = {'shared_minus_independent': (ARMS[1], ARMS[0]), 'independent_minus_Frozen': (ARMS[0], 'Frozen'), 'shared_minus_Frozen': (ARMS[1], 'Frozen')}
    for name, (a, b) in pairs.items():
        parents = sorted(parent_maps[a]); delta = np.array([parent_maps[a][p] - parent_maps[b][p] for p in parents])
        idx = np.random.default_rng(20260927).integers(0, 31, size=(10000, 31))
        ci = np.quantile(delta[idx].mean(axis=1), [.025, .975], axis=0) * 100
        for j, m in enumerate(scalar.METRICS):
            err = float(np.max(np.abs(ci[:, j] - report['paired_parent_comparisons'][name][m]['bootstrap_ci95_pp'])))
            ci_error = max(ci_error, err); assert err < 2e-4
    aux.save_once(BASE / 'ROOT_SCORE_CROSSCHECK.json', {'status': 'passed', 'time': time.time(),
        'query_metric_comparisons': len(errors), 'maximum_metric_error': max(errors),
        'maximum_AUROC_error': max(auc_errors, default=0), 'maximum_CI_error_pp': ci_error,
        'report_sha': scalar.sha256_file(path), 'preflight_sha': scalar.sha256_file(BASE / 'PRE_GT_PHYSICAL_REPLAY_AUDIT.json'),
        'valid_reference_pairs': pre['valid_reference_pairs'], 'different_valid_reference_pairs': pre['different_valid_reference_pairs'],
        'target_GT_read': False, 'new_inference_or_updates': False})


def main():
    torch.set_num_threads(2)
    if '--self-test' in sys.argv:
        scalar.synthetic_self_check()
        assert pairwise_auc([0, 1], [0, 1]) == 1.
        assert pairwise_auc([0, 1], [1, 0]) == 0.
        assert pairwise_auc([0, 1], [1, 1]) == .5
        assert pairwise_auc([1, 1], [0, 1]) is None
        print('existing scalar synthetic controls + four independent AUC controls passed')
        return
    rows, pre = preflight()
    scalar.run(RUN)
    crosscheck(rows, pre)
    aux.save_once(BASE / 'SCORING_COMPLETE.json', {'status': 'completed_and_crosschecked', 'time': time.time(),
        'report_sha': scalar.sha256_file(RUN / 'independent_cpu_readback/INDEPENDENT_SCORE.json'),
        'script_sha': scalar.sha256_file(Path(__file__)), 'target_GT_read': False})
    print('full B1 reference scalar readback and second tensor crosscheck complete')


if __name__ == '__main__':
    main()
