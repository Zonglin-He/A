"""Paired arm contrasts and source influence from sealed anonymous score rows.

This is a post-score diagnostic. It neither selects a configuration nor opens
media, annotations, tensors, or weights. The source is the resampling unit.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_correction_views_common_v1 import BASE, PUBLIC, DATASETS, read, write, sha


def source_values(rows, left, right, metric='v'):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[r['source_id'], r['order'], r['condition']].append(
            r[left + '_' + metric] - r[right + '_' + metric])
    orders = collections.defaultdict(list)
    for (s, o, c), values in cells.items():
        orders[s, o].append(float(np.mean(values)))
    sources = collections.defaultdict(list)
    for (s, o), values in orders.items():
        sources[s].append(float(np.mean(values)))
    return sorted(sources), np.asarray([np.mean(sources[s]) for s in sorted(sources)])


def effect(rows, left, right, metric='v'):
    ids, values = source_values(rows, left, right, metric)
    if not len(values):
        return dict(cells=0, sources=0, mean=None, ci95=None)
    rng = np.random.default_rng(20261001)
    boot = np.concatenate([values[rng.integers(0, len(values), (100, len(values)))].mean(1)
                           for _ in range(100)])
    loo = ((values.sum() - values) / (len(values) - 1)).tolist() if len(values) > 1 else []
    result = dict(cells=len(rows), sources=len(values), mean=float(values.mean()),
                  ci95=np.quantile(boot, [.025, .975]).tolist(),
                  source_effects=[dict(source_id=s, delta=float(v)) for s, v in zip(ids, values)],
                  leave_one_source_out_range=[min(loo), max(loo)] if loo else None,
                  leave_one_source_out_sign_changes=sum(v * values.mean() < 0 for v in loo))
    if loo:
        j = int(np.argmax(np.abs(np.asarray(loo) - values.mean())))
        result['largest_source_influence'] = dict(source_id=ids[j],
            leave_out_mean=loo[j], mean_change=float(loo[j] - values.mean()))
    return result


def pooled(groups, left, right, metric='v'):
    matrices = [source_values(groups[d], left, right, metric)[1] for d in DATASETS]
    if any(not len(m) for m in matrices):
        return dict(mean=None, ci95=None)
    rng = np.random.default_rng(20261001)
    boot = []
    for _ in range(100):
        boot.extend(.5 * sum(m[rng.integers(0, len(m), (100, len(m)))].mean(1) for m in matrices))
    return dict(mean=float(np.mean([m.mean() for m in matrices])),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                dataset_means={d:float(m.mean()) for d, m in zip(DATASETS, matrices)},
                equal_dataset_weight=True, bootstrap_samples=10000)


def run(stage, split):
    directory = PUBLIC / stage / split
    root_audit = read(BASE / f'{stage}_{split}_ROOT_AUDIT.json')
    assert root_audit['status'] == 'pass'
    pairs = [('R_select', 'U_select', 'v'), ('R_select', 'R_temp', 'v'),
             ('Specific_temp', 'R_temp', 'v')] if stage == 'round1' else [
        ('T', 'A', 'v'), ('T', 'A', 't'), ('CT', 'C', 'v'),
        ('C_newacquisition', 'C', 'v'), ('CT_newacquisition', 'CT', 'v'),
        ('C', 'twoUniform5', 'v'), ('CT_newacquisition', 'twoUniform5_newtime', 'v'),
        ('R_acquisition_new', 'R_acquisition_old', 'v')]
    allrows = {d:read(directory / d / 'ROWS.json') for d in DATASETS}
    result = dict(stage=stage, split=split, purpose='post-score paired interpretation; no selection',
                  input_hashes={d:sha(directory / d / 'ROWS.json') for d in DATASETS}, contrasts={})
    for left, right, metric in pairs:
        name = f'{left}_minus_{right}_{metric}'
        target = result['contrasts'][name] = {}
        for group in ['clean', 'corruption']:
            target[group] = {}
            for sub in ['all', 'expert', 'nonexpert']:
                groups = {d:[r for r in rr if (r['condition'] == 'clean') == (group == 'clean') and
                          (sub == 'all' or r['expert_scheduled'] == (sub == 'expert'))]
                          for d, rr in allrows.items()}
                target[group][sub] = dict(
                    datasets={d:effect(rr, left, right, metric) for d, rr in groups.items()},
                    pooled=pooled(groups, left, right, metric))
    if stage == 'round2':
        result['temporal_diagnosis'] = {}
        result['acquisition_diagnosis'] = {}
        for ds in DATASETS:
            wr = read(directory / ds / 'WRITE_ROWS.json')
            old = read(PUBLIC / 'round1' / split / ds / 'WRITE_ROWS.json')
            old = {(w['condition'],w['order'],w['arrival']):w for w in old}
            result['temporal_diagnosis'][ds] = {}
            result['acquisition_diagnosis'][ds] = {}
            for group in ['clean', 'corruption']:
                ww = [w for w in wr if (w['condition'] == 'clean') == (group == 'clean')]
                result['temporal_diagnosis'][ds][group] = dict(
                    expert_cells=len(ww),
                    actual_observation_difference=float(np.mean([w['temporal']['actual_observation_difference'] for w in ww])),
                    selected_changed=sum(w['temporal']['new_selected'] != w['temporal']['original_selected'] for w in ww),
                    **{k:sum(w['temporal'][k] for w in ww) for k in [
                        'old_replaced','new_replaced','old_wrong_replacement_t','new_wrong_replacement_t',
                        'old_wrong_replacement_v','new_wrong_replacement_v']})
                items = []
                for w in ww:
                    baseline = old[w['condition'],w['order'],w['arrival']]['evidence']['R']
                    new = w['evidence']['Rnew']
                    items.append(dict(source_id=w['source_id'], order=w['order'],condition=w['condition'],
                        old_v=baseline['event_frame_precision'], new_v=new['event_frame_precision']))
                result['acquisition_diagnosis'][ds][group] = dict(
                    event_precision_new_minus_old=effect(items, 'new', 'old'),
                    note='GT diagnostic; coverage is not critic correctness or final tube utility')
    write(directory / 'CONTRASTS.json', result)
    print('Paired contrasts saved', stage, split)


if __name__ == '__main__':
    run(*sys.argv[1:])
