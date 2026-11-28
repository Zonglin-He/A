"""Measure candidate headroom only, opening references after candidate sealing."""
from collections import Counter
from itertools import combinations

import numpy as np
import torch

from compact_pool import (BASELINE, CAPACITY, FAMILIES, HERE, ORACLE_GATE, PRIOR, RUN,
                          TOL, digest, read, write)
from score_paradigms import ious

BASE = 'Minimal-D'
POOL = 'compact_oracle'
BROAD = 'historical_oracle'


def families(entry):
    return {key.split('/')[0] for key in entry['sources']}


def score_query(entry, historical, target):
    candidates = entry['pool']
    values = ious([c['box'] for c in candidates], target)
    incumbent, oracle = float(values[0]), float(values.max())
    assert abs(incumbent-historical['iou'][BASE]) < 1.e-9
    assert entry['branch'] == historical['branch'] and entry['target'] == historical['target']
    named = [families(c) for c in candidates]
    unique_winner = None
    if oracle > incumbent + TOL:
        winners = set().union(*(tag for tag, y in zip(named, values, strict=True) if y >= oracle - TOL))
        if len(winners) == 1:
            unique_winner = winners.pop()
    scores = {BASE: incumbent, POOL: oracle, BROAD: historical['iou']['candidate_oracle']}
    for tag in FAMILIES:
        scores[f'only_{tag}'] = max([incumbent, *(float(y) for y, s in zip(values, named, strict=True) if tag in s)])
        scores[f'without_{tag}'] = max(float(y) for y, s in zip(values, named, strict=True) if s - {tag})
    # Split overlapping headroom exactly by averaging marginal contributions over all family orderings.
    subsets = {}
    for size in range(len(FAMILIES)+1):
        for group in combinations(FAMILIES, size):
            chosen = frozenset(group)
            subsets[chosen] = max([incumbent, *(float(y) for y, tags in zip(values, named, strict=True) if chosen & tags)])
    shared = {}
    for tag in FAMILIES:
        other = [s for s in subsets if tag not in s]
        shared[tag] = sum((subsets[s | {tag}] - subsets[s]) /
                          (len(FAMILIES) * sum(len(t) == len(s) for t in other)) for s in other)
    assert abs(sum(shared.values()) - (oracle-incumbent)) < 1.e-10
    return dict(uid=entry['uid'], target=entry['target'], recording=historical['recording'],
                branch=entry['branch'], iou=scores, unique_winner=unique_winner, shared=shared,
                k=len(candidates), has_anchor=candidates[0]['box'] is not None,
                candidate_sources=dict(Counter(tag for s in named[1:] for tag in s)),
                actual_region_count=entry['actual_region_count'], rays=entry['audit'])


def summarize(rows, stat_panel):
    names = (BASE, POOL, BROAD, *(f'{group}_{tag}' for group in ('only','without') for tag in FAMILIES))
    result = stat_panel(rows, names, BASE)
    old, compact = (np.array([r['iou'][name] for r in rows]) for name in (BROAD, POOL))
    k = np.array([r['k'] for r in rows])
    headroom = result['metrics'][POOL]['delta_miou']
    no_family = result['metrics']
    unique = Counter(r['unique_winner'] for r in rows)
    result.update(
        capacity=dict(mean=float(k.mean()), median=float(np.median(k)), max=int(k.max()),
                      histogram=dict(sorted(Counter(map(int, k)).items())), anchor_only=int((k == 1).sum()),
                      empty_anchor=sum(not r['has_anchor'] for r in rows)),
        family={f: dict(
            only_added_pp=no_family[f'only_{f}']['delta_miou'],
            removal_loss_pp=headroom-no_family[f'without_{f}']['delta_miou'],
            sole_oracle_source=unique[f],
            queries_with_nonanchor=sum(r['candidate_sources'].get(f, 0)>0 for r in rows),
            shapley_pp=(effect := 100*float(np.mean([r['shared'][f] for r in rows]))),
            shapley_share=effect/headroom if headroom else None) for f in FAMILIES},
        historical_comparison=dict(broad_higher=int((old > compact+TOL).sum()),
                                   compact_higher=int((compact > old+TOL).sum()),
                                   equal=int((abs(compact-old) <= TOL).sum()),
                                   broad_only_acc50=int(((old > .50) & (compact <= .50)).sum()),
                                   compact_only_acc50=int(((compact > .50) & (old <= .50)).sum()),
                                   union_oracle_miou=100*float(np.maximum(old, compact).mean())),
        actual_region_counts=dict(sorted(Counter(r['actual_region_count'] for r in rows).items())),
        ray_audit={name: dict(considered=sum(r['rays'][name][0] for r in rows),
                              changed=sum(r['rays'][name][1] for r in rows)) for name in ('selection','audit')})
    return result


def main():
    torch.set_num_threads(1)
    payload = RUN / 'candidates-private.json'
    seal = read(RUN / 'candidate-seal.json')
    assert seal['payload'] == digest(payload) and not seal['labels_read']
    assert seal['generator'] == digest(HERE/'compact_pool.py')
    assert seal['protocol'] == digest(HERE/'PROTOCOL_ZH.md') and seal['registration'] == BASELINE
    predictions = {r['uid']:r for r in read(payload)}
    assert len(predictions) == seal['n'] == 702
    assert all(1 <= len(r['pool']) <= CAPACITY and 'anchor' in r['pool'][0]['sources']
               for r in predictions.values())
    history = PRIOR / 'scores-private.json'
    assert digest(history) == read(PRIOR / 'results.json')['offline_score_sha']
    roster = {r['uid']:r for r in read(history)['critic'] if r['target'] != 'mono'}
    assert len(roster) == 665 and roster.keys() <= predictions.keys()

    from shared import setup
    setup()
    from evaluate_structure import stat_panel
    from metric_image_common import inputs
    from truth_score_inputs import truth_data

    frames = inputs()
    real, _ = truth_data({u:frames[u] for u in roster})
    assert real.keys() == roster.keys()
    rows = [score_query(predictions[u], historical, real[u]['box']) for u,historical in roster.items()]
    panels = dict(development=rows)
    panels.update({t: [r for r in rows if r['target'] == t] for t in ('waymo','drone','quad','kitti')})
    panels.update({b: [r for r in rows if r['branch'] == b] for b in ('C_ray','H','A','missing_3d')})
    metrics = {name:summarize(subset, stat_panel) for name,subset in panels.items() if subset}
    oracle = metrics['development']['metrics'][POOL]['miou']
    advanced = oracle >= ORACLE_GATE
    record_file = RUN / 'offline-scores-private.json'
    write(record_file, rows)
    result = dict(stage_1=dict(decision='go' if advanced else 'no_go', gate_miou=ORACLE_GATE,
                               oracle_miou=oracle, gap_pp=oracle-ORACLE_GATE),
                  stage_2=dict(state='eligible' if advanced else 'not_run'),
                  stage_3=dict(state='pending_stage_2' if advanced else 'not_run'),
                  method_baseline=BASELINE, panels=metrics, labels_after_seal=True,
                  candidate_seal=seal, history_scores_sha=digest(history), scores_sha=digest(record_file),
                  scorer_sha=digest(__file__))
    write(RUN / 'results.json', result)
    for name, panel in metrics.items():
        print(name, panel['n'], {key: (round(row['miou'], 3), round(row['acc50'], 3))
                                for key,row in panel['metrics'].items() if key in (BASE,POOL,BROAD)}, flush=True)
    print(result['stage_1'], flush=True)


if __name__ == '__main__':
    main()
