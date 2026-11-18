"""Describe the sealed transfers and conditional sampling variation of task gains."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    path = HERE / 'results'
    assert json.loads((path / 'AUDIT.json').read_bytes())['status'] == 'PASS'
    with (path / 'task_all_rows.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    reg = json.loads((ROOT / 'research/round131/INPUT_REGISTRY_PRIVATE.json').read_bytes())
    assert len(rows) == len(reg) == 1798
    controls = ['frozen_A', 'R145', 'fusion_zero', 'shuffled_fusion_TTA']
    strata = defaultdict(set)
    by_direction = defaultdict(list)
    matches, inserted = [], []
    entrants = Counter()
    entrant_delta = defaultdict(list)
    for i, (row, e) in enumerate(zip(rows, reg)):
        assert int(row['ordinal']) == i+1 and (row['source'], row['target']) == (e['source'], e['target'])
        strata[(row['target'], row['scene_cluster'])].add(int(row['point_content_cluster']))
        by_direction[(row['source'], row['target'])].append(i)
        item = ROOT / 'crossplatform/round154/predictions' / (e['source']+'_to_'+e['target']) / (e['row']['id']+'.npz')
        with np.load(item, allow_pickle=False) as values:
            g = int(values['fusion_TTA_winner_group'])
            kind = 'new_alternate_singleton' if g >= 256 else 'matched_pair' if values['A_to_alt'][g] >= 0 else 'original_singleton'
            entrants[kind] += 1
            matches.append(int(values['n_matched']))
            inserted.append(int(values['n_unmatched_alt_in_TTA_pool']))
            entrant_delta[kind].append(i)
    clusters = np.array([int(x['point_content_cluster']) for x in rows])
    assert sum(map(len, strata.values())) == len(set(clusters)) == 384
    weights = np.array([float(x['sampling_weight']) for x in rows])
    delta = np.array([[100*(float(x['fusion_TTA'])-float(x[c])) for c in controls] for x in rows])
    def estimate(weights):
        return np.mean([np.average(delta[idx], axis=0, weights=weights[idx]) for idx in by_direction.values()], axis=0)
    rng = np.random.default_rng(154)
    # With fixed actual recordings, sampling variation arises only within each recording.
    counts = np.zeros((2000, int(clusters.max())+1), dtype=np.int32)
    for stratum in strata.values():
        ids = sorted(stratum)
        counts[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=len(counts))
    estimates = np.asarray([estimate(weights * count[clusters]) for count in counts])
    exact = estimate(weights)
    lo, hi = np.percentile(estimates, [2.5, 97.5], axis=0)
    normalized = np.zeros(len(rows))
    for ids in by_direction.values():
        normalized[ids] = weights[ids] / (6*weights[ids].sum())
    output = {
        'paired_mIoU_pp': {c: dict(point=x, conditional_lo=l, conditional_hi=h)
                           for c, x, l, h in zip(controls, exact, lo, hi)},
        'scope': 'exposed development; conditional on observed recordings; no new-scene or method-selection uncertainty',
        'eligible_B': sum(x['has_B'] == 'True' for x in rows), 'n': len(rows),
        'mean_pairs': float(np.mean(matches)), 'mean_new_singletons_in_final256': float(np.mean(inserted)),
        'winner_type_count': dict(entrants),
        'winner_type_contribution_vs_R145_pp': {kind: float((normalized[idx]*delta[idx, 1]).sum())
                                                for kind, idx in entrant_delta.items()},
    }
    (HERE / 'TRANSFER_SUMMARY.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
