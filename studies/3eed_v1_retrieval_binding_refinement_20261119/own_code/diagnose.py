"""Measure coverage gained and used without changing sealed predictions."""

from collections import Counter

import numpy as np

from execute import HERE, DIRECTIONS, SLOT_PROFILES, digest, implementation, many_iou
from execute import read, register, registered_direction, save, save_csv, sensor_truth


def main():
    protocol = read(HERE / 'protocol.json')
    rows, baseline = register(protocol)
    private = HERE / 'predictions_PRIVATE'
    seal = read(private / 'SEALED.json')
    audit = read(HERE / 'results/audit.json')
    assert audit['prediction_seal_sha256'] == digest(private / 'SEALED.json')
    assert seal['lock_sha256'] == digest(private / 'LOCK.json')
    assert read(private / 'LOCK.json')['implementation'] == implementation()
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    arrays = {}
    for source, target in DIRECTIONS:
        direction = f'{source}_to_{target}'
        path = private / f'{direction}.npz'
        assert digest(path) == seal['files'][path.name]['sha256']
        ordinals, old = registered_direction(rows, baseline, source, target)
        with np.load(path, allow_pickle=False) as record:
            np.testing.assert_array_equal(record['ordinals'], ordinals)
            np.testing.assert_array_equal(record['boxes'][:, 0], old)
            assert tuple(record['arms']) == tuple(protocol['arms'])
            arrays[direction] = {key: record[key] for key in ('ordinals', 'pool', 'slots', 'reachable', 'pairs')}

    counts = {(group, mode, tau): Counter()
              for group in (*arrays, 'all_directions')
              for mode in SLOT_PROFILES[1:] for tau in (.25, .5)}
    for direction, block in arrays.items():
        target = direction.split('_to_')[1]
        for j, ordinal in enumerate(block['ordinals']):
            row = rows[int(ordinal)]
            gt = sensor_truth(target, row['metapath'], row['caption'], row['suffix'])
            iou = many_iou(block['pool'][j], gt)
            slots = block['slots'][j, :, 0]
            original = float(iou[slots[0][slots[0] >= 0]].max())
            for index, mode in enumerate(SLOT_PROFILES[1:], 1):
                retrieved = float(iou[slots[index][slots[index] >= 0]].max())
                reachable = float(iou[slots[index][block['reachable'][j, index]]].max())
                output = float(iou[block['pairs'][j, protocol['arms'].index(mode), 0]])
                for tau in (.25, .5):
                    entered = original <= tau < retrieved
                    record = dict(total=1, newly_covered=int(entered),
                                  lost_coverage=int(retrieved <= tau < original),
                                  newly_covered_reachable=int(entered and reachable > tau),
                                  newly_covered_selected=int(entered and output > tau))
                    for group in (direction, 'all_directions'):
                        counts[group, mode, tau].update(record)
    result = [dict(direction=group, arm=mode, threshold=threshold, **counter)
              for (group, mode, threshold), counter in counts.items()]
    path = HERE / 'results/coverage_transitions.csv'
    save_csv(path, result)
    save(HERE / 'results/transition_audit.json', {
        'source_audit_sha256': digest(HERE / 'results/audit.json'),
        'source_prediction_seal_sha256': audit['prediction_seal_sha256'],
        'script_sha256': digest(HERE / 'diagnose.py'),
        'rows': seal['n'], 'diagnostic_after_seal': True, 'FINAL_access': False,
    })
    print(f'Coverage transitions checked over all {seal["n"]} sealed rows')


if __name__ == '__main__':
    main()
