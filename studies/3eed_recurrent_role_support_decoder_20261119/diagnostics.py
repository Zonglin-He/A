"""Relation identities from saved independent reviews and the source recording holdout."""
from collections import Counter, defaultdict
import sys

import numpy as np

from common import (ARMS, ORDER, PLATFORMS, PRIVATE, RESULTS, ROOT, frame, json_atomic,
                    ordinary_rows, read, sha)
from score import NAMES, PREDICTIONS, local_points, owner_for, save_csv, verify_predictions


def source_diagnostics():
    measures, curves = [], []
    for source in PLATFORMS:
        folder = PRIVATE/source
        roster = read(folder/'ROSTER_SEAL.json')['overview']
        for arm in ARMS:
            d = read(folder/f'{arm}_TRAINING.json')
            diag = d['diagnostic']
            n, n_rel = (int(diag[k]) for k in ('n', 'relation_n'))
            assert n == roster['monitor']['natural'] + n_rel
            assert n_rel == roster['monitor']['relations']
            measures.append(dict(source=source, arm=arm, train_n=d['n_train'],
                                 train_natural=roster['train']['natural'],
                                 train_synthetic_relations=roster['train']['relations'],
                                 monitor_n=n, monitor_relations=n_rel,
                                 subject_mIoU=100*diag['subject_iou']/n,
                                 subject_Acc25=100*diag['subject25']/n,
                                 subject_Acc50=100*diag['subject50']/n,
                                 subject_mean_gt_mass=diag['subject_support']/n,
                                 synthetic_reference_Acc25=(100*diag['reference25']/n_rel if n_rel else ''),
                                 synthetic_both_Acc25=(100*diag['pair25']/n_rel if n_rel else ''),
                                 synthetic_reference_mean_gt_mass=(diag['reference_support']/n_rel if n_rel else ''),
                                 updates=d['budget_updates'], params=d['parameters'],
                                 training_seconds=d['wall_seconds']))
            curves.extend(dict(source=source, arm=arm, **samples) for samples in d['train'])
    save_csv('source_diagnostics.csv', measures)
    save_csv('training_curves.csv', curves)


def target_diagnostics():
    from common160 import rows
    sys.path[:0] = [str(ROOT/p) for p in ('research/round162', 'research/round165', 'research/round158')]
    from mechanisms162 import annotation_inventory

    path = ROOT/'crossplatform/round163'
    records, annotations = path/'ROLE_REVIEW_PRIVATE.json', path/'ROLE_ADJUDICATION_PRIVATE.json'
    proof = read(ROOT/'research/round163/ROLE_SCOPE.json')
    quals = read(ROOT/'research/round163/results/ROLE_REVIEW_SUMMARY.json')
    assert sha(records) == proof['private_review_sha256'] and sha(annotations) == quals['review_sha256']
    assert not proof['FINAL_access'] and not quals['FINAL_access']
    old = rows()
    latest = {p:{(frame(e),e['caption']):i for i,e in enumerate(ordinary_rows(p))} for p in PLATFORMS}
    work = defaultdict(list)
    counts = Counter()
    frames = set()
    for item, label in zip(read(records), read(annotations), strict=True):
        assert item['case'] == label['case']
        r = old[item['panel_index']]['r']
        platform = item['target']
        key = (frame(r), r['caption'])
        if key not in latest[platform]:
            counts['not_on_R165_validation'] += 1
            continue
        direction = item['source']+'_to_'+platform
        ordinal = latest[platform][key]
        counts[label['type']] += 1
        if label['type'] != 'verified_boxes':
            continue
        standardized = ordinary_rows(platform)[ordinal]
        names, boxes = annotation_inventory(standardized['metapath'], platform,
                                            int(standardized['suffix'] or 0), standardized['object_index'])
        ref = int(label['ref'])
        assert len(boxes) == len(item['boxes']) and 0 < ref < len(boxes) and 0 in item['role_indices']
        work[direction].append((ordinal, ref, boxes))
        frames.add((platform, frame(standardized)))
    values = defaultdict(Counter)
    for direction, cases in work.items():
        root = PREDICTIONS/direction
        for filename in sorted(read(root/'SEAL.json')['files']):
            with np.load(root/filename, allow_pickle=False) as f:
                indices = f['ordinals']
                active = [c for c in cases if indices[0] <= c[0] <= indices[-1]]
                if not active:
                    continue
                seeds = f['seeds']
                boxes = f['boxes']
                refs = f['refs']
                rel = f['is_relation']
                supports = f['support']
            for idx, ref, inventory in active:
                k = idx-int(indices[0])
                assert indices[k] == idx
                x = seeds[k]
                occupancy = np.stack([local_points(x, b) for b in inventory])
                s = np.unique(x, axis=0)
                occ_unique = np.stack([local_points(s, b) for b in inventory])
                owners = np.where(occ_unique.sum(0) == 1, occ_unique.argmax(0), len(inventory))
                for j, arm in enumerate(NAMES):
                    is_locked = owner_for(boxes[k, j], s, owners, len(inventory))
                    m = values[arm]
                    m['n'] += 1
                    m['subject_correct'] += int(is_locked == 0)
                    m['subject_unknown'] += int(is_locked < 0)
                    if j:
                        other = owner_for(refs[k, j-1], s, owners, len(inventory)) if rel[k] else -1
                        m['reference_correct'] += int(other == ref)
                        m['reference_unknown'] += int(other < 0)
                        m['both_correct'] += int(other == ref and is_locked == 0)
                        m['reference_role_emitted'] += int(rel[k])
                        w = supports[k, j-1, 1].astype(np.float64)
                        m['mass'] += float(w[occupancy[ref]].sum()/w.sum()) if rel[k] else 0.
    n = counts['verified_boxes']
    result = []
    for arm in NAMES:
        v = values[arm]
        assert v['n'] == n
        result.append(dict(arm=arm, paired_review_records=n, unique_frames=len(frames),
                           subject_correct=v['subject_correct'], subject_unknown=v['subject_unknown'],
                           reference_correct=(v['reference_correct'] if arm in ARMS else ''),
                           reference_unknown=(v['reference_unknown'] if arm in ARMS else ''),
                           both_correct=(v['both_correct'] if arm in ARMS else ''),
                           reference_role_emitted=(v['reference_role_emitted'] if arm in ARMS else ''),
                           mean_reference_gt_mass=(v['mass']/n if arm in ARMS and n else '')))
    save_csv('prior_verified_identity_panel.csv', result)
    json_atomic(RESULTS/'identity_panel.json',
                dict(statuses=dict(counts), verified_direction_rows=n, physical_frames=len(frames),
                     provenance='R163 pre-existing visual review of a failure-selected panel',
                     benchmark_identity_claim=False, used_for_training=False, FINAL_access=False))
    print('POST_SEAL_PRIOR_IDENTITY', n, len(frames), flush=True)


def main():
    verify_predictions()
    source_diagnostics()
    target_diagnostics()


if __name__ == '__main__':
    main()
