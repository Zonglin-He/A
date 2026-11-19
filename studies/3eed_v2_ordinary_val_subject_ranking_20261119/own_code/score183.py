"""Score all sealed ordinary choices and measure paired threshold transitions."""

from functools import lru_cache
from pathlib import Path
import secrets

import numpy as np

from scope183 import ARMS, DIRECTIONS, HERE, atomic, canonical, check_input, digest, ordinary, read
from predict183 import BLOCK, OUT, execution
from evaluate import recording_intervals, save_csv, score_boxes
from score_ablation import rotation, WAYMO_ANGLES

RESULTS = HERE / 'results'
IOU_OUT = HERE / 'audited_iou_PRIVATE'
METRICS = ('Acc25', 'Acc50', 'mIoU')
CUTS = (.25, .50)


@lru_cache(maxsize=4096)
def truth(path, ordinal, caption, target, suffix):
    row = read(Path(path))['ground_info'][ordinal]
    assert canonical(row['caption']) == caption
    raw = np.asarray(row['bbox_3d'], dtype=np.float64)
    assert raw.shape == ((7,) if target == 'waymo' else (9,))
    box = raw[:7].copy()
    if target == 'waymo':
        angle = np.deg2rad(WAYMO_ANGLES[suffix])
        r = rotation(angle)
        box[:3] = r @ box[:3]
        box[6] += angle
    elif target == 'drone':
        box[2] += 1.8
    box = box.astype(np.float32).astype(np.float64)
    if target == 'waymo':
        box[:3] = box[:3] @ r
        box[6] -= angle
    elif target == 'drone':
        box[2] -= 1.8
    assert np.isfinite(box).all() and (box[3:6] > 0).all()
    return box


def prediction_seal():
    path = OUT / 'SEALED.json'
    pin = read(path)
    lock = OUT / 'LOCK.json'
    assert read(lock) == execution() and digest(lock) == pin['lock_sha256']
    assert pin['n'] == 18620 and pin['all_arms_same_slate']
    assert len(pin['directions']) == len(DIRECTIONS)
    assert not pin['target_labels_read'] and not pin['FINAL_access']
    return pin


def shards(key, receipt):
    part = OUT / key
    expected = [f'{i:05d}.npz' for i in range(0, receipt['n'], BLOCK)]
    assert set(receipt['files']) == set(expected)
    for path in expected:
        item = receipt['files'][path]
        disk = part / path
        assert digest(disk) == item['sha256'] and read(disk.with_suffix('.json')) == item
        with np.load(disk, allow_pickle=False) as block:
            r = {name: block[name] for name in block.files}
        required = np.arange(item['first'], item['last'])
        np.testing.assert_array_equal(r['ordinals'], required)
        assert required.size == item['n']
        yield r


def metrics(vals):
    vals = np.asarray(vals)
    return dict(Acc25=float((vals > .25).mean() * 100), Acc50=float((vals > .50).mean() * 100),
                mIoU=float(vals.mean() * 100))


def deltas(v, w):
    return {f'{m}_pp': v[m] - w[m] for m in METRICS}


def summary_metrics(blocks, name, column):
    if name == 'six_equal':
        arm = [metrics(b['ious'][:, column]) for b in blocks.values()]
        return {m: float(np.mean([val[m] for val in arm])) for m in METRICS}
    if name == 'all_pooled':
        return metrics(np.concatenate([b['ious'][:, column] for b in blocks.values()]))
    return metrics(blocks[name]['ious'][:, column])


def changes(block, column, threshold, before=0):
    selected = block['ious']
    a, b = selected[:, column] > threshold, selected[:, before] > threshold
    rescue, harm = a & ~b, b & ~a
    outside = block['outside'][:, column]
    return dict(rescued_original_four_miss=int((rescue & (block['four_best'] <= threshold)).sum()),
                rescued_by_outside=int((rescue & outside).sum()),
                harms_by_outside=int((harm & outside).sum()), total_rescues=int(rescue.sum()),
                total_harms=int(harm.sum()), net=int(rescue.sum() - harm.sum()))


def joined(blocks):
    return {key: np.concatenate([b[key] for b in blocks.values()])
            for key in next(iter(blocks.values()))}


def summarize(blocks):
    both = joined(blocks)
    directions, transitions, matched, coverage = [], [], [], []
    comparisons = (('rank_joint', 'rank_subject'), ('rank_joint', 'rank_mismatch'),
                   ('rank_joint', 'frozen_joint'), ('rank_subject', 'frozen_subject'),
                   ('rank_mismatch', 'frozen_mismatch'))
    for name, block in (*blocks.items(), ('six_equal', both), ('all_pooled', both)):
        n = len(block['ious'])
        for i, arm in enumerate(ARMS):
            value = summary_metrics(blocks, name, i)
            delta = block['ious'][:, i] - block['ious'][:, 0]
            counts = dict(switched=int((block['chosen'][:, i] != block['chosen'][:, 0]).sum()),
                          iou_better=int((delta > 1e-12).sum()), iou_worse=int((delta < -1e-12).sum()))
            directions.append(dict(direction=name, arm=arm, n=n, **value,
                                   **deltas(value, summary_metrics(blocks, name, 0)), **counts))
            for cut in CUTS:
                transitions.append(dict(direction=name, arm=arm, threshold=cut, n=n,
                                        **changes(block, i, cut)))
        for one, two in comparisons:
            i, j = ARMS.index(one), ARMS.index(two)
            first, second = summary_metrics(blocks, name, i), summary_metrics(blocks, name, j)
            effect = block['ious'][:, i] - block['ious'][:, j]
            selected = (block['chosen'][:, i] != block['chosen'][:, j])
            paired = [changes(block, i, cut, before=j) for cut in CUTS]
            matched.append(dict(direction=name, arm=one, comparison=two, n=n,
                **deltas(first, second), different_winners=int(selected.sum()),
                different_ious=int((effect != 0).sum()),
                iou_better=int((effect > 1e-12).sum()), iou_worse=int((effect < -1e-12).sum()),
                acc25_rescue=paired[0]['total_rescues'], acc25_harm=paired[0]['total_harms'],
                acc50_rescue=paired[1]['total_rescues'], acc50_harm=paired[1]['total_harms']))
        for cut in CUTS:
            has_good = block['k16_best'] > cut
            four_miss = (block['four_best'] <= cut) & has_good
            coverage.append(dict(direction=name, threshold=cut, n=n, k16_has_good=int(has_good.sum()),
                original_four_has_good=int((block['four_best'] > cut).sum()),
                k16_covers_missing_four=int(four_miss.sum()),
                **{f'selected_in_K16_with_good_{a}': int((has_good & (block['ious'][:, i] > cut)).sum())
                   for i, a in enumerate(ARMS)},
                **{f'rescued_missing_four_{a}': int((four_miss & (block['ious'][:, i] > cut)).sum())
                   for i, a in enumerate(ARMS)}))
    return directions, transitions, matched, coverage


def target_recordings(block):
    rows = []
    for index, name in enumerate(np.unique(block['recordings']), start=1):
        mask = block['recordings'] == name
        b = {k: v[mask] for k, v in block.items()}
        before = metrics(b['ious'][:, 0])
        for i, arm in enumerate(ARMS):
            value = metrics(b['ious'][:, i])
            rows.append(dict(group=f'quad_to_drone_rec{index}', arm=arm, n=int(mask.sum()),
                **value, **deltas(value, before),
                **{f'{tag}_{k}': v for tag, cutoff in (('acc25', .25), ('acc50', .50))
                   for k, v in changes(b, i, cutoff).items()}))
    return rows


def score():
    pin = prediction_seal()
    rows = ordinary()
    assert not RESULTS.exists() and not IOU_OUT.exists()
    IOU_OUT.mkdir(mode=0o700)
    blocks, hashes, anon = {}, {}, []
    maximum = 0.
    for s, t in DIRECTIONS:
        key = f'{s}_to_{t}'
        metadata = rows[t]
        records = []
        for shard in shards(key, pin['directions'][key]):
            for j, ordinal in enumerate(shard['ordinals']):
                row = metadata[int(ordinal)]
                check_input(row)
                target = truth(row['metapath'], row['object_index'], row['caption'], t, row['suffix'])
                chosen = shard['choices'][j]
                safe = np.concatenate((shard['boxes'][j], shard['four'][j]))
                ious, err = score_boxes(safe, target)
                maximum = max(maximum, err)
                base = np.asarray(ious[:16])
                previous = float(ious[16:].max())
                if not bool(shard['active'][j]):
                    assert chosen[1] == chosen[2] == chosen[3] and chosen[4] == chosen[5] == chosen[6]
                np.testing.assert_array_equal(safe[chosen[0]], shard['four'][j, chosen[0]])
                values = base[chosen]
                outside = shard['outside'][j, chosen]
                records.append(dict(ious=values, outside=outside, chosen=chosen, four_best=previous,
                                    k16_best=float(base.max()), active=bool(shard['active'][j]),
                                    recordings=row['recording']))
                anon.append(dict(source=s, target=t, reference_active=int(shard['active'][j]),
                    four_best=previous, k16_best=float(base.max()),
                    **{a: float(values[i]) for i, a in enumerate(ARMS)},
                    **{f'outside_{a}': int(outside[i]) for i, a in enumerate(ARMS)},
                    **{f'switch_{a}': int(chosen[i] != chosen[0]) for i, a in enumerate(ARMS)}))
        local = {name: np.stack([v[name] for v in records]) for name in records[0]}
        assert len(records) == len(metadata) == pin['directions'][key]['n']
        p = IOU_OUT / f'{key}.npz'
        np.savez_compressed(p, **local)
        blocks[key] = local
        hashes[p.name] = dict(sha256=digest(p), n=len(records))
        print(key, len(records), 'scored and dual-IoU audited', flush=True)
    receipt = dict(n=18620, files=hashes, box_measurements=18620*20, threshold_disagreements=0,
                   max_absolute_two_IoU_error=maximum,prediction_seal=digest(OUT/'SEALED.json'),
                   code_sha256=digest(HERE/'score183.py'),FINAL_access=False)
    atomic(IOU_OUT/'SEALED.json', receipt)

    RESULTS.mkdir()
    direction, transitions, paired, coverage = summarize(blocks)
    save_csv(RESULTS/'direction.csv', direction)
    save_csv(RESULTS/'transitions.csv', transitions)
    save_csv(RESULTS/'matched.csv', paired)
    save_csv(RESULTS/'candidate_coverage.csv', coverage)
    save_csv(RESULTS/'quad_to_drone_recordings.csv', target_recordings(blocks['quad_to_drone']))
    intervals = recording_intervals(blocks, ARMS)
    intervals.update(contrast='method minus v1, percentage points',
        intervals={arm:[[-hi,-lo] for lo,hi in vals] for arm,vals in intervals['intervals'].items()})
    atomic(RESULTS/'recording_intervals.json', intervals)
    secrets.SystemRandom().shuffle(anon)
    save_csv(RESULTS/'anonymous_rows.csv', anon)
    atomic(RESULTS/'audit.json', dict(**receipt, endpoint_count=18620*len(ARMS),
         roster_recordings={t: len({r['recording'] for r in rows[t]}) for t in rows},
         iou_seal=digest(IOU_OUT/'SEALED.json'), source_training_runs=0, target_updates=0,
         no_hyperparameter_search=True))
    print('Scored', len(anon), 'direction records and', 18620*len(ARMS), 'endpoints', flush=True)


if __name__ == '__main__':
    score()
