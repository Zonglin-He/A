"""Post-seal task scoring and independent yaw-box geometry audit for the v1 ablation."""
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/round6/source_validation_scoring'), str(ROOT / 'tools')]
from batch_yaw_iou import many_iou
from obb_iou import oriented_iou

PLATFORMS = ('waymo', 'drone', 'quad')
DIRECTIONS = tuple((s, t) for s in PLATFORMS for t in PLATFORMS if s != t)
WAYMO_ANGLES = (0, -45, 45, -90, 90)
METRICS = ('Acc25', 'Acc50', 'mIoU')
DATA = ROOT / 'crossplatform/dataset_full/3eed_dataset'


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        out.writeheader()
        out.writerows(rows)


def save_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')


def verify_seal():
    protocol = read(HERE / 'protocol.json')
    path = ROOT / protocol['input_registry']
    assert digest(path) == protocol['input_sha256']
    original = read(ROOT / protocol['input_seal'])
    assert original['input_sha256'] == protocol['input_sha256'] and original['FINAL_access'] is False
    rows = read(path)
    private = HERE / 'predictions_PRIVATE'
    seal = read(private / 'SEALED.json')
    lock = read(private / 'LOCK.json')
    assert seal['n'] == protocol['direction_records']
    assert seal['lock_sha256'] == digest(private / 'LOCK.json')
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    assert lock['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert lock['registry_sha256'] == protocol['input_sha256']
    assert digest(ROOT / 'crossplatform/CURRENT_METHOD.json') == lock['method_sha256']
    assert all(digest(ROOT / p) == h for p, h in lock['implementation'].items())
    arms = tuple(protocol['arms'])
    assert tuple(lock['arms']) == arms
    assert set(seal['files']) == {f'{s}_to_{t}.npz' for s, t in DIRECTIONS}
    pending = {}
    for s, t in DIRECTIONS:
        name = f'{s}_to_{t}.npz'
        pin = seal['files'][name]
        actual = private / name
        assert digest(actual) == pin['sha256']
        with np.load(actual, allow_pickle=False) as saved:
            assert tuple(saved['arms']) == arms
            block = {key: saved[key] for key in ('ordinals', 'boxes', 'roles', 'same')}
        expected = np.flatnonzero([r['platform'] == t for r in rows])
        np.testing.assert_array_equal(block['ordinals'], expected)
        size = protocol['per_target_expressions']
        assert expected.size == pin['n'] == size
        assert block['boxes'].shape == (size, len(arms), 7)
        assert block['roles'].shape == (size, 2) and block['same'].shape == (size, len(arms))
        assert np.isfinite(block['boxes']).all() and (block['boxes'][..., 3:6] > 0).all()
        pending[s, t] = block
    return protocol, rows, digest(private / 'SEALED.json'), pending


def normalize(caption):
    return ' '.join(caption.replace(',', ' ,').split())


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@lru_cache(maxsize=1536)
def sensor_truth(platform, metapath, caption, suffix):
    path = Path(metapath)
    assert path.name == 'meta_info.json' and path.parents[2] == DATA / platform
    labelled = read(path)['ground_info']
    matches = [r['bbox_3d'] for r in labelled if 'caption' in r and
               normalize(r['caption']) == normalize(caption)]
    if not matches:
        raise ValueError(f'No exact registered-caption match for target {platform}')
    if any(r != matches[0] for r in matches[1:]):
        raise ValueError(f'Ambiguous registered-caption match for target {platform}')
    raw = np.asarray(matches[0], np.float64)
    assert raw.shape == ((7,) if platform == 'waymo' else (9,))
    box = raw[:7].copy()
    # The historical source-validation loader stores the canonical target in
    # float32 after applying the platform transform. Undo that transform for
    # scoring the deployed sensor-frame predictions.
    if platform == 'waymo':
        angle = np.deg2rad(WAYMO_ANGLES[int(suffix or 0)])
        box[:3] = rotation(angle) @ box[:3]
        box[6] += angle
        box = box.astype(np.float32).astype(np.float64)
        box[:3] = box[:3] @ rotation(angle)
        box[6] -= angle
    elif platform == 'drone':
        box[2] += 1.8
        box = box.astype(np.float32).astype(np.float64)
        box[2] -= 1.8
    else:
        assert platform == 'quad'
        box = box.astype(np.float32).astype(np.float64)
    return box


def score_boxes(pred, gt):
    unique, order = np.unique(pred, axis=0, return_inverse=True)
    first = many_iou(unique, gt)
    gt_rotation = rotation(gt[6])
    gt_half_extent = np.abs(gt_rotation) @ gt[3:6] / 2
    second = np.zeros(len(unique))
    for i, box in enumerate(unique):
        r = rotation(box[6])
        half_extent = np.abs(r) @ box[3:6] / 2
        if np.any(np.abs(box[:3] - gt[:3]) > half_extent + gt_half_extent + 1e-10):
            continue
        second[i] = oriented_iou(box[:3], box[3:6], r, gt[:3], gt[3:6], gt_rotation)
    err = float(np.max(np.abs(first - second)))
    assert err < 1e-7 and all(np.array_equal(first > t, second > t) for t in (.25, .5))
    return first[order], err


def as_metrics(ious):
    return np.stack((ious > .25, ious > .50, ious), axis=-1).astype(np.float64)


def paired_metrics(v1, other):
    delta = v1 - other
    record = dict(iou_better=int(np.count_nonzero(delta > 1e-12)),
                  iou_worse=int(np.count_nonzero(delta < -1e-12)),
                  iou_tied=int(np.count_nonzero(np.abs(delta) <= 1e-12)))
    for threshold, tag in ((.25, '25'), (.50, '50')):
        a, b = v1 > threshold, other > threshold
        record['rescue' + tag] = int(np.count_nonzero(a & ~b))
        record['harm' + tag] = int(np.count_nonzero(~a & b))
    return record


def mask_roles(role_flags, subset):
    ref, direct = role_flags.T
    return {'all': np.ones(len(ref), bool), 'reference_mention': ref,
            'direct_left_right': direct, 'no_reference': ~ref}[subset]


def summarize(scored, arms, subsets):
    rows, changes = [], []
    i = arms.index('v1')
    groups = [('six_equal', tuple(scored)), *((key, (key,)) for key in scored)]
    for group, keys in groups:
        for subset in subsets:
            blocks = [scored[key]['ious'][mask_roles(scored[key]['roles'], subset)] for key in keys]
            assert all(len(x) for x in blocks)
            n = sum(map(len, blocks))
            metric = np.mean([as_metrics(x).mean(axis=0) for x in blocks], axis=0) * 100
            pooled = np.concatenate(blocks)
            for j, arm in enumerate(arms):
                rows.append(dict(group=group, subset=subset, arm=arm, n=n,
                                 **dict(zip(METRICS, metric[j]))))
                if i == j:
                    continue
                effect = dict(zip(('delta_' + x + '_pp' for x in METRICS), metric[i] - metric[j]))
                changes.append(dict(group=group, subset=subset, comparison=arm, n=n,
                                    **effect, **paired_metrics(pooled[:, i], pooled[:, j])))
    return rows, changes


def recording_intervals(scored, arms, repeats=2000):
    rng = np.random.default_rng(17420261119)
    by_target = []
    for target in PLATFORMS:
        choices = [scored[f'{source}_to_{target}'] for source in PLATFORMS if source != target]
        np.testing.assert_array_equal(choices[0]['recordings'], choices[1]['recordings'])
        scene, which = np.unique(choices[0]['recordings'], return_inverse=True)
        counts = np.bincount(which)
        values = np.stack([as_metrics(x['ious']) for x in choices], axis=1)
        sums = np.zeros((len(scene),) + values.shape[1:], np.float64)
        np.add.at(sums, which, values)
        weights = rng.multinomial(len(scene), np.repeat(1 / len(scene), len(scene)), size=repeats)
        recorded = np.einsum('rs,sdak->rdak', weights, sums)
        by_target.append((recorded / (weights @ counts)[:, None, None, None]).mean(axis=1))
    samples = np.mean(by_target, axis=0)
    deltas = (samples[:, arms.index('v1'), None] - samples) * 100
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    return dict(method='shared target-recording bootstrap for the two source directions',
                replicates=repeats, descriptive_only=True, metrics=list(METRICS),
                intervals={arm: bounds[:, j, :].T.tolist() for j, arm in enumerate(arms)})


def main():
    protocol, records, seal_sha, inputs = verify_seal()
    arms = tuple(protocol['arms'])
    score_dir = HERE / 'results'
    score_dir.mkdir(mode=0o755)
    scored = {}
    people = []
    evidence = []
    max_error = 0.
    for (source, target), block in inputs.items():
        direction = f'{source}_to_{target}'
        measurements = []
        record_group = []
        n = len(block['ordinals'])
        for i, ordinal in enumerate(block['ordinals']):
            row = records[int(ordinal)]
            gt = sensor_truth(target, row['metapath'], row['caption'], row['suffix'])
            error = score_boxes(block['boxes'][i], gt)
            values, discrepancy = error
            max_error = max(max_error, discrepancy)
            record_group.append(row['scene_cluster'])
            measurements.append(values)
            people.append(dict(source=source, target=target,
                               reference=int(block['roles'][i, 0]), direct_left_right=int(block['roles'][i, 1]),
                               **{a: float(values[j]) for j, a in enumerate(arms)}))
        scored[direction] = dict(ious=np.stack(measurements), roles=block['roles'],
                                 recordings=np.asarray(record_group))
        evidence.append(dict(direction=direction, n=n, reference_mentions=int(block['roles'][:, 0].sum()),
                             direct_left_right=int(block['roles'][:, 1].sum()),
                             v1_direct_left_right_same_box=int(
                                 np.count_nonzero(block['roles'][:, 1] & block['same'][:, arms.index('v1')])),
                             recordings=len(set(record_group))))
        print('Independent geometry verified', direction, n, 'direction records', flush=True)
    aggregates, changes = summarize(scored, arms, protocol['score_subsets'])
    save_csv(score_dir / 'endpoints.csv', aggregates)
    save_csv(score_dir / 'paired.csv', changes)
    save_csv(score_dir / 'coverage.csv', evidence)
    # Publication row numbers do not correspond to private manifest order.
    secrets.SystemRandom().shuffle(people)
    save_csv(score_dir / 'anonymous_rows.csv', people)
    save_json(score_dir / 'recording_intervals.json', recording_intervals(scored, arms))
    save_json(score_dir / 'audit.json', dict(
        n=len(people), arms=len(arms), endpoints=len(people) * len(arms),
        full_seal_sha256=seal_sha, scorer_sha256=digest(Path(__file__)),
        unique_truth_queries=sensor_truth.cache_info().misses,
        max_independent_absolute_iou_error=max_error, threshold_disagreements=0,
        FINAL_access=False, all_predictions_verified_before_grounding_labels=True,
        recordings=Counter(t for t, _ in {(r['platform'], r['scene_cluster']) for r in records}),
        published_only_aggregate_and_anonymous_metrics=True))


if __name__ == '__main__':
    main()
