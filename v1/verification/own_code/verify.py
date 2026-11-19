"""Check the simplified v1 entry point and score only after predictions are sealed."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent / '3eed_v1_component_ablation_20261119'
PRIVATE = HERE / 'predictions_PRIVATE'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRIOR))
from score_ablation import (DIRECTIONS, PLATFORMS, recording_intervals, save_csv,
                            score_boxes, sensor_truth, summarize)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    draft = path.with_suffix('.tmp')
    draft.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(draft, path)


def inputs(protocol):
    roster = ROOT / protocol['input_registry']
    seal = read(ROOT / protocol['input_seal'])
    assert digest(roster) == seal['input_sha256'] == protocol['input_sha256']
    assert not seal['FINAL_access'] and not seal['target_labels_used_for_selection']
    assert digest(PRIOR / 'score_ablation.py') == protocol['score_helpers_sha256']
    rows = read(roster)
    assert len(rows) == len({row['id'] for row in rows}) == seal['expressions']
    assert all(sum(row['platform'] == target for row in rows) == protocol['per_target_expressions']
               for target in PLATFORMS)
    return rows


def baseline(protocol, rows):
    path = PRIOR / 'predictions_PRIVATE'
    seal = read(path / 'SEALED.json')
    assert digest(path / 'SEALED.json') == protocol['baseline_prediction_seal_sha256']
    assert seal['n'] == protocol['direction_records']
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    assert digest(path / 'LOCK.json') == seal['lock_sha256']
    assert read(path / 'LOCK.json')['registry_sha256'] == protocol['input_sha256']
    result = {}
    for source, target in DIRECTIONS:
        name = f'{source}_to_{target}'
        block = path / f'{name}.npz'
        assert digest(block) == seal['files'][block.name]['sha256']
        expected = [i for i, row in enumerate(rows) if row['platform'] == target]
        with np.load(block, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['ordinals'], expected)
            arms = data['arms'].tolist()
            subset = [arms.index(value) for value in protocol['cached_arms'].values()]
            result[name] = dict(ordinals=data['ordinals'], roles=data['roles'],
                                boxes=data['boxes'][:, subset])
    return result


def implementation():
    files = (Path(__file__), ROOT / 'crossplatform/ground.py',
             ROOT / 'crossplatform/candidate_frontend.py', ROOT / 'crossplatform/v1_evidence.py',
             ROOT / 'crossplatform/v1_decoder.py', ROOT / 'crossplatform/single_frame_sources.json')
    decoders = read(ROOT / 'crossplatform/CURRENT_METHOD.json')['decoders']
    return dict(files={str(path.relative_to(ROOT)): digest(path) for path in files},
                decoders=decoders)


def observation(row, seed):
    from crossplatform.candidate_frontend import WAYMO_VIEWS

    target = row['platform']
    view = WAYMO_VIEWS[int(row['suffix'] or 0)] if target == 'waymo' else 'F'
    return dict(target=target, points=row['pointpath'], rgb=row['imagepath'],
                calibration=row['metapath'], text=row['caption'], seed=seed, waymo_view=view)


def legacy_matcher(frontend):
    """Construct the archived control only for the six-row equivalence check."""
    paths = (ROOT / 'research/round90', ROOT / 'research/round76')
    sys.path[:0] = [str(path) for path in paths]
    from core90 import Fusion90
    from model76 import restore

    saved = read(ROOT / 'crossplatform/single_frame_sources.json')[frontend.source]['matcher']
    comparator = Fusion90(frontend.base.contrastive_align_projection_image)
    comparator.cuda().eval().requires_grad_(False)
    with np.load(frontend.asset_root / saved, allow_pickle=False) as prepared:
        restore(comparator, prepared)
    comparator.route = None
    comparator.use_interface = False
    return comparator


def check_row(model, old, frame):
    import torch
    from model76 import tensor_prefix
    from PIL import Image

    frontend = model._frontend
    point_probability = frontend._point_probability
    checks = []

    def compare(raw):
        p = point_probability(raw)
        device = p.device
        data = dict(image=torch.full((1, 3, 256, 384), .5, device=device))
        for key, count in (('p2i', raw['vis0'].shape[1]), ('i2p', 1536)):
            data[key] = torch.full((count, 1), -1, dtype=torch.int64, device=device)
            data[key + '_weight'] = torch.zeros_like(data[key], dtype=torch.float32)
        reference, _, detail = old.main(frontend.base, tensor_prefix(raw), data, 'independent')
        assert not detail['late_RGB_delta'].count_nonzero().item()
        torch.testing.assert_close(p, reference, rtol=0, atol=0)
        checks.append(int(p.numel()))
        return p

    with patch.object(frontend, '_point_probability', compare):
        selected = model.predict(**frame)
    assert checks == [256]
    with TemporaryDirectory(prefix='v1-camera-dimensions-') as temp:
        replacement = Path(temp) / 'blank.png'
        with Image.open(frame['rgb']) as image:
            Image.new('RGB', image.size, 'black').save(replacement)
        np.testing.assert_array_equal(selected, model.predict(**dict(frame, rgb=replacement)))


def predict(protocol, rows, prior):
    import torch
    from crossplatform.ground import Grounder

    PRIVATE.mkdir(mode=0o700)
    lock = dict(protocol_sha256=digest(HERE / 'protocol.json'), implementation=implementation(),
                input_sha256=protocol['input_sha256'], FINAL_access=False, target_labels_read=False)
    write_json(PRIVATE / 'LOCK.json', lock)
    files = {}
    checked = []
    started = perf_counter()
    for source in PLATFORMS:
        model = Grounder(source)
        old = legacy_matcher(model._frontend)
        for target in PLATFORMS:
            if target == source:
                continue
            direction = f'{source}_to_{target}'
            ordinals = prior[direction]['ordinals']
            check_row(model, old, observation(rows[int(ordinals[0])], protocol['point_seed']))
            checked.append(direction)
            print('Point probabilities and image-size-only behavior verified', direction, flush=True)
        del old
        for target in PLATFORMS:
            if target == source:
                continue
            direction = f'{source}_to_{target}'
            ordinals = prior[direction]['ordinals']
            boxes = []
            for index, ordinal in enumerate(ordinals, 1):
                boxes.append(model.predict(**observation(rows[int(ordinal)], protocol['point_seed'])))
                if index % 128 == 0 or index == len(ordinals):
                    print(direction, index, '/', len(ordinals), round(perf_counter() - started, 1),
                          flush=True)
            path = PRIVATE / f'{direction}.npz'
            boxes = np.stack(boxes)
            assert boxes.shape == (len(ordinals), 7)
            assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, ordinals=ordinals, boxes=boxes)
            os.replace(path.with_suffix('.tmp'), path)
            files[path.name] = dict(n=len(ordinals), sha256=digest(path))
        del model
        torch.cuda.empty_cache()
    assert sum(value['n'] for value in files.values()) == protocol['direction_records']
    assert implementation() == lock['implementation']
    write_json(PRIVATE / 'SEALED.json', dict(
        n=protocol['direction_records'], files=files, lock_sha256=digest(PRIVATE / 'LOCK.json'),
        parity_and_rgb_replacement_checked=checked, elapsed_seconds=perf_counter() - started,
        FINAL_access=False, target_labels_read=False))


def score(protocol, rows, prior):
    import secrets

    lock = read(PRIVATE / 'LOCK.json')
    seal = read(PRIVATE / 'SEALED.json')
    assert lock['implementation'] == implementation()
    assert lock['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert lock['input_sha256'] == protocol['input_sha256']
    assert seal['n'] == protocol['direction_records']
    assert seal['lock_sha256'] == digest(PRIVATE / 'LOCK.json')
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    assert len(set(seal['parity_and_rgb_replacement_checked'])) == len(DIRECTIONS)
    predictions = {}
    for direction, previous in prior.items():
        path = PRIVATE / f'{direction}.npz'
        entry = seal['files'][path.name]
        assert entry['sha256'] == digest(path) and entry['n'] == len(previous['ordinals'])
        with np.load(path, allow_pickle=False) as values:
            np.testing.assert_array_equal(values['ordinals'], previous['ordinals'])
            boxes = values['boxes']
        assert boxes.shape == (len(previous['ordinals']), 7)
        assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
        predictions[direction] = np.concatenate((previous['boxes'], boxes[:, None]), axis=1)

    arms = (*protocol['cached_arms'], 'v1')
    max_error = 0.
    evaluated = {}
    anonymous = []
    for direction, boxes in predictions.items():
        source, target = direction.split('_to_')
        block = prior[direction]
        ious, recordings = [], []
        for j, ordinal in enumerate(block['ordinals']):
            row = rows[int(ordinal)]
            gt = sensor_truth(target, row['metapath'], row['caption'], row['suffix'])
            values, error = score_boxes(boxes[j], gt)
            max_error = max(max_error, error)
            ious.append(values)
            recordings.append(row['scene_cluster'])
            ref, left_right = block['roles'][j]
            anonymous.append(dict(source=source, target=target, reference=int(ref),
                                  direct_left_right=int(left_right),
                                  **{arm: float(value) for arm, value in zip(arms, values)}))
        evaluated[direction] = dict(ious=np.stack(ious), roles=block['roles'],
                                    recordings=np.asarray(recordings))
        print('Independently scored', direction, len(ious), 'records', flush=True)
    out = HERE / 'results'
    out.mkdir()
    endpoints, paired = summarize(evaluated, arms, protocol['score_subsets'])
    save_csv(out / 'endpoints.csv', endpoints)
    save_csv(out / 'paired.csv', paired)
    secrets.SystemRandom().shuffle(anonymous)
    save_csv(out / 'anonymous_rows.csv', anonymous)
    write_json(out / 'recording_intervals.json', recording_intervals(evaluated, arms))
    write_json(out / 'audit.json', dict(
        directions=len(DIRECTIONS), n=len(anonymous), endpoints=len(anonymous) * len(arms),
        new_predictions=len(anonymous), threshold_disagreements=0,
        max_independent_absolute_iou_error=max_error,
        unique_truth_queries=sensor_truth.cache_info().misses,
        baseline_seal_sha256=protocol['baseline_prediction_seal_sha256'],
        new_seal_sha256=digest(PRIVATE / 'SEALED.json'),
        source_probability_parity=seal['parity_and_rgb_replacement_checked'],
        rgb_pixel_replacement_invariance=seal['parity_and_rgb_replacement_checked'],
        all_predictions_verified_before_grounding_labels=True,
        source_training_runs=0, target_parameter_updates=0, FINAL_access=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('predict', 'score'))
    args = cli.parse_args()
    protocol = read(HERE / 'protocol.json')
    rows = inputs(protocol)
    prior = baseline(protocol, rows)
    action = predict if args.phase == 'predict' else score
    action(protocol, rows, prior)
