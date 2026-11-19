"""Capture features on the sealed R180 slates and label source-only competitors."""

import argparse
from time import perf_counter

import numpy as np
import torch

from ranker182 import HERE, R181, ARMS, FeatureGrounder, K, protocol
from capture181 import R180, TARGET, SOURCE, load_r180, manifest as r181_manifest, write_npz
from evaluate import DIRECTIONS, to_sensor
from features import FEATURES, observation
from readout181 import load_retriever
from roles import inventory
from scope import ROOT, PLATFORMS, evaluation_rows, file_sha, read, save, source_rows
from train import many_iou, prepare_device, read_features

FIT_FEATURES = HERE / 'source_PRIVATE'
TARGET_FEATURES = HERE / 'target_PRIVATE'


def r181_seal(root, key):
    path = root / 'SEALED.json'
    config = protocol()
    assert file_sha(path) == config[key]
    assert file_sha(R181 / 'PROTOCOL.json') == config['r181_protocol_sha256']
    seal = read(path)
    assert seal['implementation'] == r181_manifest()
    assert not seal['FINAL_access']
    return seal


def archive_block(directory, name, seal):
    path = directory / name
    assert file_sha(path) == seal['files'][name]['sha256']
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def capture_fingerprint():
    paths = [HERE / name for name in ('PROTOCOL.json', 'ranker182.py', 'capture182.py')]
    first_study = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
    paths += [first_study / name for name in ('features.py', 'roles.py', 'scope.py', 'train.py')]
    values = {str(path.relative_to(ROOT)): file_sha(path) for path in paths}
    values.update(r181_manifest())
    return values


def frozen_outputs(row, previous, i):
    np.testing.assert_array_equal(row['chosen'], previous['chosen'][i])
    assert row['incumbent'] == previous['incumbent'][i]
    np.testing.assert_allclose(row['scores'], previous['scores'][i], rtol=0, atol=2e-6)
    np.testing.assert_allclose(row['terms'], previous['terms'][i], rtol=0, atol=2e-6)


def annotated_competitors(boxes, objects, categories, scores):
    overlaps = np.stack([many_iou(boxes, obj) for obj in objects], axis=1)
    best = overlaps.argmax(axis=1)
    peak = overlaps[np.arange(len(boxes)), best]
    unique = np.isclose(peak[:, None], overlaps, atol=1e-6, rtol=0).sum(axis=1) == 1
    separated = many_iou(objects, objects[0])[best] <= .10
    target_iou = overlaps[:, 0]
    negative = target_iou <= .25
    same = negative & unique & (best != 0) & (peak > .25) & separated
    same &= categories[best] == categories[0]
    high = np.argsort(-np.where(negative, scores, -np.inf), kind='stable')[:4]
    hard = np.zeros(len(boxes), dtype=bool)
    hard[high] = same[high]
    return target_iou, same, hard


def source_expected():
    cfg = protocol()['source']
    return cfg['fit_per_source'] * len(PLATFORMS) + sum(cfg['heldout_per_source'].values())


def source_statistics(rows, inputs, same_class, hard):
    has_good = (inputs['ious'] > .25).any(axis=1)
    return dict(recordings=len({r['recording'] for r in rows}),
                positive_rows=int(has_good.sum()), same_class_negatives=int(same_class.sum()),
                high_scoring_same_class_negatives=int(hard.sum()),
                rows_with_positive_and_hard=int((has_good & hard.any(axis=1)).sum()),
                hard_pairs=int(((inputs['ious'] > .25).sum(axis=1) * hard.sum(axis=1)).sum()))


@torch.inference_mode()
def capture_source(smoke=False):
    cfg = protocol()['source']
    pin = r181_seal(SOURCE, 'r181_source_terms_sha256')
    assert pin['source_only'] and not pin['target_labels_used']
    start = perf_counter()
    output = HERE / 'source_smoke_PRIVATE' if smoke else FIT_FEATURES
    output.mkdir(mode=0o700)
    files, split_receipt = {}, {}
    for platform in PLATFORMS:
        fit_rows, cal_rows = (source_rows(platform, name) for name in ('fit', 'calibration'))
        assert not {x['recording'] for x in fit_rows} & {x['recording'] for x in cal_rows}
        assert not {x['point_sha256'] for x in fit_rows} & {x['point_sha256'] for x in cal_rows}
        grounder, retriever = FeatureGrounder(platform), load_retriever(platform)
        for name, rows in (('fit', fit_rows), ('calibration', cal_rows)):
            key = f'{platform}_{name}.npz'
            cached = read_features(FEATURES, platform, name)
            natural = torch.where(cached['origin'] < 2)[0]
            expected = cfg['fit_per_source'] if name == 'fit' else cfg['heldout_per_source'][platform]
            assert len(rows) == len(natural) == expected
            q = {k: cached[k][natural] for k in ('anchor_query', 'token_proj', 'anchors', 'prior')}
            ranks = retriever(q).cpu().numpy()
            pool = q['anchors'].cpu().numpy()
            target = cached['gt'][natural, 0].cpu().numpy()
            incumbent = cached['v1_box'][natural].cpu().numpy()
            del cached, q
            old = archive_block(SOURCE, key, pin)
            n = 4 if smoke else expected
            features, same_class, hard = [], [], []
            for i, row in enumerate(rows[:n]):
                assert file_sha(row['pointpath']) == row['point_sha256']
                current = grounder.collect(observation(row), ranking=ranks[i])
                frozen_outputs(current, old, i)
                original = current['boxes']
                np.testing.assert_array_equal(original, pool[i, current['slate']])
                np.testing.assert_array_equal(original[current['incumbent']], incumbent[i])
                boxes, classes = inventory(row)
                np.testing.assert_array_equal(boxes[0], target[i])
                ious, same, high = annotated_competitors(original, boxes, classes, old['scores'][i, :, 0])
                np.testing.assert_allclose(ious, old['ious'][i], rtol=0, atol=1e-12)
                np.testing.assert_array_equal(ious > .25, old['ious'][i] > .25)
                np.testing.assert_array_equal(ious > .50, old['ious'][i] > .50)
                features.append(current['features'])
                same_class.append(same)
                hard.append(high)
                if (i + 1) % 80 == 0 or i + 1 == n:
                    print(platform, name, i + 1, n, round(perf_counter() - start, 1), flush=True)
            same_class, hard = np.stack(same_class), np.stack(hard)
            path = output / key
            inputs = {k: old[k][:n] for k in ('ious', 'scores', 'chosen', 'incumbent', 'has_reference')}
            write_npz(path, dict(inputs, features=np.stack(features), same_class=same_class, hard=hard))
            files[key] = dict(n=n, sha256=file_sha(path))
            split_receipt[key] = source_statistics(rows[:n], inputs, same_class, hard)
            print(platform, name, split_receipt[key], flush=True)
        del grounder, retriever
        torch.cuda.empty_cache()
    n = sum(v['n'] for v in files.values())
    assert n == (24 if smoke else source_expected())
    save(output / 'SEALED.json', dict(n=n, files=files, implementation=capture_fingerprint(),
         sources=split_receipt, source_only=True, target_labels_read=False, FINAL_access=False,
         r181_source_seal=protocol()['r181_source_terms_sha256'], elapsed_seconds=perf_counter() - start))


def recover_source():
    """Seal complete source captures after the original total-count check failed."""
    start = perf_counter()
    pin = r181_seal(SOURCE, 'r181_source_terms_sha256')
    assert pin['source_only'] and not pin['target_labels_used']
    assert not (FIT_FEATURES / 'SEALED.json').exists()
    original = HERE / 'capture182_attempt002.py'
    fields = ('ious', 'scores', 'chosen', 'incumbent', 'has_reference')
    files, summaries = {}, {}
    for platform in PLATFORMS:
        for split in ('fit', 'calibration'):
            name = f'{platform}_{split}.npz'
            rows = source_rows(platform, split)
            old = archive_block(SOURCE, name, pin)
            path = FIT_FEATURES / name
            with np.load(path, allow_pickle=False) as block:
                assert set(block.files) == {*fields, 'features', 'same_class', 'hard'}
                for key in fields:
                    np.testing.assert_array_equal(block[key], old[key])
                assert block['features'].shape == (len(rows), K, protocol()['ranker']['input_width'])
                assert np.isfinite(block['features']).all()
                same, hard = block['same_class'], block['hard']
                assert same.shape == hard.shape == (len(rows), K)
                assert not (hard & ~same).any() and not (same & (block['ious'] > .25)).any()
                assert (hard.sum(axis=1) <= 4).all()
                summaries[name] = source_statistics(rows, block, same, hard)
            files[name] = dict(n=len(rows), sha256=file_sha(path))
    n = sum(x['n'] for x in files.values())
    assert n == source_expected() == pin['n']
    assert len(files) == len(list(FIT_FEATURES.glob('*.npz'))) == 6
    save(FIT_FEATURES / 'SEALED.json', dict(n=n, files=files, implementation=capture_fingerprint(),
         sources=summaries, source_only=True, target_labels_read=False, FINAL_access=False,
         r181_source_seal=protocol()['r181_source_terms_sha256'], recovered_complete_outputs=True,
         original_producer_sha256=file_sha(original), elapsed_seconds=perf_counter() - start))
    print('Validated all six already-written source blocks', n, flush=True)


def capture_target(smoke=False):
    cfg = protocol()
    prior = r181_seal(TARGET, 'r181_target_terms_sha256')
    assert not prior['target_labels_read'] and tuple(cfg['arms']) == ARMS
    assert prior['earlier_seal_sha256'] == cfg['r180_target_slate_sha256']
    rows = evaluation_rows()
    fixed = load_r180(rows)
    start = perf_counter()
    output = HERE / 'target_smoke_PRIVATE' if smoke else TARGET_FEATURES
    output.mkdir(mode=0o700)
    files, points_checked = {}, set()
    for platform in PLATFORMS:
        grounder = FeatureGrounder(platform)
        for source, target in DIRECTIONS:
            if source != platform:
                continue
            key = f'{source}_to_{target}.npz'
            archive = archive_block(TARGET, key, prior)
            old = fixed[source, target]
            np.testing.assert_array_equal(archive['ordinals'], old['ordinals'])
            n = 2 if smoke else len(old['ordinals'])
            features = []
            for i, row_id in enumerate(old['ordinals'][:n]):
                row = rows[int(row_id)]
                point = row['pointpath'], row['point_sha256']
                if point not in points_checked:
                    assert file_sha(point[0]) == point[1]
                    points_checked.add(point)
                indices = old['slate'][i, 6, :K]
                current = grounder.collect(observation(row), fixed=indices)
                frozen_outputs(current, archive, i)
                np.testing.assert_array_equal(current['base_slots'], old['slate'][i, 0, :4])
                frozen_boxes = to_sensor(current['boxes'], target, current['view'])
                np.testing.assert_array_equal(frozen_boxes, old['pool'][i, indices])
                assert indices[current['incumbent']] == old['chosen'][i, 0]
                assert indices[current['chosen'][0]] == old['chosen'][i, 6]
                features.append(current['features'])
                if (i + 1) % 64 == 0 or i + 1 == n:
                    print(source, target, i + 1, n, round(perf_counter() - start, 1), flush=True)
            path = output / key
            write_npz(path, dict(ordinals=old['ordinals'][:n], features=np.stack(features)))
            files[key] = dict(n=n, sha256=file_sha(path))
        del grounder
        torch.cuda.empty_cache()
    n = sum(v['n'] for v in files.values())
    assert n == (12 if smoke else cfg['evaluation']['direction_rows'])
    save(output / 'SEALED.json', dict(n=n, files=files, implementation=capture_fingerprint(),
         all_original_boxes_identical=n, all_v1_incumbents_identical=n, all_R181_frozen_choices_identical=n,
         r181_components_seal=cfg['r181_target_terms_sha256'], r180_slate_seal=cfg['r180_target_slate_sha256'],
         target_labels_read=False, FINAL_access=False, elapsed_seconds=perf_counter() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('source-smoke', 'source', 'seal-source', 'target-smoke', 'target'))
    phase = parser.parse_args().phase
    prepare_device()
    if phase == 'seal-source':
        recover_source()
    elif phase.startswith('source'):
        capture_source(phase.endswith('smoke'))
    else:
        capture_target(phase.endswith('smoke'))
