"""Seal all decisions, then join them with the verified same-box IoU receipts."""

import argparse
import csv

import secrets

import numpy as np
import torch

from ranker182 import HERE, R181, WIDTH, K, ARMS, protocol, readouts
from capture182 import TARGET, TARGET_FEATURES, archive_block, capture_fingerprint, r181_seal
from capture181 import write_npz
from fit182 import TRAINED, evidence_scores, fitted, load_heads
from score181 import TRUTH, transitions
from scope import PLATFORMS, file_sha, read, save
from evaluate import DIRECTIONS, recording_intervals, save_csv
from train import metrics, prepare_device

PREDICTIONS = HERE / 'predictions_PRIVATE'
RESULTS = HERE / 'results'


def feature_seal():
    cfg = protocol()
    seal = read(TARGET_FEATURES / 'SEALED.json')
    assert seal['implementation'] == capture_fingerprint()
    assert seal['n'] == cfg['evaluation']['direction_rows']
    assert seal['r180_slate_seal'] == cfg['r180_target_slate_sha256']
    assert seal['r181_components_seal'] == cfg['r181_target_terms_sha256']
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    return seal


def predict_pin():
    return {p: file_sha(HERE / p) for p in ('PROTOCOL.json', 'ranker182.py', 'fit182.py', 'replay182.py')}


def rebuild_choice(residual, generic, frozen):
    res = readouts(residual, generic, frozen['scores'], frozen['incumbent'])
    return np.column_stack((frozen['incumbent'], frozen['chosen'],
                            *(res[name] for name in ARMS[4:]))).astype(np.int8)


@torch.inference_mode()
def predict():
    config = protocol()
    assert tuple(config['arms']) == ARMS and config['backbone_updates'] == 0
    fit = fitted()
    seal = feature_seal()
    old = r181_seal(TARGET, 'r181_target_terms_sha256')
    assert not old['target_labels_read']
    PREDICTIONS.mkdir(mode=0o700)
    record = {}
    for source in PLATFORMS:
        head = load_heads(source)
        for s, t in DIRECTIONS:
            if s != source:
                continue
            name = f'{s}_to_{t}.npz'
            data = archive_block(TARGET_FEATURES, name, seal)
            frozen = archive_block(TARGET, name, old)
            np.testing.assert_array_equal(data['ordinals'], frozen['ordinals'])
            n = config['evaluation']['rows_per_direction']
            assert data['features'].shape == (n, K, WIDTH)
            hard, generic = (evidence_scores(head[k], data) for k in ('hard', 'generic'))
            assert np.isfinite(hard).all() and np.isfinite(generic).all()
            chosen = rebuild_choice(hard, generic, frozen)
            assert chosen.shape == (n, len(ARMS)) and ((chosen >= 0) & (chosen < K)).all()
            path = PREDICTIONS / name
            write_npz(path, dict(chosen=chosen, ordinals=data['ordinals'], hard_residual=hard,
                                 generic_residual=generic))
            record[name] = dict(n=n, sha256=file_sha(path))
            print(s, t, n, 'predictions', flush=True)
    assert len(record) == len(DIRECTIONS) and sum(x['n'] for x in record.values()) == seal['n']
    save(PREDICTIONS / 'SEALED.json', dict(n=seal['n'], files=record, arms=list(ARMS),
         target_features_seal=file_sha(TARGET_FEATURES / 'SEALED.json'),
         old_terms_seal=config['r181_target_terms_sha256'], fitted_seal=file_sha(TRAINED / 'FITTED.json'),
         source_only=fit['source_only'], implementation=predict_pin(), backbone_updates=0, target_updates=0,
         target_labels_read=False, FINAL_access=False))


def read_predictions():
    root = PREDICTIONS / 'SEALED.json'
    pin = read(root)
    assert pin['implementation'] == predict_pin() and tuple(pin['arms']) == ARMS
    assert pin['fitted_seal'] == file_sha(TRAINED / 'FITTED.json')
    assert pin['target_features_seal'] == file_sha(TARGET_FEATURES / 'SEALED.json')
    assert pin['old_terms_seal'] == protocol()['r181_target_terms_sha256']
    assert pin['n'] == protocol()['evaluation']['direction_rows']
    assert pin['source_only'] and not pin['target_labels_read'] and not pin['FINAL_access']
    return pin


def compare(values, indices, first, second):
    i, j = ARMS.index(first), ARMS.index(second)
    a, b = values[:, i], values[:, j]
    return dict(arm=first, against=second, n=len(a), different_winner=int((indices[:, i] != indices[:, j]).sum()),
         different_iou=int((a != b).sum()), iou_gains=int((a > b + 1e-12).sum()),
         iou_harms=int((a < b - 1e-12).sum()), delta_mIoU=float((a - b).mean() * 100),
         **{f'{name}_{k}': int(np.count_nonzero(x)) for name, th in (('Acc25', .25), ('Acc50', .50))
            for k, x in (('rescue', (b <= th) & (a > th)), ('harm', (b > th) & (a <= th)))})


def fixed_cohort(data):
    good = data['all_ious'] > .25
    four_miss = (data['original_best'] <= .25) & good.any(axis=1)
    reachable = four_miss & (good & data['reachable']).any(axis=1)
    return dict(reachable_count=int(reachable.sum()), covered_four_miss=int(four_miss.sum()),
                passes_381={a: int((data['ious'][reachable, i] > .25).sum()) for i, a in enumerate(ARMS)},
                passes_covered_four_miss={a: int((data['ious'][four_miss, i] > .25).sum())
                                          for i, a in enumerate(ARMS)})


def score():
    config = protocol()
    pred = read_predictions()
    prior = r181_seal(TARGET, 'r181_target_terms_sha256')
    features = feature_seal()
    truth_path = TRUTH / 'SEALED.json'
    assert file_sha(truth_path) == config['r181_audited_iou_sha256']
    truth = read(truth_path)
    assert truth['terms_sha256'] == pred['old_terms_seal'] and truth['n'] == pred['n'] == 3072
    assert not truth['FINAL_access'] and truth['threshold_disagreements'] == 0
    truth_files = dict(files={name: dict(sha256=digest) for name, digest in truth['files'].items()})
    blocks, anonymous = {}, []
    for s, t in DIRECTIONS:
        d = f'{s}_to_{t}'
        name = d + '.npz'
        prepared = archive_block(PREDICTIONS, name, pred)
        frozen = archive_block(TARGET, name, prior)
        gt = archive_block(TRUTH, name, truth_files)
        rows = np.arange(len(prepared['chosen']))
        np.testing.assert_array_equal(prepared['ordinals'], frozen['ordinals'])
        idx = rebuild_choice(prepared['hard_residual'], prepared['generic_residual'], frozen)
        np.testing.assert_array_equal(prepared['chosen'], idx)
        assert gt['ious'].shape == (len(rows), K)
        selected = gt['ious'][rows[:, None], idx]
        outside = gt['outside'][rows[:, None], idx]
        changed = idx != idx[:, :1]
        blocks[d] = dict(ious=selected, selected=idx, actual=changed, outside=outside,
                         original_best=gt['original_best'], recordings=gt['recordings'],
                         all_ious=gt['ious'], reachable=frozen['can_win'])
        for j in rows:
            anonymous.append(dict(source=s, target=t, reference_active=int(frozen['has_reference'][j]),
                original_best=float(gt['original_best'][j]),
                **{a: float(selected[j, i]) for i, a in enumerate(ARMS)},
                **{f'switch_{a}': int(changed[j, i]) for i, a in enumerate(ARMS)},
                **{f'outside_{a}': int(outside[j, i]) for i, a in enumerate(ARMS)}))
    merged = {k: np.concatenate([b[k] for b in blocks.values()]) for k in next(iter(blocks.values()))}
    totals, paired, matched, cohort = [], [], [], {}
    comparisons = [('rank_joint', other) for other in
                   ('v1', 'frozen_joint', 'rank_subject', 'rank_mismatch', 'generic_joint')]
    for direction, block in (*blocks.items(), ('six_equal', merged)):
        for i, arm in enumerate(ARMS):
            val = block['ious'][:, i]
            delta = val - block['ious'][:, 0]
            totals.append(dict(direction=direction, arm=arm, **metrics(val),
                 switches=int(block['actual'][:, i].sum()), iou_better=int((delta > 1e-12).sum()),
                 iou_worse=int((delta < -1e-12).sum())))
            for th in (.25, .50):
                paired.append(dict(direction=direction, arm=arm, threshold=th,
                                   n=len(val), **transitions(block, i, th)))
        for first, second in comparisons:
            matched.append(dict(direction=direction, **compare(block['ious'], block['selected'], first, second)))
        cohort[direction] = fixed_cohort(block)
    assert cohort['six_equal']['reachable_count'] == 381 and cohort['six_equal']['covered_four_miss'] == 664
    with (R181 / 'results/direction.csv').open() as stream:
        old = {(x['direction'], x['arm']): x for x in csv.DictReader(stream)}
    for row in totals:
        if row['arm'] in ARMS[:4]:
            for metric in ('Acc25', 'Acc50', 'mIoU'):
                assert row[metric] == float(old[row['direction'], row['arm']][metric])
    ci = recording_intervals(blocks, ARMS)
    ci['contrast'] = 'method minus v1, percentage points'
    ci['intervals'] = {a: [[-hi, -lo] for lo, hi in val] for a, val in ci['intervals'].items()}
    RESULTS.mkdir()
    save_csv(RESULTS / 'direction.csv', totals)
    save_csv(RESULTS / 'transitions.csv', paired)
    save_csv(RESULTS / 'matched.csv', matched)
    save(RESULTS / 'cohorts.json', cohort)
    save(RESULTS / 'recording_intervals.json', ci)
    secrets.SystemRandom().shuffle(anonymous)
    save_csv(RESULTS / 'anonymous_rows.csv', anonymous)
    save(RESULTS / 'audit.json', dict(n=len(anonymous), endpoint_count=len(anonymous) * len(ARMS),
         all_box_slates_match_R180=features['all_original_boxes_identical'],
         all_v1_winners_match=features['all_v1_incumbents_identical'],
         preserved_R181_frozen_winners=features['all_R181_frozen_choices_identical'],
         independently_audited_R181_candidate_measurements=truth['exact_iou_checks'],
         threshold_disagreements=truth['threshold_disagreements'],
         reused_candidate_iou_seal=file_sha(truth_path), predictions_seal=file_sha(PREDICTIONS / 'SEALED.json'),
         recording_counts={t: len(np.unique(blocks[f'{s}_to_{t}']['recordings'])) for s, t in DIRECTIONS},
         fit_only_on_source=True, target_updates=0, FINAL_access=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('predict', 'score'))
    prepare_device()
    {'predict': predict, 'score': score}[parser.parse_args().mode]()
