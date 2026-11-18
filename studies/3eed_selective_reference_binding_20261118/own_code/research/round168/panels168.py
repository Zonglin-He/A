"""Score fixed-bank left/right interventions against archived blind identities."""
import numpy as np

from scope168 import ARMS, HERE, ORDER, OUT, R167, WORK, atomic_json, read, roster, sha
from challenge167 import DIR, frozen_receipt
from geometry162 import CameraX
from labels166 import RIGHT, joint_status, occupied_owners, physical_labels, truth_status
from review167 import REGISTRY, TRUTH, camera, locate
from score165 import many_iou, sensor_values, write_table
from score168 import archived, verify_seals


def pair_rows(scope, case, source, target, row, evidence, truth, review_status):
    names, boxes, seeds, ids = physical_labels(row, evidence['seeds'])
    states = {}
    if truth is not None:
        assert len(set(truth)) == 3 and all(0 <= i < len(names) for i in truth)
        for side, box_key, gold in zip(('left', 'right'), ('left_box', 'right_box'), truth[:2], strict=True):
            result = evidence[box_key]
            owners = occupied_owners(result, seeds, ids, len(names))
            ious = many_iou(result[:, 0], boxes[gold])
            other = sensor_values(result[:, 0], boxes[gold], target, int(row['suffix'] or 0))
            assert np.max(np.abs(ious - other)) < 1.e-7
            assert all(np.array_equal(ious > t, other > t) for t in (.25, .5))
            states[side] = owners, ious, gold
    for i, arm in enumerate(ARMS):
        out = dict(scope=scope, case=case, source=source, target=target, arm=arm,
                   status=review_status, valid_pair=int(truth is not None))
        for side, key in (('left', 'left_chosen'), ('right', 'right_chosen')):
            choice = evidence[key][i]
            out[side + '_refused'] = int(choice[evidence['role']] < 0)
            out.update({side + '_' + field: '' for field in ('iou', 'subject', 'reference', 'both')})
            if truth is not None:
                owners, ious, gold = states[side]
                sub = truth_status(owners[i, 0], gold)
                ref = truth_status(owners[i, evidence['role']], truth[2])
                out.update({side + '_iou': float(ious[i]), side + '_subject': sub,
                            side + '_reference': ref, side + '_both': joint_status(sub, ref)})
        out['subject_box_changed'] = int(evidence['left_chosen'][i, 0] != evidence['right_chosen'][i, 0])
        out['both_subjects'] = int(out['left_subject'] == out['right_subject'] == RIGHT) if truth is not None else ''
        out['both_endpoints_twice'] = int(out['left_both'] == out['right_both'] == RIGHT) if truth is not None else ''
        yield out


def unpack(d, k, seeds, role, original_sign):
    a, b = ('', 'flip_') if original_sign == 1 else ('flip_', '')
    return dict(seeds=seeds, role=role, left_box=d[a + 'boxes'][k], right_box=d[b + 'boxes'][k],
                left_chosen=d[a + 'chosen'][k], right_chosen=d[b + 'chosen'][k])


def check_truth(row, old, k, evidence, truth):
    _, boxes, _, _ = physical_labels(row, evidence['seeds'])
    cam = camera(old, k, ())
    proj = CameraX(cam.camera_intrinsic, cam.model_to_camera, cam.distortion, cam.width)
    left, right, reference = [proj.project(boxes[i, :3])[0] for i in truth]
    assert 0 <= left < reference < right <= 1


def score():
    ordinary = verify_seals()
    lock = read(HERE / 'EVALUATION_LOCK.json')
    assert lock['panels168.py'] == sha(HERE / 'panels168.py')
    assert lock['PREDICTION_SEAL.json'] == sha(HERE / 'PREDICTION_SEAL.json')
    assert lock['CONTROLLED_PREDICTION_SEAL.json'] == sha(HERE / 'CONTROLLED_PREDICTION_SEAL.json')
    assert sha(REGISTRY) == read(R167 / 'REVIEW_PANEL_SEAL.json')['registry']
    assert sha(TRUTH) == read(R167 / 'VISUAL_REVIEW_AUDIT.json')['review_digest']
    assert read(R167 / 'CONTROLLED_LOCK.json') == frozen_receipt()
    original = read(REGISTRY)
    labels = read(TRUTH)
    controlled = read(HERE / 'CONTROLLED_PREDICTION_SEAL.json')
    results = []
    bycase = {a['case']: a for a in original}
    for source, target in ORDER:
        items = {c['ordinal']: (c, g) for c, g in zip(original, labels, strict=True) if c['target'] == target}
        assert all(c['case'] == g['case'] for c, g in items.values())
        for d, _, _, old in archived(ordinary, source, target):
            for k, ordinal in enumerate(d['ordinals']):
                pair = items.get(ordinal)
                if pair is None:
                    continue
                case, label = pair
                row = roster()[target][int(ordinal)]
                assert d['cf'][k]
                item = unpack(d, k, old['seeds'][k], case['role'], case['sign'])
                truth = None
                if label['status'] == 'verified_pair':
                    truth = label['left'], label['right'], label['ref']
                    assert (truth[0] if case['sign'] == 1 else truth[1]) == 0
                    check_truth(row, old, k, item, truth)
                results.extend(pair_rows('archived_blind', case['case'], source, target, row, item, truth, label['status']))
        for label in read(DIR / 'ANSWERS.json'):
            case = bycase[label['case']]
            if case['target'] != target:
                continue
            k, _, old = locate(source, target, case['ordinal'])
            row = roster()[target][case['ordinal']]
            name = f"{case['case']:02d}_{source}_to_{target}.npz"
            path = WORK / 'controlled' / name
            assert sha(path) == controlled['files'][name]
            with np.load(path, allow_pickle=False) as d:
                assert tuple(d['arms']) == ARMS and d['cf'][0]
                item = unpack(d, 0, d['seeds'], 1, 1)
            truth = label['left'], label['right'], label['ref']
            check_truth(row, old, k, item, truth)
            results.extend(pair_rows('authored_pair', case['case'], source, target, row, item, truth, 'verified_pair'))

    summary = []
    for scope, attempted, valid in (('archived_blind', 40, 2), ('authored_pair', 10, 10)):
        for arm in ARMS:
            sample = [r for r in results if r['scope'] == scope and r['arm'] == arm]
            verified = [r for r in sample if r['valid_pair']]
            assert len(sample) == attempted and len(verified) == valid
            item = dict(scope=scope, arm=arm, all_attempted=attempted, valid_pairs=valid,
                        both_subjects=sum(r['both_subjects'] for r in verified),
                        both_endpoints_twice=sum(r['both_endpoints_twice'] for r in verified))
            for side in ('left', 'right'):
                item.update({side + '_' + field + '_correct': sum(r[side + '_' + field] == RIGHT for r in verified)
                             for field in ('subject', 'reference', 'both')})
                item[side + '_refused'] = sum(r[side + '_refused'] for r in sample)
                item[side + '_acc25'] = sum(r[side + '_iou'] > .25 for r in verified)
                item[side + '_acc50'] = sum(r[side + '_iou'] > .50 for r in verified)
            summary.append(item)
    details = OUT / 'identity_counterfactual_rows.csv'
    aggregate = OUT / 'identity_counterfactual_summary.csv'
    write_table(details, results)
    write_table(aggregate, summary)
    atomic_json(HERE / 'PANELS_AUDIT.json', dict(n=len(results), authored_direction_pairs=10,
                archived_attempted_direction_pairs=40, archived_verified_direction_pairs=2,
                both_sensor_sides_audited=True, visual_review_digest=sha(TRUTH),
                authored_answers_digest=sha(DIR / 'ANSWERS.json'), files={p.name: sha(p) for p in (details, aggregate)},
                FINAL_access=False))
    print('R168 verified paired identity interventions', len(results), flush=True)


if __name__ == '__main__':
    score()
