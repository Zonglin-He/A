"""Post-seal geometry, endpoint identity, and coverage on the ordinary R165 list."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scope167 import ARMS, HERE, ORDER, OUT, PLATFORMS, R166, WORK, atomic_json, lock, read, roster, sha, status
from audit167 import blocks, check_block, row_edges
from retrieval167 import support_matrix
from labels166 import MISSING_TRUTH, NO_OUTPUT, RIGHT, UNKNOWN, WRONG, joint_status, key_for, label_pair, mask_owners, occupied_owners, physical_labels, truth_status
from mechanisms162 import box_owner
from score165 import anonymizer, many_iou, sensor_values, table, write_table

METRICS = ('mIoU', 'Acc25', 'Acc50', 'subject_identity', 'subject_box_proxy')
SCOPES = ('all', 'parsed_direct', 'camera_horizontal', 'active_factor')
STAGES = ('a256', 'r88', 'old_joint_shortlist', 'repaired')
REAL_ARMS = ARMS[1:]


def eval_lock():
    lock()
    files = [HERE / f for f in ('audit167.py', 'score167.py', 'review167.py', 'experiment/EVALUATION.json')]
    files.append(R166 / 'EVALUATION_AUDIT.json')
    value = dict(files={str(f.relative_to(HERE.parent)): sha(f) for f in files}, FINAL_access=False)
    path = HERE / 'EVALUATION_LOCK.json'
    if path.exists():
        assert read(path) == value
    else:
        assert (HERE / 'PREDICTION_SEAL.json').exists()
        atomic_json(path, value)


def port_scope(xs, key):
    return xs if key == 'all' else [r for r in xs if int(r[key])]


def endpoint_row(base, present, found, old, raw, earlier, k, direct, edges, truth, seeds, seed_ids, names):
    # Every stage uses the same physical instance assignment as R166.
    novel = raw['novel'][k]
    novel = novel[np.isfinite(novel).all(1)]
    originals = len(old)
    geometry = np.concatenate((np.c_[old, np.zeros(originals)], novel))
    bank = raw['bank'][k]
    native = mask_owners(support_matrix(geometry, seeds, originals), seed_ids, len(names))
    archived = occupied_owners(earlier['starts'], seeds, seed_ids, len(names))
    camera = SimpleNamespace(edges=edges, role_spans=earlier['spans'], camera_intrinsic=earlier['camera_intrinsic'],
                             model_to_camera=earlier['model_to_camera'], distortion=earlier['distortion'], width=int(earlier['width']))
    output = []
    for index, (role, side) in enumerate(direct, 1):
        truth_type, referent = label_pair(truth, camera, 0, role, side, names, present)
        result = {key: base[key] for key in ('source', 'target', 'recording', 'frame', 'expression')}
        result.update(edge=index, truth=truth_type, active_factor=base['active_factor'],
                      reference_bank_size=int((bank[role] >= 0).sum()),
                      reference_novel_slots=int((bank[role] >= originals).sum()),
                      joint_changed_fixed_reference=int(raw['winners'][k, 3, role] != raw['winners'][k, 2, role]),
                      joint_changed_fixed_subject=int(raw['winners'][k, 3, 0] != raw['winners'][k, 2, 0]))
        for i, arm in enumerate(ARMS):
            s = truth_status(found[i, 0], 0)
            r = NO_OUTPUT if i == 0 else MISSING_TRUTH if referent < 0 else truth_status(found[i, role], referent)
            result.update({arm + '_subject': s, arm + '_ref': r,
                           arm + '_both': r if r in (NO_OUTPUT, MISSING_TRUTH) else joint_status(s, r),
                           arm + '_relation_observed': int(raw['satisfied'][k, 0, i, role])})
        for stage, (s, r) in {
            'a256': (native[:originals], native[:originals]),
            'r88': tuple(native[earlier['routes'][m][earlier['routes'][m] >= 0]] for m in (0, role)),
            'old_joint_shortlist': tuple(archived[m, earlier['in_beam'][m]] for m in (0, role)),
            'repaired': tuple(native[bank[m, bank[m] >= 0]] for m in (0, role)),
        }.items():
            s, r = int((s == 0).any()), int((r == referent).any()) if referent >= 0 else MISSING_TRUTH
            result.update({stage + '_subject': s, stage + '_ref': r, stage + '_both': s * r if r >= 0 else MISSING_TRUTH})
        output.append(result)
    return output


def flip_row(base, pred, i, role, ori, flipped, accepted):
    meta = {k: base[k] for k in ('source', 'target', 'recording', 'frame', 'expression')}
    meta['truth'] = accepted['truth']
    results = []
    for k, arm in enumerate(REAL_ARMS, 1):
        b, c = pred['winners'][i, k], pred['flip_winners'][i, k]
        main, other = (pred['satisfied'][i, m, k, role] for m in (0, 1))
        results.append(dict(**meta, arm=arm, subject_box_switched=int(b[0] != c[0]),
                            reference_box_switched=int(b[role] != c[role]),
                            subject_owner_changed=(int(ori[k, 0] != flipped[k, 0]) if min(ori[k, 0], flipped[k, 0]) >= 0 else -1),
                            stable_reference_owner=(int(ori[k, role] == flipped[k, role]) if min(ori[k, role], flipped[k, role]) >= 0 else -1),
                            original_relation_sign=int(main), flipped_relation_sign=int(other),
                            original_subject_identity=int(base[arm + '_subject']),
                            original_both_identity=int(accepted[arm + '_both']),
                            novel_subject=int(c[0] >= 256), novel_reference=int(c[role] >= 256)))
    return results


def direction(source, target):
    name = f'{source}_to_{target}'
    alias = anonymizer()
    previous = table(R166 / 'results/all_rows.csv')
    old = {r['expression']: r for r in previous if (r['source'], r['target']) == (source, target)}
    rows, pairs, cf = [], [], []
    error, baseline_error = 0., 0.
    inference_checks = Counter()
    params = ('seeds', 'spans', 'camera_intrinsic', 'model_to_camera', 'distortion', 'width', 'routes', 'starts', 'in_beam')
    for _, a, b, c in blocks('ordinary', source, target):
        np.testing.assert_array_equal(a['ordinals'], b['ordinals'])
        inference_checks.update(check_block(a, b, c))
        for k, index in enumerate(a['ordinals']):
            row = roster()[target][int(index)]
            _, path, slot = key_for(row)
            frame = Path(path)
            expression = alias('obs', (str(frame), slot))
            known = old[expression]
            edges, direct = row_edges(c, k)
            prev = {f: c[f][k] for f in params}
            names, truth, seeds, seed_ids = physical_labels(row, prev['seeds'])
            subject = a['boxes'][k, :, 0]
            pc = many_iou(subject, truth[0])
            alternative = sensor_values(subject, truth[0], target, int(row['suffix'] or 0))
            error = max(error, float(np.abs(pc - alternative).max()))
            assert error < 1.e-7 and all(np.array_equal(pc > u, alternative > u) for u in (.25, .50))
            baseline_error = max(baseline_error, abs(float(pc[0]) - float(known['r158_pca_iou'])))
            assert baseline_error < 1.e-11 and int(bool(direct)) == int(known['camera_horizontal'])
            owners = occupied_owners(a['boxes'][k], seeds, seed_ids, len(names))
            result = dict(source=source, target=target, recording=alias('rec', frame.parent.parent.as_posix()),
                          frame=alias('frame', frame), expression=expression, parsed_direct=int(known['parsed_direct']),
                          camera_horizontal=int(bool(direct)), active_factor=int(a['active'][k]), flip=int(a['flip'][k]),
                          subject_bank=int((a['bank'][k, 0] >= 0).sum()),
                          novel_available=int(np.isfinite(a['novel'][k]).all(1).sum()),
                          novel_in_bank=int((a['bank'][k] >= 256).any()))
            for i, arm in enumerate(ARMS):
                result.update({arm + '_iou': float(pc[i]), arm + '_subject': truth_status(owners[i, 0], 0),
                               arm + '_subject_box': truth_status(box_owner(subject[i], truth), 0),
                               arm + '_new_subject': int(a['winners'][k, i, 0] >= 256),
                               arm + '_new_ref': int((a['winners'][k, i, 1:] >= 256).any())})
            assert result['r158_pca_subject'] == int(known['r158_pca_subject'])
            rows.append(result)
            if direct:
                ep = endpoint_row(result, truth, owners, b['raw'][k], a, prev, k, direct, edges, row, seeds, seed_ids, names)
                pairs.extend(ep)
                if a['flip'][k]:
                    fl = occupied_owners(a['flip_boxes'][k], seeds, seed_ids, len(names))
                    cf.extend(flip_row(result, a, k, direct[0][0], owners, fl, ep[0]))
        if len(rows) % 1024 == 0:
            print('R167 scored', name, len(rows), flush=True)
    assert len(rows) == len(old) == len(roster()[target])
    folder = WORK / 'scalars' / name
    folder.mkdir(parents=True, exist_ok=True)
    for fname, output in (('rows.csv', rows), ('edges.csv', pairs), ('counterfactual.csv', cf)):
        write_table(folder / fname, output)
    print('R167 scored', name, len(rows), len(pairs), len(cf) // len(REAL_ARMS), flush=True)
    return dict(n=len(rows), edges=len(pairs), flips=len(cf) // len(REAL_ARMS), sensor_error=error,
                baseline_error=baseline_error, inference=dict(inference_checks),
                files={f.name: sha(f) for f in folder.glob('*.csv')})


def feature(row, arm):
    i = float(row[arm + '_iou'])
    return (i, i > .25, i > .5, int(row[arm + '_subject']) == RIGHT, int(row[arm + '_subject_box']) == RIGHT)


def group_records(rows):
    parts = {(a, b): [r for r in rows if (r['source'], r['target']) == (a, b)] for a, b in ORDER}
    defined = {f'{a}_to_{b}': [p] for (a, b), p in parts.items() if p}
    defined.update({f'target_{t}': [p for (a, b), p in parts.items() if b == t and p] for t in PLATFORMS})
    return dict(six_equal=[p for p in parts.values() if p], pooled=[rows], **defined)


def summarize(rows, edges, flips):
    metrics, paired, identity, coverage, counterfactual = [], [], [], [], []
    for scope in SCOPES:
        for name, groups in group_records(port_scope(rows, scope)).items():
            base = dict(scope=scope, group=name, n=sum(map(len, groups)))
            data = {a: [np.asarray([feature(r, a) for r in part], float) for part in groups] for a in ARMS}
            means = {a: np.mean([x.mean(0) for x in arr], 0) * 100 for a, arr in data.items()}
            for arm in ARMS:
                metrics.append(dict(**base, arm=arm, **dict(zip(METRICS, means[arm])),
                                    subject_unknown=sum(int(r[arm + '_subject']) == UNKNOWN for group in groups for r in group)))
            for control in (a for a in ARMS if a != 'joint'):
                delta = np.concatenate([x - y for x, y in zip(data['joint'], data[control])])
                values = {key + '_pp': means['joint'][k] - means[control][k] for k, key in enumerate(METRICS)}
                for k, metric in enumerate(METRICS):
                    values.update({metric + '_up': int((delta[:, k] > 1.e-10).sum()),
                                   metric + '_down': int((delta[:, k] < -1.e-10).sum())})
                paired.append(dict(**base, control=control, **values))
    for label in ('eligible_all', 'annotation_proxy', 'visual_verified'):
        truth = [r for r in edges if r['truth'] in ('annotation_proxy', 'visual_verified') and
                 (label == 'eligible_all' or r['truth'] == label)]
        for group, parts in group_records(truth).items():
            cohort = [r for p in parts for r in p]
            if not cohort:
                continue
            base = dict(truth=label, group=group, n=len(cohort))
            for arm in ARMS:
                fields = {}
                for role in ('subject', 'ref', 'both'):
                    c = Counter(int(r[arm + '_' + role]) for r in cohort)
                    fields.update({role + '_correct': c[RIGHT], role + '_wrong': c[WRONG],
                                   role + '_unknown': c[UNKNOWN], role + '_not_reported': c[NO_OUTPUT],
                                   role + '_n': len(cohort) - c[NO_OUTPUT] - c[MISSING_TRUTH]})
                identity.append(dict(**base, arm=arm, **fields))
            for stage in STAGES:
                coverage.append(dict(**base, stage=stage, **{role: sum(int(r[stage + '_' + role]) == 1 for r in cohort)
                                                          for role in ('subject', 'ref', 'both')}))
    for arm in REAL_ARMS:
        for group, parts in group_records([r for r in flips if r['arm'] == arm]).items():
            cohort = [r for p in parts for r in p]
            if not cohort:
                continue
            def count(predicate):
                return sum(predicate(r) for r in cohort)
            counterfactual.append(dict(arm=arm, group=group, n=len(cohort),
                subject_box_switched=count(lambda r: int(r['subject_box_switched']) == 1),
                reference_box_switched=count(lambda r: int(r['reference_box_switched']) == 1),
                subject_owner_changed=count(lambda r: int(r['subject_owner_changed']) == 1),
                subject_owner_unknown=count(lambda r: int(r['subject_owner_changed']) == -1),
                reference_owner_stable=count(lambda r: int(r['stable_reference_owner']) == 1),
                reference_owner_unknown=count(lambda r: int(r['stable_reference_owner']) == -1),
                original_observed_correct_sign=count(lambda r: int(r['original_relation_sign']) == 1),
                flipped_observed_correct_sign=count(lambda r: int(r['flipped_relation_sign']) == 1),
                original_sign_unknown=count(lambda r: int(r['original_relation_sign']) == 0),
                flipped_sign_unknown=count(lambda r: int(r['flipped_relation_sign']) == 0),
                both_signs_consistent=count(lambda r: int(r['original_relation_sign']) == int(r['flipped_relation_sign']) == 1)))
    return dict(metrics=metrics, paired=paired, identity=identity, coverage=coverage, counterfactual=counterfactual)


def intervals(rows):
    results = []
    for scope in ('all', 'camera_horizontal'):
        cohort = port_scope(rows, scope)
        for arm in ('fixed_reference', 'independent'):
            rng = np.random.default_rng(167)
            table_by = defaultdict(list)
            for row in cohort:
                diff = np.asarray(feature(row, 'joint'), float) - np.asarray(feature(row, arm), float)
                table_by[row['target'], row['recording'], row['source']].append(diff)
            draws = []
            clusters = {}
            for target in PLATFORMS:
                scenes = sorted({k[1] for k in table_by if k[0] == target})
                clusters[target] = len(scenes)
                if len(scenes) < 2:
                    continue
                bootstrap = rng.multinomial(len(scenes), np.full(len(scenes), 1 / len(scenes)), size=5000)
                for s in (x for x in PLATFORMS if x != target):
                    records = [table_by[target, sc, s] for sc in scenes]
                    size = np.asarray([len(v) for v in records])
                    sums = np.stack([np.sum(v, axis=0) if len(v) else np.zeros(len(METRICS)) for v in records])
                    draws.append((bootstrap @ sums) / np.maximum(bootstrap @ size, 1)[:, None])
            sampled = np.mean(draws, 0) * 100 if len(draws) == 6 else None
            for k, metric in enumerate(METRICS):
                lo, hi = np.percentile(sampled[:, k], (2.5, 97.5)) if sampled is not None else ('', '')
                results.append(dict(scope=scope, control=arm, metric=metric, lo_pp=lo, hi_pp=hi,
                                    waymo_recordings=clusters['waymo'], drone_recordings=clusters['drone'],
                                    quad_recordings=clusters['quad']))
    return results


def score():
    eval_lock()
    sealed = read(HERE / 'PREDICTION_SEAL.json')
    assert sealed['n'] == 18620 and not sealed['target_label_read'] and not sealed['FINAL_access']
    status('scoring')
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        pending = {d: pool.submit(direction, *d) for d in ORDER}
        receipts = {f'{a}_to_{b}': pending[a, b].result() for a, b in ORDER}
    rows, edges, flips = [], [], []
    for d, r in receipts.items():
        src = WORK / 'scalars' / d
        for name, destination in (('rows.csv', rows), ('edges.csv', edges), ('counterfactual.csv', flips)):
            assert sha(src / name) == r['files'][name]
            destination.extend(table(src / name))
    assert len(rows) == 18620 and len(edges) == 11378 and sum(int(r['camera_horizontal']) for r in rows) == 6864
    assert sum(r['truth'] in ('annotation_proxy', 'visual_verified') for r in edges) == 750
    assert len(flips) == sum(int(r['flip']) for r in rows) * len(REAL_ARMS)
    OUT.mkdir(exist_ok=True)
    tables = dict(all_rows=rows, relation_edges=edges, counterfactual_rows=flips,
                  paired_intervals=intervals(rows), **summarize(rows, edges, flips))
    for name, items in tables.items():
        write_table(OUT / (name + '.csv'), items)
    audit = dict(n=len(rows), edges=len(edges), flip_pairs=len(flips) // len(REAL_ARMS),
                 independently_scored_outputs=len(rows) * len(ARMS),
                 max_sensor_iou_error=max(r['sensor_error'] for r in receipts.values()),
                 max_r165_baseline_difference=max(r['baseline_error'] for r in receipts.values()),
                 no_negative_or_unknown_rows_excluded=True, directions=receipts,
                 files={f.name: sha(f) for f in OUT.glob('*.csv')},
                 prediction_seal=sha(HERE / 'PREDICTION_SEAL.json'), FINAL_access=False)
    atomic_json(HERE / 'EVALUATION_AUDIT.json', audit)
    status('counterfactual_review', sealed_predictions=18620)
    print('R167 ordinary evaluation sealed', audit['n'], audit['edges'], audit['flip_pairs'], flush=True)


if __name__ == '__main__':
    score()
