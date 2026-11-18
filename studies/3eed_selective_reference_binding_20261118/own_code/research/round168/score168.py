"""Score sealed fixed-bank replays, retaining explicit refusal and unknown identity."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scope168 import ARMS, HERE, ORDER, OUT, PLATFORMS, R166, R167, WORK, atomic_json, lock, read, roster, sha, status
from audit167 import blocks, row_edges
from replay168 import oriented
from retrieval167 import support_matrix
from labels166 import MISSING_TRUTH, RIGHT, UNKNOWN, joint_status, key_for, label_pair, mask_owners, occupied_owners, physical_labels, truth_status
from score165 import anonymizer, many_iou, sensor_values, table, write_table
from score167 import group_records
from common160 import array_sha
from predict165 import points_for

METRICS = ('mIoU', 'Acc25', 'Acc50', 'subject_identity')
TERMS = ('subject', 'reference', 'left', 'overlap')
SCOPES = ('all', 'parsed_direct', 'camera_horizontal', 'active_factor')


def verify_seals():
    lock()
    basic = read(HERE / 'PREDICTION_SEAL.json')
    cf = read(HERE / 'CONTROLLED_PREDICTION_SEAL.json')
    rule_sha = sha(HERE / 'ABSTENTION_RULE.json')
    assert basic['n'] == 18620 and cf['n'] == 10 and basic['exact_r167_replay']
    assert basic['rule'] == cf['rule'] == rule_sha and not (basic['target_label_read'] or cf['target_label_read'])
    assert not (basic['FINAL_access'] or cf['FINAL_access'])
    return basic


@lru_cache(maxsize=1)
def new_independent_truth():
    from review166 import PANEL, TRUTH
    from review167 import REGISTRY, TRUTH as TRUTH167

    resolved = defaultdict(set)
    registry = PANEL / 'REGISTRY_PRIVATE.json'
    assert sha(registry) == read(R166 / 'REVIEW_PANEL_SEAL.json')['digest']
    for case, answer in zip(read(registry), read(TRUTH), strict=True):
        assert case['case'] == answer['case']
        if answer['status'] == 'verified_boxes':
            resolved[case['target'], case['ordinal'], case['edge_number']].add(answer['reference'])
    assert sha(REGISTRY) == read(R167 / 'REVIEW_PANEL_SEAL.json')['registry']
    assert sha(TRUTH167) == read(R167 / 'VISUAL_REVIEW_AUDIT.json')['review_digest']
    for case, answer in zip(read(REGISTRY), read(TRUTH167), strict=True):
        assert case['case'] == answer['case']
        if answer['status'] == 'verified_pair':
            resolved[case['target'], case['ordinal'], 1].add(answer['ref'])
    return resolved


def archived(meta, source, target):
    folder = WORK / 'ordinary' / f'{source}_to_{target}'
    tag = f'{source}_to_{target}'
    assert sha(folder / 'SEAL.json') == meta['directions'][tag]['seal']
    receipt = read(folder / 'SEAL.json')
    for name, a, b, c in blocks('ordinary', source, target):
        path = folder / name
        assert sha(path) == receipt['files'][name]
        with np.load(path, allow_pickle=False) as new:
            d = dict(new)
        assert tuple(d['arms']) == ARMS
        np.testing.assert_array_equal(d['ordinals'], a['ordinals'])
        np.testing.assert_array_equal(d['chosen'][:, :2], a['winners'][:, [1, 3]])
        np.testing.assert_array_equal(d['boxes'][:, :2], a['boxes'][:, [1, 3]])
        yield d, a, b, c


def score_pair_status(record, owner, truth_ref, i, arm, role, selected):
    subject = truth_status(owner[i, 0], 0)
    ref = MISSING_TRUTH if truth_ref < 0 else truth_status(owner[i, role], truth_ref)
    record.update({arm + '_subject': subject, arm + '_ref': ref,
                   arm + '_both': ref if ref == MISSING_TRUTH else joint_status(subject, ref),
                   arm + '_physical_owner_unknown': int(selected >= 0 and owner[i, role] == UNKNOWN)})


def reference_order(score, candidates, owned, truth):
    if truth < 0 or not len(candidates) or not np.any(owned[candidates] == truth):
        return '', ''
    correct = owned[candidates] == truth
    ranked = np.argsort(-score, kind='stable')
    rank = int(np.flatnonzero(correct[ranked])[0] + 1)
    margin = float(score[correct].max() - score[~correct].max()) if (~correct).any() else ''
    return rank, margin


def oracle_row(row, old, raw, get_pts, orig, reference, role, sign, seeds, seed_ids, boxes, names, truth_type):
    """GT-side diagnostic only. Keep the best-scored geometry for the verified identity."""
    report = dict(**orig, truth_type=truth_type, reference_in_bank=0)
    for title in ('independent', 'predicted_reference', 'identity_oracle'):
        report.update({title + '_iou': '', title + '_subject': ''})
    allowed = np.flatnonzero(old['bank'][role] >= 0)
    physical = old['bank'][role, allowed]
    subjects = np.flatnonzero(old['bank'][0] >= 0)
    base = oriented(row, old, raw, np.array([old['bank'][0, subjects[0]]]), get_pts)[0]
    report['independent_iou'] = float(many_iou(base[None], boxes[0])[0])
    report['independent_subject'] = truth_status(occupied_owners(base[None], seeds, seed_ids, len(names))[0], 0)
    if not len(physical):
        return report
    geometries = oriented(row, old, raw, physical, get_pts)
    ids = occupied_owners(geometries, seeds, seed_ids, len(names))
    match = allowed[ids == reference]
    report['reference_in_bank'] = int(len(match)>0)
    nominees = [('predicted_reference', int(allowed[old['score'][role, allowed].argmax()]))]
    if len(match):
        nominees.append(('identity_oracle', int(match[old['score'][role, match].argmax()])))
    for title, loc in nominees:
        u = old['score'][0, subjects] + sign * old['left'][role - 1, subjects, loc] - old['overlap'][role - 1, subjects, loc]
        chosen = int(old['bank'][0, subjects[u.argmax()]])
        box = oriented(row, old, raw, np.array([chosen]), get_pts)
        report[title + '_iou'] = float(many_iou(box, boxes[0])[0])
        report[title + '_subject'] = truth_status(occupied_owners(box, seeds, seed_ids, len(names))[0], 0)
    return report


def relation_rows(row, base, old, native, earlier, new, k, original, owner, seeds, seed_ids, boxes, names, get_pts):
    edges, direct = row_edges(earlier, k)
    cam = SimpleNamespace(edges=edges, role_spans=earlier['spans'][k], width=int(earlier['width'][k]),
             camera_intrinsic=earlier['camera_intrinsic'][k], model_to_camera=earlier['model_to_camera'][k],
             distortion=earlier['distortion'][k])
    extras = old['novel'][np.isfinite(old['novel']).all(1)]
    geometry = np.concatenate((np.c_[native, np.zeros(len(native))], extras))
    bank_owner = mask_owners(support_matrix(geometry, seeds, len(native)), seed_ids, len(names))
    extra_truth = new_independent_truth()
    meta = {key: base[key] for key in ('source', 'target', 'recording', 'frame', 'expression')}
    result, paired, oracles = [], [], []
    for ix, (role, sign) in enumerate(direct, 1):
        label_type, reference = label_pair(row, cam, 0, role, sign, names, boxes)
        found = set(extra_truth.get((base['target'], int(earlier['ordinals'][k]), ix), set()))
        if label_type == 'visual_verified':
            found.add(reference)
        evaluation = dict(**meta, edge=ix, truth=label_type, independently_verified=int(len(found)==1),
                          visual_conflict=int(len(found)>1))
        r = old['bank'][role] >= 0
        pool = old['bank'][role, r]
        evaluation['physical_pool_size'] = len(pool)
        evaluation['calibrated_availability'] = float(new['gate_p'][k, role]) if len(pool) else ''
        if len(found) == 1:
            known_reference = next(iter(found))
            assert 0 < known_reference < len(names)
            kind = 'prior_image_verified' if label_type == 'visual_verified' else 'subsequent_image_verified'
            oracles.append(oracle_row(row, old, native, get_pts, dict(**meta, edge=ix), known_reference, role, sign,
                                      seeds, seed_ids, boxes, names, kind))
        else:
            known_reference = -1
        evaluation['old_rank'], evaluation['old_margin'] = reference_order(old['score'][role, r], pool, bank_owner, reference)
        evaluation['new_rank'], evaluation['new_margin'] = reference_order(new['new_score'][k, role, r], pool, bank_owner, reference)
        evaluation['visual_old_rank'], evaluation['visual_old_margin'] = reference_order(old['score'][role, r], pool, bank_owner, known_reference)
        evaluation['visual_new_rank'], evaluation['visual_new_margin'] = reference_order(new['new_score'][k, role, r], pool, bank_owner, known_reference)
        for i, arm in enumerate(ARMS):
            selected = int(new['chosen'][k, i, role])
            score_pair_status(evaluation, owner, reference, i, arm, role, selected)
            evaluation[arm + '_active_refusal'] = int(selected < 0 and len(pool)>0)
            evaluation[arm + '_empty_refusal'] = int(selected < 0 and len(pool)==0)
            v = MISSING_TRUTH if known_reference < 0 else truth_status(owner[i, role], known_reference)
            evaluation[arm + '_visual_ref'] = v
            evaluation[arm + '_visual_both'] = v if v == MISSING_TRUTH else joint_status(truth_status(owner[i, 0], 0), v)
            if new['cf'][k]:
                s_before = owner[i, 0]
                r_before = owner[i, role]
                s_after, r_after = original[i, 0], original[i, role]
                c0, c1 = new['chosen'][k, i], new['flip_chosen'][k, i]
                # Pick the candidate slots selected under each sign and re-check the cached physical direction.
                def geom(c, side):
                    u = int(np.flatnonzero(old['bank'][0] == c[0])[0])
                    v = int(np.flatnonzero(old['bank'][role] == c[role])[0])
                    return int(np.sign(side * old['left'][role - 1, u, v]))
                both_signs = (geom(c0, sign) == geom(c1, -sign) == 1)
                stable = min(r_before, r_after) >= 0 and r_before == r_after
                distinct = min(s_before, s_after) >= 0 and s_before != s_after
                paired.append(dict(**meta, arm=arm, both_signs=int(both_signs), same_known_anchor=int(stable),
                        different_known_subject=int(distinct), conjunction=int(both_signs and stable and distinct),
                        both_physically_taken=int(c0[role]>=0 and c1[role]>=0),
                        unchanged_ref_box=int(c0[role]==c1[role] and c0[role]>=0),
                        ref_owner_unknown=int(min(r_before, r_after)<0),
                        subject_box_changed=int(c0[0] != c1[0])))
        result.append(evaluation)
    return result, paired, oracles


def direction(source, target):
    seal = verify_seals()
    tag = f'{source}_to_{target}'
    alias = anonymizer()
    prior = {v['expression']: v for v in table(R167 / 'results/all_rows.csv') if (v['source'], v['target']) == (source, target)}
    rows, edges, flips, oracles, terms = [], [], [], [], []
    max_error, max_replay = 0., 0.
    for new, previous, base, earlier in archived(seal, source, target):
        for k, ordinal in enumerate(new['ordinals']):
            row = roster()[target][int(ordinal)]
            _, f, i = key_for(row)
            frame = Path(f)
            expression = alias('obs', (f, i))
            before = prior[expression]
            info = dict(source=source, target=target, expression=expression,
                        frame=alias('frame', frame), recording=alias('rec', frame.parent.parent.as_posix()),
                        **{x: int(before[x]) for x in SCOPES[1:]},
                        physical_roles=int(new['n_physical_roles'][k]),
                        effective_permutations=int(new['effective_permutations'][k]),
                        legacy_worst_physical_advantage=(float(new['legacy_advantage'][k]) if np.isfinite(new['legacy_advantage'][k]) else ''))
            names, gt, seeds, seed_ids = physical_labels(row, earlier['seeds'][k])
            outputs = new['boxes'][k]
            iou = many_iou(outputs[:, 0], gt[0])
            audited = sensor_values(outputs[:, 0], gt[0], target, int(row['suffix'] or 0))
            error = float(np.abs(audited - iou).max())
            assert error < 1.e-7 and all(np.array_equal(audited > t, iou > t) for t in (.25, .50))
            max_error = max(max_error, error)
            replay_error = float(np.abs(iou[:2] - [float(before[k+'_iou']) for k in ('independent', 'joint')]).max())
            max_replay = max(max_replay, replay_error)
            assert replay_error < 1.e-11
            owner = occupied_owners(outputs, seeds, seed_ids, len(names))
            bank = previous['bank'][k]
            physical = (bank >= 0).any(1)
            ref_roles = np.flatnonzero((bank == -1).any(1))
            chosen = new['chosen'][k]
            for n, arm in enumerate(ARMS):
                switched = chosen[n, 0] != chosen[0, 0]
                empty = chosen[n, ref_roles] < 0
                info.update({arm + '_iou': float(iou[n]), arm + '_subject': truth_status(owner[n, 0], 0),
                             arm + '_subject_switch': int(switched),
                             arm + '_active_refusals': int((empty & physical[ref_roles]).sum()),
                             arm + '_no_physical_pool': int((empty & ~physical[ref_roles]).sum()),
                             arm + '_physical_owner_unknown': int(((chosen[n, ref_roles]>=0) & (owner[n, ref_roles]==UNKNOWN)).sum())})
                if switched:
                    terms.append(dict(source=source, target=target, expression=expression, arm=arm,
                                iou_change=float(iou[n] - iou[0]), strong25_rescue=int(iou[n]>.25 and iou[0]<=.25),
                                strong25_harm=int(iou[n]<=.25 and iou[0]>.25),
                                strong50_rescue=int(iou[n]>.50 and iou[0]<=.50),
                                strong50_harm=int(iou[n]<=.50 and iou[0]>.50),
                                **{key: float(val) for key, val in zip(TERMS, new['decomposition'][k, n], strict=True)}))
            rows.append(info)
            if info['camera_horizontal']:
                old = {m: previous[m][k] for m in ('bank','score','left','overlap','winners','flip_winners','boxes','flip_boxes','novel')}

                def get_pts(row):
                    cloud, ix = points_for(row)
                    assert bytes(base['points_sha'][k]).hex() == array_sha(cloud)
                    assert bytes(base['indices_sha'][k]).hex() == array_sha(ix)
                    return cloud

                cf = occupied_owners(new['flip_boxes'][k], seeds, seed_ids, len(names)) if new['cf'][k] else owner
                added = relation_rows(row, info, old, base['raw'][k], earlier, new, k, cf, owner,
                                      seeds, seed_ids, gt, names, get_pts)
                edges.extend(added[0])
                flips.extend(added[1])
                oracles.extend(added[2])
    assert len(rows) == len(prior)
    folder = WORK / 'scalars' / tag
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in zip(('rows','edges','flips','oracle','term_deltas'), (rows,edges,flips,oracles,terms), strict=True):
        f = folder / (name + '.csv')
        if values:
            write_table(f, values)
            files[name] = sha(f)
    print('R168 scored', tag, len(rows), len(edges), len(oracles), len(terms), flush=True)
    return dict(n=len(rows), edges=len(edges), max_sensor_error=max_error, max_exact_replay_error=max_replay, files=files)


def feature(r, a):
    u = float(r[a + '_iou'])
    return u, float(u>.25), float(u>.50), float(int(r[a+'_subject'])==RIGHT)


def summaries(rows, edges, flips, oracles, terms):
    m, p, identity, cf, oracle, damage, ranks, oracle_pairs = [], [], [], [], [], [], [], []
    for kind in SCOPES:
        values = rows if kind == 'all' else [r for r in rows if r[kind] == '1']
        for group, grouped in group_records(values).items():
            if not grouped:
                continue
            data = {a: [np.asarray([feature(r, a) for r in g]) for g in grouped] for a in ARMS}
            mean = {a: 100 * np.mean([v.mean(0) for v in items], axis=0) for a, items in data.items()}
            merged = [v for part in grouped for v in part]
            for arm in ARMS:
                m.append(dict(scope=kind, group=group, arm=arm, n=len(merged), **dict(zip(METRICS, mean[arm])),
                             physical_roles=sum(int(r['physical_roles']) for r in merged),
                             active_refusals=sum(int(r[arm+'_active_refusals']) for r in merged),
                             empty_pool_count=sum(int(r[arm+'_no_physical_pool']) for r in merged),
                             emitted_owner_unknown=sum(int(r[arm+'_physical_owner_unknown']) for r in merged),
                             subject_switches=sum(int(r[arm+'_subject_switch']) for r in merged)))
                for control in ('independent','r167_joint','old_ref_abstain','scrambled_ref_abstain'):
                    if arm == control:
                        continue
                    x = np.concatenate([a-b for a,b in zip(data[arm],data[control],strict=True)])
                    d = dict(scope=kind, group=group, arm=arm, control=control, n=len(merged))
                    for j,k in enumerate(METRICS):
                        d.update({k+'_pp': mean[arm][j] - mean[control][j], k+'_rescues': int((x[:,j]>1.e-10).sum()),
                                  k+'_harms': int((x[:,j]< -1.e-10).sum())})
                    p.append(d)
    for truth in ('original_750', 'visual_independent'):
        sample = [e for e in edges if (e['truth'] in ('annotation_proxy','visual_verified') if truth=='original_750'
                  else e['independently_verified']=='1')]
        for group, sets in group_records(sample).items():
            cohort = [r for g in sets for r in g]
            if not cohort:
                continue
            for arm in ARMS:
                s = arm + ('_visual' if truth=='visual_independent' else '')
                identity.append(dict(scope=truth, group=group, arm=arm, n=len(cohort),
                    subject_correct=sum(r[arm+'_subject']=='1' for r in cohort),
                    ref_correct=sum(r[s+'_ref']=='1' for r in cohort),
                    both_correct=sum(r[s+'_both']=='1' for r in cohort),
                    active_refusals=sum(r[arm+'_active_refusal']=='1' for r in cohort),
                    physical_owner_unknown=sum(r[arm+'_physical_owner_unknown']=='1' for r in cohort)))
            prefix = 'visual_' if truth == 'visual_independent' else ''
            within = [r for r in cohort if r[prefix+'old_rank']!='']
            assert all(r[prefix+'new_rank']!='' for r in within)
            old = np.asarray([int(r[prefix+'old_rank']) for r in within], dtype=float)
            new = np.asarray([int(r[prefix+'new_rank']) for r in within], dtype=float)
            margins = [r for r in within if r[prefix+'old_margin']!='']
            ranks.append(dict(scope=truth, group=group, n=len(cohort), rankable=len(within), margins=len(margins),
                              old_top1=int((old==1).sum()), new_top1=int((new==1).sum()),
                              new_outranks_old=int((new<old).sum()), old_outranks_new=int((old<new).sum()),
                              old_mean_reciprocal=float((1/old).mean()) if len(old) else '',
                              new_mean_reciprocal=float((1/new).mean()) if len(new) else '',
                              old_mean_margin=float(np.mean([float(r[prefix+'old_margin']) for r in margins])) if margins else '',
                              new_mean_margin=float(np.mean([float(r[prefix+'new_margin']) for r in margins])) if margins else ''))
    for arm in ARMS:
        for group, sets in group_records([r for r in flips if r['arm']==arm]).items():
            cohort = [r for g in sets for r in g]
            cf.append(dict(group=group, arm=arm, n=len(cohort), **{field: sum(r[field]=='1' for r in cohort)
                for field in ('both_signs','same_known_anchor','different_known_subject','conjunction',
                              'both_physically_taken','ref_owner_unknown','subject_box_changed')}))
        for scope, qualifier in (('all_changes',None),('Acc25_rescue','strong25_rescue'),('Acc25_harm','strong25_harm'),
                                 ('Acc50_rescue','strong50_rescue'),('Acc50_harm','strong50_harm')):
            z = [r for r in terms if r['arm']==arm and (qualifier is None or r[qualifier]=='1')]
            damage.append(dict(scope=scope, arm=arm, n=len(z), **{key: float(np.mean([float(r[key]) for r in z])) if z else ''
                                                                 for key in TERMS}))
    for phase, pool in (('all_verified', oracles), ('verified_reference_in_bank', [v for v in oracles if v['reference_in_bank']=='1'])):
        for name in ('independent','predicted_reference','identity_oracle'):
            data = [r for r in pool if r[name+'_iou']!='']
            v = np.array([float(x[name+'_iou']) for x in data])
            oracle.append(dict(scope=phase, arm=name, all_verified=len(pool), served=len(data),
                               mIoU=float(v.mean()*100) if len(v) else '', acc25_correct=int((v>.25).sum()),
                               acc50_correct=int((v>.5).sum()), subject_correct=sum(x[name+'_subject']=='1' for x in data)))
    for arm in ('predicted_reference','identity_oracle'):
        sample = [r for r in oracles if r['reference_in_bank']=='1']
        residual = np.asarray([[float(r[a+'_iou']) for a in ('independent',arm)] for r in sample]).reshape(-1,2)
        result = dict(arm=arm,n=len(sample))
        for name, hit in [('IoU',residual),('Acc25',residual>.25),('Acc50',residual>.5),
                          ('subject_identity', np.asarray([[r[a+'_subject']=='1' for a in ('independent',arm)] for r in sample]).reshape(-1,2))]:
            diff=hit[:,1].astype(float)-hit[:,0].astype(float)
            result.update({name+'_rescue':int((diff>1.e-10).sum()), name+'_harm':int((diff< -1.e-10).sum())})
        oracle_pairs.append(result)
    return dict(metrics=m, paired=p, identity=identity, counterfactual=cf, oracle=oracle, oracle_paired=oracle_pairs,
                decomposition=damage, reference_ranks=ranks)


def score():
    seal = verify_seals()
    audit_path = HERE / 'EVALUATION_LOCK.json'
    digests = {name: sha(HERE / name) for name in ('score168.py', 'panels168.py', 'PREDICTION_SEAL.json', 'CONTROLLED_PREDICTION_SEAL.json')}
    if audit_path.exists():
        assert read(audit_path) == digests
    else:
        atomic_json(audit_path, digests)
    status('scoring')
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        pending = {(s,t):pool.submit(direction,s,t) for s,t in ORDER}
        receipts = {f'{s}_to_{t}': f.result() for (s,t),f in pending.items()}
    all_rows = defaultdict(list)
    for name, r in receipts.items():
        for key, value in r['files'].items():
            f = WORK / 'scalars' / name / (key + '.csv')
            assert sha(f) == value
            all_rows[key].extend(table(f))
    rows, edges, flips, oracles, terms = [all_rows[k] for k in ('rows','edges','flips','oracle','term_deltas')]
    assert len(rows) == 18620 and len(edges) == 11378 and len(flips) == 2526*len(ARMS)
    assert sum(r['truth'] in ('annotation_proxy','visual_verified') for r in edges) == 750
    assert sum(int(r['physical_roles']) for r in rows) == sum(x['physical_roles'] for x in seal['directions'].values())
    found = [float(r['legacy_worst_physical_advantage']) for r in rows if r['legacy_worst_physical_advantage']!='']
    assert min(found) > 0.
    out = dict(all_rows=rows, relation_edges=edges, counterfactual_rows=flips, oracle_rows=oracles,
               term_deltas=terms, **summaries(rows, edges, flips, oracles, terms))
    OUT.mkdir(exist_ok=True)
    for name, values in out.items():
        write_table(OUT / (name+'.csv'),values)
    atomic_json(HERE/'EVALUATION_AUDIT.json',dict(n=len(rows),edges=len(edges),oracles=len(oracles),
                flips=len(flips)//len(ARMS),sensor_ious=len(rows)*len(ARMS),
                max_sensor_error=max(v['max_sensor_error'] for v in receipts.values()),
                exact_r167_max_error=max(v['max_exact_replay_error'] for v in receipts.values()),
                old_null_min_dominated_margin=min(found),
                old_nonempty_roles=sum(int(r['physical_roles']) for r in rows),
                verified_reference_conflicts=sum(r['visual_conflict']=='1' for r in edges),
                all_negatives_retained=True, FINAL_access=False, directions=receipts,
                files={name+'.csv':sha(OUT/(name+'.csv')) for name in out}))
    print('R168 full evaluation sealed',len(rows),len(edges),len(oracles),flush=True)


if __name__=='__main__':
    score()
