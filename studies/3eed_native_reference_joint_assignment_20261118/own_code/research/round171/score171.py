"""Evaluate sealed predictions on official targets and separately verified endpoints."""
from math import fsum

import numpy as np

from scope171 import ARMS, HERE, INPUT, ORDER, OUT, PLATFORMS, PRIOR_INPUT, R170, WORK, atomic_json, lock, read, sha, status
from labels166 import RIGHT, box_masks, occupied_owners, physical_labels
from predict171 import FILES, MODES, REFERENCE_BANKS, camera_relation, read_npz
from review170 import ADJUDICATION as PRIOR_TRUTH, LIST as PRIOR_KEYS
from review171 import KEYS, TRUTH, verify_predictions
from score165 import many_iou, sensor_values, table, write_table
from score170 import label, status_for


PAIRS = (
    ('D_native_joint', 'D_mapped_joint'),
    ('D_native_joint', 'D_native_independent'),
    ('D_native_joint', 'D_native_fixed_reference'),
    ('D_native_joint', 'D_native_linked_joint'),
    ('D_native_joint', 'r158_pca'),
    ('D_native_joint', 'C_mapped_joint'),
    ('D_native_independent', 'D_mapped_independent'),
    ('D_mapped_joint', 'D_mapped_independent'),
)
LABELS = ('mIoU', 'Acc25', 'Acc50')
KINDS = ('official_iou', 'verified_subject_iou')


def reference_fields(d, arm):
    if arm.endswith('_mapped'):
        return d['mapped_left'], d['mapped_stats'], d[arm + '_unary']
    return d['native_left'], d['native_stats'], d[arm + '_unary']


def rank(owners, gold, order):
    if gold < 0:
        return ''
    hits = np.flatnonzero(owners[order] == gold)
    return int(hits[0] + 1) if len(hits) else 0


def candidate_owners(boxes, seeds, labels, n):
    return occupied_owners(boxes, seeds, labels, n) if len(boxes) else np.zeros(0, dtype=int)


def select_manifest():
    current = read(HERE / 'VISUAL_TRUTH_SEAL.json')
    archived = read(R170 / 'VISUAL_TRUTH_SEAL.json')
    assert current['adjudication'] == sha(TRUTH) and archived['adjudication'] == sha(PRIOR_TRUTH)
    assert current['panel_seal'] == sha(HERE / 'REVIEW_PANEL_SEAL.json')
    assert archived['certificate'] == sha(R170 / 'REVIEW_PANEL_SEAL.json')
    assert current['prediction_and_official_subject_slot_blind'] and not current['FINAL_access']
    assert archived['blinded_to_predictions_and_official_subject_slot'] and not archived['FINAL_access']
    assert read(HERE / 'REVIEW_PANEL_SEAL.json')['keys'] == sha(KEYS)
    assert read(R170 / 'REVIEW_PANEL_SEAL.json')['registry'] == sha(PRIOR_KEYS)
    return dict(replay170=zip(read(PRIOR_INPUT), read(PRIOR_TRUTH), read(PRIOR_KEYS), strict=True),
                new_frames=zip(read(INPUT), read(TRUTH), read(KEYS), strict=True))


def evaluate(entry, blind, key, source, cohort, digest, archived):
    row, case, target = entry['row'], entry['case'], entry['target']
    assert case == blind['case'] == key['case']
    d = read_npz(WORK / 'predictions' / cohort / f'{case:02d}_{source}.npz')
    assert str(d['design']) == digest and tuple(d['arms']) == ARMS
    sub, ref = label(key, blind['subject']), label(key, blind['reference'])
    pair_known = min(sub, ref) >= 0
    assert not pair_known or sub != ref
    names, gt, seeds, owner_labels = physical_labels(row, d['seeds'])
    assert len(names) == key['inventory_size'] and max(sub, ref) < len(names)
    predictions = d['method_boxes']
    owners = occupied_owners(predictions, seeds, owner_labels, len(names))
    ious = many_iou(predictions[:, 0], gt[0])
    error = 0.
    for g in (0, sub) if sub > 0 else (0,):
        value = many_iou(predictions[:, 0], gt[g])
        sensor = sensor_values(predictions[:, 0], gt[g], target, int(row['suffix'] or 0))
        error = max(error, float(np.max(np.abs(value - sensor))))
        assert all(np.array_equal(value > t, sensor > t) for t in (.25, .50))
    assert error < 1e-7
    after_blind = many_iou(predictions[:, 0], gt[sub]) if sub >= 0 else [''] * len(ARMS)
    have_ref = np.isfinite(predictions[:, 1]).all(1)
    ref_iou = np.zeros(len(ARMS))
    if ref >= 0 and have_ref.any():
        ref_iou[have_ref] = many_iou(predictions[have_ref, 1], gt[ref])
    if pair_known:
        masks = box_masks(gt[[sub, ref]], d['seeds'])
        gt_left, _, _ = camera_relation(masks[:1], masks[1:], d['uv'], d['visible'])
        truth_sign = int(np.sign(int(d['sign']) * gt_left[0, 0]))
    else:
        truth_sign = ''

    meta = dict(cohort=cohort, case=case, source=source, target=target, recording=entry['recording'])
    subjects = candidate_owners(d['subject_boxes'], seeds, owner_labels, len(names))
    common = candidate_owners(d['mapped_boxes'], seeds, owner_labels, len(names))
    arm_rows = []
    for i, (arm, bank, mode) in enumerate(zip(ARMS, REFERENCE_BANKS, MODES, strict=True)):
        j, k = d['choices'][i]
        ref_status = status_for(ref, owners[i, 1], have_ref[i])
        sub_status = status_for(sub, owners[i, 0])
        rel, n_points, visible, width, confidence = ('',) * 5
        if have_ref[i]:
            geometry, cover, _ = reference_fields(d, bank)
            rel = float(geometry[j, k] * int(d['sign']))
            n_points, visible, width, confidence = cover[k]
        else:
            assert j == 0
        same = int(owners[i, 0] == owners[i, 1]) if have_ref[i] and min(owners[i]) >= 0 else ''
        arm_rows.append(dict(**meta, arm=arm, official_iou=float(ious[i]), verified_subject_iou=after_blind[i],
            verified_reference_iou=float(ref_iou[i]) if ref >= 0 and bank else '',
            subject_identity=sub_status, reference_identity=ref_status,
            dual_correct=int(sub_status == ref_status == RIGHT) if pair_known and bank else '',
            has_reference=int(have_ref[i]), fallback=int(bool(bank) and not have_ref[i]),
            subject_changed=int(j != 0), same_known_object=same, signed_relation=rel,
            relation_used=int(mode in ('joint', 'fixed_reference') and have_ref[i]),
            ref_support_seeds=n_points, ref_visible_seeds=visible, ref_width_fraction=width,
            ref_relation_confidence=confidence, subject_position=int(j), reference_position=int(k)))
    flow = []
    for variant in ('C', 'D'):
        natural = candidate_owners(d[variant + '_natural_boxes'], seeds, owner_labels, len(names))
        u = d[variant + '_mapped_unary'][:-1]
        positions = np.flatnonzero(np.isfinite(u))
        mapped_rank = rank(common, ref, positions[np.argsort(-u[positions], kind='stable')])
        native_rank = rank(natural, ref, np.arange(len(natural)))
        linked = d[variant + '_links'] >= 0
        linked_rank = rank(natural, ref, np.flatnonzero(linked))
        if archived:
            assert str(native_rank) == archived[variant]['native_rank']
            assert str(mapped_rank) == archived[variant]['mapped_rank']
        is_lost = int(native_rank not in ('', 0) and mapped_rank == 0)
        good_both = (subjects == sub)[:, None] & (natural == ref)[None, :] if pair_known else None
        positive_joint = (int(np.any(good_both & (d['native_left'][:, :-1] * int(d['sign']) > 0)))
                          if variant == 'D' and pair_known else '')
        flow.append(dict(**meta, variant=variant, native_candidates=len(natural), linked_native_candidates=int(linked.sum()),
            mapped_candidates=len(positions), native_rank=native_rank, mapped_rank=mapped_rank, linked_native_rank=linked_rank,
            lost_at_mapping=is_lost, receiver_bank_has_correct=int((common == ref).any()) if ref >= 0 else '',
            unlinked_correct_native_candidates=int(((natural == ref) & ~linked).sum()) if ref >= 0 else '',
            correct_subject_and_native_reference_available=int(good_both.any()) if pair_known else '',
            correct_pair_with_positive_factor_available=positive_joint))
    case_row = dict(**meta, stratum=entry.get('stratum', 'r170_original_screen'),
        recording_in_r170=int(entry.get('recording_in_r170', True)), rewrite=entry['rewrite']['status'],
        subject_label=blind['subject']['status'], reference_label=blind['reference']['status'],
        subject_agrees_with_official=int(sub == 0) if sub >= 0 else '', verified_pair=int(pair_known),
        verified_pair_signed_geometry=truth_sign)
    return case_row, arm_rows, flow, error


def grouped(rows):
    by_direction = {(s, t): [x for x in rows if (x['source'], x['target']) == (s, t)] for s, t in ORDER}
    out = {'six_equal': [g for g in by_direction.values() if g], 'pooled': [rows]}
    out.update({t: [r for (s, v), r in by_direction.items() if v == t and r] for t in PLATFORMS})
    out.update({f'{s}_to_{t}': [r] for (s, t), r in by_direction.items() if r})
    return {key: val for key, val in out.items() if val and val[0]}


def by_key(row):
    return row['cohort'], row['case'], row['source']


def official(rows):
    y = np.array([float(row['official_iou']) for row in rows])
    return np.c_[y, y > .25, y > .50]


def mean_stats(groups):
    return 100 * np.mean([values.mean(0) for values in groups], axis=0)


def paired(a, b):
    diff = [official(x) - official(y) for x, y in zip(a, b, strict=True)]
    delta = mean_stats(diff)
    a_flat, b_flat = ([r for g in z for r in g] for z in (a, b))
    rows = a_flat
    item = dict(n=len(rows))
    within = np.concatenate(diff)
    for j, name in enumerate(LABELS):
        gains = [r for r, x in zip(rows, within[:, j], strict=True) if x > 1e-10]
        losses = [r for r, x in zip(rows, within[:, j], strict=True) if x < -1e-10]
        item.update({name+'_pp': float(delta[j]), name+'_gains': len(gains), name+'_losses': len(losses),
                     name+'_gain_frames': len({r['case'] for r in gains}),
                     name+'_gain_recordings': len({r['recording'] for r in gains}),
                     name+'_gains_with_correct_pair': sum(r['dual_correct'] == 1 for r in gains)})
    for name, v in (('subject_identity', RIGHT), ('reference_identity', RIGHT), ('dual_correct', 1)):
        x = [aa[name] == v for aa in a_flat]
        y = [bb[name] == v for bb in b_flat]
        item.update({name+'_gains': sum(p and not q for p, q in zip(x, y, strict=True)),
                     name+'_losses': sum(q and not p for p, q in zip(x, y, strict=True))})
    item['subject_choice_changes'] = sum(x['subject_position'] != y['subject_position'] for x, y in zip(a_flat, b_flat, strict=True))
    native_comparison = all(r['arm'].startswith('D_native') for r in (a_flat[0], b_flat[0]))
    item['reference_choice_changes_same_native_pool'] = (
        sum(x['reference_position'] != y['reference_position'] for x, y in zip(a_flat, b_flat, strict=True))
        if native_comparison else '')
    return item


def summaries(cases, lookup):
    metrics, deltas, deletion = [], [], []
    for cohort in ('replay170', 'new_frames'):
        panel = [row for row in cases if row['cohort'] == cohort]
        for section, rows in (('all', panel), ('supported_rewrite', [r for r in panel if r['rewrite'] == 'supported']),
                              ('both_verified', [r for r in panel if r['verified_pair']])):
            for name, pieces in grouped(rows).items():
                flat = [r for group in pieces for r in group]
                fields = dict(cohort=cohort, scope=section, group=name, n=len(flat),
                              recordings=len({r['recording'] for r in flat}), frames=len({r['case'] for r in flat}),
                              directions=len({(r['source'], r['target']) for r in flat}))
                data = {arm: [[lookup[(*by_key(x), arm)] for x in group] for group in pieces] for arm in ARMS}
                for arm, a in data.items():
                    flat_a = [x for c in a for x in c]
                    metric = dict(zip(LABELS, mean_stats([official(c) for c in a]), strict=True))
                    s = [x['verified_subject_iou'] for x in flat_a if x['verified_subject_iou'] != '']
                    r = [x['verified_reference_iou'] for x in flat_a if x['verified_reference_iou'] != '']
                    rel = [x['signed_relation'] for x in flat_a if x['signed_relation'] != '']
                    metrics.append(dict(**fields, arm=arm, **metric, subject_n=len(s),
                     subject_right=sum(x['subject_identity'] == RIGHT for x in flat_a),
                     verified_subject_mIoU=100 * fsum(s)/len(s) if s else '',
                     verified_subject_Acc25=sum(x > .25 for x in s), verified_subject_Acc50=sum(x > .50 for x in s),
                     reference_n=sum(x['reference_label'] == 'verified' for x in flat), reference_right=sum(x['reference_identity'] == RIGHT for x in flat_a),
                     verified_reference_mIoU=100 * fsum(r)/len(r) if r else '',
                     reference_emitted=sum(x['has_reference'] for x in flat_a),
                     pair_n=sum(x['verified_pair'] for x in flat),
                     dual_right=sum(x['dual_correct'] == 1 for x in flat_a) if arm != 'r158_pca' else '',
                     fallback=sum(x['fallback'] for x in flat_a), same_known_object=sum(x['same_known_object'] == 1 for x in flat_a),
                     selected_positive_relation=sum(x > 0 for x in rel), selected_negative_relation=sum(x < 0 for x in rel),
                     selected_zero_relation=sum(x == 0 for x in rel)))
                for a, b in PAIRS:
                    fields_no_n = {k: v for k, v in fields.items() if k != 'n'}
                    deltas.append(dict(**fields_no_n, arm=a, control=b, **paired(data[a], data[b])))
        for recording in sorted({x['recording'] for x in panel}):
            subset = grouped([x for x in panel if x['recording'] != recording]).get('six_equal', [])
            for a, b in PAIRS[:4]:
                ys = [official([lookup[(*by_key(x), a)] for x in g]) - official([lookup[(*by_key(x), b)] for x in g]) for g in subset]
                means = mean_stats(ys) if len(ys) == 6 else ('',) * len(LABELS)
                deletion.append(dict(cohort=cohort, excluded=recording, n=sum(map(len, subset)), directions=len(ys),
                     arm=a, control=b, **dict(zip((k+'_pp' for k in LABELS), means, strict=True))))
    return metrics, deltas, deletion


def coverage(cases, flow):
    stats = []
    for cohort in ('replay170', 'new_frames'):
        for group in ('pooled', *PLATFORMS):
            selected = [x for x in cases if x['cohort'] == cohort and (group == 'pooled' or x['target'] == group)]
            truths = [x for x in selected if x['verified_pair']]
            for variant in ('C', 'D'):
                rows = [r for r in flow if r['cohort'] == cohort and (group == 'pooled' or r['target'] == group) and r['variant'] == variant]
                ref = [r for r in rows if r['native_rank'] != '']
                g = [r for r in ref if r['lost_at_mapping']]
                stats.append(dict(cohort=cohort, group=group, variant=variant, directions=len(selected), ref_n=len(ref),
                                 native_coverage=sum(r['native_rank'] > 0 for r in ref),
                                 mapped_coverage=sum(r['mapped_rank'] > 0 for r in ref),
                                 linked_native_coverage=sum(r['linked_native_rank'] > 0 for r in ref),
                                 native_top1=sum(r['native_rank'] == 1 for r in ref), mapped_top1=sum(r['mapped_rank'] == 1 for r in ref),
                                 lost_at_mapping=len(g), lost_missing_receiver=sum(r['receiver_bank_has_correct'] == 0 for r in g),
                                 pairs_evaluable=len(truths), correct_pair_available=sum(r['correct_subject_and_native_reference_available'] == 1 for r in ref),
                                 correct_pair_positive_available=sum(r['correct_pair_with_positive_factor_available'] == 1 for r in ref) if variant=='D' else '',
                                 verified_gt_positive=sum(x['verified_pair_signed_geometry'] == 1 for x in truths),
                                 verified_gt_negative=sum(x['verified_pair_signed_geometry'] == -1 for x in truths),
                                 verified_gt_zero=sum(x['verified_pair_signed_geometry'] == 0 for x in truths)))
    return stats


def score():
    pin = lock('scoring', (*FILES, 'review171.py', 'score171.py'))
    proof = verify_predictions()
    panels = select_manifest()
    status('scoring', records=proof['rows'])
    prior_rows = table(R170 / 'results/directions_and_cases.csv')
    prior_flow = table(R170 / 'results/mapping_cases.csv')
    prior_checks = read(R170 / 'EVALUATION_AUDIT.json')['files']
    assert all(sha(R170 / 'results' / name) == prior_checks[name] for name in ('directions_and_cases.csv', 'mapping_cases.csv'))
    archived = {(str(r['case']), r['source'], r['variant']): r for r in prior_flow}
    old_results = {(str(r['case']), r['source']): r for r in prior_rows}
    samples, arms, flow = [], [], []
    error = 0.
    for cohort, panel in panels.items():
        for entry, blind, key in panel:
            for source in PLATFORMS:
                if source == entry['target']:
                    continue
                k = str(entry['case']), source
                handoff = {v: archived[(*k, v)] for v in ('C', 'D')} if cohort == 'replay170' else None
                case, records, f, e = evaluate(entry, blind, key, source, cohort, proof['design'], handoff)
                if handoff:
                    prior = old_results[k]
                    for i, old_arm in enumerate(('r158_pca', 'C_independent', 'C_joint', 'D_independent', 'D_joint')):
                        assert abs(records[i]['official_iou'] - float(prior[old_arm+'_iou'])) < 1e-12
                        assert str(records[i]['subject_identity']) == prior[old_arm+'_subject_identity']
                        assert str(records[i]['reference_identity']) == prior[old_arm+'_reference_identity']
                samples.append(case)
                arms.extend(records)
                flow.extend(f)
                error = max(error, e)
    assert len(samples) == 168 and len(arms) == 168 * len(ARMS) and len(flow) == 336
    assert len({by_key(row) for row in samples}) == len(samples)
    assert sum(x['cohort'] == 'replay170' and x['variant'] == 'D' and x['lost_at_mapping'] for x in flow) == 5
    lookup = {(*by_key(x), x['arm']): x for x in arms}
    bundles, deltas, deletion = summaries(samples, lookup)
    flows = coverage(samples, flow)
    OUT.mkdir(exist_ok=True)
    files = {}
    scalar_arms = [{k: v for k, v in row.items() if k not in ('subject_position', 'reference_position')} for row in arms]
    for name, rows in (('cases', samples), ('arms', scalar_arms), ('reference_flow', flow), ('coverage', flows),
                       ('metrics', bundles), ('paired', deltas), ('recording_deletion', deletion)):
        path = OUT / (name+'.csv')
        write_table(path, rows)
        files[path.name] = sha(path)
    audit = dict(predictions=sha(HERE / 'PREDICTION_SEAL.json'), new_truth=sha(HERE / 'VISUAL_TRUTH_SEAL.json'),
                 prior_truth=sha(R170 / 'VISUAL_TRUTH_SEAL.json'), lock=pin, rows=proof['rows'], strict_thresholds=[.25,.50],
                 maximum_sensor_iou_error=error, r170_original_lost_references=5, files=files, FINAL_access=False)
    atomic_json(HERE / 'EVALUATION_AUDIT.json', audit)
    print('R171 score', proof['rows'], 'sensor parity', error, flush=True)


if __name__ == '__main__':
    score()
