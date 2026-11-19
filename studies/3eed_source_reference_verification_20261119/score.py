"""Evaluate frozen D and source-only cue calibration by held-out recording."""
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from predict import points_in
from scope import HERE, PLATFORMS, PRIVATE, RESULTS, SPEC, atomic_json, read, sha


PRED = PRIVATE / 'predictions'
PLAN = HERE / 'scoring_plan.json'
ARMS = ('D', 'D_only', 'aligned', 'mismatched')
KNOWN = frozenset(('covered', 'natural_miss'))
PENALTY = 0.15


@dataclass
class Example:
    case: str
    platform: str
    recording: str
    review: str
    gold: int
    verified_foils: tuple
    owners: np.ndarray
    unary: np.ndarray
    design: dict
    confidence: np.ndarray
    status: str
    reference_points: int
    positive_side_supported: tuple

    @property
    def present_foils(self):
        return tuple(obj for obj in self.verified_foils if obj in self.owners)

    @property
    def hard(self):
        return self.status == 'covered' and bool(self.present_foils)


def check_receipts():
    truth_file = PRIVATE / 'VERIFIED_ROSTER.json'
    truth = read(PRIVATE / 'TRUTH_SEAL.json')
    pred = read(PRED / 'SEAL.json')
    assert truth['file'] == sha(truth_file)
    assert truth['review_program'] == sha(HERE / 'review.py')
    assert truth['choices'] == sha(PRIVATE / 'review' / 'BLIND_CHOICES.json')
    assert truth['single_internal_blind_reviewer'] and not truth['model_results_open_before_adjudication']
    assert not truth['protected_data_access'] and not pred['protected_data_access']
    assert pred['source_parameters_unchanged'] and not pred['prediction_reads_truth']
    assert pred['pin'] == sha(PRED / 'LOCK.json')
    lock = read(PRED / 'LOCK.json')
    assert lock['protocol'] == sha(SPEC) and lock['program'] == sha(HERE / 'predict.py')
    assert lock['inputs'] == {s: sha(PRIVATE / 'prep' / (s + '.json')) for s in PLATFORMS}
    rows = read(truth_file)
    keys = {row['case'] for row in rows}
    assert len(keys) == truth['n'] == len(rows) == len(pred['files'])
    assert keys == set(pred['files']) == {p.stem for p in PRED.glob('*.npz')}
    for key, checksum in pred['files'].items():
        assert sha(PRED / (key + '.npz')) == checksum
    return rows, pred['pin']


def owner_coverage(gold, inventory, points, memberships, oriented, raw, masks, owners):
    from score154 import many_iou

    ref_mask = memberships == gold
    ref_points = int(ref_mask.sum())
    if np.any(owners == gold):
        return 'covered', ref_points
    if ref_points < 6:
        return 'ambiguous_point_support', ref_points
    overlap = np.max(many_iou(np.r_[oriented, raw], inventory[gold]))
    hits = max(masks[:, ref_mask].sum(1).max(), points_in(raw, points[ref_mask]).sum(1).max())
    status = ('ambiguous_candidate_coverage' if overlap >= .15 or
              (hits >= 3 and hits >= .15 * ref_points) else 'natural_miss')
    return status, ref_points


def physical_owners(points, inventory, oriented):
    from labels166 import mask_owners
    from mechanisms162 import seed_owners

    memberships = seed_owners(points, inventory)
    masks = points_in(oriented, points)
    owners = mask_owners(masks, memberships, len(inventory))
    for slot, obj in enumerate(owners):
        if obj >= 0 and np.count_nonzero(masks[slot] & (memberships == obj)) < 3:
            owners[slot] = -1
    return memberships, masks, owners


def load_example(row, pin, prediction_dir=PRED):
    from calibrate168 import image_geometry
    from geometry162 import in_box
    from mechanisms162 import annotation_inventory
    from predict171 import camera_relation, read_npz
    from visual162 import projection

    r = row['row']
    case = row['case']
    gold = int(row['reference_identity'])
    foils = tuple(row['verified_foil_ids'])
    d = read_npz(prediction_dir / (case + '.npz'))
    assert str(d['pin']) == pin and int(d['frozen_query_count']) == 3
    oriented = d['boxes']
    raw = np.c_[d['raw'], np.zeros(len(d['raw']))]
    candidates = len(oriented)
    assert candidates == len(raw) == len(d['unaries']) - 1 and 1 <= candidates <= 32
    unary = d['unaries'][:-1]
    assert unary[0] == 0. and unary.argmax() == 0 and d['unaries'][-1] == -2.
    assert np.isfinite(oriented).all() and np.isfinite(raw).all() and np.isfinite(unary).all()
    designs = dict(aligned=d['evidence'], mismatched=d['mismatched'])
    assert all(x.shape == (candidates, 3) and np.isfinite(x).all() for x in designs.values())
    confidence = d['reliability']
    if gold < 0:
        return Example(case, r['platform'], row['recording'], row['verification_status'], gold, foils,
                       np.full(candidates, -1), unary, designs, confidence, 'unresolved_review', 0, ())

    names, inventory = annotation_inventory(r['metapath'], r['platform'], int(r['suffix'] or 0), r['object_index'])
    assert len(names) == row['inventory_size']
    entities = (gold, *foils)
    assert len(set(entities)) == len(entities) and 0 not in entities
    assert all(names[obj] == row['reference_class'] for obj in entities)
    xy = projection(image_geometry(r), inventory[:,:3])
    assert all(row['sign'] * (xy[obj, 0] - xy[0, 0]) >= .014 for obj in entities)
    memberships, masks, owners = physical_owners(d['points'], inventory, oriented)
    status, ref_points = owner_coverage(gold, inventory, d['points'], memberships, oriented, raw, masks, owners)

    side, _, _ = camera_relation(in_box(d['seeds'], inventory[0])[None, :],
                                 points_in(raw, d['seeds']), d['uv'], d['visible'])
    supported = tuple(int(obj) for obj in entities if np.any((owners == obj) & (row['sign'] * side[0, :-1] > 0.)))
    return Example(case, r['platform'], row['recording'], row['verification_status'], gold, foils,
                   owners, unary, designs, confidence, status, ref_points, supported)


def fit_rank(train, arm):
    batches = []
    for e in train:
        if not e.hard:
            continue
        positions = [int(np.argmax(np.where(e.owners == obj, e.unary, -np.inf)))
                     for obj in (e.gold, *e.present_foils)]
        top, *foils = positions
        x = e.design[arm]
        batches.append((e.unary[top] - e.unary[foils], x[top] - x[foils]))
    if not batches:
        return np.zeros(3), 0
    offsets = np.concatenate([offset for offset, _ in batches])
    design = np.concatenate([pair for _, pair in batches])
    weights = np.concatenate([np.full(len(offset), 1. / (len(batches) * len(offset))) for offset, _ in batches])

    def objective(beta):
        z = offsets + design @ beta
        val = weights @ np.logaddexp(0., -z) + PENALTY * (beta @ beta) / 2.
        grad = -design.T @ (weights * expit(-z)) + PENALTY * beta
        return val, grad

    opt = minimize(objective, np.zeros(3), jac=True, bounds=[(-2., 2.)] * 3, method='L-BFGS-B')
    assert opt.success and np.isfinite(opt.x).all()
    return opt.x, len(batches)


def rejection_features(example, arm):
    return np.r_[example.confidence, example.design[arm][0]] if arm in example.design else example.confidence


def fit_rejection(train, arm):
    available = [e for e in train if e.status in KNOWN]
    y = np.array([e.status == 'natural_miss' for e in available], dtype=float)
    absent = int(y.sum())
    covered = len(y) - absent
    if not absent or not covered:
        def constant(e):
            return float(bool(absent))
        return constant, covered, absent, None

    x = np.stack([rejection_features(e, arm) for e in available])
    center = x.mean(0)
    scale = x.std(0)
    scale[scale < 1.e-6] = 1.
    z = np.c_[np.ones(len(x)), (x - center) / scale]
    weights = np.where(y > 0., .5 / absent, .5 / covered)

    def objective(beta):
        logits = z @ beta
        reg = np.r_[0., beta[1:]] * PENALTY
        val = weights @ (np.logaddexp(0., logits) - y * logits) + .5 * (beta @ reg)
        grad = z.T @ (weights * (expit(logits) - y)) + reg
        return val, grad

    opt = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B')
    assert opt.success and np.isfinite(opt.x).all()

    def apply(e):
        return float(expit(opt.x[0] + opt.x[1:] @ ((rejection_features(e, arm) - center) / scale)))

    return apply, covered, absent, dict(coefficients=opt.x.tolist(), center=center.tolist(), scale=scale.tolist())


def evaluate(e, scores, p_miss, arm):
    reject = p_miss is not None and p_miss >= .5
    k = int(scores.argmax())
    covered = e.status == 'covered'
    position = None
    pair = dict(wins=0, ties=0, losses=0, hard_frame=None, all_foils_beaten=None,
                geometric_wins=0, geometric_ties=0, geometric_losses=0)
    if covered:
        correct = float(scores[e.owners == e.gold].max())
        position = 1 + int((scores > correct).sum())
        differences = np.array([correct - scores[e.owners == obj].max() for obj in e.present_foils])
        if len(differences):
            pair['wins'] = int((differences > 1.e-10).sum())
            pair['ties'] = int((abs(differences) <= 1.e-10).sum())
            pair['losses'] = int((differences < -1.e-10).sum())
            pair['hard_frame'] = (pair['wins'] + .5 * pair['ties']) / len(differences)
            pair['all_foils_beaten'] = bool(pair['wins'] == len(differences))
            geo = np.array([e.gold in e.positive_side_supported and obj in e.positive_side_supported
                            for obj in e.present_foils])
            pair['geometric_wins'] = int(((differences > 1.e-10) & geo).sum())
            pair['geometric_ties'] = int(((abs(differences) <= 1.e-10) & geo).sum())
            pair['geometric_losses'] = int(((differences < -1.e-10) & geo).sum())
    raw_identity = bool(e.gold >= 0 and e.owners[k] == e.gold)
    return dict(case=e.case, platform=e.platform, recording=e.recording, arm=arm,
                status=e.status, review=e.review, verified_foils=len(e.verified_foils), represented_foils=len(e.present_foils),
                candidates=len(scores), reference_points=e.reference_points, rank=position, raw_correct=raw_identity,
                reject=reject, correct_accept=raw_identity and not reject, missing_probability=p_miss, **pair)


def ordered_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.recording].append(row)
    return groups


def folds(rows):
    by_recording = ordered_groups(rows)
    outputs, fit_records = [], []
    for holding, test in sorted(by_recording.items()):
        fit = [other for rec, group in by_recording.items() if rec != holding for other in group]
        assert not {e.case for e in fit} & {e.case for e in test}
        assert holding not in {e.recording for e in fit}
        outputs.extend(evaluate(e, e.unary, None, 'D') for e in test)
        if not fit:
            continue
        for arm in ARMS[1:]:
            reject, covered, absent, params = fit_rejection(fit, arm)
            if arm == 'D_only':
                beta, pairs = None, 0
            else:
                beta, pairs = fit_rank(fit, arm)
            fit_records.append(dict(platform=test[0].platform, holdout=holding, arm=arm, paired_training_frames=pairs,
                                    covered_train=covered, absent_train=absent, fit_cases=[e.case for e in fit],
                                    held_cases=[e.case for e in test], weights=None if beta is None else beta.tolist(),
                                    rejection=params))
            for e in test:
                scores = e.unary if beta is None else e.unary + e.design[arm] @ beta
                outputs.append(evaluate(e, scores, reject(e), arm))
    return outputs, fit_records


def summarize(rows):
    covered = [r for r in rows if r['status'] == 'covered']
    absent = [r for r in rows if r['status'] == 'natural_miss']
    pairs = [r for r in rows if r['hard_frame'] is not None]
    return dict(n=len(rows), cover=len(covered), miss=len(absent), hard_frames=len(pairs),
                recording_count=len({r['recording'] for r in rows}),
                hard_recordings=len({r['recording'] for r in pairs}),
                miss_recordings=len({r['recording'] for r in absent}),
                hard_concordance=None if not pairs else float(np.mean([r['hard_frame'] for r in pairs])),
                hard_all_foils_won=sum(r['all_foils_beaten'] for r in pairs),
                pair_wins=sum(r['wins'] for r in pairs), pair_ties=sum(r['ties'] for r in pairs),
                pair_losses=sum(r['losses'] for r in pairs),
                geometric_wins=sum(r['geometric_wins'] for r in pairs),
                geometric_ties=sum(r['geometric_ties'] for r in pairs),
                geometric_losses=sum(r['geometric_losses'] for r in pairs),
                covered_top1=sum(r['raw_correct'] for r in covered),
                covered_mean_reciprocal_rank=None if not covered else float(np.mean([1. / r['rank'] for r in covered])),
                missed_rejected=sum(r['reject'] for r in absent),
                covered_rejected=sum(r['reject'] for r in covered),
                covered_correct_accepted=sum(r['correct_accept'] for r in covered),
                total_rejected=sum(r['reject'] for r in rows))


def report(examples, rows, fit):
    structure, metrics, gates, comparisons = {}, {}, {}, {}
    for source in PLATFORMS:
        local = [r for r in rows if r['platform'] == source]
        pop = [e for e in examples if e.platform == source]
        structure[source] = dict(invitations=len(pop), recordings=len({e.recording for e in pop}),
                                 visually_verified=sum(e.gold >= 0 for e in pop),
                                 visually_verified_with_foil=sum(bool(e.verified_foils) for e in pop),
                                 statuses=dict(Counter(e.status for e in pop)))
        arms = {arm: summarize([r for r in local if r['arm'] == arm])
                for arm in ARMS if any(r['arm'] == arm for r in local)}
        metrics[source] = arms
        base = arms['D']
        a = arms.get('aligned')
        eligible = bool(a and a['hard_recordings'] > 1 and a['miss_recordings'] > 1)
        rank_pass = bool(a and a['hard_frames'] and a['hard_concordance'] >
                         max(base['hard_concordance'], arms['mismatched']['hard_concordance']) + 1.e-12)
        reject_pass = bool(a and a['missed_rejected'] > arms['D_only']['missed_rejected'] and
                           a['covered_correct_accepted'] >= arms['D_only']['covered_correct_accepted'])
        gates[source] = dict(recording_support=eligible, rank_improved=rank_pass, rejection_improved=reject_pass,
                             advance=eligible and rank_pass and reject_pass)
        if not a:
            continue
        aligned = {r['case']: r for r in local if r['arm'] == 'aligned'}
        inventory = {}
        for arm in ('D', 'D_only', 'mismatched'):
            other = [r for r in local if r['arm'] == arm]
            deltas = [aligned[r['case']]['hard_frame'] - r['hard_frame'] for r in other if r['hard_frame'] is not None]
            inventory[arm] = dict(hard_better=sum(v > 1.e-12 for v in deltas), hard_worse=sum(v < -1.e-12 for v in deltas),
                                  hard_equal=sum(abs(v) <= 1.e-12 for v in deltas),
                                  covered_identity_recovered=sum(aligned[r['case']]['raw_correct'] and not r['raw_correct']
                                                                 for r in other if r['status'] == 'covered'),
                                  covered_identity_lost=sum(not aligned[r['case']]['raw_correct'] and r['raw_correct']
                                                            for r in other if r['status'] == 'covered'))
        comparisons[source] = inventory
    fold_diagnostics = {}
    for arm in ARMS[1:]:
        fold_diagnostics[arm] = {
            p: dict(folds=len(group),
                    no_rejection_class=sum(f['covered_train'] == 0 or f['absent_train'] == 0 for f in group),
                    no_hard_training=sum(f['paired_training_frames'] == 0 for f in group) if arm != 'D_only' else None)
            for p in PLATFORMS if (group := [f for f in fit if f['arm'] == arm and f['platform'] == p])
        }
    return dict(structure=structure, metrics=metrics, paired_comparisons=comparisons, gates=gates,
                fold_diagnostics=fold_diagnostics, joint_gate_passed=any(g['advance'] for g in gates.values()))


def write_rows(path, rows):
    with path.open('x', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    RESULTS.mkdir(exist_ok=True)
    destination = RESULTS / 'SUMMARY.json'
    assert not destination.exists()
    truth, pin = check_receipts()
    examples = [load_example(row, pin) for row in truth]
    outputs, fit = [], []
    for source in PLATFORMS:
        obs = [e for e in examples if e.platform == source]
        population, coefficients = folds(obs)
        outputs.extend(population)
        fit.extend(coefficients)
    cases = {e.case: f'A{i:03d}' for i, e in enumerate(sorted(examples, key=lambda e: e.case), 1)}
    recs = {v: 'G' + str(i + 1).zfill(2) for i, v in enumerate(sorted({e.recording for e in examples}))}
    private_eval = PRIVATE / 'evaluation'
    private_eval.mkdir(mode=0o700, exist_ok=True)
    atomic_json(private_eval / 'FIT.json', fit)
    atomic_json(private_eval / 'ROWS.json', outputs)
    scrubbed = [dict(r, case=cases[r['case']], recording=recs[r['recording']]) for r in outputs]
    write_rows(RESULTS / 'reference_outcomes.csv', scrubbed)
    compact = []
    for source in PLATFORMS:
        for rec in sorted({r['recording'] for r in scrubbed if r['platform'] == source}):
            for arm in ARMS:
                group = [r for r in scrubbed if r['recording'] == rec and r['arm'] == arm]
                if group:
                    compact.append(dict(platform=source, recording=rec, arm=arm, **summarize(group)))
    write_rows(RESULTS / 'recording_outcomes.csv', compact)
    summary = report(examples, outputs, fit)
    summary['provenance'] = dict(truth=sha(PRIVATE / 'TRUTH_SEAL.json'), predictions=sha(PRED / 'SEAL.json'),
                                 scorer=sha(Path(__file__)), plan=sha(PLAN), source_weights_updated=False,
                                 protected_data_access=False)
    atomic_json(destination, summary)
    print('Completed 108 source invitations. Joint gate passed:', summary['joint_gate_passed'], flush=True)


if __name__ == '__main__':
    run()
