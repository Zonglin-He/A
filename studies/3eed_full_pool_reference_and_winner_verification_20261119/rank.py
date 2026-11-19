"""Fit and score the same max-over-boxes physical instance readout."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from inputs import previous


RANKS = ('D', 'legacy_pair', 'pair_entity', 'pool_same_frames', 'pool_all', 'pool_centered', 'pool_shuffled')
REGULARIZATION = .15


@dataclass
class Fit:
    weights: np.ndarray
    frames: int
    entities: int
    excluded_unknown_boxes: int
    optimizer: str
    steps: int = 0


def features(x, arm):
    if arm in ('D', 'pool_centered'):
        return x.centered
    if arm == 'pool_shuffled':
        return x.shifted
    return x.features


def training_groups(cases, arm):
    for x in cases:
        e = x.example
        if e.status != 'covered' or (arm in ('pair_entity', 'pool_same_frames') and not e.hard):
            continue
        rivals = e.present_foils if arm == 'pair_entity' else sorted(set(x.owners) - {-1, e.gold})
        if len(rivals):
            groups = [np.flatnonzero(x.owners == owner) for owner in (e.gold, *rivals)]
            assert all(len(g) and (x.owners[g] >= 0).all() for g in groups)
            yield x, groups


def offsets(sizes):
    return np.cumsum([0, *sizes[:-1]])


def rank_loss(samples, pair):
    counts = [len(groups) for _, groups, _ in samples]
    frame_first = offsets(counts)
    group_lengths = [len(g) for _, groups, _ in samples for g in groups]
    entity_first = offsets(group_lengths)
    base = np.concatenate([x.example.unary[np.concatenate(groups)] for x, groups, _ in samples])
    design = np.concatenate([z[np.concatenate(groups)] for _, groups, z in samples])
    assert len(base) == len(design) == sum(group_lengths)

    def objective(weights):
        score = np.maximum.reduceat(base + design @ weights, entity_first)
        positive = score[frame_first]
        if pair:
            margins = score - np.repeat(positive, counts)
            losses = np.logaddexp(0., margins)
            losses[frame_first] = 0.
            loss = np.mean(np.add.reduceat(losses, frame_first) / (np.asarray(counts) - 1))
        else:
            ceiling = np.maximum.reduceat(score, frame_first)
            normal = np.add.reduceat(np.exp(score - np.repeat(ceiling, counts)), frame_first)
            loss = np.mean(ceiling + np.log(normal) - positive)
        return float(loss + REGULARIZATION * .5 * (weights @ weights))

    return objective


def fit_rank(train, arm):
    if arm == 'D':
        return Fit(np.zeros(3), 0, 0, 0, 'fixed')
    if arm == 'legacy_pair':
        data = [x for x in train if x.example.hard]
        w, n = previous.fit_rank([x.example for x in train], 'aligned')
        assert n == len(data)
        return Fit(w, n, sum(1 + len(x.example.present_foils) for x in data),
                   sum((x.owners == -1).sum().item() for x in data), 'legacy')
    tagged = [(x, groups, features(x, arm)) for x, groups in training_groups(train, arm)]
    if not tagged:
        return Fit(np.zeros(3), 0, 0, 0, 'empty')

    objective = rank_loss(tagged, pair=arm == 'pair_entity')
    starting = (np.zeros(3), np.array([.4, 0., 0.]), np.array([-.4, 0., 0.]))
    best = min(starting, key=objective)
    best_loss = objective(best)
    evaluations, any_converged = 0, False
    for init in starting:
        solution = minimize(objective, init, method='Powell', bounds=[(-2., 2.)] * 3,
                            options={'xtol': 1.e-5, 'ftol': 1.e-8, 'maxfev': 240})
        evaluations += solution.nfev
        any_converged |= bool(solution.success)
        assert np.isfinite(solution.fun) and np.isfinite(solution.x).all()
        if solution.fun < best_loss:
            best, best_loss = solution.x, solution.fun
    assert np.abs(best).max() <= 2. + 1.e-12 and best_loss <= objective(np.zeros(3)) + 1.e-12
    return Fit(best, len(tagged), sum(len(g) for _, g, _ in tagged),
               sum((x.owners == -1).sum().item() for x, _, _ in tagged),
               'powell' if any_converged else 'powell_budget', evaluations)


def scores_for(x, fit, arm):
    return x.example.unary + features(x, arm) @ fit.weights


def winner_status(x, index):
    e = x.example
    if e.status == 'natural_miss':
        return 'natural_miss'
    if e.status == 'covered':
        owner = x.owners[index]
        return 'correct' if owner == e.gold else ('wrong_covered' if owner >= 0 else 'unassigned_covered')
    return e.status


def instance_stats(x, scores):
    e = x.example
    known = set(x.owners) - {-1}
    correct_rank, box_rank, unassigned_ahead, hard = None, None, None, None
    if e.status == 'covered':
        pos = scores[x.owners == e.gold].max()
        unassigned_ahead = int(((x.owners == -1) & (scores > pos)).sum())
        correct_rank = 1 + unassigned_ahead + sum(int(scores[x.owners == k].max() > pos) for k in known - {e.gold})
        box_rank = 1 + int((scores > pos).sum())
        if e.hard:
            foil = np.array([scores[x.owners == k].max() for k in e.present_foils])
            delta = pos - foil
            hard = float(((delta > 1.e-10).sum() + .5 * (np.abs(delta) <= 1.e-10).sum()) / len(delta))
    return dict(instance_rank=correct_rank, box_rank=box_rank, unassigned_ahead=unassigned_ahead,
                represented_foils=len(e.present_foils), hard_concordance=hard,
                pool_boxes=len(scores), known_instances=len(known), unknown_boxes=int((x.owners == -1).sum()))


def rank_row(x, arm, scores):
    e = x.example
    selected = int(scores.argmax())
    outcome = winner_status(x, selected)
    original = winner_status(x, 0)
    return dict(case=x.alias, recording=x.rec, cohort=x.cohort, platform=e.platform,
                arm=arm, status=e.status, chosen_type=x.owner_types[int(x.owners[selected])],
                winner_status=outcome, selected_unchanged=selected == 0,
                recovered=outcome == 'correct' and original != 'correct',
                lost=outcome != 'correct' and original == 'correct', **instance_stats(x, scores))


def rank_summary(rows):
    n = len(rows)
    coverage = [r for r in rows if r['status'] == 'covered']
    stats = Counter(r['winner_status'] for r in coverage)
    mrr = None if not coverage else float(np.mean([1. / r['instance_rank'] for r in coverage]))
    box_mrr = None if not coverage else float(np.mean([1. / r['box_rank'] for r in coverage]))
    return dict(n=n, recordings=len({r['recording'] for r in rows}), covered=len(coverage),
                correct=stats['correct'], wrong_covered=stats['wrong_covered'],
                unassigned_covered=stats['unassigned_covered'], instance_mrr=mrr, box_mrr=box_mrr,
                missing=sum(r['status'] == 'natural_miss' for r in rows), recovered=sum(r['recovered'] for r in rows),
                lost=sum(r['lost'] for r in rows),
                covered_with_unassigned_above=sum(r['unassigned_ahead'] > 0 for r in coverage),
                eligible_pair_frames=sum(r['hard_concordance'] is not None for r in rows))
