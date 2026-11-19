"""Calibrate coverage and the correctness of the actual selected candidate separately."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from inputs import Input
from rank import features, fit_rank, scores_for, winner_status


RANK_MODELS = ('D', 'pool_centered', 'pool_shuffled')
RULES = {
    'D_keep': ('D', 'keep', 'actual'),
    'D_winner': ('D', 'winner', 'actual'),
    'pool_keep': ('pool_centered', 'keep', 'actual'),
    'pool_coverage_D_slot': ('pool_centered', 'coverage', 'D'),
    'pool_coverage_actual': ('pool_centered', 'coverage', 'actual'),
    'pool_winner_D_slot': ('pool_centered', 'winner', 'D'),
    'pool_winner_actual': ('pool_centered', 'winner', 'actual'),
    'shuffle_winner_actual': ('pool_shuffled', 'winner', 'actual'),
}


@dataclass
class CandidateReadout:
    example: Input
    arm: str
    logits: np.ndarray
    index: int
    outcome: str


def readout(x, arm, f):
    logits = scores_for(x, f, arm)
    index = int(logits.argmax())
    return CandidateReadout(x, arm, logits, index, winner_status(x, index))


def observation(entry, slot):
    x = entry.example
    k = entry.index if slot == 'actual' else 0
    if len(entry.logits) > 1:
        gap = entry.logits[k] - np.max(np.delete(entry.logits, k))
    else:
        gap = 0.
    return np.r_[x.example.confidence, features(x, entry.arm)[k],
                 np.clip(x.example.unary[k], -8., 0.), np.clip(gap, -8., 8.)]


def training_label(entry, task):
    if task == 'coverage':
        if entry.example.example.status in ('covered', 'natural_miss'):
            return int(entry.example.example.status == 'natural_miss')
    elif task == 'winner':
        if entry.outcome in ('correct', 'wrong_covered', 'natural_miss'):
            return int(entry.outcome != 'correct')
    return None


def fit_binary(rows, task, slot):
    labelled = [(observation(x, slot), y) for x in rows if (y := training_label(x, task)) is not None]
    labels = np.array([y for _, y in labelled], dtype=float)
    bad = int(labels.sum())
    good = len(labels) - bad
    if not bad or not good:
        return (lambda _: float(bool(bad))), dict(negative=good, positive=bad, mode='single_class_or_no_labels')

    actual = np.stack([v for v, _ in labelled])
    center = actual.mean(axis=0)
    scale = actual.std(axis=0)
    scale[scale <= 1.e-6] = 1.
    standardized = np.c_[np.ones(len(actual)), (actual - center) / scale]
    balance = np.where(labels == 1., .5 / bad, .5 / good)
    ridge = np.r_[0., np.full(standardized.shape[1] - 1, .15)]

    def objective(w):
        score = standardized @ w
        regularizer = ridge * w
        cost = balance @ (np.logaddexp(0., score) - labels * score) + .5 * (w @ regularizer)
        gradient = standardized.T @ (balance * (expit(score) - labels)) + regularizer
        return cost, gradient

    opt = minimize(objective, np.zeros(standardized.shape[1]), jac=True, method='L-BFGS-B')
    assert opt.success and np.isfinite(opt.x).all()

    def predict(row):
        return float(expit(opt.x[0] + opt.x[1:] @ ((observation(row, slot) - center) / scale)))

    evidence = dict(mode='fitted', negative=good, positive=bad, coefficients=opt.x.tolist(),
                    center=center.tolist(), scale=scale.tolist())
    return predict, evidence


def inner_partitions(training):
    by_recording = defaultdict(list)
    for row in training:
        by_recording[row.rec].append(row)
    roles = sorted(by_recording, key=lambda k: hashlib.sha256(('winner-fit-v1|' + k).encode()).digest())
    quota = min(5, len(roles))
    for i in range(quota):
        held = set(roles[i::quota])
        yield [row for r in roles if r not in held for row in by_recording[r]], [
            row for r in roles if r in held for row in by_recording[r]]


def inner_ranks(training):
    out = {arm: [] for arm in RANK_MODELS}
    proof = []
    for fit, held in inner_partitions(training):
        assert not {x.rec for x in fit} & {x.rec for x in held}
        proof.append(dict(fit_cases=[x.alias for x in fit], held_cases=[x.alias for x in held],
                          rank_fallback_to_D=not bool(fit)))
        for arm in RANK_MODELS:
            weights = fit_rank(fit, arm)
            out[arm].extend(readout(x, arm, weights) for x in held)
    for rows in out.values():
        assert sorted(row.example.alias for row in rows) == sorted(x.alias for x in training)
    return out, proof


def gate_row(rule, entry, probability):
    x = entry.example
    reject = probability is not None and probability >= .5
    return dict(case=x.alias, recording=x.rec, cohort=x.cohort, platform=x.example.platform,
                rule=rule, rank_arm=entry.arm, status=x.example.status,
                winner_status=entry.outcome, reject=reject,
                correct_accepted=entry.outcome == 'correct' and not reject, risk_score=probability)


def evaluate(training, testing, ranks):
    if not training:
        return [gate_row('D_keep', readout(x, 'D', ranks['D']), None) for x in testing], None
    nested, partitions = inner_ranks(training)
    actual = {arm: [readout(x, arm, ranks[arm]) for x in testing] for arm in RANK_MODELS}
    values, fits = [], dict(inner_folds=partitions, rules={})
    for rule, (arm, target, slot) in RULES.items():
        if target == 'keep':
            values.extend(gate_row(rule, row, None) for row in actual[arm])
            continue
        prediction, stats = fit_binary(nested[arm], target, slot)
        fits['rules'][rule] = stats
        values.extend(gate_row(rule, row, prediction(row)) for row in actual[arm])
    return values, fits


def auc(correct, wrong):
    if (not correct or not wrong or
            any(r['risk_score'] is None for r in [*correct, *wrong])):
        return None
    p = np.array([r['risk_score'] for r in wrong])
    n = np.array([r['risk_score'] for r in correct])
    return float(np.mean((p[:, None] > n) + .5 * (p[:, None] == n)))


def verifier_summary(rows):
    available = defaultdict(list)
    for x in rows:
        available[x['winner_status']].append(x)
    good, wrong, miss = (available[k] for k in ('correct','wrong_covered','natural_miss'))
    unassigned = available['unassigned_covered']
    good_accept = sum(not r['reject'] for r in good)
    bad_reject = sum(r['reject'] for r in wrong)
    return dict(n=len(rows), recordings=len({r['recording'] for r in rows}), covered_correct=len(good),
                correct_accepted=good_accept, correct_rejected=len(good)-good_accept,
                covered_wrong=len(wrong), covered_wrong_rejected=bad_reject,
                covered_unassigned=len(unassigned), covered_unassigned_rejected=sum(r['reject'] for r in unassigned),
                natural_miss=len(miss), natural_miss_rejected=sum(r['reject'] for r in miss),
                total_rejected=sum(r['reject'] for r in rows), winner_auc_covered=auc(good,wrong),
                winner_balanced_accuracy_covered=None if (not good or not wrong) else
                    .5 * (good_accept / len(good) + bad_reject / len(wrong)))
