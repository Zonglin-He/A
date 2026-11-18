"""Replay and perturb the archived camera interval calculation without moving any box."""
from itertools import product

import numpy as np

from labels166 import box_masks
from predict171 import camera_relation
from retrieval167 import components, support_intervals, support_matrix


TOL = 1e-10


def recover(d):
    """Give every subject its distinct compatible pre-PCA support masks."""
    subject = d['subject_boxes']
    donors = d['D_natural_boxes']
    seeds = d['seeds']
    native = support_matrix(np.c_[donors[:, :6], np.zeros(len(donors))], seeds, len(donors))
    flags = [[] for _ in subject]
    axis = support_matrix(np.c_[subject[:, :6], np.zeros(len(subject))], seeds, len(subject))
    for masks in (axis, box_masks(subject, seeds)):
        selected, sub_stats, native_stats = camera_relation(masks, native, d['uv'], d['visible'])
        np.testing.assert_array_equal(native_stats, d['native_stats'])
        match = (sub_stats == d['subject_stats']).all(1) & (selected == d['native_left']).all(1)
        for i in np.flatnonzero(match):
            if not any(np.array_equal(old, masks[i]) for old in flags[i]):
                flags[i].append(masks[i])
    assert all(flags), 'An archived subject mask cannot be recovered from its observed statistics and relation row'
    return flags, native


def factor(masks, visible, uv, sign):
    lo, hi, certainty = support_intervals(masks, visible, uv)
    subjects = np.arange(0, len(lo), 2)
    refs = subjects + 1
    relation, _ = components(subjects, refs, lo, hi, certainty, np.zeros((len(lo), len(lo))))
    return sign * relation.diagonal(), (hi - lo)[refs]


def one_seed(d, recovered, winning, correct, unary_margin):
    """Hold unaries constant and perturb the same visible point on both sides of a comparison."""
    w_sub, w_ref = winning
    if w_ref >= len(d['D_natural_boxes']):
        out = dict(perturbations=0, distinct_seeds_removed=0, margin_min='', margin_max='',
                   single_drop_reverses='', single_drop_closes='')
        for p in ('winner', 'correct'):
            out.update({p+'_jackknife_relation_min': '', p+'_jackknife_relation_max': '',
                        p+'_jackknife_width_max_change': '', p+'_jackknife_sign_changes': 0,
                        p+'_jackknife_opposite_sign': 0, p+'_jackknife_zero': 0})
        return out
    subjects, donors = recovered
    sign = int(d['sign'])
    visible, uv = d['visible'], d['uv']
    visible_inputs = []
    interval_width_change = [[], []]
    relation_changes = [[], []]
    gaps = []
    if correct is None:
        variants = ((s, donors[w_ref]) for s in subjects[w_sub])
        original = (sign * d['native_left'][w_sub, w_ref],)
    else:
        c_sub, c_ref = correct
        choices = (((x, x) for x in subjects[w_sub]) if c_sub == w_sub else product(subjects[w_sub], subjects[c_sub]))
        variants = ((w, donors[w_ref], c, donors[c_ref]) for w, c in choices)
        original = (sign * d['native_left'][w_sub, w_ref], sign * d['native_left'][c_sub, c_ref])
    for variant in variants:
        masks = np.stack(variant)
        old, widths = factor(masks, visible, uv, sign)
        np.testing.assert_array_equal(old, np.asarray(original))
        perturbable = np.flatnonzero(visible & masks.any(0))
        visible_inputs.extend(perturbable.tolist())
        for j in perturbable:
            sample = visible.copy()
            sample[j] = False
            new, narrower = factor(masks, sample, uv, sign)
            for i, value in enumerate(new):
                relation_changes[i].append(float(value))
                interval_width_change[i].append(abs(float(narrower[i] - widths[i])))
            if correct is not None:
                gaps.append(float(unary_margin + new[0] - new[1]))
    out = dict(perturbations=len(relation_changes[0]), distinct_seeds_removed=len(set(visible_inputs)),
               margin_min=min(gaps) if gaps else '', margin_max=max(gaps) if gaps else '',
               single_drop_reverses=int(min(gaps) < -TOL) if gaps else '',
               single_drop_closes=int(min(gaps) <= TOL) if gaps else '')
    for i, prefix in enumerate(('winner', 'correct')):
        changed = relation_changes[i]
        sign_change = ([np.sign(x) != np.sign(original[i]) for x in changed] if changed else [])
        polarity_flip = ([x * original[i] < 0 for x in changed] if changed else [])
        out.update({prefix+'_jackknife_relation_min': min(changed) if changed else '',
                    prefix+'_jackknife_relation_max': max(changed) if changed else '',
                    prefix+'_jackknife_width_max_change': max(interval_width_change[i]) if changed else '',
                    prefix+'_jackknife_sign_changes': sum(sign_change),
                    prefix+'_jackknife_opposite_sign': sum(polarity_flip),
                    prefix+'_jackknife_zero': sum(x == 0. for x in changed)})
    return out
