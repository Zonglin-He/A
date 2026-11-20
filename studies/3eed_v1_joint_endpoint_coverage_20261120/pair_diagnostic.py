"""Attribute frozen slate decisions using independently recovered endpoints after prediction."""
import argparse
from pathlib import Path

import numpy as np

from collect import OUT, PRIVATE, ROOT, SOURCES, read, save
from endpoint_score import arrays_to_owners, verify_inputs
from joint_coverage import ARMS, affinity, assignment, bearings, normalize_pair
from predict_all import DECISIONS
from score_roles import PANELS, anon, csv_file, inventory
from diagnose_roles import COHORTS, historical_endpoints, historical_records


def components(subject_assignment, reference_assignment, a, d, separation, order):
    """M_ij * Z equals subject mass * reference mass * average separation * conditional order."""
    left = subject_assignment * a
    right = reference_assignment * d
    subj, ref = left.sum(1), right.sum(1)
    unaries = subj[:, None] * ref[None, :]
    distinct = left @ separation @ right.T
    compatible = left @ (separation * order) @ right.T
    sep = np.divide(distinct, unaries, out=np.zeros_like(distinct), where=unaries > 0)
    lr = np.divide(compatible, distinct, out=np.zeros_like(distinct), where=distinct > 0)
    np.testing.assert_allclose(unaries * sep * lr, compatible, atol=1.e-14)
    return subj, ref, sep, lr, compatible


def attribute(z, dec, seed, truth, known):
    a, d = z['a'], z['d']
    sa, dr = z['mass_a'], z['mass_d']
    k = ARMS.index('joint_nomination')
    independent = ARMS.index('independent_nomination')
    ss, rr = dec['subjects'][k], dec['references'][k]
    cs, cr = assignment(affinity(ss, a)), assignment(affinity(rr, d))
    separation = 1 - affinity(a, d)
    lr = bearings(z['ha'], z['hd'], int(z['sign']))
    p = normalize_pair(sa, dr, separation, lr)
    sm, rm, distinct, order, unnormalized = components(cs, cr, sa, dr, separation, lr)
    masses = cs @ p @ cr.T
    normalizer = (sa[:, None] * dr[None, :] * separation * lr).sum()
    np.testing.assert_allclose(unnormalized / normalizer, masses, atol=1.e-12)
    win = np.unravel_index(masses.argmax(), masses.shape)
    # A true tie between identical native boxes may select a different index with identical geometry.
    assert np.array_equal(ss[win[0]], dec['raw'][k]) and np.array_equal(rr[win[1]], dec['refs'][k])
    chunks = [a, d, ss, rr, dec['subjects'][independent], dec['references'][independent]]
    result = {}
    subj, ref = known
    for rule, (oa, od, os, ore, indep_s, indep_r) in arrays_to_owners(chunks, truth, seed).items():
        good_s, good_r = oa == subj, od == ref
        valid_slate = (os == subj)[:, None] & (ore == ref)[None, :]
        valid_pool = good_s[:, None] & good_r[None, :]
        positive = bool(valid_pool.any() and p[valid_pool].max() > 0)
        chosen_correct = bool(valid_slate[win])
        obj = dict(pool_both=int(valid_pool.any()), positive_true_pair=int(positive),
                   independent_slate_both=int((indep_s == subj).any() and (indep_r == ref).any()),
                   joint_slate_both=int(valid_slate.any()), joint_winner_both=int(chosen_correct),
                   reference_covered_in_pool=int(good_r.any()),
                   joint_subject_correct=int(os[win[0]] == subj), joint_reference_correct=int(ore[win[1]] == ref),
                   true_subject_mass=float(p[good_s].sum()), true_reference_mass=float(p[:, good_r].sum()),
                   true_pair_mass=float(p[valid_pool].sum()),
                   full_best_correct_pair_rank=int(1 + (p > p[valid_pool].max()).sum()) if positive else '',
                   marginal_subject_correct=int(os[np.argmax(cs @ p.sum(1))] == subj),
                   given_true_reference_subject_correct='', separation_or_order_zero='',
                   subject_log_gap='', reference_log_gap='', separation_log_gap='', order_log_gap='', total_log_gap='')
        conditioned = cs @ p[:, good_r].sum(1)
        if conditioned.sum() > 0:
            obj['given_true_reference_subject_correct'] = int(os[conditioned.argmax()] == subj)
        if valid_slate.any():
            best = np.unravel_index(np.where(valid_slate, masses, -1).argmax(), masses.shape)
            i, j = best
            u, v = win
            factor_good = np.array([sm[i], rm[j], distinct[i, j], order[i, j]])
            factor_win = np.array([sm[u], rm[v], distinct[u, v], order[u, v]])
            obj['separation_or_order_zero'] = int(np.any(factor_good <= 0))
            if np.min(factor_good) > 0 and np.min(factor_win) > 0:
                gaps = np.log(factor_good) - np.log(factor_win)
                total = np.log(masses[best]) - np.log(masses[win])
                np.testing.assert_allclose(gaps.sum(), total, atol=1.e-10, rtol=0)
                obj.update(zip(('subject_log_gap','reference_log_gap','separation_log_gap','order_log_gap'), gaps))
                obj['total_log_gap'] = float(total)
        result[rule] = obj
    return result


def run(repo):
    verify_inputs()
    history = historical_records(repo)
    rows = []
    for panel, cohort in COHORTS.items():
        for i, entry in enumerate(read(PANELS[panel])):
            row = entry['row']
            target = row['platform']
            truth, _ = inventory(row['metapath'], target, int(row['suffix'] or 0), int(row['object_index']))
            known, _ = historical_endpoints(cohort, entry, truth, history)
            if None in known:
                continue
            for src in SOURCES:
                if src == target:
                    continue
                name = f'{panel}_{i:03d}_{src}.npz'
                tag = f"{cohort}/{entry['case']:02d}_{src}.npz"
                with np.load(PRIVATE/name, allow_pickle=False) as z, np.load(DECISIONS/name, allow_pickle=False) as dec:
                    if not dec['applied']:
                        continue
                    with np.load(ROOT/'crossplatform/round171/predictions'/tag, allow_pickle=False) as old:
                        judgements = attribute(z, dec, old['seeds'], truth, known)
                    for rule, result in judgements.items():
                        rows.append(dict(group=panel, case=anon(f'{panel}|{i}'), source=src, target=target, rule=rule, **result))
    csv_file(OUT/'pair_attribution.csv', rows)
    summary = {}
    for rule in ('unique_iou', 'seed_ownership'):
        group = [r for r in rows if r['rule'] == rule]
        ranks = [r['full_best_correct_pair_rank'] for r in group if r['positive_true_pair']]
        can_condition = [r for r in group if r['given_true_reference_subject_correct'] != '']
        simultaneous = [r for r in group if r['joint_slate_both']]
        lost = [r for r in simultaneous if not r['joint_winner_both']]
        rescued = [r for r in simultaneous if not r['independent_slate_both']]
        summary[rule] = dict(active_known=len(group), full_both=sum(r['pool_both'] for r in group),
            positive=len(ranks), median_best_rank=(float(np.median(ranks)) if ranks else None),
            slate_both=len(simultaneous), new_over_independent=len(rescued),
            lost_to_independent=sum(r['independent_slate_both'] and not r['joint_slate_both'] for r in group),
            new_slate_converted=sum(r['joint_winner_both'] for r in rescued), lost_covered_pairs=len(lost),
            structural_zero=sum(r['separation_or_order_zero'] for r in lost),
            correct_given_ref_eligible=len(can_condition),
            marginal_correct=sum(r['marginal_subject_correct'] for r in can_condition),
            given_verified_ref_correct=sum(r['given_true_reference_subject_correct'] for r in can_condition),
            ref_rescues=sum(r['given_true_reference_subject_correct'] > r['marginal_subject_correct'] for r in can_condition),
            ref_harms=sum(r['given_true_reference_subject_correct'] < r['marginal_subject_correct'] for r in can_condition))
    save(OUT/'attribution_summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    run(parser.parse_args().archive)
