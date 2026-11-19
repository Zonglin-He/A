"""Replay the published pair re-ranker and identify the actual winners it displaced."""
from collections import Counter, defaultdict

import numpy as np

from inputs import PRIVATE, RESULTS, atomic_json, load, outer_folds
from inputs import previous, write_table


def candidate(x, slot):
    owner = int(x.owners[slot])
    entity = owner in (x.example.gold, *x.example.present_foils) and owner >= 0
    group = np.flatnonzero(x.owners == owner)
    original_rep = bool(x.example.hard and entity and slot == group[x.example.unary[group].argmax()])
    return dict(owner=owner, type=x.owner_types[owner], c=bool(x.has_c[slot]), noun=bool(x.has_noun[slot]),
                in_old_pair_objective=bool(x.example.hard and entity), old_representative=original_rep)


def val(score_row, key):
    return float(score_row[key]) if score_row[key] else None


def describe(x, arm, values, weights, old):
    e = x.example
    original = previous.evaluate(e, e.unary, None, 'D')
    scored = previous.evaluate(e, values, None, arm)
    assert (old['rank'] or None) == (str(scored['rank']) if scored['rank'] is not None else None)
    assert (old['raw_correct'] == 'True') == scored['raw_correct']
    assert val(old, 'hard_frame') == scored['hard_frame']
    current = int(values.argmax())
    v = candidate(x, current)
    change = ('retained_correct' if scored['raw_correct'] else 'top1_lost') if original['raw_correct'] else (
        'top1_recovered' if scored['raw_correct'] else 'both_unverified_or_wrong')
    feature = x.centered if arm == 'aligned' else x.shifted
    centered = e.unary + feature @ weights
    replay = previous.evaluate(e, centered, None, 'same_coefficients_centered')
    rank0, rank1 = original['rank'], scored['rank']
    hard0, hard1 = original['hard_frame'], scored['hard_frame']
    blind = dict(case=x.alias, recording=x.rec, cohort=x.cohort, platform=e.platform, arm=arm, status=e.status,
                 original_rank=rank0, new_rank=rank1,
                 before_type=candidate(x, 0)['type'], after_type=v['type'],
                 missing_c=not v['c'], missing_bare_noun=not v['noun'],
                 winner_eligible_entity_in_prior_pair_loss=v['in_old_pair_objective'],
                 winner_eligible_representative_in_prior_pair_loss=v['old_representative'],
                 chosen_candidate_changed=current != 0, known_outcome=change,
                 hard_concordance_before=hard0, hard_concordance_after=hard1,
                 hard_improved=hard0 is not None and hard1 > hard0,
                 whole_pool_rank_worsened=rank0 is not None and rank1 > rank0,
                 positive_c_weight=bool(weights[0] > 0.))
    local = dict(case=x.alias, candidate_slot=current, dense_owner=v['owner'],
                 previous_slot=0, previous_owner=candidate(x, 0)['owner'],
                 original_d=float(e.unary[current]), old_delta=float(values[current] - e.unary[current]),
                 actual_c_weight=float(weights[0]), n_known_entities=len(set(x.owners) - {-1}),
                 flipped_after_centering=bool(centered.argmax() != current),
                 correct_after_centering=replay['raw_correct'], rank_after_centering=replay['rank'])
    blind.update(centered_fixed_weight_changes_winner=local['flipped_after_centering'],
                 centered_fixed_weight_selected_true=replay['raw_correct'])
    return blind, local


def report():
    data, published, provenance = load()
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / 'WINNER_AUDIT.json'
    assert not path.exists()
    rows, exact = [], []
    for train, tests in outer_folds(data):
        if not train:
            continue
        train_examples = [x.example for x in train]
        for arm in ('aligned', 'mismatched'):
            beta, _ = previous.fit_rank(train_examples, arm)
            for x in tests:
                values = x.example.unary + x.example.design[arm] @ beta
                audit, details = describe(x, arm, values, beta, published[(x.alias, arm)])
                if arm == 'aligned':
                    rows.append(audit)
                    exact.append(details)
    by_group = defaultdict(list)
    for a in rows:
        by_group[f"{a['cohort']}_{a['platform']}"].append(a)
    overview = {}
    for g, subset in by_group.items():
        lost = [r for r in subset if r['known_outcome'] == 'top1_lost']
        opposite = [r for r in subset if r['hard_improved'] and r['whole_pool_rank_worsened']]
        overview[g] = dict(n=len(subset), changed_winner=sum(r['chosen_candidate_changed'] for r in subset),
                           top1_lost=len(lost), top1_recovered=sum(r['known_outcome']=='top1_recovered' for r in subset),
                           loss_winner_owners=dict(Counter(r['after_type'] for r in lost)),
                           loss_winner_eligible_for_paired_supervision=sum(
                               r['winner_eligible_representative_in_prior_pair_loss'] for r in lost),
                           losses_missing_c=sum(r['missing_c'] for r in lost),
                           losses_missing_c_with_positive_coefficient=sum(
                               r['missing_c'] and r['positive_c_weight'] for r in lost),
                           losses_reversed_with_fixed_coefficient_centering=sum(
                               r['centered_fixed_weight_selected_true'] for r in lost),
                           local_win_global_rank_down=[r['case'] for r in opposite])
    write_table(RESULTS / 'winner_replay.csv', rows)
    atomic_json(PRIVATE / 'WINNER_DETAILS.json', exact)
    atomic_json(path, dict(old_aligned_and_misaligned_replays=2 * len(rows),
                           groups=overview, audit_examples=[r for r in rows if r['case'] in ('A078','A080')],
                           provenance=provenance))
    print('Exactly replayed old aligned and mismatched rows',2 * len(rows),flush=True)


if __name__ == '__main__':
    report()
