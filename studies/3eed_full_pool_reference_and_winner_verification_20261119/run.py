"""Run locked source-only complete-pool ranking and nested winner calibration."""
from collections import Counter, defaultdict
from dataclasses import asdict

from inputs import (HOME, PRIVATE, RESULTS, PROTOCOL, atomic_json, load, outer_folds, read, sha, write_table)
from rank import RANKS, fit_rank, rank_row, rank_summary, scores_for
from verify_winner import RULES, evaluate, verifier_summary


PROGRAM = ('inputs.py','rank.py','verify_winner.py','run.py')


def metrics(rows, key, variants, summary):
    groups = defaultdict(list)
    for row in rows:
        unit = row['cohort'] + '_' + row['platform']
        groups[unit].append(row)
        if row['platform'] == 'waymo':
            groups['waymo_pooled'].append(row)
    output = {}
    for name, group in sorted(groups.items()):
        present = {v[key] for v in group}
        output[name] = {variant: summary([row for row in group if row[key] == variant])
                        for variant in variants if variant in present}
    return output


def gates(rank, verify):
    all_waymo = rank['waymo_pooled']
    raw, candidate = all_waymo['D'], all_waymo['pool_centered']
    fixed_comparisons = (raw, all_waymo['legacy_pair'], all_waymo['pool_shuffled'])
    by_panel = [rank[p] for p in ('initial_waymo','enriched_waymo')]
    ranking = dict(
        exceeds_correct_against_all_controls=candidate['correct'] > max(r['correct'] for r in fixed_comparisons),
        exceeds_mrr_against_all_controls=candidate['instance_mrr'] > max(r['instance_mrr'] for r in fixed_comparisons),
        no_panel_loses_to_D=all(m['pool_centered']['correct'] >= m['D']['correct'] for m in by_panel),
    )
    holdout = verify['enriched_waymo']
    winner, coverage, shuffled = (holdout[r] for r in (
        'pool_winner_actual','pool_coverage_actual','shuffle_winner_actual'))
    wba, sba = winner['winner_balanced_accuracy_covered'], shuffled['winner_balanced_accuracy_covered']
    controls = [holdout[r] for r in ('D_keep','D_winner','pool_coverage_actual')]
    rejection = dict(
        rejects_more_known_wrong_covered=winner['covered_wrong_rejected'] > coverage['covered_wrong_rejected'],
        exceeds_mismatched_balanced_accuracy=wba is not None and sba is not None and wba > sba,
        retains_correct_accepts=winner['correct_accepted'] >= max(r['correct_accepted'] for r in controls),
    )
    rank_pass = all(ranking.values())
    verify_pass = all(rejection.values())
    return dict(ranking=ranking, winner_reliability=rejection, rank_pass=rank_pass,
                winner_reliability_pass=verify_pass, joint_experiment_eligible=rank_pass and verify_pass)


def fit_record(f):
    fields = asdict(f)
    fields['weights'] = f.weights.tolist()
    return fields


def run():
    RESULTS.mkdir(exist_ok=True)
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    audit = read(RESULTS / 'WINNER_AUDIT.json')
    assert audit['old_aligned_and_misaligned_replays'] == 216
    marker = PRIVATE / 'RUN_LOCK.json'
    assert not marker.exists()
    cohort, published, provenance = load()
    assert audit['provenance'] == provenance
    lock = dict(program={f:sha(HOME / f) for f in PROGRAM}, protocol=sha(PROTOCOL), inputs=provenance,
                old_replay=sha(RESULTS / 'WINNER_AUDIT.json'), score_rows=144,
                source_only=True, protected_data_access=False, forward_passes=0)
    atomic_json(marker, lock)
    rankings, verifications, training_log = [], [], []
    for i, (fit, tested) in enumerate(outer_folds(cohort), 1):
        arms = RANKS if fit else ('D',)
        fits = {arm: fit_rank(fit, arm) for arm in arms}
        for x in tested:
            for arm, f in fits.items():
                row = rank_row(x, arm, scores_for(x, f, arm))
                if arm in ('D','legacy_pair'):
                    old = published[(row['case'], 'D' if arm == 'D' else 'aligned')]
                    assert (old['raw_correct'] == 'True') == (row['winner_status'] == 'correct')
                    assert (int(old['rank']) if old['rank'] else None) == row['box_rank']
                rankings.append(row)
        batch, nested = evaluate(fit, tested, fits)
        verifications.extend(batch)
        training_log.append(dict(held=tested[0].rec, cohort=tested[0].cohort, platform=tested[0].example.platform,
                                 cases=[x.alias for x in tested], calibration=[x.alias for x in fit],
                                 rank={k:fit_record(f) for k,f in fits.items()}, verification=nested))
        if i % 10 == 0:
            print('Completed source recording folds',i,flush=True)
    assert len(training_log) == 58
    write_table(RESULTS / 'rank_rows.csv', rankings)
    write_table(RESULTS / 'verification_rows.csv', verifications)
    atomic_json(PRIVATE / 'FIT_DETAILS.json', training_log)
    rank = metrics(rankings,'arm', RANKS, rank_summary)
    verify = metrics(verifications,'rule', RULES, verifier_summary)
    verdict = gates(rank, verify)
    opt_state = Counter(v['optimizer'] for f in training_log for v in f['rank'].values())
    payload = dict(frames=144, recordings=58, rank=rank, verification=verify,
                   ranking_optimizer_statuses=dict(opt_state), decision=verdict, run_lock=sha(marker))
    atomic_json(RESULTS / 'SUMMARY.json', payload)
    print('Source decision',verdict,flush=True)


if __name__ == '__main__':
    run()
