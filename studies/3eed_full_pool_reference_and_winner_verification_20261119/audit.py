"""Independent verification of held-recording boundaries, slot scores, losses and totals."""
from collections import Counter, defaultdict
from pathlib import Path
import math

import numpy as np
from scipy.special import expit, logsumexp

from inputs import HOME, PRIVATE, RESULTS, atomic_json, csv_rows, load, read, sha
from rank import rank_loss, training_groups


SHAPES = {
    'D_keep': ('D', None), 'D_winner': ('D', 'actual'), 'pool_keep': ('pool_centered', None),
    'pool_coverage_D_slot': ('pool_centered', 'D'), 'pool_coverage_actual': ('pool_centered', 'actual'),
    'pool_winner_D_slot': ('pool_centered', 'D'), 'pool_winner_actual': ('pool_centered', 'actual'),
    'shuffle_winner_actual': ('pool_shuffled', 'actual'),
}


def values_for(sample, arm):
    if arm in ('D','pool_centered'):
        return sample.centered
    if arm == 'pool_shuffled':
        return sample.shifted
    return sample.example.design['aligned']


def verify_splits(fold, population):
    testing = {i for i,x in population.items() if x.rec == fold['held']}
    fitting = {i for i,x in population.items() if x.rec != fold['held'] and x.example.platform == fold['platform']
               and (fold['cohort'] == 'enriched' or x.cohort == 'initial')}
    assert set(fold['cases']) == testing and len(fold['cases']) == len(testing)
    assert set(fold['calibration']) == fitting and len(fold['calibration']) == len(fitting)
    assert testing and not (testing & fitting)
    v = fold['verification']
    if v is None:
        assert not fitting and fold['platform'] == 'quad'
        return 0
    counts = Counter()
    for inner in v['inner_folds']:
        a, b = set(inner['fit_cases']), set(inner['held_cases'])
        assert a | b == fitting and not a & b
        assert not {population[x].rec for x in a} & {population[y].rec for y in b}
        counts.update(b)
    assert counts == dict.fromkeys(fitting,1)
    return len(v['inner_folds'])


def independent_winner(sample, chosen):
    e = sample.example
    if e.status != 'covered':
        return e.status
    k = sample.owners[chosen]
    if k < 0:
        return 'unassigned_covered'
    return 'correct' if e.gold == k else 'wrong_covered'


def audit_cases(folds, data, rank_rows, risk_rows):
    m = {x.alias:x for x in data}
    expected_rank, expected_risk = {}, {}
    inner = 0
    for f in folds:
        inner += verify_splits(f, m)
        for key in f['cases']:
            x = m[key]
            base = x.example.unary
            for arm, fit in f['rank'].items():
                q = x.features if arm == 'legacy_pair' else values_for(x,arm)
                scores = base + q @ np.asarray(fit['weights'])
                slot = np.argsort(-scores,kind='stable')[0]
                placed = independent_winner(x, slot)
                rank, n_unassigned = None, None
                if x.example.status == 'covered':
                    correct = max(scores[x.owners == x.example.gold])
                    grouped = {}
                    for i, (s, owner) in enumerate(zip(scores,x.owners)):
                        tag = ('owner', owner) if owner >= 0 else ('unknown', i)
                        grouped[tag] = max(grouped.get(tag,-math.inf),s)
                    rank = 1 + sum(s > correct for s in map(float,grouped.values()))
                    n_unassigned = sum(i[0]=='unknown' and s > correct for i,s in grouped.items())
                expected_rank[(key,arm)] = (placed, rank, n_unassigned)
            for rule in (SHAPES if f['verification'] else ('D_keep',)):
                arm, focus = SHAPES[rule]
                design = values_for(x,arm)
                scores = base + design @ np.asarray(f['rank'][arm]['weights'])
                winner = int(np.argmax(scores))
                p = None
                if focus is not None:
                    k = winner if focus == 'actual' else 0
                    fit = f['verification']['rules'][rule]
                    if fit['mode'] == 'fitted':
                        margin = max(-8.,min(8.,scores[k] - max(np.delete(scores,k)))) if len(scores)>1 else 0.
                        features = np.array([*x.example.confidence,*design[k],max(-8.,min(0.,base[k])),margin])
                        z = (features - np.asarray(fit['center'])) / np.asarray(fit['scale'])
                        p = float(expit(np.dot([1.,*z],fit['coefficients'])))
                    else:
                        assert not fit['negative'] or not fit['positive']
                        p = float(bool(fit['positive']))
                expected_risk[(key,rule)] = (independent_winner(x,winner),p)

    assert len(expected_rank) == len(rank_rows) == len({(r['case'],r['arm']) for r in rank_rows})
    for r in rank_rows:
        placed, rank, n_unassigned = expected_rank[r['case'],r['arm']]
        assert (r['winner_status'],r['instance_rank'],r['unassigned_ahead']) == (
            placed,'' if rank is None else str(rank),'' if n_unassigned is None else str(n_unassigned))
    assert len(expected_risk) == len(risk_rows) == len({(r['case'],r['rule']) for r in risk_rows})
    for r in risk_rows:
        placed, p = expected_risk[r['case'],r['rule']]
        assert placed == r['winner_status']
        assert r['reject'] == str(p is not None and p >= .5)
        assert r['correct_accepted'] == str(placed=='correct' and r['reject']=='False')
        if p is None:
            assert not r['risk_score']
        else:
            assert np.isclose(float(r['risk_score']),p,atol=1.e-12,rtol=0.)
    return inner, len(expected_rank), len(expected_risk)


def slow_objective(cases, arm, weights):
    losses, count, unassigned, instance_count = [], 0, 0, 0
    for a in cases:
        e = a.example
        known = {int(k) for k in e.owners if k >= 0}
        ref = e.gold
        foils = set(e.verified_foils) & known
        if e.status!='covered':
            continue
        if arm in ('pair_entity','pool_same_frames') and not foils:
            continue
        choices = foils | {ref} if arm=='pair_entity' else known
        if len(choices) < 2:
            continue
        c = e.unary + values_for(a,arm) @ weights
        logits = {i:max(float(c[j]) for j, owner in enumerate(e.owners) if owner==i) for i in choices}
        truth = logits.pop(ref)
        if arm=='pair_entity':
            losses.append(np.mean([np.logaddexp(0.,v-truth) for v in logits.values()]))
        else:
            losses.append(logsumexp(list(logits.values())+[truth])-truth)
        count += 1
        instance_count += len(choices)
        unassigned += sum(owner == -1 for owner in e.owners)
    return float(np.mean(losses)+.075*(weights @ weights)), count, instance_count, int(unassigned)


def audit_rank_objectives(data):
    panels = [[x for x in data if x.example.platform==s] for s in ('waymo','drone','quad')]
    checked = 0
    for panel in panels:
        for arm in ('pair_entity','pool_same_frames','pool_all','pool_centered','pool_shuffled'):
            members = [(a, groups, values_for(a,arm)) for a,groups in training_groups(panel,arm)]
            if not members:
                continue
            obj = rank_loss(members, pair=arm=='pair_entity')
            for trial in ((0.,0.,0.),(.3,-.4,.6),(-.32,.8,-.12)):
                b = np.asarray(trial)
                cost, count, instances, unknown = slow_objective(panel,arm,b)
                assert count == len(members)
                assert len([g for _,groups,_ in members for g in groups])==instances
                assert unknown==sum(np.count_nonzero(a.owners == -1) for a,_,_ in members)
                assert all((a.owners[g]>=0).all() for a,groups,_ in members for g in groups)
                assert abs(cost-obj(b)) < 1.e-10
                checked += 1
    return checked


def summary_arithmetic(rank_rows, risk_rows, out):
    for scope, arms in out['rank'].items():
        for arm, m in arms.items():
            r=[x for x in rank_rows if (
                x['platform']=='waymo' if scope=='waymo_pooled' else x['cohort']+'_'+x['platform']==scope)
               and x['arm']==arm]
            vals=Counter(x['winner_status'] for x in r)
            assert len(r)==m['n']
            assert sum(x['status']=='covered' for x in r)==m['covered']
            assert vals['correct']==m['correct'] and vals['unassigned_covered']==m['unassigned_covered']
            assert sum(x['recovered']=='True' for x in r)==m['recovered']
            assert sum(x['lost']=='True' for x in r)==m['lost']
            if m['covered']:
                assert abs(sum(1. / int(x['instance_rank']) for x in r if x['instance_rank'])/m['covered']-m['instance_mrr'])<1.e-10
    for scope, rules in out['verification'].items():
        for rule, m in rules.items():
            a=[x for x in risk_rows if (
                x['platform']=='waymo' if scope=='waymo_pooled' else x['cohort']+'_'+x['platform']==scope)
               and x['rule']==rule]
            assert len(a)==m['n']
            assert sum(x['correct_accepted']=='True' for x in a)==m['correct_accepted']
            for row,type in (('natural_miss','natural_miss'),('covered_wrong','wrong_covered')):
                member=[r for r in a if r['winner_status']==type]
                assert len(member)==m[row]
                assert sum(r['reject']=='True' for r in member)==m[row+'_rejected']


def main():
    lockpath = PRIVATE / 'RUN_LOCK.json'
    lock = read(lockpath)
    summary = read(RESULTS / 'SUMMARY.json')
    assert summary['run_lock'] == sha(lockpath) and not lock['protected_data_access']
    assert set(lock['program']) == {'inputs.py','rank.py','verify_winner.py','run.py'}
    assert all(sha(HOME/f)==v for f,v in lock['program'].items())
    assert lock['old_replay']==sha(RESULTS/'WINNER_AUDIT.json')
    assert lock['protocol']==sha(HOME/'protocol.json')
    data, _, proof = load()
    assert proof == lock['inputs'] and lock['forward_passes']==0 and lock['source_only']
    fits = read(PRIVATE/'FIT_DETAILS.json')
    assert len(fits) == len({f['held'] for f in fits}) == 58
    rank_rows = csv_rows(RESULTS/'rank_rows.csv')
    risk_rows = csv_rows(RESULTS/'verification_rows.csv')
    inner, rank_n, verify_n = audit_cases(fits,data,rank_rows,risk_rows)
    n_obj = audit_rank_objectives(data)
    summary_arithmetic(rank_rows,risk_rows,summary)
    numeric_file = RESULTS/'NUMERICAL_CHECK.json'
    numeric = read(numeric_file)
    high = PRIVATE/'numeric_budget'
    rerun = read(high/'SUMMARY.json')
    high_lock = read(high/'RUN_LOCK.json')
    assert numeric['larger_budget_summary']==sha(high/'SUMMARY.json')
    assert numeric['original_completed_summary']==sha(RESULTS/'SUMMARY.json')
    assert rerun['run_lock']==sha(high/'RUN_LOCK.json') and high_lock['inputs']==lock['inputs']
    assert not high_lock['protected_data_access'] and high_lock['forward_passes']==0
    assert numeric['decision_unchanged']==(rerun['decision']==summary['decision'])
    extra_rank = csv_rows(high/'rank_rows.csv')
    extra_risk = csv_rows(high/'verification_rows.csv')
    assert audit_cases(read(high/'FIT_DETAILS.json'),data,extra_rank,extra_risk)==(inner,rank_n,verify_n)
    summary_arithmetic(extra_rank,extra_risk,rerun)
    names=('SUMMARY.json','WINNER_AUDIT.json','winner_replay.csv','rank_rows.csv','verification_rows.csv',
           'NUMERICAL_CHECK.json')
    atomic_json(RESULTS/'AUDIT.json',dict(status='PASS',frames=len(data),outer_folds=len(fits),
                 inner_partitions=inner,rank_rows=rank_n,verification_rows=verify_n,
                 independently_reduced_objective_probes=n_obj,higher_budget_replay_verified=True,
                 protected_data_access=False,new_backbone_calls=0,
                 files={f:sha(RESULTS/f) for f in names}))
    print('Independent fit-isolation, loss, ranking and rejection readout audit passed',flush=True)


if __name__ == '__main__':
    main()
