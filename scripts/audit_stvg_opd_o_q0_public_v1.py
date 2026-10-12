"""Recompute all published O/Q0 parent statistics without models or private data."""
import gzip
import json
import math
import sys
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        return sum(equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        return sum(equal(a, b) for a, b in zip(actual, expected))
    if expected is None or isinstance(expected, (str, bool, int)):
        assert actual == expected
    else:
        assert math.isfinite(actual) and abs(actual-expected) <= 2e-12*(1+abs(expected))
    return 1


def statistics(data, field, kind):
    parents = {}
    for row in data:
        if row.get(field) is not None:
            parents.setdefault(row['parent_id'], []).append(float(row[field]))
    values = np.asarray([math.fsum(value)/len(value) for value in parents.values()])
    if not len(values):
        return dict(parents=0, mean=None, ci95=None)
    rng = np.random.default_rng(20261012)
    boot = np.empty(10000)
    for start in range(0, 10000, 100):
        selected = rng.integers(len(values), size=(100, len(values)))
        boot[start:start+100] = np.sum(values[selected], axis=1)/len(values)
    result = dict(parents=len(values), mean=math.fsum(values)/len(values),
                  ci95=np.percentile(boot, [2.5, 97.5]).tolist())
    if kind == 'O':
        result.update(negative_parent_tail_5pp=int(np.count_nonzero(values < -.05)),
                      positive_parent_tail_5pp=int(np.count_nonzero(values > .05)),
                      p05=float(np.percentile(values, 5)))
    else:
        result.update(harm_5pp_parents=int(np.count_nonzero(values < -.05)),
                      harm_20pp_parents=int(np.count_nonzero(values < -.20)))
    return result


def audit_o(directory):
    directory = Path(directory)
    data = rows(directory/'ROWS.jsonl.gz')
    assert len(data) == 1792 and len({r['public_cell'] for r in data}) == 1792
    assert len({r['parent_id'] for r in data}) == 256
    assert all('query_ordinal' not in r and 'source_id' not in r for r in data)
    summary = read(directory/'SUMMARY.json')
    independent = read(directory/'INDEPENDENT_ROOT_STATISTICS.json')
    checks = 0
    for setting, block in summary['settings'].items():
        name, population = setting.rsplit('/', 1)
        selected = [r for r in data if r['setting'] == name and
                    (population == 'all_historical' or not r['development'])]
        assert len(selected) == block['cells'] == independent['settings'][setting]['cells']
        for field, value in block['statistics'].items():
            actual = statistics(selected, field, 'O')
            checks += equal(actual, value)
            checks += equal(actual, independent['settings'][setting]['statistics'][field])
        for order, values in block['per_order'].items():
            ordered = [r for r in selected if r['order'] == order]
            for field, value in values.items():
                actual = statistics(ordered, field, 'O')
                checks += equal(actual, value)
                checks += equal(actual, independent['settings'][setting]['per_order'][order][field])
        for field, value in independent['settings'][setting]['propagation'].items():
            checks += equal(statistics(selected, field, 'O'), value)
    return dict(status='pass', scope='all anonymous O scalar rows, all populations/orders/arms/propagation/CI/tails',
                rows=len(data), parents=256, scalar_checks=checks, bootstrap=10000,
                seed=20261012, private_data_read=False, model_calls=0)


def audit_q0(directory):
    directory = Path(directory)
    data = rows(directory/'ROWS.jsonl.gz')
    probes = rows(directory/'CANDIDATE_ROWS.jsonl.gz')
    summary = read(directory/'SUMMARY.json')
    assert len(data) == 192 and len(probes) == 2808
    assert len({r['public_query'] for r in data}) == 64
    assert len({r['parent_id'] for r in data}) == 64
    assert all('query_ordinal' not in r and 'development_parent' not in r for r in data+probes)
    groups = ('Q_random', 'Q_observable', 'LN_observable')
    indexed = {(r['dataset'], r['parent_id'], r['group']): r for r in data}
    assert len(indexed) == len(data)
    candidates = {}
    for row in probes:
        candidates.setdefault((row['dataset'], row['parent_id'], row['group']), []).append(row)
    checks = 0
    for row in data:
        rr = sorted(candidates[row['dataset'], row['parent_id'], row['group']],
                    key=lambda candidate: candidate['candidate'])
        assert len(rr) == row['candidates']
        assert [p['candidate'] for p in rr] == list(range(len(rr)))
        assert int(np.argmax([p['v'] for p in rr])) == row['oracle_index']
        assert int(np.argmax([p['reward'] for p in rr])) == row['expert_index']
        for label, index in [('oracle', row['oracle_index']), ('expert_top1', row['expert_index'])]:
            for metric in ('v', 't', 's'):
                checks += equal(rr[index][metric], row[label+'_'+metric])
                checks += equal(rr[index][metric]-rr[0][metric], row[label+'_delta_'+metric])
        for metric in ('v', 't', 's'):
            checks += equal(row['parameter_mean_'+metric]-rr[0][metric], row['parameter_mean_delta_'+metric])
        rewards = [p['reward'] for p in rr]
        assert sum(value == max(rewards) for value in rewards) == row['expert_top_tie_count']
        assert (len(set(rewards)) == 1) == row['all_rewards_equal']
        assert sum(p['budget_attained'] for p in rr[1:]) == row['attained_directions']
        assert len(rr)-1 == row['total_signed_directions']
        assert abs(math.fsum(p['weight'] for p in rr)-1) < 2e-10
    for dataset, block in summary['datasets'].items():
        for group, content in block['groups'].items():
            selected = [r for r in data if r['dataset'] == dataset and r['group'] == group]
            assert len(selected) == content['cells'] == 32
            for field, value in content['statistics'].items():
                checks += equal(statistics(selected, field, 'Q0'), value)
            for field, key in [('no_expert', 'no_expert'), ('all_rewards_equal', 'all_equal_rewards'),
                               ('attained_directions', 'attained_directions'),
                               ('total_signed_directions', 'total_signed_directions')]:
                assert sum(r[field] for r in selected) == content[key]
            assert sum(r['expert_top_tie_count'] > 1 for r in selected) == content['reward_ties']
        for first, second in [('Q_observable', 'Q_random'), ('Q_observable', 'LN_observable')]:
            old = block['paired_contrasts'][first+'_minus_'+second]
            paired = []
            for row in data:
                if row['dataset'] != dataset or row['group'] != first:
                    continue
                other = indexed[dataset, row['parent_id'], second]
                paired.append(dict(parent_id=row['parent_id'], **{
                    field: row[field]-other[field] if row[field] is not None and other[field] is not None else None
                    for field in old}))
            for field, value in old.items():
                checks += equal(statistics(paired, field, 'Q0'), value)
    return dict(status='pass', scope='all64 anonymous Q0 groups/candidates/oracle/expert/true-mean rows and paired parent CI/tails',
                queries=64, group_rows=len(data), candidate_rows=len(probes), scalar_checks=checks,
                bootstrap=10000, seed=20261012, private_data_read=False, model_calls=0)


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    result = dict(status='pass', O=audit_o(root/'results/stvg_opd_optimization_geometry/2026-10-12/common_state_public'),
                  Q0=audit_q0(root/'results/stvg_opd_query_observability_q0/2026-10-12/Native_support_scope003_public'))
    print(json.dumps(result, allow_nan=False))
