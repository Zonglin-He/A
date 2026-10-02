"""Independently reproduce published scalars and source-bootstrap summaries."""
import json
import sys
import collections
from pathlib import Path
import numpy as np


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    rows, summary, geometry = read('ROWS.json'), read('SUMMARY.json'), read('WRITE_GEOMETRY.json')
    config, resources = read('CONFIG.json'), read('RESOURCES.json')
    checks = 0
    def check(value):
        nonlocal checks
        assert value
        checks += 1
    def close(actual, expected):
        check(abs(actual-expected) < 1e-11)
    def reproduce(selected, recorded):
        check(recorded['cells'] == len(selected))
        if not selected:
            check(recorded['sources'] == 0 and not recorded['metrics'])
            return
        fields = list(recorded['metrics'])
        cells = collections.defaultdict(list)
        for row in selected:
            cells[row['source_id'], row['order'], row['condition']].append([row[field] for field in fields])
        orders = collections.defaultdict(list)
        for (source, order, condition), values in cells.items():
            orders[source, order].append(np.mean(values, axis=0))
        sources, order_values = collections.defaultdict(list), collections.defaultdict(list)
        for (source, order), values in orders.items():
            value = np.mean(values, axis=0)
            sources[source].append(value); order_values[order].append(value)
        matrix = np.array([np.mean(sources[source], axis=0) for source in sorted(sources)])
        check(recorded['sources'] == len(matrix))
        rng = np.random.default_rng(20261001)
        samples = np.concatenate([matrix[rng.integers(0, len(matrix), (100, len(matrix)))].mean(1)
                                  for _ in range(100)])
        confidence = np.quantile(samples, [.025, .975], axis=0)
        for j, field in enumerate(fields):
            target = recorded['metrics'][field]
            close(target['mean'], matrix[:, j].mean())
            close(target['query_macro'], np.mean([row[field] for row in selected]))
            for value, expected in zip(target['ci95'], confidence[:, j]):
                close(value, expected)
            means = np.array([np.mean(order_values[order], axis=0)[j] for order in sorted(order_values)])
            check(len(target['order_values']) == len(means))
            for value, expected in zip(target['order_values'], means):
                close(value, expected)
            if len(means) > 1:
                close(target['order_sample_SD'], means.std(ddof=1))
            else:
                check(target['order_sample_SD'] is None)
            if field.startswith('delta_'):
                check(target['harm_gt5pp_sources'] == int((matrix[:, j] < -.05).sum()))
    def audit_arm(selected, recorded):
        reproduce(selected, recorded)
        counts = recorded['counts']
        check(counts['positive'] == sum(row['delta_last_all_v'] > 1e-12 for row in selected))
        check(counts['negative'] == sum(row['delta_last_all_v'] < -1e-12 for row in selected))
        check(counts['zero'] == sum(abs(row['delta_last_all_v']) <= 1e-12 for row in selected))
        check(counts['harm_gt5pp'] == sum(row['delta_last_all_v'] < -.05 for row in selected))
        check(counts['latest_noops'] == sum(not row['latest_write_updated'] for row in selected))
        check(counts['intervals_changed'] == sum(row['last_interval'] != row['all_interval'] for row in selected))
        if selected:
            gross = [{**row, 'gross_gain': max(row['delta_last_all_v'], 0.),
                      'gross_loss': max(-row['delta_last_all_v'], 0.)} for row in selected]
            reproduce(gross, recorded['gross_source_macro'])
        for threshold in [.3, .5]:
            values = recorded['correctness'][str(threshold)]
            for name, expected in dict(
                all_correct=sum(row['all_v'] > threshold for row in selected),
                last_correct=sum(row['last_v'] > threshold for row in selected),
                rescued=sum(row['all_v'] <= threshold and row['last_v'] > threshold for row in selected),
                destroyed=sum(row['all_v'] > threshold and row['last_v'] <= threshold for row in selected)).items():
                check(values[name] == expected)
    check(len(rows) == 576)
    check(len({(row['arm'], row['condition'], row['order'], row['arrival']) for row in rows}) == 576)
    for arm in ['A', 'R']:
        selected = [row for row in rows if row['arm'] == arm]
        check(len(selected) == 288 and len({row['source_id'] for row in selected}) == 30)
        check(sum(row['condition'] != 'clean' for row in selected) == 240)
    for row in rows:
        check(row['arrival'] % 4 != 0 and row['latest_write_arrival'] == row['arrival']//4*4)
        check(row['prior_writes'] == row['arrival']//4+1 and row['lag'] in [1, 2, 3])
        for phase in ['source', 'all', 'last']:
            for field in ['v', 't', 's']:
                check(0 <= row[phase+'_'+field] <= 1+1e-12)
            close(row[phase+'_correct30'], float(row[phase+'_v'] > .3))
            close(row[phase+'_correct50'], float(row[phase+'_v'] > .5))
        for field, a, b in [('delta_all_source_v','all_v','source_v'),
                            ('delta_last_source_v','last_v','source_v'),
                            ('delta_last_all_v','last_v','all_v'),
                            ('delta_last_all_fixed_v','last_fixed_v','all_fixed_v'),
                            ('delta_last_all_s','last_s','all_s'),
                            ('delta_last_all_t','last_t','all_t')]:
            close(row[field], row[a]-row[b])
        if row['prior_writes'] == 1:
            close(row['delta_last_all_v'], 0.)
        if not row['latest_write_updated']:
            close(row['delta_last_source_v'], 0.)
    for group, data in summary.items():
        selected = [row for row in rows if (row['condition'] != 'clean' if group == 'corruption' else row['condition'] == group)]
        for arm in ['A','R']:
            take = [row for row in selected if row['arm'] == arm]
            audit_arm(take, data[arm])
            audit_arm([row for row in take if row['prior_writes'] >= 2], data[arm]['at_least_two_prior_writes'])
            audit_arm([row for row in take if row['latest_write_updated']], data[arm]['latest_nonzero'])
            audit_arm([row for row in take if not row['latest_write_updated']], data[arm]['latest_noop'])
            for count in range(1,9):
                audit_arm([row for row in take if row['prior_writes'] == count], data[arm]['by_prior_write_count'][str(count)])
        uniform = {(row['condition'],row['order'],row['arrival']):row for row in selected if row['arm']=='A'}
        paired = []
        for routed in [row for row in selected if row['arm']=='R']:
            other = uniform[routed['condition'],routed['order'],routed['arrival']]
            paired.append({**routed,'delta_R_A_all_v':routed['all_v']-other['all_v'],
                           'delta_R_A_last_v':routed['last_v']-other['last_v'],
                           'delta_R_A_last_all':routed['delta_last_all_v']-other['delta_last_all_v']})
        reproduce(paired, data['paired_R_minus_A'])
    check(len(geometry)==24)
    for stream in geometry:
        check(len(stream['write_norms'])==8)
        positive = [i for i,norm in enumerate(stream['write_norms']) if norm>0]
        for i, norm in enumerate(stream['write_norms']):
            close(stream['norm_sum_prefix'][i], sum(stream['write_norms'][:i+1]))
            if stream['norm_sum_prefix'][i]>0:
                close(stream['prefix_cancellation'][i],stream['prefix_norms'][i]/stream['norm_sum_prefix'][i])
                check(0<=stream['prefix_cancellation'][i]<=1+1e-12)
            else:
                check(stream['prefix_cancellation'][i] is None)
            for j, other in enumerate(stream['write_norms']):
                cosine=stream['cosine_gram'][i][j]
                if norm==0 or other==0:
                    check(cosine is None)
                else:
                    check(-1-1e-12<=cosine<=1+1e-12)
                    close(cosine,stream['cosine_gram'][j][i])
                    if i==j:close(cosine,1.)
        if positive:
            gram=np.array([[stream['cosine_gram'][i][j] for j in positive] for i in positive])
            check(np.linalg.eigvalsh(gram).min()>-1e-10)
    check(resources['bitwise_all_parity']==resources['bitwise_source_parity']==576)
    check(resources['new_backbone_forwards']==resources['new_experts']==resources['backward_calls']==resources['parameter_updates']==0)
    result = dict(status='pass', independent_public_scalar_checks=checks, targets=len(rows),
                  bootstrap_resamples=10000, bootstrap_unit='target source',
                  private_inputs_read=False)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    audit(sys.argv[1])
