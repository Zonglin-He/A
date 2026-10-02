"""Independent anonymous-public-row reconstruction, with no private inputs."""
import json
import sys
import collections
from pathlib import Path
import numpy as np


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    rows, writes = read('ROWS.json'), read('WRITE_ROWS.json')
    summary, recorded_contrasts = read('SUMMARY.json'), read('CONTRAST_ROWS.json')
    config, resources = read('CONFIG.json'), read('RESOURCES.json')
    checks = 0
    def check(value):
        nonlocal checks
        assert value
        checks += 1
    def close(value, expected, tolerance=1e-11):
        check(abs(value-expected) < tolerance)
    def reproduce(selected, target):
        check(target['cells'] == len(selected))
        if not selected:
            check(target['sources'] == 0 and not target['metrics'])
            return
        fields = list(target['metrics'])
        cells = collections.defaultdict(list)
        for row in selected:
            cells[row['source_id'], row['order'], row['condition']].append([row[f] for f in fields])
        order_cells = collections.defaultdict(list)
        for (source, order, condition), values in cells.items():
            order_cells[source, order].append(np.mean(values, axis=0))
        sources, order_values = collections.defaultdict(list), collections.defaultdict(list)
        for (source, order), values in order_cells.items():
            value = np.mean(values, axis=0); sources[source].append(value); order_values[order].append(value)
        matrix = np.array([np.mean(sources[s], axis=0) for s in sorted(sources)])
        check(target['sources'] == len(matrix))
        rng = np.random.default_rng(20261001)
        bootstrap = np.concatenate([matrix[rng.integers(0, len(matrix), (100, len(matrix)))].mean(1) for _ in range(100)])
        confidence = np.quantile(bootstrap, [.025, .975], axis=0)
        means = np.array([np.mean(order_values[o], axis=0) for o in sorted(order_values)])
        for j, field in enumerate(fields):
            m = target['metrics'][field]
            close(m['mean'], matrix[:, j].mean()); close(m['query_macro'], np.mean([r[field] for r in selected]))
            for a, b in zip(m['ci95'], confidence[:, j]):
                close(a, b)
            check(len(m['order_values']) == len(means))
            for a, b in zip(m['order_values'], means[:, j]):
                close(a, b)
            if len(means) > 1:
                close(m['order_sample_SD'], means[:, j].std(ddof=1))
            else:
                check(m['order_sample_SD'] is None)
            if field.startswith('delta_'):
                check(m['harm_gt5pp_sources'] == int((matrix[:, j] < -.05).sum()))
    branches = ['U', 'R', 'Specific']
    def stats(selected, target):
        reproduce(selected, target)
        check(target['coverage']['target_cells'] == len(selected))
        check(target['coverage']['both_nonempty'] == sum(r['both_nonempty'] for r in selected))
        check(target['coverage']['donor_sources'] == len({r['source_id'] for r in selected}))
        check(target['coverage']['target_sources'] == len({r['target_source_id'] for r in selected}))
        for branch in branches:
            field = 'delta_'+branch+'_v'; effect = target['effects'][branch]
            check(effect['positive'] == sum(r[field] > 1e-12 for r in selected))
            check(effect['negative'] == sum(r[field] < -1e-12 for r in selected))
            check(effect['zero'] == sum(abs(r[field]) <= 1e-12 for r in selected))
            check(effect['harm_gt5pp'] == sum(r[field] < -.05 for r in selected))
            check(effect['intervals_changed'] == sum(r[branch+'_interval'] != r['pre_interval'] for r in selected))
            reproduce([{**r, 'gross_gain': max(r[field], 0.), 'gross_loss': max(-r[field], 0.)} for r in selected],
                      effect['gross_source_macro'])
            for t in [.3, .5]:
                c = effect['correctness'][str(t)]
                check(c['rescued'] == sum(r['pre_v'] <= t < r[branch+'_v'] for r in selected))
                check(c['destroyed'] == sum(r[branch+'_v'] <= t < r['pre_v'] for r in selected))
    check(len(rows) == 1392 and len(writes) == 96)
    keys = [(r['condition'], r['order'], r['donor_arrival'], r['target_arrival']) for r in rows]
    check(len(set(keys)) == 1392)
    buckets = collections.defaultdict(list)
    for row in rows:
        check(row['donor_arrival'] % 4 == 0 and 0 <= row['donor_arrival'] <= 28)
        check(row['lag'] == row['target_arrival']-row['donor_arrival'])
        check(row['source_id'] == row['donor_source_id'])
        if row['broader']:
            check(row['target_arrival'] > row['donor_arrival'] and row['target_arrival'] % 4 != 0)
            check(row['target_source_id'] != row['source_id'])
        else:
            check(row['target_arrival'] == row['donor_arrival'] and 'self' in row['roles'])
        for branch in ['pre', *branches]:
            for metric in ['v', 's', 't', 'free_v', 'free_t']:
                check(0 <= row[branch+'_'+metric] <= 1+1e-12)
            close(row[branch+'_correct30'], float(row[branch+'_v'] > .3))
            close(row[branch+'_correct50'], float(row[branch+'_v'] > .5))
        for branch in branches:
            for metric in ['v', 's', 'free_v', 'free_t']:
                close(row['delta_'+branch+'_'+metric], row[branch+'_'+metric]-row['pre_'+metric])
        close(row['delta_R_U_v'], row['R_v']-row['U_v']); close(row['delta_R_U_s'], row['R_s']-row['U_s'])
        buckets[row['condition'], row['order'], row['donor_arrival']].append(row)
    check(len(buckets) == 96)
    rebuilt_contrasts = []
    for key, selected in buckets.items():
        condition, order, i = key; future = [r for r in selected if r['broader']]
        check(sorted(r['target_arrival'] for r in future) == [j for j in range(i+1, 32) if j % 4 != 0])
        near = min(future, key=lambda r: (-r['cosine'], r['target_arrival']))
        far = min(future, key=lambda r: (r['cosine'], r['target_arrival']))
        expected = dict(self=i, next=i+1, near=near['target_arrival'], far=far['target_arrival'])
        for row in selected:
            check(row['roles'] == [role for role, j in expected.items() if j == row['target_arrival']])
        record = {k: selected[0][k] for k in ['condition', 'order', 'donor_arrival', 'source_id']}
        role_map = {role: next(r for r in selected if role in r['roles']) for role in expected}
        for branch in branches:
            field = 'delta_'+branch+'_v'; broad = np.mean([r[field] for r in future])
            for role in ['self', 'near', 'next', 'far']:
                record['delta_'+branch+'_'+role+'_broader_v'] = role_map[role][field]-broad
            record['delta_'+branch+'_self_far_v'] = role_map['self'][field]-role_map['far'][field]
            record['delta_'+branch+'_near_far_v'] = role_map['near'][field]-role_map['far'][field]
        rebuilt_contrasts.append(record)
    contrast_map = {(r['condition'], r['order'], r['donor_arrival']): r for r in recorded_contrasts}
    check(len(contrast_map) == 96)
    for record in rebuilt_contrasts:
        saved = contrast_map[record['condition'], record['order'], record['donor_arrival']]
        for field, value in record.items():
            if field.startswith('delta_'):
                close(saved[field], value)
            else:
                check(saved[field] == value)
    for group, data in summary.items():
        selected = [r for r in rows if (r['condition'] != 'clean' if group == 'corruption' else r['condition'] == group)]
        for role in ['self', 'next', 'near', 'far', 'broader']:
            take = [r for r in selected if (r['broader'] if role == 'broader' else role in r['roles'])]
            stats(take, data['roles'][role])
            stats([r for r in take if r['both_nonempty']], data['roles'][role]['both_nonempty_only'])
            reproduce([{**r, 'source_id': r['target_source_id']} for r in take],
                      data['roles'][role]['target_clustered_sensitivity'])
        reproduce([r for r in rebuilt_contrasts if (r['condition'] != 'clean' if group == 'corruption' else r['condition'] == group)], data['contrasts'])
    # Continued below: teacher distributions and gradient norm identities.
    for wr in writes:
        key = wr['condition'], wr['order'], wr['donor_arrival']
        check(key in buckets and wr['source_id'] == buckets[key][0]['source_id'])
        check(wr['both_nonempty'] == (wr['U']['available'] and wr['R']['available']))
        g = wr['geometry']; nu, nr = g['norm_U'], g['norm_R']
        close(g['norm_specific']**2, nu**2+nr**2-2*g['dot_U_R'], 1e-9)
        if nu > 0 and nr > 0:
            close(g['cosine_U_R'], g['dot_U_R']/(nu*nr)); check(g['cosine_defined'])
            check(-1-1e-12 <= g['cosine_U_R'] <= 1+1e-12)
        else:
            check(g['cosine_U_R'] is None and not g['cosine_defined'])
        for branch in ['U', 'R']:
            meta = wr[branch]
            check(len(meta['positions']) == 5)
            if meta['available']:
                reward = np.asarray(meta['rewards']); ranks = np.zeros(9)
                order = np.argsort(-reward, kind='stable'); i = 0
                while i < 9:
                    j = i+1
                    while j < 9 and reward[order[i]]-reward[order[j]] <= 1e-12:
                        j += 1
                    ranks[order[i:j]] = (i+j-1)/2; i = j
                exponent = -ranks/config['params']['teacher_temperature']; q = np.exp(exponent-exponent.max()); q /= q.sum()
                p = np.asarray(meta['p']); saved_q = np.asarray(meta['q'])
                check(np.max(abs(saved_q-q)) < 2e-6)
                close(float(p.sum()), 1., 2e-6)
                close(meta['loss'], float(np.sum(p*(np.log(p)-np.log(saved_q)))), 3e-6)
                check(meta['selected_index'] == int(np.argmax(reward)))
                check(meta['flat_rewards'] == bool(np.ptp(reward) == 0))
            else:
                check(meta['rewards'] is None and meta['loss'] is None and meta['selected_index'] == 0)
            close(meta['selected_v'], wr['probe_v'][meta['selected_index']])
            table = meta['critic_pairwise']; strict = decisive = 0; score_sum = 0.
            reward = np.zeros(9) if meta['rewards'] is None else np.asarray(meta['rewards'])
            u = np.asarray(wr['probe_v']); pairs = table['pairs']; check(len(pairs) == 36)
            for p0 in pairs:
                i, j = p0['i'], p0['j']; a, b = reward[i]-reward[j], u[i]-u[j]
                ea = int(a > 1e-12)-int(a < -1e-12); gb = int(b > 1e-12)-int(b < -1e-12)
                close(p0['reward_difference'], a); close(p0['utility_difference'], b)
                check(p0['expert_sign'] == ea and p0['GT_sign'] == gb)
                if gb:
                    strict += 1; decisive += ea != 0; accuracy = .5 if ea == 0 else float(ea == gb)
                    close(p0['accuracy'], accuracy); score_sum += accuracy
                else:
                    check(p0['accuracy'] is None)
            check(table['strict_GT_pairs'] == strict and table['decisive_pairs'] == decisive)
            if strict:
                close(table['pairwise_accuracy'], score_sum/strict); close(table['decisive_coverage'], decisive/strict)
            else:
                check(table['pairwise_accuracy'] is None and table['decisive_coverage'] is None)
    check(resources['new_writes'] == 96 and resources['expert_cache_reads'] == 192)
    check(resources['new_experts'] == resources['new_backbone_forwards'] == 0)
    check(resources['backward_calls'] == sum(w[b]['available'] for w in writes for b in ['U', 'R']))
    check(resources['checkpoint_restored'] and config['params']['steps'] == 1 and config['historical_steps'] == 8)
    sensitivity = read('SENSITIVITY.json')
    for role in ['self', 'broader']:
        take = [r for r in rows if r['condition'] != 'clean' and (r['broader'] if role == 'broader' else role in r['roles'])]
        for record in sensitivity[role]['leave_one_donor_out']:
            selected = [r for r in take if r['source_id'] != record['omitted_donor_source_id']]
            reproduce(selected, dict(cells=len(selected), sources=len({r['source_id'] for r in selected}),
                                    metrics={'delta_Specific_v': {k: v for k, v in record.items() if k != 'omitted_donor_source_id'}}))
    result = dict(status='pass', independent_public_scalar_checks=checks, donors=96,
                  targets=1392, bootstrap_resamples=10000, private_inputs_read=False)
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    audit(sys.argv[1])
