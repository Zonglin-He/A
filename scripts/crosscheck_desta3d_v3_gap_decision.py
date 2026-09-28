"""Independent CPU-only reductions of saved gap geometry and locked decisions."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/desta3d_v3/latent_oracle_v1/oracle_mixer_gap_v1'
P = D.parent / 'gap_followups_v1/decision_readback_v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    import numpy as np
    start = time.monotonic()
    out = P / 'ROOT_DECISION_CROSSCHECK.json'
    assert not out.exists()
    assert read(P / 'COMPLETE.json')['seal_sha'] == sha(P / 'SEAL.json')
    for name, digest in read(P / 'SEAL.json')['files'].items():
        assert sha(P / name) == digest
    rdir = D / 'independent_readback_v1'
    r = read(rdir / 'REPORT.json')
    assert read(rdir / 'COMPLETE.json')['report_sha'] == sha(rdir / 'REPORT.json')
    assert read(rdir / 'ROOT_SUMMARY_CROSSCHECK.json')['status'] == 'passed'
    g = read(rdir / 'GEOMETRY_CASES.json')
    decision = read(P / 'DECISION_TABLE.json')
    alignment = read(P / 'COEFFICIENT_ALIGNMENT.json')
    cases = read(P / 'QUADRANT_CASES.json')
    seeds = ('seed20260928', 'seed20260929')
    rows = {a: r['arms'][a]['rows'] for a in ('B1', *seeds, 'oracle')}
    assert len(g) == len(alignment) == len(rows['B1']) == 447
    assert len(cases) == 894
    parents = sorted({x['source'] for x in rows['B1']})
    errors = []

    def close(a, b):
        if a is None or b is None:
            assert a is None and b is None
        else:
            e = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            errors.append(e)
            assert e < 2e-8, (a, b, e)

    def desc(values, saved):
        x = np.array([v for v in values if v is not None], dtype=np.float64)
        expected = dict(defined=len(x), undefined=len(values)-len(x),
                        mean=float(x.mean()) if len(x) else None,
                        median=float(np.median(x)) if len(x) else None,
                        min=float(x.min()) if len(x) else None,
                        max=float(x.max()) if len(x) else None,
                        positive=int((x > 0).sum()), negative=int((x < 0).sum()),
                        zero=int((x == 0).sum()))
        assert set(expected) == set(saved)
        for k, v in expected.items():
            close(v, saved[k])

    groups = {'all': list(range(447))}
    for metric in ('vIoU', 'tIoU'):
        for suffix, good in (('good', True), ('other', False)):
            groups[metric+'_B1_'+suffix] = [i for i, x in enumerate(rows['B1'])
                                           if (x['metrics'][metric] > .5) == good]
    for name, indices in groups.items():
        block = r['geometry_groups'][name]
        close(len(indices), block['queries'])
        close(len({rows['B1'][i]['source'] for i in indices}), block['parents'])
        for arm in (*seeds, 'oracle'):
            s = block['stats'][arm]
            desc([g[i]['cosine_to_oracle'][arm] for i in indices], s['cosine'])
            desc([g[i]['norm_over_cap'][arm] for i in indices], s['norm_over_cap'])
            for branch, label in (('event', 'T'), ('spatial', 'S')):
                desc([g[i]['descent_dot'][branch][arm] for i in indices], s[label+'_descent'])
            desc([100*(rows[arm][i]['metrics']['vIoU']-rows['B1'][i]['metrics']['vIoU'])
                  for i in indices], s['delta_v_pp'])
            if arm != 'oracle':
                desc([g[i]['mixer'][arm]['fraction_abs_above_099'] for i in indices], s['tanh_abs_gt099'])

    features = []
    quadrants = {}
    case_lookup = {(x['index'], x['seed']): x for x in cases}
    assert len(case_lookup) == 894
    for metric in ('vIoU', 'sIoU', 'tIoU'):
        values = np.asarray([rows['oracle'][i]['metrics'][metric]-rows['B1'][i]['metrics'][metric] for i in range(447)])
        close(int((values < -.05).sum()), r['query_tails']['oracle'][metric]['loss_below_minus5pp'])
        close(int((values > .05).sum()), r['query_tails']['oracle'][metric]['gain_above5pp'])
    for arm in seeds:
        table = decision['table'][arm]
        for metric in ('vIoU', 'sIoU', 'tIoU'):
            diffs = np.asarray([np.mean([rows['oracle'][i]['metrics'][metric]-rows[arm][i]['metrics'][metric]
                                        for i in range(447) if rows['B1'][i]['source'] == p]) for p in parents])
            draws = np.random.default_rng(20260927).integers(0, 31, (10000, 31))
            c = r['oracle_vs_learned'][arm][metric]
            close(diffs.mean()*100, c['mean_delta_pp'])
            close(np.quantile(diffs[draws].mean(1), [.025, .975])*100, c['bootstrap_ci95_pp'])
        counts = {a+'/'+b: 0 for a in ('gain', 'harm', 'flat') for b in ('gain', 'harm', 'flat')}
        for i in range(447):
            o = rows['oracle'][i]['metrics']['vIoU']-rows['B1'][i]['metrics']['vIoU']
            l = rows[arm][i]['metrics']['vIoU']-rows['B1'][i]['metrics']['vIoU']
            sign = lambda x: 'gain' if x > 1e-10 else ('harm' if x < -1e-10 else 'flat')
            label = sign(o)+'/'+sign(l)
            counts[label] += 1
            assert case_lookup[i, arm]['quadrant'] == label
            close(100*o, case_lookup[i, arm]['oracle_delta_v_pp'])
            close(100*l, case_lookup[i, arm]['learned_delta_v_pp'])
            assert alignment[i]['index'] == i
            close(alignment[i]['cosine'][arm], g[i]['cosine_to_oracle'][arm])
        for label, n in counts.items():
            close(n, table['quadrants'].get(label, 0))
        quadrants[arm] = counts
        for metric in ('vIoU', 'sIoU', 'tIoU'):
            values = np.asarray([rows[arm][i]['metrics'][metric]-rows['B1'][i]['metrics'][metric] for i in range(447)])
            close(int((values < -.05).sum()), r['query_tails'][arm][metric]['loss_below_minus5pp'])
            close(int((values > .05).sum()), r['query_tails'][arm][metric]['gain_above5pp'])
        cos = [x['cosine_to_oracle'][arm] for x in g if x['cosine_to_oracle'][arm] is not None]
        f = dict(cos_mean=float(np.mean(cos)), cos_median=float(np.median(cos)),
                 learned_descent={}, oracle_descent={},
                 saturated_fraction=sum(x['norm_over_cap'][arm] >= .99 for x in g)/447,
                 v_mean=r['comparisons'][arm]['vIoU']['mean_delta_pp'],
                 oracle_minus_learned_lower=r['oracle_vs_learned'][arm]['vIoU']['bootstrap_ci95_pp'][0])
        for branch in ('event', 'spatial'):
            eligible = [x for x in g if x['gradient_norms'][branch] > 0]
            for key, a in (('learned_descent', arm), ('oracle_descent', 'oracle')):
                f[key][branch] = sum(x['descent_dot'][branch][a] > 0 for x in eligible)/len(eligible)
                close(f[key][branch], table['screen_features'][key][branch])
        for key in ('cos_mean', 'cos_median', 'saturated_fraction', 'v_mean', 'oracle_minus_learned_lower'):
            close(f[key], table['screen_features'][key])
        features.append(f)
    o = r['comparisons']['oracle']
    advantage = o['vIoU']['mean_delta_pp'] > 0 and o['vIoU']['bootstrap_ci95_pp'][0] > 0 and all(o[m]['mean_delta_pp'] >= 0 for m in ('tIoU', 'sIoU'))
    mapping = advantage and all(f['oracle_minus_learned_lower'] > 0 and
                               ((f['cos_mean'] <= .1 and f['cos_median'] <= .1) or
                                max(f['oracle_descent'][b]-f['learned_descent'][b] for b in ('event', 'spatial')) >= .20)
                               for f in features)
    trust = advantage and all(f['cos_mean'] >= .30 and f['cos_median'] >= .30 and
                             min(f['learned_descent'].values()) >= .75 and
                             f['saturated_fraction'] >= .90 and f['v_mean'] <= 0 for f in features)
    expected = 'A' if mapping and not trust else ('B' if trust and not mapping else 'INCONCLUSIVE')
    assert decision['decision']['DECISION'] == expected
    assert not expected.startswith('C'), 'Rescue remains unmeasured'
    result = dict(status='passed', checks=len(errors), max_abs=max(errors),
                  DECISION=expected, quadrants=quadrants, report_sha=sha(rdir/'REPORT.json'),
                  decision_table_sha=sha(P/'DECISION_TABLE.json'), auditor_sha=sha(Path(__file__)),
                  CPU_seconds=time.monotonic()-start, GPU=False, new_GT_read=False,
                  scope='Independent reductions from all saved cases; no outcome selection or new model execution.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
