"""Independent anonymous scalar audit; no model, fit cache or raw GT access."""
import collections
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
REL = 'results/stvg_opd_mechanism_review/2026-10-13/LN_complete'
ARMS = ['full', 'alpha0', 'alpha1', 'LN_only']
ORDERS = ['order3', 'order4', 'order5']
FIELDS = ['After_v', 'Before_v', 'Frozen_v', 'delta_inherited_v', 'delta_current_v',
          'Full_minus_alpha0_v', 'Full_minus_alpha1_v', 'Full_minus_LN_only_v']
ARM_FIELDS = ['After_v', 'Before_v', 'Frozen_v', 'After_s', 'After_t', 'After_R30', 'After_R50',
              'delta_total_v', 'delta_current_v', 'delta_inherited_v']


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(part, fields):
    groups = collections.defaultdict(list)
    seen = set()
    for r in part:
        key = (r['parent_id'], r['order'], r['condition'])
        assert key not in seen
        seen.add(key)
        groups[r['parent_id']].append(r)
    ids = sorted(groups)
    assert ids and len({len(g) for g in groups.values()}) == 1
    matrix = np.array([[math.fsum(r[f] for r in groups[i])/len(groups[i]) for f in fields] for i in ids])
    rng = np.random.default_rng(20261011)
    boot = np.concatenate([matrix[rng.integers(0, len(ids), (100, len(ids)))].mean(1) for _ in range(100)])
    ci = np.quantile(boot, [.025, .975], axis=0)
    return dict(sources=len(ids), cells=len(part), metrics={f: dict(
        mean=float(matrix[:, j].mean()), query_macro=math.fsum(r[f] for r in part)/len(part),
        ci95=ci[:, j].tolist(), parent_values=[dict(parent_id=i, value=float(matrix[k, j])) for k, i in enumerate(ids)],
        negative_parent_count=int((matrix[:, j] < 0).sum()), positive_parent_count=int((matrix[:, j] > 0).sum()),
        harm_gt5pp_sources=int((matrix[:, j] < -.05).sum()), harm_gt20pp_sources=int((matrix[:, j] < -.2).sum()),
        worst_parent=float(matrix[:, j].min()), best_parent=float(matrix[:, j].max())) for j, f in enumerate(fields)})


def compare(actual, expected):
    assert actual['sources'] == expected['sources'] and actual['cells'] == expected['cells']
    checks = 2
    for f, x in actual['metrics'].items():
        y = expected['metrics'][f]
        for k in ['mean', 'query_macro', 'worst_parent', 'best_parent']:
            assert abs(x[k]-y[k]) < 4e-12, (f, k)
            checks += 1
        assert len(x['ci95']) == 2 and max(abs(a-b) for a, b in zip(x['ci95'], y['ci95'])) < 4e-12
        checks += 2
        for k in ['negative_parent_count', 'positive_parent_count', 'harm_gt5pp_sources', 'harm_gt20pp_sources']:
            assert x[k] == y[k], (f, k)
            checks += 1
        assert len(x['parent_values']) == len(y['parent_values'])
        for a, b in zip(x['parent_values'], y['parent_values']):
            assert a['parent_id'] == b['parent_id'] and abs(a['value']-b['value']) < 4e-12
            checks += 1
    return checks


def paired(part):
    cells = {(r['arm'], r['order'], r['parent_id']): r for r in part}
    assert len(cells) == len(part)
    out = []
    for r in part:
        assert r['query_reset'] and r['Adam_reset'] and r['source_reset_per_arm_order']
        for m in ['v', 's', 't']:
            assert abs(r['delta_total_'+m]-r['delta_current_'+m]-r['delta_inherited_'+m]) < 2e-10
            for label in ['Before', 'After', 'Frozen']:
                assert r[label+'_'+m] == r['values'][label][m]
        assert r['After_R30'] == float(r['After_v'] > .3)
        assert r['After_R50'] == float(r['After_v'] > .5)
        if r['arm'] != 'full':
            continue
        p = dict(r)
        for a in ARMS[1:]:
            c = cells[a, r['order'], r['parent_id']]
            assert c['arrival'] == r['arrival'] and c['Frozen_v'] == r['Frozen_v'] and c['After_t'] == r['After_t']
            for m in ['v', 's', 't']:
                p['Full_minus_'+a+'_'+m] = r['After_'+m]-c['After_'+m]
        out.append(p)
    return out


def parent_table(rr):
    output = []
    for ds in ['hc2', 'vidstg']:
        pp = paired([r for r in rr if r['dataset'] == ds])
        for pid in sorted({r['parent_id'] for r in pp}):
            group = {r['order']: r for r in pp if r['parent_id'] == pid}
            assert set(group) == set(ORDERS)
            row = dict(dataset=ds, parent_id=pid)
            for o in ORDERS:
                for f in FIELDS:
                    row[o+'_'+f] = group[o][f]
            for f in FIELDS:
                row['pooled_'+f] = math.fsum(group[o][f] for o in ORDERS)/3
            output.append(row)
    return output


def run(root):
    out = Path(root)/REL
    rr = rows(out/'ROWS.jsonl.gz')
    assert len(rr) == 3072 and len({r['parent_id'] for r in rr}) == 256
    assert all('source_id' not in r and 'query_ordinal' not in r for r in rr)
    saved = read(out/'FOUR_ARM_ORDER_STATISTICS.json.gz')
    extended = read(out/'ALL_ARM_STATISTICS.json.gz')
    checks = 0
    for ds, name in [('hc2', 'M_hc2_cross_clean'), ('vidstg', 'M_vidstg_cross_clean')]:
        part = [r for r in rr if r['dataset'] == ds]
        assert len(part) == 1536 and len({r['parent_id'] for r in part}) == 128
        assert set(r['arm'] for r in part) == set(ARMS) and set(r['order'] for r in part) == set(ORDERS)
        pp = paired(part)
        s = saved['statistics'][name]
        for order in ['pooled']+ORDERS:
            chosen = pp if order == 'pooled' else [r for r in pp if r['order'] == order]
            stat = s['pooled'] if order == 'pooled' else s['per_order'][order]
            checks += compare(stat, independent(chosen, FIELDS))
            if order != 'pooled':
                for a in ARMS[1:]:
                    v = stat['metrics']['Full_minus_'+a+'_v']
                    entry = next(x for x in saved['per_order_table'] if x['stage'] == name and x['order'] == order and x['contrast'] == 'Full - '+a)
                    assert abs(entry['mean_pp']-v['mean']*100) < 4e-12
                    assert entry['ci95_pp'] == [b*100 for b in v['ci95']]
            for a in ARMS:
                chosen = [r for r in part if r['arm'] == a and (order == 'pooled' or r['order'] == order)]
                checks += compare(extended['statistics'][name][a][order], independent(chosen, ARM_FIELDS))
        for a in ARMS:
            checks += compare(s['arms'][a], independent([r for r in part if r['arm'] == a], ['Before_v', 'After_v', 'Frozen_v', 'delta_current_v']))
        for a, first in enumerate(s['first_parent_per_order']):
            row = next(r for r in pp if r['order'] == ORDERS[a] and r['arrival'] == 0)
            assert all(abs(first[f]-row[f]) < 4e-12 for f in FIELDS)
    expected = parent_table(rr)
    with gzip.open(out/'ALL_PARENT_CONTRASTS.csv.gz', 'rt') as f:
        table = list(csv.DictReader(f))
    assert len(table) == len(expected) == 256
    for a, b in zip(table, expected):
        assert a['dataset'] == b['dataset'] and int(a['parent_id']) == b['parent_id']
        for k, v in b.items():
            if isinstance(v, float):
                assert abs(float(a[k])-v) < 4e-12
                checks += 1
    d = read(out/'DESIGN_PUBLIC.json')
    assert d['new_formal_fits'] == 768 and d['reused_control_fits'] == 2304 and d['new_DINO_calls'] == 0
    assert d['GT_only_after_all_new_formal_global_seal'] and not d['production_promoted']
    return dict(status='pass', scope='all3072 anonymous outcomes/256 paired historical parents/four real arms/three actual orders/full absolute and current tails/independent10000-parent bootstrap',
        complete_rows=3072, new_LN_formal_rows=768, reused_control_rows=2304, parents=256,
        scalar_comparisons=checks, bootstrap=10000, seed=20261011,
        rows_sha256=digest(out/'ROWS.jsonl.gz'), statistics_sha256=digest(out/'FOUR_ARM_ORDER_STATISTICS.json.gz'),
        all_arm_statistics_sha256=digest(out/'ALL_ARM_STATISTICS.json.gz'),
        CPU_only=True, GT_read=False, raw_GT_read=False, model_calls=0, new_DINO_calls=0)


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])))
