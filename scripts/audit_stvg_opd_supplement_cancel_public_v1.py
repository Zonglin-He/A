"""Portable literal scalar audit of the finite post-cancellation supplement."""
import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_stvg_opd_LN_complete_public_v1 import independent, compare

REL = 'results/stvg_opd_supplement/2026-10-13/after_cancel_complete'
ARMS = ['full', 'alpha0', 'alpha1']
ORDERS = ['order3', 'order4', 'order5']
FIELDS = ['After_v', 'Before_v', 'Frozen_v', 'delta_current_v',
          'delta_inherited_v', 'Full_minus_alpha0_v', 'Full_minus_alpha1_v']
ABS_FIELDS = ['After_v', 'Before_v', 'Frozen_v', 'After_s', 'After_t', 'After_R30',
              'After_R50', 'delta_total_v', 'delta_current_v', 'delta_inherited_v']
DEV_FIELDS = ['After_v', 'Before_v', 'After_s', 'After_t', 'After_R30', 'After_R50', 'delta_v']

def read(p):
    if p.suffix == '.gz':
        with gzip.open(p, 'rt') as f: return json.load(f)
    return json.loads(p.read_text())

def rows(p):
    with gzip.open(p, 'rt') as f: return [json.loads(line) for line in f]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def development(part):
    groups = collections.defaultdict(list)
    for r in part: groups[r['parent_id']].append(r)
    ids = sorted(groups)
    assert len(ids) == 32 and len(part) == 64 and all(len(g) == 2 for g in groups.values())
    matrix = np.array([[math.fsum(r[f] for r in groups[i])/len(groups[i]) for f in DEV_FIELDS] for i in ids])
    rng = np.random.default_rng(20261011)
    boot = np.concatenate([matrix[rng.integers(0, len(ids), (100, len(ids)))].mean(1) for _ in range(100)])
    ci = np.quantile(boot, [.025, .975], axis=0)
    ratios = sorted(x for r in part for x in r['entropy_ratios'])
    return dict(parents=len(ids), cells=len(part), values={f: dict(
        parent_macro=float(matrix[:, j].mean()), query_macro=math.fsum(r[f] for r in part)/len(part), ci95=ci[:, j].tolist()) for j, f in enumerate(DEV_FIELDS)},
        harm_gt20pp_parents=int((matrix[:, -1] < -.2).sum()),
        actual_update_arrivals=sum(r['updates'] > 0 for r in part),
        optimizer_steps=sum(r['updates'] for r in part), backward_steps=sum(r['backwards'] for r in part),
        skip_steps=sum(r['skips'] for r in part), recoveries=sum(r['recoveries'] for r in part),
        update_rate=sum(r['updates'] > 0 for r in part)/len(part),
        entropy_ratio_median=float(np.quantile(ratios, .5)), entropy_ratio_p95=float(np.quantile(ratios, .95)))

def check_dev(actual, expected):
    checks = 0
    for k, x in actual.items():
        if k == 'values':
            for f, z in x.items():
                for q in ['parent_macro', 'query_macro']:
                    assert abs(z[q]-expected[k][f][q]) < 4e-12; checks += 1
                assert max(abs(a-b) for a, b in zip(z['ci95'], expected[k][f]['ci95'])) < 4e-12; checks += 2
        else:
            assert abs(x-expected[k]) < 4e-12, k; checks += 1
    return checks

def saved_effect(part, field):
    groups = collections.defaultdict(list)
    for r in part:
        if r[field] is not None: groups[r['parent_id']].append(r[field])
    ids = sorted(groups)
    values = np.array([math.fsum(groups[i])/len(groups[i]) for i in ids])
    rng = np.random.default_rng(20261011)
    boot = np.concatenate([values[rng.integers(0, len(ids), (100, len(ids)))].mean(1) for _ in range(100)])
    return dict(parents=len(ids), eligible_cells=sum(len(g) for g in groups.values()),
        missing_cells=sum(r[field] is None for r in part), mean=float(values.mean()),
        ci95=np.quantile(boot, [.025, .975]).tolist(), positive_parents=int((values > 0).sum()),
        negative_parents=int((values < 0).sum()), parent_values=[dict(parent_id=i, value=float(v)) for i, v in zip(ids, values)])

def run(root):
    out = Path(root)/REL; checks = 0
    rr = rows(out/'ORDER_ROWS.jsonl.gz')
    assert len(rr) == 2304 and len({r['parent_id'] for r in rr}) == 256
    assert all('source_id' not in r and 'query_ordinal' not in r for r in rr)
    summary = read(out/'ORDER_STATISTICS.json.gz')
    for ds in ['hc2', 'vidstg']:
        part = [r for r in rr if r['dataset'] == ds]
        assert len(part) == 1152 and set(r['arm'] for r in part) == set(ARMS)
        cells = {(r['arm'], r['order'], r['parent_id']): r for r in part}; assert len(cells) == 1152
        paired = []
        for r in part:
            assert r['query_reset'] and r['Adam_reset'] and r['source_reset_per_arm_order']
            assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v']) < 2e-10
            if r['arm'] == 'full':
                z = dict(r)
                for arm in ARMS[1:]:
                    c = cells[arm, r['order'], r['parent_id']]
                    assert c['arrival'] == r['arrival'] and c['Frozen_v'] == r['Frozen_v'] and c['After_t'] == r['After_t']
                    z['Full_minus_'+arm+'_v'] = r['After_v']-c['After_v']
                paired.append(z)
        for order in ['pooled']+ORDERS:
            chosen = paired if order == 'pooled' else [r for r in paired if r['order'] == order]
            checks += compare(summary['statistics'][ds]['contrasts'][order], independent(chosen, FIELDS))
            for arm in ARMS:
                chosen = [r for r in part if r['arm'] == arm and (order == 'pooled' or r['order'] == order)]
                checks += compare(summary['statistics'][ds]['all_arms'][arm][order], independent(chosen, ABS_FIELDS))
    dev = rows(out/'DEVELOPMENT_ROWS.jsonl.gz'); assert len(dev) == 1152
    stats = read(out/'DEVELOPMENT_STATISTICS.json')
    selected = read(out/'SELECTED_CONFIGS.json')['selected']
    for ds in ['hc2', 'vidstg']:
        scores = {}
        for candidate, entry in stats['statistics'][ds].items():
            part = [r for r in dev if r['dataset'] == ds and r['candidate'] == candidate]
            assert len({r['parent_id'] for r in part}) == 32 and all(r['is_development_parent'] for r in part)
            for r in part: assert abs(r['After_v']-r['Before_v']-r['delta_v']) < 2e-10
            scores[candidate] = development(part); checks += check_dev(scores[candidate], entry['all_population'])
        assert len(scores) == 9
        for method in ['TENT', 'SAR']:
            candidates = [n for n in scores if n.startswith(method+'_')]
            best = max(scores[n]['values']['After_v']['parent_macro'] for n in candidates)
            tied = [n for n in candidates if best-scores[n]['values']['After_v']['parent_macro'] <= 1e-12]
            configs = {n: next(r['config'] for r in dev if r['dataset'] == ds and r['candidate'] == n) for n in candidates}
            winner = min(tied, key=lambda n: (scores[n]['harm_gt20pp_parents'], configs[n]['lr'], configs[n].get('margin_fraction', 0)))
            assert selected[ds][method]['candidate'] == winner and selected[ds][method]['config'] == configs[winner]
            assert abs(selected[ds][method]['dev_score']-scores[winner]['values']['After_v']['parent_macro']) < 4e-12
            checks += 3
    saved = rows(out/'SAVED_FRAME_EFFECT_ROWS.jsonl.gz'); assert len(saved) == 1244
    effects = read(out/'UNOBSERVED_FRAME_EFFECTS.json')
    for label, entry in effects['records'].items():
        part = [r for r in saved if r['cohort'] == label]; assert len(part) == entry['rows']
        for field, key in [('unobserved_iou_delta', 'unobserved'), ('observed_iou_delta', 'observed')]:
            actual = saved_effect(part, field); expected = entry[key]
            for k in ['parents', 'eligible_cells', 'missing_cells', 'positive_parents', 'negative_parents']:
                assert actual[k] == expected[k]; checks += 1
            assert abs(actual['mean']-expected['mean']) < 4e-12
            assert max(abs(a-b) for a, b in zip(actual['ci95'], expected['ci95'])) < 4e-12; checks += 3
            for a, b in zip(actual['parent_values'], expected['parent_values']):
                assert a['parent_id'] == b['parent_id'] and abs(a['value']-b['value']) < 4e-12; checks += 1
    table = read(out/'TABLE_ROUNDING_CORRECTION.json')
    table_rows = rows(out/'TABLE_ROUNDING_ROWS.jsonl.gz')
    groups = collections.defaultdict(list)
    for r in table_rows: groups[r['parent_id']].append(r['After_v'])
    exact = math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups)*100
    assert len(groups) == 237 and len(table_rows) == table['logical_rows']
    assert abs(exact-table['exact_parent_macro_percent']) < 4e-12 and f'{exact:.3f}' == table['correct_percent_display'] == '34.229'; checks += 3
    d = read(out/'DESIGN_PUBLIC.json')
    assert d['order_fits'] == 2304 and d['development_arrivals'] == 1152 and d['Vid_TENT_accepted_prefix'] == 13747
    assert not d['formal82710_global_completed'] and not d['remaining_formal_GT_scoring_executed'] and not d['production_promoted']
    return dict(status='pass', scope='all2304 anonymous order outcomes/all1152 dev grid outcomes/all1244 saved observed-unobserved outcomes/table correction/independent10000 parent bootstrap and exact locked dev selection',
        order_rows=2304, development_rows=1152, saved_frame_effect_rows=1244, table_rounding_rows=len(table_rows),
        scalar_comparisons=checks, bootstrap=10000, all_negative_neutral_unknown_retained=True,
        order_rows_sha256=digest(out/'ORDER_ROWS.jsonl.gz'), development_rows_sha256=digest(out/'DEVELOPMENT_ROWS.jsonl.gz'),
        CPU_only=True, GT_read=False, raw_GT_read=False, model_calls=0, new_DINO_calls=0)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])))
