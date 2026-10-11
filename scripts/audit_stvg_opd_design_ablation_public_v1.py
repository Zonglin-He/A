"""Portable independent arithmetic for all anonymous component/coverage results.

Only JSON, gzip and NumPy are required. No private GT, media, weights or fit
payload is read. This audit establishes public arithmetic, not decoder proofs.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with gzip.open(path, 'rt') as f: return [json.loads(line) for line in f]


def compare(a, b, count, path=''):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for key in b: compare(a[key], b[key], count, path+'/'+key)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, count, path+'/'+str(i))
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        assert math.isfinite(a) and math.isfinite(b), path
        assert abs(a-b) <= 3e-12*max(1., abs(b)), (path, a, b)
        count[0] += 1
    else:
        assert a == b, (path, a, b); count[0] += 1


def aggregate(data, fields):
    cells = set(); parents = collections.defaultdict(list)
    for r in data:
        key = (r['query_ordinal'], r['order'], r['condition'])
        assert key not in cells; cells.add(key); parents[r['source_id']].append(r)
    if not parents: return dict(sources=0, cells=0, metrics={f:None for f in fields})
    assert len({len(v) for v in parents.values()}) == 1
    identifiers = sorted(parents)
    mat = np.asarray([[math.fsum(r[f] for r in parents[p])/len(parents[p])
                       for f in fields] for p in identifiers])
    generator = np.random.default_rng(20261011)
    dist = np.concatenate([mat[generator.integers(0, len(mat), (50, len(mat)))].mean(axis=1)
                           for _ in range(200)])
    bounds = np.quantile(dist, [.025, .975], axis=0)
    metrics = {}
    for j, field in enumerate(fields):
        vec = mat[:,j]
        metrics[field] = dict(mean=float(np.mean(vec)), ci95=bounds[:,j].tolist(),
            query_macro=math.fsum(r[field] for r in data)/len(data),
            parent_values=[dict(source_id=p, value=float(vec[i])) for i,p in enumerate(identifiers)],
            negative_parent_count=int(sum(v < 0 for v in vec)), positive_parent_count=int(sum(v > 0 for v in vec)),
            harm_gt5pp_sources=int(sum(v < -.05 for v in vec)), harm_gt20pp_sources=int(sum(v < -.2 for v in vec)),
            worst_parent=float(min(vec)), best_parent=float(max(vec)))
    return dict(sources=len(parents), cells=len(data), metrics=metrics)


def audit(directory):
    directory = Path(directory); counter = [0]
    table = read(directory/'COMPONENT_TABLE.json'); stats = read(directory/'ALL_STATISTICS.json')
    cost = read(directory/'COST_AND_EXPERT_QUALITY.json'); coverage = read(directory/'COVERAGE_ANALYSIS.json')
    design = read(directory/'receipts/DESIGN_LOCK.json')
    fields = ['After_v','Before_v','Frozen_v','After_s','After_t','After_R30','After_R50',
              'Full_minus_control_v','delta_total_v','delta_current_v','delta_inherited_v']
    names = ['A_hc2_cross_clean','A_vidstg_cross_clean','A_hc2_same_5percent','A_vidstg_same_5percent']
    labels = {'whole_video':'w/o Event-Scoped Sampling','carry_query':'w/o Fast-State Reset',
              'full_LN_carry':'w/o Slow-State Attenuation'}
    total = 0; reproduced = {}; allrows = {}
    for name in names:
        data = rows(directory/name/'ROWS.jsonl.gz'); allrows[name] = data; total += len(data)
        assert len({(r['query_ordinal'],r['condition'],r['order'],r['arm']) for r in data}) == len(data)
        assert len(data) == design['stages'][name]['expected_new_fits']
        assert len({r['source_id'] for r in data}) == len({r['query_ordinal'] for r in data}) == 128
        for r in data:
            assert r['active_parameters'] == 1792 and r['Adam_reset']
            assert r['query_reset'] == (r['arm'] != 'carry_query')
            assert r['observation_scope'] == ('whole_observed_video' if r['arm']=='whole_video' else 'native_event')
            assert r['config'] == design['stages'][name]['variant_configs'][r['arm']]
            assert r['After_t'] == r['Before_t'] == r['Frozen_t'] == r['Event_Full_t']
            assert 0 <= r['native_event_coverage'] <= 1
            for metric in ['v','t','s']:
                for label in ['After','Before','Frozen','Event_Full']:
                    compare(r[label+'_'+metric],r['values'][label][metric],counter)
                    assert 0 <= r[label+'_'+metric] <= 1+2e-10
                compare(r['Full_minus_control_'+metric],r['Event_Full_'+metric]-r['After_'+metric],counter)
                compare(r['delta_total_'+metric],r['After_'+metric]-r['Frozen_'+metric],counter)
                compare(r['delta_current_'+metric],r['After_'+metric]-r['Before_'+metric],counter)
                compare(r['delta_inherited_'+metric],r['Before_'+metric]-r['Frozen_'+metric],counter)
            for label in ['After','Before','Frozen','Event_Full']:
                for cut in [.3,.5]: compare(r[label+'_R'+str(int(cut*100))],float(r[label+'_v']>cut),counter)
            assert r['admitted_references_with_GT'] <= r['admitted_references'] <= r['attempted_expert_calls'] <= r['nominal_selected_observations']
            assert (r['admitted_expert_GT_IoU'] is None) == (r['admitted_references_with_GT'] == 0)
        reproduced[name] = {}
        for arm, label in labels.items():
            part = [r for r in data if r['arm']==arm]; s = design['stages'][name]
            reproduced[name][arm] = dict(pooled=aggregate(part,fields),
                per_condition={c:aggregate([r for r in part if r['condition']==c],fields) for c in s['conditions']},
                per_order={o:aggregate([r for r in part if r['order']==o],fields) for o in s['orders']})
            compare(reproduced[name][arm],stats[name][arm],counter)
            cell = table['table'][label][names.index(name)]; pooled = reproduced[name][arm]['pooled']
            compare(cell['statistics'],pooled,counter)
            compare(cell['mean_percent'],pooled['metrics']['After_v']['mean']*100,counter)
            compare(cell['Full_minus_control_pp'],pooled['metrics']['Full_minus_control_v']['mean']*100,counter)
            compare(cell['Full_minus_control_ci95_pp'],[x*100 for x in pooled['metrics']['Full_minus_control_v']['ci95']],counter)
            assert cell['rows_sha256'] == sha(directory/name/'ROWS.jsonl.gz')
            valid = [r for r in part if r['admitted_expert_GT_IoU'] is not None]
            computed = dict(arrivals=len(part), nominal_selected_observations=sum(r['nominal_selected_observations'] for r in part),
                attempted_observations=sum(r['attempted_expert_calls'] for r in part),
                admitted_references=sum(r['admitted_references'] for r in part),
                admitted_references_with_GT=sum(r['admitted_references_with_GT'] for r in part),
                expert_GT_IoU_mean_over_available_arrivals=math.fsum(r['admitted_expert_GT_IoU'] for r in valid)/len(valid) if valid else None,
                actual_new_DINO_calls=sum(r['compute']['actual_new_DINO_calls'] for r in part) if arm=='whole_video' else 0,
                fit_GPU_seconds_per_arrival=math.fsum(r['compute']['fit_GPU_seconds'] for r in part)/len(part),
                CPU_math_seconds_per_arrival=math.fsum(r['compute']['CPU_math_seconds'] for r in part)/len(part),
                shared_capture_counted_once=True,state_arms_reuse_original_expert=True)
            compare(computed,cost[name][arm],counter)
        if name.endswith('cross_clean'):
            part = [r for r in data if r['arm']=='whole_video']; byp = collections.defaultdict(list)
            for r in part: byp[r['source_id']].append(r['native_event_coverage'])
            assert all(len(v)==2 and v[0]==v[1] for v in byp.values())
            for i,b in enumerate(coverage['bins'][name]):
                subset = [r for r in part if b['lo']<=r['native_event_coverage'] and
                    (r['native_event_coverage']<=b['hi'] if b['last_inclusive'] else r['native_event_coverage']<b['hi'])]
                expected = aggregate(subset,['Full_minus_control_v'])
                compare(expected,b['statistics'],counter)
                assert b['n_cells']==len(subset) and b['n_parents']==expected['sources']
                assert b['lo']==design['coverage_bins'][i] and b['hi']==design['coverage_bins'][i+1]
                assert b['last_inclusive']==(i==3)
            assert sum(b['n_parents'] for b in coverage['bins'][name])==128
    assert total == 5376
    specs = [('Full method','P3','on_policy','After_v'),
        ('w/o Action-Reward Correspondence','P2','shuffled_feedback','After_v'),
        ('w/o On-Policy Refresh','P2','frozen_rollout','After_v'),
        ('w/o Query Residual','P3','LN_only','After_v'),
        ('w/o LayerNorm Adaptation','P3','query_only','After_v'),
        ('w/o Slow-State Consolidation','P3','joint_alpha0','After_v'),
        ('Frozen source','P3','on_policy','Frozen_v')]
    historical_cells = 0
    for label,phase,arm,field in specs:
        for i,(ds,split) in enumerate([('hc2','cross_clean'),('vidstg','cross_clean'),('hc2','same_5percent'),('vidstg','same_5percent')]):
            path = directory/'historical'/f'{phase}_{ds}_{split}'/'ROWS.jsonl.gz'
            data = [r for r in rows(path) if r['arm']==arm]; old = aggregate(data,[field])
            cell = table['table'][label][i]
            compare(cell['statistics'],old,counter); compare(cell['mean_percent'],old['metrics'][field]['mean']*100,counter)
            assert sha(path)==cell['rows_sha256']; historical_cells += 1
    corrections = table['historical_display_corrections']
    assert len(corrections)==2 and corrections[0]['actual_rendered_percent']=='9.516' and corrections[1]['actual_rendered_percent']=='29.094'
    markdown = (directory/'COMPONENT_TABLE.md').read_text(); latex = (directory/'COMPONENT_TABLE.tex').read_text()
    for label,cells in table['table'].items():
        values = [f"{cell['mean_percent']:.3f}" for cell in cells]
        assert '| '+label+' | '+' | '.join(values)+' |' in markdown
        assert label+' & '+' & '.join(values)+' \\\\' in latex
    return dict(status='pass',scope='all new anonymous rows, old table cells, paired parent bootstrap, coverage bins, tails and cost arithmetic',
        new_formal_rows=total, historical_table_cells=historical_cells, scalar_comparisons=counter[0],
        parent_sources_per_target=128, bootstrap=10000, seed=20261011,
        no_private_model_or_GT_payload_reads=True, all_negative_rows_retained=True,
        decoder_or_GPU_precision_proof=False)


if __name__=='__main__': print(json.dumps(audit(sys.argv[1]),sort_keys=True))
