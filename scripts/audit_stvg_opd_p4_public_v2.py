"""Portable independent arithmetic of all 15504 anonymous P4 robustness arrivals.

Only JSON/gzip/NumPy are used. This verifies public arithmetic, not private
model fits, decoder Jacobians, RGB identities or label provenance.
"""
import collections
import gzip
import json
import math
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, counter, path=''):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for k in expected: compare(actual[k], expected[k], counter, path+'/'+k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)): compare(a, b, counter, path+'/'+str(i))
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert abs(actual-expected) <= 3e-12*max(1., abs(expected)), (path, actual, expected)
        counter[0] += 1
    else:
        assert actual == expected, (path, actual, expected)
        counter[0] += 1


def aggregate(rows, fields):
    byq = collections.defaultdict(list)
    cells = set()
    for r in rows:
        key = (r['query_ordinal'], r['condition'], r['order'])
        assert key not in cells, 'Repeated query/condition/order'
        cells.add(key); byq[r['query_ordinal']].append(r)
    assert byq and len({len(v) for v in byq.values()}) == 1
    byp = collections.defaultdict(list); qm = []
    for q in sorted(byq):
        group = byq[q]; assert len({r['source_id'] for r in group}) == 1
        v = [math.fsum(r[f] for r in group)/len(group) for f in fields]
        qm.append(v); byp[group[0]['source_id']].append(v)
    ids = sorted(byp)
    mat = np.array([[math.fsum(v[j] for v in byp[p])/len(byp[p])
                     for j in range(len(fields))] for p in ids])
    rng = np.random.default_rng(20261006)
    dist = np.concatenate([mat[rng.integers(0, len(ids), (50, len(ids)))].mean(1) for _ in range(200)])
    ci = np.quantile(dist, [.025, .975], axis=0)
    metrics = {f:dict(mean=float(mat[:,j].mean()), ci95=ci[:,j].tolist(),
        query_macro=float(np.asarray(qm)[:,j].mean()),
        harm_gt5pp_sources=int((mat[:,j]<-.05).sum()) if f.startswith(('delta_', 'Full_minus_')) else None,
        harm_gt20pp_sources=int((mat[:,j]<-.2).sum()) if f.startswith(('delta_', 'Full_minus_')) else None)
        for j,f in enumerate(fields)}
    return dict(queries=len(byq),sources=len(ids),cells=len(rows),metrics=metrics), ids, mat


def paired(full, control):
    lookup={(r['query_ordinal'],r['condition'],r['order']):r for r in full}
    assert len(lookup)==len(control)==len(full)
    result=[]
    for r in control:
        f=lookup[r['query_ordinal'],r['condition'],r['order']]
        assert f['source_id']==r['source_id'] and f['arrival']==r['arrival']
        assert all(abs(f['Frozen_'+m]-r['Frozen_'+m])<2e-10 for m in ['v','t','s'])
        result.append(dict(query_ordinal=r['query_ordinal'],source_id=r['source_id'],
            condition=r['condition'],order=r['order'],
            **{'Full_minus_'+m:f['After_'+m]-r['After_'+m] for m in ['v','t','s','R30','R50']}))
    return aggregate(result,['Full_minus_'+m for m in ['v','t','s','R30','R50']])[0]


def strata(rows):
    def group(rr):
        parents=collections.defaultdict(list)
        for r in rr: parents[r['source_id']].append(r)
        return dict(arrivals=len(rr),parents=len(parents),**{
            'parent_mean_delta_'+k+'_v':math.fsum(math.fsum(r['delta_'+k+'_v'] for r in v)/len(v)
                for v in parents.values())/len(parents) if parents else None for k in ['total','current']})
    quality={n:group([r for r in rows if fn(r['admitted_expert_GT_IoU'])]) for n,fn in [
        ('missing_GT_evaluable_expert',lambda x:x is None),('expert_IoU_below_0.3',lambda x:x is not None and x<.3),
        ('expert_IoU_0.3_to_0.5',lambda x:x is not None and .3<=x<.5),('expert_IoU_at_least_0.5',lambda x:x is not None and x>=.5)]}
    first={r['query_ordinal']:r for r in rows};features={};cuts={}
    for field in ['event_duration_fraction','normalized_target_motion']:
        q=np.quantile([r[field] for r in first.values()],[.25,.5,.75]);cuts[field]=q.tolist()
        features[field]={'quartile_'+str(i+1):group([r for r in rows if np.searchsorted(q,r[field],side='right')==i]) for i in range(4)}
    types=sorted({json.dumps(r['query_type'],sort_keys=True) for r in rows})
    return dict(quality=quality,features=features,cutpoints=cuts,
        query_type={t:group([r for r in rows if json.dumps(r['query_type'],sort_keys=True)==t]) for t in types},
        descriptive_only_not_an_intervention=True)


def costs(rr):
    components=[r['compute'].get('input_capture_components') for r in rr]
    observed=[v for v in components if v is not None]
    def mean(field): return math.fsum(r['compute'][field] for r in rr)/len(rr)
    return dict(logical_arrivals=len(rr), real_fit_seconds_per_arrival=mean('fit_GPU_seconds'),
        shared_capture_seconds_per_arrival=mean('shared_capture_seconds'), independent_CPU_math_seconds_per_arrival=mean('CPU_math_seconds'),
        actual_backward_rounds_mean=math.fsum(r['gradient_calls'] for r in rr)/len(rr),
        actual_new_DINO_calls_in_recorded_original_execution=sum(r['compute']['new_DINO_calls'] for r in rr),
        nominal_observation_budget=sorted({r['compute']['DINO_observation_budget'] for r in rr}),
        maximum_CUDA_peak_allocated=max(r['compute']['CUDA_peak_allocated'] for r in rr),
        mean_component_seconds={f:math.fsum(r[f] for r in observed)/len(observed) if observed else None
            for f in ['decode_corruption_seconds','STVG_frozen_forward_seconds','recorded_expert_forward_seconds']},
        capture_component_rows_available=len(observed),exact_stream_reuse_rows=sum(r['reused_complete_identical_stream'] for r in rr),
        reused_stream_cost_is_original_actual_measurement_not_new_GPU_time=True,
        recorded_expert_forward_may_be_reused_not_cold_latency=True,shared_capture_not_added_once_per_variant=True,
        planned_steps=sorted({r['config']['steps'] for r in rr}),trainable_parameters=sorted({r['active_parameters'] for r in rr}))


def audit(directory):
    directory=Path(directory); expected=read(directory/'ROOT_STATISTICS.json')
    stages=list(expected['by_stage']); assert set(stages)=={'P4_hc2','P4_vidstg'}
    counts_by_stage={'P4_hc2':237,'P4_vidstg':732}
    conditions=['clean']+[f'{family}_{severity}' for severity in ['2.5','5','10'] for family in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']]
    count=[0]; summaries={};contrasts={};pool={};effects={};cost={};diagnosis={};data={}
    for stage in stages:
        parent=directory.parent/stage
        if not parent.exists():parent=directory/'stages'/stage
        with gzip.open(parent/'ROWS.jsonl.gz','rt') as f: rr=list(map(json.loads,f))
        saved=read(parent/'SUMMARY.json');data[stage]=rr
        assert len({(r['query_ordinal'],r['condition'],r['order'],r['arm']) for r in rr})==len(rr)
        summaries[stage]={};contrasts[stage]={};effects[stage]={};diagnosis[stage]={}
        arms=list(next(iter(saved['conditions'].values())))
        assert len({r['query_ordinal'] for r in rr})==len({r['source_id'] for r in rr})==counts_by_stage[stage]
        assert list(saved['conditions'])==conditions and arms==['on_policy']
        assert {r['condition'] for r in rr}==set(conditions) and {r['order'] for r in rr}=={'order1'}
        assert all(r['arm']=='on_policy' and not r['reused_complete_identical_stream'] for r in rr)
        assert len(rr)==counts_by_stage[stage]*16
        for condition in conditions:
            chosen=[r for r in rr if r['condition']==condition]
            assert len(chosen)==counts_by_stage[stage] and len({r['query_ordinal'] for r in chosen})==len(chosen)
            assert len({r['source_id'] for r in chosen})==len(chosen)
        for r in rr:
            assert r['Frozen_t']==r['Before_t']==r['After_t']
            for m in ['v','t','s']:
                for label in ['Frozen','Before','After']:
                    compare(r[label+'_'+m],r['values'][label][m],count)
                    assert 0<=r[label+'_'+m]<=1+2e-10
                compare(r['delta_total_'+m],r['After_'+m]-r['Frozen_'+m],count)
                compare(r['delta_current_'+m],r['After_'+m]-r['Before_'+m],count)
                compare(r['delta_inherited_'+m],r['Before_'+m]-r['Frozen_'+m],count)
            for label in ['Frozen','Before','After']:
                for cut in [.3,.5]:compare(r[label+'_R'+str(int(cut*100))],float(r[label+'_v']>cut),count)
            for cut in [.3,.5]:
                b,a=r['Frozen_v']>cut,r['After_v']>cut
                compare(r['correct_to_wrong_'+str(cut)],float(b and not a),count)
                compare(r['wrong_to_correct_'+str(cut)],float(not b and a),count)
        for condition, oldarms in saved['conditions'].items():
            part=[r for r in rr if r['condition']==condition];full=[r for r in part if r['arm']=='on_policy']
            summaries[stage][condition]={};contrasts[stage][condition]={};effects[stage][condition]={}
            for arm,old in oldarms.items():
                chosen=[r for r in part if r['arm']==arm];fields=list(old['metrics'])
                result,ids,mat=aggregate(chosen,fields)
                score_summary=dict(sources=result['sources'],cells=result['cells'],metrics={})
                for j,f in enumerate(fields):
                    metric=dict(result['metrics'][f]);is_delta=f.startswith('delta_')
                    metric.update(order_values=[math.fsum(r[f] for r in chosen if r['order']==o)/sum(r['order']==o for r in chosen)
                        for o in sorted({r['order'] for r in chosen})],
                        gross_gain_pp=float(np.maximum(mat[:,j],0).mean()*100) if is_delta else None,
                        gross_loss_pp=float(-np.minimum(mat[:,j],0).mean()*100) if is_delta else None,
                        harm_gt5pp_cells=sum(r[f]<-.05 for r in chosen) if is_delta else None,
                        harm_gt20pp_cells=sum(r[f]<-.2 for r in chosen) if is_delta else None)
                    score_summary['metrics'][f]=metric
                compare(score_summary,old,count)
                summaries[stage][condition][arm]=result;contrasts[stage][condition][arm]=paired(full,chosen)
                j=fields.index('delta_total_v')
                effects[stage][condition][arm]=[dict(source_id=p,delta_total_v=float(mat[i,j])) for i,p in enumerate(ids)]
            diagnosis[stage][condition]=strata(full)
        pool[stage]={a:paired([r for r in rr if r['arm']=='on_policy'],[r for r in rr if r['arm']==a]) for a in arms}
        cost[stage]={a:costs([r for r in rr if r['arm']==a]) for a in arms}
    for key,val in [('by_stage',summaries),('Full_minus_controls',contrasts),('Full_minus_controls_condition_mean',pool)]:compare(val,expected[key],count)
    for file,val in [('ALL_PARENT_EFFECTS.json',effects),('COST.json',cost),('FAILURE_STRATA.json',diagnosis)]:compare(val,read(directory/file),count)
    assert sum(map(len,data.values()))==expected['anonymous_logical_rows']==15504
    return dict(status='pass',scope='all anonymous rows, complete source/bootstrap/contrast/tail/strata/cost arithmetic; no private fit certification',
        logical_rows=15504,distinct_parent_sources_across_datasets=969,scalar_comparisons=count[0],
        bootstrap=10000,seed=20261006,all_negative_rows_retained=True,no_model_or_GT_payload_reads=True),data,pool


if __name__=='__main__':print(json.dumps(audit(sys.argv[1])[0],sort_keys=True))
