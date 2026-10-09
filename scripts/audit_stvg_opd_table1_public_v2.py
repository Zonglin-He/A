"""Portable full-population scalar verification using only anonymous public rows.

Python/NumPy only: no workspace imports, models, media, captions or annotations.
Recomputes all official-query and parent macros, paired bootstrap, decomposition,
negative tails, descriptive strata, costs and deterministic case selection.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

FIELDS=['After_v','After_t','After_s','After_R30','After_R50','delta_total_v','delta_current_v','delta_inherited_v']


def read(p): return json.loads(Path(p).read_text())


def rows(p):
    with gzip.open(p,'rt') as f: return [json.loads(s) for s in f]


def summary(rr, fields=FIELDS):
    query=collections.defaultdict(list)
    for r in rr: query[r['query_ordinal']].append(r)
    parent=collections.defaultdict(list); queries=[]
    for q in sorted(query):
        group=query[q]
        assert len(group)==3 and len({r['order'] for r in group})==3
        assert len({r['source_id'] for r in group})==1
        values=[math.fsum(r[k] for r in group)/3 for k in fields]
        queries.append(values);parent[group[0]['source_id']].append(values)
    sources=sorted(parent)
    matrix=np.array([[math.fsum(v[j] for v in parent[s])/len(parent[s]) for j in range(len(fields))] for s in sources])
    rng=np.random.default_rng(20261008)
    bootstrap=np.concatenate([matrix[rng.integers(0,len(sources),(50,len(sources)))].mean(1) for _ in range(200)])
    ci=np.quantile(bootstrap,[.025,.975],axis=0)
    return dict(queries=len(query),parent_sources=len(sources),logical_arrivals=len(rr),metrics={
        k:dict(source_macro=float(matrix[:,j].mean()),official_query_macro=float(np.asarray(queries)[:,j].mean()),
               paired_parent_ci95=ci[:,j].tolist(),
               harm_gt5pp_parent_sources=int((matrix[:,j]<-.05).sum()) if k.startswith('delta_') else None,
               harm_gt20pp_parent_sources=int((matrix[:,j]<-.2).sum()) if k.startswith('delta_') else None)
        for j,k in enumerate(fields)}),sources,matrix


def strata(rr):
    def group(part):
        bysource=collections.defaultdict(list)
        for r in part:bysource[r['source_id']].append(r)
        return dict(arrivals=len(part),parents=len(bysource),
            parent_mean_delta_total_v=float(np.mean([np.mean([r['delta_total_v'] for r in v]) for v in bysource.values()])) if bysource else None,
            parent_mean_delta_current_v=float(np.mean([np.mean([r['delta_current_v'] for r in v]) for v in bysource.values()])) if bysource else None)
    quality={name:group([r for r in rr if predicate(r['admitted_expert_GT_IoU'])]) for name,predicate in [
        ('missing_GT_evaluable_expert',lambda v:v is None),('expert_IoU_below_0.3',lambda v:v is not None and v<.3),
        ('expert_IoU_0.3_to_0.5',lambda v:v is not None and .3<=v<.5),('expert_IoU_at_least_0.5',lambda v:v is not None and v>=.5)]}
    first={r['query_ordinal']:r for r in rr};features={};cutpoints={}
    for key in ['event_duration_fraction','normalized_target_motion']:
        cuts=np.quantile([r[key] for r in first.values()],[.25,.5,.75]);cutpoints[key]=cuts.tolist()
        features[key]={f'quartile_{j+1}':group([r for r in rr if np.searchsorted(cuts,r[key],side='right')==j]) for j in range(4)}
    types=sorted({json.dumps(r['query_type'],sort_keys=True) for r in rr})
    return dict(quality=quality,features=features,cutpoints=cutpoints,
        query_type={kind:group([r for r in rr if json.dumps(r['query_type'],sort_keys=True)==kind]) for kind in types},
        descriptive_only_not_an_intervention=True)


def run(directory):
    p=Path(directory);checks=0;maximum=0.;logical=0
    def compare(new,old):
        nonlocal checks,maximum
        if isinstance(new,dict):
            for k,v in new.items():assert k in old,k;compare(v,old[k])
        elif isinstance(new,list):
            assert len(new)==len(old);checks+=1
            for a,b in zip(new,old):compare(a,b)
        elif isinstance(new,(float,int)) and not isinstance(new,bool):
            error=abs(new-old);maximum=max(maximum,error);assert error<3e-12,(new,old,error);checks+=1
        else:assert new==old,(new,old);checks+=1
    assembled=read(p/'TABLE1_SUMMARY.json');root=read(p/'TABLE1_ROOT_INDEPENDENT_AUDIT.json')
    actual=read(p/'TABLE1_ACTUAL_ROOT_POPULATION_READBACK.json');effects=read(p/'TABLE1_ALL_PARENT_EFFECTS.json')
    failures=read(p/'TABLE1_FAILURE_STRATA.json');cost=read(p/'TABLE1_COST.json')
    cases=read(p/'TABLE1_ACTUAL_ROOT_CASE_SELECTION.json');chain=read(p/'TABLE1_ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json')
    for ds in ['hc2','vidstg']:
        methods={f.parent.name:rows(f) for f in sorted((p/'anonymous_rows'/ds).glob('*/ROWS.jsonl.gz'))}
        assert set(methods)==set(assembled['datasets'][ds])
        full=methods['spatial_opd'];match={(r['query_ordinal'],r['order']):r for r in full}
        assert len(match)==(10446 if ds=='hc2' else 30909)
        dev=set(read(p/('P1_'+ds)/'SUMMARY.json')['excluded_development_anonymous_source_ids']);assert len(dev)==32
        for name,rr in methods.items():
            assert len(rr)==len(match);logical+=len(rr)
            for r in rr:
                ref=match[r['query_ordinal'],r['order']]
                compare([r['source_id'],r['arrival']],[ref['source_id'],ref['arrival']])
                for key in ['v','t','s']:compare(r['Frozen_'+key],ref['Frozen_'+key])
                compare(r['delta_total_v'],r['delta_current_v']+r['delta_inherited_v'])
                for name2 in ['Frozen','Before','After']:
                    for threshold in [.3,.5]:
                        key=name2+'_R'+str(int(threshold*100))
                        if key in r:compare(r[key],float(r[name2+'_v']>threshold))
            for subset,part in [('all_official_queries',rr),('excluding_32_tuning_parent_sources',[r for r in rr if r['source_id'] not in dev])]:
                result,sources,matrix=summary(part)
                compare(result,assembled['datasets'][ds][name][subset])
                compare(result,root['datasets'][ds][name][subset])
                simple={**result,'metrics':{k:{x:v[x] for x in ['source_macro','official_query_macro','paired_parent_ci95']} for k,v in result['metrics'].items()}}
                compare(simple,actual['datasets'][ds][name][subset])
                if subset=='all_official_queries':
                    compare([dict(source_id=s,delta_total_v=float(matrix[i,FIELDS.index('delta_total_v')])) for i,s in enumerate(sources)],effects[ds][name])
            paired=[dict(query_ordinal=r['query_ordinal'],source_id=r['source_id'],order=r['order'],
                **{'OPD_minus_'+k:match[r['query_ordinal'],r['order']]['After_'+k]-r['After_'+k] for k in ['v','t','s','R30','R50']}) for r in rr]
            result,_,_=summary(paired,['OPD_minus_'+k for k in ['v','t','s','R30','R50']])
            compare(result,root['paired_OPD_vs_methods'][ds][name])
        compare(strata(full),failures[ds])
        for r in full:compare(r['Frozen_t'],r['Before_t']);compare(r['Frozen_t'],r['After_t'])
        derived=dict(arrivals=len(full),synchronized_fit_wall_seconds=math.fsum(r['compute']['fit_GPU_seconds'] for r in full),
            synchronized_fit_wall_seconds_per_arrival=float(np.mean([r['compute']['fit_GPU_seconds'] for r in full])),
            shared_capture_wall_seconds=math.fsum(r['compute']['shared_capture_seconds'] for r in full),
            independent_CPU_math_seconds=math.fsum(r['compute']['CPU_math_seconds'] for r in full),
            actual_new_DINO_calls=sum(r['compute']['new_DINO_calls'] for r in full),
            max_CUDA_peak_allocated=max(r['compute']['CUDA_peak_allocated'] for r in full),
            actual_backward_calls=sum(r['gradient_calls'] for r in full))
        # Floating sum order on seconds is allowed one nanosecond; metrics use 3e-12.
        for key,value in derived.items():
            old=cost[ds][key];error=abs(value-old);assert error<1e-9,(key,error);checks+=1
        candidates=[r for r in full if r['observed_frames'] and r['central_expert_reward_round_mean_delta'] is not None]
        used=set()
        for case in cases['datasets'][ds]:
            kind=case['kind']
            group=[r for r in candidates if kind!='expert_reward_task_mismatch' or (r['delta_current_v']<0 and r['central_expert_reward_round_mean_delta']>0)]
            group=sorted(group,key=lambda r:((-r['delta_current_v'] if kind=='success' else r['delta_current_v']),r['query_ordinal'],r['order']))
            chosen=next(r for r in group if r['query_ordinal'] not in used);used.add(chosen['query_ordinal'])
            compare({k:chosen[k] for k in ['query_ordinal','order','arrival','source_id','delta_current_v','delta_inherited_v']},case)
            item=next(r for r in chain['datasets'][ds] if r['kind']==kind)
            assert len(item['rounds'])==(40 if ds=='hc2' else 10)
            for key in ['sample_best_gt_iou','teacher_vs_uniform_gt_iou','central_gt_delta','reward_variance','ESS','weight_max']:
                compare(float(np.mean([r[key] for r in item['rounds']])),chosen[key])
    assert logical==root['anonymous_logical_rows']==actual['anonymous_logical_rows']==258576
    tensor_receipt=read(p/'TABLE1_ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')
    assert tensor_receipt['status']=='pass' and tensor_receipt['counts']['arrivals']==tensor_receipt['counts']['math_dictionary_exact']==41355
    return dict(status='pass',scope='portable anonymous all-population scalar recomputation only',
        logical_rows=logical,scalar_comparisons=checks,maximum_metric_error=maximum,
        bootstrap=10000,seed=20261008,all_positive_negative_and_no_update_rows=True,
        GT_media_models_read=False,private_math_tensor_payload_read=False,
        dense_and_tensor_audit_receipts_are_root_evidence_not_reproduced_from_scalar_rows=True)


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),sort_keys=True))
