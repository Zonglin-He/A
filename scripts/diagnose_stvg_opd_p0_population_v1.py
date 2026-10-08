"""Descriptive failure attribution on all sealed P0 rows, without selecting a method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

FIELDS=['delta_total_v','delta_current_v','delta_inherited_v','delta_total_s',
    'observed_iou_delta','unobserved_iou_delta','teacher_vs_uniform_gt_iou',
    'sample_best_gt_iou','central_expert_reward_round_mean_delta']

def descriptive(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id']].append(r)
    result=dict(arrivals=len(rows),parent_sources=len(groups),descriptive_postseal_not_an_intervention=True,metrics={})
    for field in FIELDS:
        values=[]
        for source in sorted(groups):
            available=[r[field] for r in groups[source] if r[field] is not None]
            if available:values.append(float(np.mean(available)))
        if not values:
            result['metrics'][field]=dict(mean=None,parent_sources_with_value=0);continue
        values=np.array(values);rng=np.random.default_rng(20261008)
        bootstrap=np.concatenate([values[rng.integers(0,len(values),(100,len(values)))].mean(1) for _ in range(100)])
        result['metrics'][field]=dict(mean=float(values.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
            parent_sources_with_value=len(values),missing_values_not_imputed=True)
    return result

def run():
    verify();assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    design=read(BASE/'DESIGN_LOCK.json');results={};lines=['# All-source P0 failure attribution','',
        'These are postseal descriptive associations, not interventions or new parameter selections. Every positive, negative and no-update row is retained. Quality cutpoints are .3/.5 IoU; duration/motion strata use outcome-independent target-feature quartiles. Figures/cases do not replace the full matched efficacy estimates.','']
    for ds in DATASETS:
        with gzip.open(PUB/('P0_'+ds)/'ROWS.jsonl.gz','rt') as f:rows=[r for r in map(json.loads,f) if r['arm']=='on_policy']
        assert len(rows)==256 and len({r['source_id'] for r in rows})==128
        # Quality evidence is shared across the two independent state orders.
        parents=collections.defaultdict(list)
        for r in rows:parents[r['source_id']].append(r)
        assert all(len(rr)==2 and rr[0]['admitted_expert_GT_IoU']==rr[1]['admitted_expert_GT_IoU'] for rr in parents.values())
        quality={name:[r for r in rows if test(r['admitted_expert_GT_IoU'])] for name,test in [
            ('missing_GT_evaluable_expert',lambda x:x is None),('expert_IoU_below_0.3',lambda x:x is not None and x<.3),
            ('expert_IoU_0.3_to_0.5',lambda x:x is not None and .3<=x<.5),('expert_IoU_at_least_0.5',lambda x:x is not None and x>=.5)]}
        strata={};cutpoints={}
        for feature in ['event_duration_fraction','normalized_target_motion']:
            cuts=np.quantile([rr[0][feature] for rr in parents.values()],[.25,.5,.75]);cutpoints[feature]=cuts.tolist()
            strata[feature]={f'quartile_{j+1}':descriptive([r for r in rows if int(np.searchsorted(cuts,r[feature],side='right'))==j]) for j in range(4)}
        kinds=sorted({json.dumps(r['query_type'],sort_keys=True) for r in rows})
        strata['query_type']={kind:descriptive([r for r in rows if json.dumps(r['query_type'],sort_keys=True)==kind]) for kind in kinds}
        effects=[dict(source_id=source,**{field:float(np.mean([r[field] for r in rr])) for field in ['delta_total_v','delta_current_v','delta_inherited_v']}) for source,rr in parents.items()]
        sorted_effects=sorted(effects,key=lambda x:x['delta_total_v'])
        result=dict(all=descriptive(rows),quality_groups={k:descriptive(rr) for k,rr in quality.items()},strata=strata,feature_quartile_cutpoints=cutpoints,
            all_128_parent_effects_sorted=sorted_effects,empty_support_arrivals=sum(r['empty'] for r in rows),
            no_backward_arrivals=sum(r['gradient_calls']==0 for r in rows),
            configured_rounds=design['datasets'][ds]['config']['steps'],
            average_actual_backward_calls=float(np.mean([r['gradient_calls'] for r in rows])),
            expert_reward_gain_with_current_task_harm_arrivals=sum(r['central_expert_reward_round_mean_delta'] is not None and r['central_expert_reward_round_mean_delta']>0 and r['delta_current_v']<0 for r in rows),
            not_a_proof_of_a_population_causal_quality_effect=True)
        results[ds]=result
        lines+=['## '+ds,'','| Expert quality | Parents | Total ΔvIoU pp | Current pp | Inherited pp | Teacher vs uniform GT IoU pp |','|---|---:|---:|---:|---:|---:|']
        for name,x in result['quality_groups'].items():
            def val(field):
                mean=x['metrics'].get(field,{}).get('mean');return 'n/a' if mean is None else f'{mean*100:+.3f}'
            lines.append(f"| {name} | {x['parent_sources']} | {val('delta_total_v')} | {val('delta_current_v')} | {val('delta_inherited_v')} | {val('teacher_vs_uniform_gt_iou')} |")
        lines += ['',f"Empty support: {result['empty_support_arrivals']}/256 arrivals; zero-backward arrivals: {result['no_backward_arrivals']}/256. Expert-reward gain with negative current vIoU: {result['expert_reward_gain_with_current_task_harm_arrivals']}/256. Neither time readout nor optimizer configuration changed.",'']
    write(PUB/'P0_POPULATION_FAILURE_ATTRIBUTION.json',dict(status='complete_all_P0_parents',datasets=results,
        all_arms_already_sealed=True,configurations_unchanged=True,no_new_GPU_fit=True,no_confirmation_retuning=True,
        descriptive_not_a_fresh_confirmation=True,time=time.time()))
    (PUB/'P0_POPULATION_FAILURE_ATTRIBUTION.md').write_text('\n'.join(lines)+'\n')
    print('P0_FULL_POPULATION_ATTRIBUTION',sum(r['all']['parent_sources'] for r in results.values()))

if __name__=='__main__':run()
