"""Score native XYZ interventions after freezing predictions on the clock-only panel."""
from collections import Counter
import csv

import numpy as np

from download import read_json, sha256, write_json
from native_xyz_r5 import ARMS, CONFIG, R4, RUN, index_rows, locked_roster, validate_inputs
from object_pair_r4_geometry import clock_error
from object_pair_r4_score import metrics, paired, parse, references, score
from score import jsonl


def frozen_predictions(rows):
    seal=read_json(RUN/'prediction_seal.json')
    assert not seal['labels'] and seal['training_steps']==0
    assert seal['input']==sha256(RUN/'input_seal.json')
    assert seal['rows']==len(rows)==107
    for path,digest in seal['files'].items():
        assert sha256(RUN/path)==digest
    predictions=index_rows(RUN/'predictions.jsonl')
    assert rows.keys()==predictions.keys()
    return predictions,seal


def same_data_arm(inputs,predictions,refs,history):
    rows=[]
    for q,row in inputs.items():
        selected=predictions[q]
        truth,text=refs[q]
        responses=selected['response']
        old=history[f'{q}/with_frame']['response']
        replayed=parse(responses['native'],'direction')
        original=parse(old,'direction')
        assert selected['r4_native_equal']==(responses['native']==old)
        active=row['status']=='intervened'
        if not active:
            assert selected['feature_audit']=={} and row['anchors'] is None
            assert responses['assigned']==responses['swapped']==responses['native']
        else:
            assert clock_error(row['direct_assigned'],row['direct_swapped'])==6
            for arm in ('assigned','swapped','assignment_vs_swap'):
                for part in ('depth','fusion'):
                    vals=selected['feature_audit'][arm][part]
                    assert vals['changed_tokens']>0 and vals['outside_changed_tokens']==0
        values={arm:parse(responses[arm],'direction') for arm in ARMS}
        values.update(r4_native=original,
            direct_assigned=row['direct_assigned'] if active else replayed,
            direct_swapped=row['direct_swapped'] if active else replayed,
            r4_direct=row['r4_direct'] if row['r4_direct'] is not None else original)
        rows.append(dict(id=q,image=row['image'],status=row['status'],truth=truth,reference=text,
            response=responses,replayed_matches_r4_text=selected['r4_native_equal'],
            feature_audit=selected['feature_audit'],exclusive_counts=row['exclusive_counts'],
            scores={arm:score(v,truth,'direction') for arm,v in values.items()}))
    return rows


def counterfactual(rows):
    counts=Counter()
    steps=Counter()
    for r in rows:
        scores=r['scores']
        a=scores['assigned']['estimate']
        b=scores['swapped']['estimate']
        d0=scores['direct_assigned']['estimate']
        d1=scores['direct_swapped']['estimate']
        assert clock_error(d0,d1)==6
        counts['active']+=1
        counts['assigned_agrees_within_hour']+=a is not None and clock_error(a,d0)<=1
        counts['swapped_agrees_within_hour']+=b is not None and clock_error(b,d1)<=1
        counts['assigned_agrees_exactly']+=a==d0
        counts['swapped_agrees_exactly']+=b==d1
        if a is not None and b is not None:
            gap=clock_error(a,b)
            counts['both_parsed']+=1
            counts['same_hour']+=gap==0
            counts['changed_hour']+=gap>0
            counts['exact_180']+=gap==6
            counts['within_hour_of_180']+=gap>=5
            counts['both_follow_own_geometry']+=clock_error(a,d0)<=1 and clock_error(b,d1)<=1
            steps[gap*30]+=1
    return dict(counts=dict(counts),circular_answer_change_deg=dict(sorted(steps.items())))


def feature_summary(active):
    details=[r['feature_audit'] for r in active]
    counts=[d['target_tokens'] for d in details]
    summary=dict(evaluated=len(details),target_token_footprint_median=float(np.median(counts)),
                 target_token_footprint_min=min(counts),target_token_footprint_max=max(counts))
    for arm in ('assigned','swapped','assignment_vs_swap'):
        summary[arm]={part:dict(
            median_changed_tokens=float(np.median([r[arm][part]['changed_tokens'] for r in details])),
            total_outside_changed_tokens=sum(r[arm][part]['outside_changed_tokens'] for r in details),
            median_delta_rms_in_target=float(np.median([r[arm][part]['delta_rms_inside'] for r in details])),
            max_absolute_change=max(r[arm][part]['delta_peak'] for r in details))
            for part in ('depth','fusion')}
    return summary


def audit():
    inputs=validate_inputs()
    predicted,seal=frozen_predictions(inputs)
    roster={q:r for q,r in locked_roster().items() if r['category']=='direction'}
    assert inputs.keys()==roster.keys()
    refs,n=references(roster)
    assert n==107
    history=index_rows(R4/'direction_controls.jsonl')
    rows=same_data_arm(inputs,predicted,refs,history)
    active=[r for r in rows if r['status']=='intervened']
    checks=[('r4_native','native'),('native','assigned'),('native','swapped'),
           ('assigned','swapped'),('native','direct_assigned'),('native','direct_swapped'),
           ('direct_assigned','assigned'),('direct_swapped','swapped')]
    cuts={'all':rows,'intervened':active}
    results={name:{arm:metrics(group,arm) for arm in rows[0]['scores']} for name,group in cuts.items()}
    comparisons=[dict(scope=name,**paired(group,before,after))
                 for name,group in cuts.items() for before,after in checks]
    jsonl(RUN/'scored_rows.jsonl',rows)
    with (RUN/'paired.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(comparisons[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(comparisons)
    summary=dict(n=len(rows),source_prompt_agreement=n,metrics=results,paired=comparisons,
        eligibility=dict(Counter(r['status'] for r in rows)),counterfactual=counterfactual(active),
        feature_audit=feature_summary(active),
        native_replay=dict(same_raw=sum(r['replayed_matches_r4_text'] for r in rows),
                           same_parsed=sum(r['scores']['r4_native']['estimate']==r['scores']['native']['estimate'] for r in rows)),
        runtime=read_json(RUN/'runtime.json'),prediction_seal=sha256(RUN/'prediction_seal.json'),
        input_seal=sha256(RUN/'input_seal.json'),scored_rows=sha256(RUN/'scored_rows.jsonl'))
    write_json(RUN/'summary.json',summary)
    print('results', {k:{a:(m['correct'],m['parsed']) for a,m in group.items()} for k,group in results.items()},flush=True)
    print('counterfactual',summary['counterfactual'],'prior baseline replay',summary['native_replay'],flush=True)


if __name__=='__main__':
    audit()
