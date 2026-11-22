"""Grade the locked R4 pair audit and emit the predeclared per-task continuation decision."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import re
import unicodedata

import numpy as np

from download import STUDY, read_json, sha256, write_json
from height_measurements import TOKEN, metric, nfloat
from height_transfer import COLORS, source_files
from object_pair_r4 import CONFIG, GEOMETRY_SEAL, MODEL_SEAL, RUN, check_seal, index_rows, locked_roster
from object_pair_r4_geometry import ARMS, BOX_ARMS, CATEGORIES, clock_error
from score import jsonl


HOURS=dict(one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10,eleven=11,twelve=12)
CLOCK=re.compile(rf"\b(1[012]|[1-9]|{'|'.join(HOURS)})\s*-?\s*(?:o\s*['’]?\s*clock|oclock)\b",re.I)


def parse(text,category):
    clean=unicodedata.normalize('NFKC',text.casefold()).replace('’',"'").replace('′',"'").replace('″','"')
    clean=re.sub(r'\b(is|are|of|at)(?=[+-]?\d)',r'\1 ',clean)
    if category!='direction':
        value=metric(clean)
        if value is None:
            pieces=list(TOKEN.finditer(clean))
            if pieces and all(nfloat(m[1])==0 for m in pieces):
                value=0.0
        return value
    parts=CLOCK.findall(clean)
    candidates={int(token) if token.isdigit() else HOURS[token] for token in parts}
    if len(candidates)==1:
        return candidates.pop()
    if not candidates:
        plain=re.fullmatch(r'\s*(1[012]|[1-9])\s*[.!]?\s*(?:<eos>)?\s*',clean)
        if plain:
            return int(plain[1])
    return None


def references(roster):
    import pyarrow.parquet as pq
    results={}
    exact=0
    for part in source_files():
        for batch in pq.ParquetFile(part).iter_batches(batch_size=128,columns=['id','qa_info','conversations']):
            for raw in batch.to_pylist():
                q=raw['id']
                if q not in roster:
                    continue
                qa=ast.literal_eval(raw['qa_info'])
                conv=ast.literal_eval(raw['conversations'])
                prompt=conv[0]['value']
                for noun,color in zip(qa['class'],COLORS):
                    prompt=prompt.replace('<mask>',f'the {noun} (highlighted by a {color} box)',1)
                assert prompt==roster[q]['prompt'],q
                exact+=1
                number=parse(conv[1]['value'],roster[q]['category'])
                assert number is not None,('unparseable reference',q)
                results[q]=(number,conv[1]['value'])
    assert results.keys()==roster.keys() and exact==len(roster)
    return results,exact


def score(value,truth,category):
    if value is None:
        return dict(estimate=None,success=False,error=None)
    if category=='direction':
        gap=clock_error(value,truth)
        hit=gap<=1
        error=30*gap
    else:
        hit=.75*truth-1e-12<=value<=1.25*truth+1e-12
        error=abs(value-truth)/(truth+1e-4)
    return dict(estimate=value,success=bool(hit),error=error)


def metrics(rows,arm):
    values=[r['scores'][arm] for r in rows]
    scores=[f['error'] for f in values if f['error'] is not None]
    count=sum(f['success'] for f in values)
    return dict(n=len(rows),parsed=len(scores),correct=count,accuracy_pct=100.*count/len(rows),
                mean_error=float(np.mean(scores)) if scores else None,
                median_error=float(np.median(scores)) if scores else None)


def paired(rows,before,after):
    base=[r['scores'][before] for r in rows]
    new=[r['scores'][after] for r in rows]
    changes=np.array([int(b['success'])-int(a['success']) for a,b in zip(base,new,strict=True)])
    indices=defaultdict(list)
    for i,row in enumerate(rows):
        indices[row['image']].append(i)
    totals=np.array([len(v) for v in indices.values()])
    values=np.array([changes[v].sum() for v in indices.values()])
    rng=np.random.default_rng(20261122)
    s=rng.integers(len(totals),size=(4000,len(totals)))
    interval=np.percentile(100*values[s].sum(axis=1)/totals[s].sum(axis=1),[2.5,97.5]).tolist()
    common=np.array([(l['error'],r['error']) for l,r in zip(base,new,strict=True)
                     if l['error'] is not None and r['error'] is not None],dtype=float).reshape(-1,2)
    return dict(before=before,after=after,n=len(rows),physical_images=len(totals),
        base_correct=sum(q['success'] for q in base),arm_correct=sum(q['success'] for q in new),
        rescues=int((changes==1).sum()),harms=int((changes==-1).sum()),delta_pp=float(100*changes.mean()),
        ci95_pp=interval,common_parsed=len(common),common_mean_before=float(common[:,0].mean()) if len(common) else None,
        common_mean_after=float(common[:,1].mean()) if len(common) else None)


def hypothesis_diagnostic(rows,category):
    q=[row for row in rows if len(row['geometry']['multi_hypothesis']['alternatives'])>1]
    inside=[]
    spread=[]
    each_count=[]
    at_least_one=[]
    for row in q:
        values=row['geometry']['multi_hypothesis']['alternatives']
        each_count.append(len(values))
        truth=row['reference']
        at_least_one.append(any(score(value,truth,category)['success'] for value in values))
        if category=='direction':
            inside.append(any(clock_error(v,truth)<=1 for v in values))
            spread.append(30*max(clock_error(a,b) for a in values for b in values))
        else:
            inside.append(min(values)<=truth<=max(values))
            spread.append(max(values)-min(values))
    return dict(pairs_with_multiple_candidates=len(q),reference_inside_envelope_or_direction_tolerance=sum(inside),
                typical_range_m_or_deg=float(np.median(spread)) if spread else None,
                offline_has_successful_candidate=sum(at_least_one),
                chosen_candidate_successes=sum(row['scores']['multi_hypothesis']['success'] for row in q),
                native_successes=sum(row['scores']['native']['success'] for row in q),
                median_candidates=float(np.median(each_count)) if each_count else None)


def common_inputs():
    roster=locked_roster()
    native_seal=check_seal(MODEL_SEAL)
    geo_seal=check_seal(GEOMETRY_SEAL)
    assert geo_seal['native_seal']==sha256(RUN/MODEL_SEAL)
    assert geo_seal['code']==sha256(STUDY/'object_pair_r4_geometry.py')
    native=index_rows(RUN/'native.jsonl')
    geometry=index_rows(RUN/'geometry.jsonl')
    assert roster.keys()==native.keys()==geometry.keys()
    assert len(roster)==native_seal['rows']==geo_seal['rows']==483
    return roster,native,geometry


def audit():
    roster,native,geometry=common_inputs()
    ref,exact=references(roster)
    rows=[]
    arm_pairs=[]
    summary={}
    eligible=[]
    expected=read_json(CONFIG)
    for category in CATEGORIES:
        subset=[]
        arms=[a for a in ARMS if (category=='distance_data' or a!='support_nn') and
                                (category!='distance_data' or a!='camera_robust')]
        for q,r in roster.items():
            if r['category']!=category:
                continue
            truth,raw_ground=ref[q]
            answer=parse(native[q]['response'],category)
            raw=geometry[q]['geometry']
            scores={'native':score(answer,truth,category)}
            for arm in arms:
                value=raw[arm]['estimate']
                scores[arm]=score(answer if value is None else value,truth,category)
                if value is not None:
                    assert value>=0 and raw[arm]['evidence'] is not None
            item=dict(id=q,category=category,image=r['image'],source=r['source'],reference=truth,
                      reference_text=raw_ground,native_response=native[q]['response'],
                      geometry=raw,support=geometry[q]['support'],scores=scores)
            subset.append(item)
        availability={}
        comparisons=[]
        for arm in arms:
            covered=[r for r in subset if r['geometry'][arm]['estimate'] is not None]
            availability[arm]=dict(available=len(covered),fallback=len(subset)-len(covered),
                native_when_available=metrics(covered,'native') if covered else None,
                direct_when_available=metrics(covered,arm) if covered else None)
            record=paired(subset,'native',arm)
            clearance=(arm in BOX_ARMS and record['ci95_pp'][0]>0 and
                record['common_mean_after'] is not None and record['common_mean_after']<record['common_mean_before'])
            comparisons.append(dict(category=category,cohort='all',gate_eligible=clearance,**record))
            if covered:
                comparisons.append(dict(category=category,cohort='geometry_available',gate_eligible=False,
                                        **paired(covered,'native',arm)))
            if clearance:
                eligible.append(dict(category=category,arm=arm,**record))
        arm_pairs.extend(comparisons)
        basic={a:metrics(subset,a) for a in ('native',*arms)}
        common_masks=[r for r in subset if all(r['geometry'][arm]['estimate'] is not None
                                                   for arm in ('mask_robust','window_robust'))]
        sources={source:{arm:metrics([row for row in subset if row['source']==source],arm) for arm in ('native',*arms)}
                 for source in sorted({row['source'] for row in subset})}
        summary[category]=dict(metrics=basic,availability=availability,sources=sources,
            scope=dict(rows=len(subset),physical_images=len({r['image'] for r in subset}),
                with_ground_axis=sum(r['support']['up'] is not None for r in subset),
                both_with_regions=sum(min(r['support']['region_counts'])>0 for r in subset),
                same_selected_region=sum(r['support']['region_ids'][0] is not None and
                    r['support']['region_ids'][0]==r['support']['region_ids'][1] for r in subset)),
            mask_vs_window_on_common=paired(common_masks,'window_robust','mask_robust') if common_masks else None,
            hypotheses=hypothesis_diagnostic(subset,category))
        rows.extend(subset)
    predictions={}
    for category in CATEGORIES:
        matching=[r for r in eligible if r['category']==category]
        if matching:
            chosen=min(matching,key=lambda r:(-r['delta_pp'],r['common_mean_after'],r['arm']))
            predictions[category]=chosen['arm']
    jsonl(RUN/'scored_rows.jsonl',rows)
    with (RUN/'paired.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(arm_pairs[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(arm_pairs)
    gate=dict(advance=predictions,rule=expected['gate'],eligible=eligible,labels_used_for_task_level_arm_selection=True,
              model_seal=sha256(RUN/MODEL_SEAL),geo_seal=sha256(RUN/GEOMETRY_SEAL))
    write_json(RUN/'gate.json',gate)
    export=dict(questions=len(rows),official_prompt_parity=exact,summary=summary,paired=arm_pairs,gate=gate,
        native_runtime=read_json(RUN/'native_runtime.json'),geometry_runtime=read_json(RUN/'geometry_runtime.json'),
        scored_rows_sha256=sha256(RUN/'scored_rows.jsonl'),gate_sha256=sha256(RUN/'gate.json'))
    write_json(RUN/'summary.json',export)
    for category,status in summary.items():
        print(category,{k:(v['correct'],v['parsed']) for k,v in status['metrics'].items()},
              {a:v['available'] for a,v in status['availability'].items()},flush=True)
    print('advance',predictions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    audit()
