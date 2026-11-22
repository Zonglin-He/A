"""Test whether frozen HiSpatial uses qualifying object-pair direction evidence."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from download import STUDY, read_json, sha256, write_json
from object_pair_r4 import GEOMETRY_SEAL, MODEL_SEAL, RUN, FrozenRunner, check_seal, index_rows, locked_roster
from object_pair_r4_direction import CONFIG as CLOCK_CONFIG, SEAL as CLOCK_SEAL
from object_pair_r4_geometry import UP, clock_error, relationship
from object_pair_r4_score import common_inputs, metrics, paired, parse, references, score
from score import jsonl


CONFIG=STUDY/'R4_DIRECTION_CONDITIONING.json'
SEAL='direction_condition_seal.json'
ARMS=('positions','positions_swapped','offset','offset_swapped')


def clock(pair):
    v=np.insert(np.asarray(pair,dtype=np.float64),1,0,axis=1)
    return relationship(v[0],v[1],'direction',UP)[0]


def reader_inputs(roster,geometry,arm,frame_text):
    jobs=[]
    allocation=[]
    for q,row in roster.items():
        if row['category']!='direction':
            continue
        base=geometry[q]['geometry'][arm]
        assignment=dict(id=q,direct_unrounded=base['estimate'],direct_prompt={})
        allocation.append(assignment)
        if base['estimate'] is None:
            continue
        coords=np.asarray(base['evidence'],dtype=np.float64)
        assert coords.shape==(2,2)
        delta=np.round(coords[1]-coords[0],4)
        coords=np.round(coords,4)
        covers={'positions':coords,'positions_swapped':coords[::-1],
                'offset':np.stack((np.zeros(2),delta)),'offset_swapped':np.stack((np.zeros(2),-delta))}
        if any(clock(pair) is None for pair in covers.values()):
            continue
        for mode,points in covers.items():
            if mode.startswith('positions'):
                a,b=points
                note=(f'Estimated (rightward, forward) coordinates in meters. '
                      f'Red region ({a[0]:.4f}, {a[1]:.4f}). Blue region ({b[0]:.4f}, {b[1]:.4f}).')
            else:
                x,y=points[1]
                note=f'Estimated (rightward, forward) offset from the red region to the blue region, in meters ({x:.4f}, {y:.4f}).'
            assignment['direct_prompt'][mode]=clock(points)
            jobs.append(dict(id=f'{q}/{mode}',image=row['image'],boxes=row['boxes'],
                prompt=row['prompt']+'\n'+note+'\n'+frame_text))
    return jobs,allocation


def eligibility():
    grouped=read_json(RUN/'direction_summary.json')
    clock_check=check_seal(CLOCK_SEAL)
    assert grouped['seal_sha256']==sha256(RUN/CLOCK_SEAL) and clock_check['config']==sha256(CLOCK_CONFIG)
    arm=grouped['advance_arm']
    if arm is None:
        raise ValueError('No direction arm qualified')
    return arm


def predict(cache):
    if (RUN/SEAL).exists():
        raise FileExistsError(RUN/SEAL)
    selected=eligibility()
    inputs=locked_roster()
    check_seal(GEOMETRY_SEAL)
    check_seal(MODEL_SEAL)
    geometry=index_rows(RUN/'geometry.jsonl')
    frame=read_json(CLOCK_CONFIG)['with_frame']
    tasks,assignment=reader_inputs(inputs,geometry,selected,frame)
    assert len(assignment)==107 and len(tasks)==4*sum(bool(v['direct_prompt']) for v in assignment)
    source=read_json(RUN/'pointmap_manifest.json')
    for key in {row['image'] for row in tasks}:
        assert sha256(RUN/'pointmaps'/f'{key}.npy')==source[key]
    jsonl(RUN/'direction_condition_inputs.jsonl',tasks)
    jsonl(RUN/'direction_condition_assignment.jsonl',assignment)
    started=time.monotonic()
    model=FrozenRunner(cache)
    assert model.runtime==read_json(RUN/'native_runtime.json')['model']
    size,agreement,n=model.parity(tasks)
    print('conditional batch parity',agreement,'/',n,flush=True)
    answers=model.infer(tasks,size)
    assert len(answers)==len(tasks)
    model.assert_frozen()
    jsonl(RUN/'direction_condition_responses.jsonl',[dict(id=t['id'],response=answers[t['id']]) for t in tasks])
    write_json(RUN/SEAL,dict(input_seal=sha256(RUN/'input_seal.json'),labels=False,config=sha256(CONFIG),
        geometry_seal=sha256(RUN/GEOMETRY_SEAL),clock_seal=sha256(RUN/CLOCK_SEAL),
        gate_summary_sha256=sha256(RUN/'direction_summary.json'),arm=selected,
        calls=len(tasks),answered=len(answers),coverage=len(tasks)//4,task_seconds=time.monotonic()-started,
        parity_equal=agreement,parity_compared=n,batch_size=size,peak_cuda_bytes=model.torch.cuda.max_memory_allocated(),
        files={f:sha256(RUN/f) for f in ('direction_condition_inputs.jsonl','direction_condition_assignment.jsonl',
                                          'direction_condition_responses.jsonl')}))
    print('sealed conditioned outputs',len(answers),flush=True)


def audit():
    seal=check_seal(SEAL)
    assert seal['config']==sha256(CONFIG) and seal['gate_summary_sha256']==sha256(RUN/'direction_summary.json')
    assert seal['geometry_seal']==sha256(RUN/GEOMETRY_SEAL) and seal['clock_seal']==sha256(RUN/CLOCK_SEAL)
    roster,_,_=common_inputs()
    references_by_id,official_exact=references(roster)
    assert official_exact==483
    assignment=index_rows(RUN/'direction_condition_assignment.jsonl')
    responses=index_rows(RUN/'direction_condition_responses.jsonl')
    control=index_rows(RUN/'direction_controls.jsonl')
    rows=[]
    behavior={}
    for arm in ARMS:
        behavior[arm]=Counter()
    swap_behavior={style:Counter() for style in ('positions','offset')}
    for q,di in assignment.items():
        item=roster[q]
        truth=references_by_id[q][0]
        fallback=control[f'{q}/with_frame']['response']
        native=parse(fallback,'direction')
        base={'native_frame':score(native,truth,'direction')}
        base['direct_unrounded']=score(di['direct_unrounded'] if di['direct_unrounded'] is not None else native,truth,'direction')
        seen=bool(di['direct_prompt'])
        for mode in ARMS:
            record=responses[f'{q}/{mode}']['response'] if seen else fallback
            hour=parse(record,'direction')
            base[mode]=score(hour,truth,'direction')
            if not seen:
                if mode in ('positions','offset'):
                    base[f'direct_{mode}']=base['native_frame']
                continue
            geometry=di['direct_prompt'][mode]
            count=behavior[mode]
            count['present']+=1
            count['parsed']+=hour is not None
            count['agrees_geometry_within_hour']+=hour is not None and clock_error(hour,geometry)<=1
            count['agrees_geometry_exactly']+=hour==geometry
            if mode in ('positions','offset'):
                base[f'direct_{mode}']=score(geometry,truth,'direction')
                count['rounded_hour_changed']+=geometry!=di['direct_unrounded']
        if seen:
            for style,c in swap_behavior.items():
                a,b=base[style]['estimate'],base[style+'_swapped']['estimate']
                if a is not None and b is not None:
                    c['parsed_pairs']+=1
                    c['within_one_hour_of_180_degree_change']+=clock_error(a,b)>=5
                    c['same_output_hour']+=a==b
        rows.append(dict(id=q,category='direction',image=item['image'],source=item['source'],scores=base,geometry_used=seen))
    assert len(rows)==107 and len(responses)==4*sum(r['geometry_used'] for r in rows)==seal['answered']
    active=[r for r in rows if r['geometry_used']]
    comparisons=[('native_frame',arm) for arm in ARMS]+[
        ('native_frame','direct_unrounded'),('positions_swapped','positions'),('offset_swapped','offset'),
        ('direct_positions','positions'),('direct_offset','offset'),('positions','offset')]
    export=dict(arm=seal['arm'],scope=len(rows),qualified_measurement=len(active),
                metrics={name:metrics(rows,name) for name in rows[0]['scores']},
                active_metrics={name:metrics(active,name) for name in rows[0]['scores']},
                paired=[dict(scope=name,**paired(group,a,b)) for name,group in (('all',rows),('geometry_used',active))
                         for a,b in comparisons],
                interface_agreement={k:dict(v) for k,v in behavior.items()},
                response_to_counterfactual_swap={k:dict(v) for k,v in swap_behavior.items()},
                execution=seal)
    jsonl(RUN/'direction_condition_scored_rows.jsonl',rows)
    export['scored_rows_sha256']=sha256(RUN/'direction_condition_scored_rows.jsonl')
    write_json(RUN/'direction_condition_summary.json',export)
    print({k:(v['correct'],v['parsed']) for k,v in export['metrics'].items()},flush=True)
    print('counterfactual response',export['response_to_counterfactual_swap'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('predict','audit'))
    parser.add_argument('--model-cache',type=Path)
    args=parser.parse_args()
    if args.action=='predict':
        if args.model_cache is None:
            parser.error('--model-cache is required for predict')
        predict(args.model_cache)
    else:
        audit()
