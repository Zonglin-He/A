"""Clarify clock format without supplying any geometry before R4 continuation decisions."""
import argparse
from pathlib import Path
import time

from download import STUDY, read_json, sha256, write_json
from object_pair_r4 import MODEL_SEAL, RUN, FrozenRunner, check_seal, index_rows, locked_roster
from object_pair_r4_geometry import BOX_ARMS
from object_pair_r4_score import common_inputs, metrics, paired, parse, references, score
from score import jsonl


CONFIG=STUDY/'R4_DIRECTION_FORMAT.json'
SEAL='direction_control_seal.json'
CONTROLS=('format_only','with_frame')


def tasks_for(roster,config):
    return [dict(id=f'{q}/{arm}',image=row['image'],boxes=row['boxes'],
                 prompt=row['prompt']+'\n'+config[arm])
            for q,row in roster.items() if row['category']=='direction' for arm in CONTROLS]


def predict(model_cache):
    if (RUN/SEAL).exists():
        raise FileExistsError(RUN/SEAL)
    locked=locked_roster()
    check_seal(MODEL_SEAL)
    maps=read_json(RUN/'pointmap_manifest.json')
    tasks=tasks_for(locked,read_json(CONFIG))
    assert len(tasks)==214
    for key in {row['image'] for row in tasks}:
        assert sha256(RUN/'pointmaps'/f'{key}.npy')==maps[key]
    jsonl(RUN/'direction_control_inputs.jsonl',tasks)
    started=time.monotonic()
    model=FrozenRunner(model_cache)
    assert model.runtime==read_json(RUN/'native_runtime.json')['model']
    size,agreement,checks=model.parity(tasks)
    print('clock format batch parity',agreement,'/',checks,flush=True)
    outputs=model.infer(tasks,size)
    assert len(outputs)==len(tasks)
    model.assert_frozen()
    jsonl(RUN/'direction_controls.jsonl',[dict(id=r['id'],response=outputs[r['id']]) for r in tasks])
    write_json(RUN/SEAL,dict(rows=len(outputs),labels=False,input_seal=sha256(RUN/'input_seal.json'),
        native_seal=sha256(RUN/MODEL_SEAL),config=sha256(CONFIG),stage_seconds=time.monotonic()-started,
        parity_equal=agreement,parity_compared=checks,batch_size=size,
        files={f:sha256(RUN/f) for f in ('direction_controls.jsonl','direction_control_inputs.jsonl')}))


def audit():
    seal=check_seal(SEAL)
    roster,native,geo=common_inputs()
    assert seal['config']==sha256(CONFIG) and seal['native_seal']==sha256(RUN/MODEL_SEAL)
    refs,exact=references(roster)
    assert exact==483
    c=index_rows(RUN/'direction_controls.jsonl')
    rows=[]
    arms=[a for a in BOX_ARMS if a!='support_nn']+['camera_robust','mask_robust']
    for q,row in roster.items():
        if row['category']!='direction':
            continue
        v=parse(native[q]['response'],'direction')
        truth=refs[q][0]
        outcomes={'native_verbatim':score(v,truth,'direction')}
        for cond in CONTROLS:
            v=parse(c[f'{q}/{cond}']['response'],'direction')
            outcomes[cond]=score(v,truth,'direction')
            for arm in arms:
                measurement=geo[q]['geometry'][arm]['estimate']
                outcomes[f'{arm}__{cond}']=score(v if measurement is None else measurement,truth,'direction')
        rows.append(dict(id=q,category='direction',image=row['image'],source=row['source'],scores=outcomes))
    assert len(rows)==107 and 2*len(rows)==len(c)
    results={}
    eligible=[]
    for arm in arms:
        outcomes={cond:paired(rows,cond,f'{arm}__{cond}') for cond in CONTROLS}
        results[arm]=outcomes
        if arm in BOX_ARMS and all(o['ci95_pp'][0]>0 and o['common_mean_after'] is not None and
                                   o['common_mean_after']<o['common_mean_before'] for o in outcomes.values()):
            eligible.append(arm)
    selected=min(eligible,key=lambda arm:(-results[arm]['with_frame']['delta_pp'],
                                          results[arm]['with_frame']['common_mean_after'],arm)) if eligible else None
    jsonl(RUN/'direction_scored_rows.jsonl',rows)
    export=dict(metrics={arm:metrics(rows,arm) for arm in rows[0]['scores']},paired=results,
                eligible=eligible,advance_arm=selected,rule=read_json(CONFIG)['decision'],
                seal_sha256=sha256(RUN/SEAL),scored_rows_sha256=sha256(RUN/'direction_scored_rows.jsonl'))
    write_json(RUN/'direction_summary.json',export)
    print({a:(m['correct'],m['parsed']) for a,m in export['metrics'].items()},flush=True)
    print('direction advance',selected,flush=True)


def audit_two_dimensional_controls():
    check_seal(SEAL)
    roster,_,geometry=common_inputs()
    controls=index_rows(RUN/'direction_scored_rows.jsonl')
    summary=read_json(RUN/'direction_summary.json')
    assert sha256(RUN/'direction_scored_rows.jsonl')==summary['scored_rows_sha256']

    def lateral_hour(dx):
        return 3 if dx>0 else 9 if dx<0 else None

    raw={}
    for q,item in roster.items():
        if item['category']!='direction':
            continue
        base=geometry[q]['geometry']['window_robust']
        a,b=item['boxes'][:2]
        dx=b[0]+b[2]-a[0]-a[2]
        evidence=base['evidence']
        lateral=lateral_hour(evidence[1][0]-evidence[0][0]) if evidence is not None else None
        raw[q]=dict(bbox_lateral=lateral_hour(dx),zero_forward=lateral,covered=base['estimate'] is not None)
    out=RUN/'direction_2d_offline_readouts.json'
    write_json(out,raw)
    truth,_=references(roster)
    rows=[]
    for q,item in raw.items():
        fallback=controls[q]['scores']['with_frame']
        values=dict(frame=fallback,direct=controls[q]['scores']['window_robust__with_frame'])
        for key in ('bbox_lateral','zero_forward'):
            estimate=item[key]
            values[key]=score(estimate,truth[q][0],'direction') if estimate is not None else fallback
        rows.append(dict(image=roster[q]['image'],scores=values,covered=item['covered']))
    covered=[r for r in rows if r['covered']]
    paired_rows=[dict(scope=k,**paired(active,before,'direct'))
                 for k,active in (('all',rows),('same_covered',covered))
                 for before in ('bbox_lateral','zero_forward')]
    export=dict(posthoc_diagnostic=True,
        descriptions={'bbox_lateral':'Clock 3 or 9 using the sign of the annotated 2D box center difference only',
                      'zero_forward':'Same geometry and coverage as the window readout, set relative forward coordinate to zero'},
        metrics={a:metrics(rows,a) for a in rows[0]['scores']},
        covered_metrics={a:metrics(covered,a) for a in covered[0]['scores']},
        paired=paired_rows,predictions_sha256=sha256(out),direction_format_seal=sha256(RUN/SEAL))
    write_json(RUN/'direction_2d_control.json',export)
    print({a:m['correct'] for a,m in export['covered_metrics'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','audit','audit-2d'))
    p.add_argument('--model-cache',type=Path)
    a=p.parse_args()
    if a.action=='predict':
        if a.model_cache is None:
            p.error('--model-cache is required for predict')
        predict(a.model_cache)
    elif a.action=='audit':
        audit()
    else:
        audit_two_dimensional_controls()
