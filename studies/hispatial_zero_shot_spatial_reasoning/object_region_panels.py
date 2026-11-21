"""Diagnose marker vocabulary effects with groups of up to three proposals and rejection."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from download import read_json, sha256, write_json
from geometry_cohort import read_jsonl, stable
from object_region_plan import CONFIG, DIAG, MARKS, RUN, index_rows, read_rgb, render_regions
from score import jsonl


MICROPANEL = {
    'roster': 'The same 96 EmbSpatial questions reserved for marker reversal before predictions',
    'reason': 'The first 443 raw full-overlay generations contained zero choices from E through H before any scoring or labels',
    'groups': 'Consecutive disjoint groups of at most three from the same marked proposal list',
    'rounds': 'One single-letter query per group with a no-match option. If multiple proposals survive, show them in one final grouped query.',
    'shared': 'Frozen HiSpatial weights, pointmap values, source RGB, object names, region supports and marker anchor pixels',
    'annotations': False,
}


def predict(model_cache):
    from functools import lru_cache
    import cv2
    import torch
    from geometry_audit import select_letter
    from object_region_infer import concrete, mark_prompt
    from predict import BATCH_N, batch_query, cached_single, feature, release

    assert not (RUN/'panel_predictions_seal.json').exists()
    cv2.setNumThreads(1)
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    i=read_json(RUN/'input_seal.json')
    assert i['label_access'] is False and i['protocol']==sha256(CONFIG)
    for file,digest in i['files'].items():
        assert sha256(RUN/file)==digest
    roster={k:r for k,r in index_rows(RUN/'roster.jsonl').items() if r['order_control']}
    assert len(roster)==96
    regions=read_json(RUN/'regions.json')
    assets=read_json(RUN/'views_sha256.json')
    frames=read_json(DIAG/'frames.json')
    refs={}
    first=[]
    for q in roster.values():
        options=[(j,name) for j,name in enumerate(q['options']) if concrete(name)]
        available=tuple(range(len(regions[q['key']]['marked'])))
        groups=[available[k:k+3] for k in range(0,len(available),3)]
        for j,name in options:
            key=q['key']
            ids=[]
            for k,group in enumerate(groups):
                pid=f"{q['id']}/{j}/{k}"
                first.append(dict(id=pid,key=key,shown=list(group),prompt=mark_prompt(name,len(group))))
                ids.append(pid)
            refs[q['id'],j]=dict(name=name,key=key,ids=ids)
    write_json(RUN/'panel_protocol.json',MICROPANEL)
    jsonl(RUN/'panel_first_inputs.jsonl',first)
    write_json(RUN/'panel_input_seal.json',dict(inputs=sha256(RUN/'panel_first_inputs.jsonl'),
        protocol=sha256(RUN/'panel_protocol.json'),parent=sha256(RUN/'input_seal.json'),labels=False))
    model,_,runtime=release(model_cache)
    assert runtime==read_json(RUN/'runtime.json')['runtime']

    @lru_cache(maxsize=8)
    def load(key):
        rgb,_=read_rgb(frames[key])
        xyzfile=DIAG/'pointmaps'/f'{key}.npy'
        assert sha256(xyzfile)==regions[key]['xyz_sha256']
        xyz=torch.from_numpy(np.load(xyzfile,allow_pickle=False))
        packed=RUN/'views'/f'{key}.support.npy'
        assert sha256(packed)==assets[key]['supports']
        packed=np.load(packed,allow_pickle=False)
        masks=np.unpackbits(packed,axis=1,count=448*448).astype(bool).reshape(len(packed),448,448)
        return rgb,xyz,masks

    @lru_cache(maxsize=12)
    def prepare(key,ids):
        rgb,xyz,masks=load(key)
        anchors=[tuple(regions[key]['marked'][j]['anchor']) for j in ids]
        image=render_regions(rgb,masks[list(ids)],MARKS[:len(ids)],anchors)
        return image,xyz,feature(model,image,xyz)

    def infer(tasks,path,size):
        ordered=sorted(tasks,key=lambda r:(r['key'],r['shown'],r['id']))
        responses={}
        start=time.monotonic()
        next_log=100
        for k in range(0,len(ordered),size):
            block=ordered[k:k+size]
            prepared=[prepare(r['key'],tuple(r['shown'])) for r in block]
            raw=batch_query(model,prepared,block) if len(block)>1 else [
                cached_single(model,prepared[0],block[0]['prompt'])]
            for r,text in zip(block,raw,strict=True):
                letter=select_letter(text,MARKS[:len(r['shown'])+1])
                selection=r['shown'][MARKS.index(letter)] if letter is not None and letter in MARKS[:len(r['shown'])] else None
                responses[r['id']]=dict(id=r['id'],task=r['shown'],text=text,selection=selection)
            if len(responses)>=next_log or len(responses)==len(ordered):
                print(path,len(responses),'/',len(ordered),round(time.monotonic()-start,1),'s',flush=True)
                next_log+=100
        jsonl(RUN/path,list(responses.values()))
        return responses

    check=sorted(first,key=lambda r:stable('panel_parity/'+r['id']))[:16]
    parity=0
    for j in range(0,len(check),BATCH_N):
        block=check[j:j+BATCH_N]
        p=[prepare(r['key'],tuple(r['shown'])) for r in block]
        single=[cached_single(model,v,r['prompt']) for v,r in zip(p,block,strict=True)]
        batched=batch_query(model,p,block)
        parity+=sum(x==y for x,y in zip(single,batched,strict=True))
    batch=BATCH_N if parity==len(check) else 1
    print('grouped mark parity',parity,'/',len(check), 'batch',batch,flush=True)
    early=infer(first,'panel_first_predictions.jsonl',batch)
    final=[]
    outcomes=defaultdict(dict)
    for (question,option),ref in refs.items():
        passed=[early[i]['selection'] for i in ref['ids'] if early[i]['selection'] is not None]
        result={'passed':passed,'selected':passed[0] if len(passed)==1 else None}
        outcomes[question][str(option)]=result
        if len(passed)>1:
            final.append(dict(id=f'{question}/{option}/final',key=ref['key'],shown=passed,
                              prompt=mark_prompt(ref['name'],len(passed)),question=question,option=option))
    jsonl(RUN/'panel_final_inputs.jsonl',final)
    late=infer(final,'panel_final_predictions.jsonl',batch)
    for f in final:
        outcomes[f['question']][str(f['option'])]['selected']=late[f['id']]['selection']
    output=[dict(id=k,bindings=v) for k,v in outcomes.items()]
    jsonl(RUN/'panel_predictions.jsonl',output)
    write_json(RUN/'panel_predictions_seal.json',dict(
        inputs=sha256(RUN/'panel_input_seal.json'),labels=False,question_rows=len(output),batch_anchor_matches=parity,
        inference_rows=len(early)+len(late),first_rows=len(early),final_rows=len(late),
        files={name:sha256(RUN/name) for name in ('panel_first_inputs.jsonl','panel_first_predictions.jsonl',
               'panel_final_inputs.jsonl','panel_final_predictions.jsonl','panel_predictions.jsonl')}))


def audit():
    from object_region_audit import ious, option_box

    seal=read_json(RUN/'panel_predictions_seal.json')
    main=read_json(RUN/'summary.json')
    for file,digest in seal['files'].items():
        assert sha256(RUN/file)==digest
    assert sha256(RUN/'panel_input_seal.json')==seal['inputs'] and not seal['labels']
    assert main['artifacts']['scored_bindings.jsonl']==sha256(RUN/'scored_bindings.jsonl')
    diag=read_json(DIAG/'audit_summary.json')
    assert sha256(DIAG/'emb_box_depth.jsonl')==diag['result_files']['emb_box_depth.jsonl']
    roster=index_rows(RUN/'roster.jsonl')
    predictions=index_rows(RUN/'panel_predictions.jsonl')
    geo=index_rows(DIAG/'emb_box_depth.jsonl')
    frames=read_json(RUN/'regions.json')
    previous={(r['id'],str(r['option']-1)):r for r in read_jsonl(RUN/'scored_bindings.jsonl')
              if r['concrete'] and r['annotated'] and r['id'] in predictions}
    counts=Counter()
    details=[]
    for q,p in predictions.items():
        frame=frames[roster[q]['key']]
        boxes=[m['box'] for m in frame['marked']]
        g=geo[q]
        per=Counter()
        assigned=set()
        for option,result in p['bindings'].items():
            o=int(option)
            target=option_box(g['boxes'][o],g['size'])
            if target is None:
                continue
            iou=ious(target,boxes)
            reachable=bool(any(iou>=.25))
            initial=previous[q,option]['selected_iou']
            shortlist=max((iou[i] for i in result['passed']),default=0.0)
            j=result['selected']
            picked=iou[j] if j is not None else 0.0
            counts['evaluated_options']+=1
            counts['available25']+=reachable
            counts['eight_overlay_hit25']+=initial>=.25
            counts['grouped_shortlist_hit25']+=shortlist>=.25
            counts['grouped_final_hit25']+=picked>=.25
            counts['available50']+=bool(any(iou>=.50))
            counts['eight_overlay_hit50']+=initial>=.50
            counts['grouped_final_hit50']+=picked>=.50
            counts['rescued25']+=initial<.25<=picked
            counts['lost25']+=picked<.25<=initial
            per['num']+=1
            per['chosen']+=j is not None
            per['hit']+=picked>=.25
            if j is not None:
                assigned.add(j)
            details.append(dict(id=q,option=o+1,reachable=reachable,wide_overlay_iou=initial,
                                grouped_shortlist_iou=float(shortlist),grouped_selected_iou=float(picked),
                                chosen_proposal_index=j))
        counts['questions_bound_every_option']+=0<per['num']==per['chosen']
        counts['questions_distinct_every_option']+=0<per['num']==len(assigned)
        counts['questions_correct_every_option25']+=0<per['num']==per['hit']
    jsonl(RUN/'panel_scored_bindings.jsonl',details)
    write_json(RUN/'panel_summary.json',dict(questions=len(predictions),counts={k:int(v) for k,v in counts.items()},
        same_candidate_pool=True,prediction_seal=sha256(RUN/'panel_predictions_seal.json'),
        scored_bindings=sha256(RUN/'panel_scored_bindings.jsonl')))
    print(dict(counts),flush=True)


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
