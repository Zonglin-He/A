"""Post-seal blinded-reader and strict field-recovery diagnostics."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import re
import numpy as np
from common import BASE,HERE,OUT,PRIVATE,digest,inputs,read,write
from predict_span import process
from score import canonical_box,truth_for_slot
from score_span import metric,scores
from score_region import save
from span import TOL,axes

def recover_fields(text):
    # Only intact, unique named fields from the original response. Never
    # interpolate a number or impute a part from ambiguous types.
    decoder=json.JSONDecoder();out={}
    for key in ('part','left','right','other_left','other_right'):
        tails=list(re.finditer(r'"'+key+r'"\s*:\s*',text))
        if len(tails)!=1:
            continue
        try:
            value,_=decoder.raw_decode(text[tails[0].end():])
        except ValueError:
            continue
        if key=='part':
            if value in ('headlamp','taillamp','bumper'):
                out[key]=value
        elif isinstance(value,list) and len(value)==2 and all(type(x) in (float,int) and 0<=x<=1000 for x in value):
            out[key]=value
    if not all(k in out for k in ('part','left','right')) or out['left'][0]>=out['right'][0]:
        return None
    return out|dict(status='pair')

def compare_panel(roster,marks,outputs,frozen,estimates,oracle,tag):
    rows=[]
    for code,review in roster.items():
        uid=review['uid']
        manual=marks[code];reader=outputs[uid]
        observed=read(PRIVATE/'replies'/(uid+'.json'))['parsed']
        crop=np.array(review['crop']); side=crop[2:]-crop[:2]
        r=dict(panel=tag,case=code,uid=uid,recoverable50=oracle[uid]['recoverable50'],
               reader_status=manual['status'],reason=manual.get('reason'),qwen_status=observed['status'],
               reader_part=manual.get('part'),qwen_part=observed.get('part'),
               anchor='physical_anchor' in reader,geometry=reader['status'],extended=reader.get('new_span_m',0)>TOL,
               moved_vs_anchor=abs(reader['offsets']['pair']-reader['offsets']['anchor'])>TOL)
        if manual['status']==observed['status']=='pair':
            err=[float(np.linalg.norm((np.array(manual[k])-observed[k])*side/1000)) for k in ('left','right')]
            r.update(max_mark_distance_px=max(err),both_marks_within_4px=max(err)<=4,
                     same_part=manual['part']==observed['part'])
        rows.append(r)
    return rows

def run():
    panel,frozen=inputs()
    primary=read(PRIVATE/'predictions.json');oracle=read(PRIVATE/'oracle.json')
    assert digest(PRIVATE/'predictions.json')==read(OUT/'prediction_seal.json')['private_sha']
    assert digest(PRIVATE/'mixed_geometry.json')==read(OUT/'mixed_review_seal.json')['private_sha']
    mix=read(PRIVATE/'mixed_geometry.json')
    visual=compare_panel(read(PRIVATE/'review_roster.json'),read(PRIVATE/'independent_marks.json'),
             {k:v['independent'] for k,v in primary.items() if 'independent' in v},frozen,primary,oracle,'recording_balanced_24')
    visual+=compare_panel(read(PRIVATE/'mixed_roster.json'),read(PRIVATE/'mixed_marks.json'),mix,frozen,primary,oracle,'postseal_mixed_65')
    save(OUT/'visual_review.csv',visual)
    jobs=[]
    for f in panel:
        uid=f['uid']
        if primary[uid]['read_status']!='invalid_json':
            continue
        reply=read(PRIVATE/'replies'/(uid+'.json'))
        parsed=recover_fields(reply['answer'])
        if parsed is not None:
            jobs.append((f,frozen[uid],reply|dict(parsed=parsed),None))
    with ProcessPoolExecutor(max_workers=4) as pool:
        extra={j[0]['uid']:v for j,v in zip(jobs,pool.map(process,jobs),strict=True)}
    write(PRIVATE/'recovered_geometry.json',extra)
    reread={k:extra.get(k,v) for k,v in primary.items()}
    arm_rows=[];cases=[]
    for f in panel:
        uid=f['uid'];old=frozen[uid];res=reread[uid];orig=primary[uid]
        r=dict(uid=uid,target=f['target'],frozen=oracle[uid]['baseline'], recovered=0.,primary=0.)
        if old['boxes']['H'] is not None:
            box=np.array(old['boxes']['H']);n,width=axes(box)
            gt=canonical_box(truth_for_slot(f,read(f['calibration'])['ground_info']))
            v,_=scores(box,n,dict(recovered=res['offsets']['pair'],primary=orig['offsets']['pair']),gt)
            r.update(v)
            if any(abs(orig['offsets'][a])>TOL for a in orig['offsets']):
                ng,_=axes(gt)
                delta=np.arccos(np.clip(abs(ng@n),-1,1))
                cases.append(dict(uid=uid,target=f['target'],primary_shift=orig['offsets']['pair'],
                     anchor_shift=orig['offsets']['anchor'],copy_shift=orig['offsets']['copy'],
                     frozen_iou=r['frozen'],pair_iou=r['primary'],axis_error_deg=float(np.rad2deg(delta)),
                     input_width=width,gt_projected_width=float(gt[4]*np.cos(delta)+gt[3]*np.sin(delta)),
                     true_center_offset=float((gt[:3]-box[:3])@n),
                     existing_interval=orig.get('real_interval'),new_interval=orig.get('interval')))
        arm_rows.append(r)
    save(OUT/'changed_cases.csv',cases)
    summary=dict(postseal=True,primary_unmodified=True,new_model_calls=0,
        invalid_json=sum(v['read_status']=='invalid_json' for v in primary.values()),
        additional_intact_pairs=len(extra),oracle_opportunities_recovered_by_field_parser=sum(oracle[k]['recoverable50'] for k in extra),
        additional_anchors=sum('physical_anchor' in v for v in extra.values()),
        additional_feasible=sum(v['status']=='applied_or_already_inside' for v in extra.values()),
        additional_extensions=sum(v.get('new_span_m',0)>TOL for v in extra.values()),
        total_rescue50=sum(r['recovered']>.5 and r['frozen']<=.5 for r in arm_rows),
        total_harm50=sum(r['recovered']<=.5 and r['frozen']>.5 for r in arm_rows),
        outputs_differ_from_primary=sum(abs(r['recovered']-r['primary'])>1.e-9 for r in arm_rows),
        scores=metric(arm_rows,'recovered'))
    for name,g in [('balanced',visual[:24]),('opportunities',[r for r in visual[24:] if r['recoverable50']]),
                   ('additional_controls',[r for r in visual[24:] if not r['recoverable50']])]:
        both=[r for r in g if 'same_part' in r]
        summary[name]=dict(n=len(g),reader_pairs=sum(r['reader_status']=='pair' for r in g),
                unknown_reasons=dict(Counter(r['reason'] for r in g if r['reason'])),
                qwen_pairs=sum(r['qwen_status']=='pair' for r in g),both_pair=len(both),
                both_same_part=sum(r['same_part'] for r in both),
                both_marks_within_4px=sum(r['both_marks_within_4px'] for r in both),
                anchors=sum(r['anchor'] for r in g),geometry=sum(r['geometry']=='applied_or_already_inside' for r in g),
                extended=sum(r['extended'] for r in g),moved_vs_anchor=sum(r['moved_vs_anchor'] for r in g))
    write(OUT/'diagnostics.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    run()
