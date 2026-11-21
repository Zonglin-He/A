"""Score sealed physical binding and depth-readout controls with offline labels."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from download import DATA, read_json, sha256, write_json
from geometry_audit import paired_stats
from object_region_plan import BASELINE, CONFIG, DIAG, RUN, index_rows
import score


METHODS = ('original_hispatial','whole_map_depth','window_depth','region_depth')
THRESHOLDS = (0.25,0.50)


def replay_official(official_source, roster, predictions):
    where = BASELINE / 'embspatial'
    seals = read_json(where/'predictions_seal.json')
    original_path, prompts_path = where/'predictions.jsonl',where/'inputs.jsonl'
    assert sha256(original_path)==seals['predictions_sha256'] and sha256(prompts_path)==seals['inputs_sha256']
    source_file = DATA / 'embspatial' / 'source.json'
    assert sha256(source_file)==read_json(DIAG/'inputs_seal.json')['baseline']['embspatial']['dataset_source']
    for f in read_json(source_file)['files']:
        assert sha256(source_file.parent / f['file'])==f['sha256']
    original = index_rows(original_path)
    prompts = {i:r['prompt'] for i,r in index_rows(prompts_path).items()}
    full_ids = {i.removeprefix('embspatial/'):i for i in roster}
    assert all(prompts[old]==roster[new]['prompt'] for old,new in full_ids.items())
    module,digest = score.import_official('embspatial',official_source)
    records = {}
    for method in METHODS:
        patched = {**original,**{old:dict(response=predictions[i]['answers'][method]) for old,i in full_ids.items()}}
        out=RUN/'official_replay'/method
        (out/'official_replay').mkdir(parents=True,exist_ok=True,mode=0o700)
        rows,_,metrics=score.score_emb(module,out,patched,prompts)
        write_json(out/'official_metrics.json',metrics)
        focus = {full_ids[r['id']]:r for r in rows if r['id'] in full_ids}
        assert focus.keys() == roster.keys()
        records[method] = focus
    assert sum(x['correct'] for x in records['original_hispatial'].values())==786
    return records,digest


def option_box(raw, size):
    if raw is None:
        return None
    x,y,w,h = raw
    W,H = size
    a = np.clip([448*x/W,448*y/H,448*(x+w)/W,448*(y+h)/H],0,448)
    return a if (a[2:]>a[:2]).all() else None


def ious(true_box, predictions):
    if not predictions:
        return np.empty(0,dtype=np.float64)
    boxes=np.asarray(predictions,dtype=np.float64)
    minimum=np.maximum(boxes[:,:2],true_box[:2])
    maximum=np.minimum(boxes[:,2:],true_box[2:])
    hit=np.maximum(maximum-minimum,0).prod(axis=1)
    area=(boxes[:,2:]-boxes[:,:2]).prod(axis=1)
    ref=(true_box[2:]-true_box[:2]).prod()
    return hit/(area+ref-hit)


def bound_mask_share(mask, box):
    ys,xs=np.nonzero(mask)
    if not len(xs):
        return None
    ok=(xs+.5>=box[0]) & (xs+.5<box[2]) & (ys+.5>=box[1]) & (ys+.5<box[3])
    return float(ok.mean())


def all_distinct_supported(matrix, threshold):
    if matrix.shape[0]>matrix.shape[1]:
        return False
    valid = matrix>=threshold
    a,b=linear_sum_assignment(valid,maximize=True)
    return len(a)==matrix.shape[0] and bool(valid[a,b].all())


def evaluate(roster,regions,predictions,records,geo,assets):
    @lru_cache(maxsize=4)
    def masks(key):
        path=RUN/'views'/f'{key}.support.npy'
        assert sha256(path)==assets[key]['supports']
        packed=np.load(path,allow_pickle=False)
        return np.unpackbits(packed,axis=1,count=448*448).astype(bool).reshape(len(packed),448,448)

    scored=[]
    by_object=[]
    for q,item in roster.items():
        g=geo[q]
        assert g['options']==item['options']
        key=item['key']
        frame=regions[key]
        assert frame['size']==g['size']
        picked=frame['marked']
        marked_boxes={r['id']:r['box'] for r in picked}
        bank_boxes=[marked_boxes.get(r['id'],r['box']) for r in frame['bank']]
        ids=[r['id'] for r in picked]
        previous=records['original_hispatial'][q]
        gold=previous['official_answer']
        assert gold==g['official_answer'] and previous['correct']==g['aligned_correct']
        targets=[option_box(r,frame['size']) for r in g['boxes']]
        result=predictions[q]
        assert result['key']==key
        binding=result['bindings']['marks']
        reverse=result['bindings'].get('reversed')
        selected_ids=[b['region'] for b in binding if b['status']!='abstract_option']
        expected=len([b for b in binding if b['status']!='abstract_option'])
        assert expected>=3
        depths_valid=all(i is not None for i in selected_ids)
        distinct=len(set(selected_ids))==expected
        if result['decision']=='complete':
            assert depths_valid and distinct
        else:
            assert all(result['answers'][name]==result['answers']['original_hispatial']
                       for name in ('region_depth','window_depth'))
        available=[]
        available_full=[]
        chosen=[]
        geometry_correct=(g['bbox_moge_prediction'] == g['bbox_center_prediction'] == gold)
        for n,(target,b) in enumerate(zip(targets,binding,strict=True)):
            is_concrete=b['status']!='abstract_option'
            picked_id=b['region']
            left=dict(id=q,option=n+1,name=b['option'],topic=item['topic'],source=g['source'],
                      annotated=target is not None,concrete=is_concrete,gold='ABCD'[n]==gold,
                      baseline_correct=previous['correct'],selected=picked_id,
                      binding_status=b['status'],decoded_letter=b['letter'],reply=b['response'])
            if reverse:
                other=reverse[n]
                left.update(reverse_selected=other['region'],reverse_letter=other['letter'],reverse_status=other['status'])
            if is_concrete and target is not None:
                full=ious(target,bank_boxes)
                small=ious(target,list(marked_boxes.values()))
                index=ids.index(picked_id) if picked_id is not None else None
                selected_iou=float(small[index]) if index is not None else 0.0
                sub=[small[ids.index(j)] if j is not None else 0.0 for j in selected_ids]
                rev=reverse[n]['region'] if reverse else None
                rev_iou=float(small[ids.index(rev)]) if rev is not None else 0.0
                roi=masks(key)[index] if index is not None else None
                left.update(full_iou=float(full.max(initial=0)),marked_iou=float(small.max(initial=0)),
                            selected_iou=selected_iou,rank_only_iou=float(small[0]) if len(small) else 0.0,
                            reverse_iou=rev_iou if reverse else None,
                            selected_support_in_gt=bound_mask_share(roi,target) if roi is not None else None,
                            uniform_hit25=float((small>=.25).mean()) if len(small) else 0.0,
                            shuffled_binding_hit25=float(np.mean(np.array(sub)>=.25)) if sub else 0.0)
                available.append(small)
                available_full.append(full)
                chosen.append(selected_iou)
            by_object.append(left)
        enough=len(available)==expected
        pair=dict(id=q,topic=item['topic'],source=g['source'],scene=g['scene'],gt=gold,status=result['decision'],
                  original_wrong_and_known_depth_correct=bool(not previous['correct'] and geometry_correct),
                  conditions={m:dict(correct=records[m][q]['correct'],response=result['answers'][m]) for m in METHODS})
        for t in THRESHOLDS:
            tag=f'{int(t*100):02}'
            pair['all_covered_marked_'+tag]=bool(enough and all_distinct_supported(np.asarray(available),t))
            pair['all_covered_full_'+tag]=bool(enough and all_distinct_supported(np.asarray(available_full),t))
            pair['all_bound_correct_'+tag]=bool(enough and distinct and all(s>=t for s in chosen))
        scored.append(pair)
    return scored,by_object


def object_summary(obj):
    summary={}
    for mode,items in [('all_options',obj),('gold_option',[o for o in obj if o['gold']]),
                       ('baseline_wrong',[o for o in obj if not o['baseline_correct']]),
                       ('baseline_right',[o for o in obj if o['baseline_correct']])]:
        eligible=[o for o in items if o['concrete'] and o['annotated']]
        focus=dict(options=len(items),annotated_concrete=len(eligible),
                   status=dict(Counter(r['binding_status'] for r in items)),
                   uniform_expected_hits25=sum(o['uniform_hit25'] for o in eligible),
                   shuffled_expected_hits25=sum(o['shuffled_binding_hit25'] for o in eligible),
                   selected_letter_counts=dict(Counter(o['decoded_letter'] for o in eligible if o['decoded_letter'] is not None)))
        for t in THRESHOLDS:
            tag=f'{int(t*100):02}'
            focus['at_'+tag]={
                'bank_reachable':sum(o['full_iou']>=t for o in eligible),
                'marked_reachable':sum(o['marked_iou']>=t for o in eligible),
                'selected_hit':sum(o['selected_iou']>=t for o in eligible),
                'rank_only_hit':sum(o['rank_only_iou']>=t for o in eligible),
                'bank_miss':sum(o['full_iou']<t for o in eligible),
                'displayed_pool_miss':sum(o['full_iou']>=t and o['marked_iou']<t for o in eligible),
                'semantic_miss_when_marked_reachable':sum(o['marked_iou']>=t and o['selected_iou']<t for o in eligible)}
        summary[mode]=focus
    reverse=[o for o in obj if 'reverse_selected' in o and o['concrete']]
    both=[o for o in reverse if o['selected'] is not None and o['reverse_selected'] is not None]
    summary['marker_reversal']=dict(concrete_options=len(reverse),both_bound=len(both),
        same_region=sum(o['selected']==o['reverse_selected'] for o in both),
        same_literal_letter=sum(o['decoded_letter']==o['reverse_letter'] for o in both),
        primary_hit25=sum(o['selected_iou']>=.25 for o in reverse if o['annotated']),
        reverse_hit25=sum(o['reverse_iou']>=.25 for o in reverse if o['annotated']))
    return summary


def question_summary(rows):
    s={}
    for mode,items in [('all',rows),('close',[r for r in rows if r['topic']=='close']),
                       ('far',[r for r in rows if r['topic']=='far']),
                       ('baseline_right',[r for r in rows if r['conditions']['original_hispatial']['correct']]),
                       ('baseline_wrong',[r for r in rows if not r['conditions']['original_hispatial']['correct']]),
                       ('baseline_wrong_oracle_depth_correct',[r for r in rows if r['original_wrong_and_known_depth_correct']])]:
        s[mode]=dict(n=len(items),answer_correct={m:sum(r['conditions'][m]['correct'] for r in items) for m in METHODS},
                     readout_status=dict(Counter(r['status'] for r in items)),
                     distinct_marked_support25=sum(r['all_covered_marked_25'] for r in items),
                     distinct_bank_support25=sum(r['all_covered_full_25'] for r in items),
                     all_named_objects_bound25=sum(r['all_bound_correct_25'] for r in items),
                     distinct_marked_support50=sum(r['all_covered_marked_50'] for r in items),
                     distinct_bank_support50=sum(r['all_covered_full_50'] for r in items),
                     all_named_objects_bound50=sum(r['all_bound_correct_50'] for r in items))
    return s


def run(source):
    seal=read_json(RUN/'prediction_seal.json')
    input_seal=read_json(RUN/'input_seal.json')
    assert seal['inputs']==sha256(RUN/'input_seal.json') and input_seal['protocol']==sha256(CONFIG)
    assert seal['label_access'] is False and seal['training_steps']==0 and input_seal['label_access'] is False
    assert sha256(DIAG/'predictions_seal.json')==input_seal['diagnosis_predictions']
    assert sha256(DIAG/'inputs_seal.json')==input_seal['diagnosis_input']
    for s in (seal,input_seal):
        for f,d in s['files'].items():
            assert sha256(RUN/f)==d
    roster=index_rows(RUN/'roster.jsonl')
    predictions=index_rows(RUN/'predictions.jsonl')
    assert len(roster)==len(predictions)==seal['rows']==input_seal['rows']==1206
    assert roster.keys()==predictions.keys()
    regions=read_json(RUN/'regions.json')
    assets=read_json(RUN/'views_sha256.json')
    records,source_hash=replay_official(source,roster,predictions)
    diagnostic=read_json(DIAG/'audit_summary.json')
    assert diagnostic['result_files']['emb_box_depth.jsonl']==sha256(DIAG/'emb_box_depth.jsonl')
    geo=index_rows(DIAG/'emb_box_depth.jsonl')
    scored,bind=evaluate(roster,regions,predictions,records,geo,assets)
    paired=[]
    for subset in ('all','close','far'):
        items=[r for r in scored if subset=='all' or r['topic']==subset]
        for left,right in list(zip([METHODS[0]]*3,METHODS[1:]))+[('window_depth','region_depth')]:
            paired.append(dict(scope=subset,**paired_stats(items,'correct',left,right)))
    for name,rows in (('scored_questions',scored),('scored_bindings',bind)):
        score.jsonl(RUN/f'{name}.jsonl',rows)
    with (RUN/'paired_accuracy.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(paired[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(paired)
    result=dict(
        official_scorer_sha256=source_hash,input_seal_sha256=sha256(RUN/'input_seal.json'),
        prediction_seal_sha256=sha256(RUN/'prediction_seal.json'), aggregate_score_verified=True,
        frames=len(regions),rows=len(scored),queries=seal['queries'],object_queries=len(bind),
        runtime=read_json(RUN/'runtime.json'), paired_accuracy=paired, questions=question_summary(scored),
        objects=object_summary(bind),
        artifacts={name:sha256(RUN/name) for name in ('scored_questions.jsonl','scored_bindings.jsonl','paired_accuracy.csv')})
    write_json(RUN/'summary.json',result)
    print({k:v['answer_correct'] for k,v in result['questions'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source',type=Path,required=True)
    run(parser.parse_args().official_source)
