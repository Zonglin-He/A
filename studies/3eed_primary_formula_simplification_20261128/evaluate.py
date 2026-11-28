"""Evaluate the blind formula variants on the same GT references."""
from pathlib import Path
import sys

import numpy as np

from ablate import ARMS, EPS, FIT, HERE, MONO, REFERENCE, REGISTRATION, RELOC, RUN, STUDIES, check, digest, publish, read
from evaluate_structure import stat_panel
from shared import canonical, score_boxes, setup


def score_unique(boxes, gt):
    key = {arm:None if b is None else tuple(b) for arm,b in boxes.items()}
    unique = {v:v for v in key.values() if v is not None}
    # Canonical inputs and exact duplicates are common across one-change arms.
    results = score_boxes(unique, gt)
    return {a:0. if value is None else results[value]['iou'] for a,value in key.items()}


def audits(rows):
    result = {}
    for tag in ('formal',*FIT,'E_algebra'):
        vals = [r for case in rows if tag in case['audit'] for r in case['audit'][tag]]
        result[tag] = dict(candidates=len(vals),non_nested=sum(r['non_nested'] for r in vals),
            measured_lost=sum(r['measured_lost'] for r in vals),entries_lost=sum(r['entries_lost'] for r in vals))
        for name in ('selection','audit'):
            result[tag][name] = {key:sum(r[name][key] for r in vals) for key in ('n','changed','gained','lost')}
    return result


def score():
    path = RUN/'blind.json'
    seal = check(path,'payload',RUN/'prediction-seal.json')
    assert not seal['labels_read'] and not seal['training'] and not seal['model_calls']
    assert seal['source']['registration']==digest(REGISTRATION) and tuple(seal['arms'])==ARMS
    assert all(digest(HERE/k)==v for k,v in seal['files'].items())
    assert seal['source']['history']==digest(REFERENCE/'predictions-private.json')
    setup()
    from metric_image_common import inputs
    from truth_score_inputs import truth_data
    sys.path.insert(0,str(STUDIES/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel

    refs = check(REFERENCE/'scored-private.json','score_file',REFERENCE/'results.json')
    assert refs['gt_used_only_after_seal']
    comp={r['uid']:r for r in read(REFERENCE/'scored-private.json')}
    src=inputs()
    gt,_=truth_data({u:src[u] for u,r in comp.items() if r['target']!='mono'})
    mono,sha=labels_for_panel(read(MONO/'mono-input-seal.json'))
    assert sha==read(MONO/'mono-results.json')['official_kitti_label_sha256']
    strict=set(read(MONO/'mono-strict-roster-private.json'))
    assert len(strict)==118
    rows=read(path)
    assert len(rows)==seal['n']==873
    scores=[]
    for row in rows:
        uid=row['uid']
        if uid not in comp:
            continue
        ref=comp[uid]; truth=(mono if row['target']=='mono' else gt)[uid]
        iou=score_unique(row['boxes'],canonical(truth['box']))
        verify={'formal':'axis_2d','F_height':'height_2d','R_ray_only':'axis_ray'}
        assert all(abs(iou[out]-ref['iou'][old])<1.e-10 for out,old in verify.items()),uid
        origin=row['boxes']['formal']
        changes={a:(0. if b is None and origin is None else None if (b is None or origin is None)
                    else float(np.max(abs(np.asarray(b)-np.asarray(origin))))) for a,b in row['boxes'].items()}
        scores.append(dict(uid=uid,target=row['target'],recording=ref['recording'],branch=row['route']['formal'],
                           iou=iou,changes=changes,route=row['route']))
    assert len(scores)==836
    results_path=RUN/'scores-private.json'
    publish(results_path,scores)
    base=[r for r in scores if r['target']!='mono']
    ext=[r for r in scores if r['uid'] in strict]
    panels={'development':base,'mono_strict':ext,'mono_all':[r for r in scores if r['target']=='mono'],
            '3eed':[r for r in base if r['target']!='kitti']}
    panels.update({p:[r for r in base if r['target']==p] for p in ('waymo','drone','quad','kitti')})
    for key,group in (('development',base),('mono_strict',ext)):
        for branch in ('C_ray','H','A'):
            panels[key+'_'+branch]=[r for r in group if r['branch']==branch]
    byd={r['uid']:r for r in rows}
    assert len(byd)==873
    proof=audits(rows)
    result={}
    for key,panel in panels.items():
        stats=stat_panel(panel,ARMS,'formal')
        stats['changes']={a:dict(boxes=sum(r['changes'][a] is None or r['changes'][a]>EPS for r in panel),
                    at_1e12=sum(r['changes'][a] is None or r['changes'][a]>1.e-12 for r in panel),
                    route_changed=sum(r['route'][a]!=r['branch'] for r in panel),
                    max_abs_finite=float(max((r['changes'][a] for r in panel if r['changes'][a] is not None),default=0.))) for a in ARMS}
        result[key]=stats
        print(key,len(panel),{a:round(v['delta_miou'],3) for a,v in stats.get('metrics',{}).items()},flush=True)
    summary=dict(n=len(scores),model_calls=0,training=False,target_labels_only_after_seal=True,arms=ARMS,panels=result,
                candidate_witness_audit_873=proof,controls_matched=836,score_sha=digest(results_path),prediction=seal)
    publish(RUN/'results.json',summary)
    print('Scored and cross-checked three historical arms.',flush=True)


if __name__=='__main__':
    score()
