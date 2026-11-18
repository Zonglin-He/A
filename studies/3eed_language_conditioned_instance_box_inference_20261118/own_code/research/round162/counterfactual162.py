"""Reciprocal subject/reference prompts with independent camera-ordered box labels."""
from collections import Counter
from dataclasses import replace
import gc
import re
import sys

import numpy as np

from evidence162 import EvidenceGrounder
from geometry162 import CameraX, axes
from manifest162 import HERE, ORDER, PLATFORMS, WORK, atomic_json, read, sensor_row, sha, target_rows
from mechanisms162 import annotation_inventory, box_owner, read_evidence
from run162 import SETTINGS, array_file, seal
from score162 import put_table
from solver162 import Options, solve
from support162 import SceneSupport
from score154 import many_iou


def interval(box, camera):
    offsets = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    points = (box[3:6] * .5 * offsets) @ axes(box[6]).T + box[:3]
    values = np.array([camera.project(p)[0] for p in points])
    if np.isfinite(values).all() and values.min()>=0 and values.max()<=1:
        return values.min(),values.max()
    return None


def register():
    if not (HERE/'DEV_SEAL.json').is_file():
        raise RuntimeError('Counterfactual labels are generated only after DEV is sealed')
    old = target_rows()
    registered = []
    selections = Counter()
    seen = set()
    for i in read(WORK/'AUDIT_PANEL_PRIVATE.json'):
        row = old[i]
        src,tgt = row['source'],row['target']
        key=src,tgt,row['point_sha']
        if selections[src,tgt]>=4 or key in seen:
            continue
        r = row['r']
        names, boxes = annotation_inventory(r['metapath'],tgt,int(r['suffix'] or 0),int(r['object_index']))
        names = [name.replace('_',' ').replace('-',' ').strip() for name in names]
        counts = Counter(names)
        if counts[names[0]]!=1 or not re.fullmatch('[a-z]+(?: [a-z]+){0,2}',names[0]):
            continue
        with np.load(WORK/'DEV'/f'{i:05d}.npz',allow_pickle=False) as p:
            camera = CameraX(p['ev_camera_intrinsic'],p['ev_model_to_camera'],p['ev_distortion'],int(p['ev_width']))
        left = interval(boxes[0],camera)
        if left is None:
            continue
        for j in range(1,len(boxes)):
            if counts[names[j]]!=1 or not re.fullmatch('[a-z]+(?: [a-z]+){0,2}',names[j]):
                continue
            right=interval(boxes[j],camera)
            if right is None:
                continue
            if left[1]+.02<right[0]:
                a,b=0,j
            elif right[1]+.02<left[0]:
                a,b=j,0
            else:
                continue
            first=f'The {names[a]} is to the left of the {names[b]} in the image.'
            second=f'The {names[b]} is to the right of the {names[a]} in the image.'
            registered.append(dict(index=i,source=src,target=tgt,expressions=[first,second],
                                   labels=[boxes[a].tolist(),boxes[b].tolist()]))
            selections[src,tgt]+=1
            seen.add(key)
            break
    atomic_json(WORK/'COUNTERFACTUAL_PRIVATE.json',registered)
    atomic_json(HERE/'COUNTERFACTUAL_DESIGN.json',dict(
        reciprocal_pairs=len(registered), counts={s+'_'+t:selections[s,t] for s,t in ORDER},
        criterion='Both annotated classes unique in that frame inventory. Both OBBs fully inside horizontal image bounds with at least 2% gap. Swap identified entities and LEFT/RIGHT; both labels independently originate in dataset annotations.',
        configuration_unchanged=True, uses_annotation_class_to_construct_test_text_only=True,
        challenge='distinct documented classes, full-text reciprocal test; annotation inventory can miss unnamed objects',
        sha256=sha(WORK/'COUNTERFACTUAL_PRIVATE.json'),FINAL_access=False))
    print('Registered reciprocal prompts',len(registered),flush=True)


def predict():
    import torch

    test=read(WORK/'COUNTERFACTUAL_PRIVATE.json')
    assert sha(WORK/'COUNTERFACTUAL_PRIVATE.json')==read(HERE/'COUNTERFACTUAL_DESIGN.json')['sha256']
    if not test:
        return
    fixed=read(SETTINGS)['sources']
    rows=target_rows()
    folder=WORK/'COUNTERFACTUAL'
    folder.mkdir(exist_ok=True)
    for s in PLATFORMS:
        cases=[(i,x) for i,x in enumerate(test) if x['source']==s]
        if not cases:
            continue
        model=EvidenceGrounder(s)
        settings=fixed[s]
        opt=Options(**settings['options'])
        prior,sigma=(np.asarray(settings[k]) for k in ('log_dimensions','log_scale'))
        for i,record in cases:
            frames,parser,heldout=[],[],[]
            for text in record['expressions']:
                ev=model.evidence_frame(sensor_row(rows[record['index']]['r'],text))
                scene=SceneSupport(ev)
                normal=solve(scene,opt,prior,sigma)
                ablation=solve(scene,replace(opt,relation_weight=0.),prior,sigma)
                frames.append(np.stack([ev.old_pca,normal.box,ablation.box]))
                heldout.append(normal.role_boxes[1] if len(normal.role_boxes)>1 else np.full(7,np.nan))
                parser.append([len(ev.role_mask),int(any(a==0 or b==0 for a,b,_ in ev.edges))])
            array_file(folder/f'{i:05d}.npz',boxes=np.stack(frames),reference=np.stack(heldout),parser=np.array(parser))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    seal('COUNTERFACTUAL',folder,range(len(test)),source_parameters_sha256=sha(SETTINGS),
         altered_valid_text_with_known_new_targets=True)


def score():
    from score162 import verified

    test=read(WORK/'COUNTERFACTUAL_PRIVATE.json')
    if not test:
        return
    assert sha(WORK/'COUNTERFACTUAL_PRIVATE.json')==read(HERE/'COUNTERFACTUAL_DESIGN.json')['sha256']
    ids=verified('COUNTERFACTUAL')
    rows=[]
    for i in ids:
        reg=test[i]
        gt=np.asarray(reg['labels'])
        with np.load(WORK/'COUNTERFACTUAL'/f'{i:05d}.npz',allow_pickle=False) as p:
            for j,name in enumerate(('r158','lcib','lcib_no_relation')):
                selected=p['boxes'][:,j]
                iou=[float(many_iou(b[None],g)[0]) for b,g in zip(selected,gt)]
                owner=[box_owner(b,gt) for b in selected]
                roles=[box_owner(b,gt) for b in p['reference']] if name=='lcib' else [-1,-1]
                rows.append(dict(ordinal=i+1,source=reg['source'],target=reg['target'],arm=name,
                                 iou_left=iou[0],iou_right=iou[1],both_over25=min(iou)>.25,both_over50=min(iou)>.50,
                                 role_switch_correct=owner==[0,1],reference_switch_correct=roles==[1,0] if name=='lcib' else '',
                                 explicit_edges_both_present=bool(p['parser'][:,1].all())))
    put_table(HERE/'results/DEV/counterfactual.csv',rows)
    atomic_json(HERE/'results/DEV/COUNTERFACTUAL_RESULT.json',dict(n=len(test),
        rows=len(rows),arms={k:{f:sum(r[f] for r in rows if r['arm']==k)
            for f in ('both_over25','both_over50','role_switch_correct','explicit_edges_both_present')}
            for k in ('r158','lcib','lcib_no_relation')},
        no_prediction_or_model_selection_using_new_GT=True,FINAL_access=False))


if __name__=='__main__':
    for name in sys.argv[1:]:
        {'register':register,'predict':predict,'score':score}[name]()
