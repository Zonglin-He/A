"""Blindly check the original-scene referent for every possible final-column rescue."""
import argparse
from collections import Counter
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from study import ORDER, PREDICTIONS, PRIVATE, RESULTS, THRESHOLDS, frame, json_atomic, ordinary_rows, read, sha, stable
from analyse_factorial import blocks, choice_with_mask, verify_seal
from predict import NAMES as OLD_NAMES
from reference_truth import assigned_owners, library, role_phrase
from camera_projection import CameraSpec, project_model_points, read_camera_allowlist

PANEL = PRIVATE/'blind_reference'
PLAN = PANEL.parents[1]/'visual_review_plan.json'
EDGE_S = tuple((i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4))
OFFSET = np.asarray([(x,y,z) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])


def columns_in_image(row, boxes, size):
    camera = CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(row['metapath']),
                                             waymo_view_suffix=int(row['suffix'] or 0),
                                             missing_distortion_policy='assume_zero_declared')
    corners = OFFSET[None]*boxes[:,None,3:6]
    x,y = corners[...,0].copy(),corners[...,1].copy()
    u,v = np.cos(boxes[:,6,None]),np.sin(boxes[:,6,None])
    corners[...,0],corners[...,1] = x*u-y*v,x*v+y*u
    corners += boxes[:,None,:3]
    xy = project_model_points(corners.reshape(-1,3),camera,size)['uv'].reshape(len(boxes),8,2)
    centers = project_model_points(boxes[:,:3],camera,size)['uv']
    return xy/np.asarray(size),centers/np.asarray(size)


def render(key,row,names,boxes,picture):
    with Image.open(row['imagepath']) as raw:
        size = raw.size
        rgb = raw.convert('RGB')
    rgb.thumbnail((1250,730),Image.Resampling.LANCZOS)
    xy,centers = columns_in_image(row,boxes,size)
    xy *= (rgb.width,rgb.height)
    centers *= (rgb.width,rgb.height)
    scene = ImageDraw.Draw(rgb)
    color = ('#fff085','#f3abff','#a0dfff','#ffbd85','#ff9bad')
    font = ImageFont.truetype('DejaVuSans.ttf',18)
    small = ImageFont.truetype('DejaVuSans.ttf',16)
    for i,(verts,at) in enumerate(zip(xy,centers,strict=True)):
        ink = '#59ff87' if i==0 else color[(i-1)%len(color)]
        for a,b in EDGE_S:
            if np.isfinite(verts[[a,b]]).all():
                scene.line((tuple(verts[a]),tuple(verts[b])),fill=ink,width=2)
        if 0<=at[0]<rgb.width and 0<=at[1]<rgb.height:
            badge = (at[0]-10,at[1]-11,at[0]+18,at[1]+10)
            scene.rectangle(badge,fill='#152540')
            scene.text(tuple(at),str(i),fill=ink,font=small,anchor='lm')
    phrase = ' '.join(role_phrase(row))
    lines = [f'{key}   {row["platform"]}   reference expression: {phrase}',
             *textwrap.wrap(row['caption'],122),
             *textwrap.wrap('Annotations   '+'   '.join(f'{i} {k}' for i,k in enumerate(names)),138)]
    height = 22*(len(lines)+1)
    canvas = Image.new('RGB',(max(rgb.width,1250),rgb.height+height),'#fafafa')
    canvas.paste(rgb,((canvas.width-rgb.width)//2,height))
    overlay = ImageDraw.Draw(canvas)
    for i,line in enumerate(lines):
        overlay.text((12,8+i*22),line,fill='#172336',font=font if i==0 else small)
    canvas.save(picture,quality=91)


def prepare():
    proof = verify_seal()
    assert not proof['FINAL_access']
    lookup = {}
    for s,t in ORDER:
        with np.load(PRIVATE/'score'/f'{s}_to_{t}.npz',allow_pickle=False) as p:
            j = p['values'][:,-1]
            d = p['diag']
            hit = np.stack([(j<=q)&(d[:,1]>q)&(d[:,3]>q) for q in THRESHOLDS],axis=1)
        for i in np.flatnonzero(hit.any(1)):
            lookup.setdefault((t,int(i)),[]).append([s,*map(bool,hit[i])])
    assert sum(sum(x[1] for x in seq) for seq in lookup.values())==31
    assert sum(sum(x[2] for x in seq) for seq in lookup.values())==4
    panel = []
    _,inventory = library()
    PANEL.mkdir(mode=0o700)
    for j,(t,i) in enumerate(sorted(lookup,key=lambda a:stable('blind-ref-sort',*a)),1):
        key = f'V{j:02d}'
        row=ordinary_rows(t)[i]
        names,boxes=inventory(row['metapath'],t,int(row['suffix'] or 0),row['object_index'])
        image = PANEL/f'{key}.jpg'
        render(key,row,names,boxes,image)
        panel.append(dict(case=key,target=t,ordinal=i,inventory=len(names),file_sha=sha(image),
                          qualifying=lookup[t,i]))
    registry=PANEL/'REGISTRY_PRIVATE.json'
    json_atomic(registry,panel)
    json_atomic(PANEL/'PANEL_SEAL.json',dict(n=len(panel),plan=sha(PLAN),registry=sha(registry),
                                                  predictions=sha(PREDICTIONS/'PREDICTION_SEAL.json'),
                                                  blind_to_model_outputs=True,FINAL_access=False))
    print('BLINDED_REFERENCE_PANEL',len(panel),flush=True)


def score():
    verify_seal()
    lock=read(PANEL/'PANEL_SEAL.json')
    assert lock['registry']==sha(PANEL/'REGISTRY_PRIVATE.json') and sha(PLAN)==lock['plan']
    panel=read(PANEL/'REGISTRY_PRIVATE.json')
    decisions=read(PANEL/'REVIEWS_PRIVATE.json')
    assert len(panel)==len(decisions)==lock['n']
    stats = [Counter(),Counter()]
    statuses = Counter()
    local_rows = []
    _,inventory = library()
    joint=OLD_NAMES.index('joint')-1
    for case,item in zip(panel,decisions,strict=True):
        assert case['case']==item['case']
        assert sha(PANEL/f'{case["case"]}.jpg')==case['file_sha']
        status=item['status']
        assert status in ('verified','ambiguous','non_instance','no_annotated_reference','subject_or_role_ambiguous')
        statuses[status]+=1
        ref=int(item['reference'])
        assert (0<=ref<case['inventory'])==(status=='verified')
        t,k=case['target'],case['ordinal']
        row=ordinary_rows(t)[k]
        _,boxes=inventory(row['metapath'],t,int(row['suffix'] or 0),row['object_index'])
        for s,*hit in case['qualifying']:
            direction=f'{s}_to_{t}'
            for block,orig in blocks(direction):
                lo=k-int(block['ordinals'][0])
                if not 0<=lo<len(block['ordinals']):
                    continue
                assert block['ordinals'][lo]==k
                owners=assigned_owners(orig['seeds'][lo],block['replay_refs'][lo],boxes)
                native=owners[orig['choices'][lo,joint,1]]
                valid=(owners==ref)&(ref>=0)
                select=choice_with_mask(block['joint_score'][lo],valid) if valid.any() else None
                break
            else:
                raise ValueError('Unable to locate sealed panel row')
            path=PRIVATE/'score'/f'{direction}.npz'
            assert sha(path)==read(PRIVATE/'SCORING_SEAL.json')['private_scores'][path.name]
            with np.load(path,allow_pickle=False) as p:
                recovered=float(p['slots'][k,select]) if select is not None else -1.
            for t_index,(uses,cut) in enumerate(zip(hit,THRESHOLDS,strict=True)):
                if not uses:
                    continue
                count=stats[t_index]
                count['reviewed_switchable']+=1
                count['identity_verified']+=int(ref>=0)
                count['identity_unresolved']+=int(ref<0)
                count['wrong_native_owner']+=int(ref>=0 and native>=0 and native!=ref)
                count['unknown_native_owner']+=int(ref>=0 and native<0)
                count['correct_native_owner']+=int(ref>=0 and native==ref)
                count['correct_ref_available']+=int(ref>=0 and valid.any())
                count['wrong_and_correct_ref_rescued']+=int(ref>=0 and native>=0 and native!=ref and recovered>cut)
                count['unknown_native_with_correct_ref_rescued']+=int(ref>=0 and native<0 and recovered>cut)
                count['correct_native_other_slot_same_ref_rescued']+=int(ref>=0 and native==ref and recovered>cut)
                local_rows.append(dict(case=case['case'],threshold=cut,status=status,
                                       native_owner=int(native),truth=ref,gold_intervention_iou=recovered))
    json_atomic(PANEL/'REVIEW_SCORED_PRIVATE.json',local_rows)
    out=dict(n=len(panel), by_resolution_status=dict(statuses), thresholds={
             str(cut):dict(m) for cut,m in zip(THRESHOLDS,stats,strict=True)},
             panel_sha256=sha(PANEL/'PANEL_SEAL.json'), labels_sha256=sha(PANEL/'REVIEWS_PRIVATE.json'),
             same_final_output_feasibility=True, single_local_reviewer=True, FINAL_access=False)
    json_atomic(RESULTS/'blind_switchable_review.json',out)
    print(out,flush=True)


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('stage',choices=('prepare','score'))
    {'prepare':prepare,'score':score}[args.parse_args().stage]()
