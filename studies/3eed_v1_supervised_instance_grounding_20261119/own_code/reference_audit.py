"""Prepare a blinded render and score the small natural-reference diagnostic."""

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate import NATIVE, VOTES, PRIVATE, fingerprint, many_iou, old_boxes, score_boxes, to_sensor
from crossplatform.round7.camera_projection import (CameraSpec, project_model_points,
                                                    read_camera_allowlist)
from roles import CORNERS, inventory
from scope import HERE, PLATFORMS, canonical, evaluation_rows, file_sha, physical_key
from scope import possible_relation, protocol, read, save, stable

OUT = HERE/'reference_audit_PRIVATE'
LIMIT = 4


def select(rows):
    selected = []
    for platform in PLATFORMS:
        recordings = defaultdict(list)
        for index,row in enumerate(rows):
            if row['platform']==platform and possible_relation(row['caption']):
                recordings[row['scene_cluster']].append((stable('v1-independent-reference',row['id']),index))
        queues = [deque(k for _,k in sorted(recordings[key]))
                  for key in sorted(recordings,key=lambda x:stable('review-recording',x))]
        n,seen = 0,set()
        while n<LIMIT and any(queues):
            for entries in queues:
                while entries:
                    i=entries.popleft()
                    key=physical_key(platform,rows[i]['metapath'])
                    content=rows[i]['point_content_cluster']
                    if key in seen or content in seen:
                        continue
                    seen.update((key,content))
                    selected.append(i)
                    n+=1
                    break
                if n==LIMIT:
                    break
        if n!=LIMIT:
            raise RuntimeError(f'{platform} has too few direct-relation observations')
    return selected


def primary_index(row):
    metadata = read(row['metapath'])
    subject = [i for i,g in enumerate(metadata['ground_info'])
               if canonical(g['caption'])==canonical(row['caption'])]
    if len(subject)!=1:
        return None
    return subject[0]


def display(index,row,path):
    match=primary_index(row)
    font=ImageFont.truetype('DejaVuSans.ttf',15)
    bold=ImageFont.truetype('DejaVuSans-Bold.ttf',16)
    with Image.open(row['imagepath']) as original:
        wh=original.size
        frame=original.convert('RGB')
    frame.thumbnail((624,344))
    canvas=Image.new('RGB',(640,560),'white')
    at=((640-frame.width)//2,30)
    canvas.paste(frame,at)
    draw=ImageDraw.Draw(canvas)
    draw.text((10,6),f'{index:02d}  {row["platform"]}  A = annotated subject',font=bold,fill='black')
    draw.multiline_text((10,385),'\n'.join(textwrap.wrap(row['caption'],width=81))[:950],
                        font=font,fill='black',spacing=2)
    choices = {}
    if match is not None:
        target=dict(row,object_index=match)
        boxes,classes=inventory(target)
        spec=CameraSpec.from_camera_fields(
            row['platform'],read_camera_allowlist(row['metapath']),
            waymo_view_suffix=int(row['suffix'] or 0),missing_distortion_policy='assume_zero_declared')
        local=CORNERS[None]*boxes[:,None,3:6]
        x,y=local[...,0].copy(),local[...,1].copy()
        c,s=np.cos(boxes[:,6,None]),np.sin(boxes[:,6,None])
        local[...,0],local[...,1]=c*x-s*y,s*x+c*y
        uv=project_model_points((local+boxes[:,None,:3]).reshape(-1,3),spec,wh)['uv'].reshape(len(boxes),8,2)
        for j,pixels in enumerate(uv):
            if not np.isfinite(pixels).all():
                continue
            low,hi=pixels.min(0),pixels.max(0)
            if not ((hi > 0)&(low < np.asarray(wh))).all():
                continue
            tag='A' if j==0 else str(j)
            square=np.clip(np.r_[low,hi].reshape(2,2),0,np.asarray(wh)-1)
            square=square*np.array([frame.width/wh[0],frame.height/wh[1]])+at
            color='#21e3ff' if j==0 else '#ffe451'
            x0,y0=square[0]; x1,y1=square[1]
            draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
            draw.rectangle((x0,y0-16,x0+26,y0),fill=color)
            draw.text((x0+1,y0-16),tag,font=font,fill='black')
            if j>0:
                choices[tag]=dict(box=to_sensor(boxes[j:j+1],row['platform'],int(row['suffix'] or 0))[0].tolist(),
                                  annotated_category=str(classes[j]))
    canvas.save(path)
    return choices


def verify():
    lock=read(PRIVATE/'LOCK.json')
    seal=read(PRIVATE/'SEALED.json')
    assert seal['n']==protocol()['evaluation']['records']
    assert seal['lock_sha256']==file_sha(PRIVATE/'LOCK.json')
    assert lock['implementation']==fingerprint()
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    return seal


def prepare():
    verify()
    rows=evaluation_rows()
    slots=select(rows)
    OUT.mkdir(mode=0o700)
    cases=[]
    for number,ordinal in enumerate(slots,1):
        image=OUT/f'case_{number:02d}.png'
        label_boxes=display(number,rows[ordinal],image)
        cases.append(dict(case=number,ordinal=ordinal,reference_inventory=label_boxes))
    for start in range(0,len(cases),4):
        page=Image.new('RGB',(1280,1120),'white')
        for offset,case in enumerate(cases[start:start+4]):
            with Image.open(OUT/f'case_{case["case"]:02d}.png') as img:
                page.paste(img,((offset%2)*640,(offset//2)*560))
        page.save(OUT/f'contact_{start//4+1}.png')
    save(OUT/'BLIND_ROSTER.json',cases)
    save(OUT/'BLIND_SEAL.json',dict(n=len(slots),protocol_sha256=file_sha(HERE/'reference_audit_protocol.json'),
         n_platforms=3,roster_sha256=file_sha(OUT/'BLIND_ROSTER.json'),
         prediction_seal_sha256=file_sha(PRIVATE/'SEALED.json'),prediction_disclosed_to_reader=False,
         FINAL_access=False))


def score():
    from score_ablation import sensor_truth

    seal=verify()
    blind=read(OUT/'BLIND_SEAL.json')
    assert blind['roster_sha256']==file_sha(OUT/'BLIND_ROSTER.json')
    assert blind['prediction_seal_sha256']==file_sha(PRIVATE/'SEALED.json')
    labels=read(OUT/'JUDGMENTS.json')
    cases=read(OUT/'BLIND_ROSTER.json')
    rows=evaluation_rows()
    assert set(labels)=={str(c['case']) for c in cases}
    arms=protocol()['comparisons']
    scoring={}
    for source in PLATFORMS:
        for target in PLATFORMS:
            if source==target:
                continue
            file=PRIVATE/f'{source}_to_{target}.npz'
            assert file_sha(file)==seal['files'][file.name]['sha256']
            order,before=old_boxes(rows,source,target)
            with np.load(file,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['ordinals'],order)
                np.testing.assert_array_equal(data['picked'][:,0],before)
                scoring[source,target]=dict(row={int(x):i for i,x in enumerate(order)},
                    boxes=data['picked'],references=data['referents'],known=data['reference_known'],
                    pools=data['pool'],roles=data['role_flags'])
    ledger={x:Counter() for x in arms}
    coverage=Counter()
    reviewed=Counter()
    verified=0
    for case in cases:
        index=case['ordinal']
        row=rows[index]
        label=labels[str(case['case'])]
        if label is None:
            reviewed[f'{row["platform"]}_unknown']+=1
            continue
        target_box=np.asarray(case['reference_inventory'][str(label)]['box'])
        subject=sensor_truth(row['platform'],row['metapath'],row['caption'],row['suffix'])
        verified+=1
        reviewed[f'{row["platform"]}_verified']+=1
        for source in PLATFORMS:
            if source==row['platform']:
                continue
            data=scoring[source,row['platform']]
            k=data['row'][index]
            a,_=score_boxes(data['boxes'][k],subject)
            b,_=score_boxes(data['references'][k],target_box)
            total=NATIVE+VOTES
            raw=many_iou(data['pools'][k],target_box)
            selected=data['pools'][k,2*total:]
            best=2*total+raw[2*total:].argmax()
            independent,_=score_boxes(data['pools'][k,best:best+1],target_box)
            np.testing.assert_allclose(raw[best],independent[0],atol=1e-10,rtol=0)
            subject_support=many_iou(selected,subject).max()>.25
            ref_support=raw[best]>.25
            coverage.update(n=1,ref_phrase=int(data['roles'][k,0]),
                      active=int(data['roles'][k,2]),
                      v1_a256_ref=int(raw[:NATIVE].max()>.25),
                      vote_union_ref=int(raw[:total].max()>.25),
                      continuous_union_ref=int(raw[:2*total].max()>.25),
                      selected_pool_ref=int(ref_support),
                      selected_pool_both=int(subject_support and ref_support))
            for j,name in enumerate(arms):
                present=bool(data['known'][k,j])
                ok_ref=present and b[j]>.25
                ledger[name].update(n=1,subject=int(a[j]>.25),reference=int(ok_ref),
                                    dual=int(ok_ref and a[j]>.25),unknown=int(not present))
    save(HERE/'results/reference_audit.json',
         dict(panel_physical_rows=len(cases),verified_physical=verified,unknown_physical=len(cases)-verified,
              reviewed_by_platform=reviewed,verified_coverage=coverage,
              scores=ledger,reviewer_count=1,presealed_predictions=True,descriptive=True,FINAL_access=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('prepare','score'))
    if cli.parse_args().phase=='prepare':
        prepare()
    else:
        score()
