"""Mine identity supervision from source inventories and generate diverse, source-only pairs."""
from argparse import ArgumentParser
from collections import Counter, defaultdict, deque
from functools import lru_cache
import re
import time

import numpy as np
from PIL import Image

from scope175 import (OLD_PRIVATE, PLATFORMS, PRIVATE, RELATIONS, SPEC, camera_view, frame,
                      json_atomic, read, recording, register, sha, stable, state)
from roles175 import CODES, roles_batch
from prepare import inventory, meta
from camera_projection import CAMERA_KEYS, CameraSpec, project_model_points

NOUNS = {
    'car': {'car','vehicle','sedan','suv','van','pickup','automobile','taxi','coupe','minivan'},
    'truck': {'truck','lorry'},
    'bus': {'bus','shuttle','coach'},
    'pedestrian': {'pedestrian','person','man','woman','boy','girl','child','individual','walker','worker','passerby'},
    'cyclist': {'cyclist','biker','bike','bicycle','rider'},
    'othervehicle': {'tractor','trailer','machine','machinery'}}
PLURALS = {'people','men','women','children','cars','vehicles','pedestrians','persons','buses','trucks'}
CORNERS = np.asarray([(i,j,k) for i in (-.5,.5) for j in (-.5,.5) for k in (-.5,.5)])


@lru_cache(maxsize=64)
def scene(path, image, platform, view):
    row=dict(metapath=path,imagepath=image,platform=platform,suffix=view)
    fields=meta(path)
    entries=inventory(row,fields['ground_info'])
    if not entries:
        return None
    names, boxes = zip(*entries,strict=True)
    b=np.stack(boxes)
    cam=CameraSpec.from_camera_fields(platform,{k:fields[k] for k in CAMERA_KEYS if k in fields},
                                     waymo_view_suffix=view,missing_distortion_policy='assume_zero_declared')
    with Image.open(image) as im:
        size=im.size
    corners=CORNERS[None]*b[:,None,3:6]
    a,x,y=b[:,6,None],corners[...,0].copy(),corners[...,1].copy()
    c,s=np.cos(a),np.sin(a)
    corners[...,0]=c*x-s*y
    corners[...,1]=s*x+c*y
    p=project_model_points((corners+b[:,None,:3]).reshape(-1,3),cam,size)
    middle=project_model_points(b[:,:3],cam,size)
    xy=p['uv'].reshape(-1,8,2)/np.asarray(size)
    depth=p['depth'].reshape(-1,8)
    visible=middle['in_image'] & np.isfinite(xy).all((1,2))
    fields=np.concatenate((xy,depth[...,None]),axis=-1)
    return np.asarray(names),b,visible,fields.min(1),fields.max(1),np.c_[middle['uv']/size,middle['depth']]


def inventory_for(row):
    return scene(row['metapath'],row['imagepath'],row['platform'],camera_view(row))


def satisfaction(subject, reference, kind, boxes, lo, hi, middle):
    if kind==CODES['near']:
        return np.linalg.norm(boxes[subject,:3]-boxes[reference,:3],axis=-1)
    axis = (kind-1)//2 if kind<=CODES['below'] else 2
    backwards = kind in (CODES['left'],CODES['above'],CODES['closer'])
    if backwards:
        gap=lo[reference,axis]-hi[subject,axis]
        centres=middle[reference,axis]-middle[subject,axis]
    else:
        gap=lo[subject,axis]-hi[reference,axis]
        centres=middle[subject,axis]-middle[reference,axis]
    return gap,centres


def natural_label(row, role):
    if role is None:
        return dict(row,relation=0,origin='natural_subject_only'), 'relation_unresolved'
    kind=role['relation']
    out=dict(row,relation=kind,spans=role['spans'],origin='natural_subject_only')
    world=inventory_for(row)
    head=role['head'].lower()
    if world is None or head in PLURALS:
        return out,'inventory_or_singular_reference_missing'
    names,boxes,vis,lo,hi,mid=world
    if row['platform']!='waymo' and head in (NOUNS['truck']|NOUNS['bus']):
        types={'car'}
    else:
        types={name for name, forms in NOUNS.items() if head in forms}
        if head=='vehicle' and row['platform']=='waymo':
            types |= {'truck','bus','othervehicle'}
    if not types:
        return out,'reference_category_not_annotated'
    ref_span=role['spans'][1]
    phrase=row['caption'][ref_span[0]:ref_span[1]].lower()
    if re.search(r'\b(?:two|three|several|group|row|multiple|both|pair|half|other side)\b',phrase):
        return out,'nonsingular_reference'
    subject=np.asarray(row['boxes'][0])
    # The authoritative subject box is already joined to its natural utterance.
    overlap=np.linalg.norm(boxes[:,:3]-subject[:3],axis=1)
    target=np.argmin(overlap)
    if overlap[target]>.15 or not vis[target]:
        return out,'subject_not_camera_supported'
    candidates=np.flatnonzero(vis & np.isin(names,list(types)) & (np.arange(len(boxes))!=target))
    if not len(candidates):
        return out,'no_annotated_reference'
    if kind==CODES['near']:
        d=satisfaction(target,candidates,kind,boxes,lo,hi,mid)
        order=np.argsort(d,kind='stable')
        if len(order)>1 and d[order[1]]-d[order[0]]<.75:
            return out,'ambiguous_reference'
        ref=int(candidates[order[0]])
    else:
        gap,_=satisfaction(target,candidates,kind,boxes,lo,hi,mid)
        margin=.012 if kind<=CODES['below'] else .35
        viable=candidates[gap>margin]
        if len(viable)!=1:
            return out,'ambiguous_or_inconsistent_reference'
        ref=int(viable[0])
    out.update(boxes=[row['boxes'][0],boxes[ref].tolist()],known_reference=True,
               origin='natural_unique_inventory')
    return out,'resolved'


def unique_anchors(ids, lo, hi, center):
    if len(ids)==1:
        return [(ids[0],'')]
    rank=ids[np.argsort(center[ids,0],kind='stable')]
    out=[]
    if hi[rank[0],0]+.012<lo[rank[1],0]:
        out.append((rank[0],'leftmost visible '))
    if hi[rank[-2],0]+.012<lo[rank[-1],0]:
        out.append((rank[-1],'rightmost visible '))
    return out


def templates(kind, variant):
    lead=('Identify the ','Find the ','Select the ')[variant]
    link = {
        'left': (' nearest on the left of ',' just left of ',' closest on the left side of '),
        'right': (' nearest on the right of ',' just right of ',' closest on the right side of '),
        'above': (' nearest above ',' closest above ',' directly above '),
        'below': (' nearest below ',' closest below ',' directly below '),
        'near': (' nearest to ',' closest to ',' at the shortest distance from '),
        'closer': (' closest among those closer to the camera than ',' just nearer to the camera than ',
                   ' nearest on the camera side of '),
        'farther': (' closest among those farther from the camera than ',' just deeper in view than ',
                    ' nearest on the far side of '),
    }[kind][variant]
    return lead, link, ' in three-dimensional space.' if kind=='near' else ' in this image.'


def synthetic_options(row):
    world=inventory_for(row)
    if world is None:
        return {}
    names,boxes,vis,lo,hi,mid=world
    if vis.sum()<3:
        return {}
    by_class={x:np.flatnonzero((names==x)&vis) for x in sorted(set(names[vis]))}
    options={}
    for category, ids in by_class.items():
        for ref,qualifier in unique_anchors(ids,lo,hi,mid):
            for nominal, subject_pool in by_class.items():
                ids_s=subject_pool[subject_pool!=ref]
                if len(ids_s)<2:
                    continue
                for kind in range(1,len(RELATIONS)):
                    measurement=satisfaction(ids_s,ref,kind,boxes,lo,hi,mid)
                    if kind==CODES['near']:
                        d=measurement
                        order=np.argsort(d,kind='stable')
                        if d[order[1]]-d[order[0]]<.75:
                            continue
                        k=order[0]
                    else:
                        gap,centres=measurement
                        margin=.012 if kind<=CODES['below'] else .35
                        # The language selects the nearest on the named side.
                        possible=np.flatnonzero(centres>0.)
                        if not len(possible):
                            continue
                        order=possible[np.argsort(centres[possible],kind='stable')]
                        k=order[0]
                        if gap[k]<=margin or (len(order)>1 and centres[order[1]]-centres[k]<=margin):
                            continue
                    sub=ids_s[k]
                    score=stable('R175:pair',frame(row),camera_view(row),int(sub),int(ref),kind)
                    # Prefer examples with same-category distractors at both ends.
                    candidate = (len(ids)<2,score,int(sub),int(ref),nominal,qualifier+category)
                    if kind not in options or candidate<options[kind]:
                        options[kind]=candidate
    out={}
    for k, (_,key,s,r,noun,ref_phrase) in options.items():
        a,b,c=templates(RELATIONS[k],int(key[:8],16)%3)
        lead=f'{a}{noun}'
        ref=f'the {ref_phrase}'
        x,y,z=len(a),len(lead),len(lead)+len(b)
        out[k]=dict(row,caption=lead+b+ref+c,spans=[[x,y],[z,z+len(ref)],[y,z]],relation=k,
                    known_reference=True,boxes=[boxes[s].tolist(),boxes[r].tolist()],
                    origin='synthetic_labelled_pair',
                    same_class_reference_foil=bool(sum(vis & (names==names[r]))>1))
    return out


def synthesize(natural,cap):
    alternatives=defaultdict(dict)
    for row in natural:
        alternatives[recording(row)].setdefault(frame(row),row)
    pending={g:deque(p for _,p in sorted((stable('R175:frame',k),r) for k,r in q.items()))
             for g,q in alternatives.items()}
    domains=sorted(pending,key=lambda g:stable('R175:recording',g))
    result,counts,inspected=[],Counter(),0
    while pending and len(result)<cap:
        for g in domains:
            if g not in pending:
                continue
            row=pending[g].popleft()
            if not pending[g]:
                del pending[g]
            inspected+=1
            options=synthetic_options(row)
            if options:
                k=min(options,key=lambda p:(counts[p],stable('R175:predicate',frame(row),p)))
                result.append(options[k])
                counts[k]+=1
            if len(result)==cap:
                break
    return result,inspected


def describe(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['origin']].append(row)
    return {name:dict(n=len(part),frames=len({frame(x) for x in part}),
                      recordings=len({recording(x) for x in part}),
                      predicates=dict(Counter(RELATIONS[x['relation']] for x in part)),
                      same_class_reference_foil=sum(x.get('same_class_reference_foil',False) for x in part))
            for name,part in grouped.items()}


def main():
    p=ArgumentParser()
    p.add_argument('sources',choices=PLATFORMS,nargs='+')
    args=p.parse_args()
    register()
    state('source_supervision','Build source-only linked identities and unique generated frame pairs.')
    for source in args.sources:
        start=time.perf_counter()
        old=OLD_PRIVATE/source
        folder=PRIVATE/source
        folder.mkdir(exist_ok=True,mode=0o700)
        contract=read(old/'ROSTER_SEAL.json')
        counts, files, groups = {}, {}, {}
        for split in ('train','monitor'):
            name=f'{split}_PRIVATE.json'
            assert sha(old/name)==contract['rosters'][name]
            natural=[r for r in read(old/name) if not r['known_reference']]
            groups[split]={recording(r) for r in natural}
            rows,reason=[],Counter()
            for start_batch in range(0,len(natural),64):
                part=natural[start_batch:start_batch+64]
                for row,role in zip(part,roles_batch(part,source_only_fast=True),strict=True):
                    record, why=natural_label(row,role)
                    rows.append(record)
                    reason[why]+=1
            synth,inspected=synthesize(natural,read(SPEC)['synthetic_caps'][split])
            assert len({frame(r) for r in synth})==len(synth)
            assert all(r['platform']==source for r in rows+synth)
            json_atomic(folder/name,rows+synth)
            files[name]=sha(folder/name)
            counts[split]=dict(origins=describe(rows+synth),natural_references=dict(reason),inspected_frames=inspected,
                               physical_frames=len({frame(r) for r in rows}),recordings=len(groups[split]))
            print('ROSTER',source,split,counts[split],round(time.perf_counter()-start,1),flush=True)
        assert not (groups['train'] & groups['monitor'])
        json_atomic(folder/'ROSTER_SEAL.json',dict(source=source,counts=counts,files=files,
                    input=sha(PRIVATE/'INPUT_LOCK.json'),recording_disjoint=True,FINAL_access=False))


if __name__=='__main__':
    main()
