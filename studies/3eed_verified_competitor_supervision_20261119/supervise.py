"""Certify wrong high-score competitors separately from object ownership and box validity."""
from argparse import ArgumentParser
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from catalog import FULL, HOME, PRIVATE, PROTOCOL, RESULTS, atomic_json, label_truths, read, sha, source_roster, write_table


STAGING = PRIVATE/'review'
VISUAL_CASES = 18


def slab_gap(reference, candidates):
    """Signed largest separating rectangle or height gap in meters."""
    a = np.array(reference,dtype=float,copy=True)
    b = np.atleast_2d(np.asarray(candidates,dtype=float))
    a[3:6] = a[3:6]*1.2 + .4

    def horizontal_axis(yaw):
        u = np.stack((np.cos(yaw),np.sin(yaw)),axis=-1)
        return np.stack((u,np.stack((-u[...,1],u[...,0]),axis=-1)),axis=-2)

    axes_a = horizontal_axis(a[6])
    axes_b = horizontal_axis(b[:,6])
    axes = np.concatenate((np.broadcast_to(axes_a,(len(b),2,2)),axes_b),axis=1)
    shift = np.abs(np.einsum('ki,kji->kj',b[:,:2]-a[:2],axes))
    left = np.abs(axes @ axes_a.T) @ (a[3:5]/2.)
    right = np.einsum('kij,kj->ki',np.abs(axes @ np.swapaxes(axes_b,-1,-2)),b[:,3:5]/2.)
    plane = (shift-left-right).max(1)
    height = np.abs(b[:,2]-a[2]) - (b[:,5]+a[5])/2.
    return np.maximum(plane,height)


def shortlist(x, fit):
    base = x.example.unary
    weights = (fit.get('legacy_pair'),fit.get('pool_centered'))
    scores = [base] + [base + design @ item['weights'] for item,design in
                      zip(weights,(x.features,x.centered),strict=True) if item is not None]
    nominated = {int(np.argmax(s)) for s in scores if x.owners[int(np.argmax(s))] < 0}
    if not nominated:
        worst_position = np.min([np.argsort(np.argsort(-s,kind='stable'),kind='stable') for s in scores],axis=0)
        nominees = np.flatnonzero((x.owners == -1)&(worst_position < 3))
        if len(nominees):
            nominated.add(int(nominees[worst_position[nominees].argmin()]))
    return sorted(nominated)


def label_pack(x,item,path,fit):
    from mechanisms162 import annotation_inventory, seed_owners
    from predict171 import read_npz
    from score import points_in

    base = x.example
    pkt = read_npz(path)
    n = len(x.owners)
    fields = dict(alias=x.alias, platform=base.platform, recording=x.rec,
                  owner=x.owners.tolist(), reference_match=np.full(n,-1).tolist(),
                  instance_validity=np.full(n,-1).tolist(), shortlisted=[], geometry={})
    if base.gold<0:
        return fields
    raw = item['row']
    _, inventory = annotation_inventory(raw['metapath'],base.platform,int(raw['suffix'] or 0),raw['object_index'])
    unique = seed_owners(pkt['points'],inventory)
    ref = inventory[base.gold]
    match = np.where(x.owners<0,-1,np.where(x.owners==base.gold,1,0))
    nominated = shortlist(x,fit)
    for j in nominated:
        masks = points_in(np.atleast_2d(pkt['boxes'][j]),pkt['points'])[0]
        support = int(np.sum(masks))
        ref_support = int((masks&(unique==base.gold)).sum())
        available = int((unique==base.gold).sum())
        gap = float(slab_gap(ref,pkt['boxes'][j])[0])
        verified = available>=6 and gap>.02 and support>=3 and ref_support==0
        if verified:
            match[j]=0
        fields['geometry'][str(j)] = dict(gap=gap,support=support,reference_support=ref_support,
                                         reference_available=available,verified_exclusion=verified)
    fields['reference_match']=match.tolist()
    fields['shortlisted']=nominated
    return fields


def stable(a):
    return hashlib.sha256(('winner-blind-v1|'+str(a)).encode()).digest()


def render(key,row,packet,j):
    from calibrate168 import image_geometry
    from mechanisms162 import annotation_inventory
    from predict171 import read_npz
    from visual162 import LINES, corners, projection

    raw = row['row']
    assert sha(Path(raw['imagepath']))==raw['image_sha256']
    _, inv = annotation_inventory(raw['metapath'],raw['platform'],int(raw['suffix'] or 0),raw['object_index'])
    data = read_npz(packet)
    ref, query = inv[row['reference_identity']],data['boxes'][j]
    camera = image_geometry(raw)
    with Image.open(raw['imagepath']) as p:
        image=p.convert('RGB')
    corners_uv=[projection(camera,corners(x))*[image.width,image.height] for x in (ref,query)]
    valid = [z[np.isfinite(z).all(1)] for z in corners_uv]
    missing = not all(len(v) for v in valid)
    if any(len(v) for v in valid):
        extent = np.concatenate([v for v in valid if len(v)],axis=0)
        lower = np.floor(extent.min(0)-32).astype(int)
        upper = np.ceil(extent.max(0)+32).astype(int)
        x0,y0 = np.maximum(lower,0)
        x1,y1 = np.minimum(upper,[image.width,image.height])
        if x1<=x0 or y1<=y0:
            x0,y0,x1,y1=0,0,image.width,image.height
    else:
        x0,y0,x1,y1=0,0,image.width,image.height
    image=image.crop((x0,y0,x1,y1))
    rate=min(755/image.width,350/image.height)
    image=image.resize((round(image.width*rate),round(image.height*rate)),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(1120,400),'#171b23')
    out.paste(image,(12,40))
    d=ImageDraw.Draw(out)
    pen=ImageFont.truetype('DejaVuSans-Bold.ttf',17)
    d.text((12,10),key,font=pen,fill='white')
    d.text((95,10),'Verified reference',font=pen,fill='#34dffc')
    d.text((300,10),'Candidate',font=pen,fill='#ff99ba')
    if missing:
        d.text((495,10),'Projection partly unavailable',font=pen,fill='#d2d7e0')
    palette=('#34dffc','#ff99ba')
    for color,xy in zip(palette,corners_uv,strict=True):
        xy=(xy-[x0,y0])*rate+[12,40]
        for a,b in LINES:
            if np.isfinite(xy[[a,b]]).all():
                d.line([tuple(xy[a]),tuple(xy[b])],fill=color,width=3)
    location=(ref[:2]+query[:2])/2.
    radius=max(.7,np.linalg.norm(ref[:2]-query[:2])/2.+max(ref[3:5].max(),query[3:5].max()))
    def xy_top(points):
        return (points-location)*[145/radius,-145/radius]+[930,213]
    d.rectangle((780,61,1079,363),fill='#26323d')
    cloud = data['points'][np.max(np.abs(data['points'][:,:2]-location),axis=1)<radius]
    for x,y in xy_top(cloud[:,:2]):
        d.point((x,y),fill='#7991a0')
    for color,box in zip(palette,(ref,query),strict=True):
        footprint = xy_top(corners(box)[::2,:2])[[0,1,3,2,0]]
        d.line([tuple(p) for p in footprint],fill=color,width=3)
    return out


def prepare():
    data, truth, inherited = label_truths()
    roster, baseline=source_roster()
    fits = {key: f['rank'] for f in read(FULL/'private/FIT_DETAILS.json') for key in f['cases']}
    assert set(fits)==set(roster)
    STAGING.mkdir(mode=0o700,parents=True,exist_ok=True)
    staged = STAGING/'GEOMETRIC_CANDIDATES.json'
    assert not staged.exists()
    all_labels = [label_pack(x,truth[x.alias],roster[x.alias][1],fits[x.alias]) for x in data]
    by_key = {r['alias']:r for r in all_labels}
    candidates = [(r['alias'],j) for r in all_labels for j in r['shortlisted']]
    # Review a small, preregistered mixture of the four earlier losses, uncertain cases
    # and well-separated candidates. The viewer sees no original case or method.
    designated = {r['case'] for r in read(FULL/'private/WINNER_DETAILS.json')
                  if r['case'] in {'A086','A102','A108','B34'}}
    essential = [ij for ij in candidates if ij[0] in designated]
    selected = essential.copy()
    for i in (False,True):
        eligible = [ij for ij in candidates if ij not in selected and
                    by_key[ij[0]]['geometry'][str(ij[1])]['verified_exclusion']==i]
        selected.extend(sorted(eligible,key=stable)[:max(0,(VISUAL_CASES-len(essential))//2)])
    for ij in sorted(set(candidates)-set(selected),key=stable):
        if len(selected) == VISUAL_CASES:
            break
        selected.append(ij)
    assert len(selected) <= VISUAL_CASES
    plates, group = {}, []
    hidden = {}
    for k,(case,j) in enumerate(sorted(selected,key=stable),1):
        token=f'K{k:02d}'
        hidden[token]=dict(alias=case,slot=j)
        group.append(render(token,truth[case],roster[case][1],j))
        if len(group)==3 or k==len(selected):
            page=Image.new('RGB',(1120,400*len(group)))
            for z,tile in enumerate(group):
                page.paste(tile,(0,z*400))
            name=f'review_{len(plates)+1:02d}.jpg'
            page.save(STAGING/name,quality=91)
            plates[name]=sha(STAGING/name)
            group.clear()
    atomic_json(staged,all_labels)
    atomic_json(STAGING/'BLIND_JOIN_PRIVATE.json',hidden)
    atomic_json(STAGING/'REVIEW_LOCK.json',dict(producer=sha(Path(__file__)),protocol=sha(PROTOCOL),
        data=sha(staged),join=sha(STAGING/'BLIND_JOIN_PRIVATE.json'),previous=inherited,baselines=baseline,
        score_visibility=False, method_visibility=False, protected_data_access=False, plates=plates))
    print('Shortlisted',len(candidates),'blinded audits',len(hidden),
          'geometry confirmed',sum(v['verified_exclusion'] for r in all_labels for v in r['geometry'].values()),flush=True)


def seal():
    lock=read(STAGING/'REVIEW_LOCK.json')
    assert lock['producer']==sha(Path(__file__)) and lock['protocol']==sha(PROTOCOL)
    staged=STAGING/'GEOMETRIC_CANDIDATES.json'
    stitch=STAGING/'BLIND_JOIN_PRIVATE.json'
    assert lock['data']==sha(staged) and lock['join']==sha(stitch)
    assert all(sha(STAGING/n)==h for n,h in lock['plates'].items())
    annotations=STAGING/'BLIND_LABELS.json'
    labels=read(annotations)
    hidden=read(stitch)
    assert set(hidden)==set(labels)
    full=read(staged)
    by_key={r['alias']:r for r in full}
    valid=Counter()
    for key,r in hidden.items():
        value=labels[key]
        assert value in ('single','mixed_background','unknown','veto')
        i=by_key[r['alias']]
        j=r['slot']
        assert i['owner'][j]==-1 and j in i['shortlisted']
        if value=='single':
            i['instance_validity'][j]=1
        elif value=='mixed_background':
            i['instance_validity'][j]=0
        elif value=='veto':
            i['reference_match'][j]=-1
        valid[value]+=1
    result=PRIVATE/'CANDIDATE_LABELS.json'
    output=RESULTS/'SUPERVISION.json'
    assert not result.exists() and not output.exists()
    atomic_json(result,full)
    status={}
    rows=[]
    for r in full:
        first=sum(i>=0 for i in r['owner'])
        decided=[i for i in r['shortlisted'] if r['reference_match'][i]==0]
        rows.append(dict(case=r['alias'],platform=r['platform'],recording=r['recording'],candidates=len(r['owner']),
                         known_owner=first,shortlisted=len(r['shortlisted']),new_negative=len(decided),
                         shortlisted_unknown=len(r['shortlisted'])-len(decided),
                         valid_single=r['instance_validity'].count(1),
                         checked_mixed_or_background=r['instance_validity'].count(0)))
    for source in ('waymo','drone','quad'):
        cases=[r for r in rows if r['platform']==source]
        status[source]=dict(frames=len(cases),shortlisted=sum(r['shortlisted'] for r in cases),
                            new_negative=sum(r['new_negative'] for r in cases),
                            cases_with_new_negative=sum(r['new_negative']>0 for r in cases),
                            verified_negative_recordings=len({r['recording'] for r in cases if r['new_negative']}),
                            remain_unknown=sum(r['shortlisted_unknown'] for r in cases))
    atomic_json(PRIVATE/'LABEL_SEAL.json',dict(table=sha(result),audit=sha(annotations),stage=sha(STAGING/'REVIEW_LOCK.json'),
                   reviewed=len(hidden),producer=sha(Path(__file__)),protected_data_access=False, truth_unchanged=True))
    RESULTS.mkdir(exist_ok=True)
    write_table(RESULTS/'label_coverage.csv',rows)
    atomic_json(output,dict(source=status,blinded_projection_audit=dict(valid),frames=len(full),
                           selection_used_frozen_ranks=True,unknown_unmatched_retained=True,
                           protected_data_access=False))
    print('Verified added competitors by source',status,flush=True)


if __name__ == '__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prepare','seal'))
    prepare() if ap.parse_args().phase=='prepare' else seal()
