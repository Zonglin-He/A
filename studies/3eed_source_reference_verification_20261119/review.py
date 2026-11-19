"""Make source identity plates with no prediction, then seal the blind choices."""
from collections import Counter
from pathlib import Path
import argparse
import math
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scope import HERE, PLATFORMS, PRIVATE, SPEC, atomic_json, read, sha, stable


COLORS = ('#42d9ff','#ffd743','#80fa8c','#ff9dd0','#a497ff','#ff9d54','#ffeeeb')
WIDTH=1120
ROWS_PER_PAGE=4


def confirmed_prep(source):
    p = PRIVATE/'prep'/f'{source}.json'
    receipt = read(p.with_name(source+'_seal.json'))
    assert receipt['protocol']==sha(SPEC) and not receipt['predictions_used'] and not receipt['protected_data_access']
    assert receipt['file']==sha(p)
    rows=read(p)
    assert receipt['counts'].get('invitations',0)==len(rows)
    return rows


def bounding(xy,from_x,from_y,pad):
    low = np.maximum(np.floor(xy.min(0)-pad),0).astype(int)
    high = np.minimum(np.ceil(xy.max(0)+pad),[from_x,from_y]).astype(int)
    if (high<=low).any():
        return (0,0,from_x,from_y)
    return (*low,*high)


def render(entry, mapping):
    from calibrate168 import image_geometry
    from mechanisms162 import annotation_inventory
    from visual162 import LINES, corners, projection

    r=entry['row']
    path=Path(r['imagepath'])
    assert sha(path)==r['image_sha256']
    cam=image_geometry(r)
    names,boxes=annotation_inventory(r['metapath'],r['platform'],int(r['suffix'] or 0),r['object_index'])
    assert len(names)==entry['inventory_size']
    with Image.open(path) as pic:
        raw=pic.convert('RGB')
    font=ImageFont.truetype('DejaVuSans.ttf',14)
    bold=ImageFont.truetype('DejaVuSans-Bold.ttf',17)
    m=list(mapping)
    ids=[0,*[mapping[k] for k in m]]
    polys=[projection(cam,corners(boxes[i]))*[raw.width,raw.height] for i in ids]
    assert all(np.isfinite(p).all() for p in polys[1:])
    extent=np.vstack([p[np.isfinite(p).all(1)] for p in polys])
    box=bounding(extent,raw.width,raw.height,max(25,.10*np.ptp(extent,axis=0).max()))
    view=raw.crop(box)
    scale=min(680/view.width,230/view.height)
    view=view.resize((round(view.width*scale),round(view.height*scale)),Image.Resampling.LANCZOS)
    overlay=ImageDraw.Draw(view)
    occupied=[]
    for i,poly in enumerate(polys):
        color=COLORS[i%len(COLORS)]
        vs=(poly-np.asarray(box[:2]))*scale
        for a,b in LINES:
            if np.isfinite(vs[[a,b]]).all():
                overlay.line([tuple(vs[a]),tuple(vs[b])],fill=color,width=2)
        center=projection(cam,boxes[ids[i]:ids[i]+1,:3])[0]*[raw.width,raw.height]
        x,y=(center-np.asarray(box[:2]))*scale
        if not np.isfinite([x,y]).all():
            continue
        tag='S' if i==0 else m[i-1]
        label_location = next(((max(0,min(view.width-24,int(x+dx))),max(0,min(view.height-24,int(y+dy))))
                               for dx,dy in [(0,-23),(18,-3),(-25,-13),(3,7),(-20,9)]
                               if all(abs(x+dx-a)>24 or abs(y+dy-b)>24 for a,b in occupied)),(int(x),int(y)))
        lx,ly=label_location
        occupied.append(label_location)
        overlay.line([(x,y),(lx+9,ly+9)],fill=color,width=2)
        overlay.rectangle((lx,ly,lx+23,ly+22),fill='#16191f',outline=color)
        overlay.text((lx+5,ly),tag,font=bold,fill=color)

    caption=textwrap.wrap(r['caption'],width=145)
    height=60+len(caption)*18+max(234,74*math.ceil(len(m)/3))
    out=Image.new('RGB',(WIDTH,height),'#f7f8fb')
    d=ImageDraw.Draw(out)
    d.text((12,8),f"{entry['case']}    Ref phrase    {entry['views']['C']['text']}",font=bold,fill='#171923')
    for line,message in enumerate(caption):
        d.text((12,37+line*18),message,font=font,fill='#242634')
    y0=48+len(caption)*18
    out.paste(view,(12,y0))
    overview=raw.copy()
    overview.thumbnail((158,70),Image.Resampling.BILINEAR)
    out.paste(overview,(500,y0+max(0,230-overview.height)))
    for j,tag in enumerate(m):
        poly=polys[j+1]
        crop=raw.crop(bounding(poly,raw.width,raw.height,5))
        scale=min(118/crop.width,48/crop.height)
        crop=crop.resize((max(1,round(scale*crop.width)),max(1,round(scale*crop.height))),Image.Resampling.BICUBIC)
        x=701+136*(j%3)
        y=y0+74*(j//3)
        d.text((x,y),f'{tag}  {entry["reference_class"]}',font=font,fill='#242634')
        out.paste(crop,(x,y+18))
        d.rectangle((x,y+18,x+crop.width,y+18+crop.height),outline=COLORS[(j+1)%len(COLORS)],width=2)
    return out


def prepare(source):
    rows=confirmed_prep(source)
    home=PRIVATE/'review'/source
    home.mkdir(parents=True,mode=0o700,exist_ok=True)
    assert not (home/'KEY.json').exists()
    mapping={}
    files={}
    current=[]
    for row in rows:
        ranked=sorted(row['choices'],key=lambda i:stable('review-blind-role-v1',row['case'],i))
        frame={f'P{k+1}':i for k,i in enumerate(ranked)}
        mapping[row['case']]=frame
        card=render(row,frame)
        file=home/(row['case']+'.jpg')
        card.save(file,quality=93)
        files[file.name]=sha(file)
        current.append(card)
        if len(current)==ROWS_PER_PAGE or row is rows[-1]:
            page=Image.new('RGB',(WIDTH,sum(c.height for c in current)+len(current)*5),'#d8dce8')
            y=0
            for c in current:
                page.paste(c,(0,y))
                y+=c.height+5
            file=home/f'sheet_{(len(mapping)-1)//ROWS_PER_PAGE+1:02d}.jpg'
            page.save(file,quality=93)
            files[file.name]=sha(file)
            current=[]
    atomic_json(home/'KEY.json',mapping)
    atomic_json(home/'PLATE_SEAL.json',dict(pages=files,keys=sha(home/'KEY.json'),source=source,
        input=sha(PRIVATE/'prep'/f'{source}.json'),model_predictions_shown=False,protected_data_access=False))


def seal():
    labels=PRIVATE/'review'/'BLIND_CHOICES.json'
    opinion=read(labels)
    all_cases=[]
    statuses=Counter()
    with_foils=Counter()
    maps={}
    media=set()
    for source in PLATFORMS:
        home=PRIVATE/'review'/source
        p=read(home/'PLATE_SEAL.json')
        key=read(home/'KEY.json')
        assert not p['model_predictions_shown'] and not p['protected_data_access']
        assert sha(home/'KEY.json')==p['keys'] and all(sha(home/n)==d for n,d in p['pages'].items())
        maps[source]=sha(home/'PLATE_SEAL.json')
        for row in confirmed_prep(source):
            case=row['case']
            item=opinion[case]
            choice=item['choice']
            assert choice=='UNKNOWN' or choice in key[case]
            assert item['basis'] in ('distinct_appearance','distinct_relation_or_position','ambiguous','unclear_or_contradictory')
            known=choice!='UNKNOWN'
            assert known==(item['basis'] in ('distinct_appearance','distinct_relation_or_position'))
            foils=item['foils']
            assert len(foils)==len(set(foils)) and all(p in key[case] and p!=choice for p in foils)
            assert known or not foils
            row['reference_identity']=key[case][choice] if known else -1
            row['verified_foil_ids']=[key[case][p] for p in foils]
            assert set(row['verified_foil_ids']) <= set(row['choices'])
            row['verification_status']=item['basis']
            statuses[(source,item['basis'])]+=1
            with_foils[source]+=bool(foils)
            all_cases.append(row)
            for m in ('point_sha256','image_sha256'):
                x=row['row'][m]
                assert x not in media
                media.add(x)
    assert set(opinion)=={x['case'] for x in all_cases}
    result=PRIVATE/'VERIFIED_ROSTER.json'
    assert not result.exists()
    atomic_json(result,all_cases)
    atomic_json(PRIVATE/'TRUTH_SEAL.json',dict(file=sha(result),choices=sha(labels),plates=maps,
                review_program=sha(Path(__file__)),visually_verified_foil_frames=dict(with_foils),
                single_internal_blind_reviewer=True,model_results_open_before_adjudication=False,
                n=len(all_cases),statuses={s:{k:n for (p,k),n in statuses.items() if p==s} for s in PLATFORMS},
                protected_data_access=False))
    print('Sealed blind identity decisions',dict(statuses),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=(*PLATFORMS,'seal'))
    p=cli.parse_args().phase
    seal() if p=='seal' else prepare(p)
