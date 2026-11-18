"""Small fixed, prediction-blind textual and image review of reference identities."""
import argparse
from collections import Counter,defaultdict,deque
from functools import lru_cache
import hashlib
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from scope166 import ARMS,BLOCK,HERE,ORDER,PROTOCOL,WORK,atomic_json,read,sha
from score165 import anonymizer,table,write_table
from score166 import PAIR_ARMS,verified_prediction_inputs
from labels166 import joint_status,label_pair,occupied_owners,physical_labels,truth_status
from visual162 import corners,LINES,projection

PANEL=WORK/'reference_review_PRIVATE'
PLAN=HERE/'experiment/REVIEW_PLAN.json'
TRUTH=WORK/'REFERENCE_ADJUDICATION_PRIVATE.json'


def sample(group,count):
    physical={}
    salt=b'R166-proxy-identity-review-v1'
    def key(r):
        return hashlib.sha256(salt+r['expression'].encode()).digest()
    for r in sorted(group,key=lambda x:(x['expression'],x['source'],int(x['edge_number']))):
        physical.setdefault(r['expression'],r)
    clusters=defaultdict(deque)
    for row in sorted(physical.values(),key=key):
        clusters[row['recording']].append(row)
    captures=list(clusters.values())
    out=[]
    while len(out)<count and any(captures):
        for q in captures:
            if q:
                out.append(q.popleft())
                if len(out)==count:
                    break
    return out


@lru_cache(maxsize=4)
def open_block(source,target,which):
    p=WORK/'ordinary'/f'{source}_to_{target}'
    name=f'{which:04d}.npz'
    assert sha(p/name)==read(p/'SEAL.json')['files'][name]
    with np.load(p/name,allow_pickle=False) as f:
        return dict(f)


def evidence(source,target,ordinal,edge_number):
    data=open_block(source,target,ordinal//BLOCK)
    i=ordinal%BLOCK
    assert int(data['ordinals'][i])==ordinal
    b={k:data[k][i] for k in ('seeds','boxes','spans','diagnostic','camera_intrinsic','model_to_camera','distortion','width')}
    e=data['edges'][data['edge_offsets'][i]:data['edge_offsets'][i+1]]
    direct=list(dict.fromkeys((int(z if a==0 else a),int(k if a==0 else -k)) for a,z,k in e if (a==0)!=(z==0)))
    return b,e,direct[edge_number-1]


def render(case,row,pointer,labeled):
    with Image.open(row['imagepath']) as im:
        rgb=im.convert('RGB')
    w,h=rgb.size
    rgb.thumbnail((1350,740))
    coord=SimpleNamespace(camera_intrinsic=pointer['camera_intrinsic'],model_to_camera=pointer['model_to_camera'],
                          distortion=pointer['distortion'],width=w,height=h)
    font=ImageFont.truetype('DejaVuSans.ttf',16)
    small=ImageFont.truetype('DejaVuSans.ttf',13)
    draw=ImageDraw.Draw(rgb)
    for i,b in enumerate(labeled):
        vertices=projection(coord,corners(b))*[rgb.width,rgb.height]
        if not np.isfinite(vertices).any():
            continue
        col='#46ff83' if i==0 else '#ffd462'
        for a,c in LINES:
            if np.isfinite(vertices[[a,c]]).all():
                draw.line([tuple(vertices[a]),tuple(vertices[c])],fill=col,width=2)
        xy=projection(coord,b[None,:3])[0]*[rgb.width,rgb.height]
        if (0<=xy[0]<rgb.width) and (0<=xy[1]<rgb.height):
            caption=f'{i}'
            draw.rectangle((xy[0]-3,xy[1]-8,xy[0]+9+6*len(caption),xy[1]+13),fill='#182322')
            draw.text(tuple(xy),caption,fill=col,font=small,anchor='lm')
    desc=textwrap.wrap(row['caption'],145)
    legend=textwrap.wrap('   '.join(f'{i} {k}' for i,k in enumerate(case['categories'])),140)
    height=62+(len(desc)+len(legend))*22
    canvas=Image.new('RGB',(1350,rgb.height+height),'#f4f5f4')
    canvas.paste(rgb,((1350-rgb.width)//2,height))
    page=ImageDraw.Draw(canvas)
    lines=[f"Case {case['case']:02} · {case['target']} · reference span: {case['reference_phrase']}",*desc,*legend]
    for i,l in enumerate(lines):
        page.text((16,12+i*22),l,fill='#171717',font=font)
    canvas.save(PANEL/f"{case['case']:02}.jpg",quality=90)


def prepare():
    ordinary,seal=verified_prediction_inputs()
    edge_table=HERE/'results/relation_edges.csv'
    assert sha(edge_table)==read(HERE/'EVALUATION_AUDIT.json')['files'][edge_table.name]
    aliases=anonymizer()
    by_key={}
    for target,rows in ordinary.items():
        for i,row in enumerate(rows):
            path=Path(row['metapath']).resolve()
            key=aliases('obs',(str(path),row['object_index']))
            assert (target,key) not in by_key
            by_key[target,key]=i
    edges=table(edge_table)
    PANEL.mkdir(parents=True,exist_ok=True)
    cases=[]
    for target in ('waymo','drone','quad'):
        subset=[e for e in edges if e['target']==target and e['truth']!='visual_verified']
        used=set()
        for eligible in (True,False):
            take=sample([e for e in subset if (e['truth']=='annotation_proxy')==eligible and e['expression'] not in used],3)
            for item in take:
                s=item['source']
                i=by_key[target,item['expression']]
                row=ordinary[target][i]
                n=int(item['edge_number'])
                b,edges0,(role,side)=evidence(s,target,i,n)
                names,labels,_,_=physical_labels(row,b['seeds'])
                lo,hi=b['spans'][role]
                case=dict(case=len(cases)+1,target=target,source=s,ordinal=i,edge_number=n,
                          expression=item['expression'],reference_phrase=row['caption'][lo:hi],categories=names,
                          selection_bin='proxy' if eligible else 'unresolved')
                render(case,row,b,labels)
                cases.append(case)
                used.add(item['expression'])
    registry=PANEL/'REGISTRY_PRIVATE.json'
    atomic_json(registry,cases)
    atomic_json(HERE/'REVIEW_PANEL_SEAL.json',dict(n=len(cases),digest=sha(registry),plan=sha(PLAN),
                prediction_seal=sha(HERE/'PREDICTION_SEAL.json'),counts=dict(Counter(c['target'] for c in cases)),
                boxes_and_model_outcomes_hidden=True,FINAL_access=False))
    print('Reference identity panel prepared',len(cases),flush=True)


def score():
    rows,_=verified_prediction_inputs()
    refs=read(PANEL/'REGISTRY_PRIVATE.json')
    manual=read(TRUTH)
    assert sha(PANEL/'REGISTRY_PRIVATE.json')==read(HERE/'REVIEW_PANEL_SEAL.json')['digest']
    assert [c['case'] for c in refs]==[c['case'] for c in manual]
    results=[]
    accuracy=Counter()
    for case,judgment in zip(refs,manual,strict=True):
        t,i,n=(case[k] for k in ('target','ordinal','edge_number'))
        row=rows[t][i]
        ref=judgment.get('reference',-1)
        accuracy[judgment['status']]+=1
        assert (0<ref<len(case['categories']))==(judgment['status']=='verified_boxes')
        for s,to in ORDER:
            if to!=t:
                continue
            b,e,(r,side)=evidence(s,t,i,n)
            names,labeled,seeds,seed_ids=physical_labels(row,b['seeds'])
            assert names==case['categories']
            item=dict(case=case['case'],source=s,target=t,review=judgment['status'],label_selection_bin=case['selection_bin'])
            if ref>0:
                packed=SimpleNamespace(edges=e,role_spans=b['spans'],width=b['width'],camera_intrinsic=b['camera_intrinsic'],
                                      model_to_camera=b['model_to_camera'],distortion=b['distortion'])
                source0,auto=label_pair(row,packed,0,r,side,names,labeled)
                item.update(annotation_proxy_present=int(auto>0 and source0=='annotation_proxy'),
                            annotation_proxy_matches_visual=int(auto==ref and source0=='annotation_proxy'))
                prediction=np.concatenate((b['boxes'],b['diagnostic'][None]),axis=0)
                identities=occupied_owners(prediction,seeds,seed_ids,len(names))
                for k,owners in zip(PAIR_ARMS,identities):
                    first,second=truth_status(owners[0],0),truth_status(owners[r],ref)
                    item.update({k+'_both':'' if k=='r158_pca' else joint_status(first,second)})
            else:
                item.update(annotation_proxy_present='',annotation_proxy_matches_visual='',**{a+'_both':'' for a in PAIR_ARMS})
            results.append(item)
    out=HERE/'results/visual_review.csv'
    write_table(out,results)
    x=[r for r in results if r['review']=='verified_boxes']
    metrics={a:{'n':len(x),'correct':sum(r[a+'_both']==1 for r in x)} for a in PAIR_ARMS if a!='r158_pca'}
    atomic_json(HERE/'VISUAL_REVIEW_AUDIT.json',dict(review_cases=len(refs),counts=dict(accuracy),
                verified_direction_edges=len(x),results=metrics,proxy_matches=sum(r['annotation_proxy_matches_visual'] for r in x),
                proxy_eligible=sum(r['annotation_proxy_present'] for r in x),out_sha256=sha(out),FINAL_access=False))
    print('Visual reference identities scored',dict(accuracy),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('prepare','score'))
    (prepare if p.parse_args().phase=='prepare' else score)()
