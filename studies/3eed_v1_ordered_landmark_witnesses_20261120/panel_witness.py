"""Create and seal paired evaluation captions independently of frozen model outputs."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import numpy as np
from paths_witness import WORK, LAST, OUT, SPEC, SOURCES, name, read, save, digest, tag
from registry_swap import ordinary, frame
from projection_witness import projection
from score_roles import transform_gt, many_iou

WORDS = {'pedestrian','car','bus','truck','cyclist'}
SIDES = ('left','right')
MARGIN = .015

def inventory(row):
    all_items = []
    for item in read(row['metapath'])['ground_info']:
        all_items.append((item['class'],item['bbox_3d']))
        all_items.extend((obj['class_other'],obj['bbox_3d_other']) for obj in item.get('others',()))
    keys, labels, boxes = {}, [], []
    conflict = False
    target = row['platform']
    for noun, raw in all_items:
        if raw is None or len(raw)<7:
            continue
        key = tuple(map(float,raw[:7]))
        if not np.isfinite(key).all() or min(key[3:6])<=0:
            continue
        noun = noun.lower().strip()
        if key in keys:
            conflict |= labels[keys[key]] != noun
            continue
        box = transform_gt(np.asarray(key if target=='waymo' else (*key,0.,0.)),target,int(row['suffix'] or 0)).astype(float)
        xy = np.flatnonzero(many_iou(np.stack(boxes),box)>.96) if boxes else []
        if len(xy):
            keys[key] = int(xy[0]); conflict |= labels[keys[key]] != noun
        else:
            keys[key] = len(boxes); labels.append(noun); boxes.append(box)
    return np.asarray(labels),np.asarray(boxes),conflict

def nearest(noun,ref,side,classes,px):
    ids = np.flatnonzero((classes==noun) & (px[:,7]>0))
    dist = (px[ids,0]-px[ref,0]) * (-1 if side=='left' else 1)
    ordered = np.argsort(dist,kind='stable')
    ordered = ordered[dist[ordered]>0]
    if not len(ordered):
        return None
    i = ids[ordered[0]]
    gap = px[ref,2]-px[i,3] if side=='left' else px[i,2]-px[ref,3]
    if gap<=MARGIN or (len(ordered)>1 and dist[ordered[1]]-dist[ordered[0]]<=MARGIN):
        return None
    return int(i)

def trials(classes,px):
    seen = defaultdict(list)
    for i in np.flatnonzero(px[:,7]>0):
        if classes[i] in WORDS:
            seen[str(classes[i])].append(i)
    refs = [('name',(a,n),(b,m)) for (n,[a]),(m,[b]) in combinations(
            [(k,v) for k,v in sorted(seen.items()) if len(v)==1],2)]
    for noun, ids in sorted(seen.items()):
        if len(ids)<2: continue
        order = sorted(ids,key=lambda i:px[i,1]); near,far = order[0],order[-1]
        if px[near,5]>=min(px[i,4] for i in order[1:]) or px[far,4]<=max(px[i,5] for i in order[:-1]):
            continue
        if px[near,6]>=min(px[i,6] for i in order[1:]) or px[far,6]<=max(px[i,6] for i in order[:-1]):
            continue
        refs.append(('depth',(near,noun+' closest to the camera'),(far,noun+' farthest from the camera')))
    for family,(a,r1),(b,r2) in refs:
        for noun, inds in sorted(seen.items()):
            if len(inds)<2: continue
            for side in SIDES:
                i,j = (nearest(noun,k,side,classes,px) for k in (a,b))
                if None in (i,j) or len({i,j,a,b})!=4: continue
                texts = ['Find the '+noun+' immediately to the '+side+' of the '+r+' in the camera image.' for r in (r1,r2)]
                yield family, dict(texts=texts, subjects=[i,j], refs=[int(a),int(b)])

def select():
    previous=read(LAST/'private/roster.json')
    previous_paths={str(Path(r['frame']['points']).resolve()) for r in previous}
    cached = lru_cache(maxsize=None)(digest)
    past={cached(p) for p in previous_paths}
    seen=set(past)
    roster=[]; truth={}; coverage=Counter(); allocations=Counter()
    for target,rows in ordinary().items():
        frames={}
        for idx,row in enumerate(rows):
            k=(str(Path(row['metapath']).resolve()),int(row['suffix'] or 0))
            if k not in frames: frames[k]=idx
        good=defaultdict(lambda:defaultdict(list))
        for k,idx in frames.items():
            row=rows[idx]; coverage[target,'screened']+=1
            p=str(Path(row['pointpath']).resolve())
            if p in previous_paths: coverage[target,'previous_path']+=1;continue
            classes,b,conflict=inventory(row)
            if conflict or len(b)<4: coverage[target,'conflicting_or_under_four']+=1;continue
            px=projection(b,target,row['suffix'],row['metapath'],row['imagepath'])
            possible=defaultdict(list)
            for fam,item in trials(classes,px):
                possible[fam].append(item)
            for fam,items in possible.items():
                item=min(items,key=lambda x:tag((k,x['texts'])))
                good[fam][row['recording']].append((tag((target,idx,fam)),idx,p,item))
                coverage[target,'eligible_'+fam]+=1
        for fam in ('name','depth'):
            queues=good[fam]
            recs=sorted(queues,key=lambda k:tag((target,fam,k)))
            for q in queues.values():q.sort(reverse=True)
            n=0
            while n<48:
                before=n
                for recording in recs:
                    q=queues[recording]
                    while q:
                        _,idx,p,item=q.pop()
                        f=cached(p)
                        if f in seen:continue
                        seen.add(f); n+=1
                        row=rows[idx]; pair=tag((target,fam,f))
                        _,boxes,_=inventory(row)
                        for which in (0,1):
                            uid=tag((pair,which))
                            inp=dict(frame(row),text=item['texts'][which],seed=int(f[:7],16))
                            inp={k:str(v) if isinstance(v,Path) else v for k,v in inp.items()}
                            truth[uid]=dict(truth=boxes.tolist(),subject=item['subjects'][which],reference=item['refs'][which],pair=pair,
                                         half=which,group=fam,recording=tag((target,recording)),frame=tag(f))
                            for src in SOURCES:
                                if src!=target: roster.append(dict(uid=uid,source=src,frame=inp))
                        break
                    if n==48:break
                if before==n:break
            allocations[target,fam]=n
        print(target,[(k,v) for k,v in allocations.items() if k[0]==target],flush=True)
    for old in previous:
        if old['panel']=='source_proxy': continue
        target=old['frame']['target']; panel=old['panel']; idx=old['case']
        uid=tag((panel,target,idx))
        roster.append(dict(uid=uid,source=old['source'],frame=old['frame']))
        truth[uid]=dict(group=panel,case=idx,target=target)
    assert len({name(r) for r in roster})==len(roster)
    save(WORK/'roster.json',roster);save(WORK/'labels.json',truth)
    audit=dict(protocol=digest(SPEC),observations=digest(WORK/'roster.json'),labels=digest(WORK/'labels.json'),
        screens={'|'.join(k):v for k,v in coverage.items()},frames={'|'.join(k):v for k,v in allocations.items()},
        total_rows=len(roster),natural_rows=sum(1 for r in previous if r['panel']!='source_proxy'),past_clouds_excluded=len(past))
    save(OUT/'panel.json',audit)
    print(audit,flush=True)

if __name__=='__main__':select()
