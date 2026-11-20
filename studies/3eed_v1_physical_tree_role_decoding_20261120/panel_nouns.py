"""Author matched single-word counterfactuals without reading model predictions."""
from collections import Counter,defaultdict
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import numpy as np
from paths_tree import *
from scope183 import ordinary,frame
from panel_witness import inventory,projection,nearest,WORDS
from shared_paths import anonymous

EXTRA=WORK/'noun_only'
PIN=OUT/'noun_pair_protocol.json'

def options(classes,px):
    grouped=defaultdict(list)
    for i in np.flatnonzero(px[:,7]>0):
        if classes[i] in WORDS:grouped[str(classes[i])].append(int(i))
    for sign,qual in [(1,'closest'),(-1,'farthest')]:
        refs=[]
        for noun,items in sorted(grouped.items()):
            ids=sorted(items,key=lambda i:sign*px[i,1]);r=ids[0]
            if len(ids)>1:
                near_ok=px[r,5]<min(px[i,4] for i in ids[1:]) and px[r,6]<min(px[i,6] for i in ids[1:])
                far_ok=px[r,4]>max(px[i,5] for i in ids[1:]) and px[r,6]>max(px[i,6] for i in ids[1:])
                if not (near_ok if sign==1 else far_ok):continue
            refs.append((r,noun))
        for (r0,n0),(r1,n1) in combinations(refs,2):
            for subject,items in sorted(grouped.items()):
                if len(items)<2:continue
                for side in ('left','right'):
                    a,b=[nearest(subject,r,side,classes,px) for r in (r0,r1)]
                    if a is None or b is None or len({a,b,r0,r1})!=4:continue
                    text=['Find the '+subject+' immediately to the '+side+' of the '+n+' '+qual+
                          ' to the camera in the camera image.' for n in (n0,n1)]
                    # The registered parser uses "from the camera" after "farthest".
                    if sign<0:text=[t.replace('farthest to the','farthest from the') for t in text]
                    assert sum(u!=v for u,v in zip(text[0].split(),text[1].split()))==1
                    yield text,[a,b],[r0,r1],bool((px[a,0]-.5)*(px[b,0]-.5)>0)

def run():
    assert read(PIN)['parent']==digest(PRED/'SEALED.json')
    EXTRA.mkdir(exist_ok=True,mode=0o700)
    oldrows=observations();hashpath=lru_cache(maxsize=None)(digest)
    oldpaths={str(Path(row[0]['frame']['points']).resolve()) for row in oldrows}
    used={hashpath(path) for path in oldpaths};seenrecs=set()
    tables=ordinary()
    for target,items in tables.items():
        for item in items:
            if str(Path(item['pointpath']).resolve()) in oldpaths:seenrecs.add((target,item['recording']))
    labels={};roster=[];counter=Counter()
    for tgt,items in tables.items():
        seen=set();queues=defaultdict(list)
        for ix,v in enumerate(items):
            view=(v['metapath'],v['suffix'])
            if view in seen:continue
            seen.add(view);counter[tgt,'views']+=1
            if str(Path(v['pointpath']).resolve()) in oldpaths:continue
            classes,b,conflict=inventory(v)
            if conflict or len(b)<4:continue
            px=projection(b,tgt,v['suffix'],v['metapath'],v['imagepath'])
            potential=list(options(classes,px))
            if not potential:continue
            trial=min(potential,key=lambda u:anonymous((view,u[0])))
            queues[v['recording']].append((anonymous((view,trial[0])),ix,trial))
            counter[tgt,'eligible_views']+=1
        for q in queues.values():q.sort(reverse=True)
        total=0
        for exposed in (False,True):
            recs=sorted([rec for rec in queues if ((tgt,rec) in seenrecs)==exposed],key=anonymous)
            while total<32:
                before=total
                for rec in recs:
                    while queues[rec]:
                        _,ix,(texts,subjects,refs,same)=queues[rec].pop();v=items[ix];h=hashpath(v['pointpath'])
                        if h in used:continue
                        used.add(h);total+=1;counter[tgt,'chosen']+=1;counter[tgt,'previous_recording']+=int(exposed)
                        _,gt,_=inventory(v);pair=anonymous(('noun-only',tgt,h))
                        for half in range(2):
                            uid=anonymous((pair,half));f=dict(frame(v),text=texts[half],seed=int(h[:7],16))
                            labels[uid]=dict(group='noun_only',pair=pair,half=half,truth=gt.tolist(),subject=subjects[half],
                                reference=refs[half],frame=anonymous(h),recording=anonymous((tgt,rec)),same_half=same,
                                previously_sampled_recording=exposed)
                            for src in SOURCES:
                                if src!=tgt:roster.append(dict(uid=uid,source=src,frame=f))
                        break
                    if total==32:break
                if before==total:break
        print(tgt,total,flush=True)
    save(EXTRA/'roster.json',roster);save(EXTRA/'labels.json',labels)
    save(OUT/'noun_panel.json',dict(protocol=digest(PIN),code=digest(HERE/'panel_nouns.py'),n=len(roster),pairs=len(labels)//2,
        observations=digest(EXTRA/'roster.json'),labels=digest(EXTRA/'labels.json'),
        counts={'|'.join(k):v for k,v in counter.items()},training=False,FINAL_access=False))
if __name__=='__main__':run()
