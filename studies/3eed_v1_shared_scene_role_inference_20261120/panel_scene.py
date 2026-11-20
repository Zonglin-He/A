"""Choose extra left/right subject replacements from ordinary, with the same named landmark."""
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from shared_paths import *
from registry_swap import ordinary,frame
from panel_witness import WORDS,nearest,inventory,projection

def cases(classes,px):
    visible=px[:,7]>0
    nouns={str(k):np.flatnonzero((classes==k)&visible) for k in sorted(set(classes)&WORDS)}
    refs=[]
    for noun,ids in nouns.items():
        if len(ids)==1:
            refs.append((ids[0],noun));continue
        if len(ids)<2:continue
        order=ids[np.argsort(px[ids,1])]
        for i,rivals,qual,sign in [(order[0],order[1:],' closest to the camera',1),
                                   (order[-1],order[:-1],' farthest from the camera',-1)]:
            if sign==1:
                distinct=px[i,5]<px[rivals,4].min() and px[i,6]<px[rivals,6].min()
            else:
                distinct=px[i,4]>px[rivals,5].max() and px[i,6]>px[rivals,6].max()
            if distinct:refs.append((i,noun+qual))
    for ref,phrase in refs:
        for noun,ids in nouns.items():
            if len(ids)<2:continue
            sides=[nearest(noun,ref,side,classes,px) for side in ('left','right')]
            if None in sides or len({*sides,ref})!=3:continue
            words=['Find the '+noun+' immediately to the '+side+' of the '+phrase+' in the camera image.' for side in ('left','right')]
            yield words,list(map(int,sides)),int(ref)


def run():
    NEW.mkdir(exist_ok=True,mode=0o700)
    rows=panel()
    checksum=lru_cache(maxsize=None)(digest)
    paths={str(Path(r['frame']['points']).resolve()) for r in rows}
    used={checksum(p) for p in paths}
    tables=ordinary()
    # Use the official recording field of all historical ordinary frames, before reading task results.
    rec_by_path={str(Path(v['pointpath']).resolve()):(target,v['recording']) for target,q in tables.items() for v in q}
    before={rec_by_path[p] for p in paths if p in rec_by_path}
    files=[];labels={};counts=Counter()
    for target,table in tables.items():
        seen=set();bins=defaultdict(list)
        for idx,v in enumerate(table):
            key=str(Path(v['metapath']).resolve()),int(v['suffix'] or 0)
            if key in seen:continue
            seen.add(key);counts[target,'scanned']+=1
            path=str(Path(v['pointpath']).resolve())
            if path in paths:continue
            cls,b,conflict=inventory(v)
            if conflict or len(b)<3:continue
            px=projection(b,target,v['suffix'],v['metapath'],v['imagepath'])
            cand=list(cases(cls,px))
            if not cand:continue
            words,s,r=min(cand,key=lambda item:anonymous((key,item[0])))
            bins[v['recording']].append((anonymous((target,idx)),idx,path,words,s,r))
            counts[target,'eligible']+=1
        for queue in bins.values():queue.sort(reverse=True)
        chosen=0
        for exposed in (False,True):
            recs=sorted([r for r in bins if ((target,r) in before)==exposed],key=lambda r:anonymous((target,r)))
            while chosen<32:
                old=chosen
                for rec in recs:
                    while bins[rec]:
                        _,i,path,words,subject,ref=bins[rec].pop()
                        tag=checksum(path)
                        if tag in used:continue
                        used.add(tag);chosen+=1;counts[target,('reused_recording' if exposed else 'new_recording')]+=1
                        v=table[i];pair=anonymous((target,tag))
                        cls,b,_=inventory(v)
                        for half in range(2):
                            uid=anonymous((pair,half));f=dict(frame(v),text=words[half],seed=int(tag[:7],16))
                            f={k:str(x) if isinstance(x,Path) else x for k,x in f.items()}
                            labels[uid]=dict(group='new_side',pair=pair,half=half,subject=subject[half],reference=ref,
                                truth=b.tolist(),frame=anonymous(tag),recording=anonymous((target,rec)),
                                exposed_recording=exposed,target=target)
                            for src in SOURCES:
                                if src!=target:files.append(dict(uid=uid,source=src,frame=f))
                        break
                    if chosen==32:break
                if old==chosen:break
        print(target,chosen,flush=True)
    assert len({name(r) for r in files})==len(files)
    save(NEW/'roster.json',files);save(NEW/'labels.json',labels)
    save(OUT/'panel.json',dict(observations=digest(NEW/'roster.json'),labels=digest(NEW/'labels.json'),
         protocol=digest(SPEC),extra_records=len(files),old_records=len(rows),old_scans=len(paths),
         historical_recordings_linked=len(before),historical_paths_linked=len(paths & rec_by_path.keys()),
         counts={'|'.join(k):v for k,v in counts.items()}))

if __name__=='__main__':run()
