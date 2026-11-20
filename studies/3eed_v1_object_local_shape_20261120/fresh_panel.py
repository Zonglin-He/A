"""Author content-disjoint trials with the predecessor's exact single-noun eligibility rules."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
from local_paths import *
from panel_nouns import options
from panel_witness import inventory,projection
from scope183 import ordinary,frame
from shared_paths import anonymous

EXTRA=WORK/'fresh'
def make():
    parent=catalog();table=ordinary();hashed=lru_cache(maxsize=None)(digest)
    old_paths={str(Path(r[0]['frame']['points']).resolve()) for r in old.observations()}
    nouns=previous.noun_inputs();noun_paths={str(Path(r['frame']['points']).resolve()) for r in nouns}
    used={hashed(k) for k in old_paths|noun_paths}
    noun_recordings={(s,v['recording']) for s,items in table.items() for v in items if str(Path(v['pointpath']).resolve()) in noun_paths}
    counter=Counter();rows=[];labels={};known=set()
    for tgt,items in table.items():
        for v in items:
            key=tgt,str(v['metapath']),v['suffix']
            if key in known:continue
            known.add(key);counter[tgt+'_views']+=1
            p=str(Path(v['pointpath']).resolve())
            if p in old_paths or p in noun_paths:continue
            cls,gt,bad=inventory(v)
            if bad or len(gt)<4:continue
            proj=projection(gt,tgt,v['suffix'],v['metapath'],v['imagepath']);possible=list(options(cls,proj))
            if not possible:continue
            h=hashed(p)
            if h in used:continue
            used.add(h);counter[tgt+'_frames']+=1
            texts,subjects,refs,same=min(possible,key=lambda a:anonymous(('separate-shape',key,a[0])))
            pair=anonymous(('local-residual-pair',h));rec=anonymous((tgt,v['recording']))
            for half,text in enumerate(texts):
                uid=anonymous((pair,half));f=dict(frame(v),seed=int(h[:7],16),text=text)
                for src in SOURCES:
                    if src!=tgt:rows.append(dict(uid=uid,source=src,frame=f))
                labels[uid]=dict(pair=pair,half=half,truth=gt.tolist(),subject=int(subjects[half]),reference=int(refs[half]),
                    frame=anonymous(h),recording=rec,prior_noun_recording=int((tgt,v['recording']) in noun_recordings),same_half=bool(same))
    EXTRA.mkdir(mode=0o700,exist_ok=True);save(EXTRA/'input.json',rows);save(EXTRA/'score_only.json',labels)
    save(OUT/'fresh_panel.json',dict(protocol=digest(HERE/'FRESH_PROTOCOL.json'),code=digest(Path(__file__)),roster=digest(EXTRA/'input.json'),
        scoring=digest(EXTRA/'score_only.json'),direction_records=len(rows),physical_frames=len(labels)//2,counts=dict(counter),
        recordings=len({v['recording'] for v in labels.values()}),outside_old_noun_recordings=len({v['frame'] for v in labels.values() if not v['prior_noun_recording']})))
    print('new physical frames',len(labels)//2,'direction records',len(rows),flush=True)
if __name__=='__main__':make()
