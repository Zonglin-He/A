"""Select from the sealed ordinary roster without using GT or model scores."""
from collections import defaultdict, deque
from functools import lru_cache
import sys
from pathlib import Path
import ijson
from paths import ROOT, PRIV, OUT, SPEC, digest, key, write

sys.path.insert(0,str(ROOT/'studies/3eed_v2_ordinary_val_subject_ranking_20261119'))
from scope183 import ordinary, canonical

def main():
    PRIV.mkdir(exist_ok=True, mode=0o700)
    OUT.mkdir(exist_ok=True)
    physical_hash=lru_cache(None)(digest)
    roster=[]
    for platform,rows in ordinary().items():
        groups=defaultdict(list)
        for j,row in enumerate(rows):
            groups[physical_hash(row['pointpath'])].append((j,row))
        by_rec=defaultdict(list)
        for signature,options in groups.items():
            j,row=min(options,key=lambda item:key('frozenq-utterance',platform,signature,item[1]['seed']))
            by_rec[row['recording']].append((signature,j,row))
        pool={rec:deque(sorted(rc,key=lambda x:key('frozenq-frame',platform,x[0]))) for rec,rc in by_rec.items()}
        count=0
        while count<32:
            for rec in sorted(pool,key=lambda k:key('frozenq-recording',platform,k)):
                if not pool[rec]:continue
                signature,j,row=pool[rec].popleft()
                with Path(row['metapath']).open('rb') as f:
                    text=next(s for k,s in enumerate(ijson.items(f,'ground_info.item.caption')) if k==row['object_index'])
                assert canonical(text)==row['caption']
                roster.append(dict(uid=key('frozenq-id',platform,signature),recording=key('frozenq-rec',platform,rec),
                                   target=platform,ordinal=j,slot=row['object_index'],rgb=row['imagepath'],points=row['pointpath'],
                                   calibration=row['metapath'],suffix=row['suffix'],text=text,point_sha=signature,
                                   rgb_sha=physical_hash(row['imagepath'])))
                count+=1
                if count==32:break
    assert len(roster)==len({r['point_sha'] for r in roster})==96
    write(PRIV/'panel.json',roster)
    write(OUT/'panel_seal.json',dict(protocol=digest(SPEC),panel=digest(PRIV/'panel.json'),
        frame_count=len(roster),recordings={t:len({r['recording'] for r in roster if r['target']==t}) for t in ordinary()},
        selection='physical content, recording, deterministic hash; no box labels or method scores',FINAL_access=False))

if __name__=='__main__':main()
