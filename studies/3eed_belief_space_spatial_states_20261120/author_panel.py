"""Author noun-only questions using ordinary annotations. Do not inspect model outputs."""
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import sys
import numpy as np
from io_scene import ROOT, PRIVATE, OUT, SPEC, digest, key, read, write, PLATFORMS

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'studies/3eed_eight_scale_component_ablation_20261120'))
from common import prior
from scope183 import ordinary, frame
from panel_witness import inventory
from crossplatform.physical_regions.regions import box_projection
from crossplatform.physical_regions.language import CLASSES

def run():
    cache_hash = lru_cache(maxsize=None)(digest)
    examples, answers, stats = [], {}, {}
    used=set()
    for target, table in ordinary().items():
        # A physical scan can occur under several descriptions or cameras.
        paths = defaultdict(list)
        for row in table:
            paths[str(Path(row['pointpath']).resolve())].append(row)
        groups = defaultdict(list)
        for path, rows in paths.items():
            # Works with multiple file paths for the same bytes (including duplicate copies).
            signature = cache_hash(path)
            groups[signature].extend(rows)
        queues = defaultdict(list)
        for signature, rows in groups.items():
            row = min(rows, key=lambda r:key('view',signature,r['suffix'],r['seed']))
            queues[row['recording']].append((signature,row))
        queues0 = queues
        stats[target]={}
        geometric = {}
        def metadata(row, signature):
            if signature not in geometric:
                classes, boxes, conflict = inventory(row)
                visible = np.zeros(len(boxes), bool)
                if not conflict and len(boxes)>=2:
                    px = box_projection(boxes,target,row['suffix'],row['metapath'],row['imagepath'])
                    visible = px[:,7]>0
                geometric[signature] = classes, boxes, conflict, visible
            return geometric[signature]
        for cohort,quota in [('unique',48),('height',24)]:
            queues={r:deque(sorted(v,key=lambda z:key('draw',target,z[0]) if cohort=='unique' else key('height_draw',target,z[0]))) for r,v in queues0.items()}
            order=sorted(queues,key=lambda r:key('rec',target,r))
            counts=Counter(); count=0
            while count<quota and any(queues.values()):
                for rec in order:
                    while queues[rec]:
                        signature,row=queues[rec].popleft()
                        if signature in used:continue
                        counts['screened']+=1
                        classes,boxes,conflict,visible=metadata(row,signature)
                        if conflict or len(boxes)<2:
                            counts['conflict_or_fewer_than_two_boxes']+=1
                            continue
                        grouped = {c:np.flatnonzero((classes==c)&visible) for c in CLASSES[:-1]}
                        higher=0
                        if cohort=='unique':
                            nouns=sorted(c for c,ids in grouped.items() if len(ids)==1)
                            if len(nouns)<2:
                                counts['fewer_than_two_unique_visible_nouns']+=1
                                continue
                            s,r=min(permutations(nouns,2),key=lambda v:key('roles',signature,*v))
                            si,ri=int(grouped[s][0]),int(grouped[r][0])
                        else:
                            nouns=[c for c,ids in grouped.items() if len(ids)==2 and abs(np.diff(boxes[ids,5])[0])>.10]
                            if not nouns:
                                counts['no_unique_height_pair']+=1
                                continue
                            s=r=min(nouns,key=lambda c:key('heightclass',signature,c))
                            low,high=grouped[s][np.argsort(boxes[grouped[s],5])]
                            higher=1 if int(key('higher',signature),16)%2 else -1
                            si,ri=map(int,(high,low) if higher>0 else (low,high))
                        uid=key('physical',target,signature)
                        used.add(signature); assert uid not in answers
                        f=frame(row); f.pop('text')
                        examples.append(dict(uid=uid,recording=key('recording',target,rec),cohort=cohort,
                            subject=s,reference=r,higher=higher,frame=f))
                        answers[uid]=dict(boxes=boxes.tolist(),s=si,r=ri)
                        count+=1
                        break
                    if count==quota:break
            stats[target][cohort]=dict(available_physical=len(groups),selected=count,
                recordings=len({a['recording'] for a in examples if a['frame']['target']==target and a['cohort']==cohort}),
                **counts)
        print(target, stats[target], flush=True)
    assert len(answers)==len(examples) and [stats[t]['unique']['selected'] for t in PLATFORMS]==[48,23,0] and all(stats[t]['height']['selected'] for t in PLATFORMS) and len(answers)<=144
    write(PRIVATE/'questions.json',examples)
    write(PRIVATE/'answer_key.json',answers)
    write(OUT/'panel_seal.json',dict(protocol=digest(SPEC),questions=digest(PRIVATE/'questions.json'),
          author=digest(__file__),truth=digest(PRIVATE/'answer_key.json'),platforms=stats,
          model_outputs_accessed=False,FINAL_access=False))
if __name__=='__main__':run()
