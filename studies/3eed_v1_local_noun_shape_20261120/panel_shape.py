"""Author a challenge in which image side alone cannot identify both tagged subjects."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

from paths_shape import WORK, OUT, SPEC, previous, SOURCES, tag, read, save, digest, rec_key, ordinary, frame, name
from panel_scene import cases
from panel_witness import trials, inventory, projection


def alternatives(classes, px):
    for texts, subjects, ref in cases(classes, px):
        yield 'switch_sign', texts, subjects, [ref, ref]
    for _, data in trials(classes, px):
        yield 'switch_landmark', data['texts'], data['subjects'], data['refs']


def run():
    previous_rows = previous.observed_roster()
    checks = lru_cache(maxsize=None)(digest)
    past_paths = {str(Path(r['frame']['points']).resolve()) for r in previous_rows}
    used = {checks(p) for p in past_paths}
    past_rec = {rec_key(r['frame']) for r in previous_rows}
    labels, rows = {}, []
    counters = Counter()
    for target, table in ordinary().items():
        seen = set()
        groups = defaultdict(list)
        for idx, v in enumerate(table):
            path = str(Path(v['pointpath']).resolve())
            key = (str(Path(v['metapath']).resolve()), int(v['suffix'] or 0))
            if key in seen:
                continue
            seen.add(key)
            counters[target,'all_frame_views'] += 1
            if path in past_paths:
                continue
            classes, boxes, conflict = inventory(v)
            if conflict or len(boxes) < 3:
                continue
            px = projection(boxes,target,v['suffix'],v['metapath'],v['imagepath'])
            options = defaultdict(list)
            for family, texts, sub, ref in alternatives(classes,px):
                if (px[sub[0],0]-.5)*(px[sub[1],0]-.5) <= 0:
                    continue
                whole_half = max(px[sub,3]) < .5 or min(px[sub,2]) > .5
                options[family].append((tag((key,texts)),texts,sub,ref,bool(whole_half)))
            for family, items in options.items():
                _, words, sub, ref, whole_half = min(items)
                groups[family,v['recording']].append((tag((target,idx)),idx,path,words,sub,ref,whole_half))
                counters[target,family+'_eligible_views'] += 1
        for queue in groups.values():
            queue.sort(reverse=True)
        # Prefer landmark replacements: sign and absolute image direction are both constant.
        for family in ('switch_landmark','switch_sign'):
            limit = read(SPEC)['per_target_family_limit']
            count = 0
            for exposure in (False,True):
                recs = sorted([rec for fam,rec in groups if fam==family and ((target,rec) in past_rec)==exposure],
                              key=lambda r:tag((target,r)))
                while count < limit:
                    before=count
                    for recording in recs:
                        queue = groups[family,recording]
                        while queue:
                            _,idx,path,words,sub,ref,whole_half = queue.pop()
                            signature = checks(path)
                            if signature in used:
                                continue
                            used.add(signature)
                            count+=1
                            v = table[idx]
                            pair=tag((target,signature,family))
                            _,boxes,_=inventory(v)
                            for half in (0,1):
                                uid=tag((pair,half))
                                f=dict(frame(v),text=words[half],seed=int(signature[:7],16))
                                f={k:str(x) if isinstance(x,Path) else x for k,x in f.items()}
                                assert (rec_key(f) in past_rec)==exposure
                                labels[uid]=dict(group=family,pair=pair,half=half,subject=sub[half],reference=ref[half],
                                    truth=boxes.tolist(),frame=tag(signature),recording=tag((target,recording)),
                                    exposed_recording=exposure,full_extent_same_half=whole_half)
                                rows.extend(dict(uid=uid,source=s,frame=f) for s in SOURCES if s!=target)
                            counters[target,family+'_old_recording' if exposure else family+'_new_recording']+=1
                            break
                        if count==limit:
                            break
                    if count==before:
                        break
            print(target,family,count,flush=True)
    if len({name(r) for r in rows})!=len(rows):
        raise ValueError('Repeated direction-caption entry')
    save(WORK/'roster.json',rows)
    save(WORK/'labels.json',labels)
    save(OUT/'panel.json',dict(protocol=digest(SPEC),roster=digest(WORK/'roster.json'),labels=digest(WORK/'labels.json'),
        previous_scans=len(past_paths),previous_content_count=len(used)-len(labels)//2,
        counts={'|'.join(k):v for k,v in sorted(counters.items())},new_rows=len(rows)))

if __name__=='__main__':
    run()
