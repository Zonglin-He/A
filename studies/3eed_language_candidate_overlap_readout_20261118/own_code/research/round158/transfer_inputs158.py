"""Content disjointness from both observations; selection uses only the ordinary sensor inventory."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'crossplatform/round158'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=read(ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json')
    point_used=set();image_used=set()
    for e in old:
        sensors=[e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else [])
        for f in sensors:
            point_used.add(f['point_sha256']);image_used.add(f['image_sha256'])
    dr=ROOT/'research/round131'
    path=dr/'CONTENT_GROUPS_PRIVATE.json'
    if digest(path)!=read(dr/'experiment/INPUT_LOCK.json')['files'][path.name]:raise ValueError('inventory pin')
    pop=read(path)
    targeted={}
    for t in ('drone','quad','waymo'):
        by_meta=defaultdict(list)
        for r in read(ROOT/'research/round97/input'/('TARGET_'+t.upper()+'_PRIVATE.json')):
            by_meta[str(Path(r['metapath']).resolve())].append(r)
        targeted[t]=by_meta
    reg=[];groups=0
    for g in pop:
        if g['old_development_content'] or g['protected_content_group']:continue
        frames=g['frames']
        if any((not f['eligible_for_extra_media'] or f['protected_image_content_alias'] or
                f['point_sha256'] in point_used or f['image_sha256'] in image_used) for f in frames):continue
        groups+=1;tgt=g['platform']
        for f in frames:
            rr=targeted[tgt][str(Path(f['metapath']).resolve())]
            if not rr:raise ValueError('ordinary frame lacks registered caption')
            for row in rr:
                for src in ('drone','quad','waymo'):
                    if src!=tgt:
                        reg.append(dict(source=src,target=tgt,cohort='EXT',row=row,A_sensor=f,
                                        protected_alias_fallback=False,group=g['content_group_PRIVATE']))
    if len({(e['source'],e['target'],e['row']['id']) for e in reg})!=len(reg):raise ValueError('repeated direction/row')
    p=WORK/'EXTRA_REGISTRY_PRIVATE.json';p.write_text(json.dumps(reg)+'\n')
    seal=dict(n=len(reg),n_groups=groups,sha256=digest(p),control='A only',inventory_sha256=digest(path),
              target_labels_used_for_selection=False,FINAL_access=False,shared_point_or_image_hashes=0)
    (WORK/'EXTRA_INPUT_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(seal)

if __name__=='__main__':main()
