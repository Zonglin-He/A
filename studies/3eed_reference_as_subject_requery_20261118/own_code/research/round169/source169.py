"""Examine additional source annotation associations with no model output in the plates."""
import argparse
from collections import Counter, defaultdict
import hashlib

from PIL import Image

from scope169 import CAL, HERE, OUT, PLATFORMS, R168, SOURCE_PANEL, atomic_json, lock, read, sha
from calibrate168 import image_geometry, parsed, recording_key, render_path
from mechanisms162 import annotation_inventory
from score165 import write_table


def prepare():
    design=lock('source_review',('source169.py',))
    prior=CAL/'VISUAL_ADJUDICATION_PRIVATE.json'
    assert sha(prior)==read(R168/'SOURCE_VISUAL_AUDIT.json')['private_adjudication_sha256']
    shown={x['case'] for x in read(prior)}
    registry=CAL/'ROSTER.json'
    assert sha(registry)==read(R168/'SOURCE_INPUT_SEAL.json')['source_roster_sha256']
    bank=read(registry)
    dest=SOURCE_PANEL/'plates'
    dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    reserved=SOURCE_PANEL/'PANEL_PRIVATE.json'
    assert not reserved.exists()
    local, receipts = [], {}
    for source in PLATFORMS:
        candidates=defaultdict(list)
        for x in bank:
            if x['source']!=source or x['case'] in shown:
                continue
            row=x['row']
            names,_=annotation_inventory(row['metapath'],source,int(row['suffix'] or 0),row['object_index'])
            role=max(x['truth'],key=lambda k:(names.count(names[x['truth'][k]]),-int(k)))
            proxy=x['truth'][role]
            distractor=int(names.count(names[proxy])>1)
            tie=hashlib.sha256(f"R169|{source}|{x['case']}".encode()).hexdigest()
            candidates[recording_key(row)].append((distractor,tie,int(role),x))
        for batch in candidates.values():
            batch.sort(key=lambda x:(-x[0],x[1]))
        seen=Counter()
        for _ in range(6):
            available=[r for r,rows in candidates.items() if rows]
            r=min(available,key=lambda k:(seen[k],-candidates[k][0][0],candidates[k][0][1]))
            seen[r]+=1
            distractor,_,role,x=candidates[r].pop(0)
            row=x['row']
            categories,boxes=annotation_inventory(row['metapath'],source,int(row['suffix'] or 0),row['object_index'])
            words=parsed(row)['graph']['entities'][role]['phrase']
            p=len(local)+1
            with Image.open(row['imagepath']) as im:
                rgb=im.convert('RGB')
            render_path(p,row,image_geometry(row),categories,boxes,[f'R{role} | {words}'],rgb,dest)
            local.append(dict(panel=p,source=source,parent_source_case=x['case'],role=role,same_class_distractor=distractor,
                              categories=categories))
            for f in (dest/f'{p:02}.jpg',dest/f'{p:02}_raw.jpg'):
                receipts[f.name]=sha(f)
    assert len(local)==18 and not (set(x['parent_source_case'] for x in local)&shown)
    atomic_json(reserved,local)
    atomic_json(HERE/'SOURCE_PANEL_SEAL.json',dict(n=len(local), files=receipts, panel=sha(reserved),design_lock=design,
                prior_cases_disjoint=True, contains_no_prediction_or_proxy_overlays=True, FINAL_access=False))
    print('R169 source visual panel',dict(Counter(x['source'] for x in local)),flush=True)


def evaluate():
    rule=lock('source_review',('source169.py',))
    seal=read(HERE/'SOURCE_PANEL_SEAL.json')
    panel=SOURCE_PANEL/'PANEL_PRIVATE.json'
    truth=SOURCE_PANEL/'ADJUDICATION_PRIVATE.json'
    assert seal['panel']==sha(panel) and seal['design_lock']==rule and not seal['FINAL_access']
    assert all(sha(SOURCE_PANEL/'plates'/name)==v for name,v in seal['files'].items())
    cases=read(panel)
    verdicts=read(truth)
    parent={x['case']:x for x in read(CAL/'ROSTER.json')}
    result=[]
    for case,j in zip(cases,verdicts,strict=True):
        status=j['status']
        assert case['panel']==j['panel'] and status in {'verified','unboxed','ambiguous','insufficient_visibility'}
        answer=j.get('identity',-1)
        assert (0<answer<len(case['categories']))==(status=='verified')
        proxy=parent[case['parent_source_case']]['truth'][str(case['role'])]
        outcome=int(answer==proxy) if status=='verified' else (0 if status=='unboxed' else '')
        result.append(dict(panel=case['panel'],source=case['source'],status=status,
                           potential_same_class_distractor=case['same_class_distractor'],proxy_matches=outcome))
    assert len(result)==18
    dest=OUT/'source_image_audit.csv'
    write_table(dest,result)
    stats=dict(n=len(result),confirmed=sum(x['proxy_matches']==1 for x in result), contradicted=sum(x['proxy_matches']==0 for x in result),
               unknown=sum(x['proxy_matches']=='' for x in result),statuses=dict(Counter(x['status'] for x in result)),
               examples_with_same_class_distractor=sum(x['potential_same_class_distractor'] for x in result),
               output_blind=True, image_readers=1, labels_used_to_fit_or_select_method=False,
               private_labels=sha(truth), files={dest.name:sha(dest)}, FINAL_access=False)
    atomic_json(HERE/'SOURCE_IMAGE_AUDIT.json',stats)
    print('R169 source visual audit',stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('prepare','evaluate'))
    {'prepare':prepare,'evaluate':evaluate}[p.parse_args().phase]()
