"""Build candidate-independent source invitations for natural-expression visual verification."""
import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from scope import HERE, PLATFORMS, PRIVATE, SPEC, atomic_json, former_frames, read, sha, stable, train_path

DIRECTION = re.compile(r'\b(?:left|right)\b',re.I)
RELATION_PRIORITY = re.compile(
    r'\b(?:left|right)(?:\s+side)?\s+of\b|\b(?:on|to|at)\s+(?:its|his|her|their)\s+(?:left|right)\b',
    re.I)
LIMIT = 36


def source_row(raw, folder):
    p = Path(raw['metapath']).resolve()
    assert Path(raw['pointpath']).resolve().is_file() and Path(raw['imagepath']).resolve().is_file()
    meta = read(p)['ground_info']
    matches = [i for i,a in enumerate(meta) if a['caption'].strip() == raw['caption'].strip()]
    if len(matches)!=1:
        return None
    subject = matches[0]
    counts = Counter(str(t['class_other']).casefold() for a in meta for t in a.get('others', [])
                     if t.get('bbox_3d_other') is not None)
    counts.update(str(a['class']).casefold() for j,a in enumerate(meta) if j!=subject)
    if max(counts.values(),default=0)<2:
        return None
    row = {k:raw[k] for k in ('platform','suffix','caption','pointpath','metapath','imagepath','scene')}
    row.update(caption=raw['caption'].strip(), object_index=subject, seed=0,
               id=stable('source-reference-record',p,subject,raw['suffix']))
    return row


def eligible(row, parse):
    from calibrate168 import edges_of, image_geometry
    from decode160 import WAYMO_SYNONYMS, M3ED_SYNONYMS
    from mechanisms162 import annotation_inventory
    from panel169 import rewrite
    from retrieval167 import direct_edges
    from visual162 import corners, projection

    parsed = parse(row['caption'])
    edges = direct_edges(edges_of(parsed['graph']))
    if not edges:
        return 'no_direct_horizontal', None
    options = []
    for role, sign in sorted(edges):
        view = rewrite(row['caption'],parsed,edges,role,sign)
        if view['status'] != 'supported':
            continue
        head = parsed['graph']['entities'][role]['head'].casefold().strip()
        if not re.fullmatch(r'[a-z][a-z -]*',head):
            continue
        # An additional word provides a visual or descriptive cue to adjudicate a same-class pair.
        named = re.findall(r'[a-z]+',view['C']['text'].casefold())[1:]
        if len(named) <= len(head.split()):
            continue
        options.append((role,sign,view,head))
    if not options:
        return 'no_distinctive_reference_phrase',None
    p = row['platform']
    names, boxes = annotation_inventory(row['metapath'],p,int(row['suffix'] or 0),row['object_index'])
    camera = image_geometry(row)
    xy = projection(camera,boxes[:,:3])
    centers = np.isfinite(xy).all(1) & (xy>=.015).all(1) & (xy<=.985).all(1)
    if not centers[0]:
        return 'subject_center_not_visible',None
    syn = WAYMO_SYNONYMS if p=='waymo' else M3ED_SYNONYMS
    possible = []
    for role,sign,view,head in options:
        groups = defaultdict(list)
        for i,n in enumerate(names[1:],start=1):
            synonyms = {n, *(str(w).casefold() for w in syn.get(n,()))}
            if head not in synonyms or not centers[i] or sign*(xy[i,0]-xy[0,0])<.014:
                continue
            projected = projection(camera,corners(boxes[i])) * [camera.width,camera.height]
            if not np.isfinite(projected).all() or np.ptp(projected,axis=0).min()<7:
                continue
            groups[n].append(i)
        for name, members in groups.items():
            if len(members)>=2:
                possible.append((role,sign,view,head,name,members))
    if not possible:
        return 'no_same_class_same_side_inventory', None
    role,sign,views,head,cls,candidates=min(possible,key=lambda x:(x[0],len(x[-1]),stable(row['id'],x[4])))
    noun = f'The {head}.'
    views['noun'] = dict(text=noun,spans=[[0,len(noun)-1]],sign=0)
    return 'invitations',dict(row=row,role=role,sign=sign,views=views,reference_class=cls,
                             choices=candidates,recording=stable('source-ref-rec',Path(row['metapath']).resolve().parent.parent),
                             inventory_size=len(names))


def run(source):
    from select170 import parser

    out = PRIVATE/'prep'
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = out/f'{source}.json'
    assert not path.exists() and not (out/f'{source}_seal.json').exists()
    started = perf_counter()
    train = train_path(source)
    old_frames, old_paths, old_points, old_images = former_frames()
    grouped = defaultdict(list)
    findings = Counter()
    for r in read(train):
        findings['total_index'] += 1
        if not r['positive'] or not DIRECTION.search(r['caption']):
            continue
        meta = Path(r['metapath']).resolve()
        if meta in old_frames or any(Path(r[k]).resolve() in old_paths for k in ('pointpath','imagepath')):
            findings['previous_frame_or_media'] += 1
            continue
        grouped[meta.parent.parent].append(r)
    for rows in grouped.values():
        rows.sort(key=lambda x:(not bool(RELATION_PRIORITY.search(x['caption'])),stable('natural-reference-v1',x['metapath'],x['caption'])))
    pool = {g:deque(x) for g,x in grouped.items() if x}
    n = Counter()
    seen = set()
    parse = parser()
    rows=[]
    while pool and len(rows)<LIMIT:
        recording=min(pool,key=lambda k:(n[k],stable('reference-source-site-v1',k)))
        raw=pool[recording].popleft()
        if not pool[recording]:
            del pool[recording]
        meta=Path(raw['metapath']).resolve()
        if meta in seen:
            findings['duplicate_source_frame']+=1
            continue
        row=source_row(raw,recording)
        if row is None:
            findings['lacks_exact_subject_or_peer_inventory']+=1
            continue
        status,entry=eligible(row,parse)
        if entry is None:
            findings[status]+=1
            continue
        scan, image=sha(Path(row['pointpath'])),sha(Path(row['imagepath']))
        if scan in old_points or image in old_images:
            findings['duplicate_media']+=1
            continue
        old_points.add(scan)
        old_images.add(image)
        seen.add(meta)
        row.update(point_sha256=scan,image_sha256=image)
        n[recording]+=1
        entry['case']=f'{source[0].upper()}{len(rows)+1:02d}'
        rows.append(entry)
        findings[status]+=1
        if len(rows)%8==0:
            print(source, 'review invitations',len(rows),'recordings',len(n),flush=True)
    atomic_json(path,rows)
    atomic_json(out/f'{source}_seal.json',dict(index=sha(train),protocol=sha(SPEC),selection=sha(Path(__file__)),
        file=sha(path),counts=dict(findings),recordings=len(n),gpu_used=False,predictions_used=False,
        protected_data_access=False,selection_seconds=perf_counter()-started))
    print(source,dict(findings),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',choices=PLATFORMS)
    run(ap.parse_args().source)
