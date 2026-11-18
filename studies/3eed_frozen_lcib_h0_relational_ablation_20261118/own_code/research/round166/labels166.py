"""Post-seal identity labels, independent of the methods' output boxes and scores."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from scope166 import ROOT, W97, read, sha
from manifest162 import target_rows as historical_rows
from mechanisms162 import annotation_inventory, find_expected_ref, seed_owners
from geometry162 import in_box

UNKNOWN, WRONG, RIGHT, MISSING_TRUTH, NO_OUTPUT=-1,0,1,-2,-3
PLURAL=re.compile(r'\b(?:two|three|four|five|six|seven|several|multiple|many|group of|row of|line of)\b',re.I)
IRREGULAR_PLURALS={'people','men','women','children','teeth','feet'}


def key_for(row):
    return row['platform'],str(Path(row['metapath']).resolve()),int(row['object_index'])


@lru_cache(maxsize=1)
def prior_review():
    """Reuse genuinely verified labels without reusing either prior prediction."""
    base=ROOT/'crossplatform/round163'
    config=ROOT/'research/round163'
    r=base/'ROLE_REVIEW_PRIVATE.json'
    a=base/'ROLE_ADJUDICATION_PRIVATE.json'
    assert sha(r)==read(config/'ROLE_SCOPE.json')['private_review_sha256']
    assert sha(a)==read(config/'results/ROLE_REVIEW_SUMMARY.json')['review_sha256']
    records=historical_rows()
    grouped=defaultdict(list)
    for entry,label in zip(read(r),read(a),strict=True):
        assert entry['case']==label['case']
        roles=entry['role_indices']
        if (0 in roles) and roles[0]!=roles[1]:
            row=records[entry['panel_index']]['r']
            word=entry['role_phrases'][1 if roles[0]==0 else 0].casefold()
            side=(1 if entry['parsed']=='LEFT' else -1)*(1 if roles[0]==0 else -1)
            grouped[key_for(row),word,side].append(label)
    return grouped


@lru_cache(maxsize=512)
def parser_record(rid):
    return read(W97/'roles_PRIVATE'/f'{rid}.json')


def label_pair(row, ev, a, b, sign, names, boxes):
    ref=b if a==0 else a
    lo,hi=ev.role_spans[ref]
    mention=row['caption'][int(lo):int(hi)]
    side=sign if a==0 else -sign
    before=prior_review().get((key_for(row),mention.casefold(),side),[])
    verified={int(r['ref']) for r in before if r['type']=='verified_boxes'}
    if verified:
        if len(verified)!=1 or not 0<next(iter(verified))<len(names):
            return 'review_conflict',-1
        return 'visual_verified',next(iter(verified))
    if before:
        return 'review_no_3d_reference',-1
    if not mention.strip() or PLURAL.search(mention):
        return 'non_singleton',-1
    doc=parser_record(row['id'])['doc']
    inside=[w for s in doc['sentences'] for w in s['words']
            if isinstance(w.get('start_char'),int) and isinstance(w.get('end_char'),int)
            and lo<=w['start_char'] and w['end_char']<=hi and w['upos'] in ('NOUN','PROPN')]
    def plural(w):
        surface=w['text'].casefold()
        lemma=(w.get('lemma') or surface).casefold()
        return surface in IRREGULAR_PLURALS or (surface.endswith('s') and lemma!=surface)

    if not inside or any(map(plural,inside)):
        return 'non_singleton',-1
    j=find_expected_ref(ev,names,boxes,row['caption'],row['platform'],ref)
    return ('annotation_proxy',j) if j>0 else ('unresolved_reference',-1)


def mask_owners(masks, seed_ids, objects):
    """Assign a unique annotated owner with counts measured against all contained seeds."""
    masks=np.asarray(masks,bool)
    counts=masks.sum(1)
    masses=np.stack([masks[:,seed_ids==j].sum(1) for j in range(objects)],axis=1)
    winner=masses.argmax(1)
    hits=masses[np.arange(len(masks)),winner]
    masses[np.arange(len(masks)),winner]=0
    second=np.maximum(masses.max(1),counts-hits-masses.sum(1))
    valid=(counts>=2)&(hits>=.55*counts)&(hits-second>=.15*counts)
    return np.where(valid,winner,-1)


def box_masks(boxes, seeds):
    return np.stack([in_box(seeds,b) if np.isfinite(b).all() else np.zeros(len(seeds),bool) for b in boxes])


def occupied_owners(boxes, seeds, seed_ids, objects):
    flat=boxes.reshape(-1,7)
    return mask_owners(box_masks(flat,seeds),seed_ids,objects).reshape(boxes.shape[:-1])


def truth_status(owner,gold):
    return UNKNOWN if owner<0 else RIGHT if owner==gold else WRONG


def joint_status(first,second):
    if first==WRONG or second==WRONG:
        return WRONG
    return RIGHT if first==RIGHT and second==RIGHT else UNKNOWN


def physical_labels(row,seeds):
    view=int(row['suffix'] or 0)
    names,boxes=annotation_inventory(row['metapath'],row['platform'],view,row['object_index'])
    assert len(boxes)>0
    seeds=np.unique(seeds,axis=0)
    labels=seed_owners(seeds,boxes)
    return names,boxes,seeds,labels
