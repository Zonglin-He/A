"""Post-seal linkage of reference roles to existing ordinary-only identity evidence."""
from collections import Counter, defaultdict
from functools import lru_cache
import re
import sys
from types import SimpleNamespace

import numpy as np

from study import PLATFORMS, ROOT, frame, ordinary_rows, read, sha
from evidence import natural_graphs
from score import local_points, owner_for

R163 = ROOT/'research/round163'
W163 = ROOT/'crossplatform/round163'
R166 = ROOT/'research/round166'
W166 = ROOT/'crossplatform/round166'
VISUAL163 = W163/'ROLE_ADJUDICATION_PRIVATE.json'
VISUAL166 = W166/'REFERENCE_ADJUDICATION_PRIVATE.json'
PANEL163 = W163/'ROLE_REVIEW_PRIVATE.json'
PANEL166 = W166/'reference_review_PRIVATE/REGISTRY_PRIVATE.json'


def truth_pins():
    files = [VISUAL163, VISUAL166, PANEL163, PANEL166, R163/'ROLE_SCOPE.json',
             R163/'results/ROLE_REVIEW_SUMMARY.json', R166/'REVIEW_PANEL_SEAL.json',
             R166/'VISUAL_REVIEW_AUDIT.json', R166/'PREDICTION_SEAL.json',
             R166/'labels166.py', ROOT/'research/round162/mechanisms162.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def phrase_key(phrase):
    words = re.findall(r'[\w]+', phrase.casefold())
    if words and words[0] in ('the','a','an'):
        words = words[1:]
    return tuple(words)


def role_phrase(row):
    """Resolve the same one optional reference slot role as the frozen evidence builder."""
    graph = natural_graphs([row])[0]['graph']
    entities = graph['entities']
    edges = [e for e in graph['relations'] if 'T' in (e['subject'], e['reference'])]
    selected = entities[1] if len(entities)>1 else None
    if len(edges)==1:
        direct = edges[0]
        other = direct['reference'] if direct['subject']=='T' else direct['subject']
        selected = next((item for item in entities if item['role']==other), selected)
    if selected is None:
        return ()
    a,b = selected['span']
    return phrase_key(row['caption'][a:b])


@lru_cache(maxsize=1)
def verified_refs():
    """Reuse both blind visual review panels, keeping only their exact joined expressions."""
    from common160 import rows

    cert163 = read(R163/'ROLE_SCOPE.json')
    results163 = read(R163/'results/ROLE_REVIEW_SUMMARY.json')
    cert166 = read(R166/'REVIEW_PANEL_SEAL.json')
    results166 = read(R166/'VISUAL_REVIEW_AUDIT.json')
    assert not any(cert['FINAL_access'] for cert in (cert163, results163, cert166, results166))
    assert sha(PANEL163)==cert163['private_review_sha256'] and sha(VISUAL163)==results163['review_sha256']
    assert sha(PANEL166)==cert166['digest']
    old = rows()
    current = {p:{(frame(r),r['caption']):i for i,r in enumerate(ordinary_rows(p))} for p in PLATFORMS}
    known = defaultdict(list)
    for case,mark in zip(read(PANEL163), read(VISUAL163), strict=True):
        assert case['case']==mark['case']
        prior_row = old[case['panel_index']]['r']
        if mark['type']!='verified_boxes':
            continue
        target = case['target']
        idx = current[target].get((frame(prior_row),prior_row['caption']))
        if idx is None:
            continue
        roles = case['role_indices']
        assert 0 in roles and roles[0]!=roles[1]
        phrase = case['role_phrases'][int(roles[0]==0)]
        known[target,idx].append((phrase_key(phrase),int(mark['ref']),len(case['boxes']),'R163'))
    raw166 = read(VISUAL166)
    assert dict(Counter(x['status'] for x in raw166))==results166['counts']
    for panel,mark in zip(read(PANEL166),raw166,strict=True):
        assert panel['case']==mark['case']
        if mark['status']=='verified_boxes':
            target, idx = panel['target'], int(panel['ordinal'])
            assert 0<=idx<len(ordinary_rows(target))
            known[target,idx].append((phrase_key(panel['reference_phrase']),int(mark['reference']),
                                      len(panel['categories']),'R166'))
    return dict(known)


@lru_cache(maxsize=6)
def archived_seal(direction):
    receipt = read(R166/'PREDICTION_SEAL.json')
    assert receipt['n']==18620 and not receipt['FINAL_access'] and not receipt['target_label_read']
    p = W166/'ordinary'/direction/'SEAL.json'
    old = read(p)
    assert sha(p)==receipt['directions'][direction]['seal']
    assert not old['FINAL_access'] and not old['target_label_read']
    return old


@lru_cache(maxsize=2)
def proxy_block(direction, name):
    path = W166/'ordinary'/direction/name
    assert sha(path)==archived_seal(direction)['files'][name]
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in ('ordinals','edges','edge_offsets','spans','camera_intrinsic',
                                    'model_to_camera','distortion','width')}


@lru_cache(maxsize=1)
def library():
    sys.path[:0] = [str(ROOT/f'research/round{i}') for i in (166,162,165,158)]
    from labels166 import label_pair
    from mechanisms162 import annotation_inventory
    return label_pair, annotation_inventory


def assigned_owners(seed, ref_boxes, inventory):
    points = np.unique(seed,axis=0)
    inside = np.stack([local_points(points,b) for b in inventory])
    ids = np.where(inside.sum(0)==1,inside.argmax(0),len(inventory))
    return np.asarray([owner_for(b,points,ids,len(inventory)) for b in ref_boxes],dtype=np.int16)


def identity_context(source, target, row, ordinal, seed_xyz, ref_boxes, include_proxy=True):
    label_pair, annotation_inventory = library()
    names, boxes = annotation_inventory(row['metapath'],target,int(row['suffix'] or 0),row['object_index'])
    owner = assigned_owners(seed_xyz,ref_boxes,boxes)
    phrase = role_phrase(row)
    panel = verified_refs().get((target,ordinal),())
    human = {(ref,n) for key,ref,n,_ in panel if key==phrase and phrase}
    if len(human)==1:
        ref,n = next(iter(human))
        assert len(boxes)==n and 0<ref<len(boxes)
        return owner, ref, 2, bool(panel)
    if panel or not (include_proxy and phrase):
        return owner,-1,0,bool(panel)
    bank = proxy_block(f'{source}_to_{target}',f'{ordinal//128:04d}.npz')
    i = ordinal%128
    assert int(bank['ordinals'][i])==ordinal
    edges = bank['edges'][bank['edge_offsets'][i]:bank['edge_offsets'][i+1]]
    links = set()
    for a,b,sign in edges:
        if (a==0)==(b==0):
            continue
        role,side = (b,sign) if a==0 else (a,-sign)
        lo,hi = bank['spans'][i,role]
        if phrase_key(row['caption'][lo:hi])==phrase:
            links.add((int(role),int(side)))
    if len(links)!=1:
        return owner,-1,0,False
    role,side = next(iter(links))
    data = SimpleNamespace(edges=edges,role_spans=bank['spans'][i],camera_intrinsic=bank['camera_intrinsic'][i],
                           model_to_camera=bank['model_to_camera'][i],distortion=bank['distortion'][i],
                           width=int(bank['width'][i]))
    provenance, ref = label_pair(row,data,0,role,side,names,boxes)
    # Label proxy relies on the observed inventory and explicit projected relation.
    if provenance=='annotation_proxy':
        assert 0<ref<len(boxes)
        return owner,int(ref),1,False
    return owner,-1,0,False
