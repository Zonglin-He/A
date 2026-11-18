"""Prepare the archived visual panel and format a matched original/swapped edge."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import re

import numpy as np

from scope169 import HERE, ORDER, PANEL, PANEL_N, R168, atomic_json, lock, read, roster, sha
from cache169 import observation
from audit167 import blocks, row_edges
from labels166 import IRREGULAR_PLURALS, PLURAL, parser_record
from scope166 import BLOCK
from score165 import table

ARTICLE = re.compile(r'^(?:the|an?|another)\s+', re.I)
COREF = re.compile(r'^(?:it|he|she|they|this|that|these|those|same|former|latter|one|ones)\b', re.I)
APPEARANCE = re.compile(r'\bmain\s+appearance\s+features\s*:\s*(.*?)(?=;|\bposition\s+information\s*:|'
                        r'\brelationship\s+with\b|\.(?:\s|$)|$)', re.I)
ADJACENT_MODIFIER = re.compile(r'^\s+(?P<modifier>(?:with|wearing)\s+[^,.;:]+)', re.I)
LOCATION_BREAK = re.compile(r'\b(?:located|standing|sitting|parked|positioned)\s+(?:in|on|at|near|to)\b|'
                            r'\b(?:to the (?:left|right) of|in front of|behind it|next to)\b', re.I)
FRAME_SHIFT = re.compile(r'\b(?:its|his|her|their)\s+(?:own\s+)?(?:left|right)\b|'
                         r'\b(?:from|in)\s+(?:its|his|her|their|the (?:vehicle|object)\x27s)\s+(?:own\s+)?(?:view|perspective)\b', re.I)


def endpoint(caption, entity, doc, main):
    lo, hi = entity['span']
    raw = caption[lo:hi].strip(' ,.;')
    body = ARTICLE.sub('', raw)
    if len(body.split())==0 or COREF.match(body) or PLURAL.search(body):
        return None
    head = entity.get('head','').casefold()
    if head in {'side','edge','view','position','part','image','background','foreground','one','area','unknown',''}:
        return None
    nouns = [(s,w) for s in doc['sentences'] for w in s['words'] if w['upos'] in ('NOUN','PROPN') and
             isinstance(w.get('start_char'),int) and isinstance(w.get('end_char'),int) and
             lo <= w['start_char'] < w['end_char'] <= hi and head in (w['text'].casefold(), (w.get('lemma') or '').casefold())]
    if len(nouns)!=1:
        return None
    sentence, word = nouns[0]
    name = word['text'].casefold()
    if (name in IRREGULAR_PLURALS or 'Number=Plur' in str(word.get('feats')) or
        (name.endswith('s') and name!=str(word.get('lemma') or name).casefold())):
        return None
    if any(w['head']==word['id'] and w['upos'] in ('NOUN','PROPN') and w['deprel']=='conj' and
           lo<=w.get('start_char',-1)<hi for w in sentence['words'] if isinstance(w.get('start_char'),int)):
        return None
    first = body.split()[0]
    if first.istitle() and word['upos'] != 'PROPN':
        body = body[:1].lower()+body[1:]
    extra = APPEARANCE.search(caption) if main else None
    if extra:
        attribute = extra[1].strip(' .,:;')
    else:
        tail = ADJACENT_MODIFIER.match(caption[hi:])
        attribute = LOCATION_BREAK.split(tail['modifier'])[0].strip() if tail else ''
    if attribute:
        body += ' ('+attribute+')'
    return ' '.join(body.split())


def canonical(first, second, sign):
    direction = 'left' if sign==1 else 'right'
    reach = f'The {first}'
    gap = f' to the {direction} of '
    context = f'the {second}'
    return dict(text=reach+gap+context+' in the camera image.',
                spans=[[0,len(reach)],[len(reach+gap),len(reach+gap+context)]],
                predicate_pos=len(reach+' to the '),sign=int(sign))


def rewrite(caption, parsed, direct, role, sign):
    graph, doc = parsed['graph'], parsed['doc']
    e = graph['entities']
    if not 0 < role < len(e) or e[0]['role']!='T':
        return dict(status='unresolved_endpoints')
    if sum(r==role for r,_ in direct)!=1:
        return dict(status='conflicting_horizontal_predicates')
    names = {e[0]['role'],e[role]['role']}
    edges = [x for x in graph['relations'] if {x['subject'],x['reference']}==names and
             x['predicate'] in ('LEFT','RIGHT') and x['known_frame']]
    if not edges:
        return dict(status='unsupported_coordinate_frame')
    for item in edges:
        if 'sentence' not in item or not 0<=item['sentence']<len(doc['sentences']):
            return dict(status='unsupported_coordinate_frame')
        words = doc['sentences'][item['sentence']]['words']
        context = ' '.join(w['text'] for w in words).replace(' ’', "'").replace(" 's", "'s")
        if FRAME_SHIFT.search(context):
            return dict(status='unsupported_coordinate_frame')
    first, second = (endpoint(caption,e[i],doc,i==0) for i in (0,role))
    if first is None or second is None or first.casefold()==second.casefold():
        return dict(status='ambiguous_or_nonsingular_endpoint')
    return dict(status='supported',B=canonical(first,second,sign),
                C=dict(text=f'The {second}.',spans=[[0,len('The '+second)]],sign=0),
                D=canonical(second,first,-sign))


@lru_cache(maxsize=8)
def target_block(source,target,shard):
    (_, old, base, earlier), = blocks('ordinary',source,target,shard)
    return old,base,earlier


def lookup(entry):
    source,target,ordinal = (entry[x] for x in ('source','target','ordinal'))
    old,base,earlier = target_block(source,target,f'{ordinal//BLOCK:04d}.npz')
    k, = np.flatnonzero(old['ordinals']==ordinal)
    return int(k),old,base,earlier


def prepare():
    design = lock('panel_input', ('panel169.py',))
    assert sha(R168/'results/oracle_rows.csv')==read(R168/'EVALUATION_AUDIT.json')['files']['oracle_rows.csv']
    members = {(x['source'],x['target'],x['expression'],int(x['edge'])) for x in table(R168/'results/oracle_rows.csv')}
    assert len(members)==PANEL_N
    panel, matched = [], set()
    counters = defaultdict(list)
    counts = Counter()
    for source,target in ORDER:
        rows = roster()[target]
        slots = defaultdict(list)
        for s,t,x,i in members:
            if (s,t)==(source,target):
                slots[x].append(i)
        for ordinal,row in enumerate(rows):
            info = observation(row,source,target)
            for edge in sorted(slots.get(info['expression'],[])):
                k,_,_,earlier = lookup(dict(**info,ordinal=ordinal))
                _, direct = row_edges(earlier,k)
                role,sign = direct[edge-1]
                parsed = parser_record(row['id'])
                assert hashlib.sha256(row['caption'].encode()).hexdigest()==parsed['caption_sha256']
                e = dict(**info,ordinal=ordinal,edge=edge,role=role,sign=sign)
                rw = rewrite(row['caption'],parsed,direct,role,sign)
                panel.append(dict(**e,**rw))
                matched.add((source,target,info['expression'],edge))
                counters[target,ordinal,edge].append(rw)
                counts[target,rw['status']] += 1
    assert matched==members and len(panel)==PANEL_N
    assert all(len(x)==2 and x[0]==x[1] for x in counters.values())
    PANEL.mkdir(parents=True, exist_ok=True, mode=0o700)
    path=PANEL/'INPUT_PRIVATE.json'
    assert not path.exists()
    atomic_json(path,panel)
    atomic_json(HERE/'PANEL_INPUT_SEAL.json',dict(n=PANEL_N,physical_frames=len({x['frame'] for x in panel}),
                 verified_membership_diagnostic=True, membership=sha(R168/'results/oracle_rows.csv'),
                 rewrites_FROM_GT=0, roster=sha(path),design_lock=design, target_identity_read=False,
                 counts={f'{t}:{label}':n for (t,label),n in sorted(counts.items())}, FINAL_access=False))
    print('R169 panel',dict(counts),flush=True)


if __name__=='__main__':
    prepare()
