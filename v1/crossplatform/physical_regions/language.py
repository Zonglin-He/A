"""Frozen parser gates and two noun phrases used by the regional readout."""
import re


PLURAL=re.compile(r'\b(?:two|three|four|five|six|seven|several|multiple|many|group of|row of|line of)\b',re.I)

IRREGULAR_PLURALS={'people','men','women','children','teeth','feet'}


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


NOUNS = {
    'car': {'car', 'vehicle', 'sedan', 'suv', 'van', 'pickup', 'automobile', 'taxi', 'coupe', 'minivan'},
    'pedestrian': {'pedestrian', 'person', 'man', 'woman', 'boy', 'girl', 'child', 'individual',
                   'walker', 'worker', 'passerby'},
    'truck': {'truck', 'lorry'},
    'bus': {'bus', 'shuttle', 'coach'},
    'cyclist': {'cyclist', 'biker', 'bike', 'bicycle', 'rider'},
    'othervehicle': {'tractor', 'trailer', 'machine', 'machinery'},
}

WORDS = {word: category for category, variants in NOUNS.items() for word in variants}

PLURALS = {'cars', 'vehicles', 'pedestrians', 'persons', 'people', 'men', 'women', 'children',
           'trucks', 'buses', 'cyclists', 'bikes', 'group', 'pair'}

def nominal(phrase):
    words = re.findall('[a-z]+', phrase.lower())
    if set(words) & (PLURALS | {'two','three','several','both','its','their','own'}):
        return None
    categories = {WORDS[x] for x in words if x in WORDS}
    return next(iter(categories)) if len(categories) == 1 else None


def converse(text,info):
    graph,doc=info['graph'],info['document']
    entities={e['role']:i for i,e in enumerate(graph['entities'])}
    edges=set()
    for e in graph['relations']:
        s,r=e['subject'],e['reference']
        if (s=='T')==(r=='T') or e['predicate'] not in ('LEFT','RIGHT'):
            continue
        other=entities.get(r if s=='T' else s)
        if other is not None:
            sign=1 if e['predicate']=='LEFT' else -1
            edges.add((other,sign if s=='T' else -sign))
    if len(edges)!=1:
        return None
    role,sign=next(iter(edges))
    regions={'area','background','bottom','center','edge','foreground','frame','ground','image','lane','middle',
             'picture','region','road','scene','screen','side','sky','top','view'}
    if graph['entities'][role]['head'].lower() in regions:
        return None
    views=rewrite(text,dict(graph=graph,doc=doc),list(edges),role,sign)
    return (sign,views['D']['text']) if views['status']=='supported' else None


PATTERN = re.compile(r'^find the (\w+) immediately to the (left|right) of the (\w+)( closest to the camera| farthest from the camera)? in the camera image\.$',re.I)

def interpret(text,cap):
    matched=PATTERN.fullmatch(text.strip())
    if matched:
        subject,side,ref,qualifier=matched.groups()
        depth=0 if not qualifier else (1 if 'closest' in qualifier.lower() else -1)
        return subject,ref,(-1 if side.lower()=='left' else 1),depth,True
    relation=converse(text,cap)
    if relation is None:return None
    roles={e['role']:e for e in cap['graph']['entities']}
    edges={e['reference'] if e['subject']=='T' else e['subject'] for e in cap['graph']['relations']
           if e['predicate'] in ('LEFT','RIGHT') and 'T' in (e['subject'],e['reference'])}
    if len(edges)!=1:return None
    def phrase(role):
        a,b=roles[role]['span']
        s=text[a:b].strip(' ,.')
        return s if a<b and roles[role]['head'].lower() in s.lower() else ''
    s,r=phrase('T'),phrase(next(iter(edges)))
    if not s or not r:return None
    immediate=bool(re.search(r'\bimmediately (?:to the |on the )?(?:left|right)\b',text,re.I))
    return s,r,-relation[0],0,immediate

def role_text(phrase):
    return re.sub(r'^(?:the|a|an)\s+','',phrase,flags=re.I).strip()


CLASSES = ("car", "pedestrian", "bus", "other vehicle", "truck", "cyclist", "background")
