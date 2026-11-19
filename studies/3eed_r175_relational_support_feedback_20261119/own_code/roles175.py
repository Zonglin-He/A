"""Text-only role and relation decoding. Unknown or ambiguous relations stay inactive."""
import re

from evidence import natural_graphs
from scope175 import RELATIONS

MISSING = (0,0)
CODES = {name:k for k,name in enumerate(RELATIONS)}
INVERSE = dict(left='right',right='left',above='below',below='above',near='near',
               closer='farther',farther='closer')
SCREEN = re.compile(r'\b(?:image|view|picture|frame|screen)\b',re.I)
TEXTS = {
    'left':r'(?:\b(?:to the )?left(?: side)? of\b)',
    'right':r'(?:\b(?:to the )?right(?: side)? of\b)',
    'above':r'\b(?:directly )?above\b',
    'below':r'\b(?:directly )?below\b',
    'near':r'\b(?:closest|nearest) to\b',
    'closer':r'\bcloser to (?:the )?camera than\b',
    'farther':r'\b(?:farther|further) from (?:the )?camera than\b',
}
PATTERNS = {k:re.compile(v,re.I) for k,v in TEXTS.items()}
PREFILTER = re.compile(r'\b(?:left|right|above|below|closest|nearest|closer|farther|further)\b',re.I)
INTRINSIC = re.compile(r"\b(?:its|their|own|facing|pointing|oriented|not|never)\b",re.I)


def roles(row, record):
    text = row['caption']
    graph = record['graph']
    entities = graph['entities']
    subject = next((x for x in entities if x['role']=='T'),None)
    if subject is None:
        return dict(spans=[(0,len(text)),MISSING,MISSING],relation=0,head='')
    direct = [e for e in graph['relations'] if 'T' in (e['subject'],e['reference'])]
    reference = entities[1] if len(entities)>1 else None
    if len(direct)==1:
        edge = direct[0]
        other = edge['reference'] if edge['subject']=='T' else edge['subject']
        reference = next((x for x in entities if x['role']==other),None)
    if reference is None:
        return dict(spans=[subject['span'],MISSING,MISSING],relation=0,head='')
    result = dict(spans=[subject['span'],reference['span'],MISSING],
                  relation=0,head=reference['head'])
    if len(direct)==1 and direct[0]['known_frame']:
        edge = direct[0]
        name = edge['predicate'].lower()
        words = record['doc']['sentences'][edge['sentence']]['words']
        word = next((w for w in words if w['id']==edge['governor_word']),None)
        if name in ('left','right') and word and word.get('start_char') is not None and word.get('end_char') is not None:
            if edge['subject']!='T':
                name=INVERSE[name]
            result['spans'][2]=(word['start_char'],word['end_char'])
            result['relation']=CODES[name]
            return result
    if direct:
        return result
    start,stop = subject['span'][1],reference['span'][0]
    if stop<=start:
        return result
    between=text[start:stop]
    if INTRINSIC.search(between):
        return result
    for name,pattern in PATTERNS.items():
        hits=list(pattern.finditer(between))
        if len(hits)!=1 or between[hits[0].end():].strip():
            continue
        if name in ('left','right','above','below') and not SCREEN.search(text):
            continue
        if name=='near' and not re.search(r'\b(?:3d|three.dimensional)\b',text,re.I):
            continue
        match=hits[0]
        result['spans'][2]=(start+match.start(),start+match.end())
        result['relation']=CODES[name]
        return result
    return result


def roles_batch(rows, source_only_fast=False):
    if not source_only_fast:
        return [roles(r,d) for r,d in zip(rows,natural_graphs(rows),strict=True)]
    use=[j for j,r in enumerate(rows) if PREFILTER.search(r['caption'])]
    found=[None]*len(rows)
    eligible=[rows[j] for j in use]
    for j,r,d in zip(use,eligible,natural_graphs(eligible),strict=True):
        role=roles(r,d)
        if role['relation']:
            found[j]=role
    return found
