"""General dependency argument inventory; no sample IDs or object-word blacklist."""
import re
from evidence86 import relations86
from evidence85 import color_words
SPATIAL={'left','right','front','back','behind','near','close','beside','between','next','above','below','under','outside','inside'}
COORDINATE=SPATIAL|{'top','bottom','center','centre','middle','side'}
CAMERA={'view','image','picture','observer','viewer','screen','frame'}
def repair_inventory87(caption,doc,oldgraph):
 if not oldgraph['usable']:return oldgraph
 headers=[(m.start(1),m.end(1)) for m in re.finditer(r'(?:^|[;\n])\s*([^:;\n]{1,80}):',caption)];target=dict(oldgraph['entities'][0]);target['role']='T';entities=[target];ranked=[]
 for si,sentence in enumerate(doc['sentences']):
  words=sentence['words'];by={w['id']:w for w in words};children={w['id']:[x for x in words if x['head']==w['id']] for w in words}
  def lemma(w):return str(w.get('lemma') or w['text']).casefold()
  def path(w):
   out=[w];seen={w['id']}
   while out[-1]['head'] in by and out[-1]['head'] not in seen:out.append(by[out[-1]['head']]);seen.add(out[-1]['id'])
   return out
  def cases(w):return [lemma(x) for x in children[w['id']] if x['deprel'].split(':')[0]=='case']
  candidates=[]
  for w in words:
   if any(a<=w.get('start_char',-1)<b for a,b in headers):continue
   if w['upos'] not in ['NOUN','PROPN'] or not isinstance(w.get('start_char'),int) or not isinstance(w.get('end_char'),int):continue
   if target['span'][0]<=w['start_char']<w['end_char']<=target['span'][1] or lemma(w) in COORDINATE|CAMERA or w['deprel'].split(':')[0] in ['compound','flat','amod']:continue
   p=path(w);base=w['deprel'].split(':')[0];existential=base in ['nsubj','root','conj'] and any(any(x['deprel']=='expl' and lemma(x)=='there' for x in children[n['id']]) for n in p)
   explicit=bool(set(cases(w))&SPATIAL) or (w['head'] in by and lemma(by[w['head']]) in SPATIAL and bool(cases(w)))
   if existential or explicit:candidates.append(w)
  chosen={w['id'] for w in candidates}
  for _ in range(len(words)):
   added={w['id'] for w in words if w['upos'] in ['NOUN','PROPN'] and w['deprel'].split(':')[0]=='conj' and w['head'] in chosen};new=added-chosen
   if not new:break
   chosen|=new
  for w in words:
   if w['id'] not in chosen or lemma(w) in COORDINATE|CAMERA:continue
   members={w['id']};todo=[w['id']]
   while todo:
    hd=todo.pop()
    for x in children[hd]:
     dep=x['deprel'].split(':')[0];adjacent_nominal_apposition=dep=='appos' and x['upos'] in ['NOUN','PROPN'] and ((by[hd]['end_char']<=x['start_char'] and not caption[by[hd]['end_char']:x['start_char']].strip()) or (x['end_char']<=by[hd]['start_char'] and not caption[x['end_char']:by[hd]['start_char']].strip()));include=dep in ['amod','compound','det','nummod','flat','fixed'] or adjacent_nominal_apposition or (dep=='nmod' and bool(set(cases(x))&{'with','of'}) and lemma(x) not in SPATIAL|CAMERA)
     if include and x['id'] not in members:members.add(x['id']);todo.append(x['id'])
   group=[by[k] for k in members if isinstance(by[k].get('start_char'),int) and isinstance(by[k].get('end_char'),int)];lo=min(x['start_char'] for x in group);hi=max(x['end_char'] for x in group);phrase=caption[lo:hi]
   if lo<target['span'][1] and hi>target['span'][0]:continue
   ranked.append({'role':'X'+str(len(ranked)),'phrase':phrase,'span':[lo,hi],'attributes':color_words(phrase),'head':w['text'],'selection':'dependency_existential_or_spatial_argument','sentence':si})
 allgraph={'usable':True,'entities':[target]+ranked,'relations':[]};edges=relations86(caption,doc,allgraph);degree={e['role']:0 for e in ranked}
 for e in edges:
  for role in [e['subject'],e['reference']]:
   if role in degree:degree[role]+=2 if 'T' in [e['subject'],e['reference']] else 1
 ranked=sorted(ranked,key=lambda e:(-degree[e['role']],e['span'][0]));seen=set()
 for e in ranked:
  if tuple(e['span']) in seen:continue
  seen.add(tuple(e['span']));e=dict(e);e['role']='R'+str(len(entities)-1);entities.append(e)
  if len(entities)==5:break
 graph={'usable':True,'entities':entities,'relations':[],'reason':'dependency_argument_inventory_not_all_nouns'};graph['relations']=relations86(caption,doc,graph);return graph
