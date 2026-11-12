"""Caption-only conservative role scopes; no task labels or visual evidence."""
import re,hashlib,numpy as np
CONTENT={'ADJ','NOUN','PROPN','VERB','ADV','NUM'}
RELATION=re.compile(r'\b(?:left\s+of|right\s+of|behind|next\s+to|beside|near|in\s+front\s+of|between)\b',re.I)
def spanmask(offsets,spans):
 out=np.zeros(len(offsets),bool)
 for lo,hi in spans:out|=(offsets[:,1]>lo)&(offsets[:,0]<hi)&(offsets[:,1]>offsets[:,0])
 return out
def roles(caption,document,subject,offsets,native,seedkey):
 offsets=np.asarray(offsets);native=np.asarray(native,bool);T=len(offsets);bg=np.zeros(T,bool);bg[-2:]=True;empty=np.zeros(T,bool)
 def unknown(reason):return {'usable':False,'reason':reason,'name':native,'attribute':empty,'background':bg,'mismatch':empty,'attribute_count':0}
 if subject['status']!='FOUND':return unknown('subject_'+subject['reason'])
 name=spanmask(offsets,[subject['head_span']]+subject.get('compound_spans',[]));head=spanmask(offsets,[subject['head_span']])
 if not head.any() or not (head&native).any() or (native&~name).any():return unknown('native_noun_and_caption_subject_disagree')
 if (native&bg).any():return unknown('native_noun_overlaps_background')
 words=document['sentences'][subject['sentence_index']]['words'];hid=subject['head_word_id'];attributes=[];kind='direct_dependency'
 if subject.get('structured'):
  m=re.search(r'\bmain\s+appearance\s+features\s*:\s*([^;\n\r]+)',caption,re.I)
  if m:
   lo,hi=m.start(1),m.end(1)
   # Explicit field ownership is used only as supplied by text. Embedded
   # spatial reference clauses cannot be silently converted to attributes.
   if RELATION.search(m.group(1)):return unknown('relational_clause_in_appearance_field')
   attributes=[(w['start_char'],w['end_char']) for s in document['sentences'] for w in s['words'] if type(w.get('start_char')) is int and lo<=w['start_char']<w['end_char']<=hi and (w.get('upos') in CONTENT or w.get('deprel')=='neg')];kind='explicit_subject_appearance_field'
 else:
  chosen={w['id'] for w in words if w.get('head')==hid and str(w.get('deprel','')).split(':')[0] in {'amod','nummod'}}
  parent=next((w for w in words if w['id']==next(w['head'] for w in words if w['id']==hid)),None)
  if parent is not None and parent.get('upos')=='ADJ' and next(w for w in words if w['id']==hid).get('deprel','').startswith('nsubj'):chosen.add(parent['id'])
  for _ in range(len(words)):
   more={w['id'] for w in words if w.get('head') in chosen and str(w.get('deprel','')).split(':')[0] in {'advmod','conj','neg'} and w.get('upos') in {'ADJ','ADV','NUM','PART'}}
   if more<=chosen:break
   chosen|=more
  attributes=[(w['start_char'],w['end_char']) for w in words if w['id'] in chosen and type(w.get('start_char')) is int]
 attr=spanmask(offsets,attributes)&~native&~bg
 # Spatial relation operators are excluded from ordinary attributes even
 # when the dependency parser tags their predicate as adjectival.
 relation_spans=[(m.start(),m.end()) for m in RELATION.finditer(caption)]
 attr &= ~spanmask(offsets,relation_spans)
 real=np.array([hi>lo and hi<=len(caption) and bool(re.search(r'\w',caption[lo:hi])) for lo,hi in offsets]);pool=np.flatnonzero(real&~native&~bg&~attr);k=int(attr.sum());mismatch=empty.copy()
 if k:
  if len(pool)<k:return unknown('insufficient_equal_count_mismatch_pool')
  rng=np.random.default_rng(int(hashlib.sha256(('R83roles|'+seedkey).encode()).hexdigest()[:16],16));mismatch[rng.permutation(pool)[:k]]=True
 assert not (attr&bg).any() and mismatch.sum()==attr.sum()
 return {'usable':True,'reason':'parsed_'+kind,'name':native,'attribute':attr,'background':bg,'mismatch':mismatch,'attribute_count':k}
