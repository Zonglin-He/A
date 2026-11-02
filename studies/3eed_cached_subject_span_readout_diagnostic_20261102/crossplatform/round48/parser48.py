"""Caption-only subject-head parser. No I/O, classes, boxes or model imports.

The caller supplies the frozen Stanza dependency document for this exact caption.
Only head_span participates in the readout; compound_spans are diagnostic only.
"""
import re
NOMINAL={'NOUN','PROPN'}
META={'image','picture','photo','photograph','scene','view','frame','description','object','target','entity','type','information','appearance','feature','features','position','environment'}
DIRECTIVES={'find','locate','identify','select','choose','show','pick','spot'}
TYPE_MARKER=re.compile(r'\bobject\s+type\s*:',re.I)
NEXT_FIELD=re.compile(r'\b(?:main\s+appearance\s+features|position\s+information|relationship\s+with(?:\s+the)?\s+surrounding\s+environment)\s*:',re.I)
def dependency_document(doc):
 """Serialize only caption-derived words; called after explicit CPU-NLP auth."""
 return {'sentences':[{'words':[{k:getattr(w,k,None) for k in ('id','text','lemma','upos','head','deprel','start_char','end_char')} for w in s.words]} for s in doc.sentences]}
def unknown(reason,structured=False,**extra):
 return dict(status='UNKNOWN',reason=reason,structured=structured,head_span=None,head_text=None,compound_spans=[],**extra)
def _nominal(w):return w.get('upos') in NOMINAL and str(w.get('lemma') or w.get('text','')).casefold() not in META
def _offset(caption,w):
 lo,hi=w.get('start_char'),w.get('end_char')
 return type(lo) is int and type(hi) is int and 0<=lo<hi<=len(caption) and caption[lo:hi]==w.get('text')
def _field(caption):
 markers=list(TYPE_MARKER.finditer(caption))
 if not markers:return None,None
 if len(markers)!=1:return None,'multiple_object_type_fields'
 lo=markers[0].end()
 # Delimiter positions are relative to the post-marker substring.
 ends=[len(caption)]+[lo+m.start() for m in re.finditer(r'[;\n\r]',caption[lo:])]+[m.start() for m in NEXT_FIELD.finditer(caption,lo)]
 hi=min(ends)
 while lo<hi and caption[lo].isspace():lo+=1
 while hi>lo and caption[hi-1].isspace():hi-=1
 return (lo,hi),None if hi>lo else 'empty_object_type_field'
def _accept(caption,words,head,kind,sentence,structured,field_span=None):
 if not _offset(caption,head):return unknown('selected_head_missing_or_nonexact_raw_offsets',structured,selection_kind=kind)
 if any(w.get('head')==head['id'] and str(w.get('deprel','')).split(':')[0]=='conj' and _nominal(w) for w in words):return unknown('coordinated_referents',structured,selection_kind=kind)
 byid={w['id']:w for w in words};diag=[];stack=[head['id']];seen={head['id']}
 while stack:
  parent=stack.pop()
  for w in words:
   if w.get('head')==parent and str(w.get('deprel','')).split(':')[0] in {'compound','flat','fixed'} and w['id'] not in seen:
    seen.add(w['id']);stack.append(w['id'])
    if _offset(caption,w):diag.append([w['start_char'],w['end_char']])
 return {'status':'FOUND','reason':'unique_caption_dependency_head','structured':structured,'selection_kind':kind,'sentence_index':sentence,'head_word_id':head['id'],'head_span':[head['start_char'],head['end_char']],'head_text':head['text'],'compound_spans':sorted(diag),'compound_not_used_in_primary_mask':True,'field_span':list(field_span) if field_span else None}
def parse_subject(caption,document):
 """Deterministic rules frozen before labels. Unknown falls back outside here."""
 assert isinstance(caption,str) and isinstance(document,dict)
 sentences=document.get('sentences',[])
 field,error=_field(caption)
 if error:return unknown(error,True)
 # Do not remove an unlocated true nominal head and promote its dependent.
 # Invalid offsets make the dependency-to-character mapping uncertifiable.
 if any(w.get('upos') in NOMINAL and not _offset(caption,w) for s in sentences for w in s['words']):
  return unknown('nominal_missing_or_nonexact_raw_offsets',bool(field))
 if field:
  lo,hi=field;eligible=[]
  for si,s in enumerate(sentences):
   words=s['words'];inside=[w for w in words if _offset(caption,w) and lo<=w['start_char']<w['end_char']<=hi];ids={w['id'] for w in inside};byid={w['id']:w for w in inside}
   for w in inside:
    if not _nominal(w):continue
    parent=w.get('head');visited=set();nominal_ancestor=False
    while parent in ids and parent not in visited:
     visited.add(parent);ancestor=byid[parent]
     if ancestor.get('upos') in NOMINAL:nominal_ancestor=True;break
     parent=ancestor.get('head')
    if not nominal_ancestor:eligible.append((si,words,w))
  if len(eligible)!=1:return unknown('explicit_field_not_one_nominal_head',True,field_span=list(field),candidate_heads=len(eligible))
  si,words,w=eligible[0];return _accept(caption,words,w,'explicit_object_type_head',si,True,field)
 for si,s in enumerate(sentences):
  words=s['words'];roots=[w for w in words if w.get('head')==0]
  if len(roots)!=1:
   if any(_nominal(w) for w in words):return unknown('nonunique_matrix_root',False,sentence_index=si)
   continue
  root=roots[0];subjects=[w for w in words if w.get('head')==root['id'] and str(w.get('deprel','')).split(':')[0] in {'nsubj','csubj'}];heads=[w for w in subjects if _nominal(w)]
  if len(heads)>1:return unknown('multiple_matrix_nominal_subjects',False,sentence_index=si)
  if len(heads)==1:return _accept(caption,words,heads[0],'matrix_'+str(heads[0]['deprel']).split(':')[0],si,False)
  if any(str(w.get('deprel','')).split(':')[0]=='csubj' for w in subjects):return unknown('matrix_clausal_subject_not_unique_nominal',False,sentence_index=si)
  # A lexical nominal nsubj wins above. A bare/deictic-copular nominal root
  # handles e.g. "This is a person" without selecting the pronoun "This".
  if _nominal(root):return _accept(caption,words,root,'root_nominal_or_deictic_copular',si,False)
  lemma=str(root.get('lemma') or root.get('text','')).casefold();imperative=root.get('upos')=='VERB' and lemma in DIRECTIVES and all(w.get('upos')=='PRON' and str(w.get('lemma') or w.get('text','')).casefold()=='you' for w in subjects)
  if imperative:
   objects=[w for w in words if w.get('head')==root['id'] and str(w.get('deprel','')).split(':')[0]=='obj' and _nominal(w)]
   if len(objects)==1:return _accept(caption,words,objects[0],'imperative_direct_object',si,False)
   return unknown('imperative_not_one_nominal_object',False,sentence_index=si)
  # Only leading sentences containing metadata nouns exclusively may be skipped.
  # Do not scan later reference sentences after an unresolved referential clause.
  if any(_nominal(w) for w in words):return unknown('first_referential_sentence_no_unique_matrix_role',False,sentence_index=si)
 return unknown('no_caption_subject_head',False)
