"""Full literal subject-field binding plus neural noun-phrase references.
No free-form model JSON. Unknown spatial frames remain unknown.
"""
import sys,re
from common import *
STOP={'side','front','back','rear','left','right','view','image','picture','observer','viewer','center','centre','middle','edge','corner','position','information','appearance','feature','features','environment','type','line','lines','body','roof','rack','taillight','taillights','shirt','top','shorts','pants','jacket','sleeve','hand','arm','leg','head','face','wheel','window','color','colour','light'}
def field(text,label):
 m=re.search(r'\b'+label+r'\s*:\s*([^;]+)',text,re.I)
 return {'text':m.group(1).strip(),'span':[m.start(1)+len(m.group(1))-len(m.group(1).lstrip()),m.start(1)+len(m.group(1).rstrip())]} if m else None
def parse(caption,nlp):
 typ=field(caption,r'object\s+type');app=field(caption,r'main\s+appearance\s+features');pos=field(caption,r'position\s+information');rel=field(caption,r'relationship\s+with\s+the\s+surrounding\s+environment');doc=nlp(caption);words=[w for s in doc.sentences for w in s.words]
 if typ:
  category=typ['text'];subject=typ;own=app['text'] if app else '';subject_query=category+' '+own;ref_start=rel['span'][0] if rel else len(caption);binding='explicit_subject_fields'
 else:
  first=doc.sentences[0];heads=[w for w in first.words if w.upos in ('NOUN','PROPN') and w.text.casefold() not in STOP];head=heads[0] if heads else None
  if head is None:return {'status':'unknown_subject','subject_query':'','references':[],'relations':[]}
  end=min([m.start() for m in re.finditer(r'\b(?:located|positioned|situated)\b',caption,re.I) if m.start()>head.start_char]+[first.tokens[-1].end_char]);start=0;m=re.match(r'\s*(?:this\s+is|it\s+is|there\s+is)\s+',caption,re.I)
  if m:start=m.end()
  subject={'text':caption[start:end].strip(' ,.'),'span':[start,end]};category=head.text;own=subject['text'];subject_query=own;ref_start=first.tokens[-1].end_char;binding='main_sentence_literal_prefix_with_neural_head'
  pos={'text':caption[end:ref_start],'span':[end,ref_start]}
 refs=[]
 for sent in doc.sentences:
  byid={w.id:w for w in sent.words};children={w.id:[] for w in sent.words}
  for w in sent.words:
   if w.head in children:children[w.head].append(w)
  for w in sent.words:
   if w.start_char<ref_start or w.upos not in ('NOUN','PROPN') or w.text.casefold() in STOP or w.deprel in ('compound','amod'):continue
   members=[w]
   def add(head):
    for c in children[head.id]:
     if c.deprel.split(':')[0] in ('amod','compound','det','nummod','flat','fixed'):members.append(c);add(c)
   add(w);lo=min(t.start_char for t in members);hi=max(t.end_char for t in members);phrase=caption[lo:hi]
   if any(r['span']==[lo,hi] for r in refs):continue
   refs.append({'entity':'R'+str(len(refs)),'phrase':phrase,'span':[lo,hi],'head':w.text,'own_query':phrase,'noun_dependency':w.deprel})
 refs=refs[:6];relations=[]
 # View-relative subject location is kept separately from body orientation.
 ptext=pos['text'] if pos else '';pstart=pos['span'][0] if pos else 0
 if re.search(r'\b(view|image|picture|frame|screen)\b',ptext,re.I):
  no_orientation=re.split(r'\b(?:with\s+(?:its|the)\s+(?:front|back)|front\s+pointing|facing)\b',ptext,flags=re.I)[0]
  if re.search(r'\b(close\s+to|near)\s+(?:the\s+)?(?:middle|center|centre)\b',no_orientation,re.I):pred='GLOBAL_CENTER'
  elif re.search(r'\b(?:left|right)\b.*\b(?:center|centre|middle)\b',no_orientation,re.I):pred='RIGHT_OF_CENTER' if re.search(r'\bright\b',no_orientation,re.I) else 'LEFT_OF_CENTER'
  elif re.search(r'\b(?:upper|lower|far)?\s*(?:left|right)\b',no_orientation,re.I):pred='GLOBAL_RIGHT' if re.search(r'\bright\b',no_orientation,re.I) else 'GLOBAL_LEFT'
  elif re.search(r'\b(center|centre|middle)\b',no_orientation,re.I):pred='GLOBAL_CENTER'
  else:pred='UNKNOWN'
  if pred!='UNKNOWN':relations.append({'subject':'T','reference':None,'predicate':pred,'frame':'camera_image','quote':no_orientation.strip(),'origin':'literal_subject_position_field'})
 for i,rr in enumerate(refs):
  end=refs[i+1]['span'][0] if i+1<len(refs) else len(caption);tail=caption[rr['span'][1]:end];direction='LEFT_OF' if re.search(r'\bleft\b',tail,re.I) else 'RIGHT_OF' if re.search(r'\bright\b',tail,re.I) else 'UNKNOWN';explicit=bool(re.search(r'\b(image|view|picture|screen|frame)\b',tail,re.I)) and not re.search(r'\bits\b',tail,re.I)
  if direction!='UNKNOWN':relations.append({'subject':rr['entity'],'reference':'T','predicate':direction,'frame':'camera_image' if explicit else 'UNKNOWN','quote':tail.strip(),'origin':'reference_clause_literal_direction'})
 return {'status':'parsed','subject':subject,'category':category,'subject_query':subject_query.strip(),'own_attributes_literal':own,'attribute_binding':binding,'explicit_appearance_field':app,'position_field':pos,'references':refs,'relations':relations,'complete_caption_involved':True,'unknown_unimplemented_clauses_not_satisfied':True}
def pipeline():
 sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'));import stanza,torch
 torch.set_num_threads(4);receipt=read(ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json');return stanza.Pipeline('en',dir=str(ROOT/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**receipt['kwargs'])
def main():
 locked('TEXT_LOCK.json');out=ROOT/'crossplatform/round20/text144_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();nlp=pipeline();counts={}
 for i,r in enumerate(read(EXP.parent/'input/ALL144_PRIVATE.json')):
  g=parse(r['caption'],nlp)
  if g.get('explicit_appearance_field'):assert g['explicit_appearance_field']['text'] in g['subject_query']
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),g);counts[g['attribute_binding']]=counts.get(g['attribute_binding'],0)+1
 dump(out/'COUNTS.json',counts);seal(out,'R20_full_subject_fields_neural_references_no_GT_or_free_JSON',n=144,counts=counts);print('TEXT144_COMPLETE',counts)
if __name__=='__main__':main()
