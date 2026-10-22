"""Recompose all same-subject predicates into one joint positive native caption."""
import re
from research.native_event_composition_v1.core import from_words

def joint_variants(words,caption):
 dc=from_words(words,caption)
 if not dc['supported']:return dict(supported=False,parent=dc)
 by={w['id']:w for w in words};children={i:[]for i in by}
 for w in words:
  if w['head']in children:children[w['head']].append(w['id'])
 verbs=set(dc['main_verbs']);vv=sorted(verbs);root=next(w['id']for w in words if w['head']==0)
 subject=next(i for i in children[root]if by[i]['rel']=='nsubj')
 def subtree(i,exclude):
  if i in exclude:return set()
  z={i}
  for ch in children[i]:z|=subtree(ch,exclude)
  return z
 subj=subtree(subject,verbs);real=[];swapped=[];implicit=[];slots=[]
 def render(ids,replace=None):
  text=' '.join((replace or{}).get(i,by[i]['text'])for i in sorted(ids))
  return re.sub(r'\s+([,.;:!?])',r'\1',text).strip(' ,.;:!?')
 for j,v in enumerate(vv):
  keep=subtree(v,verbs-{v})|subj
  keep={i for i in keep if by[i]['rel']not in {'cc','punct'}and not(by[i]['rel']=='nsubj'and i!=subject)}
  assert v in keep
  slots.append(sorted(keep));real.append(render(keep));swapped.append(render(keep,{v:by[vv[(j+1)%len(vv)]]['text']}));implicit.append(render(keep if j==0 else keep-subj))
 assert [x+'.'for x in real]==dc['clauses']
 return dict(supported=True,parent=dc,canonical=' and '.join(real)+'.',predicate_binding_control=' and '.join(swapped)+'.',implicit_shared_subject=' and '.join(implicit)+'.',clause_word_IDs=slots,main_predicate_word_IDs=vv,nominal_subject_word_IDs=sorted(subj))

def from_parser(caption,parser):
 doc=parser.nlp(caption)
 if len(doc.sentences)!=1:return dict(supported=False,parent=dict(supported=False,reason='multiple_sentences',clauses=[caption]))
 words=[dict(id=w.id,head=w.head,upos=w.upos,rel=w.deprel.split(':')[0],lemma=w.lemma,text=w.text)for w in doc.sentences[0].words]
 return joint_variants(words,caption)
