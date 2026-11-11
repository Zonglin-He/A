"""Literal local parsing; no LLM/external semantic output or GT spans."""
import re,sys,numpy as np
from common72 import *
sys.path.insert(0,str(ROOT/'research/round66'));from groups66 import LEXICON
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from attributes51 import COLOR_RGB_PROTOTYPES,COLOR_ALIASES
COLORS=list(COLOR_RGB_PROTOTYPES);entities=LEXICON+('wall','tree','pole','building','fence','bench','chair','sign','motorcycle','lamp','barrier','door','window')
COLOR_RE=re.compile(r'\b(?:'+'|'.join(COLORS+list(COLOR_ALIASES))+r')\b',re.I)
REL_RE=re.compile(r'\b(?:(?:to|on)\s+(?:the\s+)?)?(left|right)\s+(?:side\s+)?of\b',re.I)
ALLREL=re.compile(r'\b(?:next to|beside|behind|in front of|near|closest to|nearest to|between|(?:to (?:the )?)?(?:left|right) of)\b',re.I)
ENT_RE=re.compile(r'\b(?:'+'|'.join(entities)+r')s?\b',re.I)
GARMENT=re.compile(r'\b(shirt|t-shirt|jacket|coat|hoodie|blouse|sweater|jersey|vest|pants|trousers|jeans|shorts)\b',re.I)
_TOKENIZER=None
def tokenizer():
 global _TOKENIZER
 if _TOKENIZER is None:
  from transformers import AutoTokenizer
  _TOKENIZER=AutoTokenizer.from_pretrained(str(ROOT/'crossplatform/upstream/data/roberta_base'),local_files_only=True,use_fast=True)
 return _TOKENIZER

def request(fragment,kind):
 colors=list(dict.fromkeys(COLOR_ALIASES.get(x.group(0).lower(),x.group(0).lower()) for x in COLOR_RE.finditer(fragment)));neg=bool(re.search(r'\b(not|no|without|except)\b',fragment,re.I));g=GARMENT.search(fragment);person=bool(re.search(r'\b(person|pedestrian|man|woman|people|boy|girl|child|worker)\b',kind,re.I));scope='whole'
 if person:scope='upper' if g and g.group(1).lower() not in ['pants','trousers','jeans','shorts'] else 'lower' if g else 'UNKNOWN_person_garment_scope'
 return {'color':colors[0] if len(colors)==1 and not neg else None,'scope':scope,'negation':neg,'multi_color':len(colors)>1}

def parse(caption,ntokens):
 text=caption+' . not mentioned';tok=tokenizer()(text,return_offsets_mapping=True);off=np.asarray(tok['offset_mapping']);assert len(off)==ntokens,('token_alignment',len(off),ntokens);first=ENT_RE.search(caption);cut=ALLREL.search(caption);end=cut.start() if cut and first and cut.start()>first.end() else len(caption);target=caption[:end];kind=first.group(0).lower() if first else '';q=request(target,kind);edge=REL_RE.search(caption);refmask=np.zeros(ntokens,bool);rreq={'color':None,'scope':'whole','negation':False,'multi_color':False};reftext='';relation=None
 if edge and first and edge.start()>first.end():
  a=edge.end();tail=caption[a:];stop=re.search(r'[,.;]|\b(?:which|that|who|next to|behind|in front of|between)\b',tail,re.I);b=a+(stop.start() if stop else len(tail));reftext=caption[a:b].strip();rm=ENT_RE.search(reftext)
  if rm:
   rreq=request(reftext,rm.group(0).lower());refmask=(off[:,1]>a)&(off[:,0]<b)&(off[:,1]>off[:,0]);frame=bool(re.search(r'\b(?:image|picture|screen|photo|camera view)\b',caption,re.I));relation=edge.group(1).lower() if frame else None
 return {'target':q,'reference':rreq,'ref_mask':refmask,'relation':relation,'has_relation_literal':edge is not None,'explicit_image_frame':relation is not None,'target_literal_PRIVATE':target,'reference_literal_PRIVATE':reftext,'offsets_PRIVATE':off,'target_entity_known':first is not None}
