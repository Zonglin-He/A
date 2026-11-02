"""Independent scalar span primitives. No dataset/model/GT/file access at import."""
import ast,re
import numpy as np
def dictionary_from_source(source_text,name):
 tree=ast.parse(source_text)
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
 raise ValueError('Missing exact literal synonym dictionary')
def choose_official_charspan(caption,gt_class,synonyms):
 # Literal source-training format and loop order. This function is diagnostic:
 # gt_class is forbidden from the text-only parser and any deployment fallback.
 formatted=' '+' '.join(caption.replace(',',' ,').split())+' ';positions=[]
 for word in synonyms.get(gt_class.lower(),[gt_class.lower()]):positions.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',formatted,re.I))
 return formatted,(positions[0] if positions else None)
def positive_mask(tokenized,begin,end,width=256):
 # Match get_positive_map fallback sequence, then evaluator (>0) binarization.
 beg=tokenized.char_to_token(int(begin));last=tokenized.char_to_token(int(end)-1)
 if beg is None:
  try:
   beg=tokenized.char_to_token(int(begin)+1)
   if beg is None:beg=tokenized.char_to_token(int(begin)+2)
  except Exception:beg=None
 if last is None:
  try:
   last=tokenized.char_to_token(int(end)-2)
   if last is None:last=tokenized.char_to_token(int(end)-3)
  except Exception:last=None
 out=np.zeros(width,bool)
 if beg is not None and last is not None:out[beg:last+1]=True
 return out
def span_pair_masks(caption,gt_class,synonyms,tokenizer,actual_token_length):
 formatted,chosen=choose_official_charspan(caption,gt_class,synonyms);actual=caption+' . not mentioned';f=tokenizer([formatted],return_tensors='np');a=tokenizer([actual],return_tensors='np');assert len(a['input_ids'][0])==actual_token_length
 if chosen is None:return None,None,{'status':'no_synonym_match','GT_class_diagnostic_only':True}
 begin,end,word=chosen;literal=positive_mask(f,begin,end);assert caption==' '.join(caption.replace(',',' ,').split());assert 1<=begin<end<=len(caption)+1;remapped=positive_mask(a,begin-1,end-1)
 return literal,remapped,{'status':'matched','chosen_synonym':word,'official_charspan':(begin,end),'actual_charspan':(begin-1,end-1),'formatted_actual_token_ids_equal_without_sentinel_claimed':False,'GT_class_diagnostic_only':True}
def shared_probabilities(query,tokens):
 x=np.asarray(query,np.float64)@np.asarray(tokens,np.float64).T/.07;x-=x.max(-1,keepdims=True);x=np.exp(x);return x/x.sum(-1,keepdims=True)
def score_mask(p,mask):
 mask=np.asarray(mask,bool);assert len(mask)>=p.shape[1];return p@mask[:p.shape[1]]
