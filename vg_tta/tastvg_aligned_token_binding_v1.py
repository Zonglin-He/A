"""Explicit lexical phrase sets and max patch interaction; no native-H cosine."""
import numpy as np
def phrases(words):
 """Deterministic dependency rules over one sentence's Stanza word records."""
 by={w['id']:w for w in words};nouns={'NOUN','PROPN'};wh=[w for w in words if w['lemma'].lower() in {'which','what'} and w['deprel']=='det' and w['head'] in by and by[w['head']]['upos'] in nouns]
 nominal=by[wh[0]['head']] if wh else None;rule='wh_determiner' if wh else None;generic=False
 if nominal is None and any(w['lemma'].lower() in {'who','whom'} for w in words):generic=True;rule='generic_who'
 if nominal is None and not generic:
  nominal=next((w for w in words if w['deprel'].startswith('nsubj') and w['upos'] in nouns),None);rule='nominal_subject' if nominal else None
 if nominal is None and not generic:nominal=next((w for w in words if w['upos'] in nouns),None);rule='first_noun_fallback' if nominal else 'empty_object'
 def noun_phrase(w):
  keep=[x for x in words if x['id']==w['id'] or x['head']==w['id'] and x['deprel'] in {'amod','compound','nummod'} and x['upos'] not in {'PUNCT','DET'}]
  return ' '.join(x['text'].lower() for x in sorted(keep,key=lambda x:x['id']))
 obj=['person'] if generic else [noun_phrase(nominal)] if nominal else [];attrverbs=set()
 if nominal:
  for w in words:
   if w['head']==nominal['id'] and w['deprel'].startswith('nmod') and w['upos'] in nouns:obj.append(noun_phrase(w))
   if w['head']==nominal['id'] and w['deprel'].startswith('acl') and w['lemma'].lower()=='wear':
    attrverbs.add(w['id']);obj.extend(noun_phrase(v) for v in words if v['head']==w['id'] and v['deprel'] in {'obj','obl'} and v['upos'] in nouns)
 evt=[]
 for w in words:
  if w['upos']=='VERB' and w['id'] not in attrverbs:
   modifiers=[x for x in words if x['head']==w['id'] and x['deprel'] in {'advmod','compound:prt'} and x['upos'] in {'ADV','PART'}]
   evt.append(' '.join(x['text'].lower() for x in sorted([w]+modifiers,key=lambda x:x['id'])))
   evt.extend(noun_phrase(v) for v in words if v['head']==w['id'] and (v['deprel'] in {'obj','iobj'} or v['deprel'].startswith('obl')) and v['upos'] in nouns)
 return dict(object=list(dict.fromkeys(x for x in obj if x)),event=list(dict.fromkeys(x for x in evt if x)),referent_rule=rule,attribute_verbs=sorted(attrverbs))

def roi(box,h=14,w=14):
 bb=np.asarray(box,dtype=np.float64);a=np.clip(bb[:2]-bb[2:]/2,0,1);b=np.clip(bb[:2]+bb[2:]/2,0,1)
 x=np.maximum(0,np.minimum((np.arange(w)+1)/w,b[0])-np.maximum(np.arange(w)/w,a[0]));y=np.maximum(0,np.minimum((np.arange(h)+1)/h,b[1])-np.maximum(np.arange(h)/h,a[1]))
 return (y[:,None]*x[None,:]>0).reshape(-1)
def normalize(x):
 x=np.asarray(x,dtype=np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True)
 assert np.isfinite(x).all() and np.all(n>0)
 return x/n
def score(patches,q_obj,q_evt,tubes,interval):
 patches=normalize(patches);objects=normalize(q_obj) if len(q_obj) else np.empty((0,patches.shape[-1]));events=normalize(q_evt) if len(q_evt) else np.empty((0,patches.shape[-1]));qs=np.concatenate([objects,events]);frames,hp,d=patches.shape;side=int(np.sqrt(hp));assert side*side==hp and len(tubes)==9
 sim=(patches.reshape(-1,d)@qs.T).reshape(frames,hp,len(qs)).transpose(0,2,1) if len(qs) else np.empty((frames,0,hp));inside=np.arange(frames);inside=(inside>=interval[0])&(inside<=interval[1]);rows=[];object_scores=[];binding_scores=[];signatures=[]
 for tube in tubes:
  mask=np.stack([roi(b,side,side) for b in tube]);selected=np.where(mask[:,None,:],sim,-np.inf);valid=mask.any(1);inn=inside&valid;out=~inside&valid
  a=selected[inn].max((0,2)) if inn.any() and len(qs) else None;b=selected[out].max((0,2)) if out.any() and len(qs) else None
  obj=float(a[:len(objects)].mean()) if a is not None and len(objects) else None;bind=float((a[len(objects):]-b[len(objects):]).mean()) if a is not None and b is not None and len(events) else None
  object_scores.append(obj);binding_scores.append(bind);signatures.append(np.packbits(mask).tobytes())
  rows.append(dict(inside_valid_frames=int(inn.sum()),outside_valid_frames=int(out.sum()),empty_roi_frames=int((~valid).sum()),inside_patches=int(mask[inside].sum()),outside_patches=int(mask[~inside].sum()),object_tokens=len(objects),event_tokens=len(events),token_in_max=a.tolist() if a is not None else None,token_out_max=b.tolist() if b is not None else None))
 return dict(object_scores=object_scores,binding_scores=binding_scores,availability=rows,unique_ROI_signatures=len(set(signatures)),unique_binding_scores=len(set(x for x in binding_scores if x is not None)),all9_available=all(x is not None for x in binding_scores))

def independent_score(patches,q_obj,q_evt,tubes,interval):
 """Scalar maxima over explicit coordinates, independent of vector mask/einsum."""
 p=normalize(patches);obj=normalize(q_obj) if len(q_obj) else [];evt=normalize(q_evt) if len(q_evt) else [];side=int(np.sqrt(p.shape[1]));out=[]
 for tube in tubes:
  candidates=[[],[]]
  for t,bb in enumerate(tube):
   bb=np.asarray(bb,dtype=np.float64);a=np.maximum(bb[:2]-bb[2:]/2,0);b=np.minimum(bb[:2]+bb[2:]/2,1)
   for y in range(side):
    for x in range(side):
     if min((x+1)/side,b[0])>max(x/side,a[0]) and min((y+1)/side,b[1])>max(y/side,a[1]):candidates[0 if interval[0]<=t<=interval[1] else 1].append(p[t,y*side+x])
  def maxima(q,pool):return [max(float(np.dot(z,v)) for v in pool) for z in q]
  so=float(np.mean(maxima(obj,candidates[0]))) if len(obj) and candidates[0] else None
  st=float(np.mean(np.array(maxima(evt,candidates[0]))-maxima(evt,candidates[1]))) if len(evt) and all(candidates) else None
  out.append((so,st))
 return out
