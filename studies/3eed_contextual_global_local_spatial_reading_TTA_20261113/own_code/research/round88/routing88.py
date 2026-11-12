"""Multiple spatial hypotheses and owned roles, fixed before updates; no GT."""
import numpy as np,re,hashlib

def token_roles88(base,r,z,graph,doc):
 text=r['caption']+' . not mentioned';tok=base.tokenizer(text,return_offsets_mapping=True);offset=np.array(tok['offset_mapping']);assert len(offset)==z['lang0'].shape[1]
 roles=np.zeros((len(graph['entities']),len(offset)),bool);assigned=set()
 for e,entity in enumerate(graph['entities']):
  spans=[entity['span']]
  # Contextual encoding remains full-sentence; route modifiers owned by dependency, too.
  for s in doc['sentences']:
   words=s['words'];roots=[w['id'] for w in words if w.get('start_char',-1)>=entity['span'][0] and w.get('end_char',10**9)<=entity['span'][1] and w['upos'] in ['NOUN','PROPN']];members=set(roots)
   for _ in words:
    new={w['id'] for w in words if w['head'] in members and w['deprel'].split(':')[0] in ['amod','compound','det','nummod','flat','fixed','acl','obj','advmod'] and not any(other['span'][0]<=w.get('start_char',-1)<other['span'][1] for j,other in enumerate(graph['entities']) if j!=e)}-members
    if not new:break
    members|=new
   spans += [[w['start_char'],w['end_char']] for w in words if w['id'] in members and 'start_char' in w]
  if e==0:
   match=re.search(r'\bmain\s+appearance\s+features\s*:\s*([^;\n]+)',r['caption'],re.I)
   if match:spans.append([match.start(1),match.end(1)])
  for i,(lo,hi) in enumerate(offset):
   if hi>lo and any(lo<b and hi>a for a,b in spans) and i not in assigned:roles[e,i]=True;assigned.add(i)
 return roles,offset

def build88(r,z,d,graph,roles,attention):
 xyz=z['xyz'][0];boxes=z['boxes'];inside=(np.abs(xyz[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1);E=len(roles);H=3;support=np.zeros((E,H,1024),np.float32);weight=np.zeros((E,H),np.float32);hyp=np.full((E,H),-1,int);moments=np.zeros((E,H,3));spread=np.zeros((E,H,3));interval=np.zeros((E,H,2));confidence=np.zeros((E,H));visible=d['visible'];uv=d['uv_grid'][:,0]/48
 for e,mask in enumerate(roles):
  if not mask.any():continue
  att=attention[mask].mean(0);mass=inside@att;score=mass/np.sqrt(np.maximum(inside.sum(1),1));chosen=[]
  for q in np.argsort(-score,kind='stable'):
   if inside[q].sum()<2:continue
   if any((inside[q]&inside[j]).sum()/max(1,(inside[q]|inside[j]).sum())>.5 for j in chosen):continue
   chosen.append(int(q))
   if len(chosen)==H:break
  for h,q in enumerate(chosen):
   hyp[e,h]=q;support[e,h]=inside[q];weight[e,h]=max(float(mass[q]),1e-8);p=att*inside[q];p=p/max(float(p.sum()),1e-12);mu=p@xyz;moments[e,h]=mu;spread[e,h]=(p[:,None]*(xyz-mu)**2).sum(0);vv=visible&inside[q]
   if vv.sum()>=3:interval[e,h]=np.quantile(uv[vv],[.1,.9]);confidence[e,h]=len(np.unique(d['cell'][vv]))/(len(np.unique(d['cell'][vv]))+8)
  if weight[e].sum():weight[e]/=weight[e].sum()
 plain=weight.copy();owned=0;roles0=[x['role'] for x in graph['entities']];factors=[]
 for edge in graph['relations']:
  if not edge['known_frame'] or edge['predicate'] not in ['LEFT','RIGHT'] or edge['subject'] not in roles0 or edge['reference'] not in roles0:continue
  a,b=roles0.index(edge['subject']),roles0.index(edge['reference']);lo=interval[a,:,0,None];hi=interval[a,:,1,None];rl=interval[b,None,:,0];rh=interval[b,None,:,1]
  separation=(rl-hi) if edge['predicate']=='LEFT' else (lo-rh);reliability=np.sqrt(confidence[a,:,None]*confidence[b,None,:]);factor=np.exp(reliability*np.tanh(separation/.1));factors.append((a,b,factor));owned+=1
 # One symmetric evidence exchange; no identity truth or fixed reference Top1.
 for a,b,f in factors:weight[a]*=f@plain[b];weight[b]*=f.T@plain[a]
 for e in range(E):
  if weight[e].sum():weight[e]/=weight[e].sum()
 def decoder(w,swap=False):
  allreg=np.max(support,axis=(0,1)) if E else np.zeros(1024);out=np.ones((256,1024),np.float32)
  target=(w[1]@support[1]) if swap and E>1 else (w[0]@support[0] if E else np.zeros(1024))
  if E and target.max()>0:
   for q in range(256):
    # Preserve candidate-local evidence and all reference context. Never prune other queries.
    out[q]=.5*inside[q]+.25*allreg+.25*target
  return out
 # Partial relation-only pair: comparison is conditional on all retained visible reference hypotheses.
 pairs=[];pairweights=[];rank_candidates=np.argsort(-z['frozen_probability'],kind='stable')[:8]
 for a,b,f in factors:
  if a!=0 and b!=0:continue
  ref=b if a==0 else a;refs=np.flatnonzero((plain[ref]>0)&(confidence[ref]>.25))
  if not len(refs):continue
  isleft=next(ed['predicate']=='LEFT' for ed in graph['relations'] if ed['known_frame'] and ed['subject']==roles0[a] and ed['reference']==roles0[b] and ed['predicate'] in ['LEFT','RIGHT']);sgn=(1 if isleft else -1)*(1 if a==0 else -1)
  compat={}
  for q in rank_candidates:
   ix=visible&inside[q]
   if ix.sum()<3:continue
   lo,hi=np.quantile(uv[ix],[.1,.9]);dif=[(interval[ref,h,0]-hi if sgn>0 else lo-interval[ref,h,1]) for h in refs];compat[int(q)]=dif
  good=[q for q,v in compat.items() if min(v)>.02];bad=[q for q,v in compat.items() if max(v)<-.02]
  if good and bad:
   good=max(good,key=lambda q:z['frozen_probability'][q]);bad=max(bad,key=lambda q:z['frozen_probability'][q]);pairs.append([good,bad]);pairweights.append(float(min(confidence[ref,refs])))
 region=support.any((0,1));image=np.zeros(1536,bool);image[d['cell'][region&visible]]=True;seed=int(hashlib.sha256(('R88spaceperm|'+r['id']).encode()).hexdigest()[:16],16)
 return {'token_roles':roles,'supports':support,'weights_plain':plain,'weights_relation':weight,'hypotheses_PRIVATE':hyp,'support_centers_PRIVATE':moments,'support_variances_PRIVATE':spread,'support_intervals_PRIVATE':interval,'decoder_plain':decoder(plain),'decoder_relation':decoder(weight),'decoder_swapped':decoder(weight,True),'point_perm':np.random.default_rng(seed).permutation(1024),'rank_pairs':np.array(pairs,np.int64).reshape(-1,2),'rank_weights':np.array(pairweights,np.float32),'region_point':region,'region_image':image}, {'roles':E,'routed_tokens':int(roles.sum()),'hypotheses':int((hyp>=0).sum()),'executable_relations':owned,'partial_pairs':len(pairs),'unknown_support':int(not region.any())}
