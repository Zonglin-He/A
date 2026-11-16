"""Dependency-owned relations and support/counter/unknown conditional feedback."""
import re,hashlib
import numpy as np
COLORS=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black']
def relations86(caption,document,graph):
 if not graph['usable']:return []
 output=[];camera=bool(re.search(r'\b(view|image|picture|observer|viewer|screen|frame)\b',caption,re.I))
 for si,sentence in enumerate(document['sentences']):
  words=sentence['words'];byid={w['id']:w for w in words};children={w['id']:[x for x in words if x['head']==w['id']] for w in words};role={}
  for e in graph['entities']:
   nom=[w for w in words if isinstance(w.get('start_char'),int) and e['span'][0]<=w['start_char']<w['end_char']<=e['span'][1] and w['upos'] in {'NOUN','PROPN'} and w['text'].casefold()==e['head'].casefold()]
   if len(nom)==1:role[nom[0]['id']]=e['role']
  def ancestors(w):
   out=[];seen={w['id']};hd=w['head']
   while hd in byid and hd not in seen:seen.add(hd);out.append(byid[hd]);hd=byid[hd]['head']
   return out
  def subject_owner(w,ref):
   for node in [w]+ancestors(w):
    if node['id'] in role and role[node['id']]!=ref:return role[node['id']]
    subjects=[role[x['id']] for x in children[node['id']] if x['id'] in role and x['deprel'].split(':')[0] in {'nsubj','csubj'} and role[x['id']]!=ref]
    if len(set(subjects))==1:return subjects[0]
   return None
  for w in words:
   pred=str(w.get('lemma') or w['text']).casefold()
   if pred not in {'left','right'}:continue
   path=[w]+ancestors(w)
   if any(str(x.get('lemma') or x['text']).casefold() in {'face','facing','point','pointing','turn','orient'} for x in path):continue
   if any(x['deprel']=='neg' or x['text'].casefold() in {'not','never','its','their','own'} for node in path for x in children[node['id']] if node['id']==w['id'] or (str(node.get('lemma') or node['text']).lower() in {'side','left','right'})):continue
   refs=[];has_of=False
   for child in children[w['id']]:
    cases=[x['text'].casefold() for x in children[child['id']] if x['deprel'].split(':')[0]=='case']
    if 'of' in cases:
     has_of=True
     if child['id'] in role:refs.append(role[child['id']])
   # UD also permits LEFT as a case/adverbial modifier of the reference noun.
   if not refs and w['head'] in role and w['deprel'].split(':')[0] in {'case','advmod'}:
    refs=[role[w['head']]]
   if len(set(refs))==1:
    ref=refs[0];owner=subject_owner(w,ref)
    if owner and owner!=ref:output.append({'subject':owner,'reference':ref,'predicate':pred.upper(),'known_frame':camera,'origin':'direct_dependency_governor','sentence':si,'governor_word':w['id']})
   elif not refs and not has_of:
    # A single existential surrounding object may be located relative to T.
    owner=subject_owner(w,None);existential=any(x['text'].casefold()=='there' and x['deprel']=='expl' for x in words)
    if existential and owner and owner!='T' :
     output.append({'subject':owner,'reference':'T','predicate':pred.upper(),'known_frame':camera,'origin':'existential_surrounding_clause','sentence':si,'governor_word':w['id']})
 unique={}
 for e in output:
  key=(e['subject'],e['reference'],e['predicate']);unique.setdefault(key,e)
 return list(unique.values())

def group_attribute86(means,counts,cover,request,labels):
 G=int(labels.max())+1;s=np.zeros(G);c=np.zeros(G)
 if len(request)!=1 or request[0] not in COLORS:return s,c
 k=COLORS.index(request[0]);alt=np.max(np.delete(means,k,axis=-1),axis=-1);wanted=means[:,:,k];rel=counts.min(0)/(counts.min(0)+8)*np.minimum(1,cover.min(0)/.3);support=np.maximum(0,(wanted-alt).min(0))*rel;counter=np.maximum(0,(alt-wanted).min(0))*rel
 for j in range(G):s[j]=np.median(support[labels==j]);c[j]=np.median(counter[labels==j])
 return s,c

def transfer86(prior,support,counter):
 """Move mass only counter -> support; all UNKNOWN entries bit-identical."""
 p=np.array(prior,float,copy=True);S=support>1e-10;N=counter>1e-10;known=S|N
 if not S.any() or not N.any():return p,False
 removed=p[N]*(-np.expm1(-counter[N]));amount=removed.sum();p[N]-=removed;allocation=p[S]*support[S];den=allocation.sum()
 if den<=0:return np.array(prior,float,copy=True),False
 p[S]+=amount*allocation/den
 assert np.array_equal(p[~known],np.asarray(prior)[~known]) and abs(p.sum()-np.sum(prior))<1e-12 and (p>=0).all()
 return p,True

def group_geometry86(ev):
 labels=ev['labels'];G=len(ev['reps']);lo=np.zeros(G);hi=np.zeros(G);r=np.zeros(G)
 for g in range(G):
  ix=np.flatnonzero((labels==g)&(ev['reliability']>0))
  if len(ix):lo[g]=np.median(ev['interval'][ix,0]);hi[g]=np.median(ev['interval'][ix,1]);r[g]=np.median(ev['reliability'][ix])
 return lo,hi,r

def relation_masses86(ev,predicate):
 lo,hi,r=group_geometry86(ev);left_gap=lo[None,:]-hi[:,None]-.01;right_gap=lo[:,None]-hi[None,:]-.01;rel=np.sqrt(r[:,None]*r[None,:]);positive=rel*np.tanh(np.maximum(0,left_gap if predicate=='LEFT' else right_gap)/.1);negative=rel*np.tanh(np.maximum(0,right_gap if predicate=='LEFT' else left_gap)/.1);np.fill_diagonal(positive,0);np.fill_diagonal(negative,0);return positive,negative

def factors86(graph,entity_scores,ev):
 labels=ev['labels'];G=len(ev['reps']);out=[];roles=[e['role'] for e in graph['entities']];priors={};attrs={}
 if not graph['usable']:return out,{'reference_edges':0}
 for n,e in enumerate(graph['entities']):
  score=np.array([np.logaddexp.reduce(entity_scores[n,labels==g])-np.log((labels==g).sum()) for g in range(G)]);p=np.exp(score-score.max());p/=p.sum();s,c=group_attribute86(ev['means'],ev['counts'],ev['coverage'],e['attributes'],labels);attrs[e['role']]=(s,c);priors[e['role']]=transfer86(p,s,c)[0]
 s,c=attrs['T']
 if (s>1e-10).any() and (c>1e-10).any():out.append({'kind':'target_attribute','role':'T','support':s,'counter':c})
 edges=[e for e in graph['relations'] if e['known_frame'] and e['predicate'] in {'LEFT','RIGHT'} and e['subject'] in roles and e['reference'] in roles]
 # One synchronous reference-reference message pass, not a global identity oracle.
 messages={k:[] for k in priors if k!='T'};refedges=0
 for e in edges:
  a,b=e['subject'],e['reference']
  if a=='T' or b=='T':continue
  pos,neg=relation_masses86(ev,e['predicate']);refedges+=1
  for owner,other,P,N in [(a,b,pos,neg),(b,a,pos.T,neg.T)]:
   signed=(P-N)@priors[other];messages[owner].append((np.maximum(0,signed),np.maximum(0,-signed)))
 updated=dict(priors)
 for role,ms in messages.items():
  for s,c in ms:updated[role]=transfer86(updated[role],s,c)[0]
 for e in edges:
  a,b=e['subject'],e['reference']
  if 'T' not in [a,b]:continue
  pos,neg=relation_masses86(ev,e['predicate']);rr=b if a=='T' else a
  if b=='T':pos,neg=pos.T,neg.T
  positive=pos@updated[rr];negative=neg@updated[rr];signed=positive-negative;s=np.maximum(0,signed);c=np.maximum(0,-signed)
  if (s>1e-10).any() and (c>1e-10).any():out.append({'kind':'owned_relation','role':rr,'support':s,'counter':c,'ambiguous_reference_mass':np.minimum(positive,negative)*2})
 return out,{'reference_edges':refedges,'direct_target_edges':sum('T' in [e['subject'],e['reference']] for e in edges)}

def target86(prior,factors,labels,seedkey,mismatch=False):
 prior=np.array(prior,float);p=prior.copy();base=int(prior.argmax());bg=int(labels[base]);eligible=labels==bg;G=int(labels.max())+1;perm=np.random.default_rng(int(hashlib.sha256(('R86mismatch|'+seedkey).encode()).hexdigest()[:16],16)).permutation(G);logs=[];union=np.zeros(len(p),bool)
 for f in factors:
  s=f['support'][perm] if mismatch else f['support'];c=f['counter'][perm] if mismatch else f['counter'];ss=s[labels];cc=c[labels];known=(ss>1e-10)|(cc>1e-10);before=p.copy();p,active=transfer86(p,ss,cc)
  if active:
   union|=known
   if c[bg]>1e-10:eligible|=ss>1e-10
  logs.append({'kind':f['kind'],'support_groups':int((s>1e-10).sum()),'counter_groups':int((c>1e-10).sum()),'unknown_groups':int(((s<=1e-10)&(c<=1e-10)).sum()),'active':active,'unknown_probability_error':float(np.max(abs(p[~known]-before[~known]))) if (~known).any() else 0.,'base_group_has_counter':bool(c[bg]>1e-10)})
 assert np.array_equal(p[~union],prior[~union]) and abs(p.sum()-prior.sum())<1e-12
 return p,eligible,logs,union

def guarded86(probability,eligible):
 # Expose the raw arm too: a local preference never silently becomes global identity.
 p=np.array(probability,copy=True);p[~eligible]=-1;return p
