"""No-label competing-candidate sampling; coordinates never transformed spatially."""
import numpy as np,hashlib
BUDGET=16384

def seed(sid,v):return int(hashlib.sha256(('R65_sampling_v1|'+sid+'|'+v).encode()).hexdigest()[:16],16)
def inside(x,b,gain=1.):return ((np.abs(x-b[:3])<=b[3:]*gain/2).all(1)&bool((b[3:]>0).all()))
def overlap(a,b):
 i=np.maximum(np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2),0).prod();d=np.maximum(a[3:],0).prod()+np.maximum(b[3:],0).prod()-i;return float(i/d) if d>0 else 0.
def choose(boxes,scores,k):
 ans=[]
 for j in np.argsort(-scores,kind='stable'):
  if not (boxes[j,3:]>0).all():continue
  if any(overlap(boxes[j],boxes[h])>=.5 for h in ans):continue
  ans.append(int(j))
  if len(ans)==k:break
 return ans

def design(x,z,reference_mask):
 boxes=z['A_boxes'].astype(float);targets=choose(boxes,z['A_scores'],4);refs=[]
 if reference_mask.any() and (~reference_mask).any():
  from scipy.special import logsumexp
  l=z['A_query'].astype(float)@z['A_tokens'].astype(float).T/.07;s=logsumexp(l[:,reference_mask],1)-logsumexp(l[:,~reference_mask],1);refs=choose(boxes,s,2)
 pools=[np.flatnonzero(inside(x,boxes[j],1.25)) for j in targets];shell=np.zeros(len(x),bool);reference=np.zeros(len(x),bool)
 for j in targets:shell|=inside(x,boxes[j],2.5)&~inside(x,boxes[j],1.25)
 for j in refs:reference|=inside(x,boxes[j],1.5)
 return {'targets':targets,'refs':refs,'pools':pools,'shell':np.flatnonzero(shell),'reference':np.flatnonzero(reference),'focus_union':np.unique(np.concatenate(pools)) if pools else np.array([],np.int64),'context_union':np.flatnonzero(shell|reference)}

def draw(n,d,sid,v,mode):
 rng=np.random.default_rng(seed(sid,v));whole=np.arange(n,dtype=np.int64);fallback=0
 def pick(pool,k):
  nonlocal fallback
  if not len(pool):fallback+=k;pool=whole
  return rng.choice(pool,k,replace=True)
 if mode=='random':indices=pick(whole,BUDGET)
 elif mode=='coverage':
  assert n<=BUDGET;indices=np.concatenate([rng.permutation(whole),pick(whole,BUDGET-n)])
 else:
  assert mode=='directed';parts=[pick(whole,8192)];k=len(d['pools'])
  if k:
   for j,pool in enumerate(d['pools']):parts.append(pick(pool,4096//k+int(j<4096%k)))
  else:parts.append(pick(np.array([],np.int64),4096))
  if len(d['reference']):parts.extend([pick(d['reference'],2048),pick(d['shell'],2048)])
  else:parts.append(pick(d['shell'],4096))
  indices=np.concatenate(parts)
 assert len(indices)==BUDGET and indices.min()>=0 and indices.max()<n;perm=np.random.default_rng(seed(sid,v)^0x65A17).permutation(BUDGET);return indices[perm].astype(np.int64),fallback

def counts(indices,old_indices,d,n,seed_xyz,boxes):
 unique=np.unique(indices);old=np.unique(old_indices);focus=d['focus_union'];context=d['context_union'];seedf=np.zeros(len(seed_xyz),bool);seedc=np.zeros(len(seed_xyz),bool)
 for j in d['targets']:seedf|=inside(seed_xyz,boxes[j],1.25);seedc|=inside(seed_xyz,boxes[j],2.5)&~inside(seed_xyz,boxes[j],1.25)
 for j in d['refs']:seedc|=inside(seed_xyz,boxes[j],1.5)
 return {'raw_n':n,'unique_input_n':len(unique),'new_unique_vs_A':len(np.setdiff1d(unique,old)),'raw_fraction_covered':len(unique)/n,'focus_raw_n':len(focus),'focus_draws':int(np.isin(indices,focus).sum()),'focus_unique_n':int(np.isin(unique,focus).sum()),'context_raw_n':len(context),'context_draws':int(np.isin(indices,context).sum()),'context_unique_n':int(np.isin(unique,context).sum()),'seed_n':len(seed_xyz),'focus_seed_n':int(seedf.sum()),'context_seed_n':int(seedc.sum())}
