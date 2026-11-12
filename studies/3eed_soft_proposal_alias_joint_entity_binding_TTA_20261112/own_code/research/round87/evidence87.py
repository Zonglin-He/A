"""Soft proposal aliasing and finite sum-product entity binding; no labels."""
import numpy as np,hashlib
from scipy.special import logsumexp
def norm87(x):
 x=np.maximum(np.asarray(x,float),1e-300);return x/x.sum()
def alias87(boxes,support,features,cells,visible):
 lo=boxes[:,:3]-boxes[:,3:]/2;hi=boxes[:,:3]+boxes[:,3:]/2;inter=np.maximum(0,np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None])).prod(-1);vol=boxes[:,3:].prod(-1);iou=inter/np.maximum(1e-12,vol[:,None]+vol[None]-inter)
 s=support.astype(float);n=s.sum(1);dice=2*(s@s.T)/np.maximum(1,n[:,None]+n[None]);f=s@features/np.maximum(1,n[:,None]);f/=np.maximum(1e-12,np.linalg.norm(f,axis=1,keepdims=True));cos=np.maximum(0,f@f.T);pixels=np.zeros((len(boxes),1536))
 for i in range(len(boxes)):pixels[i,np.unique(cells[support[i]&visible])]=1
 pn=pixels.sum(1);pdice=2*(pixels@pixels.T)/np.maximum(1,pn[:,None]+pn[None]);geometric=np.where((pn[:,None]>0)&(pn[None]>0),np.sqrt(dice*pdice),dice);A=np.clip(iou*geometric*cos,0,1);np.fill_diagonal(A,1);assert np.allclose(A,A.T);return A
def attributes87(ev,requests,A):
 colors=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black'];N=len(A);S=[];C=[];weight=A/A.sum(1,keepdims=True)
 for request in requests:
  s=np.zeros(N);c=s.copy()
  if len(request)==1 and request[0] in colors:
   k=colors.index(request[0]);v=ev['means'][:,:,k];alt=np.delete(ev['means'],k,axis=-1).max(-1);rel=ev['counts'].min(0)/(ev['counts'].min(0)+8)*np.minimum(1,ev['coverage'].min(0)/.3);s=np.maximum(0,(v-alt).min(0))*rel;c=np.maximum(0,(alt-v).min(0))*rel
  S.append(weight@s);C.append(weight@c)
 return np.array(S),np.array(C)
def target87(prior,entity_scores,graph,ev,A,mode='full',seed='',mismatch=False):
 N=len(prior);R=len(graph['entities']);p0=norm87(prior)
 if not graph['usable'] or not R:return p0,{'active':False,'roles':R,'edges':0,'reference_null_mean':None},np.zeros((R,N+1))
 es=np.asarray(entity_scores,float).copy();lo,hi=ev['interval'].T;rel=ev['reliability'].copy();A=A.copy();ss,cc=attributes87(ev,[e['attributes'] for e in graph['entities']],A)
 if mode=='hard_alias':A=(ev['labels'][:,None]==ev['labels'][None,:]).astype(float);ss,cc=attributes87(ev,[e['attributes'] for e in graph['entities']],A)
 if mode=='no_alias':A=np.eye(N);ss,cc=attributes87(ev,[e['attributes'] for e in graph['entities']],A)
 if mode=='no_RGB':ss*=0;cc*=0
 if mismatch:
  rng=np.random.default_rng(int(hashlib.sha256(('R87all_feedback|'+seed).encode()).hexdigest()[:16],16));perm=rng.permutation(N);A=A[perm][:,perm];es=es[:,perm];ss=ss[:,perm];cc=cc[:,perm];lo=lo[perm];hi=hi[perm];rel=rel[perm]
 density=A.sum(1);W=A/density[:,None];unary=[]
 for r in range(R):
  semantic=norm87(W@np.exp(es[r]-es[r].max()))
  u=norm87(np.sqrt(p0*semantic)) if r==0 and mode!='no_subject' else p0.copy() if r==0 else semantic
  u=norm87(u*np.exp(ss[r]-cc[r])/density)
  if r==0:unary.append(np.r_[u,0.])
  elif mode=='no_null':unary.append(np.r_[u,0.])
  else:unary.append(np.r_[u*.5,.5])
 roles={e['role']:i for i,e in enumerate(graph['entities'])};pairs={(a,b):np.zeros((N,N)) for a in range(R) for b in range(a+1,R)};sign_edges=0;known_pairs=0
 for e in graph['relations']:
  if mode=='no_relation' or not e['known_frame'] or e['predicate'] not in ['LEFT','RIGHT'] or e['subject'] not in roles or e['reference'] not in roles:continue
  a,b=roles[e['subject']],roles[e['reference']]
  if a==b:continue
  left=lo[None,:]-hi[:,None]-.01;right=lo[:,None]-hi[None,:]-.01;geo=np.sqrt(rel[:,None]*rel[None,:])*(np.tanh(np.maximum(0,left)/.1)-np.tanh(np.maximum(0,right)/.1));geo=geo if e['predicate']=='LEFT' else -geo
  if a>b:a,b=b,a;geo=geo.T
  key=(a,b);pairs[key]=pairs.get(key,0)+geo;sign_edges+=1;known_pairs+=int((abs(geo)>1e-10).sum())
 potentials={};neighbors={r:[] for r in range(R)}
 for (a,b),score in pairs.items():
  potential=np.ones((N+1,N+1));potential[:N,:N]=np.exp(score)*(np.ones_like(A) if mode=='no_exclusivity' else np.maximum(0,1-A));potentials[(a,b)]=potential;potentials[(b,a)]=potential.T;neighbors[a].append(b);neighbors[b].append(a)
 messages={(a,b):np.ones(N+1) for a,b in potentials}
 if mode=='independent':
  # References remain frozen; no target-conditioned or reference-reference binding.
  for a in neighbors[0]:messages[(a,0)]=potentials[(a,0)].T@unary[a]
 else:
  for _ in range(4):
   new={}
   for (a,b),K in potentials.items():
    v=unary[a].copy()
    for c in neighbors[a]:
     if c!=b:v*=messages[(c,a)]
    new[(a,b)]=norm87(K.T@norm87(v))
   messages=new
 beliefs=[]
 for r in range(R):
  v=unary[r].copy()
  for c in neighbors[r]:v*=messages[(c,r)]
  beliefs.append(norm87(v))
 beliefs=np.array(beliefs);p=norm87(beliefs[0,:N]);return p,{'active':bool(np.max(abs(p-p0))>1e-10),'roles':R,'edges':sign_edges,'known_pair_comparisons':known_pairs,'reference_null_mean':float(beliefs[1:,-1].mean()) if R>1 else None,'soft_alias_offdiagonal_mass':float(A.sum()-N),'target_RGB_support_queries':int((ss[0]>1e-10).sum()),'target_RGB_counter_queries':int((cc[0]>1e-10).sum()),'max_target_change':float(abs(p-p0).max())},beliefs
