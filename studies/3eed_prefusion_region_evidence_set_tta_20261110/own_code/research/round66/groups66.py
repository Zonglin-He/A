"""Spatial computation groups; no identity truth, text-model caches or C view."""
import numpy as np,re
from scipy.sparse import csr_matrix
from scipy.special import logsumexp,softmax
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
def category(text):
 m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text,re.I);return m.group(0) if m else None

def overlap(a,b):
 lo=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);hi=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);i=np.maximum(hi-lo,0).prod(-1);u=np.maximum(a[:,3:],0).prod(-1)[:,None]+np.maximum(b[:,3:],0).prod(-1)[None]-i;return np.divide(i,u,out=np.zeros_like(i),where=u>0)
def support(x,boxes):return ((np.abs(x[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1)&(boxes[:,3:]>0).all(1)[:,None])
def affinity(a,b,x):
 aa=support(x,a).astype(np.float32);bb=support(x,b).astype(np.float32);shared=(csr_matrix(aa)@csr_matrix(bb).T).toarray();union=aa.sum(1)[:,None]+bb.sum(1)[None]-shared;j=np.divide(shared,union,out=np.zeros_like(shared),where=union>0);w=np.sqrt(np.maximum(j*overlap(a,b),0));w[shared<4]=0;return w.astype(np.float32),j,shared

def native(z,prefix=''):
 l=np.einsum('ik,jk->ij',z[prefix+'query'].astype(float),z[prefix+'tokens'].astype(float))/.07;m=z[prefix+'token_mask'];return softmax(logsumexp(l[:,m],1)-logsumexp(l[:,~m],1))
def transport(p,W,mass,base):
 q=W@p;return mass*q/max(float(q.sum()),1e-300)+(1-mass)*base

def make_groups(boxes,xyz,pAB):
 _,j,shared=affinity(boxes,boxes,xyz);io=overlap(boxes,boxes);unused=np.ones(256,bool);groups=[]
 for rep in np.argsort(-pAB,kind='stable'):
  if not unused[rep]:continue
  members=np.flatnonzero(unused&(io[rep]>=.5)&(j[rep]>=.5)&(shared[rep]>=4));members=np.unique(np.r_[rep,members]);unused[members]=False;groups.append(members.astype(np.int64))
 assert not unused.any() and len(np.unique(np.concatenate(groups)))==256
 order=sorted(range(len(groups)),key=lambda k:(-float(pAB[groups[k]].sum()),int(groups[k][0])))
 return groups,order[:4]

def masks(xyz,boxes,groups,pool):
 region=[support(xyz,boxes[groups[g]]).any(0) for g in pool];union=np.any(region,axis=0);outside=np.flatnonzero(~union);out=[]
 for mask in region:
  n=int(mask.sum());control=np.zeros(len(xyz),bool);valid=4<=n<=len(xyz)-4 and len(outside)>=n
  if valid:
   center=xyz[mask].mean(0);seed=outside[np.argmax(np.linalg.norm(xyz[outside]-center,axis=1))];ids=outside[np.argsort(np.linalg.norm(xyz[outside]-xyz[seed],axis=1),kind='stable')[:n]];control[ids]=True;assert not (mask&control).any() and mask.sum()==control.sum()
  out.append((mask,control,valid))
 return out
