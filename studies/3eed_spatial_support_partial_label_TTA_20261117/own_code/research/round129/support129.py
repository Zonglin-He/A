"""Spatially qualified partial-label supervision; no model or task-GT imports.

Supports are unique A-seed coordinates inside immutable source boxes, not full
object extents or attention masks. Complete-link duplicates need BOTH high box
IoU and high support Jaccard. Every group receives one probability vote, using
log-mean-exp rather than a count-dependent log-sum-exp.
"""
import numpy as np,torch,hashlib
from torch.nn import functional as F

def geometry(xyz,boxes,valid):
 xyz=np.asarray(xyz);boxes=np.asarray(boxes);valid=np.asarray(valid,bool)
 unique,inverse=np.unique(xyz,axis=0,return_inverse=True);vb=np.zeros(len(unique),bool);np.logical_or.at(vb,inverse,valid)
 size=np.maximum(boxes[:,3:6],1e-6);inside=(abs(unique[None]-boxes[:,None,:3])<=size[:,None]/2).all(2);counts=inside.sum(1);lo=boxes[:,:3]-size/2;hi=boxes[:,:3]+size/2
 inter=np.maximum(np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None]),0).prod(2);vol=size.prod(1);iou=inter/np.maximum(vol[:,None]+vol[None]-inter,1e-30)
 its=inside.astype(np.float32)@inside.astype(np.float32).T;union=counts[:,None]+counts[None]-its;jacc=np.divide(its,union,out=np.zeros_like(its),where=union>0);same=(iou>=.8)&(jacc>=.8);np.fill_diagonal(same,True)
 groups=[]
 for i in range(256):
  for g in groups:
   if same[i,g].all():g.append(i);break
  else:groups.append([i])
 gs=np.array([inside[g].any(0) for g in groups]);n=gs.sum(1);nb=(gs&vb[None]).sum(1);coverage=nb/np.maximum(n,1);assignment=np.empty(256,np.int64)
 for j,g in enumerate(groups):assignment[g]=j
 return {'groups':groups,'assignment':assignment,'n_A':n,'n_B':nb,'coverage_B':coverage,'unique_seed_count':len(unique),'valid_unique_seed_count':int(vb.sum()),'support':gs,'unique_xyz':unique}

def glogits(logits,groups):return torch.stack([torch.logsumexp(logits[g],0)-np.log(len(g)) for g in groups])
def glogits_np(prob,groups):
 p=np.asarray(prob,dtype=np.float64);x=np.array([p[g].mean() for g in groups]);return np.log(np.maximum(x,1e-300))
def qualify(geo,minimum,coverage):return (geo['n_A']>=minimum)&(geo['n_B']>=minimum)&(geo['coverage_B']>=coverage)
def soft(x):x=x-x.max();p=np.exp(x);return p/p.sum()
def sets(pA,pB,geo,cal):
 S=qualify(geo,cal['minimum_support_points'],cal['minimum_registered_fraction']);C=np.zeros_like(S);T=np.zeros(len(S),np.float64);diag={'S_groups':int(S.sum()),'C_groups':0,'excluded_groups':0,'unobservable_groups':int((~S).sum()),'has_partial_information':False}
 if S.sum()>=2:
  aa=glogits_np(pA,geo['groups']);bb=glogits_np(pB,geo['groups']);pa,pb=soft(aa[S]),soft(bb[S]);T[S]=(pa+pb)/2;ca=(pa.max()-pa)<=cal['tau'];cb=(pb.max()-pb)<=cal['tau'];C[S]=ca|cb;diag.update(C_groups=int(C.sum()),excluded_groups=int(S.sum()-C.sum()),has_partial_information=bool(C.any() and (S&~C).any()),teacher_A_retained_groups=int(ca.sum()),teacher_B_retained_groups=int(cb.sum()))
 return S,C,T,diag
def taskloss(logits,groups,S,C,T,kind):
 if kind=='global':return (T*(T.clamp_min(1e-300).log()-logits)).sum()
 gs=glogits(logits,groups);ls=gs[S].log_softmax(0)
 if kind=='conditional':t=T[S];return (t*(t.clamp_min(1e-300).log()-ls)).sum()
 # If C=S this returns exact algebraic zero and direct derivative zero.
 return torch.logsumexp(gs[S],0)-torch.logsumexp(gs[C],0)
def collapse(prob,query_S):return torch.cat([prob[~query_S],prob[query_S].sum()[None]])
def keep_kl(reference,current):return (reference*(reference.clamp_min(1e-300).log()-current.clamp_min(1e-300).log())).sum().clamp_min(0)
def direct(prob,groups,S,C,T,kind):
 p=np.asarray(prob,dtype=np.float64).copy();ids=np.concatenate([groups[j] for j in np.flatnonzero(S)]) if S.any() else np.array([],int)
 if S.sum()<2:return p
 mass=p[ids].sum()
 if kind=='conditional':
  weights=np.array([len(groups[j])*T[j] for j in np.flatnonzero(S)]);weights/=weights.sum()
  for j,w in zip(np.flatnonzero(S),weights):g=groups[j];p[g]*=mass*w/max(p[g].sum(),1e-300)
 elif C.any() and (S&~C).any():
  allowed=np.concatenate([groups[j] for j in np.flatnonzero(C)]);old=p[allowed].sum();p[ids]=0;p[allowed]=np.asarray(prob)[allowed]*mass/max(old,1e-300)
 return p/p.sum()
def wrong_sets(groups,S,C,salt):
 # Permute groups only within identical multiplicities, preserving query and
 # hypothesis cardinalities. Identity bins are retained rather than invented.
 rng=np.random.default_rng(int(hashlib.sha256((salt+'|support_derangement').encode()).hexdigest()[:16],16));perm=np.arange(len(groups))
 for n in sorted(set(map(len,groups))):
  ids=np.array([j for j,g in enumerate(groups) if len(g)==n]);order=rng.permutation(ids)
  if len(order)>1:perm[order]=np.roll(order,1)
 return S[perm],C[perm],perm
