"""Uni-Adapter candidate/text-pair port with invariant binary role semantics.

Source-prepared native 3EED pair embedding substitutes the original classification
VLFM. Caches contain referent/nonreferent pairs, never permanent query numbers.
Exact graph solve is used (the author's old-cache path), not recursive inverse.
"""
import torch
from torch.nn import functional as F
class UniCache:
 def __init__(self,capacity=8):self.cache={0:[],1:[]};self.capacity=capacity;self.writes=0
 def forward(self,features,prob,classifier=None,query_features=None):
  features=F.normalize(features,dim=1);entropy=-(prob*prob.clamp_min(1e-30).log()).sum(1);conf=torch.exp(-150*entropy.double()).clamp_min(1e-100)
  for x,p,c in zip(features,prob,conf):
   k=int(p.argmax());bank=self.cache[k]
   if len(bank)<self.capacity:bank.append([x.detach().clone(),c,p.detach().clone(),1])
   else:
    j=int(torch.stack([z[0]@x for z in bank]).argmax());old,weight,op,count=bank[j];den=weight*count+c;new=F.normalize(((old.double()*weight*count+x.double()*c)/den).float(),dim=0);newprob=classifier(new[None]).softmax(1)[0] if classifier is not None else (op*count+p.detach())/(count+1);bank[j]=[new,den/(count+1),newprob,count+1]
   self.writes+=1
  items=self.cache[0]+self.cache[1];K=torch.stack([v[0] for v in items]);P=torch.stack([v[2] for v in items]);W=K@K.T;W=torch.where(W>=.5,W,0);dinv=(W.sum(1)+1e-8).rsqrt();L=torch.eye(len(K),device=K.device)-dinv[:,None]*W*dinv[None];Y=torch.linalg.solve(L+.26*torch.eye(len(K),device=K.device),.26*P);Y=Y/Y.sum(1,keepdim=True).clamp_min(1e-20);labels=Y.argmax(1);values=F.one_hot(labels,2).float();values/=values.sum(0,keepdim=True)+1e-6;query=features if query_features is None else F.normalize(query_features,dim=1);logit=query@K.T@values;pc=logit.softmax(1);hc=-(pc*pc.clamp_min(1e-30).log()).sum(1);hs=entropy;out=(hc[:,None]*prob+hs[:,None]*pc)/(hs+hc)[:,None].clamp_min(1e-20)
  return out,{'cache_items':len(items),'cache_writes':self.writes,'classes_present':[k for k,v in self.cache.items() if v]}
