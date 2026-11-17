"""Candidate-group physical image availability, no class/GT confidence rules."""
from common150 import *
import numpy as np

def supports(xyz,boxes,assignment,va,vb):
 unique,inv=np.unique(xyz,axis=0,return_inverse=True);A=np.zeros(len(unique),bool);BB=A.copy();np.logical_or.at(A,inv,va);np.logical_or.at(BB,inv,vb);inside=(abs(unique[None]-boxes[:,None,:3])<=np.maximum(boxes[:,None,3:6],1e-6)/2).all(-1);gs=[np.flatnonzero(assignment==g) for g in range(int(assignment.max()+1))];sp=np.array([inside[g].any(0) for g in gs]);return gs,(sp&A).sum(1),(sp&BB).sum(1),sp.sum(1)
def route_prob(pA,pB,groups,scope,alpha):
 # Unobserved groups retain frozen A-image prediction, never a zero penalty.
 out=pA.copy()
 for g in np.flatnonzero(scope):ix=groups[g];out[ix]=alpha[g]*pA[ix]+(1-alpha[g])*pB[ix]
 return out/out.sum()
def permutation(groups,scope,salt):
 rng=np.random.default_rng(int(hashlib.sha256((salt+'|R150_visibility_route').encode()).hexdigest()[:16],16));out=np.arange(len(groups))
 for size in sorted({len(groups[g]) for g in np.flatnonzero(scope)}):
  ids=np.array([g for g in np.flatnonzero(scope) if len(groups[g])==size],int);order=rng.permutation(ids)
  if len(order)>1:out[order]=np.roll(order,1)
 return out
def evidence150(pA,pB,groups,S,nA,nB,salt):
 a=nA>=4;b=nB>=4;scope=S&(a|b);den=a.astype(float)+b;alpha=np.divide(a,den,out=np.ones(len(S)),where=den>0);aa=np.where(a,nA,0);bb=np.where(b,nB,0);count=np.divide(aa,aa+bb,out=np.ones(len(S),float),where=aa+bb>0);perm=permutation(groups,scope,salt);wrong=alpha[perm];wrongcount=count[perm];ga=float(np.mean(alpha[scope])) if scope.any() else 1.;gc=float(np.sum(aa[scope])/max(np.sum(aa[scope]+bb[scope]),1));return dict(S_rgb=scope,nA=nA,nB=nB,alpha_equal=alpha,alpha_count=count,permutation=perm,alpha_wrong_equal=wrong,alpha_wrong_count=wrongcount,alpha_global_equal=np.full(len(S),ga),alpha_global_count=np.full(len(S),gc))
def force_C(p,groups,C):
 out=np.zeros_like(p);ids=np.concatenate([groups[g] for g in np.flatnonzero(C)]);out[ids]=p[ids];return out/out.sum()
