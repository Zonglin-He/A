"""Fixed two-view unit-representation fusion; no fit or C-view input."""
from common74 import *
from scipy.special import softmax,logsumexp
def unit(x,fallback=None):
 n=np.linalg.norm(x,axis=1,keepdims=True);out=np.divide(x,n,out=np.zeros_like(x),where=n>1e-12);return np.where(n>1e-12,out,fallback) if fallback is not None else out
def score(q,z):
 l=q@z['tokens'].astype(float).T/.07;m=z['token_mask'];return softmax(logsumexp(l[:,m],1)-logsumexp(l[:,~m],1))
def fused_AB(z,e):
 W=e['B_W'];qa=unit(z['A']['query'].astype(float));qb=unit(z['B']['query'].astype(float));col=W.sum(0);af=W/np.maximum(1-col[None],1e-12);rho=np.clip(af.max(1),0,1);row=W.sum(1);qbmean=np.divide(W@qb,row[:,None],out=np.zeros_like(qa),where=row[:,None]>0);target=unit(qa+rho[:,None]*qbmean,qa);fa=unit((1-rho[:,None])*qa+rho[:,None]*target,qa);u=np.divide(W.T@target,col[:,None],out=np.zeros_like(qb),where=col[:,None]>0);good=np.linalg.norm(u,axis=1)>1e-12;conf=np.clip(af.max(0),0,1)*good;fb=unit((1-conf[:,None])*qb+conf[:,None]*unit(u,qb),qb);pb=score(fb,z['B']);mass=float(e['B_fixed_mass']);b=W@pb;b=mass*b/max(b.sum(),1e-300)+(1-mass)*e['base'];return (score(fa,z['A'])+b)/2
