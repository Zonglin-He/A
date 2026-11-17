"""Frozen teacher-only numerical calibration. Student full readout is untouched."""
from common149 import *
import numpy as np,torch
from torch.nn import functional as F

def base_logits(a,W):
 x=torch.from_numpy(a['h'].copy())[None]
 for j in [0,2]:x=F.relu(F.linear(x,W['match.'+str(j)+'.weight'],W['match.'+str(j)+'.bias']))
 q=F.normalize(F.linear(x,W['match.4.weight'],W['match.4.bias'])[0]+torch.from_numpy(a['late'].copy()),dim=-1);v=q@torch.from_numpy(a['u'].copy()).T/.07;m=torch.from_numpy(a['mask'].copy()).bool();return (v[:,m].logsumexp(-1)-v[:,~m].logsumexp(-1)).double().numpy()
def calibrated(p,b,alpha):
 if alpha==1:return p.copy()
 z=alpha*np.log(np.maximum(p,1e-300))+(1-alpha)*b;z-=z.max();v=np.exp(z);return v/v.sum()
def groups(a):return [np.flatnonzero(a==i) for i in range(int(a.max()+1))]
def original_C(pA,pB,gs,S,k,has):
 C=np.zeros_like(S)
 if has and S.sum()>=2:
  for p in [pA,pB]:
   v=np.log(np.maximum([p[g].mean() for g in gs],1e-300))[S];C[S]|=v.max()-v<=k
 return C
