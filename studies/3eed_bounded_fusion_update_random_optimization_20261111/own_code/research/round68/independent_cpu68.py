"""Independent functional head and outer-only gamma fusion."""
import torch,numpy as np
def norm(a,fallback=None):
 n=np.linalg.norm(a,axis=-1,keepdims=True);x=np.divide(a,n,out=np.zeros_like(a),where=n>1e-12);return np.where(n>1e-12,x,fallback) if fallback is not None else x

def head(h,w):
 x=torch.tensor(h)[None]
 for j in ['0','2']:x=torch.nn.functional.linear(x,torch.tensor(w[j+'.weight']),torch.tensor(w[j+'.bias'])).relu()
 y=torch.nn.functional.linear(x,torch.tensor(w['4.weight']),torch.tensor(w['4.bias']))[0].numpy();y64=np.column_stack([x[0].numpy().astype(float),np.ones(256)])@np.column_stack([w['4.weight'],w['4.bias']]).astype(float).T;return y,y64

def distribution(raw,z,g):
 out=[]
 for v in 'ABC':
  q=torch.nn.functional.normalize(torch.tensor(np.asarray(raw[v],np.float32))[None],dim=-1);l=q@torch.tensor(z[v+'_tokens'])[None].transpose(-1,-2)/.07;m=torch.tensor(z[v+'_token_mask']).bool()[None,None];d=(torch.logsumexp(l.masked_fill(~m,-torch.inf),-1)-torch.logsumexp(l.masked_fill(m,-torch.inf),-1))[0].double().softmax(-1).numpy()
  if v!='A':d=g[v+'_W']@d;d=float(g[v+'_fixed_mass'])*d/max(float(d.sum()),1e-300)+(1-float(g[v+'_fixed_mass']))*g['base']
  out.append(d)
 return sum(out)/3

def readouts(z,g,w,gamma=1.):
 y={};y64={}
 for v in 'ABC':y[v],y64[v]=head(z[v+'_h'],w)
 qq={v:norm(y64[v]) for v in 'ABC'};W=g['B_W'];m=W.sum(1);qB=np.divide(W@qq['B'],m[:,None],out=np.zeros_like(qq['A']),where=m[:,None]>0);ta=norm(qq['A']+g['B_affinity'].max(1)[:,None]*qB,qq['A']);targets={'A':np.linalg.norm(y64['A'],axis=1)[:,None]*ta};rho={'A':g['B_affinity'].max(1)}
 for v in 'BC':
  W=g[v+'_W'];m=W.sum(0);u=np.divide(W.T@ta,m[:,None],out=np.zeros_like(qq[v]),where=m[:,None]>0);rho[v]=g[v+'_affinity'].max(0)*(np.linalg.norm(u,axis=1)>1e-12);targets[v]=np.linalg.norm(y64[v],axis=1)[:,None]*norm(u,qq[v])
 strength={v:np.minimum(1.,gamma*rho[v]) for v in 'ABC'};fused={v:(1-strength[v][:,None])*y[v]+strength[v][:,None]*targets[v] for v in 'ABC'};return distribution(y,z,g),distribution(fused,z,g)
