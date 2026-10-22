"""Joint cached temporal context; exact original per-node inputs, frozen weights."""
from contextlib import contextmanager
import numpy as np
import torch
from scipy.special import logsumexp

AXES={'query_tgt':0,'query_content':0,'query_time':0,'query_mask':1,'encoded_feature':1,'encoded_pos':1,'encoded_mask':0}

def interleave(offsets):
 assert len(offsets)==2 and all(set(x)==set(AXES)for x in offsets)
 ns=[x['query_tgt'].shape[0]for x in offsets];assert ns[0]==ns[1]+1
 n=sum(ns);order=torch.tensor([i//2+(0 if i%2==0 else ns[0])for i in range(n)],device=offsets[0]['query_tgt'].device)
 merged={k:torch.cat([x[k]for x in offsets],dim=a).index_select(a,order)for k,a in AXES.items()}
 # Every actual native per-node value, including original time slots and padding,
 # must recover exactly. Intervened65-node tensor layout is intentionally new.
 for o,old in enumerate(offsets):
  ix=torch.arange(o,n,2,device=order.device)
  for k,a in AXES.items():assert torch.equal(merged[k].index_select(a,ix),old[k]),(o,k)
 return merged

@contextmanager
def parity_block(decoder,n):
 calls=[];hooks=[]
 def hook(m,args,kwargs):
  assert len(args)==2 and args[0].shape[0]==n and args[1].shape[0]==n
  assert 'attn_mask'not in kwargs
  parity=torch.arange(n,device=args[0].device)%2;mask=parity[:,None]!=parity[None,:]
  assert not mask.diag().any()and bool(mask[0,1])and not bool(mask[0,2])
  out=dict(kwargs);out['attn_mask']=mask;calls.append(dict(nodes=n,blocked=int(mask.sum())));return args,out
 try:
  for layer in decoder.layers:hooks.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
  yield calls
 finally:
  for h in hooks:h.remove()
 assert len(calls)==len(decoder.layers)

def single_posterior(logits):
 if hasattr(logits,'detach'):logits=logits.detach().cpu().numpy()
 z=np.asarray(logits);assert z.dtype==np.float32 and z.ndim==4 and z.shape[1]==1 and z.shape[0]==6 and np.isfinite(z).all()
 n=z.shape[2];i,j=np.triu_indices(n,1);tables=[]
 for layer in z:
  x=layer[0].astype(np.float64);v=x[i,0]+x[j,1];p=np.exp(v-logsumexp(v));out=np.zeros((n,n));out[i,j]=p;assert abs(out.sum()-1)<1e-12;tables.append(out)
 return np.mean(tables,axis=0)

def independent_marginal(logits):
 ns=[z.shape[2]for z in logits];n=sum(ns);q=np.zeros((n,n))
 for o,z in enumerate(logits):
  idx=np.arange(o,n,2);assert len(idx)==ns[o];q[np.ix_(idx,idx)]+=single_posterior(z)/2
 assert abs(q.sum()-1)<1e-12;return q
