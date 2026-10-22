"""Scoped original time-decoder dropout and matched marginal-noise controls."""
from contextlib import contextmanager
import hashlib
import numpy as np

@contextmanager
def stochastic_time(decoder,zero=False):
 import torch
 modules=[m for m in decoder.modules()if isinstance(m,(torch.nn.Dropout,torch.nn.MultiheadAttention))]
 assert modules
 saved=[(m,m.training,m.p if isinstance(m,torch.nn.Dropout)else m.dropout)for m in modules]
 try:
  for m,_,_ in saved:
   m.training=True
   if zero:
    if isinstance(m,torch.nn.Dropout):m.p=0.
    else:m.dropout=0.
  yield [dict(kind=type(m).__name__,probability=p)for m,_,p in saved]
 finally:
  for m,flag,p in saved:
   m.training=flag
   if isinstance(m,torch.nn.Dropout):m.p=p
   else:m.dropout=p

def input_seed(item,seed=20260920):
 import json
 raw=json.dumps(item,sort_keys=True,separators=(',',':')).encode()
 return (seed+int(hashlib.sha256(raw).hexdigest()[:16],16))%(2**63-1)

def gaussian_delta(delta,seed):
 """Four draws: exact empirical coordinate mean/std, independent correlations."""
 a=np.asarray(delta,dtype=np.float64);assert len(a)==4
 mu=a.mean(0);sd=a.std(0);z=np.random.default_rng(seed).standard_normal(a.shape)
 z-=z.mean(0);norm=z.std(0);assert np.all(norm>0)
 result=mu+sd*(z/norm)
 assert np.max(abs(result.mean(0)-mu))<1e-12
 assert np.max(abs(result.std(0)-sd))<1e-12
 return result.astype(np.float32)

def wrong_delta(delta):
 # [draw,layer,batch,frame,endpoint]
 return np.roll(delta,delta.shape[-2]//2,axis=-2)
