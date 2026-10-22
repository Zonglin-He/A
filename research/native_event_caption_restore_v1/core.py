"""Observe existing caption attention and restore equal-strength feature context."""
from contextlib import contextmanager
import torch
from research.native_event_support_view_v1.core import keep_region,inside_fill

def caption_positions(tokenizer,text):
 s=text['subject']+' '+text['caption'];shift=len(text['subject'])+1
 tok=tokenizer([s],return_offsets_mapping=True,return_tensors='pt')
 assert torch.equal(tok['input_ids'],text['full_input_ids'])
 pos=[i for i,(a,b)in enumerate(tok['offset_mapping'][0].tolist())if a<b and a>=shift and b<=len(s)]
 assert pos
 return pos,tok['input_ids'].shape[1]

@contextmanager
def observe(module,positions,hw,textlen,out):
 def hook(m,args,result):
  assert isinstance(result,tuple)and result[1]is not None,'Native weights absent; do not change kernel'
  a=result[1];assert a.ndim==3 and a.shape[1:]==(2*hw+textlen,2*hw+textlen)
  rows=a[:,[hw+i for i in positions]]
  z=(rows[:,:,:hw].float()+rows[:,:,hw+textlen:].float()).mean(1)
  assert bool(torch.isfinite(z).all())and bool((z>=0).all())
  out.append(z.detach().cpu())
 h=module.register_forward_hook(hook)
 try:yield
 finally:h.remove()

def weights(attention,keep,valid):
 assert attention.shape==keep.shape==valid.shape and bool((keep.sum((1,2))>0).all())
 a=attention.double().masked_fill(~valid,0);maximum=a.flatten(1).max(1).values
 zero=maximum<=0;w=(a/maximum.clamp_min(1e-300)[:,None,None]).clamp(0,1)
 w=torch.where(keep|zero[:,None,None],torch.ones_like(w),w);w=w.masked_fill(~valid,0)
 uniform=w.clone();spatial=w.clone();temporal=w.clone();stats=[]
 for t in range(len(w)):
  outside=valid[t]&~keep[t];vals=w[t][outside];n=len(vals)
  if n:
   uniform[t][outside]=vals.mean();spatial[t][outside]=vals.flip(0)
   # Reassign CURRENT-frame weight multiset by reversed-time spatial attention rank.
   rank=torch.argsort(a[len(w)-1-t][outside],stable=True);ordered=vals.sort(stable=True).values
   placed=torch.empty_like(vals);placed[rank]=ordered;temporal[t][outside]=placed
   for x in [uniform,spatial,temporal]:assert abs(float(x[t][outside].sum()-vals.sum()))<1e-10
   for x in [spatial,temporal]:assert torch.equal(x[t][outside].sort().values,vals.sort().values)
  stats.append(dict(outside=n,mean_restoration=float(vals.mean())if n else None,max_attention=float(maximum[t]),zero_fallback=bool(zero[t]),semantic_spatial_L1=float((spatial[t]-w[t]).abs().sum()),semantic_temporal_L1=float((temporal[t]-w[t]).abs().sum())))
 return dict(Caption=w,Uniform=uniform,SpatialPermutation=spatial,TemporalRank=temporal),stats

def restore(x,dense,w,keep,valid):
 assert x.shape==dense.shape and w.shape==keep.shape==valid.shape
 y=(w[:,None].to(device=x.device,dtype=torch.float32)*x.float()+(1-w[:,None].to(device=x.device,dtype=torch.float32))*dense.float()).to(x.dtype)
 # Preserve exact inside and original padding, not roundtrip-generated copies.
 y=torch.where((keep|~valid)[:,None],x,y)
 assert torch.equal(y.permute(0,2,3,1)[keep|~valid],x.permute(0,2,3,1)[keep|~valid])
 return y
