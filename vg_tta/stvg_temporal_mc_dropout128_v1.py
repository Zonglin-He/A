"""Frozen conditional stochastic time decoder; paired native head/output controls."""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

@contextmanager
def random_decoder(decoder,zero=False):
 selected=[x for x in decoder.modules()if isinstance(x,(torch.nn.Dropout,torch.nn.MultiheadAttention))]
 flags=[(x,x.training,x.p if isinstance(x,torch.nn.Dropout)else x.dropout)for x in selected];counts=dict(MHA=0,Dropout=0);hooks=[]
 def call(m,args,out):counts['MHA'if isinstance(m,torch.nn.MultiheadAttention)else'Dropout']+=1
 try:
  assert len(selected)==36 and all(not flag for m,flag,p in flags)
  for m,flag,p in flags:
   assert abs(p-.1)<1e-8;m.training=True
   if zero:
    if isinstance(m,torch.nn.Dropout):m.p=0.
    else:m.dropout=0.
   hooks.append(m.register_forward_hook(call))
  yield counts
 finally:
  for h in hooks:h.remove()
  for m,flag,p in flags:
   m.training=flag
   if isinstance(m,torch.nn.Dropout):m.p=p
   else:m.dropout=p
  assert all(m.training==flag and (m.p if isinstance(m,torch.nn.Dropout)else m.dropout)==p for m,flag,p in flags)

@torch.no_grad()
def stochastic_hidden(decoder,kw,seed,zero=False):
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(seed)
  with random_decoder(decoder,zero)as counts:h=decoder(**kw)
 return h,counts

@torch.no_grad()
def paired_head(head,h,seed,zero=False):
 leaf=head.dropout;assert isinstance(leaf,torch.nn.Dropout)and not leaf.training and leaf.p==.3 and head.num_layers==2
 old=(leaf.training,leaf.p);calls=[];last=[];handles=[]
 def pre(m,args):
  assert len(calls)<2;torch.manual_seed(seed+len(calls));calls.append(len(calls));return None
 def post(m,args,out):
  c=len(calls)-1
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(seed+c);mask=F.dropout(torch.ones_like(args[0]),p=m.p,training=True)
  scaled=(out-args[0]*mask).abs()/args[0].abs().clamp_min(1.);assert scaled.max().item()<=16*torch.finfo(torch.float32).eps
  digest=hashlib.sha256((mask!=0).cpu().numpy().tobytes()).hexdigest()
  last.append(dict(mask_sha256=digest,p=m.p,scaled_error=scaled.max().item()))
  if c==1:last[-1].update(mask=mask[-1].cpu(),pre_output=args[0][-1].cpu())
 try:
  leaf.training=True
  if zero:leaf.p=0.
  handles=[leaf.register_forward_pre_hook(pre),leaf.register_forward_hook(post)]
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):out=head(h)
  assert calls==[0,1]and len(last)==2
  return out,last
 finally:
  for hdl in handles:hdl.remove()
  leaf.training,leaf.p=old

@torch.no_grad()
def one_draw(model,kw,native_h,native_logits,seed):
 h,c=stochastic_hidden(model.ground_decoder.time_decoder,kw,seed+3)
 full,f=paired_head(model.temp_embed,h,seed+100)
 head,hh=paired_head(model.temp_embed,native_h,seed+100)
 assert all(a['mask_sha256']==b['mask_sha256']for a,b in zip(f,hh))
 output=native_logits[-1].cpu()*f[1]['mask']
 assert torch.isfinite(full).all()and torch.isfinite(head).all()and torch.isfinite(output).all()
 return dict(FullMC=full[-1].cpu(),HeadOnlyMC=head[-1].cpu(),OutputOnlyMC=output,head_masks=[v['mask_sha256']for v in f],head_control_masks=[v['mask_sha256']for v in hh],output_mask=f[1]['mask'],full_pre_output=f[1]['pre_output'],head_pre_output=hh[1]['pre_output'],max_head_mask_scaled_error=max(v['scaled_error']for v in f+hh),calls=c)

def integrate(records,draws,q0):
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 out={};removed={};n=len(draws)
 assert n==48
 for arm in ['FullMC','HeadOnlyMC','OutputOnlyMC']:
  ps=[];drop=[]
  for d in draws:
   ids,q=mass_table(probabilities(records,[v[arm]for v in d['offsets']]));mask=q0>0;drop.append(float(q[~mask].sum()));q[~mask]=0;q/=q.sum();assert np.array_equal(q>0,mask);ps.append(q)
  out[arm]=np.mean(ps,axis=0);removed[arm]=dict(mean=float(np.mean(drop)),max=max(drop))
  assert abs(out[arm].sum()-1)<1e-12
 return out,removed
