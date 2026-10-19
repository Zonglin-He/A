"""Per-frame object prior on native temporal memory; never modifies spatial output."""
import numpy as np
import torch
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap

def prior_mask(native,valid,boxes,h,w):
 l=h*w;assert native.ndim==2 and native.shape[0]==len(boxes)and valid.shape==(len(boxes),l)
 weights=[];empty=[]
 for i,(b,v)in enumerate(zip(boxes,valid)):
  vv=v.reshape(h,w);vh=int(vv.any(1).sum());vw=int(vv.any(0).sum());assert vh>0 and vw>0 and vv[:vh,:vw].all()
  c=torch.zeros(h,w,device=b.device,dtype=torch.float64);c[:vh,:vw]=cell_overlap(b[None],vh,vw)[0];c=c.flatten().to(dtype=torch.float32)
  allowed=~native[i,-l:];c=c*allowed;mass=c.sum();N=allowed.sum();is_empty=bool(mass<=0 or N==0)
  wi=torch.ones_like(c)if is_empty else .5+.5*c*(N/mass)
  wi=torch.where(allowed,wi,torch.ones_like(wi));weights.append(wi);empty.append(is_empty)
 ww=torch.stack(weights);bias=torch.zeros_like(native,dtype=torch.float32);bias[:,-l:]=ww.log();bias.masked_fill_(native,float('-inf'))
 assert torch.isfinite(ww).all()and(ww>0).all()
 return bias,dict(weights=ww.detach().cpu(),empty=empty,native_allowed_motion=(~native[:,-l:]).sum(-1).cpu(),padding_mask_disagreement=int((native[:,-l:]!=~valid).sum()))

@torch.no_grad()
def replay(model,kwargs,boxes,valid,h,w):
 native=kwargs['encoded_mask'];assert native.dtype==torch.bool
 bias,meta=prior_mask(native,valid,boxes,h,w);kw=dict(kwargs);kw['encoded_mask']=bias
 # Only additive cross-attention mask changes; query/content/time/memory untouched.
 hidden=model.ground_decoder.time_decoder(**kw);logits=model.temp_embed(hidden)[-1]
 assert torch.isfinite(logits).all()
 return logits.detach().cpu(),meta

def properties():
 mask=torch.zeros(2,8,dtype=torch.bool);valid=torch.ones(2,6,dtype=torch.bool);b=torch.tensor([[.5,.5,1.,1.],[.5,.5,1.,1.]],dtype=torch.float64)
 prior,m=prior_mask(mask,valid,b,2,3);assert torch.equal(prior,torch.zeros_like(prior));assert torch.equal(m['weights'],torch.ones(2,6))
 b[0]=torch.tensor([2.,2.,.1,.1]);p,m=prior_mask(mask,valid,b,2,3);assert m['empty'][0]and not p[0].any()
 b[0]=torch.tensor([.25,.5,.5,1.]);p,m=prior_mask(mask,valid,b,2,3);assert torch.allclose(m['weights'].sum(1),torch.tensor([6.,6.]));assert torch.equal(p[:,:2],torch.zeros(2,2));assert p[0,-6:].max()>p[0,-6:].min()
 return dict(status='pass',full_image_zero_bias=True,empty_noop=True,motion_prior_mean1=True,text_prior1=True,spatial_output_not_in_module=True)
