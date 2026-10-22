"""Positive native feature views preserve inside support and original padding."""
import torch
from research.native_event_spatial_text_v1.core import footprint,reflected

def keep_region(box,valid):
 b=box.clone().double();b[2:]*=1.5;keep=(footprint(b,*valid.shape,valid)>0)&valid
 fallback=not bool(keep.any())
 if fallback:keep=valid.clone()
 assert bool(keep.any())and not bool((keep&~valid).any())
 return keep,fallback

def inside_fill(x,keep,valid):
 assert keep.shape==valid.shape==(x.shape[0],x.shape[2],x.shape[3])and bool((keep.sum((1,2))>0).all())and not bool((keep&~valid).any())
 mean=(x.float()*keep[:,None]).sum((2,3))/keep.sum((1,2))[:,None]
 out=torch.where((valid&~keep)[:,None],mean[:,:,None,None],x.float()).to(x.dtype)
 assert torch.equal(out.permute(0,2,3,1)[keep|~valid],x.permute(0,2,3,1)[keep|~valid]);return out
