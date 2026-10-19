"""Temporal-mean motion input intervention; no trained component or update."""
import numpy as np
import torch
from vg_tta.stvg_temporal_semantic_contrast128_v1 import posteriors

def temporal_mean(captured,view):
 x=captured['src'];h,w=view['info']['fea_map_size'];l=h*w;t=x.shape[1]
 mask=captured['kwargs']['src_key_padding_mask'][:,-l:];assert x.ndim==3 and x.device.type=='cpu' and x.dtype==torch.float32
 assert torch.equal(mask,mask[:1].expand_as(mask));valid=~mask[0];motion=x[-l:];mean=motion.float().mean(1,keepdim=True).to(x.dtype);out=x.clone()
 # A constant sequence is a strict identity, not merely close after reduction.
 constant=torch.all(motion==motion[:,:1],dim=(1,2));mean[constant]=motion[constant,:1]
 out[-l:][valid]=mean[valid].expand(-1,t,-1)
 assert torch.equal(out[:-l],x[:-l]) and torch.equal(out[-l:][~valid],motion[~valid])
 mm=out[-l:][valid];assert torch.equal(mm,mm[:,:1].expand_as(mm))
 changed=int(torch.count_nonzero(out!=x));before=float(motion[valid].norm());after=float(mm.norm())
 return dict(captured,src=out),dict(time_steps=t,motion_cells=l,valid_cells=int(valid.sum()),changed_scalar_values=changed,motion_norm_before=before,motion_norm_after=after,nonmotion_exact=True,padded_motion_exact=True,positions_and_masks_unmodified=True,constant_sequence_exact=True)

def properties():
 x=torch.arange(5*4*3,dtype=torch.float32).reshape(5,4,3);m=torch.zeros(4,5,dtype=torch.bool);m[:,-1]=True;c=dict(src=x,kwargs=dict(src_key_padding_mask=m));v=dict(info=dict(fea_map_size=(1,2)))
 o,s=temporal_mean(c,v);assert torch.equal(o['src'][:3],x[:3]) and torch.equal(o['src'][-1],x[-1]);cc=dict(c,src=x[:,:1].expand_as(x).clone());zz,_=temporal_mean(cc,v);assert torch.equal(zz['src'],cc['src'])
 p=np.array([[0.,.1,.2],[0.,0.,.7],[0.,0.,0.]]);assert np.array_equal(posteriors(p,p)['Contrast'],p);assert np.allclose(posteriors(p,(p>0)/3)['Contrast'],posteriors(p,None)['PriorSquare'],atol=1e-15)
 return dict(status='pass',constant_motion_identity=True,only_valid_motion_changed=True,same_posterior_identity=True,uniform_counterfactual_equals_PriorSquare=True,no_new_parameters=True)
