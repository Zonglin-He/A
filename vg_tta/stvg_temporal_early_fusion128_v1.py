"""Move existing motion-region prior to early fusion; frozen native replay."""
from contextlib import contextmanager
import torch
from methods.decota_final_simplified_v1.replay import SpatialReplay
from methods.decota_final_simplified_v1.tensors import floating32
from methods.stvg_referdino_opd_v1._capture import transfer
from vg_tta.stvg_temporal_object_memory128_v1 import prior_mask,properties as old_properties

class FrozenPrefixReplay(SpatialReplay):
 def __init__(self,model,views,n):
  self.model=model;self.n=n;self.norm=model.ground_encoder.encoder.norm;self.decoder=model.ground_decoder;self.head=model.temp_embed;self.delta=torch.zeros(256,device=next(model.parameters()).device);self.views=transfer(floating32(views),'cuda')

@torch.no_grad()
def decode_prefix(model,views,n):
 replay=FrozenPrefixReplay(model,views,n);values=replay.reference_values(capture=False)
 return transfer(values,'cpu')

@contextmanager
def encoder_prior(model,bias):
 handles=[];calls=[0,[]]
 def pre(module,args,kwargs):
  assert kwargs.get('attn_mask')is None;native=kwargs['key_padding_mask'];assert native.dtype==torch.bool and native.shape==bias.shape
  assert torch.equal(native,torch.isneginf(bias))
  n,t=args[0].shape[:2];assert n==bias.shape[1]and t==bias.shape[0];heads=module.num_heads
  # Fold native padding into this encoder-only attention mask. Downstream
  # encoded_mask is never changed or used for ROI a second time.
  kwargs=dict(kwargs);kwargs['key_padding_mask']=None;actual=bias[:,None,None,:].expand(t,heads,n,n).reshape(t*heads,n,n);kwargs['attn_mask']=actual;matrix=actual.reshape(t,heads,n,n);want=bias[:,None,:].expand(t,heads,n);assert torch.equal(matrix[:,:,0,:],want)and torch.equal(matrix[:,:,-1,:],want);assert not actual.requires_grad;calls[0]+=1;calls[1].append(dict(shape=list(actual.shape),heads=heads,frame_batch=t,keys=n,first_last_query_rows_same_bias=True,padding_folded_only_in_encoder=True,original_queries_values_unchanged=True))
  return args,kwargs
 try:
  for layer in model.ground_encoder.encoder.spatial_layers:handles.append(layer.self_attn.register_forward_pre_hook(pre,with_kwargs=True))
  yield calls
 finally:
  for h in handles:h.remove()

@torch.no_grad()
def fuse(model,captured,view,boxes=None):
 x=transfer(captured['src'],'cuda');kw=transfer(captured['kwargs'],'cuda');native=kw['src_key_padding_mask'];assert native.dtype==torch.bool
 h,w=view['info']['fea_map_size'];l=h*w;valid=~native[:,-l:];meta=None;bias=None
 if boxes is not None:
  bias,meta=prior_mask(native,valid,torch.as_tensor(boxes,device='cuda',dtype=torch.float64),h,w)
  if torch.equal(meta['weights'],torch.ones_like(meta['weights'])):bias=None
 layout=[];pref=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,args:pref.append(args[0].detach().cpu()))
 try:
  with torch.autocast('cuda',dtype=torch.float16):
   if bias is None:out=model.ground_encoder.encoder(x,**kw);calls=0
   else:
    with encoder_prior(model,bias)as counter:out=model.ground_encoder.encoder(x,**kw)
    calls=counter[0];layout=counter[1];assert calls==len(model.ground_encoder.encoder.spatial_layers)
 finally:hook.remove()
 assert len(pref)==1 and torch.isfinite(pref[0]).all();assert torch.equal(native,transfer(captured['kwargs']['src_key_padding_mask'],'cuda'))
 newview=dict(view,prefix=pref[0]);newview['info']=dict(view['info']);assert newview['info']['encoded_mask'].dtype==torch.bool
 return newview,dict(prior=meta,encoder_attention_calls=calls,actual_attention_layout_checks=layout,downstream_bool_mask_unchanged=True,appearance_text_key_prior1=True)

def properties():
 out=old_properties();out.pop('spatial_output_not_in_module',None);out.update(encoder_prior_only=True,no_new_trainable_state=True,query_gates_recomputed_native=True)
 return out
