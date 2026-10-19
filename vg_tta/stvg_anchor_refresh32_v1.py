"""Four cached anchor references refreshed before each frozen spatial layer.

Counterfactual E-anchor control removes re-decoding/reseeding's common effect.
No network update, teacher forcing labels or new visual/expert observations.
"""
import numpy as np
import torch
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
ARMS=['Source_MBR','E_MBR','Anchor_local','Initial_delta','Persistent_delta','Persistent_shuffle','Persistent_allnodes']

@torch.no_grad()
def anchored_forward(decoder,kwargs,local_positions,anchor_boxes,persistent):
 query_tgt=kwargs['query_tgt'];pred_boxes=kwargs['pred_boxes'];query_time=kwargs['query_time'];query_mask=kwargs['query_mask'];feature=kwargs['encoded_feature'];pos=kwargs['encoded_pos'];mask=kwargs['encoded_mask'];outputs=[];trace=[]
 sine=decoder.forward.__func__.__globals__['gen_sineembed_for_position']
 for i,layer in enumerate(decoder.layers):
  before=pred_boxes
  if local_positions and(persistent or i==0):
   pred_boxes=pred_boxes.clone();pred_boxes[local_positions,0]=anchor_boxes.to(pred_boxes)
  used=pred_boxes;emb=sine(pred_boxes);qpos=decoder.ref_point_head(emb);scale=1 if i==0 else decoder.query_scale(query_tgt);emb=emb[...,:decoder.d_model]*scale
  query_tgt,_=layer(query_tgt=query_tgt,query_pos=qpos,query_time_embed=query_time,query_sine_embed=emb,query_mask=query_mask,encoded_feature=feature,encoded_pos=pos,encoded_mask=mask,good_rf=None,is_first=i==0)
  assert decoder.bbox_embed is not None
  pred_boxes=decoder.bbox_embed(query_tgt).sigmoid();outputs.append(pred_boxes);trace.append(dict(layer=i,before_reference=before.cpu(),used_reference=used.cpu(),raw_output=pred_boxes.cpu()));pred_boxes=pred_boxes.detach()
 # Native's final norms affect an unused intermediate list, not these outputs.
 return torch.stack(outputs).transpose(1,2),trace

@torch.no_grad()
def run(decoder,inputs,n,positions,boxes,persistent):
 allout=[];trace=[]
 for view,x in enumerate(inputs):
  indices=[j for j,p in enumerate(positions)if p%2==view];local=[positions[j]//2 for j in indices];b=torch.as_tensor(np.asarray(boxes)[indices],dtype=x['pred_boxes'].dtype,device=x['pred_boxes'].device).reshape(-1,4);out,tr=anchored_forward(decoder,x,local,b,persistent);allout.append(out.flatten(1,2)[-1]);trace.append(dict(view=view,local_positions=local,layers=tr))
 return torch.stack([allout[i%2][i//2]for i in range(n)]).float().cpu(),trace

def paired_readout(new,control,E,planned,only_unobserved=True):
 # E + (DINO-conditioned - E-conditioned) in center/log-size space.
 out=corrected(new,control,E)
 if only_unobserved:out[list(planned)]=torch.as_tensor(E,dtype=torch.float64)[list(planned)]
 assert torch.isfinite(out).all()and(out[:,2:]>0).all()
 return out

def properties():
 E=torch.tensor([[.5,.5,.2,.2],[.4,.3,.1,.3]],dtype=torch.float64);raw=torch.tensor([[.4,.2,.3,.1],[.3,.5,.2,.2]],dtype=torch.float32)
 assert torch.equal(paired_readout(raw,raw,E,[0]),E)
 changed=raw.clone();changed[1,0]+=.02;out=paired_readout(changed,raw,E,[0]);assert torch.equal(out[0],E[0])and out[1,0]>E[1,0]
 z=paired_readout(changed,raw,E,[0],False);assert torch.equal(z[0],E[0])
 return dict(status='pass',same_conditional_inputs_exact_E=True,planned_nodes_preserved=True,paired_center_logsize_readout=True,no_ref_support_noop_by_protocol=True)
