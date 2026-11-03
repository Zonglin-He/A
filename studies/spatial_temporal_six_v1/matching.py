"""VideoWalk-style Source-feature label propagation, not its learned encoder.

Reference: ajabri/videowalk 047f3f40135a4b1be2f837793b89c3dbfe7a6683,
code/utils/test_utils.py::mem_efficient_batched_affinity and code/test.py.
Adaptations: same existing Source visual tokens, all eight known masks as fixed
memory, past20 predicted unobserved masks in each direction, average the two
soft foreground predictions. No E, Source boxes, target labels or Ours DP.
No claim of a strong baseline precedes actual source validation.
"""
import numpy as np
import torch
import torch.nn.functional as F

def affinity(query,keys,valid,temperature,topk):
 logits=keys@query.T/float(temperature);logits[~valid]=-torch.inf
 assert valid.sum()>=topk
 values,indices=logits.topk(int(topk),dim=0);weights=values.softmax(dim=0)
 return weights,indices

def propagate(features,valid,positions,seeds,native_boxes,shape,image_size,*,temperature,topk,context=20):
 raw=torch.as_tensor(features,dtype=torch.float32);cells=torch.as_tensor(valid,dtype=torch.bool);pos=list(map(int,positions));soft=torch.as_tensor(seeds,dtype=torch.float32)
 T,N,C=raw.shape;H,W=map(int,shape);assert N==H*W and cells.shape==(T,N) and len(pos)==8 and len(set(pos))==8
 assert soft.shape==(8,H,W) and torch.isfinite(raw).all() and torch.isfinite(soft).all()and ((soft>=0)&(soft<=1)).all()
 feat=F.normalize(raw,dim=-1,eps=1e-12);given={p:soft[j].flatten()*cells[p]for j,p in enumerate(pos)}
 fixed_keys=feat[pos].reshape(-1,C);fixed_valid=cells[pos].flatten();fixed_labels=torch.stack([given[p]for p in pos]).flatten();predictions=[]
 for order in [range(T),range(T-1,-1,-1)]:
  history=[];labels={};outputs=torch.zeros(T,N,dtype=torch.float32)
  for t in order:
   if t in given:outputs[t]=given[t];continue
   history=history[-int(context):]
   kk=torch.cat([fixed_keys,feat[history].reshape(-1,C)],dim=0)if history else fixed_keys
   vv=torch.cat([fixed_valid,cells[history].flatten()])if history else fixed_valid
   ll=torch.cat([fixed_labels,torch.stack([labels[h]for h in history]).flatten()])if history else fixed_labels
   weights,indices=affinity(feat[t],kk,vv,temperature,topk);fg=(ll[indices]*weights).sum(0)*cells[t];assert torch.isfinite(fg).all();outputs[t]=fg;labels[t]=fg;history.append(t)
  predictions.append(outputs)
 fg=(predictions[0]+predictions[1])/2;fg=fg.reshape(T,H,W).numpy();boxes=np.zeros((T,4),dtype=np.float64);ih,iw=map(int,image_size);empty=0
 import cv2
 for t in range(T):
  if t in given:boxes[t]=np.asarray(native_boxes)[pos.index(t)];continue
  # Official single-object readout: bilinear resize two-class scores followed
  # by argmax. For fg/bg=fg/(1-fg), the foreground test is strictly fg>.5.
  mask=cv2.resize(fg[t],(iw,ih),interpolation=cv2.INTER_LINEAR)>.5;ys,xs=np.nonzero(mask)
  if len(xs)==0:empty+=1;continue
  lo=np.array([xs.min()/iw,ys.min()/ih]);hi=np.array([(xs.max()+1)/iw,(ys.max()+1)/ih]);boxes[t]=np.r_[(lo+hi)/2,hi-lo]
 assert np.array_equal(boxes[pos],np.asarray(native_boxes,dtype=np.float64))
 return boxes,dict(empty_unobserved_masks=empty,all_eight_boxes_exact=True,Source_E_boxes_or_DP_used=False,feature_layer='same postfusion visual prefix as Current SpatialGeometry',temperature=float(temperature),topk=int(topk),context=int(context),two_direction_soft_average=True)
