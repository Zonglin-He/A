"""Deterministic observation rule and frozen native ReferDINO wrapper."""
from pathlib import Path
import os,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RT=ROOT/'.cache/referdino_runtime_v1';EXT=ROOT/'external/ReferDINO'

def positions8(ids,original):
 ids=np.asarray(ids,float);s=set(map(int,original));assert len(s)==4 and len(ids)>=8
 while len(s)<8:s.add(max((i for i in range(len(ids))if i not in s),key=lambda i:(min(abs(ids[i]-ids[j])for j in s),-ids[i])))
 return sorted(s)

def interpolate(ids,selected,boxes):
 b=np.asarray(boxes,float);xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2));x=np.column_stack([np.interp(ids,np.asarray(ids)[selected],xy[:,j])for j in range(4)])
 out=np.column_stack(((x[:,:2]+x[:,2:])/2,x[:,2:]-x[:,:2]));assert (out[:,2:]>0).all();return out

class NativeRefer:
 def __init__(self):
  import torch
  sys.path[:0]=[str(RT/'python'),str(RT/'cuda_build'),str(EXT)]
  from ruamel.yaml import YAML
  from easydict import EasyDict
  from models import build_model
  with open(EXT/'configs/mevis_swinb.yaml')as f:c=YAML(typ='safe',pure=True).load(f)
  args=EasyDict({k:v['value']for k,v in c.items()});args.device='cuda';args.GroundingDINO.pretrained_path=str(ROOT/'checkpoints/groundingdino-b/groundingdino_swinb_cogcoor.pth');args.GroundingDINO.text_encoder_type=str(RT/'bert-from-frozen-teacher');args.GroundingDINO.tracking_alpha=.1
  self.model,criterion,post=build_model(args);del criterion,post
  ck=torch.load(ROOT/'checkpoints/referdino/ryt_mevis_swinb.pth',map_location='cpu',weights_only=False);sd=ck['model_state_dict'];result=self.model.load_state_dict(sd,strict=False);learned=set(dict(self.model.named_parameters()));assert not (set(result.missing_keys)&learned),result;assert not [k for k in result.unexpected_keys if not k.endswith('position_ids')],result
  self.load_audit=dict(missing=result.missing_keys,unexpected=result.unexpected_keys,all_learned_parameters_covered=True)
  self.model=self.model.cuda().eval().requires_grad_(False);self.params=sum(p.numel()for p in self.model.parameters())
  import torchvision.transforms as T
  self.transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 @__import__('torch').no_grad()
 def __call__(self,frames,caption):
  import torch
  from PIL import Image
  from misc import nested_tensor_from_videos_list
  torch.cuda.synchronize();start=time.perf_counter();imgs=torch.stack([self.transform(Image.fromarray(f))for f in frames]).cuda();samples=nested_tensor_from_videos_list(imgs[None],size_divisibility=1);h,w=imgs.shape[-2:];target={'size':torch.as_tensor([h,w]).cuda()};exp=' '.join(caption.lower().split());torch.cuda.synchronize();p=time.perf_counter();out=self.model.infer(samples,[exp],[target]);torch.cuda.synchronize();f=time.perf_counter()
  logits=out['pred_logits'][0];scores=logits.sigmoid().mean(0).max(-1).values;idx=int(scores.argmax());b=out['pred_boxes'][0,:,idx].float().cpu();m=out['pred_masks'][0,:,idx];mstats=dict(shape=list(m.shape),finite=bool(torch.isfinite(m).all()),foreground_fraction_at_native_grid=float((m>0).float().mean()))
  assert torch.isfinite(b).all()and(b[:,2:]>0).all();torch.cuda.synchronize();end=time.perf_counter()
  return dict(boxes=b,selected_query=idx,native_query_scores=scores.float().cpu(),selected_frame_scores=logits[:,idx].sigmoid().float().cpu(),mask_summary=mstats,text_sha256=hashlib.sha256(exp.encode()).hexdigest(),input_shape=list(imgs.shape),cost=dict(expert_seconds=end-start,preprocess_seconds=p-start,forward_seconds=f-p,postprocess_seconds=end-f),native_same_track_all8=True)
