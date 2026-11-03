"""Official expert checkpoint construction and explicit output interfaces."""
from pathlib import Path
import sys,time,json,importlib.util
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_subset_expansion_v1';OLD=R/'outputs/expert_generalization_v1'
def construct(expert):
 import torch
 from transformers import RobertaModel,RobertaConfig,RobertaTokenizerFast
 assets=OLD/'vendor/roberta-base';config=RobertaConfig.from_pretrained(str(assets),local_files_only=True);tok=RobertaTokenizerFast.from_pretrained(str(assets),local_files_only=True)
 class Text:
  @staticmethod
  def from_pretrained(name,*a,**kw):assert name=='roberta-base';return RobertaModel(config)
 class Token:
  @staticmethod
  def from_pretrained(name,*a,**kw):assert name=='roberta-base';return tok
 sys.path[:0]=[str(OLD/'runtime/python'),str(R/'runtime/referdino/cuda_build')]
 if expert=='referformer_r101':
  sys.path.insert(0,str(OLD/'vendor/ReferFormer'));import opts,models
  import models.backbone as backbone,models.referformer as ref
  args=opts.get_args_parser().parse_args([]);args.device='cpu';args.masks=True;args.binary=True;args.with_box_refine=True;args.freeze_text_encoder=True;args.backbone='resnet101'
  previous=(backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast);backbone.is_main_process=lambda:False;ref.RobertaModel=Text;ref.RobertaTokenizerFast=Token
  try:model,criterion,post=models.build_model(args)
  finally:backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast=previous
  checkpoint=D/'vendor/referformer_r101_refytvos.pth';key='model'
 else:
  assert expert=='mttr';import yaml,argparse
  V=D/'vendor/MTTR';sys.path.insert(0,str(V));import models
  import models.multimodal_transformer as transform
  cfg={k:v['value']for k,v in yaml.safe_load((V/'configs/refer_youtube_vos.yaml').read_text()).items()};cfg.update(device='cpu',running_mode='eval')
  previous=(transform.RobertaModel,transform.RobertaTokenizerFast);transform.RobertaModel=Text;transform.RobertaTokenizerFast=Token
  try:model,criterion,post=models.build_model(argparse.Namespace(**cfg))
  finally:transform.RobertaModel,transform.RobertaTokenizerFast=previous
  checkpoint=D/'vendor/mttr_refytvos_window12.pth.tar';key='model_state_dict'
 del criterion,post
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);state=state[key]if key in state else state
 result=model.load_state_dict(state,strict=False);allowed=lambda k:k.endswith(('position_ids','token_type_ids'))
 assert all(allowed(k)for k in result.missing_keys+result.unexpected_keys),result
 assert not(set(result.missing_keys)&set(dict(model.named_parameters())))
 for k,v in model.state_dict().items():
  if k in state:assert torch.equal(v,state[k]),k
  else:assert allowed(k)
 for k in result.unexpected_keys:
  v=state[k];assert torch.equal(v,torch.arange(v.shape[-1],dtype=v.dtype).expand_as(v))if k.endswith('position_ids')else torch.equal(v,torch.zeros_like(v))
 model.eval().requires_grad_(False)
 return model,tok,dict(status='all_learned_and_shared_tensors_exact_checkpoint',missing_deterministic_buffers=result.missing_keys,unexpected_verified_deterministic_buffers=result.unexpected_keys,parameters=sum(x.numel()for x in model.parameters()),official_vendor_unchanged=True)
def preprocess(expert,rgb):
 import torch
 from PIL import Image
 from torchvision import transforms as T
 if expert=='referformer_r101':transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 else:
  from torchvision.transforms import functional as TF
  # Exact official MTTR eval transform: DETR short360, long<=640 resizing.
  def resize(im):
   width,height=im.size;size=360;short=min(width,height);long=max(width,height)
   if long/short*size>640:size=int(round(640*short/long))
   return TF.resize(im,size)
  transform=T.Compose([resize,T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 return torch.stack([transform(Image.fromarray(f))for f in rgb])
def forward(model,expert,cpu,expr):
 import torch
 x=cpu.cuda()
 if expert=='referformer_r101':
  from util.misc import nested_tensor_from_videos_list
  samples=nested_tensor_from_videos_list([x]);assert not samples.mask.any();return model(samples,[expr],[dict(size=torch.tensor(cpu.shape[-2:],device='cuda'))])
 from misc import NestedTensor
 samples=NestedTensor(x[:,None],torch.zeros((8,1,*x.shape[-2:]),dtype=torch.bool,device='cuda'))
 return model(samples,torch.arange(8,device='cuda'),[expr])
def select(expert,outputs,input_shape,original_shape):
 import torch,numpy as np
 if expert=='referformer_r101':
  spec=importlib.util.spec_from_file_location('_original_refer_output_interface',R/'studies/expert_generalization_v1/interface.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);out=m.selected_eight(outputs,input_shape);out['expert_model']='ReferFormer-R101-RefYTVOS';out['empty_box_conversion']=0;return out
 logits=outputs['pred_is_referred'].detach().cpu();masks=outputs['pred_masks'].detach().cpu();assert logits.shape[:2]==masks.shape[:2]==(8,1)
 assert logits.shape[-1]==2 and torch.isfinite(logits).all()and torch.isfinite(masks).all()
 scores=logits.softmax(-1)[...,0];track=scores.mean(0)[0];j=int(track.argmax());native=masks[:,0,j].float().contiguous()
 import torch.nn.functional as F
 binary=F.interpolate(native[:,None],size=input_shape[-2:],mode='bilinear',align_corners=False).sigmoid()>.5
 binary=F.interpolate(binary.float(),size=original_shape,mode='nearest')[:,0].bool();h,w=original_shape;boxes=[];empty=[]
 for i,b in enumerate(binary):
  y,x=torch.where(b)
  if len(x):l,r=int(x.min()),int(x.max())+1;t,bottom=int(y.min()),int(y.max())+1
  else:l,r,t,bottom=0,w,0,h;empty.append(i)
  boxes.append([(l+r)/(2*w),(t+bottom)/(2*h),(r-l)/w,(bottom-t)/h])
 return dict(boxes=torch.tensor(boxes,dtype=torch.float32),native_mask_logits=native,selected_query=j,native_query_scores=track,selected_frame_scores=scores[:,0,j,None],input_shape=list(input_shape),nested_mask_all_false=True,native_same_track_all8=True,expert_model='MTTR-RefYTVOS-window12-eight-observation-adapter',expert_combination='one referring-video model, no detector/segmenter',box_interface='official selected-track .5 binary mask, original-size nearest, halfopen bbox; empty=explicit uninformative full-frame box, no Source/GT fallback',empty_box_conversion=len(empty),empty_indices=empty,GT_read=False)
if __name__=='__main__':
 import argparse,hashlib,os
 a=argparse.ArgumentParser();a.add_argument('expert',choices=['referformer_r101','mttr']);q=a.parse_args();import torch;torch.set_num_threads(1);start=time.perf_counter();model,tok,receipt=construct(q.expert);assert not torch.cuda.is_initialized();receipt.update(actual_model_CPU_construction=1,actual_GPU_or_forward=0,elapsed_seconds=time.perf_counter()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(D/(q.expert+'_CPU_model_doctor.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
