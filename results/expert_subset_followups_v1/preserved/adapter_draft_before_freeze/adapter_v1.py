"""Pinned official models; explicit sparse-eight STVG interfaces, no training."""
from pathlib import Path
import sys,time,json,hashlib
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_subset_followups_v1';OLD=R/'outputs/expert_generalization_v1'
EXPERTS=['onlinerefer_r50','sgmg_videoswin_t']
CHECKPOINTS={'onlinerefer_r50':'onlinerefer_r50_refytvos.pth','sgmg_videoswin_t':'sgmg_videoswin_t_refytvos.pth'}
def construct(expert):
 import torch
 from transformers import RobertaModel,RobertaConfig,RobertaTokenizerFast
 assert expert in EXPERTS
 assets=OLD/'vendor/roberta-base';config=RobertaConfig.from_pretrained(str(assets),local_files_only=True);tok=RobertaTokenizerFast.from_pretrained(str(assets),local_files_only=True)
 class Text:
  @staticmethod
  def from_pretrained(name,*a,**kw):
   assert name in ['roberta-base','/data/wudongming/referformer/roberta_base/'];return RobertaModel(config)
 class Token:
  @staticmethod
  def from_pretrained(name,*a,**kw):
   assert name in ['roberta-base','/data/wudongming/referformer/roberta_base/'];return tok
 sys.path[:0]=[str(OLD/'runtime/python'),str(R/'runtime/referdino/cuda_build')]
 V=D/'vendor'/('OnlineRefer'if expert=='onlinerefer_r50'else'SgMg');sys.path.insert(0,str(V))
 import opts,models
 args=opts.get_args_parser().parse_args([]);args.device='cpu';args.masks=True;args.binary=True;args.with_box_refine=True;args.freeze_text_encoder=True;args.dataset_file='ytvos'
 if expert=='onlinerefer_r50':
  import models.backbone as backbone,models.onlinerefer as ref
  args.online=True;args.semi_online=False;args.backbone='resnet50'
  previous=(backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast);backbone.is_main_process=lambda:False;ref.RobertaModel=Text;ref.RobertaTokenizerFast=Token
  try:model,criterion,post=models.build_model(args)
  finally:backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast=previous
 else:
  import models.text_encoder.text_encoder as text,models.text_encoder.tokenizer as token
  args.eval=True;args.amp=False;args.backbone='video_swin_t_p4w7';args.backbone_pretrained=None;args.use_checkpoint=False
  previous=(text.RobertaModel,token.RobertaTokenizerFast);text.RobertaModel=Text;token.RobertaTokenizerFast=Token
  try:model,criterion,post=models.build_model(args)
  finally:text.RobertaModel,token.RobertaTokenizerFast=previous
 del criterion,post
 checkpoint=D/'vendor'/CHECKPOINTS[expert];state=torch.load(checkpoint,map_location='cpu',weights_only=False);state=state['model']
 result=model.load_state_dict(state,strict=False);allowed=lambda k:k.endswith(('position_ids','token_type_ids'))
 assert all(allowed(k)for k in result.missing_keys+result.unexpected_keys),result
 assert not(set(result.missing_keys)&set(dict(model.named_parameters())))
 for k,v in model.state_dict().items():
  if k in state:assert torch.equal(v,state[k]),k
  else:assert allowed(k),k
 for k in result.unexpected_keys:
  v=state[k];assert torch.equal(v,torch.arange(v.shape[-1],dtype=v.dtype).expand_as(v))if k.endswith('position_ids')else torch.equal(v,torch.zeros_like(v))
 model.eval().requires_grad_(False)
 return model,tok,dict(status='all_learned_and_shared_tensors_exact_official_checkpoint',missing_deterministic_buffers=result.missing_keys,unexpected_verified_deterministic_buffers=result.unexpected_keys,parameters=sum(x.numel()for x in model.parameters()),official_vendor_unchanged=True,args=vars(args))
def preprocess(expert,rgb):
 import torch
 from PIL import Image
 from torchvision import transforms as T
 assert expert in EXPERTS and len(rgb)==8
 transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 return torch.stack([transform(Image.fromarray(f))for f in rgb])
def cpu_tree(x):
 import torch
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return{k:cpu_tree(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu_tree(v)for v in x)
 return x
def forward(model,expert,cpu,expr,checkpoint_step=None):
 import torch
 assert cpu.shape[0]==8
 if expert=='sgmg_videoswin_t':
  x=cpu.cuda();return model([x],[expr],[dict(size=torch.tensor(cpu.shape[-2:],device='cuda'))])
 assert expert=='onlinerefer_r50'and checkpoint_step is not None
 track=model.generate_empty_tracks();initial=cpu_tree(track);steps=[]
 for i in range(8):
  x=cpu[i:i+1].cuda();before=cpu_tree(track);out=model.inference([x],track,[expr],[dict(size=torch.tensor(cpu.shape[-2:],device='cuda'))]);torch.cuda.synchronize()
  # Preserve every paid frame BEFORE native tracking readout/state update.
  cold=cpu_tree(out);checkpoint_step(i,dict(outputs=cold,track_before=before,frame_index=i,native_state_update_not_yet_applied=True))
  track=model.post_process_single_image(out,track,is_last=False)
  steps.append(dict(outputs=cold,track_before=before,track_after=cpu_tree(track),frame_index=i));del out,x
 return dict(steps=steps,initial_track=initial,final_track=cpu_tree(track),model_steps=8,only_supplied_eight_frames=True,episodic_reset=True)
def binary_mask_boxes(binary):
 import torch
 assert binary.ndim==3 and binary.shape[0]==8;h,w=binary.shape[-2:];boxes=[];empty=[]
 for i,b in enumerate(binary):
  y,x=torch.where(b)
  if len(x):l,r=int(x.min()),int(x.max())+1;t,bottom=int(y.min()),int(y.max())+1
  else:l,r,t,bottom=0,w,0,h;empty.append(i)
  boxes.append([(l+r)/(2*w),(t+bottom)/(2*h),(r-l)/w,(bottom-t)/h])
 return torch.tensor(boxes,dtype=torch.float32),empty
def select(expert,outputs,input_shape,original_shape):
 import torch
 import torch.nn.functional as F
 if expert=='onlinerefer_r50':
  assert len(outputs['steps'])==outputs['model_steps']==8 and outputs['episodic_reset'];boxes=[];masks=[];indices=[];all_scores=[];chosen_scores=[]
  for i,s in enumerate(outputs['steps']):
   assert s['frame_index']==i
   o=s['outputs'];logits=o['pred_logits'][0];bb=o['pred_boxes'][0];mm=o['pred_masks'][0];assert logits.shape[0]==bb.shape[0]==mm.shape[0]==1
   assert all(torch.isfinite(v).all()for v in [logits,bb,mm]);scores=logits.sigmoid().mean(0).max(-1).values;j=int(scores.argmax());indices.append(j);all_scores.append(scores);chosen_scores.append(logits[0,j].sigmoid().max())
   boxes.append(bb[0,j]);masks.append(mm[0,j])
   assert torch.equal(o['pos_embedding'][j:j+1],s['track_after']['pos_embedding'])
   if i:assert all(torch.equal(s['track_before'][k],outputs['steps'][i-1]['track_after'][k])for k in s['track_before'])
  native=torch.stack(masks).float().contiguous();box=torch.stack(boxes).float().contiguous();assert ((box[:,2:]>0)&(box[:,2:]<=1)).all()
  return dict(boxes=box,native_mask_logits=native,selected_queries=indices,native_query_scores=torch.stack(all_scores),selected_frame_scores=torch.stack(chosen_scores)[:,None],native_same_track_all8=False,native_causal_state_carried_all8=True,episodic_reset=True,model_steps=8,box_interface='official causal per-step selected direct normalized boxhead; track updated after every supplied frame; no extra frames or global reselection',empty_box_conversion=0,native_empty_masks_not_used_for_boxhead=True,input_shape=list(input_shape),nested_mask_all_false=True,expert_model='OnlineRefer-R50-RefYTVOS-sparse-eight-causal-adapter',GT_read=False)
 assert expert=='sgmg_videoswin_t'
 logits=outputs['pred_logits'][0];masks=outputs['pred_masks'][0];assert logits.shape[:2]==masks.shape[:2]and logits.shape[0]==8
 assert torch.isfinite(logits).all()and torch.isfinite(masks).all();scores=logits.sigmoid().mean(0).max(-1).values;j=int(scores.argmax());native=masks[:,j].float().contiguous()
 # Exact official inference: crop padded output then original-size bilinear and >.5.
 cropped=native[None,:,:input_shape[-2],:input_shape[-1]];binary=(F.interpolate(cropped,size=original_shape,mode='bilinear',align_corners=False).sigmoid()>.5)[0]
 boxes,empty=binary_mask_boxes(binary)
 return dict(boxes=boxes,native_mask_logits=native,selected_query=j,native_query_scores=scores,selected_frame_scores=logits[:,j].sigmoid().max(-1).values[:,None],native_same_track_all8=True,model_steps=1,box_interface='official globally selected refined mask, crop then bilinear to original size >.5, halfopen bbox; empty=full-frame neutral box, no Source/GT/detector fallback',empty_box_conversion=len(empty),empty_indices=empty,input_shape=list(input_shape),nested_mask_all_false=True,expert_model='SgMg-Video-Swin-T-RefYTVOS-joint-eight-adapter',GT_read=False)
if __name__=='__main__':
 import argparse,os,torch
 p=argparse.ArgumentParser();p.add_argument('expert',choices=EXPERTS);q=p.parse_args();torch.set_num_threads(1);t=time.perf_counter();model,tok,rec=construct(q.expert);assert not torch.cuda.is_initialized();rec.update(actual_model_CPU_construction=1,actual_GPU_or_forward=0,elapsed_seconds=time.perf_counter()-t,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid());(D/(q.expert+'_CPU_model_doctor.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
