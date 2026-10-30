"""Frozen OpenAI CLIP ViT-B/32 crop/text similarity for R8.

This encoder has no 3D output, trainable adapter, EMA, text prompt generator or
implicit tight/context fusion. Callers supply original RGB crops and literal
full descriptions. Similarity is a normalized-feature cosine, not calibrated
instance correctness or visibility. Assets are local, pinned and hash verified.
"""
from pathlib import Path
import hashlib,json,time
from typing import Sequence
import numpy as np
from PIL import Image
import torch
from transformers import CLIPConfig,CLIPModel,CLIPImageProcessor,CLIPTokenizerFast
from safetensors.torch import load_file

REPO_ID='openai/clip-vit-base-patch32'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
DEFAULT_ASSETS=Path(__file__).resolve().parent/'assets/clip-vit-base-patch32'
CONTEXT_LENGTH=77
FEATURE_DIM=512

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def rgb_image(image):
 """Accept PIL or explicit uint8 RGB; reject already normalized tensors/floats."""
 if isinstance(image,Image.Image):return image.convert('RGB')
 if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise TypeError('Expected PIL image or uint8 HxWx3 RGB array')
 if min(image.shape[:2])<1:raise ValueError('Empty image')
 return Image.fromarray(np.ascontiguousarray(image),'RGB')

def crop_rgb(image,boxes_xyxy):
 """Explicit crop helper: floor lower/ceil upper half-open bounds, image clip.
 No implicit context expansion: tight and context bounds must be supplied by G.
 Empty/outside crops fail instead of substituting the whole image.
 """
 image=rgb_image(image);w,h=image.size;out=[];receipt=[]
 for i,box in enumerate(boxes_xyxy):
  a=np.asarray(box,dtype=np.float64)
  if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Nonfinite/invalid crop bounds')
  x0,y0,x1,y1=int(np.floor(a[0])),int(np.floor(a[1])),int(np.ceil(a[2])),int(np.ceil(a[3]))
  x0,y0,x1,y1=max(0,x0),max(0,y0),min(w,x1),min(h,y1)
  if x1<=x0 or y1<=y0:raise ValueError(f'Empty crop index {i}')
  out.append(image.crop((x0,y0,x1,y1)));receipt.append({'crop_index':i,'width':x1-x0,'height':y1-y0,'source_image_bounds_clipped':bool(a[0]<0 or a[1]<0 or a[2]>w or a[3]>h)})
 return out,receipt

class FrozenSemanticEncoder:
 def __init__(self,asset_dir=DEFAULT_ASSETS,device='cpu',batch_size=32):
  self.asset_dir=Path(asset_dir);self.device=torch.device(device);self.batch_size=int(batch_size)
  if self.batch_size<1:raise ValueError('batch_size must be positive')
  if self.device.type not in ('cpu','cuda'):raise ValueError('Only explicit CPU/CUDA supported; never automatic device selection')
  receipt=json.loads((self.asset_dir/'SAFE_CONVERSION_RECEIPT.json').read_text())
  assert receipt['state']=='safe_weights_only_official_to_local_safetensors_strict_model_verified'
  assert receipt['repo_id']==REPO_ID and receipt['revision']==REVISION
  for name,digest in receipt['runtime_files_sha256'].items():assert file_sha(self.asset_dir/name)==digest,('Asset hash mismatch',name)
  self.processor=CLIPImageProcessor.from_pretrained(str(self.asset_dir),local_files_only=True)
  self.tokenizer=CLIPTokenizerFast.from_pretrained(str(self.asset_dir),local_files_only=True)
  config=CLIPConfig.from_pretrained(str(self.asset_dir),local_files_only=True)
  assert config.text_config.max_position_embeddings==CONTEXT_LENGTH and config.projection_dim==FEATURE_DIM
  self.model=CLIPModel(config)
  state=load_file(str(self.asset_dir/'model.safetensors'),device='cpu')
  self.model.load_state_dict(state,strict=True);del state
  self.model=self.model.to(device=self.device,dtype=torch.float32).eval();self.model.requires_grad_(False)
  self.resource={'repo_id':REPO_ID,'revision':REVISION,'derived_safetensors_sha256':receipt['derived_safetensors_sha256'],'device':str(self.device),'dtype':'float32','inference_only':True,'no_3D_box_or_instance_correctness_output':True}
  self._assert_frozen()

 def _assert_frozen(self):
  assert not self.model.training and all(not m.training for m in self.model.modules())
  assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())

 def tokenize_texts(self,texts):
  texts=list(texts)
  if any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Each full description must be a nonempty literal string')
  untruncated=self.tokenizer(texts,add_special_tokens=True,padding=False,truncation=False)['input_ids'] if texts else []
  lengths=[len(t) for t in untruncated]
  tokens=self.tokenizer(texts,padding=True,truncation=True,max_length=CONTEXT_LENGTH,return_tensors='pt') if texts else None
  used=tokens['attention_mask'].sum(1).tolist() if texts else []
  receipt={'text_count':len(texts),'context_length':CONTEXT_LENGTH,'includes_start_end_special_tokens':True,'literal_full_description_no_prompt_template_or_noun_replacement':True,'original_token_lengths':lengths,'used_token_lengths':used,'truncated_count':sum(n>CONTEXT_LENGTH for n in lengths),'truncated_indices':[i for i,n in enumerate(lengths) if n>CONTEXT_LENGTH],'dropped_token_count':sum(max(0,n-CONTEXT_LENGTH) for n in lengths),'truncation_policy':'right truncation to77 with tokenizer-preserved end token; no silent overflow','padding':'actual batch longest sequence, attention_mask provided'}
  return tokens,receipt

 @torch.inference_mode()
 def encode_texts(self,texts,batch_size=None):
  self._assert_frozen();texts=list(texts);tokens,receipt=self.tokenize_texts(texts);batch=int(batch_size or self.batch_size);out=[]
  for start in range(0,len(texts),batch):
   inputs={k:v[start:start+batch].to(self.device) for k,v in tokens.items() if k in ('input_ids','attention_mask')}
   features=self.model.get_text_features(**inputs)
   norms=features.norm(dim=-1,keepdim=True);assert torch.isfinite(features).all() and (norms>0).all()
   out.append((features/norms).float().cpu())
  self._assert_frozen()
  return torch.cat(out,0) if out else torch.empty((0,FEATURE_DIM),dtype=torch.float32),receipt

 def preprocess_crops(self,crops):
  images=[rgb_image(x) for x in crops]
  pixels=self.processor(images=images,return_tensors='pt')['pixel_values'] if images else torch.empty((0,3,224,224),dtype=torch.float32)
  assert pixels.dtype==torch.float32 and tuple(pixels.shape[1:])==(3,224,224) and torch.isfinite(pixels).all()
  return pixels

 @torch.inference_mode()
 def encode_crops(self,crops,batch_size=None):
  self._assert_frozen();crops=list(crops);batch=int(batch_size or self.batch_size);out=[]
  for start in range(0,len(crops),batch):
   pixels=self.preprocess_crops(crops[start:start+batch]).to(self.device)
   features=self.model.get_image_features(pixel_values=pixels);norms=features.norm(dim=-1,keepdim=True)
   assert torch.isfinite(features).all() and (norms>0).all()
   out.append((features/norms).float().cpu())
  self._assert_frozen()
  return torch.cat(out,0) if out else torch.empty((0,FEATURE_DIM),dtype=torch.float32)

 @torch.inference_mode()
 def match(self,crops,texts,batch_size=None,crop_roles=None):
  """Return raw cosines[Ncrop,Ntext], never softmax or implicit view fusion.
  Tight/context role tags, when supplied, are preserved for order auditing only.
  """
  crops=list(crops);texts=list(texts)
  if crop_roles is not None and len(crop_roles)!=len(crops):raise ValueError('crop_roles length mismatch')
  started=time.perf_counter();images=self.encode_crops(crops,batch_size);text,text_receipt=self.encode_texts(texts,batch_size)
  cosines=(images@text.T).numpy()
  assert np.isfinite(cosines).all() and (np.abs(cosines)<=1.00001).all()
  return {'cosines':cosines,'text_receipt':text_receipt,'crop_receipt':{'crop_count':len(crops),'roles':list(crop_roles) if crop_roles is not None else None,'tight_context_fusion':'none; caller owns explicit fusion','original_crop_sizes_wh':[list(rgb_image(x).size) for x in crops],'preprocessing':'pinned HF CLIPImageProcessor; RGB, shortest-edge224 bicubic, center224 crop, rescale1/255, official mean/std'},'resource':dict(self.resource),'seconds':time.perf_counter()-started}
