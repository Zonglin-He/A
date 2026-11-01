"""Equivalent strict local Qwen loader without optional accelerate device_map."""
import json
from common37 import ROOT,EXP,sha,read
from association_runner import FrozenVLM,model_snap
class LocalFrozenVLM(FrozenVLM):
 def __init__(self,model_dir,config):
  import torch
  from transformers import AutoProcessor,AutoTokenizer,Qwen2_5_VLForConditionalGeneration
  self.torch=torch;self.config=config;torch.manual_seed(config.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
  self.processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=config.min_pixels,max_pixels=config.max_pixels);self.processor.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
  self.model,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
  if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('Strict load failed '+json.dumps(info))
  self.model=self.model.to('cuda');self.loading_info=info;self.model.eval().requires_grad_(False);self.snap=model_snap(self.model);assert not self.snap['training'] and not any(p[-1] for p in self.snap['params']);torch.cuda.synchronize()
def check_amendment():
 a=read(EXP/'MEASUREMENT_AMENDMENT_LOCK.json');assert a['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for n,h in a['program_sha256'].items():assert sha(ROOT/n)==h,n
 return a
