"""Strict official-state CPU construction, later separately gated GPU use."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_generalization_v1';V=D/'vendor/ReferFormer'
def build_cpu():
 import torch
 sys.path[:0]=[str(V),str(D/'runtime/python'),str(R/'runtime/referdino/cuda_build')]
 import opts,models
 import models.backbone as backbone
 import models.referformer as ref
 from transformers import RobertaModel,RobertaConfig,RobertaTokenizerFast
 args=opts.get_args_parser().parse_args([]);args.device='cpu';args.masks=True;args.binary=True;args.with_box_refine=True;args.freeze_text_encoder=True
 assert args.backbone=='resnet50'and args.num_queries==5
 assets=D/'vendor/roberta-base';config=RobertaConfig.from_pretrained(str(assets),local_files_only=True);tokenizer=RobertaTokenizerFast.from_pretrained(str(assets),local_files_only=True)
 class TextFactory:
  @staticmethod
  def from_pretrained(name,*a,**kw):assert name=='roberta-base'and not a;return RobertaModel(config)
 class TokenFactory:
  @staticmethod
  def from_pretrained(name,*a,**kw):assert name=='roberta-base'and not a;return tokenizer
 old=(backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast)
 backbone.is_main_process=lambda:False;ref.RobertaModel=TextFactory;ref.RobertaTokenizerFast=TokenFactory
 start=time.perf_counter()
 try:model,criterion,post=models.build_model(args)
 finally:backbone.is_main_process,ref.RobertaModel,ref.RobertaTokenizerFast=old
 del criterion,post
 state=torch.load(D/'vendor/r50_refytvos_finetune.pth',map_location='cpu',weights_only=False)['model'];result=model.load_state_dict(state,strict=False);learned=set(dict(model.named_parameters()));assert not(set(result.missing_keys)&learned),result
 allowed=lambda k:k.endswith(('position_ids','token_type_ids'))
 assert all(allowed(k)for k in result.missing_keys)and all(allowed(k)for k in result.unexpected_keys),result
 current=model.state_dict()
 for k,t in current.items():
  if k in state:assert torch.equal(t,state[k]),k
  else:assert allowed(k),k
 for k in result.unexpected_keys:
  t=state[k]
  if k.endswith('position_ids'):assert torch.equal(t,torch.arange(t.shape[-1],dtype=t.dtype).expand_as(t)),k
  else:assert torch.equal(t,torch.zeros_like(t)),k
 model.eval().requires_grad_(False);assert all(not p.is_cuda for p in model.parameters())
 return model,tokenizer,dict(all_learned_parameters_covered=True,all_shared_state_tensors_array_exact=True,missing_deterministic_buffers=list(result.missing_keys),unexpected_verified_deterministic_buffers=list(result.unexpected_keys),redundant_Imagenet_and_Roberta_weight_preloads_suppressed=True,official_vendor_source_unmodified=True,model_parameters=sum(p.numel()for p in model.parameters()),elapsed_CPU_construction_and_load_seconds=time.perf_counter()-start,actual_forward=0)
