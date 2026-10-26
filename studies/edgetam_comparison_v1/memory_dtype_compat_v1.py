"""Preserve official BF16 memory storage, explicitly satisfy FP32 V-linear input dtype."""
def hook_factory(counters):
 def before(module,args):
  import torch
  assert len(args)==1 and module.weight.dtype==torch.float32
  x=args[0]
  if x.dtype==torch.float32:return args
  assert x.dtype==torch.bfloat16,'Unexpected precision, no fallback';y=x.to(torch.float32);assert torch.equal(y.to(torch.bfloat16),x);counters['lossless_BF16_storage_to_FP32_V_projection']+=1;return(y,)
 return before

def install(model):
 counts=dict(lossless_BF16_storage_to_FP32_V_projection=0);handles=[]
 for layer in model.memory_attention.layers:handles.append(layer.cross_attn_image.v_proj.register_forward_pre_hook(hook_factory(counts)))
 assert handles
 return handles,counts

def contract():
 import torch
 from types import SimpleNamespace
 c=dict(lossless_BF16_storage_to_FP32_V_projection=0);h=hook_factory(c);m=SimpleNamespace(weight=torch.ones(2,3,dtype=torch.float32));f=torch.tensor([[1.,2.,3.]],dtype=torch.float32);args=(f,);assert h(m,args)is args;b=f.to(torch.bfloat16);v=h(m,(b,))[0];assert torch.equal(v,b.float())and torch.equal(torch.nn.functional.linear(v,m.weight),torch.nn.functional.linear(b.float(),m.weight));assert c['lossless_BF16_storage_to_FP32_V_projection']==1
 return dict(FP32_input_object_unchanged=True,BF16_stored_values_losslessly_represented_in_FP32=True,explicit_reference_linear_exact=True,no_autocast_or_weight_change=True)
