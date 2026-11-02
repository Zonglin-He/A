"""R51 Q/V-only packed-MHA LoRA. No model/data/GPU action at import.

PyTorch parametrization retains the installed original MHA forward, masks,
dropout/need_weights and projection/bias. Root must test Bzero source bridge
before fitting. This is a task interface, not a claimed paper reproduction.
"""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize

MODULE_NAME='decoder.5.cross_v'


class PackedQV(nn.Module):
 def __init__(self,weight,rank=4,alpha=4,seed=0):
  super().__init__();assert tuple(weight.shape)==(864,288) and rank==alpha==4
  self.rank=rank;self.alpha=alpha;d=288
  self.A_q=nn.Parameter(weight.new_empty(rank,d));self.B_q=nn.Parameter(weight.new_zeros(d,rank))
  self.A_v=nn.Parameter(weight.new_empty(rank,d));self.B_v=nn.Parameter(weight.new_zeros(d,rank))
  devices=[weight.device.index] if weight.is_cuda else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(seed)
   nn.init.kaiming_uniform_(self.A_q,a=math.sqrt(5));nn.init.kaiming_uniform_(self.A_v,a=math.sqrt(5))

 def forward(self,original):
  q=(self.B_q@self.A_q)*(self.alpha/self.rank)
  v=(self.B_v@self.A_v)*(self.alpha/self.rank)
  delta=torch.cat([q,torch.zeros_like(q),v],dim=0)
  return original+delta


class QVLoRA:
 def __init__(self,model,rank=4,alpha=4,seed=0):
  self.model=model;model.eval().requires_grad_(False)
  assert len(model.decoder)==6
  module=model.decoder[5].cross_v
  assert isinstance(module,nn.MultiheadAttention) and module.embed_dim==288 and module.num_heads==8
  assert not module.batch_first and module._qkv_same_embed_dim
  assert module.q_proj_weight is module.k_proj_weight is module.v_proj_weight is None
  assert tuple(module.in_proj_weight.shape)==(864,288) and tuple(module.in_proj_bias.shape)==(864,)
  assert not parametrize.is_parametrized(module,'in_proj_weight')
  self.module=module;self.adapter=PackedQV(module.in_proj_weight,rank,alpha,seed)
  parametrize.register_parametrization(module,'in_proj_weight',self.adapter)
  model.eval()
  self.source={n:p.detach().clone() for n,p in self.adapter.named_parameters()}
  self.names=[MODULE_NAME+'.parametrizations.in_proj_weight.0.'+n for n in self.source]
  trainable=[n for n,p in model.named_parameters() if p.requires_grad]
  assert sorted(trainable)==sorted(self.names)
  assert sum(p.numel() for p in self.adapter.parameters())==4608
  self.frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
  self.assert_source_identity()

 def parameters(self):return list(self.adapter.parameters())

 def reset(self):
  with torch.no_grad():
   for name,p in self.adapter.named_parameters():p.copy_(self.source[name]);p.grad=None
  self.model.eval();self.assert_source_identity()

 def assert_source_identity(self):
  assert not torch.count_nonzero(self.adapter.B_q) and not torch.count_nonzero(self.adapter.B_v)
  assert torch.equal(self.module.in_proj_weight,self.module.parametrizations.in_proj_weight.original)

 def check_frozen(self):
  for name,p in self.model.named_parameters():
   if name in self.frozen_versions:
    assert not p.requires_grad and p.grad is None and p._version==self.frozen_versions[name],name
  base=self.module.parametrizations.in_proj_weight.original
  assert torch.equal(self.module.in_proj_weight[288:576],base[288:576]), 'K must remain exact frozen source'
  assert all(not m.training for m in self.model.modules())

 def report(self):
  self.check_frozen();rows=[]
  for name,p in self.adapter.named_parameters():
   grad=p.grad;delta=p.detach()-self.source[name]
   rows.append({'module':MODULE_NAME,'parameter':name,'shape':list(p.shape),'numel':p.numel(),
    'gradient_is_none':grad is None,'gradient_L2':None if grad is None else float(grad.detach().norm()),
    'gradient_nonzero_elements':0 if grad is None else int(torch.count_nonzero(grad)),
    'source_delta_L2':float(delta.norm()),'different_source_elements':int(torch.count_nonzero(delta))})
  return {'interface':'LAST_DECODER_CROSS_V_PACKED_Q_V_LORA_ONLY','module':MODULE_NAME,
   'rank':4,'alpha':4,'scaling':1.,'trainable_parameters':4608,'parameter_names':self.names,
   'K_bias_outproj_other_modules_frozen':True,'initial_A_gradient_can_be_zero_when_B_is_zero':True,
   'parameters':rows}

 def remove(self):
  self.reset();self.check_frozen()
  parametrize.remove_parametrizations(self.module,'in_proj_weight',leave_parametrized=False)
  self.model.requires_grad_(False)


class MatchingMLP:
 """Alternative existing 184960-parameter interface; no LoRA+MLP joint arm."""
 def __init__(self,model):
  assert not parametrize.is_parametrized(model.decoder[5].cross_v,'in_proj_weight'), 'Use separate alternatives, not extra mixed scope'
  self.model=model;model.eval().requires_grad_(False)
  self.module=model.contrastive_align_projection_image;self.module.requires_grad_(True)
  assert sum(p.numel() for p in self.module.parameters())==184960
  self.source={n:p.detach().clone() for n,p in self.module.named_parameters()}
  self.frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}

 def parameters(self):return list(self.module.parameters())

 def reset(self):
  with torch.no_grad():
   for name,p in self.module.named_parameters():p.copy_(self.source[name]);p.grad=None
  self.model.eval()

 def report(self):
  assert all(not m.training for m in self.model.modules())
  for name,p in self.model.named_parameters():
   if name in self.frozen_versions:
    assert not p.requires_grad and p.grad is None and p._version==self.frozen_versions[name],name
  rows=[]
  for name,p in self.module.named_parameters():
   grad=p.grad;delta=p.detach()-self.source[name]
   rows.append({'parameter':name,'numel':p.numel(),'gradient_is_none':grad is None,
    'gradient_L2':None if grad is None else float(grad.detach().norm()),
    'different_source_elements':int(torch.count_nonzero(delta)),'source_delta_L2':float(delta.norm())})
  return {'interface':'EXISTING_IMAGE_MATCHING_MLP_ONLY','trainable_parameters':184960,'parameters':rows}
