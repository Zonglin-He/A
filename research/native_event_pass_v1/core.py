"""Observe existing native decoder outputs without replacing any tensor or query."""
from contextlib import contextmanager
import torch

@contextmanager
def collect_passes(module):
 outputs=[]
 def hook(m,args,result):
  assert isinstance(result,tuple)and len(result)==2
  hidden=result[1];assert torch.is_tensor(hidden)and hidden.ndim==4 and hidden.shape[0]==6 and hidden.shape[1]==1 and hidden.shape[-1]==256
  outputs.append(hidden.detach().clone().cpu())
 h=module.register_forward_hook(hook)
 try:yield outputs
 finally:h.remove()
 assert len(outputs)==4,'Expected original first,second per each offset'

def split_passes(outputs):
 assert len(outputs)==4
 return [outputs[0],outputs[2]],[outputs[1],outputs[3]]
