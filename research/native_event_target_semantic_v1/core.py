"""Read original native word-class head on fixed target-support attention keys."""
from contextlib import contextmanager
import torch
from research.native_event_support_view_v1.core import keep_region,reflected

def restrict_prob(prob,keep):
 assert prob.ndim==4 and keep.ndim==2 and prob.shape[0]==keep.shape[0]and prob.shape[-1]==keep.shape[-1];assert bool(keep.any(1).all())
 if bool(keep.all()):return prob
 x=prob.masked_fill(~keep[:,None,None,:],0);z=x.sum(-1,keepdim=True);assert bool((z>0).all()),'No numerical zero-support fallback allowed';return x/z

@contextmanager
def support_keys(head,keep):
 hooks=[];calls=[]
 def hook(module,args):
  p=args[0];calls.append(dict(keep_mass=p.mul(keep[:,None,None,:]).sum(-1).detach().cpu()))
  if bool(keep.all()):return None
  return (restrict_prob(p,keep),)
 try:
  for layer in head.layer_ca:hooks.append(layer.attention.self.dropout.register_forward_pre_hook(hook))
  yield calls
 finally:
  for h in hooks:h.remove()
 assert len(calls)==len(head.layer_ca)

def masks(aux,offset,n,h,w,valid,kind):
 result=[];stats=[]
 for j in range(n):
  index=offset+2*j;source_index=len(aux)-1-index if kind=='WrongTimeContext'else index;m,fb=keep_region(aux[source_index],valid[j]);control=reflected(m,valid[j]);take=control if kind=='ReflectedContext'else m;result.append(take);stats.append(dict(frame_index=index,source_index=source_index,selected_cells=int(take.sum()),valid_cells=int(valid[j].sum()),target_control_overlap=float((m&control).sum()/m.sum()),empty_ROI_fallback=fb))
 return torch.stack(result).flatten(1),stats
