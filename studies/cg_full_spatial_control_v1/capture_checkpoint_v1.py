"""Observe the unmodified Source call; persist full output before postprocessing."""
from spatial_common_v1 import *
from contextlib import contextmanager

def cpu_copy(x):
 import torch,copy
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return{k:cpu_copy(v)for k,v in x.items()}
 if isinstance(x,list):return[cpu_copy(v)for v in x]
 if isinstance(x,tuple):return tuple(cpu_copy(v)for v in x)
 return copy.deepcopy(x)

@contextmanager
def checkpoint_before_postprocess(model,path,context):
 import torch
 assert 'forward'not in model.__dict__;original=model.forward;grids={};calls=[0]
 def pre(module,args,kwargs):
  videos=kwargs['videos'];grids['projected']=videos.tensors.detach().cpu().clone();grids['valid']=(~videos.mask).cpu().clone()
 def after(module,args,kwargs,out):
  gh,gw=out['fea_map_size'];grids['spatial']=out['encoded_feature'][:gh*gw].permute(1,0,2).detach().cpu().clone();grids['spatial_valid']=(~out['encoded_mask'][:,:gh*gw]).cpu().clone();grids['shape']=(gh,gw)
 handles=[model.ground_encoder.register_forward_pre_hook(pre,with_kwargs=True),model.ground_encoder.register_forward_hook(after,with_kwargs=True)]
 def observed(*args,**kwargs):
  tick=time.perf_counter();out=original(*args,**kwargs);torch.cuda.synchronize();forward=time.perf_counter()-tick
  # No native readout, box validation or teacher access before this complete
  # returned Source output and both native grids are saved.
  calls[0]+=1;tick=time.perf_counter();save(path,dict(outputs=cpu_copy(out),grids=cpu_copy(grids),context=context,GT_read=False));io=time.perf_counter()-tick
  atomic(path.with_suffix('.json'),dict(path=str(path.relative_to(D)),sha256=sha(path),Source_forwards=calls[0],forward_with_observer_transfers_seconds=forward,full_output_checkpoint_IO_seconds=io,before_PostProcess_and_native_assertions=True,before_teacher_and_ROI=True,context=context,GT_read=False))
  assert calls[0]==1;return out
 model.forward=observed
 try:yield calls
 finally:
  del model.forward
  for h in handles:h.remove()
  assert 'forward'not in model.__dict__
